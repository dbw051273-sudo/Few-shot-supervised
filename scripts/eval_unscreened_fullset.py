"""Zero-retrain unscreened raw full-testset evaluator (P2 Experiment 8).
Evaluates trained clean-480 (+ 20-shot) weights or external baseline generated PNGs directly
on raw unscreened datasets (e.g., Brain 841, Kidney 561, liver 591, Breast strict unseen)
with persistent Cellpose GT mask caching, locked KID subset_size=50, KID(x10^-3) reporting,
and multi-model side-by-side summary merging.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpu_guard import acquire_runtime_gpu_guard, preparse_gpu_and_env  # noqa: E402

_PRE_ARGS = preparse_gpu_and_env('Workstation fullset eval')

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
from cellpose import models  # noqa: E402
from torchmetrics.image.fid import FrechetInceptionDistance  # noqa: E402
from torchmetrics.image.kid import KernelInceptionDistance  # noqa: E402
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402

from round1_losses_and_data import (  # noqa: E402
    apply_od_scale,
    build_or_load_lomo_splits,
    encode_all,
    infer_to_pngs,
    mean_od,
    pngs,
    rgb,
    tensor_from_paths,
)
from round1_models import Codec, build_flow_model  # noqa: E402
from uni_encoder import UNIEncoder  # noqa: E402
from v2_unified_eval import KID_SUBSET_SIZE, OfficialScalarMetrics, build_per_image_lpips, instance_f1_at_05  # noqa: E402

_NUM_RE = re.compile(r'(\d+)')


def _file_num(name: str) -> int | None:
    m = _NUM_RE.search(name)
    return int(m.group(1)) if m else None


def resolve_dir(explicit: Path | None, candidates: list[Path], check_sub: str, label: str) -> Path:
    pool = [explicit] if explicit is not None else candidates
    for c in pool:
        if c is not None and (c / check_sub).exists():
            return c.resolve()
    raise FileNotFoundError(f'Cannot locate {label} (checked {pool}). Pass path explicitly.')


def resolve_raw_pair_dirs(raw_root: Path, organ: str) -> tuple[Path, Path]:
    if (raw_root / organ / 'test' / 'input').is_dir() and (raw_root / organ / 'test' / 'target').is_dir():
        return raw_root / organ / 'test' / 'input', raw_root / organ / 'test' / 'target'
    if (raw_root / organ / 'input').is_dir() and (raw_root / organ / 'target').is_dir():
        return raw_root / organ / 'input', raw_root / organ / 'target'
    raise FileNotFoundError(f'Cannot find raw input/target dirs for {organ} under {raw_root}')


def load_seen_exclusions(clean_root: Path, organ: str) -> tuple[set[str], set[int]]:
    seen_names: set[str] = set()
    seen_nums: set[int] = set()
    for split_name in ('train', 'val'):
        d = clean_root / organ / split_name / 'input'
        if not d.is_dir():
            continue
        for p in d.glob('*.png'):
            seen_names.add(p.name)
            if (n := _file_num(p.name)) is not None:
                seen_nums.add(n)
    return seen_names, seen_nums


def filter_unseen_names(names: list[str], seen_names: set[str], seen_nums: set[int]) -> list[str]:
    seen_adj = {sn + d for sn in seen_nums for d in (-1, 0, 1)}
    out = []
    for nm in names:
        if nm in seen_names:
            continue
        num = _file_num(nm)
        if num is not None and num in seen_adj:
            continue
        out.append(nm)
    return out


def compute_center_scale_fallback(
    codec: Codec,
    clean_root: Path,
    target_organ: str,
) -> tuple[float, float]:
    pis = []
    for org in ('Brain', 'Breast', 'Kidney', 'liver'):
        if org == target_organ:
            continue
        tr_dir = clean_root / org / 'train' / 'input'
        files = sorted(tr_dir.glob('*.png'))[:480]
        pis.append(tensor_from_paths(files))
    with torch.no_grad():
        lat_pi = encode_all(codec, torch.cat(pis))
    return float(lat_pi.mean()), float(lat_pi.std())


def load_weights_for_eval(
    work_dir: Path,
    clean_root: Path,
    fold: str,
    arch: str,
    dim: int,
    stage: str,
) -> tuple[torch.nn.Module, Codec, float, float, str]:
    ckpt_dir = work_dir / 'checkpoints' / fold
    base_tag = f'{fold}__{arch}_d{dim}'
    flow = build_flow_model(arch, dim=dim)
    codec = Codec()

    if stage in ('0shot', '0shot_halfOD'):
        pre_path = ckpt_dir / f'{base_tag}_pretrain.pth'
        if not pre_path.exists():
            raise FileNotFoundError(f'Missing pretrain checkpoint: {pre_path}')
        obj = torch.load(pre_path, map_location='cpu', weights_only=True)
        flow.load_state_dict(obj['swad'])
        pref_codec = ckpt_dir / str(obj.get('codec_name', '')) if obj.get('codec_name') else None
        if pref_codec is not None and pref_codec.exists():
            codec_path = pref_codec
        else:
            cands = sorted(ckpt_dir.glob('codec_rae*.pth')) + sorted(ckpt_dir.glob('codec_s*.pth'))
            if not cands:
                raise FileNotFoundError(f'Missing codec checkpoint in {ckpt_dir}')
            codec_path = cands[0]
        codec.load_state_dict(torch.load(codec_path, map_location='cpu', weights_only=True))
        src_desc = f'{pre_path.name} + {codec_path.name}'
    else:
        ad_path = ckpt_dir / f'{base_tag}__{stage}.pth'
        if not ad_path.exists():
            raise FileNotFoundError(f'Missing adapted checkpoint: {ad_path}')
        obj = torch.load(ad_path, map_location='cpu', weights_only=True)
        flow.load_state_dict(obj['flow'])
        codec.load_state_dict(obj['codec'])
        src_desc = ad_path.name

    flow.cuda().eval()
    codec.cuda().eval()
    if 'center' in obj and 'scale' in obj:
        center, scale = float(obj['center']), float(obj['scale'])
    else:
        center, scale = compute_center_scale_fallback(codec, clean_root, fold)
    return flow, codec, center, scale, src_desc


def load_raw_gt_mask_cache(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        return {}
    with np.load(npz_path) as data:
        return {str(k): m for k, m in zip(data['keys'], data['masks'])}


def save_raw_gt_mask_cache(npz_path: Path, cache: dict[str, np.ndarray]) -> None:
    if not cache:
        return
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache.keys())
    masks = np.stack([cache[k].astype(np.uint16) for k in keys], axis=0)
    tmp_path = npz_path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp_path, keys=np.array(keys), masks=masks)
    tmp_path.replace(npz_path)


@torch.no_grad()
def evaluate_unscreened_subset(
    label: str,
    names: list[str],
    in_dir: Path,
    gt_dir: Path,
    flow: torch.nn.Module | None,
    codec: Codec | None,
    uni_enc: UNIEncoder | None,
    cell_model,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    center: float,
    scale: float,
    device: torch.device,
    gt_mask_cache: dict[str, np.ndarray],
    gt_cache_path: Path,
    steps: int = 4,
    invert_pi: bool = False,
    half_od_scale: np.ndarray | None = None,
    external_pred_dir: Path | None = None,
    save_dir: Path | None = None,
    batch_size: int = 32,
) -> dict:
    if not names:
        raise ValueError(f'Empty test subset for {label}')
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    preprocess_fid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    preprocess_lp = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    use_uni = flow is not None and bool(getattr(flow, 'use_uni_2d', False))
    use_fid_kid = len(names) >= 2
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device) if use_fid_kid else None
    kid = KernelInceptionDistance(subset_size=min(KID_SUBSET_SIZE, len(names)), normalize=True).to(device) if use_fid_kid else None

    psnrs, ssims, lps, pcc_gts, pcc_pis, dices, f1s, gt_cnts, pr_cnts = [], [], [], [], [], [], [], [], []
    dt = 1.0 / float(steps)
    t0 = time.perf_counter()
    cache_dirty = False

    for i in range(0, len(names), batch_size):
        bn = names[i:i + batch_size]
        pi_arr, gt_arr = [], []
        for nm in bn:
            with Image.open(in_dir / nm) as im:
                a = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)
                if invert_pi:
                    a = (255 - a).astype(np.uint8)
                pi_arr.append(a)
            with Image.open(gt_dir / nm) as im:
                b = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)
                gt_arr.append(b)

        if external_pred_dir is not None:
            pred_arr = []
            for nm in bn:
                pp = external_pred_dir / nm
                if not pp.exists():
                    raise FileNotFoundError(f'Missing external prediction PNG: {pp}')
                with Image.open(pp) as im:
                    pred_arr.append(np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8))
        else:
            bx = torch.from_numpy(np.stack(pi_arr)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).cuda()
            uni_tok = None
            if use_uni and uni_enc is not None:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    uni_tok = uni_enc(bx)[:, 1:].float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z0 = (codec.encode(bx).float() - center) / scale
            z = z0
            for k in range(steps):
                t_in = z.new_full((len(z),), k * dt)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    v = flow(z, z0, t_in, uni_tok)
                z = z + v.float() * dt
            with torch.autocast('cuda', dtype=torch.bfloat16):
                rec = codec.decode(z * scale + center).float().clamp(-1, 1)
            pred_arr = list(((rec + 1.0) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
            if half_od_scale is not None:
                pred_arr = [
                    np.rint((im.astype(np.float32) + apply_od_scale(im, half_od_scale)) / 2.0).clip(0, 255).astype(np.uint8)
                    for im in pred_arr
                ]
            if save_dir is not None:
                for nm, im in zip(bn, pred_arr):
                    Image.fromarray(im).save(save_dir / nm)

        if use_fid_kid:
            ft = torch.stack([preprocess_fid(im) for im in gt_arr]).to(device)
            fg = torch.stack([preprocess_fid(im) for im in pred_arr]).to(device)
            fid.update(ft, real=True)
            fid.update(fg, real=False)
            kid.update(ft, real=True)
            kid.update(fg, real=False)

        lt = torch.stack([preprocess_lp(im) for im in gt_arr]).to(device)
        lg = torch.stack([preprocess_lp(im) for im in pred_arr]).to(device)
        lps.extend(lpips_fn(lg, lt).view(-1).cpu().tolist())

        # Reuse cached GT Cellpose masks if available; compute only missing items
        miss_idx = [j for j, nm in enumerate(bn) if nm not in gt_mask_cache]
        if miss_idx:
            miss_imgs = [gt_arr[j] for j in miss_idx]
            mb, _, _ = cell_model.eval(miss_imgs, diameter=None, channels=[0, 0], batch_size=len(miss_imgs))
            for j, m_mask in zip(miss_idx, mb):
                gt_mask_cache[bn[j]] = m_mask.astype(np.uint16)
            cache_dirty = True
        gt_masks_b = [gt_mask_cache[nm] for nm in bn]

        pr_masks_b, _, _ = cell_model.eval(pred_arr, diameter=None, channels=[0, 0], batch_size=len(pred_arr))

        for pa, ga, pr, gm, pm in zip(pi_arr, gt_arr, pred_arr, gt_masks_b, pr_masks_b):
            pb, gb, prb = pa[:, :, ::-1], ga[:, :, ::-1], pr[:, :, ::-1]
            psnrs.append(OfficialScalarMetrics.psnr(gb, prb))
            ssims.append(OfficialScalarMetrics.compute_ssim_official(
                ga.transpose(2, 0, 1).astype(np.float32) / 255.0,
                pr.transpose(2, 0, 1).astype(np.float32) / 255.0,
            ))
            pcc_gts.append(OfficialScalarMetrics.calculate_generic_pcc(prb, gb))
            pcc_pis.append(OfficialScalarMetrics.calculate_generic_pcc(prb, pb))
            f1, _tp, gn, pn = instance_f1_at_05(gm.astype(np.uint16), pm.astype(np.uint16))
            dice = OfficialScalarMetrics.compute_dice_from_bins((gm > 0).astype(np.float32), (pm > 0).astype(np.float32))
            dices.append(dice)
            f1s.append(f1)
            gt_cnts.append(gn)
            pr_cnts.append(pn)

        if (i + len(bn)) % 320 == 0 or (i + len(bn)) == len(names):
            print(f'[{label}] {i + len(bn)}/{len(names)} (gt_cached={len(bn) - len(miss_idx)}/{len(bn)}) ({time.perf_counter()-t0:.1f}s)', flush=True)

    if cache_dirty:
        save_raw_gt_mask_cache(gt_cache_path, gt_mask_cache)
        print(f'[CACHE] Updated GT Cellpose cache ({len(gt_mask_cache)} masks) -> {gt_cache_path}', flush=True)

    gt_c = np.array(gt_cnts, dtype=np.float64)
    pr_c = np.array(pr_cnts, dtype=np.float64)
    cnt_r = float(np.corrcoef(gt_c, pr_c)[0, 1]) if (gt_c.std() > 0 and pr_c.std() > 0) else 0.0
    elapsed = time.perf_counter() - t0
    metrics = {
        'psnr': float(np.mean(psnrs)),
        'ssim': float(np.mean(ssims)),
        'lpips': float(np.mean(lps)),
        'fid': float(fid.compute().item()) if use_fid_kid else 0.0,
        'kid': float(kid.compute()[0].item()) if use_fid_kid else 0.0,
        'nuclei_dice': float(np.mean(dices)),
        'nuclei_f1_05': float(np.mean(f1s)),
        'pcc_gt': float(np.mean(pcc_gts)),
        'pcc_pi': float(np.mean(pcc_pis)),
        'nuclei_count_r': cnt_r,
    }
    print(
        f"[RAW_FULLSET] {label:28s} n={len(names):4d} | PSNR={metrics['psnr']:5.2f} SSIM={metrics['ssim']:.4f} "
        f"LPIPS={metrics['lpips']:.4f} KIDx1e3={metrics['kid']*1000.0:5.2f} Dice={metrics['nuclei_dice']:.4f} "
        f"F1={metrics['nuclei_f1_05']:.4f} ({elapsed:.1f}s)",
        flush=True,
    )
    return {
        'label': label,
        'n': len(names),
        'invert_pi': invert_pi,
        'wall_seconds': round(elapsed, 2),
        'metrics': metrics,
    }


def write_summary_files(out_dir: Path, raw_root: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'unscreened_fullset_summary.json'
    json_path.write_text(
        json.dumps({'updated_at': datetime.now().isoformat(), 'raw_root': str(raw_root), 'runs': rows}, indent=2),
        encoding='utf-8',
    )
    csv_cols = [
        'label', 'fold', 'arch', 'dim', 'stage', 'steps', 'n', 'invert_pi',
        'psnr', 'ssim', 'lpips', 'kid', 'kid_x1e3', 'nuclei_dice', 'nuclei_f1_05',
        'fid', 'pcc_gt', 'pcc_pi', 'nuclei_count_r', 'wall_seconds',
    ]
    with (out_dir / 'unscreened_fullset_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in rows:
            flat = {k: r.get(k, '') for k in csv_cols}
            for mk, mv in r['metrics'].items():
                if mk in flat:
                    flat[mk] = round(mv, 6)
            flat['kid_x1e3'] = round(r['metrics']['kid'] * 1000.0, 3)
            w.writerow(flat)

    md = [
        '# Unscreened Raw Full-Testset Zero-Retrain Evaluation (P2 Experiment 8)',
        '',
        f'- Updated: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}` | Raw Source: `{raw_root}`',
        f'- Note: `KID` values are scaled by `10^3` (`subset_size={KID_SUBSET_SIZE}`). `FID` depends on test sample count `N` and is omitted from cross-`N` main tables (retained in JSON/CSV for same-`N` diagnostics only).',
        '',
        '| Split | Fold | Model / Arch | Stage | N | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID (×10⁻³) ↓ | Dice ↑ | F1@0.5 ↑ |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in sorted(rows, key=lambda x: (x['fold'], x['label'], str(x['arch']), str(x['stage']))):
        m = r['metrics']
        md.append(
            f"| `{r['label']}` | {r['fold']} | {r['arch']} | {r['stage']} | {r['n']} | {m['psnr']:.2f} | "
            f"{m['ssim']:.4f} | {m['lpips']:.4f} | {m['kid'] * 1000.0:.2f} | "
            f"{m['nuclei_dice']:.4f} | {m['nuclei_f1_05']:.4f} |"
        )
    (out_dir / 'unscreened_fullset_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def main() -> None:
    default_work = SCRIPT_DIR.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu-id', type=str, default='2', help='Physical CUDA GPU index (default: 2)')
    ap.add_argument('--allow-other-gpu', action='store_true', default=False)
    ap.add_argument('--allow-gpu-sharing', action='store_true', default=False)
    ap.add_argument('--online-hf', action='store_true', default=False)
    ap.add_argument('--work-dir', type=Path, default=default_work)
    ap.add_argument('--cache-dir', type=Path, default=None, help='Dir storing reusable raw_gt_nuclei_256_*.npz GT mask caches')
    ap.add_argument('--data-dir', type=Path, default=None, help='Clean UniPrior_v2 dir (for +/-1 seen split exclusion)')
    ap.add_argument('--raw-data-dir', type=Path, default=None, help='Raw unscreened Datasets_processed dir')
    ap.add_argument('--folds', nargs='+', default=['Breast', 'Brain', 'Kidney', 'liver'])
    ap.add_argument('--archs', nargs='+', default=None, help='One or more trained backbones to compare (default: [u1_up64_only, u0_base])')
    ap.add_argument('--arch', type=str, default=None, help='Alias for single arch')
    ap.add_argument('--dim', type=int, default=192)
    ap.add_argument('--stage', type=str, default='m6_k20', help='m6_k20 | naive_ft_k20 | m6_k10 | m6_k5 | 0shot | 0shot_halfOD')
    ap.add_argument('--steps', type=int, default=4, help='Locked NFE sampling steps (default: 4)')
    ap.add_argument('--external-pred-dir', type=Path, default=None, help='Optional pre-generated PNG folder for external baselines (single fold)')
    ap.add_argument('--external-name', type=str, default='ExternalBaseline', help='Display arch/method name when using --external-pred-dir')
    ap.add_argument('--no-invert-liver', action='store_true', default=False)
    ap.add_argument('--include-raw-all', action='store_true', default=False, help='Also evaluate literal raw directories alongside +/-1 unseen')
    ap.add_argument('--include-all-breast', action='store_true', default=False, help='Alias to include full raw Breast 5636')
    ap.add_argument('--overwrite-summary', action='store_true', default=False, help='Reset existing unscreened_fullset_summary rows instead of merging')
    ap.add_argument('--save-images', action='store_true', default=False)
    ap.add_argument('--max-images', type=int, default=None, help='Optional cap per split for fast smoke verification')
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    acquire_runtime_gpu_guard(args.gpu_id, allow_sharing=args.allow_gpu_sharing)
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.resolve()
    elif (work_dir / 'eval_cache').is_dir():
        cache_dir = (work_dir / 'eval_cache').resolve()
    else:
        cache_dir = (SCRIPT_DIR.parent / 'eval_cache').resolve()

    clean_root = resolve_dir(
        args.data_dir,
        [work_dir.parent / 'UniPrior_v2', work_dir / 'UniPrior_v2', work_dir.parent / 'UniPrior_四器官洁净均衡集_v2'],
        'Breast/train/input',
        'clean UniPrior_v2 dataset',
    )
    raw_root = resolve_dir(
        args.raw_data_dir,
        [
            work_dir.parent / 'Datasets_processed',
            clean_root.parent / 'Datasets_processed',
            Path.home() / 'Desktop' / 'Datasets_processed',
        ],
        'Brain/test/input',
        'raw unscreened Datasets_processed',
    )

    if args.external_pred_dir is not None and len(args.folds) != 1:
        raise ValueError('--external-pred-dir expects a single fold via --folds <Organ>')
    archs_list = args.archs if args.archs is not None else ([args.arch] if args.arch is not None else ['u1_up64_only', 'u0_base'])

    device = torch.device('cuda:0')
    uni_enc = UNIEncoder(freeze=True).cuda().eval() if args.external_pred_dir is None else None
    lpips_fn = build_per_image_lpips(net_type='alex', normalize=True).cuda()
    cell_model = models.CellposeModel(gpu=True, model_type='nuclei')

    out_dir = work_dir / 'eval_results'
    summary_json = out_dir / 'unscreened_fullset_summary.json'
    merged_map: dict[tuple[str, str, int, str], dict] = {}
    if not args.overwrite_summary and summary_json.exists():
        try:
            old_runs = json.loads(summary_json.read_text(encoding='utf-8')).get('runs', [])
            for r in old_runs:
                key = (str(r['label']), str(r['arch']), int(r.get('dim', 0)), str(r['stage']))
                merged_map[key] = r
        except Exception:
            pass

    splits = None
    if args.stage == '0shot_halfOD' and args.external_pred_dir is None:
        splits = build_or_load_lomo_splits(clean_root, cache_dir, out_dir / 'round1_splits.json', seed=2026)

    for fold in args.folds:
        in_dir, gt_dir = resolve_raw_pair_dirs(raw_root, fold)
        all_names = sorted(pngs(in_dir).keys() & pngs(gt_dir).keys())
        seen_names, seen_nums = load_seen_exclusions(clean_root, fold)
        unseen_names = filter_unseen_names(all_names, seen_names, seen_nums)
        inv_pi = (fold == 'liver') and (not args.no_invert_liver)

        targets = [(f'{fold}_unseen_raw', unseen_names)]
        if (args.include_raw_all or (fold == 'Breast' and args.include_all_breast)) and len(all_names) != len(unseen_names):
            targets.append((f'{fold}_raw_all', all_names))

        gt_cache_path = cache_dir / f'raw_gt_nuclei_256_{fold}.npz'
        gt_mask_cache = load_raw_gt_mask_cache(gt_cache_path)
        print(f'[CACHE] {fold}: loaded {len(gt_mask_cache)} cached GT masks from {gt_cache_path.name}', flush=True)

        run_models = [(args.external_name, 0, True)] if args.external_pred_dir is not None else [
            (a, args.dim, False) for a in archs_list
        ]
        for arch_name, dim_val, is_ext in run_models:
            if is_ext:
                flow = codec = None
                center, scale, ckpt_desc, half_od_sc = 0.0, 1.0, str(args.external_pred_dir), None
            else:
                flow, codec, center, scale, ckpt_desc = load_weights_for_eval(work_dir, clean_root, fold, arch_name, dim_val, args.stage)
                half_od_sc = None
                if args.stage == '0shot_halfOD':
                    refs_ids = splits['organs'][fold]['refs_5']
                    refs_pi_t = tensor_from_paths(clean_root / fold / 'val' / 'input' / n for n in refs_ids)
                    refs_gt_od = mean_od([rgb(clean_root / fold / 'val' / 'target' / n) for n in refs_ids], f'{fold}_refs_gt')
                    refs_pred = infer_to_pngs(
                        flow, codec, refs_pi_t, uni_enc, center, scale, refs_ids,
                        work_dir / 'generated_unscreened' / f'_tmp_refs_{fold}_{arch_name}', steps=args.steps,
                    )
                    half_od_sc = refs_gt_od / mean_od([refs_pred[n] for n in refs_ids], f'{fold}_refs_pred')

            for label, nm_list in targets:
                use_names = nm_list[:args.max_images] if args.max_images is not None else nm_list
                sdir = (work_dir / 'generated_unscreened' / f'{fold}__{arch_name}_d{dim_val}__{args.stage}' / label) if (args.save_images and not is_ext) else None
                res = evaluate_unscreened_subset(
                    label, use_names, in_dir, gt_dir, flow, codec, uni_enc, cell_model, lpips_fn,
                    center, scale, device, gt_mask_cache, gt_cache_path,
                    steps=args.steps, invert_pi=inv_pi, half_od_scale=half_od_sc,
                    external_pred_dir=args.external_pred_dir if is_ext else None, save_dir=sdir,
                )
                res.update({'fold': fold, 'arch': arch_name, 'dim': dim_val, 'stage': args.stage, 'steps': args.steps, 'ckpt': ckpt_desc})
                merged_map[(label, arch_name, dim_val, args.stage)] = res
                write_summary_files(out_dir, raw_root, list(merged_map.values()))


if __name__ == '__main__':
    main()
