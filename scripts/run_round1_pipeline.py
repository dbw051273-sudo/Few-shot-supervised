"""End-to-end serialized Round-1 runner strictly locked to Workstation GPU 2:
For each fold/arm: Train -> Infer PNGs -> Unified Eval (6 core + supp + CSV) -> Visuals -> Update Summary -> Next -> Auto-Pack.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpu_guard import (  # noqa: E402
    acquire_runtime_gpu_guard,
    metric_matches_hash,
    preparse_gpu_and_env,
    stable_hash,
)

_PRE_ARGS = preparse_gpu_and_env('Workstation Round-1')

import numpy as np  # noqa: E402
from cellpose import models  # noqa: E402
import torch  # noqa: E402
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: E402

from pack_round1_outputs import build_archive  # noqa: E402
from round1_losses_and_data import (  # noqa: E402
    adapt_flow,
    apply_od_scale,
    build_or_load_lomo_splits,
    encode_all,
    load_fold_tensors,
    mean_od,
    precompute_uni_4views,
    pretrain_flow,
    save_visual_sheet,
    train_fold_codec,
    infer_to_pngs,
)
from round1_models import Codec, build_flow_model  # noqa: E402
from uni_encoder import UNIEncoder  # noqa: E402
from v2_unified_eval import PAPER_CORE_KEYS, build_per_image_lpips, evaluate_split, load_gt_instance_masks, sparse_grid_laplacian  # noqa: E402


def resolve_data_dir(explicit: Path | None, work_dir: Path) -> Path:
    candidates = [explicit] if explicit is not None else [
        work_dir.parent / 'UniPrior_v2',
        work_dir / 'UniPrior_v2',
        work_dir.parent / 'UniPrior_四器官洁净均衡集_v2',
    ]
    for c in candidates:
        if c is not None and (c / 'Breast' / 'train' / 'input').is_dir() and (c / 'liver' / 'test' / 'input').is_dir():
            return c.resolve()
    raise FileNotFoundError(
        'Cannot locate UniPrior_v2 dataset (expected sibling ../UniPrior_v2 with Brain/Breast/Kidney/liver). Pass --data-dir explicitly.'
    )


def _gpu_physical_info(gpu_id: str) -> str:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
            text=True,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return ' | '.join(lines)
    except Exception:
        return 'nvidia-smi unavailable'


def update_summary_files(eval_dir: Path) -> None:
    """Aggregate all eval_results/metrics/*.json into CSV, JSON, and Markdown tables."""
    metrics_dir = eval_dir / 'metrics'
    records = []
    for jp in sorted(metrics_dir.glob('*.json')):
        records.append(json.loads(jp.read_text(encoding='utf-8')))

    summary_json = eval_dir / 'round1_summary.json'
    summary_json.write_text(json.dumps({'updated_at': datetime.now().isoformat(), 'runs': records}, indent=2), encoding='utf-8')

    fieldnames = [
        'run_tag', 'fold', 'arch', 'dim', 'stage', 'strategy', 'k_shot',
        'psnr', 'ssim', 'lpips', 'kid', 'kid_x1e3', 'nuclei_dice', 'nuclei_f1_05',
        'fid', 'nuclei_dice_nonempty', 'nuclei_f1_05_nonempty', 'empty_gt_count',
        'pcc_gt', 'pcc_pi', 'nuclei_count_r', 'sparse_bg_lap', 'gt_bg_lap',
        'trainable_params_m', 'latency_bs1_ms', 'wall_seconds',
    ]
    summary_csv = eval_dir / 'round1_summary.csv'
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            m = r['metrics']
            row = {k: r.get(k, '') for k in fieldnames}
            for mk in (
                'psnr', 'ssim', 'lpips', 'kid', 'nuclei_dice', 'nuclei_f1_05',
                'fid', 'nuclei_dice_nonempty', 'nuclei_f1_05_nonempty', 'empty_gt_count',
                'pcc_gt', 'pcc_pi', 'nuclei_count_r',
            ):
                row[mk] = round(m[mk], 6) if isinstance(m.get(mk), float) else m.get(mk, '')
            row['kid_x1e3'] = round(m['kid'] * 1000.0, 3) if isinstance(m.get('kid'), float) else ''
            w.writerow(row)

    md_lines = [
        '# Round-1 Serialized Evaluation Summary',
        '',
        f'- Updated: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`',
        '- Core Metrics: `PSNR ↑`, `SSIM ↑`, `LPIPS ↓`, `KID (×10⁻³) ↓`, `Cellpose-proxy Dice ↑`, `Cellpose-proxy F1@0.5 (1:1) ↑`',
        '- Note: `KID` values are scaled by `10^3` (`subset_size=50`). `FID` depends on test size `N` and is omitted from main/cross-`N` tables (retained in JSON/CSV only for same-`N` supplementary diagnostics).',
        '',
        '| Run Tag | Fold | Arch | Dim | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID (×10⁻³) ↓ | Dice ↑ | F1@0.5 ↑ | bg_lap | gt_lap | Params(M) | ms/img |',
        '| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in records:
        m = r['metrics']
        pm = r.get('trainable_params_m', '')
        ms = r.get('latency_bs1_ms', '')
        pm_s = f'{pm:.2f}' if isinstance(pm, (int, float)) else ''
        ms_s = f'{ms:.2f}' if isinstance(ms, (int, float)) else ''
        md_lines.append(
            f"| `{r['run_tag']}` | {r['fold']} | {r['arch']} | {r['dim']} | {r['stage']} | "
            f"{m['psnr']:.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['kid'] * 1000.0:.2f} | "
            f"{m['nuclei_dice']:.4f} | {m['nuclei_f1_05']:.4f} | "
            f"{r.get('sparse_bg_lap', 0.0):.2f} | {r.get('gt_bg_lap', 0.0):.2f} | {pm_s} | {ms_s} |"
        )
    (eval_dir / 'round1_summary.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')


def make_schedule(
    preset: str,
    archs_filter: list[str] | None = None,
    folds_filter: list[str] | None = None,
    adapt_only_k20: bool = False,
) -> list[dict]:
    """Define exact serialized Round-1 job schedule (u1_up64_only locked as primary Ours)."""
    if preset == 'smoke':
        jobs = [{
            'fold': 'Breast', 'arch': 'u1_up64_only', 'dim': 128,
            'pretrain_steps': 8, 'adapt_steps': 6, 'codec_steps': 8,
            'batch_pre': 4, 'batch_ad': 2, 'adapt_jobs': [('m6', 20)], 'eval_limit': 8,
        }]
    elif preset == 'screen':
        jobs = [
            {'fold': 'Breast', 'arch': 'u1_up64_only', 'dim': 128, 'pretrain_steps': 3600, 'adapt_steps': 1000, 'codec_steps': 8800, 'batch_pre': 8, 'batch_ad': 4, 'adapt_jobs': [('m6', 20), ('naive_ft', 20)], 'eval_limit': None},
            {'fold': 'Breast', 'arch': 'u1_uni2d_1x1', 'dim': 128, 'pretrain_steps': 3600, 'adapt_steps': 1000, 'codec_steps': 8800, 'batch_pre': 8, 'batch_ad': 4, 'adapt_jobs': [('m6', 20)], 'eval_limit': None},
            {'fold': 'Breast', 'arch': 'u0_base', 'dim': 128, 'pretrain_steps': 3600, 'adapt_steps': 1000, 'codec_steps': 8800, 'batch_pre': 8, 'batch_ad': 4, 'adapt_jobs': [('m6', 20)], 'eval_limit': None},
        ]
    else:
        # Default 'round1': Streamlined 4-fold symmetric LOMO (dim=192) locking u1_up64_only vs u0_base (+ Breast 1x1 & k-shot/NaiveFT ablations)
        jobs = []
        for fold in ('Breast', 'Brain', 'Kidney', 'liver'):
            archs = ('u1_up64_only', 'u1_uni2d_1x1', 'u0_base') if fold == 'Breast' else ('u1_up64_only', 'u0_base')
            for arch in archs:
                ad_jobs = [('m6', 20)]
                if fold == 'Breast' and arch == 'u1_up64_only':
                    ad_jobs = [('m6', 20), ('naive_ft', 20), ('m6', 10), ('m6', 5)]
                jobs.append({
                    'fold': fold, 'arch': arch, 'dim': 192,
                    'pretrain_steps': 9600, 'adapt_steps': 1500, 'codec_steps': 8800,
                    'batch_pre': 8, 'batch_ad': 4, 'adapt_jobs': ad_jobs, 'eval_limit': None,
                })

    if folds_filter:
        fset = set(folds_filter)
        jobs = [j for j in jobs if j['fold'] in fset]
    if archs_filter:
        aset = set(archs_filter)
        jobs = [j for j in jobs if j['arch'] in aset]
    if adapt_only_k20:
        for j in jobs:
            j['adapt_jobs'] = [('m6', 20)]
    return jobs


def _find_reusable_ckpt(
    ckpt_fold: Path,
    reuse_dir: Path | None,
    fold: str,
    filename: str,
    allow_local: bool = True,
) -> Path | None:
    if allow_local:
        local_p = ckpt_fold / filename
        if local_p.exists():
            return local_p
    if reuse_dir is not None:
        for cand in (reuse_dir / 'checkpoints' / fold / filename, reuse_dir / fold / filename):
            if cand.exists():
                return cand
    return None


@torch.no_grad()
def measure_bs1_latency_ms(
    flow: torch.nn.Module,
    codec: Codec,
    pi_cpu: torch.Tensor,
    uni_enc: UNIEncoder | None,
    center: float,
    scale: float,
    steps: int = 4,
    n_bench: int = 8,
) -> float:
    flow.cuda().eval()
    codec.cuda().eval()
    use_uni = bool(getattr(flow, 'use_uni_2d', False)) and uni_enc is not None
    n_run = min(n_bench, len(pi_cpu))
    if n_run <= 0:
        return 0.0

    def _one(bx: torch.Tensor) -> None:
        uni_tok = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if use_uni:
                uni_tok = uni_enc(bx)[:, 1:].float()
            z0 = (codec.encode(bx).float() - center) / scale
        z = z0
        dt = 1.0 / float(steps)
        for k in range(steps):
            t_in = z.new_full((len(z),), k * dt)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v = flow(z, z0, t_in, uni_tok)
            z = z + v.float() * dt
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _ = codec.decode(z * scale + center)

    _one(pi_cpu[:1].cuda())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_run):
        _one(pi_cpu[i:i + 1].cuda())
    torch.cuda.synchronize()
    return round(((time.perf_counter() - t0) * 1000.0) / float(n_run), 2)


def eval_and_record(
    run_tag: str,
    fold: str,
    arch: str,
    dim: int,
    stage: str,
    strategy: str,
    k_shot: int,
    img_map: dict[str, np.ndarray],
    fold_data: dict,
    gt_masks: dict[str, np.ndarray],
    cell_model,
    lpips_fn,
    device: torch.device,
    work_dir: Path,
    eval_names: list[str],
    zoom_name: str,
    elapsed_s: float,
    trainable_params_m: float = 0.0,
    latency_bs1_ms: float = 0.0,
    config_hash: str = '',
    split_hash: str = '',
) -> dict:
    eval_dir = work_dir / 'eval_results'
    csv_path = eval_dir / 'per_image' / f'{run_tag}.csv'
    json_path = eval_dir / 'metrics' / f'{run_tag}.json'
    metrics = evaluate_split(
        eval_names,
        img_map,
        fold_data['pi_rgb_map'],
        fold_data['gt_rgb_map'],
        gt_masks,
        cell_model,
        lpips_fn,
        device,
        per_image_csv=csv_path,
    )
    bg_lap = sparse_grid_laplacian(img_map[zoom_name])
    gt_lap = sparse_grid_laplacian(fold_data['gt_rgb_map'][zoom_name])
    payload = {
        'run_tag': run_tag,
        'fold': fold,
        'arch': arch,
        'dim': dim,
        'stage': stage,
        'strategy': strategy,
        'k_shot': k_shot,
        'eval_n': len(eval_names),
        'config_hash': config_hash,
        'split_hash': split_hash,
        'sparse_bg_lap': round(bg_lap, 4),
        'gt_bg_lap': round(gt_lap, 4),
        'trainable_params_m': round(trainable_params_m, 4),
        'latency_bs1_ms': round(latency_bs1_ms, 2),
        'wall_seconds': round(elapsed_s, 2),
        'metrics': metrics,
        'paper_core_keys': list(PAPER_CORE_KEYS),
        'timestamp': datetime.now().isoformat(),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    update_summary_files(eval_dir)
    print(
        f"[DONE+EVAL] {run_tag:42s} | PSNR={metrics['psnr']:5.2f} SSIM={metrics['ssim']:.4f} "
        f"LPIPS={metrics['lpips']:.4f} KIDx1e3={metrics['kid']*1000.0:5.2f} Dice={metrics['nuclei_dice']:.4f} "
        f"F1={metrics['nuclei_f1_05']:.4f} | bg_lap={bg_lap:.2f} | {trainable_params_m:.2f}M {latency_bs1_ms:.1f}ms | ({elapsed_s:.1f}s)",
        flush=True,
    )
    return payload


def main() -> None:
    default_work = SCRIPT_DIR.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu-id', type=str, default='2', help='Physical CUDA GPU index (default: 2)')
    ap.add_argument('--allow-other-gpu', action='store_true', default=False)
    ap.add_argument('--allow-gpu-sharing', action='store_true', default=False)
    ap.add_argument('--online-hf', action='store_true', default=False)
    ap.add_argument('--work-dir', type=Path, default=default_work)
    ap.add_argument('--data-dir', type=Path, default=None)
    ap.add_argument('--cache-dir', type=Path, default=None)
    ap.add_argument('--preset', choices=('smoke', 'screen', 'round1'), default='round1')
    ap.add_argument('--folds', nargs='+', default=None, help='Optional organ fold filter (e.g. Breast Brain Kidney liver)')
    ap.add_argument('--archs', nargs='+', default=None, help='Optional arch filter (e.g. u1_up64_only)')
    ap.add_argument('--adapt-only-k20', action='store_true', default=False, help='Skip Breast extra k5/k10/naive_ft runs and run only m6 k20')
    ap.add_argument('--skip-0shot', action='store_true', default=False, help='Skip 0shot/0shot_halfOD eval (for decoder LR ablations reusing pretrain)')
    ap.add_argument('--reuse-ckpt-dir', type=Path, default=None, help='Optional main work_dir whose checkpoints/<fold>/ codec/pretrain are reused')
    ap.add_argument('--skip-done', action='store_true', default=True)
    ap.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    ap.add_argument('--skip-cellpose', action='store_true', default=False, help='Skip Cellpose during train/gen (or set VSTAIN_SKIP_CELLPOSE=1)')
    ap.add_argument('--pack-every-fold', action='store_true', default=True)
    ap.add_argument('--codec-noise-max', type=float, default=0.08, help='RAE Codec latent noise max fraction of batch std (default: 0.08)')
    ap.add_argument('--decoder-lr-mult', type=float, default=0.1, help='Codec decoder LR multiplier vs Flow in adapt (default: 0.1)')
    ap.add_argument('--rollout-steps', type=int, default=4, help='Locked Euler NFE steps for inference and adapt rollout (default: 4)')
    ap.add_argument('--use_on_policy_rollout', action='store_true', default=False, help='Ablation switch: add N-step on-policy rollout loss in pretrain second half')
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    acquire_runtime_gpu_guard(args.gpu_id, allow_sharing=args.allow_gpu_sharing)
    data_dir = resolve_data_dir(args.data_dir, work_dir)
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.resolve()
    elif (work_dir / 'eval_cache' / 'gt_nuclei_instances_256.npz').exists():
        cache_dir = (work_dir / 'eval_cache').resolve()
    else:
        cache_dir = (SCRIPT_DIR.parent / 'eval_cache').resolve()
    reuse_dir = args.reuse_ckpt_dir.resolve() if args.reuse_ckpt_dir is not None else None
    npz_path = cache_dir / 'gt_nuclei_instances_256.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f'Missing GT instance cache: {npz_path}')

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')

    gpu_info = f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | visible_device_0={torch.cuda.get_device_name(0)}"
    print(f'[INIT] {platform.platform()} | {gpu_info}', flush=True)
    print(f'[INIT] WORK_DIR={work_dir}', flush=True)
    print(f'[INIT] DATA_DIR={data_dir}', flush=True)

    eval_dir = work_dir / 'eval_results'
    eval_dir.mkdir(parents=True, exist_ok=True)
    splits = build_or_load_lomo_splits(data_dir, cache_dir, eval_dir / 'round1_splits.json', seed=2026)

    manifest = {
        'started_at': datetime.now().isoformat(),
        'preset': args.preset,
        'gpu_id_requested': args.gpu_id,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_device_name': torch.cuda.get_device_name(0),
        'nvidia_smi': _gpu_physical_info(args.gpu_id),
        'codec_noise_max': args.codec_noise_max,
        'decoder_lr_mult': args.decoder_lr_mult,
        'rollout_steps': args.rollout_steps,
        'use_on_policy_rollout': args.use_on_policy_rollout,
        'work_dir': str(work_dir),
        'data_dir': str(data_dir),
    }
    (eval_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    schedule = make_schedule(args.preset, args.archs, args.folds, args.adapt_only_k20)
    uni_enc = UNIEncoder(freeze=True).cuda().eval()
    lpips_fn = build_per_image_lpips(net_type='alex', normalize=True).cuda()
    skip_cp = args.skip_cellpose or os.environ.get('VSTAIN_SKIP_CELLPOSE') == '1'
    cell_model = None if skip_cp else models.CellposeModel(gpu=True, model_type='nuclei')

    fold_cache_name = None
    fold_ctx = None
    seen_folds = set()

    for idx, job in enumerate(schedule, start=1):
        fold = job['fold']
        arch = job['arch']
        dim = job['dim']
        base_tag = f"{fold}__{arch}_d{dim}"
        split_hash = stable_hash({'target': fold, 'organs': splits['organs']})
        codec_name = (
            f"codec_rae{args.codec_noise_max:.2f}_s{job['codec_steps']}.pth"
            if args.codec_noise_max > 0.0
            else f"codec_s{job['codec_steps']}.pth"
        )
        pre_fp_dict = {
            'fold': fold,
            'arch': arch,
            'dim': dim,
            'split_hash': split_hash,
            'codec_name': codec_name,
            'codec_noise_max': float(args.codec_noise_max),
            'pretrain_steps': int(job['pretrain_steps']),
            'batch_pre': int(job['batch_pre']),
            'use_on_policy_rollout': bool(args.use_on_policy_rollout),
            'rollout_steps_pre': int(args.rollout_steps) if args.use_on_policy_rollout else 0,
        }
        pre_hash = stable_hash(pre_fp_dict)
        zero_hash = stable_hash({**pre_fp_dict, 'rollout_steps_eval': int(args.rollout_steps), 'eval_limit': job['eval_limit']})

        def _adapt_hash(strat: str, k_shot: int) -> str:
            return stable_hash({
                **pre_fp_dict,
                'strategy': strat,
                'k_shot': int(k_shot),
                'decoder_lr_mult': float(args.decoder_lr_mult),
                'rollout_steps': int(args.rollout_steps),
                'adapt_steps': int(job['adapt_steps']),
                'batch_ad': int(job['batch_ad']),
                'eval_limit': job['eval_limit'],
            })

        print(f"\n========== [{idx}/{len(schedule)}] Job: {base_tag} ==========", flush=True)

        # Skip entire job only if every expected metric JSON exists AND matches exact config_hash
        expected_checks = (
            [] if args.skip_0shot else [(f"{base_tag}__0shot", zero_hash), (f"{base_tag}__0shot_halfOD", zero_hash)]
        ) + [(f"{base_tag}__{strat}_k{k}", _adapt_hash(strat, k)) for strat, k in job['adapt_jobs']]
        if args.skip_done and all(metric_matches_hash(eval_dir / 'metrics' / f'{t}.json', h) for t, h in expected_checks):
            print(f"[SKIP] All {len(expected_checks)} verified metric files match config_hash for {base_tag}", flush=True)
            continue

        # Load fold tensors + codec + UNI caches once per fold
        if fold_cache_name != fold:
            fold_ctx = None
            gc.collect()
            torch.cuda.empty_cache()

            fd20 = load_fold_tensors(data_dir, splits, fold, k_shot=20)
            src_pi, src_he = fd20['src_pi'], fd20['src_he']
            if job['eval_limit'] is not None:
                lim = job['eval_limit']
                eval_names = fd20['test_ids'][:lim]
            else:
                eval_names = fd20['test_ids']

            ckpt_fold = work_dir / 'checkpoints' / fold
            ckpt_fold.mkdir(parents=True, exist_ok=True)
            codec_path = ckpt_fold / codec_name
            reused_codec = _find_reusable_ckpt(ckpt_fold, reuse_dir, fold, codec_name, allow_local=args.skip_done)
            codec = Codec().cuda()
            if reused_codec is not None:
                cmeta_p = reused_codec.with_name(f'{reused_codec.name}.meta.json')
                if not cmeta_p.exists():
                    raise RuntimeError(
                        f"Codec reuse missing split metadata ({cmeta_p}). "
                        "Legacy/untagged codecs are blocked; train cleanly in a fresh --work-dir."
                    )
                cmeta = json.loads(cmeta_p.read_text(encoding='utf-8'))
                if cmeta.get('split_hash') != split_hash:
                    raise RuntimeError(f"Codec split mismatch on {reused_codec}: {cmeta.get('split_hash')} != {split_hash}")
                codec.load_state_dict(torch.load(reused_codec, map_location='cuda', weights_only=True))
                codec.eval().requires_grad_(False)
                print(f"[REUSE] Codec: {reused_codec}", flush=True)
            else:
                codec, cinfo = train_fold_codec(
                    src_pi, src_he, steps=job['codec_steps'], seed=7000,
                    detail_weight=0.15, noise_max=args.codec_noise_max,
                )
                torch.save(codec.state_dict(), codec_path)
                codec_path.with_name(f'{codec_name}.meta.json').write_text(
                    json.dumps({'split_hash': split_hash, 'codec_noise_max': args.codec_noise_max, 'codec_steps': job['codec_steps']}, indent=2),
                    encoding='utf-8',
                )
                print(f"[CODEC] {fold} ({codec_name}): {cinfo}", flush=True)

            with torch.no_grad():
                lat_pi = encode_all(codec, src_pi)
                center, scale = float(lat_pi.mean()), float(lat_pi.std())
                src_lat_pi = (lat_pi - center) / scale
                src_lat_he = (encode_all(codec, src_he) - center) / scale
                val_lat_pi = (encode_all(codec, fd20['val_pi']) - center) / scale
                val_lat_he = (encode_all(codec, fd20['val_he']) - center) / scale

            src_uni_4v = precompute_uni_4views(uni_enc, src_pi)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                val_uni_v0 = torch.cat(
                    [uni_enc(fd20['val_pi'][i:i + 32].cuda())[:, 1:].half().cpu() for i in range(0, len(fd20['val_pi']), 32)],
                    dim=0,
                )

            organ_pools = [
                (src_pi[[i for i, g in enumerate(fd20['src_groups']) if g == o]].cuda(),
                 src_he[[i for i, g in enumerate(fd20['src_groups']) if g == o]].cuda())
                for o in sorted(set(fd20['src_groups']))
            ]
            gt_masks = load_gt_instance_masks(npz_path, fold)
            zoom_name = 'file_4278.png' if ('file_4278.png' in fd20['infer_ids']) else eval_names[-1]
            fold_ctx = {
                'fd': fd20,
                'eval_names': eval_names,
                'zoom_name': zoom_name,
                'codec': codec,
                'codec_name': codec_name,
                'center': center,
                'scale': scale,
                'src_lat_pi': src_lat_pi,
                'src_lat_he': src_lat_he,
                'val_lat_pi': val_lat_pi,
                'val_lat_he': val_lat_he,
                'src_uni_4v': src_uni_4v,
                'val_uni_v0': val_uni_v0,
                'organ_pools': organ_pools,
                'gt_masks': gt_masks,
            }
            fold_cache_name = fold

        fd = fold_ctx['fd']
        eval_names = fold_ctx['eval_names']
        zoom_name = fold_ctx['zoom_name']
        infer_subset = eval_names + ([zoom_name] if zoom_name not in eval_names else [])
        infer_idx = [fd['infer_ids'].index(n) for n in infer_subset]
        test_pi_sub = fd['test_pi_t'][infer_idx]
        codec = fold_ctx['codec']
        center, scale = fold_ctx['center'], fold_ctx['scale']
        gt_masks = fold_ctx['gt_masks']

        ckpt_fold = work_dir / 'checkpoints' / fold
        pre_filename = f'{base_tag}_pretrain.pth'
        ckpt_pre = ckpt_fold / pre_filename
        pre_t0 = time.perf_counter()
        reused_pre = (
            None
            if args.use_on_policy_rollout
            else _find_reusable_ckpt(ckpt_fold, reuse_dir, fold, pre_filename, allow_local=args.skip_done)
        )
        if reused_pre is None and args.skip_done and ckpt_pre.exists():
            reused_pre = ckpt_pre
        if reused_pre is not None:
            pre_obj = torch.load(reused_pre, map_location='cpu', weights_only=True)
            if pre_obj.get('codec_name') != fold_ctx['codec_name']:
                raise RuntimeError(
                    f"Pretrain reuse codec mismatch on {reused_pre}: ckpt={pre_obj.get('codec_name')} vs current={fold_ctx['codec_name']}. "
                    "Use a clean --work-dir or delete stale checkpoints."
                )
            if 'center' not in pre_obj or 'scale' not in pre_obj or (
                abs(float(pre_obj['center']) - center) > 1e-3 or abs(float(pre_obj['scale']) - scale) > 1e-3
            ):
                raise RuntimeError(
                    f"Pretrain reuse latent stats missing/mismatch on {reused_pre}: "
                    f"ckpt=({pre_obj.get('center')}, {pre_obj.get('scale')}) vs current=({center:.5f}, {scale:.5f})."
                )
            if pre_obj.get('split_hash') != split_hash:
                raise RuntimeError(
                    f"Pretrain reuse split_hash missing/mismatch on {reused_pre}: ckpt={pre_obj.get('split_hash')} vs {split_hash}."
                )
            if pre_obj.get('pretrain_hash') != pre_hash:
                raise RuntimeError(
                    f"Pretrain reuse config mismatch on {reused_pre}: ckpt={pre_obj.get('pretrain_hash')} vs current={pre_hash}. "
                    "Use a dedicated --work-dir or --no-skip-done."
                )
            swad_st = pre_obj['swad']
            pre_s = 0.0
            print(f"[REUSE] Pretrain checkpoint (verified): {reused_pre}", flush=True)
        else:
            ckpt_pre.parent.mkdir(parents=True, exist_ok=True)
            fl_pre = build_flow_model(arch, dim=dim)
            use_uni_arm = getattr(fl_pre, 'use_uni_2d', False)
            swad_st, pinfo = pretrain_flow(
                base_tag,
                fl_pre,
                codec,
                center,
                scale,
                fold_ctx['src_lat_pi'],
                fold_ctx['src_lat_he'],
                fd['src_pi'],
                fd['src_he'],
                fold_ctx['src_uni_4v'] if use_uni_arm else None,
                fold_ctx['val_lat_pi'],
                fold_ctx['val_lat_he'],
                fold_ctx['val_uni_v0'] if use_uni_arm else None,
                steps=job['pretrain_steps'],
                batch=job['batch_pre'],
                seed=7102,
                use_on_policy_rollout=args.use_on_policy_rollout,
                rollout_steps=args.rollout_steps,
                lpips_fn=lpips_fn,
            )
            torch.save(
                {
                    'swad': swad_st,
                    'pinfo': pinfo,
                    'center': center,
                    'scale': scale,
                    'codec_name': fold_ctx['codec_name'],
                    'split_hash': split_hash,
                    'pretrain_hash': pre_hash,
                    'pretrain_fp': pre_fp_dict,
                },
                ckpt_pre,
            )
            pre_s = time.perf_counter() - pre_t0
            del fl_pre

        visual_maps: dict[str, dict[str, np.ndarray]] = {}
        flow_proto = build_flow_model(arch, dim=dim)
        flow_params_m = sum(p.numel() for p in flow_proto.parameters()) / 1e6
        dec_params_m = sum(p.numel() for p in codec.decoder.parameters()) / 1e6
        del flow_proto

        # Stage 1: 0-shot -> infer PNGs -> immediately evaluate
        tag_0 = f'{base_tag}__0shot'
        zs_map = None
        lat0_ms = 0.0
        if not args.skip_0shot and not (args.skip_done and metric_matches_hash(eval_dir / 'metrics' / f'{tag_0}.json', zero_hash)):
            fl0 = build_flow_model(arch, dim=dim)
            fl0.load_state_dict(swad_st)
            lat0_ms = measure_bs1_latency_ms(fl0, codec, test_pi_sub, uni_enc, center, scale, steps=args.rollout_steps)
            t_inf = time.perf_counter()
            zs_map = infer_to_pngs(
                fl0, codec, test_pi_sub, uni_enc, center, scale, infer_subset,
                work_dir / 'generated' / fold / tag_0, steps=args.rollout_steps,
            )
            eval_and_record(
                tag_0, fold, arch, dim, '0shot', 'pretrain', 0, zs_map, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                elapsed_s=pre_s + (time.perf_counter() - t_inf),
                trainable_params_m=flow_params_m, latency_bs1_ms=lat0_ms,
                config_hash=zero_hash, split_hash=split_hash,
            )
            visual_maps['0shot'] = zs_map
            del fl0
            torch.cuda.empty_cache()

        # Stage 2: 0-shot + 5-ref halfOD -> save PNGs -> immediately evaluate
        tag_od = f'{base_tag}__0shot_halfOD'
        if not args.skip_0shot and not (args.skip_done and metric_matches_hash(eval_dir / 'metrics' / f'{tag_od}.json', zero_hash)):
            fl0 = build_flow_model(arch, dim=dim)
            fl0.load_state_dict(swad_st)
            if lat0_ms <= 0.0:
                lat0_ms = measure_bs1_latency_ms(fl0, codec, test_pi_sub, uni_enc, center, scale, steps=args.rollout_steps)
            t_od = time.perf_counter()
            if zs_map is None:
                zs_map = infer_to_pngs(
                    fl0, codec, test_pi_sub, uni_enc, center, scale, infer_subset,
                    work_dir / 'generated' / fold / tag_0, steps=args.rollout_steps,
                )
            zs_refs = infer_to_pngs(
                fl0, codec, fd['refs_pi_t'], uni_enc, center, scale, fd['refs_ids'],
                work_dir / 'generated' / fold / f'_tmp_refs_{base_tag}', steps=args.rollout_steps,
            )
            sc = fd['refs_od'] / mean_od([zs_refs[n] for n in fd['refs_ids']], tag_od)
            od_dir = work_dir / 'generated' / fold / tag_od
            od_dir.mkdir(parents=True, exist_ok=True)
            half_map = {}
            for nm in infer_subset:
                im = np.rint((zs_map[nm].astype(np.float32) + apply_od_scale(zs_map[nm], sc)) / 2.0).clip(0, 255).astype(np.uint8)
                half_map[nm] = im
                from PIL import Image
                Image.fromarray(im).save(od_dir / nm)
            eval_and_record(
                tag_od, fold, arch, dim, '0shot_halfOD', 'halfOD', 5, half_map, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                elapsed_s=time.perf_counter() - t_od,
                trainable_params_m=flow_params_m, latency_bs1_ms=lat0_ms,
                config_hash=zero_hash, split_hash=split_hash,
            )
            visual_maps['0shot_halfOD'] = half_map
            del fl0
            torch.cuda.empty_cache()

        # Stage 3+: k-shot supervised adaptations -> infer PNGs -> immediately evaluate
        ad_trainable_m = flow_params_m + (dec_params_m if args.decoder_lr_mult > 0.0 else 0.0)
        for strat, k_shot in job['adapt_jobs']:
            tag_ad = f"{base_tag}__{strat}_k{k_shot}"
            ad_hash = _adapt_hash(strat, k_shot)
            if args.skip_done and metric_matches_hash(eval_dir / 'metrics' / f'{tag_ad}.json', ad_hash):
                continue
            fd_k = fd if k_shot == 20 else load_fold_tensors(data_dir, splits, fold, k_shot=k_shot)
            t_ad = time.perf_counter()
            fl_ad, co_ad, _ad_info = adapt_flow(
                tag_ad, strat, build_flow_model(arch, dim=dim), swad_st, codec, center, scale,
                fold_ctx['organ_pools'], fd_k['adapt_pi_t'], fd_k['adapt_he_t'],
                lpips_fn, uni_enc, steps=job['adapt_steps'], batch=job['batch_ad'], seed=7202,
                decoder_lr_mult=args.decoder_lr_mult, rollout_steps=args.rollout_steps,
            )
            torch.save(
                {
                    'flow': fl_ad.state_dict(),
                    'codec': co_ad.state_dict(),
                    'center': center,
                    'scale': scale,
                    'config_hash': ad_hash,
                    'split_hash': split_hash,
                },
                work_dir / 'checkpoints' / fold / f'{tag_ad}.pth',
            )
            lat_ad_ms = measure_bs1_latency_ms(fl_ad, co_ad, test_pi_sub, uni_enc, center, scale, steps=args.rollout_steps)
            ad_map = infer_to_pngs(
                fl_ad, co_ad, test_pi_sub, uni_enc, center, scale, infer_subset,
                work_dir / 'generated' / fold / tag_ad, steps=args.rollout_steps,
            )
            eval_and_record(
                tag_ad, fold, arch, dim, f'{k_shot}shot', strat, k_shot, ad_map, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                elapsed_s=time.perf_counter() - t_ad,
                trainable_params_m=ad_trainable_m, latency_bs1_ms=lat_ad_ms,
                config_hash=ad_hash, split_hash=split_hash,
            )
            visual_maps[f'{strat}_k{k_shot}'] = ad_map
            del fl_ad, co_ad
            gc.collect()
            torch.cuda.empty_cache()

        if visual_maps:
            spread_idx = [0, len(eval_names) // 3, (2 * len(eval_names)) // 3, len(eval_names) - 1]
            show_ids = [eval_names[i] for i in spread_idx]
            need_ids = show_ids + ([zoom_name] if zoom_name not in show_ids else [])
            ordered_cols = [('0shot', tag_0), ('0shot_halfOD', tag_od)] + [
                (f'{strat}_k{k_shot}', f'{base_tag}__{strat}_k{k_shot}') for strat, k_shot in job['adapt_jobs']
            ]
            from PIL import Image
            full_visuals: dict[str, dict[str, np.ndarray]] = {}
            for col_name, run_t in ordered_cols:
                if col_name in visual_maps:
                    full_visuals[col_name] = visual_maps[col_name]
                else:
                    gdir = work_dir / 'generated' / fold / run_t
                    if all((gdir / nm).exists() for nm in need_ids):
                        full_visuals[col_name] = {nm: np.asarray(Image.open(gdir / nm).convert('RGB'), dtype=np.uint8) for nm in need_ids}
            save_visual_sheet(
                work_dir / 'visuals' / fold / f'{base_tag}.png',
                show_ids,
                fd['pi_rgb_map'],
                fd['gt_rgb_map'],
                full_visuals,
                zoom_name,
            )
            print(f"[VISUAL] Saved visuals/{fold}/{base_tag}.png", flush=True)

        seen_folds.add(fold)
        is_last_of_fold = (idx == len(schedule)) or (schedule[idx]['fold'] != fold)
        if args.pack_every_fold and is_last_of_fold and args.preset != 'smoke':
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            build_archive(work_dir, work_dir / 'exports' / f'{args.preset}_{fold}_checkpoint_eval_{ts}.zip', include_generated=False)

    # Final pack at end of full run
    ts_final = datetime.now().strftime('%Y%m%d_%H%M%S')
    build_archive(work_dir, work_dir / 'exports' / f'{args.preset}_FINAL_metrics_visuals_{ts_final}.zip', include_generated=False)
    build_archive(work_dir, work_dir / 'exports' / f'{args.preset}_FINAL_with_images_{ts_final}.zip', include_generated=True)
    print(f"\n[COMPLETE] Preset={args.preset} finished cleanly on GPU {args.gpu_id}.", flush=True)


if __name__ == '__main__':
    main()
