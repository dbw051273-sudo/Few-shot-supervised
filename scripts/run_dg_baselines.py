"""Classical Domain Generalization Baselines on Unified UniFlow (dim=192) Backbone.

Supports four controlled DG baselines sharing the exact RAE Codec and u0_base Flow backbone:
  - mixstyle : Cross-organ feature statistics mixing (Zhou et al., ICLR 2021)
  - stainaug : Beer-Lambert H&E optical-density domain randomization
  - dann     : Gradient-reversal 3-organ domain adversarial alignment (DomainBed DANN)
  - mldg     : First-order meta-learning across source organs (DomainBed MLDG)

All methods share the identical 4-fold symmetric LOMO splits (`round1_splits.json`),
Source-Val selection, SWAD averaging, 20-shot adaptation (`adapt_flow`), and `v2_unified_eval.py`.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpu_guard import (  # noqa: E402
    acquire_runtime_gpu_guard,
    metric_matches_hash,
    preparse_gpu_and_env,
    stable_hash,
)

_PRE_ARGS = preparse_gpu_and_env('Workstation/Laptop DG Baselines')

import numpy as np  # noqa: E402
from cellpose import models  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from round1_losses_and_data import (  # noqa: E402
    ORGANS,
    adapt_flow,
    apply_od_scale,
    apply_view,
    build_or_load_lomo_splits,
    encode_all,
    infer_to_pngs,
    load_fold_tensors,
    mean_od,
    pi_acq_jitter,
    rgb,
    save_visual_sheet,
    stain_aug_source_he,
    train_fold_codec,
)
from round1_models import Codec, SwadAccumulator, build_flow_model  # noqa: E402
from v2_unified_eval import (  # noqa: E402
    PAPER_CORE_KEYS,
    build_per_image_lpips,
    evaluate_split,
    load_gt_instance_masks,
    sparse_grid_laplacian,
)

DG_METHODS = ('mixstyle', 'stainaug', 'dann', 'mldg')


def resolve_data_dir(explicit: Path | None, work_dir: Path) -> Path:
    candidates = [explicit] if explicit is not None else [
        work_dir.parent / 'UniPrior_v2',
        work_dir / 'UniPrior_v2',
        work_dir.parent / 'UniPrior_四器官洁净均衡集_v2',
        work_dir.parent / '03_洁净数据集与重建包' / 'UniPrior_四器官洁净均衡集_v2',
    ]
    for c in candidates:
        if c is not None and (c / 'Breast' / 'train' / 'input').is_dir() and (c / 'liver' / 'test' / 'input').is_dir():
            return c.resolve()
    raise FileNotFoundError('Cannot locate UniPrior_v2 dataset. Pass --data-dir explicitly.')


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return -ctx.lambd * grad_out, None


def _mixstyle_tokens(feat: torch.Tensor, domain_ids: torch.Tensor, alpha: float = 0.1, p: float = 0.5) -> torch.Tensor:
    """MixStyle across tokens [B, N, C] favoring cross-organ pairs when present."""
    if not feat.requires_grad or random.random() > p or feat.shape[0] < 2:
        return feat
    b = feat.shape[0]
    mu = feat.mean(dim=1, keepdim=True)
    sig = (feat.var(dim=1, unbiased=False, keepdim=True) + 1e-6).sqrt()
    normed = (feat - mu) / sig

    perm = torch.randperm(b, device=feat.device)
    if (domain_ids != domain_ids[0]).any():
        # Prefer cross-domain partner for each sample when batch spans >=2 organs.
        idx_list = []
        for i in range(b):
            diff = torch.nonzero(domain_ids != domain_ids[i], as_tuple=False).view(-1)
            j = int(diff[torch.randint(len(diff), (1,)).item()]) if len(diff) > 0 else int(perm[i].item())
            idx_list.append(j)
        perm = torch.tensor(idx_list, device=feat.device, dtype=torch.long)

    lam = torch.distributions.Beta(alpha, alpha).sample((b, 1, 1)).to(device=feat.device, dtype=feat.dtype)
    mu_mix = lam * mu + (1.0 - lam) * mu[perm]
    sig_mix = lam * sig + (1.0 - lam) * sig[perm]
    return normed * sig_mix + mu_mix


def _flow_forward_dg(
    flow: nn.Module,
    current: torch.Tensor,
    source: torch.Tensor,
    t: torch.Tensor,
    dg_method: str,
    domain_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass through ModularUniFlow(u0_base) exposing pooled hidden state & optional MixStyle."""
    angle = t[:, None] * flow.frequency[None]
    time_emb = flow.time(torch.cat((angle.sin(), angle.cos()), dim=-1))[:, None]
    a = flow.state_proj(current).flatten(2).transpose(1, 2) + flow.position + flow.roles[0] + time_emb
    b = flow.pi_proj(source).flatten(2).transpose(1, 2) + flow.position + flow.roles[1] + time_emb
    a = flow.blocks[0](a)
    b = flow.blocks[1](b)
    if dg_method == 'mixstyle' and domain_ids is not None:
        a = _mixstyle_tokens(a, domain_ids)
        b = _mixstyle_tokens(b, domain_ids)
    hidden = torch.cat((a, b), dim=1)
    n_tok = a.shape[1]
    hidden = flow.blocks[2](hidden)
    hidden = flow.blocks[3](hidden)
    vel = flow._decode_head(hidden[:, :n_tok])
    pooled = hidden[:, :n_tok].mean(dim=1)
    return vel, pooled


def pretrain_flow_dg(
    tag: str,
    dg_method: str,
    flow: nn.Module,
    codec: Codec,
    center: float,
    scale: float,
    src_lat_pi: torch.Tensor,
    src_lat_he: torch.Tensor,
    src_pi: torch.Tensor,
    src_he: torch.Tensor,
    src_groups: list[str],
    val_lat_pi: torch.Tensor,
    val_lat_he: torch.Tensor,
    steps: int = 9600,
    batch: int = 8,
    seed: int = 7102,
) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    rng = torch.Generator(device='cuda').manual_seed(seed)
    cpu_rng = torch.Generator().manual_seed(seed)
    py_rng = random.Random(seed)
    flow.cuda().train()

    org_names = sorted(set(src_groups))
    org_to_id = {o: i for i, o in enumerate(org_names)}
    dom_ids_all = torch.tensor([org_to_id[g] for g in src_groups], dtype=torch.long)
    org_indices = [torch.tensor([i for i, g in enumerate(src_groups) if g == o], dtype=torch.long) for o in org_names]

    domain_clf = None
    params = list(flow.parameters())
    if dg_method == 'dann':
        domain_clf = nn.Sequential(
            nn.Linear(flow.dim, flow.dim),
            nn.SiLU(),
            nn.Linear(flow.dim, len(org_names)),
        ).cuda()
        params = params + list(domain_clf.parameters())

    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)
    swad = SwadAccumulator({'flow': flow}, start_step=max(1, int(0.5 * steps)), every=max(1, min(20, steps // 10)))

    val_pi_c, val_he_c = val_lat_pi.cuda(), val_lat_he.cuda()
    val_t_fixed = torch.linspace(0.1, 0.9, len(val_pi_c), device='cuda')
    best_val, best_step, val_history = 1e9, 0, []
    log_every = max(1, min(900, steps // 4))
    src_he_gpu = src_he.cuda() if dg_method == 'stainaug' else None
    t0 = time.perf_counter()

    def _batch_z(ids: torch.Tensor, v_idx: int, apply_stain_aug: bool = False):
        if v_idx == 0 and not apply_stain_aug:
            return src_lat_pi[ids].cuda(), src_lat_he[ids].cuda()
        x_raw = apply_view(src_pi[ids].cuda(), v_idx)
        y_raw = apply_view(src_he[ids].cuda(), v_idx)
        if apply_stain_aug and src_he_gpu is not None:
            x_raw = pi_acq_jitter(x_raw, py_rng, strength=0.08)
            y_raw = stain_aug_source_he(y_raw, src_he_gpu, py_rng)
        with torch.no_grad():
            z0 = (codec.encode(x_raw).float() - center) / scale
            z1 = (codec.encode(y_raw).float() - center) / scale
        return z0, z1

    for step in range(1, steps + 1):
        v_idx = py_rng.randrange(4)
        opt.zero_grad(set_to_none=True)

        if dg_method == 'mldg' and len(org_indices) >= 2:
            val_dom = py_rng.randrange(len(org_indices))
            tr_doms = [d for d in range(len(org_indices)) if d != val_dom]
            tr_per_dom = max(2, batch // len(tr_doms))
            tr_ids = torch.cat([org_indices[d][torch.randint(len(org_indices[d]), (tr_per_dom,), generator=cpu_rng)] for d in tr_doms])
            va_ids = org_indices[val_dom][torch.randint(len(org_indices[val_dom]), (max(2, batch // 2),), generator=cpu_rng)]

            z0_tr, z1_tr = _batch_z(tr_ids, v_idx)
            t_tr = torch.rand(len(z0_tr), generator=rng, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pv_tr = flow(torch.lerp(z0_tr, z1_tr, t_tr[:, None, None, None]), z0_tr, t_tr, None)
            loss_tr = F.mse_loss(pv_tr.float(), z1_tr - z0_tr)
            loss_tr.backward()
            grads_tr = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in flow.parameters()]

            inner_lr = opt.param_groups[0]['lr']
            inner_flow = copy.deepcopy(flow).train()
            with torch.no_grad():
                for p_in, g_in in zip(inner_flow.parameters(), grads_tr):
                    p_in.add_(g_in, alpha=-inner_lr)

            z0_va, z1_va = _batch_z(va_ids, v_idx)
            t_va = torch.rand(len(z0_va), generator=rng, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pv_va = inner_flow(torch.lerp(z0_va, z1_va, t_va[:, None, None, None]), z0_va, t_va, None)
            loss_va = F.mse_loss(pv_va.float(), z1_va - z0_va)
            loss_va.backward()
            with torch.no_grad():
                for p_main, g_tr, p_in in zip(flow.parameters(), grads_tr, inner_flow.parameters()):
                    g_va = p_in.grad if p_in.grad is not None else torch.zeros_like(p_main)
                    p_main.grad.copy_(g_tr + 0.5 * g_va)
            del inner_flow
            loss = loss_tr + 0.5 * loss_va
        else:
            ids = torch.randint(len(src_pi), (batch,), generator=cpu_rng)
            z0, z1 = _batch_z(ids, v_idx, apply_stain_aug=(dg_method == 'stainaug'))
            dom_b = dom_ids_all[ids].cuda()
            t = torch.rand(batch, generator=rng, device='cuda')
            mixed = torch.lerp(z0, z1, t[:, None, None, None])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred_v, pooled = _flow_forward_dg(flow, mixed, z0, t, dg_method, domain_ids=dom_b)
            loss = F.mse_loss(pred_v.float(), z1 - z0)
            if dg_method == 'dann' and domain_clf is not None:
                p = step / float(max(1, steps))
                grl_alpha = float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
                logits_d = domain_clf(_GradReverse.apply(pooled.float(), grl_alpha))
                loss = loss + 0.10 * F.cross_entropy(logits_d, dom_b)
            loss.backward()

        nn.utils.clip_grad_norm_(flow.parameters(), 1.5)
        opt.step()
        sched.step()
        swad.maybe_update(step, steps)

        if step % log_every == 0 or step == steps:
            flow.eval()
            with torch.no_grad():
                v_losses = []
                for j in range(0, len(val_pi_c), 32):
                    vp, vh, vt = val_pi_c[j:j + 32], val_he_c[j:j + 32], val_t_fixed[j:j + 32]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        vv = flow(torch.lerp(vp, vh, vt[:, None, None, None]), vp, vt, None)
                    v_losses.append(F.mse_loss(vv.float(), vh - vp).item())
                vmse = float(np.mean(v_losses))
            val_history.append({'step': step, 'val_mse': round(vmse, 5)})
            if vmse < best_val:
                best_val, best_step = vmse, step
            print(f'[{tag}] pretrain {step}/{steps}: train={loss.item():.4f}, val_mse={vmse:.5f} ({time.perf_counter()-t0:.1f}s)', flush=True)
            flow.train()

    swad.apply()
    if domain_clf is not None:
        del domain_clf
    return {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}, {
        'best_val_mse': best_val,
        'best_val_step': best_step,
        'val_history': val_history,
        'pretrain_seconds': round(time.perf_counter() - t0, 2),
    }


@torch.no_grad()
def _measure_bs1_latency_ms(flow: nn.Module, codec: Codec, pi_cpu: torch.Tensor, center: float, scale: float, steps: int = 4, n_bench: int = 8) -> float:
    flow.cuda().eval()
    codec.cuda().eval()
    n_run = min(n_bench, len(pi_cpu))
    if n_run <= 0:
        return 0.0

    def _one(bx: torch.Tensor) -> None:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z0 = (codec.encode(bx).float() - center) / scale
        z = z0
        dt = 1.0 / float(steps)
        for k in range(steps):
            t_in = z.new_full((len(z),), k * dt)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v = flow(z, z0, t_in, None)
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


def update_dg_summary_files(eval_dir: Path) -> None:
    metrics_dir = eval_dir / 'dg_metrics'
    records = [json.loads(jp.read_text(encoding='utf-8')) for jp in sorted(metrics_dir.glob('*.json'))]
    (eval_dir / 'dg_summary.json').write_text(
        json.dumps({'updated_at': datetime.now().isoformat(), 'runs': records}, indent=2),
        encoding='utf-8',
    )
    fieldnames = [
        'run_tag', 'fold', 'method', 'dim', 'stage', 'k_shot',
        'psnr', 'ssim', 'lpips', 'kid', 'kid_x1e3', 'nuclei_dice', 'nuclei_f1_05',
        'fid', 'nuclei_dice_nonempty', 'nuclei_f1_05_nonempty', 'empty_gt_count',
        'pcc_gt', 'pcc_pi', 'nuclei_count_r', 'sparse_bg_lap', 'gt_bg_lap',
        'trainable_params_m', 'latency_bs1_ms', 'wall_seconds',
    ]
    with (eval_dir / 'dg_summary.csv').open('w', newline='', encoding='utf-8') as f:
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

    md = [
        '# Classical DG Baselines Summary (UniFlow d192 Backbone)',
        '',
        f'- Updated: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`',
        '- Core Metrics: `PSNR ↑`, `SSIM ↑`, `LPIPS ↓`, `KID (×10⁻³) ↓`, `Cellpose-proxy Dice ↑`, `Cellpose-proxy F1@0.5 ↑`',
        '',
        '| Run Tag | Fold | DG Method | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID (×10⁻³) ↓ | Dice ↑ | F1@0.5 ↑ | bg_lap | gt_lap | Params(M) | ms/img |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in records:
        m = r['metrics']
        md.append(
            f"| `{r['run_tag']}` | {r['fold']} | {r['method']} | {r['stage']} | "
            f"{m['psnr']:.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['kid'] * 1000.0:.2f} | "
            f"{m['nuclei_dice']:.4f} | {m['nuclei_f1_05']:.4f} | "
            f"{r.get('sparse_bg_lap', 0.0):.2f} | {r.get('gt_bg_lap', 0.0):.2f} | "
            f"{r.get('trainable_params_m', 0.0):.2f} | {r.get('latency_bs1_ms', 0.0):.2f} |"
        )
    (eval_dir / 'dg_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def record_dg_eval(
    run_tag: str,
    fold: str,
    method: str,
    dim: int,
    stage: str,
    k_shot: int,
    img_map: dict[str, np.ndarray],
    fd: dict,
    gt_masks: dict,
    cell_model,
    lpips_fn,
    device: torch.device,
    work_dir: Path,
    eval_names: list[str],
    zoom_name: str,
    wall_seconds: float,
    trainable_params_m: float,
    latency_bs1_ms: float,
    config_hash: str,
    split_hash: str,
) -> dict:
    eval_dir = work_dir / 'eval_results'
    (eval_dir / 'per_image').mkdir(parents=True, exist_ok=True)
    (eval_dir / 'dg_metrics').mkdir(parents=True, exist_ok=True)

    res = evaluate_split(
        eval_names, img_map, fd['pi_rgb_map'], fd['gt_rgb_map'], gt_masks,
        cell_model, lpips_fn, device, per_image_csv=eval_dir / 'per_image' / f'{run_tag}.csv',
    )
    bg_lap = round(sparse_grid_laplacian(img_map[zoom_name]), 3)
    gt_lap = round(sparse_grid_laplacian(fd['gt_rgb_map'][zoom_name]), 3)
    payload = {
        'run_tag': run_tag,
        'fold': fold,
        'method': method,
        'dim': dim,
        'stage': stage,
        'k_shot': k_shot,
        'n_test': len(eval_names),
        'config_hash': config_hash,
        'split_hash': split_hash,
        'sparse_bg_lap': bg_lap,
        'gt_bg_lap': gt_lap,
        'trainable_params_m': round(trainable_params_m, 3),
        'latency_bs1_ms': round(latency_bs1_ms, 2),
        'wall_seconds': round(wall_seconds, 2),
        'metrics': {k: float(res[k]) for k in list(PAPER_CORE_KEYS) + [
            'fid', 'nuclei_dice_nonempty', 'nuclei_f1_05_nonempty', 'empty_gt_count', 'pcc_gt', 'pcc_pi', 'nuclei_count_r',
        ]},
    }
    (eval_dir / 'dg_metrics' / f'{run_tag}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    update_dg_summary_files(eval_dir)
    m = payload['metrics']
    print(
        f"[DG_EVAL] {run_tag:34s} | PSNR={m['psnr']:5.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} "
        f"KIDx1e3={m['kid']*1000.0:5.2f} Dice={m['nuclei_dice']:.4f} F1={m['nuclei_f1_05']:.4f} | "
        f"{trainable_params_m:.2f}M {latency_bs1_ms:.2f}ms",
        flush=True,
    )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu-id', type=str, default='2')
    ap.add_argument('--allow-other-gpu', action='store_true', default=False)
    ap.add_argument('--allow-gpu-sharing', action='store_true', default=False)
    ap.add_argument('--online-hf', action='store_true', default=False)
    ap.add_argument('--work-dir', type=Path, default=PACKAGE_ROOT)
    ap.add_argument('--reuse-ckpt-dir', type=Path, default=None, help='Reuse Round-1 Codec checkpoints/<fold>/codec_rae0.08_s8800.pth')
    ap.add_argument('--cache-dir', type=Path, default=None)
    ap.add_argument('--data-dir', type=Path, default=None)
    ap.add_argument('--preset', choices=['smoke', 'dg'], default='dg')
    ap.add_argument('--methods', nargs='+', choices=list(DG_METHODS), default=list(DG_METHODS))
    ap.add_argument('--folds', nargs='+', choices=list(ORGANS), default=['Breast', 'Brain', 'Kidney', 'liver'])
    ap.add_argument('--dim', type=int, default=192)
    ap.add_argument('--codec-noise-max', type=float, default=0.08)
    ap.add_argument('--decoder-lr-mult', type=float, default=0.1)
    ap.add_argument('--rollout-steps', type=int, default=4)
    ap.add_argument('--skip-done', dest='skip_done', action='store_true', default=True)
    ap.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    ap.add_argument('--skip-cellpose', action='store_true', default=False)
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    acquire_runtime_gpu_guard(args.gpu_id, allow_sharing=args.allow_gpu_sharing)
    cache_dir = (args.cache_dir or (work_dir / 'eval_cache' if (work_dir / 'eval_cache').is_dir() else PACKAGE_ROOT / 'eval_cache')).resolve()
    data_dir = resolve_data_dir(args.data_dir, work_dir)
    reuse_dir = args.reuse_ckpt_dir.resolve() if args.reuse_ckpt_dir is not None else None
    eval_dir = work_dir / 'eval_results'
    eval_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72, flush=True)
    print(f'[BOOT] DG Baselines | preset={args.preset} | methods={args.methods} | folds={args.folds} | dim={args.dim}', flush=True)
    print(f'[BOOT] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} | torch_dev={torch.cuda.get_device_name(0)}', flush=True)
    print('=' * 72, flush=True)

    splits = build_or_load_lomo_splits(data_dir, cache_dir, eval_dir / 'round1_splits.json', seed=2026)
    npz_path = cache_dir / 'gt_nuclei_instances_256.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f'Missing GT nuclei mask cache: {npz_path}')

    device = torch.device('cuda:0')
    lpips_fn = build_per_image_lpips(net_type='alex', normalize=True).cuda()
    skip_cp = args.skip_cellpose or os.environ.get('VSTAIN_SKIP_CELLPOSE') == '1'
    cell_model = None if skip_cp else models.CellposeModel(gpu=True, model_type='nuclei')

    run_folds = ['Breast'] if args.preset == 'smoke' and args.folds == ['Breast', 'Brain', 'Kidney', 'liver'] else args.folds
    codec_steps = 8 if args.preset == 'smoke' else 8800
    pre_steps = 8 if args.preset == 'smoke' else 9600
    ad_steps = 6 if args.preset == 'smoke' else 1500
    dim_run = 128 if args.preset == 'smoke' else args.dim

    for fold in run_folds:
        fd = load_fold_tensors(data_dir, splits, fold, k_shot=20)
        eval_names = list(fd['test_ids'])[:8] if args.preset == 'smoke' else list(fd['test_ids'])
        zoom_name = 'file_4278.png' if ('file_4278.png' in fd['infer_ids']) else eval_names[-1]
        infer_subset = eval_names + ([zoom_name] if zoom_name not in eval_names else [])
        infer_idx = [fd['infer_ids'].index(n) for n in infer_subset]
        test_pi_sub = fd['test_pi_t'][infer_idx]
        gt_masks = load_gt_instance_masks(npz_path, fold)
        split_hash = stable_hash({'target': fold, 'organs': splits['organs']})

        ckpt_fold = work_dir / 'checkpoints' / 'dg' / fold
        ckpt_fold.mkdir(parents=True, exist_ok=True)
        codec_name = f'codec_rae{args.codec_noise_max:g}_s{codec_steps}.pth'
        codec = Codec().cuda()
        reused_codec = None
        for cand in [ckpt_fold / codec_name] + (
            [reuse_dir / 'checkpoints' / fold / codec_name, reuse_dir / fold / codec_name] if reuse_dir is not None else []
        ):
            if cand.exists():
                reused_codec = cand
                break
        if reused_codec is not None:
            meta_p = reused_codec.with_name(f'{reused_codec.name}.meta.json')
            if not meta_p.exists():
                raise RuntimeError(f'Missing codec split metadata: {meta_p}')
            meta = json.loads(meta_p.read_text(encoding='utf-8'))
            if meta.get('split_hash') != split_hash:
                raise RuntimeError(f'Codec split mismatch on {reused_codec}')
            codec.load_state_dict(torch.load(reused_codec, map_location='cuda', weights_only=True))
            codec.eval().requires_grad_(False)
            print(f'[REUSE] Codec: {reused_codec}', flush=True)
        else:
            codec, _ = train_fold_codec(fd['src_pi'], fd['src_he'], steps=codec_steps, seed=7000, detail_weight=0.15, noise_max=args.codec_noise_max)
            torch.save(codec.state_dict(), ckpt_fold / codec_name)
            (ckpt_fold / f'{codec_name}.meta.json').write_text(
                json.dumps({'split_hash': split_hash, 'codec_noise_max': args.codec_noise_max, 'codec_steps': codec_steps}, indent=2),
                encoding='utf-8',
            )

        with torch.no_grad():
            lat_pi = encode_all(codec, fd['src_pi'])
            center, scale = float(lat_pi.mean()), float(lat_pi.std())
            src_lat_pi = (lat_pi - center) / scale
            src_lat_he = (encode_all(codec, fd['src_he']) - center) / scale
            val_lat_pi = (encode_all(codec, fd['val_pi']) - center) / scale
            val_lat_he = (encode_all(codec, fd['val_he']) - center) / scale

        organ_pools = [
            (fd['src_pi'][[i for i, g in enumerate(fd['src_groups']) if g == o]].cuda(),
             fd['src_he'][[i for i, g in enumerate(fd['src_groups']) if g == o]].cuda())
            for o in sorted(set(fd['src_groups']))
        ]

        for method in args.methods:
            base_tag = f'{fold}__{method}_d{dim_run}'
            tag_0 = f'{base_tag}__0shot'
            tag_od = f'{base_tag}__0shot_halfOD'
            tag_20 = f'{base_tag}__m6_k20'
            cfg_hash = stable_hash({
                'fold': fold, 'method': method, 'dim': dim_run, 'split_hash': split_hash,
                'codec_name': codec_name, 'pre_steps': pre_steps, 'ad_steps': ad_steps,
                'decoder_lr_mult': args.decoder_lr_mult, 'rollout_steps': args.rollout_steps, 'preset': args.preset,
            })
            if args.skip_done and all(metric_matches_hash(eval_dir / 'dg_metrics' / f'{t}.json', cfg_hash) for t in (tag_0, tag_od, tag_20)):
                print(f'[SKIP] {base_tag} already evaluated (verified config_hash={cfg_hash}).', flush=True)
                continue

            t_pre = time.perf_counter()
            fl = build_flow_model('u0_base', dim=dim_run)
            flow_params_m = sum(p.numel() for p in fl.parameters()) / 1e6
            dec_params_m = sum(p.numel() for p in codec.decoder.parameters()) / 1e6
            swad_st, _ = pretrain_flow_dg(
                base_tag, method, fl, codec, center, scale,
                src_lat_pi, src_lat_he, fd['src_pi'], fd['src_he'], fd['src_groups'],
                val_lat_pi, val_lat_he, steps=pre_steps, batch=4 if args.preset == 'smoke' else 8,
            )
            pre_s = time.perf_counter() - t_pre
            torch.save({'swad': swad_st, 'center': center, 'scale': scale, 'split_hash': split_hash, 'config_hash': cfg_hash}, ckpt_fold / f'{base_tag}_pretrain.pth')
            del fl

            fl0 = build_flow_model('u0_base', dim=dim_run)
            fl0.load_state_dict(swad_st)
            lat0 = _measure_bs1_latency_ms(fl0, codec, test_pi_sub, center, scale, steps=args.rollout_steps)
            t_inf = time.perf_counter()
            map_0 = infer_to_pngs(fl0, codec, test_pi_sub, None, center, scale, infer_subset, work_dir / 'generated' / fold / tag_0, steps=args.rollout_steps)
            record_dg_eval(
                tag_0, fold, method, dim_run, '0shot', 0, map_0, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=pre_s + (time.perf_counter() - t_inf),
                trainable_params_m=flow_params_m, latency_bs1_ms=lat0, config_hash=cfg_hash, split_hash=split_hash,
            )

            t_od = time.perf_counter()
            refs_0 = infer_to_pngs(fl0, codec, fd['refs_pi_t'], None, center, scale, fd['refs_ids'], work_dir / 'generated' / fold / f'_tmp_refs_{base_tag}', steps=args.rollout_steps)
            try:
                sc = fd['refs_od'] / np.maximum(mean_od([refs_0[n] for n in fd['refs_ids']], tag_od), 1e-4)
            except ValueError as exc:
                print(f'[WARN] {tag_od}: {exc}; falling back to identity OD scale=1.0', flush=True)
                sc = np.ones(2, dtype=np.float64)
            od_dir = work_dir / 'generated' / fold / tag_od
            od_dir.mkdir(parents=True, exist_ok=True)
            map_od = {}
            for nm in infer_subset:
                im = np.rint((map_0[nm].astype(np.float32) + apply_od_scale(map_0[nm], sc)) / 2.0).clip(0, 255).astype(np.uint8)
                map_od[nm] = im
                Image.fromarray(im).save(od_dir / nm)
            record_dg_eval(
                tag_od, fold, method, dim_run, '0shot_halfOD', 5, map_od, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=time.perf_counter() - t_od,
                trainable_params_m=flow_params_m, latency_bs1_ms=lat0, config_hash=cfg_hash, split_hash=split_hash,
            )
            del fl0
            torch.cuda.empty_cache()

            t_ad = time.perf_counter()
            fl_ad, co_ad, _ = adapt_flow(
                tag_20, 'm6', build_flow_model('u0_base', dim=dim_run), swad_st, codec, center, scale,
                organ_pools, fd['adapt_pi_t'], fd['adapt_he_t'], lpips_fn, None,
                steps=ad_steps, batch=2 if args.preset == 'smoke' else 4,
                decoder_lr_mult=args.decoder_lr_mult, rollout_steps=args.rollout_steps,
            )
            torch.save({'flow': fl_ad.state_dict(), 'codec': co_ad.state_dict(), 'center': center, 'scale': scale, 'config_hash': cfg_hash}, ckpt_fold / f'{tag_20}.pth')
            lat20 = _measure_bs1_latency_ms(fl_ad, co_ad, test_pi_sub, center, scale, steps=args.rollout_steps)
            map_20 = infer_to_pngs(fl_ad, co_ad, test_pi_sub, None, center, scale, infer_subset, work_dir / 'generated' / fold / tag_20, steps=args.rollout_steps)
            ad_params_m = flow_params_m + (dec_params_m if args.decoder_lr_mult > 0.0 else 0.0)
            record_dg_eval(
                tag_20, fold, method, dim_run, '20shot', 20, map_20, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=time.perf_counter() - t_ad,
                trainable_params_m=ad_params_m, latency_bs1_ms=lat20, config_hash=cfg_hash, split_hash=split_hash,
            )

            spread_idx = [0, len(eval_names) // 3, (2 * len(eval_names)) // 3, len(eval_names) - 1]
            show_ids = [eval_names[i] for i in spread_idx]
            save_visual_sheet(
                work_dir / 'visuals' / fold / f'{base_tag}.png',
                show_ids, fd['pi_rgb_map'], fd['gt_rgb_map'],
                {'0shot': map_0, '0shot_halfOD': map_od, '20shot': map_20},
                zoom_name,
            )
            print(f'[VISUAL] Saved visuals/{fold}/{base_tag}.png', flush=True)
            del fl_ad, co_ad
            gc.collect()
            torch.cuda.empty_cache()

    print(f'\n[COMPLETE] DG preset={args.preset} finished cleanly on GPU {args.gpu_id}.', flush=True)


if __name__ == '__main__':
    main()
