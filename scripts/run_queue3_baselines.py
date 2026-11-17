"""Queue-3 External SOTA Baselines Runner: CUT (ECCV 2020) and I2SB (ICML 2023).

Uses verbatim official upstream GitHub modules from `third_party_official/`:
  - taesungp/contrastive-unpaired-translation: CUTModel (ResNet-9 + PatchNCE)
  - NVlabs/I2SB: guided_diffusion UNet (cond_x1) + i2sb.diffusion.Diffusion

Protocol alignment with Round-1 & Queue-2:
  - Identical 4-fold LOMO zero-adjacency splits (`round1_splits.json`, 480 train / 80 val / 120 test)
  - Second-half LR decay + Source-Val (`240 pairs`) best-ckpt selection & early stopping
  - Matched 20-shot adaptation on target organ `adapt_20` (zero test leakage)
  - Unified evaluation via `v2_unified_eval.py` (KID subset_size=50 scaled by 1e3; Trainable Params & bs=1 ms/img)
  - Supports `--skip-cellpose` / `VSTAIN_SKIP_CELLPOSE=1` and `--reeval-only`
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
THIRD_PARTY_DIR = PACKAGE_ROOT / 'third_party_official'
CUT_REPO = THIRD_PARTY_DIR / 'CUT'
I2SB_REPO = THIRD_PARTY_DIR / 'I2SB'

# Ensure CUT's `models` / `options` / `util` resolve without colliding with Queue-2 repos.
for p in (SCRIPT_DIR, CUT_REPO, I2SB_REPO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from gpu_guard import (  # noqa: E402
    acquire_runtime_gpu_guard,
    metric_matches_hash,
    preparse_gpu_and_env,
    stable_hash,
)

_PRE_ARGS = preparse_gpu_and_env('Workstation/Laptop Queue-3')

import numpy as np  # noqa: E402
from cellpose import models  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: E402

from round1_losses_and_data import (  # noqa: E402
    ORGANS,
    apply_od_scale,
    build_or_load_lomo_splits,
    load_fold_tensors,
    mean_od,
    rgb,
    save_visual_sheet,
)
from v2_unified_eval import (  # noqa: E402
    PAPER_CORE_KEYS,
    build_per_image_lpips,
    evaluate_split,
    load_gt_instance_masks,
    sparse_grid_laplacian,
)

from models.cut_model import CUTModel  # noqa: E402  # official taesungp/contrastive-unpaired-translation


def _load_i2sb_modules():
    """Import official I2SB UNet and Diffusion cleanly without optional wandb/ipdb loggers."""
    for mod_name in ('ipdb', 'wandb', 'prefetch_generator', 'tensorboard', 'torch.utils.tensorboard'):
        sys.modules.setdefault(
            mod_name,
            types.SimpleNamespace(
                set_trace=lambda *a, **k: None,
                SummaryWriter=object,
                BackgroundGenerator=lambda x: x,
            ),
        )
    if 'i2sb' not in sys.modules:
        pkg = types.ModuleType('i2sb')
        pkg.__path__ = [str(I2SB_REPO / 'i2sb')]
        sys.modules['i2sb'] = pkg
    if 'i2sb.util' not in sys.modules:
        util_mod = types.ModuleType('i2sb.util')
        util_mod.unsqueeze_xdim = lambda z, xdim: z[(...,) + (None,) * len(xdim)]

        def _space_indices(num_steps: int, count: int) -> list[int]:
            assert count <= num_steps
            stride = 1.0 if count <= 1 else (num_steps - 1) / float(count - 1)
            return [round(i * stride) for i in range(count)]

        util_mod.space_indices = _space_indices
        sys.modules['i2sb.util'] = util_mod

    diffusion_mod = importlib.import_module('i2sb.diffusion')
    script_util_mod = importlib.import_module('guided_diffusion.script_util')
    return diffusion_mod.Diffusion, script_util_mod.create_model, sys.modules['i2sb.util'].space_indices


I2SBDiffusion, i2sb_create_model, i2sb_space_indices = _load_i2sb_modules()

UPSTREAM_REPOS = {
    'cut': 'taesungp/contrastive-unpaired-translation',
    'i2sb': 'NVlabs/I2SB',
}


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _count_params_m(module: nn.Module) -> float:
    return round(sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6, 3)


def _clone_cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _gpu_physical_info() -> str:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
            text=True,
            timeout=5,
        )
        return ' | '.join(ln.strip() for ln in out.splitlines() if ln.strip())
    except Exception:
        return 'nvidia-smi unavailable'


# ---------------------------------------------------------------------------
# Official CUT Builder & Train / Adapt Loops
# ---------------------------------------------------------------------------

def _make_cut_opt(
    is_train: bool,
    checkpoints_dir: Path,
    name: str,
    max_epochs: int,
    batch_size: int,
    lr: float,
    gpu_ids: list[int],
) -> SimpleNamespace:
    n_ep = max(1, max_epochs // 2)
    n_decay = max(1, max_epochs - n_ep)
    return SimpleNamespace(
        isTrain=is_train,
        gpu_ids=gpu_ids,
        checkpoints_dir=str(checkpoints_dir),
        name=name,
        preprocess='none',
        verbose=False,
        continue_train=False,
        load_iter=0,
        epoch='latest',
        init_type='xavier',
        init_gain=0.02,
        lr_policy='linear',
        epoch_count=1,
        n_epochs=n_ep,
        n_epochs_decay=n_decay,
        lr_decay_iters=50,
        input_nc=3,
        output_nc=3,
        ngf=64,
        ndf=64,
        netD='basic',
        netG='resnet_9blocks',
        n_layers_D=3,
        normG='instance',
        normD='instance',
        no_dropout=True,
        no_antialias=False,
        no_antialias_up=False,
        gan_mode='lsgan',
        pool_size=0,
        direction='AtoB',
        beta1=0.5,
        beta2=0.999,
        lr=lr,
        batch_size=batch_size,
        stylegan2_G_num_downsampling=1,
        CUT_mode='CUT',
        lambda_GAN=1.0,
        lambda_NCE=1.0,
        nce_idt=True,
        nce_layers='0,4,8,12,16',
        nce_includes_all_negatives_from_minibatch=False,
        netF='mlp_sample',
        netF_nc=256,
        nce_T=0.07,
        num_patches=256,
        flip_equivariance=False,
        pretrained_name=None,
    )


def build_cut_model(
    max_epochs: int,
    batch_size: int,
    lr: float,
    ckpt_dir: Path,
    name: str,
    is_train: bool = True,
    gpu_ids: list[int] | None = None,
) -> tuple[CUTModel, SimpleNamespace]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gids = [0] if (gpu_ids is None and torch.cuda.is_available()) else (gpu_ids or [])
    opt = _make_cut_opt(is_train, ckpt_dir, name, max_epochs, batch_size, lr, gids)
    model = CUTModel(opt)
    return model, opt


@torch.no_grad()
def _eval_source_val_score(
    predict_fn,
    val_pi: torch.Tensor,
    val_he: torch.Tensor,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    batch: int = 8,
) -> tuple[float, float, float]:
    l1_list, lp_list = [], []
    for i in range(0, len(val_pi), batch):
        x = val_pi[i:i + batch].cuda()
        y = val_he[i:i + batch].cuda()
        pred = predict_fn(x).clamp(-1.0, 1.0)
        l1_list.append(F.l1_loss(pred, y, reduction='none').mean(dim=(1, 2, 3)).cpu())
        lp_list.append(lpips_fn((pred + 1.0) * 0.5, (y + 1.0) * 0.5).view(-1).cpu())
    l1_m = float(torch.cat(l1_list).mean().item())
    lp_m = float(torch.cat(lp_list).mean().item())
    return l1_m + lp_m, l1_m, lp_m


def _train_cut_pretrain(
    fold: str,
    src_pi: torch.Tensor,
    src_he: torch.Tensor,
    val_pi: torch.Tensor,
    val_he: torch.Tensor,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    ckpt_dir: Path,
    max_epochs: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    patience: int,
    max_batches_per_epoch: int | None = None,
    seed: int = 2026,
) -> tuple[dict[str, torch.Tensor], dict]:
    _set_seed(seed)
    tag = f'{fold}__cut'
    model, opt = build_cut_model(max_epochs, batch_size, lr, ckpt_dir, tag, is_train=True)
    n_src = len(src_pi)
    initialized = False
    best_score, best_epoch = 1e9, 0
    best_gen_state = _clone_cpu_state(model.netG)
    best_full_state: dict[str, dict[str, torch.Tensor]] = {}
    no_improve = 0
    val_hist = []
    t0 = time.perf_counter()

    for ep in range(1, max_epochs + 1):
        model.netG.train()
        perm_a = torch.randperm(n_src)
        perm_b = torch.randperm(n_src)
        n_steps = 0
        for s in range(0, n_src, batch_size):
            idx_a = perm_a[s:s + batch_size]
            idx_b = perm_b[s:s + batch_size]
            # CUT PatchNCELoss reshapes using opt.batch_size strictly.
            if len(idx_a) != batch_size:
                continue
            batch_data = {
                'A': src_pi[idx_a],
                'B': src_he[idx_b],
                'A_paths': [''] * batch_size,
                'B_paths': [''] * batch_size,
            }
            if not initialized:
                model.data_dependent_initialize(batch_data)
                model.setup(opt)
                initialized = True
                best_full_state = {nm: _clone_cpu_state(getattr(model, 'net' + nm)) for nm in model.model_names}
            model.set_input(batch_data)
            model.optimize_parameters()
            n_steps += 1
            if max_batches_per_epoch is not None and n_steps >= max_batches_per_epoch:
                break
        model.update_learning_rate()

        if ep % eval_every == 0 or ep == max_epochs:
            model.netG.eval()
            val_score, val_l1, val_lp = _eval_source_val_score(
                lambda x: model.netG(x), val_pi, val_he, lpips_fn, batch=16,
            )
            model.netG.train()
            val_hist.append({'epoch': ep, 'val_score': round(val_score, 5), 'val_l1': round(val_l1, 5), 'val_lpips': round(val_lp, 5)})
            if val_score < best_score:
                best_score, best_epoch = val_score, ep
                best_gen_state = _clone_cpu_state(model.netG)
                best_full_state = {nm: _clone_cpu_state(getattr(model, 'net' + nm)) for nm in model.model_names}
                no_improve = 0
            elif ep >= max(1, max_epochs // 2):
                no_improve += 1
            print(
                f'[{tag}] pretrain ep {ep}/{max_epochs} | val_score={val_score:.4f} '
                f'(L1={val_l1:.4f}, LPIPS={val_lp:.4f}, best_ep={best_epoch}) ({time.perf_counter()-t0:.1f}s)',
                flush=True,
            )
            if patience > 0 and ep >= max(1, max_epochs // 2) and no_improve >= patience:
                print(f'[{tag}] Source-Val early stop at ep {ep} (best_ep={best_epoch})', flush=True)
                break

    params_m = _count_params_m(model.netG)
    total_train_m = round(sum(_count_params_m(getattr(model, 'net' + nm)) for nm in model.model_names), 3)
    del model
    torch.cuda.empty_cache()
    return best_gen_state, {
        'best_epoch': best_epoch,
        'best_val_score': best_score,
        'val_history': val_hist,
        'trainable_params_m': params_m,
        'total_train_params_m': total_train_m,
        'wall_seconds': round(time.perf_counter() - t0, 2),
        'full_model_state': best_full_state,
    }


def _adapt_cut_k20(
    fold: str,
    full_model_state: dict[str, dict[str, torch.Tensor]],
    adapt_pi: torch.Tensor,
    adapt_he: torch.Tensor,
    ckpt_dir: Path,
    adapt_steps: int,
    batch_size: int,
    base_lr: float,
    seed: int = 7202,
) -> tuple[dict[str, torch.Tensor], float]:
    _set_seed(seed)
    bs = min(batch_size, len(adapt_pi))
    lr_ad = base_lr * 0.25
    approx_epochs = max(2, adapt_steps // max(1, len(adapt_pi) // bs))
    model, opt = build_cut_model(approx_epochs, bs, lr_ad, ckpt_dir, f'{fold}__cut__20shot', is_train=True)

    # Materialize netF MLPs first, then load pretrained G/F/D states before setup.
    init_batch = {'A': adapt_pi[:bs], 'B': adapt_he[:bs], 'A_paths': [''] * bs, 'B_paths': [''] * bs}
    model.data_dependent_initialize(init_batch)
    for nm, st in full_model_state.items():
        if hasattr(model, 'net' + nm):
            getattr(model, 'net' + nm).load_state_dict(st)
    model.setup(opt)

    model.netG.train()
    n_ad = len(adapt_pi)
    t0 = time.perf_counter()
    for _step in range(1, adapt_steps + 1):
        idx_a = torch.randperm(n_ad)[:bs]
        idx_b = torch.randperm(n_ad)[:bs]
        model.set_input({'A': adapt_pi[idx_a], 'B': adapt_he[idx_b], 'A_paths': [''] * bs, 'B_paths': [''] * bs})
        model.optimize_parameters()
    out_state = _clone_cpu_state(model.netG)
    elapsed = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return out_state, elapsed


# ---------------------------------------------------------------------------
# Official I2SB Builder, Training, Sampling & 20-shot Adaptation
# ---------------------------------------------------------------------------

def _make_beta_schedule(n_timestep: int = 1000, linear_start: float = 1e-4, linear_end: float = 2e-2) -> np.ndarray:
    betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


def build_i2sb_unet(num_channels: int = 128) -> nn.Module:
    """Official guided_diffusion ADM UNet with cond_x1 concatenated input ([xt, x1], 6ch)."""
    return i2sb_create_model(
        image_size=256,
        num_channels=num_channels,
        num_res_blocks=2,
        channel_mult='',
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions='32,16,8',
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        in_channels=6,
    )


class _I2SBBridgeHelper:
    """Wraps official i2sb.diffusion.Diffusion schedule and ddpm_sampling."""

    def __init__(self, device: torch.device, interval: int = 1000, t0: float = 1e-4, T: float = 1.0, beta_max: float = 0.3):
        betas = _make_beta_schedule(n_timestep=interval, linear_end=beta_max / interval)
        betas = np.concatenate([betas[:interval // 2], np.flip(betas[:interval // 2])])
        self.interval = interval
        self.device = device
        self.diffusion = I2SBDiffusion(betas, device)
        self.noise_levels = torch.linspace(t0, T, interval, device=device) * interval

    def train_loss(self, net: nn.Module, x0_he: torch.Tensor, x1_pi: torch.Tensor, cpu_rng: torch.Generator) -> torch.Tensor:
        step = torch.randint(0, self.interval, (len(x0_he),), generator=cpu_rng)
        xt = self.diffusion.q_sample(step, x0_he, x1_pi, ot_ode=False)
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0_he.shape[1:])
        label = (xt - x0_he) / std_fwd
        t_in = self.noise_levels[step]
        # guided_diffusion AttentionBlock uses custom gradient checkpointing incompatible with autocast
        pred = net(torch.cat([xt, x1_pi], dim=1), t_in)
        return F.mse_loss(pred, label)

    @torch.no_grad()
    def sample(self, net: nn.Module, x1_pi: torch.Tensor, nfe: int = 20, seed: int = 2026) -> torch.Tensor:
        net.eval()
        _set_seed(seed)
        steps = i2sb_space_indices(self.interval, nfe + 1)

        def _pred_x0(xt: torch.Tensor, step_int: int) -> torch.Tensor:
            st = torch.full((xt.shape[0],), step_int, device=xt.device, dtype=torch.long)
            t_in = self.noise_levels[st]
            out = net(torch.cat([xt, x1_pi], dim=1), t_in)
            std_fwd = self.diffusion.get_std_fwd(st, xdim=xt.shape[1:])
            return (xt - std_fwd * out).clamp(-1.0, 1.0)

        xs, _ = self.diffusion.ddpm_sampling(
            steps, _pred_x0, x1_pi, mask=None, ot_ode=False, log_steps=[0], verbose=False,
        )
        return xs[:, 0].to(self.device).clamp(-1.0, 1.0)


def _update_ema(ema_state: dict[str, torch.Tensor], module: nn.Module, decay: float = 0.999) -> None:
    with torch.no_grad():
        for k, v in module.state_dict().items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(decay).add_(v.detach().cpu(), alpha=1.0 - decay)
            else:
                ema_state[k].copy_(v.detach().cpu())


def _train_i2sb_pretrain(
    fold: str,
    src_pi: torch.Tensor,
    src_he: torch.Tensor,
    val_pi: torch.Tensor,
    val_he: torch.Tensor,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    max_epochs: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    patience: int,
    nfe: int = 20,
    max_batches_per_epoch: int | None = None,
    seed: int = 2026,
) -> tuple[dict[str, torch.Tensor], dict]:
    _set_seed(seed)
    tag = f'{fold}__i2sb'
    device = torch.device('cuda:0')
    bridge = _I2SBBridgeHelper(device)
    net = build_i2sb_unet().cuda().train()
    ema_state = _clone_cpu_state(net)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    total_steps_est = max(1, max_epochs * (max_batches_per_epoch or max(1, len(src_pi) // batch_size)))

    n_src = len(src_pi)
    cpu_rng = torch.Generator().manual_seed(seed)
    best_score, best_epoch = 1e9, 0
    best_ema = {k: v.clone() for k, v in ema_state.items()}
    val_sub_idx = torch.linspace(0, len(val_pi) - 1, min(32, len(val_pi))).long()
    val_sub_pi, val_sub_he = val_pi[val_sub_idx], val_he[val_sub_idx]

    no_improve = 0
    val_hist = []
    global_step = 0
    t0 = time.perf_counter()

    for ep in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(n_src)
        n_steps = 0
        for s in range(0, n_src, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:
                continue
            global_step += 1
            if total_steps_est > 2 and global_step > total_steps_est // 2:
                prog = (global_step - total_steps_est // 2) / float(max(1, total_steps_est - total_steps_est // 2))
                lr_cur = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))
                for pg in opt.param_groups:
                    pg['lr'] = lr_cur

            x0 = src_he[idx].cuda()
            x1 = src_pi[idx].cuda()
            loss = bridge.train_loss(net, x0, x1, cpu_rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            _update_ema(ema_state, net, decay=0.999)
            n_steps += 1
            if max_batches_per_epoch is not None and n_steps >= max_batches_per_epoch:
                break

        if ep % eval_every == 0 or ep == max_epochs:
            net_eval = build_i2sb_unet().cuda().eval()
            net_eval.load_state_dict(ema_state)
            val_score, val_l1, val_lp = _eval_source_val_score(
                lambda x: bridge.sample(net_eval, x, nfe=nfe), val_sub_pi, val_sub_he, lpips_fn, batch=8,
            )
            del net_eval
            val_hist.append({'epoch': ep, 'val_score': round(val_score, 5), 'val_l1': round(val_l1, 5), 'val_lpips': round(val_lp, 5)})
            if val_score < best_score:
                best_score, best_epoch = val_score, ep
                best_ema = {k: v.clone() for k, v in ema_state.items()}
                no_improve = 0
            elif ep >= max(1, max_epochs // 2):
                no_improve += 1
            print(
                f'[{tag}] pretrain ep {ep}/{max_epochs} | val_score={val_score:.4f} '
                f'(L1={val_l1:.4f}, LPIPS={val_lp:.4f}, best_ep={best_epoch}) ({time.perf_counter()-t0:.1f}s)',
                flush=True,
            )
            if patience > 0 and ep >= max(1, max_epochs // 2) and no_improve >= patience:
                print(f'[{tag}] Source-Val early stop at ep {ep} (best_ep={best_epoch})', flush=True)
                break

    params_m = _count_params_m(net)
    del net
    torch.cuda.empty_cache()
    return best_ema, {
        'best_epoch': best_epoch,
        'best_val_score': best_score,
        'val_history': val_hist,
        'trainable_params_m': params_m,
        'total_train_params_m': params_m,
        'wall_seconds': round(time.perf_counter() - t0, 2),
    }


def _adapt_i2sb_k20(
    pretrain_state: dict[str, torch.Tensor],
    adapt_pi: torch.Tensor,
    adapt_he: torch.Tensor,
    adapt_steps: int,
    batch_size: int,
    base_lr: float,
    seed: int = 7202,
) -> tuple[dict[str, torch.Tensor], float]:
    _set_seed(seed)
    device = torch.device('cuda:0')
    bridge = _I2SBBridgeHelper(device)
    net = build_i2sb_unet().cuda().train()
    net.load_state_dict(pretrain_state)
    ema_state = _clone_cpu_state(net)
    opt = torch.optim.AdamW(net.parameters(), lr=base_lr * 0.25, weight_decay=0.0)
    cpu_rng = torch.Generator().manual_seed(seed)
    n_ad = len(adapt_pi)
    bs = min(batch_size, n_ad)
    t0 = time.perf_counter()
    for _step in range(1, adapt_steps + 1):
        idx = torch.randperm(n_ad)[:bs]
        loss = bridge.train_loss(net, adapt_he[idx].cuda(), adapt_pi[idx].cuda(), cpu_rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        _update_ema(ema_state, net, decay=0.98)
    elapsed = time.perf_counter() - t0
    del net
    torch.cuda.empty_cache()
    return ema_state, elapsed


# ---------------------------------------------------------------------------
# Inference, Latency, Summaries & Evaluation
# ---------------------------------------------------------------------------

def _make_predict_fn(method: str, net_module: nn.Module, i2sb_nfe: int = 20):
    net_module.eval()
    if method == 'cut':
        return lambda x_256: net_module(x_256)
    bridge = _I2SBBridgeHelper(torch.device('cuda:0'))
    return lambda x_256: bridge.sample(net_module, x_256, nfe=i2sb_nfe)


@torch.no_grad()
def _measure_bs1_latency_ms(predict_fn, pi_cpu: torch.Tensor, n_bench: int = 8) -> float:
    n_run = min(n_bench, len(pi_cpu))
    if n_run <= 0:
        return 0.0
    _ = predict_fn(pi_cpu[:1].cuda())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_run):
        _ = predict_fn(pi_cpu[i:i + 1].cuda())
    torch.cuda.synchronize()
    return round(((time.perf_counter() - t0) * 1000.0) / float(n_run), 2)


@torch.no_grad()
def _infer_to_pngs(predict_fn, pi_tensor: torch.Tensor, names: list[str], out_dir: Path, batch: int = 8) -> dict[str, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_map: dict[str, np.ndarray] = {}
    for i in range(0, len(pi_tensor), batch):
        pred = predict_fn(pi_tensor[i:i + batch].cuda()).clamp(-1.0, 1.0)
        arr = np.rint(((pred.permute(0, 2, 3, 1).cpu().numpy() + 1.0) * 127.5)).clip(0, 255).astype(np.uint8)
        for j, nm in enumerate(names[i:i + batch]):
            pred_map[nm] = arr[j]
            Image.fromarray(arr[j]).save(out_dir / nm)
    return pred_map


def update_queue3_summary_files(eval_dir: Path) -> None:
    metrics_dir = eval_dir / 'queue3_metrics'
    records = [json.loads(jp.read_text(encoding='utf-8')) for jp in sorted(metrics_dir.glob('*.json'))]
    (eval_dir / 'queue3_summary.json').write_text(
        json.dumps({'updated_at': datetime.now().isoformat(), 'runs': records}, indent=2),
        encoding='utf-8',
    )
    fieldnames = [
        'run_tag', 'fold', 'method', 'upstream_repo', 'stage', 'k_shot',
        'psnr', 'ssim', 'lpips', 'kid', 'kid_x1e3', 'nuclei_dice', 'nuclei_f1_05',
        'fid', 'nuclei_dice_nonempty', 'nuclei_f1_05_nonempty', 'empty_gt_count',
        'pcc_gt', 'pcc_pi', 'nuclei_count_r', 'sparse_bg_lap', 'gt_bg_lap',
        'trainable_params_m', 'total_train_params_m', 'latency_bs1_ms', 'best_val_epoch', 'wall_seconds',
    ]
    with (eval_dir / 'queue3_summary.csv').open('w', newline='', encoding='utf-8') as f:
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
        '# Queue-3 External SOTA Baselines Summary (CUT & I2SB)',
        '',
        f'- Updated: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`',
        '- Official Sources: `CUT` (`taesungp/contrastive-unpaired-translation`), `I2SB` (`NVlabs/I2SB`)',
        '- Note: `KID` scaled by `10^3` (`subset_size=50`). `FID` omitted from cross-`N` main tables.',
        '',
        '| Run Tag | Fold | Method | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID (×10⁻³) ↓ | Dice ↑ | F1@0.5 ↑ | bg_lap | gt_lap | Params(M) | ms/img |',
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
    (eval_dir / 'queue3_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def record_queue3_eval(
    run_tag: str,
    fold: str,
    method: str,
    upstream_repo: str,
    stage: str,
    k_shot: int,
    pred_map: dict[str, np.ndarray],
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
    total_train_params_m: float,
    latency_bs1_ms: float,
    best_val_epoch: int,
    config_hash: str = '',
    split_hash: str = '',
) -> dict:
    eval_dir = work_dir / 'eval_results'
    (eval_dir / 'per_image').mkdir(parents=True, exist_ok=True)
    (eval_dir / 'queue3_metrics').mkdir(parents=True, exist_ok=True)

    res = evaluate_split(
        eval_names, pred_map, fd['pi_rgb_map'], fd['gt_rgb_map'], gt_masks,
        cell_model, lpips_fn, device, per_image_csv=eval_dir / 'per_image' / f'{run_tag}.csv',
    )
    sparse_lap = round(sparse_grid_laplacian(pred_map[zoom_name]), 3)
    gt_lap = round(sparse_grid_laplacian(fd['gt_rgb_map'][zoom_name]), 3)
    payload = {
        'run_tag': run_tag,
        'fold': fold,
        'method': method,
        'upstream_repo': upstream_repo,
        'stage': stage,
        'k_shot': k_shot,
        'n_test': len(eval_names),
        'config_hash': config_hash,
        'split_hash': split_hash,
        'sparse_bg_lap': sparse_lap,
        'gt_bg_lap': gt_lap,
        'trainable_params_m': round(trainable_params_m, 3),
        'total_train_params_m': round(total_train_params_m, 3),
        'latency_bs1_ms': round(latency_bs1_ms, 2),
        'best_val_epoch': best_val_epoch,
        'wall_seconds': round(wall_seconds, 2),
        'metrics': {k: float(res[k]) for k in list(PAPER_CORE_KEYS) + [
            'fid', 'nuclei_dice_nonempty', 'nuclei_f1_05_nonempty', 'empty_gt_count', 'pcc_gt', 'pcc_pi', 'nuclei_count_r',
        ]},
    }
    (eval_dir / 'queue3_metrics' / f'{run_tag}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    update_queue3_summary_files(eval_dir)
    m = payload['metrics']
    print(
        f"[Q3_EVAL] {run_tag:28s} | PSNR={m['psnr']:5.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} "
        f"KIDx1e3={m['kid']*1000.0:5.2f} Dice={m['nuclei_dice']:.4f} F1={m['nuclei_f1_05']:.4f} | "
        f"Params={trainable_params_m:.2f}M {latency_bs1_ms:.2f}ms",
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
    ap.add_argument('--cache-dir', type=Path, default=None)
    ap.add_argument('--data-dir', type=Path, default=None)
    ap.add_argument('--preset', choices=['smoke', 'queue3'], default='queue3')
    ap.add_argument('--methods', nargs='+', choices=['cut', 'i2sb'], default=['cut', 'i2sb'])
    ap.add_argument('--folds', nargs='+', choices=list(ORGANS), default=['Breast', 'Brain', 'Kidney', 'liver'])
    ap.add_argument('--cut-epochs', type=int, default=120)
    ap.add_argument('--i2sb-epochs', type=int, default=85)
    ap.add_argument('--cut-batch', type=int, default=4)
    ap.add_argument('--i2sb-batch', type=int, default=4)
    ap.add_argument('--i2sb-nfe', type=int, default=20)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--cut-adapt-steps', type=int, default=300)
    ap.add_argument('--i2sb-adapt-steps', type=int, default=600)
    ap.add_argument('--skip-done', dest='skip_done', action='store_true', default=True)
    ap.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    ap.add_argument('--skip-cellpose', action='store_true', default=False)
    ap.add_argument('--reeval-only', action='store_true', default=False)
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    acquire_runtime_gpu_guard(args.gpu_id, allow_sharing=args.allow_gpu_sharing)
    cache_dir = (args.cache_dir or (work_dir / 'eval_cache' if (work_dir / 'eval_cache').is_dir() else PACKAGE_ROOT / 'eval_cache')).resolve()
    data_dir = resolve_data_dir(args.data_dir, work_dir)
    eval_dir = work_dir / 'eval_results'
    eval_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72, flush=True)
    print(f'[BOOT] Queue-3 SOTA Baselines | preset={args.preset} | methods={args.methods} | folds={args.folds}', flush=True)
    print(f'[BOOT] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} | torch_dev={torch.cuda.get_device_name(0)}', flush=True)
    print(f'[BOOT] GPUs: {_gpu_physical_info()}', flush=True)
    print('=' * 72, flush=True)

    splits = build_or_load_lomo_splits(data_dir, cache_dir, eval_dir / 'round1_splits.json', seed=2026)
    npz_path = cache_dir / 'gt_nuclei_instances_256.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f'Missing GT nuclei mask cache: {npz_path}')

    device = torch.device('cuda:0')
    lpips_fn = build_per_image_lpips(net_type='alex', normalize=True).cuda()
    skip_cp = args.skip_cellpose or os.environ.get('VSTAIN_SKIP_CELLPOSE') == '1'
    cell_model = None if skip_cp else models.CellposeModel(gpu=True, model_type='nuclei')

    if args.reeval_only:
        if cell_model is None:
            raise RuntimeError('--reeval-only requires Cellpose (unset VSTAIN_SKIP_CELLPOSE / omit --skip-cellpose)')
        jps = sorted((eval_dir / 'queue3_metrics').glob('*.json'))
        if not jps:
            raise FileNotFoundError(f'No Queue-3 metric JSONs found under {eval_dir / "queue3_metrics"}')
        fd_cache, gt_cache = {}, {}
        for jp in jps:
            old = json.loads(jp.read_text(encoding='utf-8'))
            fold_r, tag_r = old['fold'], old['run_tag']
            gdir = work_dir / 'generated' / fold_r / tag_r
            if fold_r not in fd_cache:
                t_in = data_dir / fold_r / 'test' / 'input'
                t_gt = data_dir / fold_r / 'test' / 'target'
                t_ids = splits['organs'][fold_r]['test']
                ex = ['file_4278.png'] if (fold_r == 'Breast' and 'file_4278.png' not in t_ids and (t_in / 'file_4278.png').exists()) else []
                inf_ids = t_ids + ex
                fd_cache[fold_r] = {
                    'test_ids': t_ids,
                    'infer_ids': inf_ids,
                    'pi_rgb_map': {n: rgb(t_in / n) for n in inf_ids},
                    'gt_rgb_map': {n: rgb(t_gt / n) for n in inf_ids},
                }
                gt_cache[fold_r] = load_gt_instance_masks(npz_path, fold_r)
            fd_r, gt_m = fd_cache[fold_r], gt_cache[fold_r]
            eval_n = list(fd_r['test_ids'])[:old.get('n_test', 120)]
            zn = 'file_4278.png' if 'file_4278.png' in fd_r['infer_ids'] else eval_n[-1]
            need_n = eval_n + ([zn] if zn not in eval_n else [])
            pmap = {nm: rgb(gdir / nm) for nm in need_n}
            record_queue3_eval(
                tag_r, fold_r, old['method'], old['upstream_repo'], old['stage'], old['k_shot'],
                pmap, fd_r, gt_m, cell_model, lpips_fn, device, work_dir, eval_n, zn,
                wall_seconds=old.get('wall_seconds', 0.0),
                trainable_params_m=old.get('trainable_params_m', 0.0),
                total_train_params_m=old.get('total_train_params_m', 0.0),
                latency_bs1_ms=old.get('latency_bs1_ms', 0.0),
                best_val_epoch=old.get('best_val_epoch', 0),
                config_hash=old.get('config_hash', ''), split_hash=old.get('split_hash', ''),
            )
        print('[COMPLETE] Queue-3 --reeval-only finished cleanly.', flush=True)
        return

    run_folds = ['Breast'] if args.preset == 'smoke' and args.folds == ['Breast', 'Brain', 'Kidney', 'liver'] else args.folds
    max_batches = 2 if args.preset == 'smoke' else None

    for fold in run_folds:
        fd = load_fold_tensors(data_dir, splits, fold, k_shot=20)
        eval_names = list(fd['test_ids'])[:8] if args.preset == 'smoke' else list(fd['test_ids'])
        zoom_name = 'file_4278.png' if ('file_4278.png' in fd['infer_ids']) else eval_names[-1]
        infer_subset = eval_names + ([zoom_name] if zoom_name not in eval_names else [])
        infer_idx = [fd['infer_ids'].index(n) for n in infer_subset]
        test_pi_sub = fd['test_pi_t'][infer_idx]
        gt_masks = load_gt_instance_masks(npz_path, fold)

        ckpt_fold = work_dir / 'checkpoints' / 'queue3' / fold
        ckpt_fold.mkdir(parents=True, exist_ok=True)
        split_hash = stable_hash({'target': fold, 'organs': splits['organs']})

        for method in args.methods:
            base_tag = f'{fold}__{method}'
            tag_0 = f'{base_tag}__0shot'
            tag_od = f'{base_tag}__0shot_halfOD'
            tag_20 = f'{base_tag}__20shot'
            is_cut = (method == 'cut')
            upstream_repo = UPSTREAM_REPOS[method]
            ep_cap = 1 if args.preset == 'smoke' else (args.cut_epochs if is_cut else args.i2sb_epochs)
            ad_steps = 2 if args.preset == 'smoke' else (args.cut_adapt_steps if is_cut else args.i2sb_adapt_steps)
            bs_m = args.cut_batch if is_cut else args.i2sb_batch
            nfe_m = 2 if args.preset == 'smoke' else args.i2sb_nfe

            cfg_hash = stable_hash({
                'fold': fold, 'method': method, 'upstream_repo': upstream_repo,
                'split_hash': split_hash, 'preset': args.preset,
                'ep_cap': int(ep_cap), 'batch_size': int(bs_m), 'ad_steps': int(ad_steps),
                'nfe': 1 if is_cut else int(nfe_m),
            })
            if args.skip_done and all(metric_matches_hash(eval_dir / 'queue3_metrics' / f'{t}.json', cfg_hash) for t in (tag_0, tag_od, tag_20)):
                print(f'[SKIP] {base_tag} already evaluated (verified config_hash={cfg_hash}).', flush=True)
                continue

            ckpt_0 = ckpt_fold / f'{tag_0}.pth'
            if is_cut:
                gen_state, pre_info = _train_cut_pretrain(
                    fold, fd['src_pi'], fd['src_he'], fd['val_pi'], fd['val_he'],
                    lpips_fn, ckpt_fold, max_epochs=ep_cap, batch_size=bs_m, lr=2e-4,
                    eval_every=1 if args.preset == 'smoke' else args.eval_every,
                    patience=0 if args.preset == 'smoke' else args.patience,
                    max_batches_per_epoch=max_batches,
                )
                torch.save({'gen': gen_state, 'config_hash': cfg_hash, 'split_hash': split_hash}, ckpt_0)
                eval_model, _ = build_cut_model(ep_cap, bs_m, 2e-4, ckpt_fold, f'{tag_0}_eval', is_train=False)
                net_eval = eval_model.netG.cuda()
                net_eval.load_state_dict(gen_state)
            else:
                gen_state, pre_info = _train_i2sb_pretrain(
                    fold, fd['src_pi'], fd['src_he'], fd['val_pi'], fd['val_he'],
                    lpips_fn, max_epochs=ep_cap, batch_size=bs_m, lr=5e-5,
                    eval_every=1 if args.preset == 'smoke' else args.eval_every,
                    patience=0 if args.preset == 'smoke' else args.patience,
                    nfe=nfe_m, max_batches_per_epoch=max_batches,
                )
                torch.save({'gen': gen_state, 'config_hash': cfg_hash, 'split_hash': split_hash}, ckpt_0)
                net_eval = build_i2sb_unet().cuda()
                net_eval.load_state_dict(gen_state)

            pred_fn0 = _make_predict_fn(method, net_eval, i2sb_nfe=nfe_m)
            lat0_ms = _measure_bs1_latency_ms(pred_fn0, test_pi_sub)
            t_inf = time.perf_counter()
            map_0 = _infer_to_pngs(pred_fn0, test_pi_sub, infer_subset, work_dir / 'generated' / fold / tag_0)
            record_queue3_eval(
                tag_0, fold, method, upstream_repo, '0shot', 0, map_0, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=pre_info['wall_seconds'] + (time.perf_counter() - t_inf),
                trainable_params_m=pre_info['trainable_params_m'],
                total_train_params_m=pre_info['total_train_params_m'],
                latency_bs1_ms=lat0_ms, best_val_epoch=pre_info['best_epoch'],
                config_hash=cfg_hash, split_hash=split_hash,
            )

            t_od = time.perf_counter()
            refs_0 = _infer_to_pngs(pred_fn0, fd['refs_pi_t'], fd['refs_ids'], work_dir / 'generated' / fold / f'_tmp_refs_{base_tag}')
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
            record_queue3_eval(
                tag_od, fold, method, upstream_repo, '0shot_halfOD', 5, map_od, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=time.perf_counter() - t_od,
                trainable_params_m=pre_info['trainable_params_m'],
                total_train_params_m=pre_info['total_train_params_m'],
                latency_bs1_ms=lat0_ms, best_val_epoch=pre_info['best_epoch'],
                config_hash=cfg_hash, split_hash=split_hash,
            )

            del net_eval
            if is_cut:
                del eval_model
            torch.cuda.empty_cache()

            if is_cut:
                state_20, ad_s = _adapt_cut_k20(
                    fold, pre_info['full_model_state'], fd['adapt_pi_t'], fd['adapt_he_t'],
                    ckpt_fold, adapt_steps=ad_steps, batch_size=bs_m, base_lr=2e-4,
                )
                pre_info['full_model_state'].clear()
                eval_20, _ = build_cut_model(ep_cap, bs_m, 2e-4, ckpt_fold, f'{tag_20}_eval', is_train=False)
                net_20 = eval_20.netG.cuda()
            else:
                state_20, ad_s = _adapt_i2sb_k20(
                    gen_state, fd['adapt_pi_t'], fd['adapt_he_t'],
                    adapt_steps=ad_steps, batch_size=bs_m, base_lr=5e-5,
                )
                net_20 = build_i2sb_unet().cuda()

            net_20.load_state_dict(state_20)
            torch.save({'gen': state_20, 'config_hash': cfg_hash, 'split_hash': split_hash}, ckpt_fold / f'{tag_20}.pth')

            pred_fn20 = _make_predict_fn(method, net_20, i2sb_nfe=nfe_m)
            lat20_ms = _measure_bs1_latency_ms(pred_fn20, test_pi_sub)
            t_inf20 = time.perf_counter()
            map_20 = _infer_to_pngs(pred_fn20, test_pi_sub, infer_subset, work_dir / 'generated' / fold / tag_20)
            record_queue3_eval(
                tag_20, fold, method, upstream_repo, '20shot', 20, map_20, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=ad_s + (time.perf_counter() - t_inf20),
                trainable_params_m=pre_info['trainable_params_m'],
                total_train_params_m=pre_info['total_train_params_m'],
                latency_bs1_ms=lat20_ms, best_val_epoch=pre_info['best_epoch'],
                config_hash=cfg_hash, split_hash=split_hash,
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

            del net_20
            if is_cut:
                del eval_20
            gc.collect()
            torch.cuda.empty_cache()

    print(f'\n[COMPLETE] Queue-3 preset={args.preset} finished cleanly on GPU {args.gpu_id}.', flush=True)


if __name__ == '__main__':
    main()
