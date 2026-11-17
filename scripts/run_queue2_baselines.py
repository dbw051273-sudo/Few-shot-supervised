"""End-to-end Queue-2 Official Baselines Runner (Pix2Pix, CycleGAN, UNIStainNet), locked to GPU 2.

Uses verbatim official upstream GitHub modules from `third_party_official/`:
  - junyanz/pytorch-CycleGAN-and-pix2pix (commit 2a7afba): Pix2PixModel, CycleGANModel
  - facevoid/UNIStainNet (commit bf3576b): SPADEUNetGenerator, PatchDiscriminator, Hinge+R1+FM

Protocol alignment with Round-1:
  - Identical 4-fold LOMO zero-adjacency splits (`round1_splits.json`, 480 train / 80 val / 120 test)
  - Generous epoch upper-bound + second-half LR decay + Source-Val (240 pairs) selection & early stopping
  - Matched 20-shot fine-tuning on target organ `adapt_20` (zero test leakage)
  - Unified evaluation via `v2_unified_eval.py` (KID subset_size=50 scaled by 1e3; Trainable Params & bs=1 ms/img)
  - Direct raw-folder inference CLI (`--infer-raw-dir`) for `eval_unscreened_fullset.py --external-pred-dir`
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
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
THIRD_PARTY_DIR = PACKAGE_ROOT / 'third_party_official'
P2P_REPO = THIRD_PARTY_DIR / 'pytorch_CycleGAN_and_pix2pix'
UNISTAIN_REPO = THIRD_PARTY_DIR / 'UNIStainNet'

for p in (SCRIPT_DIR, P2P_REPO, UNISTAIN_REPO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from gpu_guard import (  # noqa: E402
    acquire_runtime_gpu_guard,
    metric_matches_hash,
    preparse_gpu_and_env,
    stable_hash,
)

_PRE_ARGS = preparse_gpu_and_env('Workstation Queue-2')

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
    pngs,
    rgb,
    save_visual_sheet,
)
from uni_encoder import UNIEncoder  # noqa: E402
from v2_unified_eval import PAPER_CORE_KEYS, build_per_image_lpips, evaluate_split, load_gt_instance_masks, sparse_grid_laplacian  # noqa: E402

from models.pix2pix_model import Pix2PixModel  # noqa: E402  # official junyanz/pytorch-CycleGAN-and-pix2pix
from models.cycle_gan_model import CycleGANModel  # noqa: E402  # official junyanz/pytorch-CycleGAN-and-pix2pix
from src.models.generator import SPADEUNetGenerator  # noqa: E402  # official facevoid/UNIStainNet
from src.models.discriminator import (  # noqa: E402  # official facevoid/UNIStainNet
    PatchDiscriminator,
    feature_matching_loss,
    hinge_loss_d,
    hinge_loss_g,
)


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
        'Cannot locate UniPrior_v2 dataset (expected sibling ../UniPrior_v2). Pass --data-dir explicitly.'
    )


def _count_params_m(module: nn.Module) -> float:
    return round(sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6, 3)


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clone_cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


# ---------------------------------------------------------------------------
# Official Pix2Pix & CycleGAN builders
# ---------------------------------------------------------------------------

def _make_p2p_opt(
    method: str,
    is_train: bool,
    checkpoints_dir: Path,
    name: str,
    max_epochs: int,
    lr: float,
) -> SimpleNamespace:
    n_ep = max(1, max_epochs // 2)
    n_decay = max(1, max_epochs - n_ep)
    is_cgan = (method == 'cyclegan')
    return SimpleNamespace(
        isTrain=is_train,
        checkpoints_dir=str(checkpoints_dir),
        name=name,
        device=torch.device('cuda:0'),
        preprocess='none',
        verbose=False,
        continue_train=False,
        load_iter=0,
        epoch='latest',
        init_type='normal',
        init_gain=0.02,
        lr_policy='linear',
        epoch_count=1,
        n_epochs=n_ep,
        n_epochs_decay=n_decay,
        input_nc=3,
        output_nc=3,
        ngf=64,
        ndf=64,
        netD='basic',
        n_layers_D=3,
        direction='AtoB',
        beta1=0.5,
        lr=lr,
        netG='resnet_9blocks' if is_cgan else 'unet_256',
        norm='instance' if is_cgan else 'batch',
        no_dropout=True if is_cgan else False,
        gan_mode='lsgan' if is_cgan else 'vanilla',
        lambda_L1=100.0,
        lambda_A=10.0,
        lambda_B=10.0,
        lambda_identity=0.5,
        pool_size=50 if is_cgan else 0,
    )


def build_p2p_or_cyclegan(method: str, max_epochs: int, lr: float, ckpt_dir: Path, name: str, is_train: bool = True):
    opt = _make_p2p_opt(method, is_train=is_train, checkpoints_dir=ckpt_dir, name=name, max_epochs=max_epochs, lr=lr)
    model = CycleGANModel(opt) if method == 'cyclegan' else Pix2PixModel(opt)
    if is_train:
        model.setup(opt)
    return model


def _get_gen_G(model, method: str) -> nn.Module:
    return model.netG_A if method == 'cyclegan' else model.netG


# ---------------------------------------------------------------------------
# Official UNIStainNet helpers (SPADEUNetGenerator + PatchDiscriminator + Hinge+R1+FM)
# ---------------------------------------------------------------------------

def _uni_14_to_32(tokens_14: torch.Tensor) -> torch.Tensor:
    """Interpolate UNI [B, 196, 1024] (14x14) tokens to SPADEUNet [B, 1024, 1024] (32x32)."""
    b, _, c = tokens_14.shape
    grid = tokens_14.transpose(1, 2).reshape(b, c, 14, 14)
    up = F.interpolate(grid, size=(32, 32), mode='bilinear', align_corners=False)
    return up.reshape(b, c, 32 * 32).transpose(1, 2).contiguous()


def _sobel_edge_loss(gen: torch.Tensor, he_in: torch.Tensor, sobel_x: torch.Tensor, sobel_y: torch.Tensor) -> torch.Tensor:
    """Official UNIStainNet compute_he_edge_loss across full & half scale."""
    g_gray = ((gen.float() + 1.0) * 0.5).mean(dim=1, keepdim=True)
    h_gray = ((he_in.float() + 1.0) * 0.5).mean(dim=1, keepdim=True)
    loss = gen.new_zeros(())
    full_sz = g_gray.shape[-1]
    for sz in (full_sz, full_sz // 2):
        gg = g_gray if sz == full_sz else F.interpolate(g_gray, size=sz, mode='bilinear', align_corners=False)
        hh = h_gray if sz == full_sz else F.interpolate(h_gray, size=sz, mode='bilinear', align_corners=False)
        gx_g, gy_g = F.conv2d(gg, sobel_x, padding=1), F.conv2d(gg, sobel_y, padding=1)
        gx_h, gy_h = F.conv2d(hh, sobel_x, padding=1), F.conv2d(hh, sobel_y, padding=1)
        eg = (gx_g ** 2 + gy_g ** 2 + 1e-8).sqrt()
        eh = (gx_h ** 2 + gy_h ** 2 + 1e-8).sqrt()
        loss = loss + F.l1_loss(eg, eh.detach())
    return loss * 0.5


def build_unistainnet_generator() -> SPADEUNetGenerator:
    """Official paper config from UNIStainNet scripts/train/train_bci.py."""
    return SPADEUNetGenerator(
        num_classes=5,
        class_dim=64,
        uni_dim=1024,
        input_skip=True,
        edge_encoder='v2',
        edge_base_ch=32,
        uni_spatial_size=32,
        image_size=512,
    )


@torch.no_grad()
def extract_uni_patch14(uni_enc: UNIEncoder, imgs_256: torch.Tensor, batch: int = 32) -> torch.Tensor:
    outs = []
    for i in range(0, len(imgs_256), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = uni_enc(imgs_256[i:i + batch].cuda())[:, 1:, :]
        outs.append(tok.half().cpu())
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Unified Validation & Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_source_val_score(
    predict_fn,
    val_pi: torch.Tensor,
    val_he: torch.Tensor,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    val_uni_14: torch.Tensor | None = None,
    batch: int = 16,
) -> tuple[float, float, float]:
    l1s, lps = [], []
    for i in range(0, len(val_pi), batch):
        x = val_pi[i:i + batch].cuda()
        y = val_he[i:i + batch].cuda()
        u = val_uni_14[i:i + batch].cuda().float() if val_uni_14 is not None else None
        pred = predict_fn(x, u).clamp(-1.0, 1.0)
        l1_val = F.l1_loss(pred, y, reduction='none').mean(dim=(1, 2, 3))
        lp_val = lpips_fn((pred + 1.0) * 0.5, (y + 1.0) * 0.5)
        l1s.extend(l1_val.cpu().tolist())
        lps.extend(lp_val.cpu().tolist())
    m_l1 = float(np.mean(l1s))
    m_lp = float(np.mean(lps))
    return m_l1 + m_lp, m_l1, m_lp


@torch.no_grad()
def _extract_uni_tokens_eval(uni_enc: UNIEncoder, imgs: torch.Tensor) -> torch.Tensor:
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return uni_enc(imgs)[:, 1:, :].float()


@torch.no_grad()
def measure_bs1_latency_ms(
    predict_fn,
    sample_pi: torch.Tensor,
    uni_enc: UNIEncoder | None,
    needs_uni: bool = False,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> float:
    x0 = sample_pi[:1].cuda()
    for _ in range(n_warmup):
        u0 = _extract_uni_tokens_eval(uni_enc, x0) if (needs_uni and uni_enc is not None) else None
        _ = predict_fn(x0, u0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        u0 = _extract_uni_tokens_eval(uni_enc, x0) if (needs_uni and uni_enc is not None) else None
        _ = predict_fn(x0, u0)
    torch.cuda.synchronize()
    return round((time.perf_counter() - t0) * 1000.0 / float(n_runs), 2)


@torch.no_grad()
def infer_queue2_to_pngs(
    predict_fn,
    pi_tensor: torch.Tensor,
    names: list[str],
    out_dir: Path,
    uni_enc: UNIEncoder | None = None,
    needs_uni: bool = False,
    batch: int = 16,
) -> dict[str, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_map: dict[str, np.ndarray] = {}
    for i in range(0, len(pi_tensor), batch):
        x = pi_tensor[i:i + batch].cuda()
        u = _extract_uni_tokens_eval(uni_enc, x) if (needs_uni and uni_enc is not None) else None
        pred = predict_fn(x, u).clamp(-1.0, 1.0)
        arr = np.rint(((pred.permute(0, 2, 3, 1).cpu().numpy() + 1.0) * 127.5)).clip(0, 255).astype(np.uint8)
        for j, nm in enumerate(names[i:i + batch]):
            pred_map[nm] = arr[j]
            Image.fromarray(arr[j]).save(out_dir / nm)
    return pred_map


def _make_predict_fn(method: str, gen_module: nn.Module):
    gen_module.eval()
    if method in ('pix2pix', 'cyclegan'):
        def _pred(x_256: torch.Tensor, _u=None) -> torch.Tensor:
            return gen_module(x_256)
        return _pred

    def _pred_uni(x_256: torch.Tensor, u_14: torch.Tensor | None) -> torch.Tensor:
        if u_14 is None:
            raise ValueError('UNIStainNet predictor requires UNI patch tokens')
        x_512 = F.interpolate(x_256, size=(512, 512), mode='bilinear', align_corners=False)
        u_32 = _uni_14_to_32(u_14)
        labels = torch.zeros(x_256.shape[0], dtype=torch.long, device=x_256.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y_512 = gen_module(x_512, u_32, labels)
        return F.interpolate(y_512.float(), size=(256, 256), mode='bilinear', align_corners=False)

    return _pred_uni


# ---------------------------------------------------------------------------
# Pretrain & 20-shot Adaptation Loops
# ---------------------------------------------------------------------------

def train_p2p_or_cyclegan_pretrain(
    method: str,
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
    tag = f'{fold}__{method}'
    model = build_p2p_or_cyclegan(method, max_epochs=max_epochs, lr=lr, ckpt_dir=ckpt_dir, name=tag, is_train=True)
    gen_g = _get_gen_G(model, method)
    n_src = len(src_pi)
    best_score, best_epoch = 1e9, 0
    best_gen_state = _clone_cpu_state(gen_g)
    best_model_state = {nm: _clone_cpu_state(getattr(model, 'net' + nm)) for nm in model.model_names}
    no_improve_evals = 0
    val_hist = []
    t0 = time.perf_counter()

    for ep in range(1, max_epochs + 1):
        gen_g.train()
        perm_a = torch.randperm(n_src)
        perm_b = torch.randperm(n_src) if method == 'cyclegan' else perm_a
        n_steps_ep = 0
        for s in range(0, n_src, batch_size):
            idx_a = perm_a[s:s + batch_size]
            idx_b = perm_b[s:s + batch_size]
            if len(idx_a) < 2:
                continue
            model.set_input({
                'A': src_pi[idx_a],
                'B': src_he[idx_b],
                'A_paths': [''] * len(idx_a),
                'B_paths': [''] * len(idx_b),
            })
            model.optimize_parameters()
            n_steps_ep += 1
            if max_batches_per_epoch is not None and n_steps_ep >= max_batches_per_epoch:
                break
        model.update_learning_rate()

        if ep % eval_every == 0 or ep == max_epochs:
            val_score, val_l1, val_lp = _eval_source_val_score(
                _make_predict_fn(method, gen_g), val_pi, val_he, lpips_fn, batch=16,
            )
            gen_g.train()
            val_hist.append({'epoch': ep, 'val_score': round(val_score, 5), 'val_l1': round(val_l1, 5), 'val_lpips': round(val_lp, 5)})
            if val_score < best_score:
                best_score, best_epoch = val_score, ep
                best_gen_state = _clone_cpu_state(gen_g)
                best_model_state = {nm: _clone_cpu_state(getattr(model, 'net' + nm)) for nm in model.model_names}
                no_improve_evals = 0
            elif ep >= max(1, max_epochs // 2):
                no_improve_evals += 1
            print(
                f'[{tag}] pretrain ep {ep}/{max_epochs} | val_score={val_score:.4f} (L1={val_l1:.4f}, LPIPS={val_lp:.4f}, best_ep={best_epoch}) ({time.perf_counter()-t0:.1f}s)',
                flush=True,
            )
            if patience > 0 and ep >= max(1, max_epochs // 2) and no_improve_evals >= patience:
                print(f'[{tag}] Source-Val early stop at ep {ep} (best_ep={best_epoch})', flush=True)
                break

    params_m = _count_params_m(gen_g)
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
        'full_model_state': best_model_state,
    }


def adapt_p2p_or_cyclegan_k20(
    method: str,
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
    lr_ad = base_lr * 0.25
    approx_epochs = max(2, adapt_steps // max(1, len(adapt_pi) // min(batch_size, len(adapt_pi))))
    model = build_p2p_or_cyclegan(method, max_epochs=approx_epochs, lr=lr_ad, ckpt_dir=ckpt_dir, name=f'{fold}__{method}__20shot', is_train=True)
    for nm, st in full_model_state.items():
        if hasattr(model, 'net' + nm):
            getattr(model, 'net' + nm).load_state_dict(st)

    gen_g = _get_gen_G(model, method)
    gen_g.train()
    n_ad = len(adapt_pi)
    bs = min(batch_size, n_ad)
    t0 = time.perf_counter()
    for step in range(1, adapt_steps + 1):
        idx_a = torch.randperm(n_ad)[:bs]
        idx_b = torch.randperm(n_ad)[:bs] if method == 'cyclegan' else idx_a
        model.set_input({
            'A': adapt_pi[idx_a],
            'B': adapt_he[idx_b],
            'A_paths': [''] * bs,
            'B_paths': [''] * bs,
        })
        model.optimize_parameters()
    out_state = _clone_cpu_state(gen_g)
    elapsed = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return out_state, elapsed


def _unistainnet_step(
    step_idx: int,
    total_steps: int,
    gen: SPADEUNetGenerator,
    gen_ema: SPADEUNetGenerator,
    disc: PatchDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    he_256: torch.Tensor,
    gt_256: torch.Tensor,
    uni_14: torch.Tensor,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
    gen_lr: float,
    disc_lr: float,
    warmup_steps: int,
    adv_start_step: int,
    use_cosine_decay: bool = True,
    r1_every: int = 16,
    r1_weight: float = 10.0,
    ema_decay: float = 0.999,
) -> float:
    b = he_256.shape[0]
    he_512 = F.interpolate(he_256.cuda(), size=(512, 512), mode='bilinear', align_corners=False)
    gt_512 = F.interpolate(gt_256.cuda(), size=(512, 512), mode='bilinear', align_corners=False)
    uni_32 = _uni_14_to_32(uni_14.cuda().float())

    # CFG dropout from official UNIStainNetTrainer
    labels = torch.zeros(b, dtype=torch.long, device=he_512.device)
    r = torch.rand(b, device=he_512.device)
    drop_both = r < 0.05
    drop_cls = (r >= 0.05) & (r < 0.15)
    drop_uni = (r >= 0.15) & (r < 0.25)
    labels_in = labels.clone()
    labels_in[drop_both | drop_cls] = 4  # null_class=4
    uni_in = uni_32.clone()
    uni_in[drop_both | drop_uni] = 0.0

    # Generator & discriminator LR schedule (linear warmup + second-half cosine decay to 0.1x)
    if step_idx < warmup_steps:
        lr_scale = step_idx / float(max(1, warmup_steps))
    elif use_cosine_decay and total_steps > 2 and step_idx > total_steps // 2:
        prog = (step_idx - total_steps // 2) / float(max(1, total_steps - total_steps // 2))
        lr_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    else:
        lr_scale = 1.0
    for pg in opt_g.param_groups:
        pg['lr'] = gen_lr * lr_scale
    for pg in opt_d.param_groups:
        pg['lr'] = disc_lr * lr_scale

    with torch.autocast('cuda', dtype=torch.bfloat16):
        gen_512 = gen(he_512, uni_in, labels_in)

    gen_f = gen_512.float()
    gen_128 = F.interpolate(gen_f, size=128, mode='bilinear', align_corners=False).clamp(-1.0, 1.0)
    gt_128 = F.interpolate(gt_512, size=128, mode='bilinear', align_corners=False).clamp(-1.0, 1.0)
    gen_256 = F.interpolate(gen_f, size=256, mode='bilinear', align_corners=False).clamp(-1.0, 1.0)
    gt_256_c = gt_256.cuda().clamp(-1.0, 1.0)
    gen_64 = F.interpolate(gen_f, size=64, mode='bilinear', align_corners=False)
    gt_64 = F.interpolate(gt_512, size=64, mode='bilinear', align_corners=False)

    loss_lp128 = lpips_fn((gen_128 + 1.0) * 0.5, (gt_128 + 1.0) * 0.5).mean()
    loss_lp256 = lpips_fn((gen_256 + 1.0) * 0.5, (gt_256_c + 1.0) * 0.5).mean()
    loss_l1_lr = F.l1_loss(gen_64, gt_64)
    loss_edge = _sobel_edge_loss(gen_f, he_512, sobel_x, sobel_y)
    loss_g = 1.0 * loss_lp128 + 0.5 * loss_lp256 + 1.0 * loss_l1_lr + 0.5 * loss_edge

    if step_idx >= adv_start_step:
        was_disc_train = disc.training
        disc.eval()
        for p in disc.parameters():
            p.requires_grad_(False)
        d_fake, feats_fake = disc(gen_f, return_features=True)
        with torch.no_grad():
            _, feats_real = disc(gt_512, return_features=True)
        loss_adv = hinge_loss_g(d_fake)
        loss_fm = feature_matching_loss(feats_fake, feats_real)
        loss_g = loss_g + 1.0 * loss_adv + 10.0 * loss_fm
        for p in disc.parameters():
            p.requires_grad_(True)
        if was_disc_train:
            disc.train()

    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
    opt_g.step()

    with torch.no_grad():
        for p_ema, p in zip(gen_ema.parameters(), gen.parameters()):
            p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

    # Discriminator update
    if step_idx >= adv_start_step:
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                fake_det = gen(he_512, uni_in, labels_in).float()
        d_real = disc(gt_512)
        d_fake_d = disc(fake_det)
        loss_d = hinge_loss_d(d_real, d_fake_d)
        if step_idx % r1_every == 0:
            real_r1 = gt_512.detach().requires_grad_(True)
            out_r1 = disc(real_r1)
            grad_r1 = torch.autograd.grad(outputs=out_r1.sum(), inputs=real_r1, create_graph=True)[0]
            loss_d = loss_d + r1_weight * grad_r1.pow(2).mean()
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

    return float(loss_g.item())


def train_unistainnet_pretrain(
    fold: str,
    src_pi: torch.Tensor,
    src_he: torch.Tensor,
    src_uni_14: torch.Tensor,
    val_pi: torch.Tensor,
    val_he: torch.Tensor,
    val_uni_14: torch.Tensor,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    max_epochs: int,
    batch_size: int,
    gen_lr: float,
    disc_lr: float,
    eval_every: int,
    patience: int,
    max_batches_per_epoch: int | None = None,
    seed: int = 2026,
) -> tuple[dict[str, torch.Tensor], dict]:
    _set_seed(seed)
    tag = f'{fold}__unistainnet'
    gen = build_unistainnet_generator().cuda()
    gen_ema = copy.deepcopy(gen).cuda().eval()
    gen_ema.requires_grad_(False)
    disc = PatchDiscriminator(in_channels=3, ndf=64, n_layers=3).cuda()

    opt_g = torch.optim.Adam(gen.parameters(), lr=gen_lr, betas=(0.0, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=disc_lr, betas=(0.0, 0.999))
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device='cuda').view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    n_src = len(src_pi)
    steps_per_ep = max(1, (n_src // batch_size) if max_batches_per_epoch is None else min(max_batches_per_epoch, n_src // batch_size))
    total_steps = max_epochs * steps_per_ep
    warmup_steps = min(500, max(1, total_steps // 10))
    adv_start_step = 1 if max_epochs <= 2 else min(360, steps_per_ep)

    best_score, best_epoch = 1e9, 0
    best_ema_state = _clone_cpu_state(gen_ema)
    best_disc_state = _clone_cpu_state(disc)
    no_improve_evals = 0
    val_hist = []
    step_idx = 0
    t0 = time.perf_counter()

    for ep in range(1, max_epochs + 1):
        gen.train()
        disc.train()
        perm = torch.randperm(n_src)
        n_batches_ep = 0
        for s in range(0, n_src, batch_size):
            idx = perm[s:s + batch_size]
            if len(idx) < 2:
                continue
            step_idx += 1
            _unistainnet_step(
                step_idx, total_steps, gen, gen_ema, disc, opt_g, opt_d,
                src_pi[idx], src_he[idx], src_uni_14[idx],
                lpips_fn, sobel_x, sobel_y,
                gen_lr=gen_lr, disc_lr=disc_lr, warmup_steps=warmup_steps,
                adv_start_step=adv_start_step, use_cosine_decay=True,
            )
            n_batches_ep += 1
            if max_batches_per_epoch is not None and n_batches_ep >= max_batches_per_epoch:
                break

        if ep % eval_every == 0 or ep == max_epochs:
            val_score, val_l1, val_lp = _eval_source_val_score(
                _make_predict_fn('unistainnet', gen_ema), val_pi, val_he, lpips_fn, val_uni_14=val_uni_14, batch=16,
            )
            val_hist.append({'epoch': ep, 'val_score': round(val_score, 5), 'val_l1': round(val_l1, 5), 'val_lpips': round(val_lp, 5)})
            if val_score < best_score:
                best_score, best_epoch = val_score, ep
                best_ema_state = _clone_cpu_state(gen_ema)
                best_disc_state = _clone_cpu_state(disc)
                no_improve_evals = 0
            elif ep >= max(1, max_epochs // 2):
                no_improve_evals += 1
            print(
                f'[{tag}] pretrain ep {ep}/{max_epochs} | val_score={val_score:.4f} (L1={val_l1:.4f}, LPIPS={val_lp:.4f}, best_ep={best_epoch}) ({time.perf_counter()-t0:.1f}s)',
                flush=True,
            )
            if patience > 0 and ep >= max(1, max_epochs // 2) and no_improve_evals >= patience:
                print(f'[{tag}] Source-Val early stop at ep {ep} (best_ep={best_epoch})', flush=True)
                break

    params_m = _count_params_m(gen)
    disc_m = _count_params_m(disc)
    del gen, gen_ema, disc
    torch.cuda.empty_cache()
    return best_ema_state, {
        'best_epoch': best_epoch,
        'best_val_score': best_score,
        'val_history': val_hist,
        'trainable_params_m': params_m,
        'total_train_params_m': round(params_m + disc_m, 3),
        'wall_seconds': round(time.perf_counter() - t0, 2),
        'disc_state': best_disc_state,
    }


def adapt_unistainnet_k20(
    gen_state: dict[str, torch.Tensor],
    disc_state: dict[str, torch.Tensor],
    adapt_pi: torch.Tensor,
    adapt_he: torch.Tensor,
    adapt_uni_14: torch.Tensor,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    adapt_steps: int,
    batch_size: int,
    gen_lr: float,
    disc_lr: float,
    seed: int = 7202,
) -> tuple[dict[str, torch.Tensor], float]:
    _set_seed(seed)
    gen = build_unistainnet_generator().cuda()
    gen.load_state_dict(gen_state)
    gen_ema = copy.deepcopy(gen).cuda().eval()
    gen_ema.requires_grad_(False)
    disc = PatchDiscriminator(in_channels=3, ndf=64, n_layers=3).cuda()
    disc.load_state_dict(disc_state)

    lr_g_ad = gen_lr * 0.25
    lr_d_ad = disc_lr * 0.25
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g_ad, betas=(0.0, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d_ad, betas=(0.0, 0.999))
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device='cuda').view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    n_ad = len(adapt_pi)
    bs = min(batch_size, n_ad)
    t0 = time.perf_counter()
    gen.train()
    disc.train()
    for step in range(1, adapt_steps + 1):
        idx = torch.randperm(n_ad)[:bs]
        _unistainnet_step(
            step, adapt_steps, gen, gen_ema, disc, opt_g, opt_d,
            adapt_pi[idx], adapt_he[idx], adapt_uni_14[idx],
            lpips_fn, sobel_x, sobel_y,
            gen_lr=lr_g_ad, disc_lr=lr_d_ad, warmup_steps=1,
            adv_start_step=1, use_cosine_decay=False, ema_decay=0.98,
        )
    out_state = _clone_cpu_state(gen_ema)
    elapsed = time.perf_counter() - t0
    del gen, gen_ema, disc
    torch.cuda.empty_cache()
    return out_state, elapsed


# ---------------------------------------------------------------------------
# Summary & Metric Recording (matches Round-1 KIDx1e3 schema)
# ---------------------------------------------------------------------------

def update_queue2_summary_files(eval_dir: Path) -> None:
    metrics_dir = eval_dir / 'queue2_metrics'
    records = [json.loads(jp.read_text(encoding='utf-8')) for jp in sorted(metrics_dir.glob('*.json'))]
    (eval_dir / 'queue2_summary.json').write_text(
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
    with (eval_dir / 'queue2_summary.csv').open('w', newline='', encoding='utf-8') as f:
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
        '# Queue-2 Official GitHub Baselines Evaluation Summary',
        '',
        f'- Updated: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`',
        '- Official Sources: `Pix2Pix` & `CycleGAN` (`junyanz/pytorch-CycleGAN-and-pix2pix@2a7afba`), `UNIStainNet` (`facevoid/UNIStainNet@bf3576b`)',
        '- Note: `KID` values are scaled by `10^3` (`subset_size=50`). `FID` depends on test size `N` and is omitted from main/cross-`N` tables (retained in JSON/CSV for same-`N` diagnostics).',
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
    (eval_dir / 'queue2_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def record_queue2_eval(
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
    (eval_dir / 'queue2_metrics').mkdir(parents=True, exist_ok=True)

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
    (eval_dir / 'queue2_metrics' / f'{run_tag}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    update_queue2_summary_files(eval_dir)
    m = payload['metrics']
    print(
        f"[Q2_EVAL] {run_tag:28s} | PSNR={m['psnr']:5.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} "
        f"KIDx1e3={m['kid']*1000.0:5.2f} Dice={m['nuclei_dice']:.4f} F1={m['nuclei_f1_05']:.4f} | "
        f"Params={trainable_params_m:.2f}M {latency_bs1_ms:.2f}ms",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# Raw Directory Inference Helper (for P2 Exp-8 eval_unscreened_fullset.py)
# ---------------------------------------------------------------------------

def run_raw_dir_inference(args: argparse.Namespace, work_dir: Path) -> None:
    if len(args.folds) != 1 or len(args.methods) != 1:
        raise ValueError('--infer-raw-dir expects exactly one --folds <Organ> and one --methods <method>')
    fold = args.folds[0]
    method = args.methods[0]
    stage_tag = '20shot' if args.infer_stage == '20shot' else '0shot'
    ckpt_path = work_dir / 'checkpoints' / 'queue2' / fold / f'{fold}__{method}__{stage_tag}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Missing Queue-2 checkpoint: {ckpt_path}')
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    if method in ('pix2pix', 'cyclegan'):
        opt = _make_p2p_opt(method, is_train=False, checkpoints_dir=ckpt_path.parent, name='infer', max_epochs=2, lr=2e-4)
        model = CycleGANModel(opt) if method == 'cyclegan' else Pix2PixModel(opt)
        gen = _get_gen_G(model, method).cuda()
        gen.load_state_dict(obj['gen'])
        uni_enc = None
        needs_uni = False
    else:
        gen = build_unistainnet_generator().cuda()
        gen.load_state_dict(obj['gen'])
        uni_enc = UNIEncoder(freeze=True).cuda().eval()
        needs_uni = True

    predict_fn = _make_predict_fn(method, gen)
    in_arg = args.infer_raw_dir.resolve()
    if (in_arg / fold / 'test' / 'input').is_dir():
        in_dir = in_arg / fold / 'test' / 'input'
    elif (in_arg / fold / 'input').is_dir():
        in_dir = in_arg / fold / 'input'
    else:
        in_dir = in_arg
    out_dir = args.infer_out_dir.resolve() if args.infer_out_dir else (work_dir / 'generated_unscreened' / f'{fold}__{method}__{stage_tag}')
    out_dir.mkdir(parents=True, exist_ok=True)
    invert_pi = (fold == 'liver') and (not args.no_invert_liver)

    names = sorted(pngs(in_dir).keys())
    if args.max_infer is not None:
        names = names[:args.max_infer]
    for i in range(0, len(names), 16):
        bn = names[i:i + 16]
        arrs = []
        for nm in bn:
            a = rgb(in_dir / nm)
            if invert_pi:
                a = (255 - a).astype(np.uint8)
            arrs.append(a)
        pi_t = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        infer_queue2_to_pngs(predict_fn, pi_t, bn, out_dir, uni_enc=uni_enc, needs_uni=needs_uni, batch=len(bn))
    print(f'[RAW_INFER] Saved {len(names)} PNGs ({fold}/{method}/{stage_tag}) -> {out_dir}', flush=True)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

UPSTREAM_REPOS = {
    'pix2pix': 'junyanz/pytorch-CycleGAN-and-pix2pix@2a7afba',
    'cyclegan': 'junyanz/pytorch-CycleGAN-and-pix2pix@2a7afba',
    'unistainnet': 'facevoid/UNIStainNet@bf3576b',
}


def main() -> None:
    default_work = PACKAGE_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu-id', type=str, default='2', help='Must be 2 unless --allow-other-gpu')
    ap.add_argument('--allow-other-gpu', action='store_true', default=False)
    ap.add_argument('--allow-gpu-sharing', action='store_true', default=False)
    ap.add_argument('--online-hf', action='store_true', default=False)
    ap.add_argument('--work-dir', type=Path, default=default_work)
    ap.add_argument('--cache-dir', type=Path, default=None)
    ap.add_argument('--data-dir', type=Path, default=None)
    ap.add_argument('--preset', choices=['smoke', 'queue2'], default='queue2')
    ap.add_argument('--methods', nargs='+', choices=['pix2pix', 'cyclegan', 'unistainnet'], default=['pix2pix', 'cyclegan', 'unistainnet'])
    ap.add_argument('--folds', nargs='+', choices=list(ORGANS), default=['Breast', 'Brain', 'Kidney', 'liver'])
    ap.add_argument('--p2p-epochs', type=int, default=120, help='Upper-bound epochs for Pix2Pix & CycleGAN')
    ap.add_argument('--uni-epochs', type=int, default=80, help='Upper-bound epochs for UNIStainNet')
    ap.add_argument('--p2p-batch', type=int, default=8)
    ap.add_argument('--cycle-batch', type=int, default=4)
    ap.add_argument('--uni-batch', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=5, help='Source-Val eval cadence (epochs)')
    ap.add_argument('--patience', type=int, default=6, help='Early-stop patience in val rounds (0=disable)')
    ap.add_argument('--adapt-steps', type=int, default=300, help='20-shot fine-tuning steps for Pix2Pix/CycleGAN')
    ap.add_argument('--uni-adapt-steps', type=int, default=200, help='20-shot fine-tuning steps for UNIStainNet')
    ap.add_argument('--skip-done', dest='skip_done', action='store_true', default=True)
    ap.add_argument('--no-skip-done', dest='skip_done', action='store_false')
    ap.add_argument('--skip-cellpose', action='store_true', default=False, help='Skip Cellpose during overnight train/gen; re-eval saved PNGs later')
    ap.add_argument('--reeval-only', action='store_true', default=False, help='Re-evaluate already generated PNGs with Cellpose without retraining')
    # Optional raw unscreened inference mode
    ap.add_argument('--infer-raw-dir', type=Path, default=None)
    ap.add_argument('--infer-out-dir', type=Path, default=None)
    ap.add_argument('--infer-stage', choices=['0shot', '20shot'], default='20shot')
    ap.add_argument('--max-infer', type=int, default=None)
    ap.add_argument('--no-invert-liver', action='store_true', default=False)
    args = ap.parse_args()

    work_dir = args.work_dir.resolve()
    acquire_runtime_gpu_guard(args.gpu_id, allow_sharing=args.allow_gpu_sharing)
    if args.infer_raw_dir is not None:
        run_raw_dir_inference(args, work_dir)
        return

    cache_dir = (args.cache_dir or (work_dir / 'eval_cache' if (work_dir / 'eval_cache').is_dir() else PACKAGE_ROOT / 'eval_cache')).resolve()
    data_dir = resolve_data_dir(args.data_dir, work_dir)
    eval_dir = work_dir / 'eval_results'
    eval_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72, flush=True)
    print(f'[BOOT] Queue-2 Official Baselines | preset={args.preset} | methods={args.methods} | folds={args.folds}', flush=True)
    print(f'[BOOT] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} | torch_dev={torch.cuda.get_device_name(0)}', flush=True)
    print(f'[BOOT] GPUs: {_gpu_physical_info()}', flush=True)
    print('=' * 72, flush=True)

    splits = build_or_load_lomo_splits(data_dir, cache_dir, eval_dir / 'round1_splits.json', seed=2026)
    npz_path = cache_dir / 'gt_nuclei_instances_256.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f'Missing GT nuclei mask cache: {npz_path}')

    device = torch.device('cuda:0')
    need_uni_enc = ('unistainnet' in args.methods) and not args.reeval_only
    uni_enc = UNIEncoder(freeze=True).cuda().eval() if need_uni_enc else None
    lpips_fn = build_per_image_lpips(net_type='alex', normalize=True).cuda()
    skip_cp = args.skip_cellpose or os.environ.get('VSTAIN_SKIP_CELLPOSE') == '1'
    cell_model = None if skip_cp else models.CellposeModel(gpu=True, model_type='nuclei')
    if args.reeval_only:
        if cell_model is None:
            raise RuntimeError('--reeval-only requires Cellpose (unset VSTAIN_SKIP_CELLPOSE / omit --skip-cellpose)')
        jps = sorted((eval_dir / 'queue2_metrics').glob('*.json'))
        if not jps:
            raise FileNotFoundError(f'No Queue-2 metric JSONs found under {eval_dir / "queue2_metrics"}')
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
            record_queue2_eval(
                tag_r, fold_r, old['method'], old['upstream_repo'], old['stage'], old['k_shot'],
                pmap, fd_r, gt_m, cell_model, lpips_fn, device, work_dir, eval_n, zn,
                wall_seconds=old.get('wall_seconds', 0.0),
                trainable_params_m=old.get('trainable_params_m', 0.0),
                total_train_params_m=old.get('total_train_params_m', 0.0),
                latency_bs1_ms=old.get('latency_bs1_ms', 0.0),
                best_val_epoch=old.get('best_val_epoch', 0),
                config_hash=old.get('config_hash', ''), split_hash=old.get('split_hash', ''),
            )
        print('[COMPLETE] Queue-2 --reeval-only finished cleanly.', flush=True)
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

        src_uni_14 = extract_uni_patch14(uni_enc, fd['src_pi']) if uni_enc is not None else None
        val_uni_14 = extract_uni_patch14(uni_enc, fd['val_pi']) if uni_enc is not None else None
        ad_uni_14 = extract_uni_patch14(uni_enc, fd['adapt_pi_t']) if uni_enc is not None else None

        ckpt_fold = work_dir / 'checkpoints' / 'queue2' / fold
        ckpt_fold.mkdir(parents=True, exist_ok=True)

        split_hash = stable_hash({'target': fold, 'organs': splits['organs']})
        for method in args.methods:
            base_tag = f'{fold}__{method}'
            tag_0 = f'{base_tag}__0shot'
            tag_od = f'{base_tag}__0shot_halfOD'
            tag_20 = f'{base_tag}__20shot'
            needs_uni = (method == 'unistainnet')
            upstream_repo = UPSTREAM_REPOS[method]
            ep_cap = 1 if args.preset == 'smoke' else (args.uni_epochs if needs_uni else args.p2p_epochs)
            ad_steps = 2 if args.preset == 'smoke' else (args.uni_adapt_steps if needs_uni else args.adapt_steps)
            bs_m = args.uni_batch if needs_uni else (args.cycle_batch if method == 'cyclegan' else args.p2p_batch)
            cfg_hash = stable_hash({
                'fold': fold,
                'method': method,
                'upstream_repo': upstream_repo,
                'split_hash': split_hash,
                'preset': args.preset,
                'ep_cap': int(ep_cap),
                'batch_size': int(bs_m),
                'ad_steps': int(ad_steps),
                'eval_every': int(1 if args.preset == 'smoke' else args.eval_every),
                'patience': int(0 if args.preset == 'smoke' else args.patience),
                'schedule': 'second_half_decay_and_patience',
            })
            if args.skip_done and all(metric_matches_hash(eval_dir / 'queue2_metrics' / f'{t}.json', cfg_hash) for t in (tag_0, tag_od, tag_20)):
                print(f'[SKIP] {base_tag} already evaluated (verified config_hash={cfg_hash}).', flush=True)
                continue

            ckpt_0 = ckpt_fold / f'{tag_0}.pth'

            if method in ('pix2pix', 'cyclegan'):
                gen_state, pre_info = train_p2p_or_cyclegan_pretrain(
                    method, fold, fd['src_pi'], fd['src_he'], fd['val_pi'], fd['val_he'],
                    lpips_fn, ckpt_fold, max_epochs=ep_cap, batch_size=bs_m, lr=2e-4,
                    eval_every=1 if args.preset == 'smoke' else args.eval_every,
                    patience=0 if args.preset == 'smoke' else args.patience,
                    max_batches_per_epoch=max_batches,
                )
                torch.save(
                    {'gen': gen_state, 'config_hash': cfg_hash, 'split_hash': split_hash, 'pre_info': {k: v for k, v in pre_info.items() if k != 'full_model_state'}},
                    ckpt_0,
                )
                gen_eval = _get_gen_G(build_p2p_or_cyclegan(method, ep_cap, 2e-4, ckpt_fold, 'eval', is_train=False), method).cuda()
                gen_eval.load_state_dict(gen_state)
            else:
                gen_state, pre_info = train_unistainnet_pretrain(
                    fold, fd['src_pi'], fd['src_he'], src_uni_14,
                    fd['val_pi'], fd['val_he'], val_uni_14,
                    lpips_fn, max_epochs=ep_cap, batch_size=bs_m, gen_lr=1e-4, disc_lr=4e-4,
                    eval_every=1 if args.preset == 'smoke' else args.eval_every,
                    patience=0 if args.preset == 'smoke' else args.patience,
                    max_batches_per_epoch=max_batches,
                )
                torch.save(
                    {'gen': gen_state, 'config_hash': cfg_hash, 'split_hash': split_hash, 'pre_info': {k: v for k, v in pre_info.items() if k != 'disc_state'}},
                    ckpt_0,
                )
                gen_eval = build_unistainnet_generator().cuda()
                gen_eval.load_state_dict(gen_state)

            pred_fn0 = _make_predict_fn(method, gen_eval)
            lat0_ms = measure_bs1_latency_ms(pred_fn0, test_pi_sub, uni_enc, needs_uni=needs_uni)
            t_inf = time.perf_counter()
            map_0 = infer_queue2_to_pngs(
                pred_fn0, test_pi_sub, infer_subset, work_dir / 'generated' / fold / tag_0,
                uni_enc=uni_enc, needs_uni=needs_uni,
            )
            record_queue2_eval(
                tag_0, fold, method, upstream_repo, '0shot', 0, map_0, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=pre_info['wall_seconds'] + (time.perf_counter() - t_inf),
                trainable_params_m=pre_info['trainable_params_m'],
                total_train_params_m=pre_info['total_train_params_m'],
                latency_bs1_ms=lat0_ms, best_val_epoch=pre_info['best_epoch'],
                config_hash=cfg_hash, split_hash=split_hash,
            )

            t_od = time.perf_counter()
            tmp_refs_dir = work_dir / 'generated' / fold / f'_tmp_refs_{base_tag}'
            refs_0 = infer_queue2_to_pngs(
                pred_fn0, fd['refs_pi_t'], fd['refs_ids'], tmp_refs_dir,
                uni_enc=uni_enc, needs_uni=needs_uni,
            )
            try:
                sc = fd['refs_od'] / np.maximum(mean_od([refs_0[n] for n in fd['refs_ids']], tag_od), 1e-4)
            except ValueError as exc:
                print(f'[WARN] {tag_od}: {exc}; falling back to identity OD scale=1.0', flush=True)
                sc = np.ones(2, dtype=np.float64)
            shutil.rmtree(tmp_refs_dir, ignore_errors=True)
            od_dir = work_dir / 'generated' / fold / tag_od
            od_dir.mkdir(parents=True, exist_ok=True)
            map_od = {}
            for nm in infer_subset:
                im = np.rint((map_0[nm].astype(np.float32) + apply_od_scale(map_0[nm], sc)) / 2.0).clip(0, 255).astype(np.uint8)
                map_od[nm] = im
                Image.fromarray(im).save(od_dir / nm)
            record_queue2_eval(
                tag_od, fold, method, upstream_repo, '0shot_halfOD', 5, map_od, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=time.perf_counter() - t_od,
                trainable_params_m=pre_info['trainable_params_m'],
                total_train_params_m=pre_info['total_train_params_m'],
                latency_bs1_ms=lat0_ms, best_val_epoch=pre_info['best_epoch'],
                config_hash=cfg_hash, split_hash=split_hash,
            )
            del gen_eval
            torch.cuda.empty_cache()

            # Stage 3: 20-shot adaptation
            if method in ('pix2pix', 'cyclegan'):
                ad_state, ad_s = adapt_p2p_or_cyclegan_k20(
                    method, fold, pre_info['full_model_state'], fd['adapt_pi_t'], fd['adapt_he_t'],
                    ckpt_fold, adapt_steps=ad_steps, batch_size=bs_m, base_lr=2e-4,
                )
                torch.save({'gen': ad_state, 'config_hash': cfg_hash, 'split_hash': split_hash}, ckpt_fold / f'{tag_20}.pth')
                gen_ad = _get_gen_G(build_p2p_or_cyclegan(method, 2, 2e-4, ckpt_fold, 'eval_ad', is_train=False), method).cuda()
                gen_ad.load_state_dict(ad_state)
            else:
                ad_state, ad_s = adapt_unistainnet_k20(
                    gen_state, pre_info['disc_state'], fd['adapt_pi_t'], fd['adapt_he_t'], ad_uni_14,
                    lpips_fn, adapt_steps=ad_steps, batch_size=bs_m, gen_lr=1e-4, disc_lr=4e-4,
                )
                torch.save({'gen': ad_state, 'config_hash': cfg_hash, 'split_hash': split_hash}, ckpt_fold / f'{tag_20}.pth')
                gen_ad = build_unistainnet_generator().cuda()
                gen_ad.load_state_dict(ad_state)

            pred_fn20 = _make_predict_fn(method, gen_ad)
            lat20_ms = measure_bs1_latency_ms(pred_fn20, test_pi_sub, uni_enc, needs_uni=needs_uni)
            t_inf20 = time.perf_counter()
            map_20 = infer_queue2_to_pngs(
                pred_fn20, test_pi_sub, infer_subset, work_dir / 'generated' / fold / tag_20,
                uni_enc=uni_enc, needs_uni=needs_uni,
            )
            record_queue2_eval(
                tag_20, fold, method, upstream_repo, '20shot', 20, map_20, fd, gt_masks,
                cell_model, lpips_fn, device, work_dir, eval_names, zoom_name,
                wall_seconds=ad_s + (time.perf_counter() - t_inf20),
                trainable_params_m=pre_info['trainable_params_m'],
                total_train_params_m=pre_info['total_train_params_m'],
                latency_bs1_ms=lat20_ms, best_val_epoch=pre_info['best_epoch'],
                config_hash=cfg_hash, split_hash=split_hash,
            )
            del gen_ad
            gc.collect()
            torch.cuda.empty_cache()

            spread_idx = [0, len(eval_names) // 3, (2 * len(eval_names)) // 3, len(eval_names) - 1]
            show_ids = [eval_names[i] for i in spread_idx]
            save_visual_sheet(
                work_dir / 'visuals' / fold / f'{base_tag}.png',
                show_ids,
                fd['pi_rgb_map'],
                fd['gt_rgb_map'],
                {'0shot': map_0, '0shot_halfOD': map_od, '20shot': map_20},
                zoom_name,
            )
            print(f'[VISUAL] Saved visuals/{fold}/{base_tag}.png', flush=True)

    print(f'\n[COMPLETE] Queue-2 preset={args.preset} finished cleanly on GPU {args.gpu_id}.', flush=True)


if __name__ == '__main__':
    main()
