"""Self-contained data loading, zero-adjacency LOMO splits, training, inference, and visual helpers."""
from __future__ import annotations

import copy
import json
import random
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F

from round1_models import Codec, SwadAccumulator
from uni_encoder import UNIEncoder

ORGANS = ('Brain', 'Breast', 'Kidney', 'liver')
H_REF = np.array([0.65, 0.70, 0.29], dtype=np.float64)
E_REF = np.array([0.07, 0.99, 0.11], dtype=np.float64)
STAIN = np.stack([H_REF, E_REF], axis=1)
PINV = np.linalg.pinv(STAIN)
STAIN_CPU = torch.tensor([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]], dtype=torch.float32)
PINV_CPU = torch.linalg.pinv(STAIN_CPU)
_NUM_RE = re.compile(r'(\d+)')


def _file_num(name: str) -> int | None:
    m = _NUM_RE.search(name)
    return int(m.group(1)) if m else None


def pngs(folder: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(folder.glob('*.png')) if p.is_file()}


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)


def tensor_from_paths(paths: Iterable[Path]) -> torch.Tensor:
    array = np.stack([rgb(p) for p in paths])
    return torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def od_concentrations(img: np.ndarray) -> tuple[float, float] | None:
    od = -np.log((img.astype(np.float64) + 1.0) / 256.0).reshape(-1, 3)
    mask = od.sum(axis=1) > 0.15
    if int(mask.sum()) < 100:
        return None
    c = PINV @ od[mask].T
    return float(c[0].mean()), float(c[1].mean())


def mean_od(imgs: list[np.ndarray], label: str) -> np.ndarray:
    vals = [v for im in imgs if (v := od_concentrations(im)) is not None]
    if not vals:
        raise ValueError(f'No valid tissue pixels for {label}')
    return np.asarray(vals, dtype=np.float64).mean(axis=0)


def apply_od_scale(image: np.ndarray, scale: np.ndarray) -> np.ndarray:
    od = -np.log((image.astype(np.float64) + 1.0) / 256.0)
    flat = od.reshape(-1, 3)
    mask = flat.sum(axis=1) > 0.15
    if mask.any():
        coeff = PINV @ flat[mask].T
        coeff = np.clip(coeff * scale[:, None], 0.0, None)
        flat[mask] = (STAIN @ coeff).T
    out = image.copy().reshape(-1, 3)
    out[mask] = np.rint(np.clip(256.0 * np.exp(-flat[mask]) - 1.0, 0, 255)).astype(np.uint8)
    return out.reshape(image.shape)


def load_font(size: int = 12):
    for candidate in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def build_or_load_lomo_splits(data_dir: Path, cache_dir: Path, out_json: Path, seed: int = 2026) -> dict:
    """Create or load deterministic 0-adjacency LOMO splits for all 4 organs."""
    if out_json.exists():
        return json.loads(out_json.read_text(encoding='utf-8'))

    breast_ref = cache_dir / 'breast_symmetric_splits.json'
    if not breast_ref.exists() and (data_dir / 'breast_symmetric_splits.json').exists():
        breast_ref = data_dir / 'breast_symmetric_splits.json'
    breast_info = json.loads(breast_ref.read_text(encoding='utf-8')) if breast_ref.exists() else {}

    rng = random.Random(seed)
    splits: dict[str, dict] = {'seed': seed, 'data_dir': str(data_dir), 'organs': {}}
    for organ in ORGANS:
        train_in = pngs(data_dir / organ / 'train' / 'input')
        train_gt = pngs(data_dir / organ / 'train' / 'target')
        val_in = pngs(data_dir / organ / 'val' / 'input')
        val_gt = pngs(data_dir / organ / 'val' / 'target')
        test_in = pngs(data_dir / organ / 'test' / 'input')
        test_gt = pngs(data_dir / organ / 'test' / 'target')

        train_ids = sorted(train_in.keys() & train_gt.keys())[:480]
        val_ids = sorted(val_in.keys() & val_gt.keys())[:80]
        if organ == 'Breast' and 'breast_test_120' in breast_info:
            test_ids = [n for n in breast_info['breast_test_120'] if n in test_in and n in test_gt]
        else:
            test_ids = sorted(test_in.keys() & test_gt.keys())[:120]

        if len(train_ids) < 100 or len(val_ids) < 30 or len(test_ids) < 100:
            raise ValueError(f'Insufficient pairs for {organ}: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}')

        test_nums = {n for name in test_ids if (n := _file_num(name)) is not None}
        clean_val = [
            v for v in val_ids
            if v not in set(test_ids)
            and ((vn := _file_num(v)) is None or all(abs(vn - tn) > 1 for tn in test_nums))
        ]
        if len(clean_val) < 25:
            raise ValueError(f'{organ}: only {len(clean_val)} val candidates satisfy +/-1 isolation vs test')

        pool = list(clean_val)
        rng.shuffle(pool)
        adapt_20 = sorted(pool[:20])
        refs_5 = sorted(pool[20:25])
        adj_pairs = sum(
            1 for v in clean_val for t in test_ids
            if (_file_num(v) is not None and _file_num(t) is not None and abs(_file_num(v) - _file_num(t)) <= 1)
        )
        splits['organs'][organ] = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'adapt_20': adapt_20,
            'adapt_10': adapt_20[:10],
            'adapt_5': adapt_20[:5],
            'refs_5': refs_5,
            'val_test_adj_pairs': adj_pairs,
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(splits, indent=2), encoding='utf-8')
    return splits


def load_fold_tensors(data_dir: Path, splits: dict, target_organ: str, k_shot: int = 20):
    """Load source train/val tensors, target val few-shot tensors, and target test maps."""
    src_pi, src_he, src_groups = [], [], []
    val_pi, val_he = [], []
    for organ in ORGANS:
        if organ == target_organ:
            continue
        tr_ids = splits['organs'][organ]['train']
        va_ids = splits['organs'][organ]['val']
        src_pi.append(tensor_from_paths(data_dir / organ / 'train' / 'input' / n for n in tr_ids))
        src_he.append(tensor_from_paths(data_dir / organ / 'train' / 'target' / n for n in tr_ids))
        src_groups.extend([organ] * len(tr_ids))
        val_pi.append(tensor_from_paths(data_dir / organ / 'val' / 'input' / n for n in va_ids))
        val_he.append(tensor_from_paths(data_dir / organ / 'val' / 'target' / n for n in va_ids))

    org_sp = splits['organs'][target_organ]
    test_ids = org_sp['test']
    adapt_key = {5: 'adapt_5', 10: 'adapt_10', 20: 'adapt_20'}[k_shot]
    adapt_ids = org_sp[adapt_key]
    refs_ids = org_sp['refs_5']

    test_in_dir = data_dir / target_organ / 'test' / 'input'
    test_gt_dir = data_dir / target_organ / 'test' / 'target'
    val_in_dir = data_dir / target_organ / 'val' / 'input'
    val_gt_dir = data_dir / target_organ / 'val' / 'target'

    extra_vis = []
    if target_organ == 'Breast' and 'file_4278.png' not in test_ids and (test_in_dir / 'file_4278.png').exists():
        extra_vis = ['file_4278.png']
    infer_ids = test_ids + extra_vis

    pi_rgb_map = {n: rgb(test_in_dir / n) for n in infer_ids}
    gt_rgb_map = {n: rgb(test_gt_dir / n) for n in infer_ids}
    refs_od = mean_od([rgb(val_gt_dir / n) for n in refs_ids], f'{target_organ}_refs')

    return {
        'src_pi': torch.cat(src_pi),
        'src_he': torch.cat(src_he),
        'src_groups': src_groups,
        'val_pi': torch.cat(val_pi),
        'val_he': torch.cat(val_he),
        'test_ids': test_ids,
        'infer_ids': infer_ids,
        'adapt_ids': adapt_ids,
        'refs_ids': refs_ids,
        'refs_od': refs_od,
        'pi_rgb_map': pi_rgb_map,
        'gt_rgb_map': gt_rgb_map,
        'test_pi_t': tensor_from_paths(test_in_dir / n for n in infer_ids),
        'adapt_pi_t': tensor_from_paths(val_in_dir / n for n in adapt_ids),
        'adapt_he_t': tensor_from_paths(val_gt_dir / n for n in adapt_ids),
        'refs_pi_t': tensor_from_paths(val_in_dir / n for n in refs_ids),
    }


def grad_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred.diff(dim=-1), target.diff(dim=-1)) + F.l1_loss(pred.diff(dim=-2), target.diff(dim=-2))


def augment_pair(x: torch.Tensor, y: torch.Tensor, py_rng: random.Random):
    if py_rng.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if py_rng.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    k = py_rng.randrange(4)
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
        y = torch.rot90(y, k, dims=(-2, -1))
    return x, y


def apply_view(x: torch.Tensor, v_idx: int) -> torch.Tensor:
    if v_idx == 1:
        return x.flip(-1)
    if v_idx == 2:
        return x.flip(-2)
    if v_idx == 3:
        return x.flip(-1).flip(-2)
    return x


def od_stain_anchor_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pinv = PINV_CPU.to(pred.device)
    cp = torch.einsum('kc,bchw->bkhw', pinv, -torch.log(((pred.float() + 1.0) * 0.5).clamp(1.0 / 256.0, 1.0)))
    ct = torch.einsum('kc,bchw->bkhw', pinv, -torch.log(((target.float() + 1.0) * 0.5).clamp(1.0 / 256.0, 1.0)))
    return F.l1_loss(cp, ct) + 0.5 * F.l1_loss(cp.mean(dim=(2, 3)), ct.mean(dim=(2, 3)))


def stain_aug_source_he(ys: torch.Tensor, yt_pool: torch.Tensor, py_rng: random.Random) -> torch.Tensor:
    device = ys.device
    stain, pinv = STAIN_CPU.to(device), PINV_CPU.to(device)
    od = -torch.log(((ys.float() + 1.0) * 0.5).clamp(1.0 / 256.0, 1.0))
    coeff = torch.einsum('kc,bchw->bkhw', pinv, od)
    od_res = od - torch.einsum('ck,bkhw->bchw', stain, coeff)
    b = ys.size(0)
    scale = torch.ones((b, 2, 1, 1), device=device, dtype=torch.float32)
    if py_rng.random() < 0.5:
        ridx = torch.randint(len(yt_pool), (b,), device=device)
        t_od = -torch.log(((yt_pool[ridx].float() + 1.0) * 0.5).clamp(1.0 / 256.0, 1.0))
        t_coeff = torch.einsum('kc,bchw->bkhw', pinv, t_od)
        s_mask = (od.sum(dim=1, keepdim=True) > 0.15).float()
        t_mask = (t_od.sum(dim=1, keepdim=True) > 0.15).float()
        s_mean = (coeff * s_mask).sum(dim=(2, 3), keepdim=True) / s_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        t_mean = (t_coeff * t_mask).sum(dim=(2, 3), keepdim=True) / t_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        scale = (t_mean / s_mean.clamp_min(0.05)).clamp(0.55, 1.85)
    jitter = torch.empty((b, 2, 1, 1), device=device).uniform_(-0.18, 0.18).exp()
    coeff_aug = (coeff * scale * jitter).clamp(min=0.0)
    od_aug = (torch.einsum('ck,bkhw->bchw', stain, coeff_aug) + od_res).clamp(0.0, 5.5)
    return (2.0 * torch.exp(-od_aug) - 1.0).clamp(-1.0, 1.0)


def pi_acq_jitter(x: torch.Tensor, py_rng: random.Random, strength: float = 0.08) -> torch.Tensor:
    b = x.size(0)
    contrast = torch.empty((b, 1, 1, 1), device=x.device).uniform_(1.0 - strength, 1.0 + strength)
    shift = torch.empty((b, 1, 1, 1), device=x.device).uniform_(-0.5 * strength, 0.5 * strength)
    mean = x.mean(dim=(2, 3), keepdim=True)
    out = (x - mean) * contrast + mean + shift
    if py_rng.random() < 0.5:
        out = out + torch.randn_like(out) * (0.15 * strength)
    return out.clamp(-1.0, 1.0)


@torch.no_grad()
def encode_all(codec: Codec, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([codec.encode(v.cuda()).float().cpu() for v in x.split(32)])


def train_fold_codec(
    src_pi: torch.Tensor,
    src_he: torch.Tensor,
    steps: int = 8800,
    seed: int = 7000,
    detail_weight: float = 0.15,
    noise_max: float = 0.08,
) -> tuple[Codec, dict]:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    cuda_rng = torch.Generator(device='cuda').manual_seed(seed)
    codec = Codec().cuda().train()
    opt = torch.optim.AdamW(codec.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(1, steps + 1):
        idx = torch.randint(len(src_pi), (8,), generator=rng)
        img = torch.cat([src_pi[idx], src_he[idx]]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = codec.encode(img)
            if noise_max > 0.0:
                z_std = z.detach().float().std().clamp_min(1e-6)
                sigma = torch.empty((z.size(0), 1, 1, 1), device=z.device).uniform_(0.0, noise_max, generator=cuda_rng) * z_std
                eps = torch.randn(z.shape, dtype=z.dtype, device=z.device, generator=cuda_rng)
                z_dec = z + sigma.to(z.dtype) * eps
            else:
                z_dec = z
            rec = codec.decode(z_dec)
        delta = rec.float() - img
        loss = delta.abs().mean() + 0.001 * z.float().square().mean()
        if detail_weight:
            loss = loss + detail_weight * (delta.diff(dim=-2).abs().mean() + delta.diff(dim=-1).abs().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
    codec.eval().requires_grad_(False)
    return codec, {
        'codec_steps': steps,
        'detail_weight': detail_weight,
        'noise_max': noise_max,
        'last_loss': last_loss,
        'codec_seconds': round(time.perf_counter() - t0, 2),
    }


@torch.no_grad()
def precompute_uni_4views(uni_enc: UNIEncoder, images_cpu: torch.Tensor) -> torch.Tensor:
    n = len(images_cpu)
    out = torch.empty((4, n, 196, 1024), dtype=torch.float16, device='cpu')
    for v_idx in range(4):
        chunks = []
        for i in range(0, n, 32):
            bx = apply_view(images_cpu[i:i + 32].cuda(), v_idx)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tok = uni_enc(bx)[:, 1:].half().cpu()
            chunks.append(tok)
        out[v_idx] = torch.cat(chunks, dim=0)
    return out


def pretrain_flow(
    tag: str,
    flow: nn.Module,
    codec: Codec,
    center: float,
    scale: float,
    src_lat_pi: torch.Tensor,
    src_lat_he: torch.Tensor,
    src_pi: torch.Tensor,
    src_he: torch.Tensor,
    src_uni_4v: torch.Tensor | None,
    val_lat_pi: torch.Tensor,
    val_lat_he: torch.Tensor,
    val_uni_v0: torch.Tensor | None,
    steps: int = 3600,
    batch: int = 8,
    seed: int = 7102,
    use_on_policy_rollout: bool = False,
    rollout_steps: int = 4,
    lpips_fn=None,
) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    rng = torch.Generator(device='cuda').manual_seed(seed)
    cpu_rng = torch.Generator().manual_seed(seed)
    py_rng = random.Random(seed)
    flow.cuda().train()
    opt = torch.optim.AdamW(flow.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)
    swad = SwadAccumulator({'flow': flow}, start_step=max(1, int(0.5 * steps)), every=max(1, min(20, steps // 10)))

    val_pi_c, val_he_c = val_lat_pi.cuda(), val_lat_he.cuda()
    val_u_c = val_uni_v0.cuda().float() if val_uni_v0 is not None else None
    val_t_fixed = torch.linspace(0.1, 0.9, len(val_pi_c), device='cuda')
    best_val, best_step, val_history = 1e9, 0, []
    log_every = max(1, min(900, steps // 4))
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        do_rollout = use_on_policy_rollout and step >= max(1, steps // 2)
        v_idx = py_rng.randrange(4)
        ids = torch.randint(len(src_pi), (batch,), generator=cpu_rng)
        if v_idx == 0:
            z0, z1 = src_lat_pi[ids].cuda(), src_lat_he[ids].cuda()
            y_tgt = src_he[ids].cuda() if do_rollout else None
        else:
            x_aug = apply_view(src_pi[ids].cuda(), v_idx)
            y_aug = apply_view(src_he[ids].cuda(), v_idx)
            y_tgt = y_aug if do_rollout else None
            with torch.no_grad():
                z0 = (codec.encode(x_aug).float() - center) / scale
                z1 = (codec.encode(y_aug).float() - center) / scale
        uni_tok = src_uni_4v[v_idx, ids].cuda().float() if src_uni_4v is not None else None
        t = torch.rand(batch, generator=rng, device='cuda')
        mixed = torch.lerp(z0, z1, t[:, None, None, None])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred_v = flow(mixed, z0, t, uni_tok)
        loss = F.mse_loss(pred_v.float(), z1 - z0)
        if do_rollout and y_tgt is not None:
            z_roll = z0
            dt = 1.0 / float(rollout_steps)
            with torch.no_grad():
                for k in range(rollout_steps - 1):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z_roll = z_roll + flow(z_roll, z0, z_roll.new_full((batch,), k * dt), uni_tok).float() * dt
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v_last = flow(z_roll, z0, z_roll.new_full((batch,), (rollout_steps - 1) * dt), uni_tok)
                z_hat_1 = z_roll + v_last.float() * dt
                pred_img = codec.decode(z_hat_1 * scale + center)
            loss = (
                loss
                + 0.15 * grad_l1(pred_img.float(), y_tgt)
                + 0.50 * F.l1_loss(pred_img.float(), y_tgt)
                + 0.15 * od_stain_anchor_loss(pred_img.float(), y_tgt)
            )
            if lpips_fn is not None:
                loss = loss + 0.20 * lpips_fn((pred_img.float().clamp(-1, 1) + 1.0) * 0.5, (y_tgt.clamp(-1, 1) + 1.0) * 0.5).mean()
        opt.zero_grad(set_to_none=True)
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
                    vu = val_u_c[j:j + 32] if val_u_c is not None else None
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        vv = flow(torch.lerp(vp, vh, vt[:, None, None, None]), vp, vt, vu)
                    v_losses.append(F.mse_loss(vv.float(), vh - vp).item())
                vmse = float(np.mean(v_losses))
            val_history.append({'step': step, 'val_mse': round(vmse, 5)})
            if vmse < best_val:
                best_val, best_step = vmse, step
            print(f'[{tag}] pretrain {step}/{steps}: train={loss.item():.4f}, val_mse={vmse:.5f} ({time.perf_counter()-t0:.1f}s)', flush=True)
            flow.train()

    swad.apply()
    return {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}, {
        'best_val_mse': best_val,
        'best_val_step': best_step,
        'val_history': val_history,
        'pretrain_seconds': round(time.perf_counter() - t0, 2),
    }


def adapt_flow(
    tag: str,
    strategy: str,
    flow: nn.Module,
    swad_st: dict,
    init_codec: Codec,
    center: float,
    scale: float,
    src_pools: list[tuple[torch.Tensor, torch.Tensor]],
    tgt_pi: torch.Tensor,
    tgt_he: torch.Tensor,
    lpips_fn,
    uni_enc: UNIEncoder | None,
    steps: int = 1000,
    batch: int = 4,
    seed: int = 7202,
    decoder_lr_mult: float = 0.1,
    rollout_steps: int = 4,
) -> tuple[nn.Module, Codec, dict]:
    """Adapt pretrained flow via 'm6' (replay + SWAD + LPIPS + OD anchor) or 'naive_ft' ablation."""
    if strategy not in ('m6', 'naive_ft'):
        raise ValueError(f'Unknown adapt strategy: {strategy}')
    torch.manual_seed(seed)
    rng = torch.Generator(device='cuda').manual_seed(seed)
    py_rng = random.Random(seed)
    flow.load_state_dict(swad_st)
    flow.cuda().train()
    codec = copy.deepcopy(init_codec).cuda()
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    codec.encoder.eval()

    freeze_dec = decoder_lr_mult <= 0.0
    for p in codec.decoder.parameters():
        p.requires_grad_(not freeze_dec)
    if freeze_dec:
        codec.decoder.eval()
        params = list(flow.parameters())
        opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
        swad_mods = {'flow': flow}
    else:
        codec.decoder.train()
        params = list(flow.parameters()) + list(codec.decoder.parameters())
        opt = torch.optim.AdamW(
            [
                {'params': list(flow.parameters()), 'lr': 2e-4},
                {'params': list(codec.decoder.parameters()), 'lr': 2e-4 * decoder_lr_mult},
            ],
            weight_decay=1e-4,
        )
        swad_mods = {'flow': flow, 'decoder': codec.decoder}

    use_swad = strategy == 'm6'
    swad = SwadAccumulator(swad_mods, start_step=max(1, int(0.4 * steps)), every=max(1, min(10, steps // 10)))
    t_pi, t_he = tgt_pi.cuda(), tgt_he.cuda()
    log_every = max(1, min(500, steps // 2))
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        ti = torch.randint(len(t_pi), (batch,), generator=rng, device='cuda')
        xt, yt = augment_pair(t_pi[ti], t_he[ti], py_rng)
        xt_in = pi_acq_jitter(xt, py_rng, strength=0.05) if strategy == 'm6' else xt

        with torch.no_grad():
            z0_t = (codec.encode(xt_in).float() - center) / scale
            z1_t = (codec.encode(yt).float() - center) / scale
            ut_t = None
            if uni_enc is not None and getattr(flow, 'use_uni_2d', False):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    ut_t = uni_enc(xt_in)[:, 1:].float()

        tt = torch.rand(batch, generator=rng, device='cuda')
        dt = 1.0 / float(rollout_steps)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            v_t = flow(torch.lerp(z0_t, z1_t, tt[:, None, None, None]), z0_t, tt, ut_t)
            z = z0_t
            for k in range(rollout_steps):
                z = z + flow(z, z0_t, z.new_full((batch,), k * dt), ut_t) * dt
            pred_t = codec.decode(z * scale + center)

        loss_vel = F.mse_loss(v_t.float(), z1_t - z0_t)
        loss_img = 4.0 * F.l1_loss(pred_t.float(), yt) + 2.0 * grad_l1(pred_t.float(), yt)
        loss_lp = lpips_fn((pred_t.float().clamp(-1, 1) + 1.0) * 0.5, (yt.clamp(-1, 1) + 1.0) * 0.5).mean()
        loss = loss_vel + loss_img + 0.50 * loss_lp

        if strategy == 'm6':
            op_pi, op_he = src_pools[py_rng.randrange(len(src_pools))]
            si = torch.randint(len(op_pi), (batch,), generator=rng, device='cuda')
            xs, ys = augment_pair(op_pi[si], op_he[si], py_rng)
            xs_in = pi_acq_jitter(xs, py_rng, strength=0.08)
            ys_in = stain_aug_source_he(ys, t_he, py_rng)
            with torch.no_grad():
                z0_s = (codec.encode(xs_in).float() - center) / scale
                z1_s = (codec.encode(ys_in).float() - center) / scale
                ut_s = None
                if uni_enc is not None and getattr(flow, 'use_uni_2d', False):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        ut_s = uni_enc(xs_in)[:, 1:].float()
            ts = torch.rand(batch, generator=rng, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v_s = flow(torch.lerp(z0_s, z1_s, ts[:, None, None, None]), z0_s, ts, ut_s)
                rec_s = codec.decode(z1_s * scale + center) if not freeze_dec else None
            l_s = 0.6 * F.mse_loss(v_s.float(), z1_s - z0_s) + (
                0.0 if freeze_dec else 0.6 * F.l1_loss(rec_s.float(), ys_in)
            )
            loss = loss + 0.25 * od_stain_anchor_loss(pred_t.float(), yt) + 0.40 * l_s

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.5)
        opt.step()
        if use_swad:
            swad.maybe_update(step, steps)

        if step % log_every == 0 or step == steps:
            print(f'[{tag}] {strategy} {step}/{steps}: img={loss_img.item():.4f} ({time.perf_counter()-t0:.1f}s)', flush=True)

    if use_swad:
        swad.apply()
    return flow.eval().cpu(), codec.eval().cpu(), {
        'adapt_strategy': strategy,
        'decoder_lr_mult': decoder_lr_mult,
        'adapt_seconds': round(time.perf_counter() - t0, 2),
    }


@torch.no_grad()
def infer_to_pngs(
    flow: nn.Module,
    codec: Codec,
    pi_cpu: torch.Tensor,
    uni_enc: UNIEncoder | None,
    center: float,
    scale: float,
    names: list[str],
    out_dir: Path,
    steps: int = 4,
) -> dict[str, np.ndarray]:
    """Run flow inference and save 256x256 RGB PNGs cleanly to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    flow.cuda().eval()
    codec.cuda().eval()
    use_uni = bool(getattr(flow, 'use_uni_2d', False)) and uni_enc is not None
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(pi_cpu), 16):
        bx = pi_cpu[i:i + 16].cuda()
        uni_tok = None
        if use_uni:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                uni_tok = uni_enc(bx)[:, 1:].float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z0 = (codec.encode(bx).float() - center) / scale
        z = z0
        for k in range(steps):
            t_in = z.new_full((len(z),), k / float(steps))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v = flow(z, z0, t_in, uni_tok)
            z = z + v.float() / float(steps)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            rec = codec.decode(z * scale + center).float().clamp(-1, 1)
        arr = ((rec + 1.0) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        batch_names = names[i:i + 16]
        assert len(batch_names) == len(arr), f'Length mismatch: {len(batch_names)} != {len(arr)}'
        for nm, im in zip(batch_names, arr):
            out[nm] = im
            Image.fromarray(im).save(out_dir / nm)
    return out


def save_visual_sheet(
    out_path: Path,
    ids: list[str],
    pi_map: dict[str, np.ndarray],
    gt_map: dict[str, np.ndarray],
    variant_maps: dict[str, dict[str, np.ndarray]],
    zoom_name: str,
) -> None:
    """Save crisp visual comparison sheet with 1:1 nearest zoom on sparse margin region."""
    cols = [('PI', pi_map)] + list(variant_maps.items()) + [('GT H&E', gt_map)]
    show_ids = ids[:4]
    wc, hc = 176, 200
    view = Image.new('RGB', (len(cols) * (wc + 6) + 8, 34 + (len(show_ids) + 1) * (hc + 6)), 'white')
    ink = ImageDraw.Draw(view)
    font = load_font(11)
    for ci, (cname, _) in enumerate(cols):
        ink.text((6 + ci * (wc + 6), 8), cname, fill=(20, 20, 20), font=font)
    for ri, nm in enumerate(show_ids):
        y0 = 32 + ri * (hc + 6)
        for ci, (_, cmap) in enumerate(cols):
            x0 = 6 + ci * (wc + 6)
            ink.text((x0, y0), nm, fill=(40, 40, 40), font=font)
            view.paste(Image.fromarray(cmap[nm]).resize((wc, wc), Image.Resampling.BILINEAR), (x0, y0 + 18))

    y0 = 32 + len(show_ids) * (hc + 6)
    for ci, (_, cmap) in enumerate(cols):
        x0 = 6 + ci * (wc + 6)
        ink.text((x0, y0), f'{zoom_name} ZOOM [40:168,0:128]', fill=(150, 20, 20), font=font)
        crop = cmap[zoom_name][40:168, 0:128]
        view.paste(Image.fromarray(crop).resize((wc, wc), Image.Resampling.NEAREST), (x0, y0 + 18))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    view.save(out_path)
