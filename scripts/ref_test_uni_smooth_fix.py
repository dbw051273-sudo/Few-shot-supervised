"""Controlled validation of 3x3 spatial smoother (+ optional PI gating) to eliminate
14x14 ViT patch grid artifacts in u1_uni2d_spatial, plus uint16 Cellpose Nuclei F1@0.5
evaluation across held_240 and zero-adjacency breast_test_120.
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F
from cellpose import models
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

REPO = Path(r'C:\Users\felix\Desktop\bbdm-swin1')
V1_SCRIPTS = Path(r'C:\Users\felix\Desktop\03_实验消融与可视化\UniPrior_四器官精筛试验集\scripts')
V2_ROOT = Path(r'C:\Users\felix\Desktop\03_实验消融与可视化\UniPrior_四器官洁净均衡集_v2')
sys.path.insert(0, str(V2_ROOT / 'scripts'))
sys.path.insert(0, str(V1_SCRIPTS))
sys.path.append(str(REPO))

from model.BrownianBridge.uni_encoder import UNIEncoder  # noqa: E402
from fusion_pilot import Block, Codec, encode, load_source, tensor  # noqa: E402
from pilot_eval import apply_scale, pngs, rgb  # noqa: E402
from uni_ablation_v2 import (  # noqa: E402
    SwadAccumulator,
    UniFlow,
    _mean_od,
    apply_view,
    augment_pair,
    grad_l1,
    infer_uniflow,
    od_stain_anchor_loss,
    pi_acq_jitter,
    precompute_uni_4views,
    stain_aug_source_he,
)
from v2_unified_eval import PAPER_CORE_KEYS, evaluate_split, load_gt_instance_masks  # noqa: E402


class SmoothUniFlow(nn.Module):
    """UniFlow with 3x3 replicate-padded GroupNorm spatial smoother (and optional PI gate)."""

    def __init__(self, dim: int = 128, gated: bool = False):
        super().__init__()
        patch = 4
        self.dim = dim
        self.use_uni_2d = True
        self.gated = gated
        self.state_proj = nn.Conv2d(12, dim, patch, stride=patch)
        self.pi_proj = nn.Conv2d(12, dim, patch, stride=patch)
        self.position = nn.Parameter(torch.randn(1, 16 * 16, dim) * 0.02)
        self.roles = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(32, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer('frequency', torch.exp(torch.linspace(0.0, 6.0, 16)))
        self.blocks = nn.ModuleList(Block(dim) for _ in range(4))
        self.uni_proj = nn.Sequential(
            nn.Conv2d(1024, dim, 1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
        )
        self.uni_zero1 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode='replicate')
        self.uni_zero2 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode='replicate')
        nn.init.zeros_(self.uni_zero1.weight)
        nn.init.zeros_(self.uni_zero1.bias)
        nn.init.zeros_(self.uni_zero2.weight)
        nn.init.zeros_(self.uni_zero2.bias)
        if gated:
            self.uni_gate = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, padding=1, padding_mode='replicate'),
                nn.SiLU(),
                nn.Conv2d(dim // 4, 1, 1),
            )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 12 * patch * patch))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _uni_grid(self, uni_tokens: torch.Tensor) -> torch.Tensor:
        b = uni_tokens.shape[0]
        feat = uni_tokens.transpose(1, 2).reshape(b, 1024, 14, 14)
        feat = F.interpolate(feat, size=(16, 16), mode='bilinear', align_corners=False)
        return self.uni_proj(feat)

    def forward(self, current, source, t, uni_tokens: torch.Tensor | None = None):
        angle = t[:, None] * self.frequency[None]
        time_emb = self.time(torch.cat((angle.sin(), angle.cos()), dim=-1))[:, None]
        a = self.state_proj(current).flatten(2).transpose(1, 2) + self.position + self.roles[0] + time_emb
        b = self.pi_proj(source).flatten(2).transpose(1, 2) + self.position + self.roles[1] + time_emb
        a = self.blocks[0](a)
        b = self.blocks[1](b)
        hidden = torch.cat((a, b), dim=1)
        n_tok = a.shape[1]
        if uni_tokens is not None:
            u_feat = self._uni_grid(uni_tokens)
            if self.gated:
                b_map = b.transpose(1, 2).reshape(-1, self.dim, 16, 16)
                u_feat = u_feat * torch.sigmoid(self.uni_gate(b_map))
            u1 = self.uni_zero1(u_feat).flatten(2).transpose(1, 2)
            hidden = torch.cat((hidden[:, :n_tok] + u1, hidden[:, n_tok:]), dim=1)
        hidden = self.blocks[2](hidden)
        if uni_tokens is not None:
            u2 = self.uni_zero2(u_feat).flatten(2).transpose(1, 2)
            hidden = torch.cat((hidden[:, :n_tok] + u2, hidden[:, n_tok:]), dim=1)
        hidden = self.blocks[3](hidden)
        velocity = self.head(hidden[:, :n_tok]).transpose(1, 2).reshape(-1, 192, 16, 16)
        return F.pixel_shuffle(velocity, 4)


def train_smooth_pretrain(arm_name: str, flow: nn.Module, codec: Codec, center: float, scale: float,
                        src_lat_pi, src_lat_he, src_pi, src_he, src_uni_4v,
                        val_lat_pi, val_lat_he, val_uni_v0, steps: int = 3600, batch: int = 8, seed: int = 7102):
    torch.manual_seed(seed)
    rng = torch.Generator(device='cuda').manual_seed(seed)
    cpu_rng = torch.Generator().manual_seed(seed)
    py_rng = random.Random(seed)
    flow.cuda().train()
    opt = torch.optim.AdamW(flow.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)
    swad = SwadAccumulator({'flow': flow}, start_step=int(0.5 * steps), every=20)
    val_pi_c, val_he_c, val_u_c = val_lat_pi.cuda(), val_lat_he.cuda(), val_uni_v0.cuda().float()
    val_t_fixed = torch.linspace(0.1, 0.9, len(val_pi_c), device='cuda')
    best_val, best_step, val_history = 1e9, 0, []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        v_idx = py_rng.randrange(4)
        ids = torch.randint(len(src_pi), (batch,), generator=cpu_rng)
        if v_idx == 0:
            z0, z1 = src_lat_pi[ids].cuda(), src_lat_he[ids].cuda()
        else:
            x_aug = apply_view(src_pi[ids].cuda(), v_idx)
            y_aug = apply_view(src_he[ids].cuda(), v_idx)
            with torch.no_grad():
                z0 = (codec.encode(x_aug).float() - center) / scale
                z1 = (codec.encode(y_aug).float() - center) / scale
        uni_tok = src_uni_4v[v_idx, ids].cuda().float()
        t = torch.rand(batch, generator=rng, device='cuda')
        mixed = torch.lerp(z0, z1, t[:, None, None, None])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred_v = flow(mixed, z0, t, uni_tok)
        loss = F.mse_loss(pred_v.float(), z1 - z0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), 1.5)
        opt.step()
        sched.step()
        swad.maybe_update(step, steps)
        if step % 900 == 0 or step == steps:
            flow.eval()
            with torch.no_grad():
                v_losses = []
                for j in range(0, len(val_pi_c), 32):
                    vp, vh, vt, vu = val_pi_c[j:j + 32], val_he_c[j:j + 32], val_t_fixed[j:j + 32], val_u_c[j:j + 32]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        vv = flow(torch.lerp(vp, vh, vt[:, None, None, None]), vp, vt, vu)
                    v_losses.append(F.mse_loss(vv.float(), vh - vp).item())
                vmse = float(np.mean(v_losses))
            val_history.append({'step': step, 'val_mse': round(vmse, 5)})
            if vmse < best_val:
                best_val, best_step = vmse, step
            print(f'[{arm_name}] step {step}/{steps}: train_loss={loss.item():.4f}, val_mse={vmse:.5f} ({time.perf_counter()-t0:.1f}s)', flush=True)
            flow.train()

    swad.apply()
    return {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}, {
        'best_val_mse': best_val, 'best_val_step': best_step, 'val_history': val_history, 'pretrain_seconds': time.perf_counter() - t0,
    }


def adapt_smooth_m6(arm_name: str, flow: nn.Module, swad_st: dict, init_codec: Codec, center: float, scale: float,
                    src_pools, tgt_pi, tgt_he, lpips_fn, uni_enc: UNIEncoder, steps: int = 1000, batch: int = 4, seed: int = 7202):
    torch.manual_seed(seed)
    rng = torch.Generator(device='cuda').manual_seed(seed)
    py_rng = random.Random(seed)
    flow.load_state_dict(swad_st)
    flow.cuda().train()
    codec = copy.deepcopy(init_codec).cuda()
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    codec.encoder.eval()
    codec.decoder.train()

    params = list(flow.parameters()) + list(codec.decoder.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
    swad = SwadAccumulator({'flow': flow, 'decoder': codec.decoder}, start_step=int(0.4 * steps), every=10)
    t_pi, t_he = tgt_pi.cuda(), tgt_he.cuda()
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        ti = torch.randint(len(t_pi), (batch,), generator=rng, device='cuda')
        xt, yt = augment_pair(t_pi[ti], t_he[ti], py_rng)
        xt_in = pi_acq_jitter(xt, py_rng, strength=0.05)

        op_pi, op_he = src_pools[py_rng.randrange(len(src_pools))]
        si = torch.randint(len(op_pi), (batch,), generator=rng, device='cuda')
        xs, ys = augment_pair(op_pi[si], op_he[si], py_rng)
        xs_in = pi_acq_jitter(xs, py_rng, strength=0.08)
        ys_in = stain_aug_source_he(ys, t_he, py_rng)

        with torch.no_grad():
            z0_t = (codec.encode(xt_in).float() - center) / scale
            z1_t = (codec.encode(yt).float() - center) / scale
            z0_s = (codec.encode(xs_in).float() - center) / scale
            z1_s = (codec.encode(ys_in).float() - center) / scale
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ut_t = uni_enc(xt_in)[:, 1:].float()
                ut_s = uni_enc(xs_in)[:, 1:].float()

        tt = torch.rand(batch, generator=rng, device='cuda')
        ts = torch.rand(batch, generator=rng, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            v_s = flow(torch.lerp(z0_s, z1_s, ts[:, None, None, None]), z0_s, ts, ut_s)
            rec_s = codec.decode(z1_s * scale + center)
            v_t = flow(torch.lerp(z0_t, z1_t, tt[:, None, None, None]), z0_t, tt, ut_t)
            z = z0_t
            for k in range(4):
                z = z + flow(z, z0_t, z.new_full((batch,), k / 4.0), ut_t) / 4.0
            pred_t = codec.decode(z * scale + center)

        l_s = 0.6 * F.mse_loss(v_s.float(), z1_s - z0_s) + 0.6 * F.l1_loss(rec_s.float(), ys_in)
        loss_vel = F.mse_loss(v_t.float(), z1_t - z0_t)
        loss_img = 4.0 * F.l1_loss(pred_t.float(), yt) + 2.0 * grad_l1(pred_t.float(), yt)
        loss_lp = lpips_fn((pred_t.float().clamp(-1, 1) + 1.0) * 0.5, (yt.clamp(-1, 1) + 1.0) * 0.5).mean()
        loss_od = od_stain_anchor_loss(pred_t.float(), yt)
        loss = loss_vel + loss_img + 0.50 * loss_lp + 0.25 * loss_od + 0.40 * l_s

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.5)
        opt.step()
        swad.maybe_update(step, steps)
        if step % 500 == 0 or step == steps:
            print(f'[{arm_name}_adapt] step {step}/{steps}: img={loss_img.item():.4f} ({time.perf_counter()-t0:.1f}s)', flush=True)

    swad.apply()
    return flow.eval().cpu(), codec.eval().cpu(), {'adapt_seconds': time.perf_counter() - t0}


def background_grid_energy(img: np.ndarray) -> float:
    """Measure local high-frequency patch-grid variation on sparse left margin of file_4278 [0:256, 0:112]."""
    crop = img[:, :112].astype(np.float32)
    gray = 0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
    lap = np.abs(gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1])
    return float(lap.mean())


def main():
    t_all = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    prev_dir = V2_ROOT / 'pilot' / 'uni_ablation_v2'
    out = V2_ROOT / 'pilot' / 'uni_smooth_fix'
    ckpt_dir = out / 'checkpoints'
    csv_dir = out / 'per_image'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads((V2_ROOT / 'breast_symmetric_splits.json').read_text(encoding='utf-8'))
    held = splits['held_240']
    sym120 = splits['breast_test_120']
    adapt = splits['adapt_20']
    refs = splits['refs_5']

    src_pi, src_he, src_groups = load_source(V2_ROOT, 'train')
    val_pi, val_he, _ = load_source(V2_ROOT, 'val')
    organ_pools = [
        (src_pi[[i for i, g in enumerate(src_groups) if g == o]].cuda(),
         src_he[[i for i, g in enumerate(src_groups) if g == o]].cuda())
        for o in ('liver', 'Kidney', 'Brain')
    ]

    held_in = pngs(V2_ROOT / 'Breast' / 'test' / 'input')
    held_gt = pngs(V2_ROOT / 'Breast' / 'test' / 'target')
    off_in = pngs(REPO / 'official_eval' / 'condition')
    off_gt = pngs(REPO / 'official_eval' / 'ground_truth')
    ref_paths = pngs(V2_ROOT / 'Breast' / 'calibration' / 'target')
    ref_od = _mean_od([rgb(ref_paths[n]) for n in refs], 'refs')

    pi_rgb_map = {n: rgb(held_in[n]) for n in held}
    gt_rgb_map = {n: rgb(held_gt[n]) for n in held}
    held_pi_t = tensor([held_in[n] for n in held])
    adapt_pi_t = tensor([off_in[n] for n in adapt])
    adapt_he_t = tensor([off_gt[n] for n in adapt])

    codec = Codec().cuda()
    codec.load_state_dict(torch.load(prev_dir / 'codec_v2.pth', map_location='cuda', weights_only=True))
    codec.eval()
    with torch.no_grad():
        lat_pi = encode(codec, src_pi).cpu()
        center, scale = float(lat_pi.mean()), float(lat_pi.std())
        src_lat_pi = (lat_pi - center) / scale
        src_lat_he = (encode(codec, src_he).cpu() - center) / scale
        val_lat_pi = (encode(codec, val_pi).cpu() - center) / scale
        val_lat_he = (encode(codec, val_he).cpu() - center) / scale

    print('Loading frozen MahmoodLab/UNI & precomputing 4-view tokens...', flush=True)
    uni_enc = UNIEncoder(freeze=True).cuda().eval()
    src_uni_4v = precompute_uni_4views(uni_enc, src_pi)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        val_uni_v0 = torch.cat([uni_enc(val_pi[i:i + 32].cuda())[:, 1:].half().cpu() for i in range(0, len(val_pi), 32)], dim=0)

    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True, reduction='none').cuda().eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    arms_imgs = {}
    logs = {}
    codec_cuda = copy.deepcopy(codec).cuda().eval()

    # 1) Load existing u0_base & u1_uni2d_spatial checkpoints without retraining
    for arm_name, use_u2d in (('u0_base', False), ('u1_uni2d_1x1', True)):
        ckpt_base = 'u0_base' if arm_name == 'u0_base' else 'u1_uni2d_spatial'
        pre = torch.load(prev_dir / 'checkpoints' / f'{ckpt_base}_pretrain.pth', map_location='cpu', weights_only=True)
        fl = UniFlow(dim=128, use_uni_2d=use_u2d).cuda().eval()
        fl.load_state_dict(pre['swad'])
        zs = infer_uniflow(fl, codec_cuda, held_pi_t, uni_enc if use_u2d else None, center, scale, held, use_u2d, steps=4)
        zs_ad = infer_uniflow(fl, codec_cuda, adapt_pi_t, uni_enc if use_u2d else None, center, scale, adapt, use_u2d, steps=4)
        sc = ref_od / _mean_od([zs_ad[n] for n in adapt], arm_name)
        arms_imgs[f'{arm_name}_0shot'] = zs
        arms_imgs[f'{arm_name}_0shot_halfOD'] = {n: np.rint((zs[n].astype(np.float32) + apply_scale(zs[n], sc)) / 2.0).clip(0, 255).astype(np.uint8) for n in held}
        ad_ck = torch.load(prev_dir / 'checkpoints' / f'{ckpt_base}_adapt.pth', map_location='cpu', weights_only=True)
        fl.load_state_dict(ad_ck['flow'])
        co_ad = Codec().cuda().eval()
        co_ad.load_state_dict(ad_ck['codec'])
        arms_imgs[f'{arm_name}_20shot'] = infer_uniflow(fl, co_ad, held_pi_t, uni_enc if use_u2d else None, center, scale, held, use_u2d, steps=4)
        del fl, co_ad

    # 2) Train & evaluate new 3x3 smoothed (+ optional gated) UNI 2D arms
    for arm_name, gated in (('u1_smooth3x3', False), ('u1_smooth3x3_gated', True)):
        fl_tr = SmoothUniFlow(dim=128, gated=gated)
        swad_st, pinfo = train_smooth_pretrain(
            arm_name, fl_tr, codec_cuda, center, scale,
            src_lat_pi, src_lat_he, src_pi, src_he, src_uni_4v,
            val_lat_pi, val_lat_he, val_uni_v0, steps=3600, batch=8, seed=7102,
        )
        torch.save({'swad': swad_st}, ckpt_dir / f'{arm_name}_pretrain.pth')

        fl = SmoothUniFlow(dim=128, gated=gated).cuda().eval()
        fl.load_state_dict(swad_st)
        zs = infer_uniflow(fl, codec_cuda, held_pi_t, uni_enc, center, scale, held, True, steps=4)
        zs_ad = infer_uniflow(fl, codec_cuda, adapt_pi_t, uni_enc, center, scale, adapt, True, steps=4)
        sc = ref_od / _mean_od([zs_ad[n] for n in adapt], arm_name)
        arms_imgs[f'{arm_name}_0shot'] = zs
        arms_imgs[f'{arm_name}_0shot_halfOD'] = {n: np.rint((zs[n].astype(np.float32) + apply_scale(zs[n], sc)) / 2.0).clip(0, 255).astype(np.uint8) for n in held}
        del fl

        fl_ad, co_ad, ad_info = adapt_smooth_m6(
            arm_name, SmoothUniFlow(dim=128, gated=gated), swad_st, codec, center, scale,
            organ_pools, adapt_pi_t, adapt_he_t, lpips_fn, uni_enc, steps=1000, batch=4, seed=7202,
        )
        torch.save({'flow': fl_ad.state_dict(), 'codec': co_ad.state_dict()}, ckpt_dir / f'{arm_name}_adapt.pth')
        fl_ad, co_ad = fl_ad.cuda().eval(), co_ad.cuda().eval()
        arms_imgs[f'{arm_name}_20shot'] = infer_uniflow(fl_ad, co_ad, held_pi_t, uni_enc, center, scale, held, True, steps=4)
        logs[arm_name] = {**pinfo, **ad_info}
        del fl_ad, co_ad, fl_tr
        torch.cuda.empty_cache()

    del uni_enc, codec_cuda, src_uni_4v, val_uni_v0
    torch.cuda.empty_cache()

    device = torch.device('cuda')
    gt_masks = load_gt_instance_masks(V2_ROOT / 'eval_cache' / 'gt_nuclei_instances_256.npz', 'Breast')
    cell_model = models.CellposeModel(gpu=True, model_type='nuclei')

    results_held = {}
    results_sym120 = {}
    grid_energy = {}

    for arm_key, img_map in arms_imgs.items():
        r_h = evaluate_split(held, img_map, pi_rgb_map, gt_rgb_map, gt_masks, cell_model, lpips_fn, device, csv_dir / f'{arm_key}_held240.csv')
        r_s = evaluate_split(sym120, img_map, pi_rgb_map, gt_rgb_map, gt_masks, cell_model, lpips_fn, device, csv_dir / f'{arm_key}_sym120.csv')
        results_held[arm_key] = r_h
        results_sym120[arm_key] = r_s
        grid_energy[arm_key] = round(background_grid_energy(img_map['file_4278.png']), 4)
        print(
            f"[HELD240] {arm_key:26s} | PSNR={r_h['psnr']:5.2f} SSIM={r_h['ssim']:.4f} LPIPS={r_h['lpips']:.4f} "
            f"KID={r_h['kid']:.5f} Dice={r_h['nuclei_dice']:.4f} F1={r_h['nuclei_f1_05']:.4f} "
            f"(FID={r_h['fid']:5.2f} PCC_pi={r_h['pcc_pi']:.4f} cnt_r={r_h['nuclei_count_r']:.4f} bg_lap={grid_energy[arm_key]:.2f})",
            flush=True,
        )
        print(
            f"[SYM120 ] {arm_key:26s} | PSNR={r_s['psnr']:5.2f} SSIM={r_s['ssim']:.4f} LPIPS={r_s['lpips']:.4f} "
            f"KID={r_s['kid']:.5f} Dice={r_s['nuclei_dice']:.4f} F1={r_s['nuclei_f1_05']:.4f}",
            flush=True,
        )

    payload = {
        'paper_core_keys': list(PAPER_CORE_KEYS),
        'held_240': results_held,
        'breast_test_120': results_sym120,
        'file_4278_bg_laplacian': grid_energy,
        'gt_file_4278_bg_laplacian': round(background_grid_energy(gt_rgb_map['file_4278.png']), 4),
        'logs': logs,
        'wall_seconds': time.perf_counter() - t_all,
    }
    (out / 'smooth_fix_results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    # Visual comparison focusing on file_4278.png (full + zoomed empty region) and representative samples
    show_ids = held[::60]  # includes file_4278.png as row 4!
    cols_show = ['u0_base_20shot', 'u1_uni2d_1x1_20shot', 'u1_smooth3x3_20shot', 'u1_smooth3x3_gated_20shot', 'u1_smooth3x3_0shot_halfOD']
    cols = ['PI', *cols_show, 'GT H&E']
    wc, hc = 160, 184
    total_rows = len(show_ids) + 1  # +1 extra row for 2x zoom crop of file_4278 left background!
    view = Image.new('RGB', (len(cols) * (wc + 6) + 8, 34 + total_rows * (hc + 6)), 'white')
    ink = ImageDraw.Draw(view)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 10)
    for c, lbl in enumerate(cols):
        ink.text((8 + c * (wc + 6), 8), lbl, fill='#17314a', font=font)
    for row_i, nm in enumerate(show_ids):
        y = 32 + row_i * (hc + 6)
        ink.text((8, y), nm, fill='#17314a', font=font)
        seq = [pi_rgb_map[nm]] + [arms_imgs[c][nm] for c in cols_show] + [gt_rgb_map[nm]]
        for c, im in enumerate(seq):
            view.paste(Image.fromarray(im).resize((wc, wc), Image.Resampling.LANCZOS), (8 + c * (wc + 6), y + 18))
    # Extra zoomed row: left sparse region of file_4278.png [40:168, 0:128] resized with NEAREST to inspect grid honestly
    y_z = 32 + len(show_ids) * (hc + 6)
    ink.text((8, y_z), 'file_4278 ZOOM (empty stroma [40:168, 0:128], nearest 1:1)', fill='#8b0000', font=font)
    z_seq = [pi_rgb_map['file_4278.png']] + [arms_imgs[c]['file_4278.png'] for c in cols_show] + [gt_rgb_map['file_4278.png']]
    for c, im in enumerate(z_seq):
        crop = Image.fromarray(im[40:168, 0:128]).resize((wc, wc), Image.Resampling.NEAREST)
        view.paste(crop, (8 + c * (wc + 6), y_z + 18))
    view.save(out / 'smooth_fix_comparison.png')
    print('Completed smooth fix ablation in', round(time.perf_counter() - t_all, 1), 's ->', out, flush=True)


if __name__ == '__main__':
    main()
