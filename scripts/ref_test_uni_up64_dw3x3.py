"""Quick 1-fold (Breast, breast_test_120) verification of moving UNI spatial smoothing
from coarse 16x16 to high-res 64x64 post-upsampling Depthwise 3x3:
  1) u1_up64_only: 1x1(14x14) -> Bilinear(64x64) -> DW3x3 -> Zero1x1 at 64x64 only
  2) u1_1x1_hi_dw3x3: u1_uni2d_1x1 (1x1 at 16x16) + 64x64 Bilinear UNI DW3x3 + 64x64 DW3x3 velocity refiner
Compares against rescored u1_uni2d_1x1 from smooth_fix_results.json + builds file_4278 zoom visual.
"""
from __future__ import annotations

import copy
import json
import os
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
from test_uni_smooth_fix import adapt_smooth_m6, background_grid_energy, train_smooth_pretrain  # noqa: E402
from uni_ablation_v2 import UniFlow, _mean_od, infer_uniflow  # noqa: E402
from v2_unified_eval import PAPER_CORE_KEYS, evaluate_split, load_gt_instance_masks  # noqa: E402


class Up64DwUniFlow(nn.Module):
    """UniFlow with 1x1(14x14) -> Bilinear(64x64) -> Depthwise 3x3 high-res smoothing."""

    def __init__(self, dim: int = 128, hybrid: bool = False):
        super().__init__()
        patch = 4
        self.dim = dim
        self.use_uni_2d = True
        self.hybrid = hybrid
        self.state_proj = nn.Conv2d(12, dim, patch, stride=patch)
        self.pi_proj = nn.Conv2d(12, dim, patch, stride=patch)
        self.position = nn.Parameter(torch.randn(1, 16 * 16, dim) * 0.02)
        self.roles = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(32, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.register_buffer('frequency', torch.exp(torch.linspace(0.0, 6.0, 16)))
        self.blocks = nn.ModuleList(Block(dim) for _ in range(4))

        if hybrid:
            # Keep 1x1-only injection at 16x16 inside attention blocks (0 spatial blur on 16x16)
            self.uni_proj16 = nn.Sequential(nn.Conv2d(1024, dim, 1), nn.SiLU())
            self.uni_zero1 = nn.Conv2d(dim, dim, 1)
            self.uni_zero2 = nn.Conv2d(dim, dim, 1)
            nn.init.zeros_(self.uni_zero1.weight)
            nn.init.zeros_(self.uni_zero1.bias)
            nn.init.zeros_(self.uni_zero2.weight)
            nn.init.zeros_(self.uni_zero2.bias)
            # High-res 64x64 residual seam refiner after pixel_shuffle(4)
            self.vel_dw = nn.Conv2d(12, 12, 3, padding=1, groups=12, padding_mode='replicate')
            self.vel_zero = nn.Conv2d(12, 12, 1)
            nn.init.zeros_(self.vel_zero.weight)
            nn.init.zeros_(self.vel_zero.bias)

        # High-res UNI branch: 1x1 on 14x14 -> Bilinear 64x64 -> Depthwise 3x3 -> Zero 1x1
        hi_dim = dim // 2
        self.uni_hi_1x1 = nn.Sequential(nn.Conv2d(1024, hi_dim, 1), nn.SiLU())
        self.uni_hi_dw = nn.Conv2d(hi_dim, hi_dim, 3, padding=1, groups=hi_dim, padding_mode='replicate')
        self.uni_hi_zero = nn.Conv2d(hi_dim, 12, 1)
        nn.init.zeros_(self.uni_hi_zero.weight)
        nn.init.zeros_(self.uni_hi_zero.bias)

        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 12 * patch * patch))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, current, source, t, uni_tokens: torch.Tensor | None = None):
        angle = t[:, None] * self.frequency[None]
        time_emb = self.time(torch.cat((angle.sin(), angle.cos()), dim=-1))[:, None]
        a = self.state_proj(current).flatten(2).transpose(1, 2) + self.position + self.roles[0] + time_emb
        b = self.pi_proj(source).flatten(2).transpose(1, 2) + self.position + self.roles[1] + time_emb
        a = self.blocks[0](a)
        b = self.blocks[1](b)
        hidden = torch.cat((a, b), dim=1)
        n_tok = a.shape[1]

        u_14 = None
        if uni_tokens is not None:
            bsz = uni_tokens.shape[0]
            u_14 = uni_tokens.transpose(1, 2).reshape(bsz, 1024, 14, 14)

        if self.hybrid and u_14 is not None:
            u16 = self.uni_proj16(F.interpolate(u_14, size=(16, 16), mode='bilinear', align_corners=False))
            u1 = self.uni_zero1(u16).flatten(2).transpose(1, 2)
            hidden = torch.cat((hidden[:, :n_tok] + u1, hidden[:, n_tok:]), dim=1)
        hidden = self.blocks[2](hidden)
        if self.hybrid and u_14 is not None:
            u2 = self.uni_zero2(u16).flatten(2).transpose(1, 2)
            hidden = torch.cat((hidden[:, :n_tok] + u2, hidden[:, n_tok:]), dim=1)
        hidden = self.blocks[3](hidden)

        vel = F.pixel_shuffle(self.head(hidden[:, :n_tok]).transpose(1, 2).reshape(-1, 192, 16, 16), 4)
        if self.hybrid:
            vel = vel + self.vel_zero(F.silu(self.vel_dw(vel)))
        if u_14 is not None:
            u_hi = F.interpolate(self.uni_hi_1x1(u_14), size=(64, 64), mode='bilinear', align_corners=False)
            vel = vel + self.uni_hi_zero(F.silu(self.uni_hi_dw(u_hi)))
        return vel


def main():
    t_all = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    prev_dir = V2_ROOT / 'pilot' / 'uni_ablation_v2'
    smooth_dir = V2_ROOT / 'pilot' / 'uni_smooth_fix'
    out = V2_ROOT / 'pilot' / 'uni_up64_dw3x3'
    ckpt_dir = out / 'checkpoints'
    csv_dir = out / 'per_image'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads((V2_ROOT / 'breast_symmetric_splits.json').read_text(encoding='utf-8'))
    sym120 = splits['breast_test_120']
    infer_names = sym120 + (['file_4278.png'] if 'file_4278.png' not in sym120 else [])
    adapt = splits['adapt_20']
    refs = splits['refs_5']

    src_pi, src_he, src_groups = load_source(V2_ROOT, 'train')
    val_pi, val_he, _ = load_source(V2_ROOT, 'val')
    organ_pools = [
        (src_pi[[i for i, g in enumerate(src_groups) if g == o]].cuda(),
         src_he[[i for i, g in enumerate(src_groups) if g == o]].cuda())
        for o in ('liver', 'Kidney', 'Brain')
    ]

    sym_in = pngs(V2_ROOT / 'Breast' / 'test' / 'input')
    sym_gt = pngs(V2_ROOT / 'Breast' / 'test' / 'target')
    off_in = pngs(REPO / 'official_eval' / 'condition')
    off_gt = pngs(REPO / 'official_eval' / 'ground_truth')
    ref_paths = pngs(V2_ROOT / 'Breast' / 'calibration' / 'target')
    ref_od = _mean_od([rgb(ref_paths[n]) for n in refs], 'refs')

    pi_rgb_map = {n: rgb(sym_in[n]) for n in infer_names}
    gt_rgb_map = {n: rgb(sym_gt[n]) for n in infer_names}
    sym_pi_t = tensor([sym_in[n] for n in infer_names])
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

    uni_enc = UNIEncoder(freeze=True).cuda().eval()
    src_uni_4v = precompute_uni_4views_wrapper(uni_enc, src_pi)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        val_uni_v0 = torch.cat([uni_enc(val_pi[i:i + 32].cuda())[:, 1:].half().cpu() for i in range(0, len(val_pi), 32)], dim=0)

    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True, reduction='none').cuda().eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    arms_imgs = {}
    logs = {}
    codec_cuda = copy.deepcopy(codec).cuda().eval()

    # Infer existing u1_uni2d_1x1_20shot + u1_smooth3x3_20shot ONLY for side-by-side visual grid
    fl_1x1 = UniFlow(dim=128, use_uni_2d=True).cuda().eval()
    ck_1x1 = torch.load(prev_dir / 'checkpoints' / 'u1_uni2d_spatial_adapt.pth', map_location='cpu', weights_only=True)
    fl_1x1.load_state_dict(ck_1x1['flow'])
    co_1x1 = Codec().cuda().eval()
    co_1x1.load_state_dict(ck_1x1['codec'])
    imgs_1x1_20 = infer_uniflow(fl_1x1, co_1x1, sym_pi_t, uni_enc, center, scale, infer_names, True, steps=4)
    del fl_1x1, co_1x1

    for arm_name, hybrid in (('u1_up64_only', False), ('u1_1x1_hi_dw3x3', True)):
        fl_tr = Up64DwUniFlow(dim=128, hybrid=hybrid)
        swad_st, pinfo = train_smooth_pretrain(
            arm_name, fl_tr, codec_cuda, center, scale,
            src_lat_pi, src_lat_he, src_pi, src_he, src_uni_4v,
            val_lat_pi, val_lat_he, val_uni_v0, steps=3600, batch=8, seed=7102,
        )
        torch.save({'swad': swad_st}, ckpt_dir / f'{arm_name}_pretrain.pth')

        fl = Up64DwUniFlow(dim=128, hybrid=hybrid).cuda().eval()
        fl.load_state_dict(swad_st)
        zs = infer_uniflow(fl, codec_cuda, sym_pi_t, uni_enc, center, scale, infer_names, True, steps=4)
        zs_ad = infer_uniflow(fl, codec_cuda, adapt_pi_t, uni_enc, center, scale, adapt, True, steps=4)
        sc = ref_od / _mean_od([zs_ad[n] for n in adapt], arm_name)
        arms_imgs[f'{arm_name}_0shot'] = zs
        arms_imgs[f'{arm_name}_0shot_halfOD'] = {n: np.rint((zs[n].astype(np.float32) + apply_scale(zs[n], sc)) / 2.0).clip(0, 255).astype(np.uint8) for n in infer_names}
        del fl

        fl_ad, co_ad, ad_info = adapt_smooth_m6(
            arm_name, Up64DwUniFlow(dim=128, hybrid=hybrid), swad_st, codec, center, scale,
            organ_pools, adapt_pi_t, adapt_he_t, lpips_fn, uni_enc, steps=1000, batch=4, seed=7202,
        )
        torch.save({'flow': fl_ad.state_dict(), 'codec': co_ad.state_dict()}, ckpt_dir / f'{arm_name}_adapt.pth')
        fl_ad, co_ad = fl_ad.cuda().eval(), co_ad.cuda().eval()
        arms_imgs[f'{arm_name}_20shot'] = infer_uniflow(fl_ad, co_ad, sym_pi_t, uni_enc, center, scale, infer_names, True, steps=4)
        logs[arm_name] = {**pinfo, **ad_info}
        del fl_ad, co_ad, fl_tr
        torch.cuda.empty_cache()

    del uni_enc, codec_cuda, src_uni_4v, val_uni_v0
    torch.cuda.empty_cache()

    device = torch.device('cuda')
    gt_masks = load_gt_instance_masks(V2_ROOT / 'eval_cache' / 'gt_nuclei_instances_256.npz', 'Breast')
    cell_model = models.CellposeModel(gpu=True, model_type='nuclei')

    prev_ref = json.loads((smooth_dir / 'smooth_fix_results.json').read_text(encoding='utf-8'))['breast_test_120']
    results = {
        'ref_u0_base_20shot': prev_ref['u0_base_20shot'],
        'ref_u1_uni2d_1x1_0shot': prev_ref['u1_uni2d_1x1_0shot'],
        'ref_u1_uni2d_1x1_0shot_halfOD': prev_ref['u1_uni2d_1x1_0shot_halfOD'],
        'ref_u1_uni2d_1x1_20shot': prev_ref['u1_uni2d_1x1_20shot'],
    }
    grid_energy = {'ref_u1_uni2d_1x1_20shot': round(background_grid_energy(imgs_1x1_20['file_4278.png']), 4)}

    for arm_key, img_map in arms_imgs.items():
        r_s = evaluate_split(sym120, img_map, pi_rgb_map, gt_rgb_map, gt_masks, cell_model, lpips_fn, device, csv_dir / f'{arm_key}_sym120.csv')
        results[arm_key] = r_s
        grid_energy[arm_key] = round(background_grid_energy(img_map['file_4278.png']), 4)
        print(
            f"[SYM120 ] {arm_key:26s} | PSNR={r_s['psnr']:5.2f} SSIM={r_s['ssim']:.4f} LPIPS={r_s['lpips']:.4f} "
            f"KID={r_s['kid']:.5f} Dice={r_s['nuclei_dice']:.4f} F1(1:1)={r_s['nuclei_f1_05']:.4f} "
            f"| nonempty={r_s['nuclei_dice_nonempty']:.4f}/{r_s['nuclei_f1_05_nonempty']:.4f} bg_lap={grid_energy[arm_key]:.2f}",
            flush=True,
        )

    payload = {
        'paper_core_keys': list(PAPER_CORE_KEYS),
        'breast_test_120': results,
        'file_4278_bg_laplacian': grid_energy,
        'logs': logs,
        'wall_seconds': round(time.perf_counter() - t_all, 2),
    }
    (out / 'up64_dw3x3_results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    # Save visual comparison including 1:1 nearest zoom on file_4278 sparse left stroma
    show_ids = [sym120[0], sym120[30], sym120[60], 'file_4278.png']
    cols_maps = {
        'PI': pi_rgb_map,
        '1x1_20shot': imgs_1x1_20,
        'up64_only_20shot': arms_imgs['u1_up64_only_20shot'],
        '1x1_hi_dw3x3_20shot': arms_imgs['u1_1x1_hi_dw3x3_20shot'],
        '1x1_hi_dw3x3_0shot': arms_imgs['u1_1x1_hi_dw3x3_0shot'],
        'GT H&E': gt_rgb_map,
    }
    cols = list(cols_maps.keys())
    wc, hc = 176, 200
    view = Image.new('RGB', (len(cols) * (wc + 6) + 8, 34 + (len(show_ids) + 1) * (hc + 6)), 'white')
    ink = ImageDraw.Draw(view)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 11)
    for ci, name in enumerate(cols):
        ink.text((6 + ci * (wc + 6), 8), name, fill=(20, 20, 20), font=font)
    for ri, nm in enumerate(show_ids):
        y0 = 32 + ri * (hc + 6)
        for ci, c_name in enumerate(cols):
            x0 = 6 + ci * (wc + 6)
            ink.text((x0, y0), nm, fill=(40, 40, 40), font=font)
            view.paste(Image.fromarray(cols_maps[c_name][nm]).resize((wc, wc), Image.Resampling.BILINEAR), (x0, y0 + 18))

    z_row = len(show_ids)
    y0 = 32 + z_row * (hc + 6)
    z_name = 'file_4278.png' if 'file_4278.png' in pi_rgb_map else sym120[-1]
    for ci, c_name in enumerate(cols):
        x0 = 6 + ci * (wc + 6)
        ink.text((x0, y0), f'{z_name} ZOOM [40:168,0:128]', fill=(150, 20, 20), font=font)
        crop = cols_maps[c_name][z_name][40:168, 0:128]
        view.paste(Image.fromarray(crop).resize((wc, wc), Image.Resampling.NEAREST), (x0, y0 + 18))
    view.save(out / 'up64_dw3x3_comparison.png')
    print(f"SAVED {out / 'up64_dw3x3_comparison.png'} | total wall={payload['wall_seconds']}s", flush=True)


def precompute_uni_4views_wrapper(uni_enc, src_pi):
    from uni_ablation_v2 import precompute_uni_4views
    return precompute_uni_4views(uni_enc, src_pi)


if __name__ == '__main__':
    main()
