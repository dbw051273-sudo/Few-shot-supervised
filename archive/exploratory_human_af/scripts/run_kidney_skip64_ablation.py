# -*- coding: utf-8 -*-
"""Kidney AF Ablation: overlap_conv + 64x64 Time-Conditioned Conv Skip Residual
Tests:
  Group 1 (No Discriminator, Standard SRA): BaseFlow_skip64 vs UniFlow_skip64 (0-shot, halfOD, 20-shot SRA)
  Group 2 (20-shot Lightweight PatchGAN + Feature Matching + Perceptual): BaseFlow_skip64_PG vs UniFlow_skip64_PG
Saves metrics to kidney_skip64_results.json and visual comparison to af_kidney_skip64_vis.png.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from cellpose import models as cp_models
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(r"C:\Users\felix\Desktop\Few-shot supervised（小样本有监督）\10_GitHub开源仓库_Few-shot-supervised\scripts")
SCRATCH_DIR = Path(r"C:\Users\felix\.gemini\antigravity\brain\787d0ac0-8ece-45d1-9c24-ccf762231692\scratch")
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from round1_models import Codec, ModularUniFlow
from round1_losses_and_data import (
    adapt_flow,
    apply_od_scale,
    encode_all,
    infer_to_pngs,
    mean_od,
    precompute_uni_4views,
    pretrain_flow,
    rgb,
)
from uni_encoder import UNIEncoder
from v2_unified_eval import build_per_image_lpips, evaluate_split, load_gt_instance_masks
from run_af_human_pipeline import DEFAULT_AF_DATA, DEFAULT_AF_NPZ, DEFAULT_AF_WORK, load_af_fold_tensors
from diag_af_phase2b_plus import adapt_patchgan_af


class ResFiLMBlock64(nn.Module):
    def __init__(self, ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")
        self.film = nn.Linear(t_dim, ch * 2)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate")
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        sc, sh = self.film(t_emb).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + sc[:, :, None, None]) + sh[:, :, None, None]
        h = self.conv2(self.act(h))
        return x + h


class UniFlowSkip64(ModularUniFlow):
    """ModularUniFlow(overlap_conv) augmented with a 64x64 high-res Conv skip residual branch."""

    def __init__(self, dim: int = 192, uni_mode: str = "up64_only"):
        super().__init__(dim=dim, uni_mode=uni_mode, head_mode="overlap_conv")
        self.skip_in = nn.Conv2d(24, 64, 3, padding=1, padding_mode="replicate")
        self.skip_b1 = ResFiLMBlock64(64, dim)
        self.skip_b2 = ResFiLMBlock64(64, dim)
        self.skip_out = nn.Conv2d(64, 12, 1)
        nn.init.zeros_(self.skip_out.weight)
        nn.init.zeros_(self.skip_out.bias)

    def forward(
        self,
        current: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
        uni_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vel = super().forward(current, source, t, uni_tokens)
        angle = t[:, None] * self.frequency[None]
        t_emb = self.time(torch.cat((angle.sin(), angle.cos()), dim=-1))
        hs = self.skip_in(torch.cat((current, source), dim=1))
        hs = self.skip_b1(hs, t_emb)
        hs = self.skip_b2(hs, t_emb)
        return vel + self.skip_out(hs)


def cp_stats(cp_model, pmap, test_ids):
    imgs = [pmap[n] for n in test_ids]
    preds = []
    for i in range(0, len(imgs), 24):
        mb, _, _ = cp_model.eval(imgs[i:i + 24], diameter=None, channels=[0, 0], batch_size=24)
        preds.extend(mb)
    cnts = [int(p.max()) for p in preds]
    return {
        "pr_med": float(np.median(cnts)),
        "pr_mean": float(np.mean(cnts)),
        "pr_nz": int(sum(c > 0 for c in cnts)),
    }


def eval_and_log(tag, pmap, fd, gt_masks, cp_model, lpips_fn, device, results):
    m = evaluate_split(fd["test_ids"], pmap, fd["pi_rgb_map"], fd["gt_rgb_map"], gt_masks, cp_model, lpips_fn, device)
    nc = cp_stats(cp_model, pmap, fd["test_ids"])
    results[tag] = {**m, **nc}
    print(
        f"[SKIP64] {tag:38s} | PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} "
        f"KID={m['kid']*1000:.2f} Dice={m['nuclei_dice']:.4f} Dice_ne={m['nuclei_dice_nonempty']:.4f} "
        f"| PR_nuc: med={nc['pr_med']:.1f} mean={nc['pr_mean']:.1f} nz={nc['pr_nz']}/120",
        flush=True,
    )


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    fold = "Kidney"
    raw_dir = DEFAULT_AF_WORK / "raw"
    ckpt_dir = raw_dir / "checkpoints" / fold
    splits = json.loads((DEFAULT_AF_DATA / "af_lomo_splits.json").read_text(encoding="utf-8"))
    fd = load_af_fold_tensors(DEFAULT_AF_DATA, splits, fold, "raw", k_shot=20)
    gt_masks = load_gt_instance_masks(DEFAULT_AF_NPZ, fold)
    lpips_fn = build_per_image_lpips(net_type="alex", normalize=True).to(device)
    cp_model = cp_models.CellposeModel(gpu=True, model_type="nuclei")
    uni_enc = UNIEncoder(freeze=True).cuda().eval()

    codec = Codec().cuda()
    codec.load_state_dict(torch.load(ckpt_dir / "codec_rae0.08_s8800.pth", map_location="cuda", weights_only=True))
    codec.eval().requires_grad_(False)

    with torch.no_grad():
        lat_pi = encode_all(codec, fd["src_pi"])
        center, scale = float(lat_pi.mean()), float(lat_pi.std())
        src_lat_pi = (lat_pi - center) / scale
        src_lat_he = (encode_all(codec, fd["src_he"]) - center) / scale
        val_lat_pi = (encode_all(codec, fd["val_pi"]) - center) / scale
        val_lat_he = (encode_all(codec, fd["val_he"]) - center) / scale
        val_uni_v0 = torch.cat(
            [uni_enc(fd["val_pi"][i:i + 32].cuda())[:, 1:].half().cpu() for i in range(0, len(fd["val_pi"]), 32)],
            dim=0,
        )
    src_uni_4v = precompute_uni_4views(uni_enc, fd["src_pi"])
    organ_pools = [
        (fd["src_pi"][[i for i, g in enumerate(fd["src_groups"]) if g == o]].cuda(),
         fd["src_he"][[i for i, g in enumerate(fd["src_groups"]) if g == o]].cuda())
        for o in sorted(set(fd["src_groups"]))
    ]

    results = {}
    vis_maps = {}

    for arch_tag, uni_mode in (("u0_skip64", "none"), ("u1_skip64", "up64_only")):
        m_pre = UniFlowSkip64(dim=192, uni_mode=uni_mode)
        swad_st, pinfo = pretrain_flow(
            f"raw_Kidney_{arch_tag}", m_pre, codec, center, scale,
            src_lat_pi, src_lat_he, fd["src_pi"], fd["src_he"], src_uni_4v,
            val_lat_pi, val_lat_he, val_uni_v0,
            steps=9600, batch=8, seed=7102, use_on_policy_rollout=False, rollout_steps=4, lpips_fn=lpips_fn,
        )
        torch.save({"swad": swad_st, "pinfo": pinfo, "center": center, "scale": scale}, ckpt_dir / f"Kidney__{arch_tag}_d192_pretrain.pth")

        fl0 = UniFlowSkip64(dim=192, uni_mode=uni_mode)
        fl0.load_state_dict(swad_st)
        zs_map = infer_to_pngs(fl0, codec, fd["test_pi_t"], uni_enc, center, scale, fd["test_ids"], raw_dir / "generated" / fold / f"Kidney__{arch_tag}_d192__0shot", steps=4)
        eval_and_log(f"Kidney__{arch_tag}__0shot", zs_map, fd, gt_masks, cp_model, lpips_fn, device, results)

        zs_refs = infer_to_pngs(fl0, codec, fd["refs_pi_t"], uni_enc, center, scale, fd["refs_ids"], raw_dir / "generated" / fold / f"_tmp_refs_{arch_tag}", steps=4)
        sc = fd["refs_od"] / mean_od([zs_refs[n] for n in fd["refs_ids"]], arch_tag)
        od_dir = raw_dir / "generated" / fold / f"Kidney__{arch_tag}_d192__0shot_halfOD"
        od_dir.mkdir(parents=True, exist_ok=True)
        half_map = {}
        for nm in fd["test_ids"]:
            im = np.rint((zs_map[nm].astype(np.float32) + apply_od_scale(zs_map[nm], sc).astype(np.float32)) / 2.0).clip(0, 255).astype(np.uint8)
            half_map[nm] = im
            Image.fromarray(im).save(od_dir / nm)
        eval_and_log(f"Kidney__{arch_tag}__0shot_halfOD", half_map, fd, gt_masks, cp_model, lpips_fn, device, results)
        del fl0

        # Group 1: 20-shot SRA (No Discriminator)
        fl_ad, co_ad, _ = adapt_flow(
            f"raw_Kidney_{arch_tag}_m6", "m6",
            UniFlowSkip64(dim=192, uni_mode=uni_mode), swad_st, codec, center, scale,
            organ_pools, fd["adapt_pi_t"], fd["adapt_he_t"],
            lpips_fn, uni_enc, steps=1500, batch=4, seed=7202, decoder_lr_mult=0.1, rollout_steps=4, od_weight=0.0,
        )
        ad_map = infer_to_pngs(fl_ad, co_ad, fd["test_pi_t"], uni_enc, center, scale, fd["test_ids"], raw_dir / "generated" / fold / f"Kidney__{arch_tag}_d192__m6_k20", steps=4)
        eval_and_log(f"Kidney__{arch_tag}__m6_k20", ad_map, fd, gt_masks, cp_model, lpips_fn, device, results)
        vis_maps[f"{arch_tag} SRA"] = ad_map
        del fl_ad, co_ad

        # Group 2: 20-shot Perceptual + Lightweight PatchGAN
        fl_pg, co_pg = adapt_patchgan_af(
            f"raw_Kidney_{arch_tag}_PG",
            UniFlowSkip64(dim=192, uni_mode=uni_mode), swad_st, codec, center, scale,
            organ_pools, fd["adapt_pi_t"], fd["adapt_he_t"],
            lpips_fn, uni_enc, w_l1=1.0, w_grad=1.0, w_lpips=2.0, w_adv=0.10, w_fm=1.5, steps=1500,
        )
        pg_map = infer_to_pngs(fl_pg, co_pg, fd["test_pi_t"], uni_enc, center, scale, fd["test_ids"], raw_dir / "generated" / fold / f"Kidney__{arch_tag}_d192__m6_k20_PG", steps=4)
        eval_and_log(f"Kidney__{arch_tag}__m6_k20_PG", pg_map, fd, gt_masks, cp_model, lpips_fn, device, results)
        vis_maps[f"{arch_tag} PG"] = pg_map
        del fl_pg, co_pg

        gc.collect()
        torch.cuda.empty_cache()

    picks = [fd["test_ids"][i] for i in [0, 20, 60, 100]]
    p2p_20 = {n: rgb(raw_dir / "generated" / fold / "Kidney__pix2pix__20shot" / n) for n in picks}
    cols = [
        ("AF (raw)", fd["pi_rgb_map"]),
        ("GT H&E", fd["gt_rgb_map"]),
        ("Pix2Pix 20shot", p2p_20),
        ("BaseFlow skip64 SRA", vis_maps["u0_skip64 SRA"]),
        ("UniFlow skip64 SRA", vis_maps["u1_skip64 SRA"]),
        ("BaseFlow skip64 PG", vis_maps["u0_skip64 PG"]),
        ("UniFlow skip64 PG", vis_maps["u1_skip64 PG"]),
    ]
    W, H, pad, header = 256, 256, 4, 28
    canvas = Image.new("RGB", (len(cols) * (W + pad), header + len(picks) * (H + pad)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, (t, _) in enumerate(cols):
        draw.text((ci * (W + pad) + 6, 6), t, fill=(0, 0, 0))
    for ri, nm in enumerate(picks):
        for ci, (_, mp) in enumerate(cols):
            canvas.paste(Image.fromarray(mp[nm]), (ci * (W + pad), header + ri * (H + pad)))
    vis_p = SCRATCH_DIR / "af_kidney_skip64_vis.png"
    canvas.save(vis_p)

    out_json = SCRATCH_DIR / "kidney_skip64_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {out_json} & {vis_p}", flush=True)


if __name__ == "__main__":
    main()
