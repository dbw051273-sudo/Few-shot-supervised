# -*- coding: utf-8 -*-
"""Phase 2B+ 10-min Lightweight Ablation on Human Breast AF->H&E:
  Part A (raw Breast): Test adding spectral-norm PatchGAN + Feature Matching in 20-shot Stage III
    on existing raw pretrained checkpoints (BaseFlow u0_base vs UniFlow u1_up64_only).
  Part B (organ_q Breast): Test organ-level global [p0.5, p99.5] quantile normalization
    (computed strictly from train for source organs and adapt_20 for target Breast — zero test leakage)
    with UniFlow (0-shot + 20-shot perc+PatchGAN).
"""
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from cellpose import models as cp_models

SCRIPT_DIR = Path(r"C:\Users\felix\Desktop\Few-shot supervised（小样本有监督）\10_GitHub开源仓库_Few-shot-supervised\scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from round1_models import Codec, SwadAccumulator, build_flow_model
from round1_losses_and_data import (
    augment_pair,
    encode_all,
    grad_l1,
    infer_to_pngs,
    mean_od,
    pi_acq_jitter,
    precompute_uni_4views,
    pretrain_flow,
    rgb,
    stain_aug_source_he,
    train_fold_codec,
)
from uni_encoder import UNIEncoder
from v2_unified_eval import build_per_image_lpips, evaluate_split, load_gt_instance_masks
from run_af_human_pipeline import AF_ORGANS, DEFAULT_AF_DATA, DEFAULT_AF_NPZ, DEFAULT_AF_WORK


class PatchDiscriminator(nn.Module):
    """Lightweight 3-layer spectral-norm 70x70 PatchGAN discriminator with multi-scale feature extraction."""

    def __init__(self, in_ch: int = 3, ndf: int = 64):
        super().__init__()
        self.b1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.b2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.b3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.b4 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 1, bias=False)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 1))

    def forward(self, x: torch.Tensor):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        out = self.head(f4)
        return out, [f1, f2, f3, f4]


def adapt_patchgan_af(
    tag: str,
    flow: nn.Module,
    swad_st: dict,
    init_codec: Codec,
    center: float,
    scale: float,
    src_pools: list[tuple[torch.Tensor, torch.Tensor]],
    tgt_pi: torch.Tensor,
    tgt_he: torch.Tensor,
    lpips_fn,
    uni_enc: UNIEncoder,
    w_l1: float = 1.0,
    w_grad: float = 1.0,
    w_lpips: float = 2.0,
    w_adv: float = 0.08,
    w_fm: float = 1.5,
    steps: int = 1500,
    batch: int = 4,
    seed: int = 7202,
):
    torch.manual_seed(seed)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    py_rng = random.Random(seed)
    t0 = time.perf_counter()

    flow.load_state_dict(swad_st)
    flow.cuda().train()
    codec = copy.deepcopy(init_codec).cuda()
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    codec.encoder.eval()
    codec.decoder.train()
    for p in codec.decoder.parameters():
        p.requires_grad_(True)

    disc = PatchDiscriminator(in_ch=3, ndf=64).cuda().train()
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4)

    g_params = list(flow.parameters()) + list(codec.decoder.parameters())
    opt_g = torch.optim.AdamW(
        [{"params": list(flow.parameters()), "lr": 2e-4}, {"params": list(codec.decoder.parameters()), "lr": 2e-5}],
        betas=(0.5, 0.999),
        weight_decay=1e-4,
    )
    swad = SwadAccumulator({"flow": flow, "decoder": codec.decoder}, start_step=600, every=10)
    t_pi, t_he = tgt_pi.cuda(), tgt_he.cuda()

    for step in range(1, steps + 1):
        ti = torch.randint(len(t_pi), (batch,), generator=rng, device="cuda")
        xt, yt = augment_pair(t_pi[ti], t_he[ti], py_rng)
        xt_in = pi_acq_jitter(xt, py_rng, strength=0.05)

        with torch.no_grad():
            z0_t = (codec.encode(xt_in).float() - center) / scale
            z1_t = (codec.encode(yt).float() - center) / scale
            ut_t = uni_enc(xt_in)[:, 1:].float() if getattr(flow, "use_uni_2d", False) else None

        tt = torch.rand(batch, generator=rng, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_t = flow(torch.lerp(z0_t, z1_t, tt[:, None, None, None]), z0_t, tt, ut_t)
            z = z0_t
            for k in range(4):
                z = z + flow(z, z0_t, z.new_full((batch,), k * 0.25), ut_t) * 0.25
            pred_t = codec.decode(z * scale + center)

        # 1) Update PatchDiscriminator (real = yt, fake = pred_t.detach())
        pred_det = pred_t.float().clamp(-1.0, 1.0).detach()
        d_real, _ = disc(yt)
        d_fake, _ = disc(pred_det)
        loss_d = 0.5 * (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
        opt_d.step()

        # 2) Update Flow + Decoder
        l_vel = F.mse_loss(v_t.float(), z1_t - z0_t)
        l_img = w_l1 * F.l1_loss(pred_t.float(), yt) + w_grad * grad_l1(pred_t.float(), yt)
        l_lp = lpips_fn((pred_t.float().clamp(-1, 1) + 1.0) * 0.5, (yt.clamp(-1, 1) + 1.0) * 0.5).mean()

        g_fake, f_fake = disc(pred_t.float().clamp(-1.0, 1.0))
        with torch.no_grad():
            _, f_real = disc(yt)
        l_adv = -g_fake.mean()
        l_fm = sum(F.l1_loss(ff, fr) for ff, fr in zip(f_fake, f_real)) / float(len(f_fake))

        loss = l_vel + l_img + w_lpips * l_lp + w_adv * l_adv + w_fm * l_fm

        op_pi, op_he = src_pools[py_rng.randrange(len(src_pools))]
        si = torch.randint(len(op_pi), (batch,), generator=rng, device="cuda")
        xs, ys = augment_pair(op_pi[si], op_he[si], py_rng)
        xs_in = pi_acq_jitter(xs, py_rng, strength=0.08)
        ys_in = stain_aug_source_he(ys, t_he, py_rng)
        with torch.no_grad():
            z0_s = (codec.encode(xs_in).float() - center) / scale
            z1_s = (codec.encode(ys_in).float() - center) / scale
            ut_s = uni_enc(xs_in)[:, 1:].float() if getattr(flow, "use_uni_2d", False) else None
        ts = torch.rand(batch, generator=rng, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_s = flow(torch.lerp(z0_s, z1_s, ts[:, None, None, None]), z0_s, ts, ut_s)
            rec_s = codec.decode(z1_s * scale + center)
        l_s = 0.6 * F.mse_loss(v_s.float(), z1_s - z0_s) + 0.6 * F.l1_loss(rec_s.float(), ys_in)
        loss = loss + 0.40 * l_s

        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(g_params, 1.5)
        opt_g.step()
        swad.maybe_update(step, steps)

        if step in (750, 1500):
            print(
                f"[{tag}] step={step}/{steps} d={loss_d.item():.3f} adv={l_adv.item():.3f} "
                f"fm={l_fm.item():.3f} lp={l_lp.item():.3f} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    swad.apply()
    return flow.eval().cpu(), codec.eval().cpu()


def eval_and_report(label: str, test_ids, pmap, pi_rgb_map, gt_rgb_map, gt_masks, cp_model, lpips_fn, device):
    m = evaluate_split(test_ids, pmap, pi_rgb_map, gt_rgb_map, gt_masks, cp_model, lpips_fn, device)
    imgs = [pmap[n] for n in test_ids]
    preds = []
    for i in range(0, len(imgs), 24):
        mb, _, _ = cp_model.eval(imgs[i:i + 24], diameter=None, channels=[0, 0], batch_size=24)
        preds.extend(mb)
    pr_cnts = [int(p.max()) for p in preds]
    nz = sum(c > 0 for c in pr_cnts)
    med = float(np.median(pr_cnts))
    mean_c = float(np.mean(pr_cnts))
    print(
        f"[RESULT] {label:28s} | PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} "
        f"KID={m['kid']*1000:.2f} Dice={m['nuclei_dice']:.4f} Dice_ne={m['nuclei_dice_nonempty']:.4f} "
        f"F1={m['nuclei_f1_05']:.4f} | PR_nuc: med={med:.1f} mean={mean_c:.1f} nz={nz}/{len(test_ids)}",
        flush=True,
    )
    return m, {"pr_med": med, "pr_mean": mean_c, "pr_nz": nz}


def load_breast_fold_raw_or_organq(data_dir: Path, splits: dict, mode: str, k_shot: int = 20):
    assert mode in ("raw", "organ_q")
    q_bounds = {}
    for org in AF_ORGANS:
        sp = splits["organs"][org]
        ref_sub, ref_ids = ("adapt_pool", sp["adapt_20"]) if org == "Breast" else ("train", sp["train"])
        if mode == "raw":
            q_bounds[org] = (0.0, 255.0)
        else:
            arr = np.concatenate([rgb(data_dir / org / ref_sub / "input" / n)[..., 0].ravel() for n in ref_ids])
            lo, hi = float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))
            q_bounds[org] = (lo, max(hi, lo + 24.0))

    def af_norm_batch(paths, org):
        lo, hi = q_bounds[org]
        arrs = [np.clip((rgb(p).astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0 for p in paths]
        return torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).float()

    def he_batch(paths):
        arrs = [rgb(p) for p in paths]
        return torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)

    src_pi, src_he, src_groups, val_pi, val_he = [], [], [], [], []
    for org in AF_ORGANS:
        if org == "Breast":
            continue
        sp = splits["organs"][org]
        src_pi.append(af_norm_batch([data_dir / org / "train" / "input" / n for n in sp["train"]], org))
        src_he.append(he_batch([data_dir / org / "train" / "target" / n for n in sp["train"]]))
        src_groups.extend([org] * len(sp["train"]))
        val_pi.append(af_norm_batch([data_dir / org / "val" / "input" / n for n in sp["val"]], org))
        val_he.append(he_batch([data_dir / org / "val" / "target" / n for n in sp["val"]]))

    sp_t = splits["organs"]["Breast"]
    ad_ids = sp_t[f"adapt_{k_shot}"]
    test_ids = sp_t["test"]
    return {
        "q_bounds": q_bounds,
        "src_pi": torch.cat(src_pi, dim=0),
        "src_he": torch.cat(src_he, dim=0),
        "src_groups": src_groups,
        "val_pi": torch.cat(val_pi, dim=0),
        "val_he": torch.cat(val_he, dim=0),
        "adapt_pi_t": af_norm_batch([data_dir / "Breast" / "adapt_pool" / "input" / n for n in ad_ids], "Breast"),
        "adapt_he_t": he_batch([data_dir / "Breast" / "adapt_pool" / "target" / n for n in ad_ids]),
        "test_ids": test_ids,
        "test_pi_t": af_norm_batch([data_dir / "Breast" / "test" / "input" / n for n in test_ids], "Breast"),
        "pi_rgb_map": {n: rgb(data_dir / "Breast" / "test" / "input" / n) for n in test_ids},
        "gt_rgb_map": {n: rgb(data_dir / "Breast" / "test" / "target" / n) for n in test_ids},
    }


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    splits = json.loads((DEFAULT_AF_DATA / "af_lomo_splits.json").read_text(encoding="utf-8"))
    uni_enc = UNIEncoder(freeze=True).cuda().eval()
    lpips_fn = build_per_image_lpips(net_type="alex", normalize=True).to(device)
    cp_model = cp_models.CellposeModel(gpu=True, model_type="nuclei")
    gt_masks = load_gt_instance_masks(DEFAULT_AF_NPZ, "Breast")
    summary = {}

    # ---------------- PART A: raw Breast PatchGAN on existing pretrain ----------------
    fd_raw = load_breast_fold_raw_or_organq(DEFAULT_AF_DATA, splits, "raw", k_shot=20)
    raw_ckpt = DEFAULT_AF_WORK / "raw" / "checkpoints" / "Breast"
    codec_raw = Codec().cuda()
    codec_raw.load_state_dict(torch.load(raw_ckpt / "codec_rae0.08_s8800.pth", map_location="cuda", weights_only=True))
    codec_raw.eval().requires_grad_(False)
    pools_raw = [
        (fd_raw["src_pi"][[i for i, g in enumerate(fd_raw["src_groups"]) if g == o]].cuda(),
         fd_raw["src_he"][[i for i, g in enumerate(fd_raw["src_groups"]) if g == o]].cuda())
        for o in sorted(set(fd_raw["src_groups"]))
    ]
    for arch in ("u0_base", "u1_up64_only"):
        pre = torch.load(raw_ckpt / f"Breast__{arch}_d192_pretrain.pth", map_location="cpu", weights_only=False)
        fl_ad, co_ad = adapt_patchgan_af(
            f"raw_{arch}_patchgan", build_flow_model(arch, dim=192), pre["swad"], codec_raw,
            pre["center"], pre["scale"], pools_raw, fd_raw["adapt_pi_t"], fd_raw["adapt_he_t"],
            lpips_fn, uni_enc, w_l1=1.0, w_grad=1.0, w_lpips=2.0, w_adv=0.08, w_fm=1.5, steps=1500,
        )
        out_dir = DEFAULT_AF_WORK / "raw" / "generated" / "Breast" / f"Breast__{arch}_d192__m6_k20_patchgan"
        pmap = infer_to_pngs(fl_ad, co_ad, fd_raw["test_pi_t"], uni_enc, pre["center"], pre["scale"], fd_raw["test_ids"], out_dir, steps=4)
        m, nc = eval_and_report(f"raw_{arch}_patchgan", fd_raw["test_ids"], pmap, fd_raw["pi_rgb_map"], fd_raw["gt_rgb_map"], gt_masks, cp_model, lpips_fn, device)
        summary[f"raw_{arch}_patchgan"] = {**m, **nc}

    # ---------------- PART B: organ_q Breast (UniFlow 0-shot + 20-shot PatchGAN) ----------------
    fd_oq = load_breast_fold_raw_or_organq(DEFAULT_AF_DATA, splits, "organ_q", k_shot=20)
    print(f"[ORGAN_Q BOUNDS] {fd_oq['q_bounds']}", flush=True)
    oq_ckpt = DEFAULT_AF_WORK / "organ_q" / "checkpoints" / "Breast"
    oq_ckpt.mkdir(parents=True, exist_ok=True)
    codec_oq, cinfo = train_fold_codec(fd_oq["src_pi"], fd_oq["src_he"], steps=8800, seed=7000, detail_weight=0.15, noise_max=0.08)
    torch.save(codec_oq.state_dict(), oq_ckpt / "codec_rae0.08_s8800.pth")
    print(f"[ORGAN_Q CODEC] {cinfo}", flush=True)

    with torch.no_grad():
        lat_pi = encode_all(codec_oq, fd_oq["src_pi"])
        c_oq, s_oq = float(lat_pi.mean()), float(lat_pi.std())
        src_lat_pi = (lat_pi - c_oq) / s_oq
        src_lat_he = (encode_all(codec_oq, fd_oq["src_he"]) - c_oq) / s_oq
        val_lat_pi = (encode_all(codec_oq, fd_oq["val_pi"]) - c_oq) / s_oq
        val_lat_he = (encode_all(codec_oq, fd_oq["val_he"]) - c_oq) / s_oq
        val_uni_v0 = torch.cat([uni_enc(fd_oq["val_pi"][i:i + 32].cuda())[:, 1:].half().cpu() for i in range(0, len(fd_oq["val_pi"]), 32)], dim=0)
    src_uni_4v = precompute_uni_4views(uni_enc, fd_oq["src_pi"])
    pools_oq = [
        (fd_oq["src_pi"][[i for i, g in enumerate(fd_oq["src_groups"]) if g == o]].cuda(),
         fd_oq["src_he"][[i for i, g in enumerate(fd_oq["src_groups"]) if g == o]].cuda())
        for o in sorted(set(fd_oq["src_groups"]))
    ]

    fl_oq = build_flow_model("u1_up64_only", dim=192)
    swad_oq, pinfo = pretrain_flow(
        "organ_q_u1", fl_oq, codec_oq, c_oq, s_oq, src_lat_pi, src_lat_he,
        fd_oq["src_pi"], fd_oq["src_he"], src_uni_4v, val_lat_pi, val_lat_he, val_uni_v0,
        steps=3600, batch=8, seed=7102, use_on_policy_rollout=True, rollout_steps=4, lpips_fn=lpips_fn,
    )
    torch.save({"swad": swad_oq, "center": c_oq, "scale": s_oq, "info": pinfo}, oq_ckpt / "Breast__u1_up64_only_d192_pretrain.pth")
    fl_0 = build_flow_model("u1_up64_only", dim=192)
    fl_0.load_state_dict(swad_oq)
    pmap0 = infer_to_pngs(fl_0, codec_oq, fd_oq["test_pi_t"], uni_enc, c_oq, s_oq, fd_oq["test_ids"], DEFAULT_AF_WORK / "organ_q" / "generated" / "Breast" / "Breast__u1_up64_only_d192__0shot", steps=4)
    m0, nc0 = eval_and_report("organ_q_u1_0shot", fd_oq["test_ids"], pmap0, fd_oq["pi_rgb_map"], fd_oq["gt_rgb_map"], gt_masks, cp_model, lpips_fn, device)
    summary["organ_q_u1_0shot"] = {**m0, **nc0, "best_val_mse": pinfo["best_val_mse"]}

    fl_oq_ad, co_oq_ad = adapt_patchgan_af(
        "organ_q_u1_patchgan", build_flow_model("u1_up64_only", dim=192), swad_oq, codec_oq,
        c_oq, s_oq, pools_oq, fd_oq["adapt_pi_t"], fd_oq["adapt_he_t"],
        lpips_fn, uni_enc, w_l1=1.0, w_grad=1.0, w_lpips=2.0, w_adv=0.08, w_fm=1.5, steps=1500,
    )
    pmap_ad = infer_to_pngs(fl_oq_ad, co_oq_ad, fd_oq["test_pi_t"], uni_enc, c_oq, s_oq, fd_oq["test_ids"], DEFAULT_AF_WORK / "organ_q" / "generated" / "Breast" / "Breast__u1_up64_only_d192__m6_k20_patchgan", steps=4)
    ma, nca = eval_and_report("organ_q_u1_patchgan", fd_oq["test_ids"], pmap_ad, fd_oq["pi_rgb_map"], fd_oq["gt_rgb_map"], gt_masks, cp_model, lpips_fn, device)
    summary["organ_q_u1_patchgan"] = {**ma, **nca}

    out_json = Path(r"C:\Users\felix\.gemini\antigravity\brain\787d0ac0-8ece-45d1-9c24-ccf762231692\scratch\phase2b_plus_results.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_json}", flush=True)


if __name__ == "__main__":
    main()
