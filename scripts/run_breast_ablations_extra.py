# -*- coding: utf-8 -*-
"""Comprehensive Breast single-fold ablation suite for IEEE TMI/MIA review:
1) Euler sampling steps S in {1, 2, 4, 8, 16} on 0-shot and 20-shot UniFlow
2) OASRA component-wise ablation (NaiveFT+SWA+jitter, w/o L_OD, w/o SourceReplay)
   evaluated on both Breast target test (N=120) and Source-Val anti-forgetting (N=240)
3) Terminal 3x3 DWConv ablation (UniFlow-up64 w/o dwconv) + 2D FFT periodic grid energy
4) 3-seed variance (seeds 7202, 8202, 9202) on BaseFlow, NaiveFT, and OASRA (20-shot)
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from cellpose import models

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from round1_models import Codec, SwadAccumulator, build_flow_model
from round1_losses_and_data import (
    augment_pair,
    build_or_load_lomo_splits,
    encode_all,
    grad_l1,
    load_fold_tensors,
    od_stain_anchor_loss,
    pi_acq_jitter,
    precompute_uni_4views,
    pretrain_flow,
    stain_aug_source_he,
    tensor_from_paths,
)
from uni_encoder import UNIEncoder
from v2_unified_eval import (
    OfficialScalarMetrics,
    build_per_image_lpips,
    evaluate_split,
    load_gt_instance_masks,
    sparse_grid_laplacian,
)

BASE = Path(r"C:\Users\felix\Desktop\Few-shot supervised（小样本有监督）\07_本地3080Ti_Round1定型版产出")
DATA_DIR = Path(r"C:\Users\felix\Desktop\Few-shot supervised（小样本有监督）\03_洁净数据集与重建包\UniPrior_四器官洁净均衡集_v2")
NPZ_PATH = Path(r"C:\Users\felix\Desktop\Few-shot supervised（小样本有监督）\01_第一轮5880发车包与核心代码\小样本有监督_第一轮5880发车包\eval_cache\gt_nuclei_instances_256.npz")
OUT_JSON = BASE / "eval_results" / "breast_extra_ablations.json"
SPLITS_JSON = BASE / "eval_results" / "round1_splits.json"
CKPT_DIR = BASE / "checkpoints" / "Breast"


def load_results() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {"steps_ablation": {}, "oasra_components": {}, "dwconv_ablation": {}, "seed_variance": {}}


def save_results(res: dict) -> None:
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def infer_map(flow, codec, pi_cpu, ids_list, uni_enc, center, scale, steps=4, batch=24):
    flow.cuda().eval()
    codec.cuda().eval()
    dt = 1.0 / float(steps)
    preds = []
    use_uni = getattr(flow, "use_uni_2d", False) and uni_enc is not None
    for i in range(0, len(pi_cpu), batch):
        bx = pi_cpu[i:i + batch].cuda()
        ut = None
        if use_uni:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ut = uni_enc(bx)[:, 1:].float()
        z0 = (codec.encode(bx).float() - center) / scale
        z = z0
        for k in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = flow(z, z0, z.new_full((len(bx),), k * dt), ut)
            z = z + v.float() * dt
        dec = codec.decode(z * scale + center).clamp(-1.0, 1.0)
        arr = ((dec.cpu().numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
        for j in range(len(arr)):
            preds.append(arr[j].transpose(1, 2, 0))
    return {nm: im for nm, im in zip(ids_list, preds)}


@torch.no_grad()
def bench_latency_ms(flow, codec, x1, uni_enc, center, scale, steps=4, warmup=8, runs=30) -> float:
    flow.cuda().eval()
    codec.cuda().eval()
    bx = x1[:1].cuda()
    dt = 1.0 / float(steps)
    use_uni = getattr(flow, "use_uni_2d", False) and uni_enc is not None
    def run_once():
        ut = None
        if use_uni:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ut = uni_enc(bx)[:, 1:].float()
        z0 = (codec.encode(bx).float() - center) / scale
        z = z0
        for k in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = flow(z, z0, z.new_full((1,), k * dt), ut)
            z = z + v.float() * dt
        _ = codec.decode(z * scale + center)
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        run_once()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / runs


def eval_breast_target(pred_map, fd, gt_masks, cell_model, lpips_fn, device):
    m = evaluate_split(
        names=fd["test_ids"],
        pred_rgb_map=pred_map,
        pi_rgb_map=fd["pi_rgb_map"],
        gt_rgb_map=fd["gt_rgb_map"],
        gt_masks=gt_masks,
        cell_model=cell_model,
        lpips_fn=lpips_fn,
        device=device,
    )
    bg_lap = sparse_grid_laplacian(pred_map["file_4278.png"]) if "file_4278.png" in pred_map else 0.0
    return {
        "psnr": round(float(m["psnr"]), 2),
        "ssim": round(float(m["ssim"]), 4),
        "lpips": round(float(m["lpips"]), 4),
        "kid_x1e3": round(float(m["kid"]) * 1000.0, 2),
        "dice": round(float(m["nuclei_dice"]), 4),
        "f1": round(float(m["nuclei_f1_05"]), 4),
        "bg_lap": round(float(bg_lap), 2),
    }


def compute_2d_fft_energy(pred_map: dict[str, np.ndarray], gt_map: dict[str, np.ndarray], test_ids: list[str]) -> tuple[float, float]:
    bg_patches = []
    for n in test_ids:
        gt_g = gt_map[n].astype(np.float64).mean(axis=2)
        for y in range(0, 256, 64):
            for x in range(0, 256, 64):
                p = gt_g[y:y+64, x:x+64]
                if p.mean() > 210 and p.std() < 15:
                    bg_patches.append((n, y, x))
    k4_list, k16_list = [], []
    win = np.outer(np.hanning(64), np.hanning(64))
    cy, cx = 32, 32
    for n, y, x in bg_patches:
        patch = pred_map[n].astype(np.float64).mean(axis=2)[y:y+64, x:x+64]
        zm = patch - patch.mean()
        mag2 = np.abs(np.fft.fftshift(np.fft.fft2(zm * win))) ** 2
        ac = mag2.sum() - mag2[cy-1:cy+2, cx-1:cx+2].sum() + 1e-12
        e16 = (mag2[cy-4, cx] + mag2[cy+4, cx] + mag2[cy, cx-4] + mag2[cy, cx+4] +
               mag2[cy-8, cx] + mag2[cy+8, cx] + mag2[cy, cx-8] + mag2[cy, cx+8] +
               mag2[cy-16, cx] + mag2[cy+16, cx] + mag2[cy, cx-16] + mag2[cy, cx+16]) / ac
        e_p16 = sum(mag2[cy+dy, cx+dx] for dy in (-4, 0, 4, -8, 8) for dx in (-4, 0, 4, -8, 8) if not (dy==0 and dx==0)) / ac
        k4_list.append(e_p16 * 100.0)
        k16_list.append(e16 * 100.0)
    return round(float(np.mean(k4_list)), 2), round(float(np.mean(k16_list)), 2)


def custom_adapt(
    tag: str,
    flow: nn.Module,
    init_st: dict,
    init_codec: Codec,
    center: float,
    scale: float,
    src_pools: list[tuple[torch.Tensor, torch.Tensor]],
    tgt_pi: torch.Tensor,
    tgt_he: torch.Tensor,
    lpips_fn,
    uni_enc: UNIEncoder | None,
    use_od: bool,
    use_replay: bool,
    use_swa: bool,
    use_jitter: bool,
    seed: int = 7202,
    steps: int = 1500,
    batch: int = 4,
    decoder_lr_mult: float = 0.1,
    rollout_steps: int = 4,
):
    torch.manual_seed(seed)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    py_rng = random.Random(seed)
    flow.load_state_dict(init_st)
    flow.cuda().train()
    codec = copy.deepcopy(init_codec).cuda()
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    codec.encoder.eval()
    codec.decoder.train()
    for p in codec.decoder.parameters():
        p.requires_grad_(True)

    params = list(flow.parameters()) + list(codec.decoder.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": list(flow.parameters()), "lr": 2e-4},
            {"params": list(codec.decoder.parameters()), "lr": 2e-4 * decoder_lr_mult},
        ],
        weight_decay=1e-4,
    )
    swad = SwadAccumulator({"flow": flow, "decoder": codec.decoder}, start_step=max(1, int(0.4 * steps)), every=10)
    t_pi, t_he = tgt_pi.cuda(), tgt_he.cuda()
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        ti = torch.randint(len(t_pi), (batch,), generator=rng, device="cuda")
        xt, yt = augment_pair(t_pi[ti], t_he[ti], py_rng)
        xt_in = pi_acq_jitter(xt, py_rng, strength=0.05) if use_jitter else xt
        with torch.no_grad():
            z0_t = (codec.encode(xt_in).float() - center) / scale
            z1_t = (codec.encode(yt).float() - center) / scale
            ut_t = None
            if uni_enc is not None and getattr(flow, "use_uni_2d", False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ut_t = uni_enc(xt_in)[:, 1:].float()

        tt = torch.rand(batch, generator=rng, device="cuda")
        dt = 1.0 / float(rollout_steps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_t = flow(torch.lerp(z0_t, z1_t, tt[:, None, None, None]), z0_t, tt, ut_t)
            z = z0_t
            for k in range(rollout_steps):
                z = z + flow(z, z0_t, z.new_full((batch,), k * dt), ut_t) * dt
            pred_t = codec.decode(z * scale + center)

        loss = F.mse_loss(v_t.float(), z1_t - z0_t) + 4.0 * F.l1_loss(pred_t.float(), yt) + 2.0 * grad_l1(pred_t.float(), yt)
        loss = loss + 0.50 * lpips_fn((pred_t.float().clamp(-1, 1) + 1.0) * 0.5, (yt.clamp(-1, 1) + 1.0) * 0.5).mean()
        if use_od:
            loss = loss + 0.25 * od_stain_anchor_loss(pred_t.float(), yt)
        if use_replay:
            op_pi, op_he = src_pools[py_rng.randrange(len(src_pools))]
            si = torch.randint(len(op_pi), (batch,), generator=rng, device="cuda").cpu()
            xs, ys = augment_pair(op_pi[si].cuda(), op_he[si].cuda(), py_rng)
            xs_in = pi_acq_jitter(xs, py_rng, strength=0.08) if use_jitter else xs
            ys_in = stain_aug_source_he(ys, t_he, py_rng)
            with torch.no_grad():
                z0_s = (codec.encode(xs_in).float() - center) / scale
                z1_s = (codec.encode(ys_in).float() - center) / scale
                ut_s = None
                if uni_enc is not None and getattr(flow, "use_uni_2d", False):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ut_s = uni_enc(xs_in)[:, 1:].float()
            ts = torch.rand(batch, generator=rng, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_s = flow(torch.lerp(z0_s, z1_s, ts[:, None, None, None]), z0_s, ts, ut_s)
                rec_s = codec.decode(z1_s * scale + center)
            l_s = 0.6 * F.mse_loss(v_s.float(), z1_s - z0_s) + 0.6 * F.l1_loss(rec_s.float(), ys_in)
            loss = loss + 0.40 * l_s

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.5)
        opt.step()
        if use_swa:
            swad.maybe_update(step, steps)

    if use_swa:
        swad.apply()
    print(f"[{tag}] adapted in {time.perf_counter()-t0:.1f}s", flush=True)
    return flow.eval().cpu(), codec.eval().cpu()


def main():
    results = load_results()
    splits = build_or_load_lomo_splits(DATA_DIR, BASE / "eval_results", SPLITS_JSON, seed=2026)
    fd = load_fold_tensors(DATA_DIR, splits, "Breast", k_shot=20)
    src_pools = []
    for org in sorted(set(fd["src_groups"])):
        idx = [i for i, g in enumerate(fd["src_groups"]) if g == org]
        src_pools.append((fd["src_pi"][idx], fd["src_he"][idx]))

    gt_masks = load_gt_instance_masks(NPZ_PATH, "Breast")
    device = torch.device("cuda:0")
    uni_enc = UNIEncoder(freeze=True).to(device).eval()
    lpips_fn = build_per_image_lpips(net_type="alex", normalize=True).to(device)
    cell_model = models.CellposeModel(gpu=True, model_type="nuclei")

    codec_ckpt = torch.load(CKPT_DIR / "codec_rae0.08_s8800.pth", map_location="cpu", weights_only=False)
    codec0 = Codec()
    codec0.load_state_dict(codec_ckpt["state_dict"] if "state_dict" in codec_ckpt else codec_ckpt)

    ckpt_pre = torch.load(CKPT_DIR / "Breast__u1_up64_only_d192_pretrain.pth", map_location="cpu", weights_only=False)
    center, scale = ckpt_pre["center"], ckpt_pre["scale"]

    # Source validation setup (N=240)
    src_organs = ["Brain", "Kidney", "liver"]
    pi_paths, he_paths = [], []
    for org in src_organs:
        for n in splits["organs"][org]["val"]:
            pi_paths.append(DATA_DIR / org / "val" / "input" / n)
            he_paths.append(DATA_DIR / org / "val" / "target" / n)
    src_val_pi = tensor_from_paths(pi_paths)
    src_val_he = tensor_from_paths(he_paths)
    src_val_gt_hwc = [((src_val_he[i].numpy().transpose(1, 2, 0) + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8) for i in range(len(src_val_he))]
    src_gt_masks = []
    for i in range(0, len(src_val_gt_hwc), 24):
        masks, _, _ = cell_model.eval(src_val_gt_hwc[i:i+24], channels=[0, 0], diameter=None, batch_size=24)
        src_gt_masks.extend(masks)

    @torch.no_grad()
    def eval_source_val(flow, codec):
        pmap = infer_map(flow, codec, src_val_pi, [str(i) for i in range(len(src_val_pi))], uni_enc, center, scale, steps=4, batch=24)
        preds_hwc = [pmap[str(i)] for i in range(len(src_val_pi))]
        pr_masks = []
        for i in range(0, len(preds_hwc), 24):
            masks, _, _ = cell_model.eval(preds_hwc[i:i+24], channels=[0, 0], diameter=None, batch_size=24)
            pr_masks.extend(masks)
        psnrs, lps, dices = [], [], []
        for idx in range(len(preds_hwc)):
            psnrs.append(OfficialScalarMetrics.psnr(src_val_gt_hwc[idx], preds_hwc[idx]))
            dices.append(OfficialScalarMetrics.compute_dice_from_bins((src_gt_masks[idx] > 0).astype(np.float32), (pr_masks[idx] > 0).astype(np.float32)))
        for i in range(0, len(preds_hwc), 24):
            p_t = torch.from_numpy(np.stack(preds_hwc[i:i+24])).permute(0, 3, 1, 2).float().to(device) / 255.0
            g_t = (src_val_he[i:i+24].to(device) + 1.0) * 0.5
            lps.extend(lpips_fn(p_t, g_t).cpu().numpy().tolist())
        return {
            "src_psnr": round(float(np.mean(psnrs)), 2),
            "src_lpips": round(float(np.mean(lps)), 4),
            "src_dice": round(float(np.mean(dices)), 4),
        }

    # =========================================================================
    # EXP 1: Euler Sampling Steps S in {1, 2, 4, 8, 16} (Zero training!)
    # =========================================================================
    if len(results["steps_ablation"]) < 5:
        print("\n=== EXP 1: EULER SAMPLING STEPS S IN {1, 2, 4, 8, 16} ===", flush=True)
        flow0 = build_flow_model("u1_up64_only", dim=192)
        flow0.load_state_dict(ckpt_pre["swad"])
        ckpt_20 = torch.load(CKPT_DIR / "Breast__u1_up64_only_d192__m6_k20.pth", map_location="cpu", weights_only=False)
        flow20 = build_flow_model("u1_up64_only", dim=192)
        flow20.load_state_dict(ckpt_20["flow"])
        codec20 = Codec()
        codec20.load_state_dict(ckpt_20["codec"])

        for S in [1, 2, 4, 8, 16]:
            key = f"S={S}"
            if key in results["steps_ablation"]:
                continue
            ms = bench_latency_ms(flow20, codec20, fd["test_pi_t"], uni_enc, center, scale, steps=S)
            pm0 = infer_map(flow0, codec0, fd["test_pi_t"], fd["infer_ids"], uni_enc, center, scale, steps=S)
            m0 = eval_breast_target(pm0, fd, gt_masks, cell_model, lpips_fn, device)
            pm20 = infer_map(flow20, codec20, fd["test_pi_t"], fd["infer_ids"], uni_enc, center, scale, steps=S)
            m20 = eval_breast_target(pm20, fd, gt_masks, cell_model, lpips_fn, device)
            results["steps_ablation"][key] = {"latency_ms": round(ms, 2), "0shot": m0, "20shot": m20}
            print(f"S={S:2d} ({ms:5.2f}ms) | 0shot: PSNR={m0['psnr']} LPIPS={m0['lpips']} KID={m0['kid_x1e3']} | 20shot: PSNR={m20['psnr']} LPIPS={m20['lpips']} KID={m20['kid_x1e3']} Dice={m20['dice']}", flush=True)
            save_results(results)

    # =========================================================================
    # EXP 2: OASRA Component Ablation (Breast 20-shot + Source-Val N=240)
    # =========================================================================
    comp_specs = [
        ("NaiveFT+SWA+Jitter", False, False, True, True),
        ("OASRA_wo_LOD", False, True, True, True),
        ("OASRA_wo_Replay", True, False, True, True),
    ]
    for cname, u_od, u_rep, u_swa, u_jit in comp_specs:
        if cname in results["oasra_components"]:
            continue
        print(f"\n=== EXP 2: OASRA COMPONENT {cname} ===", flush=True)
        fl, co = custom_adapt(
            cname, build_flow_model("u1_up64_only", dim=192), ckpt_pre["swad"], codec0, center, scale,
            src_pools, fd["adapt_pi_t"], fd["adapt_he_t"], lpips_fn, uni_enc,
            use_od=u_od, use_replay=u_rep, use_swa=u_swa, use_jitter=u_jit, seed=7202, steps=1500,
        )
        pm = infer_map(fl, co, fd["test_pi_t"], fd["infer_ids"], uni_enc, center, scale, steps=4)
        tgt_m = eval_breast_target(pm, fd, gt_masks, cell_model, lpips_fn, device)
        fft16, ffth = compute_2d_fft_energy(pm, fd["gt_rgb_map"], fd["test_ids"])
        tgt_m["fft_16px"] = fft16
        tgt_m["fft_harm"] = ffth
        src_m = eval_source_val(fl, co)
        results["oasra_components"][cname] = {"target": tgt_m, "source_val": src_m}
        print(f"{cname:20s} | Tgt: PSNR={tgt_m['psnr']} KID={tgt_m['kid_x1e3']} Dice={tgt_m['dice']} | Src: PSNR={src_m['src_psnr']} Dice={src_m['src_dice']}", flush=True)
        save_results(results)

    # =========================================================================
    # EXP 3: Terminal 3x3 DWConv Ablation (UniFlow-up64 w/o dwconv)
    # =========================================================================
    if not results["dwconv_ablation"]:
        print("\n=== EXP 3: TERMINAL 3x3 DWCONV ABLATION (w/o dwconv) ===", flush=True)
        fl_nodw = build_flow_model("u1_up64_only", dim=192)
        fl_nodw.uni_hi_dw = nn.Identity()
        codec_eval = copy.deepcopy(codec0).cuda().eval()
        with torch.no_grad():
            src_lat_pi = (encode_all(codec_eval, fd["src_pi"]) - center) / scale
            src_lat_he = (encode_all(codec_eval, fd["src_he"]) - center) / scale
            val_lat_pi = (encode_all(codec_eval, fd["val_pi"]) - center) / scale
            val_lat_he = (encode_all(codec_eval, fd["val_he"]) - center) / scale
        src_uni_4v = precompute_uni_4views(uni_enc, fd["src_pi"])
        with torch.no_grad():
            val_u_list = []
            for i in range(0, len(fd["val_pi"]), 32):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    val_u_list.append(uni_enc(fd["val_pi"][i:i+32].cuda())[:, 1:].half().cpu())
            val_uni_v0 = torch.cat(val_u_list, dim=0)

        swad_st, _ = pretrain_flow(
            "Breast_up64_no_dwconv", fl_nodw, codec_eval, center, scale,
            src_lat_pi, src_lat_he, fd["src_pi"], fd["src_he"], src_uni_4v,
            val_lat_pi, val_lat_he, val_uni_v0,
            steps=9600, batch=8, seed=7102, use_on_policy_rollout=False, rollout_steps=4, lpips_fn=lpips_fn,
        )
        fl_0 = build_flow_model("u1_up64_only", dim=192)
        fl_0.uni_hi_dw = nn.Identity()
        fl_0.load_state_dict(swad_st)
        pm_0 = infer_map(fl_0, codec0, fd["test_pi_t"], fd["infer_ids"], uni_enc, center, scale, steps=4)
        m_0 = eval_breast_target(pm_0, fd, gt_masks, cell_model, lpips_fn, device)
        f16_0, fh_0 = compute_2d_fft_energy(pm_0, fd["gt_rgb_map"], fd["test_ids"])
        m_0["fft_16px"], m_0["fft_harm"] = f16_0, fh_0

        fl_ad, co_ad = custom_adapt(
            "no_dwconv_20shot", fl_0, swad_st, codec0, center, scale,
            src_pools, fd["adapt_pi_t"], fd["adapt_he_t"], lpips_fn, uni_enc,
            use_od=True, use_replay=True, use_swa=True, use_jitter=True, seed=7202, steps=1500,
        )
        pm_20 = infer_map(fl_ad, co_ad, fd["test_pi_t"], fd["infer_ids"], uni_enc, center, scale, steps=4)
        m_20 = eval_breast_target(pm_20, fd, gt_masks, cell_model, lpips_fn, device)
        f16_20, fh_20 = compute_2d_fft_energy(pm_20, fd["gt_rgb_map"], fd["test_ids"])
        m_20["fft_16px"], m_20["fft_harm"] = f16_20, fh_20
        results["dwconv_ablation"] = {"0shot": m_0, "20shot": m_20}
        print(f"w/o dwconv 0shot : PSNR={m_0['psnr']} bg_lap={m_0['bg_lap']} 16px-FFT={f16_0}% Harm-FFT={fh_0}%", flush=True)
        print(f"w/o dwconv 20shot: PSNR={m_20['psnr']} KID={m_20['kid_x1e3']} Dice={m_20['dice']} bg_lap={m_20['bg_lap']} 16px-FFT={f16_20}% Harm-FFT={fh_20}%", flush=True)
        save_results(results)

    # =========================================================================
    # EXP 4: 3-Seed Variance on Breast 20-shot (BaseFlow, NaiveFT, OASRA)
    # =========================================================================
    ckpt_base_pre = torch.load(CKPT_DIR / "Breast__u0_base_d192_pretrain.pth", map_location="cpu", weights_only=False)
    seed_jobs = [
        ("BaseFlow", "u0_base", ckpt_base_pre["swad"], ckpt_base_pre["center"], ckpt_base_pre["scale"], True, True, True, True),
        ("NaiveFT", "u1_up64_only", ckpt_pre["swad"], center, scale, False, False, False, False),
        ("OASRA", "u1_up64_only", ckpt_pre["swad"], center, scale, True, True, True, True),
    ]
    for mname, arch, st_pre, mc, ms_c, u_od, u_rep, u_swa, u_jit in seed_jobs:
        if mname not in results["seed_variance"]:
            results["seed_variance"][mname] = {}
        for sd in [8202, 9202]:
            sk = f"seed_{sd}"
            if sk in results["seed_variance"][mname]:
                continue
            print(f"\n=== EXP 4: 3-SEED VARIANCE {mname} {sk} ===", flush=True)
            fl, co = custom_adapt(
                f"{mname}_{sk}", build_flow_model(arch, dim=192), st_pre, codec0, mc, ms_c,
                src_pools, fd["adapt_pi_t"], fd["adapt_he_t"], lpips_fn, uni_enc,
                use_od=u_od, use_replay=u_rep, use_swa=u_swa, use_jitter=u_jit, seed=sd, steps=1500,
            )
            pm = infer_map(fl, co, fd["test_pi_t"], fd["infer_ids"], uni_enc, mc, ms_c, steps=4)
            tgt_m = eval_breast_target(pm, fd, gt_masks, cell_model, lpips_fn, device)
            results["seed_variance"][mname][sk] = tgt_m
            print(f"{mname} {sk} | PSNR={tgt_m['psnr']} SSIM={tgt_m['ssim']} LPIPS={tgt_m['lpips']} KID={tgt_m['kid_x1e3']} Dice={tgt_m['dice']}", flush=True)
            save_results(results)

    print("\n[ALL_EXTRA_ABLATIONS_COMPLETE] Saved ->", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
