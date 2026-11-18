# -*- coding: utf-8 -*-
"""Human Clinical AF->H&E LOMO Pipeline (HKUST UniStain Clean-256).

Supports:
  - Phase 2B Pilot: Compare AF normalization ('raw' vs 'p1p99') on Breast for BaseFlow & UniFlow.
  - Phase 2C Full LOMO: Run all 4 human organs (Breast, Kidney, Liver, Lung) with BaseFlow, UniFlow
    (0-shot, halfOD, 5/10/20-shot SRA, NaiveFT).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from cellpose import models as cp_models
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WS_ROOT = REPO_ROOT.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from round1_models import Codec, build_flow_model
from round1_losses_and_data import (
    adapt_flow,
    apply_od_scale,
    encode_all,
    infer_to_pngs,
    mean_od,
    precompute_uni_4views,
    pretrain_flow,
    rgb,
    train_fold_codec,
)
from uni_encoder import UNIEncoder
from v2_unified_eval import PAPER_CORE_KEYS, build_per_image_lpips, evaluate_split, load_gt_instance_masks

AF_ORGANS = ("Breast", "Kidney", "Liver", "Lung")
DEFAULT_AF_DATA = WS_ROOT / "03_洁净数据集与重建包" / "HKUST_UniStain_AF_clean256"
DEFAULT_AF_WORK = WS_ROOT / "03_洁净数据集与重建包" / "HKUST_UniStain_AF_runs"
DEFAULT_AF_NPZ = REPO_ROOT / "eval_cache" / "af_gt_nuclei_instances_256.npz"


def norm_af_array(arr_u8: np.ndarray, mode: str) -> np.ndarray:
    arr = arr_u8.astype(np.float32)
    if mode == "raw":
        return arr / 127.5 - 1.0
    if mode == "p1p99":
        g = arr.mean(axis=2)
        p1 = float(np.percentile(g, 1))
        p99 = float(np.percentile(g, 99))
        denom = max(p99 - p1, 12.0)
        scaled = np.clip((arr - p1) / denom, 0.0, 1.0)
        return scaled * 2.0 - 1.0
    raise ValueError(f"Unknown AF norm mode: {mode}")


def af_tensor_from_paths(paths, mode: str) -> torch.Tensor:
    arrs = [norm_af_array(rgb(p), mode) for p in paths]
    t = torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).float()
    return t


def he_tensor_from_paths(paths) -> torch.Tensor:
    arrs = [rgb(p) for p in paths]
    return torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def ensure_af_gt_masks(data_dir: Path, splits: dict, npz_path: Path, cell_model) -> None:
    if npz_path.exists():
        return
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    keys, masks_list = [], []
    for org in AF_ORGANS:
        ids = splits["organs"][org]["test"]
        imgs = [rgb(data_dir / org / "test" / "target" / nm) for nm in ids]
        org_masks = []
        for i in range(0, len(imgs), 24):
            mb, _, _ = cell_model.eval(imgs[i:i + 24], diameter=None, channels=[0, 0], batch_size=24)
            org_masks.extend(m.astype(np.uint16) for m in mb)
        for nm, m in zip(ids, org_masks):
            keys.append(f"{org}/{nm}")
            masks_list.append(m)
        print(f"[AF_GT_INST] {org:6s}: n={len(ids)} median_nuclei={float(np.median([m.max() for m in org_masks])):.1f}", flush=True)
    np.savez_compressed(npz_path, keys=np.array(keys), masks=np.stack(masks_list, axis=0))
    print(f"[AF_GT_INST] Saved -> {npz_path}", flush=True)


def load_af_fold_tensors(data_dir: Path, splits: dict, target_organ: str, af_mode: str, k_shot: int = 20) -> dict:
    src_pi, src_he, src_groups = [], [], []
    val_pi, val_he = [], []
    for org in AF_ORGANS:
        if org == target_organ:
            continue
        tr_ids = splits["organs"][org]["train"]
        va_ids = splits["organs"][org]["val"]
        src_pi.append(af_tensor_from_paths([data_dir / org / "train" / "input" / n for n in tr_ids], af_mode))
        src_he.append(he_tensor_from_paths([data_dir / org / "train" / "target" / n for n in tr_ids]))
        src_groups.extend([org] * len(tr_ids))
        val_pi.append(af_tensor_from_paths([data_dir / org / "val" / "input" / n for n in va_ids], af_mode))
        val_he.append(he_tensor_from_paths([data_dir / org / "val" / "target" / n for n in va_ids]))

    sp_t = splits["organs"][target_organ]
    ad_ids = sp_t[f"adapt_{k_shot}"]
    ref_ids = sp_t["refs_5"]
    test_ids = sp_t["test"]
    return {
        "src_pi": torch.cat(src_pi, dim=0),
        "src_he": torch.cat(src_he, dim=0),
        "src_groups": src_groups,
        "val_pi": torch.cat(val_pi, dim=0),
        "val_he": torch.cat(val_he, dim=0),
        "adapt_pi_t": af_tensor_from_paths([data_dir / target_organ / "adapt_pool" / "input" / n for n in ad_ids], af_mode),
        "adapt_he_t": he_tensor_from_paths([data_dir / target_organ / "adapt_pool" / "target" / n for n in ad_ids]),
        "refs_ids": ref_ids,
        "refs_pi_t": af_tensor_from_paths([data_dir / target_organ / "adapt_pool" / "input" / n for n in ref_ids], af_mode),
        "refs_od": mean_od([rgb(data_dir / target_organ / "adapt_pool" / "target" / n) for n in ref_ids], f"{target_organ}_refs5"),
        "test_ids": test_ids,
        "test_pi_t": af_tensor_from_paths([data_dir / target_organ / "test" / "input" / n for n in test_ids], af_mode),
        "pi_rgb_map": {n: rgb(data_dir / target_organ / "test" / "input" / n) for n in test_ids},
        "gt_rgb_map": {n: rgb(data_dir / target_organ / "test" / "target" / n) for n in test_ids},
    }


def update_af_summary(eval_dir: Path) -> None:
    metrics_dir = eval_dir / "metrics"
    records = [json.loads(jp.read_text(encoding="utf-8")) for jp in sorted(metrics_dir.glob("*.json"))]
    (eval_dir / "af_summary.json").write_text(json.dumps({"updated_at": datetime.now().isoformat(), "runs": records}, indent=2), encoding="utf-8")
    fields = [
        "run_tag", "af_mode", "fold", "arch", "stage", "strategy", "k_shot",
        "psnr", "ssim", "lpips", "kid_x1e3", "nuclei_dice", "nuclei_f1_05", "fid",
        "nuclei_dice_nonempty", "pcc_gt", "wall_seconds",
    ]
    with (eval_dir / "af_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            m = r["metrics"]
            w.writerow({
                "run_tag": r["run_tag"], "af_mode": r["af_mode"], "fold": r["fold"],
                "arch": r["arch"], "stage": r["stage"], "strategy": r["strategy"], "k_shot": r["k_shot"],
                "psnr": round(m["psnr"], 4), "ssim": round(m["ssim"], 4), "lpips": round(m["lpips"], 4),
                "kid_x1e3": round(m["kid"] * 1000.0, 3), "nuclei_dice": round(m["nuclei_dice"], 4),
                "nuclei_f1_05": round(m["nuclei_f1_05"], 4), "fid": round(m["fid"], 3),
                "nuclei_dice_nonempty": round(m["nuclei_dice_nonempty"], 4),
                "pcc_gt": round(m["pcc_gt"], 4), "wall_seconds": r["wall_seconds"],
            })
    md = [
        "# Human AF->H&E (HKUST UniStain Clean-256) Evaluation Summary",
        "",
        "| Run Tag | AF Mode | Fold | Arch | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID (×10⁻³) ↓ | Dice ↑ | F1@0.5 ↑ | FID ↓ |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        m = r["metrics"]
        md.append(
            f"| `{r['run_tag']}` | {r['af_mode']} | {r['fold']} | {r['arch']} | {r['stage']} | "
            f"{m['psnr']:.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['kid']*1000.0:.2f} | "
            f"{m['nuclei_dice']:.4f} | {m['nuclei_f1_05']:.4f} | {m['fid']:.2f} |"
        )
    (eval_dir / "af_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_AF_DATA)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_AF_WORK)
    ap.add_argument("--npz-path", type=Path, default=DEFAULT_AF_NPZ)
    ap.add_argument("--modes", nargs="+", choices=("raw", "p1p99"), default=["p1p99"])
    ap.add_argument("--folds", nargs="+", choices=AF_ORGANS, default=["Breast"])
    ap.add_argument("--archs", nargs="+", choices=("u0_base", "u1_up64_only"), default=["u0_base", "u1_up64_only"])
    ap.add_argument("--include-k5-k10-naive", action="store_true", default=False)
    ap.add_argument("--skip-done", action="store_true", default=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    splits = json.loads((args.data_dir / "af_lomo_splits.json").read_text(encoding="utf-8"))
    uni_enc = UNIEncoder(freeze=True).cuda().eval()
    lpips_fn = build_per_image_lpips(net_type="alex", normalize=True).to(device)
    cell_model = cp_models.CellposeModel(gpu=True, model_type="nuclei")
    ensure_af_gt_masks(args.data_dir, splits, args.npz_path, cell_model)

    for af_mode in args.modes:
        mode_dir = args.work_dir / af_mode
        eval_dir = mode_dir / "eval_results"
        for fold in args.folds:
            print(f"\n==================== AF MODE={af_mode} | FOLD={fold} ====================", flush=True)
            fd = load_af_fold_tensors(args.data_dir, splits, fold, af_mode, k_shot=20)
            gt_masks = load_gt_instance_masks(args.npz_path, fold)
            eval_names = fd["test_ids"]

            ckpt_fold = mode_dir / "checkpoints" / fold
            ckpt_fold.mkdir(parents=True, exist_ok=True)
            codec_path = ckpt_fold / "codec_rae0.08_s8800.pth"
            if args.skip_done and codec_path.exists():
                codec = Codec().cuda()
                codec.load_state_dict(torch.load(codec_path, map_location="cuda", weights_only=True))
                codec.eval().requires_grad_(False)
                print(f"[REUSE CODEC] {codec_path}", flush=True)
            else:
                codec, cinfo = train_fold_codec(fd["src_pi"], fd["src_he"], steps=8800, seed=7000, detail_weight=0.15, noise_max=0.08)
                torch.save(codec.state_dict(), codec_path)
                print(f"[CODEC] {af_mode}/{fold}: {cinfo}", flush=True)

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

            for arch in args.archs:
                base_tag = f"{fold}__{arch}_d192"
                pre_path = ckpt_fold / f"{base_tag}_pretrain.pth"
                if args.skip_done and pre_path.exists():
                    swad_st = torch.load(pre_path, map_location="cpu", weights_only=False)["swad"]
                    pre_s = 0.0
                    print(f"[REUSE PRETRAIN] {pre_path}", flush=True)
                else:
                    t0 = time.perf_counter()
                    swad_st, pinfo = pretrain_flow(
                        f"{af_mode}_{base_tag}", build_flow_model(arch, dim=192), codec, center, scale,
                        src_lat_pi, src_lat_he, fd["src_pi"], fd["src_he"], src_uni_4v,
                        val_lat_pi, val_lat_he, val_uni_v0,
                        steps=9600, batch=8, seed=7102, use_on_policy_rollout=False, rollout_steps=4, lpips_fn=lpips_fn,
                    )
                    torch.save({"swad": swad_st, "pinfo": pinfo, "center": center, "scale": scale}, pre_path)
                    pre_s = time.perf_counter() - t0

                def _record(tag: str, stage: str, strat: str, k_sh: int, pmap: dict, ws: float) -> None:
                    jp = eval_dir / "metrics" / f"{tag}.json"
                    if args.skip_done and jp.exists():
                        return
                    m = evaluate_split(eval_names, pmap, fd["pi_rgb_map"], fd["gt_rgb_map"], gt_masks, cell_model, lpips_fn, device, per_image_csv=eval_dir / "per_image" / f"{tag}.csv")
                    pay = {
                        "run_tag": tag, "af_mode": af_mode, "fold": fold, "arch": arch,
                        "stage": stage, "strategy": strat, "k_shot": k_sh, "wall_seconds": round(ws, 2),
                        "metrics": m,
                    }
                    jp.parent.mkdir(parents=True, exist_ok=True)
                    jp.write_text(json.dumps(pay, indent=2), encoding="utf-8")
                    update_af_summary(eval_dir)
                    print(f"[AF_EVAL] {af_mode:5s} | {tag:34s} | PSNR={m['psnr']:5.2f} SSIM={m['ssim']:.4f} LPIPS={m['lpips']:.4f} KID={m['kid']*1000:5.2f} Dice={m['nuclei_dice']:.4f} F1={m['nuclei_f1_05']:.4f}", flush=True)

                # 0-shot
                fl0 = build_flow_model(arch, dim=192)
                fl0.load_state_dict(swad_st)
                tag_0 = f"{base_tag}__0shot"
                if not (args.skip_done and (eval_dir / "metrics" / f"{tag_0}.json").exists()):
                    t_inf = time.perf_counter()
                    zs_map = infer_to_pngs(fl0, codec, fd["test_pi_t"], uni_enc, center, scale, eval_names, mode_dir / "generated" / fold / tag_0, steps=4)
                    _record(tag_0, "0shot", "pretrain", 0, zs_map, pre_s + (time.perf_counter() - t_inf))
                else:
                    zs_map = {nm: rgb(mode_dir / "generated" / fold / tag_0 / nm) for nm in eval_names}

                # 0-shot + halfOD
                tag_od = f"{base_tag}__0shot_halfOD"
                if not (args.skip_done and (eval_dir / "metrics" / f"{tag_od}.json").exists()):
                    t_od = time.perf_counter()
                    zs_refs = infer_to_pngs(fl0, codec, fd["refs_pi_t"], uni_enc, center, scale, fd["refs_ids"], mode_dir / "generated" / fold / f"_tmp_refs_{base_tag}", steps=4)
                    sc = fd["refs_od"] / mean_od([zs_refs[n] for n in fd["refs_ids"]], tag_od)
                    shutil.rmtree(mode_dir / "generated" / fold / f"_tmp_refs_{base_tag}", ignore_errors=True)
                    od_dir = mode_dir / "generated" / fold / tag_od
                    od_dir.mkdir(parents=True, exist_ok=True)
                    half_map = {}
                    for nm in eval_names:
                        im = np.rint((zs_map[nm].astype(np.float32) + apply_od_scale(zs_map[nm], sc).astype(np.float32)) / 2.0).clip(0, 255).astype(np.uint8)
                        half_map[nm] = im
                        Image.fromarray(im).save(od_dir / nm)
                    _record(tag_od, "0shot_halfOD", "halfOD", 5, half_map, time.perf_counter() - t_od)
                del fl0

                # Adaptations
                ad_jobs = [("m6", 20)]
                if args.include_k5_k10_naive and arch == "u1_up64_only":
                    ad_jobs.extend([("m6", 10), ("m6", 5), ("naive_ft", 20)])
                for strat, k_sh in ad_jobs:
                    tag_ad = f"{base_tag}__{strat}_k{k_sh}"
                    if args.skip_done and (eval_dir / "metrics" / f"{tag_ad}.json").exists():
                        continue
                    fd_k = fd if k_sh == 20 else load_af_fold_tensors(args.data_dir, splits, fold, af_mode, k_shot=k_sh)
                    t_ad = time.perf_counter()
                    fl_ad, co_ad, _ = adapt_flow(
                        f"{af_mode}_{tag_ad}", strat, build_flow_model(arch, dim=192), swad_st, codec, center, scale,
                        organ_pools, fd_k["adapt_pi_t"], fd_k["adapt_he_t"], lpips_fn, uni_enc,
                        steps=1500, batch=4, seed=7202, decoder_lr_mult=0.1, rollout_steps=4, od_weight=0.0,
                    )
                    torch.save({"flow": fl_ad.state_dict(), "codec": co_ad.state_dict(), "center": center, "scale": scale}, ckpt_fold / f"{tag_ad}.pth")
                    ad_map = infer_to_pngs(fl_ad, co_ad, fd["test_pi_t"], uni_enc, center, scale, eval_names, mode_dir / "generated" / fold / tag_ad, steps=4)
                    _record(tag_ad, f"{k_sh}shot", strat, k_sh, ad_map, time.perf_counter() - t_ad)
                    del fl_ad, co_ad
                    gc.collect()
                    torch.cuda.empty_cache()

            del codec, fd, src_lat_pi, src_lat_he, val_lat_pi, val_lat_he, src_uni_4v, val_uni_v0, organ_pools
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
