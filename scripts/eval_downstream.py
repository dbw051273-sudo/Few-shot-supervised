# -*- coding: utf-8 -*-
"""Objective, human-free downstream validation suite for UniFlow cross-organ virtual staining.

Implements:
  A) Independent Pathology VLMs (PLIP `vinid/plip` + CONCH `conch_ViT-B-16`, strictly excluding UNI):
     1. Paired image cosine similarity (CosSim) vs Ground-Truth H&E (per-organ & 4-organ mean)
     2. Cross-image retrieval (R@1, R@5, MRR) within-organ (N=120) and 4-organ pooled (N=480)
     3. 4-organ tissue classification consistency (Train-fit Linear Probe, Nearest-Centroid, Zero-shot Text)
  B) Nuclear Morphology & Compartment-Specific Stain OD (Cellpose + regionprops + Ruifrok deconvolution):
     1. Nuclear Area, Eccentricity, Solidity Pearson r, Spearman r_s & MAE vs GT (jointly detected nuclei)
     2. Compartment-specific Nuclear Hematoxylin OD & Extra-nuclear Stromal Eosin OD Pearson r, r_s & MAE

Results archived to `results/downstream_mouse/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from cellpose import models as cp_models
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from skimage.measure import regionprops
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WS_ROOT = REPO_ROOT.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from round1_losses_and_data import ORGANS, PINV, build_or_load_lomo_splits, rgb
from v2_unified_eval import load_gt_instance_masks

DEFAULT_DATA_DIR = WS_ROOT / "03_洁净数据集与重建包" / "UniPrior_四器官洁净均衡集_v2"
DEFAULT_R1_DIR = WS_ROOT / "07_本地3080Ti_Round1定型版产出"
DEFAULT_DG_DIR = WS_ROOT / "08_本地3080Ti_DG与Queue3产出"
DEFAULT_NPZ = WS_ROOT / "01_第一轮5880发车包与核心代码" / "小样本有监督_第一轮5880发车包" / "eval_cache" / "gt_nuclei_instances_256.npz"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "downstream_mouse"

ORGAN_DISPLAY = {"Breast": "Breast", "Brain": "Brain", "Kidney": "Kidney", "liver": "Liver"}
ORGAN_TEXT_PROMPTS = {
    "Breast": [
        "an H&E stained histopathology image of mouse breast mammary tissue",
        "histology slide showing mammary gland ducts and adipose stroma",
    ],
    "Brain": [
        "an H&E stained histopathology image of mouse brain cerebral tissue",
        "histology slide showing brain neuropil and neuronal nuclei",
    ],
    "Kidney": [
        "an H&E stained histopathology image of mouse kidney renal tissue",
        "histology slide showing renal tubules and glomeruli",
    ],
    "liver": [
        "an H&E stained histopathology image of mouse liver hepatic tissue",
        "histology slide showing liver hepatocytes and sinusoids",
    ],
}


def find_conch_ckpt() -> Path | None:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--MahmoodLab--CONCH" / "snapshots"
    if not hub.exists():
        return None
    hits = sorted(hub.glob("*/pytorch_model.bin"))
    return hits[-1] if hits else None


def discover_model_specs(r1_dir: Path, dg_dir: Path, extra_roots: list[Path]) -> list[dict]:
    """Discover all complete 4-organ (480-image) model cohorts on disk."""
    candidates = [
        ("UniFlow (0-shot)", "UniFlow", "0shot", r1_dir / "generated", "{org}__u1_up64_only_d192__0shot"),
        ("UniFlow (halfOD)", "UniFlow", "halfOD", r1_dir / "generated", "{org}__u1_up64_only_d192__0shot_halfOD"),
        ("UniFlow (20-shot)", "UniFlow", "20shot", r1_dir / "generated", "{org}__u1_up64_only_d192__m6_k20"),
        ("BaseFlow (0-shot)", "BaseFlow", "0shot", r1_dir / "generated", "{org}__u0_base_d192__0shot"),
        ("BaseFlow (halfOD)", "BaseFlow", "halfOD", r1_dir / "generated", "{org}__u0_base_d192__0shot_halfOD"),
        ("BaseFlow (20-shot)", "BaseFlow", "20shot", r1_dir / "generated", "{org}__u0_base_d192__m6_k20"),
        ("MixStyle (20-shot)", "MixStyle", "20shot", dg_dir / "generated", "{org}__mixstyle_d192__m6_k20"),
        ("StainAug (20-shot)", "StainAug", "20shot", dg_dir / "generated", "{org}__stainaug_d192__m6_k20"),
        ("DANN (20-shot)", "DANN", "20shot", dg_dir / "generated", "{org}__dann_d192__m6_k20"),
        ("MLDG (20-shot)", "MLDG", "20shot", dg_dir / "generated", "{org}__mldg_d192__m6_k20"),
    ]
    ext_patterns = [
        ("Pix2Pix (20-shot)", "Pix2Pix", "20shot", "{org}__pix2pix__20shot"),
        ("Pix2Pix (20-shot)", "Pix2Pix", "20shot", "{org}__pix2pix__m6_k20"),
        ("CycleGAN (20-shot)", "CycleGAN", "20shot", "{org}__cyclegan__20shot"),
        ("CycleGAN (20-shot)", "CycleGAN", "20shot", "{org}__cyclegan__m6_k20"),
        ("UNIStainNet (20-shot)", "UNIStainNet", "20shot", "{org}__unistainnet__20shot"),
        ("UNIStainNet (20-shot)", "UNIStainNet", "20shot", "{org}__unistainnet__m6_k20"),
        ("CUT (20-shot)", "CUT", "20shot", "{org}__cut__20shot"),
        ("I2SB (20-shot)", "I2SB", "20shot", "{org}__i2sb__20shot"),
    ]
    for root in [r1_dir / "generated", dg_dir / "generated"] + extra_roots:
        if not root.exists():
            continue
        for label, fam, st, pat in ext_patterns:
            candidates.append((label, fam, st, root, pat))

    seen_labels = set()
    found = []
    for label, fam, st, root, pat in candidates:
        if label in seen_labels or not root.exists():
            continue
        dirs = {org: root / org / pat.format(org=org) for org in ORGANS}
        if all(d.is_dir() and len(list(d.glob("*.png"))) >= 120 for d in dirs.values()):
            seen_labels.add(label)
            found.append({"label": label, "family": fam, "stage": st, "dirs": dirs})
    return found


@torch.no_grad()
def encode_plip_images(model: CLIPModel, processor: CLIPProcessor, img_paths: list[Path], device: torch.device, batch_size: int = 48) -> np.ndarray:
    embs = []
    for i in range(0, len(img_paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in img_paths[i:i + batch_size]]
        pix = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        feat = model.get_image_features(pixel_values=pix)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embs.append(feat.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_plip_text(model: CLIPModel, processor: CLIPProcessor, device: torch.device) -> np.ndarray:
    out = []
    for org in ORGANS:
        inputs = processor(text=ORGAN_TEXT_PROMPTS[org], return_tensors="pt", padding=True, truncation=True).to(device)
        tf = model.get_text_features(**inputs)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        mtf = tf.mean(dim=0)
        out.append((mtf / mtf.norm()).float().cpu().numpy())
    return np.stack(out, axis=0)


@torch.no_grad()
def encode_conch_images(model, preprocess, img_paths: list[Path], device: torch.device, batch_size: int = 48) -> np.ndarray:
    embs = []
    for i in range(0, len(img_paths), batch_size):
        tensors = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in img_paths[i:i + batch_size]]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            feat = model.encode_image(tensors)
        feat = feat.float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embs.append(feat.cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_conch_text(model, device: torch.device) -> np.ndarray:
    from conch.open_clip_custom import get_tokenizer, tokenize
    tok = get_tokenizer()
    out = []
    for org in ORGANS:
        t_in = tokenize(texts=ORGAN_TEXT_PROMPTS[org], tokenizer=tok).to(device)
        tf = model.encode_text(t_in).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
        mtf = tf.mean(dim=0)
        out.append((mtf / mtf.norm()).cpu().numpy())
    return np.stack(out, axis=0)


def compute_retrieval_metrics(q_emb: np.ndarray, g_emb: np.ndarray) -> dict[str, float]:
    sim = q_emb @ g_emb.T
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    targets = np.arange(n)[:, None]
    ranks = np.where(order == targets)[1] + 1
    return {
        "r1": round(float((ranks <= 1).mean() * 100.0), 2),
        "r5": round(float((ranks <= 5).mean() * 100.0), 2),
        "mrr": round(float((1.0 / ranks).mean()), 4),
    }


def eval_vlm_suite(
    q_emb: np.ndarray,
    gt_emb: np.ndarray,
    y_test: np.ndarray,
    lin_probe: LogisticRegression,
    centroids: np.ndarray,
    text_protos: np.ndarray,
) -> dict:
    paired_cos = np.sum(q_emb * gt_emb, axis=1)
    cos_by_org, within_list = {}, []
    for o_idx, org in enumerate(ORGANS):
        idx = np.where(y_test == o_idx)[0]
        cos_by_org[ORGAN_DISPLAY[org]] = round(float(paired_cos[idx].mean()), 4)
        within_list.append(compute_retrieval_metrics(q_emb[idx], gt_emb[idx]))
    within_mean = {
        "r1": round(float(np.mean([w["r1"] for w in within_list])), 2),
        "r5": round(float(np.mean([w["r5"] for w in within_list])), 2),
        "mrr": round(float(np.mean([w["mrr"] for w in within_list])), 4),
        "by_organ": {ORGAN_DISPLAY[o]: w for o, w in zip(ORGANS, within_list)},
    }
    pooled_ret = compute_retrieval_metrics(q_emb, gt_emb)
    p_lin = lin_probe.predict(q_emb)
    p_cen = np.argmax(q_emb @ centroids.T, axis=1)
    p_txt = np.argmax(q_emb @ text_protos.T, axis=1)
    return {
        "cossim_mean": round(float(paired_cos.mean()), 4),
        "cossim_by_organ": cos_by_org,
        "retrieval_within_120": within_mean,
        "retrieval_pooled_480": pooled_ret,
        "organ_cls": {
            "linear_probe": {"acc": round(float(accuracy_score(y_test, p_lin) * 100.0), 2), "f1": round(float(f1_score(y_test, p_lin, average="macro") * 100.0), 2)},
            "centroid": {"acc": round(float(accuracy_score(y_test, p_cen) * 100.0), 2), "f1": round(float(f1_score(y_test, p_cen, average="macro") * 100.0), 2)},
            "zero_shot_text": {"acc": round(float(accuracy_score(y_test, p_txt) * 100.0), 2), "f1": round(float(f1_score(y_test, p_txt, average="macro") * 100.0), 2)},
        },
    }


def fit_vlm_classifiers(x_train: np.ndarray, y_train: np.ndarray) -> tuple[LogisticRegression, np.ndarray]:
    lin_probe = LogisticRegression(C=10.0, max_iter=1000, random_state=2026)
    lin_probe.fit(x_train, y_train)
    centroids = np.stack([x_train[y_train == c].mean(axis=0) for c in range(len(ORGANS))], axis=0)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    return lin_probe, centroids


def extract_nuclei_morphology(inst_mask: np.ndarray) -> tuple[float, float, float, int]:
    if int(inst_mask.max()) == 0:
        return 0.0, 0.0, 0.0, 0
    props = regionprops(inst_mask.astype(np.int32))
    if not props:
        return 0.0, 0.0, 0.0, 0
    return (
        float(np.mean([p.area for p in props])),
        float(np.mean([p.eccentricity for p in props])),
        float(np.mean([p.solidity for p in props])),
        len(props),
    )


def extract_compartment_od(img_rgb: np.ndarray, nuc_mask: np.ndarray) -> tuple[float, float]:
    od_map = -np.log((img_rgb.astype(np.float64) + 1.0) / 256.0)
    tissue = od_map.sum(axis=2) > 0.15
    conc = (PINV @ od_map.reshape(-1, 3).T).T.reshape(256, 256, 2)
    h_ch = np.clip(conc[:, :, 0], 0.0, None)
    e_ch = np.clip(conc[:, :, 1], 0.0, None)
    in_nuc = nuc_mask > 0
    in_stroma = (~in_nuc) & tissue
    nuc_h = float(h_ch[in_nuc].mean()) if in_nuc.any() else float(h_ch[tissue].mean() if tissue.any() else 0.0)
    str_e = float(e_ch[in_stroma].mean()) if in_stroma.any() else float(e_ch[tissue].mean() if tissue.any() else 0.0)
    return nuc_h, str_e


def corr_and_mae(gt_vals: np.ndarray, pr_vals: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(pr_vals - gt_vals)))
    if len(gt_vals) < 3 or np.std(gt_vals) < 1e-8 or np.std(pr_vals) < 1e-8:
        return {"r": 0.0, "rs": 0.0, "mae": round(mae, 4)}
    r, _ = pearsonr(gt_vals, pr_vals)
    rs, _ = spearmanr(gt_vals, pr_vals)
    return {
        "r": 0.0 if np.isnan(r) else round(float(r), 4),
        "rs": 0.0 if np.isnan(rs) else round(float(rs), 4),
        "mae": round(mae, 4),
    }


def run_cellpose_cached(cp_model, img_paths: list[Path], cache_npz: Path) -> dict[str, np.ndarray]:
    if cache_npz.exists():
        data = np.load(cache_npz)
        return {str(k): m for k, m in zip(data["keys"], data["masks"])}
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    imgs = [rgb(p) for p in img_paths]
    masks_all = []
    for i in range(0, len(imgs), 24):
        masks, _, _ = cp_model.eval(imgs[i:i + 24], channels=[0, 0], diameter=None, batch_size=24)
        masks_all.extend([m.astype(np.uint16) for m in masks])
    keys_arr = np.array([f"{p.parent.parent.name}/{p.name}" for p in img_paths])
    masks_arr = np.stack(masks_all, axis=0)
    np.savez_compressed(cache_npz, keys=keys_arr, masks=masks_arr)
    return {str(k): m for k, m in zip(keys_arr, masks_arr)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--r1-dir", type=Path, default=DEFAULT_R1_DIR)
    ap.add_argument("--dg-dir", type=Path, default=DEFAULT_DG_DIR)
    ap.add_argument("--extra-dirs", nargs="*", type=Path, default=[])
    ap.add_argument("--splits-json", type=Path, default=DEFAULT_R1_DIR / "eval_results" / "round1_splits.json")
    ap.add_argument("--gt-npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-cache", action="store_true", default=False)
    ap.add_argument("--no-conch", action="store_true", default=False)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    cache_dir = out_dir / "cache"
    per_img_dir = out_dir / "per_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_img_dir.mkdir(parents=True, exist_ok=True)

    splits = build_or_load_lomo_splits(args.data_dir, REPO_ROOT / "eval_cache", args.splits_json, seed=2026)
    test_items: list[tuple[str, str, int, Path]] = []
    tr_paths, tr_y = [], []
    for o_idx, org in enumerate(ORGANS):
        for nm in splits["organs"][org]["train"]:
            tr_paths.append(args.data_dir / org / "train" / "target" / nm)
            tr_y.append(o_idx)
        for nm in splits["organs"][org]["test"]:
            test_items.append((org, nm, o_idx, args.data_dir / org / "test" / "target" / nm))
    gt_paths = [it[3] for it in test_items]
    y_train = np.array(tr_y)
    y_test = np.array([it[2] for it in test_items])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup PLIP
    plip_model = CLIPModel.from_pretrained("vinid/plip", local_files_only=True).to(device).eval()
    plip_proc = CLIPProcessor.from_pretrained("vinid/plip", local_files_only=True)
    plip_tr_npz = cache_dir / "plip_train_gt_1920.npz"
    plip_te_npz = cache_dir / "plip_test_gt_480.npz"
    x_tr_plip = np.load(plip_tr_npz)["emb"] if (plip_tr_npz.exists() and not args.no_cache) else encode_plip_images(plip_model, plip_proc, tr_paths, device)
    if not plip_tr_npz.exists() or args.no_cache:
        np.savez_compressed(plip_tr_npz, emb=x_tr_plip, y=y_train)
    gt_plip = np.load(plip_te_npz)["emb"] if (plip_te_npz.exists() and not args.no_cache) else encode_plip_images(plip_model, plip_proc, gt_paths, device)
    if not plip_te_npz.exists() or args.no_cache:
        np.savez_compressed(plip_te_npz, emb=gt_plip)
    plip_lin, plip_cen = fit_vlm_classifiers(x_tr_plip, y_train)
    plip_txt = encode_plip_text(plip_model, plip_proc, device)

    # Setup CONCH (optional but preferred dual-VLM verification)
    conch_ckpt = None if args.no_conch else find_conch_ckpt()
    conch_ok = False
    conch_model = conch_pp = conch_lin = conch_cen = conch_txt = gt_conch = None
    if conch_ckpt is not None:
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            conch_model, conch_pp = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(conch_ckpt), device=str(device))
            conch_model.eval()
            conch_tr_npz = cache_dir / "conch_train_gt_1920.npz"
            conch_te_npz = cache_dir / "conch_test_gt_480.npz"
            x_tr_conch = np.load(conch_tr_npz)["emb"] if (conch_tr_npz.exists() and not args.no_cache) else encode_conch_images(conch_model, conch_pp, tr_paths, device)
            if not conch_tr_npz.exists() or args.no_cache:
                np.savez_compressed(conch_tr_npz, emb=x_tr_conch, y=y_train)
            gt_conch = np.load(conch_te_npz)["emb"] if (conch_te_npz.exists() and not args.no_cache) else encode_conch_images(conch_model, conch_pp, gt_paths, device)
            if not conch_te_npz.exists() or args.no_cache:
                np.savez_compressed(conch_te_npz, emb=gt_conch)
            conch_lin, conch_cen = fit_vlm_classifiers(x_tr_conch, y_train)
            conch_txt = encode_conch_text(conch_model, device)
            conch_ok = True
        except Exception as exc:
            print(f"[WARN] CONCH unavailable ({exc}); running PLIP only.", flush=True)
    print(f"[INIT] PLIP=True | CONCH={conch_ok}", flush=True)

    # GT morphology & OD
    gt_masks_by_org = {org: load_gt_instance_masks(args.gt_npz, org) for org in ORGANS}
    gt_imgs = [rgb(p) for p in gt_paths]
    gt_morph = np.asarray([extract_nuclei_morphology(gt_masks_by_org[o][nm]) for (o, nm, _, _) in test_items], dtype=np.float64)
    gt_od = np.asarray([extract_compartment_od(im, gt_masks_by_org[o][nm]) for (o, nm, _, _), im in zip(test_items, gt_imgs)], dtype=np.float64)
    gt_ne = gt_morph[:, 3] > 0

    gt_plip_eval = eval_vlm_suite(gt_plip, gt_plip, y_test, plip_lin, plip_cen, plip_txt)
    gt_conch_eval = eval_vlm_suite(gt_conch, gt_conch, y_test, conch_lin, conch_cen, conch_txt) if conch_ok else None
    print(f"[GT PLIP ] Lin={gt_plip_eval['organ_cls']['linear_probe']['acc']}% Cen={gt_plip_eval['organ_cls']['centroid']['acc']}% TextZS={gt_plip_eval['organ_cls']['zero_shot_text']['acc']}%", flush=True)
    if conch_ok:
        print(f"[GT CONCH] Lin={gt_conch_eval['organ_cls']['linear_probe']['acc']}% Cen={gt_conch_eval['organ_cls']['centroid']['acc']}% TextZS={gt_conch_eval['organ_cls']['zero_shot_text']['acc']}%", flush=True)

    specs = discover_model_specs(args.r1_dir, args.dg_dir, args.extra_dirs)
    cp_model = cp_models.CellposeModel(gpu=torch.cuda.is_available(), model_type="nuclei")

    summary_records = [{
        "label": "Real H&E (GT)", "family": "GT", "stage": "GT", "n_test": 480,
        "plip": gt_plip_eval,
        "conch": gt_conch_eval,
        "morphology": {"nonempty_gt_n": int(gt_ne.sum()), "gt_means": {"area": round(float(gt_morph[gt_ne, 0].mean()), 2), "ecc": round(float(gt_morph[gt_ne, 1].mean()), 4), "sol": round(float(gt_morph[gt_ne, 2].mean()), 4)}},
        "compartment_od": {"gt_means": {"nuc_h": round(float(gt_od[:, 0].mean()), 4), "stroma_e": round(float(gt_od[:, 1].mean()), 4)}},
    }]

    for spec in specs:
        label = spec["label"]
        safe_key = label.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        pred_paths = [spec["dirs"][org] / nm for org, nm, _, _ in test_items]

        # PLIP
        emb_plip_p = cache_dir / f"plip_{safe_key}.npz"
        q_plip = np.load(emb_plip_p)["emb"] if (emb_plip_p.exists() and not args.no_cache) else encode_plip_images(plip_model, plip_proc, pred_paths, device)
        if not emb_plip_p.exists() or args.no_cache:
            np.savez_compressed(emb_plip_p, emb=q_plip)
        plip_res = eval_vlm_suite(q_plip, gt_plip, y_test, plip_lin, plip_cen, plip_txt)

        # CONCH
        conch_res = None
        if conch_ok:
            emb_conch_p = cache_dir / f"conch_{safe_key}.npz"
            q_conch = np.load(emb_conch_p)["emb"] if (emb_conch_p.exists() and not args.no_cache) else encode_conch_images(conch_model, conch_pp, pred_paths, device)
            if not emb_conch_p.exists() or args.no_cache:
                np.savez_compressed(emb_conch_p, emb=q_conch)
            conch_res = eval_vlm_suite(q_conch, gt_conch, y_test, conch_lin, conch_cen, conch_txt)

        # Cellpose Morphology + Compartment OD
        pr_masks = run_cellpose_cached(cp_model, pred_paths, cache_dir / f"cellpose_{safe_key}.npz")
        pr_morph, pr_od_g, pr_od_s, per_rows = [], [], [], []
        paired_cos_p = np.sum(q_plip * gt_plip, axis=1)
        paired_cos_c = np.sum(q_conch * gt_conch, axis=1) if conch_ok else np.zeros(480)
        for idx_i, ((org, nm, _, _), im_p) in enumerate(zip(test_items, pred_paths)):
            im_pr = rgb(im_p)
            pm = pr_masks[f"{org}/{nm}"]
            gm = gt_masks_by_org[org][nm]
            area, ecc, sol, cnt = extract_nuclei_morphology(pm)
            nh_g, se_g = extract_compartment_od(im_pr, gm)
            nh_s, se_s = extract_compartment_od(im_pr, pm)
            pr_morph.append((area, ecc, sol, cnt))
            pr_od_g.append((nh_g, se_g))
            pr_od_s.append((nh_s, se_s))
            per_rows.append({
                "organ": org, "name": nm,
                "plip_cossim": round(float(paired_cos_p[idx_i]), 5),
                "conch_cossim": round(float(paired_cos_c[idx_i]), 5) if conch_ok else "",
                "gt_cnt": int(gt_morph[idx_i, 3]), "pr_cnt": cnt,
                "gt_area": round(gt_morph[idx_i, 0], 3), "pr_area": round(area, 3),
                "gt_ecc": round(gt_morph[idx_i, 1], 4), "pr_ecc": round(ecc, 4),
                "gt_sol": round(gt_morph[idx_i, 2], 4), "pr_sol": round(sol, 4),
                "gt_nuc_h": round(gt_od[idx_i, 0], 4), "pr_nuc_h_gtmask": round(nh_g, 4),
                "gt_str_e": round(gt_od[idx_i, 1], 4), "pr_str_e_gtmask": round(se_g, 4),
            })

        with (per_img_dir / f"{safe_key}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            w.writeheader()
            w.writerows(per_rows)

        pr_morph_arr = np.asarray(pr_morph, dtype=np.float64)
        pr_od_g_arr = np.asarray(pr_od_g, dtype=np.float64)
        pr_od_s_arr = np.asarray(pr_od_s, dtype=np.float64)
        joint_valid = gt_ne & (pr_morph_arr[:, 3] > 0)

        area_st = corr_and_mae(gt_morph[joint_valid, 0], pr_morph_arr[joint_valid, 0])
        ecc_st = corr_and_mae(gt_morph[joint_valid, 1], pr_morph_arr[joint_valid, 1])
        sol_st = corr_and_mae(gt_morph[joint_valid, 2], pr_morph_arr[joint_valid, 2])
        nh_st = corr_and_mae(gt_od[:, 0], pr_od_g_arr[:, 0])
        se_st = corr_and_mae(gt_od[:, 1], pr_od_g_arr[:, 1])
        nhs_st = corr_and_mae(gt_od[:, 0], pr_od_s_arr[:, 0])
        ses_st = corr_and_mae(gt_od[:, 1], pr_od_s_arr[:, 1])

        rec = {
            "label": label, "family": spec["family"], "stage": spec["stage"], "n_test": 480,
            "plip": plip_res,
            "conch": conch_res,
            "morphology": {
                "nonempty_gt_n": int(gt_ne.sum()),
                "joint_valid_n": int(joint_valid.sum()),
                "area": area_st, "eccentricity": ecc_st, "solidity": sol_st,
            },
            "compartment_od": {
                "gt_mask_nuc_h": nh_st, "gt_mask_stroma_e": se_st,
                "self_mask_nuc_h": nhs_st, "self_mask_stroma_e": ses_st,
            },
        }
        summary_records.append(rec)
        c_msg = (
            f"CONCH Cos={conch_res['cossim_mean']:.4f} R480={conch_res['retrieval_pooled_480']['r1']:5.2f}% Lin={conch_res['organ_cls']['linear_probe']['acc']:5.2f}% ZS={conch_res['organ_cls']['zero_shot_text']['acc']:5.2f}%"
            if conch_ok else ""
        )
        print(
            f"[DOWNSTREAM] {label:20s} | PLIP Cos={plip_res['cossim_mean']:.4f} R480={plip_res['retrieval_pooled_480']['r1']:5.2f}% Lin={plip_res['organ_cls']['linear_probe']['acc']:5.2f}% | "
            f"{c_msg} | Area rs={area_st['rs']:.4f} Ecc r={ecc_st['r']:.4f} Sol r={sol_st['r']:.4f} | "
            f"NucH r={nh_st['r']:.4f} StrE r={se_st['r']:.4f}",
            flush=True,
        )

    payload = {
        "updated_at": datetime.now().isoformat(),
        "vlms": ["vinid/plip"] + (["MahmoodLab/CONCH (conch_ViT-B-16)"] if conch_ok else []),
        "runs": summary_records,
    }
    (out_dir / "downstream_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save flat CSV & Markdown
    fields = [
        "label", "family", "stage",
        "plip_cossim", "plip_ret120_r1", "plip_ret120_r5", "plip_ret120_mrr", "plip_ret480_r1", "plip_ret480_r5", "plip_ret480_mrr",
        "plip_lin_acc", "plip_lin_f1", "plip_cen_acc", "plip_cen_f1",
        "conch_cossim", "conch_ret120_r1", "conch_ret120_r5", "conch_ret120_mrr", "conch_ret480_r1", "conch_ret480_r5", "conch_ret480_mrr",
        "conch_lin_acc", "conch_lin_f1", "conch_cen_acc", "conch_txt_acc", "conch_txt_f1",
        "joint_valid_n", "area_r", "area_rs", "area_mae", "ecc_r", "ecc_rs", "ecc_mae", "sol_r", "sol_rs", "sol_mae",
        "nuc_h_r", "nuc_h_mae", "stroma_e_r", "stroma_e_mae",
    ]
    with (out_dir / "downstream_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_records:
            if r["family"] == "GT":
                continue
            pl, co, mo, od = r["plip"], r["conch"], r["morphology"], r["compartment_od"]
            w.writerow({
                "label": r["label"], "family": r["family"], "stage": r["stage"],
                "plip_cossim": pl["cossim_mean"],
                "plip_ret120_r1": pl["retrieval_within_120"]["r1"], "plip_ret120_r5": pl["retrieval_within_120"]["r5"], "plip_ret120_mrr": pl["retrieval_within_120"]["mrr"],
                "plip_ret480_r1": pl["retrieval_pooled_480"]["r1"], "plip_ret480_r5": pl["retrieval_pooled_480"]["r5"], "plip_ret480_mrr": pl["retrieval_pooled_480"]["mrr"],
                "plip_lin_acc": pl["organ_cls"]["linear_probe"]["acc"], "plip_lin_f1": pl["organ_cls"]["linear_probe"]["f1"],
                "plip_cen_acc": pl["organ_cls"]["centroid"]["acc"], "plip_cen_f1": pl["organ_cls"]["centroid"]["f1"],
                "conch_cossim": co["cossim_mean"] if co else "",
                "conch_ret120_r1": co["retrieval_within_120"]["r1"] if co else "", "conch_ret120_r5": co["retrieval_within_120"]["r5"] if co else "", "conch_ret120_mrr": co["retrieval_within_120"]["mrr"] if co else "",
                "conch_ret480_r1": co["retrieval_pooled_480"]["r1"] if co else "", "conch_ret480_r5": co["retrieval_pooled_480"]["r5"] if co else "", "conch_ret480_mrr": co["retrieval_pooled_480"]["mrr"] if co else "",
                "conch_lin_acc": co["organ_cls"]["linear_probe"]["acc"] if co else "", "conch_lin_f1": co["organ_cls"]["linear_probe"]["f1"] if co else "",
                "conch_cen_acc": co["organ_cls"]["centroid"]["acc"] if co else "",
                "conch_txt_acc": co["organ_cls"]["zero_shot_text"]["acc"] if co else "", "conch_txt_f1": co["organ_cls"]["zero_shot_text"]["f1"] if co else "",
                "joint_valid_n": mo["joint_valid_n"],
                "area_r": mo["area"]["r"], "area_rs": mo["area"]["rs"], "area_mae": mo["area"]["mae"],
                "ecc_r": mo["eccentricity"]["r"], "ecc_rs": mo["eccentricity"]["rs"], "ecc_mae": mo["eccentricity"]["mae"],
                "sol_r": mo["solidity"]["r"], "sol_rs": mo["solidity"]["rs"], "sol_mae": mo["solidity"]["mae"],
                "nuc_h_r": od["gt_mask_nuc_h"]["r"], "nuc_h_mae": od["gt_mask_nuc_h"]["mae"],
                "stroma_e_r": od["gt_mask_stroma_e"]["r"], "stroma_e_mae": od["gt_mask_stroma_e"]["mae"],
            })

    md = [
        "# Mouse 4-Organ Downstream Verification Summary (N = 480)",
        "",
        f"- Independent VLMs: `PLIP (vinid/plip)` | `CONCH (conch_ViT-B-16)` = `{conch_ok}`",
        f"- GT PLIP Organ Acc: Linear `{gt_plip_eval['organ_cls']['linear_probe']['acc']:.2f}%` | Centroid `{gt_plip_eval['organ_cls']['centroid']['acc']:.2f}%`",
    ]
    if conch_ok:
        md.append(
            f"- GT CONCH Organ Acc: Linear `{gt_conch_eval['organ_cls']['linear_probe']['acc']:.2f}%` | "
            f"Centroid `{gt_conch_eval['organ_cls']['centroid']['acc']:.2f}%` | "
            f"Zero-Shot Text `{gt_conch_eval['organ_cls']['zero_shot_text']['acc']:.2f}%`"
        )
    md.extend([
        "",
        "## A. Independent VLM Semantic Similarity, Retrieval & Organ Classification",
        "| Model / Stage | PLIP CosSim | PLIP R@1(120/480) | PLIP LinAcc | CONCH CosSim | CONCH R@1(120/480) | CONCH MRR(480) | CONCH LinAcc / ZS-Text |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in summary_records:
        if r["family"] == "GT":
            continue
        pl, co = r["plip"], r["conch"]
        co_str = (
            f"{co['cossim_mean']:.4f} | {co['retrieval_within_120']['r1']:.2f}% / {co['retrieval_pooled_480']['r1']:.2f}% | "
            f"{co['retrieval_pooled_480']['mrr']:.4f} | {co['organ_cls']['linear_probe']['acc']:.2f}% / {co['organ_cls']['zero_shot_text']['acc']:.2f}%"
            if co else "-- | -- | -- | --"
        )
        md.append(
            f"| `{r['label']}` | {pl['cossim_mean']:.4f} | "
            f"{pl['retrieval_within_120']['r1']:.2f}% / {pl['retrieval_pooled_480']['r1']:.2f}% | "
            f"{pl['organ_cls']['linear_probe']['acc']:.2f}% | {co_str} |"
        )
    md.extend([
        "",
        "## B. Cellpose Nuclear Morphology & Compartment-Specific OD",
        "| Model / Stage | Valid N | Area (r_s / MAE) | Eccentricity (r / MAE) | Solidity (r / MAE) | Nuc-H OD (r / MAE) | Stroma-E OD (r / MAE) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in summary_records:
        if r["family"] == "GT":
            continue
        mo, od = r["morphology"], r["compartment_od"]
        md.append(
            f"| `{r['label']}` | {mo['joint_valid_n']}/{mo['nonempty_gt_n']} | "
            f"{mo['area']['rs']:.4f} / {mo['area']['mae']:.2f} | "
            f"{mo['eccentricity']['r']:.4f} / {mo['eccentricity']['mae']:.4f} | "
            f"{mo['solidity']['r']:.4f} / {mo['solidity']['mae']:.4f} | "
            f"{od['gt_mask_nuc_h']['r']:.4f} / {od['gt_mask_nuc_h']['mae']:.4f} | "
            f"{od['gt_mask_stroma_e']['r']:.4f} / {od['gt_mask_stroma_e']['mae']:.4f} |"
        )
    (out_dir / "downstream_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[DOWNSTREAM_COMPLETE] Saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
