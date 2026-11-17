# -*- coding: utf-8 -*-
"""Generate Nature-style publication figures for UniFlow using Alto-R/nature-figure-craft."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
NFC_ASSETS = HERE / "nature-figure-craft" / "nature-figure-production" / "assets"
sys.path.insert(0, str(NFC_ASSETS))

import nature_style as ns  # noqa: E402

ROOT = HERE.parent
DATA_DIR = ROOT / "03_洁净数据集与重建包" / "UniPrior_四器官洁净均衡集_v2"
R1_DIR = ROOT / "07_本地3080Ti_Round1定型版产出"
DG_DIR = ROOT / "08_本地3080Ti_DG与Queue3产出"
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ns.set_outdir(OUT_DIR)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)


def _select_tissue_and_roi(splits: dict, organ: str) -> tuple[str, tuple[int, int, int, int]]:
    """Select a representative non-blank test sample and a high-gradient 64x64 ROI."""
    best_name = splits["organs"][organ]["test"][0]
    best_score = -1.0
    for nm in splits["organs"][organ]["test"][:25]:
        gt = _load_rgb(DATA_DIR / organ / "test" / "target" / nm)
        tissue_frac = float((gt.min(axis=-1) < 225).mean())
        if tissue_frac < 0.45:
            continue
        p_our = _load_rgb(R1_DIR / "generated" / organ / f"{organ}__u1_up64_only_d192__m6_k20" / nm)
        mse = float(np.mean((gt.astype(np.float32) - p_our.astype(np.float32)) ** 2))
        psnr = 10.0 * np.log10((255.0 ** 2) / max(mse, 1e-6))
        if abs(tissue_frac - 0.72) < 0.35 and psnr > best_score:
            best_score = psnr
            best_name = nm
    gt = _load_rgb(DATA_DIR / organ / "test" / "target" / best_name).astype(np.float32)
    gray = gt.mean(axis=-1)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = gx + gy
    best_g, best_rc = -1.0, (48, 48)
    for r in range(16, 176, 16):
        for c in range(16, 176, 16):
            g = float(grad[r:r + 64, c:c + 64].mean())
            if g > best_g:
                best_g, best_rc = g, (r, c)
    return best_name, (best_rc[0], best_rc[0] + 64, best_rc[1], best_rc[1] + 64)


def _embed_zoom(img: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Draw red 64x64 ROI rectangle and bottom-right 2x zoomed inset."""
    r0, r1, c0, c1 = roi
    canvas = img.copy()
    # Zoom inset size: 96x96 in bottom-right corner
    z_sz = 96
    crop = Image.fromarray(img[r0:r1, c0:c1]).resize((z_sz, z_sz), Image.Resampling.BILINEAR)
    y0, x0 = 256 - z_sz - 4, 256 - z_sz - 4
    canvas[y0 - 2:y0 + z_sz + 2, x0 - 2:x0 + z_sz + 2] = np.array([255, 255, 255], dtype=np.uint8)
    canvas[y0:y0 + z_sz, x0:x0 + z_sz] = np.asarray(crop)
    # Red borders
    red = np.array([214, 39, 40], dtype=np.uint8)
    canvas[r0:r0 + 2, c0:c1] = red
    canvas[r1 - 2:r1, c0:c1] = red
    canvas[r0:r1, c0:c0 + 2] = red
    canvas[r0:r1, c1 - 2:c1] = red
    canvas[y0 - 2:y0, x0 - 2:x0 + z_sz + 2] = red
    canvas[y0 + z_sz:y0 + z_sz + 2, x0 - 2:x0 + z_sz + 2] = red
    canvas[y0 - 2:y0 + z_sz + 2, x0 - 2:x0] = red
    canvas[y0 - 2:y0 + z_sz + 2, x0 + z_sz:x0 + z_sz + 2] = red
    return canvas


def build_fig1_arch() -> None:
    """Nature-style 3-panel architecture, pretraining, and K-shot adaptation diagram."""
    p = ns.panel(placed_mm=180, aspect=0.60, body_pt=8.0)
    ns.apply_style(p)
    fig, ax = plt.subplots(figsize=p.figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    c_blue = "#3C5488"
    c_blue_bg = "#EDF1F7"
    c_teal = "#00A087"
    c_teal_bg = "#E8F6F3"
    c_orange = "#E64B35"
    c_orange_bg = "#FDEEEB"
    c_slate = "#4A5568"
    c_gray_bg = "#F7F8FA"

    def box(x, y, w, h, title, sub="", ec=c_blue, fc=c_blue_bg, lw=0.9, fs=5.7):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2
        )
        ax.add_patch(rect)
        if sub:
            ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=fs, fontweight="bold", color="#1A202C", zorder=3)
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 0.85, color="#4A5568", zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color="#1A202C", zorder=3)

    def arrow(x0, y0, x1, y1, color=c_slate, ls="-", lw=0.9):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", color=color, lw=lw, linestyle=ls),
            zorder=4
        )

    # Panel containers
    p_a = patches.FancyBboxPatch((0.8, 25.5), 98.4, 33.5, boxstyle="round,pad=0.3,rounding_size=1.0", lw=0.7, ec="#CBD5E0", fc="#FFFFFF", zorder=0)
    p_b = patches.FancyBboxPatch((0.8, 12.8), 98.4, 11.4, boxstyle="round,pad=0.3,rounding_size=1.0", lw=0.7, ec="#CBD5E0", fc="#FFFFFF", zorder=0)
    p_c = patches.FancyBboxPatch((0.8, 0.8), 98.4, 10.8, boxstyle="round,pad=0.3,rounding_size=1.0", lw=0.7, ec="#CBD5E0", fc="#FFFFFF", zorder=0)
    for pa in (p_a, p_b, p_c):
        ax.add_patch(pa)

    # Panel headers (Nature lowercase bold)
    ax.text(1.8, 57.5, "a", fontsize=8.5, fontweight="bold", color="#000000", va="top")
    ax.text(3.8, 57.5, "UniFlow latent rectified-flow architecture & terminal UNI injection (UniFlow vs UniFlow-Tok)", fontsize=7.0, fontweight="bold", color="#1A202C", va="top")
    ax.text(1.8, 23.2, "b", fontsize=8.5, fontweight="bold", color="#000000", va="top")
    ax.text(3.8, 23.2, "Source-domain NR-AE codec & rectified-flow pretraining (LOMO source organs, SWA)", fontsize=7.0, fontweight="bold", color="#1A202C", va="top")
    ax.text(1.8, 10.6, "c", fontsize=8.5, fontweight="bold", color="#000000", va="top")
    ax.text(3.8, 10.6, "Three-stage cross-organ deployment: 0-shot -> halfOD calibration -> SRA K-shot adaptation", fontsize=7.0, fontweight="bold", color="#1A202C", va="top")

    # --- Panel a: Main Architecture ---
    box(2.0, 42.5, 12.2, 7.4, "Input PI x", "256x256x3", ec=c_slate, fc=c_gray_bg)
    box(16.5, 46.8, 15.2, 6.2, "NR-AE Encoder E_psi", "4x -> 64x64x12 (z0)", ec=c_blue, fc=c_blue_bg)
    box(16.5, 38.0, 15.2, 6.2, "State z_t + Cond z_0", "Patch 4x4 (256 tok)", ec=c_blue, fc=c_blue_bg)
    box(34.2, 40.5, 20.8, 11.4, "4-Block Transformer v_theta", "Blocks B0..B3 (d=192)\nRole + Pos + Time emb", ec=c_blue, fc=c_blue_bg)
    box(57.5, 44.5, 14.2, 7.4, "PixelShuffle(4) Head", "16x16 -> 64x64x12\nBase velocity v_base", ec=c_blue, fc=c_blue_bg)
    box(74.0, 44.5, 10.0, 7.4, "Residual +", "64x64x12\nEuler ODE", ec=c_orange, fc=c_orange_bg)
    box(86.3, 44.5, 12.0, 7.4, "Decoder D_phi", "H&E y_hat\n256x256x3", ec=c_blue, fc=c_blue_bg)

    # UNI stream at bottom of panel a
    box(16.5, 27.2, 22.0, 7.6, "Frozen UNI (ViT-L/16)", "224x224 -> 14x14x1024 grid\nComputed 1x, cached", ec=c_teal, fc=c_teal_bg)
    box(41.5, 27.0, 31.5, 8.2, "UniFlow Terminal Prior Branch", "1x1 (1024->96) + SiLU + Bilinear(64)\n+ Zero-Init 1x1 (96->12) Residual", ec=c_teal, fc=c_teal_bg, lw=1.1)
    box(75.5, 27.2, 22.8, 7.6, "Ablation: UniFlow-Tok", "Added before B2/B3 (16x16)\nPeriodic grid artifact risk", ec="#718096", fc=c_gray_bg, lw=0.7)

    arrow(14.2, 46.5, 16.5, 49.6)
    arrow(24.1, 46.8, 24.1, 44.2)
    arrow(31.7, 42.5, 34.2, 45.0)
    arrow(55.0, 48.0, 57.5, 48.0)
    arrow(71.7, 48.0, 74.0, 48.0)
    arrow(84.0, 48.0, 86.3, 48.0)
    arrow(8.1, 42.5, 16.5, 31.0, color=c_teal)
    arrow(38.5, 31.0, 41.5, 31.0, color=c_teal)
    arrow(73.0, 31.4, 79.0, 44.5, color=c_teal, lw=1.1)
    arrow(38.5, 33.8, 42.5, 40.5, color="#718096", ls="--", lw=0.7)
    ax.text(42.5, 37.8, "Tok branch\n(ablated)", fontsize=4.8, color="#718096")

    # --- Panel b: Pretraining ---
    box(2.0, 14.2, 16.5, 6.0, "Source Organ Pairs", "3 organs x 480 pairs\n(strict LOMO split)", ec=c_slate, fc=c_gray_bg, fs=5.4)
    box(21.2, 14.2, 18.8, 6.0, "Stage 1: NR-AE Pretrain", "Noise in [0, 0.08*std]\nL1 + 0.15*Grad + Lat L2", ec=c_blue, fc=c_blue_bg, fs=5.4)
    box(42.7, 14.2, 25.0, 6.0, "Stage 2: Flow Pretrain", "z_t = (1-t)z_0 + t z_1\nMSE(v_theta, z_1-z_0), 9.6k", ec=c_blue, fc=c_blue_bg, fs=5.4)
    box(70.4, 14.2, 27.8, 6.0, "SWA & Source-Val Select", "240 source-val pairs\nAvg weights over [0.5T, T]", ec=c_teal, fc=c_teal_bg, fs=5.4)
    arrow(18.5, 17.1, 21.2, 17.1)
    arrow(40.0, 17.1, 42.7, 17.1)
    arrow(67.7, 17.1, 70.4, 17.1)

    # --- Panel c: Deployment & Adaptation ---
    box(2.0, 2.0, 21.5, 6.0, "Stage I: Strict 0-shot", "4-step Euler (cached UNI)\nTrainable 1.79M", ec=c_blue, fc=c_blue_bg, fs=5.4)
    box(26.2, 2.0, 25.2, 6.0, "Stage II: 0-shot + halfOD", "5 unpaired refs (alpha=0.5)\n0.5*0-shot + 0.5*fullOD", ec=c_teal, fc=c_teal_bg, fs=5.4)
    box(54.1, 2.0, 44.1, 6.0, "Stage III: SRA K-shot Adaptation (K = 5, 10, 20)", "L_vel + 4*L1 + 2*Grad + 0.5*LPIPS + 0.4*Replay\nFlow 2e-4, Codec Dec 2e-5 (1.90M, 1.5k) + SWA", ec=c_orange, fc=c_orange_bg, lw=1.0, fs=5.4)
    arrow(23.5, 5.0, 26.2, 5.0)
    arrow(51.4, 5.0, 54.1, 5.0, color=c_orange)

    ns.save_all("fig1_arch", fig, dpi=300, formats=("pdf", "png"))
    plt.close(fig)


def build_fig2_lomo_4organs(splits: dict) -> None:
    """Nature-style 4-organ LOMO qualitative comparison grid with 64x64 ROI zoom insets."""
    p = ns.panel(placed_mm=180, aspect=0.56, body_pt=7.5)
    ns.apply_style(p)
    organs = ["Breast", "Brain", "Kidney", "liver"]
    organ_labels = ["Breast\n(Held-out)", "Brain\n(Held-out)", "Kidney\n(Held-out)", "Liver\n(Held-out)"]
    cols = [
        ("PI Input", lambda o, n: DATA_DIR / o / "test" / "input" / n),
        ("BaseFlow\n(0-shot)", lambda o, n: R1_DIR / "generated" / o / f"{o}__u0_base_d192__0shot" / n),
        ("UniFlow\n(0-shot)", lambda o, n: R1_DIR / "generated" / o / f"{o}__u1_up64_only_d192__0shot" / n),
        ("BaseFlow\n(20-shot)", lambda o, n: R1_DIR / "generated" / o / f"{o}__u0_base_d192__m6_k20" / n),
        ("MixStyle\n(20-shot)", lambda o, n: DG_DIR / "generated" / o / f"{o}__mixstyle_d192__m6_k20" / n),
        ("DANN\n(20-shot)", lambda o, n: DG_DIR / "generated" / o / f"{o}__dann_d192__m6_k20" / n),
        ("UniFlow\n(20-shot)", lambda o, n: R1_DIR / "generated" / o / f"{o}__u1_up64_only_d192__m6_k20" / n),
        ("Reference\nH&E", lambda o, n: DATA_DIR / o / "test" / "target" / n),
    ]

    fig, axes = plt.subplots(len(organs), len(cols), figsize=p.figsize)
    plt.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.02, wspace=0.04, hspace=0.06)

    for r_idx, org in enumerate(organs):
        sample_name, roi = _select_tissue_and_roi(splits, org)
        for c_idx, (col_title, path_fn) in enumerate(cols):
            ax = axes[r_idx, c_idx]
            img = _load_rgb(path_fn(org, sample_name))
            img_zoom = _embed_zoom(img, roi)
            ax.imshow(img_zoom)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#CBD5E0" if c_idx != 6 else "#00A087")
                sp.set_linewidth(0.6 if c_idx != 6 else 1.2)
            if r_idx == 0:
                ax.set_title(col_title, fontsize=6.5, fontweight="bold" if c_idx == 6 else "normal",
                             color="#00A087" if c_idx == 6 else "#1A202C", pad=4)
            if c_idx == 0:
                ax.set_ylabel(f"{organ_labels[r_idx]}\n({sample_name})", fontsize=6.0, fontweight="bold", color="#1A202C", labelpad=4)

    ns.save_all("fig2_lomo_4organs", fig, dpi=300, formats=("pdf", "png"))
    plt.close(fig)


def _load_summary_map() -> dict[str, dict]:
    data = json.loads((R1_DIR / "eval_results" / "round1_summary.json").read_text(encoding="utf-8"))
    return {r["run_tag"]: r for r in data["runs"]}


def build_fig3_breast_ablation_zoom(smap: dict[str, dict]) -> None:
    """Held-out Breast progression and Checkerboard Grid Artifact zoom on file_4278.png."""
    p = ns.panel(placed_mm=180, aspect=0.33, body_pt=7.5)
    ns.apply_style(p)
    nm = "file_4278.png"
    def _lap(tag: str) -> str:
        return f"bg_lap={smap[tag]['sparse_bg_lap']:.2f}"
    gt_lap = smap["Breast__u1_up64_only_d192__m6_k20"]["gt_bg_lap"]
    cols = [
        ("PI Input", DATA_DIR / "Breast" / "test" / "input" / nm, ""),
        ("UniFlow\n0-shot", R1_DIR / "generated" / "Breast" / "Breast__u1_up64_only_d192__0shot" / nm, _lap("Breast__u1_up64_only_d192__0shot")),
        ("UniFlow\nhalfOD", R1_DIR / "generated" / "Breast" / "Breast__u1_up64_only_d192__0shot_halfOD" / nm, _lap("Breast__u1_up64_only_d192__0shot_halfOD")),
        ("UniFlow\n5-shot", R1_DIR / "generated" / "Breast" / "Breast__u1_up64_only_d192__m6_k5" / nm, _lap("Breast__u1_up64_only_d192__m6_k5")),
        ("UniFlow\n10-shot", R1_DIR / "generated" / "Breast" / "Breast__u1_up64_only_d192__m6_k10" / nm, _lap("Breast__u1_up64_only_d192__m6_k10")),
        ("NaiveFT\n20-shot", R1_DIR / "generated" / "Breast" / "Breast__u1_up64_only_d192__naive_ft_k20" / nm, _lap("Breast__u1_up64_only_d192__naive_ft_k20")),
        ("UniFlow-Tok\n20-shot", R1_DIR / "generated" / "Breast" / "Breast__u1_uni2d_1x1_d192__m6_k20" / nm, _lap("Breast__u1_uni2d_1x1_d192__m6_k20")),
        ("UniFlow\n20-shot", R1_DIR / "generated" / "Breast" / "Breast__u1_up64_only_d192__m6_k20" / nm, _lap("Breast__u1_up64_only_d192__m6_k20")),
        ("Reference\nH&E", DATA_DIR / "Breast" / "test" / "target" / nm, f"bg_lap={gt_lap:.2f}"),
    ]

    fig, axes = plt.subplots(2, len(cols), figsize=p.figsize)
    plt.subplots_adjust(left=0.04, right=0.995, top=0.84, bottom=0.06, wspace=0.04, hspace=0.08)

    r0, r1, c0, c1 = 40, 168, 0, 128  # Exact sparse_grid_laplacian crop box
    red = np.array([214, 39, 40], dtype=np.uint8)

    for ci, (title, img_p, lap_txt) in enumerate(cols):
        full = _load_rgb(img_p)
        crop = full[r0:r1, c0:c1]
        full_box = full.copy()
        full_box[r0:r0 + 2, c0:c1] = red
        full_box[r1 - 2:r1, c0:c1] = red
        full_box[r0:r1, c0:c0 + 2] = red
        full_box[r0:r1, c1 - 2:c1] = red

        axes[0, ci].imshow(full_box)
        axes[1, ci].imshow(crop, interpolation="nearest")
        for ri in (0, 1):
            ax = axes[ri, ci]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#E64B35" if ci == 6 else ("#00A087" if ci == 7 else "#CBD5E0"))
                sp.set_linewidth(1.1 if ci in (6, 7) else 0.6)
        axes[0, ci].set_title(title, fontsize=6.0, fontweight="bold" if ci in (6, 7) else "normal",
                              color="#E64B35" if ci == 6 else ("#00A087" if ci == 7 else "#1A202C"), pad=4)
        if lap_txt:
            axes[1, ci].set_xlabel(lap_txt, fontsize=5.3, color="#E64B35" if ci == 6 else ("#00A087" if ci == 7 else "#4A5568"), labelpad=2)

    axes[0, 0].set_ylabel("Full field\n(256x256)", fontsize=6.0, fontweight="bold", labelpad=3)
    axes[1, 0].set_ylabel("1:1 Zoom\n[40:168, 0:128]", fontsize=6.0, fontweight="bold", labelpad=3)
    ns.save_all("fig3_breast_ablation_zoom", fig, dpi=300, formats=("pdf", "png"))
    plt.close(fig)


def build_fig4_quantitative_charts(smap: dict[str, dict]) -> None:
    """Nature-style 4-panel quantitative summary figure strictly built with nature_style."""
    p = ns.panel(placed_mm=180, aspect=0.42, body_pt=7.5)
    ns.apply_style(p)
    fig, axes = plt.subplots(1, 4, figsize=p.figsize)
    plt.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.19, wspace=0.34)

    c_base = "#8491B4"
    c_ours = "#00A087"
    c_red = "#E64B35"
    c_blue = "#3C5488"
    c_cyan = "#4DBBD5"

    # Data verified directly from round1_summary.json and dg_summary.md
    organs = ["Breast", "Brain", "Kidney", "Liver"]
    fold_keys = ["Breast", "Brain", "Kidney", "liver"]
    base_psnr_0 = [round(smap[f"{f}__u0_base_d192__0shot"]["metrics"]["psnr"], 2) for f in fold_keys]
    base_psnr_h = [round(smap[f"{f}__u0_base_d192__0shot_halfOD"]["metrics"]["psnr"], 2) for f in fold_keys]
    base_psnr_20 = [round(smap[f"{f}__u0_base_d192__m6_k20"]["metrics"]["psnr"], 2) for f in fold_keys]
    ours_psnr_0 = [round(smap[f"{f}__u1_up64_only_d192__0shot"]["metrics"]["psnr"], 2) for f in fold_keys]
    ours_psnr_h = [round(smap[f"{f}__u1_up64_only_d192__0shot_halfOD"]["metrics"]["psnr"], 2) for f in fold_keys]
    ours_psnr_20 = [round(smap[f"{f}__u1_up64_only_d192__m6_k20"]["metrics"]["psnr"], 2) for f in fold_keys]

    # Panel a: 4-organ 3-stage PSNR
    ax = axes[0]
    x = np.arange(len(organs))
    w = 0.25
    ax.bar(x - w / 2, base_psnr_0, w, color="#CBD5E0", label="BaseFlow 0-shot")
    ax.bar(x + w / 2, ours_psnr_0, w, color="#91D1C2", label="UniFlow 0-shot")
    ax.plot(x, base_psnr_20, "o--", color=c_blue, ms=3.5, lw=1.0, label="BaseFlow 20-shot")
    ax.plot(x, ours_psnr_20, "s-", color=c_ours, ms=3.8, lw=1.2, label="UniFlow 20-shot")
    ax.set_xticks(x)
    ax.set_xticklabels(organs, fontsize=6.0)
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(10, 27)
    ax.set_title("Stage gain (0-shot -> 20-shot)", fontsize=6.5, pad=4)
    ax.legend(fontsize=4.8, frameon=False, loc="lower right", ncol=2, handlelength=1.2, columnspacing=0.6)
    ns.clean_axes(ax)
    ns.panel_letter(ax, "a", x=-0.18, y=1.08)

    # Panel b: 4-organ 20-shot KID (lower better) comparing Base, MixStyle, DANN, UniFlow
    ax = axes[1]
    mix_kid = [10.60, 25.72, 49.01, 18.12]
    dann_kid = [22.36, 82.30, 52.77, 29.06]
    base_kid = [round(smap[f"{f}__u0_base_d192__m6_k20"]["metrics"]["kid"] * 1000.0, 2) for f in fold_keys]
    ours_kid = [round(smap[f"{f}__u1_up64_only_d192__m6_k20"]["metrics"]["kid"] * 1000.0, 2) for f in fold_keys]
    w2 = 0.19
    ax.bar(x - 1.5 * w2, dann_kid, w2, color=c_red, alpha=0.85, label="DANN")
    ax.bar(x - 0.5 * w2, mix_kid, w2, color=c_cyan, alpha=0.9, label="MixStyle")
    ax.bar(x + 0.5 * w2, base_kid, w2, color=c_blue, alpha=0.85, label="BaseFlow")
    ax.bar(x + 1.5 * w2, ours_kid, w2, color=c_ours, label="UniFlow")
    ax.set_xticks(x)
    ax.set_xticklabels(organs, fontsize=6.0)
    ax.set_ylabel("KID (x10^-3, lower better)")
    ax.set_ylim(0, 102)
    ax.set_title("20-shot distribution distance", fontsize=6.5, pad=4)
    ax.legend(fontsize=4.8, frameon=False, loc="upper right", ncol=2, handlelength=1.2)
    ns.clean_axes(ax)
    ns.panel_letter(ax, "b", x=-0.18, y=1.08)

    # Panel c: Breast K-shot sample efficiency (0, 5, 10, 20)
    ax = axes[2]
    shots = [0, 5, 10, 20]
    br_h = round(smap["Breast__u1_up64_only_d192__0shot_halfOD"]["metrics"]["psnr"], 2)
    br_nv = round(smap["Breast__u1_up64_only_d192__naive_ft_k20"]["metrics"]["psnr"], 2)
    br_b20 = round(smap["Breast__u0_base_d192__m6_k20"]["metrics"]["psnr"], 2)
    k_psnr = [
        round(smap["Breast__u1_up64_only_d192__0shot"]["metrics"]["psnr"], 2),
        round(smap["Breast__u1_up64_only_d192__m6_k5"]["metrics"]["psnr"], 2),
        round(smap["Breast__u1_up64_only_d192__m6_k10"]["metrics"]["psnr"], 2),
        round(smap["Breast__u1_up64_only_d192__m6_k20"]["metrics"]["psnr"], 2),
    ]
    l1 = ax.plot(shots, k_psnr, "s-", color=c_ours, lw=1.3, ms=4.0, label="UniFlow PSNR")
    ax.scatter([0], [br_h], marker="^", s=26, color=c_cyan, zorder=5, label=f"halfOD ({br_h:.2f})")
    ax.scatter([20], [br_nv], marker="D", s=22, color=c_blue, zorder=5, label=f"NaiveFT ({br_nv:.2f})")
    ax.axhline(br_b20, ls="--", lw=0.8, color=c_base, label=f"BaseFlow 20-shot ({br_b20:.2f})")
    ax.set_xticks(shots)
    ax.set_xlabel("Target adaptation shots (K)")
    ax.set_ylabel("Breast PSNR (dB)")
    ax.set_ylim(14.0, 25.5)
    ax.set_title("Breast sample efficiency", fontsize=6.5, pad=4)
    ax.legend(fontsize=4.6, frameon=False, loc="lower right", handlelength=1.2)
    ns.clean_axes(ax)
    ns.panel_letter(ax, "c", x=-0.18, y=1.08)

    # Panel d: Breast 20-shot PSNR vs sparse background Laplacian (bg_lap)
    ax = axes[3]
    br_tok = smap["Breast__u1_uni2d_1x1_d192__m6_k20"]
    br_nv_r = smap["Breast__u1_up64_only_d192__naive_ft_k20"]
    br_our_r = smap["Breast__u1_up64_only_d192__m6_k20"]
    br_b_r = smap["Breast__u0_base_d192__m6_k20"]
    pts = [
        ("BaseFlow", round(br_b_r["sparse_bg_lap"], 2), round(br_b_r["metrics"]["psnr"], 2), c_blue, "o"),
        ("MixStyle", 1.74, 22.90, c_cyan, "o"),
        ("StainAug", 1.71, 23.13, "#4DBBD5", "^"),
        ("MLDG", 1.75, 23.11, "#8491B4", "v"),
        ("DANN", 1.75, 22.43, c_red, "x"),
        ("NaiveFT", round(br_nv_r["sparse_bg_lap"], 2), round(br_nv_r["metrics"]["psnr"], 2), "#F39B7F", "D"),
        ("UniFlow-Tok", round(br_tok["sparse_bg_lap"], 2), round(br_tok["metrics"]["psnr"], 2), c_red, "P"),
        ("UniFlow", round(br_our_r["sparse_bg_lap"], 2), round(br_our_r["metrics"]["psnr"], 2), c_ours, "s"),
    ]
    for lbl, lx, py, col, mk in pts:
        ax.scatter([lx], [py], color=col, marker=mk, s=28 if lbl != "UniFlow" else 36, zorder=5)
        dx, dy = 0.03, 0.04
        if lbl == "BaseFlow":
            dx, dy = -0.05, -0.15
        elif lbl == "StainAug":
            dx, dy = 0.04, 0.07
        elif lbl == "MixStyle":
            dy = -0.15
        elif lbl == "MLDG":
            dx, dy = 0.04, -0.04
        elif lbl == "UniFlow-Tok":
            dx, dy = -0.60, -0.15
        elif lbl == "UniFlow":
            dx, dy = 0.04, 0.06
        ax.text(lx + dx, py + dy, lbl, fontsize=5.1, fontweight="bold" if lbl in ("UniFlow", "UniFlow-Tok") else "normal", color=col)
    ax.axvline(2.86, ls=":", lw=0.9, color="#4A5568")
    ax.text(2.83, 22.25, "GT\n2.86", fontsize=4.8, ha="right", color="#4A5568")
    ax.set_xlabel("Sparse bg_lap (grid artifact)")
    ax.set_ylabel("Breast 20-shot PSNR (dB)")
    ax.set_xlim(1.50, 2.95)
    ax.set_ylim(22.15, 24.45)
    ax.set_title("Fidelity vs grid artifact", fontsize=6.5, pad=4)
    ns.clean_axes(ax, grid_axis="both")
    ns.panel_letter(ax, "d", x=-0.18, y=1.08)

    ns.save_all("fig4_quantitative_charts", fig, dpi=300, formats=("pdf", "svg", "png"))
    plt.close(fig)


def main() -> None:
    splits = json.loads((R1_DIR / "eval_results" / "round1_splits.json").read_text(encoding="utf-8"))
    smap = _load_summary_map()
    build_fig1_arch()
    build_fig2_lomo_4organs(splits)
    build_fig3_breast_ablation_zoom(smap)
    build_fig4_quantitative_charts(smap)
    print("[FIGURES_OK] Generated all 4 figures in:", OUT_DIR)


if __name__ == "__main__":
    main()
