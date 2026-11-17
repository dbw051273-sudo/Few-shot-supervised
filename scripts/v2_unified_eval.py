"""Unified v2 evaluation module supporting uint16 Cellpose instance F1@0.5,
all 9+ research metrics saved to JSON + per_image CSV, and 6-metric paper core summary.
Aligned with official_eval transforms (ToPILImage -> Resize(299) -> ToTensor, normalize=True).
"""
from __future__ import annotations

import csv
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class OfficialScalarMetrics:
    """Self-contained exact scalar metrics matching official_eval."""

    @staticmethod
    def psnr(gt_image: np.ndarray, syn_image: np.ndarray) -> float:
        gt = gt_image.astype(np.float32) / 255.0 if gt_image.dtype == np.uint8 else gt_image
        syn = syn_image.astype(np.float32) / 255.0 if syn_image.dtype == np.uint8 else syn_image
        mse_val = np.mean((gt - syn) ** 2)
        return 100.0 if mse_val == 0 else float(20 * math.log10(1.0 / math.sqrt(mse_val)))

    @staticmethod
    def calculate_generic_pcc(gen_img: np.ndarray, src_img: np.ndarray) -> float:
        g_gen = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
        g_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
        if np.var(g_gen) < 1e-5 or np.var(g_src) < 1e-5:
            return 0.0
        corr, _ = pearsonr(g_gen, g_src)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def compute_ssim_official(tar_chw: np.ndarray, gen_chw: np.ndarray) -> float:
        return float(ssim(
            tar_chw, gen_chw, channel_axis=0, data_range=1.0,
            win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        ))

    @staticmethod
    def compute_dice_from_bins(tar_bin: np.ndarray, gen_bin: np.ndarray) -> float:
        intersection = np.sum(tar_bin * gen_bin)
        union = np.sum(tar_bin) + np.sum(gen_bin)
        return 1.0 if union == 0 else float(2.0 * intersection / (union + 1e-6))


PAPER_CORE_KEYS = ('psnr', 'ssim', 'lpips', 'kid', 'nuclei_dice', 'nuclei_f1_05')
KID_SUBSET_SIZE = 50


class _PerImageLPIPSWrapper(torch.nn.Module):
    """Compatibility wrapper when older torchmetrics disallows reduction='none'."""

    def __init__(self, net_type: str = 'alex', normalize: bool = True):
        super().__init__()
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=normalize, reduction='mean')
        self.normalize = normalize

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            img1 = img1 * 2.0 - 1.0
            img2 = img2 * 2.0 - 1.0
        return self.metric.net(img1, img2).reshape(-1)


def build_per_image_lpips(net_type: str = 'alex', normalize: bool = True) -> torch.nn.Module:
    """Create per-image LPIPS module compatible with both old and new torchmetrics."""
    try:
        mod = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=normalize, reduction='none')
    except ValueError:
        mod = _PerImageLPIPSWrapper(net_type=net_type, normalize=normalize)
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod.eval()


def load_gt_instance_masks(npz_path: Path, organ: str) -> dict[str, np.ndarray]:
    npz = np.load(npz_path)
    prefix = f'{organ.lower()}/'
    out = {}
    for k, m in zip(npz['keys'], npz['masks']):
        ks = str(k).lower()
        if ks.startswith(prefix):
            out[ks[len(prefix):]] = m
    return out


def instance_f1_at_05(gt_mask: np.ndarray, pr_mask: np.ndarray) -> tuple[float, int, int, int]:
    """Strict 1-to-1 Hungarian instance matching at IoU >= 0.5."""
    gt_n = int(gt_mask.max())
    pr_n = int(pr_mask.max())
    if gt_n == 0 and pr_n == 0:
        return 1.0, 0, 0, 0
    if gt_n == 0 or pr_n == 0:
        return 0.0, 0, gt_n, pr_n
    g_max = gt_n + 1
    p_max = pr_n + 1
    comb = gt_mask.astype(np.int64) * p_max + pr_mask.astype(np.int64)
    hist = np.bincount(comb.ravel(), minlength=g_max * p_max).reshape(g_max, p_max)
    g_area = hist.sum(axis=1)[1:]
    p_area = hist.sum(axis=0)[1:]
    inter = hist[1:, 1:]
    union = g_area[:, None] + p_area[None, :] - inter
    iou = inter / np.maximum(union, 1)
    valid_g = g_area > 0
    valid_p = p_area > 0
    if not valid_g.any() and not valid_p.any():
        return 1.0, 0, 0, 0
    if not valid_g.any() or not valid_p.any():
        return 0.0, 0, int(valid_g.sum()), int(valid_p.sum())
    iou_sub = iou[np.ix_(valid_g, valid_p)]
    eligible = iou_sub >= 0.5
    if not eligible.any():
        tp = 0
    else:
        cost = np.zeros_like(iou_sub, dtype=np.float64)
        cost[eligible] = -(1.0 + iou_sub[eligible])
        r_idx, c_idx = linear_sum_assignment(cost)
        tp = int((iou_sub[r_idx, c_idx] >= 0.5).sum())
    ng, np_cnt = int(valid_g.sum()), int(valid_p.sum())
    return float(2.0 * tp / (ng + np_cnt)), tp, ng, np_cnt


def sparse_grid_laplacian(img_rgb: np.ndarray, col_end: int = 112) -> float:
    """Measure local high-frequency patch-grid energy in sparse margin/image."""
    crop = img_rgb[:, :col_end].astype(np.float32) if img_rgb.shape[1] >= col_end else img_rgb.astype(np.float32)
    gray = 0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
    lap = np.abs(gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1])
    return float(lap.mean())


@torch.no_grad()
def evaluate_split(
    names: list[str],
    pred_rgb_map: dict[str, np.ndarray],
    pi_rgb_map: dict[str, np.ndarray],
    gt_rgb_map: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    cell_model,
    lpips_fn: LearnedPerceptualImagePatchSimilarity,
    device: torch.device,
    per_image_csv: Path | None = None,
    precomputed_gen_masks: dict[str, np.ndarray] | None = None,
    out_gen_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    preprocess_fid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    preprocess_lpips = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    tar_rgb = [gt_rgb_map[n] for n in names]
    gen_rgb = [pred_rgb_map[n] for n in names]
    src_rgb = [pi_rgb_map[n] for n in names]

    tar_bgr = [im[:, :, ::-1].copy() for im in tar_rgb]
    gen_bgr = [im[:, :, ::-1].copy() for im in gen_rgb]
    src_bgr = [im[:, :, ::-1].copy() for im in src_rgb]
    tar_chw = [im.transpose(2, 0, 1).astype(np.float32) / 255.0 for im in tar_rgb]
    gen_chw = [im.transpose(2, 0, 1).astype(np.float32) / 255.0 for im in gen_rgb]

    use_fid_kid = len(names) >= 2
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device) if use_fid_kid else None
    kid = KernelInceptionDistance(subset_size=min(KID_SUBSET_SIZE, len(names)), normalize=True).to(device) if use_fid_kid else None
    lp_vals, gen_masks = [], []

    for i in range(0, len(names), 32):
        bn = names[i:i + 32]
        bt = tar_rgb[i:i + 32]
        bg = gen_rgb[i:i + 32]
        if use_fid_kid:
            ft = torch.stack([preprocess_fid(im) for im in bt]).to(device)
            fg = torch.stack([preprocess_fid(im) for im in bg]).to(device)
            fid.update(ft, real=True)
            fid.update(fg, real=False)
            kid.update(ft, real=True)
            kid.update(fg, real=False)
        lt = torch.stack([preprocess_lpips(im) for im in bt]).to(device)
        lg = torch.stack([preprocess_lpips(im) for im in bg]).to(device)
        lp_vals.extend(lpips_fn(lg, lt).view(-1).cpu().tolist())
        if precomputed_gen_masks is not None:
            gen_masks.extend(precomputed_gen_masks[nm] for nm in bn)
        elif cell_model is None:
            gen_masks.extend(np.zeros((256, 256), dtype=np.uint16) for _ in bn)
        else:
            mb, _, _ = cell_model.eval(bg, diameter=None, channels=[0, 0], batch_size=len(bg))
            gen_masks.extend(m.astype(np.uint16) for m in mb)

    if out_gen_masks is not None:
        for nm, gm in zip(names, gen_masks):
            out_gen_masks[nm] = gm

    def _one(idx: int):
        nm = names[idx]
        gm = gt_masks[nm.lower()]
        pm = gen_masks[idx]
        tb, gb, sb = tar_bgr[idx], gen_bgr[idx], src_bgr[idx]
        f1, _tp, gt_cnt, pr_cnt = instance_f1_at_05(gm, pm)
        dice = OfficialScalarMetrics.compute_dice_from_bins((gm > 0).astype(np.float32), (pm > 0).astype(np.float32))
        return (
            OfficialScalarMetrics.psnr(tb, gb),
            OfficialScalarMetrics.compute_ssim_official(tar_chw[idx], gen_chw[idx]),
            lp_vals[idx],
            OfficialScalarMetrics.calculate_generic_pcc(gb, tb),
            OfficialScalarMetrics.calculate_generic_pcc(gb, sb),
            dice,
            f1,
            gt_cnt,
            pr_cnt,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(_one, range(len(names))))

    if per_image_csv is not None:
        per_image_csv.parent.mkdir(parents=True, exist_ok=True)
        with per_image_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['name', 'psnr', 'ssim', 'lpips', 'pcc_gt', 'pcc_pi', 'nuclei_dice', 'nuclei_f1_05', 'gt_nuclei', 'pr_nuclei'])
            for nm, row in zip(names, rows):
                w.writerow([nm, *(round(v, 6) if isinstance(v, float) else v for v in row)])

    gt_counts = np.array([r[7] for r in rows], dtype=np.float64)
    pr_counts = np.array([r[8] for r in rows], dtype=np.float64)
    cnt_r = float(np.corrcoef(gt_counts, pr_counts)[0, 1]) if (gt_counts.std() > 0 and pr_counts.std() > 0) else 0.0
    nonempty = [r for r in rows if r[7] > 0]

    torch.manual_seed(7)
    return {
        'psnr': float(np.mean([r[0] for r in rows])),
        'ssim': float(np.mean([r[1] for r in rows])),
        'lpips': float(np.mean([r[2] for r in rows])),
        'fid': float(fid.compute().item()) if use_fid_kid else 0.0,
        'kid': float(kid.compute()[0].item()) if use_fid_kid else 0.0,
        'nuclei_dice': float(np.mean([r[5] for r in rows])),
        'nuclei_f1_05': float(np.mean([r[6] for r in rows])),
        'nuclei_dice_nonempty': float(np.mean([r[5] for r in nonempty])) if nonempty else 0.0,
        'nuclei_f1_05_nonempty': float(np.mean([r[6] for r in nonempty])) if nonempty else 0.0,
        'empty_gt_count': int(len(rows) - len(nonempty)),
        'pcc_gt': float(np.mean([r[3] for r in rows])),
        'pcc_pi': float(np.mean([r[4] for r in rows])),
        'nuclei_count_r': cnt_r,
    }
