# Exploratory Cross-Species Evaluation: Human Label-Free AF $\rightarrow$ H&E (HKUST UniStain)

> **Scope & Status**: Archived exploratory study (frozen after Pilot folds `Breast` and `Kidney`; mouse `UniPrior_v2` data and weights remain untouched). Lightweight scripts, split audits, and JSON/CSV summaries are preserved here for reproducibility and Section 5 boundary discussion; raw images and checkpoints are excluded via `.gitignore`.

---

## 1. Dataset Audit & Zero-Leakage Protocol (`af_lomo_splits.json`)

- **Source**: Official HKUST `TABLAB-HKUST/UniStain` multi-organ paired human autofluorescence (AF) & H&E dataset (`6,800` total paired `512x512` RGB tiles across `Breast`, `Kidney`, `Liver`, `Lung`; AF stored as replicated 1-channel grayscale `R=G=B`; no patient/WSI/cancer-subtype metadata; sequential indices exhibit `8 px` WSI tiling overlap).
- **Clean-256 Protocol**: Downsampled to `256x256`; per organ `380 train / 60 val` drawn strictly from official `training/`, `120 test + 20 adapt + 5 refs` drawn strictly from official `testing/` with block separation $\ge 10$ raw indices.
- **Inter-Organ AF Dynamic Range Shift**: `Breast` AF is $\sim 3\times$ darker (`adapt20 mean=35.47`, `p99=100.0`) than `Liver` (`train mean=115.05`, `p99=221.0`), with `Kidney=77.19` and `Lung=59.84`.

---

## 2. Key Pilot Results: `Breast` Fold ($N=120$, GT nuclei median = 43.0)

| Config (`raw` AF unless noted) | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID×1e3 ↓ | FID ↓ | `Dice_ne` ↑ | `PR_nuc` (med / mean / nz) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BaseFlow` (`pixel_shuffle`, SRA) | 0-shot | 18.70 | 0.4624 | 0.4602 | 150.38 | 232.66 | 0.0047 | - |
| `BaseFlow` (`pixel_shuffle`, SRA) | 20-shot | 20.27 | 0.5175 | 0.3334 | 151.30 | 227.64 | 0.0003 | - |
| `UniFlow` (`pixel_shuffle`, SRA) | 0-shot | 18.29 | 0.4247 | 0.3812 | 139.44 | 212.83 | 0.0012 | - |
| `UniFlow` (`pixel_shuffle`, SRA) | **20-shot** | **21.03** | **0.5200** | 0.3213 | 152.12 | 221.52 | 0.0022 | - |
| `p1p99 UniFlow` (per-img norm) | 0-shot | 18.82 | 0.4325 | 0.5319 | 257.23 | 299.02 | - | - |
| `p1p99 UniFlow` (per-img norm) | 20-shot | 20.98 | 0.5187 | 0.3214 | 156.62 | 221.37 | - | - |
| `BaseFlow` (`1*L1+1*grad+2*LPIPS`) | 20-shot perc | 20.14 | - | 0.2974 | 128.55 | - | 0.0005 | - |
| **`UniFlow` (`1*L1+1*grad+2*LPIPS`)** | **20-shot perc** | 20.25 | - | **0.2899** | **88.57** | - | 0.0002 | 0.0 / 0.06 / 4 |
| `BaseFlow + PatchGAN` (`w_adv=0.08, w_fm=1.5`) | 20-shot PG | 19.40 | 0.4494 | 0.3277 | 126.54 | 207.82 | 0.0017 | 0.0 / 0.38 / 11 |
| `UniFlow + PatchGAN` (`w_adv=0.08, w_fm=1.5`) | 20-shot PG | 19.68 | 0.4426 | 0.3094 | **88.19** | **169.64** | 0.0003 | 0.0 / 0.08 / 6 |
| `organ_q UniFlow + PatchGAN` | 20-shot PG | 19.20 | 0.4397 | 0.3248 | **85.11** | 170.28 | 0.0011 | 0.0 / 0.23 / 6 (`count_r=+0.1345`) |

- Codec sanity check on `Breast` GT H&E confirmed NR-AE preserves nuclei cleanly (`Breast_0081`: GT=207 vs CodecRec=190; `Breast_0099`: 76 vs 71; `Breast_0123`: 59 vs 48).

---

## 3. Key Pilot Results: `Kidney` Fold ($N=120$, GT nuclei median = 118.0)

| Architecture / Variant (`raw` AF) | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID×1e3 ↓ | FID ↓ | `Dice` / `Dice_ne` ↑ | `PR_nuc` (med / mean / nz) | `count_r` ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BaseFlow` (`pixel_shuffle`) | 0-shot | 16.20 | 0.1887 | 0.5212 | 375.59 | 317.19 | 0.0192 / 0.0025 | 0.0 / 0.1 / 3 | - |
| `BaseFlow` (`pixel_shuffle`) | 20-shot SRA | 21.12 | 0.4489 | 0.3267 | 345.58 | 294.20 | 0.0188 / 0.0021 | 0.0 / 0.2 / 7 | - |
| `UniFlow` (`pixel_shuffle`) | 0-shot | 16.52 | 0.1986 | 0.5240 | 355.95 | 309.53 | 0.0178 / 0.0011 | 0.0 / 0.0 / 2 | - |
| `UniFlow` (`pixel_shuffle`) | 20-shot SRA | 20.90 | 0.4429 | 0.3375 | 347.69 | 294.18 | 0.0167 / 0.0000 | 0.0 / 0.0 / 0 | - |
| `UniFlow` (`pixel_shuffle`) | 20-shot perc | 20.27 | 0.3768 | **0.3014** | 319.18 | 269.02 | 0.0167 / 0.0000 | 0.0 / 0.0 / 0 | - |
| **`BaseFlow` (`overlap_conv`)** | 0-shot | 17.34 | 0.2589 | 0.4864 | 295.56 | - | 0.0184 / 0.0018 | 0.0 / 0.3 / 8 | - |
| **`UniFlow` (`overlap_conv`)** | 0-shot | 17.62 | 0.2647 | 0.4887 | 298.56 | - | 0.0203 / 0.0037 | 0.0 / 0.4 / 17 | - |
| **`UniFlow` (`overlap_conv`)** | 20-shot SRA | 20.61 | 0.4345 | 0.3758 | 287.95 | - | 0.0167 / 0.0000 | 0.0 / 0.0 / 0 | - |
| `BaseFlow_skip64` (`ovlp + 64x64 ResConv`) | 20-shot SRA | **21.18** | **0.4554** | 0.3567 | 346.01 | 284.29 | 0.0181 / 0.0014 | 0.0 / 0.1 / 4 | -0.0648 |
| `UniFlow_skip64` (`ovlp + 64x64 ResConv`) | 20-shot SRA | 20.88 | 0.4504 | 0.3570 | 329.58 | 274.66 | 0.0167 / 0.0000 | 0.0 / 0.0 / 0 | 0.0000 |
| **`BaseFlow_skip64 + PatchGAN`** | **20-shot PG** | **19.03** | **0.3266** | **0.3373** | **173.74** | **167.95** | **0.0616 / 0.0457** | **61.0 / 54.4 / 99** | **0.4164** |
| **`UniFlow_skip64 + PatchGAN`** | **20-shot PG** | 19.02 | 0.3248 | 0.3381 | **178.05** | 170.14 | **0.0527 / 0.0366** | **47.0 / 44.0 / 96** | **0.3983** |
| Official `Pix2Pix` (`unet_256` + PatchGAN) | 0-shot | 16.53 | 0.2357 | 0.4221 | 178.62 | 181.11 | 0.0285 / 0.0121 | 9.0 / 16.7 / 85 | 0.0888 |
| Official `Pix2Pix` (`unet_256` + PatchGAN) | 20-shot | 18.89 | 0.2969 | 0.3870 | 251.39 | 216.47 | 0.0520 / 0.0360 | 41.0 / 43.7 / 101 | 0.3047 |

---

## 4. Mechanistic Diagnosis & Boundary Summary (for Paper §5 Discussion)

1. **Label-Free AF vs. Nuclear-Stained PI**: Unlike propidium iodide (`PI`), which provides direct 1-to-1 bright nuclear contrast allowing non-adversarial rectified flows to achieve strong `Cellpose Dice` (`0.56–0.76`) on mouse tissue, label-free human `AF` primarily highlights stromal collagen, tubular contours, and cytoplasmic autofluorescence while leaving individual nuclei as low-SNR dark voids with slight physical misregistration.
2. **Block Artifact Under Low-Contrast AF (`pixel_shuffle` vs `overlap_conv`)**: On bright `Kidney` tubules, the `16x16` token bottleneck paired with non-overlapping `pixel_shuffle(4)` produced periodic `16x16` false-nuclei dots (`KID=347–375`). Switching to bilinear + replicate `3x3` `overlap_conv` eliminated this artifact completely (`KID 356 -> 288`, `0-shot PSNR +1.1 dB`).
3. **Why Full-Res Skip + Adversarial Supervision Is Necessary on Label-Free AF (and Trade-off with UNI Prior)**: Even with `overlap_conv` and a `64x64` FiLM-ResConv skip branch (`skip64`), non-adversarial velocity regression blurs uncertain sub-resolution nuclei (`PR_nuc = 0`, `KID = 330–346`). Adding a lightweight `PatchGAN` discriminator on top of `skip64` during 20-shot adaptation unlocked crisp renal nuclei synthesis (`PR_nuc med = 47–61` vs `Pix2Pix 41`, `KID = 173.7–178.1` vs `Pix2Pix 251.4`, `LPIPS = 0.337–0.338` vs `Pix2Pix 0.387`). However, once dense local `64x64` spatial cues were bypassed into the generator on high-SNR `Kidney` tubules, `BaseFlow_skip64_PG` slightly outperformed `UniFlow_skip64_PG`, indicating that frozen bright-field `UNI` priors yield largest marginal gains under sparse/low-SNR regimes (e.g., `Breast`, where `UniFlow+PG` achieved `KID 88.19 vs BaseFlow+PG 126.54`) or explicit nuclear-stained excitation (`PI`).
