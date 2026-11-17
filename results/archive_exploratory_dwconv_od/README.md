# Exploratory Archive: Terminal `3x3 DWConv` & Explicit `L_OD` (`OASRA`)

This directory preserves the pre-simplification exploratory results from our ablation study prior to applying Occam's Razor pruning:

1. **Terminal `3x3` Depthwise Separable Conv (`dwconv`)**: Previously applied after bilinear interpolation at `64x64` inside `u1_up64_only`. Removing it simplified the terminal branch to `1x1 (1024->96) + SiLU -> bilinear(64) -> zero-init 1x1 (96->12)` while matching or slightly improving 20-shot quality (`Breast` KID `8.90 -> 8.79`, Dice `0.5575 -> 0.5609`; `Brain` Dice `0.7511 -> 0.7565`) and lowering 227-patch 2D FFT periodic grid energy to `4.86%`.
2. **Explicit Beer–Lambert OD Anchor Loss (`L_OD`, `OASRA`)**: Previously added with weight `0.25` during Stage III adaptation (`m6`). Controlled ablation (`breast_extra_ablations.json` inside `round1/`) showed removing `L_OD` had negligible or slightly favorable effect (`Breast` KID `8.90 -> 8.78`, `Source-Val` PSNR `19.67 -> 19.76`, Dice `0.7231 -> 0.7234`), so Stage III was simplified to **SRA** (*Stain-Augmented Source-Replay Adaptation*, default `--od-weight 0.0`).

Readers interested in reproducing the exploratory `L_OD` setting can pass `--od-weight 0.25` to `scripts/run_round1_pipeline.py`. All canonical manuscript results and figures are derived from `results/round1/`.
