# Round-1 Serialized Evaluation Summary

- Updated: `2026-11-17 00:15:42`
- Core Metrics: `PSNR ↑`, `SSIM ↑`, `LPIPS ↓`, `KID (×10⁻³) ↓`, `Cellpose-proxy Dice ↑`, `Cellpose-proxy F1@0.5 (1:1) ↑`
- Note: `KID` values are scaled by `10^3` (`subset_size=50`). `FID` depends on test size `N` and is omitted from main/cross-`N` tables (retained in JSON/CSV only for same-`N` supplementary diagnostics).

| Run Tag | Fold | Arch | Dim | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID (×10⁻³) ↓ | Dice ↑ | F1@0.5 ↑ | bg_lap | gt_lap | Params(M) | ms/img |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Brain__u0_base_d192__0shot` | Brain | u0_base | 192 | 0shot | 12.76 | 0.6523 | 0.3874 | 92.39 | 0.7079 | 0.6749 | 4.26 | 7.28 | 1.69 | 12.18 |
| `Brain__u0_base_d192__0shot_halfOD` | Brain | u0_base | 192 | 0shot_halfOD | 12.96 | 0.6536 | 0.3801 | 92.04 | 0.7085 | 0.6801 | 4.43 | 7.28 | 1.69 | 12.18 |
| `Brain__u0_base_d192__m6_k20` | Brain | u0_base | 192 | 20shot | 23.77 | 0.7585 | 0.1265 | 23.99 | 0.7455 | 0.7160 | 4.29 | 7.28 | 1.80 | 12.89 |
| `Brain__u1_up64_only_d192__0shot` | Brain | u1_up64_only | 192 | 0shot | 13.38 | 0.6521 | 0.3539 | 102.11 | 0.6989 | 0.6549 | 5.64 | 7.28 | 1.79 | 25.16 |
| `Brain__u1_up64_only_d192__0shot_halfOD` | Brain | u1_up64_only | 192 | 0shot_halfOD | 14.20 | 0.6594 | 0.3357 | 96.93 | 0.6997 | 0.6608 | 5.88 | 7.28 | 1.79 | 25.16 |
| `Brain__u1_up64_only_d192__m6_k20` | Brain | u1_up64_only | 192 | 20shot | 24.43 | 0.7619 | 0.1148 | 21.09 | 0.7511 | 0.7233 | 4.36 | 7.28 | 1.90 | 26.26 |
| `Breast__u0_base_d192__0shot` | Breast | u0_base | 192 | 0shot | 14.89 | 0.7194 | 0.2887 | 43.58 | 0.5045 | 0.5084 | 3.78 | 2.86 | 1.69 | 12.00 |
| `Breast__u0_base_d192__0shot_halfOD` | Breast | u0_base | 192 | 0shot_halfOD | 19.56 | 0.7815 | 0.2393 | 26.75 | 0.5042 | 0.5080 | 3.80 | 2.86 | 1.69 | 12.00 |
| `Breast__u0_base_d192__m6_k20` | Breast | u0_base | 192 | 20shot | 23.04 | 0.8142 | 0.1732 | 10.62 | 0.5471 | 0.5503 | 1.64 | 2.86 | 1.80 | 14.29 |
| `Breast__u1_uni2d_1x1_d192__0shot` | Breast | u1_uni2d_1x1 | 192 | 0shot | 15.90 | 0.7280 | 0.2387 | 34.34 | 0.5177 | 0.5231 | 4.88 | 2.86 | 1.96 | 26.68 |
| `Breast__u1_uni2d_1x1_d192__0shot_halfOD` | Breast | u1_uni2d_1x1 | 192 | 0shot_halfOD | 20.57 | 0.7848 | 0.2018 | 22.11 | 0.5299 | 0.5365 | 4.32 | 2.86 | 1.96 | 26.68 |
| `Breast__u1_uni2d_1x1_d192__m6_k20` | Breast | u1_uni2d_1x1 | 192 | 20shot | 24.11 | 0.8249 | 0.1520 | 8.89 | 0.5657 | 0.5700 | 2.70 | 2.86 | 2.07 | 24.92 |
| `Breast__u1_up64_only_d192__0shot` | Breast | u1_up64_only | 192 | 0shot | 15.04 | 0.7272 | 0.2633 | 41.25 | 0.5028 | 0.5082 | 4.62 | 2.86 | 1.79 | 29.14 |
| `Breast__u1_up64_only_d192__0shot_halfOD` | Breast | u1_up64_only | 192 | 0shot_halfOD | 19.83 | 0.7877 | 0.2156 | 27.77 | 0.5027 | 0.5081 | 4.33 | 2.86 | 1.79 | 29.14 |
| `Breast__u1_up64_only_d192__m6_k10` | Breast | u1_up64_only | 192 | 10shot | 22.95 | 0.8125 | 0.1681 | 13.00 | 0.5526 | 0.5574 | 2.06 | 2.86 | 1.90 | 28.90 |
| `Breast__u1_up64_only_d192__m6_k20` | Breast | u1_up64_only | 192 | 20shot | 23.87 | 0.8244 | 0.1562 | 8.90 | 0.5575 | 0.5594 | 1.80 | 2.86 | 1.90 | 25.63 |
| `Breast__u1_up64_only_d192__m6_k5` | Breast | u1_up64_only | 192 | 5shot | 21.34 | 0.7916 | 0.1918 | 26.26 | 0.5523 | 0.5592 | 2.19 | 2.86 | 1.90 | 30.21 |
| `Breast__u1_up64_only_d192__naive_ft_k20` | Breast | u1_up64_only | 192 | 20shot | 23.62 | 0.8170 | 0.1645 | 11.03 | 0.5571 | 0.5612 | 2.00 | 2.86 | 1.90 | 29.45 |
| `Kidney__u0_base_d192__0shot` | Kidney | u0_base | 192 | 0shot | 17.13 | 0.7415 | 0.2508 | 56.02 | 0.6923 | 0.7145 | 8.56 | 10.75 | 1.69 | 14.32 |
| `Kidney__u0_base_d192__0shot_halfOD` | Kidney | u0_base | 192 | 0shot_halfOD | 17.38 | 0.7477 | 0.2355 | 53.54 | 0.6922 | 0.7173 | 8.55 | 10.75 | 1.69 | 14.32 |
| `Kidney__u0_base_d192__m6_k20` | Kidney | u0_base | 192 | 20shot | 23.19 | 0.8210 | 0.1218 | 49.38 | 0.7060 | 0.7321 | 6.57 | 10.75 | 1.80 | 13.09 |
| `Kidney__u1_up64_only_d192__0shot` | Kidney | u1_up64_only | 192 | 0shot | 19.20 | 0.7683 | 0.2057 | 44.04 | 0.6892 | 0.7056 | 8.86 | 10.75 | 1.79 | 26.93 |
| `Kidney__u1_up64_only_d192__0shot_halfOD` | Kidney | u1_up64_only | 192 | 0shot_halfOD | 19.43 | 0.7765 | 0.1916 | 42.11 | 0.6897 | 0.7077 | 8.44 | 10.75 | 1.79 | 26.93 |
| `Kidney__u1_up64_only_d192__m6_k20` | Kidney | u1_up64_only | 192 | 20shot | 23.96 | 0.8318 | 0.1103 | 41.86 | 0.7131 | 0.7412 | 6.99 | 10.75 | 1.90 | 26.99 |
| `liver__u0_base_d192__0shot` | liver | u0_base | 192 | 0shot | 17.98 | 0.7195 | 0.2539 | 24.95 | 0.5089 | 0.5258 | 8.52 | 15.74 | 1.69 | 12.82 |
| `liver__u0_base_d192__0shot_halfOD` | liver | u0_base | 192 | 0shot_halfOD | 19.00 | 0.7222 | 0.2219 | 29.45 | 0.5780 | 0.6068 | 7.91 | 15.74 | 1.69 | 12.82 |
| `liver__u0_base_d192__m6_k20` | liver | u0_base | 192 | 20shot | 23.52 | 0.7725 | 0.1231 | 18.03 | 0.7261 | 0.7478 | 7.28 | 15.74 | 1.80 | 13.21 |
| `liver__u1_up64_only_d192__0shot` | liver | u1_up64_only | 192 | 0shot | 19.92 | 0.7369 | 0.2046 | 23.90 | 0.5380 | 0.5413 | 7.73 | 15.74 | 1.79 | 26.62 |
| `liver__u1_up64_only_d192__0shot_halfOD` | liver | u1_up64_only | 192 | 0shot_halfOD | 20.67 | 0.7449 | 0.1920 | 28.61 | 0.5884 | 0.5978 | 7.53 | 15.74 | 1.79 | 26.62 |
| `liver__u1_up64_only_d192__m6_k20` | liver | u1_up64_only | 192 | 20shot | 24.72 | 0.7831 | 0.1129 | 15.68 | 0.7303 | 0.7507 | 6.89 | 15.74 | 1.90 | 28.45 |
