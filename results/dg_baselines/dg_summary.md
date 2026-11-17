# Classical DG Baselines Summary (UniFlow d192 Backbone)

- Updated: `2026-11-17 03:10:27`
- Core Metrics: `PSNR ↑`, `SSIM ↑`, `LPIPS ↓`, `KID (×10⁻³) ↓`, `Cellpose-proxy Dice ↑`, `Cellpose-proxy F1@0.5 ↑`

| Run Tag | Fold | DG Method | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID (×10⁻³) ↓ | Dice ↑ | F1@0.5 ↑ | bg_lap | gt_lap | Params(M) | ms/img |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Brain__dann_d192__0shot` | Brain | dann | 0shot | 11.97 | 0.5769 | 0.4364 | 241.32 | 0.6101 | 0.5433 | 4.07 | 7.28 | 1.69 | 13.66 |
| `Brain__dann_d192__0shot_halfOD` | Brain | dann | 0shot_halfOD | 12.73 | 0.6037 | 0.3973 | 236.95 | 0.5907 | 0.5380 | 4.04 | 7.28 | 1.69 | 13.66 |
| `Brain__dann_d192__m6_k20` | Brain | dann | 20shot | 21.33 | 0.7222 | 0.1767 | 82.30 | 0.5780 | 0.5674 | 3.53 | 7.28 | 1.80 | 14.02 |
| `Brain__mixstyle_d192__0shot` | Brain | mixstyle | 0shot | 12.91 | 0.6520 | 0.3792 | 92.53 | 0.7095 | 0.6805 | 3.70 | 7.28 | 1.69 | 14.01 |
| `Brain__mixstyle_d192__0shot_halfOD` | Brain | mixstyle | 0shot_halfOD | 13.29 | 0.6544 | 0.3671 | 89.13 | 0.7116 | 0.6865 | 3.86 | 7.28 | 1.69 | 14.01 |
| `Brain__mixstyle_d192__m6_k20` | Brain | mixstyle | 20shot | 23.54 | 0.7565 | 0.1289 | 25.72 | 0.7463 | 0.7132 | 4.40 | 7.28 | 1.80 | 12.81 |
| `Brain__mldg_d192__0shot` | Brain | mldg | 0shot | 12.66 | 0.6516 | 0.3870 | 84.70 | 0.7046 | 0.6758 | 3.95 | 7.28 | 1.69 | 12.39 |
| `Brain__mldg_d192__0shot_halfOD` | Brain | mldg | 0shot_halfOD | 12.76 | 0.6516 | 0.3831 | 85.30 | 0.7026 | 0.6791 | 3.97 | 7.28 | 1.69 | 12.39 |
| `Brain__mldg_d192__m6_k20` | Brain | mldg | 20shot | 23.80 | 0.7592 | 0.1266 | 24.23 | 0.7464 | 0.7157 | 4.40 | 7.28 | 1.80 | 12.85 |
| `Brain__stainaug_d192__0shot` | Brain | stainaug | 0shot | 12.79 | 0.6533 | 0.3878 | 104.84 | 0.7096 | 0.6763 | 4.67 | 7.28 | 1.69 | 13.02 |
| `Brain__stainaug_d192__0shot_halfOD` | Brain | stainaug | 0shot_halfOD | 13.42 | 0.6587 | 0.3706 | 97.78 | 0.7065 | 0.6799 | 4.88 | 7.28 | 1.69 | 13.02 |
| `Brain__stainaug_d192__m6_k20` | Brain | stainaug | 20shot | 23.82 | 0.7569 | 0.1266 | 26.19 | 0.7418 | 0.7129 | 4.39 | 7.28 | 1.80 | 14.59 |
| `Breast__dann_d192__0shot` | Breast | dann | 0shot | 15.92 | 0.7236 | 0.3405 | 72.13 | 0.3392 | 0.2303 | 6.69 | 2.86 | 1.69 | 12.74 |
| `Breast__dann_d192__0shot_halfOD` | Breast | dann | 0shot_halfOD | 18.24 | 0.7399 | 0.3301 | 64.24 | 0.3208 | 0.1970 | 4.67 | 2.86 | 1.69 | 12.74 |
| `Breast__dann_d192__m6_k20` | Breast | dann | 20shot | 22.43 | 0.8061 | 0.2076 | 22.36 | 0.4942 | 0.4887 | 1.75 | 2.86 | 1.80 | 14.35 |
| `Breast__mixstyle_d192__0shot` | Breast | mixstyle | 0shot | 13.60 | 0.7011 | 0.3204 | 46.22 | 0.5004 | 0.5097 | 3.68 | 2.86 | 1.69 | 15.86 |
| `Breast__mixstyle_d192__0shot_halfOD` | Breast | mixstyle | 0shot_halfOD | 18.89 | 0.7774 | 0.2661 | 31.41 | 0.5040 | 0.5100 | 3.60 | 2.86 | 1.69 | 15.86 |
| `Breast__mixstyle_d192__m6_k20` | Breast | mixstyle | 20shot | 22.90 | 0.8141 | 0.1751 | 10.60 | 0.5502 | 0.5535 | 1.74 | 2.86 | 1.80 | 12.33 |
| `Breast__mldg_d192__0shot` | Breast | mldg | 0shot | 15.55 | 0.7273 | 0.2749 | 40.75 | 0.5060 | 0.5133 | 4.32 | 2.86 | 1.69 | 12.87 |
| `Breast__mldg_d192__0shot_halfOD` | Breast | mldg | 0shot_halfOD | 19.73 | 0.7822 | 0.2295 | 24.49 | 0.5158 | 0.5208 | 4.27 | 2.86 | 1.69 | 12.87 |
| `Breast__mldg_d192__m6_k20` | Breast | mldg | 20shot | 23.11 | 0.8153 | 0.1728 | 11.32 | 0.5618 | 0.5690 | 1.75 | 2.86 | 1.80 | 14.28 |
| `Breast__stainaug_d192__0shot` | Breast | stainaug | 0shot | 14.75 | 0.7266 | 0.3130 | 48.94 | 0.5070 | 0.5119 | 3.85 | 2.86 | 1.69 | 12.88 |
| `Breast__stainaug_d192__0shot_halfOD` | Breast | stainaug | 0shot_halfOD | 19.57 | 0.7835 | 0.2616 | 33.90 | 0.5123 | 0.5195 | 3.76 | 2.86 | 1.69 | 12.88 |
| `Breast__stainaug_d192__m6_k20` | Breast | stainaug | 20shot | 23.13 | 0.8145 | 0.1726 | 10.97 | 0.5493 | 0.5521 | 1.71 | 2.86 | 1.80 | 14.82 |
| `Kidney__dann_d192__0shot` | Kidney | dann | 0shot | 17.29 | 0.7206 | 0.3727 | 109.08 | 0.4725 | 0.2624 | 8.47 | 10.76 | 1.69 | 16.73 |
| `Kidney__dann_d192__0shot_halfOD` | Kidney | dann | 0shot_halfOD | 17.53 | 0.7246 | 0.3525 | 114.26 | 0.4807 | 0.2913 | 8.19 | 10.76 | 1.69 | 16.73 |
| `Kidney__dann_d192__m6_k20` | Kidney | dann | 20shot | 21.50 | 0.8021 | 0.1524 | 52.77 | 0.6848 | 0.7019 | 7.83 | 10.76 | 1.80 | 13.01 |
| `Kidney__mixstyle_d192__0shot` | Kidney | mixstyle | 0shot | 16.09 | 0.7275 | 0.2666 | 63.89 | 0.6860 | 0.7039 | 7.29 | 10.76 | 1.69 | 12.32 |
| `Kidney__mixstyle_d192__0shot_halfOD` | Kidney | mixstyle | 0shot_halfOD | 16.92 | 0.7396 | 0.2375 | 58.86 | 0.6797 | 0.6959 | 7.41 | 10.76 | 1.69 | 12.32 |
| `Kidney__mixstyle_d192__m6_k20` | Kidney | mixstyle | 20shot | 23.13 | 0.8203 | 0.1233 | 49.01 | 0.7072 | 0.7328 | 6.29 | 10.76 | 1.80 | 13.47 |
| `Kidney__mldg_d192__0shot` | Kidney | mldg | 0shot | 16.25 | 0.7264 | 0.2648 | 58.90 | 0.6877 | 0.7070 | 8.23 | 10.76 | 1.69 | 12.71 |
| `Kidney__mldg_d192__0shot_halfOD` | Kidney | mldg | 0shot_halfOD | 16.72 | 0.7357 | 0.2407 | 54.15 | 0.6837 | 0.6971 | 8.40 | 10.76 | 1.69 | 12.71 |
| `Kidney__mldg_d192__m6_k20` | Kidney | mldg | 20shot | 23.13 | 0.8222 | 0.1209 | 49.68 | 0.7067 | 0.7334 | 6.36 | 10.76 | 1.80 | 13.93 |
| `Kidney__stainaug_d192__0shot` | Kidney | stainaug | 0shot | 16.91 | 0.7359 | 0.2486 | 66.01 | 0.6850 | 0.7030 | 7.88 | 10.76 | 1.69 | 13.18 |
| `Kidney__stainaug_d192__0shot_halfOD` | Kidney | stainaug | 0shot_halfOD | 17.60 | 0.7483 | 0.2307 | 62.36 | 0.6892 | 0.7099 | 8.04 | 10.76 | 1.69 | 13.18 |
| `Kidney__stainaug_d192__m6_k20` | Kidney | stainaug | 20shot | 23.14 | 0.8210 | 0.1222 | 48.58 | 0.7066 | 0.7334 | 6.48 | 10.76 | 1.80 | 12.07 |
| `liver__dann_d192__0shot` | liver | dann | 0shot | 15.51 | 0.6510 | 0.3229 | 105.97 | 0.3849 | 0.4022 | 7.84 | 15.74 | 1.69 | 14.05 |
| `liver__dann_d192__0shot_halfOD` | liver | dann | 0shot_halfOD | 18.37 | 0.6812 | 0.2933 | 118.26 | 0.3691 | 0.3897 | 7.62 | 15.74 | 1.69 | 14.05 |
| `liver__dann_d192__m6_k20` | liver | dann | 20shot | 22.48 | 0.7522 | 0.1584 | 29.06 | 0.6677 | 0.6990 | 6.25 | 15.74 | 1.80 | 15.21 |
| `liver__mixstyle_d192__0shot` | liver | mixstyle | 0shot | 17.13 | 0.7072 | 0.2444 | 30.37 | 0.5142 | 0.5226 | 7.68 | 15.74 | 1.69 | 12.30 |
| `liver__mixstyle_d192__0shot_halfOD` | liver | mixstyle | 0shot_halfOD | 19.46 | 0.7210 | 0.2095 | 28.60 | 0.5873 | 0.6022 | 6.83 | 15.74 | 1.69 | 12.30 |
| `liver__mixstyle_d192__m6_k20` | liver | mixstyle | 20shot | 23.94 | 0.7728 | 0.1237 | 18.12 | 0.7209 | 0.7424 | 7.24 | 15.74 | 1.80 | 13.17 |
| `liver__mldg_d192__0shot` | liver | mldg | 0shot | 17.73 | 0.7122 | 0.2492 | 22.24 | 0.5564 | 0.5659 | 7.71 | 15.74 | 1.69 | 13.82 |
| `liver__mldg_d192__0shot_halfOD` | liver | mldg | 0shot_halfOD | 18.91 | 0.7112 | 0.2146 | 27.68 | 0.6096 | 0.6332 | 6.62 | 15.74 | 1.69 | 13.82 |
| `liver__mldg_d192__m6_k20` | liver | mldg | 20shot | 23.56 | 0.7727 | 0.1229 | 19.62 | 0.7205 | 0.7391 | 7.96 | 15.74 | 1.80 | 12.02 |
| `liver__stainaug_d192__0shot` | liver | stainaug | 0shot | 18.80 | 0.7211 | 0.2239 | 26.19 | 0.5681 | 0.5803 | 7.73 | 15.74 | 1.69 | 13.74 |
| `liver__stainaug_d192__0shot_halfOD` | liver | stainaug | 0shot_halfOD | 20.47 | 0.7326 | 0.1995 | 30.51 | 0.6237 | 0.6432 | 7.32 | 15.74 | 1.69 | 13.74 |
| `liver__stainaug_d192__m6_k20` | liver | stainaug | 20shot | 23.95 | 0.7752 | 0.1229 | 19.10 | 0.7222 | 0.7449 | 7.79 | 15.74 | 1.80 | 14.40 |
