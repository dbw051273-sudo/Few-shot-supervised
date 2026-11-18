# Human AF->H&E (HKUST UniStain Clean-256) Evaluation Summary

| Run Tag | AF Mode | Fold | Arch | Stage | PSNR ↑ | SSIM ↑ | LPIPS ↓ | KID (×10⁻³) ↓ | Dice ↑ | F1@0.5 ↑ | FID ↓ |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Breast__u0_base_d192__0shot` | raw | Breast | u0_base | 0shot | 18.70 | 0.4624 | 0.4602 | 150.38 | 0.1705 | 0.1667 | 232.66 |
| `Breast__u0_base_d192__0shot_halfOD` | raw | Breast | u0_base | 0shot_halfOD | 18.09 | 0.4485 | 0.4617 | 152.30 | 0.1700 | 0.1667 | 228.81 |
| `Breast__u0_base_d192__m6_k20` | raw | Breast | u0_base | 20shot | 20.27 | 0.5175 | 0.3334 | 151.30 | 0.1836 | 0.1833 | 227.64 |
| `Breast__u1_up64_only_d192__0shot` | raw | Breast | u1_up64_only | 0shot | 18.29 | 0.4247 | 0.3812 | 139.44 | 0.1760 | 0.1750 | 212.83 |
| `Breast__u1_up64_only_d192__0shot_halfOD` | raw | Breast | u1_up64_only | 0shot_halfOD | 17.46 | 0.4071 | 0.3921 | 132.06 | 0.1679 | 0.1667 | 206.10 |
| `Breast__u1_up64_only_d192__m6_k20` | raw | Breast | u1_up64_only | 20shot | 21.03 | 0.5200 | 0.3213 | 152.12 | 0.1685 | 0.1667 | 221.52 |
| `Kidney__u0_base_d192__0shot` | raw | Kidney | u0_base | 0shot | 16.20 | 0.1887 | 0.5212 | 375.59 | 0.0192 | 0.0168 | 317.19 |
| `Kidney__u0_base_d192__0shot_halfOD` | raw | Kidney | u0_base | 0shot_halfOD | 17.59 | 0.2246 | 0.4971 | 370.55 | 0.0191 | 0.0168 | 309.54 |
| `Kidney__u0_base_d192__m6_k20` | raw | Kidney | u0_base | 20shot | 21.12 | 0.4489 | 0.3267 | 345.58 | 0.0188 | 0.0167 | 294.20 |
| `Kidney__u1_up64_only_d192__0shot` | raw | Kidney | u1_up64_only | 0shot | 16.52 | 0.1986 | 0.5240 | 355.95 | 0.0178 | 0.0167 | 309.53 |
| `Kidney__u1_up64_only_d192__0shot_halfOD` | raw | Kidney | u1_up64_only | 0shot_halfOD | 17.67 | 0.2328 | 0.5011 | 353.31 | 0.0181 | 0.0167 | 304.13 |
| `Kidney__u1_up64_only_d192__m6_k20` | raw | Kidney | u1_up64_only | 20shot | 20.90 | 0.4429 | 0.3375 | 347.69 | 0.0167 | 0.0167 | 294.18 |
