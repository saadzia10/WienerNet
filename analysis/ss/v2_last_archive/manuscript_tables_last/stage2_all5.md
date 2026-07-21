### Stage 2 (gap-filling) — all sites (5 towers, including the Redmere-1 OOD stress)

| Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 2.62 ± 1.82 | 3.81 ± 3.33 | 3.78 ± 5.32 | 2.265 ± 3.086 | 0.328 ± 0.160 | 0.865 ± 0.260 | 25 |
| WienerNet-SS (Gaussian) | 2.90 ± 1.98 | 4.55 ± 4.07 | 5.34 ± 7.68 | 4.201 ± 7.109 | 0.287 ± 0.158 | 0.897 ± 0.220 | 25 |
| WienerNet-SS (beta-NLL) | 2.61 ± 1.69 | 3.86 ± 2.82 | 3.89 ± 2.54 | 3.601 ± 5.364 | 0.305 ± 0.111 | 0.899 ± 0.205 | 25 |
| WienerNet-SS (Student-t) | 2.47 ± 1.55 | 3.53 ± 2.63 | 3.46 ± 2.96 | 2.509 ± 2.771 | 0.288 ± 0.169 | 0.868 ± 0.260 | 25 |
| WienerNet-SS (mixture) | 2.34 ± 1.52 | 3.16 ± 2.49 | 2.71 ± 1.61 | 1.650 ± 0.677 | 0.260 ± 0.072 | 0.951 ± 0.116 | 25 |
| WienerNet-SS (+residual) | 2.43 ± 1.66 | 3.32 ± 2.62 | 2.77 ± 1.06 | 2.467 ± 2.326 | 0.293 ± 0.126 | 0.900 ± 0.221 | 25 |
| WienerNet-SS (state-space) | — | — | — | — | — | — | 0 |
| WienerNet-SS (given-diurnal, Wiener) | 1.55 ± 0.26 | 1.72 ± 0.09 | 1.72 ± 0.19 | 1.601 ± 1.296 | 0.273 ± 0.073 | 0.925 ± 0.150 | 25 |
| WienerNet-SS (given-diurnal, state-sp) | 1.55 ± 0.26 | 1.72 ± 0.09 | 1.72 ± 0.19 | 1.935 ± 2.320 | 0.275 ± 0.077 | 0.945 ± 0.117 | 25 |
| WienerNet-SS (predicted-k) | — | — | — | — | — | — | 0 |
| Neural SDE (Gaussian) | 125.17 ± 348.06 | 170.19 ± 472.86 | 75.50 ± 202.98 | 3.970 ± 5.300 | 0.266 ± 0.110 | 0.922 ± 0.163 | 25 |
| Neural SDE (Student-t) | 131.60 ± 341.90 | 178.62 ± 464.48 | 78.36 ± 199.77 | 4.718 ± 6.742 | 0.251 ± 0.105 | 0.935 ± 0.160 | 25 |
| Neural SDE (ALD) | 18.69 ± 50.43 | 25.25 ± 68.55 | 12.20 ± 29.14 | 2.169 ± 1.914 | 0.308 ± 0.102 | 0.906 ± 0.219 | 25 |
| Mean-variance (Gaussian) | 53.61 ± 196.16 | 72.63 ± 266.57 | 32.46 ± 114.56 | 2.792 ± 4.751 | 0.253 ± 0.059 | 0.975 ± 0.047 | 25 |
| Mean-variance (Student-t) | 25.89 ± 72.15 | 34.97 ± 98.09 | 16.25 ± 41.86 | 2.030 ± 2.038 | 0.250 ± 0.061 | 0.953 ± 0.087 | 25 |
| Mean-variance (ALD) | 26.21 ± 75.20 | 35.36 ± 102.24 | 16.28 ± 43.71 | 1.910 ± 1.693 | 0.260 ± 0.049 | 0.975 ± 0.052 | 25 |
| Analytical SDE (Gaussian) | 1.54 ± 0.30 | 1.69 ± 0.09 | 1.68 ± 0.23 | 1.705 ± 0.118 | 0.303 ± 0.025 | 0.998 ± 0.003 | 5 |
| Analytical SDE (Student-t) | 1.54 ± 0.30 | 1.69 ± 0.09 | 1.68 ± 0.23 | 0.951 ± 0.126 | 0.177 ± 0.026 | 0.954 ± 0.026 | 5 |
| MDN (mixture) | 11.23 ± 30.01 | 14.99 ± 40.85 | 7.57 ± 17.24 | 1.686 ± 0.854 | 0.241 ± 0.048 | 0.975 ± 0.045 | 25 |
| Random Forest | 1.74 ± 0.19 | 1.37 ± 0.13 | 1.17 ± 0.22 | — | — | — | 25 |
| XGBoost | 2.00 ± 0.34 | 1.57 ± 0.34 | 1.33 ± 0.43 | — | — | — | 25 |

**Stage 2 — autoregressive gap-fill**, scored on the RAW predictive band (the +σ_struct band is a coverage device and is reported separately). mean ± 1 SD over all (site, seed) units (5 seeds × sites; the calibrated Analytical variants are seed-independent, n = 1 per site). RMSE by hours-into-gap is the stability test (flat ⇒ the physics drift tracks the nightly decline; climbing/diverging ⇒ it does not); CRPS / PIT-KS / cov90 are given at 5 h+, the discriminating horizon. `—` = arm not rolled out.  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. Because the LR schedule reacts to the held-out site, model *selection* is test-informed; reported metrics are nonetheless taken from `last.pth`.

