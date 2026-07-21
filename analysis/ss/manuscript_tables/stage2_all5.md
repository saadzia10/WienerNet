### Stage 2 (gap-filling) — all sites (5 towers, including the Redmere-1 OOD stress)

| Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 2.33 ± 1.56 | 3.11 ± 2.47 | 2.50 ± 0.96 | 1.464 ± 0.541 | 0.284 ± 0.097 | 0.888 ± 0.198 | 25 |
| WienerNet-SS (Gaussian) | 2.82 ± 1.70 | 4.38 ± 3.18 | 4.43 ± 3.37 | 3.809 ± 4.243 | 0.265 ± 0.064 | 0.928 ± 0.126 | 25 |
| WienerNet-SS (beta-NLL) | 2.48 ± 1.58 | 3.52 ± 2.52 | 3.64 ± 2.07 | 2.518 ± 1.805 | 0.271 ± 0.097 | 0.916 ± 0.149 | 25 |
| WienerNet-SS (Student-t) | 2.09 ± 1.18 | 2.74 ± 1.83 | 2.72 ± 0.92 | 1.639 ± 0.523 | 0.254 ± 0.076 | 0.927 ± 0.137 | 25 |
| WienerNet-SS (mixture) | 2.16 ± 1.37 | 2.79 ± 2.12 | 2.44 ± 0.93 | 1.588 ± 0.405 | 0.253 ± 0.051 | 0.974 ± 0.044 | 25 |
| WienerNet-SS (+residual) | 2.18 ± 1.33 | 3.01 ± 2.19 | 2.97 ± 2.23 | 1.930 ± 1.521 | 0.250 ± 0.100 | 0.883 ± 0.202 | 25 |
| WienerNet-SS (predicted-k) | 2.60 ± 1.54 | 3.51 ± 2.65 | 3.06 ± 2.96 | 2.224 ± 2.383 | 0.366 ± 0.173 | 0.854 ± 0.322 | 25 |
| WienerNet-SS (state-space) | 2.27 ± 1.14 | 2.83 ± 1.76 | 2.49 ± 1.21 | 1.669 ± 0.845 | 0.269 ± 0.080 | 0.903 ± 0.169 | 25 |
| WienerNet-SS (state-space, +residual) | 2.21 ± 1.03 | 2.93 ± 1.71 | 2.97 ± 1.75 | 2.406 ± 2.385 | 0.292 ± 0.076 | 0.912 ± 0.170 | 25 |
| WienerNet-SS (state-space, predicted-k) | 2.69 ± 1.44 | 3.84 ± 2.75 | 3.74 ± 3.91 | 2.601 ± 2.588 | 0.324 ± 0.114 | 0.926 ± 0.193 | 25 |
| WienerNet-SS (ALD + sigma_struct band) | 2.33 ± 1.56 | 3.11 ± 2.47 | 2.50 ± 0.96 | 1.481 ± 0.478 | 0.277 ± 0.083 | 0.945 ± 0.103 | 25 |
| WienerNet-SS (given-diurnal, Wiener) | 1.55 ± 0.26 | 1.72 ± 0.09 | 1.72 ± 0.19 | 1.185 ± 0.176 | 0.243 ± 0.065 | 0.925 ± 0.130 | 25 |
| WienerNet-SS (given-diurnal, state-sp) | 1.55 ± 0.26 | 1.72 ± 0.09 | 1.72 ± 0.19 | 1.183 ± 0.148 | 0.257 ± 0.061 | 0.933 ± 0.125 | 25 |
| Neural SDE (Gaussian) | 129.64 ± 287.00 | 176.49 ± 389.79 | 78.78 ± 166.75 | 4.973 ± 5.343 | 0.274 ± 0.120 | 0.908 ± 0.158 | 25 |
| Neural SDE (Student-t) | 44.68 ± 94.97 | 60.61 ± 129.00 | 27.65 ± 54.96 | 3.141 ± 3.458 | 0.242 ± 0.065 | 0.948 ± 0.068 | 25 |
| Neural SDE (ALD) | 8.23 ± 23.39 | 11.04 ± 31.79 | 6.24 ± 13.33 | 1.774 ± 0.719 | 0.254 ± 0.071 | 0.894 ± 0.155 | 25 |
| Mean-variance (Gaussian) | 17.14 ± 36.16 | 23.03 ± 49.23 | 10.99 ± 20.75 | 1.810 ± 0.811 | 0.273 ± 0.039 | 0.995 ± 0.006 | 25 |
| Mean-variance (Student-t) | 14.77 ± 32.65 | 19.80 ± 44.46 | 9.58 ± 18.73 | 1.551 ± 0.683 | 0.235 ± 0.048 | 0.973 ± 0.037 | 25 |
| Mean-variance (ALD) | 2.83 ± 4.85 | 3.54 ± 6.66 | 2.53 ± 2.68 | 1.279 ± 0.247 | 0.260 ± 0.051 | 0.972 ± 0.053 | 25 |
| Analytical SDE (Gaussian) | 1.54 ± 0.30 | 1.69 ± 0.09 | 1.68 ± 0.23 | 1.703 ± 0.115 | 0.303 ± 0.024 | 0.998 ± 0.003 | 5 |
| Analytical SDE (Student-t) | 1.54 ± 0.30 | 1.69 ± 0.09 | 1.68 ± 0.23 | 1.030 ± 0.266 | 0.193 ± 0.037 | 0.959 ± 0.032 | 5 |
| MDN (mixture) | 14.61 ± 38.30 | 19.56 ± 52.14 | 9.42 ± 22.13 | 1.668 ± 1.014 | 0.271 ± 0.037 | 0.986 ± 0.021 | 25 |
| Random Forest | 1.57 ± 0.13 | 1.58 ± 0.19 | 1.41 ± 0.30 | — | — | — | 25 |
| XGBoost | 1.84 ± 0.38 | 1.83 ± 0.46 | 1.67 ± 0.60 | — | — | — | 25 |

**Stage 2 — autoregressive gap-fill**, scored on the RAW predictive band (the +σ_struct band is a coverage device and is reported separately). mean ± 1 SD over all (site, seed) units (5 seeds × sites; the calibrated Analytical variants are seed-independent, n = 1 per site). RMSE by hours-into-gap is the stability test (flat ⇒ the physics drift tracks the nightly decline; climbing/diverging ⇒ it does not); CRPS / PIT-KS / cov90 are given at 5 h+, the discriminating horizon. `—` = arm not rolled out.  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. **These values are scored from `best.pth`, which with `data.val_frac` unset is the epoch selected ON THE HELD-OUT TEST SITE.** They are therefore an optimistic, test-selected bound rather than an honest estimate of generalisation, and they are not directly comparable to the calibrated Analytical SDE and tree arms, which have no epoch to select and gain nothing from it. The leakage-free `last.pth` scores are archived in `analysis/ss/v2_last_archive/`.

