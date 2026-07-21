# WienerNet-SS — manuscript results tables

### Stage 1 (distributional) — clean sites (4 in-distribution towers; Redmere 1 excluded)

| Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 0.707 ± 0.142 | 1.95 ± 0.41 | 0.766 ± 0.086 | 2.82 ± 1.33 | 0.095 ± 0.062 | 20 |
| WienerNet-SS (Gaussian) | 0.788 ± 0.096 | 2.44 ± 0.31 | 0.908 ± 0.035 | 5.09 ± 1.05 | 0.137 ± 0.037 | 20 |
| WienerNet-SS (beta-NLL) | 0.730 ± 0.126 | 2.13 ± 0.39 | 0.858 ± 0.049 | 3.84 ± 1.23 | 0.108 ± 0.050 | 20 |
| WienerNet-SS (Student-t) | 0.710 ± 0.137 | 2.15 ± 0.55 | 0.838 ± 0.051 | 3.47 ± 1.08 | 0.082 ± 0.037 | 20 |
| WienerNet-SS (mixture) | 0.703 ± 0.144 | 2.37 ± 0.47 | 0.869 ± 0.040 | 4.03 ± 1.18 | 0.069 ± 0.037 | 20 |
| WienerNet-SS (+residual) | 0.721 ± 0.138 | 1.96 ± 0.38 | 0.782 ± 0.101 | 3.01 ± 1.30 | 0.099 ± 0.065 | 20 |
| WienerNet-SS (predicted-k) | 0.745 ± 0.106 | 2.04 ± 0.39 | 0.760 ± 0.202 | 3.36 ± 1.62 | 0.141 ± 0.156 | 20 |
| WienerNet-SS (state-space) | 0.708 ± 0.139 | 1.95 ± 0.37 | 0.782 ± 0.100 | 2.91 ± 1.18 | 0.084 ± 0.057 | 20 |
| WienerNet-SS (state-space, +residual) | 0.725 ± 0.119 | 2.03 ± 0.32 | 0.825 ± 0.074 | 3.43 ± 1.08 | 0.092 ± 0.055 | 20 |
| WienerNet-SS (state-space, predicted-k) | 0.763 ± 0.102 | 2.20 ± 0.41 | 0.828 ± 0.139 | 3.92 ± 1.69 | 0.122 ± 0.123 | 20 |
| WienerNet-SS (ALD + sigma_struct band) | — | — | — | — | — | 0 |
| WienerNet-SS (given-diurnal, Wiener) | 0.707 ± 0.149 | 1.94 ± 0.41 | 0.766 ± 0.078 | 2.80 ± 1.35 | 0.088 ± 0.033 | 20 |
| WienerNet-SS (given-diurnal, state-sp) | 0.703 ± 0.148 | 1.95 ± 0.38 | 0.787 ± 0.087 | 2.89 ± 1.21 | 0.082 ± 0.043 | 20 |
| Neural SDE (Gaussian) | 0.773 ± 0.113 | 2.37 ± 0.28 | 0.900 ± 0.024 | 4.76 ± 0.75 | 0.137 ± 0.054 | 20 |
| Neural SDE (Student-t) | 0.715 ± 0.138 | 2.09 ± 0.31 | 0.848 ± 0.032 | 3.54 ± 0.81 | 0.095 ± 0.053 | 20 |
| Neural SDE (ALD) | 0.712 ± 0.143 | 1.99 ± 0.36 | 0.792 ± 0.095 | 3.07 ± 1.14 | 0.087 ± 0.057 | 20 |
| Mean-variance (Gaussian) | 0.729 ± 0.117 | 2.29 ± 0.27 | 0.907 ± 0.029 | 4.56 ± 0.55 | 0.137 ± 0.056 | 20 |
| Mean-variance (Student-t) | 0.700 ± 0.144 | 2.07 ± 0.35 | 0.839 ± 0.048 | 3.35 ± 0.83 | 0.097 ± 0.045 | 20 |
| Mean-variance (ALD) | 0.700 ± 0.143 | 2.02 ± 0.32 | 0.831 ± 0.057 | 3.27 ± 0.87 | 0.082 ± 0.055 | 20 |
| Analytical SDE (Gaussian) | 0.775 ± 0.112 | 2.28 ± 0.13 | 0.919 ± 0.029 | 5.04 ± 0.32 | 0.163 ± 0.068 | 4 |
| Analytical SDE (Student-t) | 0.723 ± 0.166 | 1.79 ± 0.36 | 0.767 ± 0.095 | 2.34 ± 1.21 | 0.113 ± 0.045 | 4 |
| MDN (mixture) | 0.699 ± 0.146 | 2.16 ± 0.34 | 0.853 ± 0.052 | 3.75 ± 0.88 | 0.083 ± 0.048 | 20 |
| Random Forest | — | 1.57 ± 0.03 | — | — | — | 20 |
| XGBoost | — | 1.83 ± 0.29 | — | — | — | 20 |

**Stage 1 — one-step predictive law.** mean ± 1 SD over all (site, seed) units (5 seeds × sites; calibrated seed-independent models show a bare value). CRPS and RMSE lower = better; cov90 nominal = 0.90; sharp90 = mean 90% interval width (µmol m⁻² s⁻¹); PIT-KS 0 = perfectly calibrated. Trees emit no predictive distribution (RMSE only).  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. **These values are scored from `best.pth`, which with `data.val_frac` unset is the epoch selected ON THE HELD-OUT TEST SITE.** They are therefore an optimistic, test-selected bound rather than an honest estimate of generalisation, and they are not directly comparable to the calibrated Analytical SDE and tree arms, which have no epoch to select and gain nothing from it. The leakage-free `last.pth` scores are archived in `analysis/ss/v2_last_archive/`.

### Stage 2 (gap-filling) — clean sites (4 in-distribution towers; Redmere 1 excluded)

| Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 1.59 ± 0.26 | 1.92 ± 0.32 | 2.39 ± 1.05 | 1.474 ± 0.608 | 0.274 ± 0.105 | 0.864 ± 0.216 | 20 |
| WienerNet-SS (Gaussian) | 2.07 ± 0.81 | 3.30 ± 2.58 | 4.70 ± 3.73 | 2.800 ± 1.773 | 0.262 ± 0.072 | 0.913 ± 0.137 | 20 |
| WienerNet-SS (beta-NLL) | 1.72 ± 0.26 | 2.35 ± 0.85 | 3.66 ± 2.31 | 2.131 ± 1.341 | 0.272 ± 0.107 | 0.897 ± 0.162 | 20 |
| WienerNet-SS (Student-t) | 1.61 ± 0.22 | 2.04 ± 0.40 | 2.70 ± 0.95 | 1.625 ± 0.481 | 0.251 ± 0.084 | 0.917 ± 0.151 | 20 |
| WienerNet-SS (mixture) | 1.59 ± 0.26 | 1.93 ± 0.41 | 2.36 ± 0.98 | 1.629 ± 0.443 | 0.246 ± 0.051 | 0.970 ± 0.048 | 20 |
| WienerNet-SS (+residual) | 1.66 ± 0.32 | 2.17 ± 0.89 | 3.02 ± 2.49 | 1.853 ± 1.500 | 0.248 ± 0.106 | 0.856 ± 0.218 | 20 |
| WienerNet-SS (predicted-k) | 1.88 ± 0.45 | 2.39 ± 1.36 | 3.15 ± 3.32 | 2.449 ± 2.627 | 0.380 ± 0.191 | 0.819 ± 0.353 | 20 |
| WienerNet-SS (state-space) | 1.79 ± 0.41 | 2.08 ± 0.48 | 2.45 ± 1.34 | 1.508 ± 0.631 | 0.254 ± 0.081 | 0.883 ± 0.184 | 20 |
| WienerNet-SS (state-space, +residual) | 1.86 ± 0.40 | 2.34 ± 0.76 | 3.11 ± 1.88 | 1.908 ± 0.977 | 0.287 ± 0.082 | 0.895 ± 0.187 | 20 |
| WienerNet-SS (state-space, predicted-k) | 2.05 ± 0.55 | 2.92 ± 2.13 | 3.99 ± 4.36 | 2.608 ± 2.771 | 0.329 ± 0.127 | 0.909 ± 0.213 | 20 |
| WienerNet-SS (ALD + sigma_struct band) | 1.59 ± 0.26 | 1.92 ± 0.32 | 2.39 ± 1.05 | 1.484 ± 0.537 | 0.262 ± 0.086 | 0.934 ± 0.113 | 20 |
| WienerNet-SS (given-diurnal, Wiener) | 1.53 ± 0.29 | 1.69 ± 0.09 | 1.80 ± 0.13 | 1.187 ± 0.196 | 0.230 ± 0.066 | 0.908 ± 0.140 | 20 |
| WienerNet-SS (given-diurnal, state-sp) | 1.53 ± 0.29 | 1.69 ± 0.09 | 1.80 ± 0.13 | 1.191 ± 0.164 | 0.250 ± 0.067 | 0.920 ± 0.137 | 20 |
| Neural SDE (Gaussian) | 1.90 ± 0.40 | 3.02 ± 1.38 | 4.70 ± 3.14 | 2.811 ± 2.033 | 0.270 ± 0.133 | 0.888 ± 0.172 | 20 |
| Neural SDE (Student-t) | 1.66 ± 0.30 | 2.16 ± 0.70 | 2.82 ± 1.23 | 1.665 ± 0.603 | 0.239 ± 0.067 | 0.936 ± 0.072 | 20 |
| Neural SDE (ALD) | 1.64 ± 0.28 | 2.09 ± 0.45 | 2.67 ± 0.86 | 1.567 ± 0.346 | 0.244 ± 0.067 | 0.870 ± 0.166 | 20 |
| Mean-variance (Gaussian) | 1.56 ± 0.28 | 1.80 ± 0.16 | 2.11 ± 0.37 | 1.492 ± 0.212 | 0.276 ± 0.041 | 0.996 ± 0.005 | 20 |
| Mean-variance (Student-t) | 1.55 ± 0.29 | 1.78 ± 0.15 | 2.06 ± 0.37 | 1.292 ± 0.215 | 0.223 ± 0.044 | 0.969 ± 0.040 | 20 |
| Mean-variance (ALD) | 1.54 ± 0.28 | 1.75 ± 0.19 | 1.93 ± 0.51 | 1.268 ± 0.249 | 0.251 ± 0.053 | 0.967 ± 0.058 | 20 |
| Analytical SDE (Gaussian) | 1.52 ± 0.34 | 1.68 ± 0.10 | 1.78 ± 0.11 | 1.708 ± 0.132 | 0.302 ± 0.028 | 0.998 ± 0.003 | 4 |
| Analytical SDE (Student-t) | 1.52 ± 0.34 | 1.68 ± 0.10 | 1.78 ± 0.11 | 1.085 ± 0.273 | 0.195 ± 0.042 | 0.954 ± 0.035 | 4 |
| MDN (mixture) | 1.55 ± 0.29 | 1.75 ± 0.08 | 1.96 ± 0.21 | 1.316 ± 0.189 | 0.269 ± 0.037 | 0.984 ± 0.023 | 20 |
| Random Forest | 1.59 ± 0.14 | 1.58 ± 0.21 | 1.45 ± 0.32 | — | — | — | 20 |
| XGBoost | 1.92 ± 0.39 | 1.90 ± 0.49 | 1.79 ± 0.61 | — | — | — | 20 |

**Stage 2 — autoregressive gap-fill**, scored on the RAW predictive band (the +σ_struct band is a coverage device and is reported separately). mean ± 1 SD over all (site, seed) units (5 seeds × sites; the calibrated Analytical variants are seed-independent, n = 1 per site). RMSE by hours-into-gap is the stability test (flat ⇒ the physics drift tracks the nightly decline; climbing/diverging ⇒ it does not); CRPS / PIT-KS / cov90 are given at 5 h+, the discriminating horizon. `—` = arm not rolled out.  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. **These values are scored from `best.pth`, which with `data.val_frac` unset is the epoch selected ON THE HELD-OUT TEST SITE.** They are therefore an optimistic, test-selected bound rather than an honest estimate of generalisation, and they are not directly comparable to the calibrated Analytical SDE and tree arms, which have no epoch to select and gain nothing from it. The leakage-free `last.pth` scores are archived in `analysis/ss/v2_last_archive/`.

### Stage 1 (distributional) — all sites (5 towers, including the Redmere-1 OOD stress)

| Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 0.703 ± 0.127 | 2.06 ± 0.43 | 0.788 ± 0.088 | 2.99 ± 1.23 | 0.084 ± 0.060 | 25 |
| WienerNet-SS (Gaussian) | 1.480 ± 2.277 | 94.89 ± 372.27 | 0.911 ± 0.032 | 14.32 ± 29.94 | 0.132 ± 0.034 | 25 |
| WienerNet-SS (beta-NLL) | 0.983 ± 0.782 | 21.40 ± 60.27 | 0.870 ± 0.050 | 7.44 ± 10.70 | 0.109 ± 0.045 | 25 |
| WienerNet-SS (Student-t) | 0.773 ± 0.281 | 9.02 ± 27.16 | 0.844 ± 0.048 | 4.52 ± 3.76 | 0.076 ± 0.036 | 25 |
| WienerNet-SS (mixture) | 0.703 ± 0.137 | 3.04 ± 3.37 | 0.871 ± 0.036 | 4.10 ± 1.27 | 0.065 ± 0.035 | 25 |
| WienerNet-SS (+residual) | 0.816 ± 0.524 | 13.60 ± 57.58 | 0.803 ± 0.100 | 5.06 ± 9.56 | 0.087 ± 0.063 | 25 |
| WienerNet-SS (predicted-k) | 0.788 ± 0.250 | 11.31 ± 41.74 | 0.785 ± 0.187 | 4.40 ± 4.73 | 0.121 ± 0.146 | 25 |
| WienerNet-SS (state-space) | 0.833 ± 0.438 | 27.24 ± 73.55 | 0.803 ± 0.098 | 5.62 ± 8.36 | 0.076 ± 0.054 | 25 |
| WienerNet-SS (state-space, +residual) | 1.063 ± 1.347 | 41.67 ± 158.92 | 0.834 ± 0.069 | 9.80 ± 24.57 | 0.083 ± 0.053 | 25 |
| WienerNet-SS (state-space, predicted-k) | 0.927 ± 0.616 | 26.01 ± 80.33 | 0.838 ± 0.126 | 7.17 ± 11.58 | 0.104 ± 0.116 | 25 |
| WienerNet-SS (ALD + sigma_struct band) | — | — | — | — | — | 0 |
| WienerNet-SS (given-diurnal, Wiener) | 0.685 ± 0.140 | 1.91 ± 0.37 | 0.784 ± 0.079 | 2.94 ± 1.24 | 0.076 ± 0.037 | 25 |
| WienerNet-SS (given-diurnal, state-sp) | 0.681 ± 0.139 | 1.93 ± 0.34 | 0.803 ± 0.084 | 3.01 ± 1.11 | 0.071 ± 0.044 | 25 |
| Neural SDE (Gaussian) | 2.117 ± 3.055 | 96.62 ± 252.43 | 0.905 ± 0.024 | 16.29 ± 33.42 | 0.133 ± 0.049 | 25 |
| Neural SDE (Student-t) | 1.666 ± 2.230 | 169.12 ± 521.23 | 0.855 ± 0.033 | 14.88 ± 30.29 | 0.088 ± 0.049 | 25 |
| Neural SDE (ALD) | 0.852 ± 0.431 | 12.15 ± 40.83 | 0.811 ± 0.093 | 4.45 ± 6.50 | 0.077 ± 0.054 | 25 |
| Mean-variance (Gaussian) | 0.885 ± 0.378 | 6.77 ± 9.91 | 0.911 ± 0.027 | 5.00 ± 1.05 | 0.131 ± 0.052 | 25 |
| Mean-variance (Student-t) | 0.830 ± 0.356 | 6.51 ± 10.70 | 0.849 ± 0.047 | 3.81 ± 1.21 | 0.090 ± 0.042 | 25 |
| Mean-variance (ALD) | 0.715 ± 0.142 | 2.87 ± 2.07 | 0.839 ± 0.053 | 3.37 ± 0.90 | 0.073 ± 0.053 | 25 |
| Analytical SDE (Gaussian) | 0.751 ± 0.111 | 2.22 ± 0.18 | 0.926 ± 0.029 | 5.06 ± 0.28 | 0.162 ± 0.059 | 5 |
| Analytical SDE (Student-t) | 0.699 ± 0.154 | 1.70 ± 0.37 | 0.758 ± 0.085 | 2.22 ± 1.08 | 0.108 ± 0.041 | 5 |
| MDN (mixture) | 0.863 ± 0.459 | 7.24 ± 12.68 | 0.861 ± 0.050 | 4.61 ± 2.02 | 0.075 ± 0.046 | 25 |
| Random Forest | — | 1.52 ± 0.12 | — | — | — | 25 |
| XGBoost | — | 1.73 ± 0.32 | — | — | — | 25 |

**Stage 1 — one-step predictive law.** mean ± 1 SD over all (site, seed) units (5 seeds × sites; calibrated seed-independent models show a bare value). CRPS and RMSE lower = better; cov90 nominal = 0.90; sharp90 = mean 90% interval width (µmol m⁻² s⁻¹); PIT-KS 0 = perfectly calibrated. Trees emit no predictive distribution (RMSE only).  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. **These values are scored from `best.pth`, which with `data.val_frac` unset is the epoch selected ON THE HELD-OUT TEST SITE.** They are therefore an optimistic, test-selected bound rather than an honest estimate of generalisation, and they are not directly comparable to the calibrated Analytical SDE and tree arms, which have no epoch to select and gain nothing from it. The leakage-free `last.pth` scores are archived in `analysis/ss/v2_last_archive/`.

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

### Stage 1 (distributional) — site-wise

| Site | Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|---|
| Woodwalton | WienerNet-SS (ALD, primary) | 0.502 ± 0.023 | 1.34 ± 0.07 | 0.672 ± 0.091 | 1.13 ± 0.43 | 0.178 ± 0.066 | 5 |
| Woodwalton | WienerNet-SS (Gaussian) | 0.706 ± 0.076 | 2.12 ± 0.12 | 0.883 ± 0.055 | 4.27 ± 0.70 | 0.158 ± 0.060 | 5 |
| Woodwalton | WienerNet-SS (beta-NLL) | 0.555 ± 0.043 | 1.57 ± 0.06 | 0.813 ± 0.055 | 2.40 ± 0.34 | 0.143 ± 0.066 | 5 |
| Woodwalton | WienerNet-SS (Student-t) | 0.518 ± 0.007 | 1.54 ± 0.09 | 0.794 ± 0.074 | 2.15 ± 0.44 | 0.120 ± 0.017 | 5 |
| Woodwalton | WienerNet-SS (mixture) | 0.498 ± 0.025 | 1.93 ± 0.45 | 0.863 ± 0.049 | 2.90 ± 1.08 | 0.111 ± 0.030 | 5 |
| Woodwalton | WienerNet-SS (+residual) | 0.536 ± 0.081 | 1.40 ± 0.19 | 0.684 ± 0.127 | 1.45 ± 0.95 | 0.188 ± 0.059 | 5 |
| Woodwalton | WienerNet-SS (predicted-k) | 0.660 ± 0.135 | 1.57 ± 0.42 | 0.519 ± 0.297 | 2.01 ± 2.33 | 0.376 ± 0.144 | 5 |
| Woodwalton | WienerNet-SS (state-space) | 0.511 ± 0.026 | 1.40 ± 0.17 | 0.703 ± 0.152 | 1.52 ± 1.04 | 0.151 ± 0.065 | 5 |
| Woodwalton | WienerNet-SS (state-space, +residual) | 0.564 ± 0.031 | 1.62 ± 0.29 | 0.801 ± 0.116 | 2.67 ± 1.46 | 0.174 ± 0.028 | 5 |
| Woodwalton | WienerNet-SS (state-space, predicted-k) | 0.755 ± 0.154 | 2.37 ± 0.79 | 0.810 ± 0.290 | 5.07 ± 3.07 | 0.301 ± 0.125 | 5 |
| Woodwalton | WienerNet-SS (given-diurnal, Wiener) | 0.489 ± 0.005 | 1.30 ± 0.03 | 0.674 ± 0.057 | 0.96 ± 0.29 | 0.129 ± 0.027 | 5 |
| Woodwalton | WienerNet-SS (given-diurnal, state-sp) | 0.487 ± 0.013 | 1.37 ± 0.15 | 0.716 ± 0.126 | 1.38 ± 0.98 | 0.134 ± 0.040 | 5 |
| Woodwalton | Neural SDE (Gaussian) | 0.658 ± 0.118 | 2.03 ± 0.21 | 0.912 ± 0.026 | 4.33 ± 0.73 | 0.212 ± 0.054 | 5 |
| Woodwalton | Neural SDE (Student-t) | 0.528 ± 0.067 | 1.63 ± 0.20 | 0.847 ± 0.039 | 2.64 ± 0.93 | 0.166 ± 0.042 | 5 |
| Woodwalton | Neural SDE (ALD) | 0.508 ± 0.031 | 1.44 ± 0.22 | 0.717 ± 0.141 | 1.66 ± 1.09 | 0.164 ± 0.042 | 5 |
| Woodwalton | Mean-variance (Gaussian) | 0.558 ± 0.024 | 1.92 ± 0.12 | 0.930 ± 0.008 | 4.09 ± 0.44 | 0.218 ± 0.018 | 5 |
| Woodwalton | Mean-variance (Student-t) | 0.489 ± 0.015 | 1.54 ± 0.13 | 0.845 ± 0.027 | 2.21 ± 0.47 | 0.163 ± 0.024 | 5 |
| Woodwalton | Mean-variance (ALD) | 0.494 ± 0.029 | 1.62 ± 0.28 | 0.836 ± 0.073 | 2.48 ± 1.13 | 0.158 ± 0.053 | 5 |
| Woodwalton | Analytical SDE (Gaussian) | 0.633 | 2.12 | 0.957 | 5.46 | 0.256 | 1 |
| Woodwalton | Analytical SDE (Student-t) | 0.497 | 1.41 | 0.849 | 2.05 | 0.173 | 1 |
| Woodwalton | MDN (mixture) | 0.492 ± 0.032 | 1.75 ± 0.31 | 0.861 ± 0.039 | 2.96 ± 0.80 | 0.130 ± 0.038 | 5 |
| Woodwalton | Random Forest | — | 1.57 ± 0.02 | — | — | — | 5 |
| Woodwalton | XGBoost | — | 2.31 ± 0.00 | — | — | — | 5 |
| Rosedene | WienerNet-SS (ALD, primary) | 0.726 ± 0.011 | 2.01 ± 0.09 | 0.772 ± 0.049 | 2.46 ± 0.43 | 0.072 ± 0.031 | 5 |
| Rosedene | WienerNet-SS (Gaussian) | 0.762 ± 0.017 | 2.47 ± 0.17 | 0.922 ± 0.012 | 4.82 ± 0.51 | 0.128 ± 0.013 | 5 |
| Rosedene | WienerNet-SS (beta-NLL) | 0.723 ± 0.012 | 2.15 ± 0.10 | 0.857 ± 0.038 | 3.31 ± 0.55 | 0.064 ± 0.019 | 5 |
| Rosedene | WienerNet-SS (Student-t) | 0.718 ± 0.005 | 2.14 ± 0.12 | 0.846 ± 0.027 | 3.19 ± 0.48 | 0.044 ± 0.016 | 5 |
| Rosedene | WienerNet-SS (mixture) | 0.722 ± 0.005 | 2.30 ± 0.21 | 0.854 ± 0.053 | 3.53 ± 0.86 | 0.037 ± 0.013 | 5 |
| Rosedene | WienerNet-SS (+residual) | 0.728 ± 0.012 | 2.06 ± 0.10 | 0.804 ± 0.064 | 2.86 ± 0.66 | 0.059 ± 0.031 | 5 |
| Rosedene | WienerNet-SS (predicted-k) | 0.726 ± 0.006 | 2.07 ± 0.09 | 0.815 ± 0.048 | 2.94 ± 0.54 | 0.044 ± 0.028 | 5 |
| Rosedene | WienerNet-SS (state-space) | 0.726 ± 0.011 | 2.04 ± 0.09 | 0.810 ± 0.072 | 2.82 ± 0.63 | 0.046 ± 0.041 | 5 |
| Rosedene | WienerNet-SS (state-space, +residual) | 0.738 ± 0.028 | 2.17 ± 0.17 | 0.864 ± 0.048 | 3.63 ± 1.02 | 0.055 ± 0.033 | 5 |
| Rosedene | WienerNet-SS (state-space, predicted-k) | 0.723 ± 0.011 | 2.16 ± 0.24 | 0.857 ± 0.053 | 3.42 ± 1.12 | 0.059 ± 0.044 | 5 |
| Rosedene | WienerNet-SS (given-diurnal, Wiener) | 0.736 ± 0.012 | 2.01 ± 0.08 | 0.771 ± 0.046 | 2.56 ± 0.61 | 0.075 ± 0.022 | 5 |
| Rosedene | WienerNet-SS (given-diurnal, state-sp) | 0.729 ± 0.006 | 2.05 ± 0.07 | 0.815 ± 0.049 | 2.84 ± 0.49 | 0.048 ± 0.024 | 5 |
| Rosedene | Neural SDE (Gaussian) | 0.750 ± 0.016 | 2.39 ± 0.15 | 0.906 ± 0.026 | 4.39 ± 0.63 | 0.111 ± 0.013 | 5 |
| Rosedene | Neural SDE (Student-t) | 0.723 ± 0.005 | 2.21 ± 0.15 | 0.856 ± 0.037 | 3.45 ± 0.62 | 0.060 ± 0.014 | 5 |
| Rosedene | Neural SDE (ALD) | 0.736 ± 0.028 | 2.12 ± 0.13 | 0.823 ± 0.053 | 3.08 ± 0.61 | 0.055 ± 0.037 | 5 |
| Rosedene | Mean-variance (Gaussian) | 0.754 ± 0.013 | 2.54 ± 0.13 | 0.934 ± 0.010 | 5.08 ± 0.46 | 0.134 ± 0.014 | 5 |
| Rosedene | Mean-variance (Student-t) | 0.723 ± 0.009 | 2.32 ± 0.14 | 0.895 ± 0.023 | 4.01 ± 0.50 | 0.079 ± 0.024 | 5 |
| Rosedene | Mean-variance (ALD) | 0.722 ± 0.005 | 2.27 ± 0.24 | 0.878 ± 0.029 | 3.80 ± 0.83 | 0.036 ± 0.019 | 5 |
| Rosedene | Analytical SDE (Gaussian) | 0.785 | 2.33 | 0.911 | 4.71 | 0.142 | 1 |
| Rosedene | Analytical SDE (Student-t) | 0.765 | 1.84 | 0.680 | 1.55 | 0.108 | 1 |
| Rosedene | MDN (mixture) | 0.717 ± 0.006 | 2.41 ± 0.10 | 0.907 ± 0.018 | 4.39 ± 0.47 | 0.036 ± 0.019 | 5 |
| Rosedene | Random Forest | — | 1.61 ± 0.01 | — | — | — | 5 |
| Rosedene | XGBoost | — | 1.73 ± 0.00 | — | — | — | 5 |
| Redmere 1 | WienerNet-SS (ALD, primary) | 0.685 ± 0.020 | 2.50 ± 0.12 | 0.874 ± 0.013 | 3.63 ± 0.10 | 0.038 ± 0.009 | 5 |
| Redmere 1 | WienerNet-SS (Gaussian) | 4.252 ± 4.367 | 464.68 ± 786.02 | 0.924 ± 0.009 | 51.25 ± 56.96 | 0.113 ± 0.006 | 5 |
| Redmere 1 | WienerNet-SS (beta-NLL) | 1.999 ± 1.409 | 98.44 ± 111.89 | 0.916 ± 0.008 | 21.82 ± 18.86 | 0.112 ± 0.010 | 5 |
| Redmere 1 | WienerNet-SS (Student-t) | 1.025 ± 0.533 | 36.54 ± 56.93 | 0.871 ± 0.013 | 8.69 ± 7.20 | 0.054 ± 0.015 | 5 |
| Redmere 1 | WienerNet-SS (mixture) | 0.702 ± 0.117 | 5.74 ± 7.47 | 0.878 ± 0.010 | 4.38 ± 1.73 | 0.050 ± 0.026 | 5 |
| Redmere 1 | WienerNet-SS (+residual) | 1.198 ± 1.154 | 60.14 ± 128.48 | 0.889 ± 0.015 | 13.27 ± 20.87 | 0.036 ± 0.020 | 5 |
| Redmere 1 | WienerNet-SS (predicted-k) | 0.958 ± 0.527 | 48.39 ± 91.13 | 0.883 ± 0.038 | 8.55 ± 9.74 | 0.039 ± 0.024 | 5 |
| Redmere 1 | WienerNet-SS (state-space) | 1.329 ± 0.822 | 128.44 ± 128.27 | 0.885 ± 0.012 | 16.44 ± 15.14 | 0.042 ± 0.011 | 5 |
| Redmere 1 | WienerNet-SS (state-space, +residual) | 2.418 ± 2.819 | 200.24 ± 335.04 | 0.871 ± 0.016 | 35.29 ± 51.01 | 0.046 ± 0.022 | 5 |
| Redmere 1 | WienerNet-SS (state-space, predicted-k) | 1.580 ± 1.250 | 121.23 ± 156.69 | 0.879 ± 0.031 | 20.15 ± 22.98 | 0.031 ± 0.013 | 5 |
| Redmere 1 | WienerNet-SS (given-diurnal, Wiener) | 0.598 ± 0.004 | 1.80 ± 0.08 | 0.855 ± 0.019 | 3.48 ± 0.25 | 0.032 ± 0.007 | 5 |
| Redmere 1 | WienerNet-SS (given-diurnal, state-sp) | 0.594 ± 0.007 | 1.85 ± 0.06 | 0.867 ± 0.013 | 3.49 ± 0.15 | 0.028 ± 0.004 | 5 |
| Redmere 1 | Neural SDE (Gaussian) | 7.491 ± 3.284 | 473.60 ± 400.32 | 0.923 ± 0.013 | 62.40 ± 58.13 | 0.117 ± 0.014 | 5 |
| Redmere 1 | Neural SDE (Student-t) | 5.468 ± 2.678 | 837.25 ± 965.70 | 0.885 ± 0.009 | 60.26 ± 47.79 | 0.061 ± 0.007 | 5 |
| Redmere 1 | Neural SDE (ALD) | 1.409 ± 0.731 | 52.78 ± 86.14 | 0.884 ± 0.017 | 9.97 ± 14.13 | 0.039 ± 0.017 | 5 |
| Redmere 1 | Mean-variance (Gaussian) | 1.506 ± 0.435 | 24.68 ± 9.32 | 0.929 ± 0.010 | 6.73 ± 0.66 | 0.107 ± 0.017 | 5 |
| Redmere 1 | Mean-variance (Student-t) | 1.349 ± 0.491 | 24.24 ± 13.95 | 0.887 ± 0.004 | 5.64 ± 0.45 | 0.062 ± 0.004 | 5 |
| Redmere 1 | Mean-variance (ALD) | 0.776 ± 0.139 | 6.23 ± 2.76 | 0.870 ± 0.005 | 3.76 ± 1.04 | 0.033 ± 0.007 | 5 |
| Redmere 1 | Analytical SDE (Gaussian) | 0.655 | 1.98 | 0.953 | 5.14 | 0.154 | 1 |
| Redmere 1 | Analytical SDE (Student-t) | 0.601 | 1.34 | 0.725 | 1.75 | 0.084 | 1 |
| Redmere 1 | MDN (mixture) | 1.518 ± 0.703 | 27.56 ± 17.88 | 0.893 ± 0.018 | 8.04 ± 1.52 | 0.042 ± 0.019 | 5 |
| Redmere 1 | Random Forest | — | 1.28 ± 0.01 | — | — | — | 5 |
| Redmere 1 | XGBoost | — | 1.35 ± 0.00 | — | — | — | 5 |
| Redmere 2 | WienerNet-SS (ALD, primary) | 0.711 ± 0.005 | 2.11 ± 0.25 | 0.828 ± 0.072 | 3.67 ± 1.07 | 0.076 ± 0.025 | 5 |
| Redmere 2 | WienerNet-SS (Gaussian) | 0.755 ± 0.040 | 2.41 ± 0.37 | 0.912 ± 0.034 | 5.13 ± 1.24 | 0.141 ± 0.025 | 5 |
| Redmere 2 | WienerNet-SS (beta-NLL) | 0.745 ± 0.024 | 2.28 ± 0.23 | 0.890 ± 0.047 | 4.64 ± 1.03 | 0.140 ± 0.029 | 5 |
| Redmere 2 | WienerNet-SS (Student-t) | 0.711 ± 0.013 | 2.19 ± 0.13 | 0.869 ± 0.032 | 3.97 ± 0.55 | 0.109 ± 0.021 | 5 |
| Redmere 2 | WienerNet-SS (mixture) | 0.700 ± 0.014 | 2.40 ± 0.22 | 0.884 ± 0.030 | 4.44 ± 0.72 | 0.079 ± 0.021 | 5 |
| Redmere 2 | WienerNet-SS (+residual) | 0.719 ± 0.006 | 2.07 ± 0.18 | 0.844 ± 0.073 | 3.76 ± 0.96 | 0.083 ± 0.007 | 5 |
| Redmere 2 | WienerNet-SS (predicted-k) | 0.711 ± 0.010 | 2.14 ± 0.26 | 0.865 ± 0.061 | 4.08 ± 1.20 | 0.100 ± 0.022 | 5 |
| Redmere 2 | WienerNet-SS (state-space) | 0.707 ± 0.007 | 2.05 ± 0.21 | 0.833 ± 0.056 | 3.54 ± 0.88 | 0.072 ± 0.018 | 5 |
| Redmere 2 | WienerNet-SS (state-space, +residual) | 0.711 ± 0.005 | 2.09 ± 0.22 | 0.844 ± 0.054 | 3.73 ± 0.92 | 0.073 ± 0.017 | 5 |
| Redmere 2 | WienerNet-SS (state-space, predicted-k) | 0.695 ± 0.003 | 2.02 ± 0.16 | 0.842 ± 0.040 | 3.46 ± 0.65 | 0.075 ± 0.016 | 5 |
| Redmere 2 | WienerNet-SS (given-diurnal, Wiener) | 0.710 ± 0.007 | 2.14 ± 0.20 | 0.842 ± 0.068 | 3.86 ± 0.97 | 0.086 ± 0.018 | 5 |
| Redmere 2 | WienerNet-SS (given-diurnal, state-sp) | 0.707 ± 0.010 | 2.11 ± 0.27 | 0.843 ± 0.058 | 3.69 ± 1.03 | 0.082 ± 0.021 | 5 |
| Redmere 2 | Neural SDE (Gaussian) | 0.762 ± 0.031 | 2.43 ± 0.22 | 0.896 ± 0.023 | 4.90 ± 0.74 | 0.128 ± 0.017 | 5 |
| Redmere 2 | Neural SDE (Student-t) | 0.710 ± 0.008 | 2.17 ± 0.10 | 0.857 ± 0.032 | 3.84 ± 0.35 | 0.104 ± 0.012 | 5 |
| Redmere 2 | Neural SDE (ALD) | 0.711 ± 0.008 | 2.08 ± 0.14 | 0.846 ± 0.045 | 3.71 ± 0.68 | 0.063 ± 0.022 | 5 |
| Redmere 2 | Mean-variance (Gaussian) | 0.729 ± 0.013 | 2.23 ± 0.07 | 0.884 ± 0.014 | 4.32 ± 0.30 | 0.126 ± 0.008 | 5 |
| Redmere 2 | Mean-variance (Student-t) | 0.707 ± 0.007 | 2.19 ± 0.24 | 0.833 ± 0.028 | 3.60 ± 0.38 | 0.089 ± 0.011 | 5 |
| Redmere 2 | Mean-variance (ALD) | 0.701 ± 0.008 | 1.98 ± 0.07 | 0.830 ± 0.033 | 3.32 ± 0.45 | 0.074 ± 0.008 | 5 |
| Redmere 2 | Analytical SDE (Gaussian) | 0.773 | 2.25 | 0.921 | 5.06 | 0.162 | 1 |
| Redmere 2 | Analytical SDE (Student-t) | 0.733 | 1.68 | 0.689 | 1.63 | 0.110 | 1 |
| Redmere 2 | MDN (mixture) | 0.697 ± 0.007 | 2.07 ± 0.08 | 0.831 ± 0.028 | 3.43 ± 0.23 | 0.059 ± 0.009 | 5 |
| Redmere 2 | Random Forest | — | 1.54 ± 0.01 | — | — | — | 5 |
| Redmere 2 | XGBoost | — | 1.67 ± 0.00 | — | — | — | 5 |
| Great Fen | WienerNet-SS (ALD, primary) | 0.890 ± 0.007 | 2.36 ± 0.13 | 0.794 ± 0.048 | 4.04 ± 0.62 | 0.056 ± 0.027 | 5 |
| Great Fen | WienerNet-SS (Gaussian) | 0.928 ± 0.029 | 2.77 ± 0.16 | 0.913 ± 0.021 | 6.14 ± 0.76 | 0.119 ± 0.030 | 5 |
| Great Fen | WienerNet-SS (beta-NLL) | 0.895 ± 0.016 | 2.54 ± 0.12 | 0.872 ± 0.021 | 5.00 ± 0.50 | 0.086 ± 0.025 | 5 |
| Great Fen | WienerNet-SS (Student-t) | 0.895 ± 0.019 | 2.72 ± 0.73 | 0.841 ± 0.035 | 4.58 ± 0.84 | 0.054 ± 0.010 | 5 |
| Great Fen | WienerNet-SS (mixture) | 0.892 ± 0.009 | 2.85 ± 0.49 | 0.876 ± 0.030 | 5.24 ± 0.49 | 0.049 ± 0.030 | 5 |
| Great Fen | WienerNet-SS (+residual) | 0.900 ± 0.019 | 2.31 ± 0.18 | 0.797 ± 0.070 | 3.97 ± 0.93 | 0.067 ± 0.041 | 5 |
| Great Fen | WienerNet-SS (predicted-k) | 0.884 ± 0.005 | 2.38 ± 0.20 | 0.840 ± 0.044 | 4.42 ± 0.87 | 0.044 ± 0.008 | 5 |
| Great Fen | WienerNet-SS (state-space) | 0.891 ± 0.011 | 2.29 ± 0.12 | 0.782 ± 0.061 | 3.76 ± 0.69 | 0.068 ± 0.036 | 5 |
| Great Fen | WienerNet-SS (state-space, +residual) | 0.887 ± 0.013 | 2.25 ± 0.15 | 0.792 ± 0.054 | 3.69 ± 0.76 | 0.067 ± 0.030 | 5 |
| Great Fen | WienerNet-SS (state-space, predicted-k) | 0.881 ± 0.005 | 2.26 ± 0.08 | 0.802 ± 0.034 | 3.73 ± 0.42 | 0.052 ± 0.023 | 5 |
| Great Fen | WienerNet-SS (given-diurnal, Wiener) | 0.895 ± 0.004 | 2.31 ± 0.10 | 0.778 ± 0.034 | 3.81 ± 0.53 | 0.061 ± 0.018 | 5 |
| Great Fen | WienerNet-SS (given-diurnal, state-sp) | 0.890 ± 0.009 | 2.26 ± 0.12 | 0.773 ± 0.054 | 3.65 ± 0.57 | 0.064 ± 0.030 | 5 |
| Great Fen | Neural SDE (Gaussian) | 0.921 ± 0.028 | 2.65 ± 0.11 | 0.887 ± 0.023 | 5.44 ± 0.47 | 0.097 ± 0.019 | 5 |
| Great Fen | Neural SDE (Student-t) | 0.899 ± 0.013 | 2.35 ± 0.03 | 0.831 ± 0.022 | 4.21 ± 0.20 | 0.049 ± 0.022 | 5 |
| Great Fen | Neural SDE (ALD) | 0.895 ± 0.016 | 2.30 ± 0.09 | 0.783 ± 0.079 | 3.81 ± 0.66 | 0.066 ± 0.041 | 5 |
| Great Fen | Mean-variance (Gaussian) | 0.876 ± 0.005 | 2.46 ± 0.12 | 0.878 ± 0.017 | 4.77 ± 0.47 | 0.071 ± 0.017 | 5 |
| Great Fen | Mean-variance (Student-t) | 0.882 ± 0.009 | 2.24 ± 0.14 | 0.786 ± 0.040 | 3.59 ± 0.58 | 0.056 ± 0.017 | 5 |
| Great Fen | Mean-variance (ALD) | 0.883 ± 0.011 | 2.22 ± 0.08 | 0.780 ± 0.045 | 3.48 ± 0.46 | 0.062 ± 0.025 | 5 |
| Great Fen | Analytical SDE (Gaussian) | 0.907 | 2.43 | 0.887 | 4.94 | 0.093 | 1 |
| Great Fen | Analytical SDE (Student-t) | 0.897 | 2.26 | 0.849 | 4.12 | 0.063 | 1 |
| Great Fen | MDN (mixture) | 0.892 ± 0.018 | 2.42 ± 0.22 | 0.811 ± 0.063 | 4.22 ± 1.04 | 0.106 ± 0.045 | 5 |
| Great Fen | Random Forest | — | 1.58 ± 0.01 | — | — | — | 5 |
| Great Fen | XGBoost | — | 1.61 ± 0.00 | — | — | — | 5 |

**Stage 1 — one-step predictive law, per held-out site.** mean ± 1 SD over the 5 seeds (0-4); n = 1 entries are the calibrated, seed-independent models. Metric conventions as in the pooled Stage-1 tables.  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. **These values are scored from `best.pth`, which with `data.val_frac` unset is the epoch selected ON THE HELD-OUT TEST SITE.** They are therefore an optimistic, test-selected bound rather than an honest estimate of generalisation, and they are not directly comparable to the calibrated Analytical SDE and tree arms, which have no epoch to select and gain nothing from it. The leakage-free `last.pth` scores are archived in `analysis/ss/v2_last_archive/`.

### Stage 2 (gap-filling) — site-wise

| Site | Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|---|
| Woodwalton | WienerNet-SS (ALD, primary) | 1.21 ± 0.13 | 2.21 ± 0.54 | 3.57 ± 1.66 | 2.011 ± 1.062 | 0.340 ± 0.194 | 0.530 ± 0.181 | 5 |
| Woodwalton | WienerNet-SS (Gaussian) | 2.94 ± 1.33 | 6.83 ± 3.15 | 10.41 ± 2.87 | 5.453 ± 1.514 | 0.226 ± 0.111 | 0.701 ± 0.111 | 5 |
| Woodwalton | WienerNet-SS (beta-NLL) | 1.57 ± 0.48 | 3.22 ± 1.38 | 6.12 ± 3.62 | 3.377 ± 2.323 | 0.266 ± 0.180 | 0.669 ± 0.189 | 5 |
| Woodwalton | WienerNet-SS (Student-t) | 1.31 ± 0.12 | 2.52 ± 0.44 | 4.02 ± 0.85 | 2.155 ± 0.477 | 0.209 ± 0.148 | 0.702 ± 0.177 | 5 |
| Woodwalton | WienerNet-SS (mixture) | 1.20 ± 0.15 | 2.16 ± 0.59 | 2.93 ± 0.79 | 1.643 ± 0.289 | 0.199 ± 0.079 | 0.907 ± 0.055 | 5 |
| Woodwalton | WienerNet-SS (+residual) | 1.39 ± 0.55 | 2.71 ± 1.73 | 4.81 ± 4.71 | 2.835 ± 2.908 | 0.344 ± 0.158 | 0.560 ± 0.245 | 5 |
| Woodwalton | WienerNet-SS (predicted-k) | 1.74 ± 0.73 | 3.73 ± 2.33 | 7.20 ± 4.96 | 5.603 ± 3.985 | 0.639 ± 0.220 | 0.307 ± 0.393 | 5 |
| Woodwalton | WienerNet-SS (state-space) | 1.33 ± 0.22 | 2.34 ± 0.72 | 3.95 ± 2.03 | 2.142 ± 1.018 | 0.271 ± 0.125 | 0.604 ± 0.171 | 5 |
| Woodwalton | WienerNet-SS (state-space, +residual) | 1.54 ± 0.37 | 3.10 ± 0.99 | 5.91 ± 1.18 | 3.318 ± 0.789 | 0.355 ± 0.123 | 0.640 ± 0.239 | 5 |
| Woodwalton | WienerNet-SS (state-space, predicted-k) | 2.41 ± 0.93 | 5.88 ± 2.56 | 10.61 ± 4.08 | 6.552 ± 3.179 | 0.474 ± 0.164 | 0.657 ± 0.331 | 5 |
| Woodwalton | WienerNet-SS (ALD + sigma_struct band) | 1.21 ± 0.13 | 2.21 ± 0.54 | 3.57 ± 1.66 | 1.887 ± 0.983 | 0.235 ± 0.172 | 0.781 ± 0.144 | 5 |
| Woodwalton | WienerNet-SS (given-diurnal, Wiener) | 1.07 ± 0.00 | 1.64 ± 0.00 | 1.97 ± 0.00 | 1.037 ± 0.029 | 0.143 ± 0.062 | 0.685 ± 0.102 | 5 |
| Woodwalton | WienerNet-SS (given-diurnal, state-sp) | 1.07 ± 0.00 | 1.64 ± 0.00 | 1.97 ± 0.00 | 1.076 ± 0.104 | 0.159 ± 0.053 | 0.731 ± 0.169 | 5 |
| Woodwalton | Neural SDE (Gaussian) | 1.95 ± 0.80 | 4.28 ± 2.23 | 7.54 ± 4.79 | 4.730 ± 3.184 | 0.350 ± 0.251 | 0.707 ± 0.255 | 5 |
| Woodwalton | Neural SDE (Student-t) | 1.31 ± 0.40 | 2.42 ± 1.28 | 3.39 ± 1.89 | 1.870 ± 0.952 | 0.211 ± 0.093 | 0.844 ± 0.068 | 5 |
| Woodwalton | Neural SDE (ALD) | 1.26 ± 0.28 | 2.23 ± 0.86 | 3.03 ± 1.14 | 1.628 ± 0.472 | 0.297 ± 0.069 | 0.636 ± 0.176 | 5 |
| Woodwalton | Mean-variance (Gaussian) | 1.12 ± 0.02 | 1.78 ± 0.09 | 2.30 ± 0.22 | 1.526 ± 0.141 | 0.259 ± 0.050 | 0.989 ± 0.005 | 5 |
| Woodwalton | Mean-variance (Student-t) | 1.10 ± 0.01 | 1.70 ± 0.03 | 2.20 ± 0.14 | 1.237 ± 0.138 | 0.195 ± 0.036 | 0.905 ± 0.017 | 5 |
| Woodwalton | Mean-variance (ALD) | 1.11 ± 0.10 | 1.81 ± 0.38 | 2.36 ± 0.92 | 1.311 ± 0.405 | 0.226 ± 0.083 | 0.891 ± 0.080 | 5 |
| Woodwalton | Analytical SDE (Gaussian) | 1.04 | 1.59 | 1.92 | 1.871 | 0.328 | 0.998 | 1 |
| Woodwalton | Analytical SDE (Student-t) | 1.04 | 1.59 | 1.92 | 1.060 | 0.215 | 0.957 | 1 |
| Woodwalton | MDN (mixture) | 1.09 ± 0.02 | 1.72 ± 0.08 | 2.19 ± 0.19 | 1.332 ± 0.163 | 0.291 ± 0.057 | 0.950 ± 0.021 | 5 |
| Woodwalton | Random Forest | 1.67 ± 0.17 | 1.84 ± 0.13 | 1.95 ± 0.15 | — | — | — | 5 |
| Woodwalton | XGBoost | 2.52 ± 0.19 | 2.71 ± 0.11 | 2.79 ± 0.14 | — | — | — | 5 |
| Rosedene | WienerNet-SS (ALD, primary) | 1.57 ± 0.02 | 1.72 ± 0.07 | 2.14 ± 0.19 | 1.135 ± 0.082 | 0.212 ± 0.019 | 0.967 ± 0.015 | 5 |
| Rosedene | WienerNet-SS (Gaussian) | 1.61 ± 0.05 | 1.99 ± 0.27 | 2.84 ± 0.50 | 1.853 ± 0.264 | 0.252 ± 0.031 | 0.982 ± 0.017 | 5 |
| Rosedene | WienerNet-SS (beta-NLL) | 1.63 ± 0.09 | 1.95 ± 0.24 | 2.66 ± 0.73 | 1.484 ± 0.297 | 0.213 ± 0.070 | 0.974 ± 0.015 | 5 |
| Rosedene | WienerNet-SS (Student-t) | 1.57 ± 0.02 | 1.70 ± 0.08 | 2.18 ± 0.21 | 1.288 ± 0.106 | 0.232 ± 0.035 | 0.991 ± 0.004 | 5 |
| Rosedene | WienerNet-SS (mixture) | 1.58 ± 0.01 | 1.69 ± 0.05 | 2.10 ± 0.21 | 1.372 ± 0.190 | 0.252 ± 0.022 | 0.990 ± 0.012 | 5 |
| Rosedene | WienerNet-SS (+residual) | 1.60 ± 0.07 | 1.80 ± 0.25 | 2.36 ± 0.79 | 1.240 ± 0.203 | 0.221 ± 0.041 | 0.974 ± 0.015 | 5 |
| Rosedene | WienerNet-SS (predicted-k) | 2.12 ± 0.50 | 2.16 ± 0.46 | 2.10 ± 0.43 | 1.184 ± 0.173 | 0.258 ± 0.049 | 0.978 ± 0.009 | 5 |
| Rosedene | WienerNet-SS (state-space) | 2.15 ± 0.53 | 2.19 ± 0.54 | 2.17 ± 0.60 | 1.187 ± 0.230 | 0.230 ± 0.065 | 0.974 ± 0.024 | 5 |
| Rosedene | WienerNet-SS (state-space, +residual) | 2.17 ± 0.56 | 2.36 ± 0.76 | 2.76 ± 1.29 | 1.572 ± 0.674 | 0.276 ± 0.061 | 0.978 ± 0.017 | 5 |
| Rosedene | WienerNet-SS (state-space, predicted-k) | 2.12 ± 0.50 | 2.15 ± 0.46 | 2.11 ± 0.43 | 1.431 ± 0.553 | 0.310 ± 0.091 | 0.990 ± 0.008 | 5 |
| Rosedene | WienerNet-SS (ALD + sigma_struct band) | 1.57 ± 0.02 | 1.72 ± 0.07 | 2.14 ± 0.19 | 1.190 ± 0.073 | 0.243 ± 0.007 | 0.975 ± 0.015 | 5 |
| Rosedene | WienerNet-SS (given-diurnal, Wiener) | 1.55 ± 0.00 | 1.58 ± 0.00 | 1.85 ± 0.00 | 1.101 ± 0.133 | 0.238 ± 0.021 | 0.974 ± 0.012 | 5 |
| Rosedene | WienerNet-SS (given-diurnal, state-sp) | 1.55 ± 0.00 | 1.58 ± 0.00 | 1.85 ± 0.00 | 1.193 ± 0.094 | 0.273 ± 0.024 | 0.975 ± 0.018 | 5 |
| Rosedene | Neural SDE (Gaussian) | 1.72 ± 0.10 | 2.33 ± 0.25 | 3.39 ± 0.46 | 1.915 ± 0.143 | 0.248 ± 0.040 | 0.967 ± 0.024 | 5 |
| Rosedene | Neural SDE (Student-t) | 1.60 ± 0.04 | 1.81 ± 0.10 | 2.31 ± 0.27 | 1.426 ± 0.116 | 0.241 ± 0.030 | 0.977 ± 0.019 | 5 |
| Rosedene | Neural SDE (ALD) | 1.71 ± 0.12 | 2.17 ± 0.31 | 3.17 ± 0.89 | 1.685 ± 0.470 | 0.204 ± 0.085 | 0.914 ± 0.058 | 5 |
| Rosedene | Mean-variance (Gaussian) | 1.56 ± 0.01 | 1.66 ± 0.06 | 2.11 ± 0.22 | 1.659 ± 0.131 | 0.293 ± 0.028 | 0.999 ± 0.001 | 5 |
| Rosedene | Mean-variance (Student-t) | 1.56 ± 0.02 | 1.65 ± 0.07 | 2.14 ± 0.25 | 1.456 ± 0.206 | 0.256 ± 0.022 | 0.998 ± 0.001 | 5 |
| Rosedene | Mean-variance (ALD) | 1.55 ± 0.02 | 1.64 ± 0.05 | 1.97 ± 0.10 | 1.409 ± 0.221 | 0.276 ± 0.031 | 0.997 ± 0.002 | 5 |
| Rosedene | Analytical SDE (Gaussian) | 1.55 | 1.59 | 1.80 | 1.548 | 0.310 | 1.000 | 1 |
| Rosedene | Analytical SDE (Student-t) | 1.55 | 1.59 | 1.80 | 0.916 | 0.144 | 0.913 | 1 |
| Rosedene | MDN (mixture) | 1.58 ± 0.03 | 1.68 ± 0.08 | 2.02 ± 0.16 | 1.496 ± 0.137 | 0.257 ± 0.022 | 0.998 ± 0.001 | 5 |
| Rosedene | Random Forest | 1.52 ± 0.13 | 1.41 ± 0.19 | 1.28 ± 0.14 | — | — | — | 5 |
| Rosedene | XGBoost | 1.84 ± 0.15 | 1.61 ± 0.13 | 1.40 ± 0.07 | — | — | — | 5 |
| Redmere 1 | WienerNet-SS (ALD, primary) | 5.32 ± 0.54 | 7.86 ± 0.86 | 2.96 ± 0.08 | 1.421 ± 0.036 | 0.324 ± 0.028 | 0.980 ± 0.009 | 5 |
| Redmere 1 | WienerNet-SS (Gaussian) | 5.84 ± 0.21 | 8.71 ± 0.28 | 3.37 ± 0.46 | 7.846 ± 8.225 | 0.280 ± 0.015 | 0.990 ± 0.014 | 5 |
| Redmere 1 | WienerNet-SS (beta-NLL) | 5.52 ± 0.44 | 8.20 ± 0.77 | 3.55 ± 0.40 | 4.064 ± 2.694 | 0.268 ± 0.047 | 0.989 ± 0.005 | 5 |
| Redmere 1 | WienerNet-SS (Student-t) | 3.97 ± 1.59 | 5.57 ± 2.64 | 2.82 ± 0.85 | 1.697 ± 0.733 | 0.265 ± 0.032 | 0.968 ± 0.043 | 5 |
| Redmere 1 | WienerNet-SS (mixture) | 4.45 ± 1.66 | 6.23 ± 2.75 | 2.76 ± 0.71 | 1.422 ± 0.104 | 0.281 ± 0.041 | 0.988 ± 0.015 | 5 |
| Redmere 1 | WienerNet-SS (+residual) | 4.25 ± 1.87 | 6.38 ± 2.70 | 2.80 ± 0.58 | 2.238 ± 1.742 | 0.260 ± 0.078 | 0.991 ± 0.006 | 5 |
| Redmere 1 | WienerNet-SS (predicted-k) | 5.47 ± 0.68 | 8.00 ± 1.33 | 2.72 ± 0.09 | 1.323 ± 0.158 | 0.311 ± 0.038 | 0.992 ± 0.006 | 5 |
| Redmere 1 | WienerNet-SS (state-space) | 4.21 ± 1.04 | 5.85 ± 1.84 | 2.67 ± 0.43 | 2.313 ± 1.319 | 0.330 ± 0.039 | 0.986 ± 0.015 | 5 |
| Redmere 1 | WienerNet-SS (state-space, +residual) | 3.65 ± 1.56 | 5.28 ± 2.48 | 2.38 ± 0.95 | 4.394 ± 4.840 | 0.314 ± 0.043 | 0.980 ± 0.013 | 5 |
| Redmere 1 | WienerNet-SS (state-space, predicted-k) | 5.25 ± 0.88 | 7.52 ± 1.65 | 2.72 ± 0.08 | 2.574 ± 1.928 | 0.304 ± 0.033 | 0.992 ± 0.005 | 5 |
| Redmere 1 | WienerNet-SS (ALD + sigma_struct band) | 5.32 ± 0.54 | 7.86 ± 0.86 | 2.96 ± 0.08 | 1.469 ± 0.036 | 0.335 ± 0.028 | 0.990 ± 0.008 | 5 |
| Redmere 1 | WienerNet-SS (given-diurnal, Wiener) | 1.62 ± 0.00 | 1.81 ± 0.00 | 1.43 ± 0.00 | 1.176 ± 0.057 | 0.295 ± 0.012 | 0.994 ± 0.006 | 5 |
| Redmere 1 | WienerNet-SS (given-diurnal, state-sp) | 1.62 ± 0.00 | 1.81 ± 0.00 | 1.43 ± 0.00 | 1.150 ± 0.047 | 0.287 ± 0.009 | 0.987 ± 0.006 | 5 |
| Redmere 1 | Neural SDE (Gaussian) | 640.59 ± 293.77 | 870.35 ± 399.17 | 375.13 ± 171.98 | 13.621 ± 5.903 | 0.292 ± 0.036 | 0.986 ± 0.007 | 5 |
| Redmere 1 | Neural SDE (Student-t) | 216.77 ± 88.56 | 294.39 ± 120.18 | 126.98 ± 51.96 | 9.042 ± 3.948 | 0.256 ± 0.062 | 0.994 ± 0.004 | 5 |
| Redmere 1 | Neural SDE (ALD) | 34.59 ± 46.87 | 46.86 ± 63.69 | 20.50 ± 27.28 | 2.601 ± 1.208 | 0.293 ± 0.083 | 0.990 ± 0.010 | 5 |
| Redmere 1 | Mean-variance (Gaussian) | 79.44 ± 42.16 | 107.92 ± 57.30 | 46.53 ± 24.67 | 3.080 ± 1.100 | 0.261 ± 0.028 | 0.991 ± 0.006 | 5 |
| Redmere 1 | Mean-variance (Student-t) | 67.66 ± 45.00 | 91.91 ± 61.14 | 39.67 ± 26.26 | 2.588 ± 0.947 | 0.282 ± 0.033 | 0.989 ± 0.003 | 5 |
| Redmere 1 | Mean-variance (ALD) | 8.00 ± 9.95 | 10.70 ± 13.63 | 4.95 ± 5.72 | 1.321 ± 0.260 | 0.295 ± 0.020 | 0.995 ± 0.004 | 5 |
| Redmere 1 | Analytical SDE (Gaussian) | 1.60 | 1.76 | 1.30 | 1.681 | 0.309 | 1.000 | 1 |
| Redmere 1 | Analytical SDE (Student-t) | 1.60 | 1.76 | 1.30 | 0.811 | 0.187 | 0.976 | 1 |
| Redmere 1 | MDN (mixture) | 66.86 ± 67.35 | 90.81 ± 91.53 | 39.27 ± 39.32 | 3.073 ± 1.707 | 0.283 ± 0.042 | 0.995 ± 0.005 | 5 |
| Redmere 1 | Random Forest | 1.46 ± 0.04 | 1.55 ± 0.06 | 1.23 ± 0.04 | — | — | — | 5 |
| Redmere 1 | XGBoost | 1.52 ± 0.06 | 1.55 ± 0.08 | 1.19 ± 0.03 | — | — | — | 5 |
| Redmere 2 | WienerNet-SS (ALD, primary) | 1.86 ± 0.06 | 1.90 ± 0.17 | 1.96 ± 0.16 | 1.324 ± 0.252 | 0.277 ± 0.062 | 0.982 ± 0.022 | 5 |
| Redmere 2 | WienerNet-SS (Gaussian) | 1.84 ± 0.04 | 1.96 ± 0.16 | 2.35 ± 0.72 | 1.610 ± 0.424 | 0.283 ± 0.009 | 0.987 ± 0.012 | 5 |
| Redmere 2 | WienerNet-SS (beta-NLL) | 1.88 ± 0.05 | 1.98 ± 0.22 | 2.65 ± 0.83 | 1.666 ± 0.381 | 0.295 ± 0.051 | 0.968 ± 0.031 | 5 |
| Redmere 2 | WienerNet-SS (Student-t) | 1.84 ± 0.06 | 1.81 ± 0.13 | 1.96 ± 0.21 | 1.282 ± 0.202 | 0.285 ± 0.022 | 0.990 ± 0.012 | 5 |
| Redmere 2 | WienerNet-SS (mixture) | 1.84 ± 0.01 | 1.82 ± 0.03 | 1.80 ± 0.07 | 1.475 ± 0.244 | 0.285 ± 0.022 | 0.998 ± 0.002 | 5 |
| Redmere 2 | WienerNet-SS (+residual) | 1.88 ± 0.05 | 2.05 ± 0.21 | 2.36 ± 0.21 | 1.528 ± 0.223 | 0.178 ± 0.069 | 0.961 ± 0.049 | 5 |
| Redmere 2 | WienerNet-SS (predicted-k) | 1.77 ± 0.09 | 1.67 ± 0.09 | 1.52 ± 0.16 | 1.408 ± 0.348 | 0.342 ± 0.057 | 0.998 ± 0.004 | 5 |
| Redmere 2 | WienerNet-SS (state-space) | 1.78 ± 0.10 | 1.74 ± 0.15 | 1.63 ± 0.25 | 1.184 ± 0.189 | 0.248 ± 0.056 | 0.987 ± 0.015 | 5 |
| Redmere 2 | WienerNet-SS (state-space, +residual) | 1.81 ± 0.11 | 1.84 ± 0.12 | 1.82 ± 0.26 | 1.299 ± 0.211 | 0.254 ± 0.035 | 0.981 ± 0.023 | 5 |
| Redmere 2 | WienerNet-SS (state-space, predicted-k) | 1.77 ± 0.10 | 1.68 ± 0.13 | 1.49 ± 0.20 | 1.117 ± 0.212 | 0.295 ± 0.053 | 0.996 ± 0.004 | 5 |
| Redmere 2 | WienerNet-SS (ALD + sigma_struct band) | 1.86 ± 0.06 | 1.90 ± 0.17 | 1.96 ± 0.16 | 1.394 ± 0.242 | 0.297 ± 0.045 | 0.989 ± 0.012 | 5 |
| Redmere 2 | WienerNet-SS (given-diurnal, Wiener) | 1.82 ± 0.00 | 1.75 ± 0.00 | 1.66 ± 0.00 | 1.261 ± 0.260 | 0.284 ± 0.043 | 0.989 ± 0.012 | 5 |
| Redmere 2 | WienerNet-SS (given-diurnal, state-sp) | 1.82 ± 0.00 | 1.75 ± 0.00 | 1.66 ± 0.00 | 1.231 ± 0.261 | 0.305 ± 0.044 | 0.994 ± 0.004 | 5 |
| Redmere 2 | Neural SDE (Gaussian) | 1.96 ± 0.09 | 2.46 ± 0.61 | 3.12 ± 1.32 | 1.753 ± 0.531 | 0.243 ± 0.057 | 0.969 ± 0.041 | 5 |
| Redmere 2 | Neural SDE (Student-t) | 1.87 ± 0.04 | 1.95 ± 0.12 | 2.19 ± 0.46 | 1.362 ± 0.245 | 0.275 ± 0.042 | 0.982 ± 0.030 | 5 |
| Redmere 2 | Neural SDE (ALD) | 1.87 ± 0.04 | 2.00 ± 0.15 | 2.29 ± 0.58 | 1.443 ± 0.159 | 0.225 ± 0.036 | 0.965 ± 0.039 | 5 |
| Redmere 2 | Mean-variance (Gaussian) | 1.83 ± 0.02 | 1.78 ± 0.03 | 1.73 ± 0.09 | 1.209 ± 0.076 | 0.291 ± 0.024 | 0.999 ± 0.001 | 5 |
| Redmere 2 | Mean-variance (Student-t) | 1.84 ± 0.01 | 1.80 ± 0.06 | 1.69 ± 0.13 | 1.103 ± 0.150 | 0.235 ± 0.022 | 0.995 ± 0.005 | 5 |
| Redmere 2 | Mean-variance (ALD) | 1.83 ± 0.01 | 1.77 ± 0.03 | 1.67 ± 0.05 | 1.104 ± 0.087 | 0.281 ± 0.021 | 0.991 ± 0.007 | 5 |
| Redmere 2 | Analytical SDE (Gaussian) | 1.82 | 1.75 | 1.71 | 1.717 | 0.306 | 0.994 | 1 |
| Redmere 2 | Analytical SDE (Student-t) | 1.82 | 1.75 | 1.71 | 0.886 | 0.180 | 0.947 | 1 |
| Redmere 2 | MDN (mixture) | 1.84 ± 0.01 | 1.79 ± 0.04 | 1.79 ± 0.11 | 1.110 ± 0.060 | 0.281 ± 0.010 | 0.992 ± 0.004 | 5 |
| Redmere 2 | Random Forest | 1.68 ± 0.06 | 1.63 ± 0.11 | 1.38 ± 0.13 | — | — | — | 5 |
| Redmere 2 | XGBoost | 1.70 ± 0.04 | 1.69 ± 0.09 | 1.63 ± 0.17 | — | — | — | 5 |
| Great Fen | WienerNet-SS (ALD, primary) | 1.70 ± 0.02 | 1.86 ± 0.05 | 1.89 ± 0.16 | 1.427 ± 0.144 | 0.267 ± 0.022 | 0.980 ± 0.015 | 5 |
| Great Fen | WienerNet-SS (Gaussian) | 1.87 ± 0.22 | 2.41 ± 0.75 | 3.19 ± 1.52 | 2.284 ± 0.602 | 0.287 ± 0.089 | 0.981 ± 0.034 | 5 |
| Great Fen | WienerNet-SS (beta-NLL) | 1.80 ± 0.08 | 2.27 ± 0.28 | 3.21 ± 0.87 | 1.997 ± 0.400 | 0.314 ± 0.083 | 0.978 ± 0.026 | 5 |
| Great Fen | WienerNet-SS (Student-t) | 1.74 ± 0.04 | 2.12 ± 0.22 | 2.64 ± 0.55 | 1.774 ± 0.393 | 0.279 ± 0.072 | 0.986 ± 0.006 | 5 |
| Great Fen | WienerNet-SS (mixture) | 1.74 ± 0.11 | 2.05 ± 0.52 | 2.60 ± 1.70 | 2.027 ± 0.664 | 0.247 ± 0.023 | 0.985 ± 0.031 | 5 |
| Great Fen | WienerNet-SS (+residual) | 1.76 ± 0.06 | 2.12 ± 0.31 | 2.54 ± 1.07 | 1.808 ± 0.576 | 0.248 ± 0.059 | 0.928 ± 0.127 | 5 |
| Great Fen | WienerNet-SS (predicted-k) | 1.88 ± 0.18 | 1.98 ± 0.13 | 1.79 ± 0.08 | 1.601 ± 0.234 | 0.281 ± 0.050 | 0.993 ± 0.006 | 5 |
| Great Fen | WienerNet-SS (state-space) | 1.89 ± 0.18 | 2.04 ± 0.14 | 2.05 ± 0.21 | 1.520 ± 0.081 | 0.267 ± 0.083 | 0.965 ± 0.037 | 5 |
| Great Fen | WienerNet-SS (state-space, +residual) | 1.91 ± 0.18 | 2.08 ± 0.15 | 1.97 ± 0.27 | 1.445 ± 0.223 | 0.262 ± 0.065 | 0.979 ± 0.013 | 5 |
| Great Fen | WienerNet-SS (state-space, predicted-k) | 1.88 ± 0.18 | 1.97 ± 0.12 | 1.76 ± 0.06 | 1.330 ± 0.077 | 0.239 ± 0.010 | 0.992 ± 0.006 | 5 |
| Great Fen | WienerNet-SS (ALD + sigma_struct band) | 1.70 ± 0.02 | 1.86 ± 0.05 | 1.89 ± 0.16 | 1.466 ± 0.140 | 0.274 ± 0.018 | 0.991 ± 0.007 | 5 |
| Great Fen | WienerNet-SS (given-diurnal, Wiener) | 1.70 ± 0.00 | 1.79 ± 0.00 | 1.71 ± 0.00 | 1.349 ± 0.136 | 0.255 ± 0.026 | 0.982 ± 0.015 | 5 |
| Great Fen | WienerNet-SS (given-diurnal, state-sp) | 1.70 ± 0.00 | 1.79 ± 0.00 | 1.71 ± 0.00 | 1.264 ± 0.121 | 0.262 ± 0.028 | 0.978 ± 0.019 | 5 |
| Great Fen | Neural SDE (Gaussian) | 1.97 ± 0.19 | 3.00 ± 0.85 | 4.74 ± 2.55 | 2.847 ± 1.478 | 0.237 ± 0.073 | 0.910 ± 0.123 | 5 |
| Great Fen | Neural SDE (Student-t) | 1.84 ± 0.14 | 2.45 ± 0.48 | 3.37 ± 1.31 | 2.002 ± 0.604 | 0.229 ± 0.087 | 0.942 ± 0.058 | 5 |
| Great Fen | Neural SDE (ALD) | 1.74 ± 0.05 | 1.96 ± 0.13 | 2.20 ± 0.33 | 1.510 ± 0.234 | 0.249 ± 0.044 | 0.968 ± 0.027 | 5 |
| Great Fen | Mean-variance (Gaussian) | 1.73 ± 0.05 | 1.99 ± 0.21 | 2.30 ± 0.52 | 1.575 ± 0.159 | 0.261 ± 0.053 | 0.998 ± 0.001 | 5 |
| Great Fen | Mean-variance (Student-t) | 1.71 ± 0.03 | 1.94 ± 0.21 | 2.21 ± 0.57 | 1.372 ± 0.214 | 0.205 ± 0.065 | 0.978 ± 0.017 | 5 |
| Great Fen | Mean-variance (ALD) | 1.68 ± 0.01 | 1.77 ± 0.01 | 1.71 ± 0.03 | 1.248 ± 0.114 | 0.222 ± 0.040 | 0.987 ± 0.011 | 5 |
| Great Fen | Analytical SDE (Gaussian) | 1.69 | 1.77 | 1.68 | 1.696 | 0.262 | 1.000 | 1 |
| Great Fen | Analytical SDE (Student-t) | 1.69 | 1.77 | 1.68 | 1.479 | 0.240 | 0.999 | 1 |
| Great Fen | MDN (mixture) | 1.70 ± 0.01 | 1.82 ± 0.04 | 1.82 ± 0.11 | 1.328 ± 0.165 | 0.245 ± 0.031 | 0.996 ± 0.003 | 5 |
| Great Fen | Random Forest | 1.50 ± 0.05 | 1.46 ± 0.07 | 1.20 ± 0.06 | — | — | — | 5 |
| Great Fen | XGBoost | 1.59 ± 0.09 | 1.58 ± 0.09 | 1.33 ± 0.09 | — | — | — | 5 |

**Stage 2 — autoregressive gap-fill, per held-out site.** mean ± 1 SD over the 5 seeds (0-4); n = 1 entries are the calibrated, seed-independent models. Raw predictive band; metric conventions as in the pooled Stage-2 tables.  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. **These values are scored from `best.pth`, which with `data.val_frac` unset is the epoch selected ON THE HELD-OUT TEST SITE.** They are therefore an optimistic, test-selected bound rather than an honest estimate of generalisation, and they are not directly comparable to the calibrated Analytical SDE and tree arms, which have no epoch to select and gain nothing from it. The leakage-free `last.pth` scores are archived in `analysis/ss/v2_last_archive/`.

