# WienerNet-SS — manuscript results tables

### Stage 1 (distributional) — clean sites (4 in-distribution towers; Redmere 1 excluded)

| Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 0.745 ± 0.152 | 2.16 ± 0.46 | 0.775 ± 0.139 | 3.43 ± 1.71 | 0.117 ± 0.122 | 20 |
| WienerNet-SS (Gaussian) | 0.812 ± 0.185 | 2.38 ± 0.39 | 0.846 ± 0.164 | 4.50 ± 1.69 | 0.148 ± 0.132 | 20 |
| WienerNet-SS (beta-NLL) | 0.753 ± 0.123 | 2.25 ± 0.45 | 0.843 ± 0.127 | 4.11 ± 1.61 | 0.137 ± 0.083 | 20 |
| WienerNet-SS (Student-t) | 0.748 ± 0.129 | 2.24 ± 0.73 | 0.760 ± 0.165 | 3.41 ± 1.76 | 0.136 ± 0.132 | 20 |
| WienerNet-SS (mixture) | 0.718 ± 0.138 | 2.25 ± 0.39 | 0.844 ± 0.067 | 3.91 ± 1.31 | 0.102 ± 0.075 | 20 |
| WienerNet-SS (+residual) | 0.718 ± 0.135 | 2.11 ± 0.47 | 0.792 ± 0.154 | 3.47 ± 1.60 | 0.113 ± 0.110 | 20 |
| WienerNet-SS (state-space) | 0.719 ± 0.133 | 2.19 ± 0.50 | 0.801 ± 0.130 | 3.64 ± 1.67 | 0.105 ± 0.081 | 20 |
| WienerNet-SS (given-diurnal, Wiener) | 0.711 ± 0.152 | 2.12 ± 0.52 | 0.791 ± 0.101 | 3.37 ± 1.76 | 0.089 ± 0.043 | 20 |
| WienerNet-SS (given-diurnal, state-sp) | 0.711 ± 0.149 | 2.16 ± 0.50 | 0.811 ± 0.092 | 3.54 ± 1.62 | 0.086 ± 0.043 | 20 |
| WienerNet-SS (predicted-k) | 0.736 ± 0.108 | 2.11 ± 0.48 | 0.734 ± 0.269 | 3.44 ± 1.84 | 0.158 ± 0.218 | 20 |
| Neural SDE (Gaussian) | 0.758 ± 0.122 | 2.32 ± 0.37 | 0.869 ± 0.064 | 4.39 ± 1.11 | 0.132 ± 0.077 | 20 |
| Neural SDE (Student-t) | 0.730 ± 0.143 | 2.18 ± 0.40 | 0.806 ± 0.099 | 3.57 ± 1.34 | 0.100 ± 0.072 | 20 |
| Neural SDE (ALD) | 0.719 ± 0.136 | 2.18 ± 0.44 | 0.806 ± 0.128 | 3.67 ± 1.48 | 0.105 ± 0.087 | 20 |
| Mean-variance (Gaussian) | 0.733 ± 0.145 | 2.31 ± 0.45 | 0.868 ± 0.054 | 4.22 ± 1.22 | 0.126 ± 0.050 | 20 |
| Mean-variance (Student-t) | 0.717 ± 0.158 | 2.17 ± 0.45 | 0.816 ± 0.064 | 3.45 ± 1.11 | 0.100 ± 0.043 | 20 |
| Mean-variance (ALD) | 0.713 ± 0.150 | 2.25 ± 0.40 | 0.845 ± 0.064 | 3.92 ± 1.18 | 0.088 ± 0.050 | 20 |
| Analytical SDE (Gaussian) | 0.775 ± 0.112 | 2.27 ± 0.12 | 0.919 ± 0.029 | 5.04 ± 0.33 | 0.163 ± 0.068 | 4 |
| Analytical SDE (Student-t) | 0.726 ± 0.171 | 1.73 ± 0.26 | 0.735 ± 0.078 | 1.93 ± 0.43 | 0.119 ± 0.037 | 4 |
| MDN (mixture) | 0.712 ± 0.154 | 2.17 ± 0.36 | 0.830 ± 0.073 | 3.70 ± 1.02 | 0.095 ± 0.045 | 20 |
| Random Forest | — | 1.57 ± 0.03 | — | — | — | 20 |
| XGBoost | — | 1.83 ± 0.29 | — | — | — | 20 |

**Stage 1 — one-step predictive law.** mean ± 1 SD over all (site, seed) units (5 seeds × sites; calibrated seed-independent models show a bare value). CRPS and RMSE lower = better; cov90 nominal = 0.90; sharp90 = mean 90% interval width (µmol m⁻² s⁻¹); PIT-KS 0 = perfectly calibrated. Trees emit no predictive distribution (RMSE only).  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. Because the LR schedule reacts to the held-out site, model *selection* is test-informed; reported metrics are nonetheless taken from `last.pth`.

### Stage 2 (gap-filling) — clean sites (4 in-distribution towers; Redmere 1 excluded)

| Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 1.81 ± 0.77 | 2.60 ± 2.44 | 3.93 ± 5.97 | 2.456 ± 3.441 | 0.340 ± 0.177 | 0.834 ± 0.284 | 20 |
| WienerNet-SS (Gaussian) | 2.11 ± 1.22 | 3.45 ± 3.76 | 5.79 ± 8.56 | 3.552 ± 5.750 | 0.294 ± 0.176 | 0.876 ± 0.242 | 20 |
| WienerNet-SS (beta-NLL) | 1.82 ± 0.35 | 2.67 ± 1.44 | 3.77 ± 2.78 | 2.252 ± 1.417 | 0.315 ± 0.122 | 0.878 ± 0.225 | 20 |
| WienerNet-SS (Student-t) | 1.73 ± 0.35 | 2.39 ± 1.35 | 3.52 ± 3.31 | 2.243 ± 2.094 | 0.307 ± 0.185 | 0.841 ± 0.286 | 20 |
| WienerNet-SS (mixture) | 1.63 ± 0.30 | 2.05 ± 0.88 | 2.63 ± 1.80 | 1.689 ± 0.755 | 0.260 ± 0.080 | 0.941 ± 0.128 | 20 |
| WienerNet-SS (+residual) | 1.63 ± 0.24 | 2.07 ± 0.50 | 2.67 ± 1.15 | 1.696 ± 0.787 | 0.295 ± 0.141 | 0.877 ± 0.243 | 20 |
| WienerNet-SS (state-space) | — | — | — | — | — | — | 0 |
| WienerNet-SS (given-diurnal, Wiener) | 1.53 ± 0.29 | 1.69 ± 0.09 | 1.80 ± 0.13 | 1.275 ± 0.219 | 0.260 ± 0.076 | 0.907 ± 0.164 | 20 |
| WienerNet-SS (given-diurnal, state-sp) | 1.53 ± 0.29 | 1.69 ± 0.09 | 1.80 ± 0.13 | 1.309 ± 0.274 | 0.265 ± 0.083 | 0.932 ± 0.129 | 20 |
| WienerNet-SS (predicted-k) | — | — | — | — | — | — | 0 |
| Neural SDE (Gaussian) | 1.75 ± 0.27 | 2.53 ± 0.85 | 3.78 ± 2.40 | 2.229 ± 1.754 | 0.262 ± 0.123 | 0.905 ± 0.179 | 20 |
| Neural SDE (Student-t) | 1.62 ± 0.29 | 2.02 ± 0.32 | 2.54 ± 0.65 | 1.544 ± 0.395 | 0.244 ± 0.115 | 0.923 ± 0.178 | 20 |
| Neural SDE (ALD) | 1.64 ± 0.33 | 2.06 ± 0.66 | 2.51 ± 1.29 | 1.581 ± 0.435 | 0.312 ± 0.112 | 0.883 ± 0.241 | 20 |
| Mean-variance (Gaussian) | 1.59 ± 0.29 | 1.91 ± 0.28 | 2.24 ± 0.56 | 1.486 ± 0.312 | 0.243 ± 0.061 | 0.971 ± 0.052 | 20 |
| Mean-variance (Student-t) | 1.59 ± 0.31 | 1.91 ± 0.38 | 2.27 ± 0.63 | 1.411 ± 0.250 | 0.246 ± 0.067 | 0.944 ± 0.096 | 20 |
| Mean-variance (ALD) | 1.56 ± 0.31 | 1.84 ± 0.23 | 2.05 ± 0.38 | 1.388 ± 0.258 | 0.253 ± 0.052 | 0.970 ± 0.057 | 20 |
| Analytical SDE (Gaussian) | 1.52 ± 0.34 | 1.68 ± 0.10 | 1.78 ± 0.11 | 1.709 ± 0.135 | 0.302 ± 0.028 | 0.998 ± 0.003 | 4 |
| Analytical SDE (Student-t) | 1.52 ± 0.34 | 1.68 ± 0.10 | 1.78 ± 0.11 | 0.990 ± 0.105 | 0.175 ± 0.030 | 0.949 ± 0.028 | 4 |
| MDN (mixture) | 1.56 ± 0.30 | 1.81 ± 0.16 | 2.08 ± 0.40 | 1.347 ± 0.256 | 0.237 ± 0.051 | 0.970 ± 0.049 | 20 |
| Random Forest | 1.81 ± 0.14 | 1.41 ± 0.10 | 1.25 ± 0.18 | — | — | — | 20 |
| XGBoost | 2.12 ± 0.27 | 1.64 ± 0.34 | 1.44 ± 0.41 | — | — | — | 20 |

**Stage 2 — autoregressive gap-fill**, scored on the RAW predictive band (the +σ_struct band is a coverage device and is reported separately). mean ± 1 SD over all (site, seed) units (5 seeds × sites; the calibrated Analytical variants are seed-independent, n = 1 per site). RMSE by hours-into-gap is the stability test (flat ⇒ the physics drift tracks the nightly decline; climbing/diverging ⇒ it does not); CRPS / PIT-KS / cov90 are given at 5 h+, the discriminating horizon. `—` = arm not rolled out.  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. Because the LR schedule reacts to the held-out site, model *selection* is test-informed; reported metrics are nonetheless taken from `last.pth`.

### Stage 1 (distributional) — all sites (5 towers, including the Redmere-1 OOD stress)

| Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 0.770 ± 0.202 | 3.08 ± 4.17 | 0.796 ± 0.132 | 4.17 ± 3.54 | 0.101 ± 0.113 | 25 |
| WienerNet-SS (Gaussian) | 1.412 ± 2.794 | 60.09 ± 267.60 | 0.858 ± 0.148 | 12.51 ± 36.68 | 0.138 ± 0.119 | 25 |
| WienerNet-SS (beta-NLL) | 1.519 ± 2.876 | 79.49 ± 247.42 | 0.857 ± 0.116 | 14.32 ± 37.81 | 0.132 ± 0.075 | 25 |
| WienerNet-SS (Student-t) | 1.092 ± 1.462 | 31.20 ± 111.11 | 0.778 ± 0.151 | 8.25 ± 19.94 | 0.117 ± 0.124 | 25 |
| WienerNet-SS (mixture) | 0.722 ± 0.129 | 3.17 ± 4.25 | 0.852 ± 0.062 | 4.07 ± 1.39 | 0.091 ± 0.070 | 25 |
| WienerNet-SS (+residual) | 1.279 ± 1.626 | 67.03 ± 190.17 | 0.810 ± 0.143 | 13.78 ± 29.57 | 0.099 ± 0.102 | 25 |
| WienerNet-SS (state-space) | 1.414 ± 2.188 | 93.17 ± 329.32 | 0.819 ± 0.122 | 16.38 ± 39.62 | 0.093 ± 0.076 | 25 |
| WienerNet-SS (given-diurnal, Wiener) | 0.904 ± 0.844 | 48.49 ± 173.79 | 0.811 ± 0.099 | 7.36 ± 15.52 | 0.079 ± 0.043 | 25 |
| WienerNet-SS (given-diurnal, state-sp) | 1.114 ± 1.528 | 63.28 ± 195.55 | 0.828 ± 0.090 | 11.27 ± 27.72 | 0.077 ± 0.043 | 25 |
| WienerNet-SS (predicted-k) | 1.214 ± 1.827 | 101.46 ± 344.90 | 0.765 ± 0.248 | 12.24 ± 33.22 | 0.132 ± 0.201 | 25 |
| Neural SDE (Gaussian) | 1.852 ± 3.036 | 80.64 ± 238.25 | 0.878 ± 0.060 | 13.75 ± 30.75 | 0.126 ± 0.070 | 25 |
| Neural SDE (Student-t) | 2.603 ± 3.938 | 184.75 ± 394.76 | 0.819 ± 0.092 | 26.04 ± 48.22 | 0.093 ± 0.066 | 25 |
| Neural SDE (ALD) | 1.198 ± 1.394 | 67.90 ± 232.10 | 0.824 ± 0.119 | 10.54 ± 23.62 | 0.091 ± 0.083 | 25 |
| Mean-variance (Gaussian) | 1.602 ± 2.920 | 24.56 ± 68.13 | 0.878 ± 0.052 | 5.14 ± 2.67 | 0.124 ± 0.045 | 25 |
| Mean-variance (Student-t) | 1.183 ± 1.146 | 16.39 ± 33.09 | 0.828 ± 0.062 | 4.16 ± 2.58 | 0.093 ± 0.042 | 25 |
| Mean-variance (ALD) | 0.998 ± 0.840 | 10.77 ± 22.89 | 0.852 ± 0.059 | 4.27 ± 1.47 | 0.077 ± 0.050 | 25 |
| Analytical SDE (Gaussian) | 0.751 ± 0.110 | 2.22 ± 0.16 | 0.926 ± 0.030 | 5.07 ± 0.29 | 0.162 ± 0.059 | 5 |
| Analytical SDE (Student-t) | 0.702 ± 0.158 | 1.65 ± 0.29 | 0.729 ± 0.069 | 1.87 ± 0.39 | 0.114 ± 0.034 | 5 |
| MDN (mixture) | 0.940 ± 0.534 | 9.92 ± 17.24 | 0.844 ± 0.071 | 4.82 ± 2.98 | 0.084 ± 0.046 | 25 |
| Random Forest | — | 1.52 ± 0.12 | — | — | — | 25 |
| XGBoost | — | 1.73 ± 0.32 | — | — | — | 25 |

**Stage 1 — one-step predictive law.** mean ± 1 SD over all (site, seed) units (5 seeds × sites; calibrated seed-independent models show a bare value). CRPS and RMSE lower = better; cov90 nominal = 0.90; sharp90 = mean 90% interval width (µmol m⁻² s⁻¹); PIT-KS 0 = perfectly calibrated. Trees emit no predictive distribution (RMSE only).  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. Because the LR schedule reacts to the held-out site, model *selection* is test-informed; reported metrics are nonetheless taken from `last.pth`.

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

### Stage 1 (distributional) — site-wise

| Site | Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|---|
| Woodwalton | WienerNet-SS (ALD, primary) | 0.638 ± 0.255 | 1.57 ± 0.41 | 0.589 ± 0.139 | 1.25 ± 0.44 | 0.261 ± 0.176 | 5 |
| Woodwalton | WienerNet-SS (Gaussian) | 0.838 ± 0.371 | 2.17 ± 0.57 | 0.725 ± 0.310 | 3.66 ± 2.63 | 0.266 ± 0.231 | 5 |
| Woodwalton | WienerNet-SS (beta-NLL) | 0.601 ± 0.088 | 1.59 ± 0.11 | 0.711 ± 0.203 | 2.04 ± 0.73 | 0.217 ± 0.126 | 5 |
| Woodwalton | WienerNet-SS (Student-t) | 0.603 ± 0.114 | 1.51 ± 0.22 | 0.569 ± 0.232 | 1.39 ± 1.09 | 0.307 ± 0.173 | 5 |
| Woodwalton | WienerNet-SS (mixture) | 0.533 ± 0.063 | 1.73 ± 0.32 | 0.802 ± 0.105 | 2.64 ± 1.57 | 0.181 ± 0.118 | 5 |
| Woodwalton | WienerNet-SS (+residual) | 0.534 ± 0.045 | 1.46 ± 0.27 | 0.651 ± 0.256 | 1.67 ± 1.35 | 0.248 ± 0.157 | 5 |
| Woodwalton | WienerNet-SS (state-space) | 0.531 ± 0.034 | 1.49 ± 0.28 | 0.655 ± 0.178 | 1.56 ± 1.20 | 0.215 ± 0.086 | 5 |
| Woodwalton | WienerNet-SS (given-diurnal, Wiener) | 0.489 ± 0.010 | 1.32 ± 0.04 | 0.663 ± 0.099 | 0.98 ± 0.41 | 0.138 ± 0.039 | 5 |
| Woodwalton | WienerNet-SS (given-diurnal, state-sp) | 0.492 ± 0.025 | 1.47 ± 0.32 | 0.733 ± 0.128 | 1.64 ± 1.30 | 0.131 ± 0.042 | 5 |
| Woodwalton | WienerNet-SS (predicted-k) | 0.619 ± 0.081 | 1.41 ± 0.24 | 0.389 ± 0.371 | 1.18 ± 1.67 | 0.450 ± 0.283 | 5 |
| Woodwalton | Neural SDE (Gaussian) | 0.599 ± 0.088 | 1.84 ± 0.36 | 0.857 ± 0.112 | 3.52 ± 1.53 | 0.231 ± 0.090 | 5 |
| Woodwalton | Neural SDE (Student-t) | 0.545 ± 0.088 | 1.74 ± 0.56 | 0.745 ± 0.180 | 2.60 ± 2.33 | 0.188 ± 0.095 | 5 |
| Woodwalton | Neural SDE (ALD) | 0.528 ± 0.032 | 1.54 ± 0.31 | 0.686 ± 0.207 | 1.95 ± 1.57 | 0.238 ± 0.052 | 5 |
| Woodwalton | Mean-variance (Gaussian) | 0.517 ± 0.030 | 1.68 ± 0.19 | 0.872 ± 0.037 | 2.81 ± 0.75 | 0.185 ± 0.036 | 5 |
| Woodwalton | Mean-variance (Student-t) | 0.490 ± 0.020 | 1.52 ± 0.24 | 0.795 ± 0.053 | 1.91 ± 0.74 | 0.157 ± 0.040 | 5 |
| Woodwalton | Mean-variance (ALD) | 0.500 ± 0.030 | 1.77 ± 0.42 | 0.837 ± 0.082 | 2.71 ± 1.34 | 0.154 ± 0.045 | 5 |
| Woodwalton | Analytical SDE (Gaussian) | 0.634 | 2.12 | 0.957 | 5.48 | 0.256 | 1 |
| Woodwalton | Analytical SDE (Student-t) | 0.497 | 1.40 | 0.849 | 2.05 | 0.173 | 1 |
| Woodwalton | MDN (mixture) | 0.494 ± 0.026 | 1.64 ± 0.26 | 0.807 ± 0.098 | 2.54 ± 1.09 | 0.140 ± 0.039 | 5 |
| Woodwalton | Random Forest | — | 1.57 ± 0.02 | — | — | — | 5 |
| Woodwalton | XGBoost | — | 2.31 ± 0.00 | — | — | — | 5 |
| Rosedene | WienerNet-SS (ALD, primary) | 0.726 ± 0.013 | 2.06 ± 0.13 | 0.784 ± 0.062 | 2.64 ± 0.54 | 0.063 ± 0.045 | 5 |
| Rosedene | WienerNet-SS (Gaussian) | 0.739 ± 0.020 | 2.33 ± 0.27 | 0.875 ± 0.054 | 4.01 ± 1.14 | 0.094 ± 0.034 | 5 |
| Rosedene | WienerNet-SS (beta-NLL) | 0.726 ± 0.015 | 2.24 ± 0.16 | 0.871 ± 0.047 | 3.67 ± 0.81 | 0.076 ± 0.030 | 5 |
| Rosedene | WienerNet-SS (Student-t) | 0.722 ± 0.007 | 2.07 ± 0.11 | 0.795 ± 0.041 | 2.75 ± 0.46 | 0.051 ± 0.020 | 5 |
| Rosedene | WienerNet-SS (mixture) | 0.724 ± 0.004 | 2.26 ± 0.21 | 0.851 ± 0.058 | 3.54 ± 0.99 | 0.048 ± 0.010 | 5 |
| Rosedene | WienerNet-SS (+residual) | 0.725 ± 0.010 | 2.16 ± 0.23 | 0.824 ± 0.081 | 3.21 ± 1.06 | 0.059 ± 0.037 | 5 |
| Rosedene | WienerNet-SS (state-space) | 0.726 ± 0.009 | 2.14 ± 0.22 | 0.831 ± 0.083 | 3.24 ± 1.10 | 0.053 ± 0.041 | 5 |
| Rosedene | WienerNet-SS (given-diurnal, Wiener) | 0.723 ± 0.007 | 2.06 ± 0.10 | 0.797 ± 0.040 | 2.72 ± 0.46 | 0.056 ± 0.024 | 5 |
| Rosedene | WienerNet-SS (given-diurnal, state-sp) | 0.728 ± 0.004 | 2.17 ± 0.25 | 0.833 ± 0.065 | 3.27 ± 1.07 | 0.053 ± 0.026 | 5 |
| Rosedene | WienerNet-SS (predicted-k) | 0.719 ± 0.005 | 2.15 ± 0.16 | 0.824 ± 0.037 | 3.11 ± 0.56 | 0.046 ± 0.028 | 5 |
| Rosedene | Neural SDE (Gaussian) | 0.747 ± 0.025 | 2.38 ± 0.24 | 0.894 ± 0.032 | 4.24 ± 0.94 | 0.095 ± 0.030 | 5 |
| Rosedene | Neural SDE (Student-t) | 0.727 ± 0.003 | 2.17 ± 0.15 | 0.837 ± 0.039 | 3.25 ± 0.62 | 0.052 ± 0.009 | 5 |
| Rosedene | Neural SDE (ALD) | 0.722 ± 0.006 | 2.22 ± 0.19 | 0.854 ± 0.047 | 3.53 ± 0.89 | 0.043 ± 0.019 | 5 |
| Rosedene | Mean-variance (Gaussian) | 0.769 ± 0.046 | 2.62 ± 0.36 | 0.918 ± 0.035 | 5.15 ± 1.34 | 0.127 ± 0.036 | 5 |
| Rosedene | Mean-variance (Student-t) | 0.735 ± 0.008 | 2.37 ± 0.18 | 0.880 ± 0.047 | 4.08 ± 0.75 | 0.078 ± 0.035 | 5 |
| Rosedene | Mean-variance (ALD) | 0.729 ± 0.017 | 2.42 ± 0.38 | 0.890 ± 0.032 | 4.35 ± 1.18 | 0.044 ± 0.024 | 5 |
| Rosedene | Analytical SDE (Gaussian) | 0.785 | 2.31 | 0.910 | 4.69 | 0.142 | 1 |
| Rosedene | Analytical SDE (Student-t) | 0.765 | 1.84 | 0.680 | 1.55 | 0.108 | 1 |
| Rosedene | MDN (mixture) | 0.723 ± 0.008 | 2.37 ± 0.19 | 0.892 ± 0.033 | 4.21 ± 0.74 | 0.046 ± 0.020 | 5 |
| Rosedene | Random Forest | — | 1.61 ± 0.01 | — | — | — | 5 |
| Rosedene | XGBoost | — | 1.73 ± 0.00 | — | — | — | 5 |
| Redmere 1 | WienerNet-SS (ALD, primary) | 0.866 ± 0.345 | 6.77 ± 9.06 | 0.880 ± 0.034 | 7.15 ± 6.89 | 0.039 ± 0.012 | 5 |
| Redmere 1 | WienerNet-SS (Gaussian) | 3.811 ± 6.140 | 290.93 ± 588.55 | 0.907 ± 0.008 | 44.56 ± 80.33 | 0.100 ± 0.013 | 5 |
| Redmere 1 | WienerNet-SS (beta-NLL) | 4.584 ± 5.905 | 388.48 ± 467.05 | 0.911 ± 0.005 | 55.19 ± 77.17 | 0.110 ± 0.017 | 5 |
| Redmere 1 | WienerNet-SS (Student-t) | 2.467 ± 3.130 | 147.05 ± 230.43 | 0.851 ± 0.011 | 27.63 ± 42.25 | 0.038 ± 0.012 | 5 |
| Redmere 1 | WienerNet-SS (mixture) | 0.738 ± 0.098 | 6.83 ± 9.30 | 0.885 ± 0.014 | 4.70 ± 1.68 | 0.047 ± 0.015 | 5 |
| Redmere 1 | WienerNet-SS (+residual) | 3.523 ± 2.811 | 326.72 ± 334.09 | 0.883 ± 0.013 | 55.01 ± 50.77 | 0.042 ± 0.015 | 5 |
| Redmere 1 | WienerNet-SS (state-space) | 4.197 ± 4.067 | 457.09 ± 666.17 | 0.892 ± 0.014 | 67.32 ± 73.14 | 0.044 ± 0.006 | 5 |
| Redmere 1 | WienerNet-SS (given-diurnal, Wiener) | 1.673 ± 1.799 | 233.98 ± 357.02 | 0.892 ± 0.021 | 23.31 ± 32.15 | 0.041 ± 0.006 | 5 |
| Redmere 1 | WienerNet-SS (given-diurnal, state-sp) | 2.725 ± 3.137 | 307.77 ± 368.83 | 0.895 ± 0.026 | 42.18 ± 55.72 | 0.042 ± 0.019 | 5 |
| Redmere 1 | WienerNet-SS (predicted-k) | 3.125 ± 3.776 | 498.86 ± 683.37 | 0.889 ± 0.008 | 47.44 ± 68.32 | 0.029 ± 0.005 | 5 |
| Redmere 1 | Neural SDE (Gaussian) | 6.229 ± 5.031 | 393.94 ± 432.66 | 0.910 ± 0.022 | 51.18 ± 58.97 | 0.105 ± 0.019 | 5 |
| Redmere 1 | Neural SDE (Student-t) | 10.093 ± 2.304 | 915.05 ± 318.88 | 0.870 ± 0.009 | 115.94 ± 36.25 | 0.065 ± 0.018 | 5 |
| Redmere 1 | Neural SDE (ALD) | 3.114 ± 2.415 | 330.79 ± 463.93 | 0.894 ± 0.004 | 38.01 ± 46.45 | 0.037 ± 0.007 | 5 |
| Redmere 1 | Mean-variance (Gaussian) | 5.079 ± 5.674 | 113.56 ± 124.40 | 0.918 ± 0.014 | 8.82 ± 3.81 | 0.120 ± 0.012 | 5 |
| Redmere 1 | Mean-variance (Student-t) | 3.048 ± 1.526 | 73.30 ± 38.85 | 0.873 ± 0.015 | 7.03 ± 4.61 | 0.061 ± 0.011 | 5 |
| Redmere 1 | Mean-variance (ALD) | 2.140 ± 1.445 | 44.84 ± 36.46 | 0.881 ± 0.011 | 5.66 ± 1.81 | 0.032 ± 0.008 | 5 |
| Redmere 1 | Analytical SDE (Gaussian) | 0.656 | 2.01 | 0.953 | 5.16 | 0.155 | 1 |
| Redmere 1 | Analytical SDE (Student-t) | 0.604 | 1.33 | 0.704 | 1.65 | 0.093 | 1 |
| Redmere 1 | MDN (mixture) | 1.852 ± 0.545 | 40.91 ± 16.82 | 0.901 ± 0.011 | 9.28 ± 4.16 | 0.040 ± 0.010 | 5 |
| Redmere 1 | Random Forest | — | 1.28 ± 0.01 | — | — | — | 5 |
| Redmere 1 | XGBoost | — | 1.35 ± 0.00 | — | — | — | 5 |
| Redmere 2 | WienerNet-SS (ALD, primary) | 0.720 ± 0.018 | 2.47 ± 0.18 | 0.898 ± 0.029 | 5.07 ± 0.83 | 0.101 ± 0.021 | 5 |
| Redmere 2 | WienerNet-SS (Gaussian) | 0.765 ± 0.049 | 2.48 ± 0.40 | 0.908 ± 0.051 | 5.23 ± 1.51 | 0.145 ± 0.036 | 5 |
| Redmere 2 | WienerNet-SS (beta-NLL) | 0.768 ± 0.029 | 2.49 ± 0.26 | 0.902 ± 0.048 | 5.28 ± 1.11 | 0.142 ± 0.027 | 5 |
| Redmere 2 | WienerNet-SS (Student-t) | 0.746 ± 0.034 | 2.49 ± 0.26 | 0.876 ± 0.037 | 4.74 ± 1.08 | 0.120 ± 0.021 | 5 |
| Redmere 2 | WienerNet-SS (mixture) | 0.712 ± 0.013 | 2.46 ± 0.21 | 0.882 ± 0.026 | 4.69 ± 0.64 | 0.088 ± 0.019 | 5 |
| Redmere 2 | WienerNet-SS (+residual) | 0.712 ± 0.014 | 2.28 ± 0.13 | 0.872 ± 0.036 | 4.37 ± 0.68 | 0.081 ± 0.018 | 5 |
| Redmere 2 | WienerNet-SS (state-space) | 0.726 ± 0.014 | 2.52 ± 0.14 | 0.897 ± 0.013 | 5.13 ± 0.38 | 0.100 ± 0.009 | 5 |
| Redmere 2 | WienerNet-SS (given-diurnal, Wiener) | 0.729 ± 0.022 | 2.60 ± 0.16 | 0.905 ± 0.011 | 5.36 ± 0.48 | 0.108 ± 0.009 | 5 |
| Redmere 2 | WienerNet-SS (given-diurnal, state-sp) | 0.728 ± 0.033 | 2.52 ± 0.35 | 0.887 ± 0.043 | 5.08 ± 1.17 | 0.103 ± 0.023 | 5 |
| Redmere 2 | WienerNet-SS (predicted-k) | 0.712 ± 0.013 | 2.27 ± 0.21 | 0.868 ± 0.053 | 4.33 ± 1.05 | 0.093 ± 0.027 | 5 |
| Redmere 2 | Neural SDE (Gaussian) | 0.773 ± 0.025 | 2.49 ± 0.16 | 0.891 ± 0.019 | 5.06 ± 0.51 | 0.129 ± 0.010 | 5 |
| Redmere 2 | Neural SDE (Student-t) | 0.732 ± 0.026 | 2.36 ± 0.12 | 0.856 ± 0.027 | 4.27 ± 0.46 | 0.101 ± 0.018 | 5 |
| Redmere 2 | Neural SDE (ALD) | 0.732 ± 0.038 | 2.43 ± 0.23 | 0.876 ± 0.030 | 4.76 ± 0.79 | 0.087 ± 0.023 | 5 |
| Redmere 2 | Mean-variance (Gaussian) | 0.745 ± 0.017 | 2.45 ± 0.23 | 0.872 ± 0.024 | 4.54 ± 0.49 | 0.124 ± 0.011 | 5 |
| Redmere 2 | Mean-variance (Student-t) | 0.721 ± 0.016 | 2.37 ± 0.29 | 0.843 ± 0.032 | 4.03 ± 0.47 | 0.093 ± 0.015 | 5 |
| Redmere 2 | Mean-variance (ALD) | 0.713 ± 0.013 | 2.37 ± 0.13 | 0.871 ± 0.019 | 4.53 ± 0.42 | 0.086 ± 0.006 | 5 |
| Redmere 2 | Analytical SDE (Gaussian) | 0.773 | 2.24 | 0.921 | 5.08 | 0.162 | 1 |
| Redmere 2 | Analytical SDE (Student-t) | 0.733 | 1.68 | 0.689 | 1.63 | 0.110 | 1 |
| Redmere 2 | MDN (mixture) | 0.716 ± 0.011 | 2.33 ± 0.08 | 0.853 ± 0.025 | 4.22 ± 0.40 | 0.088 ± 0.016 | 5 |
| Redmere 2 | Random Forest | — | 1.54 ± 0.01 | — | — | — | 5 |
| Redmere 2 | XGBoost | — | 1.67 ± 0.00 | — | — | — | 5 |
| Great Fen | WienerNet-SS (ALD, primary) | 0.898 ± 0.015 | 2.56 ± 0.10 | 0.828 ± 0.039 | 4.77 ± 0.55 | 0.042 ± 0.014 | 5 |
| Great Fen | WienerNet-SS (Gaussian) | 0.906 ± 0.022 | 2.55 ± 0.21 | 0.877 ± 0.029 | 5.10 ± 0.88 | 0.086 ± 0.023 | 5 |
| Great Fen | WienerNet-SS (beta-NLL) | 0.915 ± 0.027 | 2.66 ± 0.16 | 0.887 ± 0.025 | 5.44 ± 0.62 | 0.113 ± 0.042 | 5 |
| Great Fen | WienerNet-SS (Student-t) | 0.921 ± 0.022 | 2.91 ± 1.02 | 0.801 ± 0.075 | 4.77 ± 1.38 | 0.068 ± 0.024 | 5 |
| Great Fen | WienerNet-SS (mixture) | 0.904 ± 0.011 | 2.55 ± 0.12 | 0.840 ± 0.045 | 4.78 ± 0.65 | 0.092 ± 0.010 | 5 |
| Great Fen | WienerNet-SS (+residual) | 0.902 ± 0.010 | 2.54 ± 0.33 | 0.820 ± 0.069 | 4.65 ± 1.38 | 0.064 ± 0.025 | 5 |
| Great Fen | WienerNet-SS (state-space) | 0.893 ± 0.006 | 2.60 ± 0.28 | 0.820 ± 0.045 | 4.64 ± 0.94 | 0.052 ± 0.010 | 5 |
| Great Fen | WienerNet-SS (given-diurnal, Wiener) | 0.904 ± 0.006 | 2.50 ± 0.05 | 0.800 ± 0.026 | 4.44 ± 0.34 | 0.053 ± 0.012 | 5 |
| Great Fen | WienerNet-SS (given-diurnal, state-sp) | 0.897 ± 0.009 | 2.48 ± 0.10 | 0.790 ± 0.046 | 4.19 ± 0.50 | 0.058 ± 0.024 | 5 |
| Great Fen | WienerNet-SS (predicted-k) | 0.893 ± 0.011 | 2.60 ± 0.21 | 0.855 ± 0.045 | 5.13 ± 0.87 | 0.043 ± 0.018 | 5 |
| Great Fen | Neural SDE (Gaussian) | 0.911 ± 0.012 | 2.55 ± 0.19 | 0.836 ± 0.051 | 4.75 ± 0.81 | 0.072 ± 0.027 | 5 |
| Great Fen | Neural SDE (Student-t) | 0.918 ± 0.038 | 2.46 ± 0.09 | 0.787 ± 0.048 | 4.15 ± 0.39 | 0.057 ± 0.023 | 5 |
| Great Fen | Neural SDE (ALD) | 0.895 ± 0.020 | 2.50 ± 0.17 | 0.809 ± 0.065 | 4.43 ± 0.80 | 0.051 ± 0.040 | 5 |
| Great Fen | Mean-variance (Gaussian) | 0.902 ± 0.018 | 2.48 ± 0.24 | 0.808 ± 0.054 | 4.36 ± 0.85 | 0.066 ± 0.013 | 5 |
| Great Fen | Mean-variance (Student-t) | 0.923 ± 0.017 | 2.40 ± 0.29 | 0.747 ± 0.029 | 3.77 ± 0.69 | 0.074 ± 0.014 | 5 |
| Great Fen | Mean-variance (ALD) | 0.909 ± 0.011 | 2.42 ± 0.24 | 0.782 ± 0.056 | 4.09 ± 0.83 | 0.066 ± 0.030 | 5 |
| Great Fen | Analytical SDE (Gaussian) | 0.907 | 2.41 | 0.887 | 4.93 | 0.093 | 1 |
| Great Fen | Analytical SDE (Student-t) | 0.911 | 2.00 | 0.723 | 2.48 | 0.087 | 1 |
| Great Fen | MDN (mixture) | 0.917 ± 0.016 | 2.34 ± 0.19 | 0.767 ± 0.055 | 3.84 ± 0.80 | 0.104 ± 0.039 | 5 |
| Great Fen | Random Forest | — | 1.58 ± 0.01 | — | — | — | 5 |
| Great Fen | XGBoost | — | 1.61 ± 0.00 | — | — | — | 5 |

**Stage 1 — one-step predictive law, per held-out site.** mean ± 1 SD over the 5 seeds (0-4); n = 1 entries are the calibrated, seed-independent models. Metric conventions as in the pooled Stage-1 tables.  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. Because the LR schedule reacts to the held-out site, model *selection* is test-informed; reported metrics are nonetheless taken from `last.pth`.

### Stage 2 (gap-filling) — site-wise

| Site | Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|---|
| Woodwalton | WienerNet-SS (ALD, primary) | 2.07 ± 1.63 | 4.72 ± 4.55 | 9.62 ± 10.71 | 5.537 ± 6.340 | 0.496 ± 0.307 | 0.400 ± 0.253 | 5 |
| Woodwalton | WienerNet-SS (Gaussian) | 3.15 ± 2.27 | 7.46 ± 6.28 | 14.47 ± 14.71 | 8.844 ± 10.470 | 0.343 ± 0.372 | 0.589 ± 0.370 | 5 |
| Woodwalton | WienerNet-SS (beta-NLL) | 1.84 ± 0.60 | 4.20 ± 1.86 | 7.17 ± 3.04 | 3.818 ± 1.931 | 0.379 ± 0.224 | 0.562 ± 0.265 | 5 |
| Woodwalton | WienerNet-SS (Student-t) | 1.70 ± 0.72 | 3.87 ± 2.18 | 7.49 ± 5.02 | 4.566 ± 3.345 | 0.501 ± 0.300 | 0.422 ± 0.305 | 5 |
| Woodwalton | WienerNet-SS (mixture) | 1.40 ± 0.51 | 2.82 ± 1.61 | 4.52 ± 2.97 | 2.392 ± 1.284 | 0.275 ± 0.156 | 0.782 ± 0.188 | 5 |
| Woodwalton | WienerNet-SS (+residual) | 1.30 ± 0.21 | 2.48 ± 0.84 | 3.97 ± 1.57 | 2.439 ± 1.282 | 0.404 ± 0.254 | 0.556 ± 0.328 | 5 |
| Woodwalton | WienerNet-SS (given-diurnal, Wiener) | 1.07 ± 0.00 | 1.64 ± 0.00 | 1.97 ± 0.00 | 1.055 ± 0.046 | 0.164 ± 0.078 | 0.664 ± 0.167 | 5 |
| Woodwalton | WienerNet-SS (given-diurnal, state-sp) | 1.07 ± 0.00 | 1.64 ± 0.00 | 1.97 ± 0.00 | 1.145 ± 0.272 | 0.149 ± 0.065 | 0.764 ± 0.176 | 5 |
| Woodwalton | Neural SDE (Gaussian) | 1.53 ± 0.39 | 3.12 ± 1.29 | 5.73 ± 4.14 | 3.661 ± 3.301 | 0.371 ± 0.212 | 0.702 ± 0.282 | 5 |
| Woodwalton | Neural SDE (Student-t) | 1.20 ± 0.15 | 2.04 ± 0.38 | 2.91 ± 0.73 | 1.835 ± 0.599 | 0.293 ± 0.234 | 0.732 ± 0.298 | 5 |
| Woodwalton | Neural SDE (ALD) | 1.27 ± 0.38 | 2.29 ± 1.06 | 3.27 ± 1.35 | 1.971 ± 0.416 | 0.430 ± 0.179 | 0.567 ± 0.328 | 5 |
| Woodwalton | Mean-variance (Gaussian) | 1.13 ± 0.06 | 1.84 ± 0.21 | 2.43 ± 0.49 | 1.450 ± 0.316 | 0.201 ± 0.101 | 0.902 ± 0.070 | 5 |
| Woodwalton | Mean-variance (Student-t) | 1.12 ± 0.04 | 1.81 ± 0.18 | 2.43 ± 0.48 | 1.371 ± 0.263 | 0.255 ± 0.125 | 0.816 ± 0.125 | 5 |
| Woodwalton | Mean-variance (ALD) | 1.08 ± 0.02 | 1.70 ± 0.08 | 2.11 ± 0.23 | 1.260 ± 0.237 | 0.205 ± 0.066 | 0.903 ± 0.086 | 5 |
| Woodwalton | Analytical SDE (Gaussian) | 1.04 | 1.59 | 1.92 | 1.876 | 0.328 | 0.998 | 1 |
| Woodwalton | Analytical SDE (Student-t) | 1.04 | 1.59 | 1.92 | 1.060 | 0.215 | 0.957 | 1 |
| Woodwalton | MDN (mixture) | 1.07 ± 0.01 | 1.69 ± 0.04 | 2.13 ± 0.16 | 1.257 ± 0.173 | 0.240 ± 0.088 | 0.908 ± 0.070 | 5 |
| Woodwalton | Random Forest | 1.68 ± 0.02 | 1.47 ± 0.02 | 1.52 ± 0.01 | — | — | — | 5 |
| Woodwalton | XGBoost | 2.53 ± 0.00 | 2.19 ± 0.00 | 2.11 ± 0.00 | — | — | — | 5 |
| Rosedene | WienerNet-SS (ALD, primary) | 1.58 ± 0.02 | 1.79 ± 0.12 | 2.27 ± 0.34 | 1.202 ± 0.096 | 0.224 ± 0.020 | 0.951 ± 0.055 | 5 |
| Rosedene | WienerNet-SS (Gaussian) | 1.60 ± 0.05 | 1.88 ± 0.11 | 2.52 ± 0.16 | 1.578 ± 0.129 | 0.263 ± 0.026 | 0.966 ± 0.042 | 5 |
| Rosedene | WienerNet-SS (beta-NLL) | 1.61 ± 0.05 | 1.77 ± 0.11 | 2.09 ± 0.27 | 1.364 ± 0.166 | 0.243 ± 0.016 | 0.990 ± 0.011 | 5 |
| Rosedene | WienerNet-SS (Student-t) | 1.56 ± 0.03 | 1.67 ± 0.08 | 2.09 ± 0.24 | 1.197 ± 0.063 | 0.207 ± 0.039 | 0.975 ± 0.021 | 5 |
| Rosedene | WienerNet-SS (mixture) | 1.56 ± 0.02 | 1.65 ± 0.05 | 2.06 ± 0.15 | 1.334 ± 0.188 | 0.258 ± 0.024 | 0.991 ± 0.009 | 5 |
| Rosedene | WienerNet-SS (+residual) | 1.61 ± 0.06 | 1.83 ± 0.21 | 2.41 ± 0.68 | 1.338 ± 0.225 | 0.223 ± 0.057 | 0.972 ± 0.025 | 5 |
| Rosedene | WienerNet-SS (given-diurnal, Wiener) | 1.55 ± 0.00 | 1.58 ± 0.00 | 1.85 ± 0.00 | 1.128 ± 0.109 | 0.252 ± 0.027 | 0.978 ± 0.012 | 5 |
| Rosedene | WienerNet-SS (given-diurnal, state-sp) | 1.55 ± 0.00 | 1.58 ± 0.00 | 1.85 ± 0.00 | 1.306 ± 0.222 | 0.292 ± 0.036 | 0.980 ± 0.020 | 5 |
| Rosedene | Neural SDE (Gaussian) | 1.66 ± 0.08 | 2.11 ± 0.34 | 3.05 ± 0.37 | 1.774 ± 0.200 | 0.210 ± 0.051 | 0.971 ± 0.032 | 5 |
| Rosedene | Neural SDE (Student-t) | 1.59 ± 0.03 | 1.73 ± 0.05 | 2.27 ± 0.20 | 1.385 ± 0.152 | 0.214 ± 0.030 | 0.983 ± 0.017 | 5 |
| Rosedene | Neural SDE (ALD) | 1.64 ± 0.15 | 1.81 ± 0.26 | 2.12 ± 0.26 | 1.341 ± 0.221 | 0.254 ± 0.043 | 0.989 ± 0.012 | 5 |
| Rosedene | Mean-variance (Gaussian) | 1.62 ± 0.05 | 1.82 ± 0.06 | 2.29 ± 0.22 | 1.764 ± 0.344 | 0.277 ± 0.041 | 0.996 ± 0.004 | 5 |
| Rosedene | Mean-variance (Student-t) | 1.57 ± 0.03 | 1.70 ± 0.06 | 2.20 ± 0.31 | 1.549 ± 0.224 | 0.267 ± 0.025 | 0.991 ± 0.011 | 5 |
| Rosedene | Mean-variance (ALD) | 1.59 ± 0.06 | 1.72 ± 0.10 | 2.14 ± 0.21 | 1.527 ± 0.315 | 0.274 ± 0.024 | 0.996 ± 0.002 | 5 |
| Rosedene | Analytical SDE (Gaussian) | 1.55 | 1.59 | 1.80 | 1.545 | 0.310 | 1.000 | 1 |
| Rosedene | Analytical SDE (Student-t) | 1.55 | 1.59 | 1.80 | 0.916 | 0.144 | 0.913 | 1 |
| Rosedene | MDN (mixture) | 1.64 ± 0.12 | 1.85 ± 0.25 | 2.39 ± 0.32 | 1.595 ± 0.244 | 0.237 ± 0.032 | 0.989 ± 0.009 | 5 |
| Rosedene | Random Forest | 1.97 ± 0.01 | 1.24 ± 0.01 | 1.18 ± 0.01 | — | — | — | 5 |
| Rosedene | XGBoost | 2.12 ± 0.00 | 1.33 ± 0.00 | 1.25 ± 0.00 | — | — | — | 5 |
| Redmere 1 | WienerNet-SS (ALD, primary) | 5.85 ± 0.82 | 8.63 ± 1.36 | 3.16 ± 0.18 | 1.503 ± 0.079 | 0.281 ± 0.032 | 0.988 ± 0.014 | 5 |
| Redmere 1 | WienerNet-SS (Gaussian) | 6.06 ± 0.97 | 8.94 ± 1.40 | 3.56 ± 0.58 | 6.797 ± 11.647 | 0.259 ± 0.032 | 0.982 ± 0.020 | 5 |
| Redmere 1 | WienerNet-SS (beta-NLL) | 5.75 ± 1.02 | 8.62 ± 1.62 | 4.37 ± 1.31 | 8.997 ± 10.846 | 0.267 ± 0.026 | 0.985 ± 0.013 | 5 |
| Redmere 1 | WienerNet-SS (Student-t) | 5.42 ± 0.44 | 8.07 ± 0.72 | 3.21 ± 0.41 | 3.573 ± 4.846 | 0.216 ± 0.028 | 0.977 ± 0.024 | 5 |
| Redmere 1 | WienerNet-SS (mixture) | 5.17 ± 1.00 | 7.57 ± 1.74 | 3.04 ± 0.14 | 1.493 ± 0.083 | 0.260 ± 0.024 | 0.992 ± 0.007 | 5 |
| Redmere 1 | WienerNet-SS (+residual) | 5.61 ± 0.63 | 8.32 ± 0.98 | 3.20 ± 0.34 | 5.551 ± 3.827 | 0.286 ± 0.021 | 0.994 ± 0.005 | 5 |
| Redmere 1 | WienerNet-SS (given-diurnal, Wiener) | 1.62 ± 0.00 | 1.81 ± 0.00 | 1.43 ± 0.00 | 2.902 ± 2.685 | 0.328 ± 0.015 | 0.998 ± 0.002 | 5 |
| Redmere 1 | WienerNet-SS (given-diurnal, state-sp) | 1.62 ± 0.00 | 1.81 ± 0.00 | 1.43 ± 0.00 | 4.439 ± 4.706 | 0.314 ± 0.017 | 0.997 ± 0.003 | 5 |
| Redmere 1 | Neural SDE (Gaussian) | 618.88 ± 588.25 | 840.83 ± 799.27 | 362.39 ± 344.36 | 10.938 ± 8.837 | 0.278 ± 0.025 | 0.987 ± 0.008 | 5 |
| Redmere 1 | Neural SDE (Student-t) | 651.54 ± 528.19 | 885.02 ± 717.47 | 381.62 ± 309.41 | 17.411 ± 4.498 | 0.280 ± 0.039 | 0.982 ± 0.014 | 5 |
| Redmere 1 | Neural SDE (ALD) | 86.86 ± 89.41 | 117.99 ± 121.46 | 50.97 ± 52.31 | 4.523 ± 3.524 | 0.295 ± 0.042 | 0.996 ± 0.005 | 5 |
| Redmere 1 | Mean-variance (Gaussian) | 261.67 ± 404.03 | 355.48 ± 548.94 | 153.33 ± 236.47 | 8.018 ± 9.608 | 0.294 ± 0.025 | 0.991 ± 0.004 | 5 |
| Redmere 1 | Mean-variance (Student-t) | 123.09 ± 128.33 | 167.22 ± 174.35 | 72.17 ± 75.02 | 4.505 ± 3.878 | 0.267 ± 0.028 | 0.989 ± 0.006 | 5 |
| Redmere 1 | Mean-variance (ALD) | 124.78 ± 136.92 | 169.46 ± 186.08 | 73.20 ± 80.01 | 3.995 ± 3.174 | 0.284 ± 0.028 | 0.997 ± 0.001 | 5 |
| Redmere 1 | Analytical SDE (Gaussian) | 1.60 | 1.76 | 1.30 | 1.686 | 0.309 | 1.000 | 1 |
| Redmere 1 | Analytical SDE (Student-t) | 1.60 | 1.76 | 1.30 | 0.794 | 0.183 | 0.973 | 1 |
| Redmere 1 | MDN (mixture) | 49.89 ± 55.38 | 67.74 ± 75.26 | 29.52 ± 32.07 | 3.042 ± 1.089 | 0.260 ± 0.031 | 0.994 ± 0.004 | 5 |
| Redmere 1 | Random Forest | 1.45 ± 0.01 | 1.20 ± 0.01 | 0.89 ± 0.01 | — | — | — | 5 |
| Redmere 1 | XGBoost | 1.52 ± 0.00 | 1.28 ± 0.00 | 0.90 ± 0.00 | — | — | — | 5 |
| Redmere 2 | WienerNet-SS (ALD, primary) | 1.89 ± 0.07 | 1.99 ± 0.23 | 1.93 ± 0.58 | 1.530 ± 0.308 | 0.351 ± 0.027 | 0.994 ± 0.011 | 5 |
| Redmere 2 | WienerNet-SS (Gaussian) | 1.91 ± 0.16 | 2.24 ± 0.85 | 3.25 ± 2.28 | 1.798 ± 0.791 | 0.295 ± 0.040 | 0.972 ± 0.032 | 5 |
| Redmere 2 | WienerNet-SS (beta-NLL) | 1.87 ± 0.05 | 1.82 ± 0.13 | 1.81 ± 0.18 | 1.467 ± 0.324 | 0.292 ± 0.037 | 0.990 ± 0.010 | 5 |
| Redmere 2 | WienerNet-SS (Student-t) | 1.86 ± 0.07 | 1.83 ± 0.16 | 1.86 ± 0.21 | 1.393 ± 0.277 | 0.271 ± 0.034 | 0.986 ± 0.012 | 5 |
| Redmere 2 | WienerNet-SS (mixture) | 1.87 ± 0.05 | 1.87 ± 0.09 | 1.81 ± 0.11 | 1.364 ± 0.124 | 0.281 ± 0.013 | 0.998 ± 0.002 | 5 |
| Redmere 2 | WienerNet-SS (+residual) | 1.85 ± 0.05 | 1.90 ± 0.13 | 2.00 ± 0.38 | 1.361 ± 0.271 | 0.284 ± 0.021 | 0.993 ± 0.010 | 5 |
| Redmere 2 | WienerNet-SS (given-diurnal, Wiener) | 1.82 ± 0.00 | 1.75 ± 0.00 | 1.66 ± 0.00 | 1.428 ± 0.124 | 0.335 ± 0.012 | 0.999 ± 0.001 | 5 |
| Redmere 2 | WienerNet-SS (given-diurnal, state-sp) | 1.82 ± 0.00 | 1.75 ± 0.00 | 1.66 ± 0.00 | 1.416 ± 0.403 | 0.340 ± 0.033 | 0.999 ± 0.001 | 5 |
| Redmere 2 | Neural SDE (Gaussian) | 1.94 ± 0.13 | 2.33 ± 0.70 | 3.00 ± 1.52 | 1.675 ± 0.452 | 0.247 ± 0.035 | 0.979 ± 0.033 | 5 |
| Redmere 2 | Neural SDE (Student-t) | 1.92 ± 0.08 | 2.12 ± 0.31 | 2.34 ± 0.93 | 1.380 ± 0.405 | 0.250 ± 0.049 | 0.994 ± 0.006 | 5 |
| Redmere 2 | Neural SDE (ALD) | 1.95 ± 0.22 | 2.24 ± 0.79 | 2.70 ± 2.14 | 1.524 ± 0.624 | 0.296 ± 0.025 | 0.994 ± 0.008 | 5 |
| Redmere 2 | Mean-variance (Gaussian) | 1.84 ± 0.02 | 1.79 ± 0.07 | 1.73 ± 0.21 | 1.227 ± 0.108 | 0.270 ± 0.028 | 0.998 ± 0.001 | 5 |
| Redmere 2 | Mean-variance (Student-t) | 1.84 ± 0.04 | 1.88 ± 0.15 | 1.88 ± 0.32 | 1.257 ± 0.252 | 0.249 ± 0.037 | 0.994 ± 0.005 | 5 |
| Redmere 2 | Mean-variance (ALD) | 1.85 ± 0.03 | 1.87 ± 0.12 | 1.71 ± 0.25 | 1.263 ± 0.145 | 0.278 ± 0.052 | 0.999 ± 0.002 | 5 |
| Redmere 2 | Analytical SDE (Gaussian) | 1.82 | 1.75 | 1.71 | 1.722 | 0.307 | 0.994 | 1 |
| Redmere 2 | Analytical SDE (Student-t) | 1.82 | 1.75 | 1.71 | 0.886 | 0.180 | 0.947 | 1 |
| Redmere 2 | MDN (mixture) | 1.83 ± 0.02 | 1.84 ± 0.15 | 1.92 ± 0.61 | 1.263 ± 0.312 | 0.253 ± 0.030 | 0.994 ± 0.009 | 5 |
| Redmere 2 | Random Forest | 1.68 ± 0.01 | 1.49 ± 0.01 | 1.24 ± 0.01 | — | — | — | 5 |
| Redmere 2 | XGBoost | 1.85 ± 0.00 | 1.59 ± 0.00 | 1.34 ± 0.00 | — | — | — | 5 |
| Great Fen | WienerNet-SS (ALD, primary) | 1.72 ± 0.04 | 1.91 ± 0.14 | 1.90 ± 0.20 | 1.553 ± 0.125 | 0.288 ± 0.054 | 0.992 ± 0.006 | 5 |
| Great Fen | WienerNet-SS (Gaussian) | 1.80 ± 0.08 | 2.21 ± 0.34 | 2.92 ± 0.86 | 1.988 ± 0.195 | 0.276 ± 0.045 | 0.978 ± 0.029 | 5 |
| Great Fen | WienerNet-SS (beta-NLL) | 1.98 ± 0.33 | 2.90 ± 1.21 | 3.99 ± 2.13 | 2.361 ± 0.915 | 0.344 ± 0.079 | 0.969 ± 0.059 | 5 |
| Great Fen | WienerNet-SS (Student-t) | 1.79 ± 0.07 | 2.21 ± 0.22 | 2.64 ± 0.45 | 1.816 ± 0.560 | 0.247 ± 0.069 | 0.982 ± 0.027 | 5 |
| Great Fen | WienerNet-SS (mixture) | 1.70 ± 0.02 | 1.88 ± 0.17 | 2.12 ± 0.68 | 1.668 ± 0.309 | 0.228 ± 0.056 | 0.992 ± 0.010 | 5 |
| Great Fen | WienerNet-SS (+residual) | 1.76 ± 0.08 | 2.08 ± 0.31 | 2.28 ± 0.51 | 1.645 ± 0.431 | 0.268 ± 0.065 | 0.988 ± 0.012 | 5 |
| Great Fen | WienerNet-SS (given-diurnal, Wiener) | 1.70 ± 0.00 | 1.79 ± 0.00 | 1.71 ± 0.00 | 1.491 ± 0.154 | 0.288 ± 0.024 | 0.988 ± 0.009 | 5 |
| Great Fen | WienerNet-SS (given-diurnal, state-sp) | 1.70 ± 0.00 | 1.79 ± 0.00 | 1.71 ± 0.00 | 1.369 ± 0.136 | 0.279 ± 0.030 | 0.985 ± 0.008 | 5 |
| Great Fen | Neural SDE (Gaussian) | 1.85 ± 0.20 | 2.54 ± 0.70 | 3.34 ± 1.15 | 1.805 ± 0.219 | 0.222 ± 0.054 | 0.969 ± 0.041 | 5 |
| Great Fen | Neural SDE (Student-t) | 1.76 ± 0.09 | 2.17 ± 0.29 | 2.64 ± 0.49 | 1.577 ± 0.154 | 0.219 ± 0.022 | 0.983 ± 0.024 | 5 |
| Great Fen | Neural SDE (ALD) | 1.70 ± 0.01 | 1.90 ± 0.07 | 1.93 ± 0.16 | 1.486 ± 0.112 | 0.267 ± 0.032 | 0.984 ± 0.024 | 5 |
| Great Fen | Mean-variance (Gaussian) | 1.79 ± 0.11 | 2.20 ± 0.42 | 2.52 ± 0.84 | 1.504 ± 0.223 | 0.224 ± 0.016 | 0.989 ± 0.007 | 5 |
| Great Fen | Mean-variance (Student-t) | 1.81 ± 0.18 | 2.27 ± 0.64 | 2.59 ± 1.05 | 1.469 ± 0.234 | 0.212 ± 0.040 | 0.974 ± 0.018 | 5 |
| Great Fen | Mean-variance (ALD) | 1.75 ± 0.08 | 2.07 ± 0.32 | 2.25 ± 0.58 | 1.505 ± 0.243 | 0.257 ± 0.031 | 0.982 ± 0.016 | 5 |
| Great Fen | Analytical SDE (Gaussian) | 1.69 | 1.77 | 1.68 | 1.694 | 0.262 | 1.000 | 1 |
| Great Fen | Analytical SDE (Student-t) | 1.69 | 1.77 | 1.68 | 1.098 | 0.163 | 0.980 | 1 |
| Great Fen | MDN (mixture) | 1.70 ± 0.02 | 1.84 ± 0.09 | 1.86 ± 0.20 | 1.272 ± 0.150 | 0.216 ± 0.041 | 0.988 ± 0.008 | 5 |
| Great Fen | Random Forest | 1.91 ± 0.01 | 1.45 ± 0.01 | 1.04 ± 0.01 | — | — | — | 5 |
| Great Fen | XGBoost | 1.97 ± 0.00 | 1.45 ± 0.00 | 1.06 ± 0.00 | — | — | — | 5 |

**Stage 2 — autoregressive gap-fill, per held-out site.** mean ± 1 SD over the 5 seeds (0-4); n = 1 entries are the calibrated, seed-independent models. Raw predictive band; metric conventions as in the pooled Stage-2 tables.  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was re-trained under one identical schedule — `reduce_on_plateau` stepped on the held-out-site loss, no validation split carved from the training sites — so the ± is a genuine seed spread and not a mixture of protocols. Because the LR schedule reacts to the held-out site, model *selection* is test-informed; reported metrics are nonetheless taken from `last.pth`.

