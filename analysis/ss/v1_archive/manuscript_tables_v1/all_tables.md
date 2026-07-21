# WienerNet-SS — manuscript results tables

### Stage 1 (distributional) — clean sites (4 in-distribution towers; Redmere 1 excluded)

| Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 0.720 ± 0.149 | 2.16 ± 0.51 | 0.793 ± 0.109 | 3.47 ± 1.67 | 0.103 ± 0.072 | 12 |
| WienerNet-SS (Gaussian) | 0.756 ± 0.118 | 2.28 ± 0.36 | 0.859 ± 0.088 | 4.31 ± 1.23 | 0.123 ± 0.054 | 12 |
| WienerNet-SS (beta-NLL) | 0.751 ± 0.137 | 2.22 ± 0.56 | 0.823 ± 0.101 | 3.88 ± 1.77 | 0.131 ± 0.074 | 12 |
| WienerNet-SS (Student-t) | 0.754 ± 0.156 | 2.79 ± 1.18 | 0.731 ± 0.166 | 3.56 ± 2.26 | 0.142 ± 0.104 | 12 |
| WienerNet-SS (mixture) | 0.759 ± 0.118 | 2.34 ± 0.47 | 0.816 ± 0.075 | 4.07 ± 1.87 | 0.120 ± 0.076 | 12 |
| WienerNet-SS (+residual) | 0.738 ± 0.179 | 2.15 ± 0.67 | 0.777 ± 0.117 | 3.47 ± 2.25 | 0.125 ± 0.061 | 12 |
| WienerNet-SS (state-space) | 0.762 ± 0.165 | 2.45 ± 1.02 | 0.764 ± 0.210 | 4.05 ± 2.74 | 0.144 ± 0.141 | 12 |
| WienerNet-SS (given-diurnal, Wiener) | 0.712 ± 0.155 | 2.19 ± 0.59 | 0.798 ± 0.104 | 3.56 ± 1.92 | 0.093 ± 0.043 | 12 |
| WienerNet-SS (given-diurnal, state-sp) | 0.720 ± 0.141 | 2.21 ± 0.49 | 0.819 ± 0.078 | 3.83 ± 1.77 | 0.085 ± 0.054 | 12 |
| WienerNet-SS (predicted-k) | 0.850 ± 0.305 | 2.23 ± 0.43 | 0.715 ± 0.238 | 3.21 ± 1.64 | 0.171 ± 0.215 | 12 |
| Neural SDE (Gaussian) | 0.794 ± 0.221 | 2.59 ± 1.15 | 0.857 ± 0.126 | 4.95 ± 2.82 | 0.151 ± 0.063 | 12 |
| Neural SDE (Student-t) | 0.779 ± 0.182 | 2.81 ± 0.73 | 0.790 ± 0.080 | 3.82 ± 1.39 | 0.112 ± 0.113 | 12 |
| Neural SDE (ALD) | 0.734 ± 0.164 | 2.22 ± 0.51 | 0.817 ± 0.109 | 3.79 ± 1.67 | 0.105 ± 0.064 | 12 |
| Mean-variance (Gaussian) | 0.762 ± 0.138 | 2.53 ± 0.53 | 0.844 ± 0.098 | 4.57 ± 1.40 | 0.132 ± 0.055 | 12 |
| Mean-variance (Student-t) | 0.755 ± 0.156 | 3.26 ± 1.32 | 0.820 ± 0.078 | 4.46 ± 2.57 | 0.104 ± 0.069 | 12 |
| Mean-variance (ALD) | 0.729 ± 0.149 | 2.39 ± 0.56 | 0.849 ± 0.062 | 4.19 ± 1.39 | 0.094 ± 0.059 | 12 |
| Analytical SDE (Gaussian) | 0.775 ± 0.112 | 2.29 ± 0.12 | 0.919 ± 0.029 | 5.05 ± 0.33 | 0.163 ± 0.068 | 4 |
| Analytical SDE (Student-t) | 0.738 ± 0.189 | 1.71 ± 0.23 | 0.689 ± 0.127 | 1.68 ± 0.26 | 0.143 ± 0.040 | 4 |
| MDN (mixture) | 0.774 ± 0.193 | 2.67 ± 1.51 | 0.834 ± 0.077 | 4.97 ± 4.89 | 0.111 ± 0.070 | 12 |
| Random Forest | — | 1.57 ± 0.03 | — | — | — | 12 |
| XGBoost | — | 1.83 ± 0.29 | — | — | — | 12 |

**Stage 1 — one-step predictive law.** mean ± 1 SD over all (site, seed) units (3 seeds × sites; calibrated seed-independent models show a bare value). CRPS and RMSE lower = better; cov90 nominal = 0.90; sharp90 = mean 90% interval width (µmol m⁻² s⁻¹); PIT-KS 0 = perfectly calibrated. Trees emit no predictive distribution (RMSE only).  **Seed 42 uses the LR-fixed re-run** (`reduce_on_plateau`, stepped on the TRAIN loss) for every gradient-trained arm, after seed 42 was found to diverge under the original constant-LR protocol; seeds 0 and 1 remain on the original protocol, so the ± for a given row mixes two training protocols and is provisional.

### Stage 2 (gap-filling) — clean sites (4 in-distribution towers; Redmere 1 excluded)

| Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 1.64 ± 0.30 | 2.06 ± 0.33 | 2.71 ± 0.98 | 1.583 ± 0.521 | 0.284 ± 0.141 | 0.890 ± 0.207 | 12 |
| WienerNet-SS (Gaussian) | 1.76 ± 0.46 | 2.40 ± 1.52 | 3.34 ± 2.86 | 2.060 ± 1.400 | 0.280 ± 0.126 | 0.905 ± 0.200 | 12 |
| WienerNet-SS (beta-NLL) | 1.75 ± 0.23 | 2.52 ± 0.84 | 3.99 ± 2.12 | 2.285 ± 1.191 | 0.326 ± 0.105 | 0.857 ± 0.230 | 12 |
| WienerNet-SS (Student-t) | 1.66 ± 0.33 | 2.13 ± 0.65 | 2.58 ± 0.94 | 1.733 ± 0.755 | 0.273 ± 0.142 | 0.890 ± 0.229 | 12 |
| WienerNet-SS (mixture) | 1.89 ± 0.53 | 2.77 ± 2.09 | 3.74 ± 4.31 | 2.206 ± 2.264 | 0.251 ± 0.091 | 0.933 ± 0.124 | 12 |
| WienerNet-SS (+residual) | 1.72 ± 0.39 | 2.41 ± 0.69 | 3.38 ± 1.11 | 1.930 ± 0.629 | 0.315 ± 0.115 | 0.824 ± 0.254 | 12 |
| WienerNet-SS (state-space) | — | — | — | — | — | — | 0 |
| WienerNet-SS (given-diurnal, Wiener) | 1.53 ± 0.30 | 1.69 ± 0.09 | 1.80 ± 0.13 | 1.277 ± 0.300 | 0.256 ± 0.077 | 0.925 ± 0.152 | 12 |
| WienerNet-SS (given-diurnal, state-sp) | 1.53 ± 0.30 | 1.69 ± 0.09 | 1.80 ± 0.13 | 1.360 ± 0.295 | 0.274 ± 0.077 | 0.940 ± 0.110 | 12 |
| WienerNet-SS (predicted-k) | — | — | — | — | — | — | 0 |
| Neural SDE (Gaussian) | 1.80 ± 0.42 | 2.64 ± 1.10 | 4.01 ± 2.64 | 2.442 ± 1.558 | 0.321 ± 0.090 | 0.884 ± 0.194 | 12 |
| Neural SDE (Student-t) | 1.89 ± 0.80 | 2.84 ± 2.70 | 4.32 ± 6.88 | 2.967 ± 5.046 | 0.300 ± 0.163 | 0.891 ± 0.230 | 12 |
| Neural SDE (ALD) | 1.68 ± 0.42 | 2.22 ± 0.73 | 2.99 ± 1.71 | 1.614 ± 0.516 | 0.273 ± 0.082 | 0.928 ± 0.166 | 12 |
| Mean-variance (Gaussian) | 1.64 ± 0.29 | 2.09 ± 0.39 | 2.68 ± 0.67 | 1.675 ± 0.270 | 0.243 ± 0.043 | 0.947 ± 0.126 | 12 |
| Mean-variance (Student-t) | 1.60 ± 0.29 | 1.97 ± 0.39 | 2.37 ± 0.80 | 1.735 ± 0.757 | 0.279 ± 0.065 | 0.980 ± 0.028 | 12 |
| Mean-variance (ALD) | 1.59 ± 0.34 | 1.96 ± 0.38 | 2.27 ± 0.50 | 1.478 ± 0.291 | 0.244 ± 0.068 | 0.978 ± 0.032 | 12 |
| Analytical SDE (Gaussian) | 1.52 ± 0.34 | 1.68 ± 0.10 | 1.78 ± 0.11 | 1.710 ± 0.136 | 0.302 ± 0.028 | 0.998 ± 0.003 | 4 |
| Analytical SDE (Student-t) | 1.52 ± 0.34 | 1.68 ± 0.10 | 1.78 ± 0.11 | 0.953 ± 0.076 | 0.161 ± 0.048 | 0.932 ± 0.024 | 4 |
| MDN (mixture) | 1.61 ± 0.34 | 1.97 ± 0.37 | 2.32 ± 0.62 | 1.765 ± 1.336 | 0.235 ± 0.069 | 0.984 ± 0.016 | 12 |
| Random Forest | 1.81 ± 0.15 | 1.41 ± 0.10 | 1.24 ± 0.18 | — | — | — | 12 |
| XGBoost | 2.12 ± 0.27 | 1.64 ± 0.35 | 1.44 ± 0.42 | — | — | — | 12 |

**Stage 2 — autoregressive gap-fill**, scored on the RAW predictive band (the +σ_struct band is a coverage device and is reported separately). mean ± 1 SD over all (site, seed) units (3 seeds × sites; the calibrated Analytical variants are seed-independent, n = 1 per site). RMSE by hours-into-gap is the stability test (flat ⇒ the physics drift tracks the nightly decline; climbing/diverging ⇒ it does not); CRPS / PIT-KS / cov90 are given at 5 h+, the discriminating horizon. `—` = arm not rolled out.  **Seed 42 uses the LR-fixed re-run** (`reduce_on_plateau`, stepped on the TRAIN loss) for every gradient-trained arm, after seed 42 was found to diverge under the original constant-LR protocol; seeds 0 and 1 remain on the original protocol, so the ± for a given row mixes two training protocols and is provisional.

### Stage 1 (distributional) — all sites (5 towers, including the Redmere-1 OOD stress)

| Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 0.727 ± 0.134 | 2.30 ± 0.54 | 0.809 ± 0.103 | 3.63 ± 1.53 | 0.093 ± 0.067 | 15 |
| WienerNet-SS (Gaussian) | 1.207 ± 1.782 | 82.77 ± 311.61 | 0.869 ± 0.081 | 10.38 ± 23.53 | 0.120 ± 0.049 | 15 |
| WienerNet-SS (beta-NLL) | 0.790 ± 0.200 | 5.37 ± 11.88 | 0.839 ± 0.096 | 4.61 ± 2.89 | 0.128 ± 0.066 | 15 |
| WienerNet-SS (Student-t) | 0.752 ± 0.139 | 2.91 ± 1.08 | 0.756 ± 0.155 | 3.72 ± 2.03 | 0.124 ± 0.100 | 15 |
| WienerNet-SS (mixture) | 0.790 ± 0.157 | 6.77 ± 12.44 | 0.834 ± 0.078 | 5.11 ± 3.31 | 0.107 ± 0.073 | 15 |
| WienerNet-SS (+residual) | 1.173 ± 1.378 | 27.53 ± 69.96 | 0.804 ± 0.118 | 11.63 ± 25.15 | 0.112 ± 0.061 | 15 |
| WienerNet-SS (state-space) | 2.221 ± 4.240 | 226.42 ± 643.97 | 0.790 ± 0.194 | 30.38 ± 76.10 | 0.123 ± 0.132 | 15 |
| WienerNet-SS (given-diurnal, Wiener) | 0.770 ± 0.315 | 9.83 ± 29.57 | 0.818 ± 0.101 | 5.16 ± 5.98 | 0.083 ± 0.044 | 15 |
| WienerNet-SS (given-diurnal, state-sp) | 0.807 ± 0.319 | 37.23 ± 100.87 | 0.839 ± 0.080 | 5.99 ± 6.29 | 0.080 ± 0.049 | 15 |
| WienerNet-SS (predicted-k) | 0.826 ± 0.275 | 2.37 ± 0.48 | 0.749 ± 0.223 | 3.37 ± 1.50 | 0.142 ± 0.200 | 15 |
| Neural SDE (Gaussian) | 4.527 ± 12.369 | 194.80 ± 636.32 | 0.871 ± 0.115 | 36.62 ± 109.91 | 0.148 ± 0.058 | 15 |
| Neural SDE (Student-t) | 3.402 ± 6.941 | 566.75 ± 1905.36 | 0.803 ± 0.075 | 47.42 ± 118.44 | 0.096 ± 0.105 | 15 |
| Neural SDE (ALD) | 1.197 ± 1.263 | 14.01 ± 31.14 | 0.834 ± 0.103 | 6.30 ± 9.23 | 0.093 ± 0.062 | 15 |
| Mean-variance (Gaussian) | 2.099 ± 3.507 | 36.30 ± 87.55 | 0.859 ± 0.093 | 5.43 ± 2.22 | 0.130 ± 0.050 | 15 |
| Mean-variance (Student-t) | 2.478 ± 4.724 | 47.87 ± 117.59 | 0.826 ± 0.072 | 5.31 ± 3.28 | 0.095 ± 0.065 | 15 |
| Mean-variance (ALD) | 1.183 ± 1.568 | 14.61 ± 38.45 | 0.857 ± 0.058 | 4.73 ± 1.95 | 0.083 ± 0.057 | 15 |
| Analytical SDE (Gaussian) | 0.751 ± 0.110 | 2.23 ± 0.16 | 0.926 ± 0.029 | 5.07 ± 0.29 | 0.162 ± 0.059 | 5 |
| Analytical SDE (Student-t) | 0.712 ± 0.174 | 1.63 ± 0.26 | 0.690 ± 0.110 | 1.67 ± 0.22 | 0.134 ± 0.041 | 5 |
| MDN (mixture) | 0.879 ± 0.346 | 8.56 ± 15.20 | 0.846 ± 0.073 | 5.27 ± 4.42 | 0.096 ± 0.070 | 15 |
| Random Forest | — | 1.51 ± 0.12 | — | — | — | 15 |
| XGBoost | — | 1.73 ± 0.33 | — | — | — | 15 |

**Stage 1 — one-step predictive law.** mean ± 1 SD over all (site, seed) units (3 seeds × sites; calibrated seed-independent models show a bare value). CRPS and RMSE lower = better; cov90 nominal = 0.90; sharp90 = mean 90% interval width (µmol m⁻² s⁻¹); PIT-KS 0 = perfectly calibrated. Trees emit no predictive distribution (RMSE only).  **Seed 42 uses the LR-fixed re-run** (`reduce_on_plateau`, stepped on the TRAIN loss) for every gradient-trained arm, after seed 42 was found to diverge under the original constant-LR protocol; seeds 0 and 1 remain on the original protocol, so the ± for a given row mixes two training protocols and is provisional.

### Stage 2 (gap-filling) — all sites (5 towers, including the Redmere-1 OOD stress)

| Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|
| WienerNet-SS (ALD, primary) | 2.52 ± 1.86 | 3.51 ± 3.04 | 3.01 ± 1.09 | 1.648 ± 0.484 | 0.289 ± 0.126 | 0.908 ± 0.188 | 15 |
| WienerNet-SS (Gaussian) | 2.34 ± 1.37 | 3.27 ± 2.43 | 3.30 ± 2.55 | 2.817 ± 3.464 | 0.285 ± 0.113 | 0.923 ± 0.181 | 15 |
| WienerNet-SS (beta-NLL) | 2.56 ± 2.02 | 3.75 ± 3.21 | 4.24 ± 2.04 | 2.339 ± 1.098 | 0.323 ± 0.097 | 0.881 ± 0.210 | 15 |
| WienerNet-SS (Student-t) | 2.57 ± 1.95 | 3.54 ± 3.03 | 2.82 ± 0.98 | 1.762 ± 0.675 | 0.276 ± 0.127 | 0.910 ± 0.207 | 15 |
| WienerNet-SS (mixture) | 2.00 ± 0.65 | 2.83 ± 1.95 | 3.37 ± 3.90 | 2.103 ± 2.022 | 0.259 ± 0.083 | 0.946 ± 0.114 | 15 |
| WienerNet-SS (+residual) | 2.64 ± 1.94 | 3.81 ± 2.97 | 3.39 ± 0.99 | 2.537 ± 2.036 | 0.315 ± 0.102 | 0.859 ± 0.237 | 15 |
| WienerNet-SS (state-space) | — | — | — | — | — | — | 0 |
| WienerNet-SS (given-diurnal, Wiener) | 1.55 ± 0.27 | 1.72 ± 0.09 | 1.72 ± 0.19 | 1.296 ± 0.271 | 0.271 ± 0.075 | 0.939 ± 0.138 | 15 |
| WienerNet-SS (given-diurnal, state-sp) | 1.55 ± 0.27 | 1.72 ± 0.09 | 1.72 ± 0.19 | 1.432 ± 0.315 | 0.285 ± 0.072 | 0.951 ± 0.100 | 15 |
| WienerNet-SS (predicted-k) | — | — | — | — | — | — | 0 |
| Neural SDE (Gaussian) | 374.81 ± 1237.26 | 509.38 ± 1680.91 | 221.84 ± 723.72 | 7.627 ± 16.425 | 0.321 ± 0.080 | 0.905 ± 0.177 | 15 |
| Neural SDE (Student-t) | 120.88 ± 299.92 | 164.43 ± 407.37 | 73.37 ± 174.68 | 6.325 ± 9.686 | 0.292 ± 0.146 | 0.911 ± 0.208 | 15 |
| Neural SDE (ALD) | 29.06 ± 72.55 | 39.39 ± 98.55 | 18.68 ± 41.89 | 2.202 ± 1.677 | 0.277 ± 0.073 | 0.942 ± 0.150 | 15 |
| Mean-variance (Gaussian) | 115.35 ± 297.41 | 156.61 ± 404.10 | 68.92 ± 173.59 | 4.260 ± 6.849 | 0.250 ± 0.041 | 0.955 ± 0.113 | 15 |
| Mean-variance (Student-t) | 134.56 ± 395.48 | 182.65 ± 537.36 | 79.96 ± 231.15 | 4.721 ± 9.096 | 0.277 ± 0.059 | 0.983 ± 0.026 | 15 |
| Mean-variance (ALD) | 40.01 ± 130.66 | 54.19 ± 177.56 | 24.52 ± 76.18 | 2.336 ± 3.002 | 0.252 ± 0.063 | 0.981 ± 0.029 | 15 |
| Analytical SDE (Gaussian) | 1.54 ± 0.30 | 1.69 ± 0.09 | 1.68 ± 0.23 | 1.705 ± 0.118 | 0.303 ± 0.024 | 0.998 ± 0.003 | 5 |
| Analytical SDE (Student-t) | 1.54 ± 0.30 | 1.69 ± 0.09 | 1.68 ± 0.23 | 0.920 ± 0.099 | 0.164 ± 0.043 | 0.940 ± 0.027 | 5 |
| MDN (mixture) | 2.34 ± 1.77 | 2.99 ± 2.44 | 2.58 ± 0.85 | 1.743 ± 1.187 | 0.241 ± 0.062 | 0.986 ± 0.015 | 15 |
| Random Forest | 1.74 ± 0.20 | 1.37 ± 0.13 | 1.17 ± 0.21 | — | — | — | 15 |
| XGBoost | 2.00 ± 0.34 | 1.57 ± 0.34 | 1.33 ± 0.43 | — | — | — | 15 |

**Stage 2 — autoregressive gap-fill**, scored on the RAW predictive band (the +σ_struct band is a coverage device and is reported separately). mean ± 1 SD over all (site, seed) units (3 seeds × sites; the calibrated Analytical variants are seed-independent, n = 1 per site). RMSE by hours-into-gap is the stability test (flat ⇒ the physics drift tracks the nightly decline; climbing/diverging ⇒ it does not); CRPS / PIT-KS / cov90 are given at 5 h+, the discriminating horizon. `—` = arm not rolled out.  **Seed 42 uses the LR-fixed re-run** (`reduce_on_plateau`, stepped on the TRAIN loss) for every gradient-trained arm, after seed 42 was found to diverge under the original constant-LR protocol; seeds 0 and 1 remain on the original protocol, so the ± for a given row mixes two training protocols and is provisional.

### Stage 1 (distributional) — site-wise

| Site | Model / ablation | CRPS | RMSE | cov90 | sharp90 | PIT-KS | n |
|---|---|---|---|---|---|---|---|
| Woodwalton | WienerNet-SS (ALD, primary) | 0.514 ± 0.020 | 1.53 ± 0.25 | 0.735 ± 0.195 | 1.91 ± 1.28 | 0.196 ± 0.097 | 3 |
| Woodwalton | WienerNet-SS (Gaussian) | 0.634 ± 0.138 | 1.98 ± 0.59 | 0.812 ± 0.184 | 3.56 ± 2.34 | 0.193 ± 0.040 | 3 |
| Woodwalton | WienerNet-SS (beta-NLL) | 0.576 ± 0.047 | 1.48 ± 0.04 | 0.696 ± 0.121 | 1.68 ± 0.48 | 0.228 ± 0.055 | 3 |
| Woodwalton | WienerNet-SS (Student-t) | 0.593 ± 0.146 | 2.27 ± 1.56 | 0.677 ± 0.310 | 3.41 ± 4.55 | 0.260 ± 0.160 | 3 |
| Woodwalton | WienerNet-SS (mixture) | 0.671 ± 0.163 | 2.27 ± 0.74 | 0.821 ± 0.136 | 4.53 ± 3.14 | 0.211 ± 0.111 | 3 |
| Woodwalton | WienerNet-SS (+residual) | 0.504 ± 0.006 | 1.30 ± 0.02 | 0.626 ± 0.066 | 0.91 ± 0.29 | 0.219 ± 0.027 | 3 |
| Woodwalton | WienerNet-SS (state-space) | 0.562 ± 0.058 | 1.48 ± 0.30 | 0.535 ± 0.339 | 1.58 ± 1.98 | 0.346 ± 0.152 | 3 |
| Woodwalton | WienerNet-SS (given-diurnal, Wiener) | 0.490 ± 0.011 | 1.44 ± 0.19 | 0.721 ± 0.168 | 1.57 ± 1.13 | 0.140 ± 0.051 | 3 |
| Woodwalton | WienerNet-SS (given-diurnal, state-sp) | 0.527 ± 0.065 | 1.79 ± 0.69 | 0.787 ± 0.150 | 2.94 ± 2.87 | 0.149 ± 0.062 | 3 |
| Woodwalton | WienerNet-SS (predicted-k) | 1.076 ± 0.611 | 2.01 ± 0.80 | 0.372 ± 0.264 | 0.95 ± 0.74 | 0.493 ± 0.202 | 3 |
| Woodwalton | Neural SDE (Gaussian) | 0.585 ± 0.110 | 1.88 ± 0.72 | 0.781 ± 0.261 | 3.41 ± 3.09 | 0.241 ± 0.004 | 3 |
| Woodwalton | Neural SDE (Student-t) | 0.705 ± 0.358 | 2.27 ± 1.20 | 0.746 ± 0.116 | 3.19 ± 2.56 | 0.251 ± 0.167 | 3 |
| Woodwalton | Neural SDE (ALD) | 0.505 ± 0.009 | 1.67 ± 0.35 | 0.798 ± 0.208 | 2.72 ± 1.85 | 0.192 ± 0.023 | 3 |
| Woodwalton | Mean-variance (Gaussian) | 0.569 ± 0.069 | 1.96 ± 0.44 | 0.803 ± 0.192 | 3.41 ± 1.87 | 0.213 ± 0.010 | 3 |
| Woodwalton | Mean-variance (Student-t) | 0.624 ± 0.245 | 3.01 ± 2.41 | 0.857 ± 0.103 | 5.22 ± 5.66 | 0.193 ± 0.097 | 3 |
| Woodwalton | Mean-variance (ALD) | 0.534 ± 0.056 | 2.13 ± 0.71 | 0.880 ± 0.088 | 3.81 ± 2.31 | 0.176 ± 0.050 | 3 |
| Woodwalton | Analytical SDE (Gaussian) | 0.634 | 2.13 | 0.957 | 5.48 | 0.256 | 1 |
| Woodwalton | Analytical SDE (Student-t) | 0.497 | 1.40 | 0.849 | 2.05 | 0.173 | 1 |
| Woodwalton | MDN (mixture) | 0.739 ± 0.408 | 3.56 ± 3.28 | 0.913 ± 0.075 | 8.81 ± 9.95 | 0.211 ± 0.052 | 3 |
| Woodwalton | Random Forest | — | 1.55 ± 0.01 | — | — | — | 3 |
| Woodwalton | XGBoost | — | 2.31 ± 0.00 | — | — | — | 3 |
| Rosedene | WienerNet-SS (ALD, primary) | 0.736 ± 0.007 | 1.97 ± 0.06 | 0.742 ± 0.046 | 2.29 ± 0.40 | 0.081 ± 0.027 | 3 |
| Rosedene | WienerNet-SS (Gaussian) | 0.754 ± 0.013 | 2.42 ± 0.22 | 0.912 ± 0.020 | 4.61 ± 0.73 | 0.122 ± 0.018 | 3 |
| Rosedene | WienerNet-SS (beta-NLL) | 0.725 ± 0.008 | 2.18 ± 0.16 | 0.852 ± 0.040 | 3.41 ± 0.73 | 0.064 ± 0.023 | 3 |
| Rosedene | WienerNet-SS (Student-t) | 0.739 ± 0.007 | 2.23 ± 0.14 | 0.727 ± 0.031 | 2.35 ± 0.38 | 0.094 ± 0.027 | 3 |
| Rosedene | WienerNet-SS (mixture) | 0.729 ± 0.002 | 2.02 ± 0.06 | 0.770 ± 0.043 | 2.46 ± 0.43 | 0.081 ± 0.008 | 3 |
| Rosedene | WienerNet-SS (+residual) | 0.734 ± 0.009 | 2.04 ± 0.13 | 0.765 ± 0.073 | 2.53 ± 0.66 | 0.092 ± 0.011 | 3 |
| Rosedene | WienerNet-SS (state-space) | 0.733 ± 0.008 | 1.99 ± 0.08 | 0.778 ± 0.078 | 2.51 ± 0.58 | 0.067 ± 0.043 | 3 |
| Rosedene | WienerNet-SS (given-diurnal, Wiener) | 0.732 ± 0.009 | 1.98 ± 0.07 | 0.751 ± 0.047 | 2.34 ± 0.41 | 0.082 ± 0.030 | 3 |
| Rosedene | WienerNet-SS (given-diurnal, state-sp) | 0.728 ± 0.007 | 2.00 ± 0.04 | 0.795 ± 0.035 | 2.60 ± 0.34 | 0.054 ± 0.022 | 3 |
| Rosedene | WienerNet-SS (predicted-k) | 0.725 ± 0.014 | 2.06 ± 0.10 | 0.806 ± 0.051 | 2.83 ± 0.54 | 0.063 ± 0.064 | 3 |
| Rosedene | Neural SDE (Gaussian) | 0.754 ± 0.029 | 2.32 ± 0.30 | 0.876 ± 0.057 | 4.05 ± 1.37 | 0.114 ± 0.063 | 3 |
| Rosedene | Neural SDE (Student-t) | 0.735 ± 0.004 | 2.44 ± 0.09 | 0.809 ± 0.032 | 3.02 ± 0.30 | 0.041 ± 0.016 | 3 |
| Rosedene | Neural SDE (ALD) | 0.729 ± 0.004 | 2.02 ± 0.01 | 0.797 ± 0.020 | 2.66 ± 0.18 | 0.052 ± 0.020 | 3 |
| Rosedene | Mean-variance (Gaussian) | 0.750 ± 0.007 | 2.44 ± 0.17 | 0.881 ± 0.013 | 4.25 ± 0.43 | 0.101 ± 0.012 | 3 |
| Rosedene | Mean-variance (Student-t) | 0.755 ± 0.006 | 3.32 ± 1.19 | 0.838 ± 0.061 | 4.23 ± 1.28 | 0.058 ± 0.021 | 3 |
| Rosedene | Mean-variance (ALD) | 0.727 ± 0.003 | 2.27 ± 0.19 | 0.873 ± 0.033 | 3.83 ± 0.80 | 0.049 ± 0.012 | 3 |
| Rosedene | Analytical SDE (Gaussian) | 0.784 | 2.33 | 0.910 | 4.69 | 0.142 | 1 |
| Rosedene | Analytical SDE (Student-t) | 0.765 | 1.85 | 0.680 | 1.55 | 0.108 | 1 |
| Rosedene | MDN (mixture) | 0.720 ± 0.015 | 2.23 ± 0.14 | 0.842 ± 0.058 | 3.23 ± 0.83 | 0.044 ± 0.016 | 3 |
| Rosedene | Random Forest | — | 1.61 ± 0.00 | — | — | — | 3 |
| Rosedene | XGBoost | — | 1.73 ± 0.00 | — | — | — | 3 |
| Redmere 1 | WienerNet-SS (ALD, primary) | 0.755 ± 0.038 | 2.83 ± 0.25 | 0.869 ± 0.043 | 4.26 ± 0.57 | 0.052 ± 0.007 | 3 |
| Redmere 1 | WienerNet-SS (Gaussian) | 3.011 ± 4.006 | 404.71 ± 696.68 | 0.908 ± 0.010 | 34.65 ± 52.55 | 0.106 ± 0.025 | 3 |
| Redmere 1 | WienerNet-SS (beta-NLL) | 0.950 ± 0.359 | 17.98 ± 26.24 | 0.905 ± 0.005 | 7.53 ± 5.02 | 0.120 ± 0.020 | 3 |
| Redmere 1 | WienerNet-SS (Student-t) | 0.743 ± 0.028 | 3.35 ± 0.06 | 0.853 ± 0.018 | 4.35 ± 0.02 | 0.050 ± 0.019 | 3 |
| Redmere 1 | WienerNet-SS (mixture) | 0.910 ± 0.263 | 24.49 ± 22.20 | 0.910 ± 0.025 | 9.30 ± 4.97 | 0.055 ± 0.019 | 3 |
| Redmere 1 | WienerNet-SS (+residual) | 2.912 ± 2.727 | 129.01 ± 122.26 | 0.911 ± 0.013 | 44.26 ± 49.01 | 0.060 ± 0.014 | 3 |
| Redmere 1 | WienerNet-SS (state-space) | 8.060 ± 7.859 | 1122.29 ± 1182.40 | 0.895 ± 0.020 | 135.73 ± 140.32 | 0.040 ± 0.014 | 3 |
| Redmere 1 | WienerNet-SS (given-diurnal, Wiener) | 1.002 ± 0.680 | 40.38 ± 66.10 | 0.896 ± 0.025 | 11.56 ± 12.36 | 0.043 ± 0.019 | 3 |
| Redmere 1 | WienerNet-SS (given-diurnal, state-sp) | 1.153 ± 0.616 | 177.33 ± 185.53 | 0.918 ± 0.009 | 14.66 ± 10.92 | 0.058 ± 0.011 | 3 |
| Redmere 1 | WienerNet-SS (predicted-k) | 0.729 ± 0.038 | 2.93 ± 0.11 | 0.884 ± 0.014 | 4.01 ± 0.28 | 0.027 ± 0.009 | 3 |
| Redmere 1 | Neural SDE (Gaussian) | 19.459 ± 25.548 | 963.64 ± 1313.76 | 0.929 ± 0.012 | 163.32 ± 233.28 | 0.137 ± 0.029 | 3 |
| Redmere 1 | Neural SDE (Student-t) | 13.895 ± 11.431 | 2822.53 ± 3983.94 | 0.851 ± 0.009 | 221.81 ± 202.88 | 0.035 ± 0.011 | 3 |
| Redmere 1 | Neural SDE (ALD) | 3.050 ± 2.140 | 61.18 ± 51.16 | 0.901 ± 0.016 | 16.34 ± 19.79 | 0.046 ± 0.011 | 3 |
| Redmere 1 | Mean-variance (Gaussian) | 7.446 ± 5.690 | 171.42 ± 139.37 | 0.921 ± 0.009 | 8.87 ± 1.16 | 0.121 ± 0.023 | 3 |
| Redmere 1 | Mean-variance (Student-t) | 9.367 ± 8.193 | 226.30 ± 192.57 | 0.848 ± 0.043 | 8.71 ± 4.20 | 0.058 ± 0.016 | 3 |
| Redmere 1 | Mean-variance (ALD) | 3.001 ± 3.302 | 63.48 ± 76.60 | 0.893 ± 0.024 | 6.89 ± 2.66 | 0.042 ± 0.023 | 3 |
| Redmere 1 | Analytical SDE (Gaussian) | 0.657 | 2.01 | 0.954 | 5.16 | 0.155 | 1 |
| Redmere 1 | Analytical SDE (Student-t) | 0.605 | 1.32 | 0.696 | 1.62 | 0.096 | 1 |
| Redmere 1 | MDN (mixture) | 1.299 ± 0.551 | 32.10 ± 23.78 | 0.897 ± 0.008 | 6.49 ± 1.58 | 0.036 ± 0.018 | 3 |
| Redmere 1 | Random Forest | — | 1.28 ± 0.01 | — | — | — | 3 |
| Redmere 1 | XGBoost | — | 1.35 ± 0.00 | — | — | — | 3 |
| Redmere 2 | WienerNet-SS (ALD, primary) | 0.714 ± 0.011 | 2.38 ± 0.08 | 0.870 ± 0.051 | 4.53 ± 0.62 | 0.080 ± 0.016 | 3 |
| Redmere 2 | WienerNet-SS (Gaussian) | 0.731 ± 0.005 | 2.19 ± 0.04 | 0.858 ± 0.029 | 4.07 ± 0.34 | 0.117 ± 0.009 | 3 |
| Redmere 2 | WienerNet-SS (beta-NLL) | 0.766 ± 0.031 | 2.43 ± 0.33 | 0.881 ± 0.066 | 4.91 ± 1.27 | 0.136 ± 0.031 | 3 |
| Redmere 2 | WienerNet-SS (Student-t) | 0.717 ± 0.003 | 2.92 ± 0.44 | 0.861 ± 0.035 | 4.26 ± 0.55 | 0.093 ± 0.003 | 3 |
| Redmere 2 | WienerNet-SS (mixture) | 0.730 ± 0.024 | 2.44 ± 0.54 | 0.842 ± 0.075 | 4.30 ± 1.69 | 0.102 ± 0.026 | 3 |
| Redmere 2 | WienerNet-SS (+residual) | 0.733 ± 0.020 | 2.35 ± 0.26 | 0.873 ± 0.053 | 4.69 ± 1.20 | 0.086 ± 0.040 | 3 |
| Redmere 2 | WienerNet-SS (state-space) | 0.803 ± 0.158 | 2.94 ± 0.96 | 0.897 ± 0.020 | 5.51 ± 1.58 | 0.102 ± 0.020 | 3 |
| Redmere 2 | WienerNet-SS (given-diurnal, Wiener) | 0.718 ± 0.018 | 2.50 ± 0.08 | 0.886 ± 0.023 | 4.84 ± 0.33 | 0.098 ± 0.012 | 3 |
| Redmere 2 | WienerNet-SS (given-diurnal, state-sp) | 0.726 ± 0.026 | 2.36 ± 0.36 | 0.870 ± 0.050 | 4.66 ± 1.30 | 0.098 ± 0.022 | 3 |
| Redmere 2 | WienerNet-SS (predicted-k) | 0.702 ± 0.007 | 2.26 ± 0.21 | 0.862 ± 0.033 | 4.21 ± 0.60 | 0.087 ± 0.010 | 3 |
| Redmere 2 | Neural SDE (Gaussian) | 0.755 ± 0.018 | 2.35 ± 0.21 | 0.872 ± 0.044 | 4.53 ± 0.81 | 0.134 ± 0.017 | 3 |
| Redmere 2 | Neural SDE (Student-t) | 0.741 ± 0.030 | 3.35 ± 0.37 | 0.856 ± 0.038 | 4.67 ± 0.72 | 0.077 ± 0.014 | 3 |
| Redmere 2 | Neural SDE (ALD) | 0.767 ± 0.067 | 2.72 ± 0.49 | 0.897 ± 0.032 | 5.54 ± 1.22 | 0.094 ± 0.032 | 3 |
| Redmere 2 | Mean-variance (Gaussian) | 0.808 ± 0.053 | 3.06 ± 0.55 | 0.890 ± 0.028 | 5.94 ± 1.30 | 0.137 ± 0.018 | 3 |
| Redmere 2 | Mean-variance (Student-t) | 0.714 ± 0.013 | 3.12 ± 1.05 | 0.846 ± 0.061 | 4.26 ± 1.16 | 0.077 ± 0.013 | 3 |
| Redmere 2 | Mean-variance (ALD) | 0.726 ± 0.033 | 2.32 ± 0.34 | 0.850 ± 0.040 | 4.36 ± 1.10 | 0.087 ± 0.020 | 3 |
| Redmere 2 | Analytical SDE (Gaussian) | 0.773 | 2.27 | 0.921 | 5.09 | 0.162 | 1 |
| Redmere 2 | Analytical SDE (Student-t) | 0.733 | 1.69 | 0.689 | 1.63 | 0.110 | 1 |
| Redmere 2 | MDN (mixture) | 0.726 ± 0.019 | 2.42 ± 0.37 | 0.820 ± 0.052 | 3.95 ± 1.07 | 0.078 ± 0.011 | 3 |
| Redmere 2 | Random Forest | — | 1.54 ± 0.01 | — | — | — | 3 |
| Redmere 2 | XGBoost | — | 1.67 ± 0.00 | — | — | — | 3 |
| Great Fen | WienerNet-SS (ALD, primary) | 0.915 ± 0.030 | 2.76 ± 0.34 | 0.827 ± 0.058 | 5.14 ± 1.23 | 0.056 ± 0.011 | 3 |
| Great Fen | WienerNet-SS (Gaussian) | 0.904 ± 0.038 | 2.53 ± 0.14 | 0.855 ± 0.019 | 5.00 ± 0.66 | 0.062 ± 0.032 | 3 |
| Great Fen | WienerNet-SS (beta-NLL) | 0.935 ± 0.048 | 2.79 ± 0.44 | 0.863 ± 0.049 | 5.51 ± 1.31 | 0.094 ± 0.050 | 3 |
| Great Fen | WienerNet-SS (Student-t) | 0.969 ± 0.047 | 3.76 ± 1.66 | 0.659 ± 0.121 | 4.23 ± 1.81 | 0.121 ± 0.068 | 3 |
| Great Fen | WienerNet-SS (mixture) | 0.908 ± 0.035 | 2.62 ± 0.29 | 0.829 ± 0.031 | 4.98 ± 0.87 | 0.085 ± 0.045 | 3 |
| Great Fen | WienerNet-SS (+residual) | 0.980 ± 0.073 | 2.93 ± 0.55 | 0.844 ± 0.090 | 5.75 ± 2.17 | 0.104 ± 0.018 | 3 |
| Great Fen | WienerNet-SS (state-space) | 0.950 ± 0.074 | 3.40 ± 1.16 | 0.844 ± 0.069 | 6.58 ± 3.00 | 0.061 ± 0.023 | 3 |
| Great Fen | WienerNet-SS (given-diurnal, Wiener) | 0.908 ± 0.020 | 2.83 ± 0.46 | 0.835 ± 0.055 | 5.49 ± 1.60 | 0.050 ± 0.017 | 3 |
| Great Fen | WienerNet-SS (given-diurnal, state-sp) | 0.900 ± 0.018 | 2.66 ± 0.19 | 0.824 ± 0.027 | 5.12 ± 0.53 | 0.040 ± 0.017 | 3 |
| Great Fen | WienerNet-SS (predicted-k) | 0.897 ± 0.018 | 2.60 ± 0.07 | 0.820 ± 0.047 | 4.84 ± 0.47 | 0.041 ± 0.020 | 3 |
| Great Fen | Neural SDE (Gaussian) | 1.084 ± 0.241 | 3.81 ± 1.87 | 0.898 ± 0.038 | 7.78 ± 3.81 | 0.114 ± 0.035 | 3 |
| Great Fen | Neural SDE (Student-t) | 0.936 ± 0.052 | 3.18 ± 0.26 | 0.751 ± 0.083 | 4.40 ± 0.52 | 0.077 ± 0.041 | 3 |
| Great Fen | Neural SDE (ALD) | 0.933 ± 0.053 | 2.47 ± 0.34 | 0.778 ± 0.088 | 4.26 ± 1.34 | 0.084 ± 0.063 | 3 |
| Great Fen | Mean-variance (Gaussian) | 0.919 ± 0.031 | 2.63 ± 0.25 | 0.800 ± 0.065 | 4.69 ± 0.66 | 0.076 ± 0.014 | 3 |
| Great Fen | Mean-variance (Student-t) | 0.929 ± 0.010 | 3.60 ± 0.99 | 0.738 ± 0.044 | 4.15 ± 0.25 | 0.090 ± 0.014 | 3 |
| Great Fen | Mean-variance (ALD) | 0.929 ± 0.026 | 2.84 ± 0.80 | 0.791 ± 0.056 | 4.76 ± 1.59 | 0.062 ± 0.032 | 3 |
| Great Fen | Analytical SDE (Gaussian) | 0.907 | 2.42 | 0.887 | 4.94 | 0.093 | 1 |
| Great Fen | Analytical SDE (Student-t) | 0.958 | 1.90 | 0.538 | 1.48 | 0.183 | 1 |
| Great Fen | MDN (mixture) | 0.912 ± 0.012 | 2.47 ± 0.30 | 0.761 ± 0.053 | 3.89 ± 0.90 | 0.113 ± 0.013 | 3 |
| Great Fen | Random Forest | — | 1.58 ± 0.00 | — | — | — | 3 |
| Great Fen | XGBoost | — | 1.61 ± 0.00 | — | — | — | 3 |

**Stage 1 — one-step predictive law, per held-out site.** mean ± 1 SD over the 3 seeds (0/1/42); n = 1 entries are the calibrated, seed-independent models. Metric conventions as in the pooled Stage-1 tables.  **Seed 42 uses the LR-fixed re-run** (`reduce_on_plateau`, stepped on the TRAIN loss) for every gradient-trained arm, after seed 42 was found to diverge under the original constant-LR protocol; seeds 0 and 1 remain on the original protocol, so the ± for a given row mixes two training protocols and is provisional.

### Stage 2 (gap-filling) — site-wise

| Site | Model / ablation | RMSE 0-2h | RMSE 2-5h | RMSE 5h+ | CRPS 5h+ | PIT-KS 5h+ | cov90 5h+ | n |
|---|---|---|---|---|---|---|---|---|
| Woodwalton | WienerNet-SS (ALD, primary) | 1.20 ± 0.09 | 2.15 ± 0.31 | 4.04 ± 0.94 | 2.189 ± 0.559 | 0.351 ± 0.290 | 0.633 ± 0.316 | 3 |
| Woodwalton | WienerNet-SS (Gaussian) | 1.90 ± 1.05 | 3.97 ± 2.76 | 6.72 ± 4.57 | 3.740 ± 2.172 | 0.310 ± 0.281 | 0.655 ± 0.307 | 3 |
| Woodwalton | WienerNet-SS (beta-NLL) | 1.58 ± 0.21 | 3.45 ± 0.56 | 7.01 ± 0.43 | 4.036 ± 0.638 | 0.454 ± 0.115 | 0.481 ± 0.088 | 3 |
| Woodwalton | WienerNet-SS (Student-t) | 1.24 ± 0.19 | 2.38 ± 0.82 | 3.55 ± 1.16 | 2.442 ± 1.309 | 0.418 ± 0.237 | 0.671 ± 0.429 | 3 |
| Woodwalton | WienerNet-SS (mixture) | 2.27 ± 1.03 | 5.40 ± 3.13 | 9.14 ± 6.62 | 4.889 ± 3.667 | 0.276 ± 0.194 | 0.773 ± 0.182 | 3 |
| Woodwalton | WienerNet-SS (+residual) | 1.16 ± 0.01 | 2.01 ± 0.04 | 3.62 ± 0.27 | 2.233 ± 0.213 | 0.468 ± 0.058 | 0.410 ± 0.086 | 3 |
| Woodwalton | WienerNet-SS (given-diurnal, Wiener) | 1.07 ± 0.00 | 1.64 ± 0.00 | 1.97 ± 0.00 | 1.106 ± 0.099 | 0.178 ± 0.088 | 0.747 ± 0.249 | 3 |
| Woodwalton | WienerNet-SS (given-diurnal, state-sp) | 1.07 ± 0.00 | 1.64 ± 0.00 | 1.97 ± 0.00 | 1.392 ± 0.503 | 0.184 ± 0.096 | 0.813 ± 0.181 | 3 |
| Woodwalton | Neural SDE (Gaussian) | 1.36 ± 0.26 | 2.69 ± 0.95 | 4.47 ± 2.46 | 2.392 ± 1.007 | 0.401 ± 0.089 | 0.686 ± 0.337 | 3 |
| Woodwalton | Neural SDE (Student-t) | 2.10 ± 1.76 | 4.89 ± 5.51 | 10.13 ± 13.83 | 7.197 ± 10.199 | 0.419 ± 0.333 | 0.619 ± 0.375 | 3 |
| Woodwalton | Neural SDE (ALD) | 1.08 ± 0.01 | 1.70 ± 0.05 | 2.22 ± 0.17 | 1.375 ± 0.098 | 0.316 ± 0.150 | 0.798 ± 0.334 | 3 |
| Woodwalton | Mean-variance (Gaussian) | 1.24 ± 0.13 | 2.12 ± 0.31 | 2.91 ± 0.56 | 1.834 ± 0.468 | 0.253 ± 0.016 | 0.837 ± 0.248 | 3 |
| Woodwalton | Mean-variance (Student-t) | 1.16 ± 0.10 | 1.97 ± 0.40 | 2.86 ± 1.19 | 2.076 ± 1.559 | 0.318 ± 0.113 | 0.950 ± 0.044 | 3 |
| Woodwalton | Mean-variance (ALD) | 1.09 ± 0.02 | 1.69 ± 0.03 | 2.05 ± 0.02 | 1.482 ± 0.432 | 0.237 ± 0.097 | 0.959 ± 0.060 | 3 |
| Woodwalton | Analytical SDE (Gaussian) | 1.04 | 1.59 | 1.92 | 1.875 | 0.328 | 0.998 | 1 |
| Woodwalton | Analytical SDE (Student-t) | 1.04 | 1.59 | 1.92 | 1.060 | 0.215 | 0.957 | 1 |
| Woodwalton | MDN (mixture) | 1.07 ± 0.01 | 1.72 ± 0.08 | 2.25 ± 0.24 | 2.862 ± 2.640 | 0.337 ± 0.053 | 0.974 ± 0.026 | 3 |
| Woodwalton | Random Forest | 1.66 ± 0.02 | 1.45 ± 0.01 | 1.50 ± 0.02 | — | — | — | 3 |
| Woodwalton | XGBoost | 2.53 ± 0.00 | 2.19 ± 0.00 | 2.11 ± 0.00 | — | — | — | 3 |
| Rosedene | WienerNet-SS (ALD, primary) | 1.58 ± 0.02 | 1.72 ± 0.08 | 2.26 ± 0.21 | 1.131 ± 0.055 | 0.188 ± 0.035 | 0.946 ± 0.039 | 3 |
| Rosedene | WienerNet-SS (Gaussian) | 1.61 ± 0.05 | 1.88 ± 0.18 | 2.57 ± 0.93 | 1.687 ± 0.226 | 0.304 ± 0.016 | 0.991 ± 0.013 | 3 |
| Rosedene | WienerNet-SS (beta-NLL) | 1.59 ± 0.02 | 1.81 ± 0.09 | 2.54 ± 0.66 | 1.394 ± 0.244 | 0.268 ± 0.025 | 0.983 ± 0.013 | 3 |
| Rosedene | WienerNet-SS (Student-t) | 1.58 ± 0.05 | 1.68 ± 0.09 | 2.06 ± 0.23 | 1.226 ± 0.145 | 0.197 ± 0.076 | 0.962 ± 0.007 | 3 |
| Rosedene | WienerNet-SS (mixture) | 1.61 ± 0.08 | 1.75 ± 0.22 | 2.05 ± 0.10 | 1.106 ± 0.123 | 0.224 ± 0.045 | 0.969 ± 0.012 | 3 |
| Rosedene | WienerNet-SS (+residual) | 1.69 ± 0.12 | 2.11 ± 0.39 | 2.87 ± 0.65 | 1.363 ± 0.298 | 0.221 ± 0.086 | 0.948 ± 0.041 | 3 |
| Rosedene | WienerNet-SS (given-diurnal, Wiener) | 1.55 ± 0.00 | 1.58 ± 0.00 | 1.85 ± 0.00 | 1.064 ± 0.098 | 0.225 ± 0.037 | 0.963 ± 0.022 | 3 |
| Rosedene | WienerNet-SS (given-diurnal, state-sp) | 1.55 ± 0.00 | 1.58 ± 0.00 | 1.85 ± 0.00 | 1.154 ± 0.079 | 0.263 ± 0.022 | 0.965 ± 0.015 | 3 |
| Rosedene | Neural SDE (Gaussian) | 1.71 ± 0.11 | 2.22 ± 0.42 | 3.07 ± 0.73 | 1.962 ± 0.468 | 0.297 ± 0.120 | 0.908 ± 0.081 | 3 |
| Rosedene | Neural SDE (Student-t) | 1.59 ± 0.04 | 1.71 ± 0.12 | 2.15 ± 0.27 | 1.442 ± 0.122 | 0.255 ± 0.040 | 0.969 ± 0.025 | 3 |
| Rosedene | Neural SDE (ALD) | 1.69 ± 0.19 | 2.00 ± 0.51 | 2.87 ± 1.32 | 1.316 ± 0.211 | 0.216 ± 0.010 | 0.970 ± 0.018 | 3 |
| Rosedene | Mean-variance (Gaussian) | 1.56 ± 0.09 | 1.71 ± 0.21 | 2.22 ± 0.47 | 1.566 ± 0.105 | 0.264 ± 0.058 | 0.981 ± 0.022 | 3 |
| Rosedene | Mean-variance (Student-t) | 1.61 ± 0.06 | 1.73 ± 0.11 | 2.13 ± 0.14 | 1.829 ± 0.417 | 0.296 ± 0.015 | 0.985 ± 0.014 | 3 |
| Rosedene | Mean-variance (ALD) | 1.59 ± 0.03 | 1.81 ± 0.14 | 2.33 ± 0.31 | 1.487 ± 0.136 | 0.280 ± 0.023 | 0.986 ± 0.020 | 3 |
| Rosedene | Analytical SDE (Gaussian) | 1.55 | 1.59 | 1.80 | 1.544 | 0.310 | 1.000 | 1 |
| Rosedene | Analytical SDE (Student-t) | 1.55 | 1.59 | 1.80 | 0.916 | 0.144 | 0.913 | 1 |
| Rosedene | MDN (mixture) | 1.69 ± 0.10 | 1.91 ± 0.18 | 2.55 ± 0.15 | 1.443 ± 0.162 | 0.197 ± 0.032 | 0.981 ± 0.015 | 3 |
| Rosedene | Random Forest | 1.97 ± 0.01 | 1.25 ± 0.00 | 1.18 ± 0.00 | — | — | — | 3 |
| Rosedene | XGBoost | 2.12 ± 0.00 | 1.33 ± 0.00 | 1.25 ± 0.00 | — | — | — | 3 |
| Redmere 1 | WienerNet-SS (ALD, primary) | 6.07 ± 0.44 | 9.33 ± 0.81 | 4.22 ± 0.57 | 1.905 ± 0.153 | 0.310 ± 0.015 | 0.984 ± 0.004 | 3 |
| Redmere 1 | WienerNet-SS (Gaussian) | 4.64 ± 1.45 | 6.75 ± 2.47 | 3.14 ± 0.63 | 5.846 ± 7.486 | 0.303 ± 0.035 | 0.996 ± 0.001 | 3 |
| Redmere 1 | WienerNet-SS (beta-NLL) | 5.81 ± 2.93 | 8.69 ± 4.73 | 5.22 ± 1.62 | 2.551 ± 0.745 | 0.309 ± 0.063 | 0.978 ± 0.005 | 3 |
| Redmere 1 | WienerNet-SS (Student-t) | 6.23 ± 0.92 | 9.21 ± 1.23 | 3.78 ± 0.46 | 1.879 ± 0.174 | 0.284 ± 0.025 | 0.988 ± 0.004 | 3 |
| Redmere 1 | WienerNet-SS (mixture) | 2.46 ± 1.03 | 3.07 ± 1.53 | 1.90 ± 0.54 | 1.692 ± 0.303 | 0.291 ± 0.025 | 0.999 ± 0.001 | 3 |
| Redmere 1 | WienerNet-SS (+residual) | 6.32 ± 0.14 | 9.43 ± 0.10 | 3.43 ± 0.18 | 4.966 ± 3.971 | 0.315 ± 0.023 | 0.998 ± 0.002 | 3 |
| Redmere 1 | WienerNet-SS (given-diurnal, Wiener) | 1.62 ± 0.00 | 1.81 ± 0.00 | 1.43 ± 0.00 | 1.371 ± 0.094 | 0.331 ± 0.006 | 0.998 ± 0.003 | 3 |
| Redmere 1 | WienerNet-SS (given-diurnal, state-sp) | 1.62 ± 0.00 | 1.81 ± 0.00 | 1.43 ± 0.00 | 1.723 ± 0.239 | 0.329 ± 0.009 | 0.998 ± 0.002 | 3 |
| Redmere 1 | Neural SDE (Gaussian) | 1866.84 ± 2557.67 | 2536.32 ± 3474.88 | 1093.15 ± 1497.63 | 28.370 ± 32.686 | 0.321 ± 0.030 | 0.988 ± 0.016 | 3 |
| Redmere 1 | Neural SDE (Student-t) | 596.84 ± 452.67 | 810.79 ± 615.02 | 349.58 ± 265.10 | 19.756 ± 13.355 | 0.256 ± 0.029 | 0.995 ± 0.003 | 3 |
| Redmere 1 | Neural SDE (ALD) | 138.59 ± 119.80 | 188.05 ± 162.90 | 81.42 ± 69.89 | 4.555 ± 2.798 | 0.290 ± 0.016 | 0.998 ± 0.002 | 3 |
| Redmere 1 | Mean-variance (Gaussian) | 570.20 ± 480.89 | 774.67 ± 653.32 | 333.91 ± 281.57 | 14.604 ± 11.285 | 0.277 ± 0.016 | 0.991 ± 0.006 | 3 |
| Redmere 1 | Mean-variance (Student-t) | 666.42 ± 751.29 | 905.39 ± 1020.73 | 390.30 ± 439.79 | 16.669 ± 17.561 | 0.267 ± 0.030 | 0.997 ± 0.002 | 3 |
| Redmere 1 | Mean-variance (ALD) | 193.70 ± 274.26 | 263.13 ± 372.60 | 113.52 ± 160.53 | 5.769 ± 6.367 | 0.284 ± 0.022 | 0.993 ± 0.004 | 3 |
| Redmere 1 | Analytical SDE (Gaussian) | 1.60 | 1.76 | 1.30 | 1.689 | 0.309 | 1.000 | 1 |
| Redmere 1 | Analytical SDE (Student-t) | 1.60 | 1.76 | 1.30 | 0.788 | 0.180 | 0.972 | 1 |
| Redmere 1 | MDN (mixture) | 5.26 ± 2.31 | 7.06 ± 3.17 | 3.64 ± 0.94 | 1.655 ± 0.190 | 0.265 ± 0.011 | 0.997 ± 0.001 | 3 |
| Redmere 1 | Random Forest | 1.45 ± 0.01 | 1.19 ± 0.02 | 0.90 ± 0.02 | — | — | — | 3 |
| Redmere 1 | XGBoost | 1.52 ± 0.00 | 1.28 ± 0.00 | 0.90 ± 0.00 | — | — | — | 3 |
| Redmere 2 | WienerNet-SS (ALD, primary) | 1.94 ± 0.04 | 2.08 ± 0.14 | 2.20 ± 0.74 | 1.356 ± 0.322 | 0.302 ± 0.035 | 0.988 ± 0.010 | 3 |
| Redmere 2 | WienerNet-SS (Gaussian) | 1.81 ± 0.01 | 1.75 ± 0.04 | 1.85 ± 0.29 | 1.164 ± 0.136 | 0.272 ± 0.021 | 0.989 ± 0.012 | 3 |
| Redmere 2 | WienerNet-SS (beta-NLL) | 1.87 ± 0.07 | 1.96 ± 0.31 | 2.49 ± 1.26 | 1.533 ± 0.443 | 0.269 ± 0.066 | 0.978 ± 0.020 | 3 |
| Redmere 2 | WienerNet-SS (Student-t) | 1.90 ± 0.08 | 1.96 ± 0.16 | 2.12 ± 0.02 | 1.478 ± 0.213 | 0.280 ± 0.019 | 0.995 ± 0.005 | 3 |
| Redmere 2 | WienerNet-SS (mixture) | 1.97 ± 0.23 | 2.11 ± 0.49 | 2.00 ± 0.44 | 1.305 ± 0.335 | 0.263 ± 0.041 | 0.994 ± 0.007 | 3 |
| Redmere 2 | WienerNet-SS (+residual) | 1.91 ± 0.06 | 2.23 ± 0.64 | 2.71 ± 1.39 | 1.777 ± 0.594 | 0.282 ± 0.074 | 0.974 ± 0.040 | 3 |
| Redmere 2 | WienerNet-SS (given-diurnal, Wiener) | 1.82 ± 0.00 | 1.75 ± 0.00 | 1.66 ± 0.00 | 1.292 ± 0.172 | 0.321 ± 0.022 | 0.998 ± 0.002 | 3 |
| Redmere 2 | WienerNet-SS (given-diurnal, state-sp) | 1.82 ± 0.00 | 1.75 ± 0.00 | 1.66 ± 0.00 | 1.336 ± 0.299 | 0.343 ± 0.033 | 0.996 ± 0.007 | 3 |
| Redmere 2 | Neural SDE (Gaussian) | 1.87 ± 0.06 | 1.97 ± 0.19 | 2.24 ± 0.46 | 1.444 ± 0.344 | 0.295 ± 0.060 | 0.984 ± 0.020 | 3 |
| Redmere 2 | Neural SDE (Student-t) | 2.14 ± 0.28 | 2.69 ± 0.67 | 2.77 ± 1.00 | 1.626 ± 0.419 | 0.301 ± 0.022 | 0.994 ± 0.009 | 3 |
| Redmere 2 | Neural SDE (ALD) | 2.10 ± 0.30 | 2.80 ± 1.20 | 3.92 ± 3.26 | 2.000 ± 0.888 | 0.295 ± 0.076 | 0.987 ± 0.021 | 3 |
| Redmere 2 | Mean-variance (Gaussian) | 1.87 ± 0.02 | 2.01 ± 0.28 | 2.26 ± 0.73 | 1.557 ± 0.254 | 0.209 ± 0.035 | 0.999 ± 0.001 | 3 |
| Redmere 2 | Mean-variance (Student-t) | 1.80 ± 0.01 | 1.77 ± 0.05 | 1.66 ± 0.10 | 1.347 ± 0.300 | 0.249 ± 0.055 | 0.999 ± 0.001 | 3 |
| Redmere 2 | Mean-variance (ALD) | 1.94 ± 0.19 | 2.21 ± 0.71 | 2.26 ± 0.94 | 1.338 ± 0.414 | 0.258 ± 0.082 | 0.991 ± 0.010 | 3 |
| Redmere 2 | Analytical SDE (Gaussian) | 1.82 | 1.75 | 1.71 | 1.724 | 0.307 | 0.994 | 1 |
| Redmere 2 | Analytical SDE (Student-t) | 1.82 | 1.75 | 1.71 | 0.886 | 0.180 | 0.947 | 1 |
| Redmere 2 | MDN (mixture) | 1.95 ± 0.12 | 2.33 ± 0.64 | 2.59 ± 1.22 | 1.415 ± 0.591 | 0.193 ± 0.024 | 0.986 ± 0.009 | 3 |
| Redmere 2 | Random Forest | 1.68 ± 0.01 | 1.50 ± 0.02 | 1.24 ± 0.01 | — | — | — | 3 |
| Redmere 2 | XGBoost | 1.85 ± 0.00 | 1.59 ± 0.00 | 1.34 ± 0.00 | — | — | — | 3 |
| Great Fen | WienerNet-SS (ALD, primary) | 1.83 ± 0.12 | 2.29 ± 0.45 | 2.36 ± 0.54 | 1.658 ± 0.363 | 0.294 ± 0.039 | 0.993 ± 0.005 | 3 |
| Great Fen | WienerNet-SS (Gaussian) | 1.73 ± 0.01 | 1.98 ± 0.09 | 2.22 ± 0.21 | 1.648 ± 0.291 | 0.234 ± 0.050 | 0.987 ± 0.010 | 3 |
| Great Fen | WienerNet-SS (beta-NLL) | 1.97 ± 0.25 | 2.85 ± 0.89 | 3.93 ± 1.55 | 2.179 ± 0.696 | 0.314 ± 0.091 | 0.987 ± 0.007 | 3 |
| Great Fen | WienerNet-SS (Student-t) | 1.89 ± 0.35 | 2.50 ± 0.99 | 2.61 ± 1.14 | 1.785 ± 0.336 | 0.199 ± 0.027 | 0.932 ± 0.076 | 3 |
| Great Fen | WienerNet-SS (mixture) | 1.70 ± 0.02 | 1.81 ± 0.05 | 1.77 ± 0.11 | 1.523 ± 0.315 | 0.239 ± 0.040 | 0.995 ± 0.008 | 3 |
| Great Fen | WienerNet-SS (+residual) | 2.11 ± 0.31 | 3.29 ± 0.67 | 4.34 ± 1.36 | 2.346 ± 0.882 | 0.288 ± 0.071 | 0.966 ± 0.018 | 3 |
| Great Fen | WienerNet-SS (given-diurnal, Wiener) | 1.70 ± 0.00 | 1.79 ± 0.00 | 1.71 ± 0.00 | 1.647 ± 0.356 | 0.300 ± 0.053 | 0.992 ± 0.006 | 3 |
| Great Fen | WienerNet-SS (given-diurnal, state-sp) | 1.70 ± 0.00 | 1.79 ± 0.00 | 1.71 ± 0.00 | 1.556 ± 0.073 | 0.308 ± 0.015 | 0.986 ± 0.012 | 3 |
| Great Fen | Neural SDE (Gaussian) | 2.25 ± 0.50 | 3.70 ± 1.72 | 6.28 ± 4.17 | 3.970 ± 2.579 | 0.289 ± 0.072 | 0.957 ± 0.059 | 3 |
| Great Fen | Neural SDE (Student-t) | 1.74 ± 0.09 | 2.09 ± 0.32 | 2.22 ± 0.38 | 1.603 ± 0.180 | 0.226 ± 0.028 | 0.981 ± 0.022 | 3 |
| Great Fen | Neural SDE (ALD) | 1.84 ± 0.11 | 2.39 ± 0.47 | 2.96 ± 1.20 | 1.765 ± 0.384 | 0.267 ± 0.019 | 0.955 ± 0.065 | 3 |
| Great Fen | Mean-variance (Gaussian) | 1.88 ± 0.12 | 2.54 ± 0.32 | 3.31 ± 0.35 | 1.741 ± 0.155 | 0.247 ± 0.054 | 0.970 ± 0.019 | 3 |
| Great Fen | Mean-variance (Student-t) | 1.84 ± 0.12 | 2.40 ± 0.46 | 2.83 ± 0.75 | 1.686 ± 0.196 | 0.255 ± 0.045 | 0.986 ± 0.010 | 3 |
| Great Fen | Mean-variance (ALD) | 1.73 ± 0.04 | 2.11 ± 0.11 | 2.43 ± 0.51 | 1.605 ± 0.187 | 0.202 ± 0.065 | 0.976 ± 0.024 | 3 |
| Great Fen | Analytical SDE (Gaussian) | 1.69 | 1.77 | 1.68 | 1.696 | 0.262 | 1.000 | 1 |
| Great Fen | Analytical SDE (Student-t) | 1.69 | 1.77 | 1.68 | 0.951 | 0.103 | 0.910 | 1 |
| Great Fen | MDN (mixture) | 1.72 ± 0.02 | 1.93 ± 0.16 | 1.89 ± 0.24 | 1.341 ± 0.236 | 0.214 ± 0.021 | 0.994 ± 0.008 | 3 |
| Great Fen | Random Forest | 1.91 ± 0.01 | 1.45 ± 0.00 | 1.04 ± 0.01 | — | — | — | 3 |
| Great Fen | XGBoost | 1.97 ± 0.00 | 1.45 ± 0.00 | 1.06 ± 0.00 | — | — | — | 3 |

**Stage 2 — autoregressive gap-fill, per held-out site.** mean ± 1 SD over the 3 seeds (0/1/42); n = 1 entries are the calibrated, seed-independent models. Raw predictive band; metric conventions as in the pooled Stage-2 tables.  **Seed 42 uses the LR-fixed re-run** (`reduce_on_plateau`, stepped on the TRAIN loss) for every gradient-trained arm, after seed 42 was found to diverge under the original constant-LR protocol; seeds 0 and 1 remain on the original protocol, so the ± for a given row mixes two training protocols and is provisional.

