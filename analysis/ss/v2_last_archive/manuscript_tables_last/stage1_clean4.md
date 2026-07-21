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

