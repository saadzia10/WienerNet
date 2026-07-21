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

