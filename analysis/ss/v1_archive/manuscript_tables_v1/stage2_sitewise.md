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

