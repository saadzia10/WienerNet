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

