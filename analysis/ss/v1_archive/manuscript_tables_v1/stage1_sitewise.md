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

