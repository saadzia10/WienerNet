# Nighttime NEE residual: error-structure analysis

Exploratory analysis of the physics residual `NEE_obs − Reco(Ta; E0, rb)` and of
the trained SDE noise head. All read-only w.r.t. the training pipeline.

## Figures
- [figures/fig_error_structure.png](figures/fig_error_structure.png) — the residual is
  heteroscedastic (SD ≈ 0.24 + 0.30·flux, R²=0.99) and double-exponential (Laplace),
  with ~1/√N error cancellation under aggregation.
- [figures/fig_noise_head.png](figures/fig_noise_head.png) — whether the trained β-NLL /
  Student-t noise heads reproduce that structure (heteroscedastic ✓ but too shallow;
  Student-t learns ν=3.16 heavy tail, β-NLL stays Gaussian).

## Report
- [error_structure_report.html](error_structure_report.html) — manuscript-grade brief
  (open in a browser; figure embedded).

## Scripts (`scripts/`, run from repo root with `conda activate pytorch`)
| script | what it produces |
|---|---|
| `nee_driver_correlations.py` | driver↔NEE/dNEE correlations, per-site coverage |
| `temp_change_timescales.py`  | Ta/Tsoil change vs timescale; dT↔dNEE across scales |
| `residual_error_structure.py`| heteroscedasticity + Gaussian-vs-Laplace + aggregation |
| `make_figures.py`            | builds `fig_error_structure.png` |
| `noise_head_test.py`         | tests trained σ-head vs the 0.30 slope + Laplace tail |
| `make_noise_fig.py`          | builds `fig_noise_head.png` |
| `build_report.py`            | assembles `error_structure_report.html` |

Note: `noise_head_test.py` / `make_noise_fig.py` read the nll_bakeoff predictions
(`pred_noise_stds` = per-min σ). Point `BASE` at a run dir containing
`{method}/metrics/predictions.parquet` if the cached predictions are gone.
