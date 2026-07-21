# Nighttime NEE residual: error-structure analysis

Exploratory analysis of the physics residual `NEE_obs − Reco(Ta; E0, rb)` and of
the trained SDE noise head. All read-only w.r.t. the training pipeline.

## Figures
Each plot is a standalone vector PDF with a draft-caption `.txt` sidecar
(per the plot-generation conventions in `CLAUDE.md`); LaTeX `subfigure` composes
the panels. The residual is heteroscedastic (SD ≈ 0.24 + 0.30·flux, R²=0.99) and
double-exponential (Laplace), with ~1/√N error cancellation under aggregation.
- `figures/fig_error_a_residual_density.pdf` — residual density vs Gaussian/Laplace fits.
- `figures/fig_error_b_tail_exceedance.pdf` — tail exceedance P(|z|>k) tracks Laplace.
- `figures/fig_error_c_heteroscedasticity.pdf` — local residual scale ∝ predicted flux.
- `figures/fig_error_d_aggregation_cancellation.pdf` — relative uncertainty vs aggregation.

Noise-head diagnostics (whether the trained β-NLL / Student-t heads reproduce that
structure — heteroscedastic ✓ but too shallow; Student-t learns a ν≈3 heavy tail,
β-NLL stays Gaussian):
- `figures/fig_noise_a_scale_vs_flux.pdf` — predicted σ vs flux vs empirical scale.
- `figures/fig_noise_b_whitened_tail.pdf` — whitened-residual tail still heavier than assumed.

## Report
- [error_structure_report.html](error_structure_report.html) — manuscript-grade brief
  (open in a browser; figure embedded).

## Scripts (`scripts/`, run from repo root with `conda activate pytorch`)
| script | what it produces |
|---|---|
| `nee_driver_correlations.py` | driver↔NEE/dNEE correlations, per-site coverage |
| `temp_change_timescales.py`  | Ta/Tsoil change vs timescale; dT↔dNEE across scales |
| `residual_error_structure.py`| heteroscedasticity + Gaussian-vs-Laplace + aggregation |
| `make_figures.py`            | builds the 4 `fig_error_*.pdf` panels (+ caption `.txt`) |
| `noise_head_test.py`         | tests trained σ-head vs the 0.30 slope + Laplace tail |
| `make_noise_fig.py`          | builds the 2 `fig_noise_*.pdf` panels (+ caption `.txt`) |
| `build_report.py`            | assembles `error_structure_report.html` |

Note: `noise_head_test.py` / `make_noise_fig.py` read the nll_bakeoff predictions
(`pred_noise_stds` = per-min σ). Point `BASE` at a run dir containing
`{method}/metrics/predictions.parquet` if the cached predictions are gone.
