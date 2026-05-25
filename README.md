# WienerNet

Physics-informed autoencoder for forecasting nighttime ecosystem CO₂ fluxes
(NEE) at UK East Anglia flux towers. The dynamics are modelled as a
Wiener-process SDE with an analytic Lloyd-Taylor drift and a learned noise
term sampled via VAE-style reparameterisation.

Reference: Houpert C., Zia S. et al., *Physics-informed VAE for enhancing
forecast reliability of CO₂ emissions from agricultural farms.*

---

## Quickstart

```bash
# 1. Environment
conda env create -f env.yaml
conda activate pytorch
pip install hydra-core mlflow rich pytest    # if not already in env

# 2. Train the headline configuration
python scripts/train.py experiment=paper_main

# 3. Evaluate the resulting checkpoint
python scripts/evaluate.py --run outputs/*paper_main*

# 4. View metrics in the analysis notebook
jupyter notebook notebooks/02_results_analysis.ipynb
```

## Directory layout

```
configs/           Hydra config tree
wienernet/         Main package (models, losses, training, evaluation, data, utils, physics)
data_pipeline/    Site preprocessing + Lloyd-Taylor parameter fitting
scripts/           train.py + evaluate.py (Hydra entrypoints)
notebooks/         Slim analysis notebook (12 cells)
tests/             pytest suite (73 tests)
data_manipulation/  Parquet files + active parameter-estimation notebooks
.archive/          Pre-refactor code preserved for reference
```

## Training

### Torch models (WienerNet + AE/VAE family)
```bash
# Pick a model variant + loss profile
python scripts/train.py model=piae_sde_sampling loss=full
python scripts/train.py model=piae_sde_reg_sampling loss=reg
python scripts/train.py model=ae loss=ae
python scripts/train.py model=vae loss=vae
python scripts/train.py model=pivae_sde_sampling loss=full

# Override any field on the CLI
python scripts/train.py model.latent_dim=64 training.optimizer.lr=5e-4 seed=88

# Multirun (Hydra basic launcher)
python scripts/train.py -m seed=0,1,2,41,88,128,256          # 7-seed sweep
python scripts/train.py -m model=piae_sde_sampling,ae,vae    # 3 variants

# Pre-composed experiments
python scripts/train.py experiment=paper_main
python scripts/train.py -m experiment=seed_sweep
```

### Non-torch baselines (Random Forest, XGBoost)

These baselines share the data pipeline + evaluation flow but use scikit-learn /
XGBoost directly (no torch). A separate entrypoint avoids cluttering the
torch trainer with sklearn branching:

```bash
python scripts/train_baseline.py                                # default: rf, seed 42
python scripts/train_baseline.py model=xgb seed=88
python scripts/train_baseline.py -m model=rf,xgb seed=0,1,2,41,88,128,256

# Override hyperparameters on the CLI
python scripts/train_baseline.py model=rf model.n_estimators=300 model.max_depth=10
python scripts/train_baseline.py model=xgb model.learning_rate=0.05 model.n_estimators=500
```

Each run produces `outputs/<timestamp>_<variant>_seed<N>/` containing:
- `checkpoints/model.joblib` — the fitted estimator (re-loadable via `RandomForestBaseline.load(path)` or `XGBoostBaseline.load(path)`)
- `metrics.json` — full per-resolution metric table for NEE
- `predictions.parquet` — gt + preds + test_df for downstream plotting
- `feature_importance.csv` — sklearn's per-feature importance ranking
- `config.json`, `run.log`, `scaler.pkl`

Each run produces `outputs/<timestamp>_<run_name>/` containing:
- `config.json` — the fully-resolved Hydra config
- `checkpoints/best.pth` + `last.pth` — model + optimizer + scheduler + RNG state
- `history.json` — per-epoch loss components
- `scaler.pkl` — fitted StandardScaler
- `tensorboard/` — TB event files
- `run.log`

If `mlflow.enabled=true` (default), every run is also logged to `./mlruns/`.
View with `mlflow ui`.

## Evaluation

```bash
# Single run
python scripts/evaluate.py --run outputs/X

# Cross-seed summary
python scripts/evaluate.py --run outputs/*paper_main*
```

Produces:
- `outputs/X/metrics/predictions.parquet` — gt + preds + test_df columns
- `outputs/X/metrics/per_site.csv` — per-site metric breakdown
- `outputs/evaluation_summary/long.csv` — one row per (run, resolution, target, metric)
- `outputs/evaluation_summary/summary.csv` — median + MAD across runs

## Data pipeline

```bash
# Raw site Excel → canonical parquet → Lloyd-Taylor parameter fits
python -m data_pipeline.cli preprocess --site woodwalton
python -m data_pipeline.cli partition --site woodwalton --strategy astral
python -m data_pipeline.cli all --site rosedene --strategy column
```

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest          # 73 tests, ~3s
```

The `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` workaround is necessary because of a
broken langsmith plugin in the shared conda env (incompatible with pydantic v1).

## Architecture

The unified `WienerNetModel` has composable heads controlled by a YAML config:

| Variant | latent_reparameterize | predict_drift | heads (k, temp_derivative, noise) | Entrypoint |
|---|---|---|---|---|
| `piae_sde_sampling` | False | True | k+temp+noise (the full WienerNet) | `train.py` |
| `piae_sde_reg_sampling` | False | False | k+temp+noise (ablation: no SDE step) | `train.py` |
| `pivae_sde_sampling` | True | True | k+temp+noise (VAE latent) | `train.py` |
| `ae` | False | False | k only | `train.py` |
| `vae` | True | False | k only | `train.py` |
| `rf` | — | — | sklearn RandomForestRegressor on flat features | `train_baseline.py` |
| `xgb` | — | — | xgboost.XGBRegressor on flat features | `train_baseline.py` |

The 5 hand-written torch model+trainer pairs from the original codebase are
reduced to one model class + one Trainer + 5 YAML presets. The two non-torch
baselines (RF, XGB) share the same data pipeline and evaluation flow.

## Citing

Citation guidance will be added when the paper is published.

## License

Research code; license TBD.
