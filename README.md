# WienerNet

WienerNet, a stochastic physics-encoded neural network, generalises physics-encoded modelling to Wiener-type SDEs. It embeds a single Euler-Maruyama step in the graph, forecasting the full conditional law of the next observation through five interpretable blocks. This approach confers robustness and provides interpretable attributions for predicted variance.

Reference: Zia M.S., Houpert C., et al., *WienerNet: Embedding the Physics of Noise in Stochastic Digital Twins*

---

## Install

```bash
conda env create -f env.yaml
conda activate pytorch
```

## Quickstart

```bash
python scripts/train.py                          # default model + loss, seed 42
python scripts/evaluate.py --run outputs/<run>   # metrics + predictions
```

Every run writes `outputs/<timestamp>_<run_name>/` containing `config.json`,
`checkpoints/{best,last}.pth`, `history.json`, `scaler.pkl`, `run.log`,
`tensorboard/`, and Hydra's `.hydra/`. With `mlflow.enabled=true` (default) runs
are also logged to `./mlruns/` — view with `mlflow ui`.

## Training

Pick a model and a loss profile from the config groups; override any field on the
command line.

```bash
python scripts/train.py model=piae_sde_sampling loss=full
python scripts/train.py model=piae_increment_residual loss=nll_ald
python scripts/train.py model=ae loss=ae

python scripts/train.py model.latent_dim=64 training.optimizer.lr=5e-4 seed=88
```

**Models** (`configs/model/`)

| Group | Variants |
|---|---|
| WienerNet (level) | `piae_sde_sampling`, `piae_sde_reg_sampling`, `pivae_sde_sampling` |
| WienerNet (increment / SS) | `piae_increment`, `piae_increment_residual`, `piae_reg_increment` |
| Process baselines | `analytical_sde`, `neural_sde`, `hetero_mlp`, `hetero_mdn` |
| Autoencoder baselines | `ae`, `vae` |
| Tree baselines | `rf`, `xgb` (via `train_baseline.py`) |

**Losses** (`configs/loss/`): `full`, `reg`, `ae`, `vae`, `increment`,
`increment_residual`, and the likelihood profiles `nll_gaussian`, `nll_beta`,
`nll_student_t`, `nll_ald`. Loss weights are config-driven — set a term's weight
to `0` to ablate it.

**Pre-composed experiments** (`configs/experiment/`), e.g.:

```bash
python scripts/train.py experiment=paper_main
python scripts/train.py experiment=piae_increment_residual_nll
python scripts/train.py -m experiment=seed_sweep          # 7 seeds
```

### Splits

`data.split_strategy` is one of `site_fraction` (default), `year`, or
`site_holdout`. For leave-one-site-out, hold a tower out and train on the rest:

```bash
python scripts/train.py data.split_strategy=site_holdout data.holdout_site=woodwalton
```

Set `data.val_frac` to carve a validation split from the *training* sites; when
it is unset the test loader is used as the val monitor.

### Tree baselines

RF/XGB use scikit-learn directly, so they have their own entrypoint:

```bash
python scripts/train_baseline.py model=rf seed=42
python scripts/train_baseline.py model=xgb model.n_estimators=500 model.learning_rate=0.05
python scripts/train_baseline.py -m model=rf,xgb seed=0,1,2
```

These write `checkpoints/model.joblib` (reload with
`RandomForestBaseline.load(path)` / `XGBoostBaseline.load(path)`),
`metrics.json`, `predictions.parquet`, and `feature_importance.csv`.

### Sweeps

```bash
python scripts/train.py -m seed=0,1,2,41,88,128,256
python scripts/train.py -m model=piae_sde_sampling,ae,vae loss=full,ae,vae
```

## Evaluation

```bash
python scripts/evaluate.py --run outputs/X                 # single run
python scripts/evaluate.py --run outputs/X outputs/Y       # cross-seed summary
```

Produces `metrics/predictions.parquet`, `metrics/per_site.csv`, and
`metrics/probabilistic.json` per run, plus `outputs/evaluation_summary/`
(`long.csv`, `summary.csv`) across runs. `scripts/evaluate.py` handles torch
runs; tree baselines write their metrics during training.

## Data pipeline

```bash
python -m data_pipeline.cli preprocess --site woodwalton
python -m data_pipeline.cli partition  --site woodwalton --strategy astral
python -m data_pipeline.cli all        --site rosedene   --strategy column
```

Raw site spreadsheets → canonical parquet → Lloyd–Taylor parameter fits →
`final_night_data.parquet` / `final_day_data.parquet`.

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest       # 241 tests
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required: the conda env ships a langsmith
pytest plugin that is incompatible with the installed pydantic v1.

## Layout

```
configs/            Hydra config tree (data, model, training, loss, experiment)
wienernet/          Package: models, losses, physics, training, evaluation, data, baselines, utils
data_pipeline/      Site preprocessing + Lloyd-Taylor parameter fitting
scripts/            train.py, train_baseline.py, evaluate.py + sweep drivers
analysis/           Post-hoc analysis scripts and figure builders
notebooks/          Results-analysis notebook
tests/              pytest suite
outputs/, multirun/ Hydra run outputs
.archive/           Pre-refactor code, kept for reference
```

## License

Research code; license TBD.
