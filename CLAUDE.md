# CLAUDE.md

Context for future Claude Code sessions in this repository.

## What this project is

Implementation of **WienerNet** — a physics-informed autoencoder that models
nighttime CO₂ flux (NEE) as a Wiener-process SDE. The drift term is the
analytic derivative of the Lloyd-Taylor respiration model; the noise term is
sampled via a VAE-style reparameterisation head. Trained on UK East Anglia
flux tower data (Rosedene, Redmere 1+2, Great Fen).

The original codebase was a single 7900-line notebook plus 5 copy-pasted
trainer/model file pairs. The 2026-05 refactor (this layout) collapsed all of
that into a Hydra-driven training pipeline with composable model heads, one
unified Trainer, and a clean evaluation module.

## Repo layout

```
configs/                 Hydra config tree (data, model, training, loss, experiment)
wienernet/               Main package
  data/                  ClimateDataset, feature engineering, splits, build_dataloaders()
  models/                Unified WienerNetModel + 5 named presets (torch)
  losses/                MMD + composite loss assembly
  physics/               Torch Lloyd-Taylor (used inside forward())
  training/              Trainer, build_optimizer, build_scheduler
  evaluation/            Metrics + temporal aggregations + cross-seed reporter
  baselines/             RandomForest + XGBoost wrappers (non-torch baselines)
  utils/                 set_seed_globally, save/load_checkpoint, setup_logging, make_run_dir
data_pipeline/           Site XLSX -> processed_data.parquet -> final_night/day_data.parquet
scripts/                 train.py, train_baseline.py, evaluate.py (Hydra entrypoints)
notebooks/               02_results_analysis.ipynb (thin orchestration; 12 cells)
tests/                   85 pytest tests across 5 test_*.py modules (+ conftest.py)
.archive/                Pre-refactor source + notebooks + old run outputs (incl. piae_sde/)
data_manipulation/       Parquet files + 3 active notebooks (Parameter Estimation, Preprocess)
outputs/                 Hydra single-run outputs
multirun/                Hydra multirun outputs
mlruns/                  MLflow tracking server (local)
```

**Note**: there is NO active `piae_sde/` directory at repo root — it was
archived in Phase 7.9. Look in `.archive/piae_sde/` if you need the original
night/*.py, the old notebooks, or the pre-refactor checkpoint output dirs.

## How to do common things

### Train one torch model
```bash
conda activate pytorch
python scripts/train.py                                # default: piae_sde_sampling, seed 42
python scripts/train.py model=ae loss=ae               # baseline AE
python scripts/train.py model=piae_sde_sampling seed=88 training.num_epochs=200
python scripts/train.py experiment=paper_main          # the published headline config
```

### Train an RF / XGB baseline
Separate entrypoint because these models don't fit the torch Trainer's API.
```bash
python scripts/train_baseline.py model=rf seed=42
python scripts/train_baseline.py model=xgb model.n_estimators=500 model.learning_rate=0.05
python scripts/train_baseline.py -m model=rf,xgb seed=0,1,2,41,88,128,256
```
Output dir contains `checkpoints/model.joblib` (re-loadable via
`RandomForestBaseline.load(path)` / `XGBoostBaseline.load(path)`),
`metrics.json`, `predictions.parquet`, and `feature_importance.csv`.

### Multirun (torch)
```bash
python scripts/train.py -m experiment=seed_sweep                       # 7 seeds
python scripts/train.py -m model=piae_sde_sampling,ae,vae seed=0,1,2,42 loss=full,ae,vae
```

### Evaluate a run (or many)
```bash
python scripts/evaluate.py --run outputs/2026-05-25_*piae_sde_sampling*
python scripts/evaluate.py --run outputs/X/ outputs/Y/ outputs/Z/   # cross-seed summary
```

### View MLflow
```bash
mlflow ui                                              # default: http://127.0.0.1:5000
```

### Run tests
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest      # 73 tests, ~3s
```

The `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed because the pytorch conda
env has a broken langsmith pytest plugin (incompatible with the installed
pydantic v1). Don't waste time trying to fix it — just use the env var.

### Data pipeline (raw site data → training-ready parquets)
```bash
python -m data_pipeline.cli preprocess --site woodwalton
python -m data_pipeline.cli partition --site woodwalton --strategy astral
python -m data_pipeline.cli all --site rosedene --strategy column   # full Rosedene flow
```

## Conventions

- **Run directories**: one per training run, under `outputs/<timestamp>_<run_name>/`.
  Each has: `checkpoints/{best,last}.pth`, `config.json`, `history.json`,
  `scaler.pkl`, `run.log`, `tensorboard/`, and Hydra's `.hydra/`.
- **All hyperparameters live in `configs/`** — never hardcode in scripts.
- **Loss weights are config-driven**. To ablate a term, set its weight to 0.
- **Submodule names** in WienerNetModel match the original night/*.py so old
  checkpoints weight-load directly (except VAE which renamed fc_mu→latent_mu).
- **Tests are mandatory before merging refactors.** Run pytest first.

## Plot generation conventions

These apply to **every** plot produced in this repo (analysis scripts, notebooks,
figure builders). The target is always a print figure in a LaTeX manuscript.

- **No titles.** Never call `plt.title()`, `ax.set_title()`, `plt.suptitle()`, or
  any figure-level title. The LaTeX caption describes the figure; the image must
  not repeat it. (Panel *letters* A/B/C are also out — LaTeX `subcaption` adds
  them; see the one-file-per-plot rule below.)
- **Always label axes with units.** e.g. `"Latency (ms)"`, `"Predicted flux Reco
  (µmol m⁻² s⁻¹)"` — never a bare quantity name, never an unlabeled axis.
- **Readable print fonts.** Axis labels, tick labels, and legends must stay legible
  at column width — roughly **8–9 pt minimum after LaTeX scales the image down**.
  Do not ship matplotlib's tiny defaults. Size the figure for its final placement
  (a full-width figure shrunk into one column makes 10 pt text ~3 pt — too small).
- **One plot per file.** Save each plot as its own standalone file. Do **not**
  build combined multi-panel images (`plt.subplots(2,2)` merging separate
  experiments, PIL compositing, etc.). Generate each panel separately and let
  LaTeX `subfigure`/`subcaption` handle layout and panel labels.
- **Vector by default.** Export **PDF or SVG** so it stays sharp when scaled. Use
  PNG *only* when the plot has heavy rasterized content (dense scatter/heatmap),
  and then at **≥300 DPI**. Embed editable fonts for PDF: `pdf.fonttype: 42`.
- **Tight bounding box.** Always save with `bbox_inches="tight"` (no excess
  whitespace around the plot).
- **Consistent, colorblind-friendly palette.** Use a fixed palette across all plots
  in the same paper (e.g. `tab10` or `viridis`) — never default random per-plot
  colors. Keep line widths, marker sizes, and legend placement consistent across
  plots that appear together.
- **Descriptive, groupable filenames.** e.g. `fig3_latency_vs_load.pdf`,
  `fig3_throughput_vs_load.pdf` — so related panels sort/reference together.
- **Emit a draft caption per plot.** After saving `<name>.pdf` (or `.svg`/`.png`),
  print a suggested 1–2 sentence caption to the console **and** write it to a
  companion `<name>.txt` (e.g. `fig3_latency_vs_load.pdf` → `fig3_latency_vs_load.txt`)
  describing what the plot shows and its key takeaway, for later editing in LaTeX.

## Known gotchas — read before debugging

### 1. NaN dropna requirement
The original notebook silently propagated NaN through StandardScaler. The
new `build_dataloaders` requires *all driver columns* to be non-NaN, which
drops the combined dataset from 56k → 40k rows. If you see far fewer
training samples than the notebook reported, this is why. The notebook's
numbers were partially contaminated by NaN.

### 2. Physics formula — Reichstein convention
Lloyd-Taylor uses T0 = -46.02 °C. The codebase stores T0 as a positive
46.02 and writes the formula with `(T + T0)` rather than `(T - T0)`. Both
the partitioner (`data_pipeline.partitioning.lloyd_taylor`) and the
in-graph torch version (`wienernet.physics.lloyd_taylor.reco`) use this
convention. If you ever see `(T - self.T0)` in the codebase, that's a
regression to the pre-refactor bug — fix it.

### 3. VAE / PIVAE can't overfit a single batch to ~0
Their `latent_mu`/`latent_logvar` use Tanh, so logvar is bounded in
[-1, 1]. The reparameterisation noise is structurally bounded below,
which floors the loss. The overfit-single-batch test uses a 1.3× threshold
for these variants instead of the 10× used for deterministic models.

### 4. Existing checkpoints predate the physics fix
The pre-refactor `Final Experiments Other Sites/` and `seed_experiments/`
checkpoints were trained with `(T - T0)` instead of `(T + T0)`. They load
architecturally but their weights are conditioned on the wrong physics
formula. Don't trust their predictions; retrain from scratch.

### 5. MMDLoss requires equal batch sizes
`source.size(0) == target.size(0)`. This is preserved from the original
implementation. If you need unequal batches, sub-sample.

### 6. Hydra and chdir
`hydra.job.chdir: false` is set in `configs/config.yaml` so cwd stays at
the repo root. Otherwise relative paths to `data_manipulation/` break.

## Where the user spent attention

The user is the PhD researcher who runs the experiments. They care about:
- **Reproducibility** — every published number must be re-derivable from
  config + git SHA + data hash.
- **Iteration speed** — change a hyperparameter in 5 seconds, not 5 minutes.
- **Catching scientific errors** — the audit found 3 HIGH bugs the original
  pipeline was hiding (T_test leakage, broken pivae import, physics sign
  mismatch). Always investigate when something looks off; don't assume the
  notebook is right.

## What's NOT done

- `data_pipeline` notebook orchestration scripts: the 3 active notebooks
  in `data_manipulation/` still use their original logic. They can be
  rewritten as thin wrappers around the modules, but the modules already
  produce bit-identical output (verified on Rosedene 2018).
- A `scripts/sweep.py` convenience wrapper for multirun was discussed in
  the plan but not implemented. `python scripts/train.py -m experiment=seed_sweep`
  works directly.
- `scripts/evaluate.py` only handles torch runs. Baseline runs (RF/XGB) write
  their own metrics during training, but a unified cross-runtype evaluation
  script that also picks up `.joblib` checkpoints would be nice.
- The slim analysis notebook (`notebooks/02_results_analysis.ipynb`) doesn't
  yet have a "baseline vs WienerNet" comparison plot. The flat-table format
  in `evaluation_summary/long.csv` makes this a 1-cell addition when needed.
