"""Per-site cross-seed aggregation for any run type (torch or RF/XGB baseline).

Both torch and baseline runs write a `predictions.parquet` containing a `site`
column plus `gt_<target>` / `pred_<target>` columns. This script recomputes
per-site metrics from those predictions (so torch and baselines are scored
identically), then runs them through the same reporter as the global summary —
giving per-site `long`, robust `summary` (median + MAD), and `parametric_summary`
(mean ± SD/SEM + t-based 95% CI), all grouped by (variant, site, target, metric).

Why recompute instead of reading per_site.csv: only torch runs wrote one; the
baseline runs left their per-site breakdown uncomputed. Recomputing from the
shared predictions.parquet covers every run with one code path.

Usage:
    # One variant across seeds
    python scripts/aggregate_per_site.py --run multirun/2026-06-06_20-25-48/piae_sde_sampling_seed*/

    # All methods together -> one table, a row per (variant, site, target, metric)
    python scripts/aggregate_per_site.py \
        --run multirun/2026-06-06_20-25-48/piae_sde_sampling_seed*/ \
              multirun/2026-06-06_20-33-22/rf_seed*/ \
              multirun/2026-06-06_20-33-41/xgb_seed*/ \
        --out multirun/per_site_summary

Writes (to --out):
    per_site_long.csv               — one row per (run, site, target, metric)
    per_site_summary.csv            — n_runs, median, mad, q05, q95
    per_site_parametric_summary.csv — n_runs, mean, sd, sem, ci95_lo, ci95_hi
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wienernet.evaluation import parametric_summary, standard_errors
from wienernet.evaluation.metrics import evaluate_predictions
from wienernet.utils import get_logger, setup_logging

log = get_logger("scripts.aggregate_per_site")

GROUP_COLS = ("variant", "site", "target", "metric")


def _load_config(run_dir: Path):
    hydra_path = run_dir / ".hydra" / "config.yaml"
    if hydra_path.exists():
        return OmegaConf.load(hydra_path)
    json_path = run_dir / "config.json"
    if json_path.exists():
        return OmegaConf.create(json.loads(json_path.read_text()))
    raise FileNotFoundError(f"No config found in {run_dir}")


def _find_predictions(run_dir: Path) -> Path | None:
    """Torch runs write metrics/predictions.parquet; baselines write it at root."""
    for cand in (run_dir / "metrics" / "predictions.parquet", run_dir / "predictions.parquet"):
        if cand.exists():
            return cand
    return None


def _per_site_rows(preds_df: pd.DataFrame, run_meta: dict) -> list[dict]:
    """Recompute per-site metrics for every target shared by gt_/pred_ columns."""
    gt = {c[len("gt_"):]: preds_df[c].to_numpy()
          for c in preds_df.columns if c.startswith("gt_")}
    preds = {c[len("pred_"):]: preds_df[c].to_numpy()
             for c in preds_df.columns if c.startswith("pred_")}
    targets = [t for t in ("nee", "bnee", "E0", "rb", "dtemp", "f")
               if t in gt and t in preds]

    rows: list[dict] = []
    if "site" not in preds_df.columns:
        log.warning("no `site` column in predictions for %s — skipping", run_meta["run"])
        return rows

    for site, idx in preds_df.groupby("site").groups.items():
        mask = np.zeros(len(preds_df), dtype=bool)
        mask[idx] = True
        site_metrics = evaluate_predictions(
            gt, preds, targets=targets, mask=mask, include_mmd=True, mmd_subsample=2000,
        )
        for target, mdict in site_metrics.items():
            n = mdict.get("n")
            for metric, value in mdict.items():
                if metric == "n" or value is None:
                    continue
                rows.append({
                    **run_meta,
                    "site": site,
                    "target": target,
                    "metric": metric,
                    "value": float(value),
                    "n": int(n) if n is not None else None,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-site cross-seed aggregation (torch + baselines)")
    parser.add_argument("--run", nargs="+", required=True,
                        help="Run dirs (each with predictions.parquet + config)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir. Default: alongside the first run.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    setup_logging(None, console_level=getattr(logging, args.log_level.upper()))

    run_dirs: list[Path] = []
    for spec in args.run:
        p = Path(spec)
        if p.is_dir():
            run_dirs.append(p.resolve())
        else:
            run_dirs.extend(sorted(Path().glob(spec)))
    if not run_dirs:
        raise SystemExit(f"No runs found from {args.run}")

    out_dir = args.out or (run_dirs[0].parent / "per_site_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for run_dir in run_dirs:
        preds_path = _find_predictions(run_dir)
        if preds_path is None:
            log.warning("skipping %s: no predictions.parquet", run_dir)
            continue
        cfg = _load_config(run_dir)
        run_meta = {
            "run": run_dir.name,
            "variant": str(cfg.model.variant),
            "seed": int(cfg.seed),
        }
        rows = _per_site_rows(pd.read_parquet(preds_path), run_meta)
        all_rows.extend(rows)
        log.info("scored %s (variant=%s, seed=%s, %d per-site rows)",
                 run_dir.name, run_meta["variant"], run_meta["seed"], len(rows))

    if not all_rows:
        raise SystemExit("No per-site metrics could be computed")

    long_df = pd.DataFrame(all_rows)
    long_path = out_dir / "per_site_long.csv"
    summary_path = out_dir / "per_site_summary.csv"
    parametric_path = out_dir / "per_site_parametric_summary.csv"
    long_df.to_csv(long_path, index=False)
    standard_errors(long_df, group_cols=GROUP_COLS).to_csv(summary_path, index=False)
    parametric_summary(long_df, group_cols=GROUP_COLS).to_csv(parametric_path, index=False)
    log.info("\n=== per-site summary saved to %s ===", out_dir)
    for kind, path in {"long": long_path, "summary": summary_path,
                       "parametric": parametric_path}.items():
        log.info("  %-10s %s", kind + ":", path)

    # Compact preview: nee RMSE/R2 mean across seeds, per (variant, site)
    nee = long_df[(long_df.target == "nee") & (long_df.metric.isin(["rmse", "r2", "mae"]))]
    if not nee.empty:
        print("\nnee metrics (mean across seeds), per variant x site:")
        pivot = nee.pivot_table(index=["variant", "site"], columns="metric",
                                values="value", aggfunc="mean")
        print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
