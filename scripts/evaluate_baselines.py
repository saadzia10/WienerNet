"""Aggregate already-computed metrics across baseline (RF/XGB) runs.

The torch `evaluate.py` rebuilds a WienerNetModel and re-runs inference, which
doesn't apply to the non-torch baselines: they have no `model.heads`, store a
`.joblib` instead of `best.pth`, and already write a full `metrics.json` (every
resolution, every target) at train time. This script just collects those
`metrics.json` files and runs them through the same cross-seed reporter, so RF
and XGB get the identical long / summary / parametric_summary tables that the
torch runs get.

Usage:
    # One variant, all seeds
    python scripts/evaluate_baselines.py --run multirun/2026-06-06_20-33-22/rf_seed*/

    # RF and XGB together -> one summary with a row per variant
    python scripts/evaluate_baselines.py \
        --run multirun/2026-06-06_20-33-22/rf_seed*/ \
              multirun/2026-06-06_20-33-41/xgb_seed*/ \
        --out multirun/baseline_evaluation_summary

Writes (to --out, default: <first run>/../evaluation_summary):
    long.csv                — one row per (run, resolution, target, metric)
    summary.csv             — robust: n_runs, median, mad, q05, q95
    parametric_summary.csv  — n_runs, mean, sd, sem, ci95_lo, ci95_hi
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wienernet.evaluation import metrics_to_long_dataframe, save_metrics_tables
from wienernet.utils import get_logger, setup_logging

log = get_logger("scripts.evaluate_baselines")


def _load_config(run_dir: Path):
    """Prefer Hydra's resolved config; fall back to our config.json."""
    hydra_path = run_dir / ".hydra" / "config.yaml"
    if hydra_path.exists():
        return OmegaConf.load(hydra_path)
    json_path = run_dir / "config.json"
    if json_path.exists():
        return OmegaConf.create(json.loads(json_path.read_text()))
    raise FileNotFoundError(f"No config found in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics.json across baseline (RF/XGB) runs"
    )
    parser.add_argument("--run", nargs="+", required=True,
                        help="One or more baseline run dirs (each with metrics.json + config.json)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the summary tables. Default: alongside the first run.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    setup_logging(None, console_level=getattr(logging, args.log_level.upper()))

    # Expand globs / accept directories
    run_dirs: list[Path] = []
    for spec in args.run:
        p = Path(spec)
        if p.is_dir():
            run_dirs.append(p.resolve())
        else:
            run_dirs.extend(sorted(Path().glob(spec)))
    if not run_dirs:
        raise SystemExit(f"No runs found from {args.run}")

    out_dir = args.out or (run_dirs[0].parent / "evaluation_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_metrics: dict = {}
    per_run_meta: dict = {}
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            log.warning("skipping %s: no metrics.json", run_dir)
            continue
        cfg = _load_config(run_dir)
        run_id = run_dir.name
        per_run_metrics[run_id] = json.loads(metrics_path.read_text())
        per_run_meta[run_id] = {
            "variant": str(cfg.model.variant),
            "seed": int(cfg.seed),
            "run_dir": str(run_dir),
        }
        log.info("collected %s (variant=%s, seed=%s)", run_id, cfg.model.variant, cfg.seed)

    if not per_run_metrics:
        raise SystemExit("No runs had a metrics.json to aggregate")

    long_df = metrics_to_long_dataframe(per_run_metrics, extra_columns=per_run_meta)
    paths = save_metrics_tables(long_df, out_dir=out_dir)
    log.info("\n=== summary saved to %s ===", out_dir)
    for kind, path in paths.items():
        log.info("  %-10s %s", kind + ":", path)

    # Compact preview: mean across seeds at raw resolution, per variant
    raw = long_df[long_df["resolution"] == "raw"]
    if not raw.empty:
        print("\nMetrics (raw resolution, mean across seeds):")
        pivot = raw.pivot_table(
            index=["variant", "target"], columns="metric", values="value", aggfunc="mean",
        )
        print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
