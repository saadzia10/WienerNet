"""Cross-run aggregation.

Replaces the `standard_errors` function from All_Results_Analysis.ipynb and
gives one consistent flat-table format that pandas + the analysis notebook
can pivot however the user wants.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

log = logging.getLogger("wienernet.evaluation.reporter")


# ---------------------------------------------------------------------------
# Flat-table representation: one row per (run, resolution, target, metric)
# ---------------------------------------------------------------------------

def metrics_to_long_dataframe(
    metrics_per_run: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float | int | None]]]],
    *,
    extra_columns: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Flatten nested {run: {resolution: {target: {metric: value}}}}.

    Args:
        metrics_per_run: outer keys are run IDs (e.g. "ae_seed42"), values
            are the dict returned by `evaluate_at_resolutions`.
        extra_columns: per-run metadata to merge into each row. e.g.
            {"ae_seed42": {"variant": "ae", "seed": 42, "run_dir": "..."}}.

    Returns:
        DataFrame with columns
            [run, variant?, seed?, resolution, target, metric, value, n]
        Easy to pivot with `.pivot_table(index=..., columns=..., values='value')`.
    """
    rows: list[dict[str, Any]] = []
    for run_id, per_resolution in metrics_per_run.items():
        meta = (extra_columns or {}).get(run_id, {})
        for resolution, per_target in per_resolution.items():
            for target, metric_dict in per_target.items():
                n = metric_dict.get("n")
                for metric, value in metric_dict.items():
                    if metric == "n" or value is None:
                        continue
                    rows.append({
                        "run": run_id,
                        **meta,
                        "resolution": resolution,
                        "target": target,
                        "metric": metric,
                        "value": float(value),
                        "n": int(n) if n is not None else None,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-seed summary statistics
# ---------------------------------------------------------------------------

def standard_errors(
    long_df: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("variant", "resolution", "target", "metric"),
    value_col: str = "value",
) -> pd.DataFrame:
    """Median and (1.4826 * MAD) per group — robust analogue of the original
    `standard_errors` function but using MAD instead of standard error so
    outlier seeds don't dominate.

    Returns a DataFrame indexed by `group_cols` with columns:
        n_runs, median, mad, q05, q95
    """
    group_cols = list(group_cols)

    def _median(s: pd.Series) -> float:
        v = s.dropna().to_numpy()
        return float(np.median(v)) if len(v) else np.nan

    def _mad(s: pd.Series) -> float:
        v = s.dropna().to_numpy()
        if len(v) == 0:
            return np.nan
        return float(1.4826 * np.median(np.abs(v - np.median(v))))

    def _q(quantile: float):
        def fn(s: pd.Series) -> float:
            v = s.dropna().to_numpy()
            return float(np.quantile(v, quantile)) if len(v) else np.nan
        return fn

    summary = long_df.groupby(group_cols, sort=False)[value_col].agg(
        n_runs="count",
        median=_median,
        mad=_mad,
        q05=_q(0.05),
        q95=_q(0.95),
    ).reset_index()
    return summary


def cross_seed_pivot(
    long_df: pd.DataFrame,
    *,
    target: str,
    resolution: str = "raw",
    index: str = "variant",
    columns: str = "metric",
    aggfunc: str = "median",
) -> pd.DataFrame:
    """Wide-table view for paper-style comparison.

    Filters long_df to a single (target, resolution), then pivots — gives
    one row per variant, one column per metric (median across seeds).
    """
    df = long_df[(long_df["target"] == target) & (long_df["resolution"] == resolution)]
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(
        index=index, columns=columns, values="value", aggfunc=aggfunc,
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_metrics_tables(
    long_df: pd.DataFrame,
    *,
    out_dir: str | Path,
    standard_error_groups: Iterable[str] = ("variant", "resolution", "target", "metric"),
) -> dict[str, Path]:
    """Write a `long.csv` and a `summary.csv` (median+MAD) to out_dir.

    Returns paths so the caller can log them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "long.csv"
    summary_path = out_dir / "summary.csv"
    long_df.to_csv(long_path, index=False)
    standard_errors(long_df, group_cols=standard_error_groups).to_csv(summary_path, index=False)
    log.info("wrote %s (%d rows) and %s", long_path, len(long_df), summary_path)
    return {"long": long_path, "summary": summary_path}
