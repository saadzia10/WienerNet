"""Temporal windowing for the per-resolution metric tables.

The analysis notebook computes metrics on daily / weekly / monthly / quarterly
subsets of the test set. This module gives one helper per resolution plus a
combined `evaluate_at_resolutions` that does all four in one call.
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .metrics import evaluate_predictions

log = logging.getLogger("wienernet.evaluation.aggregations")

Resolution = Literal["raw", "daily", "weekly", "monthly", "quarterly"]


# ---------------------------------------------------------------------------
# Time-bucket helpers
# ---------------------------------------------------------------------------

def add_time_buckets(
    df: pd.DataFrame,
    *,
    datetime_column: str = "DateTime",
) -> pd.DataFrame:
    """Add `daily`, `weekly`, `monthly`, `quarterly` integer-bucket columns.

    Buckets are computed from the timestamp (period_start of the bucket). They
    work as group keys for `groupby` and as row masks (`df['daily'] == day_id`).
    """
    out = df.copy()
    dt = pd.to_datetime(out[datetime_column])
    out["daily"] = dt.dt.normalize().astype("int64") // 10**9
    out["weekly"] = dt.dt.to_period("W").apply(lambda p: p.start_time).astype("int64") // 10**9
    out["monthly"] = dt.dt.to_period("M").apply(lambda p: p.start_time).astype("int64") // 10**9
    out["quarterly"] = dt.dt.to_period("Q").apply(lambda p: p.start_time).astype("int64") // 10**9
    return out


# ---------------------------------------------------------------------------
# Per-resolution aggregation of predictions
# ---------------------------------------------------------------------------

def aggregate_by_bucket(
    df: pd.DataFrame,
    gt: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    *,
    bucket_column: str,
    targets: Iterable[str],
    statistic: str = "mean",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    """Aggregate the per-sample gt/pred arrays into one value per bucket.

    Args:
        df: dataframe whose row order matches gt/preds. Must include the bucket column.
        bucket_column: e.g. 'daily', 'weekly', 'monthly', 'quarterly'.
        targets: which gt/pred keys to aggregate.
        statistic: 'mean' or 'sum'.

    Returns:
        (gt_agg, pred_agg, bucket_df). `bucket_df` has one row per bucket
        with the bucket ID column and a `n_samples` count.
    """
    if len(df) != len(next(iter(gt.values()))):
        raise ValueError("df length must equal gt array length")

    work = df[[bucket_column]].copy()
    for target in targets:
        if target in gt:
            work[f"_gt_{target}"] = gt[target]
        if target in preds:
            work[f"_pred_{target}"] = preds[target]

    grouped = work.groupby(bucket_column, sort=True)
    if statistic == "mean":
        agg = grouped.mean(numeric_only=True)
    elif statistic == "sum":
        agg = grouped.sum(numeric_only=True)
    else:
        raise ValueError(f"Unknown statistic {statistic!r}")
    counts = grouped.size().rename("n_samples")

    gt_agg = {
        target: agg[f"_gt_{target}"].to_numpy()
        for target in targets if f"_gt_{target}" in agg.columns
    }
    pred_agg = {
        target: agg[f"_pred_{target}"].to_numpy()
        for target in targets if f"_pred_{target}" in agg.columns
    }
    bucket_df = pd.DataFrame({
        bucket_column: agg.index.to_numpy(),
        "n_samples": counts.to_numpy(),
    })
    return gt_agg, pred_agg, bucket_df


# ---------------------------------------------------------------------------
# Top-level: compute metrics at every requested resolution
# ---------------------------------------------------------------------------

def evaluate_at_resolutions(
    df: pd.DataFrame,
    gt: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    *,
    resolutions: Iterable[Resolution] = ("raw", "daily", "weekly", "monthly", "quarterly"),
    targets: Iterable[str] = ("nee", "bnee", "E0", "rb", "dtemp", "f"),
    datetime_column: str = "DateTime",
    statistic: str = "mean",
    include_mmd: bool = True,
    mmd_subsample: int | None = 2000,
) -> dict[Resolution, dict[str, dict[str, float | int | None]]]:
    """Compute the full per-resolution metric table.

    Returns: {resolution: {target: {metric: value}}}

    The 'raw' resolution computes metrics on the original per-sample arrays.
    """
    df_b = add_time_buckets(df, datetime_column=datetime_column)
    out: dict[Resolution, dict[str, dict[str, float | int | None]]] = {}

    for res in resolutions:
        if res == "raw":
            out["raw"] = evaluate_predictions(
                gt, preds,
                targets=targets,
                include_mmd=include_mmd,
                mmd_subsample=mmd_subsample,
            )
            continue
        gt_agg, pred_agg, _ = aggregate_by_bucket(
            df_b, gt, preds,
            bucket_column=res,
            targets=targets,
            statistic=statistic,
        )
        out[res] = evaluate_predictions(
            gt_agg, pred_agg,
            targets=targets,
            include_mmd=include_mmd,
            mmd_subsample=mmd_subsample,
        )
    return out
