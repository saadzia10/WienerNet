"""Tests for the evaluation pipeline: metrics, aggregations, reporter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wienernet.evaluation import (
    add_time_buckets,
    aggregate_by_bucket,
    compute_metric_bundle,
    cross_seed_pivot,
    evaluate_at_resolutions,
    evaluate_predictions,
    kl_divergence_histogram,
    mae,
    metrics_to_long_dataframe,
    mmd_rbf,
    r2,
    rmse,
    standard_errors,
    wasserstein,
)


# ---------------------------------------------------------------------------
# Single-target metrics
# ---------------------------------------------------------------------------

def test_mae_perfect_prediction_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_rmse_known_value():
    y = np.array([0., 0., 0.])
    y_hat = np.array([1., 1., 1.])
    assert rmse(y, y_hat) == pytest.approx(1.0)


def test_r2_perfect_prediction_is_one():
    y = np.array([1., 2., 3., 4., 5.])
    assert r2(y, y) == pytest.approx(1.0)


def test_mmd_self_is_small():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    assert mmd_rbf(y, y) < 1e-6


def test_mmd_grows_with_distribution_shift():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    y_shifted = y + 5.0
    assert mmd_rbf(y, y_shifted) > mmd_rbf(y, y + 0.1)


def test_kl_divergence_self_is_small():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(2000)
    # Same distribution → KL ~ 0
    assert kl_divergence_histogram(y, y) < 0.05


def test_wasserstein_known_value():
    """W1 between shifted Diracs equals the shift."""
    a = np.array([0.0])
    b = np.array([5.0])
    assert wasserstein(a, b) == pytest.approx(5.0)


def test_compute_metric_bundle_returns_full_dict():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(100)
    y_hat = y + rng.normal(0, 0.1, 100)
    bundle = compute_metric_bundle(y, y_hat)
    d = bundle.as_dict()
    assert set(d) >= {"n", "mae", "rmse", "r2", "mmd", "kl", "wasserstein"}
    assert d["n"] == 100
    assert d["r2"] > 0.9  # Should fit well — noise is small


def test_evaluate_predictions_with_mask():
    """Mask should restrict the metric computation to the chosen rows."""
    y = np.arange(10, dtype=float)
    y_hat = y.copy()
    y_hat[5:] = -100.0  # corrupt the second half
    mask = np.zeros(10, dtype=bool); mask[:5] = True
    out = evaluate_predictions({"x": y}, {"x": y_hat}, targets=["x"], mask=mask,
                               include_mmd=False)
    assert out["x"]["mae"] == 0.0  # only the perfect prefix counted


# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------

def test_add_time_buckets_produces_expected_columns():
    df = pd.DataFrame({"DateTime": pd.date_range("2020-01-01", periods=200, freq="h")})
    out = add_time_buckets(df)
    for col in ("daily", "weekly", "monthly", "quarterly"):
        assert col in out.columns
    # daily should change every 24 hours
    assert out["daily"].nunique() == 9  # 200h ≈ 9 day-buckets


def test_aggregate_by_bucket_averages_correctly():
    df = pd.DataFrame({
        "DateTime": pd.date_range("2020-01-01", periods=48, freq="30min"),
    })
    df = add_time_buckets(df)
    gt = {"nee": np.arange(48, dtype=float)}
    preds = {"nee": np.arange(48, dtype=float) + 1.0}
    gt_agg, pred_agg, buckets = aggregate_by_bucket(
        df, gt, preds, bucket_column="daily", targets=["nee"], statistic="mean",
    )
    # 48 samples / 48 per day = 1 day
    assert len(buckets) == 1
    assert gt_agg["nee"][0] == pytest.approx(np.arange(48).mean())
    assert pred_agg["nee"][0] == pytest.approx(np.arange(48).mean() + 1.0)


def test_evaluate_at_resolutions_runs_for_all():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"DateTime": pd.date_range("2020-01-01", periods=n, freq="h")})
    gt = {"nee": rng.standard_normal(n)}
    preds = {"nee": gt["nee"] + rng.normal(0, 0.5, n)}
    out = evaluate_at_resolutions(df, gt, preds, targets=["nee"], include_mmd=False)
    assert set(out) == {"raw", "daily", "weekly", "monthly", "quarterly"}
    for res in out:
        assert "nee" in out[res], f"missing nee at resolution {res}"


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

def test_metrics_to_long_dataframe_flattens_correctly():
    metrics_per_run = {
        "ae_seed42": {"raw": {"nee": {"n": 100, "mae": 1.0, "r2": 0.5}}},
        "ae_seed88": {"raw": {"nee": {"n": 100, "mae": 1.2, "r2": 0.45}}},
    }
    extra = {"ae_seed42": {"variant": "ae", "seed": 42},
             "ae_seed88": {"variant": "ae", "seed": 88}}
    df = metrics_to_long_dataframe(metrics_per_run, extra_columns=extra)
    assert len(df) == 4  # 2 runs × 1 resolution × 1 target × 2 non-n metrics
    assert set(df.columns) >= {"run", "variant", "seed", "resolution", "target", "metric", "value", "n"}


def test_standard_errors_aggregates_across_seeds():
    df = pd.DataFrame({
        "variant": ["ae"] * 4,
        "resolution": ["raw"] * 4,
        "target": ["nee"] * 4,
        "metric": ["mae"] * 4,
        "value": [1.0, 1.1, 0.9, 1.2],
        "n": [100] * 4,
    })
    out = standard_errors(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_runs"] == 4
    assert row["median"] == pytest.approx(1.05)


def test_cross_seed_pivot_returns_wide_table():
    df = pd.DataFrame({
        "variant": ["ae"] * 3 + ["piae"] * 3,
        "resolution": ["raw"] * 6,
        "target": ["nee"] * 6,
        "metric": ["mae", "r2", "mmd"] * 2,
        "value": [1.0, 0.5, 0.3, 0.5, 0.8, 0.1],
        "n": [100] * 6,
    })
    pivot = cross_seed_pivot(df, target="nee", resolution="raw")
    assert set(pivot.index) == {"ae", "piae"}
    assert set(pivot.columns) == {"mae", "r2", "mmd"}


def test_increment_correlation_by_lag_perfect_and_contiguous():
    """dNEE proportional to dReco -> |corr|=1; a gap breaks contiguity."""
    from wienernet.evaluation import increment_correlation_by_lag

    n = 400
    times = pd.date_range("2020-01-01", periods=n, freq="30min")
    rng = np.random.default_rng(0)
    Ta = np.cumsum(rng.normal(0, 0.3, n)) + 10.0
    E0 = np.full(n, 300.0)
    rb = np.full(n, 2.0)
    from wienernet.data.features import physics_nee_numpy
    reco = physics_nee_numpy(E0, rb, Ta)
    df = pd.DataFrame({"DateTime": times, "NEE": 2.0 * reco, "Ta": Ta, "E0": E0, "rb": rb})

    res = increment_correlation_by_lag({"s": df}, lags=(1, 2, 4), min_pairs=10)
    # NEE = 2*Reco exactly -> increment correlates perfectly with the Reco increment
    for _, row in res[res.site == "pooled"].iterrows():
        assert abs(abs(row["corr_dReco"]) - 1.0) < 1e-6
    assert set(res["lag"]) == {1, 2, 4}
    assert (res["hours"] == res["lag"] * 0.5).all()

    # Insert a 1h gap (skip a timestamp) -> those spanning pairs are excluded
    df2 = df.drop(index=100).reset_index(drop=True)
    res2 = increment_correlation_by_lag({"s": df2}, lags=(1,), min_pairs=10)
    assert res2[res2.site == "s"]["n"].iloc[0] < len(df2) - 1  # at least the gap pair dropped

