"""Smoke tests for the manuscript plotting functions.

We don't pixel-compare; we just verify the functions return a Figure with
non-zero axes/lines for sensible inputs, and save to disk without errors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from wienernet.evaluation import (
    compute_gt_noise,
    daily_window_mask,
    monthly_window_mask,
    plot_all_temporal_scales,
    plot_feature_importance,
    plot_nee_daily,
    plot_nee_distribution_grid,
    plot_nee_monthly,
    plot_nee_quarterly,
    plot_nee_weekly,
    plot_noise_comparison_overall,
    plot_noise_comparison_per_site,
    plot_noise_distribution_grid,
    plot_sample_counts_per_site,
    quarterly_window_mask,
    weekly_window_mask,
)


@pytest.fixture
def synthetic_site_dfs():
    rng = np.random.default_rng(0)
    out = {}
    for site in ("Redmere1", "Redmere2", "Rosedene"):
        n = 500
        out[site] = pd.DataFrame({
            "DateTime": pd.date_range("2018-01-01", periods=n, freq="30min"),
            "NEE":      rng.standard_normal(n) * 2,
            "Ta":       rng.uniform(0, 20, n),
            "E0":       rng.uniform(100, 300, n),
            "rb":       rng.uniform(1, 5, n),
        })
    return out


@pytest.fixture
def synthetic_test_run():
    """Mimics what scripts/evaluate.py writes: gt_*, pred_*, plus test_data columns."""
    rng = np.random.default_rng(0)
    n = 1000
    times = pd.date_range("2018-01-01", periods=n, freq="30min")
    NEE = rng.standard_normal(n) * 2
    test_data = pd.DataFrame({
        "DateTime": times,
        "site": np.repeat(["redmere_1", "great_fen"], n // 2),
        "NEE":  NEE,
        "Ta":   rng.uniform(0, 20, n),
        "E0":   rng.uniform(100, 300, n),
        "rb":   rng.uniform(1, 5, n),
        "season": (times.month % 4),
    })
    gt = {
        "nee": NEE,
        "bnee": NEE - rng.normal(0, 0.5, n),
        "E0": test_data["E0"].values,
        "rb": test_data["rb"].values,
        "dtemp": rng.standard_normal(n) * 0.1,
        "f": rng.standard_normal(n) * 0.05,
    }
    preds = {k: v + rng.normal(0, 0.3, n) for k, v in gt.items()}
    preds["noise"]      = rng.normal(0, 1, n)
    preds["noise_stds"] = np.abs(rng.normal(1, 0.1, n))
    return test_data, gt, preds


# ===========================================================================
# Data-level plots
# ===========================================================================

def test_plot_sample_counts_per_site(tmp_path):
    counts = {"Redmere1": 100, "Redmere2": 200, "Rosedene": 500}
    fig = plot_sample_counts_per_site(counts, save_path=tmp_path / "x.png")
    ax = fig.axes[0]
    assert len(ax.patches) == 3                 # 3 bars
    assert (tmp_path / "x.png").exists()
    plt.close(fig)


def test_plot_nee_distribution_grid(synthetic_site_dfs, tmp_path):
    fig = plot_nee_distribution_grid(synthetic_site_dfs, save_path=tmp_path / "y.png")
    # 3 sites in a 2-col grid -> 2 rows × 2 cols = 4 axes (one hidden)
    assert len(fig.axes) >= 3
    assert (tmp_path / "y.png").exists()
    plt.close(fig)


def test_plot_noise_distribution_grid(synthetic_site_dfs, tmp_path):
    fig = plot_noise_distribution_grid(synthetic_site_dfs, save_path=tmp_path / "z.png")
    assert (tmp_path / "z.png").exists()
    plt.close(fig)


# ===========================================================================
# Mask helpers
# ===========================================================================

def test_daily_window_mask_picks_one_contiguous_block(synthetic_test_run):
    test_data, _, _ = synthetic_test_run
    # Inject some gaps so we have multiple daily groups
    td = test_data.copy()
    td.loc[200:299, "DateTime"] = td.loc[200:299, "DateTime"] + pd.Timedelta(days=10)
    mask = daily_window_mask(td, group_id=0)
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_weekly_window_mask(synthetic_test_run):
    test_data, _, _ = synthetic_test_run
    mask = weekly_window_mask(test_data, group_id=0)
    assert mask.dtype == bool


def test_monthly_window_mask(synthetic_test_run):
    test_data, _, _ = synthetic_test_run
    mask = monthly_window_mask(test_data, year=2018, month=1)
    assert mask.dtype == bool
    assert mask.sum() > 0  # synthetic data starts Jan 2018


def test_quarterly_window_mask_requires_season_column(synthetic_test_run):
    test_data, _, _ = synthetic_test_run
    mask = quarterly_window_mask(test_data, year=2018, season=1)
    assert mask.dtype == bool


def test_quarterly_window_mask_raises_without_season():
    df = pd.DataFrame({"DateTime": pd.date_range("2018-01-01", periods=10, freq="h")})
    with pytest.raises(KeyError, match="season"):
        quarterly_window_mask(df, year=2018, season=1)


# ===========================================================================
# Temporal-scale plots
# ===========================================================================

def test_plot_nee_daily_minimal(synthetic_test_run, tmp_path):
    test_data, gt, preds = synthetic_test_run
    fig = plot_nee_daily(gt, preds, test_data, model_name="Test",
                         group_id=0, save_path=tmp_path / "d.png")
    assert fig.axes[0].get_title().startswith("Test:")
    assert (tmp_path / "d.png").exists()
    plt.close(fig)


def test_plot_nee_daily_with_extras(synthetic_test_run, tmp_path):
    test_data, gt, preds = synthetic_test_run
    # Add the physics column the daily plot can show
    from wienernet.data import physics_nee_numpy
    test_data["NEE_phy"] = physics_nee_numpy(test_data["E0"].values,
                                              test_data["rb"].values,
                                              test_data["Ta"].values)
    fig = plot_nee_daily(
        gt, preds, test_data, model_name="PIAE", group_id=0,
        show_predictions_no_noise=True, show_physics=True, show_ci=True,
        save_path=tmp_path / "d_extras.png",
    )
    # GT + Pred + Pred-no-noise + Physics = 4 lines
    assert len([l for l in fig.axes[0].lines if l.get_linewidth() == 4]) >= 4
    plt.close(fig)


def test_plot_nee_weekly(synthetic_test_run, tmp_path):
    test_data, gt, preds = synthetic_test_run
    fig = plot_nee_weekly(gt, preds, test_data, model_name="Test",
                          group_id=0, save_path=tmp_path / "w.png")
    plt.close(fig)


def test_plot_nee_monthly(synthetic_test_run, tmp_path):
    test_data, gt, preds = synthetic_test_run
    fig = plot_nee_monthly(gt, preds, test_data, model_name="Test",
                            year=2018, month=1, save_path=tmp_path / "m.png")
    plt.close(fig)


def test_plot_nee_quarterly(synthetic_test_run, tmp_path):
    test_data, gt, preds = synthetic_test_run
    fig = plot_nee_quarterly(gt, preds, test_data, model_name="Test",
                              year=2018, season=1, save_path=tmp_path / "q.png")
    plt.close(fig)


def test_plot_all_temporal_scales_produces_4_figures(synthetic_test_run, tmp_path):
    test_data, gt, preds = synthetic_test_run
    figs = plot_all_temporal_scales(
        gt, preds, test_data, model_name="Test",
        daily_kwargs={"group_id": 0},
        weekly_kwargs={"group_id": 0},
        monthly_kwargs={"year": 2018, "month": 1},
        quarterly_kwargs={"year": 2018, "season": 1},
        save_dir=tmp_path, save_prefix="test",
    )
    assert set(figs) == {"daily", "weekly", "monthly", "quarterly"}
    for scale, fig in figs.items():
        assert (tmp_path / f"test_{scale}.png").exists()
        plt.close(fig)


# ===========================================================================
# Noise comparison (NEW)
# ===========================================================================

def test_compute_gt_noise_returns_array(synthetic_test_run):
    test_data, _, _ = synthetic_test_run
    out = compute_gt_noise(test_data)
    assert out.shape == (len(test_data),)
    assert np.isfinite(out).all()


def test_plot_noise_comparison_overall(synthetic_test_run, tmp_path):
    test_data, _, preds = synthetic_test_run
    gt_noise = compute_gt_noise(test_data)
    fig = plot_noise_comparison_overall(
        gt_noise, preds["noise"], model_name="PIAE",
        save_path=tmp_path / "n.png",
    )
    assert (tmp_path / "n.png").exists()
    # Two histograms -> at least two patch collections
    assert len(fig.axes[0].patches) > 0
    plt.close(fig)


def test_plot_noise_comparison_per_site(synthetic_test_run, tmp_path):
    test_data, _, preds = synthetic_test_run
    gt_noise = compute_gt_noise(test_data)
    fig = plot_noise_comparison_per_site(
        test_data, gt_noise, preds["noise"], model_name="PIAE",
        save_path=tmp_path / "ns.png",
    )
    assert (tmp_path / "ns.png").exists()
    # 2 unique sites in fixture -> 2 panels (plus hidden ones in a 2-col grid)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_noise_per_site_length_mismatch_raises(synthetic_test_run):
    test_data, _, _ = synthetic_test_run
    short = np.zeros(10)
    with pytest.raises(ValueError, match="length mismatch"):
        plot_noise_comparison_per_site(test_data, short, short, model_name="X")


# ===========================================================================
# Feature importance
# ===========================================================================

def test_plot_feature_importance(tmp_path):
    fig = plot_feature_importance(
        ["a", "b", "c", "d"], np.array([0.1, 0.3, 0.4, 0.2]),
        save_path=tmp_path / "fi.png",
    )
    assert (tmp_path / "fi.png").exists()
    plt.close(fig)


def test_plot_feature_importance_top_n(tmp_path):
    fig = plot_feature_importance(
        list("abcdef"), np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        top_n=3, save_path=tmp_path / "fi3.png",
    )
    ax = fig.axes[0]
    # Only top 3 bars
    assert len(ax.patches) == 3
    plt.close(fig)
