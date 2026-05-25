"""Tests for the data pipeline: features, splits, dataset, partitioning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from data_pipeline import (
    ParameterEstimator,
    SitePreprocessor,
    lloyd_taylor,
    make_night_mask_from_column,
)
from wienernet.data import (
    ClimateDataset,
    add_site_vector_for,
    add_time_vars,
    assemble_features,
    physics_nee_numpy,
    set_season_tag,
    split_data_by_site_fraction,
    split_data_by_year,
)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_add_time_vars_columns_and_ranges():
    df = pd.DataFrame({"DateTime": pd.date_range("2018-01-01", periods=48, freq="30min")})
    out = add_time_vars(df)
    for col in ("month_sin", "month_cos", "hour_sin", "hour_cos", "time"):
        assert col in out.columns
    # sin/cos must lie in [-1, 1]
    for col in ("month_sin", "month_cos", "hour_sin", "hour_cos"):
        assert out[col].min() >= -1.0 - 1e-9
        assert out[col].max() <= 1.0 + 1e-9
    # `time` is normalised to [0, 1)
    assert 0 <= out["time"].min() <= out["time"].max() < 1.01


def test_set_season_tag_encodes_correctly():
    df = pd.DataFrame({"DateTime": pd.to_datetime([
        "2020-01-15",  # winter (0)
        "2020-04-15",  # spring (1)
        "2020-07-15",  # summer (2)
        "2020-10-15",  # autumn (3)
    ])})
    out = set_season_tag(df)
    assert out["season"].tolist() == [0, 1, 2, 3]


def test_add_site_vector_constant_per_site():
    df = pd.DataFrame({"DateTime": pd.date_range("2020-01-01", periods=10, freq="h")})
    out = add_site_vector_for(df, "rosedene")
    assert out["site_x"].nunique() == 1
    # The vector is a unit vector
    norm = np.sqrt(out["site_x"].iloc[0]**2 + out["site_y"].iloc[0]**2 + out["site_z"].iloc[0]**2)
    assert abs(norm - 1.0) < 1e-9


def test_assemble_features_returns_correct_dim():
    df = pd.DataFrame({
        "DateTime": pd.date_range("2020-01-01", periods=10, freq="30min"),
        "Ta": np.random.randn(10), "H": np.random.randn(10), "Tau": np.random.randn(10),
        "RH": np.random.randn(10), "VPD": np.random.randn(10), "Rg": np.random.randn(10),
        "Ustar": np.random.randn(10), "Tsoil1": np.random.randn(10),
    })
    df = add_time_vars(df)
    df = set_season_tag(df)
    df = add_site_vector_for(df, "rosedene")
    drivers = ["Ta", "H", "Tau", "RH", "VPD", "Rg", "Ustar", "Tsoil1"]
    X, cols = assemble_features(df, drivers=drivers)
    # 8 drivers + 1 season + 5 time_vars + 3 site_xyz = 17
    assert X.shape == (10, 17), f"shape was {X.shape}"
    assert len(cols) == 17


def test_assemble_features_raises_on_missing_column():
    df = pd.DataFrame({"DateTime": pd.date_range("2020-01-01", periods=5, freq="h")})
    with pytest.raises(KeyError, match="Missing feature columns"):
        assemble_features(df, drivers=["Ta"])  # Ta not in df


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_split_by_site_fraction_no_leakage():
    """train and test sets must be disjoint by row index."""
    df = pd.DataFrame({
        "site": ["A"] * 100 + ["B"] * 100,
        "value": np.arange(200),
        "DateTime": pd.date_range("2020-01-01", periods=200, freq="30min"),
    })
    train, test = split_data_by_site_fraction(df, test_frac=0.3, shuffle=False)
    overlap = set(train["value"]) & set(test["value"])
    assert overlap == set(), "train and test sets overlap"


def test_split_by_site_fraction_balanced_per_site():
    df = pd.DataFrame({"site": ["A"] * 100 + ["B"] * 100,
                       "value": np.arange(200)})
    train, test = split_data_by_site_fraction(df, test_frac=0.3)
    assert (test["site"] == "A").sum() == 30
    assert (test["site"] == "B").sum() == 30


def test_split_by_site_fraction_deterministic():
    """Same args → same output (when shuffle=True with fixed random_state)."""
    df = pd.DataFrame({"site": ["A"] * 50 + ["B"] * 50,
                       "value": np.arange(100),
                       "DateTime": pd.date_range("2020-01-01", periods=100, freq="h")})
    a_train, a_test = split_data_by_site_fraction(df, shuffle=True, random_state=7)
    b_train, b_test = split_data_by_site_fraction(df, shuffle=True, random_state=7)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_split_by_year():
    df = pd.DataFrame({"DateTime": pd.to_datetime(
        ["2017-06-01", "2018-06-01", "2019-06-01", "2020-06-01"]),
        "v": [1, 2, 3, 4]})
    train, test = split_data_by_year(df, test_years=[2018, 2019])
    assert train["v"].tolist() == [1, 4]
    assert test["v"].tolist() == [2, 3]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_climate_dataset_getitem_returns_all_tensors():
    n = 8
    ds = ClimateDataset(
        X=np.random.randn(n, 17).astype(np.float32),
        k=np.random.randn(n, 2).astype(np.float32),
        T=np.random.randn(n).astype(np.float32),
        dNEE=np.random.randn(n).astype(np.float32),
        bNEE=np.random.randn(n).astype(np.float32),
        dT=np.random.randn(n).astype(np.float32),
        NEE=np.random.randn(n).astype(np.float32),
    )
    item = ds[0]
    assert set(item.keys()) == {"X", "k", "T", "dNEE", "bNEE", "dT", "NEE"}
    for v in item.values():
        assert isinstance(v, torch.Tensor)
        assert v.dtype == torch.float32


def test_dataset_with_site_ids():
    n = 8
    ds = ClimateDataset(
        X=np.zeros((n, 17), np.float32),
        k=np.zeros((n, 2), np.float32),
        T=np.zeros(n, np.float32),
        dNEE=np.zeros(n, np.float32),
        bNEE=np.zeros(n, np.float32),
        dT=np.zeros(n, np.float32),
        NEE=np.zeros(n, np.float32),
        site_ids=["a"] * 4 + ["b"] * 4,
    )
    item = ds[5]
    assert item["site_id"] == "b"


# ---------------------------------------------------------------------------
# Physics & partitioning sanity (numpy)
# ---------------------------------------------------------------------------

def test_physics_nee_numpy_matches_partitioner_lloyd_taylor():
    """Wienernet's numpy helper must match the partitioner's reference."""
    T = np.array([5., 15., 25.])
    E0, rb = 200.0, 2.5
    assert np.allclose(physics_nee_numpy(np.array([E0]*3), np.array([rb]*3), T),
                       lloyd_taylor(T, rb, E0))


def test_parameter_estimator_runs_on_minimal_data():
    """Synthetic night-only data with enough temperature range for one fit."""
    rng = np.random.default_rng(0)
    n = 100
    times = pd.date_range("2020-06-01 00:00", periods=n, freq="30min")
    T = rng.uniform(5, 25, n)
    rb_true, E0_true = 3.0, 200.0
    NEE = lloyd_taylor(T, rb_true, E0_true) + rng.normal(0, 0.1, n)
    df = pd.DataFrame({
        "DateTime": times,
        "Day/Night": [False] * n,   # all nighttime
        "Rg": np.zeros(n),
        "Ta": T,
        "NEE": NEE,
        "TER": NEE,
    })
    night_mask = make_night_mask_from_column(df)
    est = ParameterEstimator(min_data_points_per_window=10)
    out = est.fit_night(df, night_mask=night_mask)
    assert "E0" in out.columns and "rb" in out.columns
    assert not out["E0"].isna().any()
    # Recovered values should be in the right ballpark (curve_fit on noisy data)
    assert abs(out["E0"].mean() - E0_true) < E0_true * 0.5
    assert abs(out["rb"].mean() - rb_true) < rb_true * 0.5
