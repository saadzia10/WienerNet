"""Tests for the SDE process-consistency diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wienernet.evaluation import (
    drift_check,
    energy_distance,
    noise_check,
    standardized_residual_autocorr,
    variance_vs_scale,
)


def _one_night(n: int) -> tuple[pd.Series, np.ndarray]:
    """A single contiguous night of 30-min steps (no gaps -> one night id)."""
    times = pd.Series(pd.date_range("2020-06-01 20:00", periods=n, freq="30min"))
    sites = np.array(["s"] * n)
    return times, sites


# ---------------------------------------------------------------------------
# Variance-vs-scale (quadratic variation)
# ---------------------------------------------------------------------------

def test_variance_vs_scale_grows_and_matches_when_identical():
    rng = np.random.default_rng(0)
    n = 2000
    times, sites = _one_night(n)
    walk = np.cumsum(rng.standard_normal(n))  # Wiener: Var(X[t+k]-X[t]) ~ k
    tbl = variance_vs_scale(times, sites, walk, walk, ks=(1, 2, 4, 8))
    # increment variance increases with window
    assert tbl["var_obs"].is_monotonic_increasing
    # identical generated series -> ratio ~ 1 everywhere
    assert np.allclose(tbl["ratio"].to_numpy(), 1.0, atol=1e-9)
    # roughly linear scaling: var at k=4 ~ 4x var at k=1 (loose bound)
    v1 = tbl.loc[tbl.k == 1, "var_obs"].iloc[0]
    v4 = tbl.loc[tbl.k == 4, "var_obs"].iloc[0]
    assert 2.5 * v1 < v4 < 6.0 * v1


def test_variance_vs_scale_detects_wrong_diffusion():
    rng = np.random.default_rng(1)
    n = 3000
    times, sites = _one_night(n)
    obs = np.cumsum(rng.standard_normal(n))
    gen = np.cumsum(rng.standard_normal(n) * 0.5)  # half the diffusion
    tbl = variance_vs_scale(times, sites, obs, gen, ks=(1, 2, 4))
    assert (tbl["ratio"] < 0.5).all()  # under-dispersed generator


# ---------------------------------------------------------------------------
# Standardized-residual whiteness
# ---------------------------------------------------------------------------

def test_standardized_residual_white_for_iid():
    rng = np.random.default_rng(2)
    n = 4000
    times, sites = _one_night(n)
    obs = rng.standard_normal(n)
    mean = np.zeros(n)
    scale = np.ones(n)
    acf, summary = standardized_residual_autocorr(times, sites, obs, mean, scale, max_lag=5)
    assert summary["std_z"] == pytest.approx(1.0, abs=0.05)
    assert (acf["acf"].abs() < 0.06).all()  # white


def test_standardized_residual_detects_autocorrelation():
    rng = np.random.default_rng(3)
    n = 4000
    times, sites = _one_night(n)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.6 * z[t - 1] + rng.standard_normal()
    acf, _ = standardized_residual_autocorr(times, sites, z, np.zeros(n), np.ones(n), max_lag=3)
    assert acf.loc[acf.lag == 1, "acf"].iloc[0] > 0.4  # AR(1) leaves lag-1 structure


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------

def test_drift_check_perfect_backbone():
    rng = np.random.default_rng(4)
    n_windows, per = 60, 20
    window_id = np.repeat(np.arange(n_windows), per)
    m = rng.normal(0, 1, n_windows)[window_id]          # true window-mean increment
    obs_incr = m + rng.normal(0, 1, n_windows * per)     # + measurement noise
    pred_incr = m                                        # perfect drift
    res = drift_check(obs_incr, pred_incr, window_id)
    assert res["r2"] > 0.9 and abs(res["bias"]) < 0.1


def test_drift_check_biased_backbone():
    n_windows, per = 40, 25
    window_id = np.repeat(np.arange(n_windows), per)
    m = np.linspace(-1, 1, n_windows)[window_id]
    res = drift_check(m, m + 0.5, window_id)             # constant +0.5 bias
    assert res["bias"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Noise check + energy distance
# ---------------------------------------------------------------------------

def test_energy_distance_zero_for_same_dist_large_for_shift():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    c = rng.standard_normal(2000) + 3.0
    assert energy_distance(a, b) < 0.1
    assert energy_distance(a, c) > 1.0


def test_noise_check_matches_when_scale_correct():
    rng = np.random.default_rng(6)
    n = 6000
    scale = np.full(n, 1.5)
    emp_resid = rng.standard_normal(n) * scale
    res = noise_check(emp_resid, scale, family="gaussian")
    assert res["energy_distance"] < 0.1
    assert res["pit"]["ks_uniform"] < 0.03
    assert res["empirical"]["var"] == pytest.approx(res["predicted"]["var"], rel=0.1)
