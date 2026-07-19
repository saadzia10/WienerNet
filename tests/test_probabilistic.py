"""Tests for the probabilistic scoring metrics (CRPS / NLL / PIT / coverage / DM)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from wienernet.evaluation import (
    crps,
    crps_ensemble,
    crps_gaussian,
    diebold_mariano,
    ensemble_scores,
    interval_coverage,
    measurement_noise_floor,
    pit_summary,
    predictive_cdf,
    predictive_logpdf,
    probabilistic_scores,
    sample_predictive,
)


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def test_crps_gaussian_matches_ensemble():
    rng = np.random.default_rng(0)
    n = 300
    mean = rng.normal(0, 2, n)
    std = rng.uniform(0.5, 2.0, n)
    obs = mean + std * rng.standard_normal(n)
    closed = crps_gaussian(obs, mean, std)
    samples = sample_predictive(mean, std, 4000, family="gaussian", rng=rng)
    ens = crps_ensemble(obs, samples)
    assert np.allclose(closed.mean(), ens.mean(), rtol=0.02)


def test_crps_point_forecast_equals_absolute_error():
    obs = np.array([1.0, 2.0, 3.0])
    mean = np.array([1.5, 2.0, 5.0])
    got = crps(obs, mean, scale=None)
    assert np.allclose(got, np.abs(obs - mean))
    # a degenerate ensemble (all identical) reproduces |error|
    ens = crps_ensemble(obs, np.repeat(mean[:, None], 5, axis=1))
    assert np.allclose(ens, np.abs(obs - mean))


def test_crps_rewards_calibrated_over_underdispersed():
    rng = np.random.default_rng(1)
    n = 4000
    mean = np.zeros(n)
    obs = rng.standard_normal(n)
    good = crps(obs, mean, np.ones(n), family="gaussian").mean()
    too_narrow = crps(obs, mean, np.full(n, 0.2), family="gaussian").mean()
    too_wide = crps(obs, mean, np.full(n, 5.0), family="gaussian").mean()
    assert good < too_narrow and good < too_wide


# ---------------------------------------------------------------------------
# NLL / log score
# ---------------------------------------------------------------------------

def test_predictive_logpdf_matches_scipy_gaussian():
    obs = np.array([0.0, 1.0, -2.0])
    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([1.0, 2.0, 0.5])
    assert np.allclose(predictive_logpdf(obs, mean, std, family="gaussian"),
                       stats.norm.logpdf(obs, mean, std))


def test_student_t_scoring_is_finite_and_heavier_tail_wins():
    rng = np.random.default_rng(2)
    n = 5000
    mean = np.zeros(n)
    scale = np.ones(n)
    # heavy-tailed observations
    obs = stats.t.rvs(df=3, size=n, random_state=rng)
    nll_t = -np.mean(predictive_logpdf(obs, mean, scale, family="student_t", nu=3.0))
    nll_g = -np.mean(predictive_logpdf(obs, mean, scale, family="gaussian"))
    assert np.isfinite(nll_t) and nll_t < nll_g  # t fits heavy tails better


# ---------------------------------------------------------------------------
# PIT / calibration
# ---------------------------------------------------------------------------

def test_pit_uniform_when_well_specified():
    rng = np.random.default_rng(3)
    n = 8000
    mean = rng.normal(0, 1, n)
    std = rng.uniform(0.5, 2, n)
    obs = mean + std * rng.standard_normal(n)
    pit = predictive_cdf(obs, mean, std, family="gaussian")
    s = pit_summary(pit)
    assert s["ks_uniform"] < 0.03
    assert s["mean"] == pytest.approx(0.5, abs=0.02)
    assert s["var"] == pytest.approx(1 / 12, abs=0.005)


def test_pit_variance_flags_underdispersion():
    rng = np.random.default_rng(4)
    n = 8000
    mean = np.zeros(n)
    obs = rng.standard_normal(n)
    pit = predictive_cdf(obs, mean, np.full(n, 0.5), family="gaussian")  # too narrow
    s = pit_summary(pit)
    assert s["var"] > 1 / 12  # U-shape => over-confident


def test_interval_coverage_matches_nominal_when_calibrated():
    rng = np.random.default_rng(5)
    n = 20000
    mean = np.zeros(n)
    std = np.ones(n)
    obs = rng.standard_normal(n)
    cov = interval_coverage(obs, mean, std, family="gaussian", levels=(0.5, 0.9, 0.95))
    assert cov["0.90"]["coverage"] == pytest.approx(0.9, abs=0.02)
    assert cov["0.95"]["coverage"] == pytest.approx(0.95, abs=0.02)
    # narrower nominal interval is sharper (smaller width)
    assert cov["0.50"]["sharpness"] < cov["0.95"]["sharpness"]


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------

def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(6)
    n = 2000
    base = rng.uniform(0.5, 1.5, n)
    loss_a = base                 # model A
    loss_b = base + 0.3           # model B strictly worse
    res = diebold_mariano(loss_a, loss_b)
    assert res["mean_diff"] < 0            # A has lower loss
    assert res["dm_stat"] < 0 and res["p_value"] < 0.01


def test_diebold_mariano_null_for_equal_models():
    rng = np.random.default_rng(7)
    loss = rng.uniform(0, 1, 2000)
    res = diebold_mariano(loss, loss.copy())
    assert res["mean_diff"] == pytest.approx(0.0)  # identical losses -> no difference


# ---------------------------------------------------------------------------
# Measurement-noise floor
# ---------------------------------------------------------------------------

def test_measurement_noise_floor_recovers_linear_scale():
    rng = np.random.default_rng(8)
    n = 40000
    flux = rng.uniform(1, 10, n)
    sigma = 0.3 + 0.2 * flux
    resid = rng.standard_normal(n) * sigma
    out = measurement_noise_floor(resid, flux)
    assert out["b"] == pytest.approx(0.2, abs=0.05)
    assert out["a"] == pytest.approx(0.3, abs=0.1)
    assert out["rmse_floor"] == pytest.approx(np.sqrt(np.mean(sigma ** 2)), rel=0.05)


# ---------------------------------------------------------------------------
# Top-level bundle
# ---------------------------------------------------------------------------

def test_ensemble_scores_calibrated_when_well_specified():
    rng = np.random.default_rng(11)
    n = 6000
    mean = rng.normal(0, 1, n)
    std = rng.uniform(0.5, 2, n)
    obs = mean + std * rng.standard_normal(n)
    samples = sample_predictive(mean, std, 200, family="gaussian", rng=rng)
    sc = ensemble_scores(obs, samples, levels=(0.5, 0.9, 0.95))
    assert sc["pit"]["ks_uniform"] < 0.05
    assert sc["coverage"]["0.90"]["coverage"] == pytest.approx(0.9, abs=0.03)
    # ensemble CRPS agrees with the closed-form Gaussian CRPS
    assert sc["crps"] == pytest.approx(crps_gaussian(obs, mean, std).mean(), rel=0.05)


def test_ensemble_scores_flags_underdispersed_ensemble():
    rng = np.random.default_rng(12)
    n = 5000
    obs = rng.standard_normal(n)
    narrow = sample_predictive(np.zeros(n), np.full(n, 0.3), 200, rng=rng)  # too tight
    sc = ensemble_scores(obs, narrow)
    assert sc["coverage"]["0.90"]["coverage"] < 0.6
    assert sc["pit"]["var"] > 1 / 12


def test_probabilistic_scores_full_and_point():
    rng = np.random.default_rng(9)
    n = 3000
    mean = np.zeros(n)
    std = np.ones(n)
    obs = rng.standard_normal(n)
    full = probabilistic_scores(obs, mean, std, family="gaussian")
    assert full["nll"] is not None and full["pit"] is not None
    assert set(full["coverage"]) == {"0.50", "0.90", "0.95"}
    # deterministic: scale None -> CRPS only, NLL/PIT null
    point = probabilistic_scores(obs, mean, None)
    assert point["nll"] is None and point["family"] == "point"
    assert point["crps"] == pytest.approx(np.mean(np.abs(obs - mean)), rel=1e-6)
