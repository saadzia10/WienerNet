"""Tests for aggregated-scale (weekly/monthly) scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wienernet.evaluation import aggregated_scores


def _synth(n_weeks=6, per_week=48, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-06")  # a Monday
    times, obs = [], []
    for w in range(n_weeks):
        for j in range(per_week):
            times.append(start + pd.Timedelta(weeks=w) + pd.Timedelta(hours=j))
            obs.append(2.0 + 0.5 * np.sin(j) + rng.standard_normal() * 0.3)
    times = np.array(times, dtype="datetime64[ns]")
    obs = np.array(obs)
    return times, obs


def test_weekly_monthly_windows_and_shapes():
    times, obs = _synth()
    ens = obs[:, None] + np.random.default_rng(1).standard_normal((obs.size, 50)) * 0.3
    out = aggregated_scores(times, None, obs, ensemble=ens,
                            resolutions=("weekly", "monthly"), statistic="mean")
    assert out["weekly"]["n_windows"] == 6           # 6 distinct weeks
    assert out["weekly"]["n_windows"] >= out["monthly"]["n_windows"]
    for res in ("weekly", "monthly"):
        assert "point" in out[res] and "ensemble" in out[res]
        assert out[res]["ensemble"]["crps"] >= 0
        assert 0.0 <= out[res]["rel_uncertainty"]


def test_noise_cancels_under_aggregation():
    """Aggregated relative uncertainty is smaller than the raw single-step one."""
    times, obs = _synth(n_weeks=8)
    ens = obs[:, None] + np.random.default_rng(2).standard_normal((obs.size, 200)) * 0.5
    out = aggregated_scores(times, None, obs, ensemble=ens, resolutions=("weekly", "monthly"))
    raw = out["raw_rel_uncertainty"]
    assert out["weekly"]["rel_uncertainty"] < raw          # ~1/sqrt(N) cancellation
    assert out["monthly"]["rel_uncertainty"] <= out["weekly"]["rel_uncertainty"] + 1e-9


def test_perfect_mean_zero_rmse():
    times, obs = _synth()
    ens = np.repeat(obs[:, None], 30, axis=1)              # zero-spread, mean == obs
    out = aggregated_scores(times, None, obs, ensemble=ens, resolutions=("weekly",))
    assert out["weekly"]["point"]["rmse"] < 1e-9


def test_per_site_grouping_independent():
    times, obs = _synth(n_weeks=4)
    sites = np.array(["A"] * (obs.size // 2) + ["B"] * (obs.size - obs.size // 2))
    ens = obs[:, None] + np.random.default_rng(3).standard_normal((obs.size, 20)) * 0.2
    out = aggregated_scores(times, sites, obs, ensemble=ens, resolutions=("weekly",))
    # two sites -> windows are per (site, week), so more windows than single-site
    single = aggregated_scores(times, None, obs, ensemble=ens, resolutions=("weekly",))
    assert out["weekly"]["n_windows"] >= single["weekly"]["n_windows"]


def test_min_window_filter():
    times, obs = _synth(n_weeks=3, per_week=2)             # tiny weeks
    ens = obs[:, None] + np.zeros((obs.size, 10))
    out = aggregated_scores(times, None, obs, ensemble=ens, resolutions=("weekly",), min_window=5)
    assert out["weekly"]["n_windows"] == 0                 # all weeks too small -> dropped


def test_deterministic_point_only():
    times, obs = _synth()
    out = aggregated_scores(times, None, obs, ensemble=None,
                            mean=obs + 0.1, scale=np.full(obs.size, 0.3),
                            resolutions=("weekly",))
    assert "point" in out["weekly"] and "parametric" in out["weekly"]
    assert "ensemble" not in out["weekly"]                 # no ensemble supplied
