"""Tests for the increment-SDE rollout + decomposition diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wienernet.evaluation import (
    assign_night_ids,
    decomposition_correlations,
    nightly_rollout,
)


def _two_nights():
    """Site A: one 3-row night. Site B: two 2-row nights (a >30min gap splits)."""
    times = [
        "2020-01-01 22:00", "2020-01-01 22:30", "2020-01-01 23:00",   # A night 0
        "2020-06-01 23:00", "2020-06-01 23:30",                        # B night 0
        "2020-06-02 22:00", "2020-06-02 22:30",                        # B night 1 (24h gap)
    ]
    sites = ["A", "A", "A", "B", "B", "B", "B"]
    return pd.to_datetime(times), sites


def test_assign_night_ids_splits_on_gap():
    times, sites = _two_nights()
    night = assign_night_ids(times, sites)
    # Three distinct nights, contiguous rows share an id
    assert night[0] == night[1] == night[2]
    assert night[3] == night[4]
    assert night[5] == night[6]
    assert len({night[0], night[3], night[5]}) == 3


def test_nightly_rollout_is_boundary_plus_cumsum():
    times, sites = _two_nights()
    boundary = np.array([10.0, 11.0, 12.0, 5.0, 5.5, 8.0, 8.0])  # observed NEE_t
    dnee = np.array([1.0, 2.0, 99.0, 0.5, 99.0, -1.0, 99.0])     # predicted increments

    rolled = nightly_rollout(times, sites, boundary, dnee)

    # Night A: seed 10, then +1, +2 (last row's own increment doesn't add to itself)
    np.testing.assert_allclose(rolled[:3], [10.0, 11.0, 13.0])
    # Night B0: seed 5, then +0.5
    np.testing.assert_allclose(rolled[3:5], [5.0, 5.5])
    # Night B1: reseeds at its own boundary (8.0), independent of B0
    np.testing.assert_allclose(rolled[5:7], [8.0, 7.0])


def test_nightly_rollout_survives_unsorted_input():
    """Rows in arbitrary order must still integrate per night and map back."""
    times, sites = _two_nights()
    boundary = np.array([10.0, 11.0, 12.0, 5.0, 5.5, 8.0, 8.0])
    dnee = np.array([1.0, 2.0, 99.0, 0.5, 99.0, -1.0, 99.0])

    order = [6, 0, 4, 2, 5, 1, 3]
    t2 = pd.to_datetime([str(times[i]) for i in order])
    s2 = [sites[i] for i in order]
    rolled_shuf = nightly_rollout(t2, s2, boundary[order], dnee[order])

    rolled_ref = nightly_rollout(times, sites, boundary, dnee)
    np.testing.assert_allclose(rolled_shuf, rolled_ref[order])


def test_decomposition_correlations():
    rng = np.random.default_rng(0)
    ta = rng.normal(size=500)
    residual = 0.9 * ta + 0.1 * rng.normal(size=500)   # misfit tracks Ta
    noise = rng.normal(size=500)                        # pure aleatoric

    out = decomposition_correlations(ta, residual=residual, noise=noise)
    assert out["corr_residual_ta"] > 0.8     # residual carries the Ta-dependence
    assert abs(out["corr_noise_ta"]) < 0.2   # noise is ~uncorrelated with Ta
    assert abs(out["noise_mean"]) < 0.2

    # Only the provided terms are reported
    only_noise = decomposition_correlations(ta, noise=noise)
    assert "corr_residual_ta" not in only_noise
    assert "corr_noise_ta" in only_noise
