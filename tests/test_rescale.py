"""Tests for the time-scale target preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wienernet.data import rescale_to_timestep


def _one_night(start, n, nee, ta):
    t = pd.date_range(start, periods=n, freq="30min")
    return pd.DataFrame({"DateTime": t, "site": "s", "NEE": nee, "Ta": ta,
                         "NEE_next": np.nan, "dNEE": np.nan, "dTa": np.nan, "dt": np.nan})


def test_reconstructs_k_step_targets_within_night():
    # Single 10-step night; k=4 (2h) targets must be exactly the t+4 values.
    n = 10
    nee = np.arange(n, dtype=float) * 2.0        # 0,2,4,...
    ta = np.arange(n, dtype=float) * 0.5         # 0,0.5,1.0,...
    df = _one_night("2020-01-01 20:00", n, nee, ta)

    out = rescale_to_timestep(df, k=4, step_minutes=30)
    # rows 0..5 have a valid t+4 within the night; rows 6..9 run off the end
    assert len(out) == n - 4
    span = 4 * 30.0
    for i in range(len(out)):
        assert out["NEE_next"].iloc[i] == nee[i + 4]                 # NEE_{t+4}
        assert np.isclose(out["dNEE"].iloc[i], (nee[i + 4] - nee[i]) / span)
        assert np.isclose(out["dTa"].iloc[i], (ta[i + 4] - ta[i]) / span)
        assert out["dt"].iloc[i] == span
        assert out["NEE"].iloc[i] == nee[i]                           # input row unchanged


def test_does_not_cross_night_boundary():
    # Two nights of 6 steps each, separated by a ~14h day gap. A k=4 lookahead
    # from late in night 1 must NOT pick night 2's values.
    n1 = _one_night("2020-01-01 20:00", 6, np.arange(6.0), np.arange(6.0))
    n2 = _one_night("2020-01-02 20:00", 6, 100 + np.arange(6.0), 100 + np.arange(6.0))
    df = pd.concat([n1, n2], ignore_index=True)

    out = rescale_to_timestep(df, k=4, step_minutes=30)
    # Each night keeps only rows 0,1 (2 per night) -> no target >= 100 paired with a <100 input
    assert len(out) == 4
    assert (out["NEE"] < 100).sum() == 2 and (out["NEE"] >= 100).sum() == 2
    # No cross-night contamination: every target is in the same night as its input
    for _, r in out.iterrows():
        assert (r["NEE"] < 100) == (r["NEE_next"] < 100)


def test_internal_gap_splits_night():
    # A missing row (60-min gap) mid-night breaks contiguity; a k=2 step must not
    # span the gap.
    t = list(pd.date_range("2020-01-01 20:00", periods=3, freq="30min"))
    t += list(pd.date_range("2020-01-01 22:00", periods=3, freq="30min"))  # 60-min jump
    df = pd.DataFrame({"DateTime": t, "site": "s", "NEE": np.arange(6.0), "Ta": np.arange(6.0),
                       "NEE_next": np.nan, "dNEE": np.nan, "dTa": np.nan, "dt": np.nan})
    out = rescale_to_timestep(df, k=2, step_minutes=30)
    # Contiguous runs are [0,1,2] and [3,4,5]; k=2 valid only at row 0 and row 3.
    assert len(out) == 2
    assert set(out["NEE"].tolist()) == {0.0, 3.0}
    assert out.loc[out.NEE == 0.0, "NEE_next"].iloc[0] == 2.0   # within first run
    assert out.loc[out.NEE == 3.0, "NEE_next"].iloc[0] == 5.0   # within second run
