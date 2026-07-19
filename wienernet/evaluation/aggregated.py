"""Aggregated-scale evaluation (weekly / monthly).

The one-step predictions are teacher-forced (each anchored on the observed NEE_t),
so aggregating them over a window sums independent-ish per-step errors: the RANDOM
error cancels ~1/sqrt(N) while any SYSTEMATIC bias persists. Scoring the aggregated
flux therefore isolates (i) systematic bias in the mean and (ii) whether the model's
noise structure predicts the aggregated uncertainty — the scale where the physics
signal-to-noise argument (drift persists, noise cancels) is supposed to pay off and
where half-hourly measurement noise no longer floors the metric.

`aggregated_scores` groups the test rows by (site, calendar-window), aggregates the
observation and the forecast ensemble to one value per window, and returns per-
resolution point skill (RMSE/bias/R2 of the aggregated mean) + distributional skill
(ensemble CRPS/PIT/coverage on the aggregated value) + the relative-uncertainty
(error-cancellation) curve. Persisted by `scripts/evaluate.py` to `aggregated.json`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import bias, mae, r2, rmse
from .probabilistic import crps_gaussian, ensemble_scores

_RES_PERIOD = {"daily": "D", "weekly": "W", "monthly": "M", "quarterly": "Q"}


def _window_ids(times: np.ndarray, resolution: str) -> np.ndarray:
    """Integer period-start id per timestamp for the given calendar resolution."""
    if resolution not in _RES_PERIOD:
        raise ValueError(f"unknown resolution {resolution!r}; valid: {list(_RES_PERIOD)}")
    s = pd.to_datetime(pd.Series(times))
    return s.dt.to_period(_RES_PERIOD[resolution]).apply(
        lambda p: p.start_time).astype("int64").to_numpy()


def aggregated_scores(
    times: np.ndarray,
    sites: np.ndarray | None,
    obs: np.ndarray,
    *,
    ensemble: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    resolutions: tuple[str, ...] = ("weekly", "monthly"),
    statistic: str = "mean",
    min_window: int = 3,
    levels: tuple[float, ...] = (0.5, 0.9, 0.95),
) -> dict:
    """Score window-aggregated NEE (per site) at each resolution.

    Args:
        times: (N,) datetime64 per test row. sites: (N,) site id or None (single group).
        obs: (N,) observed NEE_{t+1}.
        ensemble: (N, M) forecast ensemble (repeated stochastic passes); enables the
            honest aggregated distributional score. None for deterministic models.
        mean, scale: (N,) parametric one-step location/scale; used for a parametric
            aggregated Gaussian (Sum/mean of independent -> variance adds) as a cheap
            cross-check. Optional.
        statistic: 'mean' (average flux rate over the window) or 'sum' (total flux).
        min_window: drop windows with fewer than this many half-hourly samples
            (a 1-sample "week" isn't an aggregate).

    Returns: {resolution: {n_windows, statistic, point{...}, ensemble{...},
              parametric{...}, rel_uncertainty}}, plus a top-level 'raw'
              rel_uncertainty reference (single-step) for the cancellation curve.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    n = obs.shape[0]
    site_arr = np.asarray(sites) if sites is not None else np.zeros(n, dtype=int)
    agg = (np.mean if statistic == "mean" else np.sum)
    if statistic not in ("mean", "sum"):
        raise ValueError(f"statistic must be 'mean' or 'sum', got {statistic!r}")

    out: dict = {"statistic": statistic}
    # single-step relative-uncertainty reference (mean over rows of scale/|obs|)
    if ensemble is not None:
        e = np.asarray(ensemble, dtype=float)
        out["raw_rel_uncertainty"] = float(np.nanmean(e.std(1) / (np.abs(e.mean(1)) + 1e-9)))

    for res in resolutions:
        wid = _window_ids(times, res)
        key = pd.Series(list(zip(site_arr.tolist(), wid.tolist())))
        S_obs, S_ens, S_mean, S_scale = [], [], [], []
        for _, idx in key.groupby(key).groups.items():
            idx = np.asarray(idx, dtype=int)
            if idx.size < min_window:
                continue
            S_obs.append(agg(obs[idx]))
            if ensemble is not None:
                S_ens.append(agg(e[idx, :], axis=0))                # (M,)
            if mean is not None and scale is not None:
                m = np.asarray(mean, float).ravel(); sc = np.asarray(scale, float).ravel()
                var = float(np.sum(sc[idx] ** 2))                   # independent-sum variance
                if statistic == "mean":
                    S_mean.append(float(np.mean(m[idx]))); S_scale.append(np.sqrt(var) / idx.size)
                else:
                    S_mean.append(float(np.sum(m[idx]))); S_scale.append(np.sqrt(var))
        S_obs = np.asarray(S_obs, dtype=float)
        block: dict = {"n_windows": int(S_obs.size), "statistic": statistic}
        if S_obs.size == 0:
            out[res] = block
            continue
        # point skill on the aggregated mean (ensemble mean if available, else parametric)
        point_pred = (np.asarray(S_ens).mean(1) if S_ens else
                      (np.asarray(S_mean) if S_mean else None))
        if point_pred is not None:
            block["point"] = {"rmse": rmse(S_obs, point_pred), "bias": bias(S_obs, point_pred),
                              "r2": r2(S_obs, point_pred), "mae": mae(S_obs, point_pred)}
        # distributional skill via the aggregated ensemble (the honest axis)
        if S_ens:
            se = np.asarray(S_ens)
            block["ensemble"] = ensemble_scores(S_obs, se, levels=levels)
            block["rel_uncertainty"] = float(np.nanmean(se.std(1) / (np.abs(se.mean(1)) + 1e-9)))
        # parametric aggregated-Gaussian cross-check
        if S_mean:
            sm, ss = np.asarray(S_mean), np.asarray(S_scale)
            block["parametric"] = {"crps": float(np.mean(crps_gaussian(S_obs, sm, ss)))}
        out[res] = block
    return out
