"""Free-running rollout + SDE-decomposition diagnostics for the increment model.

Because the increment model's drift and diffusion are **exogenous** (functions of
drivers, not of NEE_t), a free-running rollout is just a within-night cumulative
sum of the predicted increments seeded from the observed boundary — there is no
train/inference exposure mismatch. This module provides:

  * `nightly_rollout`  — integrate predicted dNEE within each contiguous night.
  * `decomposition_correlations` — corr(residual, Ta) vs corr(noise, Ta), the
    headline check that version A moved the physics misfit out of the noise
    (corr(noise, Ta) -> 0) and into the named residual.

Both operate on plain arrays aligned to the test-set row order (the test loader
uses shuffle=False), so they compose with `Trainer.predict` output + `test_df`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_night_ids(
    datetimes,
    sites,
    *,
    step_minutes: float = 30.0,
) -> np.ndarray:
    """Label each row with a globally-unique contiguous-night id.

    Mirrors the night-crossover rule in `wienernet.data.rescale`: within a site,
    a real-time gap larger than one step starts a new night. Returns an int array
    aligned to the input order.
    """
    t = pd.to_datetime(pd.Series(list(datetimes)))
    df = pd.DataFrame({"t": t.to_numpy(), "site": list(sites)})
    df["_ord"] = np.arange(len(df))
    labels = np.empty(len(df), dtype=object)
    for site, g in df.groupby("site", sort=False):
        g = g.sort_values("t")
        gap = (g["t"] - g["t"].shift(1)).dt.total_seconds() / 60.0
        nid = (gap > step_minutes).cumsum().fillna(0).astype(int)
        labels[g["_ord"].to_numpy()] = [f"{site}#{n}" for n in nid.to_numpy()]
    return pd.factorize(labels)[0]


def nightly_rollout(
    datetimes,
    sites,
    boundary_nee,
    pred_dnee,
    *,
    step_minutes: float = 30.0,
) -> np.ndarray:
    """Integrate predicted increments within each night from the observed boundary.

    NEE_hat[t0] = observed NEE at the night's first row; NEE_hat[t+1] =
    NEE_hat[t] + pred_dNEE[t]. Since the increment is exogenous, this equals
    seed + cumsum(pred_dNEE) shifted by one. Returns the rolled NEE aligned to
    the input row order.

    Args:
        datetimes, sites: per-row time + site, for night grouping.
        boundary_nee: observed NEE_t per row (the GT trajectory; also the seed).
        pred_dnee: model-predicted increment per row (outputs['dnee_pred'],
            i.e. the *integrated* dNEE = drift*dt + noise*sqrt(dt), NOT the rate).
        step_minutes: cadence used to detect night boundaries.
    """
    night = assign_night_ids(datetimes, sites, step_minutes=step_minutes)
    df = pd.DataFrame({
        "t": pd.to_datetime(pd.Series(list(datetimes))).to_numpy(),
        "night": night,
        "b": np.asarray(boundary_nee, dtype=float).ravel(),
        "d": np.asarray(pred_dnee, dtype=float).ravel(),
    })
    df["_ord"] = np.arange(len(df))
    rolled = np.empty(len(df), dtype=float)
    for _, g in df.groupby("night", sort=False):
        g = g.sort_values("t")
        seed = float(g["b"].iloc[0])
        traj = seed + g["d"].cumsum().shift(1).fillna(0.0)
        rolled[g["_ord"].to_numpy()] = traj.to_numpy()
    return rolled


def decomposition_correlations(
    ta,
    *,
    residual=None,
    noise=None,
) -> dict[str, float]:
    """corr(residual, Ta) / corr(noise, Ta) and the noise mean.

    The headline SDE-cleanup diagnostic (docs/increment_sde_model_plan.md §7):
      * version A: corr(noise, Ta) -> 0 and noise mean -> 0 (pure aleatoric),
        with corr(residual, Ta) carrying the temperature-dependent misfit.
      * version B: the misfit stays in the noise, so corr(noise, Ta) is large.
    Only the provided terms are reported (residual is None for B/C; noise is
    None for C).
    """
    ta = np.asarray(ta, dtype=float).ravel()
    out: dict[str, float] = {}

    def _corr(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).ravel()
        mask = np.isfinite(a) & np.isfinite(ta)
        if mask.sum() < 2 or np.std(a[mask]) == 0 or np.std(ta[mask]) == 0:
            return float("nan")
        return float(np.corrcoef(ta[mask], a[mask])[0, 1])

    if residual is not None:
        out["corr_residual_ta"] = _corr(residual)
        out["residual_mean"] = float(np.nanmean(np.asarray(residual, dtype=float)))
    if noise is not None:
        out["corr_noise_ta"] = _corr(noise)
        out["noise_mean"] = float(np.nanmean(np.asarray(noise, dtype=float)))
    return out
