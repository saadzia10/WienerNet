"""Process-consistency diagnostics for the SDE transition model.

Beyond scoring the one-step predictive distribution (see `probabilistic.py`),
these check that the *stochastic process* is captured:

  * variance_vs_scale        — quadratic-variation / diffusion scaling: does the
    increment variance grow with the aggregation window the way the observed
    series does? (A Wiener increment variance grows ~linearly with window.)
  * standardized_residual_autocorr — are the standardized one-step residuals
    z=(obs-mean)/scale white and unit-variance? Leftover autocorrelation signals
    drift structure the model missed.
  * drift_check              — the deterministic part (physics drift + residual)
    vs a denoised coarse-scale conditional-mean increment (R^2 + bias), away from
    the measurement-noise floor.
  * noise_check              — the predicted noise vs the empirical residual
    (obs - deterministic) via moments, tails, an energy distance, and a PIT.

All operate on plain arrays aligned to the test-set row order (shuffle=False),
composing with `Trainer.predict` output + `test_df`. Night grouping reuses the
same gap rule as `wienernet.data.rescale` via `assign_night_ids`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .probabilistic import pit_summary, predictive_cdf, sample_predictive
from .rollout import assign_night_ids

log = logging.getLogger("wienernet.evaluation.process_consistency")


# ---------------------------------------------------------------------------
# 1. Variance-vs-scale (quadratic variation / diffusion scaling)
# ---------------------------------------------------------------------------

def _within_night_diff(series: np.ndarray, night: np.ndarray, k: int) -> np.ndarray:
    """X[t+k] - X[t] taken only within the same night (contiguous rows)."""
    s = pd.Series(np.asarray(series, dtype=float))
    return s.groupby(night).transform(lambda x: x.shift(-k) - x).to_numpy()


def variance_vs_scale(
    times, sites, obs_series: np.ndarray, gen_series: np.ndarray,
    *, ks=(1, 2, 4, 8, 16), step_minutes: float = 30.0,
) -> pd.DataFrame:
    """Increment variance of observed vs generated NEE at growing window sizes.

    For each window k (in native steps), compute Var(X[t+k]-X[t]) within nights
    for both the observed level series and the model-generated (rolled) level
    series. A faithful diffusion reproduces the observed variance-vs-k curve.

    Returns columns: k, hours, var_obs, var_gen, ratio (gen/obs), n.
    """
    night = assign_night_ids(times, sites, step_minutes=step_minutes)
    obs_series = np.asarray(obs_series, dtype=float)
    gen_series = np.asarray(gen_series, dtype=float)
    rows = []
    for k in ks:
        do = _within_night_diff(obs_series, night, k)
        dg = _within_night_diff(gen_series, night, k)
        m = np.isfinite(do) & np.isfinite(dg)
        if m.sum() < 30:
            continue
        vo = float(np.var(do[m], ddof=1))
        vg = float(np.var(dg[m], ddof=1))
        rows.append({"k": int(k), "hours": k * step_minutes / 60.0,
                     "var_obs": vo, "var_gen": vg,
                     "ratio": vg / vo if vo > 0 else float("nan"),
                     "n": int(m.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Standardized-residual whiteness
# ---------------------------------------------------------------------------

def standardized_residual_autocorr(
    times, sites, obs: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    *, max_lag: int = 8, step_minutes: float = 30.0,
) -> tuple[pd.DataFrame, dict]:
    """Autocorrelation of the standardized one-step residual z=(obs-mean)/scale.

    Well-specified => z is white (acf ~ 0 at all lags) and unit variance
    (std_z ~ 1). Returns (acf_table[lag, hours, acf, n], summary{mean_z, std_z}).
    ACF is computed within nights only (no cross-night pairs).
    """
    night = assign_night_ids(times, sites, step_minutes=step_minutes)
    obs, mean, scale = (np.asarray(a, float).ravel() for a in (obs, mean, scale))
    z = (obs - mean) / scale
    good = np.isfinite(z)
    summary = {
        "mean_z": float(np.mean(z[good])) if good.any() else float("nan"),
        "std_z": float(np.std(z[good])) if good.any() else float("nan"),
        "n": int(good.sum()),
    }
    zc = z - summary["mean_z"]
    zc[~good] = np.nan
    zs = pd.Series(zc)
    rows = []
    for lag in range(1, max_lag + 1):
        a = zs.to_numpy()
        b = zs.groupby(night).shift(-lag).to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30:
            continue
        denom = float(np.sqrt(np.sum(a[m] ** 2) * np.sum(b[m] ** 2)))
        acf = float(np.sum(a[m] * b[m]) / denom) if denom > 0 else float("nan")
        rows.append({"lag": lag, "hours": lag * step_minutes / 60.0,
                     "acf": acf, "n": int(m.sum())})
    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# 3. Component-wise drift check
# ---------------------------------------------------------------------------

def drift_check(
    obs_increment: np.ndarray, pred_increment: np.ndarray, window_id: np.ndarray,
) -> dict[str, float]:
    """Deterministic increment vs the denoised coarse-scale conditional mean.

    Aggregates both the observed one-step increment and the model's deterministic
    increment to `window_id` means (the conditional-mean target, with measurement
    noise averaged out), then reports R^2, bias and slope of predicted-vs-observed
    window means. Validates the drift backbone away from the noise floor.
    """
    df = pd.DataFrame({
        "w": np.asarray(window_id),
        "obs": np.asarray(obs_increment, float).ravel(),
        "pred": np.asarray(pred_increment, float).ravel(),
    })
    df = df[np.isfinite(df["obs"]) & np.isfinite(df["pred"])]
    agg = df.groupby("w").mean()
    o, p = agg["obs"].to_numpy(), agg["pred"].to_numpy()
    if len(o) < 3 or np.var(o) == 0:
        return {"r2": float("nan"), "bias": float("nan"), "slope": float("nan"),
                "n_windows": int(len(o))}
    ss_res = float(np.sum((o - p) ** 2)); ss_tot = float(np.sum((o - o.mean()) ** 2))
    slope = float(np.cov(o, p, ddof=0)[0, 1] / np.var(p)) if np.var(p) > 0 else float("nan")
    return {"r2": 1.0 - ss_res / ss_tot, "bias": float(np.mean(p - o)),
            "slope": slope, "n_windows": int(len(o))}


# ---------------------------------------------------------------------------
# 4. Component-wise noise check
# ---------------------------------------------------------------------------

def energy_distance(a: np.ndarray, b: np.ndarray, *, max_n: int = 3000,
                    seed: int = 0) -> float:
    """Energy distance between two 1-D samples: 2 E|a-b| - E|a-a'| - E|b-b'|.

    0 iff distributions match. Sub-samples to `max_n` per side for tractability.
    """
    rng = np.random.default_rng(seed)
    a = _finite(a); b = _finite(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.size > max_n:
        a = rng.choice(a, max_n, replace=False)
    if b.size > max_n:
        b = rng.choice(b, max_n, replace=False)
    d_ab = np.abs(a[:, None] - b[None, :]).mean()
    d_aa = np.abs(a[:, None] - a[None, :]).mean()
    d_bb = np.abs(b[:, None] - b[None, :]).mean()
    return float(max(2 * d_ab - d_aa - d_bb, 0.0))


def noise_check(
    emp_resid: np.ndarray, scale: np.ndarray,
    *, family: str = "gaussian", nu: float | None = None, seed: int = 0,
) -> dict:
    """Compare the predicted noise to the empirical residual (obs - deterministic).

    Reports moments (var, skew, excess kurtosis, 3-sigma tail) for both the
    empirical residual and a one-draw-per-row sample of the predicted noise, plus
    an energy distance between them and a PIT of the residual under the predictive
    noise law (mean 0, per-row scale). Confirms the stochastic component matches
    the real noise shape and tails.
    """
    from scipy import stats
    r = _finite(emp_resid)
    sc = np.asarray(scale, float).ravel()
    rng = np.random.default_rng(seed)
    pred = sample_predictive(np.zeros_like(sc), sc, 1, family=family, nu=nu, rng=rng).ravel()
    pred = _finite(pred)

    def _moments(x: np.ndarray) -> dict:
        s = np.std(x) if x.size else float("nan")
        return {"var": float(np.var(x)), "skew": float(stats.skew(x)),
                "exkurt": float(stats.kurtosis(x)),
                "tail_3sd": float(np.mean(np.abs(x - np.mean(x)) > 3 * s)) if s > 0 else float("nan")}

    # PIT of the residual under predictive noise centred at 0 (per-row scale)
    m = np.isfinite(emp_resid) & np.isfinite(scale) & (np.asarray(scale, float).ravel() > 0)
    pit = predictive_cdf(np.asarray(emp_resid, float).ravel()[m], np.zeros(int(m.sum())),
                         np.asarray(scale, float).ravel()[m], family=family, nu=nu)
    return {
        "empirical": _moments(r),
        "predicted": _moments(pred),
        "energy_distance": energy_distance(r, pred, seed=seed),
        "pit": pit_summary(pit),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]
