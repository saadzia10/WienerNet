"""Probabilistic scoring of the one-step predictive distribution.

The primary evaluation axis for WienerNet is the quality of the one-step
predictive distribution over next-step NEE (the SDE transition), not point
accuracy. Every model that emits a distribution — a per-sample predictive
location `mean` and scale `scale` with a known `family` — is scored here on
proper scores (CRPS, log-score/NLL), calibration (PIT, interval coverage +
sharpness), and significance (Diebold-Mariano on paired CRPS). Deterministic
models are scored on CRPS too (it reduces to absolute error), so they sit on the
same axis and are automatically penalised for having no spread.

The predictive family is a location-scale law:
  * gaussian    — NEE_{t+1} ~ Normal(mean, scale)
  * student_t   — NEE_{t+1} ~ mean + scale * t(nu)   (scale is the t-scale, not SD)
  * laplace     — NEE_{t+1} ~ Laplace(mean, scale)

`mean` = deterministic prediction (bNEE + drift*dt), `scale` = exp(nee_log_std)
(= sigma*sqrt(dt)), both aligned to the observed target `obs` = NEE_{t+1}.

All functions are pure numpy/scipy so they are unit-testable and reusable from
scripts/evaluate.py and from ad-hoc analysis.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

log = logging.getLogger("wienernet.evaluation.probabilistic")

Family = str  # "gaussian" | "student_t" | "laplace"
_FAMILIES = ("gaussian", "student_t", "laplace")
_SQRT_PI = float(np.sqrt(np.pi))


# ---------------------------------------------------------------------------
# Predictive-distribution primitives (dispatch on family)
# ---------------------------------------------------------------------------

def _check_family(family: Family, nu: float | None) -> None:
    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r}; valid: {_FAMILIES}")
    if family == "student_t" and (nu is None or not np.isfinite(nu) or nu <= 1):
        raise ValueError("student_t needs a finite dof nu > 1 (finite mean, so CRPS/NLL/PIT defined)")


def predictive_logpdf(
    obs: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    *, family: Family = "gaussian", nu: float | None = None,
) -> np.ndarray:
    """Per-sample log predictive density of the observation."""
    _check_family(family, nu)
    obs, mean, scale = map(_flat, (obs, mean, scale))
    if family == "gaussian":
        return stats.norm.logpdf(obs, loc=mean, scale=scale)
    if family == "laplace":
        return stats.laplace.logpdf(obs, loc=mean, scale=scale)
    return stats.t.logpdf(obs, df=nu, loc=mean, scale=scale)


def predictive_cdf(
    obs: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    *, family: Family = "gaussian", nu: float | None = None,
) -> np.ndarray:
    """Predictive CDF evaluated at the observation (the PIT value)."""
    _check_family(family, nu)
    obs, mean, scale = map(_flat, (obs, mean, scale))
    if family == "gaussian":
        return stats.norm.cdf(obs, loc=mean, scale=scale)
    if family == "laplace":
        return stats.laplace.cdf(obs, loc=mean, scale=scale)
    return stats.t.cdf(obs, df=nu, loc=mean, scale=scale)


def predictive_interval(
    mean: np.ndarray, scale: np.ndarray, level: float,
    *, family: Family = "gaussian", nu: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Central predictive interval [lo, hi] at the given nominal level."""
    _check_family(family, nu)
    mean, scale = _flat(mean), _flat(scale)
    alpha = (1.0 - level) / 2.0
    if family == "gaussian":
        q = stats.norm.ppf([alpha, 1 - alpha])
    elif family == "laplace":
        q = stats.laplace.ppf([alpha, 1 - alpha])
    else:
        q = stats.t.ppf([alpha, 1 - alpha], df=nu)
    return mean + scale * q[0], mean + scale * q[1]


def sample_predictive(
    mean: np.ndarray, scale: np.ndarray, n_samples: int,
    *, family: Family = "gaussian", nu: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw an (n, n_samples) ensemble from the per-point predictive law."""
    _check_family(family, nu)
    rng = rng or np.random.default_rng(0)
    mean, scale = _flat(mean), _flat(scale)
    n = mean.shape[0]
    if family == "gaussian":
        eps = rng.standard_normal((n, n_samples))
    elif family == "laplace":
        eps = rng.laplace(0.0, 1.0, (n, n_samples))
    else:
        eps = rng.standard_t(nu, (n, n_samples))
    return mean[:, None] + scale[:, None] * eps


# ---------------------------------------------------------------------------
# CRPS — the headline proper score
# ---------------------------------------------------------------------------

def crps_gaussian(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution (per sample).

    CRPS(N(mu,sigma), y) = sigma * [ z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],
    z = (y - mu)/sigma  (Gneiting & Raftery 2007). Lower is better.
    """
    obs, mean, std = map(_flat, (obs, mean, std))
    z = (obs - mean) / std
    return std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1.0 / _SQRT_PI)


def crps_ensemble(obs: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """CRPS estimated from an ensemble of forecast samples (per sample).

    Uses the sorted representation of CRPS = E|X-y| - 0.5 E|X-X'|, which is
    O(m log m) per point and reduces to |mean-y| for a degenerate (point)
    ensemble. `samples` is (n, m); `obs` is (n,). Lower is better.
    """
    obs = _flat(obs)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[0] != obs.shape[0]:
        raise ValueError(f"samples rows {samples.shape[0]} != obs {obs.shape[0]}")
    m = samples.shape[1]
    s = np.sort(samples, axis=1)
    y = obs[:, None]
    # weights i (1-indexed): (2i - 1 - m) for the E|X-X'| term via sorted trick.
    i = np.arange(1, m + 1)[None, :]
    term1 = np.abs(s - y).mean(axis=1)                    # E|X - y|
    term2 = (s * (2 * i - 1 - m)).sum(axis=1) / (m * m)   # 0.5 E|X - X'|
    return term1 - term2


def crps(
    obs: np.ndarray, mean: np.ndarray, scale: np.ndarray | None = None,
    *, family: Family = "gaussian", nu: float | None = None,
    n_samples: int = 400, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-sample CRPS, dispatching on the predictive family.

    scale=None (or a deterministic point forecast) -> CRPS = |obs - mean|, so
    deterministic baselines are scored on the same axis. Gaussian uses the
    closed form; other families use an ensemble estimate.
    """
    obs, mean = _flat(obs), _flat(mean)
    if scale is None:
        return np.abs(obs - mean)
    scale = _flat(scale)
    if family == "gaussian":
        return crps_gaussian(obs, mean, scale)
    samples = sample_predictive(mean, scale, n_samples, family=family, nu=nu, rng=rng)
    return crps_ensemble(obs, samples)


def ensemble_scores(
    obs: np.ndarray, samples: np.ndarray,
    *, levels: Sequence[float] = (0.5, 0.9, 0.95), rng_seed: int = 0,
) -> dict:
    """Score a per-timestep forecast *ensemble* (n, m) against observations.

    Family-agnostic: works for stochastic models whose predictive law is only
    available as samples (repeated latent+noise draws), and reduces to CRPS=|err|
    for a degenerate one-sample ensemble. Returns CRPS, a rank-based PIT summary,
    and empirical interval coverage/sharpness. This is the honest axis for the
    sampling / MMD variants, whose parametric `nee_std` under-represents spread.
    """
    obs = _flat(obs)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    finite = np.isfinite(obs) & np.all(np.isfinite(samples), axis=1)
    obs, samples = obs[finite], samples[finite]
    m = samples.shape[1]
    crps_ps = crps_ensemble(obs, samples)
    # randomized rank PIT (handles ties; uniform under calibration)
    rng = np.random.default_rng(rng_seed)
    below = np.sum(samples < obs[:, None], axis=1)
    equal = np.sum(samples == obs[:, None], axis=1)
    pit = (below + rng.random(obs.size) * (equal + 1)) / (m + 1)
    out: dict = {"n": int(obs.size), "n_samples": int(m),
                 "crps": float(np.mean(crps_ps)), "pit": pit_summary(pit),
                 "coverage": {}}
    for lv in levels:
        a = (1.0 - lv) / 2.0
        lo = np.quantile(samples, a, axis=1)
        hi = np.quantile(samples, 1 - a, axis=1)
        out["coverage"][f"{lv:.2f}"] = {
            "nominal": float(lv),
            "coverage": float(np.mean((obs >= lo) & (obs <= hi))),
            "sharpness": float(np.mean(hi - lo)),
        }
    return out


# ---------------------------------------------------------------------------
# Calibration — PIT and interval coverage / sharpness
# ---------------------------------------------------------------------------

def pit_summary(pit: np.ndarray, *, n_bins: int = 20) -> dict:
    """Summarise PIT values (predictive CDF at the observation).

    A calibrated model has uniform PIT: flat histogram, mean 1/2, var 1/12.
    U-shape (var > 1/12) => under-dispersed (spread too narrow); dome
    (var < 1/12) => over-dispersed. `ks_uniform` is the KS distance from
    Uniform(0,1) (0 = perfectly calibrated).
    """
    pit = _flat(pit)
    pit = pit[np.isfinite(pit)]
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    ks = float(stats.kstest(pit, "uniform").statistic) if pit.size else float("nan")
    return {
        "n": int(pit.size),
        "mean": float(np.mean(pit)) if pit.size else float("nan"),
        "var": float(np.var(pit)) if pit.size else float("nan"),
        "var_ideal": 1.0 / 12.0,
        "ks_uniform": ks,
        "hist_counts": counts.tolist(),
        "hist_edges": edges.tolist(),
    }


def interval_coverage(
    obs: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    *, family: Family = "gaussian", nu: float | None = None,
    levels: Sequence[float] = (0.5, 0.9, 0.95),
) -> dict[str, dict[str, float]]:
    """Empirical coverage and mean interval width (sharpness) per nominal level.

    Target: empirical coverage ~ nominal with the narrowest intervals. Returns
    {"0.90": {nominal, coverage, sharpness}, ...}.
    """
    obs = _flat(obs)
    out: dict[str, dict[str, float]] = {}
    for lv in levels:
        lo, hi = predictive_interval(mean, scale, lv, family=family, nu=nu)
        inside = (obs >= lo) & (obs <= hi)
        out[f"{lv:.2f}"] = {
            "nominal": float(lv),
            "coverage": float(np.mean(inside)),
            "sharpness": float(np.mean(hi - lo)),
        }
    return out


# ---------------------------------------------------------------------------
# Significance — Diebold-Mariano on paired per-timestep scores
# ---------------------------------------------------------------------------

def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, *, h: int = 1, hac_lags: int | None = None,
) -> dict[str, float]:
    """Diebold-Mariano test on paired per-timestep losses (e.g. CRPS).

    d_t = loss_a - loss_b. Tests H0: E[d]=0 (equal predictive accuracy). A
    negative statistic with small p means model A is significantly better
    (lower loss). Uses a Newey-West (HAC) variance with `hac_lags` lags
    (default h-1) and the Harvey-Leybourne-Newbold small-sample correction;
    p-value from a t distribution with n-1 df.
    """
    d = _flat(loss_a) - _flat(loss_b)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 8:
        return {"dm_stat": float("nan"), "p_value": float("nan"),
                "mean_diff": float(np.mean(d)) if n else float("nan"), "n": int(n)}
    dbar = float(np.mean(d))
    dc = d - dbar
    lags = (h - 1) if hac_lags is None else hac_lags
    gamma0 = float(np.mean(dc * dc))
    var = gamma0
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)            # Bartlett weight
        gk = float(np.mean(dc[k:] * dc[:-k]))
        var += 2.0 * wk * gk
    var = max(var, 1e-300)
    dm = dbar / np.sqrt(var / n)
    # Harvey, Leybourne & Newbold (1997) small-sample correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corr = dm * corr
    p = 2.0 * stats.t.sf(abs(dm_corr), df=n - 1)
    return {"dm_stat": float(dm_corr), "p_value": float(p), "mean_diff": dbar, "n": int(n)}


# ---------------------------------------------------------------------------
# Supporting context — irreducible measurement-noise floor
# ---------------------------------------------------------------------------

def measurement_noise_floor(
    residual: np.ndarray, flux: np.ndarray, *, n_bins: int = 16,
) -> dict[str, float]:
    """Estimate the irreducible point-error floor from a flux-scaled random-error
    model, sigma(flux) = a + b*|flux| (Hollinger & Richardson).

    Fits the per-flux-bin residual SD vs |flux|, then reports the RMSE floor
    sqrt(mean sigma^2) and MAE floor (Gaussian: sigma*sqrt(2/pi)). This is the
    accuracy no point model can beat; report it beside RMSE/MAE to show the
    point axis can't discriminate.
    """
    r = _flat(residual); f = np.abs(_flat(flux))
    m = np.isfinite(r) & np.isfinite(f)
    r, f = r[m], f[m]
    if r.size < 3 * n_bins:
        return {"a": float("nan"), "b": float("nan"),
                "rmse_floor": float("nan"), "mae_floor": float("nan"), "n": int(r.size)}
    qs = np.unique(np.quantile(f, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(f, qs[1:-1]), 0, len(qs) - 2)
    centers, sds = [], []
    for b in range(len(qs) - 1):
        sel = idx == b
        if sel.sum() >= 20:
            centers.append(np.median(f[sel])); sds.append(np.std(r[sel], ddof=1))
    centers, sds = np.array(centers), np.array(sds)
    A = np.vstack([centers, np.ones_like(centers)]).T
    b_slope, a_int = np.linalg.lstsq(A, sds, rcond=None)[0]
    sigma = a_int + b_slope * f
    return {
        "a": float(a_int), "b": float(b_slope),
        "rmse_floor": float(np.sqrt(np.mean(sigma ** 2))),
        "mae_floor": float(np.mean(np.abs(sigma)) * np.sqrt(2.0 / np.pi)),
        "n": int(r.size),
    }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def probabilistic_scores(
    obs: np.ndarray, mean: np.ndarray, scale: np.ndarray | None,
    *, family: Family = "gaussian", nu: float | None = None,
    levels: Sequence[float] = (0.5, 0.9, 0.95),
    n_samples: int = 400, rng_seed: int = 0,
    return_per_sample: bool = False,
) -> dict:
    """Headline probabilistic scores for one (obs, predictive-dist) pair.

    Distributional model -> full set (CRPS, NLL, PIT, coverage/sharpness).
    Deterministic model (scale None) -> CRPS(=MAE) only; NLL/PIT are null.
    """
    obs, mean = _flat(obs), _flat(mean)
    finite = np.isfinite(obs) & np.isfinite(mean)
    if scale is not None:
        scale = _flat(scale)
        finite &= np.isfinite(scale) & (scale > 0)
    obs, mean = obs[finite], mean[finite]
    sc = scale[finite] if scale is not None else None
    rng = np.random.default_rng(rng_seed)

    crps_ps = crps(obs, mean, sc, family=family, nu=nu, n_samples=n_samples, rng=rng)
    out: dict = {
        "n": int(obs.size),
        "family": family if sc is not None else "point",
        "nu": float(nu) if (nu is not None and family == "student_t") else None,
        "crps": float(np.mean(crps_ps)),
    }
    if sc is None:
        out.update({"nll": None, "pit": None, "coverage": None})
        return (out, {"crps": crps_ps}) if return_per_sample else out

    out["nll"] = float(-np.mean(predictive_logpdf(obs, mean, sc, family=family, nu=nu)))
    pit = predictive_cdf(obs, mean, sc, family=family, nu=nu)
    out["pit"] = pit_summary(pit)
    out["coverage"] = interval_coverage(obs, mean, sc, family=family, nu=nu, levels=levels)
    if return_per_sample:
        return out, {"crps": crps_ps, "pit": pit, "logpdf": predictive_logpdf(
            obs, mean, sc, family=family, nu=nu)}
    return out


def stratified_scores(
    obs: np.ndarray, mean: np.ndarray, scale: np.ndarray | None,
    groups: Mapping[str, np.ndarray],
    *, family: Family = "gaussian", nu: float | None = None,
    levels: Sequence[float] = (0.5, 0.9, 0.95), min_n: int = 100,
) -> dict[str, dict[str, dict]]:
    """Compute probabilistic_scores within each level of each grouping.

    `groups` maps a dimension name (e.g. "season", "site", "hour_bin") to a
    per-sample label array. Returns {dimension: {label: scores}}. Shows whether
    calibration holds as the dynamics change (non-stationarity check).
    """
    obs, mean = _flat(obs), _flat(mean)
    sc = _flat(scale) if scale is not None else None
    out: dict[str, dict[str, dict]] = {}
    for dim, labels in groups.items():
        labels = np.asarray(labels)
        out[dim] = {}
        for lab in pd_unique(labels):
            sel = labels == lab
            if sel.sum() < min_n:
                continue
            out[dim][str(lab)] = probabilistic_scores(
                obs[sel], mean[sel], sc[sel] if sc is not None else None,
                family=family, nu=nu, levels=levels,
            )
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def pd_unique(a: np.ndarray) -> np.ndarray:
    """Unique labels preserving first-seen order, NaN-safe (avoids importing pandas
    just for this)."""
    seen: list = []
    out: list = []
    for v in a:
        key = v if not (isinstance(v, float) and np.isnan(v)) else "__nan__"
        if key not in seen:
            seen.append(key); out.append(v)
    return np.array(out, dtype=a.dtype)
