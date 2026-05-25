"""Numpy metrics for post-training evaluation.

These mirror the inline `evaluate()` function defined in cell 38 of
All_Results_Analysis.ipynb, but live in a module so they're testable and
the analysis notebook becomes a thin orchestration layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import rbf_kernel

log = logging.getLogger("wienernet.evaluation.metrics")


# ---------------------------------------------------------------------------
# Single-target metrics
# ---------------------------------------------------------------------------

def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(mean_absolute_error(_flat(y), _flat(y_hat)))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(_flat(y), _flat(y_hat))))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(r2_score(_flat(y), _flat(y_hat)))


def mmd_rbf(y: np.ndarray, y_hat: np.ndarray, *, gamma: float = 1.0) -> float:
    """Maximum mean discrepancy with an RBF kernel. Returns sqrt(MMD^2).

    Used as a single-number distributional distance — small means the two
    samples come from similar distributions.
    """
    y = _column(y)
    y_hat = _column(y_hat)
    K_yy = rbf_kernel(y, y, gamma=gamma)
    K_hh = rbf_kernel(y_hat, y_hat, gamma=gamma)
    K_yh = rbf_kernel(y, y_hat, gamma=gamma)
    val = K_yy.mean() + K_hh.mean() - 2 * K_yh.mean()
    # Numerical noise can make this very slightly negative
    return float(np.sqrt(max(val, 0.0)))


def kl_divergence_histogram(
    y: np.ndarray,
    y_hat: np.ndarray,
    *,
    bins: int = 100,
    epsilon: float = 1e-10,
) -> float:
    """Discrete KL between histograms of y and y_hat (shared bins from y)."""
    y = _flat(y)
    y_hat = _flat(y_hat)
    hist_y, edges = np.histogram(y, bins=bins, density=True)
    hist_h, _ = np.histogram(y_hat, bins=edges, density=True)
    hist_y = (hist_y + epsilon) / (hist_y.sum() + epsilon * len(hist_y))
    hist_h = (hist_h + epsilon) / (hist_h.sum() + epsilon * len(hist_h))
    return float(entropy(hist_y, hist_h))


def wasserstein(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(wasserstein_distance(_flat(y), _flat(y_hat)))


# ---------------------------------------------------------------------------
# Bundle: compute everything at once
# ---------------------------------------------------------------------------

@dataclass
class MetricBundle:
    """All single-target metrics for one (ground_truth, prediction) pair."""

    n: int                  # sample count
    mae: float
    rmse: float
    r2: float
    mmd: float | None       # may be None if MMD wasn't requested (it's O(n²) RAM)
    kl: float
    wasserstein: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mmd": self.mmd,
            "kl": self.kl,
            "wasserstein": self.wasserstein,
        }


def compute_metric_bundle(
    y: np.ndarray,
    y_hat: np.ndarray,
    *,
    include_mmd: bool = True,
    mmd_gamma: float = 1.0,
    mmd_subsample: int | None = 2000,
    kl_bins: int = 100,
) -> MetricBundle:
    """All metrics in one call.

    Args:
        include_mmd: MMD is O(n²) RAM. For n > ~5000 the kernel matrix gets
            heavy. Set False to skip, or use `mmd_subsample` for a tractable
            stratified sub-sample.
        mmd_subsample: if not None, randomly down-sample BOTH y and y_hat to
            this many points before computing MMD. Uses a deterministic seed.
    """
    y = _flat(y)
    y_hat = _flat(y_hat)
    if len(y) != len(y_hat):
        raise ValueError(f"length mismatch: y={len(y)}, y_hat={len(y_hat)}")
    n = len(y)
    if n == 0:
        raise ValueError("empty arrays")

    mmd_val: float | None = None
    if include_mmd:
        if mmd_subsample is not None and n > mmd_subsample:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, mmd_subsample, replace=False)
            mmd_val = mmd_rbf(y[idx], y_hat[idx], gamma=mmd_gamma)
        else:
            mmd_val = mmd_rbf(y, y_hat, gamma=mmd_gamma)

    return MetricBundle(
        n=n,
        mae=mae(y, y_hat),
        rmse=rmse(y, y_hat),
        r2=r2(y, y_hat),
        mmd=mmd_val,
        kl=kl_divergence_histogram(y, y_hat, bins=kl_bins),
        wasserstein=wasserstein(y, y_hat),
    )


def evaluate_predictions(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    *,
    targets: Iterable[str] = ("nee", "bnee", "E0", "rb", "dtemp", "f"),
    mask: np.ndarray | None = None,
    include_mmd: bool = True,
    mmd_gamma: float = 1.0,
    mmd_subsample: int | None = 2000,
) -> dict[str, dict[str, float | int | None]]:
    """Compute MetricBundle for every requested target that's present in both
    `gt` and `preds`, optionally masked to a subset of rows.

    Returns nested dict: {target: {metric_name: value}}.
    """
    out: dict[str, dict[str, float | int | None]] = {}
    for target in targets:
        if target not in gt or target not in preds:
            log.debug("skipping target %r (missing from gt or preds)", target)
            continue
        y = np.asarray(gt[target])
        h = np.asarray(preds[target])
        if mask is not None:
            mask_arr = np.asarray(mask)
            y = y[mask_arr]
            h = h[mask_arr]
        if len(y) == 0:
            log.warning("target %r: no rows after masking", target)
            continue
        bundle = compute_metric_bundle(
            y, h,
            include_mmd=include_mmd,
            mmd_gamma=mmd_gamma,
            mmd_subsample=mmd_subsample,
        )
        out[target] = bundle.as_dict()
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _flat(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    return arr.ravel()


def _column(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    return arr.reshape(-1, 1)
