"""Lloyd-Taylor respiration model — torch version for in-graph use.

The numpy version lives in `data_pipeline.partitioning.lloyd_taylor` and is
used for offline parameter fitting. This torch version is used inside the
model's forward pass so that gradients flow through (E0, rb, T).

Both versions use the same convention: T0 is stored as a positive number
(46.02) and the denominators become (T + T0). This is the Reichstein 2005
form with T0 = -46.02 °C.
"""

from __future__ import annotations

import torch

# Reichstein 2005 constants (in °C). T0 stored positive; used as `T + T0` in
# the denominator. See data_pipeline/config.py for the matching numpy version.
DEFAULT_TREF = 10.0
DEFAULT_T0 = 46.02


def reco(
    T: torch.Tensor,
    E0: torch.Tensor,
    rb: torch.Tensor,
    *,
    tref: float = DEFAULT_TREF,
    t0: float = DEFAULT_T0,
) -> torch.Tensor:
    """Reco(T) = rb * exp(E0 * (1/(Tref + T0) - 1/(T + T0)))."""
    exp_term = torch.exp(E0 * (1.0 / (tref + t0) - 1.0 / (T + t0)))
    return rb * exp_term


def dreco_dT(
    T: torch.Tensor,
    E0: torch.Tensor,
    rb: torch.Tensor,
    *,
    tref: float = DEFAULT_TREF,
    t0: float = DEFAULT_T0,
) -> torch.Tensor:
    """Analytic d(Reco)/dT = rb * exp(...) * E0 / (T + T0)^2."""
    exp_term = torch.exp(E0 * (1.0 / (tref + t0) - 1.0 / (T + t0)))
    return rb * (E0 / (T + t0) ** 2) * exp_term


def sde_drift(
    T: torch.Tensor,
    E0: torch.Tensor,
    rb: torch.Tensor,
    dT_dt: torch.Tensor,
    *,
    tref: float = DEFAULT_TREF,
    t0: float = DEFAULT_T0,
) -> torch.Tensor:
    """The drift term of the Wiener SDE: f = d(NEE)/dT * dT/dt.

    For nighttime: NEE = Reco (GPP=0), so this is dReco/dT * dT/dt by chain rule.
    Applied directly inside the model's forward pass — gradients flow through
    (E0, rb), allowing the k_decoder to be trained end-to-end.
    """
    return dreco_dT(T, E0, rb, tref=tref, t0=t0) * dT_dt
