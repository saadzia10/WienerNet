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

# Physical upper bound on the activation-energy parameter E0 (real fitted E0 is
# ~50-450; REddyProc bounds it below ~450). Clamping at 1000 is inert for any
# well-behaved model (the mse_E0 anchor keeps the predicted head near ~100), but
# prevents a transiently-diverging *predicted* E0 head from overflowing exp(...)
# to +inf and raising a non-finite loss. Paired with the exponent clamp below,
# it keeps both the exponential and the dReco/dT E0-factor finite.
# Physical upper bounds on the base respiration rate rb (~0.5-20) and the
# activation-energy E0 (~50-450). Clamping at 1000 is far above any real value and
# inert for a well-behaved model (the mse_E0/rb anchors keep the predicted heads
# near their GT scale), but prevents a transiently- or OOD-diverging *predicted*
# k-head from producing an overflowing / astronomically-large drift.
_E0_CLAMP_MAX = 1000.0
_RB_CLAMP_MAX = 1000.0
_EXP_CLAMP_MAX = 30.0


def reco(
    T: torch.Tensor,
    E0: torch.Tensor,
    rb: torch.Tensor,
    *,
    tref: float = DEFAULT_TREF,
    t0: float = DEFAULT_T0,
) -> torch.Tensor:
    """Reco(T) = rb * exp(E0 * (1/(Tref + T0) - 1/(T + T0)))."""
    E0 = E0.clamp(max=_E0_CLAMP_MAX)
    rb = rb.clamp(max=_RB_CLAMP_MAX)
    exponent = (E0 * (1.0 / (tref + t0) - 1.0 / (T + t0))).clamp(max=_EXP_CLAMP_MAX)
    return rb * torch.exp(exponent)


def dreco_dT(
    T: torch.Tensor,
    E0: torch.Tensor,
    rb: torch.Tensor,
    *,
    tref: float = DEFAULT_TREF,
    t0: float = DEFAULT_T0,
) -> torch.Tensor:
    """Analytic d(Reco)/dT = rb * exp(...) * E0 / (T + T0)^2."""
    E0 = E0.clamp(max=_E0_CLAMP_MAX)
    rb = rb.clamp(max=_RB_CLAMP_MAX)
    exponent = (E0 * (1.0 / (tref + t0) - 1.0 / (T + t0))).clamp(max=_EXP_CLAMP_MAX)
    return rb * (E0 / (T + t0) ** 2) * torch.exp(exponent)


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
