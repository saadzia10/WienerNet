"""Increment/level likelihood losses (Gaussian / beta-NLL / Student-t).

Replaces the ``mse_nee + mse_drift + mmd_noise`` anchoring for the stochastic
part of the model with a proper conditional likelihood of the target given the
predicted (mean, scale).

Why a likelihood and not ``MSE(mean + noise, target)``:
    E[MSE(mean + noise, target)] = (mean - target)^2 + Var(noise)  -> minimised at Var->0.
MSE actively kills the noise; only an external prior (mmd_noise) keeps it alive,
so the noise ends up matching a *prior*, not the true conditional variance. A
likelihood rewards *calibration*: the ``(target-mean)^2 / s^2`` term punishes a
too-small scale, the ``log s`` term punishes a too-large scale, so ``s`` settles
at the real conditional spread with nothing to tune. The mean and scale are
orthogonal in these families, so the deterministic drift/residual (mean) and the
aleatoric noise (scale) do not fight.

The three variants operate on the SAME ``(target, mean, log_std)`` interface — the
model maps its SDE moments to the target scale:
    increment:  mean = bNEE + (f_phys + r)*dt ,  log_std = log_sigma + 0.5*log(dt)
    level:      mean = nee_raw + drift*dt      ,  log_std = log_sigma_level
Because the NLL of ``target ~ N(mean, s^2)`` is shift-invariant, evaluating it on
NEE_{t+1} with the level mean is identical to evaluating it on the increment ΔNEE
with the increment mean — one code path serves both families.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)
# Clamp log_std so the inverse variance stays finite early in training (guards
# the variance-collapse / explosion failure mode of heteroscedastic NLL).
LOG_STD_MIN = -7.0
LOG_STD_MAX = 5.0

# Student-t degrees-of-freedom floor: nu = softplus(log_nu) + NU_FLOOR. Floored at
# 2 so the fitted law keeps a finite mean rather than collapsing to the extreme
# heavy tails (nu ~ 1.3) the raw NLL optimum prefers on this data — that collapse
# gives an infinite-variance predictive law that destabilised sampling and scoring.
# nu can still learn upward (toward Gaussian) from the floor.
NU_FLOOR = 2.0


def student_t_dof(log_nu: torch.Tensor) -> torch.Tensor:
    """Positive Student-t degrees of freedom nu = softplus(log_nu) + NU_FLOOR."""
    return F.softplus(log_nu) + NU_FLOOR


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Per-element Gaussian NLL of ``target ~ N(mean, exp(log_std)^2)``.

    Drives mean -> E[target|state] and exp(log_std) -> Std[target|state].
    """
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    inv_var = torch.exp(-2.0 * log_std)
    return 0.5 * (target - mean) ** 2 * inv_var + log_std + 0.5 * LOG_2PI


def beta_nll(
    target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, *, beta: float = 0.5
) -> torch.Tensor:
    """Gaussian NLL with per-sample gradient re-weighting by detached s^{2*beta}.

    Seitzer et al. (ICLR 2022). Fixes vanilla heteroscedastic NLL's failure mode:
    in high-variance regions the 1/s^2 factor down-weights the mean gradient, so
    the mean fit degrades. Re-weighting by stop_grad(s^{2*beta}) restores it.
    beta=0 -> plain Gaussian NLL; beta=0.5 -> balanced (default); beta=1 -> the
    mean gradient behaves like plain MSE (most stable, least sharp variance).
    """
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    inv_var = torch.exp(-2.0 * log_std)
    nll = 0.5 * (target - mean) ** 2 * inv_var + log_std + 0.5 * LOG_2PI
    with torch.no_grad():
        weight = torch.exp(2.0 * beta * log_std)   # (s^2)^beta, detached
    return weight * nll


def student_t_nll(
    target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, log_nu: torch.Tensor
) -> torch.Tensor:
    """Per-element Student-t NLL: location=mean, scale=exp(log_std), dof=nu.

    Same self-regularising mean/scale behaviour as the Gaussian but with heavy
    tails — matches the skewed / heavy-tailed residuals and helps the tail-
    sensitive metrics (KL, Wasserstein, MMD). ``log_nu`` is a single learnable
    scalar; nu = softplus(log_nu) + NU_FLOOR (>= 2, finite mean). As nu -> inf this
    recovers the Gaussian.
    """
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    nu = student_t_dof(log_nu)
    z2 = ((target - mean) * torch.exp(-log_std)) ** 2
    return (
        -torch.lgamma((nu + 1.0) / 2.0)
        + torch.lgamma(nu / 2.0)
        + 0.5 * torch.log(nu * math.pi)
        + log_std
        + 0.5 * (nu + 1.0) * torch.log1p(z2 / nu)
    )


def _gaussian_logpdf(target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Per-element Gaussian log-density (the negative of ``gaussian_nll``)."""
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    z2 = ((target - mean) * torch.exp(-log_std)) ** 2
    return -0.5 * z2 - log_std - 0.5 * LOG_2PI


def _student_t_logpdf(
    target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, log_nu: torch.Tensor
) -> torch.Tensor:
    """Per-element Student-t log-density (the negative of ``student_t_nll``)."""
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    nu = student_t_dof(log_nu)
    z2 = ((target - mean) * torch.exp(-log_std)) ** 2
    return (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu * math.pi)
        - log_std
        - 0.5 * (nu + 1.0) * torch.log1p(z2 / nu)
    )


def mixture_nll(
    target: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    logits: torch.Tensor,
    *,
    variant: str = "gaussian",
    log_nu: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """NLL of a K-component mixture density (mixture-density-network head).

    ``means`` / ``log_scales`` / ``logits`` are ``(N, K)``; the predictive law per
    row is ``sum_k softmax(logits)_k * comp(y; means_k, exp(log_scales)_k)`` with
    ``comp`` a Gaussian or (shared-dof) Student-t. The NLL is
    ``-logsumexp_k( log_softmax(logits)_k + log comp_k )`` — the proper training
    objective for the no-physics MDN baseline, which no single (mean, scale) pair
    can represent. Reduces to the plain Gaussian/Student-t NLL when K = 1.
    """
    target = target.view(-1, 1)
    log_w = F.log_softmax(logits, dim=1)
    if variant == "gaussian":
        comp = _gaussian_logpdf(target, means, log_scales)
    elif variant == "student_t":
        if log_nu is None:
            raise ValueError("student_t mixture NLL needs a learnable `log_nu` (set noise_student_dof=True)")
        comp = _student_t_logpdf(target, means, log_scales, log_nu)
    else:
        raise ValueError(f"unknown mixture NLL variant {variant!r}; valid: gaussian|student_t")
    per = -torch.logsumexp(log_w + comp, dim=1)
    if reduction == "sum":
        return per.sum()
    return per.mean()


# Clamp on log_kappa so the asymmetry stays in a sane, invertible range
# (kappa in ~[0.05, 20]); kappa=1 is the symmetric Laplace.
LOG_KAPPA_MIN, LOG_KAPPA_MAX = -3.0, 3.0


def asymmetric_laplace_nll(
    target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, log_kappa: torch.Tensor,
    *, beta: float = 0.0,
) -> torch.Tensor:
    """Per-element asymmetric-Laplace (ALD) NLL: location=mean, scale=exp(log_std),
    asymmetry kappa=exp(log_kappa).

    Density  f(x) = (1/sigma) * (kappa/(1+kappa^2)) * exp(-rho_kappa((x-mean)/sigma)),
    rho_kappa(u) = kappa*u for u>=0 else -u/kappa. kappa=1 -> symmetric Laplace;
    kappa<1 -> right-skewed (heavier right tail), matching the observed +skew. The
    asymmetric abs `rho` is the pinball loss, so this is the ML fit of a skewed,
    heavy-tailed (double-exponential) noise while the location stays the drift.

    `beta` applies the Seitzer et al. beta-NLL gradient re-weighting (detached
    s^{2*beta}), same as `beta_nll`. The pinball location gradient scales as 1/sigma,
    so when the scale is enlarged (physics-anchored scale, to fix coverage) the mean
    fit degrades; beta=0.5 restores an MSE-like (scale-independent) mean gradient.
    beta=0 (default) is the plain ALD NLL.
    """
    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    kappa = torch.exp(log_kappa.clamp(LOG_KAPPA_MIN, LOG_KAPPA_MAX))
    u = (target - mean) * torch.exp(-log_std)
    rho = torch.where(u >= 0, kappa * u, -u / kappa)
    per = log_std - torch.log(kappa) + torch.log1p(kappa ** 2) + rho
    if beta:
        with torch.no_grad():
            weight = torch.exp(2.0 * beta * log_std)
        per = weight * per
    return per


def compute_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    *,
    variant: str = "gaussian",
    beta: float = 0.5,
    log_nu: torch.Tensor | None = None,
    log_kappa: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Dispatch to the requested NLL variant and reduce.

    variant: 'gaussian' | 'beta' | 'student_t' | 'ald'. 'student_t' requires
    ``log_nu``; 'ald' requires ``log_kappa`` (learnable scalars carried on the
    model). Returns a scalar tensor.
    """
    target = target.view_as(mean)
    if variant == "gaussian":
        per = gaussian_nll(target, mean, log_std)
    elif variant == "beta":
        per = beta_nll(target, mean, log_std, beta=beta)
    elif variant == "student_t":
        if log_nu is None:
            raise ValueError("student_t NLL needs a learnable `log_nu` (set model noise_student_dof=True)")
        per = student_t_nll(target, mean, log_std, log_nu)
    elif variant == "ald":
        if log_kappa is None:
            raise ValueError("ald NLL needs a learnable `log_kappa` (set model noise_asymmetry=True)")
        per = asymmetric_laplace_nll(target, mean, log_std, log_kappa, beta=beta)
    else:
        raise ValueError(f"unknown NLL variant {variant!r}; valid: gaussian|beta|student_t|ald")
    if reduction == "sum":
        return per.sum()
    return per.mean()
