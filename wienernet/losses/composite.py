"""Loss assembly: turn batch + model_outputs + weight dict into per-term losses.

This replaces the per-trainer `loss_function` methods scattered across all
five night/*.py files. Weights of 0 disable a term, so you don't need to
delete code to ablate a loss — just edit the config.

The terms supported (any subset can be enabled):

    Point-wise (MSE):
        mse_nee              — final NEE prediction vs ground truth NEE
        mse_bnee             — boundary NEE (before drift) vs ground truth NEE
        mse_E0, mse_rb       — predicted Lloyd-Taylor parameters
        mse_temp_derivative  — predicted dT/dt
        mse_drift            — predicted f vs target dNEE (finite-difference)
        residual_l2          — L2 penalty on the increment model's residual head

    Distributional (MMD):
        mmd_nee, mmd_bnee    — match the distribution of NEE / boundary NEE
        mmd_noise            — match the noise term to a Gaussian prior

    KL (for VAE-style latent):
        kl_latent            — standard VAE KL term: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .likelihood import compute_nll, mixture_nll


def compute_losses(
    batch: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor | None],
    weights: Mapping[str, float],
    *,
    mmd_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    noise_prior_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    target_scales: Mapping[str, float] | None = None,
    likelihood: Mapping[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute every enabled loss term.

    Args:
        batch: dataloader batch with keys 'NEE', 'bNEE', 'k', 'dT', 'dNEE', 'T'.
            Tensors are expected to be on the same device as outputs.
        outputs: model forward output dict. Keys can be None for disabled heads.
        weights: dict {term_name: weight}. Terms with weight == 0 (or missing)
            are skipped, allowing cheap ablations.
        mmd_loss_fn: callable(source, target) -> scalar tensor. Required iff any
            mmd_* weight is non-zero.
        noise_prior_fn: callable(noise) -> noise_prior sample. The default
            in the reference code is `randn_like(noise) * noise_std + noise_mu`.
        target_scales: optional {mse_term: std}. Each listed MSE is divided by
            scale**2, turning it into ~"fraction of variance unexplained" (O(1)).
            Only terms present in the dict are scaled; the rest keep their raw
            scale. The loader supplies scales for the k anchors ONLY (E0~112,
            rb~4) because those live on the REddyProc-parameter scale and their
            raw MSE (~1e4) dominates the loss; the flux-scale terms (mse_nee) and
            the naturally-small physics terms (mse_temp_derivative, mse_drift) are
            left alone — normalising them to unit variance would force the weak
            30-min physics up to parity with mse_nee and hurt the fit. MMD / KL
            are never scaled.

    Returns:
        Dict {term_name: weighted_loss_tensor}. The caller sums these for the
        scalar to backprop. Returning a dict (not just the sum) lets the
        trainer log each term separately to TensorBoard/MLflow.
    """
    losses: dict[str, torch.Tensor] = {}

    def _w(name: str) -> float:
        return float(weights.get(name, 0.0))

    def _s2(name: str) -> float:
        """1 / scale**2 normaliser for MSE term `name` (1.0 if no scale given)."""
        if not target_scales:
            return 1.0
        s = float(target_scales.get(name, 1.0))
        return 1.0 / (s * s) if s > 0 else 1.0

    # ------------------------------------------------------------------
    # MSE — point-wise
    # ------------------------------------------------------------------
    if _w("mse_nee") > 0 and outputs.get("nee_pred") is not None:
        losses["mse_nee"] = _w("mse_nee") * _s2("mse_nee") * F.mse_loss(
            outputs["nee_pred"], batch["NEE"].view(-1, 1)
        )

    if _w("mse_bnee") > 0 and outputs.get("bnee") is not None:
        losses["mse_bnee"] = _w("mse_bnee") * _s2("mse_bnee") * F.mse_loss(
            outputs["bnee"], batch["bNEE"].view(-1, 1)
        )

    if outputs.get("k") is not None:
        if _w("mse_E0") > 0:
            losses["mse_E0"] = _w("mse_E0") * _s2("mse_E0") * F.mse_loss(
                outputs["k"][:, 0:1], batch["k"][:, 0:1]
            )
        if _w("mse_rb") > 0:
            losses["mse_rb"] = _w("mse_rb") * _s2("mse_rb") * F.mse_loss(
                outputs["k"][:, 1:2], batch["k"][:, 1:2]
            )

    if _w("mse_temp_derivative") > 0 and outputs.get("temp_derivative") is not None:
        # Anchor the tendency head to the physics diurnal target when the model provides
        # one (drift_tendency="learned_diurnal"), else to the observed dTa.
        _td_target = outputs.get("temp_derivative_target")
        if _td_target is None:
            _td_target = batch["dT"].view(-1, 1)
        losses["mse_temp_derivative"] = _w("mse_temp_derivative") * _s2("mse_temp_derivative") * F.mse_loss(
            outputs["temp_derivative"], _td_target
        )

    if _w("mse_drift") > 0 and outputs.get("drift") is not None:
        losses["mse_drift"] = _w("mse_drift") * _s2("mse_drift") * F.mse_loss(
            outputs["drift"], batch["dNEE"].view(-1, 1)
        )

    # L2 on the increment model's residual drift-misfit head (version A). Keeps
    # the residual a *small* correction so the physics explains first; weight 0
    # (or no residual head) makes it a no-op.
    if _w("residual_l2") > 0 and outputs.get("residual") is not None:
        losses["residual_l2"] = _w("residual_l2") * outputs["residual"].pow(2).mean()

    # ------------------------------------------------------------------
    # Likelihood (NLL) — the principled replacement for mse_nee + mse_drift +
    # mmd_noise on the stochastic part. The model exposes (nee_mean, nee_log_std)
    # on the NEE_{t+1} target scale; the NLL makes nee_mean -> E[NEE|state]
    # (drift/residual) and exp(nee_log_std) -> Std[NEE|state] (aleatoric noise),
    # with no prior to tune. Set likelihood={'variant','weight','beta'} to enable
    # and zero the mse_nee/mse_drift/mmd_noise weights. See losses/likelihood.py.
    # ------------------------------------------------------------------
    if likelihood is not None:
        w_nll = float(likelihood.get("weight", 0.0))
        variant = str(likelihood.get("variant", "gaussian"))
        # Mixture-density head (no-physics MDN baseline): a K-component mixture NLL,
        # which no single (mean, scale) can represent. Detected via the mix_* keys.
        if w_nll > 0 and outputs.get("mix_logits") is not None:
            mix_variant = "student_t" if variant == "student_t" else "gaussian"
            losses["nll"] = w_nll * mixture_nll(
                batch["NEE"].view(-1, 1),
                outputs["mix_means"],
                outputs["mix_log_scales"],
                outputs["mix_logits"],
                variant=mix_variant,
                log_nu=outputs.get("log_nu"),
            )
        elif (w_nll > 0 and outputs.get("nee_mean") is not None
                and outputs.get("nee_log_std") is not None):
            losses["nll"] = w_nll * compute_nll(
                batch["NEE"].view(-1, 1),
                outputs["nee_mean"],
                outputs["nee_log_std"],
                variant=variant,
                beta=float(likelihood.get("beta", 0.5)),
                log_nu=outputs.get("log_nu"),
                log_kappa=outputs.get("log_kappa"),
            )

    # ------------------------------------------------------------------
    # MMD — distributional
    # ------------------------------------------------------------------
    needs_mmd = any(_w(k) > 0 for k in ("mmd_nee", "mmd_bnee", "mmd_noise"))
    if needs_mmd and mmd_loss_fn is None:
        raise ValueError("mmd_loss_fn is required for any non-zero mmd_* weight")

    if _w("mmd_nee") > 0 and outputs.get("nee_pred") is not None:
        losses["mmd_nee"] = _w("mmd_nee") * mmd_loss_fn(
            batch["NEE"].view(-1, 1), outputs["nee_pred"]
        )

    if _w("mmd_bnee") > 0 and outputs.get("bnee") is not None:
        losses["mmd_bnee"] = _w("mmd_bnee") * mmd_loss_fn(
            batch["bNEE"].view(-1, 1), outputs["bnee"]
        )

    if _w("mmd_noise") > 0 and outputs.get("noise") is not None:
        if noise_prior_fn is None:
            raise ValueError("noise_prior_fn is required for mmd_noise > 0")
        prior = noise_prior_fn(outputs["noise"])
        losses["mmd_noise"] = _w("mmd_noise") * mmd_loss_fn(outputs["noise"], prior)

    # ------------------------------------------------------------------
    # KL — VAE-style latent regularizer
    # ------------------------------------------------------------------
    if _w("kl_latent") > 0 and outputs.get("latent_mu") is not None:
        mu = outputs["latent_mu"]
        logvar = outputs["latent_logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses["kl_latent"] = _w("kl_latent") * kl

    return losses


def make_gaussian_noise_prior(
    mean: float, std: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a noise_prior_fn that draws from N(mean, std^2) with the same
    shape and device as the sampled noise tensor.
    """
    def prior(noise: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(noise) * std + mean
    return prior


def make_empirical_noise_prior(
    residuals, *, center: bool = True
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a noise_prior_fn that resamples from an empirical residual pool.

    Unlike `make_gaussian_noise_prior`, this matches the noise to the *shape* of
    the real residual distribution (NEE - NEE_phy), which is skewed / heavy-
    tailed — a Gaussian target throws that away. With ``center=True`` the pool is
    de-meaned so the target is zero-mean and the deterministic head carries the
    location; matching an un-centred pool would re-impose the physics-misfit bias.

    Each call draws ``noise.shape`` samples with replacement (so MMD's equal-
    batch-size requirement holds) via the global RNG — seed with `set_seed_globally`
    for reproducibility.
    """
    pool = torch.as_tensor(np.asarray(residuals, dtype=np.float32).ravel())
    if pool.numel() == 0:
        raise ValueError("empirical noise prior needs a non-empty residual pool")
    if center:
        pool = pool - pool.mean()

    def prior(noise: torch.Tensor) -> torch.Tensor:
        idx = torch.randint(0, pool.numel(), noise.shape, device="cpu")
        return pool[idx].to(device=noise.device, dtype=noise.dtype)

    return prior
