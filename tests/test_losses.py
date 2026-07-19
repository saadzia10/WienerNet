"""Tests for MMD loss and the composite loss assembly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wienernet.losses import (
    MMDLoss,
    compute_losses,
    make_empirical_noise_prior,
    make_gaussian_noise_prior,
)


def test_mmd_self_distance_is_small():
    """MMD(X, X) should be near zero (it isn't exactly zero with multi-bandwidth)."""
    torch.manual_seed(0)
    x = torch.randn(64, 1)
    loss = MMDLoss()(x, x)
    # Multi-bandwidth MMD on the same sample produces a small but non-zero value
    assert loss.item() < 1e-5, f"MMD(X,X)={loss.item()} should be near 0"


def test_mmd_grows_with_distance():
    """MMD should increase as the two distributions are pulled apart."""
    torch.manual_seed(0)
    x = torch.randn(128, 1)
    y_near = x + torch.randn_like(x) * 0.1
    y_far = x + 10.0
    mmd_near = MMDLoss()(x, y_near).item()
    mmd_far = MMDLoss()(x, y_far).item()
    assert mmd_far > mmd_near, f"MMD didn't grow with distance: near={mmd_near}, far={mmd_far}"


def test_mmd_requires_equal_batch_sizes():
    """MMDLoss assumes |source| == |target|. This is preserved from the
    reference implementation. Document the constraint in the test."""
    torch.manual_seed(0)
    x = torch.randn(50, 1)
    y = torch.randn(80, 1)
    with pytest.raises(RuntimeError):
        MMDLoss()(x, y)


def test_compute_losses_skips_zero_weights():
    """A term with weight=0 should not appear in the result dict."""
    batch = {
        "NEE": torch.randn(8), "bNEE": torch.randn(8),
        "k": torch.randn(8, 2), "dT": torch.randn(8),
        "dNEE": torch.randn(8), "T": torch.randn(8),
    }
    outputs = {
        "nee_pred": torch.randn(8, 1, requires_grad=True),
        "bnee": torch.randn(8, 1),
        "k": torch.randn(8, 2),
        "temp_derivative": torch.randn(8, 1),
        "drift": torch.randn(8, 1),
        "noise": None, "noise_mu": None, "noise_logvar": None,
        "latent": None, "latent_mu": None, "latent_logvar": None,
        "nee_raw": None,
    }
    weights = {
        "mse_nee": 1.0,
        "mse_bnee": 0.0,         # disabled
        "mse_E0": 1.0,
        "mse_temp_derivative": 0.0,  # disabled
    }
    losses = compute_losses(batch, outputs, weights)
    assert set(losses.keys()) == {"mse_nee", "mse_E0"}, f"unexpected losses: {set(losses.keys())}"


def test_compute_losses_target_scales_normalise_only_listed_mse():
    """target_scales divides an MSE by scale**2; unlisted terms are untouched."""
    batch = {
        "NEE": torch.zeros(8), "bNEE": torch.zeros(8),
        "k": torch.zeros(8, 2), "dT": torch.zeros(8),
        "dNEE": torch.zeros(8), "T": torch.zeros(8),
    }
    outputs = {
        "nee_pred": torch.full((8, 1), 2.0),
        "bnee": None,
        "k": torch.full((8, 2), 3.0),           # error 3 on both E0 and rb
        "temp_derivative": None, "drift": None,
        "noise": None, "noise_mu": None, "noise_logvar": None,
        "latent": None, "latent_mu": None, "latent_logvar": None,
    }
    weights = {"mse_nee": 1.0, "mse_E0": 1.0, "mse_rb": 1.0}

    raw = compute_losses(batch, outputs, weights)
    scaled = compute_losses(batch, outputs, weights, target_scales={"mse_E0": 3.0})

    # mse_E0: raw 9.0 -> 9/3**2 = 1.0
    assert raw["mse_E0"].item() == pytest.approx(9.0)
    assert scaled["mse_E0"].item() == pytest.approx(1.0)
    # mse_rb (not listed) and mse_nee (not listed) are unchanged
    assert scaled["mse_rb"].item() == pytest.approx(raw["mse_rb"].item())
    assert scaled["mse_nee"].item() == pytest.approx(raw["mse_nee"].item())


def test_compute_losses_kl_for_vae():
    """kl_latent term computes only when latent_mu/logvar are present."""
    batch = {"NEE": torch.randn(8), "bNEE": torch.randn(8), "k": torch.randn(8, 2),
             "dT": torch.randn(8), "dNEE": torch.randn(8), "T": torch.randn(8)}
    outputs = {
        "nee_pred": torch.randn(8, 1),
        "latent_mu": torch.zeros(8, 32),
        "latent_logvar": torch.zeros(8, 32),    # var=1, KL should be 0
        "bnee": None, "k": None, "temp_derivative": None, "drift": None,
        "noise": None, "noise_mu": None, "noise_logvar": None,
        "latent": None, "nee_raw": None,
    }
    losses = compute_losses(batch, outputs, {"kl_latent": 1.0})
    # KL(N(0, I) || N(0, I)) = 0
    assert "kl_latent" in losses
    assert abs(losses["kl_latent"].item()) < 1e-5


def test_compute_losses_requires_mmd_fn_when_mmd_enabled():
    batch = {"NEE": torch.randn(8), "bNEE": torch.randn(8), "k": torch.randn(8, 2),
             "dT": torch.randn(8), "dNEE": torch.randn(8), "T": torch.randn(8)}
    outputs = {"nee_pred": torch.randn(8, 1), "bnee": None, "k": None,
               "temp_derivative": None, "drift": None, "noise": None,
               "noise_mu": None, "noise_logvar": None, "latent": None,
               "latent_mu": None, "latent_logvar": None, "nee_raw": None}
    with pytest.raises(ValueError, match="mmd_loss_fn is required"):
        compute_losses(batch, outputs, {"mmd_nee": 1.0})


def test_compute_losses_noise_prior_required():
    batch = {"NEE": torch.randn(8), "bNEE": torch.randn(8), "k": torch.randn(8, 2),
             "dT": torch.randn(8), "dNEE": torch.randn(8), "T": torch.randn(8)}
    outputs = {"nee_pred": torch.randn(8, 1), "noise": torch.randn(8, 1),
               "bnee": None, "k": None, "temp_derivative": None, "drift": None,
               "noise_mu": None, "noise_logvar": None, "latent": None,
               "latent_mu": None, "latent_logvar": None, "nee_raw": None}
    with pytest.raises(ValueError, match="noise_prior_fn is required"):
        compute_losses(batch, outputs, {"mmd_noise": 1.0}, mmd_loss_fn=MMDLoss())


def test_make_gaussian_noise_prior_shape_and_dtype():
    fn = make_gaussian_noise_prior(mean=-0.2, std=1.5)
    noise = torch.randn(32, 1)
    prior = fn(noise)
    assert prior.shape == noise.shape
    assert prior.dtype == noise.dtype


def test_make_empirical_noise_prior_centers_and_samples_from_pool():
    # Pool with a deliberately non-zero mean, like NEE - NEE_phy (~ -0.24)
    residuals = np.array([-3.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)  # mean 0.4
    fn = make_empirical_noise_prior(residuals, center=True)
    noise = torch.randn(4096, 1)
    prior = fn(noise)
    assert prior.shape == noise.shape
    assert prior.dtype == noise.dtype
    # Centered target -> ~zero mean over many draws
    assert abs(float(prior.mean())) < 0.1
    # Every drawn value is a member of the centered pool (shape preserved, not Gaussianized)
    centered_pool = (residuals - residuals.mean()).astype(np.float64)
    drawn = prior.flatten().numpy().astype(np.float64)
    nearest = np.min(np.abs(drawn[:, None] - centered_pool[None, :]), axis=1)
    assert np.all(nearest < 1e-4)


def test_make_empirical_noise_prior_uncentered_keeps_mean():
    residuals = np.array([-3.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)  # mean 0.4
    fn = make_empirical_noise_prior(residuals, center=False)
    prior = fn(torch.randn(8192, 1))
    assert abs(float(prior.mean()) - 0.4) < 0.15  # un-centered -> retains pool mean


def test_make_empirical_noise_prior_rejects_empty_pool():
    with pytest.raises(ValueError, match="non-empty"):
        make_empirical_noise_prior(np.array([], dtype=np.float32))
