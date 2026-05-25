"""Tests for MMD loss and the composite loss assembly."""

from __future__ import annotations

import pytest
import torch

from wienernet.losses import MMDLoss, compute_losses, make_gaussian_noise_prior


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
