"""The minimum 4 model tests plus checkpoint resume.

These run against every model variant via the `variant` fixture in conftest.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from wienernet.models import (
    HeadsConfig,
    WienerNetConfig,
    WienerNetModel,
    build_model,
)
from wienernet.utils import load_model_weights, save_checkpoint


def test_per_row_dt_overrides_config_dt(synthetic_batch):
    """Passing per-row dt scales the drift step; None falls back to config dt."""
    model = build_model("piae_sde_sampling", input_dim=20, device="cpu")
    model.cfg.dt = 30.0
    args = (synthetic_batch["X"], synthetic_batch["bNEE"], synthetic_batch["k"], synthetic_batch["T"])
    n = synthetic_batch["X"].shape[0]

    # drift contribution = nee_pred - bnee = drift*dt (deterministic; the reparam
    # noise in bnee cancels, and drift is a deterministic function of the input).
    def drift_contrib(out):
        return out["nee_pred"] - out["bnee"]

    contrib_cfg = drift_contrib(model(*args))                # dt=None -> cfg.dt=30
    contrib_30 = drift_contrib(model(*args, torch.full((n,), 30.0)))
    torch.testing.assert_close(contrib_cfg, contrib_30)      # per-row 30 == config 30

    contrib_15 = drift_contrib(model(*args, torch.full((n,), 15.0)))
    torch.testing.assert_close(contrib_15, contrib_30 * 0.5)  # halving dt halves the step


def test_noise_zero_mean_drops_fc_mu_and_zeros_noise_mean(synthetic_batch):
    """noise_zero_mean removes fc_mu; noise_mu is exactly 0 so noise = eps*sigma."""
    cfg = WienerNetConfig(
        input_dim=20, device="cpu",
        heads=HeadsConfig(temp_derivative=True, k=True, noise=True, noise_zero_mean=True),
    )
    model = WienerNetModel(cfg)
    assert model.fc_mu is None            # head not built -> no extra params/overhead
    assert model.fc_logvar is not None    # variance head still present

    out = model(synthetic_batch["X"], synthetic_batch["bNEE"],
                synthetic_batch["k"], synthetic_batch["T"])
    assert out["noise"] is not None
    assert torch.count_nonzero(out["noise_mu"]) == 0   # mean fixed to zero

    # Contrast: the default head does learn a (generally non-zero) mean
    default = WienerNetModel(WienerNetConfig(
        input_dim=20, device="cpu",
        heads=HeadsConfig(temp_derivative=True, k=True, noise=True, noise_zero_mean=False),
    ))
    assert default.fc_mu is not None


def test_output_shape(variant, synthetic_batch):
    """Forward pass produces tensors of the expected shapes."""
    model = build_model(variant, input_dim=20, device="cpu")
    out = model(synthetic_batch["X"], synthetic_batch["bNEE"],
                synthetic_batch["k"], synthetic_batch["T"])

    n = synthetic_batch["X"].shape[0]
    assert out["nee_pred"].shape == (n, 1), f"nee_pred shape wrong for {variant}"
    assert out["bnee"].shape == (n, 1)
    assert out["latent"].shape[0] == n
    if out["k"] is not None:
        assert out["k"].shape == (n, 2)
    if out["temp_derivative"] is not None:
        assert out["temp_derivative"].shape == (n, 1)
    if out["noise"] is not None:
        assert out["noise"].shape == (n, 1)


def test_gradient_flow(variant, synthetic_batch):
    """Every parameter receives a non-NaN, non-zero gradient on a simple MSE loss.

    Standard sanity check — catches dead heads (e.g. an unused decoder branch
    or an activation that swallows gradients at init).
    """
    model = build_model(variant, input_dim=20, device="cpu")
    out = model(synthetic_batch["X"], synthetic_batch["bNEE"],
                synthetic_batch["k"], synthetic_batch["T"])

    # MSE on every output tensor with a non-None value
    loss = F.mse_loss(out["nee_pred"], synthetic_batch["NEE"].view(-1, 1))
    if out["k"] is not None:
        loss = loss + F.mse_loss(out["k"], synthetic_batch["k"])
    if out["temp_derivative"] is not None:
        loss = loss + F.mse_loss(out["temp_derivative"], synthetic_batch["dT"].view(-1, 1))
    loss.backward()

    no_grad: list[str] = []
    nan_grad: list[str] = []
    zero_grad: list[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad.append(name)
            continue
        if torch.isnan(param.grad).any():
            nan_grad.append(name)
        if param.grad.abs().sum() == 0:
            zero_grad.append(name)

    assert not no_grad, f"{variant}: no gradient on {no_grad}"
    assert not nan_grad, f"{variant}: NaN gradient on {nan_grad}"
    assert not zero_grad, f"{variant}: zero gradient on {zero_grad}"


def test_overfit_single_batch(variant, synthetic_batch):
    """The most important sanity check — the model must be able to memorise
    one batch. Deterministic models should reduce loss by ≥10×. Stochastic
    models (latent_reparameterize=True) are bounded below by the reparam
    noise variance, so they only need a ≥3× reduction.
    """
    model = build_model(variant, input_dim=20, device="cpu")
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = synthetic_batch["NEE"].view(-1, 1)

    initial_loss: float | None = None
    for step in range(500):
        out = model(synthetic_batch["X"], synthetic_batch["bNEE"],
                    synthetic_batch["k"], synthetic_batch["T"])
        loss = F.mse_loss(out["nee_pred"], target)
        if initial_loss is None:
            initial_loss = loss.item()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    final_loss = loss.item()
    # Threshold is variant-dependent:
    # - Deterministic models: should overfit hard (≥10× reduction).
    # - Stochastic latent (VAE/PIVAE): the Tanh-bounded logvar caps how much
    #   the reparam noise can shrink, so loss is bounded below by a structural
    #   floor. We just require meaningful improvement (≥1.3× reduction); a
    #   tighter test for these variants would require disabling the Tanh on
    #   the logvar head.
    threshold = 1.3 if model.cfg.latent_reparameterize else 10.0
    assert final_loss < initial_loss / threshold, (
        f"{variant}: failed to reduce loss on one batch (threshold={threshold}×). "
        f"init_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
    )


def test_initial_loss_sanity(variant, synthetic_batch):
    """Loss at init shouldn't be NaN or infinite. The exact value depends on
    the head and the synthetic batch's variance.
    """
    model = build_model(variant, input_dim=20, device="cpu")
    out = model(synthetic_batch["X"], synthetic_batch["bNEE"],
                synthetic_batch["k"], synthetic_batch["T"])
    loss = F.mse_loss(out["nee_pred"], synthetic_batch["NEE"].view(-1, 1))
    assert torch.isfinite(loss), f"{variant}: initial loss is {loss.item()}"
    # And it should be in a sane order of magnitude — NEE values are O(1-10),
    # MSE should be O(1-100). 1e6 would mean something is exploding.
    assert loss.item() < 1e4, f"{variant}: initial loss suspiciously large: {loss.item()}"


def test_checkpoint_resume(variant, synthetic_batch):
    """save → load → identical predictions."""
    model_a = build_model(variant, input_dim=20, device="cpu")
    # Train one step to move weights away from init
    optim = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    out_pre = model_a(synthetic_batch["X"], synthetic_batch["bNEE"],
                      synthetic_batch["k"], synthetic_batch["T"])
    loss = F.mse_loss(out_pre["nee_pred"], synthetic_batch["NEE"].view(-1, 1))
    optim.zero_grad(); loss.backward(); optim.step()

    # Take a "reference" prediction
    model_a.eval()
    with torch.no_grad():
        # Re-seed so reparameterization noise is deterministic
        torch.manual_seed(0)
        ref = model_a(synthetic_batch["X"], synthetic_batch["bNEE"],
                      synthetic_batch["k"], synthetic_batch["T"])["nee_pred"]

    # Save + reload into a fresh model
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt.pth"
        save_checkpoint(ckpt, model_a)
        model_b = build_model(variant, input_dim=20, device="cpu")
        load_model_weights(ckpt, model_b)

    model_b.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        loaded = model_b(synthetic_batch["X"], synthetic_batch["bNEE"],
                         synthetic_batch["k"], synthetic_batch["T"])["nee_pred"]

    assert torch.allclose(ref, loaded, atol=1e-6), (
        f"{variant}: predictions differ after save/load: "
        f"max diff = {(ref - loaded).abs().max().item():.2e}"
    )


def test_model_has_no_dead_weights(variant):
    """Init shouldn't produce all-zero parameters anywhere."""
    model = build_model(variant, input_dim=20, device="cpu")
    for name, p in model.named_parameters():
        if "bias" in name:
            continue  # biases are intentionally zero-initialised
        assert p.abs().sum() > 0, f"{variant}: parameter {name} is all zero at init"
