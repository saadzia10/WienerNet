"""Tests for the increment-based Euler–Maruyama SDE model (IncrementSDEModel).

Covers the three variants (A residual / B non-residual / C reg), the
nee_pred = bNEE + dNEE integration, dt / sqrt(dt) scaling, zero-mean noise for A,
the residual_l2 loss term, and the standard overfit / gradient / resume checks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wienernet.losses.composite import compute_losses
from wienernet.models import (
    INCREMENT_VARIANTS,
    IncrementSDEConfig,
    IncrementSDEModel,
    build_increment_model,
    is_increment_variant,
)
from wienernet.utils import load_model_weights, save_checkpoint

FEATURE_DIM = 17  # exogenous X width (synthetic_batch fixture)


@pytest.fixture
def inc_batch(synthetic_batch):
    """synthetic_batch plus a per-row dt column (minutes)."""
    b = dict(synthetic_batch)
    b["dt"] = torch.full((b["X"].shape[0],), 30.0)
    return b


@pytest.fixture(params=list(INCREMENT_VARIANTS))
def inc_variant(request):
    return request.param


def _forward(model, batch):
    return model(batch["X"], batch["bNEE"], batch["k"], batch["T"], batch.get("dt"), batch.get("dT"))


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------

def test_increment_variants_recognised():
    for v in INCREMENT_VARIANTS:
        assert is_increment_variant(v)
    assert not is_increment_variant("piae_sde_sampling")


# ---------------------------------------------------------------------------
# Shapes + the core integration identity
# ---------------------------------------------------------------------------

def test_output_shapes_and_integration(inc_variant, inc_batch):
    model = build_increment_model(inc_variant, input_dim=FEATURE_DIM, device="cpu")
    out = _forward(model, inc_batch)
    n = inc_batch["X"].shape[0]

    assert out["nee_pred"].shape == (n, 1)
    assert out["dnee_pred"].shape == (n, 1)
    assert out["k"].shape == (n, 2)
    assert out["temp_derivative"].shape == (n, 1)
    assert out["bnee"] is None  # no level reconstruction

    # nee_pred = observed boundary + integrated increment (exactly)
    torch.testing.assert_close(out["nee_pred"], inc_batch["bNEE"].view(-1, 1) + out["dnee_pred"])


def test_variant_specific_heads(inc_batch):
    a = build_increment_model("piae_increment_residual", input_dim=FEATURE_DIM, device="cpu")
    b = build_increment_model("piae_increment", input_dim=FEATURE_DIM, device="cpu")
    c = build_increment_model("piae_reg_increment", input_dim=FEATURE_DIM, device="cpu")

    out_a, out_b, out_c = _forward(a, inc_batch), _forward(b, inc_batch), _forward(c, inc_batch)

    # A: residual on, zero-mean aleatoric noise
    assert out_a["residual"] is not None
    assert out_a["noise"] is not None
    assert a.noise_mu_decoder is None
    assert torch.count_nonzero(out_a["noise_mu"]) == 0

    # B: no residual, learned-mean noise
    assert out_b["residual"] is None
    assert out_b["noise"] is not None
    assert b.noise_mu_decoder is not None

    # C: deterministic — no residual, no noise
    assert out_c["residual"] is None
    assert out_c["noise"] is None
    assert out_c["dnee_pred"].shape == out_c["nee_pred"].shape


# ---------------------------------------------------------------------------
# Euler–Maruyama scaling: drift ~ dt, diffusion ~ sqrt(dt)
# ---------------------------------------------------------------------------

def test_drift_scales_linearly_with_dt(inc_batch):
    """Deterministic C: dNEE = f_phys * dt, so it scales linearly with dt."""
    model = build_increment_model("piae_reg_increment", input_dim=FEATURE_DIM, device="cpu")
    X, bNEE, k, T = (inc_batch[key] for key in ("X", "bNEE", "k", "T"))
    n = X.shape[0]

    d30 = model(X, bNEE, k, T, torch.full((n,), 30.0))["dnee_pred"]
    d15 = model(X, bNEE, k, T, torch.full((n,), 15.0))["dnee_pred"]
    torch.testing.assert_close(d15, d30 * 0.5)
    # Falls back to the config dt (=30) when per-row dt is None
    d_cfg = model(X, bNEE, k, T, None)["dnee_pred"]
    torch.testing.assert_close(d_cfg, d30)


def test_diffusion_scales_with_sqrt_dt():
    """Physics + residual off, noise on -> dNEE = eps*sigma*sqrt(dt)."""
    cfg = IncrementSDEConfig(
        input_dim=FEATURE_DIM, device="cpu",
        predict_k=False, predict_temp_derivative=False, residual=False,
        noise=True, noise_zero_mean=True,
    )
    model = IncrementSDEModel(cfg).initialize()
    g = torch.Generator().manual_seed(7)
    X = torch.randn(8, FEATURE_DIM, generator=g)
    bNEE = torch.randn(8, generator=g)
    k = torch.abs(torch.randn(8, 2, generator=g)) + 1
    T = torch.randn(8, generator=g)

    torch.manual_seed(0)
    d30 = model(X, bNEE, k, T, torch.full((8,), 30.0))["dnee_pred"]
    torch.manual_seed(0)
    d120 = model(X, bNEE, k, T, torch.full((8,), 120.0))["dnee_pred"]
    # 4x dt -> sqrt(4)=2x the diffusion increment (same eps via the re-seed)
    torch.testing.assert_close(d120, d30 * 2.0, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Loss integration (residual_l2 + the increment loss set)
# ---------------------------------------------------------------------------

def test_residual_l2_only_active_with_residual(inc_batch):
    weights = {"mse_nee": 1.0, "residual_l2": 2.0}

    a = build_increment_model("piae_increment_residual", input_dim=FEATURE_DIM, device="cpu")
    losses_a = compute_losses(inc_batch, _forward(a, inc_batch), weights)
    assert "residual_l2" in losses_a
    assert losses_a["residual_l2"].item() >= 0

    c = build_increment_model("piae_reg_increment", input_dim=FEATURE_DIM, device="cpu")
    losses_c = compute_losses(inc_batch, _forward(c, inc_batch), weights)
    assert "residual_l2" not in losses_c  # no residual head -> term skipped


def test_increment_loss_set_is_finite(inc_variant, inc_batch):
    from wienernet.losses import MMDLoss, make_gaussian_noise_prior

    weights = {
        "mse_nee": 1.0, "mse_E0": 1.0, "mse_rb": 1.0,
        "mse_temp_derivative": 1.0, "mse_drift": 1.0,
        "residual_l2": 1.0, "mmd_noise": 1.0,
    }
    model = build_increment_model(inc_variant, input_dim=FEATURE_DIM, device="cpu")
    losses = compute_losses(
        inc_batch, _forward(model, inc_batch), weights,
        mmd_loss_fn=MMDLoss(), noise_prior_fn=make_gaussian_noise_prior(0.0, 1.0),
    )
    total = sum(losses.values())
    assert torch.isfinite(total)
    assert "mse_nee" in losses  # always present


# ---------------------------------------------------------------------------
# Standard model sanity checks
# ---------------------------------------------------------------------------

def test_gradient_flow(inc_variant, inc_batch):
    model = build_increment_model(inc_variant, input_dim=FEATURE_DIM, device="cpu")
    out = _forward(model, inc_batch)
    loss = F.mse_loss(out["nee_pred"], inc_batch["NEE"].view(-1, 1))
    loss = loss + F.mse_loss(out["k"], inc_batch["k"])
    loss = loss + F.mse_loss(out["temp_derivative"], inc_batch["dT"].view(-1, 1))
    loss.backward()

    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    nan = [n for n, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    assert not dead, f"{inc_variant}: no/zero gradient on {dead}"
    assert not nan, f"{inc_variant}: NaN gradient on {nan}"


def test_overfit_single_batch(inc_variant, inc_batch):
    model = build_increment_model(inc_variant, input_dim=FEATURE_DIM, device="cpu")
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = inc_batch["NEE"].view(-1, 1)

    initial = None
    for _ in range(500):
        out = _forward(model, inc_batch)
        loss = F.mse_loss(out["nee_pred"], target)
        if initial is None:
            initial = loss.item()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    # The residual/mu heads (and dtdt) give the deterministic part full freedom,
    # and softplus lets sigma shrink to the floor, so every variant should
    # memorise one batch hard.
    assert loss.item() < initial / 10.0, (
        f"{inc_variant}: init={initial:.4f} final={loss.item():.4f}"
    )


def test_checkpoint_resume(inc_variant, inc_batch):
    model_a = build_increment_model(inc_variant, input_dim=FEATURE_DIM, device="cpu")
    optim = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    loss = F.mse_loss(_forward(model_a, inc_batch)["nee_pred"], inc_batch["NEE"].view(-1, 1))
    optim.zero_grad(); loss.backward(); optim.step()

    model_a.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        ref = _forward(model_a, inc_batch)["nee_pred"]

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt.pth"
        save_checkpoint(ckpt, model_a)
        model_b = build_increment_model(inc_variant, input_dim=FEATURE_DIM, device="cpu")
        load_model_weights(ckpt, model_b)

    model_b.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        loaded = _forward(model_b, inc_batch)["nee_pred"]
    assert torch.allclose(ref, loaded, atol=1e-6)


def test_ald_noise_model(inc_batch):
    """ALD (F.1): log_kappa present, finite NLL, drift stays the mean."""
    from wienernet.losses.composite import compute_losses
    cfg = IncrementSDEConfig(input_dim=FEATURE_DIM, residual=True, noise=True, noise_zero_mean=True,
                             physics_k_source="ground_truth", predict_k=False, noise_asymmetry=True)
    m = IncrementSDEModel(cfg).initialize()
    out = _forward(m, inc_batch)
    assert out["log_kappa"] is not None and out["mix_logits"] is None
    torch.testing.assert_close(out["nee_mean"], inc_batch["bNEE"].view(-1, 1) + out["drift"] * 30.0)
    losses = compute_losses(inc_batch, out, {}, likelihood={"variant": "ald", "weight": 1.0})
    assert torch.isfinite(losses["nll"])
    losses["nll"].backward()
    assert m.log_kappa.grad is not None


def test_mixture_noise_model(inc_batch):
    """Zero-mean mixture noise (F.2): mix params present, mixture NLL routed, drift = mean."""
    from wienernet.losses.composite import compute_losses
    cfg = IncrementSDEConfig(input_dim=FEATURE_DIM, residual=True, noise=True, noise_zero_mean=True,
                             physics_k_source="ground_truth", predict_k=False, noise_mixture_components=3)
    m = IncrementSDEModel(cfg).initialize()
    out = _forward(m, inc_batch)
    n = inc_batch["X"].shape[0]
    for key in ("mix_means", "mix_log_scales", "mix_logits"):
        assert out[key] is not None and out[key].shape == (n, 3), key
    # zero-mean mixture -> the deterministic mean is still the drift
    torch.testing.assert_close(out["nee_mean"], inc_batch["bNEE"].view(-1, 1) + out["drift"] * 30.0)
    # mixture mean (weighted component locations) == nee_mean (centering holds)
    w = torch.softmax(out["mix_logits"], dim=1)
    torch.testing.assert_close((w * out["mix_means"]).sum(1, keepdim=True), out["nee_mean"], atol=1e-5, rtol=1e-4)
    losses = compute_losses(inc_batch, out, {}, likelihood={"variant": "gaussian", "weight": 1.0})
    assert torch.isfinite(losses["nll"])
    losses["nll"].backward()
    assert all(p.grad is not None for p in m.mix_offset_decoder.parameters())


def test_physics_anchored_scale(inc_batch):
    """Physics-anchored scale (C.1): the diffusion std grows with Reco(T,E0,rb) and
    the drift stays the conditional mean (the anchor only shapes the noise)."""
    cfg = IncrementSDEConfig(input_dim=FEATURE_DIM, residual=False, noise=True, noise_zero_mean=True,
                             physics_k_source="ground_truth", predict_k=False,
                             noise_asymmetry=True, noise_physics_scale=True)
    m = IncrementSDEModel(cfg).initialize()
    assert m.scale_anchor_a is not None and m.scale_anchor_b is not None
    with torch.no_grad():
        m.scale_anchor_b.fill_(1.0)      # ensure a positive, monotone flux coupling
    # physical E0/rb so Reco stays below the OOD clamp and varies monotonically with T
    k_phys = torch.tensor([[100.0, 2.0]]).expand_as(inc_batch["k"]).contiguous()
    lo = dict(inc_batch); lo["T"] = torch.zeros_like(inc_batch["T"]); lo["k"] = k_phys
    hi = dict(inc_batch); hi["T"] = torch.full_like(inc_batch["T"], 20.0); hi["k"] = k_phys
    s_lo, out_hi = _forward(m, lo)["sigma"], _forward(m, hi)
    # higher T -> higher Reco -> larger diffusion scale on every row
    assert (out_hi["sigma"] > s_lo).all()
    # the noise scale never touches the mean: nee_mean == bNEE + drift*dt
    torch.testing.assert_close(out_hi["nee_mean"], hi["bNEE"].view(-1, 1) + out_hi["drift"] * 30.0)


def test_drift_clamp_bounds_extreme(inc_batch):
    """drift_clamp soft-bounds an exploding drift while staying ~identity for real values."""
    m = IncrementSDEModel(IncrementSDEConfig(input_dim=FEATURE_DIM, residual=True, noise=False,
                          physics_k_source="ground_truth", predict_k=False, drift_clamp=1.0)).initialize()
    # force a huge residual -> drift; the clamp caps |drift| < drift_clamp
    with torch.no_grad():
        for p in m.residual_decoder.parameters():
            p.mul_(0).add_(50.0)
    out = _forward(m, inc_batch)
    assert out["drift"].abs().max() <= 1.0 + 1e-4


def test_no_dead_weights(inc_variant):
    model = build_increment_model(inc_variant, input_dim=FEATURE_DIM, device="cpu")
    for name, p in model.named_parameters():
        if "bias" in name:
            continue
        assert p.abs().sum() > 0, f"{inc_variant}: {name} all zero at init"
