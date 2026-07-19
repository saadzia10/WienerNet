"""Tests for the process-comparison baselines (the ablation ladder).

Currently covers Baseline 1 — AnalyticalSDEModel (physics drift + constant
diffusion). Verifies: registry/dispatch, the output-dict contract shared with the
increment model, pure-physics drift, constant state-independent diffusion, the
Euler sqrt(dt) scaling, that calibrate() recovers the dt-normalised increment-
misfit spread AND sits at the Gaussian-NLL MLE, likelihood-loss integration, the
Student-t option, and checkpoint round-trip.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from wienernet.losses.composite import compute_losses
from wienernet.losses.likelihood import mixture_nll
from wienernet.models import (
    BASELINE_VARIANTS,
    AnalyticalSDEConfig,
    AnalyticalSDEModel,
    HeteroscedasticMLPConfig,
    HeteroscedasticMLPModel,
    NeuralSDEConfig,
    NeuralSDEModel,
    build_baseline_model,
    is_baseline_variant,
    is_increment_variant,
)
from wienernet.physics.lloyd_taylor import sde_drift
from wienernet.utils import load_model_weights, save_checkpoint

B2_FEATURE_DIM = 17  # synthetic_batch X width


@pytest.fixture
def base_batch(synthetic_batch):
    """synthetic_batch plus a per-row dt column (minutes)."""
    b = dict(synthetic_batch)
    b["dt"] = torch.full((b["X"].shape[0],), 30.0)
    return b


def _forward(model, batch):
    return model(batch["X"], batch["bNEE"], batch["k"], batch["T"], batch.get("dt"), batch.get("dT"))


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------

def test_baseline_variant_recognised():
    assert "analytical_sde" in BASELINE_VARIANTS
    for v in BASELINE_VARIANTS:
        assert is_baseline_variant(v)
    assert not is_baseline_variant("piae_sde_sampling")
    assert not is_baseline_variant("piae_increment")
    # baselines are a distinct family from the increment variants
    assert not is_increment_variant("analytical_sde")


# ---------------------------------------------------------------------------
# Output contract + the core integration identity
# ---------------------------------------------------------------------------

def test_output_contract_and_integration(base_batch):
    model = build_baseline_model("analytical_sde", device="cpu")
    out = _forward(model, base_batch)
    n = base_batch["X"].shape[0]

    for key in ("nee_pred", "nee_mean", "nee_log_std", "dnee_pred", "noise", "sigma",
                "drift", "latent", "f_phys"):
        assert out[key] is not None and out[key].shape == (n, 1), key
    # No encoder / no reconstruction / no learned heads
    assert out["bnee"] is None
    assert out["k"] is None
    assert out["residual"] is None
    assert out["temp_derivative"] is None

    # nee_pred = observed boundary + integrated increment (exactly)
    torch.testing.assert_close(out["nee_pred"], base_batch["bNEE"].view(-1, 1) + out["dnee_pred"])
    # nee_mean is the deterministic (noise-free) prediction
    torch.testing.assert_close(out["nee_mean"], base_batch["bNEE"].view(-1, 1) + out["drift"] * 30.0)


def test_drift_is_pure_physics(base_batch):
    """The drift is exactly the ground-truth Lloyd-Taylor sde_drift — no learning."""
    model = build_baseline_model("analytical_sde", device="cpu")
    out = _forward(model, base_batch)
    k, T, dT = base_batch["k"], base_batch["T"], base_batch["dT"]
    expected = sde_drift(T.view(-1, 1), k[:, 0:1], k[:, 1:2], dT.view(-1, 1),
                         tref=model.cfg.tref, t0=model.cfg.t0)
    torch.testing.assert_close(out["drift"], expected)


def test_constant_state_independent_diffusion(base_batch):
    """sigma is one global constant — identical across every row, and the log-std
    is log_sigma + 0.5 log dt."""
    model = build_baseline_model("analytical_sde", device="cpu")
    out = _forward(model, base_batch)
    sigma = out["sigma"]
    assert torch.allclose(sigma, sigma[0].expand_as(sigma))         # constant
    expected_log_std = model.log_sigma + 0.5 * math.log(30.0)
    torch.testing.assert_close(out["nee_log_std"], expected_log_std.expand_as(out["nee_log_std"]))


def test_diffusion_scales_with_sqrt_dt(base_batch):
    """4x dt -> 2x diffusion increment (same eps via re-seed); drift ~ dt."""
    model = build_baseline_model("analytical_sde", device="cpu")
    X, bNEE, k, T, dT = (base_batch[key] for key in ("X", "bNEE", "k", "T", "dT"))
    n = X.shape[0]

    torch.manual_seed(0)
    d30 = model(X, bNEE, k, T, torch.full((n,), 30.0), dT)
    torch.manual_seed(0)
    d120 = model(X, bNEE, k, T, torch.full((n,), 120.0), dT)
    # diffusion part = dnee - drift*dt ; scales with sqrt(dt)
    diff30 = d30["dnee_pred"] - d30["drift"] * 30.0
    diff120 = d120["dnee_pred"] - d120["drift"] * 120.0
    torch.testing.assert_close(diff120, diff30 * 2.0, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Calibration: the constant diffusion coefficient
# ---------------------------------------------------------------------------

def test_calibrate_recovers_dt_normalised_spread(base_batch):
    model = build_baseline_model("analytical_sde", device="cpu")
    model.calibrate([base_batch])                      # a one-batch iterable is a valid "loader"

    # expected sigma^2 = mean( ((NEE - bNEE) - f_phys*dt)^2 / dt )
    k, T, dT = base_batch["k"], base_batch["T"], base_batch["dT"]
    f_phys = sde_drift(T.view(-1, 1), k[:, 0:1], k[:, 1:2], dT.view(-1, 1),
                       tref=model.cfg.tref, t0=model.cfg.t0)
    resid = (base_batch["NEE"].view(-1, 1) - base_batch["bNEE"].view(-1, 1)) - f_phys * 30.0
    expected_sigma = math.sqrt(float((resid ** 2 / 30.0).mean()))
    torch.testing.assert_close(float(torch.exp(model.log_sigma)), expected_sigma, rtol=1e-5, atol=1e-6)


def test_calibration_is_gaussian_nll_mle(base_batch):
    """After calibration the Gaussian-NLL gradient wrt log_sigma is ~0 (MLE), so
    likelihood training leaves the textbook sigma untouched."""
    model = build_baseline_model("analytical_sde", device="cpu")
    model.calibrate([base_batch])
    out = _forward(model, base_batch)
    losses = compute_losses(
        base_batch, out, {}, likelihood={"variant": "gaussian", "weight": 1.0},
    )
    losses["nll"].backward()
    assert model.log_sigma.grad is not None
    assert abs(float(model.log_sigma.grad)) < 1e-4


def test_likelihood_loss_finite_and_trains_sigma(base_batch):
    model = build_baseline_model("analytical_sde", device="cpu")
    out = _forward(model, base_batch)
    losses = compute_losses(
        base_batch, out, {}, likelihood={"variant": "gaussian", "weight": 1.0},
    )
    assert "nll" in losses and torch.isfinite(losses["nll"])
    losses["nll"].backward()
    # the single free parameter receives gradient
    assert model.log_sigma.grad is not None and model.log_sigma.grad.abs() > 0


def test_student_t_option(base_batch):
    cfg = AnalyticalSDEConfig(noise_student_dof=True, device="cpu")
    model = AnalyticalSDEModel(cfg).initialize()
    out = _forward(model, base_batch)
    assert out["log_nu"] is not None
    losses = compute_losses(
        base_batch, out, {}, likelihood={"variant": "student_t", "weight": 1.0},
    )
    assert torch.isfinite(losses["nll"])
    losses["nll"].backward()
    assert model.log_sigma.grad is not None
    assert model.log_nu.grad is not None


# ---------------------------------------------------------------------------
# Per-site constant diffusion
# ---------------------------------------------------------------------------

def _persite_batch(synthetic_batch, sites):
    b = dict(synthetic_batch)
    n = b["X"].shape[0]
    b["dt"] = torch.full((n,), 30.0)
    b["site_id"] = [sites[i % len(sites)] for i in range(n)]
    return b


def test_per_site_requires_layout():
    with pytest.raises(ValueError):
        AnalyticalSDEModel(AnalyticalSDEConfig(per_site=True, site_names=(), device="cpu"))


def test_per_site_sigma_selected_and_calibrated(synthetic_batch):
    sites = ("great_fen", "rosedene")
    batch = _persite_batch(synthetic_batch, sites)
    cfg = AnalyticalSDEConfig(per_site=True, site_names=sites, device="cpu")
    model = AnalyticalSDEModel(cfg).initialize().calibrate([batch])

    # Per-site sigma matches the within-site dt-normalised MLE for each site.
    k, T, dT = batch["k"], batch["T"], batch["dT"]
    f_phys = sde_drift(T.view(-1, 1), k[:, 0:1], k[:, 1:2], dT.view(-1, 1),
                       tref=cfg.tref, t0=cfg.t0)
    resid = (batch["NEE"].view(-1, 1) - batch["bNEE"].view(-1, 1)) - f_phys * 30.0
    per_dt = (resid ** 2 / 30.0).view(-1)
    site_arr = batch["site_id"]
    for i, s in enumerate(sites):
        mask = torch.tensor([nm == s for nm in site_arr])
        expected = math.sqrt(float(per_dt[mask].mean()))
        got = float(torch.exp(model.log_sigma_site[i]))
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    # forward picks the right per-row sigma from the site argument.
    out = _forward_site(model, batch)
    for i, s in enumerate(sites):
        rows = torch.tensor([nm == s for nm in site_arr])
        sig = out["sigma"][rows]
        torch.testing.assert_close(sig, torch.exp(model.log_sigma_site[i]).expand_as(sig))


def test_per_site_unknown_site_falls_back_to_global(synthetic_batch):
    sites = ("great_fen",)
    batch = _persite_batch(synthetic_batch, ("great_fen",))
    model = AnalyticalSDEModel(
        AnalyticalSDEConfig(per_site=True, site_names=sites, device="cpu")).initialize()
    n = batch["X"].shape[0]
    unknown = ["nowhere"] * n
    out = model(batch["X"], batch["bNEE"], batch["k"], batch["T"],
                batch["dt"], batch["dT"], site=unknown)
    torch.testing.assert_close(out["sigma"], torch.exp(model.log_sigma).expand(n, 1))


def _forward_site(model, batch):
    return model(batch["X"], batch["bNEE"], batch["k"], batch["T"],
                 batch.get("dt"), batch.get("dT"), site=batch.get("site_id"))


def test_global_model_ignores_site(base_batch):
    """The uniform `site` kwarg is inert for the global model."""
    model = build_baseline_model("analytical_sde", device="cpu")
    n = base_batch["X"].shape[0]
    torch.manual_seed(0)
    a = model(base_batch["X"], base_batch["bNEE"], base_batch["k"], base_batch["T"],
              base_batch["dt"], base_batch["dT"], site=["rosedene"] * n)["nee_mean"]
    torch.manual_seed(0)
    b = _forward(model, base_batch)["nee_mean"]
    torch.testing.assert_close(a, b)


# ===========================================================================
# Baseline 2 — HeteroscedasticMLPModel (mean-variance / MDN)
# ===========================================================================

def _b2(mixture=1, student=False):
    cfg = HeteroscedasticMLPConfig(
        input_dim=B2_FEATURE_DIM, mixture_components=mixture,
        noise_student_dof=student, device="cpu",
    )
    return HeteroscedasticMLPModel(cfg).initialize()


def test_b2_variant_recognised():
    for v in ("hetero_mlp", "hetero_mdn"):
        assert v in BASELINE_VARIANTS and is_baseline_variant(v)
    m1 = build_baseline_model("hetero_mlp", input_dim=B2_FEATURE_DIM, device="cpu")
    m3 = build_baseline_model("hetero_mdn", input_dim=B2_FEATURE_DIM, device="cpu")
    assert isinstance(m1, HeteroscedasticMLPModel) and m1.cfg.mixture_components == 1
    assert isinstance(m3, HeteroscedasticMLPModel) and m3.cfg.mixture_components == 3


def test_b2_mean_variance_contract(base_batch):
    model = _b2(mixture=1)
    out = _forward(model, base_batch)
    n = base_batch["X"].shape[0]
    for key in ("nee_pred", "nee_mean", "nee_log_std", "dnee_pred", "noise", "sigma"):
        assert out[key] is not None and out[key].shape == (n, 1), key
    assert out["latent"] is not None and out["latent"].shape == (n, model.cfg.latent_dim)
    # no physics / no SDE structure
    assert out["drift"] is None and out["f_phys"] is None and out["residual"] is None
    assert out["mix_logits"] is None                     # single component -> not an MDN
    # increment anchored on the boundary; sigma = exp(log_std); NO dt term in log_std
    torch.testing.assert_close(out["nee_pred"], base_batch["bNEE"].view(-1, 1) + out["dnee_pred"])
    torch.testing.assert_close(out["sigma"], torch.exp(out["nee_log_std"]))


def test_b2_log_std_independent_of_dt(base_batch):
    """B2 has no Euler dt-scaling: the predictive scale does not depend on dt."""
    model = _b2(mixture=1)
    X, bNEE, k, T, dT = (base_batch[c] for c in ("X", "bNEE", "k", "T", "dT"))
    n = X.shape[0]
    s30 = model(X, bNEE, k, T, torch.full((n,), 30.0), dT)["nee_log_std"]
    s120 = model(X, bNEE, k, T, torch.full((n,), 120.0), dT)["nee_log_std"]
    torch.testing.assert_close(s30, s120)                # unchanged by dt


def test_b2_mean_variance_nll_trains(base_batch):
    model = _b2(mixture=1)
    out = _forward(model, base_batch)
    losses = compute_losses(base_batch, out, {}, likelihood={"variant": "gaussian", "weight": 1.0})
    assert "nll" in losses and torch.isfinite(losses["nll"])
    losses["nll"].backward()
    grads = [p.grad for n_, p in model.named_parameters()]
    assert all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)


def test_b2_mdn_emits_mixture_and_routes_mixture_nll(base_batch):
    model = _b2(mixture=3)
    out = _forward(model, base_batch)
    n = base_batch["X"].shape[0]
    for key in ("mix_means", "mix_log_scales", "mix_logits"):
        assert out[key] is not None and out[key].shape == (n, 3), key
    # moment-matched summary still present for point metrics + parametric report
    assert out["nee_mean"].shape == (n, 1) and out["nee_log_std"].shape == (n, 1)

    # compute_losses routes to the mixture NLL and matches the direct call
    losses = compute_losses(base_batch, out, {}, likelihood={"variant": "gaussian", "weight": 1.0})
    direct = mixture_nll(base_batch["NEE"].view(-1, 1), out["mix_means"],
                         out["mix_log_scales"], out["mix_logits"], variant="gaussian")
    torch.testing.assert_close(losses["nll"], direct)
    losses["nll"].backward()
    assert model.logit_decoder is not None
    assert all(p.grad is not None for p in model.logit_decoder.parameters())


def test_b2_mdn_moment_matching(base_batch):
    """nee_mean / nee_log_std are the mixture's analytic mean / std."""
    model = _b2(mixture=3)
    out = _forward(model, base_batch)
    w = torch.softmax(out["mix_logits"], dim=1)
    mu = out["mix_means"]; s = torch.exp(out["mix_log_scales"])
    mean = (w * mu).sum(1, keepdim=True)
    var = (w * (s ** 2 + mu ** 2)).sum(1, keepdim=True) - mean ** 2
    torch.testing.assert_close(out["nee_mean"], mean)
    torch.testing.assert_close(out["nee_log_std"], 0.5 * torch.log(var.clamp_min(model._VAR_FLOOR)))


def test_b2_student_t_mixture(base_batch):
    model = _b2(mixture=3, student=True)
    out = _forward(model, base_batch)
    assert out["log_nu"] is not None
    losses = compute_losses(base_batch, out, {}, likelihood={"variant": "student_t", "weight": 1.0})
    assert torch.isfinite(losses["nll"])
    losses["nll"].backward()
    assert model.log_nu.grad is not None


def test_b2_overfits_single_batch(base_batch):
    """The mean head has full freedom, so the NLL drops sharply on one batch."""
    model = _b2(mixture=1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial = None
    for _ in range(400):
        out = _forward(model, base_batch)
        loss = compute_losses(base_batch, out, {}, likelihood={"variant": "gaussian", "weight": 1.0})["nll"]
        if initial is None:
            initial = loss.item()
        optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
    assert loss.item() < initial - 0.5, f"init={initial:.3f} final={loss.item():.3f}"


def test_b2_checkpoint_resume(base_batch):
    model_a = _b2(mixture=3)
    model_a.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        ref = _forward(model_a, base_batch)["nee_mean"]     # deterministic (no sampling)
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "b2.pth"
        save_checkpoint(ckpt, model_a)
        model_b = _b2(mixture=3)
        load_model_weights(ckpt, model_b)
    model_b.eval()
    with torch.no_grad():
        loaded = _forward(model_b, base_batch)["nee_mean"]
    torch.testing.assert_close(ref, loaded, atol=1e-6, rtol=0)


# ===========================================================================
# Baseline 3 — NeuralSDEModel (free neural drift + diffusion)
# ===========================================================================

def _b3(student=False):
    cfg = NeuralSDEConfig(input_dim=B2_FEATURE_DIM, noise_student_dof=student, device="cpu")
    return NeuralSDEModel(cfg).initialize()


def test_b3_variant_recognised():
    assert "neural_sde" in BASELINE_VARIANTS and is_baseline_variant("neural_sde")
    m = build_baseline_model("neural_sde", input_dim=B2_FEATURE_DIM, device="cpu")
    assert isinstance(m, NeuralSDEModel)


def test_b3_contract_and_integration(base_batch):
    model = _b3()
    out = _forward(model, base_batch)
    n = base_batch["X"].shape[0]
    for key in ("nee_pred", "nee_mean", "nee_log_std", "dnee_pred", "noise", "sigma", "drift"):
        assert out[key] is not None and out[key].shape == (n, 1), key
    assert out["f_phys"] is None and out["residual"] is None and out["k"] is None
    # nee_pred = boundary + integrated increment; nee_mean = boundary + drift*dt
    torch.testing.assert_close(out["nee_pred"], base_batch["bNEE"].view(-1, 1) + out["dnee_pred"])
    torch.testing.assert_close(out["nee_mean"], base_batch["bNEE"].view(-1, 1) + out["drift"] * 30.0)
    # keeps the Euler sqrt(dt) structure: log_std = log(sigma) + 0.5 log dt
    torch.testing.assert_close(out["nee_log_std"], torch.log(out["sigma"]) + 0.5 * math.log(30.0))


def test_b3_euler_scaling(base_batch):
    """drift scales with dt; diffusion with sqrt(dt) (same eps via re-seed)."""
    model = _b3()
    X, bNEE, k, T, dT = (base_batch[c] for c in ("X", "bNEE", "k", "T", "dT"))
    n = X.shape[0]
    torch.manual_seed(0)
    d30 = model(X, bNEE, k, T, torch.full((n,), 30.0), dT)
    torch.manual_seed(0)
    d120 = model(X, bNEE, k, T, torch.full((n,), 120.0), dT)
    det30 = d30["drift"] * 30.0
    det120 = d120["drift"] * 120.0
    torch.testing.assert_close(det120, det30 * 4.0)                 # drift ~ dt
    diff30 = d30["dnee_pred"] - det30
    diff120 = d120["dnee_pred"] - det120
    torch.testing.assert_close(diff120, diff30 * 2.0, rtol=1e-4, atol=1e-6)   # diffusion ~ sqrt(dt)


def test_b3_nll_trains_drift_and_sigma(base_batch):
    model = _b3()
    out = _forward(model, base_batch)
    losses = compute_losses(base_batch, out, {}, likelihood={"variant": "gaussian", "weight": 1.0})
    assert "nll" in losses and torch.isfinite(losses["nll"])
    losses["nll"].backward()
    assert all(p.grad is not None for p in model.drift_decoder.parameters())
    assert all(p.grad is not None for p in model.sigma_decoder.parameters())


def test_b3_student_t(base_batch):
    model = _b3(student=True)
    out = _forward(model, base_batch)
    assert out["log_nu"] is not None
    losses = compute_losses(base_batch, out, {}, likelihood={"variant": "student_t", "weight": 1.0})
    assert torch.isfinite(losses["nll"])
    losses["nll"].backward()
    assert model.log_nu.grad is not None


def test_b3_overfits_single_batch(base_batch):
    model = _b3()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial = None
    for _ in range(400):
        out = _forward(model, base_batch)
        loss = compute_losses(base_batch, out, {}, likelihood={"variant": "gaussian", "weight": 1.0})["nll"]
        if initial is None:
            initial = loss.item()
        optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
    assert loss.item() < initial - 0.5, f"init={initial:.3f} final={loss.item():.3f}"


def test_b3_checkpoint_resume(base_batch):
    model_a = _b3()
    model_a.eval()
    with torch.no_grad():
        ref = _forward(model_a, base_batch)["nee_mean"]           # deterministic
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "b3.pth"
        save_checkpoint(ckpt, model_a)
        model_b = _b3()
        load_model_weights(ckpt, model_b)
    model_b.eval()
    with torch.no_grad():
        loaded = _forward(model_b, base_batch)["nee_mean"]
    torch.testing.assert_close(ref, loaded, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Checkpoint round-trip (B1)
# ---------------------------------------------------------------------------

def test_checkpoint_resume(base_batch):
    model_a = build_baseline_model("analytical_sde", device="cpu")
    model_a.calibrate([base_batch])
    model_a.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        ref = _forward(model_a, base_batch)["nee_pred"]

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt.pth"
        save_checkpoint(ckpt, model_a)
        model_b = build_baseline_model("analytical_sde", device="cpu")
        load_model_weights(ckpt, model_b)

    torch.testing.assert_close(model_a.log_sigma, model_b.log_sigma)
    model_b.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        loaded = _forward(model_b, base_batch)["nee_pred"]
    torch.testing.assert_close(ref, loaded, atol=1e-6, rtol=0)
