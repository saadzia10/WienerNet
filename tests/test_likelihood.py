"""Tests for the NLL likelihood losses + their model/loss wiring."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wienernet.losses import compute_losses
from wienernet.losses.likelihood import (
    beta_nll,
    compute_nll,
    gaussian_nll,
    student_t_nll,
)
from wienernet.models import HeadsConfig, build_increment_model, build_model
from wienernet.utils import load_model_weights, save_checkpoint


# ---------------------------------------------------------------------------
# Formula correctness vs torch.distributions
# ---------------------------------------------------------------------------

def test_gaussian_nll_matches_torch():
    torch.manual_seed(0)
    target = torch.randn(500)
    mean = torch.randn(500)
    log_std = torch.randn(500) * 0.3
    ours = gaussian_nll(target, mean, log_std)
    ref = -torch.distributions.Normal(mean, log_std.exp()).log_prob(target)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6)


def test_student_t_nll_matches_torch():
    torch.manual_seed(0)
    target = torch.randn(500)
    mean = torch.randn(500)
    log_std = torch.randn(500) * 0.3
    log_nu = torch.tensor(math.log(math.expm1(5.0)))       # softplus(log_nu) ~ 5
    from wienernet.losses.likelihood import student_t_dof
    nu = student_t_dof(log_nu)                              # = softplus + NU_FLOOR
    ours = student_t_nll(target, mean, log_std, log_nu)
    ref = -torch.distributions.StudentT(nu, mean, log_std.exp()).log_prob(target)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6)


def test_beta_nll_reduces_to_gaussian_at_beta0():
    torch.manual_seed(0)
    target, mean, log_std = torch.randn(200), torch.randn(200), torch.randn(200) * 0.2
    torch.testing.assert_close(
        beta_nll(target, mean, log_std, beta=0.0), gaussian_nll(target, mean, log_std)
    )


def test_compute_nll_dispatch_and_student_requires_dof():
    t, m, ls = torch.randn(64), torch.randn(64), torch.zeros(64)
    assert compute_nll(t, m, ls, variant="gaussian").ndim == 0        # scalar (mean-reduced)
    assert compute_nll(t, m, ls, variant="beta", beta=0.5).ndim == 0
    with pytest.raises(ValueError, match="student_t"):
        compute_nll(t, m, ls, variant="student_t")                    # no log_nu


# ---------------------------------------------------------------------------
# Recovery: shared params recover the conditional mean AND variance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["gaussian", "beta", "student_t"])
def test_nll_recovers_moments_with_shared_params(variant):
    torch.manual_seed(0)
    mu_true, s_true = 1.5, 2.0
    data = mu_true + s_true * torch.randn(20000)
    mean = torch.zeros((), requires_grad=True)
    log_std = torch.zeros((), requires_grad=True)
    log_nu = torch.nn.Parameter(torch.tensor(math.log(math.expm1(30.0))))  # large nu ~ Gaussian
    params = [mean, log_std] + ([log_nu] if variant == "student_t" else [])
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(1500):
        opt.zero_grad()
        compute_nll(data, mean.expand_as(data), log_std.expand_as(data),
                    variant=variant, beta=0.5, log_nu=log_nu).backward()
        opt.step()
    assert abs(mean.item() - mu_true) < 0.1, f"{variant}: mean {mean.item():.3f}"
    assert abs(log_std.exp().item() - s_true) < 0.2, f"{variant}: std {log_std.exp().item():.3f}"


# ---------------------------------------------------------------------------
# Model wiring: both families expose (nee_mean, nee_log_std); log_nu gated
# ---------------------------------------------------------------------------

def _batch(n=32):
    g = torch.Generator().manual_seed(1)
    return {
        "X": torch.randn(n, 17, generator=g), "bNEE": torch.randn(n, generator=g),
        "k": torch.abs(torch.randn(n, 2, generator=g)) + 1, "T": torch.randn(n, generator=g) * 10 + 10,
        "dT": torch.randn(n, generator=g) * 0.05, "dNEE": torch.randn(n, generator=g) * 0.06,
        "NEE": torch.randn(n, generator=g) * 3, "dt": torch.full((n,), 30.0),
    }


def _fwd(m, b):
    return m(b["X"], b["bNEE"], b["k"], b["T"], b["dt"], b["dT"])


def test_increment_exposes_nll_moments_and_dof_gated():
    b = _batch()
    m0 = build_increment_model("piae_increment_residual", input_dim=17, device="cpu",
                               physics_k_source="ground_truth", predict_k=False)
    out0 = _fwd(m0, b)
    assert out0["nee_mean"].shape == (32, 1) and out0["nee_log_std"].shape == (32, 1)
    assert out0["log_nu"] is None and not hasattr(m0, "log_nu") or m0.log_nu is None
    # nee_mean is the deterministic prediction: nee_pred == nee_mean + noise*sqrt(dt)
    noise_contrib = out0["noise"] * torch.sqrt(b["dt"].view(-1, 1))
    torch.testing.assert_close(out0["nee_pred"], out0["nee_mean"] + noise_contrib, atol=1e-5, rtol=1e-4)

    m1 = build_increment_model("piae_increment_residual", input_dim=17, device="cpu",
                               physics_k_source="ground_truth", predict_k=False,
                               noise_student_dof=True)
    assert isinstance(m1.log_nu, torch.nn.Parameter)
    assert _fwd(m1, b)["log_nu"] is not None


def test_level_model_exposes_nll_moments():
    b = _batch()
    m = build_model("piae_sde_sampling", input_dim=20, device="cpu",
                    heads=HeadsConfig(temp_derivative=True, k=True, noise=True, noise_zero_mean=True))
    out = _fwd(m, b)
    assert out["nee_mean"].shape == (32, 1) and out["nee_log_std"] is not None
    # deterministic mean = nee_pred - noise (level noise added at the level)
    torch.testing.assert_close(out["nee_pred"], out["nee_mean"] + out["noise"], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("variant", ["gaussian", "beta", "student_t"])
def test_nll_gradient_flow_increment(variant):
    b = _batch()
    m = build_increment_model("piae_increment_residual", input_dim=17, device="cpu",
                              physics_k_source="ground_truth", predict_k=False,
                              noise_student_dof=(variant == "student_t"))
    losses = compute_losses(b, _fwd(m, b), {"residual_l2": 1.0},
                            likelihood={"variant": variant, "weight": 1.0, "beta": 0.5})
    assert "nll" in losses and torch.isfinite(losses["nll"])
    sum(losses.values()).backward()
    # sigma head + residual head must receive gradient; log_nu too for student_t
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.sigma_decoder.parameters()), f"{variant}: sigma head no grad"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.residual_decoder.parameters()), f"{variant}: residual head no grad"
    if variant == "student_t":
        assert m.log_nu.grad is not None and m.log_nu.grad.abs() > 0


def test_student_dof_survives_checkpoint():
    b = _batch()
    m = build_increment_model("piae_increment_residual", input_dim=17, device="cpu",
                              physics_k_source="ground_truth", predict_k=False, noise_student_dof=True)
    with torch.no_grad():
        m.log_nu.add_(0.5)                      # move it off init
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "c.pth"
        save_checkpoint(ckpt, m)
        m2 = build_increment_model("piae_increment_residual", input_dim=17, device="cpu",
                                   physics_k_source="ground_truth", predict_k=False, noise_student_dof=True)
        load_model_weights(ckpt, m2)
    torch.testing.assert_close(m2.log_nu, m.log_nu)
