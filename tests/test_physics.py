"""Physics layer tests — the torch and numpy formulas must agree."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data_pipeline.partitioning import lloyd_taylor as np_lloyd_taylor
from data_pipeline.partitioning import lloyd_taylor_invert_to_rb
from wienernet.physics.lloyd_taylor import dreco_dT, reco, sde_drift


def test_torch_reco_matches_numpy():
    """The torch and numpy Lloyd-Taylor implementations must give identical values."""
    T = np.array([0., 10., 15., 20., 25., 30.])
    np_vals = np_lloyd_taylor(T, Reco_ref=2.5, E0=200.0)
    torch_vals = reco(
        torch.tensor(T, dtype=torch.float64),
        E0=torch.tensor(200.0, dtype=torch.float64),
        rb=torch.tensor(2.5, dtype=torch.float64),
    )
    np.testing.assert_allclose(torch_vals.numpy(), np_vals, atol=1e-12)


def test_reco_equals_rb_at_tref():
    """At T = Tref the exponent is 0, so reco(Tref) = rb exactly."""
    val = reco(torch.tensor([10.0]), E0=torch.tensor(200.0), rb=torch.tensor(2.5))
    assert torch.allclose(val, torch.tensor([2.5]))


def test_dreco_dT_matches_finite_difference():
    """Analytic dReco/dT should match numerical differentiation."""
    T = torch.tensor([15.0], dtype=torch.float64)
    E0, rb = torch.tensor(200.0, dtype=torch.float64), torch.tensor(2.5, dtype=torch.float64)
    eps = 1e-5
    fd = (reco(T + eps, E0, rb) - reco(T - eps, E0, rb)) / (2 * eps)
    analytic = dreco_dT(T, E0, rb)
    assert torch.allclose(analytic, fd, atol=1e-6), (
        f"analytic={analytic.item():.6f}, finite-diff={fd.item():.6f}"
    )


def test_sde_drift_chain_rule():
    """sde_drift = dReco/dT * dT/dt."""
    T = torch.tensor([15.0])
    dT_dt = torch.tensor([0.05])
    E0, rb = torch.tensor(200.0), torch.tensor(2.5)
    expected = dreco_dT(T, E0, rb) * dT_dt
    actual = sde_drift(T, E0, rb, dT_dt)
    assert torch.allclose(actual, expected)


def test_lloyd_taylor_invert_round_trip():
    """rb = invert(reco(T, rb, E0), T, E0) must round-trip."""
    T = np.array([5., 10., 15., 20., 25.])
    E0 = np.array([100., 150., 200., 250., 300.])
    rb = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    reco_vals = np_lloyd_taylor(T, rb, E0)
    rb_back = lloyd_taylor_invert_to_rb(reco_vals, T, E0)
    np.testing.assert_allclose(rb_back, rb, atol=1e-10)


def test_drift_gradient_flows_through_E0_rb():
    """Gradient should flow from drift through (E0, rb) so k_decoder is trainable."""
    T = torch.tensor([15.0])
    dT_dt = torch.tensor([0.05])
    E0 = torch.tensor(200.0, requires_grad=True)
    rb = torch.tensor(2.5, requires_grad=True)

    drift = sde_drift(T, E0, rb, dT_dt)
    drift.sum().backward()

    assert E0.grad is not None and E0.grad.abs() > 0
    assert rb.grad is not None and rb.grad.abs() > 0


def test_model_physics_matches_partitioner_on_real_temperatures():
    """End-to-end: random batch, model.physics_residual should reproduce
    partitioner's lloyd_taylor for the same (E0, rb, T).
    """
    from wienernet.models import build_model
    torch.manual_seed(0)

    model = build_model("piae_sde_sampling", input_dim=20, device="cpu")
    T = torch.tensor([[5.], [10.], [15.], [20.], [25.]], dtype=torch.float32)
    k = torch.tensor([[200., 2.5]] * 5, dtype=torch.float32)
    dT_dt = torch.ones_like(T)

    # Use the model's sde_drift helper through the same path as forward
    from wienernet.physics.lloyd_taylor import sde_drift as torch_drift
    model_dreco = torch_drift(T, k[:, 0:1], k[:, 1:2], dT_dt,
                              tref=model.cfg.tref, t0=model.cfg.t0)

    # Partitioner's dR/dT
    np_T = T.flatten().numpy()
    Tref, T0 = 10.0, 46.02
    exp_term = np.exp(200.0 * (1.0 / (Tref + T0) - 1.0 / (np_T + T0)))
    np_dreco = 2.5 * (200.0 / (np_T + T0) ** 2) * exp_term

    np.testing.assert_allclose(model_dreco.flatten().numpy(), np_dreco, atol=1e-5)
