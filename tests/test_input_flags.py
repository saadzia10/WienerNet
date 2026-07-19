"""Tests for the configurable encoder inputs (E0/rb/dTa) + physics k-source.

Covers the InputsConfig flags (include_bnee/include_k/include_dtemp,
scale_extra_inputs), the encoder_input_dim helper, the normalisation buffers,
and the physics_k_source switch on the analytic SDE drift.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wienernet.models import (
    HeadsConfig,
    InputsConfig,
    WienerNetConfig,
    WienerNetModel,
    build_model,
    encoder_input_dim,
)
from wienernet.physics.lloyd_taylor import sde_drift
from wienernet.utils import load_model_weights, save_checkpoint

FEATURE_DIM = 17  # X width in the synthetic_batch fixture


def _cfg(inputs: InputsConfig, **kw) -> WienerNetConfig:
    return WienerNetConfig(
        input_dim=encoder_input_dim(FEATURE_DIM, inputs),
        inputs=inputs,
        device="cpu",
        **kw,
    )


# ---------------------------------------------------------------------------
# encoder_input_dim helper
# ---------------------------------------------------------------------------

def test_encoder_input_dim_flag_combinations():
    assert encoder_input_dim(17, InputsConfig()) == 20                       # bnee + k
    assert encoder_input_dim(17, InputsConfig(include_dtemp=True)) == 21     # + dTa
    assert encoder_input_dim(17, InputsConfig(include_k=False)) == 18        # bnee only
    assert encoder_input_dim(17, InputsConfig(include_bnee=False)) == 19     # k only
    assert encoder_input_dim(
        17, InputsConfig(include_bnee=False, include_k=False, include_dtemp=False)
    ) == 17                                                                  # features only


# ---------------------------------------------------------------------------
# Default flags reproduce the legacy cat(x, bNEE, k)
# ---------------------------------------------------------------------------

def test_default_inputs_reproduce_legacy_concat(synthetic_batch):
    model = build_model("piae_sde_sampling", input_dim=20, device="cpu")
    enc = model.build_encoder_input(
        synthetic_batch["X"], synthetic_batch["bNEE"], synthetic_batch["k"]
    )
    legacy = torch.cat(
        (synthetic_batch["X"], synthetic_batch["bNEE"].view(-1, 1), synthetic_batch["k"]), dim=1
    )
    assert torch.equal(enc, legacy)


# ---------------------------------------------------------------------------
# include_dtemp: adds a column, needs dT, drift still uses the predicted head
# ---------------------------------------------------------------------------

def test_include_dtemp_adds_input_and_requires_dT(synthetic_batch):
    inputs = InputsConfig(include_dtemp=True)
    model = WienerNetModel(_cfg(inputs)).initialize()
    X, bNEE, k, T, dT = (synthetic_batch[key] for key in ("X", "bNEE", "k", "T", "dT"))

    with pytest.raises(ValueError, match="include_dtemp"):
        model.build_encoder_input(X, bNEE, k)  # dT omitted

    enc = model.build_encoder_input(X, bNEE, k, dT)
    assert enc.shape == (X.shape[0], 21)
    # dTa concatenated raw (scale off) as the last column
    torch.testing.assert_close(enc[:, -1], dT.view(-1))

    out = model(X, bNEE, k, T, None, dT)
    assert out["nee_pred"].shape == (X.shape[0], 1)
    # dTa-as-input does not feed the drift: drift still uses the predicted dT/dt head
    expected = sde_drift(
        T.view(-1, 1), out["k"][:, 0:1], out["k"][:, 1:2], out["temp_derivative"],
        tref=model.cfg.tref, t0=model.cfg.t0,
    )
    torch.testing.assert_close(out["drift"], expected)


# ---------------------------------------------------------------------------
# include_k=False: k dropped from the encoder input; head still predicts it
# ---------------------------------------------------------------------------

def test_include_k_false_drops_input_but_keeps_head(synthetic_batch):
    inputs = InputsConfig(include_k=False)     # bNEE stays in, k out
    model = WienerNetModel(_cfg(inputs)).initialize()   # heads default: k=True
    X, bNEE, k, T = (synthetic_batch[key] for key in ("X", "bNEE", "k", "T"))

    enc = model.build_encoder_input(X, bNEE, k)
    assert enc.shape == (X.shape[0], 18)       # 17 features + bNEE

    out = model(X, bNEE, k, T)
    assert out["k"] is not None                # exogenous k head unaffected by the input flag
    assert out["k"].shape == (X.shape[0], 2)


# ---------------------------------------------------------------------------
# scale_extra_inputs: buffers exist, default to identity, apply once set
# ---------------------------------------------------------------------------

def test_scale_extra_inputs_buffers_and_normalisation(synthetic_batch):
    inputs = InputsConfig(scale_extra_inputs=True)      # default include_k=True
    model = WienerNetModel(_cfg(inputs)).initialize()
    X, bNEE, k = (synthetic_batch[key] for key in ("X", "bNEE", "k"))

    assert hasattr(model, "k_norm_mean") and hasattr(model, "dtemp_norm_mean")

    # Identity buffers -> raw concat (no silent scaling before stats are set)
    legacy = torch.cat((X, bNEE.view(-1, 1), k), dim=1)
    torch.testing.assert_close(model.build_encoder_input(X, bNEE, k), legacy)

    km, ks = k.mean(0), k.std(0)
    model.set_input_norm_stats(k_mean=km, k_std=ks)
    enc = model.build_encoder_input(X, bNEE, k)
    # Column layout is [X(17), bNEE(1), k(2)], so k occupies cols 18:20.
    k_start = FEATURE_DIM + 1  # +1 for the included bNEE
    torch.testing.assert_close(enc[:, k_start:k_start + 2], (k - km) / ks)


def test_scale_off_has_no_buffers_and_rejects_stats():
    model = WienerNetModel(WienerNetConfig(input_dim=20, device="cpu"))
    assert not hasattr(model, "k_norm_mean")
    with pytest.raises(RuntimeError, match="scale_extra_inputs"):
        model.set_input_norm_stats(k_mean=[1.0, 2.0])


# ---------------------------------------------------------------------------
# physics_k_source: 'ground_truth' feeds GT (E0, rb) into the drift
# ---------------------------------------------------------------------------

def test_physics_k_source_ground_truth_uses_gt_k(synthetic_batch):
    cfg = WienerNetConfig(input_dim=20, physics_k_source="ground_truth", device="cpu")
    model = WienerNetModel(cfg).initialize()
    X, bNEE, k, T = (synthetic_batch[key] for key in ("X", "bNEE", "k", "T"))

    out = model(X, bNEE, k, T)
    gt_drift = sde_drift(
        T.view(-1, 1), k[:, 0:1], k[:, 1:2], out["temp_derivative"], tref=cfg.tref, t0=cfg.t0
    )
    torch.testing.assert_close(out["drift"], gt_drift)      # drift built from GT k

    # ...and (with the k head still on) it is NOT the predicted-k drift
    pred_drift = sde_drift(
        T.view(-1, 1), out["k"][:, 0:1], out["k"][:, 1:2], out["temp_derivative"],
        tref=cfg.tref, t0=cfg.t0,
    )
    assert not torch.allclose(out["drift"], pred_drift)


def test_physics_k_source_ground_truth_works_without_k_head(synthetic_batch):
    """GT k source lets the drift exist even when the k head is disabled."""
    cfg = WienerNetConfig(
        input_dim=20, physics_k_source="ground_truth", device="cpu",
        heads=HeadsConfig(temp_derivative=True, k=False, noise=True),
    )
    model = WienerNetModel(cfg).initialize()
    X, bNEE, k, T = (synthetic_batch[key] for key in ("X", "bNEE", "k", "T"))

    out = model(X, bNEE, k, T)
    assert out["k"] is None            # head off
    assert out["drift"] is not None    # but GT k still drives the physics


def test_invalid_physics_k_source_raises():
    with pytest.raises(ValueError, match="physics_k_source"):
        WienerNetModel(WienerNetConfig(input_dim=20, physics_k_source="bogus", device="cpu"))


# ---------------------------------------------------------------------------
# Scaled model round-trips through a checkpoint (buffers save + load strictly)
# ---------------------------------------------------------------------------

def test_scaled_model_checkpoint_roundtrip(synthetic_batch):
    inputs = InputsConfig(scale_extra_inputs=True)
    model_a = WienerNetModel(_cfg(inputs)).initialize()
    X, bNEE, k, T = (synthetic_batch[key] for key in ("X", "bNEE", "k", "T"))
    model_a.set_input_norm_stats(k_mean=k.mean(0), k_std=k.std(0))

    optim = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    loss = F.mse_loss(model_a(X, bNEE, k, T)["nee_pred"], synthetic_batch["NEE"].view(-1, 1))
    optim.zero_grad(); loss.backward(); optim.step()

    model_a.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        ref = model_a(X, bNEE, k, T)["nee_pred"]

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt.pth"
        save_checkpoint(ckpt, model_a)
        model_b = WienerNetModel(_cfg(inputs)).initialize()
        load_model_weights(ckpt, model_b)   # strict load must accept the norm buffers

    # Buffers restored from disk, not re-injected
    torch.testing.assert_close(model_b.k_norm_mean, model_a.k_norm_mean)
    model_b.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        loaded = model_b(X, bNEE, k, T)["nee_pred"]
    assert torch.allclose(ref, loaded, atol=1e-6)
