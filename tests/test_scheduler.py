"""LR-scheduler construction and stepping.

Regression tests for two silent defects:

1. `build_scheduler` forwarded **kwargs blindly, so the caller's union of options
   (patience/factor/mode) reached CosineAnnealingLR and raised TypeError — the 'cosine'
   option was unusable.
2. The trainer stepped schedulers via `try: step(val_loss) / except TypeError: step()`.
   Epoch-based schedulers accept a positional argument as `epoch`, so they did NOT raise:
   the validation loss was silently passed as the epoch number, corrupting the schedule.
"""
from __future__ import annotations

import pytest
import torch

from wienernet.training import build_scheduler


def _opt(lr=1e-3):
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(2))], lr=lr)


def test_none_returns_none():
    assert build_scheduler(_opt(), name="none") is None


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        build_scheduler(_opt(), name="not_a_scheduler")


def test_plateau_builds_with_its_options():
    s = build_scheduler(_opt(), name="reduce_on_plateau", patience=3, factor=0.25, mode="min")
    assert isinstance(s, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert s.patience == 3 and s.factor == 0.25 and s.mode == "min"


def test_cosine_builds_despite_plateau_kwargs():
    """The caller always passes patience/factor/mode; cosine must ignore them, not crash."""
    s = build_scheduler(_opt(), name="cosine", patience=10, factor=0.5, mode="min", t_max=25)
    assert isinstance(s, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert s.T_max == 25


def test_cosine_requires_t_max():
    with pytest.raises(ValueError, match="t_max"):
        build_scheduler(_opt(), name="cosine", patience=10, factor=0.5, mode="min")


def test_cosine_decays_when_stepped_without_a_metric():
    """Epoch-based stepping must lower the LR monotonically over the horizon."""
    opt = _opt(lr=1e-3)
    s = build_scheduler(opt, name="cosine", patience=10, factor=0.5, mode="min", t_max=10)
    lrs = []
    for _ in range(5):
        opt.step()
        s.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert all(b < a for a, b in zip(lrs, lrs[1:])), f"cosine LR not decaying: {lrs}"


def test_trainer_scheduler_monitor_validation():
    """`monitor` selects which loss a plateau scheduler reacts to; 'train' is leakage-safe."""
    from wienernet.training import Trainer

    net = torch.nn.Linear(1, 1)
    for good in ("val", "train", "TRAIN"):
        t = Trainer(net, _opt(), scheduler_monitor=good)
        assert t.scheduler_monitor == good.lower()
    with pytest.raises(ValueError, match="scheduler_monitor"):
        Trainer(net, _opt(), scheduler_monitor="testset")


def test_plateau_reduces_lr_after_patience_on_flat_metric():
    opt = _opt(lr=1e-3)
    s = build_scheduler(opt, name="reduce_on_plateau", patience=1, factor=0.5, mode="min")
    for _ in range(6):
        s.step(1.0)          # never improves -> must drop the LR
    assert opt.param_groups[0]["lr"] < 1e-3
