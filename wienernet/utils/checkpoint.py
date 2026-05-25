"""Checkpoint save/load.

Saves everything needed for true resumption: model, optimizer, scheduler,
AMP scaler, epoch/step counters, best metric, config, and RNG states.
`load_model_weights` is the lightweight inference-time variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .reproducibility import restore_rng_state, snapshot_rng_state


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    epoch: int | None = None,
    global_step: int | None = None,
    best_metric: float | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Persist a full training checkpoint.

    Args:
        path: Destination file (parent dirs created automatically).
        model: Module whose state_dict is saved.
        optimizer, scheduler, scaler: Optional; state_dicts saved if provided.
        epoch, global_step, best_metric: Training-loop counters.
        config: The resolved config used for the run (serialized as-is).
        extra: Anything else you want to round-trip.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "rng_state": snapshot_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if global_step is not None:
        payload["global_step"] = global_step
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if config is not None:
        payload["config"] = dict(config)
    if extra:
        payload["extra"] = dict(extra)

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a full checkpoint into the provided objects.

    Returns:
        The remaining checkpoint metadata (epoch, global_step, best_metric,
        config, extra) so the training loop can resume from where it stopped.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng and "rng_state" in payload:
        restore_rng_state(payload["rng_state"])

    meta = {
        key: payload[key]
        for key in ("epoch", "global_step", "best_metric", "config", "extra")
        if key in payload
    }
    return meta


def load_model_weights(
    path: str | Path,
    model: torch.nn.Module,
    *,
    map_location: str | torch.device | None = None,
    weights_key: str = "model_state_dict",
) -> None:
    """Inference-time loader: weights only, no optimizer/RNG/etc.

    Accepts both full-checkpoint payloads (as written by `save_checkpoint`)
    and bare state_dict files (e.g. the existing notebook's best_model.pth).
    Tries `weights_only=True` first for safety on untrusted checkpoints,
    and falls back to `weights_only=False` only for our own full payloads
    (which contain numpy RNG state that the strict loader rejects).
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # Full payloads include numpy RNG arrays; fall back to a trusted load.
        payload = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = (
        payload[weights_key]
        if isinstance(payload, dict) and weights_key in payload
        else payload
    )
    model.load_state_dict(state_dict)
