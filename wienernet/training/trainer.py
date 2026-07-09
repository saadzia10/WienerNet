"""Unified training loop.

Replaces the five copy-pasted Trainer classes in piae_sde/night/*.py. Takes a
WienerNetModel (or any Module returning the same output dict), a loss-weight
dict, and runs the train/val loop with TensorBoard logging, checkpointing,
NaN-loss detection, and optional gradient clipping.

The MLflow integration is added later as a callback (Phase 7) so this module
stays focused on the training mechanics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..losses.composite import compute_losses
from ..losses.mmd import MMDLoss
from ..utils.checkpoint import save_checkpoint
from ..utils.logging import get_logger
from ..utils.paths import best_checkpoint, last_checkpoint, tensorboard_dir

log = get_logger("training.trainer")


# Map dataset keys to training inputs (matches ClimateDataset.__getitem__).
_BATCH_KEYS = ("X", "k", "T", "dNEE", "bNEE", "dT", "NEE", "dt")


class Trainer:
    """End-to-end training driver for a WienerNetModel.

    Args:
        model: instance returning the model_outputs dict.
        optimizer: any torch optimizer.
        scheduler: optional lr scheduler (e.g. ReduceLROnPlateau).
        loss_weights: dict {term: weight}. Pass 0 (or omit) to disable a term.
            See `wienernet.losses.composite.compute_losses` for the term list.
        mmd_loss_fn: required iff any mmd_* weight > 0. Defaults to MMDLoss().
        noise_prior_fn: required iff mmd_noise > 0.
        device: torch device.
        writer: optional TensorBoard SummaryWriter. If None, no TB logging.
        clip_grad_norm: if not None, clip gradients to this max norm each step.
        nan_policy: 'raise' (default) or 'skip'. 'skip' is for debugging only.
        log_every_n_steps: console/TB log frequency within an epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        scheduler: Any = None,
        loss_weights: Mapping[str, float] | None = None,
        mmd_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        noise_prior_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: str | torch.device = "cpu",
        writer: Any = None,
        clip_grad_norm: float | None = None,
        nan_policy: str = "raise",
        log_every_n_steps: int = 50,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_weights = dict(loss_weights or {})
        self.device = torch.device(device)
        self.writer = writer
        self.clip_grad_norm = clip_grad_norm
        if nan_policy not in ("raise", "skip"):
            raise ValueError(f"nan_policy must be 'raise' or 'skip', got {nan_policy!r}")
        self.nan_policy = nan_policy
        self.log_every_n_steps = log_every_n_steps

        # Lazily build a default MMD if user enabled mmd_* without supplying one
        needs_mmd = any(self.loss_weights.get(k, 0) > 0 for k in ("mmd_nee", "mmd_bnee", "mmd_noise"))
        self.mmd_loss_fn = mmd_loss_fn or (MMDLoss() if needs_mmd else None)
        self.noise_prior_fn = noise_prior_fn

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.start_epoch = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        num_epochs: int,
        run_dir: str | Path,
        save_last: bool = True,
    ) -> dict[str, list[float]]:
        """Run the full train/val loop, saving best & last checkpoints.

        Returns a history dict keyed by metric name with per-epoch averages.
        """
        run_dir = Path(run_dir)
        history: dict[str, list[float]] = defaultdict(list)

        for epoch in range(self.start_epoch, num_epochs):
            train_metrics = self._run_epoch(train_loader, epoch, train=True)
            val_metrics = self._run_epoch(val_loader, epoch, train=False)

            for name, value in train_metrics.items():
                history[f"train/{name}"].append(value)
            for name, value in val_metrics.items():
                history[f"val/{name}"].append(value)

            self._log_epoch(epoch, train_metrics, val_metrics)

            # Save best (always) and last (optional)
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    best_checkpoint(run_dir),
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=self.best_val_loss,
                )
                log.info(
                    "New best @ epoch %d: val loss %.6f → saved best.pth", epoch, val_loss
                )

            if save_last:
                save_checkpoint(
                    last_checkpoint(run_dir),
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_metric=self.best_val_loss,
                )

            if self.scheduler is not None:
                # ReduceLROnPlateau needs a metric; cosine-style schedulers don't.
                try:
                    self.scheduler.step(val_loss)
                except TypeError:
                    self.scheduler.step()

        return dict(history)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Run inference, returning (ground_truth, predictions) as numpy dicts.

        Output structure mimics the original `predict()` methods so the
        analysis notebook can drop-in replace its call.
        """
        self.model.eval()
        gt_buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        pred_buckets: dict[str, list[np.ndarray]] = defaultdict(list)

        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.model(batch["X"], batch["bNEE"], batch["k"], batch["T"], batch.get("dt"))

            # Ground truth: rename a couple keys to match original convention
            gt_buckets["nee"].append(batch["NEE"].detach().cpu().numpy())
            gt_buckets["bnee"].append(batch["bNEE"].detach().cpu().numpy())
            gt_buckets["E0"].append(batch["k"][:, 0].detach().cpu().numpy())
            gt_buckets["rb"].append(batch["k"][:, 1].detach().cpu().numpy())
            gt_buckets["dtemp"].append(batch["dT"].detach().cpu().numpy())
            gt_buckets["f"].append(batch["dNEE"].detach().cpu().numpy())

            pred_buckets["nee"].append(outputs["nee_pred"].detach().cpu().numpy())
            if outputs.get("bnee") is not None:
                pred_buckets["bnee"].append(outputs["bnee"].detach().cpu().numpy())
            if outputs.get("k") is not None:
                pred_buckets["E0"].append(outputs["k"][:, 0].detach().cpu().numpy())
                pred_buckets["rb"].append(outputs["k"][:, 1].detach().cpu().numpy())
            if outputs.get("temp_derivative") is not None:
                pred_buckets["dtemp"].append(outputs["temp_derivative"].detach().cpu().numpy())
            if outputs.get("drift") is not None:
                pred_buckets["f"].append(outputs["drift"].detach().cpu().numpy())
            if outputs.get("noise") is not None:
                pred_buckets["noise"].append(outputs["noise"].detach().cpu().numpy())
                pred_buckets["noise_mus"].append(outputs["noise_mu"].detach().cpu().numpy())
                pred_buckets["noise_stds"].append(
                    torch.exp(0.5 * outputs["noise_logvar"]).detach().cpu().numpy()
                )
            pred_buckets["z"].append(outputs["latent"].detach().cpu().numpy())

        def _stack(buckets: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            for k, vals in buckets.items():
                arr = np.concatenate(vals, axis=0)
                if arr.ndim > 1 and arr.shape[1] == 1:
                    arr = arr.flatten()
                out[k] = arr
            return out

        return _stack(gt_buckets), _stack(pred_buckets)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        *,
        train: bool,
    ) -> dict[str, float]:
        """Run one epoch in train or eval mode. Returns avg-per-batch metrics."""
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Accumulate per-batch losses as detached GPU tensors and sync to host
        # once, at epoch end. Calling .item() per batch (and per loss term) forces
        # a CUDA sync each step, stalling the GPU between batches — for this small,
        # fast model that synchronization dominated the epoch time.
        loss_sums: dict[str, torch.Tensor] = {}
        n_batches = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(loader):
                batch = self._to_device(batch)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(batch["X"], batch["bNEE"], batch["k"], batch["T"], batch.get("dt"))
                losses = compute_losses(
                    batch,
                    outputs,
                    self.loss_weights,
                    mmd_loss_fn=self.mmd_loss_fn,
                    noise_prior_fn=self.noise_prior_fn,
                )

                total = sum(losses.values()) if losses else torch.zeros((), device=self.device)

                if torch.isnan(total) or torch.isinf(total):
                    msg = f"Non-finite loss at epoch {epoch}, batch {batch_idx}: {total.item()}"
                    if self.nan_policy == "raise":
                        raise RuntimeError(msg)
                    log.warning("%s — skipping batch", msg)
                    continue

                if train:
                    total.backward()
                    if self.clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    self.global_step += 1

                n_batches += 1
                self._accumulate(loss_sums, "loss", total)
                for name, val in losses.items():
                    self._accumulate(loss_sums, name, val)

                if train and self.log_every_n_steps and self.global_step % self.log_every_n_steps == 0:
                    self._log_step(losses, total, epoch)

        if n_batches == 0:
            return {}
        return {name: (total_sum / n_batches).item() for name, total_sum in loss_sums.items()}

    @staticmethod
    def _accumulate(sums: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
        """Add a detached scalar loss into a running GPU-side sum without a host sync."""
        detached = value.detach()
        if name in sums:
            sums[name] += detached
        else:
            # clone so we own the accumulator and don't alias `value`'s storage
            sums[name] = detached.clone()

    def _to_device(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Move all tensor batch entries to the trainer device."""
        out: dict[str, torch.Tensor] = {}
        for key in _BATCH_KEYS:
            if key not in batch:
                continue
            out[key] = batch[key].to(self.device)
        # Preserve any extras (e.g. site_id strings) untouched
        for key, val in batch.items():
            if key in out:
                continue
            out[key] = val
        return out

    def _log_step(self, losses: Mapping[str, torch.Tensor], total: torch.Tensor, epoch: int) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar("train/step/loss", total.item(), self.global_step)
        for name, val in losses.items():
            self.writer.add_scalar(f"train/step/{name}", val.item(), self.global_step)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float],
    ) -> None:
        log.info(
            "epoch=%d  train_loss=%.6f  val_loss=%.6f",
            epoch, train_metrics["loss"], val_metrics["loss"],
        )
        if self.writer is None:
            return
        for name, value in train_metrics.items():
            self.writer.add_scalar(f"train/epoch/{name}", value, epoch)
        for name, value in val_metrics.items():
            self.writer.add_scalar(f"val/epoch/{name}", value, epoch)


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
) -> Optimizer:
    """Build the optimiser from a small dict of common choices."""
    name = optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer {optimizer!r}")


def build_scheduler(
    optimizer: Optimizer,
    *,
    name: str = "reduce_on_plateau",
    **kwargs,
) -> Any:
    """Build the LR scheduler. Default matches original notebook setup."""
    name = name.lower()
    if name == "reduce_on_plateau":
        defaults = {"mode": "min", "patience": 10, "factor": 0.5}
        defaults.update(kwargs)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **defaults)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler {name!r}")
