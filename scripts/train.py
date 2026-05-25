"""Hydra entrypoint for training a single WienerNet variant.

Usage:
    # Default (PIAE_SDE_Sampling on East Anglia data, seed 42)
    python scripts/train.py

    # Override individual fields
    python scripts/train.py model=ae loss=ae seed=88 training.num_epochs=50

    # Run the paper's headline configuration
    python scripts/train.py experiment=paper_main

    # Multirun: 5 seeds × 3 models
    python scripts/train.py -m model=piae_sde_sampling,ae,vae seed=0,1,2,41,88
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

# Make the repo root importable when invoked as `python scripts/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# mlflow is optional — guard the import so missing dep doesn't break training
try:
    import mlflow
except ImportError:
    mlflow = None

from wienernet.data import build_dataloaders
from wienernet.losses import MMDLoss, make_gaussian_noise_prior
from wienernet.models import HeadsConfig, WienerNetConfig, WienerNetModel
from wienernet.training import Trainer, build_optimizer, build_scheduler
from wienernet.utils import (
    get_logger,
    save_checkpoint,
    set_seed_globally,
    setup_logging,
)


def _resolve_device(device_str: str) -> str:
    """Honour user request while warning when the asked-for device isn't available."""
    import torch
    requested = device_str.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


def _flatten_for_mlflow(cfg_dict: dict, parent_key: str = "", separator: str = ".") -> dict:
    """Flatten the resolved config into dot-notated params MLflow can consume."""
    flat: dict[str, str] = {}
    for key, value in cfg_dict.items():
        full = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_for_mlflow(value, full, separator))
        elif isinstance(value, list):
            # MLflow params must be strings; serialise lists.
            flat[full] = str(value)
        else:
            flat[full] = "null" if value is None else str(value)
    return flat


def _setup_mlflow(cfg: DictConfig, log) -> bool:
    """Configure MLflow if requested and importable. Returns True iff active."""
    if not cfg.get("mlflow", {}).get("enabled", False):
        return False
    if mlflow is None:
        log.warning("mlflow is enabled in config but not installed; skipping")
        return False
    tracking_uri = cfg.mlflow.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    log.info("MLflow tracking URI: %s  experiment: %s",
             mlflow.get_tracking_uri(), cfg.mlflow.experiment_name)
    return True


def _build_model_from_cfg(cfg: DictConfig, input_dim: int, device: str) -> WienerNetModel:
    """Map the Hydra model config block to a WienerNetConfig."""
    heads = HeadsConfig(
        nee=True,
        temp_derivative=bool(cfg.model.heads.temp_derivative),
        k=bool(cfg.model.heads.k),
        noise=bool(cfg.model.heads.noise),
        noise_dims=tuple(cfg.model.heads.get("noise_dims", [4])),
        k_activation=cfg.model.heads.get("k_activation"),
        k_activation_slope=float(cfg.model.heads.get("k_activation_slope", 0.01)),
    )
    model_cfg = WienerNetConfig(
        input_dim=input_dim,
        latent_dim=int(cfg.model.latent_dim),
        encoder_dims=tuple(cfg.model.encoder_dims),
        decoder_dims=tuple(cfg.model.decoder_dims),
        activation=str(cfg.model.activation),
        latent_reparameterize=bool(cfg.model.latent_reparameterize),
        predict_drift=bool(cfg.model.predict_drift),
        heads=heads,
        tref=float(cfg.model.tref),
        t0=float(cfg.model.t0),
        device=device,
    )
    return WienerNetModel(model_cfg).initialize()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    # ---------- Setup ----------
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).resolve()
    log = setup_logging(run_dir)
    log.info("run dir: %s", run_dir)

    set_seed_globally(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = _resolve_device(str(cfg.device))
    if device != cfg.device:
        log.warning("requested device=%s not available; falling back to %s", cfg.device, device)

    # Save the fully-resolved config as JSON for downstream eval scripts
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2, default=str))
    log.info("variant=%s seed=%d device=%s", cfg.model.variant, cfg.seed, device)

    # ---------- Data ----------
    log.info("building dataloaders...")
    scaler_path = run_dir / cfg.data.get("save_scaler_filename", "scaler.pkl")
    bundle = build_dataloaders(
        site_paths={k: v for k, v in OmegaConf.to_container(cfg.data.site_paths, resolve=True).items()},
        drivers=tuple(cfg.data.drivers),
        include_sites=tuple(cfg.data.include_sites) or None,
        exclude_sites=tuple(cfg.data.exclude_sites),
        nighttime_only=bool(cfg.data.nighttime_only),
        night_radiation_threshold=float(cfg.data.night_radiation_threshold),
        nee_target_column=str(cfg.data.nee_target_column),
        boundary_nee_column=str(cfg.data.boundary_nee_column),
        e0_column=str(cfg.data.e0_column),
        rb_column=str(cfg.data.rb_column),
        temperature_column=str(cfg.data.temperature_column),
        dtemp_column=str(cfg.data.dtemp_column),
        dnee_column=str(cfg.data.dnee_column),
        split_strategy=str(cfg.data.split_strategy),
        test_frac=float(cfg.data.test_frac),
        test_years=tuple(cfg.data.test_years),
        shuffle_split=bool(cfg.data.shuffle_split),
        split_random_state=int(cfg.data.split_random_state),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=bool(cfg.data.persistent_workers),
        save_scaler_path=scaler_path,
    )
    log.info("train batches=%d  test batches=%d  input_dim=%d",
             len(bundle.train_loader), len(bundle.test_loader), bundle.input_dim)

    # ---------- Model ----------
    model = _build_model_from_cfg(cfg, input_dim=bundle.input_dim, device=device)
    log.info("model built: %d params",
             sum(p.numel() for p in model.parameters()))

    # ---------- Optimiser, scheduler ----------
    optimizer = build_optimizer(
        model,
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
        optimizer=str(cfg.training.optimizer.name),
    )
    scheduler = build_scheduler(
        optimizer,
        name=str(cfg.training.scheduler.name),
        patience=int(cfg.training.scheduler.get("patience", 10)),
        factor=float(cfg.training.scheduler.get("factor", 0.5)),
        mode=str(cfg.training.scheduler.get("mode", "min")),
    )

    # ---------- Losses ----------
    raw_weights = OmegaConf.to_container(cfg.loss.weights, resolve=True)
    loss_weights = {k: float(v) for k, v in raw_weights.items()}

    needs_mmd = any(loss_weights.get(k, 0) > 0 for k in ("mmd_nee", "mmd_bnee", "mmd_noise"))
    mmd_fn = (
        MMDLoss(
            kernel_mul=float(cfg.loss.mmd.kernel_mul),
            kernel_num=int(cfg.loss.mmd.kernel_num),
            fix_sigma=cfg.loss.mmd.get("fix_sigma"),
        ).to(device)
        if needs_mmd
        else None
    )

    noise_prior_mean = cfg.training.noise_prior.get("mean")
    noise_prior_std = cfg.training.noise_prior.get("std")
    noise_prior_fn = (
        make_gaussian_noise_prior(
            mean=bundle.noise_mu if noise_prior_mean is None else float(noise_prior_mean),
            std=bundle.noise_std if noise_prior_std is None else float(noise_prior_std),
        )
        if loss_weights.get("mmd_noise", 0) > 0
        else None
    )

    # ---------- TensorBoard ----------
    tb_writer = SummaryWriter(run_dir / "tensorboard")

    # ---------- Trainer ----------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_weights=loss_weights,
        mmd_loss_fn=mmd_fn,
        noise_prior_fn=noise_prior_fn,
        device=device,
        writer=tb_writer,
        clip_grad_norm=cfg.training.get("clip_grad_norm"),
        nan_policy=str(cfg.training.nan_policy),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
    )

    log.info("starting training: %d epochs", cfg.training.num_epochs)

    # MLflow: param log + run-context wrapping the training loop
    mlflow_active = _setup_mlflow(cfg, log)
    mlflow_run_ctx = (
        mlflow.start_run(run_name=str(cfg.run_name)) if mlflow_active else _NullContext()
    )

    with mlflow_run_ctx:
        if mlflow_active:
            mlflow.log_params(_flatten_for_mlflow(resolved))
            mlflow.set_tag("variant", str(cfg.model.variant))
            mlflow.set_tag("seed", str(cfg.seed))

        history = trainer.fit(
            train_loader=bundle.train_loader,
            val_loader=bundle.test_loader,
            num_epochs=int(cfg.training.num_epochs),
            run_dir=run_dir,
        )

        if mlflow_active:
            for metric_name, values in history.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=step)
            mlflow.log_metric("best_val_loss", trainer.best_val_loss)
            for artefact in (run_dir / "checkpoints" / "best.pth",
                             run_dir / "config.json",
                             run_dir / "scaler.pkl"):
                if artefact.exists():
                    mlflow.log_artifact(str(artefact))

    tb_writer.close()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info("training complete. best_val_loss=%.6f", trainer.best_val_loss)

    return trainer.best_val_loss


class _NullContext:
    """No-op context manager used when MLflow is disabled or unavailable."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    main()
