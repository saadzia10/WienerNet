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

from wienernet.data import build_dataloaders, ordered_site_names
from wienernet.data.loader import DataBundle
from wienernet.losses import MMDLoss, make_empirical_noise_prior, make_gaussian_noise_prior
from wienernet.models import (
    AnalyticalSDEConfig,
    AnalyticalSDEModel,
    HeadsConfig,
    HeteroscedasticMLPConfig,
    HeteroscedasticMLPModel,
    IncrementSDEConfig,
    IncrementSDEModel,
    InputsConfig,
    NeuralSDEConfig,
    NeuralSDEModel,
    WienerNetConfig,
    WienerNetModel,
    encoder_input_dim,
    is_baseline_variant,
    is_increment_variant,
)
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


def _inputs_from_cfg(cfg: DictConfig, *, exogenous_default: bool) -> InputsConfig:
    """Build InputsConfig from the (optional) cfg.model.inputs block.

    `exogenous_default` picks the fallback when the block is absent: level models
    default to the legacy encoder (bNEE + GT k in); the increment model defaults
    to fully exogenous (nothing extra in the encoder).
    """
    default_on = not exogenous_default
    inputs_block = cfg.model.get("inputs", {}) or {}
    return InputsConfig(
        include_bnee=bool(inputs_block.get("include_bnee", default_on)),
        include_k=bool(inputs_block.get("include_k", default_on)),
        include_dtemp=bool(inputs_block.get("include_dtemp", False)),
        scale_extra_inputs=bool(inputs_block.get("scale_extra_inputs", False)),
    )


def _maybe_set_norm_stats(model, inputs: InputsConfig, bundle: DataBundle) -> None:
    if inputs.scale_extra_inputs:
        model.set_input_norm_stats(
            k_mean=bundle.k_mean, k_std=bundle.k_std,
            dtemp_mean=bundle.dtemp_mean, dtemp_std=bundle.dtemp_std,
        )


def _build_model_from_cfg(cfg: DictConfig, bundle: DataBundle, device: str) -> WienerNetModel:
    """Map the Hydra model config block to a WienerNetConfig.

    The encoder input width is derived from the scaled feature width
    (`bundle.feature_dim`) plus the `inputs` flags, so toggling E0/rb/dTa in or
    out of the encoder resizes the first layer automatically. When
    `inputs.scale_extra_inputs` is on, the model's normalisation buffers are
    populated from the train-only stats in the bundle.
    """
    heads = HeadsConfig(
        nee=True,
        temp_derivative=bool(cfg.model.heads.temp_derivative),
        k=bool(cfg.model.heads.k),
        noise=bool(cfg.model.heads.noise),
        noise_zero_mean=bool(cfg.model.heads.get("noise_zero_mean", False)),
        noise_dims=tuple(cfg.model.heads.get("noise_dims", [4])),
        k_activation=cfg.model.heads.get("k_activation"),
        k_activation_slope=float(cfg.model.heads.get("k_activation_slope", 0.01)),
    )
    inputs = _inputs_from_cfg(cfg, exogenous_default=False)
    model_cfg = WienerNetConfig(
        input_dim=encoder_input_dim(bundle.feature_dim, inputs),
        latent_dim=int(cfg.model.latent_dim),
        encoder_dims=tuple(cfg.model.encoder_dims),
        decoder_dims=tuple(cfg.model.decoder_dims),
        activation=str(cfg.model.activation),
        latent_reparameterize=bool(cfg.model.latent_reparameterize),
        predict_drift=bool(cfg.model.predict_drift),
        heads=heads,
        inputs=inputs,
        physics_k_source=str(cfg.model.get("physics_k_source", "predicted")),
        noise_student_dof=_needs_student_dof(cfg),
        tref=float(cfg.model.tref),
        t0=float(cfg.model.t0),
        dt=float(cfg.model.get("dt", 30.0)),
        device=device,
    )
    model = WienerNetModel(model_cfg).initialize()
    _maybe_set_norm_stats(model, inputs, bundle)
    return model


def _needs_student_dof(cfg: DictConfig) -> bool:
    """True if the model needs a learnable log_nu (loss uses the student_t NLL)."""
    lik = cfg.loss.get("likelihood") or {}
    return bool(cfg.model.get("noise_student_dof", False)) or str(lik.get("variant", "")) == "student_t"


def _needs_asymmetry(cfg: DictConfig) -> bool:
    """True if the model needs a learnable log_kappa (loss uses the ALD NLL)."""
    lik = cfg.loss.get("likelihood") or {}
    return bool(cfg.model.get("noise_asymmetry", False)) or str(lik.get("variant", "")) == "ald"


def _build_increment_from_cfg(cfg: DictConfig, bundle: DataBundle, device: str) -> IncrementSDEModel:
    """Map the Hydra model config block to an IncrementSDEConfig (increment variants)."""
    inputs = _inputs_from_cfg(cfg, exogenous_default=True)
    model_cfg = IncrementSDEConfig(
        input_dim=encoder_input_dim(bundle.feature_dim, inputs),
        latent_dim=int(cfg.model.latent_dim),
        encoder_dims=tuple(cfg.model.encoder_dims),
        decoder_dims=tuple(cfg.model.decoder_dims),
        activation=str(cfg.model.activation),
        latent_reparameterize=bool(cfg.model.get("latent_reparameterize", False)),
        predict_k=bool(cfg.model.get("predict_k", True)),
        predict_temp_derivative=bool(cfg.model.get("predict_temp_derivative", True)),
        residual=bool(cfg.model.get("residual", False)),
        noise=bool(cfg.model.get("noise", True)),
        noise_zero_mean=bool(cfg.model.get("noise_zero_mean", True)),
        noise_dims=tuple(cfg.model.get("noise_dims", [4])),
        residual_dims=tuple(cfg.model.get("residual_dims", [16, 16])),
        k_activation=cfg.model.get("k_activation", "leaky_relu"),
        k_activation_slope=float(cfg.model.get("k_activation_slope", 0.01)),
        inputs=inputs,
        physics_k_source=str(cfg.model.get("physics_k_source", "predicted")),
        noise_student_dof=_needs_student_dof(cfg),
        noise_asymmetry=_needs_asymmetry(cfg),
        noise_mixture_components=int(cfg.model.get("noise_mixture_components", 1)),
        noise_physics_scale=bool(cfg.model.get("noise_physics_scale", False)),
        noise_physics_scale_b=float(cfg.model.get("noise_physics_scale_b", 0.25)),
        exact_reco_drift=bool(cfg.model.get("exact_reco_drift", False)),
        drift_tendency=str(cfg.model.get("drift_tendency", "learned_observed")),
        noise_state_space=bool(cfg.model.get("noise_state_space", False)),
        drift_clamp=cfg.model.get("drift_clamp", 1.0),
        tref=float(cfg.model.get("tref", 10.0)),
        t0=float(cfg.model.get("t0", 46.02)),
        dt=float(cfg.model.get("dt", 30.0)),
        device=device,
    )
    model = IncrementSDEModel(model_cfg).initialize()
    _maybe_set_norm_stats(model, inputs, bundle)
    return model


def _ordered_site_names(cfg: DictConfig) -> tuple[str, ...]:
    """Checkpoint-stable per-site layout from the data config (train + eval agree)."""
    site_paths = OmegaConf.to_container(cfg.data.site_paths, resolve=True)
    return ordered_site_names(
        site_paths,
        include_sites=cfg.data.get("include_sites") or None,
        exclude_sites=cfg.data.get("exclude_sites") or (),
    )


def _build_baseline_from_cfg(cfg: DictConfig, bundle: DataBundle, device: str):
    """Map the Hydra model block to a process-comparison baseline (ablation ladder).

    Baselines share the increment output contract but are separate model classes.
    The analytical baseline has no encoder; its single constant sigma is set later
    by `calibrate()` on the train loader (see main()).
    """
    variant = str(cfg.model.variant)
    if variant == "analytical_sde":
        per_site = bool(cfg.model.get("per_site", False))
        site_names = _ordered_site_names(cfg) if per_site else ()
        model_cfg = AnalyticalSDEConfig(
            per_site=per_site,
            site_names=site_names,
            noise_student_dof=_needs_student_dof(cfg),
            tref=float(cfg.model.get("tref", 10.0)),
            t0=float(cfg.model.get("t0", 46.02)),
            dt=float(cfg.model.get("dt", 30.0)),
            init_sigma=float(cfg.model.get("init_sigma", 1.0)),
            device=device,
        )
        return AnalyticalSDEModel(model_cfg).initialize()
    if variant in ("hetero_mlp", "hetero_mdn"):
        # No-physics baseline: encoder over the driver matrix only (X = drivers +
        # time + site), no GT physics inputs — input_dim is exactly feature_dim.
        default_k = 3 if variant == "hetero_mdn" else 1
        model_cfg = HeteroscedasticMLPConfig(
            input_dim=bundle.feature_dim,
            latent_dim=int(cfg.model.get("latent_dim", 32)),
            encoder_dims=tuple(cfg.model.get("encoder_dims", [16, 16])),
            decoder_dims=tuple(cfg.model.get("decoder_dims", [16, 16])),
            activation=str(cfg.model.get("activation", "relu")),
            mixture_components=int(cfg.model.get("mixture_components", default_k)),
            noise_student_dof=_needs_student_dof(cfg),
            noise_asymmetry=_needs_asymmetry(cfg),
            device=device,
        )
        return HeteroscedasticMLPModel(model_cfg).initialize()
    if variant == "neural_sde":
        # Data-driven neural SDE: encoder over the driver matrix only (no physics).
        model_cfg = NeuralSDEConfig(
            input_dim=bundle.feature_dim,
            latent_dim=int(cfg.model.get("latent_dim", 32)),
            encoder_dims=tuple(cfg.model.get("encoder_dims", [16, 16])),
            decoder_dims=tuple(cfg.model.get("decoder_dims", [16, 16])),
            activation=str(cfg.model.get("activation", "relu")),
            noise_student_dof=_needs_student_dof(cfg),
            noise_asymmetry=_needs_asymmetry(cfg),
            dt=float(cfg.model.get("dt", 30.0)),
            device=device,
        )
        return NeuralSDEModel(model_cfg).initialize()
    raise ValueError(f"No builder wired for baseline variant {variant!r}")


def _build_any_model(cfg: DictConfig, bundle: DataBundle, device: str):
    """Dispatch to the level (WienerNet), increment (IncrementSDE), or process
    baseline builder based on the variant name."""
    variant = str(cfg.model.variant)
    if is_baseline_variant(variant):
        return _build_baseline_from_cfg(cfg, bundle, device)
    if is_increment_variant(variant):
        return _build_increment_from_cfg(cfg, bundle, device)
    return _build_model_from_cfg(cfg, bundle, device)


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
        holdout_site=cfg.data.get("holdout_site"),
        train_subsample_frac=cfg.data.get("train_subsample_frac"),
        train_subsample_seed=int(cfg.data.get("train_subsample_seed", 0)),
        val_frac=cfg.data.get("val_frac"),
        val_split_seed=int(cfg.data.get("val_split_seed", 0)),
        shuffle_split=bool(cfg.data.shuffle_split),
        split_random_state=int(cfg.data.split_random_state),
        time_step_k=cfg.data.get("time_step_k"),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=bool(cfg.data.persistent_workers),
        save_scaler_path=scaler_path,
    )
    log.info("train batches=%d  test batches=%d  input_dim=%d",
             len(bundle.train_loader), len(bundle.test_loader), bundle.input_dim)

    # ---------- Model ----------
    model = _build_any_model(cfg, bundle, device=device)
    log.info("model built: %s  %d params  encoder_input_dim=%d",
             type(model).__name__, sum(p.numel() for p in model.parameters()), model.cfg.input_dim)

    # Analytical-SDE baseline: set its single constant diffusion level from the
    # spread of the training increment misfit (the textbook estimate). Training
    # under the Gaussian NLL then leaves it at this MLE, so it's stable even at
    # 0 epochs.
    if hasattr(model, "calibrate"):
        model.calibrate(bundle.train_loader)
        import torch as _torch  # local: avoid a top-level torch dependency in this fn
        log.info("calibrated constant diffusion: sigma=%.4f",
                 float(_torch.exp(model.log_sigma).detach().cpu()))

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
        # cosine needs a horizon; default it to the training length.
        t_max=int(cfg.training.scheduler.get("t_max") or cfg.training.num_epochs),
        eta_min=cfg.training.scheduler.get("eta_min"),
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

    # Noise prior for the mmd_noise term. 'gaussian' (default) matches noise to a
    # parametric N(mean, std); 'empirical' matches to the (optionally centred)
    # train residual pool, preserving the true skewed noise shape (step 2).
    noise_prior_kind = str(cfg.training.noise_prior.get("kind", "gaussian"))
    if loss_weights.get("mmd_noise", 0) <= 0:
        noise_prior_fn = None
    elif noise_prior_kind == "empirical":
        noise_prior_fn = make_empirical_noise_prior(
            bundle.noise_residuals,
            center=bool(cfg.training.noise_prior.get("center", True)),
        )
        log.info("noise prior: empirical (train residual pool, n=%d, center=%s)",
                 len(bundle.noise_residuals), cfg.training.noise_prior.get("center", True))
    elif noise_prior_kind == "gaussian":
        noise_prior_mean = cfg.training.noise_prior.get("mean")
        noise_prior_std = cfg.training.noise_prior.get("std")
        noise_prior_fn = make_gaussian_noise_prior(
            mean=bundle.noise_mu if noise_prior_mean is None else float(noise_prior_mean),
            std=bundle.noise_std if noise_prior_std is None else float(noise_prior_std),
        )
    else:
        raise ValueError(f"Unknown noise_prior.kind: {noise_prior_kind!r} (expected 'gaussian' or 'empirical')")

    # ---------- TensorBoard ----------
    tb_writer = SummaryWriter(run_dir / "tensorboard")

    # Optional per-target MSE normalisation (loss.normalize_anchors). Puts every
    # MSE term on an O(1) footing so the raw-scale E0 anchor stops dominating the
    # loss and starving mse_nee / the small physics terms of gradient.
    target_scales = bundle.target_scales if bool(cfg.loss.get("normalize_anchors", False)) else None
    if target_scales is not None:
        log.info("anchor normalisation ON; target scales: %s",
                 {k: round(v, 4) for k, v in target_scales.items()})

    # Likelihood (NLL) config — when set, replaces mse_nee/mse_drift/mmd_noise.
    likelihood_cfg = OmegaConf.to_container(cfg.loss.likelihood, resolve=True) if cfg.loss.get("likelihood") else None
    if likelihood_cfg is not None and float(likelihood_cfg.get("weight", 0)) > 0:
        log.info("likelihood loss ON: %s", likelihood_cfg)

    # ---------- Trainer ----------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_monitor=str(cfg.training.scheduler.get("monitor", "val")),
        loss_weights=loss_weights,
        mmd_loss_fn=mmd_fn,
        noise_prior_fn=noise_prior_fn,
        device=device,
        writer=tb_writer,
        clip_grad_norm=cfg.training.get("clip_grad_norm"),
        nan_policy=str(cfg.training.nan_policy),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        target_scales=target_scales,
        likelihood=likelihood_cfg,
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

        # Monitor a TRAIN-SITE validation split when `data.val_frac` is set, so that
        # best.pth selection and the LR scheduler never see the held-out test site.
        # With val_frac unset this falls back to the historical behaviour (test loader).
        _clean_val = getattr(bundle, "val_loader", None) is not None
        _val_loader = bundle.val_loader if _clean_val else bundle.test_loader
        log.info("val monitor = %s", "train-site split (leakage-free)"
                 if _clean_val else "TEST loader (legacy; leaks into best.pth)")
        if not _clean_val:
            log.warning("LEAKAGE: best.pth is selected on the HELD-OUT TEST loader because "
                        "data.val_frac is unset -> report last.pth, or set data.val_frac.")
            # Only leaks when the plateau scheduler actually reads the (test) val monitor;
            # scheduler.monitor='train' keeps the LR schedule free of held-out data.
            if (str(cfg.training.scheduler.name).lower() == "reduce_on_plateau"
                    and str(cfg.training.scheduler.get("monitor", "val")).lower() != "train"):
                log.warning("LEAKAGE: scheduler 'reduce_on_plateau' steps on that same test-set "
                            "loss, so the LR schedule is driven by the test site. Set "
                            "training.scheduler.monitor=train (or data.val_frac) instead.")
        history = trainer.fit(
            train_loader=bundle.train_loader,
            val_loader=_val_loader,
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
