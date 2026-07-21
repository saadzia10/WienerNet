"""Evaluate one or more trained runs.

Usage:
    # Evaluate one run
    python scripts/evaluate.py --run outputs/2026-05-25_22-30-00_paper_main_seed42

    # Evaluate multiple runs and produce a cross-seed summary
    python scripts/evaluate.py --run outputs/2026-05-25_22-30-00_*/

    # Save metrics, predictions, and per-site breakdowns to a single dir
    python scripts/evaluate.py --run outputs/X/ --out outputs/X/metrics

For each run, loads the best checkpoint, re-runs inference on the test loader
(rebuilt from the saved config), computes metrics at every requested
resolution, and writes:
    metrics/long.csv     — one row per (run, resolution, target, metric)
    metrics/summary.csv  — median + MAD across runs in the same group
    metrics/per_site.csv — long-format per-site metrics
    predictions.parquet  — gt + preds + test_df columns for downstream plotting
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wienernet.data import build_dataloaders, ordered_site_names
from wienernet.data.features import physics_nee_numpy
from wienernet.evaluation import (
    aggregated_scores,
    crps,
    crps_ensemble,
    drift_check,
    emit_calibration_plots,
    emit_manuscript_plots,
    ensemble_scores,
    evaluate_at_resolutions,
    measurement_noise_floor,
    metrics_to_long_dataframe,
    noise_check,
    predictive_cdf,
    probabilistic_scores,
    save_metrics_tables,
    standardized_residual_autocorr,
    stratified_scores,
    variance_vs_scale,
)
from wienernet.evaluation.metrics import evaluate_predictions
from wienernet.evaluation.rollout import assign_night_ids, nightly_rollout
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
from wienernet.training import Trainer
from wienernet.utils import get_logger, load_model_weights, setup_logging

log = get_logger("scripts.evaluate")


def _load_resolved_config(run_dir: Path) -> DictConfig:
    """Prefer Hydra's .hydra/config.yaml; fall back to our config.json."""
    hydra_path = run_dir / ".hydra" / "config.yaml"
    if hydra_path.exists():
        return OmegaConf.load(hydra_path)
    json_path = run_dir / "config.json"
    if json_path.exists():
        return OmegaConf.create(json.loads(json_path.read_text()))
    raise FileNotFoundError(f"No resolved config found in {run_dir}")


def _rebuild_model(cfg: DictConfig, feature_dim: int, device: str) -> WienerNetModel:
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
    # Reconstruct the encoder input layout from the saved config so the first
    # layer width (and the scale buffers, if any) match the checkpoint. The
    # norm buffers are restored by load_model_weights, so no stats injection here.
    inputs_block = cfg.model.get("inputs", {}) or {}
    inputs = InputsConfig(
        include_bnee=bool(inputs_block.get("include_bnee", True)),
        include_k=bool(inputs_block.get("include_k", True)),
        include_dtemp=bool(inputs_block.get("include_dtemp", False)),
        scale_extra_inputs=bool(inputs_block.get("scale_extra_inputs", False)),
    )
    model_cfg = WienerNetConfig(
        input_dim=encoder_input_dim(feature_dim, inputs),
        latent_dim=int(cfg.model.latent_dim),
        encoder_dims=tuple(cfg.model.encoder_dims),
        decoder_dims=tuple(cfg.model.decoder_dims),
        activation=str(cfg.model.activation),
        latent_reparameterize=bool(cfg.model.latent_reparameterize),
        predict_drift=bool(cfg.model.predict_drift),
        heads=heads,
        inputs=inputs,
        physics_k_source=str(cfg.model.get("physics_k_source", "predicted")),
        noise_student_dof=_student_dof(cfg),
        tref=float(cfg.model.tref),
        t0=float(cfg.model.t0),
        dt=float(cfg.model.get("dt", 30.0)),
        device=device,
    )
    return WienerNetModel(model_cfg)


def _student_dof(cfg: DictConfig) -> bool:
    """Re-derive whether the model carries a log_nu param (student_t NLL variant).

    config.json stores the Hydra cfg, not the built model config, so the flag
    (set at build time from loss.likelihood.variant) has to be reconstructed here
    or strict checkpoint loading fails on the `log_nu` key.
    """
    lik = cfg.loss.get("likelihood") or {} if "loss" in cfg else {}
    return bool(cfg.model.get("noise_student_dof", False)) or str(lik.get("variant", "")) == "student_t"


def _asymmetry(cfg: DictConfig) -> bool:
    """Re-derive whether the model carries a log_kappa param (ALD NLL variant)."""
    lik = cfg.loss.get("likelihood") or {} if "loss" in cfg else {}
    return bool(cfg.model.get("noise_asymmetry", False)) or str(lik.get("variant", "")) == "ald"


def _increment_inputs(cfg: DictConfig) -> InputsConfig:
    inputs_block = cfg.model.get("inputs", {}) or {}
    return InputsConfig(
        include_bnee=bool(inputs_block.get("include_bnee", False)),
        include_k=bool(inputs_block.get("include_k", False)),
        include_dtemp=bool(inputs_block.get("include_dtemp", False)),
        scale_extra_inputs=bool(inputs_block.get("scale_extra_inputs", False)),
    )


def _rebuild_increment_model(cfg: DictConfig, feature_dim: int, device: str) -> IncrementSDEModel:
    inputs = _increment_inputs(cfg)
    model_cfg = IncrementSDEConfig(
        input_dim=encoder_input_dim(feature_dim, inputs),
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
        noise_student_dof=_student_dof(cfg),
        noise_asymmetry=_asymmetry(cfg),
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
    return IncrementSDEModel(model_cfg)


def _rebuild_baseline_model(cfg: DictConfig, feature_dim: int, device: str):
    """Rebuild a process-comparison baseline from its saved config (weights —
    e.g. the analytical baseline's calibrated log_sigma — are loaded afterwards)."""
    variant = str(cfg.model.variant)
    if variant == "analytical_sde":
        per_site = bool(cfg.model.get("per_site", False))
        site_names = ordered_site_names(
            OmegaConf.to_container(cfg.data.site_paths, resolve=True),
            include_sites=cfg.data.get("include_sites") or None,
            exclude_sites=cfg.data.get("exclude_sites") or (),
        ) if per_site else ()
        model_cfg = AnalyticalSDEConfig(
            per_site=per_site,
            site_names=site_names,
            noise_student_dof=_student_dof(cfg),
            tref=float(cfg.model.get("tref", 10.0)),
            t0=float(cfg.model.get("t0", 46.02)),
            dt=float(cfg.model.get("dt", 30.0)),
            init_sigma=float(cfg.model.get("init_sigma", 1.0)),
            device=device,
        )
        return AnalyticalSDEModel(model_cfg)
    if variant in ("hetero_mlp", "hetero_mdn"):
        default_k = 3 if variant == "hetero_mdn" else 1
        model_cfg = HeteroscedasticMLPConfig(
            input_dim=feature_dim,
            latent_dim=int(cfg.model.get("latent_dim", 32)),
            encoder_dims=tuple(cfg.model.get("encoder_dims", [16, 16])),
            decoder_dims=tuple(cfg.model.get("decoder_dims", [16, 16])),
            activation=str(cfg.model.get("activation", "relu")),
            mixture_components=int(cfg.model.get("mixture_components", default_k)),
            noise_student_dof=_student_dof(cfg),
            noise_asymmetry=_asymmetry(cfg),
            device=device,
        )
        return HeteroscedasticMLPModel(model_cfg)
    if variant == "neural_sde":
        model_cfg = NeuralSDEConfig(
            input_dim=feature_dim,
            latent_dim=int(cfg.model.get("latent_dim", 32)),
            encoder_dims=tuple(cfg.model.get("encoder_dims", [16, 16])),
            decoder_dims=tuple(cfg.model.get("decoder_dims", [16, 16])),
            activation=str(cfg.model.get("activation", "relu")),
            noise_student_dof=_student_dof(cfg),
            noise_asymmetry=_asymmetry(cfg),
            dt=float(cfg.model.get("dt", 30.0)),
            device=device,
        )
        return NeuralSDEModel(model_cfg)
    raise ValueError(f"No rebuild wired for baseline variant {variant!r}")


def _rebuild_any_model(cfg: DictConfig, feature_dim: int, device: str):
    variant = str(cfg.model.variant)
    if is_baseline_variant(variant):
        return _rebuild_baseline_model(cfg, feature_dim, device)
    if is_increment_variant(variant):
        return _rebuild_increment_model(cfg, feature_dim, device)
    return _rebuild_model(cfg, feature_dim, device)


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


def _predictive_family(model) -> tuple[str, float | None]:
    """Recover the one-step predictive family from the trained model.

    A non-None `log_nu` parameter (set for the student_t NLL variant) implies a
    Student-t predictive law with nu = softplus(log_nu)+1; otherwise Gaussian
    (covers the plain-Gaussian and beta-NLL variants, whose family is Gaussian).
    """
    import math
    from wienernet.losses.likelihood import NU_FLOOR
    log_nu = getattr(model, "log_nu", None)
    if log_nu is not None:
        nu = math.log1p(math.exp(float(log_nu.detach().cpu()))) + NU_FLOOR  # softplus + floor
        return "student_t", nu
    return "gaussian", None


def _hour_of_night_bin(dt: pd.Series) -> np.ndarray:
    """Coarse time-of-night label for stratified calibration."""
    hr = pd.to_datetime(dt).dt.hour.to_numpy()
    # nighttime data wraps midnight; bin into evening / midnight / pre-dawn / dawn
    edges = [(18, 21, "18-21"), (21, 24, "21-24"), (0, 3, "00-03"), (3, 6, "03-06")]
    out = np.full(len(hr), "other", dtype=object)
    for lo, hi, lab in edges:
        out[(hr >= lo) & (hr < hi)] = lab
    return out


def write_distribution_reports(
    model, gt: dict, preds: dict, test_df: pd.DataFrame, out_dir: Path,
    *, ensemble_samples: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Compute and persist the probabilistic + process-consistency reports, each
    at GLOBAL (pooled) and PER-SITE resolution.

    Applies to every model that emits a one-step predictive distribution
    (nee_mean + nee_std); deterministic models still get CRPS (= MAE) so they sit
    on the same axis. When `ensemble_samples` (N, M) is given (repeated stochastic
    forward passes), an ensemble-based score set is added — the honest axis for the
    sampling / MMD variants whose parametric `nee_std` under-represents spread; for
    the NLL variants it cross-checks the exact parametric score.

    Writes `probabilistic.json` (keys: global, per_site, stratified) and
    `process_consistency.json` (keys: global, per_site), and returns
    (probabilistic_report, per_sample_scores). Per-sample CRPS/PIT prefer the
    ensemble when present (fair for sampling models); these columns enable paired
    Diebold-Mariano across runs downstream.
    """
    obs = np.asarray(gt["nee"], dtype=float).ravel()
    mean = np.asarray(preds.get("nee_mean", preds.get("nee")), dtype=float).ravel()
    scale_raw = preds.get("nee_std")
    scale = np.asarray(scale_raw, dtype=float).ravel() if scale_raw is not None else None
    family, nu = _predictive_family(model) if scale is not None else ("point", None)
    fam = "gaussian" if family == "point" else family

    flux = None
    if {"E0", "rb"} <= set(gt) and "Ta" in test_df.columns:
        flux = physics_nee_numpy(np.asarray(gt["E0"]), np.asarray(gt["rb"]),
                                 test_df["Ta"].to_numpy())
    have_proc = scale is not None and "bnee" in gt and {"DateTime", "site"} <= set(test_df.columns)
    bnee = np.asarray(gt["bnee"], dtype=float).ravel() if "bnee" in gt else None
    times = pd.to_datetime(test_df["DateTime"]).to_numpy() if "DateTime" in test_df.columns else None
    sites_arr = test_df["site"].to_numpy() if "site" in test_df.columns else None
    dnee = np.asarray(preds["dnee"], dtype=float).ravel() if "dnee" in preds else None

    def _prob_block(idx: np.ndarray) -> dict:
        o = obs[idx]; m = mean[idx]; s = scale[idx] if scale is not None else None
        block = {"probabilistic": probabilistic_scores(o, m, s, family=fam, nu=nu)}
        if ensemble_samples is not None:
            block["ensemble"] = ensemble_scores(o, ensemble_samples[idx])
        if flux is not None:
            block["measurement_noise_floor"] = measurement_noise_floor(o - flux[idx], flux[idx])
        return block

    def _proc_block(idx: np.ndarray) -> dict | None:
        if not have_proc:
            return None
        o = obs[idx]; m = mean[idx]; s = scale[idx]; bn = bnee[idx]
        t = times[idx]; st = sites_arr[idx]
        obs_incr = o - bn; det_incr = m - bn; emp = obs_incr - det_incr
        night = assign_night_ids(t, st)
        acf_df, acf_sum = standardized_residual_autocorr(t, st, o, m, s)
        block = {
            "residual_autocorrelation": {"summary": acf_sum, "acf": acf_df.to_dict("records")},
            "drift_check": drift_check(obs_incr, det_incr, night),
            "noise_check": noise_check(emp, s, family=fam, nu=nu),
        }
        if dnee is not None:
            gen = nightly_rollout(t, st, bn, dnee[idx])
            block["variance_vs_scale"] = variance_vs_scale(t, st, bn, gen).to_dict("records")
        return block

    all_idx = np.ones(obs.shape[0], dtype=bool)
    site_masks: dict[str, np.ndarray] = {}
    if sites_arr is not None:
        for site in pd.unique(sites_arr):
            site_masks[str(site)] = (sites_arr == site)

    # --- probabilistic report: global + per-site (+ season/time-of-night strata) ---
    prob_report: dict = {"family": family, "nu": nu, "n": int(obs.shape[0]),
                         "global": _prob_block(all_idx),
                         "per_site": {s: _prob_block(m) for s, m in site_masks.items()}}
    if scale is not None:
        strata: dict[str, np.ndarray] = {}
        if "season" in test_df.columns:
            strata["season"] = test_df["season"].to_numpy()
        if "DateTime" in test_df.columns:
            strata["time_of_night"] = _hour_of_night_bin(test_df["DateTime"])
        if strata:
            prob_report["stratified"] = stratified_scores(obs, mean, scale, strata,
                                                          family=fam, nu=nu)
    (out_dir / "probabilistic.json").write_text(json.dumps(prob_report, indent=2, default=str))

    # --- process-consistency report: global + per-site ---
    if have_proc:
        proc_report = {"family": family, "nu": nu,
                       "global": _proc_block(all_idx),
                       "per_site": {s: _proc_block(m) for s, m in site_masks.items()}}
        (out_dir / "process_consistency.json").write_text(
            json.dumps(proc_report, indent=2, default=str))

    # --- per-sample scores for column attachment / cross-run DM (full length) ---
    if ensemble_samples is not None:
        crps_ps = crps_ensemble(obs, ensemble_samples)
        below = np.sum(np.asarray(ensemble_samples) < obs[:, None], axis=1)
        pit_ps = (below + 0.5) / (ensemble_samples.shape[1] + 1)
    else:
        crps_ps = crps(obs, mean, scale, family=fam, nu=nu)
        pit_ps = (predictive_cdf(obs, mean, scale, family=fam, nu=nu)
                  if scale is not None else np.full(obs.shape, np.nan))
    per_sample = {"crps": crps_ps, "pit": pit_ps}

    ov = prob_report["global"]["probabilistic"]
    log.info("probabilistic: family=%s sites=%d CRPS=%.4f NLL=%s cover90=%s",
             family, len(site_masks), ov["crps"],
             f"{ov['nll']:.4f}" if ov.get("nll") is not None else "n/a",
             f"{ov['coverage']['0.90']['coverage']:.3f}" if ov.get("coverage") else "n/a")
    return prob_report, per_sample


def evaluate_run(run_dir: Path, *, out_dir: Path | None, device: str, make_plots: bool = False,
                 checkpoint: str = "best") -> dict:
    """Load a run, run inference, compute metrics, save predictions+metrics.

    Returns the long-format metrics dict for this run (used for cross-run aggregation).
    When make_plots is True, also writes the manuscript decomposition plots
    (per-site temporal + daily/weekly breakdown + component histograms) under
    <out_dir>/plots/.
    """
    cfg = _load_resolved_config(run_dir)
    out_dir = out_dir or (run_dir / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("evaluating %s", run_dir)

    # Rebuild dataloaders from the saved data config
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
        shuffle_split=bool(cfg.data.shuffle_split),
        split_random_state=int(cfg.data.split_random_state),
        time_step_k=cfg.data.get("time_step_k"),
        batch_size=int(cfg.data.batch_size),
        save_scaler_path=None,
    )

    # Rebuild model and load the requested checkpoint. `last` is the leakage-free
    # choice for held-out-site / generalisation runs where `best` (min val loss)
    # would select on the test set.
    model = _rebuild_any_model(cfg, feature_dim=bundle.feature_dim, device=device)
    ckpt_path = run_dir / "checkpoints" / f"{checkpoint}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No {checkpoint}.pth in {ckpt_path.parent}")
    load_model_weights(ckpt_path, model, map_location=device)
    log.info("loaded %s (%d params)", ckpt_path, sum(p.numel() for p in model.parameters()))

    # Predict via the Trainer's predict helper (no optimizer required)
    trainer = Trainer(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.0), device=device)
    gt, preds = trainer.predict(bundle.test_loader)

    # Stash predictions for downstream plotting
    test_df = bundle.test_df.reset_index(drop=True)
    pred_frame = pd.DataFrame({f"gt_{k}": v for k, v in gt.items()})
    for k, v in preds.items():
        # 1-D arrays only; skip latent (z) which is multi-dim
        if v.ndim == 1:
            pred_frame[f"pred_{k}"] = v

    # Probabilistic (headline) + process-consistency reports. Distributional and
    # deterministic models alike are scored on CRPS; per-sample CRPS/PIT are
    # attached to the frame so paired Diebold-Mariano can be run across runs.
    # Stochastic models (noise head or reparameterised latent) also get an
    # ensemble-based score from repeated forward passes — the honest predictive
    # distribution for the sampling / MMD variants.
    ensemble_samples = None
    try:
        is_stochastic = ("noise" in preds) or bool(
            getattr(getattr(model, "cfg", None), "latent_reparameterize", False))
        if is_stochastic:
            _, ensemble_samples = trainer.predict_ensemble(bundle.test_loader, n_samples=100)
    except Exception as exc:
        log.warning("ensemble prediction failed: %s", exc)
        ensemble_samples = None
    try:
        _, per_sample = write_distribution_reports(
            model, gt, preds, test_df, out_dir, ensemble_samples=ensemble_samples)
        pred_frame["pred_crps"] = per_sample["crps"]
        pred_frame["pred_pit"] = per_sample["pit"]
    except Exception as exc:  # never let scoring break the eval run
        log.warning("distribution reports failed: %s", exc)

    # Aggregated-scale (weekly/monthly) scoring — the scale where random noise
    # cancels ~1/sqrt(N) and the physics signal is meant to show. Uses the same
    # ensemble; persisted to aggregated.json for later cross-run analysis.
    try:
        if "DateTime" in test_df.columns:
            fam_a, nu_a = _predictive_family(model) if "nee_std" in preds else ("point", None)
            agg = aggregated_scores(
                pd.to_datetime(test_df["DateTime"]).to_numpy(),
                test_df["site"].to_numpy() if "site" in test_df.columns else None,
                np.asarray(gt["nee"], float),
                ensemble=ensemble_samples,
                mean=np.asarray(preds["nee_mean"], float) if "nee_mean" in preds else None,
                scale=np.asarray(preds["nee_std"], float) if "nee_std" in preds else None,
                resolutions=("weekly", "monthly"),
                statistic="mean",
            )
            (out_dir / "aggregated.json").write_text(json.dumps(agg, indent=2, default=str))
            for res in ("weekly", "monthly"):
                b = agg.get(res, {})
                pt = b.get("point", {}); en = b.get("ensemble", {})
                log.info("aggregated %-7s n=%s  point RMSE=%s bias=%s R2=%s | ens CRPS=%s cov90=%s",
                         res, b.get("n_windows"),
                         f"{pt.get('rmse'):.3f}" if pt.get("rmse") is not None else "n/a",
                         f"{pt.get('bias'):.3f}" if pt.get("bias") is not None else "n/a",
                         f"{pt.get('r2'):.3f}" if pt.get("r2") is not None else "n/a",
                         f"{en.get('crps'):.3f}" if en.get("crps") is not None else "n/a",
                         f"{en.get('coverage',{}).get('0.90',{}).get('coverage'):.3f}"
                         if en.get("coverage") else "n/a")
    except Exception as exc:
        log.warning("aggregated scoring failed: %s", exc)

    combined = pd.concat([test_df.reset_index(drop=True), pred_frame], axis=1)
    predictions_path = out_dir / "predictions.parquet"
    combined.to_parquet(predictions_path)
    log.info("wrote %s (%d rows)", predictions_path, len(combined))

    # Compute metrics at every resolution. `nee_mean` (the deterministic point
    # estimate) is scored alongside `nee` (the stochastic sample) — for a
    # calibrated stochastic model the point accuracy belongs on the mean, while
    # the sample carries the honest aleatoric spread.
    available_targets = [t for t in ("nee", "nee_mean", "bnee", "E0", "rb", "dtemp", "f")
                         if t in preds and t in gt]
    metrics = evaluate_at_resolutions(
        test_df, gt, preds,
        targets=available_targets,
        include_mmd=True,
        mmd_subsample=2000,
    )

    # Calibration / coverage — needs the likelihood outputs (nee_mean + nee_std).
    # z = (NEE_{t+1} - mean) / std; a well-calibrated model has |z|<=1.96 ~95% of
    # the time and std(residual)/mean(std) ~ 1. (Gaussian bands; for the heavy-
    # tailed student_t the coverage under-reads a little — read calib_std_ratio.)
    if "nee_mean" in preds and "nee_std" in preds:
        tgt = np.asarray(gt["nee"]).ravel()
        mean = np.asarray(preds["nee_mean"]).ravel()
        std = np.asarray(preds["nee_std"]).ravel()
        m = np.isfinite(tgt) & np.isfinite(mean) & np.isfinite(std) & (std > 0)
        resid = tgt[m] - mean[m]
        z = resid / std[m]
        calib = {
            "n": int(m.sum()),
            "cover50": float(np.mean(np.abs(z) <= 0.6745)),
            "cover95": float(np.mean(np.abs(z) <= 1.96)),
            "calib_std_ratio": float(np.std(resid) / np.mean(std[m])),
            "mean_pred_std": float(np.mean(std[m])),
            "emp_resid_std": float(np.std(resid)),
        }
        # Written to a sidecar file (NOT into `metrics`, whose 3-level shape the
        # reporter/aggregator rely on).
        (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
        log.info("calibration: cover95=%.3f  calib_std_ratio=%.3f  (pred_std %.3f vs emp %.3f)",
                 calib["cover95"], calib["calib_std_ratio"], calib["mean_pred_std"], calib["emp_resid_std"])

    # Per-site metrics on the raw resolution
    per_site_rows: list[dict] = []
    if "site" in test_df.columns:
        for site, idx in test_df.groupby("site").groups.items():
            mask = np.zeros(len(test_df), dtype=bool)
            mask[idx] = True
            site_metrics = evaluate_predictions(
                gt, preds, targets=available_targets, mask=mask, include_mmd=True, mmd_subsample=2000,
            )
            for target, mdict in site_metrics.items():
                for metric, value in mdict.items():
                    if metric == "n" or value is None:
                        continue
                    per_site_rows.append({
                        "site": site, "target": target, "metric": metric,
                        "value": float(value), "n": int(mdict["n"]),
                    })
        per_site_df = pd.DataFrame(per_site_rows)
        per_site_df.to_csv(out_dir / "per_site.csv", index=False)
        log.info("wrote %s (%d rows)", out_dir / "per_site.csv", len(per_site_df))

    model_name = str(cfg.model.variant)
    plot_dir = out_dir / "plots"
    # Metric-illustration figures (PIT, coverage, diffusion scaling, ACF, CRPS by
    # site, aggregated skill) are emitted for EVERY run, driven from the JSON
    # reports so they're cheap + reproducible. The heavier per-site decomposition
    # plots stay behind --plots.
    try:
        prob_json = out_dir / "probabilistic.json"
        if prob_json.exists():
            prob_rep = json.loads(prob_json.read_text())
            proc_json = out_dir / "process_consistency.json"
            proc_rep = json.loads(proc_json.read_text()) if proc_json.exists() else None
            agg_json = out_dir / "aggregated.json"
            agg_rep = json.loads(agg_json.read_text()) if agg_json.exists() else None
            cal = emit_calibration_plots(prob_rep, proc_rep, plot_dir, model_name=model_name,
                                         aggregated_report=agg_rep)
            log.info("wrote %d metric plots to %s", len(cal), plot_dir)
    except Exception as exc:
        log.warning("metric plots failed: %s", exc)

    if make_plots:
        try:
            written = emit_manuscript_plots(gt, preds, test_df, model_name=model_name, out_dir=plot_dir)
            log.info("wrote %d manuscript decomposition plots to %s", len(written), plot_dir)
        except Exception as exc:
            log.warning("manuscript plots failed: %s", exc)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more WienerNet training runs")
    parser.add_argument("--run", nargs="+", required=True,
                        help="One or more run directories (containing config.json + checkpoints/best.pth)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the cross-run summary tables. Default: alongside the first run.")
    parser.add_argument("--device", default="cuda", help="'cuda' | 'cpu' | 'mps'")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best",
                        help="which checkpoint to score; use 'last' for held-out-site / "
                             "generalisation runs to avoid selecting on the test set")
    parser.add_argument("--plots", action="store_true",
                        help="also write manuscript decomposition plots (per-site temporal + "
                             "daily/weekly breakdown + component histograms) under <run>/metrics/plots/")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    setup_logging(None, console_level=getattr(logging, args.log_level.upper()))
    device = _resolve_device(args.device)

    # Expand globs / accept directories
    run_dirs: list[Path] = []
    for spec in args.run:
        p = Path(spec)
        if p.is_dir():
            run_dirs.append(p.resolve())
        else:
            run_dirs.extend(sorted(Path().glob(spec)))
    if not run_dirs:
        raise SystemExit(f"No runs found from {args.run}")

    out_dir = args.out or (run_dirs[0].parent / "evaluation_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each run, collect metrics + metadata for the cross-run summary
    per_run_metrics: dict = {}
    per_run_meta: dict = {}
    for run_dir in run_dirs:
        try:
            metrics = evaluate_run(run_dir, out_dir=None, device=device, make_plots=args.plots,
                                   checkpoint=args.checkpoint)
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", run_dir, exc)
            continue
        cfg = _load_resolved_config(run_dir)
        run_id = run_dir.name
        per_run_metrics[run_id] = metrics
        per_run_meta[run_id] = {
            "variant": str(cfg.model.variant),
            "seed": int(cfg.seed),
            "run_dir": str(run_dir),
        }

    if not per_run_metrics:
        raise SystemExit("No runs were successfully evaluated")

    # Cross-run flat table + summary
    long_df = metrics_to_long_dataframe(per_run_metrics, extra_columns=per_run_meta)
    paths = save_metrics_tables(long_df, out_dir=out_dir)
    log.info("\n=== summary saved to %s ===", out_dir)
    for kind, path in paths.items():
        log.info("  %-7s %s", kind + ":", path)

    # Print a compact preview to stdout
    raw = long_df[long_df["resolution"] == "raw"]
    if not raw.empty:
        print("\nMetrics (raw resolution, median across runs):")
        pivot = raw.pivot_table(
            index=["variant", "target"], columns="metric", values="value", aggfunc="median",
        )
        print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
