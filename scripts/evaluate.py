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

from wienernet.data import build_dataloaders
from wienernet.evaluation import (
    evaluate_at_resolutions,
    metrics_to_long_dataframe,
    save_metrics_tables,
)
from wienernet.evaluation.metrics import evaluate_predictions
from wienernet.models import HeadsConfig, WienerNetConfig, WienerNetModel
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


def _rebuild_model(cfg: DictConfig, input_dim: int, device: str) -> WienerNetModel:
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
    return WienerNetModel(model_cfg)


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


def evaluate_run(run_dir: Path, *, out_dir: Path | None, device: str) -> dict:
    """Load a run, run inference, compute metrics, save predictions+metrics.

    Returns the long-format metrics dict for this run (used for cross-run aggregation).
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
        shuffle_split=bool(cfg.data.shuffle_split),
        split_random_state=int(cfg.data.split_random_state),
        batch_size=int(cfg.data.batch_size),
        save_scaler_path=None,
    )

    # Rebuild model and load best.pth weights
    model = _rebuild_model(cfg, input_dim=bundle.input_dim, device=device)
    ckpt_path = run_dir / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best.pth in {ckpt_path.parent}")
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
    combined = pd.concat([test_df.reset_index(drop=True), pred_frame], axis=1)
    predictions_path = out_dir / "predictions.parquet"
    combined.to_parquet(predictions_path)
    log.info("wrote %s (%d rows)", predictions_path, len(combined))

    # Compute metrics at every resolution
    available_targets = [t for t in ("nee", "bnee", "E0", "rb", "dtemp", "f") if t in preds and t in gt]
    metrics = evaluate_at_resolutions(
        test_df, gt, preds,
        targets=available_targets,
        include_mmd=True,
        mmd_subsample=2000,
    )

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

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more WienerNet training runs")
    parser.add_argument("--run", nargs="+", required=True,
                        help="One or more run directories (containing config.json + checkpoints/best.pth)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write the cross-run summary tables. Default: alongside the first run.")
    parser.add_argument("--device", default="cuda", help="'cuda' | 'cpu' | 'mps'")
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
            metrics = evaluate_run(run_dir, out_dir=None, device=device)
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
