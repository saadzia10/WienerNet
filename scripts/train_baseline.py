"""Hydra entrypoint for non-torch baselines (RF, XGB).

Mirrors scripts/train.py in structure but skips the optimiser/scheduler/loss
machinery — baselines just `.fit(X, y)` then `.predict(X)`.

Usage:
    python scripts/train_baseline.py                       # default: rf, seed 42
    python scripts/train_baseline.py model=xgb seed=88
    python scripts/train_baseline.py -m model=rf,xgb seed=0,1,2,41,88,128,256
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wienernet.baselines import build_baseline, extract_flat_features
from wienernet.data import build_dataloaders
from wienernet.evaluation import evaluate_at_resolutions
from wienernet.utils import set_seed_globally, setup_logging

try:
    import mlflow
except ImportError:
    mlflow = None


def _flatten_for_mlflow(cfg_dict: dict, parent_key: str = "", separator: str = ".") -> dict:
    flat: dict[str, str] = {}
    for key, value in cfg_dict.items():
        full = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_for_mlflow(value, full, separator))
        elif isinstance(value, list):
            flat[full] = str(value)
        else:
            flat[full] = "null" if value is None else str(value)
    return flat


def _model_kwargs_from_cfg(cfg: DictConfig) -> dict:
    """Extract the baseline-specific hyperparameters from the model config block."""
    skip = {"variant", "kind"}
    out: dict[str, object] = {}
    for key, value in OmegaConf.to_container(cfg.model, resolve=True).items():
        if key in skip:
            continue
        out[key] = value
    # Inject the shared seed into the model's random_state
    out["random_state"] = int(cfg.seed)
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig) -> float:
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).resolve()
    log = setup_logging(run_dir)
    log.info("run dir: %s", run_dir)

    set_seed_globally(int(cfg.seed), deterministic=False)
    log.info("variant=%s seed=%d", cfg.model.variant, cfg.seed)

    # Persist resolved config
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2, default=str))

    # ---------- Data ----------
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
        time_step_k=cfg.data.get("time_step_k"),
        batch_size=int(cfg.data.batch_size),
        save_scaler_path=scaler_path,
    )

    X_train, y_train = extract_flat_features(bundle, split="train")
    X_test, y_test = extract_flat_features(bundle, split="test")
    log.info("flat feature shape: train=%s test=%s", X_train.shape, X_test.shape)

    # ---------- Model ----------
    model_kwargs = _model_kwargs_from_cfg(cfg)
    baseline = build_baseline(str(cfg.model.variant), **model_kwargs)

    # ---------- MLflow context ----------
    mlflow_active = bool(cfg.get("mlflow", {}).get("enabled", False)) and mlflow is not None
    if mlflow_active:
        tracking_uri = cfg.mlflow.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(str(cfg.mlflow.experiment_name))

    ctx = mlflow.start_run(run_name=str(cfg.run_name)) if mlflow_active else _NullContext()
    with ctx:
        if mlflow_active:
            mlflow.log_params(_flatten_for_mlflow(resolved))
            mlflow.set_tag("variant", str(cfg.model.variant))
            mlflow.set_tag("seed", str(cfg.seed))

        # ---------- Fit ----------
        log.info("fitting baseline...")
        baseline.fit(X_train, y_train)
        model_path = run_dir / "checkpoints" / "model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        baseline.save(model_path)
        log.info("saved %s", model_path)

        # ---------- Predict + evaluate ----------
        y_pred = baseline.predict(X_test)
        gt = {"nee": y_test.astype(np.float32)}
        preds = {"nee": y_pred.astype(np.float32)}
        metrics = evaluate_at_resolutions(
            bundle.test_df, gt, preds, targets=["nee"],
            include_mmd=True, mmd_subsample=2000,
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        log.info("raw NEE metrics: %s", metrics["raw"]["nee"])

        # Save predictions for downstream notebook plotting
        import pandas as pd
        pred_df = bundle.test_df.reset_index(drop=True).copy()
        pred_df["gt_nee"] = y_test
        pred_df["pred_nee"] = y_pred
        (run_dir / "predictions.parquet").write_bytes(b"")  # ensure path exists
        pred_df.to_parquet(run_dir / "predictions.parquet")

        # Feature importances (free with these models)
        try:
            imp = baseline.feature_importances_
            feature_names = bundle.feature_columns + ["bNEE", "E0", "rb"]
            importance_df = pd.DataFrame({"feature": feature_names, "importance": imp})
            importance_df.to_csv(run_dir / "feature_importance.csv", index=False)
            log.info("top 5 features:\n%s",
                     importance_df.sort_values("importance", ascending=False).head().to_string(index=False))
        except (AttributeError, RuntimeError):
            pass

        if mlflow_active:
            for resolution, per_target in metrics.items():
                for target, mdict in per_target.items():
                    for metric, value in mdict.items():
                        if metric == "n" or value is None:
                            continue
                        mlflow.log_metric(f"{resolution}/{target}/{metric}", float(value))
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(run_dir / "config.json"))
            mlflow.log_artifact(str(run_dir / "metrics.json"))

    # Hydra multirun optimisation target
    return float(metrics["raw"]["nee"]["mae"])


class _NullContext:
    def __enter__(self): return None
    def __exit__(self, *args): return False


if __name__ == "__main__":
    main()
