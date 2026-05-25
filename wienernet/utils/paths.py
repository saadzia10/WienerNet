"""Run-directory helpers.

Standardises where artefacts of a training run live so the rest of the
codebase doesn't sprinkle Path expressions.

Layout for one run:
    outputs/2026-05-25_18-32-04_paper_main/
        config.yaml               # full resolved config (Hydra writes this)
        run.log                   # logger output (logging.setup_logging)
        tensorboard/              # TensorBoard event files
        checkpoints/
            best.pth
            last.pth
        predictions/
            test.parquet
        metrics/
            summary.json
            per_site.csv
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUTS_ENV = "WIENERNET_OUTPUTS_DIR"
DEFAULT_OUTPUTS_NAME = "outputs"

# Standard subdirectories under each run directory
TENSORBOARD_SUBDIR = "tensorboard"
CHECKPOINTS_SUBDIR = "checkpoints"
PREDICTIONS_SUBDIR = "predictions"
METRICS_SUBDIR = "metrics"


def get_outputs_root(root: str | Path | None = None) -> Path:
    """Resolve the top-level outputs directory.

    Resolution order: explicit `root` arg > env var `WIENERNET_OUTPUTS_DIR`
    > `./outputs` in the current working directory.
    """
    if root is not None:
        return Path(root).expanduser().resolve()
    env_root = os.environ.get(DEFAULT_OUTPUTS_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd() / DEFAULT_OUTPUTS_NAME


def make_run_dir(
    root: str | Path | None = None,
    run_name: str | None = None,
    *,
    timestamp: bool = True,
    create_subdirs: bool = True,
) -> Path:
    """Create and return a fresh run directory.

    Args:
        root: Outputs root. Defaults to `get_outputs_root()`.
        run_name: Human-readable suffix (e.g. "paper_main" or "ablation_no_sde").
        timestamp: Prepend a sortable timestamp prefix.
        create_subdirs: Pre-create the standard subdirectories (tensorboard/,
            checkpoints/, predictions/, metrics/).

    Returns:
        The newly created run directory. Raises if it already exists.
    """
    outputs = get_outputs_root(root)
    outputs.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    if timestamp:
        parts.append(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if run_name:
        parts.append(run_name)
    if not parts:
        raise ValueError("Either timestamp=True or run_name must be set")

    run_dir = outputs / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=False)

    if create_subdirs:
        for sub in (TENSORBOARD_SUBDIR, CHECKPOINTS_SUBDIR, PREDICTIONS_SUBDIR, METRICS_SUBDIR):
            (run_dir / sub).mkdir(exist_ok=True)

    return run_dir


def tensorboard_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / TENSORBOARD_SUBDIR


def checkpoints_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / CHECKPOINTS_SUBDIR


def predictions_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / PREDICTIONS_SUBDIR


def metrics_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / METRICS_SUBDIR


def best_checkpoint(run_dir: str | Path) -> Path:
    return checkpoints_dir(run_dir) / "best.pth"


def last_checkpoint(run_dir: str | Path) -> Path:
    return checkpoints_dir(run_dir) / "last.pth"
