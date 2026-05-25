"""Logger setup: console (INFO) + file (DEBUG).

Call `setup_logging(run_dir)` once at the top of each training/eval script.
Subsequent calls on the same logger are idempotent — handlers are only
attached once, so importing this from notebooks doesn't multiply output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "wienernet"
_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s — %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    run_dir: str | Path | None = None,
    *,
    name: str = _LOGGER_NAME,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_filename: str = "run.log",
) -> logging.Logger:
    """Configure and return the project logger.

    Args:
        run_dir: Directory for the rotating log file. If None, no file handler.
        name: Logger name. Defaults to "wienernet" so child loggers
            (e.g. "wienernet.training.trainer") inherit settings.
        console_level: Minimum level for stderr output.
        file_level: Minimum level for the file handler.
        log_filename: File created under `run_dir`.

    Returns:
        The configured logger. Calling again with the same `name` is safe;
        existing handlers of the same kind are not duplicated.
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    # Prevent double-emission via the root logger
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_wn_kind", None) == "console" for h in logger.handlers):
        console_handler = _make_console_handler(console_level)
        logger.addHandler(console_handler)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / log_filename
        if not any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path
            for h in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(file_handler)

    return logger


def _make_console_handler(level: int) -> logging.Handler:
    """Use Rich if available for nicer console output; fall back to stderr."""
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            level=level,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    handler._wn_kind = "console"  # type: ignore[attr-defined]
    return handler


def get_logger(name: str | None = None) -> logging.Logger:
    """Child-logger accessor. Inherits the project logger's handlers.

    Use as `log = get_logger(__name__)` in any module.
    """
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    if name == _LOGGER_NAME or name.startswith(_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
