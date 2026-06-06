"""Global reproducibility helpers.

One entrypoint, `set_seed_globally`, sets every RNG that matters (python, numpy,
torch CPU/CUDA) plus PYTHONHASHSEED, and optionally enables fully deterministic
CUDA kernels. Call once at the start of a training run.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed_globally(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG and (optionally) enable deterministic CUDA kernels.

    Args:
        seed: Integer seed shared across python/numpy/torch/cuda.
        deterministic: If True, also forces deterministic algorithms in
            torch and disables cuDNN's benchmark autotuner. This is
            slower but guarantees bitwise reproducibility across runs
            with the same hardware + driver.

    Notes:
        - Sets PYTHONHASHSEED via the environment. Python's hash randomization
          is set at interpreter start, so this only affects child processes.
          For the current process, dict/set ordering on objects with hashes
          based on `id()` may still vary across runs. This is not a problem
          for ML training, only for some unit tests on hashable structures.
        - `torch.use_deterministic_algorithms(True)` can raise at runtime if a
          non-deterministic op is invoked. If that happens, set
          `deterministic=False` for that run and document why.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # With CUDA >= 10.2, deterministic cuBLAS GEMMs require a fixed
        # workspace config set *before* the first cuBLAS handle is created.
        # Setting it here (run startup) is early enough.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def snapshot_rng_state() -> dict:
    """Capture all RNG states for full-resumption checkpoints."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore RNG states previously captured by `snapshot_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
