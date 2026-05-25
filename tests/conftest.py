"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wienernet.models import build_model
from wienernet.utils import set_seed_globally


INPUT_DIM = 20  # X (17) + bNEE (1) + k (2)
BATCH = 16


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Every test starts from the same seed for reproducibility."""
    set_seed_globally(42, deterministic=False)


@pytest.fixture
def synthetic_batch() -> dict[str, torch.Tensor]:
    """One realistic-looking batch (CPU)."""
    g = torch.Generator().manual_seed(1)
    return {
        "X": torch.randn(BATCH, 17, generator=g),
        "bNEE": torch.randn(BATCH, generator=g),
        "k": torch.abs(torch.randn(BATCH, 2, generator=g)) * 50 + 100,
        "T": torch.randn(BATCH, generator=g) * 10 + 15,
        "dT": torch.randn(BATCH, generator=g) * 0.05,
        "dNEE": torch.randn(BATCH, generator=g) * 0.1,
        "NEE": torch.randn(BATCH, generator=g) * 3,
    }


@pytest.fixture(params=[
    "piae_sde_sampling",
    "piae_sde_reg_sampling",
    "pivae_sde_sampling",
    "ae",
    "vae",
])
def variant(request) -> str:
    """Parametrised across all 5 model variants."""
    return request.param
