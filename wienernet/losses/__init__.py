"""Loss functions and assembly."""

from .composite import (
    compute_losses,
    make_empirical_noise_prior,
    make_gaussian_noise_prior,
)
from .mmd import MMDLoss

__all__ = [
    "MMDLoss", "compute_losses",
    "make_gaussian_noise_prior", "make_empirical_noise_prior",
]
