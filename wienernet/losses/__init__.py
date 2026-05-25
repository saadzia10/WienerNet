"""Loss functions and assembly."""

from .composite import compute_losses, make_gaussian_noise_prior
from .mmd import MMDLoss

__all__ = ["MMDLoss", "compute_losses", "make_gaussian_noise_prior"]
