"""Reusable model building blocks.

`build_mlp` and `build_mlp_with_head` together replace the
`append_linear_modules` pattern repeated across all five night/*.py models.
`ReparamHead` factors the (mu, logvar, reparameterize) trio used by both the
noise head and the VAE latent.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.init as init


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
    *,
    activation_kwargs: dict | None = None,
) -> list[nn.Module]:
    """Return `[Linear, Act, Linear, Act, ...]` matching the notebook pattern.

    The output is a flat list so callers can append a final Linear (and
    optionally another activation) before wrapping in `nn.Sequential`.

    Args:
        in_dim: input dimension.
        hidden_dims: sizes of hidden layers.
        activation: activation class (called with no args by default).
        activation_kwargs: optional kwargs passed to `activation(**kwargs)`,
            e.g. `{"negative_slope": 0.01}` for LeakyReLU.
    """
    kwargs = activation_kwargs or {}
    modules: list[nn.Module] = []
    current = in_dim
    for dim in hidden_dims:
        modules.append(nn.Linear(current, dim))
        modules.append(activation(**kwargs))
        current = dim
    return modules


def build_mlp_with_head(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    *,
    activation_kwargs: dict | None = None,
    final_activation: type[nn.Module] | None = None,
    final_activation_kwargs: dict | None = None,
) -> nn.Sequential:
    """Build an MLP with a final Linear head and optional final activation."""
    modules = build_mlp(in_dim, hidden_dims, activation, activation_kwargs=activation_kwargs)
    last_hidden = hidden_dims[-1] if hidden_dims else in_dim
    modules.append(nn.Linear(last_hidden, out_dim))
    if final_activation is not None:
        modules.append(final_activation(**(final_activation_kwargs or {})))
    return nn.Sequential(*modules)


class ReparamHead(nn.Module):
    """Two parallel MLPs producing (mu, logvar) plus a reparameterised sample.

    Used by the noise head (PIAE_SDE_Sampling and variants) and by the VAE
    latent. Both heads share an architecture template but have independent
    weights — this matches the original implementation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.mu = build_mlp_with_head(in_dim, hidden_dims, out_dim, activation=activation)
        self.logvar = build_mlp_with_head(in_dim, hidden_dims, out_dim, activation=activation)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, logvar, sample)."""
        mu = self.mu(x)
        logvar = self.logvar(x)
        sample = self.reparameterize(mu, logvar)
        return mu, logvar, sample


def initialize_weights(layer: nn.Module) -> None:
    """Xavier-uniform Linear weights, zero biases.

    Applied via `model.apply(initialize_weights)`. Preserves the original
    initialisation strategy used by all five night/*.py model files.
    """
    if isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)
        init.zeros_(layer.bias)
