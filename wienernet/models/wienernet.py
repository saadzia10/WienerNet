"""Unified WienerNet model — replaces all 5 model classes in piae_sde/night/*.py.

A single class with composable heads that can be configured to reproduce any
of the original architectures:

  PIAE_SDE_Sampling         ≡ HeadsConfig(temp_derivative=True, k=True, noise=True),
                              latent_reparameterize=False, predict_drift=True
  PIAE_SDE_Reg_Sampling     ≡ HeadsConfig(temp_derivative=True, k=True, noise=True),
                              latent_reparameterize=False, predict_drift=False
  PIVAE_SDE_Sampling        ≡ HeadsConfig(temp_derivative=True, k=True, noise=True),
                              latent_reparameterize=True,  predict_drift=True
  AE                        ≡ HeadsConfig(temp_derivative=False, k=True, noise=False),
                              latent_reparameterize=False, predict_drift=False
  VAE                       ≡ HeadsConfig(temp_derivative=False, k=True, noise=False),
                              latent_reparameterize=True,  predict_drift=False

Submodule names match the original implementation so existing checkpoints
load (`encoder`, `nee_decoder`, `temp_derivative_decoder`, `k_decoder`,
`fc_mu` / `fc_logvar` for noise head, `latent_mu` / `latent_logvar` for VAE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from ..physics.lloyd_taylor import DEFAULT_T0, DEFAULT_TREF, sde_drift
from .components import (
    ReparamHead,
    build_mlp,
    build_mlp_with_head,
    initialize_weights,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HeadsConfig:
    """Which heads exist on the model."""

    nee: bool = True           # always on; included for symmetry / documentation
    temp_derivative: bool = True
    k: bool = True
    noise: bool = True
    noise_dims: tuple[int, ...] = (4,)
    # k_decoder uses LeakyReLU in the PIAE variants and ReLU/Tanh in AE/VAE.
    # Setting None means "use the same activation as the rest of the model".
    k_activation: str | None = "leaky_relu"
    k_activation_slope: float = 0.01


@dataclass
class WienerNetConfig:
    """Top-level model config. Mirrors the Hydra YAML in phase 5."""

    input_dim: int                              # X + bNEE + k -> see registry
    latent_dim: int = 32
    encoder_dims: tuple[int, ...] = (16, 16)
    decoder_dims: tuple[int, ...] = (16, 16)
    activation: str = "relu"                    # 'relu' | 'tanh' | 'leaky_relu'
    latent_reparameterize: bool = False         # VAE-style z = reparam(mu, logvar)
    predict_drift: bool = True                  # nee_pred = bnee + drift
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    # Physics constants — match data_pipeline.partitioning
    tref: float = DEFAULT_TREF
    t0: float = DEFAULT_T0
    device: str = "cpu"


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def _resolve_activation(name: str) -> type[nn.Module]:
    try:
        return _ACTIVATIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown activation {name!r}; valid: {list(_ACTIVATIONS)}") from exc


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WienerNetModel(nn.Module):
    """Composable PIAE/PIVAE/AE/VAE.

    Forward signature: `forward(x, b, k, T) -> dict`.
    The dict always contains keys (some may be `None` if a head is disabled):
        encoder_h        : output of the encoder backbone (before latent_mu/logvar)
        latent           : z, the latent vector used by all decoders
        latent_mu, latent_logvar : non-None iff latent_reparameterize
        nee_raw          : output of nee_decoder before noise / drift
        noise            : sampled noise (None if noise head off)
        noise_mu, noise_logvar : non-None iff noise head enabled
        k                : (E0, rb) from k_decoder (None if k head off)
        temp_derivative  : dT/dt (None if temp head off)
        drift            : f = dReco/dT * dT/dt (None if k or temp head off)
        bnee             : nee_raw + noise (boundary NEE)
        nee_pred         : bnee + (drift if predict_drift else 0)
    """

    def __init__(self, cfg: WienerNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        device = torch.device(cfg.device)
        self._device = device

        activation_cls = _resolve_activation(cfg.activation)

        # ------------------------------------------------------------------
        # Encoder backbone
        # ------------------------------------------------------------------
        if cfg.latent_reparameterize:
            # VAE-style: encoder produces a hidden representation, then two
            # separate heads produce (mu, logvar) that are reparameterized.
            backbone = build_mlp(cfg.input_dim, cfg.encoder_dims, activation_cls)
            self.encoder = nn.Sequential(*backbone)
            self.latent_mu = nn.Sequential(
                nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim),
                activation_cls(),
            )
            self.latent_logvar = nn.Sequential(
                nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim),
                activation_cls(),
            )
        else:
            # Deterministic encoder: build hidden layers then project to latent_dim.
            modules = build_mlp(cfg.input_dim, cfg.encoder_dims, activation_cls)
            modules.append(nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim))
            self.encoder = nn.Sequential(*modules)
            self.latent_mu = None
            self.latent_logvar = None

        # ------------------------------------------------------------------
        # Decoder heads
        # ------------------------------------------------------------------
        self.nee_decoder = build_mlp_with_head(
            cfg.latent_dim, cfg.decoder_dims, 1, activation=activation_cls,
        )

        if cfg.heads.temp_derivative:
            self.temp_derivative_decoder = build_mlp_with_head(
                cfg.latent_dim, cfg.decoder_dims, 1, activation=activation_cls,
            )
        else:
            self.temp_derivative_decoder = None

        if cfg.heads.k:
            # k_decoder typically uses LeakyReLU to avoid zero-E0 collapse.
            if cfg.heads.k_activation == "leaky_relu":
                k_act = nn.LeakyReLU
                k_kwargs = {"negative_slope": cfg.heads.k_activation_slope}
                final_act = nn.LeakyReLU
                final_kwargs = {"negative_slope": cfg.heads.k_activation_slope}
            else:
                k_act = activation_cls
                k_kwargs = {}
                final_act = None
                final_kwargs = None
            self.k_decoder = build_mlp_with_head(
                cfg.latent_dim, cfg.decoder_dims, 2,
                activation=k_act,
                activation_kwargs=k_kwargs,
                final_activation=final_act,
                final_activation_kwargs=final_kwargs,
            )
        else:
            self.k_decoder = None

        # Noise head: shares the original ReparamHead pattern (mu + logvar +
        # reparameterise). The submodule names fc_mu / fc_logvar match the
        # original implementation so existing checkpoints load.
        if cfg.heads.noise:
            self.fc_mu = build_mlp_with_head(
                cfg.latent_dim, cfg.heads.noise_dims, 1, activation=activation_cls,
            )
            self.fc_logvar = build_mlp_with_head(
                cfg.latent_dim, cfg.heads.noise_dims, 1, activation=activation_cls,
            )
        else:
            self.fc_mu = None
            self.fc_logvar = None

        self.to(device)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def encode(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return (latent z, latent_mu, latent_logvar). The mu/logvar are
        None unless latent_reparameterize is True."""
        if self.cfg.latent_reparameterize:
            h = self.encoder(input_tensor)
            mu = self.latent_mu(h)
            logvar = self.latent_logvar(h)
            z = self._reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            return self.encoder(input_tensor), None, None

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        k: torch.Tensor,
        T: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Full forward pass returning a dict of all (possibly-None) outputs.

        Inputs:
            x: [B, X_dim] feature matrix (drivers + time + site_xyz).
            b: [B] or [B, 1] boundary NEE (NEE at the current step).
            k: [B, 2] (E0, rb) estimates from REddyProc fit.
            T: [B] or [B, 1] temperature at the current step.
        """
        input_tensor = torch.cat(
            (x, b.view(x.shape[0], 1), k), dim=1
        ).to(self._device)

        z, latent_mu, latent_logvar = self.encode(input_tensor)

        # Noise head
        if self.fc_mu is not None and self.fc_logvar is not None:
            noise_mu = self.fc_mu(z)
            noise_logvar = self.fc_logvar(z)
            noise = self._reparameterize(noise_mu, noise_logvar)
        else:
            noise_mu = noise_logvar = noise = None

        # Decoders
        nee_raw = self.nee_decoder(z)
        bnee = nee_raw + noise if noise is not None else nee_raw

        k_pred = self.k_decoder(z) if self.k_decoder is not None else None
        temp_derivative = (
            self.temp_derivative_decoder(z) if self.temp_derivative_decoder is not None else None
        )

        # Drift via the analytic SDE operator
        if k_pred is not None and temp_derivative is not None:
            T_view = T.view(-1, 1)
            E0 = k_pred[:, 0:1]
            rb = k_pred[:, 1:2]
            drift = sde_drift(T_view, E0, rb, temp_derivative, tref=self.cfg.tref, t0=self.cfg.t0)
        else:
            drift = None

        if self.cfg.predict_drift and drift is not None:
            nee_pred = bnee + drift
        else:
            nee_pred = bnee

        return {
            "latent": z,
            "latent_mu": latent_mu,
            "latent_logvar": latent_logvar,
            "nee_raw": nee_raw,
            "noise": noise,
            "noise_mu": noise_mu,
            "noise_logvar": noise_logvar,
            "k": k_pred,
            "temp_derivative": temp_derivative,
            "drift": drift,
            "bnee": bnee,
            "nee_pred": nee_pred,
        }

    # ----------------------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------------------

    def initialize(self) -> "WienerNetModel":
        """Apply Xavier-uniform init (matches the original notebook flow)."""
        self.apply(initialize_weights)
        return self
