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

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from ..losses.likelihood import student_t_dof
from ..physics.lloyd_taylor import DEFAULT_T0, DEFAULT_TREF, sde_drift
from .components import (
    ReparamHead,
    build_mlp,
    build_mlp_with_head,
    draw_unit_noise,
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
    # Structurally fix the noise mean to zero: drop the fc_mu head so the noise
    # is eps * sigma(z), i.e. E[eps|z]=0 by construction. Forces the deterministic
    # part (nee_raw + drift) to carry the conditional mean rather than leaking it
    # into the noise. See the drift-displacement analysis.
    noise_zero_mean: bool = False
    # k_decoder uses LeakyReLU in the PIAE variants and ReLU/Tanh in AE/VAE.
    # Setting None means "use the same activation as the rest of the model".
    k_activation: str | None = "leaky_relu"
    k_activation_slope: float = 0.01


@dataclass
class InputsConfig:
    """Which ground-truth quantities are fed into the ENCODER input.

    These flags are fully orthogonal to the prediction heads (`HeadsConfig`)
    and to their anchor losses (`mse_E0`/`mse_rb`, `mse_temp_derivative`): a
    quantity can be fed as input, predicted by a head, and anchored by a loss
    all independently.

    CLEAN-PREDICTION CAVEAT: feeding a GT value here (e.g. `include_k=True`)
    *while* its head predicts it (`heads.k`) AND that prediction is anchored
    (`mse_E0`/`mse_rb`) lets the head trivially copy input→output. In that
    regime the k head is no longer a faithful driver→E0/rb estimate and any
    "clean prediction" / "where is Lloyd-Taylor biased" diagnostic is invalid.
    Turn the corresponding `include_*` flag OFF to keep the head exogenous.
    """

    # Boundary NEE (NEE at the current step). Fed as an encoder input for the
    # level models; the increment reformulation sets this False (bNEE is then
    # only the integration boundary, added post-hoc, not encoded).
    include_bnee: bool = True
    # Ground-truth (E0, rb) from the REddyProc fit.
    include_k: bool = True
    # Ground-truth dT/dt (dTa). NOTE: this only ever adds dTa to the *encoder*
    # input. The analytic SDE drift ALWAYS uses the predicted temp-derivative
    # head (never GT dTa), regardless of this flag.
    include_dtemp: bool = False
    # Standardise the optional GT inputs (k, dtemp — NOT bNEE) with train-fit
    # per-feature stats before concatenation. When False the values are
    # concatenated raw, matching the legacy behaviour so existing checkpoints
    # reproduce bit-for-bit. When True, `set_input_norm_stats()` supplies the
    # mean/std (stored as buffers so inference is self-contained). Scaling is
    # especially important for dTa, whose per-minute magnitude is tiny raw.
    scale_extra_inputs: bool = False


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
    inputs: InputsConfig = field(default_factory=InputsConfig)
    # Which (E0, rb) the analytic SDE drift consumes:
    #   "predicted"    — the k_decoder head output (default; end-to-end trained).
    #   "ground_truth" — the GT (E0, rb) from the batch, fed straight into the
    #                    physics operator. Independent of whether the k head
    #                    exists or is anchored; lets you isolate physics-misfit
    #                    from k-estimation error. dT/dt is always predicted.
    physics_k_source: str = "predicted"
    # Learnable Student-t dof (log_nu) for the heavy-tailed NLL loss variant.
    # Adds one parameter; leave False (default) unless training with student_t.
    noise_student_dof: bool = False
    # Physics constants — match data_pipeline.partitioning
    tref: float = DEFAULT_TREF
    t0: float = DEFAULT_T0
    # Euler step size (minutes). The physics drift is a per-MINUTE rate
    # (f = dReco/dT · dT/dt, and dT/dt = dTa is stored divided by the minutes to
    # the next timestep). The Euler-Maruyama step is nee_pred = bNEE + f·dt, so
    # dt un-normalises the rate back to the per-step increment. The data cadence
    # is 30 min; all kept rows have divisor 30 (whole-hour gaps drop out as
    # div-by-zero), so 30 exactly recovers NEE_next - NEE_t.
    dt: float = 30.0
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


_VALID_K_SOURCES = ("predicted", "ground_truth")


def encoder_input_dim(feature_dim: int, inputs: InputsConfig) -> int:
    """Width of the encoder input for a given scaled-feature width + input flags.

    ``feature_dim`` is the number of scaled driver/time/site columns (i.e.
    ``X.shape[1]``). The optional GT inputs add: bNEE (+1), k=(E0, rb) (+2),
    dTa (+1). The standard published pipeline is feature_dim=17, all-on-except
    dtemp → 17 + 1 + 2 = 20, matching the legacy hard-coded value.
    """
    dim = feature_dim
    if inputs.include_bnee:
        dim += 1
    if inputs.include_k:
        dim += 2
    if inputs.include_dtemp:
        dim += 1
    return dim


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
        nee_pred         : bnee + (drift * dt if predict_drift else 0)
    """

    def __init__(self, cfg: WienerNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.physics_k_source not in _VALID_K_SOURCES:
            raise ValueError(
                f"physics_k_source must be one of {_VALID_K_SOURCES}, got {cfg.physics_k_source!r}"
            )
        device = torch.device(cfg.device)
        self._device = device

        activation_cls = _resolve_activation(cfg.activation)

        # Standardisation buffers for the optional GT encoder inputs (k, dtemp).
        # Registered only when scale_extra_inputs is on so that legacy models
        # keep an identical state_dict (strict checkpoint loading stays happy).
        # Default to identity (mean 0, std 1) → no-op until real train stats are
        # injected via set_input_norm_stats().
        if cfg.inputs.scale_extra_inputs:
            self.register_buffer("k_norm_mean", torch.zeros(2))
            self.register_buffer("k_norm_std", torch.ones(2))
            self.register_buffer("dtemp_norm_mean", torch.zeros(1))
            self.register_buffer("dtemp_norm_std", torch.ones(1))

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
            # Build fc_mu first (when present) so weight-init RNG order — and thus
            # default-variant initialisation — is bit-identical to before.
            # noise_zero_mean drops fc_mu entirely -> noise = eps * sigma(z),
            # a zero-conditional-mean stochastic term.
            self.fc_mu = None if cfg.heads.noise_zero_mean else build_mlp_with_head(
                cfg.latent_dim, cfg.heads.noise_dims, 1, activation=activation_cls,
            )
            self.fc_logvar = build_mlp_with_head(
                cfg.latent_dim, cfg.heads.noise_dims, 1, activation=activation_cls,
            )
        else:
            self.fc_mu = None
            self.fc_logvar = None

        # Student-t dof (log_nu) for the heavy-tailed NLL variant; nu = softplus+1,
        # init so nu ~ 8. Only present when noise_student_dof (old ckpts unaffected).
        self.log_nu = (
            nn.Parameter(torch.tensor(float(math.log(math.expm1(7.0)))))
            if cfg.noise_student_dof else None
        )

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

    def set_input_norm_stats(
        self,
        k_mean: Sequence[float] | torch.Tensor | None = None,
        k_std: Sequence[float] | torch.Tensor | None = None,
        dtemp_mean: float | torch.Tensor | None = None,
        dtemp_std: float | torch.Tensor | None = None,
    ) -> "WienerNetModel":
        """Populate the encoder-input standardisation buffers with train-fit stats.

        No-op (with a helpful error) unless the model was built with
        `inputs.scale_extra_inputs=True`. Only the stats needed by the active
        `include_*` flags matter; passing None for a quantity leaves its buffer
        at identity.
        """
        if not self.cfg.inputs.scale_extra_inputs:
            raise RuntimeError(
                "set_input_norm_stats requires inputs.scale_extra_inputs=True; "
                "the model has no normalisation buffers."
            )
        with torch.no_grad():
            if k_mean is not None:
                self.k_norm_mean.copy_(torch.as_tensor(k_mean, dtype=torch.float32).view(-1))
            if k_std is not None:
                self.k_norm_std.copy_(torch.as_tensor(k_std, dtype=torch.float32).view(-1))
            if dtemp_mean is not None:
                self.dtemp_norm_mean.copy_(torch.as_tensor(dtemp_mean, dtype=torch.float32).view(-1))
            if dtemp_std is not None:
                self.dtemp_norm_std.copy_(torch.as_tensor(dtemp_std, dtype=torch.float32).view(-1))
        return self

    def build_encoder_input(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        k: torch.Tensor,
        dT: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the encoder input from scaled features + the flagged GT parts.

        Order is always [x, (bNEE), (k), (dTa)] so the column layout is stable.
        Optional GT parts (k, dTa) are standardised with the norm buffers when
        `inputs.scale_extra_inputs` is on. bNEE is always concatenated raw.
        """
        inputs = self.cfg.inputs
        scale = inputs.scale_extra_inputs
        parts: list[torch.Tensor] = [x]
        if inputs.include_bnee:
            parts.append(b.view(x.shape[0], 1))
        if inputs.include_k:
            k_in = (k - self.k_norm_mean) / self.k_norm_std if scale else k
            parts.append(k_in)
        if inputs.include_dtemp:
            if dT is None:
                raise ValueError(
                    "inputs.include_dtemp=True but no dT (dTa) was passed to forward()."
                )
            dtemp = dT.view(x.shape[0], 1)
            if scale:
                dtemp = (dtemp - self.dtemp_norm_mean) / self.dtemp_norm_std
            parts.append(dtemp)
        return torch.cat(parts, dim=1).to(self._device)

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        k: torch.Tensor,
        T: torch.Tensor,
        dt: torch.Tensor | None = None,
        dT: torch.Tensor | None = None,
        site: object | None = None,
        dT_diurnal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Full forward pass returning a dict of all (possibly-None) outputs.

        `site` (a per-row site identifier, e.g. the batch's site_id list) is
        accepted for a uniform model-call signature across the family — this model
        does not use it (only the per-site analytical baseline does).

        Inputs:
            x: [B, X_dim] feature matrix (drivers + time + site_xyz).
            b: [B] or [B, 1] boundary NEE (NEE at the current step). Concatenated
                into the encoder input iff `inputs.include_bnee`; always used as
                the Euler integration boundary for nee_pred.
            k: [B, 2] (E0, rb) estimates from REddyProc fit. Concatenated iff
                `inputs.include_k`; also the physics source iff
                `physics_k_source == "ground_truth"`.
            T: [B] or [B, 1] temperature at the current step.
            dt: [B] or [B, 1] per-row Euler step size (minutes to the next
                timestamp). When None, falls back to the scalar config `dt`.
            dT: [B] or [B, 1] GT dT/dt (dTa). Required iff
                `inputs.include_dtemp` (it is only ever an encoder input; the
                drift always uses the predicted temp-derivative head).
        """
        input_tensor = self.build_encoder_input(x, b, k, dT)

        z, latent_mu, latent_logvar = self.encode(input_tensor)

        # Noise head. With noise_zero_mean, fc_mu is absent and the mean is a
        # constant zero tensor, so noise = eps * sigma(z) (E[eps|z]=0).
        if self.fc_logvar is not None:
            noise_logvar = self.fc_logvar(z)
            noise_mu = self.fc_mu(z) if self.fc_mu is not None else torch.zeros_like(noise_logvar)
            # Sample the noise from the SAME family the head is trained under —
            # Student-t (dof from log_nu) for the student_t NLL, else Gaussian —
            # so the sampled predictive law matches the likelihood + scoring. The
            # VAE latent (self.encode) stays Gaussian; only the noise head changes.
            std = torch.exp(0.5 * noise_logvar)
            nu = student_t_dof(self.log_nu).detach() if self.log_nu is not None else None
            noise = noise_mu + draw_unit_noise(std.shape, std.device, nu) * std
        else:
            noise_mu = noise_logvar = noise = None

        # Decoders
        nee_raw = self.nee_decoder(z)
        bnee = nee_raw + noise if noise is not None else nee_raw

        k_pred = self.k_decoder(z) if self.k_decoder is not None else None
        temp_derivative = (
            self.temp_derivative_decoder(z) if self.temp_derivative_decoder is not None else None
        )

        # Drift via the analytic SDE operator. The (E0, rb) source is either the
        # predicted k head or the GT batch k (physics_k_source). dT/dt is always
        # the predicted temp-derivative head (never GT dTa), so drift needs the
        # temp head regardless of the k source.
        if self.cfg.physics_k_source == "ground_truth":
            k_src = k.to(self._device)          # raw GT (E0, rb) — physics wants raw
        else:
            k_src = k_pred
        if k_src is not None and temp_derivative is not None:
            T_view = T.view(-1, 1)
            E0 = k_src[:, 0:1]
            rb = k_src[:, 1:2]
            drift = sde_drift(T_view, E0, rb, temp_derivative, tref=self.cfg.tref, t0=self.cfg.t0)
        else:
            drift = None

        if self.cfg.predict_drift and drift is not None:
            # Euler-Maruyama step: bNEE + f·dt (drift is a per-minute rate).
            # Prefer the per-row dt from the data; fall back to the scalar config dt.
            step_dt = dt.view(-1, 1).to(drift.device) if dt is not None else self.cfg.dt
            nee_pred = bnee + drift * step_dt
            drift_contrib = drift * step_dt
        else:
            nee_pred = bnee
            drift_contrib = torch.zeros_like(nee_raw)

        # Likelihood sufficient statistics on the NEE target scale:
        #   nee_mean = deterministic prediction (nee_raw + drift*dt, NO noise)
        #   nee_log_std = log of the level noise std = 0.5*noise_logvar
        # For a level model the noise is added at the level, so its std IS the
        # target-scale std. Valid decomposition needs zero-mean noise.
        nee_mean = nee_raw + drift_contrib
        nee_log_std = (0.5 * noise_logvar) if noise_logvar is not None else None

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
            "nee_mean": nee_mean,          # deterministic mean for the NLL loss
            "nee_log_std": nee_log_std,    # target-scale log-std for the NLL loss
            "log_nu": self.log_nu,         # Student-t dof (None unless enabled)
        }

    # ----------------------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------------------

    def initialize(self) -> "WienerNetModel":
        """Apply Xavier-uniform init (matches the original notebook flow)."""
        self.apply(initialize_weights)
        return self
