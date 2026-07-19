"""Increment-based Euler–Maruyama SDE model — a separate reformulation of WienerNet.

Where `WienerNetModel` reconstructs the NEE *level* (nee_raw) and adds a physics
drift, this model predicts the **increment** and integrates onto the observed
boundary:

    z       = encoder(x_exogenous)              # x excludes NEE_t (no exposure bias)
    E0, rb  = k_decoder(z)                      # -> mse_E0 / mse_rb (REddyProc anchor)
    dT/dt   = temp_derivative_decoder(z)        # -> mse_temp_derivative (faithful dTa)
    f_phys  = dReco/dT(T, E0, rb) * dT/dt       # physics drift RATE (per minute)
    r       = residual_decoder(z)               # optional drift-misfit correction
    sigma   = softplus(sigma_decoder(z))        # optional diffusion scale
    dNEE    = (f_phys + r) * dt + (mu + eps*sigma) * sqrt(dt)     # Euler–Maruyama
    nee_pred = bNEE_observed + dNEE             # -> mse_nee vs NEE_{t+k}

Three named variants (see registry / configs):
  A. piae_increment_residual — residual on, zero-mean noise (aleatoric only).
     r(z) is the "where Lloyd-Taylor is biased" diagnostic; noise is pure noise.
  B. piae_increment          — no residual, noise keeps a learned mean that
     absorbs the physics misfit (the old behaviour, in increment form).
  C. piae_reg_increment      — deterministic drift only (no residual, no noise);
     the PIAE-reg analog: dNEE = f_phys * dt.

Design invariants (docs/increment_sde_model_plan.md §3):
  * bNEE is the integration boundary only — never reconstructed, and (by default)
    not fed to the encoder. Drift/diffusion are exogenous functions of state.
  * Drift scales with dt, diffusion with sqrt(dt); per-row dt from the batch so
    the model composes with the dt-scale sweep.
  * The encoder-input flags (InputsConfig) and physics_k_source are the SAME
    shared knobs as WienerNetModel, so E0/rb/dTa can be toggled in/out here too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.likelihood import LOG_KAPPA_MAX, LOG_KAPPA_MIN, student_t_dof
from ..physics.lloyd_taylor import DEFAULT_T0, DEFAULT_TREF, reco, sde_drift
from .components import (
    build_mlp,
    build_mlp_with_head,
    draw_unit_noise,
    initialize_weights,
    sample_ald_noise,
)
from .wienernet import (
    _VALID_K_SOURCES,
    InputsConfig,
    _resolve_activation,
    encoder_input_dim,
)

# Minimum diffusion scale so sigma stays strictly positive and log(sigma) finite.
_SIGMA_FLOOR = 1e-4


@dataclass
class IncrementSDEConfig:
    """Config for IncrementSDEModel. Mirrors WienerNetConfig where it overlaps.

    The default `inputs` is fully exogenous (no bNEE, no GT k, no GT dTa in the
    encoder) — the recommended increment setup. Flip individual flags to feed
    any of them; see InputsConfig for the clean-prediction caveat.
    """

    input_dim: int
    latent_dim: int = 32
    encoder_dims: tuple[int, ...] = (16, 16)
    decoder_dims: tuple[int, ...] = (16, 16)
    activation: str = "relu"
    latent_reparameterize: bool = False        # PIVAE-style latent (default off)

    # Heads / variant switches --------------------------------------------------
    predict_k: bool = True                     # E0, rb head (anchored by mse_E0/rb)
    predict_temp_derivative: bool = True        # dT/dt head (anchored by mse_temp_derivative)
    residual: bool = False                      # version A: drift-misfit correction r(z)
    noise: bool = True                          # diffusion term present (C sets False)
    noise_zero_mean: bool = True                # A: True (aleatoric); B: False (learned mean)
    noise_dims: tuple[int, ...] = (4,)
    residual_dims: tuple[int, ...] = (16, 16)
    k_activation: str | None = "leaky_relu"
    k_activation_slope: float = 0.01
    # Add a learnable Student-t degrees-of-freedom scalar (log_nu) for the
    # heavy-tailed likelihood loss. Only needed when training with the student_t
    # NLL variant; adds one parameter (old checkpoints unaffected when False).
    noise_student_dof: bool = False
    # Terminal noise-model options (Workstream F). noise_asymmetry adds a learnable
    # asymmetry scalar (log_kappa) for the asymmetric-Laplace (ALD) noise — captures
    # the +right-skew a symmetric t misses. noise_mixture_components > 1 replaces the
    # single-scale noise with a K-component ZERO-MEAN mixture noise head (the MDN's
    # flexible shape, kept on the SDE so the drift stays the mean). At most one of
    # {student_dof, asymmetry, mixture>1} should be on.
    noise_asymmetry: bool = False
    noise_mixture_components: int = 1
    # Physics-anchored heteroscedastic scale (Workstream C.1 / coverage fix). When
    # True the diffusion scale is MULTIPLIED by a learnable physics term:
    #   sigma_eff = softplus(sigma_head(z)) * (softplus(a) + softplus(b) * Reco)
    # so the noise std grows with the respiration magnitude Reco(T,E0,rb) — matching
    # the empirical heteroscedasticity SD ~ 0.24 + 0.30*Reco. This is the lever that
    # fixes the L1/pinball ALD under-coverage: the tail-robust ALD loss collapses a
    # free sigma head to the bulk MAD (5x too small on high-flux nights), but with
    # sigma structurally tied to Reco the loss grows `b` to cover the high-flux
    # spread. Reco is used DETACHED (a fixed covariate) so the noise scale reads the
    # respiration level without perturbing the drift's E0/rb. a,b >= 0 (softplus);
    # init a->1, b->`noise_physics_scale_b` so sigma starts at ~(1 + b*Reco)*head.
    noise_physics_scale: bool = False
    noise_physics_scale_b: float = 0.25    # initial softplus(b): the Reco->sigma coupling

    # ---- State-space / physics-integration variant (WienerNet-SS) --------------
    # These default to the current behaviour; the new variant switches them on.
    #
    # exact_reco_drift: the drift increment is the EXACT respiration change
    #   Reco(T+ΔT) - Reco(T) with ΔT = (dT/dt)*dt, instead of the Euler linearization
    #   dReco/dT * ΔT. Removes the convexity bias that grows with dt; reduces to the
    #   linearization as dt->0.
    exact_reco_drift: bool = False
    # drift_tendency: where the drift's dT/dt comes from.
    #   "learned_observed" (current): a learned head anchored to observed dTa.
    #   "learned_diurnal" : a learned head anchored to the physics diurnal tendency
    #                       (dTa_diurnal) — learns to predict the predictable diurnal part.
    #   "diurnal"         : use the exogenous physics diurnal tendency directly (no head).
    drift_tendency: str = "learned_observed"
    # noise_state_space: split the diffusion into an independent MEASUREMENT term
    #   (flat in dt) and a small WIENER PROCESS term (∝ dt), so the increment variance
    #   is  2*sigma_meas^2 + sigma_proc^2 * dt  — matching the empirically ~flat noise
    #   scaling (not the pure sqrt(dt) a single Wiener head assumes). Adds one head.
    noise_state_space: bool = False

    # Shared input / physics knobs (identical semantics to WienerNetConfig) ------
    inputs: InputsConfig = field(
        default_factory=lambda: InputsConfig(
            include_bnee=False, include_k=False, include_dtemp=False
        )
    )
    physics_k_source: str = "predicted"         # "predicted" | "ground_truth"

    # Soft physical bound on the per-minute drift RATE (f_phys + residual).
    # `d -> c*tanh(d/c)`: ~identity for |d| << c (real drift is ~±0.17/min) but
    # saturates at ±c, so a head that extrapolates to garbage on a far-OOD site
    # (e.g. the dTa head predicting a 600 C/min change) degrades GRACEFULLY to a
    # bounded increment instead of the exponential Lloyd-Taylor drift amplifying it
    # to a catastrophic prediction. None disables it (legacy behaviour).
    drift_clamp: float | None = 1.0

    # Physics constants + Euler step -------------------------------------------
    tref: float = DEFAULT_TREF
    t0: float = DEFAULT_T0
    dt: float = 30.0
    device: str = "cpu"


class IncrementSDEModel(nn.Module):
    """Euler–Maruyama increment predictor. Forward signature matches WienerNetModel
    (`forward(x, b, k, T, dt=None, dT=None)`) so it drops into the same Trainer.

    Output dict keys (some None depending on the variant):
        latent, latent_mu, latent_logvar
        k                : (E0, rb) from k_decoder (None if predict_k off)
        temp_derivative  : dT/dt
        f_phys           : physics drift rate dReco/dT * dT/dt (per minute)
        residual         : r(z) drift-misfit correction (None if residual off)
        drift            : drift_term = f_phys (+ r); the per-minute RATE, so
                           mse_drift compares it to the GT dNEE rate.
        sigma            : diffusion scale (None if noise off)
        noise            : reparam sample mu + eps*sigma (None if noise off)
        noise_mu, noise_logvar : for logging / mmd_noise (None if noise off)
        dnee_pred        : the integrated increment (drift*dt + noise*sqrt(dt))
        nee_pred         : bNEE_observed + dnee_pred (-> mse_nee vs NEE_{t+k})
        bnee             : always None (no level reconstruction; disables mse_bnee)
    """

    def __init__(self, cfg: IncrementSDEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.physics_k_source not in _VALID_K_SOURCES:
            raise ValueError(
                f"physics_k_source must be one of {_VALID_K_SOURCES}, got {cfg.physics_k_source!r}"
            )
        device = torch.device(cfg.device)
        self._device = device
        activation_cls = _resolve_activation(cfg.activation)

        # Encoder-input standardisation buffers (only when scaling on) — same
        # convention as WienerNetModel so the same train stats slot in.
        if cfg.inputs.scale_extra_inputs:
            self.register_buffer("k_norm_mean", torch.zeros(2))
            self.register_buffer("k_norm_std", torch.ones(2))
            self.register_buffer("dtemp_norm_mean", torch.zeros(1))
            self.register_buffer("dtemp_norm_std", torch.ones(1))

        # Encoder backbone (optionally VAE-style) ------------------------------
        if cfg.latent_reparameterize:
            self.encoder = nn.Sequential(*build_mlp(cfg.input_dim, cfg.encoder_dims, activation_cls))
            self.latent_mu = nn.Sequential(
                nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim), activation_cls()
            )
            self.latent_logvar = nn.Sequential(
                nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim), activation_cls()
            )
        else:
            modules = build_mlp(cfg.input_dim, cfg.encoder_dims, activation_cls)
            modules.append(nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim))
            self.encoder = nn.Sequential(*modules)
            self.latent_mu = None
            self.latent_logvar = None

        # Physics heads --------------------------------------------------------
        if cfg.predict_k:
            if cfg.k_activation == "leaky_relu":
                k_act, k_kwargs = nn.LeakyReLU, {"negative_slope": cfg.k_activation_slope}
                final_act, final_kwargs = nn.LeakyReLU, {"negative_slope": cfg.k_activation_slope}
            else:
                k_act, k_kwargs, final_act, final_kwargs = activation_cls, {}, None, None
            self.k_decoder = build_mlp_with_head(
                cfg.latent_dim, cfg.decoder_dims, 2,
                activation=k_act, activation_kwargs=k_kwargs,
                final_activation=final_act, final_activation_kwargs=final_kwargs,
            )
        else:
            self.k_decoder = None

        # Temperature-tendency head — present unless the drift uses the exogenous
        # physics diurnal tendency directly ("diurnal" -> no head).
        self._use_diurnal_input = cfg.drift_tendency in ("diurnal", "learned_diurnal")
        self.temp_derivative_decoder = (
            build_mlp_with_head(cfg.latent_dim, cfg.decoder_dims, 1, activation=activation_cls)
            if (cfg.predict_temp_derivative and cfg.drift_tendency != "diurnal") else None
        )

        # Residual (drift-misfit) head — version A -----------------------------
        self.residual_decoder = (
            build_mlp_with_head(cfg.latent_dim, cfg.residual_dims, 1, activation=activation_cls)
            if cfg.residual else None
        )

        # Diffusion head(s). sigma via softplus (positivity); mu only when the
        # noise is allowed a learned mean (version B).
        self.n_mix = max(1, int(cfg.noise_mixture_components))
        if cfg.noise:
            self.sigma_decoder = build_mlp_with_head(
                cfg.latent_dim, cfg.noise_dims, 1, activation=activation_cls
            )
            self.noise_mu_decoder = None if cfg.noise_zero_mean else build_mlp_with_head(
                cfg.latent_dim, cfg.noise_dims, 1, activation=activation_cls
            )
            # State-space noise: a second head for the small Wiener PROCESS scale.
            # sigma_decoder plays the MEASUREMENT scale; the increment std combines
            # them as sqrt(2*sigma_meas^2 + sigma_proc^2*dt). Only built when on.
            self.sigma_proc_decoder = build_mlp_with_head(
                cfg.latent_dim, cfg.noise_dims, 1, activation=activation_cls
            ) if cfg.noise_state_space else None
            # Zero-mean MIXTURE noise head (Workstream F.2): per-component offset,
            # log-scale and mixing logits. Component offsets are centred at forward
            # time so the mixture is zero-mean and the drift stays the conditional
            # mean. Only built when noise_mixture_components > 1.
            if self.n_mix > 1:
                self.mix_offset_decoder = build_mlp_with_head(
                    cfg.latent_dim, cfg.noise_dims, self.n_mix, activation=activation_cls)
                self.mix_logscale_decoder = build_mlp_with_head(
                    cfg.latent_dim, cfg.noise_dims, self.n_mix, activation=activation_cls)
                self.mix_logit_decoder = build_mlp_with_head(
                    cfg.latent_dim, cfg.noise_dims, self.n_mix, activation=activation_cls)
            else:
                self.mix_offset_decoder = self.mix_logscale_decoder = self.mix_logit_decoder = None
        else:
            self.sigma_decoder = None
            self.noise_mu_decoder = None
            self.sigma_proc_decoder = None
            self.mix_offset_decoder = self.mix_logscale_decoder = self.mix_logit_decoder = None

        # Student-t dof (log_nu): nu = softplus(log_nu)+1, init so nu ~ 8 (mild tails).
        self.log_nu = (
            nn.Parameter(torch.tensor(float(math.log(math.expm1(7.0)))))
            if cfg.noise_student_dof else None
        )
        # Asymmetric-Laplace asymmetry (log_kappa): kappa=exp(log_kappa), init 1 (symmetric).
        self.log_kappa = nn.Parameter(torch.tensor(0.0)) if cfg.noise_asymmetry else None

        # Physics-anchored scale coefficients: sigma_mult = softplus(a) + softplus(b)*Reco.
        # init softplus(a)=1.0, softplus(b)=0.5 (a modest flux coupling to grow from).
        if cfg.noise_physics_scale:
            b0 = float(cfg.noise_physics_scale_b)
            b_raw = math.log(math.expm1(b0)) if b0 > 0 else -10.0   # softplus_inv(b0)
            self.scale_anchor_a = nn.Parameter(torch.tensor(0.54132))    # softplus -> 1.0
            self.scale_anchor_b = nn.Parameter(torch.tensor(float(b_raw)))
        else:
            self.scale_anchor_a = self.scale_anchor_b = None

        self.to(device)

    # ------------------------------------------------------------------
    # Encoder input / latent
    # ------------------------------------------------------------------

    def set_input_norm_stats(
        self,
        k_mean: Sequence[float] | torch.Tensor | None = None,
        k_std: Sequence[float] | torch.Tensor | None = None,
        dtemp_mean: float | torch.Tensor | None = None,
        dtemp_std: float | torch.Tensor | None = None,
    ) -> "IncrementSDEModel":
        """Populate the encoder-input standardisation buffers (see WienerNetModel)."""
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
        """Assemble the encoder input. Default (all include_* off) is just x."""
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

    def encode(self, input_tensor: torch.Tensor):
        if self.cfg.latent_reparameterize:
            h = self.encoder(input_tensor)
            mu = self.latent_mu(h)
            logvar = self.latent_logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            return z, mu, logvar
        return self.encoder(input_tensor), None, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        # `site` is accepted for a uniform model-call signature across the family;
        # this model does not use it (only the per-site analytical baseline does).
        z, latent_mu, latent_logvar = self.encode(self.build_encoder_input(x, b, k, dT))

        k_pred = self.k_decoder(z) if self.k_decoder is not None else None
        temp_derivative = (
            self.temp_derivative_decoder(z) if self.temp_derivative_decoder is not None else None
        )

        # Per-row dt (minutes to next timestamp). Needed early: the exact-Reco drift and
        # the state-space noise both depend on dt.
        if dt is not None:
            step_dt = dt.view(-1, 1).to(self._device)
        else:
            step_dt = torch.full((x.shape[0], 1), float(self.cfg.dt), device=self._device)
        sqrt_dt = torch.sqrt(step_dt)

        # Drift tendency dT/dt: learned head, or the exogenous physics diurnal tendency.
        if self.cfg.drift_tendency == "diurnal":
            tendency = dT_diurnal.view(-1, 1).to(self._device) if dT_diurnal is not None else None
        else:
            tendency = temp_derivative

        # Physics drift RATE (per minute). E0/rb from the predicted head or GT k.
        if self.cfg.physics_k_source == "ground_truth":
            k_src = k.to(self._device)
        else:
            k_src = k_pred
        if k_src is not None and tendency is not None:
            T_view = T.view(-1, 1)
            E0c, rbc = k_src[:, 0:1], k_src[:, 1:2]
            if self.cfg.exact_reco_drift:
                # Exact respiration change over the step, expressed as a rate so the
                # clamp/residual/dt machinery below is unchanged:
                #   f_phys = (Reco(T + dT/dt*dt) - Reco(T)) / dt   -> dReco/dT*(dT/dt) as dt->0
                dT_step = tendency * step_dt
                f_phys = (
                    reco(T_view + dT_step, E0c, rbc, tref=self.cfg.tref, t0=self.cfg.t0)
                    - reco(T_view, E0c, rbc, tref=self.cfg.tref, t0=self.cfg.t0)
                ) / step_dt
            else:
                f_phys = sde_drift(
                    T_view, E0c, rbc, tendency, tref=self.cfg.tref, t0=self.cfg.t0,
                )
        else:
            f_phys = None

        # Residual drift-misfit correction (also a per-minute rate).
        residual = self.residual_decoder(z) if self.residual_decoder is not None else None
        if f_phys is not None:
            drift_term = f_phys + residual if residual is not None else f_phys
        else:
            drift_term = residual  # physics off but residual on -> residual is the drift

        # Soft-bound the drift rate so an out-of-distribution head can't drive the
        # exponential physics drift to a catastrophic prediction (graceful OOD).
        if drift_term is not None and self.cfg.drift_clamp is not None:
            c = float(self.cfg.drift_clamp)
            drift_term = c * torch.tanh(drift_term / c)

        # Diffusion term. sigma_eff is the effective per-minute noise std used for
        # the parametric (nee_log_std) report; `noise` is a sample from the SAME
        # family the head is trained under (Gaussian / Student-t / ALD / zero-mean
        # mixture) so the sampled predictive law is consistent with the likelihood.
        # Physics-anchored heteroscedastic scale multiplier (Workstream C.1). Grows
        # the diffusion scale with the respiration magnitude Reco(T,E0,rb); Reco is
        # DETACHED so the noise scale reads respiration without perturbing the drift.
        sigma_mult = None
        if self.scale_anchor_a is not None and k_src is not None:
            reco_level = reco(
                T.view(-1, 1), k_src[:, 0:1], k_src[:, 1:2],
                tref=self.cfg.tref, t0=self.cfg.t0,
            ).detach().clamp(0.0, 50.0)
            sigma_mult = F.softplus(self.scale_anchor_a) + F.softplus(self.scale_anchor_b) * reco_level

        _mix_off = _mix_logscale = _mix_logits = None
        _sigma_meas = _sigma_proc = None   # state-space heads (for the gap-fill band eval)
        noise_is_increment = False   # state-space single-scale path sets this True
        if self.sigma_decoder is not None and self.n_mix > 1:
            # Zero-mean MIXTURE noise (Workstream F.2). Component offsets are centred
            # against the mixing weights so E[noise]=0 and the drift stays the mean.
            off = self.mix_offset_decoder(z)                                  # (n, K)
            logsc = self.mix_logscale_decoder(z).clamp(-7.0, 5.0)             # (n, K)
            logits = self.mix_logit_decoder(z)                               # (n, K)
            w = F.softmax(logits, dim=1)                                      # (n, K)
            off = off - (w * off).sum(dim=1, keepdim=True)                    # centre -> zero-mean
            # Physics-anchored scale: scale BOTH the centred offsets and the widths so
            # the whole mixture noise grows with Reco (stays zero-mean).
            if sigma_mult is not None:
                off = off * sigma_mult
                logsc = logsc + torch.log(sigma_mult)
            scales = torch.exp(logsc)
            idx = torch.multinomial(w, 1)
            noise = torch.gather(off, 1, idx) + torch.randn(x.shape[0], 1, device=self._device) * torch.gather(scales, 1, idx)
            noise_mu = torch.zeros_like(noise)
            mix_var = (w * (scales ** 2 + off ** 2)).sum(dim=1, keepdim=True)  # E[noise^2] (zero mean)
            sigma_eff = torch.sqrt(mix_var.clamp_min(1e-8))
            noise_logvar = 2.0 * torch.log(sigma_eff)
            _mix_off, _mix_logscale, _mix_logits = off, logsc, logits
        elif self.sigma_decoder is not None:
            if self.sigma_proc_decoder is not None:
                # State-space noise: measurement (flat) + Wiener process (∝ dt). sigma_eff
                # is the INCREMENT std directly (already carries dt), so the diffusion is
                # added to dnee WITHOUT the extra sqrt(dt) below (noise_is_increment).
                sigma_meas = F.softplus(self.sigma_decoder(z)) + _SIGMA_FLOOR
                sigma_proc = F.softplus(self.sigma_proc_decoder(z)) + _SIGMA_FLOOR
                if sigma_mult is not None:
                    sigma_meas = sigma_meas * sigma_mult
                    sigma_proc = sigma_proc * sigma_mult
                incr_var = 2.0 * sigma_meas ** 2 + sigma_proc ** 2 * step_dt
                sigma_eff = torch.sqrt(incr_var.clamp_min(1e-8))
                noise_is_increment = True
                _sigma_meas, _sigma_proc = sigma_meas, sigma_proc
            else:
                sigma_eff = F.softplus(self.sigma_decoder(z)) + _SIGMA_FLOOR
                if sigma_mult is not None:
                    sigma_eff = sigma_eff * sigma_mult
                noise_is_increment = False
            noise_mu = self.noise_mu_decoder(z) if self.noise_mu_decoder is not None else torch.zeros_like(sigma_eff)
            if self.log_kappa is not None:                     # asymmetric Laplace (ALD)
                kappa = torch.exp(self.log_kappa.clamp(LOG_KAPPA_MIN, LOG_KAPPA_MAX)).detach()
                eps = sample_ald_noise(sigma_eff.shape, self._device, kappa)
            elif self.log_nu is not None:                       # Student-t
                eps = draw_unit_noise(sigma_eff.shape, self._device, student_t_dof(self.log_nu).detach())
            else:                                               # Gaussian
                eps = draw_unit_noise(sigma_eff.shape, self._device, None)
            noise = noise_mu + eps * sigma_eff
            noise_logvar = 2.0 * torch.log(sigma_eff)
        else:
            sigma_eff = noise = noise_mu = noise_logvar = None
            noise_is_increment = False
        sigma = sigma_eff

        # Euler–Maruyama step: drift ~ dt, diffusion ~ sqrt(dt) (Wiener) or ~ the combined
        # increment std directly (state-space). step_dt/sqrt_dt were computed above.
        dnee = torch.zeros((x.shape[0], 1), device=self._device)
        if drift_term is not None:
            dnee = dnee + drift_term * step_dt
        if noise is not None:
            dnee = dnee + (noise if noise_is_increment else noise * sqrt_dt)

        nee_pred = b.view(-1, 1).to(self._device) + dnee

        # Likelihood sufficient statistics on the NEE_{t+1} target scale:
        #   nee_mean = deterministic prediction (bNEE + drift*dt, NO noise)
        #   nee_log_std = log(sigma) + 0.5*log(dt)   (increment std = sigma*sqrt(dt))
        det_increment = drift_term * step_dt if drift_term is not None else torch.zeros_like(dnee)
        nee_mean = b.view(-1, 1).to(self._device) + det_increment
        if sigma is None:
            nee_log_std = None
        elif noise_is_increment:
            nee_log_std = torch.log(sigma)               # sigma already the increment std
        else:
            nee_log_std = torch.log(sigma) + 0.5 * torch.log(step_dt)   # Wiener: sigma*sqrt(dt)

        # Zero-mean mixture noise -> per-component predictive law on the NEE_{t+1}
        # scale for the mixture NLL: location_k = nee_mean + sqrt(dt)*offset_k
        # (offsets already zero-mean, so the mixture mean == nee_mean == drift),
        # scale_k = sigma_k * sqrt(dt). compute_losses routes to mixture_nll when
        # mix_logits is present.
        if _mix_logits is not None:
            mix_means = nee_mean + sqrt_dt * _mix_off
            mix_log_scales = _mix_logscale + 0.5 * torch.log(step_dt)
            mix_logits = _mix_logits
        else:
            mix_means = mix_log_scales = mix_logits = None

        return {
            "latent": z,
            "latent_mu": latent_mu,
            "latent_logvar": latent_logvar,
            "k": k_pred,
            "temp_derivative": temp_derivative,
            # When the head learns the diurnal tendency, anchor it to dTa_diurnal instead
            # of observed dTa (None -> the loss falls back to observed dT).
            "temp_derivative_target": (
                dT_diurnal.view(-1, 1).to(self._device)
                if (self.cfg.drift_tendency == "learned_diurnal" and dT_diurnal is not None)
                else None
            ),
            "f_phys": f_phys,
            "residual": residual,
            "drift": drift_term,
            "sigma": sigma,
            "sigma_meas": _sigma_meas,   # state-space measurement scale (None otherwise)
            "sigma_proc": _sigma_proc,   # state-space Wiener process scale (None otherwise)
            "noise": noise,
            "noise_mu": noise_mu,
            "noise_logvar": noise_logvar,
            "dnee_pred": dnee,
            "nee_pred": nee_pred,
            "nee_mean": nee_mean,          # deterministic mean for the NLL loss
            "nee_log_std": nee_log_std,    # target-scale log-std for the NLL loss
            "log_nu": self.log_nu,         # Student-t dof (None unless enabled)
            "log_kappa": self.log_kappa,   # ALD asymmetry (None unless enabled)
            # zero-mean mixture noise params (None unless noise_mixture_components>1)
            "mix_means": mix_means,
            "mix_log_scales": mix_log_scales,
            "mix_logits": mix_logits,
            "bnee": None,  # no level reconstruction; keeps mse_bnee / mmd_bnee inert
        }

    def initialize(self) -> "IncrementSDEModel":
        self.apply(initialize_weights)
        return self
