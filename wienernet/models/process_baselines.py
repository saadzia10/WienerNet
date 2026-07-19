"""Process-model comparison baselines (the ablation ladder).

Three baselines that each remove exactly one ingredient of the full increment
model (physics drift + learned misfit + learned state-dependent noise + Euler
integration), so together they show which ingredient earns its place:

  1. AnalyticalSDEModel  — physics drift + CONSTANT noise, NO learning.
     The textbook stochastic model: Lloyd-Taylor drift (the same dReco/dT * dTa
     used by the main model) integrated one Euler-Maruyama step, plus a constant
     diffusion level (one GLOBAL value, or one PER SITE) estimated from the spread
     of the training increment misfit. The special case of the full model with
     zero mean correction and constant diffusion — the cheapest baseline and the
     most informative for the core claim (what the learned misfit + state-
     dependent noise add over textbook physics with constant noise).

  2. HeteroscedasticMLPModel — flexible distribution, NO physics / NO SDE.  (B2)
  3. NeuralSDEModel          — learned drift + learned noise, NO physics.    (B3)

All three share the IncrementSDEModel forward signature
``forward(x, b, k, T, dt=None, dT=None, site=None)`` and emit the SAME output-dict
contract (``nee_mean`` / ``nee_log_std`` / ``nee_pred`` / ``dnee_pred`` / ``noise``
/ ``latent`` ...), so they drop straight into the unified Trainer, the likelihood
loss (``compute_losses``), ``Trainer.predict`` / ``predict_ensemble`` and the
probabilistic + process-consistency + calibration evaluation suite with no
special-casing — the fair, matched-capacity, identically-scored comparison the
manuscript needs.

Fairness (shared): same drivers, same chronological split, same normalisation,
same one-step NEE_{t+1} target and scale, same Euler step where applicable, each
emits a predictive distribution scored by CRPS/NLL/PIT/coverage, the neural ones
trained with the same likelihood objective, capacity matched to the main model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..losses.likelihood import student_t_dof
from ..physics.lloyd_taylor import DEFAULT_T0, DEFAULT_TREF, sde_drift
from .components import build_mlp, build_mlp_with_head, draw_unit_noise, initialize_weights
from .wienernet import InputsConfig, _resolve_activation

# Minimum diffusion scale so sigma stays strictly positive and log(sigma) finite
# (matches IncrementSDEModel._SIGMA_FLOOR).
_SIGMA_FLOOR = 1e-4

# Clamp the predictive log-scale (matches the likelihood loss's LOG_STD_MIN/MAX) so
# an out-of-distribution row can't exp-overflow the scale head into a huge sigma —
# the loss already clamps internally, this keeps the *sampled* prediction sane too.
_LOG_SCALE_MIN, _LOG_SCALE_MAX = -7.0, 5.0


def _site_index_tensor(site, site_to_idx: dict[str, int], device: torch.device) -> torch.Tensor:
    """Map a per-row site identifier (list/tuple/ndarray/tensor of names) to a
    LongTensor of indices; unknown or missing names map to -1 (global fallback)."""
    names = [str(s) for s in site]
    idx = [site_to_idx.get(nm, -1) for nm in names]
    return torch.tensor(idx, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Baseline 1 — Analytical SDE (constant-diffusion, textbook)
# ---------------------------------------------------------------------------

@dataclass
class AnalyticalSDEConfig:
    """Config for AnalyticalSDEModel (Baseline 1).

    The model has NO encoder and NO learned drift — the drift is pure physics from
    the ground-truth Lloyd-Taylor parameters (E0, rb) and ground-truth dTa, exactly
    the drift the main model integrates. Its only free quantity is a constant
    diffusion level: one GLOBAL scalar (``per_site=False``) or one per site
    (``per_site=True``), estimated from the training increment-misfit spread via
    ``calibrate()``.

    ``input_dim`` / ``inputs`` are accepted for a uniform builder signature but are
    unused (there is no encoder). ``site_names`` fixes the per-site parameter layout
    so checkpoints round-trip (it must match at train and eval time — the train/eval
    scripts derive it deterministically from the data config's site list).
    """

    input_dim: int = 0                       # unused (no encoder); kept for a uniform builder
    per_site: bool = False                   # one constant sigma per site vs one global
    site_names: tuple[str, ...] = ()         # fixed per-site layout (empty -> global only)
    # Student-t heavy-tailed likelihood option: add a learnable dof scalar so the
    # analytical baseline can be trained/scored with the student_t NLL too. Off by
    # default -> Gaussian constant-diffusion, the textbook version.
    noise_student_dof: bool = False
    inputs: InputsConfig = field(default_factory=InputsConfig)   # unused; uniform builder

    # Physics constants + Euler step (identical semantics to IncrementSDEConfig).
    tref: float = DEFAULT_TREF
    t0: float = DEFAULT_T0
    dt: float = 30.0
    # Initial constant diffusion level (overwritten by calibrate()); a sane
    # positive default so an un-calibrated model is still well-defined.
    init_sigma: float = 1.0
    device: str = "cpu"


class AnalyticalSDEModel(nn.Module):
    """Textbook constant-diffusion SDE: physics drift + constant sigma.

    Prediction (per row, one Euler-Maruyama step of size dt):
        f_phys   = dReco/dT(T, E0, rb) * dTa          # GT physics drift RATE
        dNEE     = f_phys*dt + sigma*sqrt(dt)*eps      # eps ~ N(0,1)
        nee_pred = bNEE + dNEE
    with the likelihood sufficient statistics on the NEE_{t+1} scale:
        nee_mean    = bNEE + f_phys*dt                 # deterministic mean
        nee_log_std = log(sigma) + 0.5*log(dt)         # increment std = sigma*sqrt(dt)

    ``sigma`` is a constant diffusion coefficient — one GLOBAL value, or one PER
    SITE when ``per_site`` is set (selected per row from the ``site`` argument).
    ``calibrate()`` sets it to the dt-normalised RMS of the training increment
    misfit, which is exactly the Gaussian-NLL MLE, so likelihood training leaves it
    put and the model is well-defined whether calibrated, trained 0 epochs, or
    trained to convergence.
    """

    def __init__(self, cfg: AnalyticalSDEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.device)
        init = max(float(cfg.init_sigma), _SIGMA_FLOOR)
        log_init = math.log(init)
        # Global constant diffusion level (also the per-site fallback for unknown
        # sites): the one free scalar of the textbook model.
        self.log_sigma = nn.Parameter(torch.tensor(log_init, dtype=torch.float32))
        # Per-site diffusion levels (a parallel constant per site), when enabled.
        self._site_to_idx: dict[str, int] = {s: i for i, s in enumerate(cfg.site_names)}
        if cfg.per_site:
            if not cfg.site_names:
                raise ValueError("per_site=True requires a non-empty site_names layout")
            self.log_sigma_site = nn.Parameter(
                torch.full((len(cfg.site_names),), log_init, dtype=torch.float32)
            )
        else:
            self.log_sigma_site = None
        # Student-t dof (nu = softplus(log_nu)+1); None unless enabled.
        self.log_nu = (
            nn.Parameter(torch.tensor(float(math.log(math.expm1(7.0)))))
            if cfg.noise_student_dof else None
        )
        self.to(self._device)

    # ------------------------------------------------------------------
    # Per-row log-sigma selection
    # ------------------------------------------------------------------

    def _row_log_sigma(self, n: int, site) -> torch.Tensor:
        """Return the (n, 1) per-row log-sigma: the per-site value where enabled
        and the site is known, else the global scalar."""
        if self.cfg.per_site and self.log_sigma_site is not None and site is not None:
            idx = _site_index_tensor(site, self._site_to_idx, self._device)
            row = self.log_sigma.expand(n).clone()
            known = idx >= 0
            if known.any():
                row = row.clone()
                row[known] = self.log_sigma_site[idx[known]]
            return row.view(n, 1)
        return self.log_sigma.expand(n, 1)

    # ------------------------------------------------------------------
    # Calibration: the constant diffusion coefficient(s)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate(self, loader: DataLoader) -> "AnalyticalSDEModel":
        """Set the constant diffusion coefficient(s) from the training increment misfit.

        The model integrates one Euler-Maruyama step, so the diffusion level enters
        the predictive increment std as ``sigma*sqrt(dt)``. The correct constant
        diffusion coefficient is therefore the dt-normalised spread of the increment
        misfit (obs increment - physics increment):

            sigma^2 = mean[ ((NEE_{t+1} - NEE_t) - f_phys*dt)^2 / dt ]

        computed globally and (when ``per_site``) within each site. This dt-
        normalised estimate is exactly the Gaussian-NLL MLE for ``log_sigma`` (the
        NLL derivative is ``1 - resid^2/(sigma^2 dt)``, zero in the mean here), so
        training under the Gaussian likelihood leaves the calibrated sigma put.
        Idempotent; safe to call before ``fit``.
        """
        tot_sum, tot_n = 0.0, 0
        site_sum: dict[str, float] = {s: 0.0 for s in self.cfg.site_names}
        site_n: dict[str, int] = {s: 0 for s in self.cfg.site_names}
        for batch in loader:
            b = batch["bNEE"].to(self._device).view(-1, 1)
            k = batch["k"].to(self._device)
            T = batch["T"].to(self._device).view(-1, 1)
            dT = batch["dT"].to(self._device).view(-1, 1)
            nee = batch["NEE"].to(self._device).view(-1, 1)
            step_dt = (batch["dt"].to(self._device).view(-1, 1)
                       if "dt" in batch else torch.full_like(b, float(self.cfg.dt)))
            f_phys = sde_drift(T, k[:, 0:1], k[:, 1:2], dT, tref=self.cfg.tref, t0=self.cfg.t0)
            resid = (nee - b) - f_phys * step_dt          # obs increment - physics increment
            per_dt = (resid ** 2 / step_dt).view(-1)      # dt-normalised -> diffusion coefficient
            finite = torch.isfinite(per_dt)
            tot_sum += float(per_dt[finite].sum().cpu())
            tot_n += int(finite.sum().cpu())
            if self.cfg.per_site and "site_id" in batch:
                names = [str(s) for s in batch["site_id"]]
                vals = per_dt.detach().cpu()
                fin = finite.detach().cpu()
                for i, nm in enumerate(names):
                    if nm in site_sum and bool(fin[i]):
                        site_sum[nm] += float(vals[i]); site_n[nm] += 1
        if tot_n == 0:
            raise ValueError("calibrate() saw no finite increment residuals")
        g_sigma = max(math.sqrt(tot_sum / tot_n), _SIGMA_FLOOR)
        self.log_sigma.data = torch.tensor(math.log(g_sigma), dtype=torch.float32, device=self._device)
        if self.cfg.per_site and self.log_sigma_site is not None:
            for s, i in self._site_to_idx.items():
                # per-site MLE where the site has data, else fall back to global
                sig = math.sqrt(site_sum[s] / site_n[s]) if site_n[s] > 0 else g_sigma
                sig = max(sig, _SIGMA_FLOOR)
                self.log_sigma_site.data[i] = math.log(sig)
        return self

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
        device = self._device
        n = x.shape[0]
        b = b.view(-1, 1).to(device)
        k = k.to(device)
        T = T.view(-1, 1).to(device)
        if dT is None:
            raise ValueError("AnalyticalSDEModel needs ground-truth dTa (dT) for the physics drift.")
        dT = dT.view(-1, 1).to(device)

        # Ground-truth physics drift RATE (per minute) — the same drift the main
        # model integrates, but with no learned correction.
        f_phys = sde_drift(T, k[:, 0:1], k[:, 1:2], dT, tref=self.cfg.tref, t0=self.cfg.t0)

        # Euler step: drift ~ dt, diffusion ~ sqrt(dt). Per-row dt when given.
        step_dt = (dt.view(-1, 1).to(device) if dt is not None
                   else torch.full((n, 1), float(self.cfg.dt), device=device))
        sqrt_dt = torch.sqrt(step_dt)

        row_log_sigma = self._row_log_sigma(n, site)      # (n, 1) global or per-site
        sigma = torch.exp(row_log_sigma)                  # constant within site
        noise = torch.randn(n, 1, device=device) * sigma  # zero-mean diffusion sample

        det_increment = f_phys * step_dt
        dnee = det_increment + noise * sqrt_dt
        nee_pred = b + dnee
        nee_mean = b + det_increment
        nee_log_std = row_log_sigma + 0.5 * torch.log(step_dt)

        return {
            "latent": torch.zeros(n, 1, device=device),   # no encoder; keeps trainer.predict happy
            "latent_mu": None,
            "latent_logvar": None,
            "k": None,
            "temp_derivative": None,
            "f_phys": f_phys,
            "residual": None,
            "drift": f_phys,                               # the (pure physics) drift rate
            "sigma": sigma,
            "noise": noise,
            "noise_mu": torch.zeros(n, 1, device=device),
            "noise_logvar": 2.0 * row_log_sigma,
            "dnee_pred": dnee,
            "nee_pred": nee_pred,
            "nee_mean": nee_mean,
            "nee_log_std": nee_log_std,
            "log_nu": self.log_nu,
            "bnee": None,                                  # no level reconstruction
        }

    def initialize(self) -> "AnalyticalSDEModel":
        # Nothing to random-initialise (sigma is set by calibrate / init_sigma);
        # method kept for a uniform build_*_model(...).initialize() interface.
        return self


# ---------------------------------------------------------------------------
# Baseline 2 — No-physics heteroscedastic model (mean-variance / mixture-density)
# ---------------------------------------------------------------------------

@dataclass
class HeteroscedasticMLPConfig:
    """Config for HeteroscedasticMLPModel (Baseline 2).

    A black-box conditional density over the next-step NEE increment, with NO
    physics, NO SDE structure (no drift, no dt / sqrt(dt) Euler scaling, no misfit
    term). The trunk mirrors the main model's encoder/decoder widths so capacity is
    comparable and any difference reflects *structure*, not size.

    ``mixture_components`` = 1 gives the mean-variance head (a single Gaussian /
    Student-t per row, which rides the exact parametric scoring). > 1 gives a
    mixture-density head (K Gaussians / Student-t), trained with the mixture NLL and
    scored honestly via the ensemble (sampling) axis — a single (mean, scale) cannot
    represent a mixture, so its parametric report is only a moment-matched summary.
    """

    input_dim: int
    latent_dim: int = 32
    encoder_dims: tuple[int, ...] = (16, 16)
    decoder_dims: tuple[int, ...] = (16, 16)
    activation: str = "relu"
    mixture_components: int = 1                  # 1 = mean-variance; >1 = MDN
    noise_student_dof: bool = False             # Student-t components (adds log_nu)
    inputs: InputsConfig = field(default_factory=InputsConfig)   # unused; uniform builder
    dt: float = 30.0                            # unused (no SDE); kept for a uniform builder
    device: str = "cpu"


class HeteroscedasticMLPModel(nn.Module):
    """No-physics heteroscedastic network / mixture-density network (Baseline 2).

    Maps the same drivers ``x`` straight to a predictive distribution over the
    next-step increment, anchored on the observed boundary:
        z        = encoder(x)
        Δμ, logσ = mean_head(z), log_scale_head(z)          # (n, K) each
        NEE_{t+1} ~ mixture_k softmax(logits_k) * comp(bNEE + Δμ_k, σ_k)
    No drift, no physics, no Euler dt-scaling — the scale directly models the
    increment spread. This isolates whether embedding physics + SDE structure buys
    anything over a flexible heteroscedastic conditional density.

    Emits the shared output contract: ``nee_mean`` / ``nee_log_std`` (the mixture's
    moment-matched mean/std for K>1, exact for K=1), a sampled ``nee_pred`` /
    ``dnee_pred`` / ``noise`` (so the ensemble axis scores it honestly), and, for
    K>1, the mixture parameters (``mix_means`` / ``mix_log_scales`` / ``mix_logits``)
    that drive the mixture NLL.
    """

    # keeps sqrt of a mixture variance finite for the moment-matched log-std
    _VAR_FLOOR = 1e-8

    def __init__(self, cfg: HeteroscedasticMLPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.device)
        if cfg.mixture_components < 1:
            raise ValueError("mixture_components must be >= 1")
        act = _resolve_activation(cfg.activation)
        K = cfg.mixture_components

        modules = build_mlp(cfg.input_dim, cfg.encoder_dims, act)
        modules.append(nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim))
        self.encoder = nn.Sequential(*modules)

        # Two (mean, log-scale) heads over K components; a mixing-logit head for K>1.
        self.mean_decoder = build_mlp_with_head(cfg.latent_dim, cfg.decoder_dims, K, activation=act)
        self.log_scale_decoder = build_mlp_with_head(cfg.latent_dim, cfg.decoder_dims, K, activation=act)
        self.logit_decoder = (
            build_mlp_with_head(cfg.latent_dim, cfg.decoder_dims, K, activation=act) if K > 1 else None
        )
        self.log_nu = (
            nn.Parameter(torch.tensor(float(math.log(math.expm1(7.0)))))
            if cfg.noise_student_dof else None
        )
        self.to(self._device)

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
        device = self._device
        n = x.shape[0]
        x = x.to(device)
        b = b.view(-1, 1).to(device)

        z = self.encoder(x)
        comp_means = b + self.mean_decoder(z)              # (n, K) on NEE_{t+1} scale
        log_scales = self.log_scale_decoder(z).clamp(_LOG_SCALE_MIN, _LOG_SCALE_MAX)   # (n, K)
        scales = torch.exp(log_scales)
        K = self.cfg.mixture_components
        nu = student_t_dof(self.log_nu).detach() if self.log_nu is not None else None

        if K == 1:
            nee_mean = comp_means                          # (n, 1)
            nee_log_std = log_scales
            nee_pred = nee_mean + draw_unit_noise((n, 1), device, nu) * scales
            mix_means = mix_log_scales = mix_logits = None
        else:
            logits = self.logit_decoder(z)                 # (n, K)
            w = F.softmax(logits, dim=1)
            # moment-matched Gaussian summary (for point metrics + parametric report;
            # the honest scoring for a mixture is the ensemble axis)
            mean = (w * comp_means).sum(dim=1, keepdim=True)
            var = (w * (scales ** 2 + comp_means ** 2)).sum(dim=1, keepdim=True) - mean ** 2
            nee_mean = mean
            nee_log_std = 0.5 * torch.log(var.clamp_min(self._VAR_FLOOR))
            # draw one mixture sample per row (component ~ w, then its Gaussian/t)
            idx = torch.multinomial(w, 1)                  # (n, 1)
            chosen_mean = torch.gather(comp_means, 1, idx)
            chosen_scale = torch.gather(scales, 1, idx)
            nee_pred = chosen_mean + draw_unit_noise((n, 1), device, nu) * chosen_scale
            mix_means, mix_log_scales, mix_logits = comp_means, log_scales, logits

        dnee = nee_pred - b
        noise = nee_pred - nee_mean                        # stochastic part (triggers ensemble)

        return {
            "latent": z,
            "latent_mu": None,
            "latent_logvar": None,
            "k": None,
            "temp_derivative": None,
            "f_phys": None,
            "residual": None,
            "drift": None,                                 # no physics / no drift rate
            "sigma": torch.exp(nee_log_std),
            "noise": noise,
            "noise_mu": torch.zeros(n, 1, device=device),
            "noise_logvar": 2.0 * nee_log_std,
            "dnee_pred": dnee,
            "nee_pred": nee_pred,
            "nee_mean": nee_mean,
            "nee_log_std": nee_log_std,
            "log_nu": self.log_nu,
            # mixture parameters for the mixture NLL (present only for K > 1)
            "mix_means": mix_means,
            "mix_log_scales": mix_log_scales,
            "mix_logits": mix_logits,
            "bnee": None,
        }

    def initialize(self) -> "HeteroscedasticMLPModel":
        self.apply(initialize_weights)
        return self


# ---------------------------------------------------------------------------
# Baseline 3 — Neural SDE (data-driven drift and diffusion)
# ---------------------------------------------------------------------------

@dataclass
class NeuralSDEConfig:
    """Config for NeuralSDEModel (Baseline 3).

    A direct (observation-space) neural SDE: both the drift and the diffusion are
    free neural functions of the state, with NO physics prior, integrated with the
    same one-step Euler-Maruyama scheme as the full model. Trunk widths mirror the
    main model so capacity is comparable.
    """

    input_dim: int
    latent_dim: int = 32
    encoder_dims: tuple[int, ...] = (16, 16)
    decoder_dims: tuple[int, ...] = (16, 16)
    activation: str = "relu"
    noise_student_dof: bool = False             # Student-t noise (adds log_nu)
    inputs: InputsConfig = field(default_factory=InputsConfig)   # unused; uniform builder
    tref: float = DEFAULT_TREF                  # unused (no physics); uniform builder
    t0: float = DEFAULT_T0                      # unused (no physics); uniform builder
    dt: float = 30.0
    device: str = "cpu"


class NeuralSDEModel(nn.Module):
    """Data-driven neural SDE (Baseline 3): free neural drift + diffusion, no physics.

    Both the drift and the diffusion scale are neural functions of the state,
    integrated by the same one-step Euler-Maruyama scheme the full model uses:
        z        = encoder(x)
        g(z)     = drift_decoder(z)                    # free neural drift RATE
        σ(z)     = softplus(sigma_decoder(z))          # free neural diffusion scale
        ΔNEE     = g(z)·dt + σ(z)·√dt·ε ,  ε ~ 𝒩(0,1)
        NEE_{t+1}= NEE_t + ΔNEE
    trained on observed next-step values with the same likelihood objective. It
    keeps the SDE dt / √dt structure (unlike the no-physics heteroscedastic model)
    but replaces the physics drift with a black box (unlike the analytical SDE) and
    has no misfit decomposition (unlike the full model). Substantiates the critique
    of purely data-driven neural SDEs: the drift carries no physical parameters and
    the diffusion tends to be under-estimated.
    """

    def __init__(self, cfg: NeuralSDEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(cfg.device)
        act = _resolve_activation(cfg.activation)

        modules = build_mlp(cfg.input_dim, cfg.encoder_dims, act)
        modules.append(nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim))
        self.encoder = nn.Sequential(*modules)

        self.drift_decoder = build_mlp_with_head(cfg.latent_dim, cfg.decoder_dims, 1, activation=act)
        self.sigma_decoder = build_mlp_with_head(cfg.latent_dim, cfg.decoder_dims, 1, activation=act)
        self.log_nu = (
            nn.Parameter(torch.tensor(float(math.log(math.expm1(7.0)))))
            if cfg.noise_student_dof else None
        )
        self.to(self._device)

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
        device = self._device
        n = x.shape[0]
        x = x.to(device)
        b = b.view(-1, 1).to(device)

        z = self.encoder(x)
        drift = self.drift_decoder(z)                          # free neural drift RATE
        sigma = F.softplus(self.sigma_decoder(z)) + _SIGMA_FLOOR

        step_dt = (dt.view(-1, 1).to(device) if dt is not None
                   else torch.full((n, 1), float(self.cfg.dt), device=device))
        sqrt_dt = torch.sqrt(step_dt)

        nu = student_t_dof(self.log_nu).detach() if self.log_nu is not None else None
        noise = draw_unit_noise((n, 1), device, nu) * sigma    # zero-mean diffusion sample
        det_increment = drift * step_dt
        dnee = det_increment + noise * sqrt_dt
        nee_pred = b + dnee
        nee_mean = b + det_increment
        nee_log_std = torch.log(sigma) + 0.5 * torch.log(step_dt)

        return {
            "latent": z,
            "latent_mu": None,
            "latent_logvar": None,
            "k": None,
            "temp_derivative": None,
            "f_phys": None,
            "residual": None,
            "drift": drift,                                     # free neural drift rate
            "sigma": sigma,
            "noise": noise,
            "noise_mu": torch.zeros(n, 1, device=device),
            "noise_logvar": 2.0 * torch.log(sigma),
            "dnee_pred": dnee,
            "nee_pred": nee_pred,
            "nee_mean": nee_mean,
            "nee_log_std": nee_log_std,
            "log_nu": self.log_nu,
            "bnee": None,
        }

    def initialize(self) -> "NeuralSDEModel":
        self.apply(initialize_weights)
        return self
