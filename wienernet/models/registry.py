"""Named presets for the 5 model variants.

Each preset returns a fully-configured `WienerNetConfig`. The presets exist so
that experiment configs can just say `model: piae_sde_sampling` instead of
specifying every flag, but you can always override individual fields.
"""

from __future__ import annotations

from dataclasses import replace

from .increment_sde import IncrementSDEConfig, IncrementSDEModel
from .process_baselines import (
    AnalyticalSDEConfig,
    AnalyticalSDEModel,
    HeteroscedasticMLPConfig,
    HeteroscedasticMLPModel,
    NeuralSDEConfig,
    NeuralSDEModel,
)
from .wienernet import HeadsConfig, WienerNetConfig, WienerNetModel

# Variant names exactly match the original night/*.py class names so existing
# checkpoints and notebooks stay grep-able.
KNOWN_VARIANTS = (
    "piae_sde_sampling",
    "piae_sde_reg_sampling",
    "pivae_sde_sampling",
    "ae",
    "vae",
)

# Increment-based Euler–Maruyama variants (separate model class). See
# wienernet/models/increment_sde.py and docs/increment_sde_model_plan.md.
INCREMENT_VARIANTS = (
    "piae_increment_residual",   # A: residual + zero-mean aleatoric noise
    "piae_increment",            # B: no residual, learned-mean noise absorbs misfit
    "piae_reg_increment",        # C: deterministic drift only (PIAE-reg analog)
)


def is_increment_variant(variant: str) -> bool:
    """True iff `variant` is an increment-SDE variant (IncrementSDEModel)."""
    return variant in INCREMENT_VARIANTS


# Process-model comparison baselines (the ablation ladder — separate model
# classes in process_baselines.py). See docs and configs/model/<name>.yaml.
BASELINE_VARIANTS = (
    "analytical_sde",            # B1: physics drift + constant diffusion, no learning
    "hetero_mlp",                # B2a: no-physics heteroscedastic mean-variance head
    "hetero_mdn",                # B2b: no-physics mixture-density network (K components)
    "neural_sde",                # B3: data-driven neural drift + diffusion (no physics)
)

# Default mixture size for the MDN baseline (overridable via cfg.model).
_DEFAULT_MDN_COMPONENTS = 3


def is_baseline_variant(variant: str) -> bool:
    """True iff `variant` is a process-comparison baseline (process_baselines.py)."""
    return variant in BASELINE_VARIANTS


def preset(variant: str, *, input_dim: int, device: str = "cpu", **overrides) -> WienerNetConfig:
    """Return the canonical config for a known variant, optionally overridden.

    Args:
        variant: one of KNOWN_VARIANTS.
        input_dim: feature dimension of the encoder input (X + bNEE + k).
            For the standard 8-driver + 5-time + 3-site_xyz + bNEE + k pipeline
            this is 17 + 1 + 2 = 20.
        device: torch device string.
        **overrides: any WienerNetConfig field to override (e.g. latent_dim=64).

    Returns:
        A WienerNetConfig ready to pass to `WienerNetModel(cfg)`.
    """
    if variant not in KNOWN_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; valid: {KNOWN_VARIANTS}")

    base = WienerNetConfig(
        input_dim=input_dim,
        latent_dim=32,
        encoder_dims=(16, 16),
        decoder_dims=(16, 16),
        device=device,
    )

    if variant == "piae_sde_sampling":
        cfg = replace(
            base,
            activation="relu",
            latent_reparameterize=False,
            predict_drift=True,
            heads=HeadsConfig(temp_derivative=True, k=True, noise=True, k_activation="leaky_relu"),
        )
    elif variant == "piae_sde_reg_sampling":
        cfg = replace(
            base,
            activation="relu",
            latent_reparameterize=False,
            predict_drift=False,
            heads=HeadsConfig(temp_derivative=True, k=True, noise=True, k_activation="leaky_relu"),
        )
    elif variant == "pivae_sde_sampling":
        cfg = replace(
            base,
            activation="tanh",
            latent_reparameterize=True,
            predict_drift=True,
            heads=HeadsConfig(temp_derivative=True, k=True, noise=True, k_activation="leaky_relu"),
        )
    elif variant == "ae":
        cfg = replace(
            base,
            activation="relu",
            latent_reparameterize=False,
            predict_drift=False,
            heads=HeadsConfig(temp_derivative=False, k=True, noise=False, k_activation=None),
        )
    elif variant == "vae":
        cfg = replace(
            base,
            activation="tanh",
            latent_reparameterize=True,
            predict_drift=False,
            heads=HeadsConfig(temp_derivative=False, k=True, noise=False, k_activation=None),
        )

    # Apply user overrides on top of the preset
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown override {key!r}; WienerNetConfig fields: {list(cfg.__dataclass_fields__)}")
        cfg = replace(cfg, **{key: val})
    return cfg


def build_model(variant: str, *, input_dim: int, device: str = "cpu", **overrides) -> WienerNetModel:
    """Instantiate, initialize, and return a ready-to-train WienerNetModel."""
    cfg = preset(variant, input_dim=input_dim, device=device, **overrides)
    return WienerNetModel(cfg).initialize()


# ---------------------------------------------------------------------------
# Increment-SDE presets
# ---------------------------------------------------------------------------

def increment_preset(
    variant: str, *, input_dim: int, device: str = "cpu", **overrides
) -> IncrementSDEConfig:
    """Return the canonical IncrementSDEConfig for an increment variant.

    Args:
        variant: one of INCREMENT_VARIANTS.
        input_dim: encoder input width. For the default fully-exogenous inputs
            this equals the scaled feature width (e.g. 17); use
            `wienernet.models.encoder_input_dim(feature_dim, inputs)` when input
            flags are toggled on.
        device: torch device string.
        **overrides: any IncrementSDEConfig field to override.
    """
    if variant not in INCREMENT_VARIANTS:
        raise ValueError(f"Unknown increment variant {variant!r}; valid: {INCREMENT_VARIANTS}")

    base = IncrementSDEConfig(
        input_dim=input_dim,
        latent_dim=32,
        encoder_dims=(16, 16),
        decoder_dims=(16, 16),
        device=device,
    )

    if variant == "piae_increment_residual":       # A
        cfg = replace(base, residual=True, noise=True, noise_zero_mean=True)
    elif variant == "piae_increment":              # B
        cfg = replace(base, residual=False, noise=True, noise_zero_mean=False)
    elif variant == "piae_reg_increment":          # C
        cfg = replace(base, residual=False, noise=False)

    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(
                f"Unknown override {key!r}; IncrementSDEConfig fields: {list(cfg.__dataclass_fields__)}"
            )
        cfg = replace(cfg, **{key: val})
    return cfg


def build_increment_model(
    variant: str, *, input_dim: int, device: str = "cpu", **overrides
) -> IncrementSDEModel:
    """Instantiate, initialize, and return a ready-to-train IncrementSDEModel."""
    cfg = increment_preset(variant, input_dim=input_dim, device=device, **overrides)
    return IncrementSDEModel(cfg).initialize()


# ---------------------------------------------------------------------------
# Process-baseline presets (the ablation ladder)
# ---------------------------------------------------------------------------

def baseline_preset(variant: str, *, input_dim: int = 0, device: str = "cpu", **overrides):
    """Return the canonical config for a process-comparison baseline.

    Args:
        variant: one of BASELINE_VARIANTS.
        input_dim: encoder input width (unused by the analytical baseline, which
            has no encoder; kept for a uniform builder signature).
        device: torch device string.
        **overrides: any config field to override.
    """
    if variant not in BASELINE_VARIANTS:
        raise ValueError(f"Unknown baseline variant {variant!r}; valid: {BASELINE_VARIANTS}")

    if variant == "analytical_sde":
        cfg = AnalyticalSDEConfig(input_dim=input_dim, device=device)
    elif variant == "hetero_mlp":
        cfg = HeteroscedasticMLPConfig(input_dim=input_dim, mixture_components=1, device=device)
    elif variant == "hetero_mdn":
        cfg = HeteroscedasticMLPConfig(
            input_dim=input_dim, mixture_components=_DEFAULT_MDN_COMPONENTS, device=device)
    elif variant == "neural_sde":
        cfg = NeuralSDEConfig(input_dim=input_dim, device=device)
    else:  # pragma: no cover - guarded by the membership check above
        raise ValueError(f"No preset wired for baseline variant {variant!r}")

    for key, val in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(
                f"Unknown override {key!r}; {type(cfg).__name__} fields: {list(cfg.__dataclass_fields__)}"
            )
        cfg = replace(cfg, **{key: val})
    return cfg


def build_baseline_model(variant: str, *, input_dim: int = 0, device: str = "cpu", **overrides):
    """Instantiate, initialize, and return a ready-to-train process baseline."""
    cfg = baseline_preset(variant, input_dim=input_dim, device=device, **overrides)
    if variant == "analytical_sde":
        return AnalyticalSDEModel(cfg).initialize()
    if variant in ("hetero_mlp", "hetero_mdn"):
        return HeteroscedasticMLPModel(cfg).initialize()
    if variant == "neural_sde":
        return NeuralSDEModel(cfg).initialize()
    raise ValueError(f"No builder wired for baseline variant {variant!r}")  # pragma: no cover
