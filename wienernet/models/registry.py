"""Named presets for the 5 model variants.

Each preset returns a fully-configured `WienerNetConfig`. The presets exist so
that experiment configs can just say `model: piae_sde_sampling` instead of
specifying every flag, but you can always override individual fields.
"""

from __future__ import annotations

from dataclasses import replace

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
