"""Model definitions and factory."""

from .components import ReparamHead, build_mlp, build_mlp_with_head, initialize_weights
from .increment_sde import IncrementSDEConfig, IncrementSDEModel
from .process_baselines import (
    AnalyticalSDEConfig,
    AnalyticalSDEModel,
    HeteroscedasticMLPConfig,
    HeteroscedasticMLPModel,
    NeuralSDEConfig,
    NeuralSDEModel,
)
from .registry import (
    BASELINE_VARIANTS,
    INCREMENT_VARIANTS,
    KNOWN_VARIANTS,
    baseline_preset,
    build_baseline_model,
    build_increment_model,
    build_model,
    increment_preset,
    is_baseline_variant,
    is_increment_variant,
    preset,
)
from .wienernet import (
    HeadsConfig,
    InputsConfig,
    WienerNetConfig,
    WienerNetModel,
    encoder_input_dim,
)

__all__ = [
    # core
    "WienerNetModel",
    "WienerNetConfig",
    "HeadsConfig",
    "InputsConfig",
    "encoder_input_dim",
    # increment-SDE model
    "IncrementSDEModel",
    "IncrementSDEConfig",
    "build_increment_model",
    "increment_preset",
    "INCREMENT_VARIANTS",
    "is_increment_variant",
    # process-comparison baselines
    "AnalyticalSDEModel",
    "AnalyticalSDEConfig",
    "HeteroscedasticMLPModel",
    "HeteroscedasticMLPConfig",
    "NeuralSDEModel",
    "NeuralSDEConfig",
    "build_baseline_model",
    "baseline_preset",
    "BASELINE_VARIANTS",
    "is_baseline_variant",
    # factory
    "build_model",
    "preset",
    "KNOWN_VARIANTS",
    # building blocks
    "build_mlp",
    "build_mlp_with_head",
    "ReparamHead",
    "initialize_weights",
]
