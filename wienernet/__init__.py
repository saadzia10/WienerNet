"""WienerNet: physics-informed autoencoder for NEE gap-filling.

Top-level package. Subpackages:
    models      — WienerNetModel and components
    losses      — MMD + composite loss assembly
    physics     — torch Lloyd-Taylor (drift used in forward)
    training    — unified Trainer
    data        — Dataset + dataloader helpers
    evaluation  — metrics + temporal aggregations
    utils       — reproducibility, checkpointing, logging, paths
"""

__version__ = "0.2.0"
