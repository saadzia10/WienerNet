"""Post-training evaluation: metrics, temporal aggregations, cross-seed summaries."""

from .aggregations import (
    Resolution,
    add_time_buckets,
    aggregate_by_bucket,
    evaluate_at_resolutions,
)
from .metrics import (
    MetricBundle,
    compute_metric_bundle,
    evaluate_predictions,
    kl_divergence_histogram,
    mae,
    mmd_rbf,
    r2,
    rmse,
    wasserstein,
)
from .reporter import (
    cross_seed_pivot,
    metrics_to_long_dataframe,
    save_metrics_tables,
    standard_errors,
)

__all__ = [
    # metrics
    "MetricBundle",
    "compute_metric_bundle",
    "evaluate_predictions",
    "mae",
    "rmse",
    "r2",
    "mmd_rbf",
    "kl_divergence_histogram",
    "wasserstein",
    # aggregations
    "Resolution",
    "add_time_buckets",
    "aggregate_by_bucket",
    "evaluate_at_resolutions",
    # reporter
    "metrics_to_long_dataframe",
    "standard_errors",
    "cross_seed_pivot",
    "save_metrics_tables",
]
