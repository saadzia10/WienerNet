"""Post-training evaluation: metrics, temporal aggregations, cross-seed summaries, plots."""

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
from .plots import (
    COLORS,
    compute_gt_noise,
    daily_window_mask,
    monthly_window_mask,
    plot_all_temporal_scales,
    plot_feature_importance,
    plot_nee_daily,
    plot_nee_distribution_grid,
    plot_nee_monthly,
    plot_nee_quarterly,
    plot_nee_weekly,
    plot_noise_comparison_overall,
    plot_noise_comparison_per_site,
    plot_noise_distribution_grid,
    plot_sample_counts_per_site,
    quarterly_window_mask,
    setup_paper_style,
    weekly_window_mask,
)
from .reporter import (
    cross_seed_pivot,
    metrics_to_long_dataframe,
    parametric_summary,
    save_metrics_tables,
    standard_errors,
)
from .timescale import increment_correlation_by_lag, plot_increment_correlation

__all__ = [
    # metrics
    "MetricBundle", "compute_metric_bundle", "evaluate_predictions",
    "mae", "rmse", "r2", "mmd_rbf", "kl_divergence_histogram", "wasserstein",
    # aggregations
    "Resolution", "add_time_buckets", "aggregate_by_bucket", "evaluate_at_resolutions",
    # reporter
    "metrics_to_long_dataframe", "standard_errors", "parametric_summary",
    "cross_seed_pivot", "save_metrics_tables",
    "increment_correlation_by_lag", "plot_increment_correlation",
    # plots: style + helpers
    "setup_paper_style", "COLORS", "compute_gt_noise",
    "daily_window_mask", "weekly_window_mask", "monthly_window_mask", "quarterly_window_mask",
    # plots: data-level
    "plot_sample_counts_per_site", "plot_nee_distribution_grid", "plot_noise_distribution_grid",
    # plots: model-level
    "plot_feature_importance",
    "plot_nee_daily", "plot_nee_weekly", "plot_nee_monthly", "plot_nee_quarterly",
    "plot_all_temporal_scales",
    "plot_noise_comparison_overall", "plot_noise_comparison_per_site",
]
