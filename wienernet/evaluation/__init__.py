"""Post-training evaluation: metrics, temporal aggregations, cross-seed summaries, plots."""

from .aggregated import aggregated_scores
from .aggregations import (
    Resolution,
    add_time_buckets,
    aggregate_by_bucket,
    evaluate_at_resolutions,
)
from .metrics import (
    MetricBundle,
    bias,
    compute_metric_bundle,
    evaluate_predictions,
    kl_divergence_histogram,
    mae,
    mmd_rbf,
    r2,
    rmse,
    wasserstein,
)
from .probabilistic import (
    crps,
    crps_ensemble,
    crps_gaussian,
    diebold_mariano,
    ensemble_scores,
    interval_coverage,
    measurement_noise_floor,
    pit_summary,
    predictive_cdf,
    predictive_interval,
    predictive_logpdf,
    probabilistic_scores,
    sample_predictive,
    stratified_scores,
)
from .process_consistency import (
    drift_check,
    energy_distance,
    noise_check,
    standardized_residual_autocorr,
    variance_vs_scale,
)
from .calibration_plots import (
    descriptive_model,
    descriptive_site,
    emit_calibration_plots,
    plot_coverage_reliability,
    plot_crps_by_site,
    plot_pit_histograms,
    plot_residual_autocorrelation,
    plot_variance_scaling,
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
from .decomposition import (
    compute_decomposition,
    emit_manuscript_plots,
    pick_site_windows,
    plot_breakdown,
    plot_breakdown_mean,
    plot_decomposition_hists,
)
from .rollout import (
    assign_night_ids,
    decomposition_correlations,
    nightly_rollout,
)
from .timescale import increment_correlation_by_lag, plot_increment_correlation

__all__ = [
    # increment-SDE rollout + decomposition diagnostics
    "assign_night_ids", "nightly_rollout", "decomposition_correlations",
    # manuscript decomposition plots
    "compute_decomposition", "plot_breakdown", "plot_breakdown_mean",
    "plot_decomposition_hists", "pick_site_windows", "emit_manuscript_plots",
    # metrics
    "MetricBundle", "compute_metric_bundle", "evaluate_predictions",
    "mae", "rmse", "bias", "r2", "mmd_rbf", "kl_divergence_histogram", "wasserstein",
    # probabilistic (headline): CRPS / NLL / PIT / coverage / DM / noise floor
    "crps", "crps_gaussian", "crps_ensemble", "ensemble_scores", "predictive_logpdf",
    "predictive_cdf", "predictive_interval", "sample_predictive", "pit_summary",
    "interval_coverage", "diebold_mariano", "measurement_noise_floor",
    "probabilistic_scores", "stratified_scores",
    # process consistency
    "variance_vs_scale", "standardized_residual_autocorr", "drift_check",
    "noise_check", "energy_distance",
    # metric illustration plots (manuscript grade)
    "emit_calibration_plots", "plot_pit_histograms", "plot_coverage_reliability",
    "plot_variance_scaling", "plot_residual_autocorrelation", "plot_crps_by_site",
    "plot_aggregated_skill", "descriptive_site", "descriptive_model",
    # aggregations
    "Resolution", "add_time_buckets", "aggregate_by_bucket", "evaluate_at_resolutions",
    "aggregated_scores",
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
