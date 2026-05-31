"""Manuscript-quality plotting functions.

Reproduces the exact visual style from the original
`piae_sde/All_Seed_Experiments_Night.ipynb` plots and adds two new plots:
  - `plot_noise_comparison_overall`: predicted vs ground-truth noise histogram
  - `plot_noise_comparison_per_site`: same, broken down per site

Every function is stateless, accepts numpy/pandas inputs, returns the matplotlib
figure for further composition, and optionally takes `save_path` for one-line
artifact saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ..data.features import physics_nee_numpy

# ---------------------------------------------------------------------------
# Style constants — preserved verbatim from the original notebook.
# Changing these will change manuscript figure aesthetics.
# ---------------------------------------------------------------------------

COLORS = {
    "gt": "#1f77b4",            # matplotlib tab:blue
    "pred": "#ff7f0e",          # matplotlib tab:orange
    "pred_no_noise": "green",
    "physics": "red",
    "noise_gt": "tab:blue",
    "noise_pred": "tab:orange",
    "bar_site": "#4F81BD",
    "ci_fill": "#ff7f0e",
}

DEFAULT_DPI_GRID = 300
DEFAULT_DPI_TIMESERIES = 400
DEFAULT_DPI_FEATURE = 200

PAPER_STYLE = "whitegrid"


def setup_paper_style() -> None:
    """Apply the seaborn whitegrid style used in the original notebook."""
    sns.set_style(PAPER_STYLE)


def _maybe_save(fig: Figure, save_path: str | Path | None, dpi: int | None = None) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)


def _format_window_dates(timestamps: pd.Series | np.ndarray) -> str:
    """Format the date range visible in a window selection for the plot title."""
    if len(timestamps) == 0:
        return "no data"
    first = pd.to_datetime(timestamps.iloc[0] if hasattr(timestamps, "iloc") else timestamps[0])
    last = pd.to_datetime(timestamps.iloc[-1] if hasattr(timestamps, "iloc") else timestamps[-1])
    if first.date() == last.date():
        return str(first.date())
    return f"{first.date()} till {last.date()}"


# ===========================================================================
# Data-level plots (no model required)
# ===========================================================================

def plot_sample_counts_per_site(
    counts: Mapping[str, int],
    *,
    title: str = "Number of Samples per Site",
    save_path: str | Path | None = None,
) -> Figure:
    """Bar chart of dataset size per site. Mirrors cell 18 in the original notebook."""
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=DEFAULT_DPI_GRID)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    pd.Series(counts).plot(kind="bar", ax=ax, color=COLORS["bar_site"], width=0.5)
    ax.set_ylabel("Number of Samples")
    ax.set_xlabel("Site")
    ax.set_title(title)
    plt.xticks(rotation=30)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_nee_distribution_grid(
    site_dfs: Mapping[str, pd.DataFrame],
    *,
    nee_col: str = "NEE",
    phy_col: str = "NEE_phy",
    bins: int = 100,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Grid of NEE vs NEE_phy histograms, one panel per site (cell 19 style)."""
    setup_paper_style()
    n_sites = len(site_dfs)
    nrows = (n_sites + ncols - 1) // ncols
    fig_size = figsize or (5 * ncols, 5 * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=fig_size, dpi=DEFAULT_DPI_GRID, squeeze=False)
    axes_flat = axs.flatten()

    for ax, (site_name, df) in zip(axes_flat, site_dfs.items()):
        if nee_col not in df.columns:
            ax.set_title(f"{site_name} (missing {nee_col})"); ax.axis("off"); continue
        if phy_col not in df.columns:
            df = df.copy()
            df[phy_col] = physics_nee_numpy(df["E0"].values, df["rb"].values, df["Ta"].values)

        sns.histplot(df[nee_col], bins=bins, alpha=0.5, label=nee_col, kde=True,
                     ax=ax, stat="density", color=COLORS["noise_gt"])
        sns.histplot(df[phy_col], bins=bins, alpha=0.5, label=phy_col, kde=True,
                     ax=ax, stat="density", color=COLORS["noise_pred"])
        ax.set_title(site_name)
        ax.legend()

    # Hide unused subplots
    for ax in axes_flat[len(site_dfs):]:
        ax.axis("off")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_noise_distribution_grid(
    site_dfs: Mapping[str, pd.DataFrame],
    *,
    nee_col: str = "NEE",
    phy_col: str = "NEE_phy",
    bins: int = 100,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Per-site histograms of `NEE_phy - NEE` (the ground-truth noise residual).
    Mirrors cell 20 of the original notebook.
    """
    setup_paper_style()
    n_sites = len(site_dfs)
    nrows = (n_sites + ncols - 1) // ncols
    fig_size = figsize or (5 * ncols, 5 * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=fig_size, dpi=DEFAULT_DPI_GRID, squeeze=False)
    axes_flat = axs.flatten()

    for ax, (site_name, df) in zip(axes_flat, site_dfs.items()):
        if phy_col not in df.columns:
            df = df.copy()
            df[phy_col] = physics_nee_numpy(df["E0"].values, df["rb"].values, df["Ta"].values)
        residual = df[phy_col] - df[nee_col]
        sns.histplot(residual.dropna(), bins=bins, alpha=0.5, label=site_name,
                     kde=True, ax=ax, stat="density")
        ax.set_title(site_name)

    for ax in axes_flat[len(site_dfs):]:
        ax.axis("off")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ===========================================================================
# Model-level plots: feature importance
# ===========================================================================

def plot_feature_importance(
    features: Iterable[str],
    importances: np.ndarray,
    *,
    top_n: int | None = None,
    title: str = "Feature Importances",
    save_path: str | Path | None = None,
) -> Figure:
    """Horizontal bar chart of feature importances (cell 72 style)."""
    setup_paper_style()
    df = pd.DataFrame({"Feature": list(features), "Importance": np.asarray(importances)})
    df = df.sort_values("Importance", ascending=False)
    if top_n is not None:
        df = df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DEFAULT_DPI_FEATURE)
    sns.barplot(x="Importance", y="Feature", data=df, ax=ax, color=COLORS["bar_site"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ===========================================================================
# Temporal-window mask helpers
# Each returns a boolean mask aligned to `test_data`'s rows.
# ===========================================================================

def daily_window_mask(
    test_data: pd.DataFrame,
    *,
    group_id: int = 5,
    gap_hours: int = 2,
    datetime_column: str = "DateTime",
) -> np.ndarray:
    """Boolean mask selecting all rows of the Nth contiguous daily window.

    A new daily window starts every time the gap between consecutive timestamps
    exceeds `gap_hours`. The original notebook uses 2h (the night-time gap).
    """
    df = test_data.copy()
    df["_time_diff"] = pd.to_datetime(df[datetime_column]).diff()
    df["_daily_group"] = (df["_time_diff"] > pd.Timedelta(hours=gap_hours)).cumsum()
    return (df["_daily_group"] == group_id).to_numpy()


def weekly_window_mask(
    test_data: pd.DataFrame,
    *,
    group_id: int = 4,
    datetime_column: str = "DateTime",
) -> np.ndarray:
    """Boolean mask selecting all rows of the Nth contiguous week.

    A new week starts when the day-of-week wraps (typical sensible week boundary).
    """
    df = test_data.copy()
    dt = pd.to_datetime(df[datetime_column])
    dow = dt.dt.dayofweek
    week_group = (dow < dow.shift()).cumsum()
    return (week_group == group_id).to_numpy()


def monthly_window_mask(
    test_data: pd.DataFrame,
    *,
    year: int = 2018,
    month: int = 9,
    datetime_column: str = "DateTime",
) -> np.ndarray:
    dt = pd.to_datetime(test_data[datetime_column])
    return ((dt.dt.year == year) & (dt.dt.month == month)).to_numpy()


def quarterly_window_mask(
    test_data: pd.DataFrame,
    *,
    year: int = 2018,
    season: int = 4,
    datetime_column: str = "DateTime",
    season_column: str = "season",
) -> np.ndarray:
    """Boolean mask for a season-of-year window (preserves the notebook's
    `season == 4` convention)."""
    dt = pd.to_datetime(test_data[datetime_column])
    if season_column not in test_data.columns:
        raise KeyError(f"season column {season_column!r} not in test_data")
    return ((dt.dt.year == year) & (test_data[season_column] == season)).to_numpy()


# ===========================================================================
# Time-series plots (one per temporal scale)
# Each takes gt + preds dicts (np arrays) + a row mask and produces a panel
# in the manuscript style.
# ===========================================================================

def _ts_axes(figsize=(12, 5), dpi=DEFAULT_DPI_TIMESERIES) -> tuple[Figure, plt.Axes]:
    setup_paper_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", which="major", labelsize=25)
    return fig, ax


def plot_nee_daily(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    group_id: int = 5,
    show_predictions_no_noise: bool = False,
    show_physics: bool = False,
    show_ci: bool = False,
    physics_column: str = "NEE_phy",
    datetime_column: str = "DateTime",
    save_path: str | Path | None = None,
) -> Figure:
    """Daily-scale NEE time-series plot (cells 82 / 95 style).

    Args:
        show_predictions_no_noise: also draw `pred_nee - pred_noise` (green line).
            Only meaningful when the model has a noise head (WienerNet variants).
        show_physics: also draw NEE_phy (red line) — requires `physics_column`
            in `test_data`.
        show_ci: shade a 95% prediction interval using `preds['noise_stds']`.
            Only meaningful for the WienerNet variants.
    """
    mask = daily_window_mask(test_data, group_id=group_id, datetime_column=datetime_column)
    gt_vis = np.asarray(gt["nee"])[mask]
    preds_vis = np.asarray(preds["nee"])[mask].flatten()
    date = _format_window_dates(pd.to_datetime(test_data[datetime_column])[mask])

    fig, ax = _ts_axes()
    handles, labels = [], []
    h_gt, = ax.plot(gt_vis, "o-", color=COLORS["gt"], linewidth=4, markersize=8)
    h_p,  = ax.plot(preds_vis, "o-", color=COLORS["pred"], linewidth=4, markersize=8)
    handles += [h_gt, h_p]; labels += ["Ground Truth", "Predictions"]

    if show_predictions_no_noise and "noise" in preds:
        preds_no_noise = preds_vis - np.asarray(preds["noise"])[mask].flatten()
        h_pn, = ax.plot(preds_no_noise, "o-", color=COLORS["pred_no_noise"],
                        linewidth=4, markersize=8)
        handles.append(h_pn); labels.append("Predictions without noise")

    if show_physics and physics_column in test_data.columns:
        phy = test_data[physics_column].to_numpy()[mask]
        h_phy, = ax.plot(phy, "o-", color=COLORS["physics"], linewidth=4, markersize=8)
        handles.append(h_phy); labels.append("Physics")

    if show_ci and "noise_stds" in preds:
        std = np.asarray(preds["noise_stds"])[mask]
        z = 1.96
        lo = preds_vis - z * std
        hi = preds_vis + z * std
        ax.fill_between(np.arange(len(preds_vis)), lo, hi,
                        color=COLORS["ci_fill"], alpha=0.2,
                        label="95% Prediction Interval")
        handles.append(ax.collections[-1]); labels.append("95% Prediction Interval")

    ax.legend(handles, labels, fontsize="x-large")
    ax.set_ylabel("NEE", fontsize=25)
    ax.set_title(f"{model_name}: NEE Daily Scale ({date})", fontsize=30)
    _maybe_save(fig, save_path)
    return fig


def plot_nee_weekly(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    group_id: int = 4,
    show_residual_ci: bool = False,
    datetime_column: str = "DateTime",
    save_path: str | Path | None = None,
) -> Figure:
    """Weekly-scale NEE time-series plot (cells 83 / 96 style)."""
    mask = weekly_window_mask(test_data, group_id=group_id, datetime_column=datetime_column)
    gt_vis = np.asarray(gt["nee"])[mask]
    preds_vis = np.asarray(preds["nee"])[mask].flatten()
    date = _format_window_dates(pd.to_datetime(test_data[datetime_column])[mask])

    fig, ax = _ts_axes()
    ax.plot(gt_vis, "-", color=COLORS["gt"], linewidth=4, markersize=8)
    ax.plot(preds_vis, "-", color=COLORS["pred"], linewidth=4, markersize=8)

    if show_residual_ci:
        residuals = gt_vis - preds_vis
        sigma = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0
        z = 1.96
        lo = preds_vis - z * sigma
        hi = preds_vis + z * sigma
        ax.fill_between(np.arange(len(preds_vis)), lo, hi,
                        color=COLORS["ci_fill"], alpha=0.2,
                        label="95% Prediction Interval")

    ax.legend(["Ground Truth", "Predictions"], fontsize="x-large")
    ax.set_ylabel("NEE", fontsize=25)
    ax.set_title(f"{model_name}: NEE Weekly Scale ({date})", fontsize=25)
    _maybe_save(fig, save_path)
    return fig


def _smoothed_ts(
    fig_ax: tuple[Figure, plt.Axes],
    gt_vis: np.ndarray,
    preds_vis: np.ndarray,
    *,
    window_size: int = 10,
) -> None:
    fig, ax = fig_ax
    gt_s = pd.Series(gt_vis).rolling(window=window_size, min_periods=1).mean()
    pr_s = pd.Series(preds_vis).rolling(window=window_size, min_periods=1).mean()
    ax.plot(gt_vis, alpha=0.4, color=COLORS["gt"], linewidth=2)
    ax.plot(preds_vis, alpha=0.4, color=COLORS["pred"], linewidth=2)
    ax.plot(gt_s, color=COLORS["gt"], label="Ground Truth", linewidth=4)
    ax.plot(pr_s, color=COLORS["pred"], label="Predictions", linewidth=4)
    ax.legend(fontsize="x-large")
    ax.set_ylabel("NEE", fontsize=25)


def plot_nee_monthly(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    year: int = 2018,
    month: int = 9,
    window_size: int = 10,
    datetime_column: str = "DateTime",
    save_path: str | Path | None = None,
) -> Figure:
    """Monthly-scale NEE plot with rolling-mean smoothing (cells 84 / 97 style)."""
    mask = monthly_window_mask(test_data, year=year, month=month, datetime_column=datetime_column)
    gt_vis = np.asarray(gt["nee"])[mask]
    preds_vis = np.asarray(preds["nee"])[mask].flatten()
    date = _format_window_dates(pd.to_datetime(test_data[datetime_column])[mask])

    fig, ax = _ts_axes()
    _smoothed_ts((fig, ax), gt_vis, preds_vis, window_size=window_size)
    ax.set_title(f"{model_name}: NEE Monthly Scale ({date})", fontsize=25)
    _maybe_save(fig, save_path)
    return fig


def plot_nee_quarterly(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    year: int = 2018,
    season: int = 4,
    window_size: int = 10,
    datetime_column: str = "DateTime",
    season_column: str = "season",
    save_path: str | Path | None = None,
) -> Figure:
    """Quarterly-scale NEE plot with rolling-mean smoothing (cells 85 / 98 style)."""
    mask = quarterly_window_mask(
        test_data, year=year, season=season,
        datetime_column=datetime_column, season_column=season_column,
    )
    gt_vis = np.asarray(gt["nee"])[mask]
    preds_vis = np.asarray(preds["nee"])[mask].flatten()
    date = _format_window_dates(pd.to_datetime(test_data[datetime_column])[mask])

    fig, ax = _ts_axes()
    _smoothed_ts((fig, ax), gt_vis, preds_vis, window_size=window_size)
    ax.set_title(f"{model_name}: NEE Quarterly Scale ({date})", fontsize=25)
    _maybe_save(fig, save_path)
    return fig


# ===========================================================================
# Convenience: emit all 4 temporal-scale plots in one call
# ===========================================================================

def plot_all_temporal_scales(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    daily_kwargs: dict | None = None,
    weekly_kwargs: dict | None = None,
    monthly_kwargs: dict | None = None,
    quarterly_kwargs: dict | None = None,
    save_dir: str | Path | None = None,
    save_prefix: str | None = None,
) -> dict[str, Figure]:
    """Run daily / weekly / monthly / quarterly NEE plots in one call.

    Args:
        save_dir: if set, write each plot as `<save_dir>/<save_prefix>_<scale>.png`.
        save_prefix: filename prefix; defaults to `model_name.lower().replace(' ', '_')`.
    """
    save_dir = Path(save_dir) if save_dir else None
    prefix = save_prefix or model_name.lower().replace(" ", "_")

    def _p(scale: str) -> Path | None:
        return save_dir / f"{prefix}_{scale}.png" if save_dir else None

    return {
        "daily":     plot_nee_daily(    gt, preds, test_data, model_name=model_name, save_path=_p("daily"),     **(daily_kwargs or {})),
        "weekly":    plot_nee_weekly(   gt, preds, test_data, model_name=model_name, save_path=_p("weekly"),    **(weekly_kwargs or {})),
        "monthly":   plot_nee_monthly(  gt, preds, test_data, model_name=model_name, save_path=_p("monthly"),   **(monthly_kwargs or {})),
        "quarterly": plot_nee_quarterly(gt, preds, test_data, model_name=model_name, save_path=_p("quarterly"), **(quarterly_kwargs or {})),
    }


# ===========================================================================
# NEW: predicted-noise vs ground-truth-noise comparison plots
# ===========================================================================

def compute_gt_noise(
    test_data: pd.DataFrame,
    *,
    nee_col: str = "NEE",
    e0_col: str = "E0",
    rb_col: str = "rb",
    t_col: str = "Ta",
) -> np.ndarray:
    """Ground-truth noise residual: NEE - NEE_phy.

    Uses the partitioner-compatible Lloyd-Taylor formula so this matches
    the model's in-graph physics term.
    """
    nee_phy = physics_nee_numpy(test_data[e0_col].values, test_data[rb_col].values, test_data[t_col].values)
    return test_data[nee_col].values - nee_phy


def plot_noise_comparison_overall(
    gt_noise: np.ndarray,
    pred_noise: np.ndarray,
    *,
    model_name: str,
    bins: int = 100,
    figsize: tuple[float, float] = (8, 5),
    save_path: str | Path | None = None,
) -> Figure:
    """Histogram of predicted noise overlaid with ground-truth noise.

    Useful diagnostic for WienerNet variants — if the model has correctly
    learned the noise distribution, the two histograms should overlap.
    """
    setup_paper_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI_GRID)
    gt_arr = np.asarray(gt_noise).ravel()
    pred_arr = np.asarray(pred_noise).ravel()
    gt_arr = gt_arr[np.isfinite(gt_arr)]
    pred_arr = pred_arr[np.isfinite(pred_arr)]

    sns.histplot(gt_arr, bins=bins, alpha=0.5, kde=True, ax=ax,
                 stat="density", color=COLORS["noise_gt"], label="GT noise (NEE − NEE_phy)")
    sns.histplot(pred_arr, bins=bins, alpha=0.5, kde=True, ax=ax,
                 stat="density", color=COLORS["noise_pred"], label="Predicted noise")
    ax.set_title(f"{model_name}: Predicted vs Ground-Truth Noise")
    ax.set_xlabel("Noise (umol m^-2 s^-1)")
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_noise_comparison_per_site(
    test_data: pd.DataFrame,
    gt_noise: np.ndarray,
    pred_noise: np.ndarray,
    *,
    model_name: str,
    site_column: str = "site",
    bins: int = 100,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """One-panel-per-site grid of predicted-vs-GT noise histograms."""
    setup_paper_style()
    if site_column not in test_data.columns:
        raise KeyError(f"site column {site_column!r} not in test_data")

    if len(test_data) != len(gt_noise) or len(test_data) != len(pred_noise):
        raise ValueError(
            f"length mismatch: test_data={len(test_data)}, "
            f"gt_noise={len(gt_noise)}, pred_noise={len(pred_noise)}"
        )

    sites = sorted(test_data[site_column].unique())
    n_sites = len(sites)
    nrows = (n_sites + ncols - 1) // ncols
    fig_size = figsize or (5 * ncols, 4 * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=fig_size, dpi=DEFAULT_DPI_GRID, squeeze=False)
    axes_flat = axs.flatten()

    gt_arr = np.asarray(gt_noise).ravel()
    pred_arr = np.asarray(pred_noise).ravel()
    site_arr = test_data[site_column].to_numpy()

    for ax, site in zip(axes_flat, sites):
        site_mask = site_arr == site
        gt_site = gt_arr[site_mask]
        pred_site = pred_arr[site_mask]
        gt_site = gt_site[np.isfinite(gt_site)]
        pred_site = pred_site[np.isfinite(pred_site)]

        if len(gt_site) == 0 or len(pred_site) == 0:
            ax.set_title(f"{site} (no data)"); ax.axis("off"); continue

        sns.histplot(gt_site, bins=bins, alpha=0.5, kde=True, ax=ax,
                     stat="density", color=COLORS["noise_gt"], label="GT noise")
        sns.histplot(pred_site, bins=bins, alpha=0.5, kde=True, ax=ax,
                     stat="density", color=COLORS["noise_pred"], label="Predicted noise")
        ax.set_title(f"{site} (n={site_mask.sum()})")
        ax.legend(fontsize="small")

    for ax in axes_flat[n_sites:]:
        ax.axis("off")

    fig.suptitle(f"{model_name}: Predicted vs GT Noise per Site", fontsize=14, y=1.01)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
