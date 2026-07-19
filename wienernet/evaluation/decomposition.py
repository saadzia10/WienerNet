"""Model-aware SDE decomposition plots for the manuscript.

Answers "is the model's decomposition of NEE actually accurate?" by breaking the
prediction into its additive parts and showing them over time + as distributions.

Two model families are handled automatically (detected from the prediction keys):

* **Level models** (WienerNet PIAE / AE / VAE): ``nee_pred = nee_raw + noise + drift*dt``.
  Components on the NEE level: the deterministic part (``nee_pred - noise``), the
  physics curve ``NEE_phy = Reco(T)``, and the additive ``noise``.

* **Increment models** (IncrementSDE A/B/C): ``nee_pred = bNEE + (f_phys + r)*dt + noise*sqrt(dt)``.
  The *increment* ΔNEE decomposes into physics ``f_phys*dt``, residual ``r*dt`` and
  noise ``noise*sqrt(dt)``. The residual should carry the state-dependent physics
  misfit; the noise should be zero-mean aleatoric. That is exactly what these
  plots let you eyeball.

Entry point: ``emit_manuscript_plots(gt, preds, test_data, model_name, out_dir)``
— called per run by ``scripts/evaluate.py --plots``. The per-site random-window
picker (previously inline in notebooks/03_manuscript_plots.ipynb) lives here now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.features import physics_nee_numpy
from .plots import (
    COLORS,
    _format_window_dates,
    _maybe_save,
    daily_window_mask,
    plot_all_temporal_scales,
    setup_paper_style,
    weekly_window_mask,
)


# ---------------------------------------------------------------------------
# Component computation (model-aware)
# ---------------------------------------------------------------------------

def compute_decomposition(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    dt_column: str = "dt",
    boundary_column: str = "NEE",
    physics_column: str = "NEE_phy",
) -> dict[str, np.ndarray | bool]:
    """Return the additive decomposition components (all NEE-scale, length N).

    Keys always present: nee_gt, nee_pred, bnee, is_increment, gt_increment,
    pred_increment, det_pred (prediction without noise), noise_contrib.
    Increment models additionally get: phys_increment, resid_increment,
    noise_increment, det_increment, phys_pred. Level models get phys_pred
    (=NEE_phy) when E0/rb/Ta are available.
    """
    def _arr(d, k, default=None):
        if k in d and d[k] is not None:
            return np.asarray(d[k]).ravel().astype(float)
        return default

    nee_gt = _arr(gt, "nee")
    nee_pred = _arr(preds, "nee")
    n = len(nee_pred)
    dt = (test_data[dt_column].to_numpy().astype(float)
          if dt_column in test_data.columns else np.full(n, 30.0))
    bnee = (test_data[boundary_column].to_numpy().astype(float)
            if boundary_column in test_data.columns else _arr(preds, "bnee", nee_pred))

    is_increment = ("residual" in preds) or ("dnee" in preds)
    comp: dict[str, np.ndarray | bool] = {
        "is_increment": is_increment,
        "nee_gt": nee_gt,
        "nee_pred": nee_pred,
        "bnee": bnee,
        "gt_increment": nee_gt - bnee,
        "pred_increment": nee_pred - bnee,
    }

    zeros = np.zeros(n)
    noise = _arr(preds, "noise", zeros)          # raw reparam sample

    if is_increment:
        f = _arr(preds, "f", zeros)              # drift_term rate = f_phys + r
        residual = _arr(preds, "residual", zeros)
        f_phys = f - residual
        sqrt_dt = np.sqrt(dt)
        comp["phys_increment"] = f_phys * dt
        comp["resid_increment"] = residual * dt
        comp["noise_increment"] = noise * sqrt_dt
        comp["det_increment"] = f * dt                 # physics + residual, no noise
        comp["phys_pred"] = bnee + f_phys * dt
        comp["det_pred"] = bnee + f * dt               # nee_pred without the noise term
        comp["noise_contrib"] = comp["noise_increment"]
        # noise 1-sigma band on the increment (sigma*sqrt(dt))
        sig = _arr(preds, "noise_stds", None)
        comp["noise_sigma_incr"] = (sig * sqrt_dt) if sig is not None else None
    else:
        comp["det_pred"] = nee_pred - noise            # nee_raw + drift*dt
        comp["noise_contrib"] = noise
        comp["resid_increment"] = None
        # level physics curve Reco(T)
        if physics_column in test_data.columns:
            comp["phys_pred"] = test_data[physics_column].to_numpy().astype(float)
        elif {"E0", "rb", "Ta"}.issubset(test_data.columns):
            comp["phys_pred"] = physics_nee_numpy(
                test_data["E0"].values, test_data["rb"].values, test_data["Ta"].values)
        else:
            comp["phys_pred"] = None
        sig = _arr(preds, "noise_stds", None)
        comp["noise_sigma_incr"] = sig
    return comp


# ---------------------------------------------------------------------------
# Breakdown time-series (daily / weekly) — the two "versions" requested
# ---------------------------------------------------------------------------

def plot_breakdown(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    scale: str = "daily",
    group_id: int = 5,
    datetime_column: str = "DateTime",
    save_path: str | Path | None = None,
):
    """Decomposition breakdown over one window. `scale` in {'daily','weekly'}.

    Increment models: two stacked panels — NEE level (GT / pred / no-noise /
    physics) and the increment ΔNEE split into physics / +residual / noise band.
    Level models: a single NEE-level panel (GT / pred / no-noise / NEE_phy / CI).
    """
    setup_paper_style()
    if scale == "daily":
        mask = daily_window_mask(test_data, group_id=group_id, datetime_column=datetime_column)
    else:
        mask = weekly_window_mask(test_data, group_id=group_id, datetime_column=datetime_column)
    comp = compute_decomposition(gt, preds, test_data)
    date = _format_window_dates(pd.to_datetime(test_data[datetime_column])[mask])
    x = np.arange(int(mask.sum()))

    def m(key):
        v = comp.get(key)
        return None if v is None or isinstance(v, bool) else np.asarray(v)[mask]

    if comp["is_increment"]:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 10), dpi=130, sharex=True)
        # --- level panel ---
        ax0.plot(x, m("nee_gt"), "o-", color=COLORS["gt"], lw=2.5, ms=5, label="Ground truth")
        ax0.plot(x, m("nee_pred"), "o-", color=COLORS["pred"], lw=2.5, ms=5, label="Prediction")
        ax0.plot(x, m("det_pred"), "-", color=COLORS["pred_no_noise"], lw=2, label="Physics + residual (no noise)")
        ax0.plot(x, m("phys_pred"), "--", color=COLORS["physics"], lw=2, label="Physics only (bNEE + f_phys·dt)")
        ax0.set_ylabel("NEE"); ax0.legend(fontsize="medium"); ax0.set_title(
            f"{model_name}: decomposition — {scale} ({date})")
        # --- increment panel ---
        ax1.axhline(0, color="0.6", lw=1)
        ax1.plot(x, m("gt_increment"), "o-", color=COLORS["gt"], lw=2, ms=4, label="ΔNEE (observed)")
        ax1.plot(x, m("phys_increment"), "--", color=COLORS["physics"], lw=2, label="physics  f_phys·dt")
        ax1.plot(x, m("det_increment"), "-", color=COLORS["pred_no_noise"], lw=2, label="physics + residual  (f_phys+r)·dt")
        band = m("noise_sigma_incr")
        det = m("det_increment")
        if band is not None:
            ax1.fill_between(x, det - 1.96 * band, det + 1.96 * band,
                             color=COLORS["ci_fill"], alpha=0.2, label="±1.96·σ·√dt (noise)")
        ax1.set_ylabel("ΔNEE increment"); ax1.set_xlabel("timestep in window")
        ax1.legend(fontsize="medium")
    else:
        fig, ax0 = plt.subplots(figsize=(13, 5.5), dpi=130)
        ax0.plot(x, m("nee_gt"), "o-", color=COLORS["gt"], lw=2.5, ms=5, label="Ground truth")
        ax0.plot(x, m("nee_pred"), "o-", color=COLORS["pred"], lw=2.5, ms=5, label="Prediction")
        noise = m("noise_contrib")
        if noise is not None and np.any(noise != 0):
            ax0.plot(x, m("det_pred"), "-", color=COLORS["pred_no_noise"], lw=2, label="Prediction without noise")
        if m("phys_pred") is not None:
            ax0.plot(x, m("phys_pred"), "--", color=COLORS["physics"], lw=2, label="Physics  Reco(T)")
        band = m("noise_sigma_incr")
        if band is not None:
            pv = m("nee_pred")
            ax0.fill_between(x, pv - 1.96 * band, pv + 1.96 * band,
                             color=COLORS["ci_fill"], alpha=0.2, label="95% PI")
        ax0.set_ylabel("NEE"); ax0.set_xlabel("timestep in window")
        ax0.legend(fontsize="medium"); ax0.set_title(f"{model_name}: NEE — {scale} ({date})")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_breakdown_mean(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    scale: str = "daily",
    group_id: int = 5,
    datetime_column: str = "DateTime",
    save_path: str | Path | None = None,
):
    """Point-prediction view: the deterministic MEAN + a 95% predictive band.

    `plot_breakdown` draws a single noisy *sample* (nee_mean + noise·√dt), which
    looks jumpy and overstates the error. This companion draws the **mean**
    (the estimate the point-accuracy metrics actually use) with a shaded
    ±1.96·σ band from the learned scale — so the picture matches the reported
    R²/calibration. For a well-calibrated model the band should cover ~95% of
    the observed NEE; for the old overconfident model it's a thin sliver.
    """
    setup_paper_style()
    if scale == "daily":
        mask = daily_window_mask(test_data, group_id=group_id, datetime_column=datetime_column)
    else:
        mask = weekly_window_mask(test_data, group_id=group_id, datetime_column=datetime_column)
    comp = compute_decomposition(gt, preds, test_data)
    date = _format_window_dates(pd.to_datetime(test_data[datetime_column])[mask])
    x = np.arange(int(mask.sum()))

    def m(key):
        v = comp.get(key)
        return None if v is None or isinstance(v, bool) else np.asarray(v)[mask]

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=130)
    ax.plot(x, m("nee_gt"), "o-", color=COLORS["gt"], lw=2.5, ms=5, label="Ground truth")
    mean = m("det_pred")
    ax.plot(x, mean, "-", color=COLORS["pred"], lw=2.5, label="Mean prediction (no noise)")
    band = m("noise_sigma_incr")
    if band is not None and mean is not None:
        ax.fill_between(x, mean - 1.96 * band, mean + 1.96 * band,
                        color=COLORS["ci_fill"], alpha=0.25, label="95% predictive band")
    if m("phys_pred") is not None:
        ax.plot(x, m("phys_pred"), "--", color=COLORS["physics"], lw=1.8, label="Physics only")
    ax.set_ylabel("NEE"); ax.set_xlabel("timestep in window")
    ax.legend(fontsize="medium")
    ax.set_title(f"{model_name}: mean + 95% band — {scale} ({date})")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Component histograms + state-dependence (accuracy of the decomposition)
# ---------------------------------------------------------------------------

def plot_decomposition_hists(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    bins: int = 80,
    temp_column: str = "Ta",
    save_path: str | Path | None = None,
):
    """Distributions of the decomposition components + their Ta-dependence.

    Increment models (the accuracy check): (1) the physics misfit ΔNEE − f_phys·dt
    that residual+noise must jointly explain, overlaid with the model's residual+
    noise; (2) residual vs noise distributions; (3) mean residual & noise binned by
    Ta — the residual should carry the Ta-slope (the misfit), the noise should be
    flat and zero-mean. Level models fall back to predicted-noise vs GT-noise.
    """
    setup_paper_style()
    comp = compute_decomposition(gt, preds, test_data)
    ta = test_data[temp_column].to_numpy().astype(float) if temp_column in test_data.columns else None

    def _finite(a):
        a = np.asarray(a, float); return a[np.isfinite(a)]

    if comp["is_increment"]:
        gt_misfit = comp["gt_increment"] - comp["phys_increment"]
        resid = comp["resid_increment"]
        noise = comp["noise_increment"]
        model_expl = resid + noise
        fig, axs = plt.subplots(1, 3, figsize=(19, 5.2), dpi=130)
        # (1) misfit vs model explanation
        ax = axs[0]
        ax.hist(_finite(gt_misfit), bins=bins, density=True, alpha=0.5,
                color=COLORS["noise_gt"], label="GT misfit  ΔNEE − f_phys·dt")
        ax.hist(_finite(model_expl), bins=bins, density=True, alpha=0.5,
                color=COLORS["noise_pred"], label="model  r·dt + noise·√dt")
        ax.axvline(0, color="0.5", lw=1); ax.legend(fontsize="small")
        ax.set_title("Physics misfit: GT vs model"); ax.set_xlabel("ΔNEE units")
        # (2) residual vs noise components
        ax = axs[1]
        ax.hist(_finite(resid), bins=bins, density=True, alpha=0.55,
                color="tab:green", label=f"residual r·dt (mean {np.nanmean(resid):+.3f})")
        ax.hist(_finite(noise), bins=bins, density=True, alpha=0.55,
                color="tab:orange", label=f"noise·√dt (mean {np.nanmean(noise):+.3f})")
        ax.axvline(0, color="0.5", lw=1); ax.legend(fontsize="small")
        ax.set_title("Residual vs noise"); ax.set_xlabel("ΔNEE units")
        # (3) Ta-dependence: mean residual & noise per Ta-decile
        ax = axs[2]
        if ta is not None:
            df = pd.DataFrame({"ta": ta, "resid": resid, "noise": noise}).dropna()
            df["bin"] = pd.qcut(df["ta"], q=10, duplicates="drop")
            g = df.groupby("bin", observed=True)
            centers = g["ta"].mean().to_numpy()
            ax.plot(centers, g["resid"].mean().to_numpy(), "o-", color="tab:green",
                    label=f"residual (corr {np.corrcoef(df.ta, df.resid)[0,1]:+.2f})")
            ax.plot(centers, g["noise"].mean().to_numpy(), "s-", color="tab:orange",
                    label=f"noise (corr {np.corrcoef(df.ta, df.noise)[0,1]:+.2f})")
            ax.axhline(0, color="0.5", lw=1); ax.legend(fontsize="small")
            ax.set_title("Mean component vs Ta"); ax.set_xlabel("Ta (°C)"); ax.set_ylabel("mean ΔNEE units")
        else:
            ax.axis("off")
        fig.suptitle(f"{model_name}: increment decomposition accuracy", fontsize=15)
    else:
        # Level model: predicted noise vs GT noise (NEE − NEE_phy)
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), dpi=130)
        noise = comp["noise_contrib"]
        gt_noise = None
        if {"E0", "rb", "Ta", "NEE"}.issubset(test_data.columns):
            gt_noise = test_data["NEE"].values - physics_nee_numpy(
                test_data["E0"].values, test_data["rb"].values, test_data["Ta"].values)
        ax = axs[0]
        if gt_noise is not None:
            ax.hist(_finite(gt_noise), bins=bins, density=True, alpha=0.5,
                    color=COLORS["noise_gt"], label="GT noise (NEE − NEE_phy)")
        ax.hist(_finite(noise), bins=bins, density=True, alpha=0.5,
                color=COLORS["noise_pred"], label=f"pred noise (mean {np.nanmean(noise):+.3f})")
        ax.axvline(0, color="0.5", lw=1); ax.legend(fontsize="small")
        ax.set_title("Noise: predicted vs GT"); ax.set_xlabel("NEE units")
        ax = axs[1]
        if ta is not None and np.any(noise != 0):
            df = pd.DataFrame({"ta": ta, "noise": noise}).dropna()
            df["bin"] = pd.qcut(df["ta"], q=10, duplicates="drop")
            g = df.groupby("bin", observed=True)
            ax.plot(g["ta"].mean().to_numpy(), g["noise"].mean().to_numpy(), "s-",
                    color="tab:orange", label=f"noise (corr {np.corrcoef(df.ta, df.noise)[0,1]:+.2f})")
            ax.axhline(0, color="0.5", lw=1); ax.legend(fontsize="small")
            ax.set_title("Mean predicted noise vs Ta"); ax.set_xlabel("Ta (°C)")
        else:
            ax.axis("off")
        fig.suptitle(f"{model_name}: noise decomposition", fontsize=15)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Per-site random window picker (extracted from 03_manuscript_plots.ipynb)
# ---------------------------------------------------------------------------

def _daily_groups(dt: pd.Series, gap_hours: int = 2) -> pd.Series:
    return (dt.diff() > pd.Timedelta(hours=gap_hours)).cumsum()


def _weekly_groups(dt: pd.Series) -> pd.Series:
    dow = dt.dt.dayofweek
    return (dow < dow.shift()).cumsum()


def pick_site_windows(
    td_site: pd.DataFrame,
    rng: np.random.Generator,
    *,
    datetime_column: str = "DateTime",
    min_daily_points: int = 20,
) -> dict:
    """Choose a random populated window at each scale for one site (reproducible).

    Returns per-scale kwargs so the temporal/breakdown plots land on real data for
    this site's test span. Prefers windows with enough points; falls back to the
    largest available window.
    """
    dt = pd.to_datetime(td_site[datetime_column])

    def _choice(labels: pd.Series, min_size: int):
        counts = labels.value_counts()
        eligible = counts[counts >= min_size].index.to_numpy()
        if len(eligible) == 0:
            eligible = counts.index.to_numpy()[:1]
        return rng.choice(eligible)

    daily_group = int(_choice(_daily_groups(dt), min_daily_points))
    weekly_group = int(_choice(_weekly_groups(dt), 5))
    ym = dt.dt.strftime("%Y-%m")
    ym_pick = str(_choice(ym, 10)); year_m, month = int(ym_pick[:4]), int(ym_pick[5:7])
    if "season" in td_site.columns:
        yq = dt.dt.year.astype(str) + "-" + td_site["season"].astype(int).astype(str)
        yq_pick = str(_choice(yq, 10)); year_q, season = (int(p) for p in yq_pick.split("-"))
    else:
        year_q, season = year_m, 0
    return {
        "daily_group": daily_group, "weekly_group": weekly_group,
        "monthly_kwargs": dict(year=year_m, month=month),
        "quarterly_kwargs": dict(year=year_q, season=season),
    }


# ---------------------------------------------------------------------------
# Orchestration — called by evaluate.py
# ---------------------------------------------------------------------------

def emit_manuscript_plots(
    gt: Mapping[str, np.ndarray],
    preds: Mapping[str, np.ndarray],
    test_data: pd.DataFrame,
    *,
    model_name: str,
    out_dir: str | Path,
    temporal_random_seed: int = 42,
    min_daily_points: int = 20,
    per_site: bool = True,
) -> list[Path]:
    """Write the full manuscript plot set for one evaluated run.

    Per site: the 4 temporal-scale NEE plots + daily & weekly decomposition
    breakdowns + the component histograms. Also an overall (all-sites) histogram.
    Returns the list of written PNG paths. Robust: a failing site/plot is logged
    and skipped, never aborts the run.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    td = test_data.reset_index(drop=True)
    if "NEE_phy" not in td.columns and {"E0", "rb", "Ta"}.issubset(td.columns):
        td = td.copy()
        td["NEE_phy"] = physics_nee_numpy(td["E0"].values, td["rb"].values, td["Ta"].values)

    def _save(fig, path):
        try:
            written.append(Path(path))
        finally:
            plt.close(fig)

    # Overall (all sites) decomposition histogram — the headline accuracy check.
    try:
        p = out_dir / "decomp_hist_overall.png"
        fig = plot_decomposition_hists(gt, preds, td, model_name=model_name, save_path=p)
        _save(fig, p)
    except Exception as exc:  # noqa: BLE001
        print(f"    [plot warn] {model_name} overall hist: {exc}")

    sites = sorted(td["site"].unique()) if (per_site and "site" in td.columns) else [None]
    for site_idx, site in enumerate(sites):
        if site is None:
            td_s, gt_s, preds_s, tag = td, gt, preds, "all"
        else:
            mask = (td["site"] == site).to_numpy()
            if mask.sum() < 5:
                continue
            td_s = td[mask].reset_index(drop=True)
            gt_s = {k: np.asarray(v)[mask] for k, v in gt.items()}
            preds_s = {k: np.asarray(v)[mask] for k, v in preds.items()}
            tag = str(site)
        rng = np.random.default_rng(temporal_random_seed + site_idx)
        try:
            win = pick_site_windows(td_s, rng, min_daily_points=min_daily_points)
        except Exception as exc:  # noqa: BLE001
            print(f"    [plot warn] {model_name}/{tag} window pick: {exc}")
            continue
        sd = out_dir / tag
        sd.mkdir(parents=True, exist_ok=True)
        # 4 temporal-scale NEE plots
        try:
            figs = plot_all_temporal_scales(
                gt_s, preds_s, td_s, model_name=f"{model_name} ({tag})",
                save_dir=sd, save_prefix=f"nee_{tag}",
                daily_kwargs=dict(group_id=win["daily_group"]),
                weekly_kwargs=dict(group_id=win["weekly_group"]),
                monthly_kwargs=win["monthly_kwargs"],
                quarterly_kwargs=win["quarterly_kwargs"],
            )
            for scale, fig in figs.items():
                written.append(sd / f"nee_{tag}_{scale}.png"); plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print(f"    [plot warn] {model_name}/{tag} temporal: {exc}")
        # daily + weekly decomposition breakdowns: the sampled-draw view AND the
        # mean + 95%-band view (point estimate + honest uncertainty).
        for scale, gid in (("daily", win["daily_group"]), ("weekly", win["weekly_group"])):
            try:
                p = sd / f"breakdown_{tag}_{scale}.png"
                fig = plot_breakdown(gt_s, preds_s, td_s, model_name=f"{model_name} ({tag})",
                                     scale=scale, group_id=gid, save_path=p)
                _save(fig, p)
            except Exception as exc:  # noqa: BLE001
                print(f"    [plot warn] {model_name}/{tag} breakdown {scale}: {exc}")
            try:
                p = sd / f"breakdown_mean_{tag}_{scale}.png"
                fig = plot_breakdown_mean(gt_s, preds_s, td_s, model_name=f"{model_name} ({tag})",
                                          scale=scale, group_id=gid, save_path=p)
                _save(fig, p)
            except Exception as exc:  # noqa: BLE001
                print(f"    [plot warn] {model_name}/{tag} breakdown_mean {scale}: {exc}")
        # per-site component histogram
        try:
            p = sd / f"decomp_hist_{tag}.png"
            fig = plot_decomposition_hists(gt_s, preds_s, td_s, model_name=f"{model_name} ({tag})", save_path=p)
            _save(fig, p)
        except Exception as exc:  # noqa: BLE001
            print(f"    [plot warn] {model_name}/{tag} hist: {exc}")

    return written
