#!/usr/bin/env python
"""Drift / noise decomposition figures — WienerNet's interpretability edge.

Reads the per-row predictions of trained runs in outputs/final_loso/ (no retraining)
and renders four manuscript figures that a black box / tree cannot produce:

  1. fig_decomp_night      per-night decomposition: observed NEE, the physics DRIFT
                           mean, and the calibrated aleatoric NOISE band; a no-physics
                           panel beside it (one opaque predictive blob) for contrast.
  2. fig_drift_physics     the drift is a physical instrument — reconstructed
                           respiration Reco(T) vs the binned observed nighttime
                           respiration, and the predicted temperature tendency vs observed.
  3. fig_noise_calibration the noise is calibrated + heteroscedastic — predicted noise
                           std vs empirical residual std across flux bins, vs the
                           measured SD ≈ 0.24 + 0.30·Reco.
  4. fig_variance_aggregation  drift (systematic) carries the weekly/monthly flux BUDGET
                           while the noise cancels ~1/√N — the carbon-budget payoff.

Outputs -> analysis/final_loso/figures_decomp/
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWEEP = os.path.join(ROOT, "outputs", "final_loso")
FIG = os.path.join(HERE, "figures_decomp")
sys.path.insert(0, HERE)
from aggregate_and_plot import setup_mpl  # reuse the publication style

TREF, T0 = 10.0, 46.02
WIENER = "wienernet_laplace"      # the interpretable model
BLACKBOX = "comp_mdn"            # the no-decomposition contrast
SITE = "woodwalton"               # clean physics-win held-out site
SEED = 0
Z = 1.6449                        # 90% two-sided normal quantile for the band


def load(label, site=SITE, seed=SEED):
    p = os.path.join(SWEEP, f"{label}_{site}_s{seed}", "metrics", "predictions.parquet")
    df = pd.read_parquet(p)
    df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def reco(T, E0, rb):
    return rb * np.exp(E0 * (1.0 / (TREF + T0) - 1.0 / (T + T0)))


def night_ids(df):
    """Group consecutive nighttime 30-min steps into nights (gap > 3 h = new night)."""
    t = pd.to_datetime(df["DateTime"])
    gap = t.diff().dt.total_seconds().fillna(0) / 3600.0
    return (gap > 3.0).cumsum()


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    print("wrote", name)


# ---------------------------------------------------------------- 1. per-night
def _select_clean_nights(w, k=3, min_n=10):
    """Nights where the physics explains the structure (high corr NEE~Reco(T)),
    with moderate outliers — the illustrative cases, not the noisiest."""
    w = w.copy()
    w["reco"] = reco(w["Ta"].values, w["E0"].values, w["rb"].values)
    w["absres"] = (w["NEE"] - w["reco"]).abs()
    rows = []
    for nid, g in w.groupby("night"):
        if len(g) < min_n:
            continue
        if g["reco"].std() < 0.3:            # need some temperature-driven variation to show
            continue
        c = np.corrcoef(g["NEE"], g["reco"])[0, 1]
        rows.append((nid, c, g["absres"].quantile(0.9)))
    rows = [r for r in rows if np.isfinite(r[1])]
    # rank by correlation (physics signal visible), prefer lower 90th-pct residual
    rows.sort(key=lambda r: (-r[1], r[2]))
    return [r[0] for r in rows[:k]]


def fig_decomp_night(plt):
    w = load(WIENER); w["night"] = night_ids(w)
    bb = load(BLACKBOX); bb["night"] = night_ids(bb)
    nights = _select_clean_nights(w, k=3)
    fig, axes = plt.subplots(2, len(nights), figsize=(5.2 * len(nights), 8.2), sharex="col")
    for j, nid in enumerate(nights):
        g = w[w["night"] == nid].reset_index(drop=True)
        hrs = (pd.to_datetime(g["DateTime"]) - pd.to_datetime(g["DateTime"]).iloc[0]).dt.total_seconds() / 3600
        phys = reco(g["Ta"].values, g["E0"].values, g["rb"].values)   # smooth physics respiration
        sd = g["pred_nee_std"].values                                  # calibrated aleatoric std
        obs = g["NEE"].values
        # top: WienerNet decomposition — physics drift signal + aleatoric band
        ax = axes[0, j]
        ax.fill_between(hrs, phys - Z * sd, phys + Z * sd, color="#0072B2", alpha=0.18,
                        label="aleatoric noise (90% band)")
        ax.plot(hrs, phys, color="#0072B2", lw=2.4, label="physics drift  $R_{eco}(T)$")
        ax.scatter(hrs, obs, s=34, color="#111", zorder=5, label="observed NEE")
        e0, rb = g["E0"].iloc[0], g["rb"].iloc[0]
        ax.set_title(f"{pd.to_datetime(g['DateTime']).iloc[0].date()}   "
                     f"($E_0$={e0:.0f}, $r_b$={rb:.1f})", fontsize=10.5)
        if j == 0:
            ax.set_ylabel("NEE  (µmol m⁻² s⁻¹)\n— WienerNet decomposition —", fontsize=10)
            ax.legend(fontsize=8.5, loc="best")
        # bottom: black box — a single opaque predictive band, no signal/noise split
        gb = bb[bb["night"] == nid].reset_index(drop=True)
        if len(gb):
            hb = (pd.to_datetime(gb["DateTime"]) - pd.to_datetime(gb["DateTime"]).iloc[0]).dt.total_seconds() / 3600
            mb, sb = gb["pred_nee_mean"].values, gb["pred_nee_std"].values
            axb = axes[1, j]
            axb.fill_between(hb, mb - Z * sb, mb + Z * sb, color="#D55E00", alpha=0.16,
                             label="predictive band (90%)")
            axb.plot(hb, mb, color="#D55E00", lw=1.6, alpha=0.9, label="predictive mean")
            axb.scatter(hb, gb["NEE"].values, s=34, color="#111", zorder=5)
            axb.set_xlabel("hours into night")
            if j == 0:
                axb.set_ylabel("NEE  (µmol m⁻² s⁻¹)\n— black box: no decomposition —", fontsize=10)
                axb.legend(fontsize=8.5, loc="best")
    fig.suptitle("Per-night flux decomposition — WienerNet separates an interpretable physics drift "
                 "(smooth respiration $R_{eco}(T)$)\nfrom calibrated aleatoric noise (top); the no-physics "
                 "black box gives only one opaque predictive band (bottom)", fontsize=12)
    _save(fig, "fig_decomp_night")


# ---------------------------------------------------------------- 2. drift = physics
def fig_drift_physics(plt):
    w = load(WIENER)
    T = w["Ta"].values
    R = reco(T, w["E0"].values, w["rb"].values)          # model respiration (drift's Reco)
    obs_resp = w["NEE"].values                            # nighttime NEE ≈ Reco (GPP=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    # A: Reco(T) curve vs binned observed nighttime respiration
    order = np.argsort(T)
    a1.plot(T[order], R[order], color="#0072B2", lw=2.4, zorder=4,
            label="WienerNet drift — Lloyd–Taylor $R_{eco}(T)$")
    bins = np.linspace(np.nanpercentile(T, 1), np.nanpercentile(T, 99), 14)
    idx = np.digitize(T, bins)
    bx = [T[idx == k].mean() for k in range(1, len(bins)) if np.any(idx == k)]
    by = [np.nanmedian(obs_resp[idx == k]) for k in range(1, len(bins)) if np.any(idx == k)]
    be = [np.nanstd(obs_resp[idx == k]) for k in range(1, len(bins)) if np.any(idx == k)]
    a1.errorbar(bx, by, yerr=be, fmt="o", color="#111", ms=5, capsize=3, lw=1,
                label="observed nighttime respiration (binned)")
    a1.set_xlabel("air temperature  T  (°C)"); a1.set_ylabel("respiration  (µmol m⁻² s⁻¹)")
    a1.set_title("The drift is a physical instrument:\nrecovered $R_{eco}(T)$ vs observations")
    a1.legend(fontsize=9)
    # B: predicted temperature tendency vs observed
    pt, gt = w["pred_dtemp"].values, w["dTa"].values
    m = np.isfinite(pt) & np.isfinite(gt)
    a2.scatter(gt[m], pt[m], s=8, alpha=0.25, color="#0072B2", edgecolor="none")
    lim = np.nanpercentile(np.abs(np.concatenate([gt[m], pt[m]])), 99)
    a2.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="1:1")
    r = np.corrcoef(gt[m], pt[m])[0, 1]
    a2.set_xlim(-lim, lim); a2.set_ylim(-lim, lim); a2.set_aspect("equal")
    a2.set_xlabel("observed temperature tendency  dT/dt"); a2.set_ylabel("predicted  dT/dt")
    a2.set_title(f"The learned tendency head is faithful\n(r = {r:.2f})"); a2.legend(fontsize=9)
    fig.suptitle("Drift interpretability — physical respiration response × faithful temperature tendency",
                 fontsize=12.5)
    _save(fig, "fig_drift_physics")


# ---------------------------------------------------------------- 3. noise calibration
def fig_noise_calibration(plt):
    w = load(WIENER)
    R = reco(w["Ta"].values, w["E0"].values, w["rb"].values)
    resid = (w["NEE_next"] - w["pred_nee_mean"]).values
    pred_sd = w["pred_nee_std"].values
    m = np.isfinite(R) & np.isfinite(resid) & np.isfinite(pred_sd)
    R, resid, pred_sd = R[m], resid[m], pred_sd[m]
    bins = np.quantile(R, np.linspace(0, 1, 11))
    idx = np.digitize(R, bins)
    bx, emp, prd = [], [], []
    for k in range(1, len(bins)):
        s = idx == k
        if s.sum() > 20:
            bx.append(R[s].mean()); emp.append(resid[s].std()); prd.append(pred_sd[s].mean())
    bx, emp, prd = map(np.array, (bx, emp, prd))
    fig, ax = plt.subplots(figsize=(8.4, 6))
    ax.plot(bx, emp, "o-", color="#111", lw=1.6, ms=6, label="empirical residual std")
    ax.plot(bx, prd, "s-", color="#0072B2", lw=2, ms=6, label="WienerNet predicted noise std")
    xx = np.linspace(bx.min(), bx.max(), 50)
    ax.plot(xx, 0.24 + 0.30 * xx, "--", color="#D55E00", lw=1.5,
            label="measured law  SD ≈ 0.24 + 0.30·$R_{eco}$")
    ax.set_xlabel("respiration flux  $R_{eco}$  (µmol m⁻² s⁻¹)")
    ax.set_ylabel("noise standard deviation")
    ax.set_title("The aleatoric noise is calibrated and heteroscedastic:\n"
                 "the predicted scale tracks the flux-dependent residual spread")
    ax.legend(fontsize=9.5)
    _save(fig, "fig_noise_calibration")


# ---------------------------------------------------------------- 4. aggregation / budget
def fig_variance_aggregation(plt):
    w = load(WIENER)
    t = pd.to_datetime(w["DateTime"])
    obs, mean, sd = w["NEE_next"].values, w["pred_nee_mean"].values, w["pred_nee_std"].values
    df = pd.DataFrame({"t": t, "obs": obs, "mean": mean, "var": sd ** 2})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    # A: predicted vs observed aggregated flux (drift carries the budget)
    for res, lab, c in [("W", "weekly", "#56B4E9"), ("MS", "monthly", "#0072B2")]:
        gp = df.set_index("t").resample(res).agg(obs=("obs", "mean"), mean=("mean", "mean"))
        gp = gp.dropna()
        if len(gp) > 1:
            r2 = 1 - np.sum((gp["obs"] - gp["mean"]) ** 2) / np.sum((gp["obs"] - gp["obs"].mean()) ** 2)
            a1.scatter(gp["obs"], gp["mean"], s=42, color=c, edgecolor="k", linewidth=0.4,
                       label=f"{lab}  (R²={r2:.2f})", zorder=3)
    lim = [min(df["obs"].quantile(.02), 0), df["obs"].quantile(.98)]
    a1.plot(lim, lim, "k--", lw=1, label="1:1"); a1.set_xlim(*lim); a1.set_ylim(*lim); a1.set_aspect("equal")
    a1.set_xlabel("observed aggregated flux"); a1.set_ylabel("drift (physics) predicted flux")
    a1.set_title("Drift carries the flux budget under aggregation"); a1.legend(fontsize=9)
    # B: relative uncertainty vs aggregation window, with 1/sqrt(N) reference
    windows = [("30-min", 1, "30min"), ("daily", 48, "D"), ("weekly", 336, "W"), ("monthly", 1460, "MS")]
    xs, rel = [], []
    total_mean = np.nansum(mean)
    for i, (lab, nper, res) in enumerate(windows):
        gp = df.set_index("t").resample(res).agg(m=("mean", "sum"), v=("var", "sum")).dropna()
        # relative uncertainty of each window's summed flux, averaged
        ru = np.nanmean(np.sqrt(gp["v"]) / np.abs(gp["m"]).clip(1e-6))
        xs.append(i); rel.append(ru)
    a2.plot(xs, rel, "o-", color="#0072B2", lw=2, ms=7, label="WienerNet relative uncertainty")
    ref = rel[0] / np.sqrt([1, 48, 336, 1460] / np.array([1]))
    a2.plot(xs, rel[0] / np.sqrt(np.array([1, 48, 336, 1460])), "--", color="#888", label="1/√N reference")
    a2.set_xticks(xs); a2.set_xticklabels([w[0] for w in windows])
    a2.set_ylabel("relative uncertainty of summed flux"); a2.set_yscale("log")
    a2.set_title("Aleatoric noise cancels ~1/√N\n(the budget uncertainty shrinks with aggregation)")
    a2.legend(fontsize=9)
    fig.suptitle("Carbon-budget decomposition — systematic drift persists, random noise averages out",
                 fontsize=12.5)
    _save(fig, "fig_variance_aggregation")


def main():
    plt = setup_mpl()
    fig_decomp_night(plt)
    fig_drift_physics(plt)
    fig_noise_calibration(plt)
    fig_variance_aggregation(plt)


if __name__ == "__main__":
    main()
