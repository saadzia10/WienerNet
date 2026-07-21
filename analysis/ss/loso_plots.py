#!/usr/bin/env python
"""Figures for the WienerNet-SS multi-site generalisation and the structural band term.
Reads the CSVs written by loso_report.py / gap_band_eval.py / autoregressive_gapfill.py and
writes PNGs to analysis/ss/figs/.

Figures:
  F1  per-site CRPS, grouped bars, error bars = std over seeds; bars are clipped at 2.0 and
      values above the clip are annotated off-chart.
  F2  per-site 90% coverage vs the 0.90 target line.
  F3  autoregressive gap-fill RMSE by hours-into-gap; values above the 4.0 clip are annotated.
  F4  gap band coverage with and without the structural-variance term.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
SITE_LBL = {"woodwalton": "Woodwalton", "rosedene": "Rosedene", "redmere_1": "Redmere 1",
            "redmere_2": "Redmere 2", "great_fen": "Great Fen"}
# colorblind-safe categorical (Okabe–Ito subset), fixed order, physics models first
PAL = {"WienerNet-SS (diurnal, state-space)": "#0072B2", "WienerNet-SS (diurnal, Wiener)": "#56B4E9",
       "MDN (no physics)": "#E69F00", "Neural SDE": "#D55E00", "Analytical SDE": "#009E73",
       "Random Forest": "#CC79A7", "XGBoost": "#8C6D31"}
SHORT = {"WienerNet-SS (diurnal, state-space)": "WN-SS (state-sp)", "WienerNet-SS (diurnal, Wiener)": "WN-SS (Wiener)",
         "MDN (no physics)": "MDN", "Neural SDE": "Neural SDE", "Analytical SDE": "Analytical",
         "Random Forest": "RF", "XGBoost": "XGB"}
ORDER = list(PAL)
# the AR gap-fill CSV uses shorter model names — alias them onto the same colours/labels
PAL.update({"WienerNet-SS (state-space)": PAL["WienerNet-SS (diurnal, state-space)"],
            "WienerNet-SS (Wiener)": PAL["WienerNet-SS (diurnal, Wiener)"]})
SHORT.update({"WienerNet-SS (state-space)": "WN-SS (state-sp)", "WienerNet-SS (Wiener)": "WN-SS (Wiener)"})
AR_ORDER = ["WienerNet-SS (state-space)", "WienerNet-SS (Wiener)", "MDN (no physics)",
            "Neural SDE", "Analytical SDE", "Random Forest", "XGBoost"]


def _mean_std(df, model, site, col):
    r = df[(df.model == model) & (df.site == site)][col].dropna()
    return (r.mean(), r.std()) if len(r) else (np.nan, np.nan)


def fig_crps_coverage(df):
    for metric, fname, ycap, target in [("crps", "F1_loso_crps.png", None, None),
                                        ("cov90", "F2_loso_coverage.png", None, 0.90)]:
        cap = 2.0 if metric == "crps" else None   # keep the competitive region readable
        fig, ax = plt.subplots(figsize=(11, 4.6))
        nS, nM = len(SITES), len(ORDER)
        w = 0.8 / nM
        for j, m in enumerate(ORDER):
            xs, ys, es, ann = [], [], [], []
            for i, s in enumerate(SITES):
                mu, sd = _mean_std(df, m, s, metric)
                x = i + (j - nM / 2) * w + w / 2
                xs.append(x)
                if cap is not None and np.isfinite(mu) and mu > cap:
                    ys.append(cap); es.append(0); ann.append((x, cap, mu))
                else:
                    ys.append(mu if np.isfinite(mu) else 0); es.append(sd if np.isfinite(sd) else 0); ann.append(None)
            ax.bar(xs, ys, w, label=SHORT[m], color=PAL[m], edgecolor="white", linewidth=0.5,
                   yerr=es, error_kw=dict(elinewidth=0.8, capsize=1.5, ecolor="#444"))
            for a in ann:
                if a:
                    ax.annotate(f"{a[2]:.0f}", (a[0], a[1]), ha="center", va="bottom",
                                fontsize=6.5, rotation=90, color=PAL[m])
        ax.set_xticks(range(nS)); ax.set_xticklabels([SITE_LBL[s] for s in SITES])
        ax.set_ylabel("CRPS (μmol m⁻² s⁻¹)" if metric == "crps" else "90% coverage")
        ax.set_title(("Probabilistic accuracy across held-out sites (CRPS, lower better)" if metric == "crps"
                      else "Calibration across held-out sites (90% coverage)"), fontsize=11)
        if target:
            ax.axhline(target, color="#333", ls="--", lw=1, zorder=0)
            ax.annotate("nominal 0.90", (nS - 0.5, target), fontsize=8, va="bottom", ha="right", color="#333")
            ax.set_ylim(0, 1.02)
        if cap is not None and metric == "crps":
            ax.set_ylim(0, cap * 1.08)
            ax.annotate("bars clipped at 2.0; value = mean CRPS",
                        (0.5, 0.98), xycoords="axes fraction", fontsize=7, ha="center", va="top", color="#666")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.grid(axis="y", color="#e8e8e8", lw=0.6, zorder=0)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, fname), dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fname)


def fig_ar_rmse():
    p = os.path.join(OUT, "ar_gapfill_loso.csv")
    if not os.path.exists(p):
        print("skip F3 (no ar_gapfill_loso.csv yet)"); return
    df = pd.read_csv(p)
    bins = ["h0-2", "h2-5", "h5+"]
    CAP = 4.0   # competitive region; larger values are annotated off-chart
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    # WN-SS (state-space) drawn last & on top; Analytical dashed — the physics-drift models
    # coincide (identical deterministic drift), so distinct styles keep both visible.
    styles = {"WienerNet-SS (state-space)": dict(lw=2.6, zorder=6),
              "Analytical SDE": dict(ls="--", lw=2, zorder=5),
              "WienerNet-SS (Wiener)": dict(lw=1.4, zorder=4, alpha=0.7)}
    for m in AR_ORDER:
        r = df[df.model == m]
        if r.empty:
            continue
        ys = np.array([r[f"rmse_{b}"].mean() for b in bins], float)
        if not np.isfinite(ys).any():
            continue
        yc = np.clip(ys, None, CAP)
        st = dict(lw=2, zorder=3); st.update(styles.get(m, {}))
        ax.plot(range(len(bins)), yc, "-o", color=PAL[m], label=SHORT[m], ms=6,
                markeredgecolor="white", markeredgewidth=0.7, **st)
        for xi, (yv, ycv) in enumerate(zip(ys, yc)):
            if np.isfinite(yv) and yv > CAP:
                ax.annotate(f"{yv:.0f}", (xi, ycv), ha="center", va="bottom",
                            fontsize=7, color=PAL[m], rotation=0)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(["0–2 h", "2–5 h", "5+ h"])
    ax.set_xlabel("hours into gap"); ax.set_ylabel("rollout RMSE (μmol m⁻² s⁻¹)")
    ax.set_title("Autoregressive gap-fill: point error vs gap length (mean over 5 sites)", fontsize=11)
    ax.set_ylim(0, CAP * 1.05)
    ax.annotate("Neural SDE blows up out-of-distribution (values off-chart);\nphysics-drift models (WN-SS ≈ Analytical) stay flat",
                (0.03, 0.60), xycoords="axes fraction", fontsize=8, ha="left", va="top", color="#555")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F3_ar_gapfill_rmse.png"), dpi=160, bbox_inches="tight")
    plt.close(fig); print("wrote F3_ar_gapfill_rmse.png")


def fig_band_fix():
    p = os.path.join(OUT, "gap_band_fix.csv")
    if not os.path.exists(p):
        print("skip F4 (no gap_band_fix.csv yet)"); return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sites = [s for s in SITES if s in set(df.site)]
    x = np.arange(len(sites))
    old = [df[(df.site == s)]["old_near"].mean() for s in sites]
    fix = [df[(df.site == s)]["fix_near"].mean() for s in sites]
    ax.bar(x - 0.2, old, 0.4, label="measurement+process band (old)", color="#D55E00", edgecolor="white")
    ax.bar(x + 0.2, fix, 0.4, label="+ structural-variance term (fixed)", color="#0072B2", edgecolor="white")
    ax.axhline(0.90, color="#333", ls="--", lw=1)
    ax.annotate("nominal 0.90", (len(sites) - 0.5, 0.90), fontsize=8, va="bottom", ha="right", color="#333")
    ax.set_xticks(x); ax.set_xticklabels([SITE_LBL[s] for s in sites])
    ax.set_ylabel("gap band coverage (first night)"); ax.set_ylim(0, 1.02)
    ax.set_title("Gap-fill band: structural-error fix restores nominal coverage", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4_band_fix.png"), dpi=160, bbox_inches="tight")
    plt.close(fig); print("wrote F4_band_fix.png")


def main():
    p = os.path.join(OUT, "loso_metrics_long.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        fig_crps_coverage(df)
    else:
        print("no loso_metrics_long.csv — run loso_report.py first")
    fig_ar_rmse()
    fig_band_fix()


if __name__ == "__main__":
    main()
