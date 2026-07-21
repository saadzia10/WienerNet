#!/usr/bin/env python
"""Figures for the Redmere 1 analysis (one standalone vector file per plot): per-driver
distribution shift, per-site Tau distribution, predicted noise scale, and RMSE by prediction type."""
from __future__ import annotations
import os, json, csv
import numpy as np
from figstyle import C, LW, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
rep = json.load(open(os.path.join(HERE, "investigate.json")))


def fig_driver_ood():
    """Per-driver standardised mean shift vs the training pool, as a horizontal lollipop
    on a log axis, with the signed value annotated."""
    od = rep["driver_ood"]["per_driver"]
    drivers = sorted(od, key=lambda d: abs(od[d]["z_meanshift"]))   # bottom -> top = ascending
    z = [max(abs(od[d]["z_meanshift"]), 1e-3) for d in drivers]
    y = np.arange(len(drivers))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for yi, zi, d in zip(y, z, drivers):
        col = C["orange"] if abs(od[d]["z_meanshift"]) > 3 else C["blue"]
        ax.plot([1e-2, zi], [yi, yi], color=col, lw=LW, solid_capstyle="round", zorder=2)
        ax.scatter([zi], [yi], s=55, color=col, zorder=3)
        ax.annotate(f"{od[d]['z_meanshift']:+.2f}σ", (zi, yi), xytext=(9, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9.5,
                    color=col, fontweight="bold" if abs(od[d]["z_meanshift"]) > 3 else "normal")
    ax.axvline(3, color=C["grey"], ls="--", lw=1, zorder=1)
    ax.annotate("3σ", (3, len(drivers) - 0.35), xytext=(3, 0), textcoords="offset points",
                color=C["grey"], fontsize=9.5, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 2.2e2)
    ax.set_yticks(y); ax.set_yticklabels(drivers)
    ax.set_xlabel("Standardised mean shift $|z|$ vs. training pool (σ, log scale)")
    ax.set_ylabel("Meteorological driver")
    ax.set_ylim(-0.7, len(drivers) - 0.3)
    ax.grid(axis="y", visible=False)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_redmere1_driver_ood",
         "Standardised mean shift of each meteorological driver at Redmere 1 relative to the training "
         "pool, on a logarithmic axis (signed value annotated). Only the momentum flux Tau is "
         "out-of-distribution (−26.5σ, ~50x the next-largest shift); every other driver, including air "
         "and soil temperature, sits well inside 1σ — the corrupted Tau channel is the single OOD input.")


def fig_tau_range(basis="model"):
    """Per-site Tau distribution on a symmetric-log axis.

    `basis="model"` uses the rows the training pipeline keeps (nighttime + every driver non-NaN);
    `basis="raw"` uses the nighttime rows before that driver-completeness dropna.
    """
    src = "tau_by_site.json" if basis == "model" else "tau_by_site_raw.json"
    summ = json.load(open(os.path.join(HERE, src)))
    pool = summ.pop("_train_pool")
    sites = sorted([s for s in summ], key=lambda s: (summ[s]["split"] == "test", -summ[s]["sd"]))
    nice = {"redmere_1": "Redmere 1", "redmere_2": "Redmere 2", "great_fen": "Great Fen",
            "rosedene": "Rosedene", "woodwalton": "Woodwalton"}
    vmax = max(abs(summ[s][k]) for s in sites for k in ("min", "max"))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for yi, s in enumerate(sites):
        v, test = summ[s], summ[s]["split"] == "test"
        col = C["orange"] if test else C["blue"]
        ax.plot([v["min"], v["max"]], [yi, yi], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot([v["p1"], v["p99"]], [yi, yi], color=col, lw=7, solid_capstyle="butt",
                alpha=.85, zorder=3)
        ax.scatter([v["p50"]], [yi], s=34, color="white", edgecolor=col, lw=1.6, zorder=4)
        ax.annotate(f"SD = {v['sd']:.2f}", (vmax * 2.6, yi), fontsize=9.5, va="center", ha="left",
                    color=col, fontweight="bold" if test else "normal")
    # the envelope every training site lives inside
    lim = max(abs(summ[s][k]) for s in sites if summ[s]["split"] == "train" for k in ("min", "max"))
    ax.axvspan(-lim, lim, color=C["grey"], alpha=.13, lw=0, zorder=0)
    ax.set_xscale("symlog", linthresh=1e-2, linscale=0.6)
    ax.set_xlim(-vmax * 2.2, vmax * 90)
    emax = int(np.floor(np.log10(vmax)))
    dec = [10.0 ** e for e in range(0, emax + 2, 2) if 10.0 ** e <= vmax * 2.2]
    ax.set_xticks([-d for d in reversed(dec)] + [0] + dec)
    ax.set_yticks(np.arange(len(sites)))
    ax.set_yticklabels([nice.get(s, s) + ("\n(held out)" if summ[s]["split"] == "test" else "")
                        for s in sites], fontsize=10)
    ax.set_xlabel("Momentum flux Tau (kg m⁻¹ s⁻², symmetric-log scale)")
    ax.set_ylabel("Flux-tower site")
    ax.set_ylim(-0.7, len(sites) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.annotate("training-site range", (lim, len(sites) - 0.55), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=C["grey"], va="center", ha="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    r1, trmax = summ["redmere_1"], max(summ[s]["sd"] for s in sites if summ[s]["split"] == "train")
    rows = ("retained by the training pipeline (nighttime rows with every driver present)"
            if basis == "model" else
            "all nighttime rows, before the pipeline's driver-completeness NaN filter")
    save(fig, "fig_redmere1_tau_distribution" + ("" if basis == "model" else "_unfiltered"),
         "Distribution of the momentum flux Tau at each East Anglia site on a symmetric-log axis: thin "
         "line = full min–max range, thick bar = 1st–99th percentile, marker = median; the shaded band "
         f"is the envelope spanned by all four training sites. Rows are {rows}. Held-out Redmere 1 spans "
         f"{r1['min']:.3g} to {r1['max']:.3g} with a standard deviation of {r1['sd']:.0f} — three to four "
         f"orders of magnitude wider than any training site (SD ≤ {trmax:.2f}) — while its median "
         f"({r1['p50']:.2f}) is ordinary, identifying the Tau channel as corrupted in its tails rather "
         "than merely extreme.")


def fig_sigma_explosion():
    order = ["ldiur_wien (PRIMARY, robust)", "diur_wien (no ldiur reg)", "diur_ss (blows up)", "ldiur_ss (worst)"]
    lbl = {"ldiur_wien (PRIMARY, robust)": "learned-diurnal\n+ Wiener\n(primary)",
           "diur_wien (no ldiur reg)": "given-diurnal\n+ Wiener",
           "diur_ss (blows up)": "given-diurnal\n+ state-space",
           "ldiur_ss (worst)": "learned-diurnal\n+ state-space"}
    tr = [rep[m]["train"]["sigma"]["max"] for m in order]
    te = [rep[m]["redmere1_test"]["sigma"]["max"] for m in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - 0.2, tr, 0.4, label="Training sites (in-distribution)", color=C["blue"], edgecolor="white")
    ax.bar(x + 0.2, te, 0.4, label="Redmere 1 (out-of-distribution)", color=C["orange"], edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Max. noise scale σ (µmol m⁻² s⁻¹)")
    ax.set_xticks(x); ax.set_xticklabels([lbl[m] for m in order], fontsize=8.5)
    for i, v in enumerate(te):
        ax.annotate(f"{v:.0f}" if v > 5 else f"{v:.1f}", (i + 0.2, v), ha="center", va="bottom",
                    fontsize=8.5, color=C["orange"], fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_redmere1_noise_scale_explosion",
         "Maximum predicted aleatoric noise scale on the training sites versus the held-out Redmere 1 "
         "test set, for four WienerNet-SS configurations. The unbounded learned noise-scale head "
         "extrapolates to 10³–10⁴ on the corrupted-Tau rows for every configuration except the primary "
         "(learned-diurnal tendency + single-Wiener noise), whose scale stays bounded (0.9).")


def _rmse(run, target):
    p = os.path.join(run, "metrics", "per_site.csv")
    if not os.path.exists(p): return np.nan
    for r in csv.DictReader(open(p)):
        if r.get("target") == target and r.get("metric") == "rmse": return float(r["value"])
    return np.nan


def fig_localisation():
    LOSO = os.path.join(ROOT, "outputs", "ss_loso"); FL = os.path.join(ROOT, "outputs", "final_loso")
    models = [("WienerNet-SS\n(primary)", f"{LOSO}/redmere_1_s0_ldiur_wien"),
              ("WienerNet-SS\ngiven-diur+ss", f"{LOSO}/redmere_1_s0_diur_ss"),
              ("MDN", f"{FL}/comp_mdn_redmere_1_s0"),
              ("Neural SDE", f"{FL}/comp_neuralsde_redmere_1_s0")]
    det = [_rmse(r, "nee_mean") for _, r in models]
    samp = [_rmse(r, "nee") for _, r in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - 0.2, det, 0.4, label="Deterministic mean (drift only)", color=C["blue"], edgecolor="white")
    ax.bar(x + 0.2, samp, 0.4, label="Sampled prediction (drift + noise)", color=C["orange"], edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Redmere 1 RMSE (µmol m⁻² s⁻¹)")
    ax.set_xticks(x); ax.set_xticklabels([m for m, _ in models], fontsize=8.5)
    ax.axhline(2.0, color=C["grey"], ls="--", lw=1)
    ax.annotate("in-range RMSE", (len(models) - 0.5, 2.2), color=C["grey"], fontsize=8.5, ha="right")
    for i, (d, s) in enumerate(zip(det, samp)):
        if np.isfinite(d): ax.annotate(f"{d:.1f}", (i - 0.2, d), ha="center", va="bottom", fontsize=8, color=C["blue"])
        if np.isfinite(s): ax.annotate(f"{s:.0f}" if s > 5 else f"{s:.1f}", (i + 0.2, s), ha="center", va="bottom", fontsize=8, color=C["orange"])
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_redmere1_error_localisation",
         "Redmere 1 one-step RMSE split into the deterministic mean (drift only) and the sampled "
         "prediction (drift + noise). For WienerNet-SS the physics-anchored mean stays stable (~1–2) and "
         "only the noise sample explodes; the black-box MDN and Neural SDE blow up in the mean itself "
         "(12.7 and 1418), having no physics to anchor the conditional mean.")


if __name__ == "__main__":
    fig_tau_range("model"); fig_tau_range("raw"); fig_driver_ood(); fig_sigma_explosion(); fig_localisation()
