#!/usr/bin/env python
"""Figures for the Redmere 1 blow-up report: the driver artifact, the noise-scale
explosion 2x2, and the drift-vs-noise localisation."""
from __future__ import annotations
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
rep = json.load(open(os.path.join(HERE, "investigate.json")))
BLUE, ORANGE, RED, GREEN, GREY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#888888"


def fig_driver_ood():
    od = rep["driver_ood"]["per_driver"]
    drivers = sorted(od, key=lambda d: -abs(od[d]["z_meanshift"]))
    z = [abs(od[d]["z_meanshift"]) for d in drivers]
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [RED if abs(od[d]["z_meanshift"]) > 3 else BLUE for d in drivers]
    ax.bar(drivers, z, color=cols, edgecolor="white")
    ax.set_ylabel("|standardised mean shift|  (σ from train pool)")
    ax.set_title("Redmere 1 is out-of-distribution in ONE driver: Tau (momentum flux)", fontsize=11)
    ax.axhline(3, color=GREY, ls="--", lw=1); ax.annotate("3σ", (len(drivers)-0.5, 3.2), color=GREY, fontsize=8)
    for i, d in enumerate(drivers):
        if abs(od[d]["z_meanshift"]) > 3:
            ax.annotate(f"{od[d]['z_meanshift']:.1f}σ\n({100*od[d]['frac_beyond_train_p1_99']:.0f}% rows extreme)",
                        (i, z[i]), ha="center", va="top", color="white", fontsize=8, fontweight="bold",
                        xytext=(0, -6), textcoords="offset points")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eee", lw=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R1_driver_ood.png"), dpi=160, bbox_inches="tight")
    plt.close(fig); print("wrote R1_driver_ood.png")


def fig_sigma_2x2():
    # sigma max train vs test for the four variants
    order = ["ldiur_wien (PRIMARY, robust)", "diur_wien (no ldiur reg)", "diur_ss (blows up)", "ldiur_ss (worst)"]
    lbl = {"ldiur_wien (PRIMARY, robust)": "learned-diur\n+ Wiener\n(PRIMARY)",
           "diur_wien (no ldiur reg)": "exog-diur\n+ Wiener",
           "diur_ss (blows up)": "exog-diur\n+ state-space",
           "ldiur_ss (worst)": "learned-diur\n+ state-space"}
    tr = [rep[m]["train"]["sigma"]["max"] for m in order]
    te = [rep[m]["redmere1_test"]["sigma"]["max"] for m in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, tr, 0.4, label="train (in-distribution)", color=BLUE, edgecolor="white")
    ax.bar(x + 0.2, te, 0.4, label="Redmere 1 (out-of-distribution)", color=RED, edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("max predicted noise scale σ  (log)")
    ax.set_title("The learned noise-scale explodes OOD — except the primary (learned-diurnal + Wiener)", fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels([lbl[m] for m in order], fontsize=8)
    for i, v in enumerate(te):
        ax.annotate(f"{v:.0f}" if v > 5 else f"{v:.1f}", (i + 0.2, v), ha="center", va="bottom",
                    fontsize=8, color=RED, fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eee", lw=0.6); ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R1_sigma_explosion.png"), dpi=160, bbox_inches="tight")
    plt.close(fig); print("wrote R1_sigma_explosion.png")


def _rmse(run, target):
    p = os.path.join(run, "metrics", "per_site.csv")
    if not os.path.exists(p): return np.nan
    for r in csv.DictReader(open(p)):
        if r.get("target") == target and r.get("metric") == "rmse": return float(r["value"])
    return np.nan


def fig_localisation():
    LOSO = os.path.join(ROOT, "outputs", "ss_loso"); FL = os.path.join(ROOT, "outputs", "final_loso")
    models = [("WN-SS ldiur+wien\n(PRIMARY)", f"{LOSO}/redmere_1_s0_ldiur_wien", BLUE),
              ("WN-SS diur+ss", f"{LOSO}/redmere_1_s0_diur_ss", ORANGE),
              ("MDN", f"{FL}/comp_mdn_redmere_1_s0", GREEN),
              ("Neural SDE", f"{FL}/comp_neuralsde_redmere_1_s0", RED)]
    det = [_rmse(r, "nee_mean") for _, r, _ in models]
    samp = [_rmse(r, "nee") for _, r, _ in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, det, 0.4, label="deterministic mean (drift only)", color=BLUE, edgecolor="white")
    ax.bar(x + 0.2, samp, 0.4, label="sampled prediction (drift + noise)", color=RED, edgecolor="white")
    ax.set_yscale("log"); ax.set_ylabel("Redmere 1 RMSE  (log)")
    ax.set_title("Where the blow-up lives: physics keeps the MEAN stable; the noise sample explodes\n"
                 "(black-box models blow up in the mean too — no physics anchor)", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([m for m, _, _ in models], fontsize=8.5)
    ax.axhline(2.0, color=GREY, ls="--", lw=1); ax.annotate("~in-range RMSE", (len(models)-0.5, 2.1), color=GREY, fontsize=8, ha="right")
    for i, (d, s) in enumerate(zip(det, samp)):
        if np.isfinite(d): ax.annotate(f"{d:.1f}", (i-0.2, d), ha="center", va="bottom", fontsize=7.5, color=BLUE)
        if np.isfinite(s): ax.annotate(f"{s:.0f}" if s > 5 else f"{s:.1f}", (i+0.2, s), ha="center", va="bottom", fontsize=7.5, color=RED)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eee", lw=0.6); ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "R1_localisation.png"), dpi=160, bbox_inches="tight")
    plt.close(fig); print("wrote R1_localisation.png")


if __name__ == "__main__":
    fig_driver_ood(); fig_sigma_2x2(); fig_localisation()
