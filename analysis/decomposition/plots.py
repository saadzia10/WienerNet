#!/usr/bin/env python
"""Decomposition-rationality figures (one standalone vector file per plot; manuscript conventions).

Produces three figures on the in-distribution leave-one-site-out test sets:
  fig_decomp_attribution              — 30-min increment variance split (physics drift vs noise)
  fig_decomp_noise_vs_temperature     — mean one-step residual vs air temperature (noise cleanliness)
  fig_decomp_drift_physicality        — model drift rate vs the analytic physics drift rate
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from figstyle import C, LW, MSZ, save
from rationality import pool, reco


def components(df):
    tgt = df["gt_nee"].to_numpy(float); bnee = df["gt_bnee"].to_numpy(float)
    mean = df["pred_nee_mean"].to_numpy(float); f = df["pred_f"].to_numpy(float)
    dt = df["dt"].to_numpy(float); Ta = df["Ta"].to_numpy(float)
    return dict(eta=tgt - mean, obs_incr=tgt - bnee, drift_incr=f * dt, f=f, Ta=Ta,
                Reco=reco(Ta, df["E0"].to_numpy(float), df["rb"].to_numpy(float)))


def main():
    import matplotlib.pyplot as plt
    prim = components(pool("{site}_s{seed}_ldiur_wien"))
    nsde = components(pool("{site}_s{seed}_base_nsde_ald"))

    # --- 1. variance attribution (physics drift vs noise), 30-min step ---
    vd = np.nanvar(prim["drift_incr"]); vn = np.nanvar(prim["eta"]); tot = vd + vn
    fd, fn = 100 * vd / tot, 100 * vn / tot
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    ax.barh([0], [fd], color=C["blue"], edgecolor="white")
    ax.barh([0], [fn], left=[fd], color=C["orange"], edgecolor="white")
    ax.annotate(f"physics drift  {fd:.1f}%", (fd + 1, 0), va="center", ha="left", fontsize=10, color=C["blue"])
    ax.annotate(f"aleatoric noise  {fn:.1f}%", (fd + fn / 2, 0), va="center", ha="center", fontsize=11, color="white", fontweight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_xlabel("Share of 30-min increment variance (%)")
    ax.grid(False)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_decomp_attribution",
         "Attribution of the 30-minute NEE-increment variance in the WienerNet-SS prediction: the "
         "analytic physics drift accounts for ~2% and the aleatoric noise ~98%, reflecting that 30-min "
         "nighttime flux change is measurement-noise-dominated rather than respiration-driven.")

    # --- 2. noise cleanliness: mean residual vs air temperature ---
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    for c, col, lab in [(prim, C["blue"], "WienerNet-SS (physics mean)"), (nsde, C["vermillion"], "Neural SDE (learned mean)")]:
        d = pd.DataFrame({"Ta": c["Ta"], "eta": c["eta"]}).dropna()
        d["bin"] = pd.qcut(d["Ta"], 12, duplicates="drop")
        g = d.groupby("bin", observed=True)
        r = float(np.corrcoef(d.Ta, d.eta)[0, 1])
        ax.plot(g["Ta"].mean(), g["eta"].mean(), "-o", color=col, lw=LW, ms=MSZ, label=f"{lab}  (r = {r:+.2f})")
    ax.axhline(0, color=C["grey"], ls="--", lw=1)
    ax.set_xlabel("Air temperature (°C)")
    ax.set_ylabel("Mean one-step residual (µmol m⁻² s⁻¹)")
    ax.legend(frameon=False, loc="lower left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_decomp_noise_vs_temperature",
         "Mean one-step residual (observation minus predicted mean) binned by air temperature. Both "
         "WienerNet-SS and the Neural SDE leave essentially no temperature structure in the noise "
         "(correlations -0.01 and -0.03), so the aleatoric term is a clean, structureless remainder.")

    # --- 3. drift physicality: model drift vs analytic physics drift ---
    phys = prim["f"]; n = min(len(phys), len(nsde["f"]))
    rng = np.random.default_rng(0); idx = rng.choice(n, min(4000, n), replace=False)
    r_nsde = float(np.corrcoef(phys[:n], nsde["f"][:n])[0, 1])
    lim = float(np.nanpercentile(np.abs(phys), 99))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(phys[:n][idx], nsde["f"][:n][idx], s=6, alpha=0.25, color=C["vermillion"],
               rasterized=True, label=f"Neural SDE drift  (r = {r_nsde:.2f})")
    ax.scatter(phys[idx], phys[idx], s=6, alpha=0.5, color=C["blue"], rasterized=True,
               label="WienerNet-SS drift  (= physics, r = 1.00)")
    ax.plot([-lim, lim], [-lim, lim], color=C["grey"], ls="--", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Analytic physics drift rate (µmol m⁻² s⁻¹ min⁻¹)")
    ax.set_ylabel("Model drift rate (µmol m⁻² s⁻¹ min⁻¹)")
    ax.legend(frameon=False, loc="upper left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_decomp_drift_physicality",
         "Each model's predicted drift rate against the analytic Lloyd-Taylor drift. WienerNet-SS lies "
         "on the identity line by construction (r = 1.00) whereas the Neural SDE's free drift is a "
         "scattered cloud (r = 0.26): only the physics-informed drift is an interpretable respiration term.")


if __name__ == "__main__":
    main()
