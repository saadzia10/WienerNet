#!/usr/bin/env python
"""How WienerNet-SS APPORTIONS its own predicted ΔNEE across its three heads, vs timescale.

Complements `drift_vs_timescale.py` (which is MODEL-FREE — the true physics explanatory power). Here
we take the MODEL's predicted increment and decompose ITS predictive variance into the three additive
heads, rolled up over growing WITHIN-NIGHT windows (the regime the increment model actually integrates
over; beyond a night the drift is the nightly Reco(T) reconstruction of drift_vs_timescale.py):

    predicted ΔNEE(t→t+k) = Σ f_phys·dt  +  Σ r·dt  +  Σ noise
                            └ physics drift ┘ └residual┘ └ noise ┘

For each window length k we compute, pooled over the 4 in-distribution sites:
  V_phys(k)  = Var over windows of the summed physics-drift increment  (coherent → grows ∝k²)
  V_resid(k) = Var over windows of the summed residual increment
  V_noise(k) = mean accumulated aleatoric variance Σ σ²           (independent → grows ∝k)
and plot each as a share of the total. The physics share should CLIMB with the window (the drift
accumulates coherently as respiration declines), while the noise share falls (it averages down); a
short-timescale residual correction falls too. This is the model's INTERNAL attribution — a check
that its heads behave as a rational decomposition, not the model-free truth.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from figstyle import C, LW, MSZ, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CLEAN = ["woodwalton", "rosedene", "redmere_2", "great_fen"]


def pool(tmpl):
    fr = []
    for s in CLEAN:
        p = os.path.join(ROOT, "outputs", "ss_loso", tmpl.format(site=s), "metrics", "predictions.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df["_site"] = s
            fr.append(df)
    return pd.concat(fr, ignore_index=True) if fr else None


def decompose(df, ks):
    """Per-window-length k: variance shares of physics / residual / noise in the predicted increment."""
    df = df.copy()
    dt = df["dt"].to_numpy(float)
    r = df["pred_residual"].to_numpy(float) if "pred_residual" in df else np.zeros(len(df))
    df["_phys"] = (df["pred_f"].to_numpy(float) - r) * dt        # physics-drift increment per step
    df["_resid"] = r * dt                                        # residual increment per step
    df["_nvar"] = df["pred_nee_std"].to_numpy(float) ** 2        # per-step aleatoric variance
    g = df.groupby(["_site", "window_id"], sort=False)
    rows = []
    for k in ks:
        phys = g["_phys"].transform(lambda s: s.rolling(k).sum())
        resid = g["_resid"].transform(lambda s: s.rolling(k).sum())
        nvar = g["_nvar"].transform(lambda s: s.rolling(k).sum())
        m = np.isfinite(phys) & np.isfinite(resid) & np.isfinite(nvar)
        Vp = float(np.nanvar(phys[m])); Vr = float(np.nanvar(resid[m])); Vn = float(np.nanmean(nvar[m]))
        tot = Vp + Vr + Vn
        rows.append(dict(k=k, hours=k * 0.5, phys=Vp / tot, resid=Vr / tot, noise=Vn / tot,
                         has_resid=np.any(r != 0), n=int(m.sum())))
    return pd.DataFrame(rows)


def main():
    ks = [1, 2, 4, 8, 16]   # 30 min, 1 h, 2 h, 4 h, 8 h (within-night)
    res = decompose(pool("{site}_s0_ldiur_res_wien"), ks)      # residual variant: all three heads
    prim = decompose(pool("{site}_s0_ldiur_wien"), ks)         # primary: physics + noise only

    print("=== Model's predicted-ΔNEE variance decomposition vs within-night window ===")
    print("  RESIDUAL variant (physics / residual / noise, % of predicted increment variance):")
    print(f"    {'window':<8}{'phys%':>8}{'resid%':>8}{'noise%':>8}")
    for _, r in res.iterrows():
        lab = {0.5: "30 min", 1: "1 h", 2: "2 h", 4: "4 h", 8: "8 h"}[r.hours]
        print(f"    {lab:<8}{100*r.phys:>8.1f}{100*r.resid:>8.1f}{100*r.noise:>8.1f}")
    print("  PRIMARY (physics / noise, residual off):")
    for _, r in prim.iterrows():
        lab = {0.5: "30 min", 1: "1 h", 2: "2 h", 4: "4 h", 8: "8 h"}[r.hours]
        print(f"    {lab:<8}{100*r.phys:>8.1f}{'—':>8}{100*r.noise:>8.1f}")

    xlab = ["30 min", "1 h", "2 h", "4 h", "8 h"]

    # --- figure 1: primary configuration (physics + noise) ---
    x = np.arange(len(prim))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(x, 100 * prim.phys, "-o", color=C["blue"], lw=LW, ms=MSZ, label="Physics drift")
    ax.plot(x, 100 * prim.noise, "-^", color=C["vermillion"], lw=LW, ms=MSZ, label="Noise head")
    ax.set_xticks(x); ax.set_xticklabels(xlab); ax.set_xlabel("Within-night window")
    ax.set_ylabel("Share of predicted increment variance (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="center right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_predicted_shares_primary",
         "Share of the WienerNet-SS predicted-increment variance carried by the physics drift and the "
         "noise head (primary, residual-off configuration) versus within-night window. The physics-drift "
         "share climbs from 7% to 26% as the coherent respiration signal accumulates while the white noise "
         "averages down from 93% to 74%.")

    # --- figure 2: residual variant (physics + residual + noise) ---
    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(x, 100 * res.phys, "-o", color=C["blue"], lw=LW, ms=MSZ, label="Physics drift")
    ax.plot(x, 100 * res.resid, "-s", color=C["green"], lw=LW, ms=MSZ, label="Residual")
    ax.plot(x, 100 * res.noise, "-^", color=C["vermillion"], lw=LW, ms=MSZ, label="Noise head")
    ax.set_xticks(x); ax.set_xticklabels(xlab); ax.set_xlabel("Within-night window")
    ax.set_ylabel("Share of predicted increment variance (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="center right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_predicted_shares_residual_variant",
         "Share of the WienerNet-SS predicted-increment variance carried by the physics drift, residual "
         "and noise heads (residual variant) versus within-night window. The physics-drift share climbs "
         "(1% to 13%) and the noise averages down (77% to 57%), while the residual holds a stable ~30% "
         "share — the signature of a persistent structural misfit that neither averages away nor overtakes "
         "the coherent physics drift.")


if __name__ == "__main__":
    main()
