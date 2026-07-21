#!/usr/bin/env python
"""Illustrate that the residual is a structured stationary misfit, not overfit white noise.

Two panels, pooled over the four in-distribution sites (residual variant ldiur_res_wien):
  A. Variance-vs-window (log-log): Var of the summed physics-drift / residual / noise increment over
     growing within-night windows, with fitted power-law slopes. The physics drift is the only
     COHERENT accumulator (slope ≈ 2); the residual and the noise both scale ~linearly (slope ≈ 1),
     but the residual sits above the noise and (panel B) is temporally structured.
  B. Within-night autocorrelation of the residual vs the aleatoric noise: the residual is persistent
     (lag-1 ≈ 0.5, decaying), the noise is white (≈ 0) — the residual carries systematic structure the
     noise head does not, which is why its variance share does not average away.
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


def pool():
    fr = []
    for s in CLEAN:
        p = os.path.join(ROOT, "outputs", "ss_loso", f"{s}_s0_ldiur_res_wien", "metrics", "predictions.parquet")
        if os.path.exists(p):
            fr.append(pd.read_parquet(p).assign(_site=s))
    return pd.concat(fr, ignore_index=True)


def main():
    w = pool()
    dt = w["dt"].to_numpy(float); r = w["pred_residual"].to_numpy(float)
    w["_phys"] = (w["pred_f"].to_numpy(float) - r) * dt
    w["_resid"] = r * dt
    w["_nvar"] = w["pred_nee_std"].to_numpy(float) ** 2
    g = w.groupby(["_site", "window_id"], sort=False)

    ks = [1, 2, 4, 8, 16]
    Vp, Vr, Vn = [], [], []
    for k in ks:
        P = g["_phys"].transform(lambda s: s.rolling(k).sum())
        R = g["_resid"].transform(lambda s: s.rolling(k).sum())
        N = g["_nvar"].transform(lambda s: s.rolling(k).sum())
        m = np.isfinite(P) & np.isfinite(R) & np.isfinite(N)
        Vp.append(np.nanvar(P[m])); Vr.append(np.nanvar(R[m])); Vn.append(np.nanmean(N[m]))
    lk = np.log(ks)
    ap = np.polyfit(lk, np.log(Vp), 1)[0]; ar = np.polyfit(lk, np.log(Vr), 1)[0]; an = np.polyfit(lk, np.log(Vn), 1)[0]

    # within-night autocorrelation of residual rate and noise sample
    def acf(col, maxlag=6):
        out = []
        for L in range(1, maxlag + 1):
            v = []
            for _, gg in g:
                x = gg[col].to_numpy(float)
                if len(x) > L + 2 and np.nanstd(x) > 0:
                    a, b = x[:-L], x[L:]
                    mm = np.isfinite(a) & np.isfinite(b)
                    if mm.sum() > 3 and np.std(a[mm]) > 0 and np.std(b[mm]) > 0:
                        v.append(np.corrcoef(a[mm], b[mm])[0, 1])
            out.append(np.nanmean(v))
        return out
    ac_r = acf("pred_residual"); ac_n = acf("pred_noise")

    print(f"slopes: physics {ap:.2f}  residual {ar:.2f}  noise {an:.2f}")
    print(f"acf lag1: residual {ac_r[0]:.2f}  noise {ac_n[0]:.2f}")

    # --- figure 1: variance-vs-window scaling (log-log) ---
    hrs = [k * 0.5 for k in ks]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(hrs, Vp, "-o", color=C["blue"], lw=LW, ms=MSZ, label=f"Physics drift  (slope {ap:.2f})")
    ax.plot(hrs, Vr, "-s", color=C["green"], lw=LW, ms=MSZ - 0.5, label=f"Residual  (slope {ar:.2f})")
    ax.plot(hrs, Vn, "-^", color=C["vermillion"], lw=LW, ms=MSZ - 0.5, label=f"Noise head  (slope {an:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(hrs); ax.set_xticklabels(["30 min", "1 h", "2 h", "4 h", "8 h"])
    ax.set_xlabel("Within-night window")
    ax.set_ylabel("Increment variance (µmol² m⁻⁴ s⁻²)")
    ax.legend(frameon=False, loc="upper left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_residual_variance_scaling",
         "Variance of the summed physics-drift, residual and noise increments versus the within-night "
         "window (log-log), with fitted power-law slopes. Only the physics drift accumulates coherently "
         "(slope 1.84); the residual (1.05) and the white noise (0.85) scale ~linearly — the residual is a "
         "stationary structured correction, not an accumulating second drift.")

    # --- figure 2: within-night autocorrelation ---
    lags = np.arange(1, len(ac_r) + 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.axhline(0, color=C["grey"], lw=1)
    ax.plot(lags, ac_r, "-s", color=C["green"], lw=LW, ms=MSZ, label="Residual  (ρ₁ = %.2f)" % ac_r[0])
    ax.plot(lags, ac_n, "-^", color=C["vermillion"], lw=LW, ms=MSZ, label="Noise head  (ρ₁ = %.2f)" % ac_n[0])
    ax.set_xlabel("Within-night lag (× 30 min)")
    ax.set_ylabel("Autocorrelation (–)")
    ax.set_xticks(lags)
    ax.legend(frameon=False, loc="upper right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_residual_autocorrelation",
         "Within-night autocorrelation of the residual and of the aleatoric noise versus lag. The residual "
         "is temporally persistent (lag-1 ≈ 0.55, decaying over ~2 h) whereas the noise head is white "
         "(≈ 0): the residual carries systematic structure the noise does not, so its variance share does "
         "not average away.")


if __name__ == "__main__":
    main()
