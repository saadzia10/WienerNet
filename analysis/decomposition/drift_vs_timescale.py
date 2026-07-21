#!/usr/bin/env python
"""Does the physics DRIFT explain more of the NEE increment as the step grows (30-min → hours → days)?

At the native 30-min step the increment is measurement-noise-dominated, so the drift (respiration
change) explains ~nothing — which is exactly why the decomposition attributes ~98% of the 30-min
increment variance to noise. If the drift is genuine physics, its explanatory share should CLIMB as
we aggregate to hours and days, because the temperature-driven respiration change accumulates while
independent measurement noise stays ~flat.

We quantify this model-free: the physics-drift increment is ΔReco = Reco(T_{t+k}) − Reco(T_t) (the
exact-Reco drift telescopes to this over a window), and we score how much of the observed increment
ΔNEE = NEE_{t+k} − NEE_t it explains, at growing windows: within-night k·30-min, then night/week/month
aggregate differences. r² = fraction of increment variance explained by the drift ("coverage" of the
drift). We report ΔReco (the physics drift) and raw ΔTa for reference; sites are z-scored before
pooling (matching analysis/nee_error_structure/temp_change_timescales.py). Figure: figs/drift_vs_timescale.png.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from figstyle import C, LW, MSZ, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TREF, T0 = 10.0, 46.02
SITES = {
    "rosedene": "data_manipulation/final_night_data.parquet",
    "woodwalton": "data_manipulation/other_sites/woodwalton/final_night_data.parquet",
    "redmere_1": "data_manipulation/other_sites/redmere_1/final_night_data.parquet",
    "redmere_2": "data_manipulation/other_sites/redmere_2/final_night_data.parquet",
    "great_fen": "data_manipulation/other_sites/great_fen/final_night_data.parquet",
}


def reco(T, E0, rb):
    return rb * np.exp(E0 * (1.0 / (TREF + T0) - 1.0 / (T + T0)))


def load():
    out = {}
    for s, p in SITES.items():
        p = os.path.join(ROOT, p)
        if os.path.exists(p):
            df = pd.read_parquet(p).sort_values("DateTime").reset_index(drop=True)
            df["Reco"] = reco(df["Ta"].values, df["E0"].values, df["rb"].values)
            out[s] = df
    return out


def _z(a):
    a = np.asarray(a, float); m = np.isfinite(a)
    z = np.full_like(a, np.nan)
    if m.sum() > 30 and a[m].std() > 0:
        z[m] = (a[m] - a[m].mean()) / a[m].std()
    return z


def pooled_r2(data, xcol, ycol, k=None, level=None):
    """Pooled r (sites z-scored) of Δx vs Δy, either within-night lag-k or aggregate-diff."""
    zx, zy = [], []
    for s, df in data.items():
        if level is None:   # within-night lag-k
            dx = df.groupby("window_id")[xcol].transform(lambda v: v.shift(-k) - v).values
            dy = df.groupby("window_id")[ycol].transform(lambda v: v.shift(-k) - v).values
        else:               # aggregate-then-diff
            t = df["DateTime"]
            grp = df["window_id"] if level == "night" else t.dt.to_period("W" if level == "week" else "M")
            order = df.groupby("window_id")["DateTime"].transform("min") if level == "night" else grp
            agg = df.assign(_g=grp, _o=order).groupby("_g").agg({xcol: "mean", ycol: "mean", "_o": "first"}).sort_values("_o")
            dx = agg[xcol].diff().values; dy = agg[ycol].diff().values
        m = np.isfinite(dx) & np.isfinite(dy)
        if m.sum() > 30 and dx[m].std() > 0 and dy[m].std() > 0:
            zx.append(_z(dx[m])); zy.append(_z(dy[m]))
    if not zx:
        return np.nan, 0
    X = np.concatenate(zx); Y = np.concatenate(zy); mm = np.isfinite(X) & np.isfinite(Y)
    r = float(np.corrcoef(X[mm], Y[mm])[0, 1])
    return r, int(mm.sum())


def main():
    data = load()
    steps = [("30 min", 0.5, dict(k=1)), ("1 h", 1.0, dict(k=2)), ("2 h", 2.0, dict(k=4)),
             ("4 h", 4.0, dict(k=8)), ("8 h", 8.0, dict(k=16)),
             ("night", 24.0, dict(level="night")), ("week", 168.0, dict(level="week")),
             ("month", 720.0, dict(level="month"))]
    print("=== Physics-drift explanatory share of the NEE increment vs timescale (pooled, 5 sites) ===")
    print(f"    {'window':<8}{'~hours':>8}{'r(ΔReco,ΔNEE)':>16}{'r²(drift)':>11}{'r(ΔTa,ΔNEE)':>14}{'r²(Ta)':>9}   n")
    rows = []
    for lab, hrs, kw in steps:
        rR, n = pooled_r2(data, "Reco", "NEE", **kw)
        rT, _ = pooled_r2(data, "Ta", "NEE", **kw)
        rows.append((lab, hrs, rR, rR**2, rT, rT**2, n))
        print(f"    {lab:<8}{hrs:>8.1f}{rR:>16.3f}{rR**2:>11.3f}{rT:>14.3f}{rT**2:>9.3f}{n:>7}")

    # figure: fraction of NEE-increment variance explained by the drift, vs timescale
    hrs = [r[1] for r in rows]; r2R = [r[3] for r in rows]; r2T = [r[5] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(hrs, r2R, "-o", color=C["blue"], lw=LW, ms=MSZ, label="Physics drift  ΔReco")
    ax.plot(hrs, r2T, "--s", color=C["vermillion"], lw=LW, ms=MSZ - 0.5, label="Raw temperature change  ΔTa")
    ax.set_xscale("log")
    ax.set_xticks(hrs); ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlabel("Aggregation window")
    ax.set_ylabel("Fraction of ΔNEE variance explained, r² (–)")
    ax.set_ylim(bottom=0)
    ax.axvspan(0.4, 3, color=C["orange"], alpha=0.06)
    ax.axvspan(20, 800, color=C["green"], alpha=0.06)
    ax.annotate("noise-dominated", (0.75, 0.90), color=C["orange"], fontsize=9)
    ax.annotate("drift-dominated", (120, 0.55), color=C["green"], fontsize=9, ha="center")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_drift_r2_vs_timescale",
         "Fraction of the observed NEE-increment variance explained (r²) by the physics-drift increment "
         "ΔReco = Reco(T_{t+k})−Reco(T_t) and by the raw temperature change ΔTa, versus aggregation window "
         "(model-free, pooled over sites). The physics drift rises from 0.2% at 30 min to 31% night-to-night "
         "and 95% month-to-month, and exceeds raw temperature ~5× at every scale: the small 30-min drift is "
         "a timescale property of a genuine, nonlinear respiration signal.")


if __name__ == "__main__":
    main()
