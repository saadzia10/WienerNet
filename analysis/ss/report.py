#!/usr/bin/env python
"""WienerNet-SS report: eval metrics for the 16 combos + baselines, plus synthetic
gap-filling (physics mean-rollout vs persistence).

Gap-filling: during a gap the TEMPERATURE is still measured, so the physics
reconstructs the flux level as Reco(T) anchored at the gap start (the mean rollout of
the exact-Reco drift telescopes to exactly this). Persistence carries the last value.
A no-physics model has no Reco to reconstruct with -> it can only persist.
"""
from __future__ import annotations
import json, os, glob, csv
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SS = os.path.join(ROOT, "outputs", "ss_sweep")
FINAL = os.path.join(ROOT, "outputs", "final_loso")
OUT = os.path.join(ROOT, "analysis", "ss")
TREF, T0 = 10.0, 46.02
def reco(T, E0, rb): return rb * np.exp(E0 * (1 / (TREF + T0) - 1 / (T + T0)))


def read_metrics(run):
    p = os.path.join(run, "metrics", "probabilistic.json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))["global"]
    e = g.get("ensemble") or g.get("probabilistic")
    if not e or e.get("crps") is None:
        return None
    out = {"crps": e["crps"]}
    cov = e.get("coverage") or {}
    out["cov90"] = (cov.get("0.90") or {}).get("coverage")
    out["cov95"] = (cov.get("0.95") or {}).get("coverage")
    out["pit_ks"] = (e.get("pit") or {}).get("ks_uniform")
    # point RMSE from per_site.csv
    ps = os.path.join(run, "metrics", "per_site.csv")
    if os.path.exists(ps):
        for row in csv.DictReader(open(ps)):
            if row.get("target") == "nee" and row.get("metric") == "rmse":
                out["rmse"] = float(row["value"])
    return out


def gap_fill(run, days=(1, 3, 7, 14)):
    """Reconstruct D-day gaps of daily-mean nighttime NEE: physics Reco(T) anchored vs
    persistence. Returns {D: (physics_rmse, persistence_rmse)}."""
    p = os.path.join(run, "metrics", "predictions.parquet")
    if not os.path.exists(p):
        return {}
    w = pd.read_parquet(p)
    if not {"NEE", "Ta", "E0", "rb", "DateTime"}.issubset(w.columns):
        return {}
    w = w.dropna(subset=["NEE", "Ta", "E0", "rb"]).copy()
    w["reco"] = reco(w["Ta"].values, w["E0"].values, w["rb"].values)
    daily = (w.assign(DateTime=pd.to_datetime(w["DateTime"]))
             .set_index("DateTime").resample("D").agg(NEE=("NEE", "mean"), reco=("reco", "mean")).dropna())
    if len(daily) < 20:
        return {}
    b, a = np.polyfit(daily["reco"], daily["NEE"], 1)   # night NEE ~ a + b*Reco
    phys = a + b * daily["reco"].values
    obs = daily["NEE"].values
    def rmse(x, y):
        m = np.isfinite(x) & np.isfinite(y); return float(np.sqrt(np.mean((x[m] - y[m]) ** 2)))
    res = {}
    for D in days:
        pers = np.concatenate([np.full(D, np.nan), obs[:-D]]) if D < len(obs) else np.full_like(obs, np.nan)
        res[D] = (rmse(phys, obs), rmse(pers, obs))
    return res


COMBOS = [f"{k}_{d}_{r}_{n}" for k in ("gtk", "pk") for d in ("diur", "ldiur")
          for r in ("res", "nores") for n in ("ss", "wien")]
BASELINES = {  # from the earlier final_loso sweep (woodwalton, seed 0)
    "WienerNet-Laplace (30m, GT-k)": os.path.join(FINAL, "wienernet_laplace_woodwalton_s0"),
    "MDN (no physics)": os.path.join(FINAL, "comp_mdn_woodwalton_s0"),
    "Analytical SDE": os.path.join(FINAL, "prior_analytical_woodwalton_s0"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name in COMBOS:
        run = os.path.join(SS, name)
        m = read_metrics(run)
        if m:
            m["combo"] = name; m["group"] = "SS"; m["gap"] = gap_fill(run); rows.append(m)
    for disp, run in BASELINES.items():
        m = read_metrics(run)
        if m:
            m["combo"] = disp; m["group"] = "baseline"; m["gap"] = gap_fill(run); rows.append(m)

    # eval-metric table (sorted by CRPS)
    rows_ok = [r for r in rows if r.get("crps") is not None]
    rows_ok.sort(key=lambda r: r["crps"])
    print(f"\n{'model':<34}{'CRPS':>8}{'cov90':>8}{'cov95':>8}{'PITks':>8}{'RMSE':>8}")
    for r in rows_ok:
        print(f"{r['combo']:<34}{r['crps']:>8.3f}"
              f"{(r.get('cov90') or float('nan')):>8.3f}{(r.get('cov95') or float('nan')):>8.3f}"
              f"{(r.get('pit_ks') or float('nan')):>8.3f}{(r.get('rmse') or float('nan')):>8.3f}")

    # gap-filling table (physics vs persistence at D days), a representative combo per k-regime
    print(f"\n=== gap-filling RMSE (physics Reco(T) reconstruction vs persistence) ===")
    print(f"    {'model':<34}" + "".join(f"{'D='+str(D):>16}" for D in (1, 7, 14)))
    for r in rows_ok:
        g = r.get("gap") or {}
        if not g:
            continue
        cells = ""
        for D in (1, 7, 14):
            if D in g:
                cells += f"  phys{g[D][0]:>5.2f}/pers{g[D][1]:>5.2f}"
            else:
                cells += f"{'-':>16}"
        print(f"    {r['combo']:<34}{cells}")

    # CSV
    with open(os.path.join(OUT, "ss_report.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "group", "crps", "cov90", "cov95", "pit_ks", "rmse",
                     "gap1_phys", "gap1_pers", "gap14_phys", "gap14_pers"])
        for r in rows_ok:
            g = r.get("gap") or {}
            wr.writerow([r["combo"], r["group"],
                         round(r["crps"], 4), r.get("cov90"), r.get("cov95"), r.get("pit_ks"), r.get("rmse"),
                         g.get(1, (None, None))[0], g.get(1, (None, None))[1],
                         g.get(14, (None, None))[0], g.get(14, (None, None))[1]])
    print("\nwrote", os.path.join(OUT, "ss_report.csv"))
    return rows_ok


if __name__ == "__main__":
    main()
