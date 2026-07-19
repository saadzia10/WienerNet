#!/usr/bin/env python
"""WienerNet-SS multi-site generalisation report.

Reads the ss_loso sweep (4 SS ablations x 5 held-out sites x 3 seeds) and the REUSED
final_loso competing methods (MDN / Neural SDE / mean-var / analytical / RF / XGB), and asks
the Task-1 question: do the Woodwalton wins hold across all five held-out sites?

Outputs (all under analysis/ss/):
  * loso_metrics_long.csv  — one row per (model, site, seed): CRPS, cov90/95, PIT-KS, RMSE
  * loso_summary.csv       — per (model, site): mean +/- std over seeds
  * printed tables         — per-site headline comparison + cross-site means
Probabilistic metrics (CRPS/coverage/PIT) come from metrics/probabilistic.json (ensemble);
point RMSE from metrics/per_site.csv (torch) or metrics.json raw.nee.rmse (trees).
"""
from __future__ import annotations
import json, os, csv
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOSO = os.path.join(ROOT, "outputs", "ss_loso")
FL = os.path.join(ROOT, "outputs", "final_loso")
OUT = os.path.join(ROOT, "analysis", "ss")
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
SEEDS = [0, 1, 42]

# display name -> (parent dir, run-name template with {site},{seed}, is_point)
SS_ABL = {
    # PRIMARY: learned-diurnal + known params, both noise laws, residual off/on ----
    "WienerNet-SS (learned-diurnal, Wiener)":          (LOSO, "{site}_s{seed}_ldiur_wien", False),
    "WienerNet-SS (learned-diurnal, state-sp)":        (LOSO, "{site}_s{seed}_ldiur_ss", False),
    "WienerNet-SS (learned-diurnal, Wiener, +resid)":  (LOSO, "{site}_s{seed}_ldiur_res_wien", False),
    "WienerNet-SS (learned-diurnal, state-sp, +resid)":(LOSO, "{site}_s{seed}_ldiur_res_ss", False),
    # exogenous-diurnal comparison arm (climatology lookup) ------------------------
    "WienerNet-SS (diurnal, state-space)":             (LOSO, "{site}_s{seed}_diur_ss", False),
    "WienerNet-SS (diurnal, Wiener)":                  (LOSO, "{site}_s{seed}_diur_wien", False),
}
COMPETING = {
    "MDN (no physics)":  (FL, "comp_mdn_{site}_s{seed}", False),
    "Neural SDE":        (FL, "comp_neuralsde_{site}_s{seed}", False),
    "Mean-Var Gaussian": (FL, "comp_meanvar_{site}_s{seed}", False),
    "Analytical SDE":    (FL, "prior_analytical_{site}_s{seed}", False),
    "Random Forest":     (FL, "comp_rf_{site}_s{seed}", True),
    "XGBoost":           (FL, "comp_xgb_{site}_s{seed}", True),
}
HEADLINE = "WienerNet-SS (learned-diurnal, Wiener)"


def read_prob(run):
    p = os.path.join(run, "metrics", "probabilistic.json")
    if not os.path.exists(p):
        return {}
    g = json.load(open(p)).get("global", {})
    e = g.get("ensemble") or g.get("probabilistic")
    if not e or e.get("crps") is None:
        return {}
    cov = e.get("coverage") or {}
    return dict(crps=e["crps"], cov90=(cov.get("0.90") or {}).get("coverage"),
                cov95=(cov.get("0.95") or {}).get("coverage"),
                pit_ks=(e.get("pit") or {}).get("ks_uniform"))


def read_rmse(run, is_point):
    if is_point:
        mj = os.path.join(run, "metrics.json")
        if os.path.exists(mj):
            try:
                return float(json.load(open(mj))["raw"]["nee"]["rmse"])
            except Exception:
                return None
        return None
    ps = os.path.join(run, "metrics", "per_site.csv")
    if os.path.exists(ps):
        for row in csv.DictReader(open(ps)):
            if row.get("target") == "nee" and row.get("metric") == "rmse":
                return float(row["value"])
    return None


def collect():
    rows = []
    for name, (parent, tmpl, is_pt) in {**SS_ABL, **COMPETING}.items():
        for site in SITES:
            for seed in SEEDS:
                run = os.path.join(parent, tmpl.format(site=site, seed=seed))
                if not os.path.isdir(run):
                    continue
                rec = dict(model=name, site=site, seed=seed, is_point=is_pt)
                rec.update(read_prob(run))
                rec["rmse"] = read_rmse(run, is_pt)
                if rec.get("crps") is not None or rec.get("rmse") is not None:
                    rows.append(rec)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = collect()
    if df.empty:
        print("no runs found yet under", LOSO); return
    df.to_csv(os.path.join(OUT, "loso_metrics_long.csv"), index=False)

    metrics = ["crps", "cov90", "cov95", "pit_ks", "rmse"]
    agg = (df.groupby(["model", "site"])[metrics].agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join([c for c in col if c]).strip("_") for col in agg.columns]
    agg.to_csv(os.path.join(OUT, "loso_summary.csv"), index=False)

    order = list(SS_ABL) + list(COMPETING)
    # ---- per-site headline comparison (CRPS / cov90 / RMSE, mean over seeds) ----
    print("\n================ PER-SITE (mean over seeds) — CRPS | cov90 | RMSE ================")
    comp_models = [HEADLINE, "WienerNet-SS (learned-diurnal, state-sp)",
                   "WienerNet-SS (learned-diurnal, Wiener, +resid)",
                   "WienerNet-SS (learned-diurnal, state-sp, +resid)",
                   "MDN (no physics)", "Neural SDE", "Analytical SDE", "Random Forest", "XGBoost"]
    for site in SITES:
        print(f"\n-- {site} --")
        print(f"{'model':<40}{'CRPS':>8}{'cov90':>8}{'RMSE':>8}   nseed")
        sub = df[df.site == site]
        for m in comp_models:
            r = sub[sub.model == m]
            if r.empty:
                print(f"{m:<40}{'—':>8}{'—':>8}{'—':>8}"); continue
            crps = r["crps"].mean() if r["crps"].notna().any() else float('nan')
            cov = r["cov90"].mean() if "cov90" in r and r["cov90"].notna().any() else float('nan')
            rmse = r["rmse"].mean() if r["rmse"].notna().any() else float('nan')
            print(f"{m:<40}{crps:>8.3f}{cov:>8.3f}{rmse:>8.2f}{len(r):>7}")

    # ---- cross-site summary (mean +/- std over all site x seed) ----
    print("\n================ CROSS-SITE (mean +/- std over all site x seed) ================")
    print(f"{'model':<40}{'CRPS':>14}{'cov90':>14}{'RMSE':>14}")
    for m in order:
        r = df[df.model == m]
        if r.empty:
            continue
        def ms(col, f):
            v = r[col].dropna()
            return f"{v.mean():.{f}f}±{v.std():.{f}f}" if len(v) else "—"
        print(f"{m:<40}{ms('crps',3):>14}{ms('cov90',3):>14}{ms('rmse',2):>14}")

    # ---- head-to-head: how often does the headline SS beat MDN per site x seed ----
    h = df[df.model == HEADLINE].set_index(["site", "seed"])
    mdn = df[df.model == "MDN (no physics)"].set_index(["site", "seed"])
    common = h.index.intersection(mdn.index)
    if len(common):
        wins_crps = int((h.loc[common, "crps"].values < mdn.loc[common, "crps"].values).sum())
        wins_rmse = int((h.loc[common, "rmse"].values < mdn.loc[common, "rmse"].values).sum())
        print(f"\nHeadline SS vs MDN over {len(common)} site×seed: "
              f"CRPS wins {wins_crps}/{len(common)}, RMSE wins {wins_rmse}/{len(common)}")
    print("\nwrote", os.path.join(OUT, "loso_metrics_long.csv"), "and loso_summary.csv")
    return df


if __name__ == "__main__":
    main()
