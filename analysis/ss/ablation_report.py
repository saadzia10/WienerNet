#!/usr/bin/env python
"""Comprehensive ablation report for the final WienerNet-SS pass.

Two things a manuscript needs:
  A. WienerNet-SS ablation axes vs the PRIMARY (learned-diurnal, known-k, single-Wiener,
     no-residual, ALD): noise law, residual on/off, diurnal tendency source, likelihood family
     (Gaussian / beta-NLL / Student-t / ALD / mixture), and parameter head (known vs predicted).
  B. FAIR comparison — architecture x loss-family: WienerNet-SS vs the process baselines
     (NeuralSDE, mean-variance, Analytical, MDN) all on the SAME likelihood family, so a skill
     difference reflects architecture, not the noise objective.

Reads outputs/ss_loso/ (all WienerNet-SS ablations + the Student-t/ALD baselines) and
outputs/final_loso/ (the Gaussian baselines + the Student-t MDN + trees). Writes CSVs to
analysis/ss/. Mean +/- std over the 3 seeds; per held-out site + cross-site.
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


def read_prob(run):
    """Full Stage-1 distributional metrics from probabilistic.json (ensemble block for CRPS/PIT/
    coverage/sharpness; parametric block for NLL; measurement-noise floor)."""
    p = os.path.join(run, "metrics", "probabilistic.json")
    if not os.path.exists(p):
        return {}
    g = json.load(open(p)).get("global", {})
    e = g.get("ensemble") or g.get("probabilistic")
    if not e or e.get("crps") is None:
        return {}
    cov = e.get("coverage") or {}
    pit = e.get("pit") or {}
    par = g.get("probabilistic") or {}
    mnf = g.get("measurement_noise_floor") or {}
    def c(lvl, field):
        return (cov.get(lvl) or {}).get(field)
    return dict(
        crps=e["crps"], nll=par.get("nll"),
        cov50=c("0.50", "coverage"), cov90=c("0.90", "coverage"), cov95=c("0.95", "coverage"),
        sharp50=c("0.50", "sharpness"), sharp90=c("0.90", "sharpness"), sharp95=c("0.95", "sharpness"),
        pit_mean=pit.get("mean"), pit_var=pit.get("var"), pit_ks=pit.get("ks_uniform"),
        rmse_floor=mnf.get("rmse_floor"))


def read_proc(run):
    """Stage-1 process-consistency diagnostics (§1.7) from process_consistency.json: standardised-
    residual whiteness (mean_z, std_z, lag-1 ACF), drift-check R², noise-shape moments + energy
    distance, and the diffusion variance-vs-scale ratio at the longest window."""
    p = os.path.join(run, "metrics", "process_consistency.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    d = d.get("global", d)
    ra = d.get("residual_autocorrelation") or {}
    summ = ra.get("summary") or {}
    acf = ra.get("acf")
    dc = d.get("drift_check") or {}
    nc = d.get("noise_check") or {}
    emp = nc.get("empirical") or {}
    vv = d.get("variance_vs_scale")
    vvr = vv[-1].get("ratio") if isinstance(vv, list) and vv else None
    return dict(
        mean_z=summ.get("mean_z"), std_z=summ.get("std_z"),
        acf1=(acf[1] if isinstance(acf, list) and len(acf) > 1 else None),
        drift_r2=dc.get("r2"),
        noise_skew=emp.get("skew"), noise_exkurt=emp.get("exkurt"), noise_tail3sd=emp.get("tail_3sd"),
        energy_distance=nc.get("energy_distance"), vvs_ratio_maxwin=vvr)


def read_rmse(run, is_point=False):
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


def collect(spec):
    """spec: name -> (parent, template, is_point). Returns long df over site x seed."""
    rows = []
    for name, (parent, tmpl, is_pt) in spec.items():
        for site in SITES:
            for seed in SEEDS:
                run = os.path.join(parent, tmpl.format(site=site, seed=seed))
                if not os.path.isdir(run):
                    continue
                rec = dict(model=name, site=site, seed=seed)
                rec.update(read_prob(run))
                rec.update(read_proc(run))
                rec["rmse"] = read_rmse(run, is_pt)
                if rec.get("crps") is not None or rec.get("rmse") is not None:
                    rows.append(rec)
    return pd.DataFrame(rows)


def ms(df, model, col, f=3):
    r = df[df.model == model][col].dropna()
    return f"{r.mean():.{f}f}±{r.std():.{f}f}" if len(r) else "—"


# ---- A. WienerNet-SS ablation axes (all in ss_loso) -----------------------------------
WNSS = {
    "PRIMARY (ldiur, GTk, Wiener, no-res, ALD)": (LOSO, "{site}_s{seed}_ldiur_wien", False),
    "  noise: state-space":                       (LOSO, "{site}_s{seed}_ldiur_ss", False),
    "  residual: ON (Wiener)":                     (LOSO, "{site}_s{seed}_ldiur_res_wien", False),
    "  residual: ON (state-space)":                (LOSO, "{site}_s{seed}_ldiur_res_ss", False),
    "  tendency: given-diurnal (Wiener)":          (LOSO, "{site}_s{seed}_diur_wien", False),
    "  tendency: given-diurnal (state-space)":     (LOSO, "{site}_s{seed}_diur_ss", False),
    "  likelihood: Gaussian":                      (LOSO, "{site}_s{seed}_ldiur_wien_gauss", False),
    "  likelihood: beta-NLL":                      (LOSO, "{site}_s{seed}_ldiur_wien_beta", False),
    "  likelihood: Student-t":                     (LOSO, "{site}_s{seed}_ldiur_wien_studt", False),
    "  likelihood: mixture(3)":                    (LOSO, "{site}_s{seed}_ldiur_wien_mix", False),
    "  params: predicted-k":                       (LOSO, "{site}_s{seed}_ldiur_wien_predk", False),
}

# ---- B. Fair comparison — architecture x loss family --------------------------------
# rows = architecture, columns = likelihood family; each cell is a run spec.
FAIR = {
    "WienerNet-SS": {
        "Gaussian":  (LOSO, "{site}_s{seed}_ldiur_wien_gauss"),
        "Student-t": (LOSO, "{site}_s{seed}_ldiur_wien_studt"),
        "ALD":       (LOSO, "{site}_s{seed}_ldiur_wien"),
        "mixture":   (LOSO, "{site}_s{seed}_ldiur_wien_mix"),
    },
    "Neural SDE": {
        "Gaussian":  (FL, "comp_neuralsde_{site}_s{seed}"),
        "Student-t": (LOSO, "{site}_s{seed}_base_nsde_studt"),
        "ALD":       (LOSO, "{site}_s{seed}_base_nsde_ald"),
    },
    "mean-variance": {
        "Gaussian":  (FL, "comp_meanvar_{site}_s{seed}"),
        "Student-t": (LOSO, "{site}_s{seed}_base_mv_studt"),
        "ALD":       (LOSO, "{site}_s{seed}_base_mv_ald"),
    },
    "Analytical SDE": {
        "Gaussian":  (FL, "prior_analytical_{site}_s{seed}"),
        "Student-t": (LOSO, "{site}_s0_base_analyt_studt"),   # seed-independent
    },
    "MDN": {
        "mixture":   (FL, "comp_mdn_{site}_s{seed}"),
    },
}
FAMILIES = ["Gaussian", "beta-NLL", "Student-t", "ALD", "mixture"]


def cell_stat(parent, tmpl, metric, sites=None):
    vals = []
    for site in (sites or SITES):
        for seed in SEEDS:
            run = os.path.join(parent, tmpl.format(site=site, seed=seed))
            if not os.path.isdir(run):
                continue
            m = read_prob(run) if metric != "rmse" else {}
            v = m.get(metric) if metric != "rmse" else read_rmse(run)
            if v is not None:
                vals.append(v)
    return np.array(vals)


# Redmere 1 is a corrupted-Tau OOD artifact (see docs/redmere1_blowup_analysis.md); the
# "clean" view over the four in-distribution sites separates architecture skill from the
# single-site OOD blow-up that dominates the 5-site mean and its variance.
SITES_CLEAN = [s for s in SITES if s != "redmere_1"]


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- A ----
    dfA = collect(WNSS)
    dfA.to_csv(os.path.join(OUT, "ablation_wnss_long.csv"), index=False)
    print("================ A. WienerNet-SS ablation axes (cross-site mean±std) ================")
    print(f"{'variant':<44}{'CRPS':>14}{'cov90':>14}{'RMSE':>13}   n")
    for m in WNSS:
        r = dfA[dfA.model == m]
        print(f"{m:<44}{ms(dfA,m,'crps'):>14}{ms(dfA,m,'cov90'):>14}{ms(dfA,m,'rmse',2):>13}{len(r):>4}")

    # full Stage-1 metric set — CLEAN 4 sites, MEDIAN. The unbounded metrics (NLL, std_z) are
    # dominated at their 5-site MEAN by the Redmere-1 OOD blow-up (e.g. primary NLL 4.4e6 there, so
    # the 5-site mean is meaningless); the clean-site median is the honest summary. Every per-run
    # value (all sites/seeds) is in ablation_wnss_long.csv, so any aggregate is recomputable.
    clean = dfA[dfA.site != "redmere_1"]
    def mnc(m, col, f=2):
        v = clean[clean.model == m][col].dropna() if col in clean.columns else pd.Series(dtype=float)
        return f"{v.median():.{f}f}" if len(v) else "—"
    print("\n---- full Stage-1 set (CLEAN 4 sites, median; unbounded metrics are OOD-dominated at the 5-site mean) ----")
    print(f"{'variant':<44}{'NLL':>8}{'cov50':>7}{'cov95':>7}{'shrp90':>8}{'PITvar':>8}{'std_z':>7}{'drR2':>7}{'exkurt':>8}{'VvS':>7}")
    for m in WNSS:
        print(f"{m:<44}{mnc(m,'nll'):>8}{mnc(m,'cov50',2):>7}{mnc(m,'cov95',2):>7}{mnc(m,'sharp90'):>8}"
              f"{mnc(m,'pit_var',3):>8}{mnc(m,'std_z'):>7}{mnc(m,'drift_r2',2):>7}{mnc(m,'noise_exkurt',1):>8}{mnc(m,'vvs_ratio_maxwin'):>7}")

    # ---- B ----
    print("\n================ B. Fair comparison — CRPS by architecture × loss family ================")
    print("   (same objective across a column → differences reflect ARCHITECTURE; mean±std over site×seed)")
    hdr = f"{'architecture':<16}" + "".join(f"{fam:>16}" for fam in FAMILIES)
    print(hdr); print("-" * len(hdr))
    csv_rows = []
    for arch, fams in FAIR.items():
        cells = {}
        for fam in FAMILIES:
            if fam in fams:
                v = cell_stat(*fams[fam], "crps")
                cells[fam] = f"{v.mean():.3f}±{v.std():.3f}" if len(v) else "—"
            else:
                cells[fam] = "—"
        print(f"{arch:<16}" + "".join(f"{cells[fam]:>16}" for fam in FAMILIES))
        csv_rows.append({"architecture": arch, **cells})
    with open(os.path.join(OUT, "ablation_fair_crps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["architecture"] + FAMILIES); w.writeheader(); w.writerows(csv_rows)

    # Same CRPS table over the 4 in-distribution sites (Redmere 1 excluded) — separates
    # architecture skill from the corrupted-Tau OOD blow-up that dominates the 5-site mean.
    print("\n---- CRPS by architecture × loss family, REDMERE 1 EXCLUDED (4 clean sites) ----")
    print(hdr); print("-" * len(hdr))
    for arch, fams in FAIR.items():
        cells = {}
        for fam in FAMILIES:
            if fam in fams:
                v = cell_stat(*fams[fam], "crps", sites=SITES_CLEAN)
                cells[fam] = f"{v.mean():.3f}±{v.std():.3f}" if len(v) else "—"
            else:
                cells[fam] = "—"
        print(f"{arch:<16}" + "".join(f"{cells[fam]:>16}" for fam in FAMILIES))

    # same table for coverage (calibration) + RMSE
    for metric, fmt in [("cov90", "calibration (90% coverage)"), ("rmse", "point RMSE")]:
        print(f"\n---- {fmt} by architecture × loss family ----")
        print(hdr)
        for arch, fams in FAIR.items():
            cells = {}
            for fam in FAMILIES:
                if fam in fams:
                    v = cell_stat(*fams[fam], metric)
                    cells[fam] = (f"{v.mean():.3f}" if metric == "cov90" else f"{v.mean():.2f}") if len(v) else "—"
                else:
                    cells[fam] = "—"
            print(f"{arch:<16}" + "".join(f"{cells[fam]:>16}" for fam in FAMILIES))
    # ---- C. Stage-2 gap-filling metrics per likelihood ablation + matched-loss baselines ----
    gap_csv = os.path.join(OUT, "gap_crps_summary.csv")
    if os.path.exists(gap_csv):
        gs = pd.read_csv(gap_csv)
        def gv(model, scope, col):
            r = gs[(gs.model == model) & (gs.scope == scope)]
            try:
                return float(r[col].iloc[0])
            except (ValueError, TypeError, IndexError):
                return float("nan")
        print("\n================ C. Gap-filling (Stage 2; 5h+ horizon, raw band; seed 0) ================")
        print("   probabilistic skill (CRPS) + calibration (PIT-KS, 0=ideal) + 90% coverage vs hours-into-gap")
        print(f"{'model':<26}{'CRPS clean':>11}{'CRPS all5':>11}{'PITks clean':>12}{'cov90 clean':>12}")
        order = ["WN-SS (ALD)", "WN-SS (Gaussian)", "WN-SS (beta-NLL)", "WN-SS (Student-t)", "WN-SS (mixture)",
                 "Neural SDE (ALD)", "mean-var (ALD)", "MDN (mixture)", "Analytical (Student-t)",
                 "Neural SDE (Gaussian)", "Analytical (Gaussian)"]
        for m in order:
            if not len(gs[gs.model == m]):
                continue
            print(f"{m:<26}{gv(m,'clean4','crpsraw_h5+'):>11.3f}{gv(m,'all5','crpsraw_h5+'):>11.3f}"
                  f"{gv(m,'clean4','pitksraw_h5+'):>12.3f}{gv(m,'clean4','covraw_h5+'):>12.3f}")
        print("  (source: analysis/ss/gap_crps_summary.csv — all metrics × horizons × sites)")

    print("\nwrote", os.path.join(OUT, "ablation_wnss_long.csv"), "and ablation_fair_crps.csv")


if __name__ == "__main__":
    main()
