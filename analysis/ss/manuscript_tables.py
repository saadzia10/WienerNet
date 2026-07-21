#!/usr/bin/env python
"""Manuscript results tables: every model / ablation x (Stage-1 distributional, Stage-2 gap-filling),
in three views — clean 4 sites, all 5 sites (including the Redmere-1 OOD stress), and site-wise.

Values are `mean ± std` (1 SD). What the ± is taken over differs by stage, because the two stages
were run at different seed depth — this is stated in every table footer rather than papered over:

  * Stage 1 (one-step distributional) — 3 seeds (0/1/42) x sites.
      pooled views : ± over all (site, seed) units.
      site-wise    : ± over the 3 seeds.
  * Stage 2 (AR gap-fill + band)      — 3 seeds (0/1/42) x sites, same depth as Stage 1.
      pooled views : ± over all (site, seed) units.
      site-wise    : ± over the 3 seeds.

Seed-independent models (the calibrated Analytical SDE) are read once; where n = 1 the value is
reported without ±.

Outputs (analysis/ss/manuscript_tables/):
  table_clean4.{csv,md}     — 4 in-distribution sites
  table_all5.{csv,md}       — all 5 sites (Redmere-1 included)
  table_sitewise.{csv,md}   — per held-out site
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ablation_report import read_prob, read_proc, read_rmse   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
LOSO = os.path.join(ROOT, "outputs", "ss_loso")
FL = os.path.join(ROOT, "outputs", "final_loso")
OUTDIR = os.path.join(HERE, "manuscript_tables")
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
CLEAN = [s for s in SITES if s != "redmere_1"]
SEEDS = [0, 1, 42]
SITE_LABEL = {"woodwalton": "Woodwalton", "rosedene": "Rosedene", "redmere_1": "Redmere 1",
              "redmere_2": "Redmere 2", "great_fen": "Great Fen"}

# display name -> (stage-1 parent, stage-1 run template, is_point_model, stage-2 name in ar_gapfill_loso.csv)
MODELS = [
    # --- WienerNet-SS: likelihood axis (the manuscript ablation) ---
    ("WienerNet-SS (ALD, primary)",             LOSO, "{site}_s{seed}_ldiur_wien",       False, "WN-SS (ALD)"),
    ("WienerNet-SS (Gaussian)",                 LOSO, "{site}_s{seed}_ldiur_wien_gauss", False, "WN-SS (Gaussian)"),
    ("WienerNet-SS (beta-NLL)",                 LOSO, "{site}_s{seed}_ldiur_wien_beta",  False, "WN-SS (beta-NLL)"),
    ("WienerNet-SS (Student-t)",                LOSO, "{site}_s{seed}_ldiur_wien_studt", False, "WN-SS (Student-t)"),
    ("WienerNet-SS (mixture)",                  LOSO, "{site}_s{seed}_ldiur_wien_mix",   False, "WN-SS (mixture)"),
    # --- WienerNet-SS: other ablation axes ---
    ("WienerNet-SS (+residual)",                LOSO, "{site}_s{seed}_ldiur_res_wien",   False, "WN-SS (+residual)"),
    ("WienerNet-SS (predicted-k)",              LOSO, "{site}_s{seed}_ldiur_wien_predk", False, "WN-SS (predicted-k)"),
    # --- state-space noise law: the matched partner of each single-Wiener arm above.
    # Only ever run at ALD -- ALD is decisively the best likelihood on single-Wiener, so the
    # likelihood axis was not crossed with the noise law (stated in the manuscript).
    ("WienerNet-SS (state-space)",              LOSO, "{site}_s{seed}_ldiur_ss",         False, "WN-SS (state-space)"),
    ("WienerNet-SS (state-space, +residual)",   LOSO, "{site}_s{seed}_ldiur_res_ss",     False, "WN-SS (state-space, +residual)"),
    ("WienerNet-SS (state-space, predicted-k)", LOSO, "{site}_s{seed}_ldiur_ss_predk",   False, "WN-SS (state-space, predicted-k)"),
    # Version A of the structural-error treatment: the SAME primary run, scored with the
    # +sigma_struct band instead of the raw one. Stage-2 only -- sigma_struct is applied to the
    # rollout band, not inside the model, so it has no Stage-1 row (shown as "—").
    ("WienerNet-SS (ALD + sigma_struct band)",  LOSO, "__stage2_only__",                 False, "WN-SS (ALD) +sigma_struct"),
    ("WienerNet-SS (given-diurnal, Wiener)",    LOSO, "{site}_s{seed}_diur_wien",        False, "WN-SS (given-diurnal, Wiener)"),
    ("WienerNet-SS (given-diurnal, state-sp)",  LOSO, "{site}_s{seed}_diur_ss",          False, "WN-SS (given-diurnal, st-sp)"),
    # --- baselines at their manuscript likelihoods ---
    ("Neural SDE (Gaussian)",       FL,   "comp_neuralsde_{site}_s{seed}",   False, "Neural SDE (Gaussian)"),
    ("Neural SDE (Student-t)",      LOSO, "{site}_s{seed}_base_nsde_studt",  False, "Neural SDE (Student-t)"),
    ("Neural SDE (ALD)",            LOSO, "{site}_s{seed}_base_nsde_ald",    False, "Neural SDE (ALD)"),
    ("Mean-variance (Gaussian)",    FL,   "comp_meanvar_{site}_s{seed}",     False, "mean-var (Gaussian)"),
    ("Mean-variance (Student-t)",   LOSO, "{site}_s{seed}_base_mv_studt",    False, "mean-var (Student-t)"),
    ("Mean-variance (ALD)",         LOSO, "{site}_s{seed}_base_mv_ald",      False, "mean-var (ALD)"),
    ("Analytical SDE (Gaussian)",   FL,   "prior_analytical_{site}_s{seed}", False, "Analytical (Gaussian)"),
    ("Analytical SDE (Student-t)",  LOSO, "{site}_s0_base_analyt_studt",     False, "Analytical (Student-t)"),
    ("MDN (mixture)",               FL,   "comp_mdn_{site}_s{seed}",         False, "MDN (mixture)"),
    ("Random Forest",               FL,   "comp_rf_{site}_s{seed}",          True,  "Random Forest"),
    ("XGBoost",                     FL,   "comp_xgb_{site}_s{seed}",         True,  "XGBoost"),
]

# ---- metric sets -------------------------------------------------------------------------------
# MAIN = the manuscript tables. Reduced on purpose:
#   NLL          dropped — unbounded, so at 5 sites it is 1e6-1e10 with a SD larger than the mean;
#                also a second proper score that duplicates the CRPS ranking.
#   cov50/cov95  dropped — near-collinear with cov90; PIT-KS already tests every quantile.
#   std_z        dropped — redundant with PIT/coverage (std_z >> 1 *is* under-dispersion), OOD-driven.
#   drift_r2     dropped — ~0 for every model (drift is a weak per-step signal at the noise floor);
#                it validates the method, it does not discriminate models -> one sentence of text.
#   noise_exkurt dropped — a property of the DATA residual, ~equal across models.
#   vvs_ratio    dropped — process diagnostic, OOD-sensitive, not readable in a comparison table.
#   gap CRPS at 0-2h/2-5h and gap sharpness dropped — the RMSE-by-horizon trend already carries the
#                stability shape, and cov90+CRPS already carry the band quality.
# FULL = everything, retained in the supplementary CSVs so nothing is lost.
MAIN_S1 = ["crps", "rmse", "cov90", "sharp90", "pit_ks"]
MAIN_S2 = ["rmse_h0-2", "rmse_h2-5", "rmse_h5+", "crpsraw_h5+", "pitksraw_h5+", "covraw_h5+"]
FULL_S1 = ["crps", "nll", "rmse", "cov50", "cov90", "cov95", "sharp90", "pit_ks",
           "std_z", "drift_r2", "noise_exkurt", "vvs_ratio_maxwin"]
FULL_S2 = ["rmse_h0-2", "rmse_h2-5", "rmse_h5+",
           "crpsraw_h0-2", "crpsraw_h2-5", "crpsraw_h5+",
           "pitksraw_h5+", "covraw_h5+", "sharpraw_h5+"]

LABEL = {"crps": "CRPS", "nll": "NLL", "rmse": "RMSE", "cov50": "cov50", "cov90": "cov90",
         "cov95": "cov95", "sharp90": "sharp90", "pit_ks": "PIT-KS", "std_z": "std z",
         "drift_r2": "drift R2", "noise_exkurt": "ex-kurt", "vvs_ratio_maxwin": "VvS ratio",
         "rmse_h0-2": "RMSE 0-2h", "rmse_h2-5": "RMSE 2-5h", "rmse_h5+": "RMSE 5h+",
         "crpsraw_h0-2": "CRPS 0-2h", "crpsraw_h2-5": "CRPS 2-5h", "crpsraw_h5+": "CRPS 5h+",
         "pitksraw_h5+": "PIT-KS 5h+", "covraw_h5+": "cov90 5h+", "sharpraw_h5+": "sharp 5h+"}
PREC = {"crps": 3, "nll": 2, "rmse": 2, "cov50": 3, "cov90": 3, "cov95": 3, "sharp90": 2,
        "pit_ks": 3, "std_z": 2, "drift_r2": 2, "noise_exkurt": 1, "vvs_ratio_maxwin": 2,
        "rmse_h0-2": 2, "rmse_h2-5": 2, "rmse_h5+": 2, "crpsraw_h0-2": 3, "crpsraw_h2-5": 3,
        "crpsraw_h5+": 3, "pitksraw_h5+": 3, "covraw_h5+": 3, "sharpraw_h5+": 2}


def fmt(v, s, n, prec):
    """mean ± std; bare value when a single unit (n<2) or std undefined; em-dash when absent."""
    if v is None or not np.isfinite(v):
        return "—"
    if n is None or n < 2 or s is None or not np.isfinite(s):
        return f"{v:.{prec}f}"
    return f"{v:.{prec}f} ± {s:.{prec}f}"


LRFIX_ROOT = os.path.join(ROOT, "outputs", "ss_lrfix")
FULL5_ROOT = os.path.join(ROOT, "outputs", "ss_full5")
USE_FULL5 = False        # set by main(); see --full5


def _resolve_run(parent, run_name, seed, use_lrfix):
    """Locate a run directory.

    --full5: every arm comes from the single uniform-protocol sweep root, which flattens the old
    ss_loso / final_loso split into one directory; anything absent there falls back to its original
    parent so a partial sweep still renders.

    --lrfix (the older, superseded mode): seed 42 was re-trained with a leakage-free LR schedule
    after it was found to diverge under the original constant-LR protocol; arms with no LR schedule
    to fix (Analytical SDE, RF/XGB) have no re-run and fall back automatically.
    """
    if USE_FULL5:
        alt = os.path.join(FULL5_ROOT, run_name)
        return alt if os.path.isdir(alt) else os.path.join(parent, run_name)
    if use_lrfix and seed == 42:
        alt = os.path.join(LRFIX_ROOT, run_name)
        if os.path.isdir(alt):
            return alt
    return os.path.join(parent, run_name)


def collect_stage1(use_lrfix=False):
    rows = []
    for name, parent, tmpl, is_pt, _ in MODELS:
        seeds = SEEDS if "{seed}" in tmpl else [0]      # calibrated models are seed-independent
        for site in SITES:
            for seed in seeds:
                run = _resolve_run(parent, tmpl.format(site=site, seed=seed), seed, use_lrfix)
                if not os.path.isdir(run):
                    continue
                rec = {"model": name, "site": site, "seed": seed}
                rec.update(read_prob(run))
                rec.update(read_proc(run))
                rec["rmse"] = read_rmse(run, is_pt)
                rows.append(rec)
    return pd.DataFrame(rows)


def collect_stage2(stage2_csv=None):
    p = stage2_csv or os.path.join(HERE, "ar_gapfill_loso.csv")
    if not os.path.isabs(p):
        p = os.path.join(HERE, p)
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    # "+sigma_struct" is the primary rolled out with the structural band: copy its rows and promote
    # every *fix_* column over its *raw_* twin, so the same metric selectors read the fixed band.
    src = df[df.model == "WN-SS (ALD)"].copy()
    if len(src):
        for c in [c for c in df.columns if "fix_" in c]:
            raw = c.replace("fix_", "raw_")
            if raw in src.columns:
                src[raw] = src[c]
        src["model"] = "WN-SS (ALD) +sigma_struct"
        df = pd.concat([df, src], ignore_index=True)
    return df


def _cells(sub, metrics):
    out = {}
    for m in metrics:
        if m not in sub.columns:
            out[LABEL[m]] = "—"; continue
        v = pd.to_numeric(sub[m], errors="coerce").dropna()
        out[LABEL[m]] = fmt(v.mean() if len(v) else np.nan,
                            v.std() if len(v) > 1 else np.nan, len(v), PREC[m])
    return out


def stage1_pooled(df1, sites, metrics):
    rows = []
    for name, _, _, _, _ in MODELS:
        a = df1[(df1.model == name) & (df1.site.isin(sites))]
        row = {"Model / ablation": name}
        row.update(_cells(a, metrics))
        row["n"] = len(a)
        rows.append(row)
    return pd.DataFrame(rows)


def stage2_pooled(df2, sites, metrics):
    rows = []
    for name, _, _, _, gname in MODELS:
        b = df2[(df2.model == gname) & (df2.site.isin(sites))] if gname and len(df2) else pd.DataFrame()
        row = {"Model / ablation": name}
        row.update(_cells(b, metrics) if len(b) else {LABEL[m]: "—" for m in metrics})
        row["n"] = len(b)
        rows.append(row)
    return pd.DataFrame(rows)


def stage1_sitewise(df1, metrics):
    rows = []
    for site in SITES:
        for name, _, _, _, _ in MODELS:
            a = df1[(df1.model == name) & (df1.site == site)]
            if not len(a):
                continue
            row = {"Site": SITE_LABEL[site], "Model / ablation": name}
            row.update(_cells(a, metrics))
            row["n"] = len(a)
            rows.append(row)
    return pd.DataFrame(rows)


def stage2_sitewise(df2, metrics):
    rows = []
    for site in SITES:
        for name, _, _, _, gname in MODELS:
            b = df2[(df2.model == gname) & (df2.site == site)] if gname and len(df2) else pd.DataFrame()
            if not len(b):
                continue
            row = {"Site": SITE_LABEL[site], "Model / ablation": name}
            row.update(_cells(b, metrics))
            row["n"] = len(b)
            rows.append(row)
    return pd.DataFrame(rows)


def to_md(df, title, note):
    hdr = list(df.columns)
    out = [f"### {title}", "", "| " + " | ".join(hdr) + " |",
           "|" + "|".join("---" for _ in hdr) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in hdr) + " |")
    out += ["", note, ""]
    return "\n".join(out)


def main(use_lrfix=False, skip_supp=False, full5=False, stage2_csv=None):
    global USE_FULL5, SEEDS
    USE_FULL5 = full5
    if full5:
        SEEDS = [0, 1, 2, 3, 4]
    os.makedirs(OUTDIR, exist_ok=True)
    df1, df2 = collect_stage1(use_lrfix), collect_stage2(stage2_csv)
    if full5:
        LRNOTE = ("  **Uniform protocol** (`outputs/ss_full5`): every gradient-trained arm was "
                  "re-trained under one identical schedule — `reduce_on_plateau` stepped on the "
                  "held-out-site loss, no validation split carved from the training sites — so the "
                  "± is a genuine seed spread and not a mixture of protocols. "
                  "**These values are scored from `best.pth`, which with `data.val_frac` unset is "
                  "the epoch selected ON THE HELD-OUT TEST SITE.** They are therefore an optimistic, "
                  "test-selected bound rather than an honest estimate of generalisation, and they are "
                  "not directly comparable to the calibrated Analytical SDE and tree arms, which have "
                  "no epoch to select and gain nothing from it. The leakage-free `last.pth` scores are "
                  "archived in `analysis/ss/v2_last_archive/`.")
        NSEED, NSEEDL = "5 seeds", "the 5 seeds (0-4)"
    else:
        LRNOTE = ("  **Seed 42 uses the LR-fixed re-run** (`reduce_on_plateau`, stepped on the TRAIN "
                  "loss) for every gradient-trained arm, after seed 42 was found to diverge under the "
                  "original constant-LR protocol; seeds 0 and 1 remain on the original protocol, so "
                  "the ± for a given row mixes two training protocols and is provisional."
                  if use_lrfix else "")
        NSEED, NSEEDL = "3 seeds", "the 3 seeds (0/1/42)"

    N1P = ("**Stage 1 — one-step predictive law.** mean ± 1 SD over all (site, seed) units "
           f"({NSEED} × sites; calibrated seed-independent models show a bare value). CRPS and RMSE "
           "lower = better; cov90 nominal = 0.90; sharp90 = mean 90% interval width (µmol m⁻² s⁻¹); "
           "PIT-KS 0 = perfectly calibrated. Trees emit no predictive distribution (RMSE only).")
    N1S = ("**Stage 1 — one-step predictive law, per held-out site.** mean ± 1 SD over "
           f"{NSEEDL}; n = 1 entries are the calibrated, seed-independent models. Metric conventions "
           "as in the pooled Stage-1 tables.")
    N2P = ("**Stage 2 — autoregressive gap-fill**, scored on the RAW predictive band (the +σ_struct "
           "band is a coverage device and is reported separately). mean ± 1 SD over all (site, seed) "
           f"units ({NSEED} × sites; the calibrated Analytical variants are seed-independent, n = 1 per "
           "site). RMSE by hours-into-gap is the stability test (flat ⇒ the physics drift tracks the "
           "nightly decline; climbing/diverging ⇒ it does not); CRPS / PIT-KS / cov90 are given at "
           "5 h+, the discriminating horizon. `—` = arm not rolled out.")
    N2S = ("**Stage 2 — autoregressive gap-fill, per held-out site.** mean ± 1 SD over "
           f"{NSEEDL}; n = 1 entries are the calibrated, seed-independent models. Raw predictive band; "
           "metric conventions as in the pooled Stage-2 tables.")

    specs = [
        ("stage1_clean4", stage1_pooled(df1, CLEAN, MAIN_S1),
         "Stage 1 (distributional) — clean sites (4 in-distribution towers; Redmere 1 excluded)", N1P),
        ("stage2_clean4", stage2_pooled(df2, CLEAN, MAIN_S2),
         "Stage 2 (gap-filling) — clean sites (4 in-distribution towers; Redmere 1 excluded)", N2P),
        ("stage1_all5", stage1_pooled(df1, SITES, MAIN_S1),
         "Stage 1 (distributional) — all sites (5 towers, including the Redmere-1 OOD stress)", N1P),
        ("stage2_all5", stage2_pooled(df2, SITES, MAIN_S2),
         "Stage 2 (gap-filling) — all sites (5 towers, including the Redmere-1 OOD stress)", N2P),
        ("stage1_sitewise", stage1_sitewise(df1, MAIN_S1),
         "Stage 1 (distributional) — site-wise", N1S),
        ("stage2_sitewise", stage2_sitewise(df2, MAIN_S2),
         "Stage 2 (gap-filling) — site-wise", N2S),
    ]
    md_all = ["# WienerNet-SS — manuscript results tables", ""]
    for stem, df, title, note in specs:
        df.to_csv(os.path.join(OUTDIR, f"{stem}.csv"), index=False)
        md = to_md(df, title, note + LRNOTE)
        open(os.path.join(OUTDIR, f"{stem}.md"), "w").write(md + "\n")
        md_all.append(md)
        print(f"wrote {stem:<18} {len(df):>3} rows x {len(df.columns)-2:>2} metrics")
    open(os.path.join(OUTDIR, "all_tables.md"), "w").write("\n".join(md_all) + "\n")

    # ---- supplementary: the FULL metric sets (nothing is lost by the main-table reduction) ----
    if skip_supp:
        print("skipped supplementary CSVs (--skip-supp): they still reflect the previous run")
        return
    for stem, df in [("supp_stage1_clean4", stage1_pooled(df1, CLEAN, FULL_S1)),
                     ("supp_stage1_all5", stage1_pooled(df1, SITES, FULL_S1)),
                     ("supp_stage1_sitewise", stage1_sitewise(df1, FULL_S1)),
                     ("supp_stage2_clean4", stage2_pooled(df2, CLEAN, FULL_S2)),
                     ("supp_stage2_all5", stage2_pooled(df2, SITES, FULL_S2)),
                     ("supp_stage2_sitewise", stage2_sitewise(df2, FULL_S2))]:
        df.to_csv(os.path.join(OUTDIR, f"{stem}.csv"), index=False)
    print("wrote all_tables.md + 6 supplementary full-metric CSVs ->", OUTDIR)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrfix", action="store_true",
                    help="use the LR-fixed seed-42 re-runs (outputs/ss_lrfix) where they exist")
    ap.add_argument("--skip-supp", action="store_true", help="do not rewrite the supp_* CSVs")
    ap.add_argument("--full5", action="store_true",
                    help="read the uniform-protocol 5-seed sweep (outputs/ss_full5, seeds 0-4)")
    ap.add_argument("--stage2-csv", default=None,
                    help="gap-fill results CSV (default analysis/ss/ar_gapfill_loso.csv)")
    a = ap.parse_args()
    main(use_lrfix=a.lrfix, skip_supp=a.skip_supp, full5=a.full5, stage2_csv=a.stage2_csv)
