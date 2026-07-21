#!/usr/bin/env python
"""How well is the *predicted diurnal* captured? — GT-k vs predicted-k.

Read-only post-hoc analysis. Touches no training code: it only reads the
`metrics/predictions.parquet` written by `scripts/evaluate.py` for two runs of
the SAME architecture (WienerNet, ALD likelihood, learned-diurnal drift
tendency, single-Wiener noise — i.e. *not* the state-space noise law):

    outputs/ss_loso/{site}_s{seed}_ldiur_wien         physics_k_source=ground_truth
    outputs/ss_loso/{site}_s{seed}_ldiur_wien_predk   physics_k_source=predicted

5 LOSO held-out sites x 3 seeds (0/1/42). Values are mean +/- SD; the SD is over
the 3 seeds for the site-wise view and over all (site, seed) units for pooled.

Four tables are written (CSV + Markdown) to analysis/diurnal/tables/:

  T1_tendency   the diurnal tendency head itself: pred_dtemp vs gt_dtemp
                (dTa/dt, K per 30-min step). RMSE / MAE / bias / R2 / corr.
  T2_composite  the mean diurnal *cycle* of NEE: observations and predictions
                composited by hour of night, then compared bin-to-bin. RMSE of
                the composite, bias, and the predicted/observed amplitude ratio.
  T3_by_hour    point + distributional NEE skill resolved by hour-of-night bin:
                RMSE, CRPS, 90% coverage, 90% sharpness, PIT-KS.
  T4_overall    the same metrics over all half-hourly points.

Scoring convention follows scripts/evaluate.py: the one-step predictive law is
scored as Gaussian(pred_nee_mean, pred_nee_std) — `_predictive_family()` returns
Gaussian for every non-Student-t head, ALD included. Per-sample CRPS and PIT are
read straight from the parquet; coverage/sharpness are recomputed from
(mean, std) with the repo's `predictive_interval`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from wienernet.evaluation.probabilistic import predictive_interval  # noqa: E402

LOSO = os.path.join(ROOT, "outputs", "ss_loso")
OUTDIR = os.path.join(HERE, "tables")

SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
SITE_LABEL = {"woodwalton": "Woodwalton", "rosedene": "Rosedene", "redmere_1": "Redmere 1",
              "redmere_2": "Redmere 2", "great_fen": "Great Fen"}
SEEDS = [0, 1, 42]
MODELS = [("GT-k (primary)", "{site}_s{seed}_ldiur_wien"),
          ("predicted-k", "{site}_s{seed}_ldiur_wien_predk")]

# Hour-of-night bins. UK sites, nighttime-only records; DateTime is UTC and the
# local offset is at most 1 h, so UTC hour is used directly (DateTimeLocal is
# NaT for some sites). Ordered evening -> dawn across the midnight wrap.
HOUR_BINS = [("16-18", (16, 17)), ("18-21", (18, 19, 20)), ("21-24", (21, 22, 23)),
             ("00-03", (0, 1, 2)), ("03-06", (3, 4, 5)), ("06-09", (6, 7, 8))]

COLS = ["DateTime", "site", "gt_nee", "gt_dtemp", "pred_dtemp", "dTa_diurnal",
        "pred_nee_mean", "pred_nee_std", "pred_crps", "pred_pit"]


# --------------------------------------------------------------------------- #
# primitive metrics
# --------------------------------------------------------------------------- #
def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _r2(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _ks_uniform(pit):
    """One-sample KS distance of the PIT values against U(0,1)."""
    p = np.sort(np.asarray(pit, dtype=float))
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    n = p.size
    cdf = np.arange(1, n + 1) / n
    return float(np.max(np.maximum(np.abs(cdf - p), np.abs(p - (np.arange(n) / n)))))


def _dist_metrics(df):
    """Point + distributional NEE metrics for one slice of one run."""
    obs = df["gt_nee"].to_numpy(dtype=float)
    mu = df["pred_nee_mean"].to_numpy(dtype=float)
    sd = df["pred_nee_std"].to_numpy(dtype=float)
    lo, hi = predictive_interval(mu, sd, 0.90, family="gaussian")
    return dict(
        n=len(df),
        rmse=_rmse(obs, mu),
        mae=float(np.mean(np.abs(obs - mu))),
        bias=float(np.mean(mu - obs)),
        crps=float(np.mean(df["pred_crps"].to_numpy(dtype=float))),
        cov90=float(np.mean((obs >= lo) & (obs <= hi))),
        sharp90=float(np.mean(hi - lo)),
        pit_ks=_ks_uniform(df["pred_pit"].to_numpy(dtype=float)),
    )


def _tendency_metrics(df):
    """Skill of the learned diurnal tendency head, in K per 30-min step.

    With `drift_tendency="learned_diurnal"` the head's training anchor is
    `dTa_diurnal` — the per-(site, month, hour) climatology of the observed
    tendency, i.e. the *predictable* diurnal part (increment_sde.py:550). The
    observed dTa is what the drift ultimately stands in for, so both targets are
    scored: `_dn` = vs the diurnal anchor, `_ob` = vs the observed tendency.
    """
    pred = df["pred_dtemp"].to_numpy(dtype=float)
    out = dict(sd_pred=float(np.nanstd(pred)))
    for tag, col in (("dn", "dTa_diurnal"), ("ob", "gt_dtemp")):
        obs = df[col].to_numpy(dtype=float)
        ok = np.isfinite(obs) & np.isfinite(pred)
        o, p = obs[ok], pred[ok]
        out[f"n_{tag}"] = int(ok.sum())
        out[f"rmse_{tag}"] = _rmse(o, p)
        out[f"mae_{tag}"] = float(np.mean(np.abs(o - p)))
        out[f"bias_{tag}"] = float(np.mean(p - o))
        out[f"r2_{tag}"] = _r2(o, p)
        out[f"corr_{tag}"] = float(np.corrcoef(o, p)[0, 1])
        out[f"sd_{tag}"] = float(o.std())
        # skill of the head against the trivial "predict the target mean" baseline
        out[f"rmse_ref_{tag}"] = _rmse(o, np.full_like(o, o.mean()))
    return out


def _composite_metrics(df):
    """Mean diurnal cycle of NEE: composite obs and pred by hour of night, then
    compare the two composites bin-to-bin. Amplitude = peak-to-trough range."""
    hb = df["hour_bin"]
    rows = []
    for lab, _ in HOUR_BINS:
        m = hb == lab
        if m.sum() == 0:
            continue
        rows.append((float(df.loc[m, "gt_nee"].mean()),
                     float(df.loc[m, "pred_nee_mean"].mean())))
    o = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    return dict(
        n_bins=len(o),
        comp_rmse=_rmse(o, p),
        comp_bias=float(np.mean(p - o)),
        amp_obs=float(o.max() - o.min()),
        amp_pred=float(p.max() - p.min()),
        amp_ratio=float((p.max() - p.min()) / (o.max() - o.min())),
        comp_r2=_r2(o, p),
    )


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def load_run(site, seed, template):
    run = os.path.join(LOSO, template.format(site=site, seed=seed))
    path = os.path.join(run, "metrics", "predictions.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=COLS)
    hr = pd.to_datetime(df["DateTime"]).dt.hour.to_numpy()
    lab = np.full(len(df), "other", dtype=object)
    for name, hours in HOUR_BINS:
        lab[np.isin(hr, hours)] = name
    df["hour_bin"] = lab
    return df[df["hour_bin"] != "other"].reset_index(drop=True)


def collect():
    """One row per (model, site, seed) x table."""
    tend, comp, overall, byhour = [], [], [], []
    for model, template in MODELS:
        for site in SITES:
            for seed in SEEDS:
                df = load_run(site, seed, template)
                if df is None:
                    print(f"  MISSING {template.format(site=site, seed=seed)}")
                    continue
                key = dict(model=model, site=site, seed=seed)
                tend.append({**key, **_tendency_metrics(df)})
                comp.append({**key, **_composite_metrics(df)})
                overall.append({**key, **_dist_metrics(df)})
                for lab, _ in HOUR_BINS:
                    sub = df[df["hour_bin"] == lab]
                    if len(sub) == 0:
                        continue
                    byhour.append({**key, "hour_bin": lab, **_dist_metrics(sub)})
    return (pd.DataFrame(tend), pd.DataFrame(comp),
            pd.DataFrame(overall), pd.DataFrame(byhour))


# --------------------------------------------------------------------------- #
# aggregation / formatting
# --------------------------------------------------------------------------- #
def _fmt(mean, sd, n, prec):
    if mean is None or not np.isfinite(mean):
        return "—"
    if n is None or n < 2 or sd is None or not np.isfinite(sd):
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} ± {sd:.{prec}f}"


def summarise(df, group, metrics, prec):
    """mean ± SD of each metric over the units inside each group."""
    out = []
    for keys, g in df.groupby(group, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group, keys))
        row["n_runs"] = len(g)
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            row[m] = _fmt(v.mean() if v.size else np.nan,
                          v.std(ddof=1) if v.size > 1 else np.nan, v.size, prec[m])
        out.append(row)
    return pd.DataFrame(out)


def write(name, table, note):
    os.makedirs(OUTDIR, exist_ok=True)
    table.to_csv(os.path.join(OUTDIR, f"{name}.csv"), index=False)
    with open(os.path.join(OUTDIR, f"{name}.md"), "w") as fh:
        fh.write(table.to_markdown(index=False) + "\n\n" + note + "\n")
    print(f"\n### {name}\n")
    print(table.to_markdown(index=False))
    print(f"\n{note}")


def main():
    print("reading runs from", LOSO)
    tend, comp, overall, byhour = collect()

    tend_m = dict(rmse_dn=4, mae_dn=4, bias_dn=4, corr_dn=3, rmse_ref_dn=4, sd_dn=4,
                  rmse_ob=4, mae_ob=4, corr_ob=3, rmse_ref_ob=4, sd_ob=4, sd_pred=4)
    comp_m = dict(comp_rmse=3, comp_bias=3, amp_obs=3, amp_pred=3, amp_ratio=3, comp_r2=3)
    dist_m = dict(rmse=3, mae=3, bias=3, crps=3, cov90=3, sharp90=2, pit_ks=3)

    clean = lambda d: d[d["site"] != "redmere_1"]  # noqa: E731
    tend_c, comp_c, overall_c, byhour_c = map(clean, (tend, comp, overall, byhour))

    for scope, group, tag in [("pooled", ["model"], "pooled"),
                              ("site-wise", ["model", "site"], "sitewise")]:
        write(f"T1_tendency_{tag}",
              summarise(tend, group, list(tend_m), tend_m),
              "Learned diurnal tendency head (K per 30-min step). _dn = vs the "
              "training anchor dTa_diurnal (the (site, month, hour) climatology "
              "of dTa); _ob = vs the observed dTa. rmse_ref_* is the trivial "
              "constant-mean predictor on the same target — the head only adds "
              "skill where rmse_* < rmse_ref_*. ± over "
              f"{'all (site, seed) units' if scope == 'pooled' else 'the 3 seeds'}.")
        write(f"T2_composite_{tag}",
              summarise(comp, group, list(comp_m), comp_m),
              "Mean diurnal cycle of NEE (µmol m⁻² s⁻¹): observed and predicted "
              "composites over 4 hour-of-night bins, compared bin-to-bin. "
              "amp_ratio = predicted / observed peak-to-trough range.")
        write(f"T4_overall_{tag}",
              summarise(overall, group, list(dist_m), dist_m),
              "All half-hourly nighttime points. RMSE/MAE/bias/CRPS/sharp90 in "
              "µmol m⁻² s⁻¹; cov90 nominal 0.90; PIT-KS 0 = perfectly calibrated.")

    for name, df_c, mset in [("T1_tendency_clean4", tend_c, tend_m),
                             ("T2_composite_clean4", comp_c, comp_m),
                             ("T4_overall_clean4", overall_c, dist_m)]:
        write(name, summarise(df_c, ["model"], list(mset), mset),
              "Pooled over the 4 in-distribution held-out sites (Redmere 1, the "
              "OOD stress site, excluded). ± over all (site, seed) units.")

    byhour["hour_bin"] = pd.Categorical(byhour["hour_bin"],
                                        [b[0] for b in HOUR_BINS], ordered=True)
    byhour = byhour.sort_values(["model", "hour_bin"])
    write("T3_by_hour_pooled",
          summarise(byhour, ["model", "hour_bin"], list(dist_m), dist_m),
          "NEE skill resolved by hour-of-night (UTC). ± over all (site, seed) "
          "units. Units as in T4.")

    # raw per-run values, for anything downstream
    for name, df in [("raw_tendency", tend), ("raw_composite", comp),
                     ("raw_overall", overall), ("raw_by_hour", byhour)]:
        df.to_csv(os.path.join(OUTDIR, f"{name}.csv"), index=False)
    print(f"\nwrote tables + raw per-run CSVs to {OUTDIR}")


if __name__ == "__main__":
    main()
