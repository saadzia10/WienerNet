#!/usr/bin/env python
"""How RATIONAL is WienerNet-SS's drift / residual / noise decomposition?

The interpretability edge: WienerNet-SS splits each increment into an analytic physics drift,
an optional named residual, and an aleatoric noise, and each term should mean what it claims.
This script quantifies that on the four in-distribution held-out sites (Redmere 1, the corrupted-
Tau OOD site, is excluded), pooling the LOSO test predictions, and contrasts it with the
data-driven Neural SDE (whose 'drift' is a free function) — which has no interpretable
decomposition to offer.

Increment identity (per row):  NEE_{t+1} = NEE_t + (f_phys + r)*dt + noise ,  noise ≈ N(0, sigma).
We read the saved component arrays (pred_f = f_phys+r, pred_residual = r, pred_noise, pred_noise_stds)
and the observed residual eta = NEE - nee_mean (what the noise must explain), and score:

  1. NOISE zero-mean          — mean(eta) ≈ 0
  2. NOISE cleanliness        — |corr(eta, Ta)|, |corr(eta, Reco)| ≈ 0  (physics took the T-signal;
                                the leftover is structureless aleatoric noise, not misfit)
  3. NOISE magnitude physical — model sigma vs the model-free measurement-error floor (Hollinger-
                                Richardson): the noise should sit at the irreducible-error scale
  4. NOISE shape physical     — skew / excess-kurtosis of eta vs the model's noise sample (the ALD
                                should reproduce the right-skew + heavy tails of real flux noise)
  5. DRIFT physicality        — is the drift the respiration derivative? For Neural SDE we correlate
                                its free drift against the physics drift; low corr = not interpretable
  6. VARIANCE attribution     — fraction of the increment variance the model assigns to drift vs noise
  7. RESIDUAL isolates misfit — corr(residual, Ta) (residual variant), and whether turning the
                                residual ON reduces the temperature structure left in the noise
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOSO = os.path.join(ROOT, "outputs", "ss_loso")
OUT = os.path.dirname(os.path.abspath(__file__))
CLEAN = ["woodwalton", "rosedene", "redmere_2", "great_fen"]   # Redmere 1 excluded (corrupted Tau)
TREF, T0 = 10.0, 46.02


def reco(T, E0, rb):
    return rb * np.exp(E0 * (1.0 / (TREF + T0) - 1.0 / (T + T0)))


def pool(tmpl, seed=0):
    """Pool the per-site LOSO test predictions for a model config over the clean sites."""
    frames = []
    for site in CLEAN:
        p = os.path.join(LOSO, tmpl.format(site=site, seed=seed), "metrics", "predictions.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def meas_floor(tmpl, seed=0):
    vals = []
    for site in CLEAN:
        jp = os.path.join(LOSO, tmpl.format(site=site, seed=seed), "metrics", "probabilistic.json")
        if os.path.exists(jp):
            mf = json.load(open(jp))["global"].get("measurement_noise_floor", {})
            if mf.get("rmse_floor") is not None:
                vals.append(mf["rmse_floor"])
    return float(np.mean(vals)) if vals else np.nan


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _mad(x):
    """Robust (median-absolute-deviation) scale; ×1.4826 → Gaussian-consistent σ."""
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def rationality(df, phys_drift_ref=None):
    """Compute the decomposition-rationality metrics for one pooled model frame."""
    tgt = df["gt_nee"].to_numpy(float)          # target NEE_{t+1}
    bnee = df["gt_bnee"].to_numpy(float)        # boundary NEE_t
    Ta = df["Ta"].to_numpy(float)
    dt = df["dt"].to_numpy(float) if "dt" in df else np.full(len(df), 30.0)
    mean = df["pred_nee_mean"].to_numpy(float)
    f = df["pred_f"].to_numpy(float)            # drift rate (f_phys + r)
    resid_rate = df["pred_residual"].to_numpy(float) if "pred_residual" in df else np.zeros(len(df))
    pred_sig = df["pred_nee_std"].to_numpy(float) if "pred_nee_std" in df else np.full(len(df), np.nan)  # parametric increment std
    noise_samp = df["pred_noise"].to_numpy(float) if "pred_noise" in df else np.full(len(df), np.nan)
    Reco = reco(Ta, df["E0"].to_numpy(float), df["rb"].to_numpy(float)) if {"E0", "rb"}.issubset(df) else np.full(len(df), np.nan)

    eta = tgt - mean                      # empirical residual the noise must explain
    drift_incr = f * dt                   # physics(+residual) increment
    obs_incr = tgt - bnee                 # observed increment
    fin = np.isfinite(eta) & np.isfinite(obs_incr)
    ef = eta[np.isfinite(eta)]

    out = {
        "n": int(fin.sum()),
        "noise_mean": float(np.nanmean(eta)),
        "corr_eta_Ta": _corr(eta, Ta),
        "corr_eta_Reco": _corr(eta, Reco),
        "eta_std": float(np.nanstd(eta)),
        "eta_mad": _mad(eta),                                   # robust bulk spread
        "model_sigma_param": float(np.nanmean(pred_sig)),       # parametric predictive std
        "eta_skew": float(stats.skew(ef)),
        "eta_exkurt": float(stats.kurtosis(ef)),
        "var_frac_drift": float(np.nanvar(drift_incr[fin]) / np.nanvar(obs_incr[fin])) if np.nanvar(obs_incr[fin]) > 0 else np.nan,
        "var_frac_noise": float(np.nanvar(eta[fin]) / np.nanvar(obs_incr[fin])) if np.nanvar(obs_incr[fin]) > 0 else np.nan,
        "corr_resid_Ta": _corr(resid_rate, Ta) if np.any(resid_rate != 0) else np.nan,
        "resid_std_incr": float(np.nanstd(resid_rate * dt)) if np.any(resid_rate != 0) else np.nan,
    }
    if phys_drift_ref is not None and len(phys_drift_ref) == len(f):
        out["corr_drift_physics"] = _corr(f, phys_drift_ref)   # positional (same rows/order)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    models = {
        "WienerNet-SS (primary, resid off)": "{site}_s{seed}_ldiur_wien",
        "WienerNet-SS (+residual)":          "{site}_s{seed}_ldiur_res_wien",
        "Neural SDE (ALD, free drift)":      "{site}_s{seed}_base_nsde_ald",
    }
    # physics-drift reference = the primary WN-SS f (pure physics, resid off), pooled clean sites
    ref_df = pool(models["WienerNet-SS (primary, resid off)"])
    phys_ref = ref_df["pred_f"].to_numpy(float) if ref_df is not None else None
    floor = meas_floor(models["WienerNet-SS (primary, resid off)"])

    rows = {}
    for name, tmpl in models.items():
        df = pool(tmpl)
        if df is None:
            print(f"(missing) {name}"); continue
        rows[name] = rationality(df, phys_drift_ref=phys_ref)

    print(f"=== DECOMPOSITION RATIONALITY — pooled over 4 in-distribution sites (Redmere 1 excluded) ===")
    print(f"    model-free measurement-error floor (Hollinger-Richardson RMSE): {floor:.2f}\n")

    short = {"WienerNet-SS (primary, resid off)": "WN-SS primary",
             "WienerNet-SS (+residual)": "WN-SS +resid",
             "Neural SDE (ALD, free drift)": "Neural SDE"}

    def row(label, key, fmt="{:.3f}", scale=1.0):
        cells = "".join(f"{fmt.format(rows[m][key]*scale):>15}" if (m in rows and np.isfinite(rows[m].get(key, np.nan))) else f"{'—':>15}" for m in models)
        print(f"  {label:<34}{cells}")

    print(f"  {'metric':<34}" + "".join(f"{short[m]:>15}" for m in models))
    print("  " + "-" * (34 + 15 * len(models)))
    print("  NOISE — is it clean aleatoric?")
    row("mean (→0 ideal)", "noise_mean")
    row("corr with Ta (→0 clean)", "corr_eta_Ta")
    row("corr with Reco (→0 clean)", "corr_eta_Reco")
    print("  NOISE — magnitude & shape physical?")
    row("empirical resid std (tail-infl)", "eta_std")
    row("empirical resid MAD (bulk)", "eta_mad")
    row("model parametric σ", "model_sigma_param")
    row("resid skew (real: right-skew)", "eta_skew")
    row("resid excess-kurt (real: heavy)", "eta_exkurt")
    print("  DRIFT — physical / interpretable?")
    row("corr(drift, physics drift)", "corr_drift_physics")
    row("variance frac → drift (small=OK)", "var_frac_drift")
    row("variance frac → noise", "var_frac_noise")
    print("  RESIDUAL — isolates the misfit?")
    row("corr(residual, Ta)", "corr_resid_Ta")
    row("residual increment std", "resid_std_incr")

    json.dump(rows, open(os.path.join(OUT, "rationality.json"), "w"), indent=2, default=float)
    print("\n  wrote", os.path.join(OUT, "rationality.json"))
    return rows


if __name__ == "__main__":
    main()
