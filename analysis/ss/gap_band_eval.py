#!/usr/bin/env python
"""Gap-fill BAND evaluation + the NEXT-2 structural-error fix.

For a trained run we reconstruct the flux across synthetic multi-day GAPS: the mean is the
physics level (temperature is measured through the gap); the question is the UNCERTAINTY
BAND. The one-step CRPS sweep cannot see this — it only scores single increments.

The band must carry EVERY source of gap uncertainty:
  * measurement noise  sigma_meas         — flat        (aleatoric head)
  * process/accumulation sigma_proc*sqrt(t)— grows sqrt(t) (state-space head)
  * STRUCTURAL error   sigma_struct        — flat        (NEW: missing soil drivers)

The state-space band used only the first two and UNDER-COVERED, because the dominant gap
uncertainty is structural (Reco(T_air) misses the soil signal). This script now reports,
per site, the fix from analysis/ss/structural.py:
  (A) +soil MEAN  : reconstruct with a + b*Reco(Ta) + c*Tsoil1  -> smaller residual
  (B) +struct BAND: add sigma_struct (out-of-sample, from TRAIN residuals) in quadrature

Bands compared: old state-space (meas+proc) vs fixed (meas+struct+proc), each with the
Reco-only mean and the +soil mean. sigma_struct is estimated on the TRAIN sites and applied
to the held-out TEST site, so the band is genuinely out-of-sample. Nominal coverage = 90%.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
from pathlib import Path
import evaluate as ev
from wienernet.data import build_dataloaders
from wienernet.utils import load_model_weights
from structural import reco, structural_sigma

Z = 1.6449   # 90% two-sided


def run_forward(run_dir, device="cpu"):
    """Load a run; return (train_df, test_df) with the model's per-test-row noise scales."""
    run_dir = Path(run_dir)
    cfg = ev._load_resolved_config(run_dir)
    bundle = build_dataloaders(
        site_paths={k: v for k, v in OmegaConf.to_container(cfg.data.site_paths, resolve=True).items()},
        drivers=tuple(cfg.data.drivers), include_sites=tuple(cfg.data.include_sites) or None,
        exclude_sites=tuple(cfg.data.exclude_sites), nighttime_only=bool(cfg.data.nighttime_only),
        night_radiation_threshold=float(cfg.data.night_radiation_threshold),
        nee_target_column=str(cfg.data.nee_target_column), boundary_nee_column=str(cfg.data.boundary_nee_column),
        e0_column=str(cfg.data.e0_column), rb_column=str(cfg.data.rb_column),
        temperature_column=str(cfg.data.temperature_column), dtemp_column=str(cfg.data.dtemp_column),
        dnee_column=str(cfg.data.dnee_column), split_strategy=str(cfg.data.split_strategy),
        test_frac=float(cfg.data.test_frac), test_years=tuple(cfg.data.test_years),
        holdout_site=cfg.data.get("holdout_site"), shuffle_split=bool(cfg.data.shuffle_split),
        split_random_state=int(cfg.data.split_random_state), time_step_k=cfg.data.get("time_step_k"),
        batch_size=4096, save_scaler_path=None, save_dataframes=True,
    )
    model = ev._rebuild_any_model(cfg, feature_dim=bundle.feature_dim, device=device)
    load_model_weights(run_dir / "checkpoints" / "last.pth", model, map_location=device)
    model.eval()
    sm, sp, sig = [], [], []
    with torch.no_grad():
        for bt in bundle.test_loader:
            out = model(bt["X"].to(device), bt["bNEE"].to(device), bt["k"].to(device), bt["T"].to(device),
                        bt.get("dt"), bt.get("dT"), site=bt.get("site_id"), dT_diurnal=bt.get("dT_diurnal"))
            n = bt["X"].shape[0]
            sig.append(torch.exp(out["nee_log_std"]).cpu().numpy().ravel() if out["nee_log_std"] is not None else np.full(n, np.nan))
            sm.append(out["sigma_meas"].cpu().numpy().ravel() if out.get("sigma_meas") is not None else np.full(n, np.nan))
            sp.append(out["sigma_proc"].cpu().numpy().ravel() if out.get("sigma_proc") is not None else np.full(n, np.nan))
    df = bundle.test_df.reset_index(drop=True).copy()
    df["sigma_incr"] = np.concatenate(sig); df["sigma_meas"] = np.concatenate(sm); df["sigma_proc"] = np.concatenate(sp)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return bundle.train_df.reset_index(drop=True).copy(), df.sort_values("DateTime").reset_index(drop=True)


def _coverage(test_df, predict_mean, sigma_struct, gap_days=(3, 7, 14)):
    """Walk synthetic gaps; return per-gaplen coverage for old (meas+proc) and fixed
    (meas+struct+proc) bands, plus the mean-reconstruction RMSE."""
    df = test_df.copy()
    df["recon"] = predict_mean(df)
    t0all = df["DateTime"].values
    res = {}
    for D in gap_days:
        span = pd.Timedelta(minutes=D * 24 * 60)
        rmse, cov_old_n, cov_old_f, cov_fix_n, cov_fix_f = [], [], [], [], []
        i, n = 0, len(df)
        while i < n:
            t0 = df["DateTime"].iloc[i]
            win = df[(df["DateTime"] >= t0) & (df["DateTime"] < t0 + span)]
            if len(win) >= 6:
                elapsed = (win["DateTime"] - t0).dt.total_seconds().values / 60.0     # minutes
                obs, rec = win["NEE"].values, win["recon"].values
                sm, sp = win["sigma_meas"].values, win["sigma_proc"].values
                rmse.append(np.sqrt(np.nanmean((obs - rec) ** 2)))
                band_old = np.sqrt(np.nan_to_num(sm) ** 2 + np.nan_to_num(sp) ** 2 * elapsed)
                band_fix = np.sqrt(np.nan_to_num(sm) ** 2 + sigma_struct ** 2 + np.nan_to_num(sp) ** 2 * elapsed)
                in_old = np.abs(obs - rec) <= Z * band_old
                in_fix = np.abs(obs - rec) <= Z * band_fix
                near, far = elapsed <= 12 * 60, elapsed > (D - 1) * 24 * 60
                if near.any(): cov_old_n.append(in_old[near].mean()); cov_fix_n.append(in_fix[near].mean())
                if far.any():  cov_old_f.append(in_old[far].mean());  cov_fix_f.append(in_fix[far].mean())
            nxt = df[df["DateTime"] >= t0 + span]
            if len(nxt) == 0: break
            i = nxt.index[0]
        mean = lambda v: float(np.nanmean(v)) if v else np.nan
        res[D] = dict(rmse=mean(rmse), old_near=mean(cov_old_n), old_far=mean(cov_old_f),
                      fix_near=mean(cov_fix_n), fix_far=mean(cov_fix_f))
    return res


def evaluate_run(run_dir, device="cpu"):
    """Full NEXT-2 comparison for one run. Returns a dict of tables."""
    train_df, test_df = run_forward(run_dir, device=device)
    sm_bar = float(np.nanmean(test_df["sigma_meas"].values))
    # sigma_struct estimated OUT-OF-SAMPLE on train sites, for Reco-only vs +soil mean
    ss_reco, lm_reco = structural_sigma(train_df, sm_bar, use_soil=False)
    ss_soil, lm_soil = structural_sigma(train_df, sm_bar, use_soil=True)
    out = dict(sigma_meas=sm_bar, sigma_struct_reco=ss_reco, sigma_struct_soil=ss_soil,
               soil_used=lm_soil["soil_used"],
               resid_std_reco=lm_reco["resid_std"], resid_std_soil=lm_soil["resid_std"])
    # Reco-only mean with old vs fixed band; +soil mean with fixed band
    out["reco_mean"] = _coverage(test_df, lm_reco["predict"], ss_reco)
    out["soil_mean"] = _coverage(test_df, lm_soil["predict"], ss_soil)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run dirs; default = the ss_loso state-space runs, one per site")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.runs:
        runs = args.runs
    else:
        LOSO = os.path.join(ROOT, "outputs", "ss_loso")
        sites = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
        runs = [os.path.join(LOSO, f"{s}_s0_diur_ss") for s in sites]

    print("=== GAP-FILL BAND — NEXT-2 structural-error fix (state-space runs, seed 0) ===")
    print("  level resid std: how far the flux scatters around the physics mean (structural error)")
    print("  cov: fraction of observed flux inside the 90% band;  near=<12h into gap, end=last day\n")
    hdr = (f"{'site':<12}{'residσ Reco':>12}{'+soil':>8}{'σ_struct':>9}"
           f"{'gap':>5}{'RMSE':>7}{'old near':>9}{'old end':>8}{'fix near':>9}{'fix end':>8}")
    print(hdr); print("-" * len(hdr))
    import csv
    csv_rows = []
    for run in runs:
        site = os.path.basename(run).split("_s")[0]
        if not os.path.isdir(run):
            print(f"{site:<12}  (missing: {run})"); continue
        try:
            r = evaluate_run(run, device=args.device)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"{site}: ERROR {e}"); continue
        for j, D in enumerate((3, 7, 14)):
            g = r["soil_mean"][D]       # headline: +soil mean + fixed band
            lead = (f"{site:<12}{r['resid_std_reco']:>12.2f}{r['resid_std_soil']:>8.2f}"
                    f"{r['sigma_struct_soil']:>9.2f}") if j == 0 else " " * 41
            print(f"{lead}{str(D)+'d':>5}{g['rmse']:>7.2f}{g['old_near']:>9.2f}{g['old_far']:>8.2f}"
                  f"{g['fix_near']:>9.2f}{g['fix_far']:>8.2f}")
            csv_rows.append(dict(site=site, gap_days=D, sigma_meas=r["sigma_meas"],
                                 sigma_struct=r["sigma_struct_soil"], resid_std_reco=r["resid_std_reco"],
                                 resid_std_soil=r["resid_std_soil"], soil_used=r["soil_used"],
                                 rmse=g["rmse"], old_near=g["old_near"], old_far=g["old_far"],
                                 fix_near=g["fix_near"], fix_far=g["fix_far"]))
    if csv_rows:
        out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap_band_fix.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
        print("\nwrote", out_csv)
    print("\n  'old' band = measurement+process only (under-covers).  'fix' band = +sigma_struct.")
    print("  +soil mean shrinks the structural residual; +struct band lifts coverage toward 0.90.")


if __name__ == "__main__":
    main()
