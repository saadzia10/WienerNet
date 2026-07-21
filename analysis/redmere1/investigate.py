#!/usr/bin/env python
"""Diagnostics for the learned-noise models on the Redmere 1 holdout.

Runs three measurements, writes `investigate.json` and prints tables:

  (1) DRIFT vs NOISE localisation — for a redmere_1-holdout run, the deterministic mean
      (nee_mean) vs the sampled prediction (nee), on the TRAIN pool vs the held-out
      Redmere 1 TEST set.

  (2) NOISE SCALE — the predicted sigma_eff (and state-space sigma_meas / sigma_proc)
      distribution (p50/p99/max) on TRAIN vs Redmere 1 TEST.

  (3) DRIVER out-of-distribution — per-driver standardised shift of Redmere 1 vs the training
      pool, plus a Mahalanobis distance of the test mean from the training distribution.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from omegaconf import OmegaConf
from pathlib import Path
import evaluate as ev
from wienernet.data import build_dataloaders
from wienernet.utils import load_model_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.dirname(os.path.abspath(__file__))


def load(run_dir):
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
    model = ev._rebuild_any_model(cfg, feature_dim=bundle.feature_dim, device=DEVICE)
    load_model_weights(run_dir / "checkpoints" / "last.pth", model, map_location=DEVICE)
    model.eval()
    return model, bundle, cfg


@torch.no_grad()
def forward_scales(model, loader):
    """Collect sigma_eff / sigma_meas / sigma_proc and |sampled-obs|, |mean-obs| over a loader."""
    sig, sm, sp, samp_err, mean_err = [], [], [], [], []
    for bt in loader:
        out = model(bt["X"].to(DEVICE), bt["bNEE"].to(DEVICE), bt["k"].to(DEVICE), bt["T"].to(DEVICE),
                    bt.get("dt"), bt.get("dT"), site=bt.get("site_id"), dT_diurnal=bt.get("dT_diurnal"))
        obs = bt["NEE"].numpy().ravel()
        if out.get("sigma") is not None:
            sig.append(out["sigma"].cpu().numpy().ravel())
        if out.get("sigma_meas") is not None:
            sm.append(out["sigma_meas"].cpu().numpy().ravel())
        if out.get("sigma_proc") is not None:
            sp.append(out["sigma_proc"].cpu().numpy().ravel())
        if out.get("nee_pred") is not None:
            samp_err.append(np.abs(out["nee_pred"].cpu().numpy().ravel() - obs))
        if out.get("nee_mean") is not None:
            mean_err.append(np.abs(out["nee_mean"].cpu().numpy().ravel() - obs))
    def cat(x): return np.concatenate(x) if x else np.array([])
    return dict(sigma=cat(sig), sigma_meas=cat(sm), sigma_proc=cat(sp),
                samp_err=cat(samp_err), mean_err=cat(mean_err))


def pstats(x):
    if x.size == 0: return None
    return {"p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)), "mean": float(np.mean(x))}


def driver_ood(bundle, cfg):
    """Per-driver standardised shift of the held-out site vs the training pool + Mahalanobis."""
    drivers = list(cfg.data.drivers)
    tr, te = bundle.train_df, bundle.test_df
    rows = {}
    mu = tr[drivers].mean(); sd = tr[drivers].std(ddof=0).replace(0, 1e-9)
    for d in drivers:
        z = float((te[d].mean() - mu[d]) / sd[d])          # standardised mean shift
        # fraction of test rows beyond the train 1–99 percentile range (tail OOD)
        lo, hi = np.nanpercentile(tr[d], 1), np.nanpercentile(tr[d], 99)
        frac_out = float(np.mean((te[d] < lo) | (te[d] > hi)))
        rows[d] = {"z_meanshift": z, "frac_beyond_train_p1_99": frac_out}
    # Mahalanobis of the test-mean vs train distribution (diagonal-regularised cov)
    X = tr[drivers].dropna().to_numpy(); C = np.cov(X.T) + 1e-6 * np.eye(len(drivers))
    diff = (te[drivers].mean().to_numpy() - X.mean(0))
    maha = float(np.sqrt(diff @ np.linalg.inv(C) @ diff))
    return {"per_driver": rows, "mahalanobis_testmean_vs_train": maha}


def main():
    os.makedirs(OUT, exist_ok=True)
    LOSO = os.path.join(ROOT, "outputs", "ss_loso")
    runs = {
        "ldiur_wien (PRIMARY, robust)": f"{LOSO}/redmere_1_s0_ldiur_wien",
        "diur_wien (no ldiur reg)":     f"{LOSO}/redmere_1_s0_diur_wien",
        "diur_ss (blows up)":           f"{LOSO}/redmere_1_s0_diur_ss",
        "ldiur_ss (worst)":             f"{LOSO}/redmere_1_s0_ldiur_ss",
    }
    report = {}
    print("=== (1)+(2) NOISE-SCALE explosion: in-distribution TRAIN vs held-out Redmere 1 TEST ===")
    hdr = f"{'model':<30}{'split':<7}{'sigma p50':>10}{'sigma p99':>10}{'sigma max':>11}{'mean|err|':>10}{'samp|err|p99':>13}"
    print(hdr); print("-" * len(hdr))
    driver_done = False
    for name, run in runs.items():
        if not os.path.isdir(run):
            print(f"{name}: missing {run}"); continue
        model, bundle, cfg = load(run)
        tr = forward_scales(model, bundle.train_loader)
        te = forward_scales(model, bundle.test_loader)
        report[name] = {"train": {k: pstats(v) for k, v in tr.items()},
                        "redmere1_test": {k: pstats(v) for k, v in te.items()}}
        for split, d in [("train", tr), ("r1 TEST", te)]:
            s = pstats(d["sigma"]) or {"p50": np.nan, "p99": np.nan, "max": np.nan}
            me = float(np.mean(d["mean_err"])) if d["mean_err"].size else np.nan
            se99 = float(np.percentile(d["samp_err"], 99)) if d["samp_err"].size else np.nan
            print(f"{name:<30}{split:<7}{s['p50']:>10.3f}{s['p99']:>10.2f}{s['max']:>11.1f}{me:>10.2f}{se99:>13.2f}")
        # state-space process/measurement split for the ss models
        if te["sigma_proc"].size:
            tp = pstats(tr["sigma_proc"]); ep = pstats(te["sigma_proc"])
            tm = pstats(tr["sigma_meas"]); em = pstats(te["sigma_meas"])
            print(f"{'  -> sigma_meas p99 (tr/te)':<30}{'':7}{tm['p99']:>10.2f}{em['p99']:>10.2f}"
                  f"{'  sigma_proc p99 (tr/te):':>21}{tp['p99']:>8.2f}{ep['p99']:>8.2f}")
        if not driver_done:
            report["driver_ood"] = driver_ood(bundle, cfg)
            driver_done = True
        print()
    # driver OOD table
    print("=== (3) DRIVER out-of-distribution (Redmere 1 vs training pool) ===")
    od = report["driver_ood"]
    print(f"Mahalanobis(test-mean vs train) = {od['mahalanobis_testmean_vs_train']:.2f}")
    print(f"{'driver':<10}{'z mean-shift':>13}{'frac beyond train p1-99':>26}")
    for d, v in sorted(od["per_driver"].items(), key=lambda kv: -abs(kv[1]["z_meanshift"])):
        print(f"{d:<10}{v['z_meanshift']:>13.2f}{v['frac_beyond_train_p1_99']:>26.3f}")
    json.dump(report, open(os.path.join(OUT, "investigate.json"), "w"), indent=2)
    print("\nwrote", os.path.join(OUT, "investigate.json"))


if __name__ == "__main__":
    main()
