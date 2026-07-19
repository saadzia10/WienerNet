#!/usr/bin/env python
"""Autoregressive within-night gap-filling for ALL methods, across ALL five held-out sites.

Protocol (user-specified): for each contiguous night, anchor at the last observed NEE before
the night, forecast NEE_{t+1}, feed that forecast back as NEE_t for the next step, using the
real drivers at every step (only NEE is synthetic). Every increment-form model uses NEE_t
(nee_pred = bNEE + drift + noise, incl. MDN comp_means = b + decoder), so this is fair.

Two outputs per (site, model):
  * deterministic rollout (feed the mean forward)   -> point RMSE by hours-into-gap
  * ensemble rollout (M noisy members)              -> 90% band coverage by hours-into-gap
    For WienerNet-SS we ALSO report the NEXT-2 structural-fix band: each member carries a
    persistent structural offset ~ N(0, sigma_struct) (sigma_struct estimated out-of-sample
    on the training sites), added on top of the aleatoric spread. This is the rollout analogue
    of the gap_band_eval fix — the band must know about the flux variation temperature can't
    explain, or it under-covers no matter how the aleatoric heads are calibrated.

Trees (RF/XGB) are driver-only regressors: they don't consume NEE_t, so their gap forecast is
their stored per-row prediction (pred_nee), aligned by DateTime. Their RMSE is therefore FLAT
with gap length by construction (a useful reference: no accumulation, but no physics drift
either) and they carry no uncertainty band (point-only).

A model whose drift tracks the within-night respiration decline keeps RMSE flat as the gap
grows; a persistence-like or noise-chasing model's RMSE climbs / blows up.
"""
from __future__ import annotations
import os, sys, csv
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
from structural import structural_sigma

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
BINS = [("h0-2", lambda s: s < 4), ("h2-5", lambda s: (s >= 4) & (s < 10)), ("h5+", lambda s: s >= 10)]


def load_torch(run_dir, want_struct=False):
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
    ds = bundle.test_df.reset_index(drop=True)
    night = (pd.to_datetime(ds["DateTime"]).diff().dt.total_seconds() / 60 != 30).cumsum().values
    sigma_struct = 0.0
    if want_struct:
        sm_bar = 0.0
        with torch.no_grad():
            for bt in bundle.test_loader:
                o = model(bt["X"].to(DEVICE), bt["bNEE"].to(DEVICE), bt["k"].to(DEVICE), bt["T"].to(DEVICE),
                          bt.get("dt"), bt.get("dT"), site=bt.get("site_id"), dT_diurnal=bt.get("dT_diurnal"))
                if o.get("sigma_meas") is not None:
                    sm_bar = float(np.nanmean(o["sigma_meas"].cpu().numpy())); break
        sigma_struct, _ = structural_sigma(bundle.train_df, sm_bar, use_soil=False)
    return model, bundle.test_dataset, night, sigma_struct


def _row(ds, i, M):
    rep = lambda t: t[i:i+1].repeat(M, 1).to(DEVICE)
    dt = ds.dt[i:i+1].repeat(M).to(DEVICE) if ds.dt is not None else None
    dtd = ds.dT_diurnal[i:i+1].repeat(M, 1).to(DEVICE) if ds.dT_diurnal is not None else None
    return rep(ds.X), rep(ds.k), rep(ds.T), dt, rep(ds.dT), dtd


@torch.no_grad()
def rollout_torch(model, ds, night, sigma_struct=0.0, M=100, min_len=6, max_nights=200, seed=0):
    torch.manual_seed(seed)
    groups = [np.where(night == nid)[0] for nid in np.unique(night)]
    groups = [g for g in groups if len(g) >= min_len]
    rng = np.random.default_rng(seed)
    if len(groups) > max_nights:
        groups = [groups[i] for i in rng.choice(len(groups), max_nights, replace=False)]
    det_err, cov_raw, cov_fix, steps = [], [], [], []
    for g in groups:
        det = ds.bNEE[g[0]].item()
        ens = torch.full((M,), ds.bNEE[g[0]].item(), device=DEVICE)
        # one persistent structural offset per member, drawn at gap start (flat in gap-time)
        offset = torch.randn(M, device=DEVICE) * sigma_struct if sigma_struct > 0 else None
        for j in range(len(g)):
            i = g[j]
            X, k, T, dt, dT, dtd = _row(ds, i, M)
            out = model(X, ens.view(-1, 1), k, T, dt, dT, dT_diurnal=dtd)
            pred = out["nee_pred"].squeeze(1)
            mean_out = model(ds.X[i:i+1].to(DEVICE), torch.tensor([[det]], device=DEVICE),
                             ds.k[i:i+1].to(DEVICE), ds.T[i:i+1].to(DEVICE),
                             ds.dt[i:i+1].to(DEVICE) if ds.dt is not None else None, ds.dT[i:i+1].to(DEVICE),
                             dT_diurnal=ds.dT_diurnal[i:i+1].to(DEVICE) if ds.dT_diurnal is not None else None)
            det_next = mean_out["nee_mean"].item()
            obs = ds.NEE[i].item()
            det_err.append((det_next - obs) ** 2); steps.append(j)
            lo, hi = torch.quantile(pred, 0.05).item(), torch.quantile(pred, 0.95).item()
            cov_raw.append(1.0 if lo <= obs <= hi else 0.0)
            if offset is not None:
                pf = pred + offset
                lo2, hi2 = torch.quantile(pf, 0.05).item(), torch.quantile(pf, 0.95).item()
                cov_fix.append(1.0 if lo2 <= obs <= hi2 else 0.0)
            ens = pred; det = det_next
    det_err, cov_raw, steps = map(np.array, (det_err, cov_raw, steps))
    cov_fix = np.array(cov_fix) if cov_fix else None
    binned = lambda vals, fn: {lab: fn(vals[m(steps)]) for lab, m in BINS if m(steps).any()}
    r = dict(rmse=binned(det_err, lambda v: float(np.sqrt(v.mean()))),
             cov_raw=binned(cov_raw, lambda v: float(v.mean())), n_nights=len(groups))
    if cov_fix is not None:
        r["cov_fix"] = binned(cov_fix, lambda v: float(v.mean()))
    return r


def rollout_tree(run_dir, night_ref_df):
    """Tree = driver-only: forecast is the stored per-row pred_nee, aligned by DateTime to the
    same test rows. RMSE binned by hours-into-gap (flat by construction; no band)."""
    p = os.path.join(run_dir, "predictions.parquet")
    if not os.path.exists(p):
        return None
    w = pd.read_parquet(p)
    if not {"pred_nee", "gt_nee", "DateTime"}.issubset(w.columns):
        return None
    w = w[["DateTime", "pred_nee", "gt_nee"]].copy()
    w["DateTime"] = pd.to_datetime(w["DateTime"])
    ref = night_ref_df.copy(); ref["DateTime"] = pd.to_datetime(ref["DateTime"])
    m = ref.merge(w, on="DateTime", how="inner")
    if len(m) < 50:
        return None
    steps = m["step"].values
    err = (m["pred_nee"].values - m["gt_nee"].values) ** 2
    binned = {lab: float(np.sqrt(err[msk(steps)].mean())) for lab, msk in BINS if msk(steps).any()}
    return dict(rmse=binned, cov_raw={}, n_nights=int(ref["night"].nunique()))


def night_ref(test_df):
    """Rows with a within-night step index, for aligning tree predictions."""
    d = test_df[["DateTime"]].copy()
    d["DateTime"] = pd.to_datetime(d["DateTime"])
    d["night"] = (d["DateTime"].diff().dt.total_seconds() / 60 != 30).cumsum()
    d["step"] = d.groupby("night").cumcount()
    return d


def main():
    LOSO = os.path.join(ROOT, "outputs", "ss_loso")
    FL = os.path.join(ROOT, "outputs", "final_loso")

    def paths(site):
        return [
            ("WN-SS (learned-diurnal, Wiener)", os.path.join(LOSO, f"{site}_s0_ldiur_wien"), "torch_struct"),
            ("WienerNet-SS (state-space)", os.path.join(LOSO, f"{site}_s0_diur_ss"), "torch_struct"),
            ("WienerNet-SS (Wiener)",      os.path.join(LOSO, f"{site}_s0_diur_wien"), "torch"),
            ("MDN (no physics)",           os.path.join(FL, f"comp_mdn_{site}_s0"), "torch"),
            ("Neural SDE",                 os.path.join(FL, f"comp_neuralsde_{site}_s0"), "torch"),
            ("Analytical SDE",             os.path.join(FL, f"prior_analytical_{site}_s0"), "torch"),
            ("Random Forest",              os.path.join(FL, f"comp_rf_{site}_s0"), "tree"),
            ("XGBoost",                    os.path.join(FL, f"comp_xgb_{site}_s0"), "tree"),
        ]

    rows = []
    for site in SITES:
        print(f"\n===== {site} =====")
        print(f"{'model':<28}{'RMSE h0-2':>10}{'h2-5':>7}{'h5+':>7}{'covRAW h0-2':>12}{'h2-5':>7}{'h5+':>7}{'covFIX h0-2':>12}{'h2-5':>7}{'h5+':>7}")
        # a reference night frame for tree alignment (from any torch run of this site)
        ref = None
        for name, run, kind in paths(site):
            if not os.path.isdir(run):
                print(f"{name:<28}  (missing)"); continue
            try:
                if kind.startswith("torch"):
                    model, ds, night, ss = load_torch(run, want_struct=(kind == "torch_struct"))
                    r = rollout_torch(model, ds, night, sigma_struct=ss)
                    if ref is None:
                        # build tree-alignment ref from this run's test predictions timeline
                        ref = _ref_from_run(run)
                else:
                    if ref is None:
                        ref = _ref_from_run(run)  # fallback: from tree's own parquet timeline
                    r = rollout_tree(run, ref)
                    if r is None:
                        print(f"{name:<28}  (no aligned rows)"); continue
                rm, cr, cf = r["rmse"], r.get("cov_raw", {}), r.get("cov_fix", {})
                g = lambda d, kx: d.get(kx, float('nan'))
                print(f"{name:<28}{g(rm,'h0-2'):>10.2f}{g(rm,'h2-5'):>7.2f}{g(rm,'h5+'):>7.2f}"
                      f"{g(cr,'h0-2'):>12.2f}{g(cr,'h2-5'):>7.2f}{g(cr,'h5+'):>7.2f}"
                      f"{g(cf,'h0-2'):>12.2f}{g(cf,'h2-5'):>7.2f}{g(cf,'h5+'):>7.2f}")
                rows.append(dict(site=site, model=name, **{f"rmse_{k}": g(rm, k) for k in ("h0-2", "h2-5", "h5+")},
                                 **{f"covraw_{k}": g(cr, k) for k in ("h0-2", "h2-5", "h5+")},
                                 **{f"covfix_{k}": g(cf, k) for k in ("h0-2", "h2-5", "h5+")}))
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"{name}: ERROR {e}")

    # cross-site aggregate + CSV
    if rows:
        out = os.path.join(ROOT, "analysis", "ss")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, "ar_gapfill_loso.csv"), index=False)
        print("\n===== CROSS-SITE MEAN (over sites) =====")
        print(f"{'model':<28}{'RMSE h0-2':>10}{'h2-5':>7}{'h5+':>7}{'covFIX h5+':>12}")
        for name, gdf in df.groupby("model", sort=False):
            print(f"{name:<28}{gdf['rmse_h0-2'].mean():>10.2f}{gdf['rmse_h2-5'].mean():>7.2f}"
                  f"{gdf['rmse_h5+'].mean():>7.2f}{gdf['covfix_h5+'].mean():>12.2f}")
        print("\nwrote", os.path.join(out, "ar_gapfill_loso.csv"))


def _ref_from_run(run):
    """Build a DateTime/night/step reference frame from a run's test predictions timeline."""
    for cand in [os.path.join(run, "metrics", "predictions.parquet"), os.path.join(run, "predictions.parquet")]:
        if os.path.exists(cand):
            w = pd.read_parquet(cand)
            if "DateTime" in w.columns:
                return night_ref(w[["DateTime"]])
    return None


if __name__ == "__main__":
    main()
