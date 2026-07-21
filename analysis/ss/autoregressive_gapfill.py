#!/usr/bin/env python
"""Autoregressive within-night gap-filling for all methods, across all five held-out sites.

Protocol: for each contiguous night, anchor at the last observed NEE before the night,
forecast NEE_{t+1}, feed that forecast back as NEE_t for the next step, using the real drivers
at every step (only NEE is synthetic). Every increment-form model uses NEE_t
(nee_pred = bNEE + drift + noise, incl. MDN comp_means = b + decoder).

Two outputs per (site, model):
  * deterministic rollout (feed the mean forward)   -> point RMSE by hours-into-gap
  * ensemble rollout (M noisy members)              -> 90% band coverage by hours-into-gap
    For WienerNet-SS a structural-band variant is also reported: each member carries a
    persistent structural offset ~ N(0, sigma_struct) (sigma_struct estimated on the training
    sites), added on top of the aleatoric spread — the rollout analogue of the gap_band_eval
    band.

Trees (RF/XGB) are rolled out under the same protocol and emit no uncertainty band
(point-only).

Writes per-seed CSVs (ar_gapfill_s<seed>.csv), the merged ar_gapfill_loso.csv and the pooled
gap_crps_summary.csv under analysis/ss/.
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
from wienernet.evaluation import crps_ensemble
from wienernet.baselines import RandomForestBaseline, XGBoostBaseline
from structural import structural_sigma

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
BINS = [("h0-2", lambda s: s < 4), ("h2-5", lambda s: (s >= 4) & (s < 10)), ("h5+", lambda s: s >= 10)]


_BUNDLE_CACHE = {}

# When set (via --prefer-root), a run of the same basename under this root wins over the
# canonical location; arms with no run there fall back to the canonical location.
PREFER_ROOT: str | None = None
# Which checkpoint the rollout loads.
CHECKPOINT: str = "last"


def _resolve(path: str) -> str:
    if PREFER_ROOT:
        alt = os.path.join(PREFER_ROOT, os.path.basename(path))
        if os.path.isdir(alt):
            return alt
    return path


def _get_bundle(cfg):
    """Build (or reuse) the leave-one-site-out dataloader bundle. Cached per (holdout_site, drivers):
    every model rolled out for a given held-out site shares the same test set / train sites, so the
    data is built once per site rather than once per model."""
    key = (str(cfg.data.get("holdout_site")), tuple(cfg.data.drivers))
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE.clear()   # site-major loop -> keep one bundle (bounds RAM across parallel seeds)
        _BUNDLE_CACHE[key] = build_dataloaders(
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
    return _BUNDLE_CACHE[key]


def load_torch(run_dir, want_struct=False):
    run_dir = Path(run_dir)
    cfg = ev._load_resolved_config(run_dir)
    bundle = _get_bundle(cfg)
    model = ev._rebuild_any_model(cfg, feature_dim=bundle.feature_dim, device=DEVICE)
    load_model_weights(run_dir / "checkpoints" / f"{CHECKPOINT}.pth", model, map_location=DEVICE)
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
    det_err, steps, obs_all, mem_raw, mem_fix = [], [], [], [], []
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
            det_err.append((det_next - obs) ** 2); steps.append(j); obs_all.append(obs)
            mem_raw.append(pred.cpu().numpy())               # ensemble members -> full gap distribution
            if offset is not None:
                mem_fix.append((pred + offset).cpu().numpy())
            ens = pred; det = det_next
    det_err, steps = np.asarray(det_err), np.asarray(steps)
    obs_all, mem_raw = np.asarray(obs_all), np.asarray(mem_raw)     # (N,), (N, M)
    mem_fix = np.asarray(mem_fix) if mem_fix else None

    def _dist(mem):
        """Full distributional scores from an (N,M) gap ensemble vs obs: CRPS, coverage@{50,90,95},
        sharpness (90% width), and rank PIT."""
        q = np.quantile(mem, [0.025, 0.05, 0.25, 0.75, 0.95, 0.975], axis=1)   # (6, N)
        return dict(crps=crps_ensemble(obs_all, mem),
                    cov50=((obs_all >= q[2]) & (obs_all <= q[3])).astype(float),
                    cov90=((obs_all >= q[1]) & (obs_all <= q[4])).astype(float),
                    cov95=((obs_all >= q[0]) & (obs_all <= q[5])).astype(float),
                    sharp=(q[4] - q[1]), pit=(mem < obs_all[:, None]).mean(axis=1))

    def _pit_ks(p):     # KS distance of the PIT from Uniform(0,1); 0 = perfectly calibrated
        p = np.sort(p); n = len(p)
        return float(np.max(np.abs((np.arange(1, n + 1) / n) - p))) if n else float("nan")

    binned = lambda vals, fn: {lab: fn(vals[m(steps)]) for lab, m in BINS if m(steps).any()}
    mean_ = lambda v: float(np.mean(v))
    r = dict(rmse=binned(det_err, lambda v: float(np.sqrt(v.mean()))), n_nights=len(groups))
    for tag, mem in (("raw", mem_raw), ("fix", mem_fix)):
        if mem is None:
            continue
        d = _dist(mem)
        r[f"crps_{tag}"] = binned(d["crps"], mean_)
        r[f"cov_{tag}"] = binned(d["cov90"], mean_)          # 90% (back-compatible key)
        r[f"cov50_{tag}"] = binned(d["cov50"], mean_)
        r[f"cov95_{tag}"] = binned(d["cov95"], mean_)
        r[f"sharp_{tag}"] = binned(d["sharp"], mean_)
        r[f"pitks_{tag}"] = binned(d["pit"], _pit_ks)
    return r


def rollout_tree(run_dir, night_ref_df=None, min_len=6, max_nights=200, seed=0):
    """Autoregressive gap-fill for a tree baseline — same protocol as `rollout_torch`.

    The trees take the current flux `bNEE` as a feature, so they are run through a gap by feeding
    each prediction back as the next step's input: anchor once on the observed NEE at the gap edge,
    then propagate. Observed DRIVERS are refreshed every step (they remain available during a gap);
    only NEE is unobserved and therefore self-supplied.

    Point predictors -> RMSE only; no ensemble, hence no CRPS/coverage/PIT.
    """
    run_dir = Path(run_dir)
    cfg = ev._load_resolved_config(run_dir)
    bundle = _get_bundle(cfg)
    ckpt = run_dir / "checkpoints" / "model.joblib"
    if not ckpt.exists():
        return None
    variant = str(cfg.get("model", {}).get("variant", "")).lower()
    if "xgb" in variant or "xgb" in run_dir.name:
        model = XGBoostBaseline.load(ckpt)
    else:
        model = RandomForestBaseline.load(ckpt)

    ds = bundle.test_dataset
    X = ds.X.numpy(); bnee = ds.bNEE.numpy().reshape(-1); k = ds.k.numpy(); obs = ds.NEE.numpy()
    dtv = pd.to_datetime(bundle.test_df["DateTime"])
    night = (dtv.diff().dt.total_seconds() / 60 != 30).cumsum().values

    groups = [np.where(night == nid)[0] for nid in np.unique(night)]
    groups = [g for g in groups if len(g) >= min_len]
    rng = np.random.default_rng(seed)
    if len(groups) > max_nights:
        groups = [groups[i] for i in rng.choice(len(groups), max_nights, replace=False)]

    # Nights are independent, so the rollout is vectorised ACROSS nights: at step j we advance every
    # still-active night in ONE predict() call. Sequential within a night (as it must be), batched
    # over nights -- ~40 batched calls instead of ~3000 single-row ones against a 350 MB forest.
    cur = np.array([bnee[g[0]] for g in groups], dtype=float)     # per-night running state
    err, steps = [], []
    for j in range(max(len(g) for g in groups)):
        act = np.array([gi for gi, g in enumerate(groups) if j < len(g)])
        if not len(act):
            break
        rows = np.array([groups[gi][j] for gi in act])
        feat = np.concatenate([X[rows], cur[act, None], k[rows]], axis=1)
        pred = np.asarray(model.predict(feat), dtype=float).reshape(-1)
        err.extend((pred - obs[rows].astype(float)) ** 2)
        steps.extend([j] * len(act))
        cur[act] = pred                              # <- feed the prediction forward
    err, steps = np.asarray(err), np.asarray(steps)
    binned = {lab: float(np.sqrt(err[msk(steps)].mean())) for lab, msk in BINS if msk(steps).any()}
    return dict(rmse=binned, cov_raw={}, n_nights=len(groups))


def night_ref(test_df):
    """Rows with a within-night step index, for aligning tree predictions."""
    d = test_df[["DateTime"]].copy()
    d["DateTime"] = pd.to_datetime(d["DateTime"])
    d["night"] = (d["DateTime"].diff().dt.total_seconds() / 60 != 30).cumsum()
    d["step"] = d.groupby("night").cumcount()
    return d


def write_gap_summary(df, out):
    """Aggregate table: all methods x {all 5 sites, 4 sites} x horizon, RMSE / CRPS (raw+fix) /
    coverage (raw+fix). Derived from the per-site ar_gapfill_loso rows."""
    HS = ["h0-2", "h2-5", "h5+"]
    METRICS = ["rmse", "crpsraw", "crpsfix", "covraw", "covfix", "cov50raw", "cov50fix",
               "cov95raw", "cov95fix", "sharpraw", "sharpfix", "pitksraw", "pitksfix"]
    rows = []
    for scope, frame in [("all5", df), ("clean4", df[df.site != "redmere_1"])]:
        for name, gdf in frame.groupby("model", sort=False):
            rec = {"scope": scope, "model": name}
            for m in METRICS:
                for h in HS:
                    col = f"{m}_{h}"
                    ok = col in gdf.columns and gdf[col].notna().any()
                    rec[col] = round(float(gdf[col].mean()), 3) if ok else ""
            rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(out, "gap_crps_summary.csv"), index=False)


def main(seed=0):
    LOSO = os.path.join(ROOT, "outputs", "ss_loso")
    FL = os.path.join(ROOT, "outputs", "final_loso")
    print(f"### AR gap-fill sweep — seed {seed}, device {DEVICE} ###")

    def paths(site):
        return [
            # --- WienerNet-SS likelihood axis: learned-diurnal, single-Wiener, GT-k, residual OFF;
            #     only the likelihood family changes. ALD uses the +sigma_struct band
            #     (torch_struct -> covFIX/crpsFIX); the rest use the raw band. ---
            ("WN-SS (ALD)",        os.path.join(LOSO, f"{site}_s{seed}_ldiur_wien"), "torch_struct"),
            ("WN-SS (Gaussian)",   os.path.join(LOSO, f"{site}_s{seed}_ldiur_wien_gauss"), "torch"),
            ("WN-SS (beta-NLL)",   os.path.join(LOSO, f"{site}_s{seed}_ldiur_wien_beta"), "torch"),
            ("WN-SS (Student-t)",  os.path.join(LOSO, f"{site}_s{seed}_ldiur_wien_studt"), "torch"),
            ("WN-SS (mixture)",    os.path.join(LOSO, f"{site}_s{seed}_ldiur_wien_mix"), "torch"),
            # --- other WN-SS arms: residual on/off, parameter head, given-diurnal ---
            ("WN-SS (+residual)",            os.path.join(LOSO, f"{site}_s{seed}_ldiur_res_wien"), "torch"),
            ("WN-SS (predicted-k)",          os.path.join(LOSO, f"{site}_s{seed}_ldiur_wien_predk"), "torch"),
            # --- state-space noise law: matched partner of each single-Wiener arm above (ALD only) ---
            ("WN-SS (state-space)",                os.path.join(LOSO, f"{site}_s{seed}_ldiur_ss"), "torch"),
            ("WN-SS (state-space, +residual)",     os.path.join(LOSO, f"{site}_s{seed}_ldiur_res_ss"), "torch"),
            ("WN-SS (state-space, predicted-k)",   os.path.join(LOSO, f"{site}_s{seed}_ldiur_ss_predk"), "torch"),
            ("WN-SS (given-diurnal, st-sp)", os.path.join(LOSO, f"{site}_s{seed}_diur_ss"), "torch_struct"),
            ("WN-SS (given-diurnal, Wiener)",os.path.join(LOSO, f"{site}_s{seed}_diur_wien"), "torch"),
            # --- baselines (matched-loss gap comparison) ---
            ("Neural SDE (Gaussian)",  os.path.join(FL, f"comp_neuralsde_{site}_s{seed}"), "torch"),
            ("Neural SDE (Student-t)", os.path.join(LOSO, f"{site}_s{seed}_base_nsde_studt"), "torch"),
            ("Neural SDE (ALD)",       os.path.join(LOSO, f"{site}_s{seed}_base_nsde_ald"), "torch"),
            ("mean-var (Gaussian)",    os.path.join(FL, f"comp_meanvar_{site}_s{seed}"), "torch"),
            ("mean-var (Student-t)",   os.path.join(LOSO, f"{site}_s{seed}_base_mv_studt"), "torch"),
            ("mean-var (ALD)",         os.path.join(LOSO, f"{site}_s{seed}_base_mv_ald"), "torch"),
            ("Analytical (Gaussian)",  os.path.join(FL, f"prior_analytical_{site}_s{seed}"), "torch"),
            ("Analytical (Student-t)", os.path.join(LOSO, f"{site}_s{seed}_base_analyt_studt"), "torch"),
            ("MDN (mixture)",          os.path.join(FL, f"comp_mdn_{site}_s{seed}"), "torch"),
            ("Random Forest",          os.path.join(FL, f"comp_rf_{site}_s{seed}"), "tree"),
            ("XGBoost",                os.path.join(FL, f"comp_xgb_{site}_s{seed}"), "tree"),
        ]

    rows = []
    for site in SITES:
        print(f"\n===== {site} =====")
        print(f"{'model':<28}{'RMSE h0-2':>10}{'h2-5':>7}{'h5+':>7}{'covRAW h0-2':>12}{'h2-5':>7}{'h5+':>7}{'covFIX h0-2':>12}{'h2-5':>7}{'h5+':>7}")
        # a reference night frame for tree alignment (from any torch run of this site)
        ref = None
        for name, run, kind in paths(site):
            run = _resolve(run)
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
                mtab = {"rmse": rm, "covraw": cr, "covfix": cf,
                        "crpsraw": r.get("crps_raw", {}), "crpsfix": r.get("crps_fix", {}),
                        "pitksraw": r.get("pitks_raw", {}), "pitksfix": r.get("pitks_fix", {}),
                        "cov50raw": r.get("cov50_raw", {}), "cov50fix": r.get("cov50_fix", {}),
                        "cov95raw": r.get("cov95_raw", {}), "cov95fix": r.get("cov95_fix", {}),
                        "sharpraw": r.get("sharp_raw", {}), "sharpfix": r.get("sharp_fix", {})}
                row = {"site": site, "seed": seed, "model": name}
                for mk, md in mtab.items():
                    for h in ("h0-2", "h2-5", "h5+"):
                        row[f"{mk}_{h}"] = g(md, h)
                rows.append(row)
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"{name}: ERROR {e}")

    # per-seed CSV; `--merge` concatenates the seeds and writes the pooled summary
    if rows:
        out = os.path.join(ROOT, "analysis", "ss")
        df = pd.DataFrame(rows)
        target = os.path.join(out, f"ar_gapfill_s{seed}.csv")
        df.to_csv(target, index=False)
        print(f"\n===== seed {seed}: CROSS-SITE MEAN (over sites) =====")
        print(f"{'model':<28}{'RMSE h0-2':>10}{'h2-5':>7}{'h5+':>7}{'CRPS h5+':>10}{'PITks h5+':>11}")
        for name, gdf in df.groupby("model", sort=False):
            gv = lambda c: gdf[c].mean() if c in gdf.columns else float("nan")
            print(f"{name:<28}{gv('rmse_h0-2'):>10.2f}{gv('rmse_h2-5'):>7.2f}{gv('rmse_h5+'):>7.2f}"
                  f"{gv('crpsraw_h5+'):>10.3f}{gv('pitksraw_h5+'):>11.3f}")
        print("\nwrote", target)


def merge_seeds():
    """Concatenate every per-seed ar_gapfill_s*.csv into ar_gapfill_loso.csv and write the pooled
    gap_crps_summary.csv (aggregating over site AND seed)."""
    import glob
    out = os.path.join(ROOT, "analysis", "ss")
    parts = sorted(glob.glob(os.path.join(out, "ar_gapfill_s*.csv")))
    if not parts:
        print("no per-seed CSVs to merge"); return
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(os.path.join(out, "ar_gapfill_loso.csv"), index=False)
    write_gap_summary(df, out)
    n_seed = df["seed"].nunique() if "seed" in df.columns else 1
    print(f"merged {len(parts)} seed files -> ar_gapfill_loso.csv "
          f"({len(df)} rows, {df['model'].nunique()} models x {df['site'].nunique()} sites x {n_seed} seeds)")
    print("wrote gap_crps_summary.csv")


def _ref_from_run(run):
    """Build a DateTime/night/step reference frame from a run's test predictions timeline."""
    for cand in [os.path.join(run, "metrics", "predictions.parquet"), os.path.join(run, "predictions.parquet")]:
        if os.path.exists(cand):
            w = pd.read_parquet(cand)
            if "DateTime" in w.columns:
                return night_ref(w[["DateTime"]])
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="training seed of the runs to roll out")
    ap.add_argument("--merge", action="store_true", help="merge per-seed CSVs + write the summary")
    ap.add_argument("--cpu-ok", action="store_true", help="allow CPU (default: require CUDA)")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="last",
                    help="which checkpoint to roll out")
    ap.add_argument("--prefer-root", default=None,
                    help="prefer same-named runs under this root (e.g. outputs/ss_lrfix)")
    a = ap.parse_args()
    CHECKPOINT = a.checkpoint
    if CHECKPOINT != "last":
        print(f"### rolling out {CHECKPOINT}.pth ###")
    if a.prefer_root:
        PREFER_ROOT = os.path.abspath(a.prefer_root)
        print(f"### preferring runs under {PREFER_ROOT} where they exist ###")
    if a.merge:
        merge_seeds()
    else:
        if DEVICE != "cuda" and not a.cpu_ok:
            raise SystemExit("CUDA not available — rerun with --cpu-ok to allow CPU (much slower).")
        main(seed=a.seed)
