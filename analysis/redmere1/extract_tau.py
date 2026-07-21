#!/usr/bin/env python
"""Dump the per-site momentum-flux (Tau) distribution, on two row bases.

  * MODEL basis  -> `tau_by_site.json` / `.npz`: the *same* filtering as training
    (`build_dataloaders` with the run's config), i.e. nighttime + every driver non-NaN.
    These are exactly the rows the model saw / was tested on.
  * RAW basis    -> `tau_by_site_raw.json` / `.npz`: the site parquets with only the
    nighttime filter (Rg < threshold) and a Tau-NaN drop, i.e. *before* the pipeline's
    driver-completeness dropna removes the most extreme Tau rows.

The two differ substantially at Redmere 1 (raw tails reach ~1e5, model-basis tails ~5e3)
because the corrupted-Tau rows often also miss another driver.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from omegaconf import OmegaConf
from pathlib import Path
import evaluate as ev
from wienernet.data import build_dataloaders

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = Path(ROOT) / "outputs" / "ss_loso" / "redmere_1_s0_ldiur_wien"
DRIVER = "Tau"


def _summarise(v):
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=0)),
            "min": float(v.min()), "max": float(v.max()),
            "p1": float(np.percentile(v, 1)), "p50": float(np.percentile(v, 50)),
            "p99": float(np.percentile(v, 99))}


def raw_basis(cfg):
    """Per-site Tau with only the nighttime filter + a Tau-NaN drop (no driver-completeness dropna)."""
    paths = OmegaConf.to_container(cfg.data.site_paths, resolve=True)
    thr = float(cfg.data.night_radiation_threshold)
    holdout = str(cfg.data.get("holdout_site") or "")
    arrays, summary = {}, {}
    for site, path in paths.items():
        df = pd.read_parquet(os.path.join(ROOT, path))
        if bool(cfg.data.nighttime_only):
            df = df[df["Rg"] < thr]
        v = df[DRIVER].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        arrays[site] = v
        summary[site] = {"split": "test" if site == holdout else "train", **_summarise(v)}
    pool = np.concatenate([arrays[s] for s in arrays if s != holdout])
    summary["_train_pool"] = {"mean": float(pool.mean()), "sd": float(pool.std(ddof=0)),
                              "p1": float(np.percentile(pool, 1)), "p99": float(np.percentile(pool, 99))}
    return arrays, summary


def main():
    cfg = ev._load_resolved_config(RUN)
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
    tr, te = bundle.train_df, bundle.test_df
    site_col = "site" if "site" in tr.columns else "site_name"
    arrays, summary = {}, {}
    for df, split in ((tr, "train"), (te, "test")):
        for site, g in df.groupby(site_col):
            v = g[DRIVER].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            arrays[str(site)] = v
            summary[str(site)] = {"split": split, **_summarise(v)}
    # training-pool moments (the reference the standardised shift is measured against)
    pool = tr[DRIVER].to_numpy(dtype=float); pool = pool[np.isfinite(pool)]
    summary["_train_pool"] = {"mean": float(pool.mean()), "sd": float(pool.std(ddof=0)),
                              "p1": float(np.percentile(pool, 1)), "p99": float(np.percentile(pool, 99))}
    np.savez_compressed(os.path.join(HERE, "tau_by_site.npz"), **arrays)
    json.dump(summary, open(os.path.join(HERE, "tau_by_site.json"), "w"), indent=2)
    print("=== MODEL basis (nighttime + all drivers non-NaN) ===")
    for s, v in summary.items():
        print(f"{s:<12}", json.dumps(v))

    raw_arrays, raw_summary = raw_basis(cfg)
    np.savez_compressed(os.path.join(HERE, "tau_by_site_raw.npz"), **raw_arrays)
    json.dump(raw_summary, open(os.path.join(HERE, "tau_by_site_raw.json"), "w"), indent=2)
    print("\n=== RAW basis (nighttime only, Tau-NaN dropped) ===")
    for s, v in raw_summary.items():
        print(f"{s:<12}", json.dumps(v))
    print("\nwrote tau_by_site{,_raw}.npz / .json")


if __name__ == "__main__":
    main()
