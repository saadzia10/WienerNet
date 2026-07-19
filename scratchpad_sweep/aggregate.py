"""Aggregate the dt-scale sweep into one comparison view (all methods x k x seed).

For torch runs, runs evaluate_run (which also writes predictions/per_site — the
"evaluate all trained models" step). For RF/XGB, reads the metrics.json written
at train time. Everything is scored on the SAME target (NEE_next) with the same
metric bundle, so methods are directly comparable.

Writes to outputs/dt_sweep/:
  comparison_long.csv     — one row per (method, k, seed) x metric
  comparison_summary.csv  — mean +/- std over seeds, per (method, k)
  comparison_r2_pivot.csv — quick method x k table of NEE R2 (mean)
and prints the R2 / RMSE pivots.
"""
from __future__ import annotations
import json, re, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/cognitia/Desktop/Work/PhD/WienerNet")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO / "outputs" / "dt_sweep"
BASELINES = {"rf", "xgb"}
METRICS = ["rmse", "mae", "r2", "mmd", "kl", "wasserstein"]
LEAF = re.compile(r"k(\d+)_s(\d+)$")

import torch
from evaluate import evaluate_run  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect() -> pd.DataFrame:
    rows = []
    for method_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        method = method_dir.name
        is_base = method in BASELINES
        for leaf in sorted(method_dir.iterdir()):
            m = LEAF.search(leaf.name)
            if not m or not leaf.is_dir():
                continue
            k, seed = int(m.group(1)), int(m.group(2))
            try:
                if is_base:
                    mj = json.loads((leaf / "metrics.json").read_text())
                    nee = mj["raw"]["nee"]
                else:
                    if not (leaf / "checkpoints" / "best.pth").exists():
                        raise FileNotFoundError("no best.pth (train failed?)")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        metrics = evaluate_run(leaf, out_dir=None, device=DEVICE)
                    nee = metrics["raw"]["nee"]
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP {method} k{k} s{seed}: {exc}")
                continue
            row = {"method": method, "k": k, "seed": seed}
            row.update({mt: (float(nee[mt]) if nee.get(mt) is not None else np.nan) for mt in METRICS})
            rows.append(row)
            print(f"  {method:18s} k{k} s{seed}  R2={row['r2']:.4f}  RMSE={row['rmse']:.4f}")
    return pd.DataFrame(rows)


def main():
    if not ROOT.exists():
        raise SystemExit(f"{ROOT} not found — run the sweep first")
    df = collect()
    if df.empty:
        raise SystemExit("no runs collected")
    df = df.sort_values(["method", "k", "seed"]).reset_index(drop=True)
    df.to_csv(ROOT / "comparison_long.csv", index=False)

    # mean +/- std over seeds, per (method, k)
    agg = df.groupby(["method", "k"])[METRICS].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(ROOT / "comparison_summary.csv", index=False)

    r2piv = df.pivot_table(index="method", columns="k", values="r2", aggfunc="mean")
    rmsepiv = df.pivot_table(index="method", columns="k", values="rmse", aggfunc="mean")
    r2piv.to_csv(ROOT / "comparison_r2_pivot.csv")

    pd.set_option("display.width", 140, "display.float_format", lambda x: f"{x:.4f}")
    print("\n================ NEE R2 (mean over seeds) — rows=method, cols=k ================")
    print(r2piv)
    print("\n================ NEE RMSE (mean over seeds) ================")
    print(rmsepiv)
    print(f"\nwrote:\n  {ROOT/'comparison_long.csv'}\n  {ROOT/'comparison_summary.csv'}\n  {ROOT/'comparison_r2_pivot.csv'}")


if __name__ == "__main__":
    main()
