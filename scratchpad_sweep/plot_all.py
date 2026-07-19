"""Generate manuscript decomposition plots for every run in outputs/dt_sweep.

Torch runs -> evaluate_run(make_plots=True) (re-predicts, writes metrics/plots/).
RF/XGB     -> level plots straight from their predictions.parquet (plots/).
Robust: a failing run is logged and skipped.
"""
from __future__ import annotations
import os, re, sys, warnings
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import pandas as pd

REPO = Path("/home/cognitia/Desktop/Work/PhD/WienerNet")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
ROOT = REPO / "outputs" / "dt_sweep"
BASELINES = {"rf", "xgb"}
LEAF = re.compile(r"k(\d+)_s(\d+)$")

import torch
from evaluate import evaluate_run
from wienernet.evaluation import emit_manuscript_plots

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

done = fail = 0
for method_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    method = method_dir.name
    is_base = method in BASELINES
    for leaf in sorted(method_dir.iterdir()):
        if not leaf.is_dir() or not LEAF.search(leaf.name):
            continue
        tag = f"{method}/{leaf.name}"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if is_base:
                    df = pd.read_parquet(leaf / "predictions.parquet")
                    gt = {"nee": df["gt_nee"].to_numpy()}
                    preds = {"nee": df["pred_nee"].to_numpy()}
                    n = emit_manuscript_plots(gt, preds, df, model_name=method,
                                              out_dir=leaf / "plots")
                else:
                    if not (leaf / "checkpoints" / "best.pth").exists():
                        raise FileNotFoundError("no best.pth")
                    evaluate_run(leaf, out_dir=None, device=DEVICE, make_plots=True)
            done += 1
            print(f"OK   {tag}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"FAIL {tag}: {exc}")

print(f"\nPLOTS DONE  ok={done} fail={fail}")
