"""Train old MSE+MMD vs the 3 NLL variants on inc_residual (seed 0, k=1), then
compare point accuracy, distributional metrics, decomposition + calibration."""
import os, subprocess, sys, json
os.environ["MKL_THREADING_LAYER"] = "GNU"   # avoid MKL/libgomp clash for child procs
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = "/home/cognitia/Desktop/Work/PhD/WienerNet"
BASE = "/tmp/claude-1000/-home-cognitia-Desktop-Work-PhD-WienerNet/b4a3f217-adb5-459c-9476-66908216cac1/scratchpad/nllcmp"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
os.chdir(REPO)

RUNS = {
    "mse_mmd_baseline": ["experiment=piae_increment_residual"],
    "nll_gaussian":     ["experiment=piae_increment_residual_nll", "loss=nll_gaussian"],
    "nll_beta":         ["experiment=piae_increment_residual_nll", "loss=nll_beta"],
    "nll_student_t":    ["experiment=piae_increment_residual_nll", "loss=nll_student_t"],
}

for name, ov in RUNS.items():
    d = f"{BASE}/{name}"
    if Path(f"{d}/checkpoints/best.pth").exists():
        print(f"skip train {name} (exists)"); continue
    cmd = ["python", "scripts/train.py", *ov, "seed=0", "device=cuda", "mlflow.enabled=false",
           "training.num_epochs=120", f"hydra.run.dir={d}"]
    print("TRAIN", name); subprocess.run(cmd, check=True,
          stdout=open(f"{BASE}_{name}.log", "w"), stderr=subprocess.STDOUT)

from evaluate import evaluate_run
from wienernet.evaluation.metrics import mmd_rbf, wasserstein

def corr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or np.std(a[m]) == 0 or np.std(b[m]) == 0: return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])

rows = []
for name in RUNS:
    d = Path(f"{BASE}/{name}")
    metrics = evaluate_run(d, out_dir=None, device="cuda")
    nee = metrics["raw"]["nee"]
    pp = pd.read_parquet(d / "metrics" / "predictions.parquet")
    Ta = pp["Ta"].to_numpy(); dt = pp["dt"].to_numpy()
    bnee = pp["NEE"].to_numpy(); nee_next = pp["NEE_next"].to_numpy()
    dnee_obs = nee_next - bnee
    resid = pp["pred_residual"].to_numpy() if "pred_residual" in pp else np.zeros(len(pp))
    noise = pp["pred_noise"].to_numpy() if "pred_noise" in pp else np.zeros(len(pp))
    f = pp["pred_f"].to_numpy()
    # deterministic mean increment = drift*dt ; empirical residual = observed - det
    det_incr = f * dt
    emp_resid = dnee_obs - det_incr                       # what the noise should cover
    # predicted increment noise std = sigma*sqrt(dt) ; sigma = pred_noise_stds
    sig = pp["pred_noise_stds"].to_numpy() if "pred_noise_stds" in pp else np.full(len(pp), np.nan)
    pred_incr_std = sig * np.sqrt(dt)
    # calibration: empirical residual std vs mean predicted std; 95% coverage
    cover = float(np.mean(np.abs(emp_resid) <= 1.96 * pred_incr_std)) if np.isfinite(pred_incr_std).all() else np.nan
    rows.append({
        "method": name,
        "nee_r2": round(nee["r2"], 4), "nee_rmse": round(nee["rmse"], 4),
        "mmd": round(nee["mmd"], 5), "wass": round(nee["wasserstein"], 4),
        "corr_noise_Ta": round(corr(noise, Ta), 3),
        "corr_resid_Ta": round(corr(resid, Ta), 3),
        "noise_mean": round(float(np.mean(noise)), 4),
        "emp_resid_std": round(float(np.std(emp_resid)), 3),
        "pred_incr_std_mean": round(float(np.nanmean(pred_incr_std)), 3),
        "cover95": round(cover, 3),
    })
df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/nll_comparison.csv", index=False)
pd.set_option("display.width", 200)
print("\n" + df.to_string(index=False))
print("\ncalibration: emp_resid_std should ~= pred_incr_std_mean; cover95 should ~= 0.95")
print("wrote", f"{BASE}/nll_comparison.csv")
