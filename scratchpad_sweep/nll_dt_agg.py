"""Evaluate + aggregate the NLL dt-sweep (β-NLL vs baseline, k=1/4/12 = 30m/2h/6h).

Per run: evaluate_run(make_plots=True) -> nee_mean point metrics + calibration.json
+ the new mean+band plots. Then aggregate the noise-vs-residual balance and the
calibration as a function of horizon.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"; os.environ.setdefault("MPLBACKEND", "Agg")
import sys, re, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = "/home/cognitia/Desktop/Work/PhD/WienerNet"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/scripts")
ROOT = Path(REPO) / "outputs" / "nll_dt_sweep"
LEAF = re.compile(r"k(\d+)_s(\d+)$")
KMIN = {1: "30m", 4: "2h", 12: "6h"}
from evaluate import evaluate_run

def r2(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

rows = []
for method_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    method = method_dir.name
    for leaf in sorted(method_dir.iterdir()):
        mm = LEAF.search(leaf.name)
        if not mm or not leaf.is_dir():
            continue
        k, s = int(mm.group(1)), int(mm.group(2))
        if not (leaf / "checkpoints" / "best.pth").exists():
            print(f"SKIP {method} k{k} s{s} (no ckpt)"); continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evaluate_run(leaf, out_dir=None, device="cuda" if torch.cuda.is_available() else "cpu", make_plots=True)
        pp = pd.read_parquet(leaf / "metrics" / "predictions.parquet")
        cal = json.loads((leaf / "metrics" / "calibration.json").read_text())
        bnee = pp["NEE"].to_numpy(); nn = pp["NEE_next"].to_numpy(); dt = pp["dt"].to_numpy()
        f = pp["pred_f"].to_numpy()
        det_incr = f * dt
        dnee_obs = nn - bnee
        noise_incr = pp["pred_noise"].to_numpy() * np.sqrt(dt) if "pred_noise" in pp else np.zeros(len(pp))
        rows.append({
            "method": method, "k": k, "gap": KMIN.get(k, f"k{k}"), "seed": s, "n": len(pp),
            "incr_R2": round(r2(dnee_obs, det_incr), 4),
            "level_R2": round(r2(nn, bnee + det_incr), 4),
            "std_dNEE": round(float(np.std(dnee_obs)), 3),
            "std_det": round(float(np.std(det_incr)), 3),
            "std_noise": round(float(np.std(noise_incr)), 3),
            "det_frac": round(float(np.var(det_incr) / np.var(dnee_obs)), 4),   # predictable variance share
            "cover95": round(cal["cover95"], 3),
            "calib_ratio": round(cal["calib_std_ratio"], 3),
        })
        print(f"  {method:16s} {KMIN.get(k):>3s} s{s}  incrR2={rows[-1]['incr_R2']:.3f} "
              f"det/noise std={rows[-1]['std_det']:.2f}/{rows[-1]['std_noise']:.2f} cover95={rows[-1]['cover95']:.2f}")

df = pd.DataFrame(rows).sort_values(["method", "k", "seed"])
df.to_csv(ROOT / "nll_dt_long.csv", index=False)
agg = df.groupby(["method", "gap", "k"]).agg(
    incr_R2=("incr_R2", "mean"), level_R2=("level_R2", "mean"),
    std_det=("std_det", "mean"), std_noise=("std_noise", "mean"),
    det_frac=("det_frac", "mean"), cover95=("cover95", "mean"), calib_ratio=("calib_ratio", "mean"),
).reset_index().sort_values(["method", "k"])
agg.to_csv(ROOT / "nll_dt_summary.csv", index=False)
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}")
print("\n================ NLL dt-sweep summary (mean over seeds) ================")
print(agg.to_string(index=False))
print("\ndet_frac = predictable variance share of the increment (physics+residual).")
print("Expect det_frac / incr_R2 to RISE and std_noise to fall (relative) as the gap grows;")
print("cover95 should stay ~0.9-0.95 (calibration holds across horizons).")
print(f"\nwrote {ROOT/'nll_dt_long.csv'} and {ROOT/'nll_dt_summary.csv'}")
