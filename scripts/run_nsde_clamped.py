#!/usr/bin/env python
"""Clamp-controlled re-run of the Neural SDE baselines (all 3 manuscript likelihoods).

WHY: WienerNetModel bounds its drift rate with c*tanh(rate/c), c=1.0, and no baseline did. The
clamp is inert in-distribution (clean-4 RMSE 1.95 clamped vs 1.97 unclamped) but decisive OOD
(Redmere 1: 2.50 clamped vs 53.50 unclamped, 2/5 seeds detonating). So the published OOD contrast
confounds "physics drift generalises" with "a bounded increment cannot explode". These runs give
the Neural SDE the SAME bound so the surviving gap is attributable to the physics.

Only the Neural SDE is re-run: mean-variance and MDN have no drift term to clamp, and the
Analytical SDE's drift is pure physics with no learned head.

Names get a `_clamp` suffix so the originals are preserved for the ablation.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
OUT = os.path.join(ROOT, "outputs", "ss_full5")
COMMON = ["mlflow.enabled=false", "data.num_workers=0", "data.split_strategy=site_holdout",
          "training.num_epochs=120", "device=cuda", "model=neural_sde",
          "model.drift_clamp=1.0",                     # <- the only change vs the published arms
          "training.scheduler.name=reduce_on_plateau", "training.scheduler.monitor=val",
          "data.val_frac=null"]
ARMS = [("comp_neuralsde_{site}_s{seed}_clamp", "loss=nll_gaussian"),
        ("{site}_s{seed}_base_nsde_studt_clamp", "loss=nll_student_t"),
        ("{site}_s{seed}_base_nsde_ald_clamp",   "loss=nll_ald")]


def job(tmpl, loss, site, seed, logdir):
    name = tmpl.format(site=site, seed=seed)
    dst = os.path.join(OUT, name)
    if os.path.exists(os.path.join(dst, "metrics", "probabilistic.json")):
        print(f"  [skip] {name}", flush=True); return (name, 0)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), *COMMON, loss,
           f"data.holdout_site={site}", f"seed={seed}", f"hydra.run.dir={dst}"]
    with open(os.path.join(logdir, f"{name}.log"), "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            rc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"),
                                 "--run", dst, "--checkpoint", "best"],
                                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"  [{'ok  ' if rc == 0 else 'FAIL'}] {name}", flush=True)
    return (name, rc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    logdir = os.path.join(ROOT, "scratchpad_sweep", "nsde_clamp_logs"); os.makedirs(logdir, exist_ok=True)
    tasks = [(t, l, s, sd) for t, l in ARMS for s in SITES for sd in range(5)]
    print(f"{len(tasks)} runs (3 likelihoods x 5 sites x 5 seeds), drift_clamp=1.0", flush=True)
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        res = list(ex.map(lambda t: job(*t, logdir), tasks))
    bad = [n for n, rc in res if rc != 0]
    print(f"\ndone: {len(res)-len(bad)}/{len(res)} ok" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
