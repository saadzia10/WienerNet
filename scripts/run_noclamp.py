#!/usr/bin/env python
"""Ablate the DRIFT CLAMP: the primary (learned-diurnal, GT-k, single-Wiener, ALD) with
`model.drift_clamp=null`.

The primary bounds its drift with `c*tanh(drift/c)`, c=1.0, which no baseline has -- so part of
its OOD advantage may be the saturation rather than the physics. This isolates that.

Overrides = `ldiur_wien` (the primary) verbatim with `model.drift_clamp=null`. Same uniform protocol
as the rest of outputs/ss_full5 (reduce_on_plateau on the held-out site, no train-side val split).
Scored from best.pth to match the current tables.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
OUT = os.path.join(ROOT, "outputs", "ss_full5")
BASE = ["mlflow.enabled=false", "data.num_workers=0", "data.split_strategy=site_holdout",
        "training.num_epochs=120", "device=cuda", "model=piae_increment",
        "model.exact_reco_drift=true", "model.predict_temp_derivative=true",
        "model.drift_tendency=learned_diurnal", "model.residual=false",
        "model.noise_state_space=false",
        "model.noise_zero_mean=true", "loss=nll_ald",
        "model.physics_k_source=ground_truth", "model.predict_k=false",
        "model.drift_clamp=null",                            # <- the ONLY change vs the primary
        "training.scheduler.name=reduce_on_plateau", "training.scheduler.monitor=val",
        "data.val_frac=null"]


def job(site, seed, logdir):
    name = f"{site}_s{seed}_ldiur_wien_noclamp"
    dst = os.path.join(OUT, name)
    if os.path.exists(os.path.join(dst, "metrics", "probabilistic.json")):
        print(f"  [skip] {name}", flush=True); return (name, 0)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), *BASE,
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
    logdir = os.path.join(ROOT, "scratchpad_sweep", "ss_predk_logs"); os.makedirs(logdir, exist_ok=True)
    tasks = [(s, sd) for s in SITES for sd in range(5)]
    print(f"{len(tasks)} runs -> {OUT}/<site>_s<seed>_ldiur_wien_noclamp", flush=True)
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        res = list(ex.map(lambda t: job(*t, logdir), tasks))
    bad = [n for n, rc in res if rc != 0]
    print(f"\ndone: {len(res)-len(bad)}/{len(res)} ok" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
