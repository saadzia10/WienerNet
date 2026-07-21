#!/usr/bin/env python
"""Re-score every torch run in outputs/ss_full5 from `best.pth` instead of `last.pth`.

WHAT THIS MEANS (read before using any number it produces)
----------------------------------------------------------
These runs were trained with `data.val_frac=null`, so the val loader IS the held-out test site and
`best.pth` is the epoch that scored best ON THE TEST SITE. Scoring it is explicit test-set model
selection: the reported metric is the minimum of a curve chosen using the same data it is reported
on. Numbers produced here are an OPTIMISTIC BOUND on each model's skill, not an honest estimate,
and they are not directly comparable to the calibrated Analytical SDE / tree arms, which have no
epoch to select. Requested deliberately to see the ceiling; label as such wherever they appear.

Metrics are overwritten IN PLACE (out_dir defaults to <run>/metrics) because the disk has no room
for a second copy of 460 predictions.parquet. The `last.pth` results are archived first as small
JSONs under analysis/ss/v2_last_archive/metrics_last/.

Trees (comp_rf / comp_xgb) have no checkpoint to choose and are skipped unchanged.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "outputs", "ss_full5")


def job(run: str, logdir: str) -> tuple[str, int]:
    name = os.path.basename(run)
    with open(os.path.join(logdir, f"{name}.log"), "w") as fh:
        rc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"),
                             "--run", run, "--checkpoint", "best"],
                            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"  [{'ok  ' if rc == 0 else 'FAIL'}] {name}", flush=True)
    return (name, rc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--skip-done-from", default=None,
                    help="prior driver log; runs it reported [ok] are skipped (resume)")
    a = ap.parse_args()

    logdir = os.path.join(ROOT, "scratchpad_sweep", "reeval_best_logs")
    os.makedirs(logdir, exist_ok=True)

    done = set()
    if a.skip_done_from and os.path.exists(a.skip_done_from):
        for line in open(a.skip_done_from):
            if "[ok  ]" in line:
                done.add(line.split("]")[-1].strip())
        print(f"resuming: {len(done)} runs already scored from best.pth")

    runs = []
    for n in sorted(os.listdir(SWEEP)):
        if n in done:
            continue
        d = os.path.join(SWEEP, n)
        if not os.path.isdir(d) or n == "evaluation_summary":
            continue
        if not os.path.exists(os.path.join(d, "checkpoints", "best.pth")):
            continue                       # trees + anything without a saved best
        runs.append(d)

    print(f"re-scoring {len(runs)} runs from best.pth, parallel={a.parallel}", flush=True)
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        res = list(ex.map(lambda r: job(r, logdir), runs))
    bad = [n for n, rc in res if rc != 0]
    print(f"\ndone: {len(res) - len(bad)}/{len(res)} ok" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
