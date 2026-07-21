#!/usr/bin/env python
"""Re-run every TRAINED seed-42 ablation with a leakage-free LR schedule.

Motivation: seed 42 diverges under the original protocol (constant LR, 120 epochs, no early
stopping) -- its held-out loss climbs to ~2x its own minimum while train loss keeps falling, and
because scoring uses `last.pth` we score the degraded checkpoint. `reduce_on_plateau` fixes it,
but stepping it on the val monitor is test leakage when `data.val_frac` is unset (the monitor IS
the held-out test loader). So we step the plateau scheduler on the TRAINING loss instead:

    training.scheduler.name=reduce_on_plateau
    training.scheduler.monitor=train        <- no held-out data touches the LR schedule

Each run replays its ORIGINAL `.hydra/overrides.yaml` verbatim, with any `training.scheduler.*`
override stripped and the two above appended -- so the scheduler is the only thing that changes.

Excluded (nothing to re-train): the Analytical SDE variants (calibrated, not gradient-trained)
and the RF/XGB trees (no optimiser, hence no LR schedule).

Outputs land in outputs/ss_lrfix/<name>, leaving the original runs untouched.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
SEED = 42
OUT_ROOT = os.path.join(ROOT, "outputs", "ss_lrfix")

# (parent under outputs/, run-name template with {site}) -- every gradient-trained arm that
# appears in analysis/ss/manuscript_tables/stage1_all5.csv
CONFIGS = [
    # --- WienerNet-SS: likelihood axis ---
    ("ss_loso", "{site}_s42_ldiur_wien"),
    ("ss_loso", "{site}_s42_ldiur_wien_gauss"),
    ("ss_loso", "{site}_s42_ldiur_wien_beta"),
    ("ss_loso", "{site}_s42_ldiur_wien_studt"),
    ("ss_loso", "{site}_s42_ldiur_wien_mix"),
    # --- WienerNet-SS: other axes ---
    ("ss_loso", "{site}_s42_ldiur_res_wien"),
    ("ss_loso", "{site}_s42_ldiur_ss"),
    ("ss_loso", "{site}_s42_diur_wien"),
    ("ss_loso", "{site}_s42_diur_ss"),
    ("ss_loso", "{site}_s42_ldiur_wien_predk"),
    # --- matched-loss neural baselines ---
    ("ss_loso", "{site}_s42_base_nsde_studt"),
    ("ss_loso", "{site}_s42_base_nsde_ald"),
    ("ss_loso", "{site}_s42_base_mv_studt"),
    ("ss_loso", "{site}_s42_base_mv_ald"),
    ("final_loso", "comp_neuralsde_{site}_s42"),
    ("final_loso", "comp_meanvar_{site}_s42"),
    ("final_loso", "comp_mdn_{site}_s42"),
]
NEW_SCHED = ["training.scheduler.name=reduce_on_plateau", "training.scheduler.monitor=train"]


def read_overrides(run_dir: str) -> list[str] | None:
    p = os.path.join(run_dir, ".hydra", "overrides.yaml")
    if not os.path.exists(p):
        return None
    out = []
    for line in open(p):
        line = line.strip()
        if not line.startswith("- "):
            continue
        ov = line[2:].strip().strip("'\"")
        # drop anything we are replacing / that would fight the new run dir
        if ov.startswith("training.scheduler.") or ov.startswith("hydra."):
            continue
        out.append(ov)
    return out


def job(parent: str, tmpl: str, site: str, logdir: str, dry: bool) -> tuple[str, int]:
    name = tmpl.format(site=site)
    src = os.path.join(ROOT, "outputs", parent, name)
    ovr = read_overrides(src)
    if ovr is None:
        return (name, 127)
    dst = os.path.join(OUT_ROOT, name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), *ovr, *NEW_SCHED,
           f"hydra.run.dir={dst}"]
    if dry:
        print("DRY:", " ".join(cmd[2:])); return (name, 0)
    with open(os.path.join(logdir, f"{name}.log"), "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"  [{'ok ' if rc == 0 else 'FAIL'}] {name}", flush=True)
    return (name, rc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on the run name")
    a = ap.parse_args()

    logdir = os.path.join(ROOT, "scratchpad_sweep", "lrfix_logs")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    tasks = [(p, t, s) for p, t in CONFIGS for s in SITES
             if a.only is None or a.only in t.format(site=s)]
    print(f"{len(tasks)} runs ({len(CONFIGS)} configs x {len(SITES)} sites), "
          f"parallel={a.parallel}, seed={SEED} -> {OUT_ROOT}")

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        res = list(ex.map(lambda t: job(*t, logdir, a.dry_run), tasks))
    bad = [n for n, rc in res if rc != 0]
    print(f"\ndone: {len(res) - len(bad)}/{len(res)} ok" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
