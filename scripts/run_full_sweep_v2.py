#!/usr/bin/env python
"""FULL manuscript sweep v2 — every model/ablation x 5 sites x 5 seeds, one uniform protocol.

Why this exists
---------------
The v1 results mixed protocols: seeds 0/1 trained at a constant LR, seed 42 was later re-trained
with `reduce_on_plateau` after it was found to diverge. Any `±` over that mix is meaningless. This
sweep re-runs EVERYTHING under one schedule so the seed spread is a real seed spread.

Protocol (identical for every gradient-trained arm)
--------------------------------------------------
    training.scheduler.name=reduce_on_plateau
    training.scheduler.monitor=val          <- the val monitor IS the held-out test site
    data.val_frac=null                      <- pinned: NO split is carved from the training sites

`data.val_frac=null` is the point, not an oversight: with it unset `build_dataloaders` returns the
held-out-site test loader as the val loader, so "validation" means the left-out site, exactly as
requested. This is knowingly test-informed model selection (an LR schedule that reacts to the
held-out loss), and every number produced here must be reported as such.

Roster: the 22 configurations behind analysis/ss/manuscript_tables/. Each one's overrides are read
verbatim from its existing seed-0 run's `.hydra/overrides.yaml` and replayed with the seed swapped,
so nothing but the seed and the LR schedule differs from the published runs.

Seed handling: the Analytical SDE arms are calibrated, not gradient-trained, so they are
seed-invariant and run at seed 0 only. Trees are seeded (bootstrap / column sampling) and run at
all 5. Resume-safe: an arm whose expected output already exists is skipped.

    python scripts/run_full_sweep_v2.py --parallel 8
    python scripts/run_full_sweep_v2.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = ["woodwalton", "rosedene", "redmere_1", "redmere_2", "great_fen"]
SEEDS = [0, 1, 2, 3, 4]
OUT_ROOT = os.path.join(ROOT, "outputs", "ss_full5")

# kind: 'torch' -> train.py + evaluate.py ; 'tree' -> train_baseline.py ; '*_1seed' -> seed 0 only
# (template-parent, name template with {site}/{seed}, kind)
CONFIGS = [
    # --- WienerNet-SS: likelihood axis ---
    ("ss_loso", "{site}_s{seed}_ldiur_wien",        "torch"),
    ("ss_loso", "{site}_s{seed}_ldiur_wien_gauss",  "torch"),
    ("ss_loso", "{site}_s{seed}_ldiur_wien_beta",   "torch"),
    ("ss_loso", "{site}_s{seed}_ldiur_wien_studt",  "torch"),
    ("ss_loso", "{site}_s{seed}_ldiur_wien_mix",    "torch"),
    # --- WienerNet-SS: other ablation axes ---
    ("ss_loso", "{site}_s{seed}_ldiur_res_wien",    "torch"),
    ("ss_loso", "{site}_s{seed}_ldiur_res_ss",      "torch"),
    ("ss_loso", "{site}_s{seed}_ldiur_ss",          "torch"),
    ("ss_loso", "{site}_s{seed}_diur_wien",         "torch"),
    ("ss_loso", "{site}_s{seed}_diur_ss",           "torch"),
    ("ss_loso", "{site}_s{seed}_ldiur_wien_predk",  "torch"),
    # --- matched-loss neural baselines ---
    ("ss_loso", "{site}_s{seed}_base_nsde_studt",   "torch"),
    ("ss_loso", "{site}_s{seed}_base_nsde_ald",     "torch"),
    ("ss_loso", "{site}_s{seed}_base_mv_studt",     "torch"),
    ("ss_loso", "{site}_s{seed}_base_mv_ald",       "torch"),
    ("final_loso", "comp_neuralsde_{site}_s{seed}", "torch"),
    ("final_loso", "comp_meanvar_{site}_s{seed}",   "torch"),
    ("final_loso", "comp_mdn_{site}_s{seed}",       "torch"),
    # --- calibrated physics references: seed-invariant ---
    ("ss_loso", "{site}_s{seed}_base_analyt_studt",  "torch_1seed"),
    ("final_loso", "prior_analytical_{site}_s{seed}", "torch_1seed"),
    # --- tree point baselines (no optimiser, hence no LR schedule) ---
    ("final_loso", "comp_rf_{site}_s{seed}",  "tree"),
    ("final_loso", "comp_xgb_{site}_s{seed}", "tree"),
]

NEW_SCHED = [
    "training.scheduler.name=reduce_on_plateau",
    "training.scheduler.monitor=val",
    "data.val_frac=null",          # explicit: val loader == held-out test site, not a train split
]


def read_overrides(run_dir: str) -> list[str] | None:
    """Replay a run's Hydra overrides, dropping the ones this sweep re-specifies."""
    p = os.path.join(run_dir, ".hydra", "overrides.yaml")
    if not os.path.exists(p):
        return None
    out = []
    for line in open(p):
        line = line.strip()
        if not line.startswith("- "):
            continue
        ov = line[2:].strip().strip("'\"")
        if ov.startswith(("training.scheduler.", "data.val_frac", "hydra.", "seed=")):
            continue
        out.append(ov)
    return out


def done(dst: str, kind: str) -> bool:
    if kind == "tree":
        return os.path.exists(os.path.join(dst, "metrics.json"))
    return os.path.exists(os.path.join(dst, "metrics", "probabilistic.json"))


def job(parent: str, tmpl: str, kind: str, site: str, seed: int,
        logdir: str, dry: bool, force: bool) -> tuple[str, int]:
    name = tmpl.format(site=site, seed=seed)
    src = os.path.join(ROOT, "outputs", parent, tmpl.format(site=site, seed=0))
    ovr = read_overrides(src)
    if ovr is None:
        print(f"  [MISS] no template overrides for {name} (looked in {src})", flush=True)
        return (name, 127)
    dst = os.path.join(OUT_ROOT, name)
    if not force and done(dst, kind):
        print(f"  [skip] {name}", flush=True)
        return (name, 0)

    entry = "train_baseline.py" if kind == "tree" else "train.py"
    sched = [] if kind == "tree" else NEW_SCHED
    cmd = [sys.executable, os.path.join(ROOT, "scripts", entry), *ovr, *sched,
           f"seed={seed}", f"hydra.run.dir={dst}"]
    if dry:
        print("DRY:", name, "|", " ".join(cmd[2:]))
        return (name, 0)

    with open(os.path.join(logdir, f"{name}.log"), "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0 and kind != "tree":
            rc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"),
                                 "--run", dst, "--checkpoint", "last"],
                                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"  [{'ok  ' if rc == 0 else 'FAIL'}] {name}", flush=True)
    return (name, rc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    ap.add_argument("--only", default=None, help="substring filter on the run name")
    a = ap.parse_args()

    logdir = os.path.join(ROOT, "scratchpad_sweep", "full5_logs")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    tasks = []
    for parent, tmpl, kind in CONFIGS:
        seeds = [0] if kind.endswith("_1seed") else SEEDS
        for site in SITES:
            for seed in seeds:
                if a.only and a.only not in tmpl.format(site=site, seed=seed):
                    continue
                tasks.append((parent, tmpl, kind, site, seed))

    print(f"{len(tasks)} runs ({len(CONFIGS)} configs x {len(SITES)} sites x <=5 seeds), "
          f"parallel={a.parallel} -> {OUT_ROOT}", flush=True)

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        res = list(ex.map(lambda t: job(*t, logdir, a.dry_run, a.force), tasks))
    bad = [n for n, rc in res if rc != 0]
    print(f"\ndone: {len(res) - len(bad)}/{len(res)} ok" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
