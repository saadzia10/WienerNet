#!/usr/bin/env python
"""Aggregate the leave-one-site-out terminal-noise sweep.

Compares the increment SDE with each terminal noise head — ALD (beta=0.5) and the
3-component Gaussian mixture, each with and without the residual drift head — against
the no-physics MDN baseline, per held-out site, over seeds. Reads the ensemble
(100-pass) scores from each run's metrics/probabilistic.json + calibration.json.

Outputs:
  analysis/holdout_noise/loso_noise_summary.csv   — per (model, site) mean+/-std
  analysis/holdout_noise/figures/loso_noise_crps.png
  analysis/holdout_noise/figures/loso_noise_cov90.png
"""
from __future__ import annotations
import json, os, glob
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEW = os.path.join(ROOT, "outputs", "loso_noise")
OLD = os.path.join(ROOT, "outputs", "loso")
SITES = ["woodwalton", "redmere_1", "redmere_2", "great_fen", "rosedene"]
SEEDS = [0, 1, 42]

# label -> (directory root, dirname template). {site},{seed} filled per run.
MODELS = {
    "ald_B":        (NEW, "ald_B_{site}_s{seed}"),
    "mix_B":        (NEW, "mix_B_{site}_s{seed}"),
    "ald_A(resid)": (NEW, "ald_A_{site}_s{seed}"),
    "mix_A(resid)": (NEW, "mix_A_{site}_s{seed}"),
    "MDN(noPhys)":  (OLD, "hetero_mdn_{site}_s{seed}"),
    "StudentT":     (OLD, "increment_B_gtk_{site}_s{seed}"),
}


def read_run(d):
    p = os.path.join(d, "metrics", "probabilistic.json")
    c = os.path.join(d, "metrics", "calibration.json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))["global"]
    e = g.get("ensemble") or g.get("probabilistic")
    out = {
        "crps": e["crps"],
        "cov90": e["coverage"]["0.90"]["coverage"],
        "cov95": e["coverage"]["0.95"]["coverage"],
        "pit_ks": e["pit"]["ks_uniform"],
    }
    if os.path.exists(c):
        out["calibR"] = json.load(open(c)).get("calib_std_ratio")
    return out


def main():
    # collect: agg[(label, site)] = list over seeds of metric dicts
    agg = defaultdict(list)
    for label, (root, tmpl) in MODELS.items():
        for site in SITES:
            for seed in SEEDS:
                d = os.path.join(root, tmpl.format(site=site, seed=seed))
                r = read_run(d)
                if r is not None:
                    agg[(label, site)].append(r)

    rows = []
    for label in MODELS:
        for site in SITES:
            runs = agg[(label, site)]
            if not runs:
                continue
            def ms(key):
                v = np.array([r[key] for r in runs if r.get(key) is not None], float)
                return (float(v.mean()), float(v.std()), len(v)) if len(v) else (np.nan, np.nan, 0)
            crps_m, crps_s, n = ms("crps")
            cov_m, cov_s, _ = ms("cov90")
            c95_m, _, _ = ms("cov95")
            ks_m, _, _ = ms("pit_ks")
            rows.append(dict(model=label, site=site, n=n, crps=crps_m, crps_sd=crps_s,
                             cov90=cov_m, cov90_sd=cov_s, cov95=c95_m, pit_ks=ks_m))

    # write CSV
    import csv
    csv_path = os.path.join(ROOT, "analysis", "holdout_noise", "loso_noise_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "site", "n", "crps", "crps_sd",
                                          "cov90", "cov90_sd", "cov95", "pit_ks"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", csv_path)

    # console table: CRPS (mean) per model x site, + physics-minus-MDN gap
    labels = [l for l in MODELS if any((l, s) in agg for s in SITES)]
    print(f"\n=== ensemble CRPS (mean over seeds) ===")
    hdr = f"{'model':<14}" + "".join(f"{s[:9]:>11}" for s in SITES)
    print(hdr)
    tab = {}
    for label in labels:
        line = f"{label:<14}"
        for site in SITES:
            runs = agg[(label, site)]
            if runs:
                m = np.mean([r["crps"] for r in runs]); tab[(label, site)] = m
                line += f"{m:>11.3f}"
            else:
                line += f"{'-':>11}"
        print(line)
    print(f"\n=== cov90 (mean over seeds; nominal 0.90) ===")
    print(hdr)
    for label in labels:
        line = f"{label:<14}"
        for site in SITES:
            runs = agg[(label, site)]
            line += f"{np.mean([r['cov90'] for r in runs]):>11.3f}" if runs else f"{'-':>11}"
        print(line)

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figdir = os.path.join(ROOT, "analysis", "holdout_noise", "figures")
        os.makedirs(figdir, exist_ok=True)
        plot_labels = [l for l in labels]
        for metric, nominal, fname in [("crps", None, "loso_noise_crps.png"),
                                       ("cov90", 0.90, "loso_noise_cov90.png")]:
            fig, ax = plt.subplots(figsize=(11, 5))
            x = np.arange(len(SITES)); wdt = 0.8 / len(plot_labels)
            for i, label in enumerate(plot_labels):
                vals = [np.mean([r[metric] for r in agg[(label, s)]]) if agg[(label, s)] else np.nan
                        for s in SITES]
                ax.bar(x + i * wdt, vals, wdt, label=label)
            if nominal is not None:
                ax.axhline(nominal, ls="--", c="k", lw=1, label=f"nominal {nominal}")
            ax.set_xticks(x + 0.4 - wdt / 2); ax.set_xticklabels(SITES, rotation=15)
            ax.set_ylabel(metric); ax.set_title(f"LOSO terminal-noise sweep — {metric}")
            ax.legend(fontsize=8, ncol=3)
            fig.tight_layout(); fig.savefig(os.path.join(figdir, fname), dpi=130)
            print("wrote", os.path.join(figdir, fname))
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
