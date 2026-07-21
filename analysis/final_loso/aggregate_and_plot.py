#!/usr/bin/env python
"""Aggregate the leave-one-site-out sweep and render its figures.

Reads outputs/final_loso/<label>_<site>_s<seed>/, maps internal labels to display
names, and produces per-site probabilistic-skill figures with error bars over
seeds. Robust to partial results (skips missing runs). Colourblind-safe (Okabe-Ito).

Outputs -> analysis/final_loso/
  summary_long.csv        one row per (model, site): mean/sd over seeds, all metrics
  point_skill.csv         RMSE/R2 incl. tree baselines
  figures/fig_crps_by_site.(png|pdf)
  figures/fig_coverage_by_site.(png|pdf)
  figures/fig_ablation_ladder.(png|pdf)
  figures/fig_reliability.(png|pdf)
  figures/fig_pit_panels.(png|pdf)
  figures/fig_sharpness_calibration.(png|pdf)
"""
from __future__ import annotations
import json, os, glob, csv
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.path.join(ROOT, "outputs", "final_loso")
OUT = os.path.join(ROOT, "analysis", "final_loso")
FIG = os.path.join(OUT, "figures")
SITES = ["woodwalton", "redmere_1", "redmere_2", "great_fen", "rosedene"]
SITE_LABEL = {"woodwalton": "Woodwalton", "redmere_1": "Redmere 1", "redmere_2": "Redmere 2",
              "great_fen": "Great Fen", "rosedene": "Rosedene"}
SEEDS = [0, 1, 42]

# internal label -> (display name, group). Order = plot order.
MODELS = [
    ("wienernet_laplace",  "WienerNet (Laplace)",            "wienernet"),
    ("wienernet_mixture",  "WienerNet (Mixture)",            "wienernet"),
    ("abl_studentt",       "Student-t diffusion",            "ablation"),
    ("abl_gaussian",       "Gaussian diffusion",             "ablation"),
    ("abl_beta",           "β-NLL Gaussian",            "ablation"),
    ("abl_laplace_b0",     "Laplace (no variance match)",    "ablation"),
    ("abl_mmdnoise",       "MMD noise",                      "ablation"),
    ("abl_laplace_resid",  "Laplace + residual",             "ablation"),
    ("abl_mixture_resid",  "Mixture + residual",             "ablation"),
    ("abl_determin",       "Deterministic (no noise)",       "ablation"),
    ("prior_analytical",   "Analytical SDE",                 "reference"),
    ("prior_old_wienernet","Original WienerNet (MMD)",       "reference"),
    ("comp_mdn",           "Mixture-density net",            "nophysics"),
    ("comp_meanvar",       "Mean-variance net",              "nophysics"),
    ("comp_neuralsde",     "Neural SDE",                     "nophysics"),
    ("comp_rf",            "Random forest",                  "tree"),
    ("comp_xgb",           "XGBoost",                        "tree"),
]
DISPLAY = {k: v for k, v, _ in MODELS}
GROUP = {k: g for k, _, g in MODELS}
# Okabe-Ito colourblind-safe
GROUP_COLOR = {"wienernet": "#0072B2", "ablation": "#56B4E9", "reference": "#999999",
               "nophysics": "#D55E00", "tree": "#E69F00"}

# noise-family ablation ladder (increasing noise sophistication)
LADDER = ["abl_mmdnoise", "abl_gaussian", "abl_beta", "abl_studentt",
          "abl_laplace_b0", "wienernet_laplace", "wienernet_mixture"]

# model set used in the main figures
MAIN = ["wienernet_laplace", "wienernet_mixture", "comp_mdn", "comp_meanvar",
        "comp_neuralsde", "prior_analytical", "abl_determin"]


def read_prob(d):
    """Ensemble probabilistic metrics from a torch run (None if absent).

    Point predictors (deterministic / analytical with no spread) may have a null
    PIT/coverage block — keep CRPS, mark the calibration fields as missing.
    """
    p = os.path.join(d, "metrics", "probabilistic.json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))["global"]
    e = g.get("ensemble") or g.get("probabilistic")
    if e is None or e.get("crps") is None:
        return None
    pit = e.get("pit") or {}
    cov = e.get("coverage") or {}
    def _cov(level, field):
        c = cov.get(level) or {}
        return c.get(field)
    hist = pit.get("hist_counts")
    return dict(crps=e["crps"],
                cov90=_cov("0.90", "coverage"), cov95=_cov("0.95", "coverage"),
                pit_ks=pit.get("ks_uniform"),
                sharp90=_cov("0.90", "sharpness"),
                pit_hist=np.array(hist, float) if hist is not None else None)


def read_point(d):
    """Half-hourly RMSE/R2 on NEE — tree metrics.json or torch per_site.csv (long format)."""
    mj = os.path.join(d, "metrics.json")           # tree baselines
    if os.path.exists(mj):
        raw = json.load(open(mj)).get("raw", {}).get("nee")
        if raw:
            return dict(rmse=raw.get("rmse"), r2=raw.get("r2"), bias=raw.get("bias"))
    ps = os.path.join(d, "metrics", "per_site.csv")  # torch runs
    if os.path.exists(ps):
        vals = {}
        for row in csv.DictReader(open(ps)):
            if row.get("target") == "nee" and row.get("metric") in ("rmse", "r2", "bias"):
                vals[row["metric"]] = float(row["value"])
        if "rmse" in vals:
            return dict(rmse=vals.get("rmse"), r2=vals.get("r2"), bias=vals.get("bias"))
    return None


def collect():
    prob = defaultdict(lambda: defaultdict(list))   # [label][site] -> list of metric dicts
    point = defaultdict(lambda: defaultdict(list))
    for label, _, grp in MODELS:
        for site in SITES:
            for seed in SEEDS:
                d = os.path.join(SWEEP, f"{label}_{site}_s{seed}")
                if not os.path.isdir(d):
                    continue
                r = read_prob(d)
                if r:
                    prob[label][site].append(r)
                pt = read_point(d)
                if pt and pt.get("rmse") is not None:
                    point[label][site].append(pt)
    return prob, point


def agg(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std()), len(v)


def write_csvs(prob, point):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "site", "n", "crps", "crps_sd", "cov90", "cov90_sd",
                    "cov95", "pit_ks", "sharp90"])
        for label, _, _ in MODELS:
            for site in SITES:
                runs = prob[label][site]
                if not runs:
                    continue
                cm, cs, n = agg([r["crps"] for r in runs])
                v90m, v90s, _ = agg([r["cov90"] for r in runs])
                v95m, _, _ = agg([r["cov95"] for r in runs])
                ksm, _, _ = agg([r["pit_ks"] for r in runs])
                shm, _, _ = agg([r["sharp90"] for r in runs])
                w.writerow([DISPLAY[label], SITE_LABEL[site], n, f"{cm:.4f}", f"{cs:.4f}",
                            f"{v90m:.4f}", f"{v90s:.4f}", f"{v95m:.4f}", f"{ksm:.4f}", f"{shm:.4f}"])
    with open(os.path.join(OUT, "point_skill.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "site", "n", "rmse", "rmse_sd", "r2"])
        for label, _, _ in MODELS:
            for site in SITES:
                runs = point[label][site]
                if not runs:
                    continue
                rm, rs, n = agg([r["rmse"] for r in runs])
                r2m, _, _ = agg([r["r2"] for r in runs])
                w.writerow([DISPLAY[label], SITE_LABEL[site], n, f"{rm:.4f}", f"{rs:.4f}", f"{r2m:.4f}"])
    print("wrote summary_long.csv, point_skill.csv")


def headline_gaps():
    """Per-site paired CRPS gap (WienerNet head - MDN) using matched seeds.
    Writes headline_gaps.csv and prints it."""
    rows = []
    for head in ("wienernet_laplace", "wienernet_mixture"):
        for site in SITES:
            gaps = []
            for seed in SEEDS:
                a = read_prob(os.path.join(SWEEP, f"{head}_{site}_s{seed}"))
                b = read_prob(os.path.join(SWEEP, f"comp_mdn_{site}_s{seed}"))
                if a and b:
                    gaps.append(a["crps"] - b["crps"])
            if not gaps:
                continue
            g = np.array(gaps)
            all_neg = bool(np.all(g < 0)); all_pos = bool(np.all(g > 0))
            verdict = "physics wins (all seeds)" if all_neg else \
                      ("black box wins (all seeds)" if all_pos else "mixed/tie")
            rows.append(dict(head=DISPLAY[head], site=SITE_LABEL[site], n=len(g),
                             mean_gap=float(g.mean()), sd=float(g.std()), verdict=verdict))
    if rows:
        with open(os.path.join(OUT, "headline_gaps.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["head", "site", "n", "mean_gap", "sd", "verdict"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "mean_gap": f"{r['mean_gap']:.4f}", "sd": f"{r['sd']:.4f}"})
        print("\n=== headline paired CRPS gap (WienerNet head − MDN), matched seeds ===")
        for r in rows:
            print(f"  {r['head']:<22} {r['site']:<12} gap {r['mean_gap']:+.3f}±{r['sd']:.3f}  {r['verdict']}")
    return rows


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 220, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans", "figure.constrained_layout.use": True,
    })
    return plt


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    print("wrote", name)


def fig_metric_by_site(prob, metric, ylabel, title, name, nominal=None, order=None, ycap=None,
                       legend_loc="upper left"):
    plt = setup_mpl()
    labels = order or [l for l, _, _ in MODELS if any(prob[l][s] for s in SITES)]
    if not labels:
        print(f"skip {name}: no data yet"); return
    fig, ax = plt.subplots(figsize=(13, 5.2))
    n = len(labels); w = 0.8 / n
    x = np.arange(len(SITES))
    for i, label in enumerate(labels):
        means, sds, capped = [], [], []
        for site in SITES:
            m, s, _ = agg([r[metric] for r in prob[label][site]])
            over = ycap is not None and not np.isnan(m) and m > ycap
            capped.append((over, m))
            means.append(min(m, ycap) if over else m)
            sds.append(0 if over else s)                # hide runaway error bars on capped bars
        ax.bar(x + i * w, means, w, yerr=sds, capsize=2, label=DISPLAY[label],
               color=GROUP_COLOR[GROUP[label]], edgecolor="white", linewidth=0.4,
               error_kw=dict(lw=0.8, alpha=0.7))
        for xi, (over, mval) in zip(x + i * w, capped):
            if over:
                ax.text(xi, ycap, f"{mval:.0f}", ha="center", va="bottom", fontsize=6.5,
                        rotation=90, color="#333")
    if nominal is not None:
        ax.axhline(nominal, ls="--", c="k", lw=1.1, zorder=0)
        ax.text(len(SITES) - 0.5, nominal, f" nominal {nominal:g}", va="bottom", ha="right", fontsize=9)
    if ycap is not None:
        ax.set_ylim(0, ycap * 1.08)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels([SITE_LABEL[s] for s in SITES])
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xlabel("held-out site")
    anchor = (1.0, 1.0) if legend_loc == "upper right" else (0, 1.0)
    ax.legend(ncol=3, fontsize=8, framealpha=0.9, loc=legend_loc, bbox_to_anchor=anchor)
    cap_note = f"  bars above {ycap:g} clipped (value shown)" if ycap else ""
    fig.text(0.995, 0.01, "error bars: ±1 s.d. over 3 seeds" + cap_note, ha="right", fontsize=8, color="#555")
    _save(fig, name)


def fig_full_ranked(prob, metric="crps", nominal=None,
                    title="All models — ensemble CRPS per held-out site",
                    xlabel="ensemble CRPS (lower better)", name="fig_crps_full_ranked"):
    """Faceted: one panel per site, horizontal bars for ALL models sorted best-first."""
    plt = setup_mpl()
    import matplotlib.patches as mp
    # fixed, grouped order (WienerNet -> ablation -> reference -> no-physics), best group on top
    order = [l for l, _, _ in MODELS if GROUP[l] != "tree" and any(prob[l][s] for s in SITES)]
    if not order:
        print(f"skip {name}: no data yet"); return
    order = order[::-1]                          # so WienerNet ends up at the top row
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, len(SITES), figsize=(3.5 * len(SITES) + 2.2, 6.6), sharey=True)
    fig.set_constrained_layout(False)
    if len(SITES) == 1:
        axes = [axes]
    for j, (ax, site) in enumerate(zip(axes, SITES)):
        means = [agg([r[metric] for r in prob[l][site]])[0] for l in order]
        sds = [agg([r[metric] for r in prob[l][site]])[1] for l in order]
        ax.barh(y, means, xerr=sds, capsize=2,
                color=[GROUP_COLOR[GROUP[l]] for l in order], edgecolor="white",
                linewidth=0.4, error_kw=dict(lw=0.7, alpha=0.7))
        if nominal is not None:
            ax.axvline(nominal, ls="--", c="k", lw=1)
        vmax = np.nanmax(means) if means else 1
        if metric == "crps" and vmax > 6:
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlim(left=0)
            ax.set_xlabel(xlabel + " (log)", fontsize=8.5)
            # annotate the out-of-range values
            for yi, mv in zip(y, means):
                if mv and mv > 6:
                    ax.text(mv, yi, f" {mv:.0f}", va="center", fontsize=7, color="#333")
        else:
            ax.set_xlim(left=0)
            ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_title(SITE_LABEL[site], fontsize=11.5)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(y); axes[0].set_yticklabels([DISPLAY[l] for l in order], fontsize=8.5)
    handles = [mp.Patch(color=c, label=g) for g, c in
               [("WienerNet (ours)", GROUP_COLOR["wienernet"]), ("physics ablation", GROUP_COLOR["ablation"]),
                ("physics reference", GROUP_COLOR["reference"]), ("no-physics baseline", GROUP_COLOR["nophysics"])]]
    fig.suptitle(title, fontsize=14, y=0.985)
    fig.legend(handles=handles, ncol=4, loc="upper center", fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.text(0.995, 0.005, "error bars: ±1 s.d. over 3 seeds", ha="right", fontsize=8, color="#555")
    fig.subplots_adjust(left=0.16, right=0.99, top=0.87, bottom=0.09, wspace=0.12)
    _save(fig, name)


def fig_ablation_ladder(prob):
    """Noise-family ladder: CRPS + cov90 vs increasing noise sophistication, averaged over sites."""
    plt = setup_mpl()
    labels = [l for l in LADDER if any(prob[l][s] for s in SITES)]
    if not labels:
        print("skip fig_ablation_ladder: no data yet"); return
    xs = np.arange(len(labels))
    crps_m, crps_s, cov_m, cov_s = [], [], [], []
    for l in labels:
        cm, cs, _ = agg([r["crps"] for s in SITES for r in prob[l][s]])
        vm, vs, _ = agg([r["cov90"] for s in SITES for r in prob[l][s]])
        crps_m.append(cm); crps_s.append(cs); cov_m.append(vm); cov_s.append(vs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.errorbar(xs, crps_m, yerr=crps_s, marker="o", lw=2, capsize=3, color="#0072B2")
    a1.set_xticks(xs); a1.set_xticklabels([DISPLAY[l] for l in labels], rotation=30, ha="right")
    a1.set_ylabel("ensemble CRPS (lower better)"); a1.set_title("Noise-family ablation — skill")
    a2.errorbar(xs, cov_m, yerr=cov_s, marker="s", lw=2, capsize=3, color="#0072B2")
    a2.axhline(0.90, ls="--", c="k", lw=1); a2.text(0, 0.90, " nominal 0.90", va="bottom", fontsize=9)
    a2.set_xticks(xs); a2.set_xticklabels([DISPLAY[l] for l in labels], rotation=30, ha="right")
    a2.set_ylabel("90% coverage"); a2.set_title("Noise-family ablation — calibration")
    fig.suptitle("Increasing noise-model sophistication (averaged over held-out sites)", fontsize=12)
    _save(fig, "fig_ablation_ladder")


def fig_reliability(prob, sites=("woodwalton", "redmere_1")):
    """Reliability curve from the pooled-seed PIT histogram (empirical CDF of PIT vs nominal)."""
    plt = setup_mpl()
    key_models = ["wienernet_laplace", "wienernet_mixture", "comp_mdn", "abl_gaussian", "abl_determin"]
    fig, axes = plt.subplots(1, len(sites), figsize=(6.2 * len(sites), 5))
    if len(sites) == 1:
        axes = [axes]
    for ax, site in zip(axes, sites):
        ax.plot([0, 1], [0, 1], ls="--", c="k", lw=1, label="ideal")
        for label in key_models:
            runs = [r for r in prob[label][site] if r.get("pit_hist") is not None]
            if not runs:
                continue
            h = np.sum([r["pit_hist"] for r in runs], axis=0)
            cdf = np.cumsum(h) / h.sum()
            edges = np.linspace(0, 1, len(h) + 1)[1:]
            ax.plot(edges, cdf, marker=".", lw=1.6, label=DISPLAY[label])
        ax.set_title(SITE_LABEL[site]); ax.set_xlabel("nominal probability")
        ax.set_ylabel("empirical (PIT ≤ p)"); ax.set_aspect("equal")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Reliability (PIT calibration) on held-out sites", fontsize=12)
    _save(fig, "fig_reliability")


def fig_sharpness_calibration(prob, sharp_cap=10.0):
    """Scatter: sharpness (mean 90% interval width) vs |cov90 - 0.90|, averaged over sites.
    Models with mean interval width above `sharp_cap` are listed off-chart rather than
    stretching the axis. Labels de-conflicted by nudging."""
    plt = setup_mpl()
    pts, off = [], []
    for label, disp, grp in MODELS:
        if grp == "tree":
            continue
        sh, _, _ = agg([r["sharp90"] for s in SITES for r in prob[label][s]])
        cv, _, _ = agg([r["cov90"] for s in SITES for r in prob[label][s]])
        if np.isnan(sh) or np.isnan(cv):
            continue
        (off if sh > sharp_cap else pts).append((disp, grp, sh, abs(cv - 0.90)))
    if not pts:
        print("skip fig_sharpness_calibration: no data"); return
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    xs = np.array([p[2] for p in pts]); ys = np.array([p[3] for p in pts])
    for disp, grp, x, y in pts:
        ax.scatter(x, y, s=90, color=GROUP_COLOR[grp], edgecolor="k", linewidth=0.6, zorder=3)
    # simple label de-confliction: nudge in data coords over a few iterations
    xr = (xs.max() - xs.min()) or 1; yr = (ys.max() - ys.min()) or 1
    lx = xs + 0.012 * xr; ly = ys + 0.012 * yr
    for _ in range(200):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = (lx[i] - lx[j]) / xr; dy = (ly[i] - ly[j]) / yr
                d = (dx * dx + dy * dy) ** 0.5
                if d < 0.12:
                    push = (0.12 - d) / 2
                    ux, uy = (dx / (d + 1e-9)), (dy / (d + 1e-9))
                    lx[i] += ux * push * xr; ly[i] += uy * push * yr
                    lx[j] -= ux * push * xr; ly[j] -= uy * push * yr
    for (disp, grp, x, y), lxi, lyi in zip(pts, lx, ly):
        ax.annotate(disp, (x, y), xytext=(lxi, lyi), fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#888"))
    ax.set_xlim(-0.3, sharp_cap); ax.set_ylim(-0.02, max(0.2, ys.max() + 0.03))
    ax.set_xlabel("sharpness — mean 90% interval width (smaller = sharper)")
    ax.set_ylabel("miscalibration  |coverage − 0.90|  (smaller = better)")
    ax.set_title("Sharpness vs calibration (averaged over held-out sites)\nbottom-left = sharp AND calibrated")
    if off:
        note = "off-chart (interval width > %g):\n" % sharp_cap + \
               "\n".join(f"  {d} — width {sh:.0f}" for d, _, sh, _ in sorted(off, key=lambda t: -t[2]))
        ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#B33", bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))
    import matplotlib.patches as mp
    handles = [mp.Patch(color=c, label=g) for g, c in
               [("WienerNet", GROUP_COLOR["wienernet"]), ("physics ablation", GROUP_COLOR["ablation"]),
                ("physics reference", GROUP_COLOR["reference"]), ("no-physics", GROUP_COLOR["nophysics"])]]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    _save(fig, "fig_sharpness_calibration")


def main():
    prob, point = collect()
    have = sum(1 for l, _, _ in MODELS for s in SITES if prob[l][s])
    print(f"collected probabilistic runs for {have} (model,site) cells")
    write_csvs(prob, point)
    headline_gaps()
    try:
        # grouped-bar figures
        fig_metric_by_site(prob, "crps", "ensemble CRPS (lower better)",
                           "Predictive skill on held-out sites (headline models)",
                           "fig_crps_by_site", order=[l for l in MAIN if any(prob[l][s] for s in SITES)],
                           ycap=1.5, legend_loc="upper right")
        fig_metric_by_site(prob, "cov90", "90% interval coverage",
                           "Calibration on held-out sites (headline models)", "fig_coverage_by_site",
                           nominal=0.90, order=[l for l in MAIN if any(prob[l][s] for s in SITES)])
        # full-roster ranked panels (all models, per site)
        fig_full_ranked(prob, "crps", None,
                        "All models — ensemble CRPS per held-out site", "ensemble CRPS (lower better)",
                        "fig_crps_full_ranked")
        fig_full_ranked(prob, "cov90", 0.90,
                        "All models — 90% coverage per held-out site (dashed = nominal)",
                        "90% interval coverage", "fig_coverage_full_ranked")
        fig_ablation_ladder(prob)
        fig_reliability(prob)
        fig_sharpness_calibration(prob)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print("figure error:", ex)


if __name__ == "__main__":
    main()
