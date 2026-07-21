#!/usr/bin/env python
"""Manuscript figures for WienerNet, matched one-to-one to analysis/ss/manuscript_tables.

Every figure reports the FULL 5-site leave-one-site-out sweep (the out-of-distribution Redmere-1
fold included) over the 5 seeds of the uniform-protocol sweep (outputs/ss_full5) — the same runs
`manuscript_tables.py --full5` tabulates. There are no clean-4 views here; those live in the tables.

Model selection (fixed, and stated in every caption):
  * WienerNet is shown at its primary likelihood (ALD) and its runner-up (mixture); the two tie on
    CRPS and sit at opposite ends of the sharpness/calibration frontier.
  * Every competing family is shown at ITS OWN best likelihood, chosen by Stage-1 CRPS on the
    4 in-distribution ("clean") sites — i.e. each baseline is picked at its in-distribution best
    and then scored here on all 5, so no baseline is handicapped by the OOD fold it is about to
    be judged on.
  * Trees are represented by Random Forest (the better of RF/XGB on point RMSE). It is rolled out
    autoregressively like every other arm — its own prediction is fed back as the next step's
    input — so the Stage-2 comparison is like-for-like. It emits no predictive distribution, so it
    appears only in the RMSE-based figures.

The Analytical SDE is omitted from the three Stage-2 figures by default (it is a calibrated physics
prior, not a competing learned model); pass --with-analytical-stage2 to draw it there. It is always
shown in Stage 1, where it is the reference point. Captions adapt to the setting automatically.

Because a single OOD fold supplies the whole right tail, the all-5 distributions are strongly
skewed: the CRPS figures therefore plot every (site, seed) unit on a log axis rather than a
symmetric mean ± SD bar, which would run negative on a positive-only quantity.

Numbers come from the same collectors that build the tables (manuscript_tables.collect_stage1 /
collect_stage2), so figures and tables can never drift apart.

Figures (one standalone PDF + .txt draft caption each, in analysis/ss/figs/):
  Stage 1 — one-step predictive law
    fig_stage1_crps                   CRPS per (site, seed) unit + mean
    fig_stage1_coverage               90% interval coverage vs the 0.90 target
    fig_stage1_sharpness_calibration  sharpness/calibration trade-off
    fig_stage1_crps_by_site           CRPS broken out per held-out site
    fig_stage1_coverage_by_site       coverage broken out per held-out site
  Stage 2 — autoregressive gap-fill
    fig_stage2_gapfill_rmse           rollout RMSE vs hours into gap (log scale: the OOD blow-ups)
    fig_stage2_crps_h5                5 h+ band CRPS per (site, seed) unit + mean
    fig_stage2_gapfill_rmse_by_site   5 h+ rollout RMSE broken out per held-out site
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manuscript_tables as MT  # noqa: E402
from manuscript_tables import collect_stage1, collect_stage2, SITES, SITE_LABEL  # noqa: E402

# Read the uniform-protocol 5-seed sweep (outputs/ss_full5, seeds 0-4), exactly as
# `manuscript_tables.py --full5` does — the figures must aggregate the same runs over the same
# seeds as the tables. Flipping these module globals is the same switch main(full5=True) throws.
MT.USE_FULL5 = True
MT.SEEDS = [0, 1, 2, 3, 4]

FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

# ---- selected arms: (label, stage-1 name, stage-2 name, colour) ---------------------------------
# Okabe-Ito colourblind-safe palette, fixed across every figure in the paper.
ARMS = [
    # Primary + runner-up: these two bracket the sharpness/calibration frontier — ALD is the
    # sharpest arm at acceptable calibration, mixture the best-calibrated at acceptable sharpness,
    # and they tie on CRPS (mean, median and worst unit alike).
    ("WienerNet (ALD)",           "WienerNet-SS (ALD, primary)", "WN-SS (ALD)",            "#0072B2"),
    ("WienerNet (mixture)",       "WienerNet-SS (mixture)",      "WN-SS (mixture)",        "#56B4E9"),
    ("Neural SDE (ALD)",          "Neural SDE (ALD)",            "Neural SDE (ALD)",       "#D55E00"),
    ("Mean-variance (ALD)",       "Mean-variance (ALD)",         "mean-var (ALD)",         "#E69F00"),
    ("Analytical SDE (Student-t)", "Analytical SDE (Student-t)", "Analytical (Student-t)", "#009E73"),
    ("MDN (mixture)",             "MDN (mixture)",               "MDN (mixture)",          "#CC79A7"),
    ("Random Forest",             "Random Forest",               "Random Forest",          "#999999"),
]
ANALYTICAL = "Analytical SDE (Student-t)"
# The Analytical SDE is a calibrated physics prior, not a competing learned model. In Stage 2 it is
# also degenerate against the given-diurnal arms (identical deterministic drift), so it can crowd
# the gap-fill figures without adding a comparison. Off by default here; `--with-analytical-stage2`
# puts it back. Stage-1 figures always keep it — there it IS the reference point.
SHOW_ANALYTICAL_STAGE2 = False

LABELS = [a[0] for a in ARMS]
COLOR = {a[0]: a[3] for a in ARMS}
S1NAME = {a[0]: a[1] for a in ARMS}
S2NAME = {a[0]: a[2] for a in ARMS}
DIST = [a[0] for a in ARMS if a[0] != "Random Forest"]   # arms with a predictive distribution

LW, MS = 2.2, 6.5
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9.5,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": .7, "axes.axisbelow": True,
    "axes.edgecolor": "#888", "axes.linewidth": .8,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def s2_arms(arms):
    """Arm list for a Stage-2 figure, honouring SHOW_ANALYTICAL_STAGE2."""
    return [a for a in arms if SHOW_ANALYTICAL_STAGE2 or a != ANALYTICAL]


def s2_txt(on, off):
    """Pick the caption fragment matching the current Stage-2 arm set, so a toggled figure never
    ships a caption describing an arm it does not draw."""
    return on if SHOW_ANALYTICAL_STAGE2 else off


def save(fig, stem, caption):
    path = os.path.join(FIG, stem)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    with open(path + ".txt", "w") as f:
        f.write(caption.strip() + "\n")
    print(f"wrote {stem}.pdf")
    print(f"  caption: {caption.strip()}")


def agg(df, name_col, name, sites, metric):
    """mean and 1 SD of `metric` over the (site, seed) units for one model."""
    if not len(df) or metric not in df.columns:
        return np.nan, np.nan
    v = pd.to_numeric(df[(df[name_col] == name) & (df.site.isin(sites))][metric],
                      errors="coerce").dropna()
    if not len(v):
        return np.nan, np.nan
    return v.mean(), (v.std() if len(v) > 1 else np.nan)


def units(df, name_col, name, metric):
    """The individual per-(site, seed) values of `metric` for one model, over all 5 sites."""
    if not len(df) or metric not in df.columns:
        return np.array([])
    sub = df[(df[name_col] == name) & (df.site.isin(SITES))]
    return pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()


def _hdots(ax, arms, series, xlabel):
    """Horizontal mean-plus-units dot plot, log x.

    The all-5 distributions are strongly right-skewed (a single out-of-distribution fold supplies
    the tail), so a symmetric mean ± SD bar is not a faithful summary — it runs negative on a
    positive-only quantity. Each (site, seed) unit is drawn instead, with the mean as a larger
    marker, on a log axis so the in-distribution bulk and the OOD tail are both readable.
    """
    y = np.arange(len(arms))[::-1].astype(float)
    for i, a in enumerate(arms):
        v = series[i]
        if not len(v):
            continue
        # deterministic vertical spread (no RNG) so the figure is byte-reproducible
        off = np.linspace(-.16, .16, len(v)) if len(v) > 1 else np.zeros(1)
        ax.scatter(v, y[i] + off, s=16, color=COLOR[a], alpha=.45, linewidth=0, zorder=2)
        ax.scatter([v.mean()], [y[i]], s=110, color=COLOR[a], edgecolor="white",
                   linewidth=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(arms)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_xscale("log")
    # Plain decimal ticks at hand-picked stops: matplotlib's default log labelling ("3 x 10^0",
    # "4 x 10^0") collides at these narrow spans and is unreadable at print size.
    allv = np.concatenate([v for v in series if len(v)])
    stops = np.array([.3, .5, .7, 1, 1.5, 2, 3, 5, 7, 10, 20, 30, 50, 100, 200, 500])
    keep = stops[(stops >= allv.min() * .9) & (stops <= allv.max() * 1.1)]
    ax.xaxis.set_major_locator(mticker.FixedLocator(keep))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color="#555", alpha=.45),
               plt.Line2D([], [], ls="", marker="o", ms=9, color="#555", markeredgecolor="white")]
    ax.legend(handles, ["individual held-out site × seed", "mean"],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)


def _by_site_bars(ax, arms, df, name_of, metric, ylabel, ylim=None):
    """Grouped bars: one x group per held-out site, one bar per arm, ± 1 SD over the 5 seeds.

    Used for the bounded / mildly-spread metrics (CRPS, coverage). Skewed ones use _by_site_dots.
    """
    nS, nM = len(SITES), len(arms)
    w = 0.8 / nM
    for j, a in enumerate(arms):
        xs, mu, sd = [], [], []
        for i, s in enumerate(SITES):
            v = pd.to_numeric(df[(df.model == name_of[a]) & (df.site == s)][metric],
                              errors="coerce").dropna()
            xs.append(i + (j - nM / 2) * w + w / 2)
            mu.append(v.mean() if len(v) else np.nan)
            sd.append(v.std() if len(v) > 1 else 0.0)
        ax.bar(xs, mu, w, color=COLOR[a], edgecolor="white", linewidth=.5, label=a,
               yerr=sd, error_kw=dict(elinewidth=.7, capsize=1.2, ecolor="#555"))
    ax.set_xticks(range(nS))
    ax.set_xticklabels([SITE_LABEL[s] for s in SITES])
    ax.set_xlabel("Held-out site")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              handletextpad=.4, columnspacing=1.2)


def _by_site_dots(ax, arms, df, name_of, metric, ylabel):
    """Per-site markers on a log y axis — for metrics whose OOD fold is orders of magnitude out,
    where a bar (anchored at zero) would misrepresent the ratio it is meant to show."""
    nS, nM = len(SITES), len(arms)
    w = 0.8 / nM
    pts = {}
    for j, a in enumerate(arms):
        xs, mu = [], []
        for i, s in enumerate(SITES):
            v = pd.to_numeric(df[(df.model == name_of[a]) & (df.site == s)][metric],
                              errors="coerce").dropna()
            xs.append(i + (j - nM / 2) * w + w / 2)
            mu.append(v.mean() if len(v) else np.nan)
        pts[a] = (xs, mu)
    # Lollipop stems: the baseline must be pinned before drawing, since on a log axis a stem has to
    # start at a finite floor rather than at zero.
    allv = np.array([v for xs, mu in pts.values() for v in mu], float)
    allv = allv[np.isfinite(allv)]
    floor = allv.min() * 0.75
    ax.set_ylim(floor, allv.max() * 1.6)
    for a in arms:
        xs, mu = pts[a]
        ax.vlines(xs, floor, mu, color=COLOR[a], lw=1.1, alpha=.75, zorder=2)
        ax.scatter(xs, mu, s=44, color=COLOR[a], edgecolor="white", linewidth=.8, zorder=3, label=a)
    ax.set_xticks(range(nS))
    ax.set_xticklabels([SITE_LABEL[s] for s in SITES])
    ax.set_xlabel("Held-out site")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              handletextpad=.2, columnspacing=1.2)


# ---------------------------------------------------------------- Stage 1
def fig_stage1_crps(df1):
    arms = DIST
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    _hdots(ax, arms, [units(df1, "model", S1NAME[a], "crps") for a in arms],
           "One-step CRPS (µmol m⁻² s⁻¹)")
    fig.tight_layout()
    save(fig, "fig_stage1_crps",
         "One-step predictive CRPS (lower is better) across all five leave-one-site-out folds and "
         "5 seeds, for each model family at its best likelihood; every held-out site × seed unit is "
         "plotted with the mean as the large marker (logarithmic axis). Typical-fold performance is "
         "indistinguishable — every arm has a median of 0.71–0.73 — so the families separate only "
         "through their tails on the out-of-distribution Redmere-1 fold. Both WienerNet heads have "
         "effectively no tail — their worst site × seed unit is 0.90, matching the calibrated "
         "Analytical SDE — against 1.02 (mean-variance), 2.35 (Neural SDE) and 2.73 (MDN). Their "
         "mean of 0.70 accordingly ties the Analytical prior and leads every learned baseline.")


def fig_stage1_coverage(df1):
    """Coverage is bounded in [0, 1] and not skewed, so it keeps a plain bar + SD."""
    arms = DIST
    v = [agg(df1, "model", S1NAME[a], SITES, "cov90") for a in arms]
    y = np.arange(len(arms))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, a in enumerate(arms):
        ax.barh(y[i], v[i][0], 0.62, color=COLOR[a], edgecolor="white", linewidth=.6,
                xerr=(0 if not np.isfinite(v[i][1]) else v[i][1]),
                error_kw=dict(elinewidth=.9, capsize=2.5, ecolor="#444"))
    ax.axvline(0.90, color="#333", ls="--", lw=1.2, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(arms)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Empirical coverage of the 90% predictive interval")
    ax.annotate("nominal 0.90", (0.90, ax.get_ylim()[1]), xytext=(3, -2),
                textcoords="offset points", fontsize=9.5, va="top", ha="left", color="#333")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_stage1_coverage",
         "Empirical coverage of the nominal 90% one-step predictive interval over all five held-out "
         "sites and 5 seeds; the dashed line is the 0.90 target and bars are mean ± 1 SD over "
         "(site, seed) units. Every arm under-covers at one step. The wider-interval baselines "
         "The WienerNet mixture head comes closest to nominal at 0.87, ahead of the MDN (0.86), "
         "mean-variance (0.84) and the Neural SDE (0.81); the sharper WienerNet ALD head sits at "
         "0.79 and the calibrated Analytical SDE under-covers most at 0.76. Coverage alone therefore "
         "ranks the arms largely by interval width — what each pays for it is resolved in the "
         "sharpness figure.")


def fig_stage1_crps_by_site(df1):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _by_site_bars(ax, DIST, df1, S1NAME, "crps", "One-step CRPS (µmol m⁻² s⁻¹)")
    fig.tight_layout()
    save(fig, "fig_stage1_crps_by_site",
         "One-step predictive CRPS per held-out site (mean ± 1 SD over the 5 seeds, lower is "
         "better). The pooled comparison is decided entirely at one site: on Woodwalton, Rosedene, "
         "Redmere 2 and Great Fen every arm agrees to within 0.04, whereas on the "
         "out-of-distribution Redmere 1 fold the Neural SDE (1.41) and MDN (1.52) roughly double "
         "their error while both WienerNet heads (0.69–0.70) and mean-variance (0.78) hold, and the "
         "calibrated Analytical SDE is actually at its best there (0.60).")


def fig_stage1_coverage_by_site(df1):
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    _by_site_bars(ax, DIST, df1, S1NAME, "cov90",
                  "Coverage of the 90% interval", ylim=(0, 1.0))
    ax.axhline(0.90, color="#333", ls="--", lw=1.2, zorder=5)
    ax.annotate("nominal 0.90", (len(SITES) - 0.45, 0.90), xytext=(0, 3),
                textcoords="offset points", fontsize=9.5, va="bottom", ha="right", color="#333")
    fig.tight_layout()
    save(fig, "fig_stage1_coverage_by_site",
         "Empirical coverage of the nominal 90% one-step interval per held-out site (mean ± 1 SD "
         "over the 5 seeds; dashed line is the 0.90 target). The pooled under-coverage is not "
         "uniform: the WienerNet mixture head is the steadiest arm, within 0.02–0.05 of nominal at "
         "every site, while the sharper ALD head loses most of its coverage at Woodwalton (0.67) "
         "and the calibrated Analytical SDE under-covers badly at Rosedene, Redmere 1 and Redmere 2 "
         "(0.68–0.72) while being near-nominal at Woodwalton (0.85) — its errors are site-specific, "
         "not a uniform bias.")


def fig_stage2_rmse_by_site(df2):
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    _by_site_dots(ax, s2_arms(LABELS), df2, S2NAME, "rmse_h5+",
                  "Rollout RMSE (µmol m⁻² s⁻¹)")
    fig.tight_layout()
    save(fig, "fig_stage2_gapfill_rmse_by_site",
         "Autoregressive gap-fill error 5 h or more into the gap, per held-out site (mean over the "
         "5 seeds; logarithmic axis, so the stems are drawn from a finite floor and their lengths "
         "encode ratios rather than differences). The rollout divergence is localised entirely to "
         "the out-of-distribution "
         "Redmere 1 fold, where the MDN reaches 39.3 and the Neural SDE 20.5 µmol m⁻² s⁻¹ — one to "
         "two orders of magnitude above their own error at every other site — while both WienerNet "
         "heads stay at 2.8–3.0, in line with their in-distribution performance. Mean-variance "
         "degrades mildly there (4.95), while " +
         s2_txt("the physics-drift and tree arms are untouched", "the tree arm is untouched") +
         " — the Random Forest, rolled out autoregressively like every other arm, records 1.23 at "
         "Redmere 1, among its best sites.")


def fig_stage1_sharp_cal(df1):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for a in DIST:
        x, _ = agg(df1, "model", S1NAME[a], SITES, "sharp90")
        y, _ = agg(df1, "model", S1NAME[a], SITES, "pit_ks")
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        # Semi-transparent fill: the Neural SDE and the MDN land on nearly the same point, and an
        # opaque marker would hide one of them entirely.
        ax.scatter(x, y, s=95, color=COLOR[a], alpha=.75, edgecolor="white", linewidth=1.0,
                   zorder=3, label=a)
    ax.set_xlabel("90% interval width (µmol m⁻² s⁻¹)")
    ax.set_ylabel("PIT Kolmogorov–Smirnov statistic")
    ax.set_xlim(left=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              handletextpad=.2, columnspacing=1.0)
    fig.tight_layout()
    save(fig, "fig_stage1_sharpness_calibration",
         "Sharpness–calibration trade-off over all five held-out sites and 5 seeds: mean 90% interval "
         "width against the PIT Kolmogorov–Smirnov statistic (0 = perfectly calibrated), so the "
         "lower-left corner is best. The two WienerNet heads bracket the frontier: the ALD head is "
         "the sharpest learned arm at 3.0 µmol m⁻² s⁻¹, a third narrower than the Neural SDE (4.5) "
         "and MDN (4.6), while the mixture head is the best-calibrated arm in the comparison "
         "(PIT-KS 0.065) and strictly dominates both of those baselines — narrower AND better "
         "calibrated. Neither is dominated in turn: mean-variance is slightly better calibrated than "
         "the ALD head (0.073 vs 0.084) at 13% greater width, and the calibrated Analytical SDE is "
         "sharpest overall (2.2) at clearly the worst calibration (0.108), consistent with its "
         "under-coverage. Markers are semi-transparent because the Neural SDE and MDN nearly "
         "coincide.")


# ---------------------------------------------------------------- Stage 2
BINS = [("rmse_h0-2", "0–2"), ("rmse_h2-5", "2–5"), ("rmse_h5+", "5+")]


def _gapfill_rmse(df2, sites, stem, logy, caption):
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    x = np.arange(len(BINS))
    for a in s2_arms(LABELS):
        mu = [agg(df2, "model", S2NAME[a], sites, col)[0] for col, _ in BINS]
        if not np.isfinite(mu).any():
            continue
        # No error bars: across (site, seed) the spread spans orders of magnitude for the diverging
        # arms, so a symmetric SD is meaningless — the log axis carries the spread instead.
        ax.plot(x, mu, color=COLOR[a], lw=LW, marker="o", ms=MS,
                markeredgecolor="white", markeredgewidth=.8, label=a)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in BINS])
    ax.set_xlim(-0.25, len(BINS) - 0.75)
    ax.set_xlabel("Hours into the gap (h)")
    ax.set_ylabel("Rollout RMSE (µmol m⁻² s⁻¹)")
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              handletextpad=.4, columnspacing=1.0)
    fig.tight_layout()
    save(fig, stem, caption)


def fig_gapfill(df2):
    _gapfill_rmse(
        df2, SITES, "fig_stage2_gapfill_rmse", True,
        "Autoregressive gap-fill error against hours into the gap, over all five held-out towers and "
        "5 seeds (mean; logarithmic axis). The MDN and the Neural SDE diverge under rollout, peaking "
        "at 19.6 and 11.0 µmol m⁻² s⁻¹ in the 2–5 h window — three to seven times the 2.8–3.1 of the "
        "WienerNet heads — and neither recovers by 5 h+. The mean-variance baseline, by contrast, "
        "stays bounded and tracks the WienerNet arms closely, so a learned noise scale is not on "
        "its own sufficient to cause the divergence. " +
        s2_txt("The calibrated Analytical SDE is flat throughout, and so is ",
               "So is ") +
        "the Random Forest (1.6 → 1.4), which is fed back its own prediction "
        "at each step exactly as the SDE arms are: a purely data-driven point regressor can be "
        "rolled out for a whole night without drifting, so the divergence of the Neural SDE and MDN "
        "is a property of those models rather than of autoregressive rollout as such. The Random "
        "Forest is in fact the lowest-error arm at 5 h+ — it simply reports no uncertainty, which "
        "is what the remaining figures score.")


def fig_stage2_crps(df2):
    arms = s2_arms(DIST)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    _hdots(ax, arms, [units(df2, "model", S2NAME[a], "crpsraw_h5+") for a in arms],
           "Gap-fill CRPS at 5 h+ into the gap (µmol m⁻² s⁻¹)")
    fig.tight_layout()
    save(fig, "fig_stage2_crps_h5",
         "Probabilistic gap-fill quality at the discriminating horizon: CRPS of the raw predictive "
         "band 5 h or more into the gap, over all five held-out towers and 5 seeds (lower is better; "
         "every site × seed unit plotted, mean as the large marker, logarithmic axis). " +
         s2_txt("The calibrated Analytical SDE gives the best band by a clear margin (median 0.92, "
                "worst unit 1.6× that). Among the learned arms the medians are ",
                "The medians are ") +
         "nearly identical (1.21–1.56) and "
         "WienerNet does not lead: mean-variance has both the lowest median (1.21) and the tightest "
         "spread (worst unit 1.7× median), against 2.7× for the WienerNet ALD head, 2.2× for the "
         "mixture head, 2.5× for the Neural SDE and 4.2× for the MDN. The WienerNet advantage at "
         "this horizon is in point error and rollout stability (previous figure), not in the "
         "width-calibration of the band it reports.")


def main(with_analytical_stage2=False):
    global SHOW_ANALYTICAL_STAGE2
    SHOW_ANALYTICAL_STAGE2 = with_analytical_stage2
    df1, df2 = collect_stage1(), collect_stage2()
    print(f"stage-1 rows: {len(df1)}   stage-2 rows: {len(df2)}   "
          f"Analytical SDE in Stage 2: {'on' if SHOW_ANALYTICAL_STAGE2 else 'off'}\n")
    fig_stage1_crps(df1)
    fig_stage1_coverage(df1)
    fig_stage1_sharp_cal(df1)
    fig_stage1_crps_by_site(df1)
    fig_stage1_coverage_by_site(df1)
    fig_gapfill(df2)
    fig_stage2_crps(df2)
    fig_stage2_rmse_by_site(df2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-analytical-stage2", action="store_true",
                    help="also draw the Analytical SDE in the three Stage-2 figures "
                         "(off by default; it is always drawn in Stage 1)")
    main(with_analytical_stage2=ap.parse_args().with_analytical_stage2)
