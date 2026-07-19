"""Manuscript-grade illustrations of the probabilistic / process-consistency
metrics.

Every figure is driven from the JSON reports written by
`scripts/evaluate.py::write_distribution_reports` (so it is reproducible without
re-running inference) and uses **descriptive** labels — full site names and
plain-language model names, never code identifiers — for publication.

Figures (only the high-value ones):
  * PIT / rank histograms          — calibration shape (per site grid)
  * coverage reliability diagram   — calibration magnitude (global + per site)
  * increment-variance scaling     — diffusion consistency (observed vs model)
  * standardized-residual ACF      — whiteness of the one-step residual
  * predictive skill by site       — CRPS per site (lower is better)

Colour palette is the colourblind-safe Okabe-Ito set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Okabe-Ito colourblind-safe palette
C_OBSERVED = "#111111"
C_MODEL = "#0072B2"
C_REFERENCE = "#8a8a8a"
C_ACCENT = "#D55E00"
C_BAND = "#009E73"
_IDEAL_PIT_VAR = 1.0 / 12.0

# Descriptive names for publication labels ------------------------------------
_SITE_NAMES = {
    "great_fen": "Great Fen", "rosedene": "Rosedene",
    "redmere_1": "Redmere 1", "redmere_2": "Redmere 2",
    "wicken_fen": "Wicken Fen", "woodwalton": "Woodwalton",
}
_MODEL_NAMES = {
    "piae_sde_sampling": "Physics-informed SDE (sampled noise)",
    "piae_sde_reg_sampling": "Physics-informed SDE with residual (sampled noise)",
    "pivae_sde_sampling": "Physics-informed VAE–SDE (sampled noise)",
    "piae_increment": "Increment SDE (physics drift)",
    "piae_increment_residual": "Increment SDE (physics drift + residual)",
    "piae_reg_increment": "Increment SDE (deterministic drift)",
    "nll_gaussian": "Increment SDE (Gaussian noise)",
    "nll_beta": "Increment SDE (β-NLL noise)",
    "nll_student_t": "Increment SDE (Student-t noise)",
    "ae": "Autoencoder (deterministic baseline)",
    "vae": "Variational autoencoder",
}


def descriptive_site(slug: str) -> str:
    return _SITE_NAMES.get(str(slug), str(slug).replace("_", " ").title())


def descriptive_model(slug: str) -> str:
    return _MODEL_NAMES.get(str(slug), str(slug).replace("_", " ").title())


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
        "axes.axisbelow": True, "axes.edgecolor": "#888", "axes.linewidth": 0.8,
        "figure.dpi": 120,
    })


def _save(fig: Figure, save_path: str | Path | None, dpi: int = 300) -> None:
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=dpi)


def _preferred_scores(block: dict) -> dict | None:
    """Prefer the honest ensemble score set when present, else the parametric one."""
    if not block:
        return None
    if "ensemble" in block and block["ensemble"]:
        return block["ensemble"]
    return block.get("probabilistic")


def _dispersion_label(var: float) -> str:
    if var != var:  # nan
        return ""
    if var > _IDEAL_PIT_VAR * 1.15:
        return "under-dispersed"
    if var < _IDEAL_PIT_VAR * 0.87:
        return "over-dispersed"
    return "well calibrated"


# ---------------------------------------------------------------------------
# 1. PIT / rank histogram — calibration shape
# ---------------------------------------------------------------------------

def plot_pit_histograms(
    panels: Mapping[str, dict], *, title: str | None = None,
    save_path: str | Path | None = None, ncols: int = 3,
) -> Figure:
    """Grid of PIT histograms. `panels` maps a descriptive panel label to a PIT
    summary dict (with hist_counts / hist_edges / var). A calibrated model gives a
    flat histogram at density 1; a ∪-shape means the predictive spread is too
    narrow (over-confident), a dome means too wide.
    """
    _apply_style()
    items = [(k, v) for k, v in panels.items() if v and v.get("hist_counts")]
    n = len(items)
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (label, s) in enumerate(items):
        ax = axes.flat[i]; ax.set_visible(True)
        counts = np.asarray(s["hist_counts"], dtype=float)
        edges = np.asarray(s["hist_edges"], dtype=float)
        total = counts.sum()
        widths = np.diff(edges)
        density = counts / total / widths if total > 0 else counts
        ax.bar(edges[:-1], density, width=widths, align="edge",
               color=C_MODEL, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axhline(1.0, color=C_REFERENCE, lw=1.4, ls="--")
        disp = _dispersion_label(s.get("var", float("nan")))
        ax.set_title(f"{label}\n({disp})" if disp else label, fontsize=9.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability integral transform")
        if i % ncols == 0:
            ax.set_ylabel("Relative frequency")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Coverage reliability diagram — calibration magnitude
# ---------------------------------------------------------------------------

def plot_coverage_reliability(
    series: Mapping[str, dict], *, title: str = "Predictive-interval calibration",
    save_path: str | Path | None = None,
) -> Figure:
    """Empirical vs nominal interval coverage. `series` maps a descriptive label
    (e.g. "All sites", or a site name) to a coverage dict
    {level: {nominal, coverage}}. Points on the 1:1 line are perfectly calibrated;
    below the line = intervals too narrow (over-confident)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], color=C_REFERENCE, lw=1.4, ls="--", label="Perfect calibration")
    palette = [C_MODEL, C_ACCENT, C_BAND, "#CC79A7", "#56B4E9", "#E69F00", "#000000"]
    for i, (label, cov) in enumerate(series.items()):
        if not cov:
            continue
        levels = sorted(cov.values(), key=lambda d: d["nominal"])
        nom = [d["nominal"] for d in levels]
        emp = [d["coverage"] for d in levels]
        emphasise = label.lower().startswith("all")
        ax.plot(nom, emp, "-o", color=(C_OBSERVED if emphasise else palette[i % len(palette)]),
                lw=2.2 if emphasise else 1.3, ms=6 if emphasise else 4,
                alpha=1.0 if emphasise else 0.75, label=label, zorder=5 if emphasise else 3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Increment-variance scaling — diffusion consistency
# ---------------------------------------------------------------------------

def plot_variance_scaling(
    records: list[dict], *, title: str = "Increment-variance scaling",
    save_path: str | Path | None = None,
) -> Figure:
    """Observed vs model-generated increment variance across aggregation windows
    (from `variance_vs_scale`). A faithful diffusion reproduces the observed curve;
    the dashed line is the linear (Wiener) growth reference."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    hours = np.array([r["hours"] for r in records], dtype=float)
    v_obs = np.array([r["var_obs"] for r in records], dtype=float)
    v_gen = np.array([r["var_gen"] for r in records], dtype=float)
    ax.plot(hours, v_obs, "-o", color=C_OBSERVED, lw=2.2, ms=6, label="Observed", zorder=5)
    ax.plot(hours, v_gen, "-s", color=C_MODEL, lw=2.0, ms=5, label="Model (generated)", zorder=4)
    # linear-diffusion reference anchored at the first observed point
    ref = v_obs[0] * hours / hours[0]
    ax.plot(hours, ref, ls="--", color=C_REFERENCE, lw=1.4, label="Linear (Wiener) reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Aggregation window (hours)")
    ax.set_ylabel("Increment variance  (µmol m⁻² s⁻¹)²")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Standardized-residual autocorrelation — whiteness
# ---------------------------------------------------------------------------

def plot_residual_autocorrelation(
    acf_records: list[dict], *, n: int | None = None,
    title: str = "Standardised-residual autocorrelation",
    save_path: str | Path | None = None,
) -> Figure:
    """Autocorrelation of the standardized one-step residual vs lag (from
    `standardized_residual_autocorr`). Bars inside the shaded 95% white-noise band
    indicate no leftover structure; bars outside signal missed drift dynamics."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    hours = np.array([r["hours"] for r in acf_records], dtype=float)
    acf = np.array([r["acf"] for r in acf_records], dtype=float)
    if n is None:
        n = min((r.get("n", 0) for r in acf_records), default=0)
    if n and n > 0:
        band = 1.96 / np.sqrt(n)
        ax.axhspan(-band, band, color=C_BAND, alpha=0.15,
                   label="95% white-noise band")
    ax.axhline(0, color=C_REFERENCE, lw=1.0)
    ax.bar(hours, acf, width=0.28, color=C_MODEL, alpha=0.9)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Predictive skill by site — CRPS
# ---------------------------------------------------------------------------

def plot_crps_by_site(
    per_site: Mapping[str, dict], *, global_crps: float | None = None,
    title: str = "Predictive skill by site", save_path: str | Path | None = None,
) -> Figure:
    """CRPS per site (lower = better). `per_site` maps site slug to its metric
    block; the ensemble CRPS is used when available, else parametric."""
    _apply_style()
    labels, values = [], []
    for site, block in per_site.items():
        sc = _preferred_scores(block)
        if sc and sc.get("crps") is not None:
            labels.append(descriptive_site(site)); values.append(sc["crps"])
    order = np.argsort(values)
    labels = [labels[i] for i in order]; values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(5.6, 0.5 * len(labels) + 1.4))
    ax.barh(labels, values, color=C_MODEL, alpha=0.9)
    if global_crps is not None:
        ax.axvline(global_crps, color=C_ACCENT, lw=1.6, ls="--",
                   label=f"All sites ({global_crps:.3f})")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("Continuous ranked probability score  (µmol m⁻² s⁻¹)")
    ax.set_title(f"{title}  (lower is better)", fontsize=12, fontweight="bold", loc="left")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def plot_aggregated_skill(agg: dict, *, save_path: str | Path | None = None,
                          title: str | None = None) -> Figure:
    """Aggregated-scale figure: (left) the relative-uncertainty cancellation curve
    raw -> weekly -> monthly (physics signal persists, random noise cancels ~1/sqrt(N));
    (right) per-resolution aggregated skill (point R^2) and ensemble CRPS."""
    _apply_style()
    res_order = [r for r in ("weekly", "monthly") if r in agg]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9))
    # left: rel-uncertainty cancellation
    xs, ys, labs = [], [], []
    if agg.get("raw_rel_uncertainty") is not None:
        xs.append(0); ys.append(agg["raw_rel_uncertainty"]); labs.append("raw (0.5h)")
    for i, r in enumerate(res_order, 1):
        ru = agg[r].get("rel_uncertainty")
        if ru is not None:
            xs.append(i); ys.append(ru); labs.append(r)
    if len(ys) >= 2:
        ax[0].plot(xs, ys, "o-", color="#0072B2", lw=1.8, ms=7)
        ax[0].set_yscale("log"); ax[0].set_xticks(xs); ax[0].set_xticklabels(labs)
    ax[0].set_ylabel("relative uncertainty (log)")
    ax[0].set_title("Noise cancels under aggregation")
    ax[0].grid(alpha=.3)
    # right: R2 and CRPS per resolution
    x = np.arange(len(res_order)); w = 0.35
    r2 = [(agg[r].get("point") or {}).get("r2") for r in res_order]
    crps = [(agg[r].get("ensemble") or {}).get("crps") for r in res_order]
    axr = ax[1]; axr2 = axr.twinx()
    axr.bar(x - w / 2, [v if v is not None else 0 for v in r2], w, color="#009E73", label="R²")
    axr2.bar(x + w / 2, [v if v is not None else 0 for v in crps], w, color="#D55E00", label="CRPS")
    axr.set_xticks(x); axr.set_xticklabels(res_order)
    axr.set_ylabel("aggregated R²", color="#009E73"); axr2.set_ylabel("ensemble CRPS", color="#D55E00")
    axr.set_title("Aggregated skill"); axr.set_ylim(0, 1.02)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def emit_calibration_plots(
    probabilistic_report: dict, process_report: dict | None,
    out_dir: str | Path, *, model_name: str = "", aggregated_report: dict | None = None,
) -> list[Path]:
    """Emit the full manuscript figure set from the JSON reports. Returns the
    written paths. Silently skips any figure whose inputs are absent (e.g. no
    coverage for a deterministic model)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    title_model = descriptive_model(model_name) if model_name else ""
    g = probabilistic_report.get("global", {})
    per_site = probabilistic_report.get("per_site", {})

    # 1. PIT histograms (global + per site)
    panels: dict[str, dict] = {}
    gs = _preferred_scores(g)
    if gs and gs.get("pit"):
        panels["All sites"] = gs["pit"]
    for site, block in per_site.items():
        sc = _preferred_scores(block)
        if sc and sc.get("pit"):
            panels[descriptive_site(site)] = sc["pit"]
    if panels:
        p = out / "calibration_pit_histograms.png"
        plot_pit_histograms(panels, save_path=p,
                            title=f"Forecast calibration — {title_model}" if title_model else None)
        plt.close("all"); written.append(p)

    # 2. Coverage reliability (global emphasised + per site)
    series: dict[str, dict] = {}
    if gs and gs.get("coverage"):
        series["All sites"] = gs["coverage"]
    for site, block in per_site.items():
        sc = _preferred_scores(block)
        if sc and sc.get("coverage"):
            series[descriptive_site(site)] = sc["coverage"]
    if series:
        p = out / "calibration_coverage_reliability.png"
        plot_coverage_reliability(series, save_path=p)
        plt.close("all"); written.append(p)

    # 3. CRPS by site
    if per_site:
        p = out / "predictive_skill_by_site.png"
        plot_crps_by_site(per_site, global_crps=(gs or {}).get("crps"), save_path=p)
        plt.close("all"); written.append(p)

    # 4/5. Process-consistency (global)
    if process_report:
        gproc = process_report.get("global") or {}
        if gproc.get("variance_vs_scale"):
            p = out / "diffusion_variance_scaling.png"
            plot_variance_scaling(gproc["variance_vs_scale"], save_path=p)
            plt.close("all"); written.append(p)
        ac = gproc.get("residual_autocorrelation")
        if ac and ac.get("acf"):
            p = out / "residual_autocorrelation.png"
            plot_residual_autocorrelation(ac["acf"], n=ac.get("summary", {}).get("n"),
                                          save_path=p)
            plt.close("all"); written.append(p)

    # 6. Aggregated-scale skill (weekly/monthly)
    if aggregated_report and any(r in aggregated_report for r in ("weekly", "monthly")):
        p = out / "aggregated_skill.png"
        plot_aggregated_skill(aggregated_report, save_path=p,
                              title=f"Aggregated-scale skill — {title_model}" if title_model else None)
        plt.close("all"); written.append(p)
    return written
