"""Smoke tests for the manuscript metric-illustration plots."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np

from wienernet.evaluation import (
    descriptive_model,
    descriptive_site,
    emit_calibration_plots,
    pit_summary,
    plot_coverage_reliability,
    plot_crps_by_site,
    plot_pit_histograms,
    plot_residual_autocorrelation,
    plot_variance_scaling,
)


def _pit(rng, narrow=False):
    x = rng.standard_normal(3000)
    scale = 0.4 if narrow else 1.0
    pit = 0.5 + 0.5 * np.tanh((x / scale) / 2)  # crude but in (0,1), varies shape
    return pit_summary(np.clip(pit, 0, 1))


def _coverage(under=False):
    base = {0.5: 0.5, 0.9: 0.9, 0.95: 0.95}
    return {f"{k:.2f}": {"nominal": k, "coverage": (k * 0.4 if under else k),
                         "sharpness": k} for k in base}


def test_descriptive_labels():
    assert descriptive_site("great_fen") == "Great Fen"
    assert descriptive_site("redmere_1") == "Redmere 1"
    assert "Student-t" in descriptive_model("nll_student_t")
    # unknown slug falls back to a title-cased label, never the raw identifier
    assert descriptive_model("some_new_model") == "Some New Model"


def test_plot_pit_histograms(tmp_path):
    rng = np.random.default_rng(0)
    panels = {"All sites": _pit(rng), "Great Fen": _pit(rng, narrow=True)}
    p = tmp_path / "pit.png"
    plot_pit_histograms(panels, save_path=p, title="Calibration")
    assert p.exists() and p.stat().st_size > 0


def test_plot_coverage_reliability(tmp_path):
    p = tmp_path / "cov.png"
    plot_coverage_reliability({"All sites": _coverage(), "Rosedene": _coverage(under=True)},
                              save_path=p)
    assert p.exists()


def test_plot_variance_scaling_and_acf(tmp_path):
    recs = [{"k": k, "hours": k * 0.5, "var_obs": k * 1.0, "var_gen": k * 0.8,
             "ratio": 0.8, "n": 500} for k in (1, 2, 4, 8)]
    p1 = tmp_path / "var.png"; plot_variance_scaling(recs, save_path=p1)
    acf = [{"lag": l, "hours": l * 0.5, "acf": 0.3 / l, "n": 1000} for l in (1, 2, 3)]
    p2 = tmp_path / "acf.png"; plot_residual_autocorrelation(acf, n=1000, save_path=p2)
    assert p1.exists() and p2.exists()


def test_plot_crps_by_site(tmp_path):
    per_site = {
        "great_fen": {"probabilistic": {"crps": 0.9}},
        "rosedene": {"ensemble": {"crps": 0.7}, "probabilistic": {"crps": 0.8}},
    }
    p = tmp_path / "crps.png"
    plot_crps_by_site(per_site, global_crps=0.8, save_path=p)
    assert p.exists()


def test_emit_calibration_plots_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    prob = {
        "family": "gaussian", "nu": None,
        "global": {"probabilistic": {"crps": 0.8, "pit": _pit(rng), "coverage": _coverage()}},
        "per_site": {
            "great_fen": {"probabilistic": {"crps": 0.9, "pit": _pit(rng), "coverage": _coverage()}},
            "rosedene": {"probabilistic": {"crps": 0.7, "pit": _pit(rng), "coverage": _coverage(under=True)}},
        },
    }
    proc = {"global": {
        "variance_vs_scale": [{"k": k, "hours": k * 0.5, "var_obs": k * 1.0,
                               "var_gen": k * 0.8, "ratio": 0.8, "n": 500} for k in (1, 2, 4)],
        "residual_autocorrelation": {
            "summary": {"n": 1000, "std_z": 1.0},
            "acf": [{"lag": l, "hours": l * 0.5, "acf": 0.2 / l, "n": 1000} for l in (1, 2, 3)]},
    }}
    written = emit_calibration_plots(prob, proc, tmp_path, model_name="nll_student_t")
    names = {p.name for p in written}
    assert "calibration_pit_histograms.png" in names
    assert "calibration_coverage_reliability.png" in names
    assert "predictive_skill_by_site.png" in names
    assert "diffusion_variance_scaling.png" in names
    assert "residual_autocorrelation.png" in names
    for p in written:
        assert p.exists() and p.stat().st_size > 0
