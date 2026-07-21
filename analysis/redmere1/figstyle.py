"""Shared manuscript-figure scaffold for the decomposition plots.

Implements the repo plot-conventions skill: no titles / panel letters, unit-bearing axis
labels, one standalone vector (PDF) file per plot, print-legible fonts, tight bbox, a fixed
Okabe-Ito colorblind palette, and a draft-caption `.txt` sidecar per figure. A preview PNG is
written alongside the PDF purely so the figure can be embedded in the analysis Markdown; the
PDF is the manuscript deliverable.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette — fixed across every decomposition figure.
C = {"black": "#111111", "blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
     "sky": "#56B4E9", "vermillion": "#E69F00", "purple": "#CC79A7", "grey": "#999999"}
LW = 2.2          # shared line width
MSZ = 6.5         # shared line-marker size

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": .7, "axes.axisbelow": True,
    "axes.edgecolor": "#888", "axes.linewidth": .8,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGDIR, exist_ok=True)


def save(fig, name, caption, preview_png=True):
    """Save one standalone vector figure (PDF) + a preview PNG + a draft-caption sidecar.

    `name` is the extensionless basename (grouped by prefix). `caption` is a 1–2 sentence
    LaTeX-ready draft written to `<name>.txt` and echoed to the console.
    """
    base = os.path.join(FIGDIR, name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    if preview_png:
        fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    cap = " ".join(caption.split())
    with open(base + ".txt", "w") as f:
        f.write(cap + "\n")
    print(f"wrote {name}.pdf  |  {cap}")
