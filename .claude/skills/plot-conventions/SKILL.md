---
name: plot-conventions
description: Manuscript plot-generation conventions for this repo. Load BEFORE writing or editing ANY plotting code — matplotlib/pyplot/plt, seaborn, plotly, savefig, plt.subplots, .plot(), charts, figures, or any script/notebook that produces a figure for the paper. Enforces no-titles, unit-bearing axis labels, one-standalone-vector-file-per-plot (PDF/SVG), print-legible fonts, tight bbox, a consistent colorblind-safe palette, and a draft-caption sidecar per figure.
---

# Plot generation conventions (WienerNet manuscript figures)

Every plot in this repo targets a **print figure in a LaTeX manuscript**. Apply
these rules whenever you create or edit plotting code, then verify the output
before finishing. These mirror the "Plot generation conventions" section of
`CLAUDE.md` — keep the two in sync if either changes.

## The rules

1. **No titles.** Never `plt.title()`, `ax.set_title()`, `plt.suptitle()`, or any
   figure-level title. The LaTeX caption describes the figure. **No panel letters**
   (A/B/C) baked into the image either — LaTeX `subcaption` adds them.
2. **Always label axes, with units.** e.g. `"Latency (ms)"`, `"Predicted flux Reco
   (µmol m⁻² s⁻¹)"` — never a bare quantity name, never an unlabeled axis.
3. **Print-legible fonts.** Labels, ticks, and legends must stay ≥ ~8–9 pt *after*
   LaTeX shrinks the image to its final width. Do not ship matplotlib defaults; size
   the figure for its final placement (a full-width figure squeezed into one column
   turns 10 pt text into ~3 pt).
4. **One plot per file.** Each plot is its own standalone file. **Never** build
   combined multi-panel images (`plt.subplots(2,2)` merging separate experiments,
   PIL compositing, etc.). Generate each panel separately; let LaTeX
   `subfigure`/`subcaption` handle layout and panel labels.
5. **Vector by default.** Export **PDF or SVG**. Use PNG *only* for heavy rasterized
   content (dense scatter/heatmap), and then at **≥ 300 DPI**. Embed editable PDF
   fonts with `pdf.fonttype: 42`.
6. **Tight bounding box.** Always save with `bbox_inches="tight"`.
7. **Consistent, colorblind-friendly palette.** One fixed palette across the whole
   paper (`tab10`, `viridis`, or the Okabe-Ito set below) — never per-plot random
   colors. Keep line widths, marker sizes, and legend placement consistent across
   plots that appear together.
8. **Descriptive, groupable filenames.** e.g. `fig3_latency_vs_load.pdf`,
   `fig3_throughput_vs_load.pdf` — related panels sort and reference together.
9. **Draft-caption sidecar.** After saving `<name>.<ext>`, print a 1–2 sentence
   caption to the console **and** write it to `<name>.txt` (e.g.
   `fig3_latency_vs_load.pdf` → `fig3_latency_vs_load.txt`) describing what the plot
   shows and its key takeaway, for later editing in LaTeX.

## Ready-to-use scaffold (matplotlib)

Reuse this preamble and `save()` helper so every figure conforms by construction.

```python
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (or use tab10 / viridis consistently)
C = {"black":"#111111","blue":"#0072B2","orange":"#D55E00","green":"#009E73",
     "sky":"#56B4E9","vermillion":"#E69F00","purple":"#CC79A7","grey":"#999999"}
LW, MS = 2.2, 28  # shared line width / marker size across co-appearing plots

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":12,
    "axes.labelsize":12,"axes.titlesize":12,
    "xtick.labelsize":10.5,"ytick.labelsize":10.5,"legend.fontsize":10,
    "axes.grid":True,"grid.color":"#e6e6e6","grid.linewidth":.7,"axes.axisbelow":True,
    "axes.edgecolor":"#888","axes.linewidth":.8,
    "pdf.fonttype":42,"svg.fonttype":"none",
})

def save(fig, path_no_ext, caption):
    """Save one standalone vector figure + write/print its draft caption."""
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")   # vector; PNG @300dpi only for heavy raster
    plt.close(fig)
    with open(path_no_ext + ".txt", "w") as f:
        f.write(caption.strip() + "\n")
    print(f"wrote {os.path.basename(path_no_ext)}.pdf")
    print(f"  caption: {caption.strip()}")

# --- one figure, one file ---
fig, ax = plt.subplots(figsize=(4.6, 3.6))         # sized for ~half-width placement
ax.plot(x, y, color=C["blue"], lw=LW, label="…")
ax.set_xlabel("Time (s)")                          # units, always
ax.set_ylabel("Throughput (req s⁻¹)")
ax.legend(frameon=False, loc="upper left")         # pin placement; don't rely on "best"
fig.tight_layout()
save(fig, os.path.join(FIGDIR, "fig3_throughput_vs_time"),
     "Throughput versus elapsed time under sustained load; it plateaus after ~2 s "
     "as the connection pool saturates.")
```

Notes:
- Save figures into a repo `figures/` dir (see the `save-figures-in-repo` memory) —
  not the ephemeral scratchpad.
- Pin `legend(loc=...)` deliberately; matplotlib's `"best"` often lands on the data.
- If a plot genuinely needs PNG (dense heatmap/scatter), save `.png` with `dpi=300`
  (or higher) plus `bbox_inches="tight"`, and still write the `.txt` sidecar.

## Before you finish

- [ ] No title / suptitle / baked-in panel letters anywhere.
- [ ] Both axes labeled, with units.
- [ ] Each plot is its own file (no merged panels).
- [ ] Vector PDF/SVG (or PNG ≥300 DPI only if heavy raster), `bbox_inches="tight"`.
- [ ] Consistent colorblind palette + line/marker/legend styling.
- [ ] Descriptive, groupable filenames.
- [ ] `<name>.txt` caption written and printed for every figure.
- [ ] Rendered the figure and eyeballed it (labels legible, legend clear of data).
