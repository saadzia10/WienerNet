"""How the temperature -> NEE-increment signal scales with the time horizon.

Motivation: WienerNet models NEE as a Wiener SDE, `dNEE = f·dt + noise`, with the
drift `f = dReco/dT · dT/dt` (Lloyd-Taylor). At the native 30-min step the drift
is near-inert — over 30 min the temperature barely moves, so the temperature-
driven increment is tiny and the actual `ΔNEE` is dominated by turbulence/noise.
This module quantifies how the deterministic (physics) fraction of the increment
grows with the horizon, by correlating the NEE increment with the temperature /
Reco increment at increasing lags. It's the empirical motivation for multi-scale
increment training.

Only *contiguous* spans are used (the gap to the lagged sample equals exactly
`lag * step_minutes`), so a lag never straddles a data gap or a day.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..data.features import physics_nee_numpy
from .plots import COLORS, _maybe_save, setup_paper_style


def increment_correlation_by_lag(
    site_dfs: Mapping[str, pd.DataFrame],
    *,
    lags: Sequence[int] = (1, 2, 4, 8, 16),
    step_minutes: int = 30,
    datetime_column: str = "DateTime",
    nee_column: str = "NEE",
    temp_column: str = "Ta",
    e0_column: str = "E0",
    rb_column: str = "rb",
    min_pairs: int = 100,
) -> pd.DataFrame:
    """Correlate the NEE increment with the temperature / Reco increment per lag.

    For each site and each lag `k`, uses only rows whose sample `k` steps ahead is
    exactly `k * step_minutes` later (contiguous), then computes Pearson r between
    `ΔNEE` and `ΔTa` / `ΔReco`. A pooled row (all sites concatenated) is added per
    lag under site name ``"pooled"``.

    Returns a long DataFrame: [site, lag, hours, n, corr_dTa, corr_dReco].
    """
    rows: list[dict] = []
    pooled: dict[int, dict[str, list[np.ndarray]]] = {
        k: {"dNEE": [], "dTa": [], "dReco": []} for k in lags
    }

    for name, df in site_dfs.items():
        d = df.sort_values(datetime_column).reset_index(drop=True)
        dt = pd.to_datetime(d[datetime_column])
        nee = d[nee_column].to_numpy(dtype=float)
        ta = d[temp_column].to_numpy(dtype=float)
        reco = physics_nee_numpy(d[e0_column].to_numpy(), d[rb_column].to_numpy(), ta)

        for k in lags:
            gap = (dt.shift(-k) - dt).dt.total_seconds().to_numpy() / 60.0
            contiguous = gap == k * step_minutes
            d_nee = np.roll(nee, -k) - nee
            d_ta = np.roll(ta, -k) - ta
            d_reco = np.roll(reco, -k) - reco
            mask = contiguous & np.isfinite(d_nee) & np.isfinite(d_ta) & np.isfinite(d_reco)
            n = int(mask.sum())
            if n >= min_pairs:
                rows.append({
                    "site": name, "lag": k, "hours": k * step_minutes / 60.0, "n": n,
                    "corr_dTa": float(np.corrcoef(d_nee[mask], d_ta[mask])[0, 1]),
                    "corr_dReco": float(np.corrcoef(d_nee[mask], d_reco[mask])[0, 1]),
                })
            pooled[k]["dNEE"].append(d_nee[mask])
            pooled[k]["dTa"].append(d_ta[mask])
            pooled[k]["dReco"].append(d_reco[mask])

    for k in lags:
        dn = np.concatenate(pooled[k]["dNEE"]) if pooled[k]["dNEE"] else np.array([])
        dta = np.concatenate(pooled[k]["dTa"]) if pooled[k]["dTa"] else np.array([])
        dre = np.concatenate(pooled[k]["dReco"]) if pooled[k]["dReco"] else np.array([])
        if len(dn) >= min_pairs:
            rows.append({
                "site": "pooled", "lag": k, "hours": k * step_minutes / 60.0, "n": int(len(dn)),
                "corr_dTa": float(np.corrcoef(dn, dta)[0, 1]),
                "corr_dReco": float(np.corrcoef(dn, dre)[0, 1]),
            })

    return pd.DataFrame(rows).sort_values(["site", "lag"]).reset_index(drop=True)


def plot_increment_correlation(
    result: pd.DataFrame,
    *,
    metric: str = "corr_dReco",
    figsize: tuple[float, float] = (7, 5),
    save_path: str | Path | None = None,
) -> Figure:
    """Plot |correlation| of the NEE increment vs horizon.

    Pooled line is drawn bold; per-site lines are faint. `metric` is
    ``"corr_dReco"`` (Lloyd-Taylor physics) or ``"corr_dTa"`` (raw temperature).
    """
    import matplotlib.pyplot as plt

    setup_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    for site, g in result.groupby("site"):
        g = g.sort_values("hours")
        if site == "pooled":
            continue
        ax.plot(g["hours"], g[metric].abs(), "-", color="0.7", linewidth=1.2, alpha=0.8, zorder=1)

    pooled = result[result["site"] == "pooled"].sort_values("hours")
    if not pooled.empty:
        ax.plot(pooled["hours"], pooled[metric].abs(), "o-",
                color=COLORS["pred"], linewidth=3, markersize=8, label="pooled", zorder=3)

    ax.set_xlabel("Increment horizon (hours)")
    ax.set_ylabel("|correlation| with NEE increment")
    ax.set_ylim(bottom=0)
    ax.legend(["per site", "pooled"], fontsize="large")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
