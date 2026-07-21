#!/usr/bin/env python
"""Quantify how NEE-related drivers correlate with nighttime NEE and its
increment (dNEE), and report data coverage per site.

Covers soil temperature depths, soil moisture (VWC), water table, precipitation,
air temperature Ta and its increment dTa. Reports per-site and pooled (each site
z-scored per variable before pooling) Pearson correlations.

Run from the repo root:

    conda activate pytorch
    python nee_driver_correlations.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# site slug -> final_night_data.parquet
SITE_FILES = {
    "rosedene": "data_manipulation/final_night_data.parquet",
    **{
        os.path.basename(os.path.dirname(p)): p
        for p in sorted(
            glob.glob("data_manipulation/other_sites/*/final_night_data.parquet")
        )
    },
}

# Canonical variable -> list of possible column names across sites.
# (Rosedene uses VWC_1/VWC2; great_fen uses VWC1..VWC4; etc.)
VAR_ALIASES = {
    "Ta        (air temp, USED in drift)": ["Ta"],
    "Tsoil1    (soil temp, USED as driver)": ["Tsoil1"],
    "Tsoil2    (soil temp, unused)": ["Tsoil2"],
    "Tsoil3    (soil temp, unused)": ["Tsoil3"],
    "Tsoil4    (soil temp, unused)": ["Tsoil4"],
    "VWC1      (soil moisture, unused)": ["VWC1", "VWC_1"],
    "VWC2      (soil moisture, unused)": ["VWC2", "VWC_2"],
    "VWC3      (soil moisture, unused)": ["VWC3"],
    "VWC4      (soil moisture, unused)": ["VWC4"],
    "WaterLevel(water table, unused)": ["WaterLevel"],
    "Precip    (precipitation, unused)": ["Precipitation"],
    "dTa       (air-temp increment, USED)": ["dTa"],
}

TARGETS = {
    "NEE": "NEE",       # respiration level
    "dNEE": "dNEE",     # increment the SDE integrates
}


def resolve(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """Pearson r over the mutually non-NaN, finite subset."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 30 or x[m].std() == 0 or y[m].std() == 0:
        return (np.nan, n)
    return (float(np.corrcoef(x[m], y[m])[0, 1]), n)


def main() -> None:
    per_site_rows = []
    pooled = {}  # canonical var -> list of (value, NEE, dNEE) arrays to pool

    for site, path in SITE_FILES.items():
        if not os.path.exists(path):
            print(f"[skip] {site}: missing {path}")
            continue
        df = pd.read_parquet(path)
        nee = df[TARGETS["NEE"]] if TARGETS["NEE"] in df else None
        dnee = df[TARGETS["dNEE"]] if TARGETS["dNEE"] in df else None

        for label, aliases in VAR_ALIASES.items():
            col = resolve(df, aliases)
            if col is None:
                per_site_rows.append(
                    dict(site=site, var=label, present=False, cover=np.nan,
                         r_NEE=np.nan, n_NEE=0, r_dNEE=np.nan, n_dNEE=0)
                )
                continue
            x = df[col].astype(float)
            cover = float(np.isfinite(x).mean())
            r_nee, n_nee = corr_pair(x, nee) if nee is not None else (np.nan, 0)
            r_dnee, n_dnee = corr_pair(x, dnee) if dnee is not None else (np.nan, 0)
            per_site_rows.append(
                dict(site=site, var=label, present=True, cover=cover,
                     r_NEE=r_nee, n_NEE=n_nee, r_dNEE=r_dnee, n_dNEE=n_dnee)
            )
            # accumulate for pooled (site-standardised so sites weight evenly)
            if nee is not None and dnee is not None:
                pooled.setdefault(label, []).append(
                    pd.DataFrame({"x": x, "NEE": nee, "dNEE": dnee})
                )

    res = pd.DataFrame(per_site_rows)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 400)

    print("\n" + "=" * 100)
    print("PER-SITE  |  coverage (frac finite) and Pearson r with NEE and dNEE")
    print("=" * 100)
    for site in SITE_FILES:
        sub = res[res.site == site]
        if sub.empty:
            continue
        print(f"\n### {site}   (n_rows-based; only present vars shown)")
        show = sub[sub.present].copy()
        if show.empty:
            print("   (no target/driver columns matched)")
            continue
        show = show[["var", "cover", "r_NEE", "n_NEE", "r_dNEE", "n_dNEE"]]
        print(
            show.to_string(
                index=False,
                float_format=lambda v: f"{v:6.3f}" if pd.notna(v) else "   NaN",
            )
        )

    # ---- Pooled across sites (each site z-scored first so no single big site dominates)
    print("\n" + "=" * 100)
    print("POOLED across sites (each site z-scored per-variable before pooling)")
    print("=" * 100)
    pooled_rows = []
    for label, frames in pooled.items():
        zs = []
        for fdf in frames:
            m = np.isfinite(fdf["x"]) & np.isfinite(fdf["NEE"]) & np.isfinite(fdf["dNEE"])
            g = fdf[m]
            if len(g) < 30 or g["x"].std() == 0:
                continue
            z = g.copy()
            z["x"] = (z["x"] - z["x"].mean()) / z["x"].std()
            zs.append(z)
        if not zs:
            continue
        allz = pd.concat(zs, ignore_index=True)
        r_nee, n_nee = corr_pair(allz["x"], allz["NEE"])
        r_dnee, n_dnee = corr_pair(allz["x"], allz["dNEE"])
        n_sites = len(zs)
        pooled_rows.append(
            dict(var=label, n_sites=n_sites, n=n_nee,
                 r_NEE=r_nee, r_dNEE=r_dnee,
                 absr_NEE=abs(r_nee), absr_dNEE=abs(r_dnee))
        )
    pdf = pd.DataFrame(pooled_rows).sort_values("absr_NEE", ascending=False)
    print(
        pdf[["var", "n_sites", "n", "r_NEE", "r_dNEE"]].to_string(
            index=False,
            float_format=lambda v: f"{v:6.3f}" if pd.notna(v) else "   NaN",
        )
    )

    print("\nNotes:")
    print(" - r_NEE  = corr with nighttime NEE level (respiration magnitude).")
    print(" - r_dNEE = corr with the 30-min NEE increment (what the SDE integrates).")
    print(" - Pooled r's z-score each site first, so a strong-but-consistent signal")
    print("   shows up even if site means differ; n = pooled sample count.")


if __name__ == "__main__":
    main()
