#!/usr/bin/env python
"""Exploratory / read-only. Where does temperature *change* enough to matter,
and at what timescale does that change actually track NEE change?

Motivation: at the native 30-min step, Ta and Tsoil barely move, so dTa/dTsoil
correlate ~0 with dNEE (see nee_driver_correlations.py). This script walks up
the timescale ladder:

  1. NATIVE 30-min   — how big is the within-night 30-min step in Ta / Tsoil?
  2. WITHIN-NIGHT k   — accumulate k*30-min steps inside one night (window_id);
                        find the k (hours) where mean |ΔT| becomes meaningful.
  3. NIGHT-TO-NIGHT   — diff of per-night mean Ta/Tsoil/NEE.
  4. WEEK-TO-WEEK     — diff of per-week mean.
  5. MONTH-TO-MONTH   — diff of per-month mean.

At every level it reports mean |ΔTa|, mean |ΔTsoil|, and the Pearson r of
dTa & dTsoil against dNEE (per-site and pooled, sites z-scored before pooling).

Standalone. Does NOT import or modify the wienernet package. Run from repo root:
    conda activate pytorch
    python temp_change_timescales.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

SITE_FILES = {
    "rosedene": "data_manipulation/final_night_data.parquet",
    **{
        os.path.basename(os.path.dirname(p)): p
        for p in sorted(
            glob.glob("data_manipulation/other_sites/*/final_night_data.parquet")
        )
    },
}

TA = "Ta"
NEE = "NEE"
TSOIL_CANDIDATES = ["Tsoil1"]
NIGHT = "window_id"
TIME = "DateTime"

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 500)


def tsoil_col(df: pd.DataFrame) -> str | None:
    for c in TSOIL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def r_finite(x, y) -> tuple[float, int]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 30 or x[m].std() == 0 or y[m].std() == 0:
        return (np.nan, n)
    return (float(np.corrcoef(x[m], y[m])[0, 1]), n)


def mad(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if x.size else np.nan


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for site, path in SITE_FILES.items():
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if TIME in df:
            df = df.sort_values(TIME).reset_index(drop=True)
        out[site] = df
    return out


# ---------------------------------------------------------------------------
# 1 + 2: native 30-min and within-night accumulation of k steps
# ---------------------------------------------------------------------------
def within_night_deltas(df: pd.DataFrame, col: str, k: int) -> pd.Series:
    """T[t+k] - T[t] taken only within the same night window (row-shift by k)."""
    if col not in df or NIGHT not in df:
        return pd.Series(dtype=float)
    return df.groupby(NIGHT)[col].transform(lambda s: s.shift(-k) - s)


def within_night_table(data: dict[str, pd.DataFrame], ks) -> pd.DataFrame:
    rows = []
    for k in ks:
        # pooled, z-scored per site
        pooled = {"dTa": [], "dTs": [], "dNEE": []}
        approx_min = []
        per_site_absTa, per_site_absTs = [], []
        for site, df in data.items():
            ts = tsoil_col(df)
            dnee = within_night_deltas(df, NEE, k)
            dta = within_night_deltas(df, TA, k)
            dts = within_night_deltas(df, ts, k) if ts else pd.Series(np.nan, index=df.index)
            # approx elapsed minutes for this k within a night
            if TIME in df:
                dmin = df.groupby(NIGHT)[TIME].transform(
                    lambda s: (s.shift(-k) - s).dt.total_seconds() / 60.0
                )
                approx_min.append(np.nanmedian(dmin.values))
            per_site_absTa.append(mad(dta))
            per_site_absTs.append(mad(dts))
            # z-score per site before pooling correlations
            for key, series in (("dTa", dta), ("dTs", dts), ("dNEE", dnee)):
                s = series.values.astype(float)
                m = np.isfinite(s)
                if m.sum() > 30 and s[m].std() > 0:
                    z = np.full_like(s, np.nan)
                    z[m] = (s[m] - s[m].mean()) / s[m].std()
                    pooled[key].append(pd.Series(z, index=df.index))
        # align pooled arrays by concatenation (index-independent)
        dta_all = np.concatenate([p.values for p in pooled["dTa"]]) if pooled["dTa"] else np.array([])
        dts_all = np.concatenate([p.values for p in pooled["dTs"]]) if pooled["dTs"] else np.array([])
        # need matching dNEE per source; rebuild aligned pairs instead:
        # (recompute pooled pairwise to keep alignment simple)
        rTa, nTa = pooled_pair(data, TA, NEE, k, within=True)
        rTs, nTs = pooled_pair(data, "TSOIL", NEE, k, within=True)
        rows.append(dict(
            k=k,
            approx_h=round(np.nanmean(approx_min) / 60.0, 2) if approx_min else np.nan,
            meanabs_dTa=np.nanmean(per_site_absTa),
            meanabs_dTsoil=np.nanmean(per_site_absTs),
            r_dTa_dNEE=rTa, r_dTsoil_dNEE=rTs, n=nTa,
        ))
    return pd.DataFrame(rows)


def pooled_pair(data, xkind, ycol, k, within):
    """Pooled Pearson r of Δx vs Δy at lag k, sites z-scored first.
    xkind: 'Ta' col name, or 'TSOIL' to resolve per site."""
    zx_all, zy_all = [], []
    for site, df in data.items():
        xcol = tsoil_col(df) if xkind == "TSOIL" else xkind
        if xcol is None:
            continue
        if within:
            dx = within_night_deltas(df, xcol, k).values.astype(float)
            dy = within_night_deltas(df, ycol, k).values.astype(float)
        else:
            dx = (df[xcol].shift(-k) - df[xcol]).values.astype(float)
            dy = (df[ycol].shift(-k) - df[ycol]).values.astype(float)
        m = np.isfinite(dx) & np.isfinite(dy)
        if m.sum() < 30 or dx[m].std() == 0 or dy[m].std() == 0:
            continue
        zx = (dx[m] - dx[m].mean()) / dx[m].std()
        zy = (dy[m] - dy[m].mean()) / dy[m].std()
        zx_all.append(zx)
        zy_all.append(zy)
    if not zx_all:
        return (np.nan, 0)
    return r_finite(np.concatenate(zx_all), np.concatenate(zy_all))


# ---------------------------------------------------------------------------
# 3/4/5: aggregate-then-diff (night, week, month)
# ---------------------------------------------------------------------------
def aggregate_diff_table(data: dict[str, pd.DataFrame], level: str) -> pd.DataFrame:
    """Per-site + pooled r of Δmean(T) vs Δmean(NEE) at a coarse aggregation."""
    per_site = []
    pooled_x_ta, pooled_x_ts, pooled_y = [], [], []
    for site, df in data.items():
        ts = tsoil_col(df)
        if TIME not in df:
            continue
        t = df[TIME]
        if level == "night":
            if NIGHT not in df:
                continue
            grp = df[NIGHT]
            order_key = df.groupby(NIGHT)[TIME].transform("min")
        elif level == "week":
            grp = t.dt.to_period("W")
            order_key = grp
        elif level == "month":
            grp = t.dt.to_period("M")
            order_key = grp
        else:
            raise ValueError(level)

        cols = {NEE: "mean", TA: "mean"}
        if ts:
            cols[ts] = "mean"
        agg = df.assign(_g=grp, _ord=order_key).groupby("_g").agg(
            {**cols, "_ord": "first"}
        ).sort_values("_ord")
        d_nee = agg[NEE].diff().values
        d_ta = agg[TA].diff().values
        d_ts = agg[ts].diff().values if ts else np.full(len(agg), np.nan)

        rta, nta = r_finite(d_ta, d_nee)
        rts, nts = r_finite(d_ts, d_nee)
        per_site.append(dict(
            site=site, n_units=len(agg),
            meanabs_dTa=mad(d_ta), meanabs_dTsoil=mad(d_ts),
            r_dTa_dNEE=rta, r_dTsoil_dNEE=rts, n=nta,
        ))
        # z-score for pooling
        for src, store in ((d_ta, pooled_x_ta), (d_ts, pooled_x_ts)):
            m = np.isfinite(src) & np.isfinite(d_nee)
            if m.sum() > 10 and src[m].std() > 0:
                store.append((src[m] - src[m].mean()) / src[m].std())
        m = np.isfinite(d_ta) & np.isfinite(d_nee)
        if m.sum() > 10 and d_nee[m].std() > 0:
            pooled_y.append((d_nee[m] - d_nee[m].mean()) / d_nee[m].std())
    df_ps = pd.DataFrame(per_site)
    # pooled correlation (recompute aligned)
    rta_p, nta_p = pooled_pair_agg(data, TA, level)
    rts_p, nts_p = pooled_pair_agg(data, "TSOIL", level)
    return df_ps, (rta_p, rts_p, nta_p)


def pooled_pair_agg(data, xkind, level):
    zx_all, zy_all = [], []
    for site, df in data.items():
        xcol = tsoil_col(df) if xkind == "TSOIL" else xkind
        if xcol is None or TIME not in df:
            continue
        t = df[TIME]
        if level == "night":
            if NIGHT not in df:
                continue
            grp = df[NIGHT]
            order_key = df.groupby(NIGHT)[TIME].transform("min")
        elif level == "week":
            grp = t.dt.to_period("W"); order_key = grp
        elif level == "month":
            grp = t.dt.to_period("M"); order_key = grp
        agg = df.assign(_g=grp, _ord=order_key).groupby("_g").agg(
            {NEE: "mean", xcol: "mean", "_ord": "first"}
        ).sort_values("_ord")
        dx = agg[xcol].diff().values
        dy = agg[NEE].diff().values
        m = np.isfinite(dx) & np.isfinite(dy)
        if m.sum() < 10 or dx[m].std() == 0 or dy[m].std() == 0:
            continue
        zx_all.append((dx[m] - dx[m].mean()) / dx[m].std())
        zy_all.append((dy[m] - dy[m].mean()) / dy[m].std())
    if not zx_all:
        return (np.nan, 0)
    return r_finite(np.concatenate(zx_all), np.concatenate(zy_all))


def fmt(df):
    return df.to_string(index=False,
                        float_format=lambda v: f"{v:7.3f}" if pd.notna(v) else "    NaN")


def main():
    data = load_all()
    print(f"Loaded sites: {list(data.keys())}")
    for s, df in data.items():
        ts = tsoil_col(df)
        print(f"  {s:12s} rows={len(df):6d}  night_col={NIGHT in df}  Tsoil={ts}")

    # ---- 1 & 2: native 30-min + within-night accumulation
    print("\n" + "=" * 104)
    print("WITHIN-NIGHT: accumulate k*30-min steps inside one night (window_id).")
    print("meanabs_dT in °C; r_* pooled across sites (z-scored per site). k=1 is the NATIVE 30-min step.")
    print("=" * 104)
    wn = within_night_table(data, ks=[1, 2, 3, 4, 6, 8, 10, 12, 16])
    print(fmt(wn))

    # ---- 3/4/5: coarse aggregation diffs
    for level, label in (("night", "NIGHT-TO-NIGHT"),
                         ("week", "WEEK-TO-WEEK"),
                         ("month", "MONTH-TO-MONTH")):
        ps, (rta_p, rts_p, n_p) = aggregate_diff_table(data, level)
        print("\n" + "=" * 104)
        print(f"{label}: diff of per-{level} MEAN Ta / Tsoil / NEE, then correlate.")
        print("=" * 104)
        print(fmt(ps))
        print(f"  POOLED (sites z-scored): r(dTa,dNEE)={rta_p:6.3f}   "
              f"r(dTsoil,dNEE)={rts_p:6.3f}   n={n_p}")

    print("\nReading guide:")
    print(" - meanabs_dT tells you how many °C the temperature actually moves at that timescale.")
    print(" - r_dTa_dNEE / r_dTsoil_dNEE: does that temperature *change* track the NEE *change*?")
    print(" - Native 30-min (k=1) is expected tiny & ~0 corr; watch where |r| climbs above ~0.2-0.3.")


if __name__ == "__main__":
    main()
