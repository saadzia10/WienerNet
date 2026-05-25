"""Feature engineering helpers.

These functions used to live as inline cells in
piae_sde/All_Seed_Experiments_Night.ipynb and piae_sde/Analysis Notebook.ipynb.
Moving them here gives us one canonical implementation that's easy to unit test.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from data_pipeline.config import SITES

log = logging.getLogger("wienernet.data.features")


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

TIME_VAR_COLUMNS = ("month_sin", "month_cos", "hour_sin", "hour_cos", "time")


def add_time_vars(df: pd.DataFrame, *, datetime_column: str = "DateTime") -> pd.DataFrame:
    """Add sin/cos cyclic encodings of month/hour plus a normalised `time` column.

    Output columns appended: month_sin, month_cos, hour_sin, hour_cos, time.
    `time` is the fractional position within a year, useful as a slow trend.
    """
    out = df.copy()
    dt = pd.to_datetime(out[datetime_column])

    month = dt.dt.month
    hour = dt.dt.hour + dt.dt.minute / 60.0

    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    day_of_year = dt.dt.dayofyear + hour / 24.0
    out["time"] = day_of_year / 366.0
    return out


# ---------------------------------------------------------------------------
# Season tagging
# ---------------------------------------------------------------------------

def set_season_tag(
    df: pd.DataFrame,
    *,
    datetime_column: str = "DateTime",
    out_column: str = "season",
    northern_hemisphere: bool = True,
) -> pd.DataFrame:
    """Append a numeric season column.

    Encoding: winter=0, spring=1, summer=2, autumn=3 (Northern Hemisphere).
    Southern hemisphere swaps winter↔summer and spring↔autumn.
    """
    out = df.copy()
    month = pd.to_datetime(out[datetime_column]).dt.month
    # Northern-hemisphere mapping
    seasons = pd.Series(0, index=out.index)
    seasons[(month >= 3) & (month <= 5)] = 1   # spring
    seasons[(month >= 6) & (month <= 8)] = 2   # summer
    seasons[(month >= 9) & (month <= 11)] = 3  # autumn
    # winter (Dec/Jan/Feb) stays 0

    if not northern_hemisphere:
        swap = {0: 2, 2: 0, 1: 3, 3: 1}
        seasons = seasons.map(swap)
    out[out_column] = seasons
    return out


# ---------------------------------------------------------------------------
# Site geography → unit-sphere coordinates
# ---------------------------------------------------------------------------

def lat_lon_to_unit_vector(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Convert latitude/longitude (degrees) to a 3D unit-sphere coordinate.

    The unit vector lets the model represent geography as continuous features
    without dealing with the wraparound discontinuity at the dateline.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return float(x), float(y), float(z)


SITE_VECTOR_COLUMNS = ("site_x", "site_y", "site_z")


def add_site_vector(df: pd.DataFrame, lat_deg: float, lon_deg: float) -> pd.DataFrame:
    """Append constant site_x, site_y, site_z columns derived from lat/lon."""
    out = df.copy()
    sx, sy, sz = lat_lon_to_unit_vector(lat_deg, lon_deg)
    out["site_x"] = sx
    out["site_y"] = sy
    out["site_z"] = sz
    return out


def add_site_vector_for(df: pd.DataFrame, site_name: str) -> pd.DataFrame:
    """Append site_x, site_y, site_z columns using the site coords in `data_pipeline.config.SITES`."""
    if site_name not in SITES:
        raise KeyError(f"Unknown site {site_name!r}. Known sites: {sorted(SITES)}")
    info = SITES[site_name]
    return add_site_vector(df, float(info["lat"]), float(info["lon"]))


# ---------------------------------------------------------------------------
# Physics-based NEE (numpy)
# ---------------------------------------------------------------------------

def physics_nee_numpy(
    E0: np.ndarray | pd.Series,
    rb: np.ndarray | pd.Series,
    T: np.ndarray | pd.Series,
    *,
    tref: float = 10.0,
    t0: float = 46.02,
) -> np.ndarray:
    """Reichstein Lloyd-Taylor in numpy. Matches data_pipeline.partitioning.lloyd_taylor."""
    return rb * np.exp(E0 * (1.0 / (tref + t0) - 1.0 / (T + t0)))


# ---------------------------------------------------------------------------
# Combined feature concatenation
# ---------------------------------------------------------------------------

def assemble_features(
    df: pd.DataFrame,
    *,
    drivers: Iterable[str],
    include_season: bool = True,
    include_time_vars: bool = True,
    include_site_xyz: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Concatenate selected feature columns into one float32 matrix.

    Returns the matrix plus a list of the corresponding column names so we
    keep an explicit record of feature order.
    """
    cols: list[str] = list(drivers)
    if include_season:
        cols.append("season")
    if include_time_vars:
        cols.extend(TIME_VAR_COLUMNS)
    if include_site_xyz:
        cols.extend(SITE_VECTOR_COLUMNS)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[cols].to_numpy(dtype=np.float32), cols


def compute_noise_statistics(
    nee: np.ndarray | pd.Series,
    nee_phys: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Mean and std of the noise term (NEE - NEE_phy) used as MMD prior."""
    residual = np.asarray(nee) - np.asarray(nee_phys)
    finite = residual[np.isfinite(residual)]
    return float(np.mean(finite)), float(np.std(finite))
