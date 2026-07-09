"""Nighttime/daytime flux partitioning and parameter fitting.

Night: fit (E0, rb) per window via Lloyd-Taylor on respiration.
Day:   fit (alpha, beta) per window via the light-response curve on GPP,
       then back out rb from TER using the previously fitted E0.

Both notebooks (Parameter Estimation.ipynb and Parameter Estimation Other
Sites.ipynb) share this logic; the only difference is how day/night is
identified:
  - main Rosedene site uses a precomputed `Day/Night` boolean column
  - other sites compute is_night from astral sunrise/sunset (timezone-aware)

The class supports both modes via `night_mask_strategy`.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .config import (
    CURVE_FIT_MAXFEV,
    DAY_WINDOW_GAP_HOURS,
    LIGHT_RESPONSE_BOUNDS,
    LLOYD_TAYLOR_BOUNDS,
    LLOYD_TAYLOR_T0_PARAM_FIT,
    LLOYD_TAYLOR_TREF,
    MIN_DATA_POINTS_PER_WINDOW,
    NAN_SENTINEL,
    NIGHT_RADIATION_THRESHOLD,
    NIGHT_WINDOW_GAP_HOURS,
    SITES,
    TEMPERATURE_RANGE_THRESHOLD,
)

log = logging.getLogger("wienernet.data_pipeline.partitioning")


# ---------------------------------------------------------------------------
# Physics functions (notebook formulas, preserved verbatim).
# These use the parameter-fit convention T0 = +46.02 -> denominator (T + 46.02).
# See note in config.py about the divergence from the night-model formula.
# ---------------------------------------------------------------------------

def lloyd_taylor(T: np.ndarray | pd.Series, Reco_ref: float, E0: float) -> np.ndarray:
    """Lloyd-Taylor respiration (notebook parameter-fit form, T in °C)."""
    T0 = LLOYD_TAYLOR_T0_PARAM_FIT
    Tref = LLOYD_TAYLOR_TREF
    return Reco_ref * np.exp(E0 * (1.0 / (Tref + T0) - 1.0 / (T + T0)))


def lloyd_taylor_invert_to_rb(Reco: np.ndarray | pd.Series, T: np.ndarray | pd.Series, E0: np.ndarray | pd.Series) -> np.ndarray:
    """Solve for rb given Reco, T, and a known E0."""
    T0 = LLOYD_TAYLOR_T0_PARAM_FIT
    Tref = LLOYD_TAYLOR_TREF
    return Reco * np.exp(-(E0 * (1.0 / (Tref + T0) - 1.0 / (T + T0))))


def light_response_curve(PAR: np.ndarray | pd.Series, alpha: float, beta: float) -> np.ndarray:
    """Hyperbolic light-response: GPP = alpha*beta*PAR / (alpha*PAR + beta)."""
    return (alpha * beta * PAR) / (alpha * PAR + beta)


# ---------------------------------------------------------------------------
# Day/Night masking strategies
# ---------------------------------------------------------------------------

def make_night_mask_from_column(df: pd.DataFrame, *, column: str = "Day/Night") -> pd.Series:
    """Strategy A: use a precomputed boolean column (True=day, False=night)."""
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found in dataframe")
    return df[column] == False  # noqa: E712 (notebook convention preserved)


def make_night_mask_from_astral(
    df: pd.DataFrame,
    *,
    site: str,
    datetime_column: str = "DateTime",
) -> pd.Series:
    """Strategy B: compute is_night from sunrise/sunset for the given site.

    Falls back to the nearest preceding date if a row's date isn't yet in the
    cached sun-times dict (handles edge cases at the year boundary).
    """
    from astral import LocationInfo
    from astral.sun import sun

    if site not in SITES:
        raise KeyError(f"Unknown site {site!r}; known: {sorted(SITES)}")
    info = SITES[site]
    city = LocationInfo(site, str(info["country"]), str(info["timezone"]), float(info["lat"]), float(info["lon"]))

    timestamps = pd.to_datetime(df[datetime_column])
    local = timestamps.dt.tz_localize(info["timezone"], ambiguous="NaT", nonexistent="shift_forward")
    dates = local.dt.date

    unique_dates: list[_dt.date] = sorted({d for d in dates.dropna().unique()})
    sun_times: dict[_dt.date, dict[str, _dt.datetime]] = {}
    for d in unique_dates:
        s = sun(city.observer, date=d)
        sun_times[d] = {"sunrise": s["sunrise"], "sunset": s["sunset"]}

    available = sorted(sun_times.keys())

    def lookup(d):
        if pd.isnull(d):
            return None, None
        pos = bisect.bisect_right(available, d)
        if pos == 0:
            return None, None
        key = available[pos - 1]
        return sun_times[key]["sunrise"], sun_times[key]["sunset"]

    sunrise = []
    sunset = []
    for d in dates:
        sr, ss = lookup(d)
        sunrise.append(sr)
        sunset.append(ss)
    sunrise_s = pd.Series(sunrise, index=df.index)
    sunset_s = pd.Series(sunset, index=df.index)

    return (local < sunrise_s) | (local > sunset_s)


# ---------------------------------------------------------------------------
# Parameter estimator
# ---------------------------------------------------------------------------

class ParameterEstimator:
    """Fit (E0, rb) on nighttime data and (alpha, beta) on daytime data.

    The class is stateful only for thresholds; each `fit_*` call returns a
    new dataframe and does not mutate inputs.
    """

    def __init__(
        self,
        *,
        night_radiation_threshold: float = NIGHT_RADIATION_THRESHOLD,
        night_window_gap_hours: int = NIGHT_WINDOW_GAP_HOURS,
        day_window_gap_hours: int = DAY_WINDOW_GAP_HOURS,
        temperature_range_threshold: float = TEMPERATURE_RANGE_THRESHOLD,
        min_data_points_per_window: int = MIN_DATA_POINTS_PER_WINDOW,
        lloyd_taylor_bounds: tuple = LLOYD_TAYLOR_BOUNDS,
        light_response_bounds: tuple = LIGHT_RESPONSE_BOUNDS,
        maxfev: int = CURVE_FIT_MAXFEV,
    ) -> None:
        self.night_radiation_threshold = night_radiation_threshold
        self.night_window_gap_hours = night_window_gap_hours
        self.day_window_gap_hours = day_window_gap_hours
        self.temperature_range_threshold = temperature_range_threshold
        self.min_data_points_per_window = min_data_points_per_window
        self.lloyd_taylor_bounds = lloyd_taylor_bounds
        self.light_response_bounds = light_response_bounds
        self.maxfev = maxfev

    # -------------------------------------------------------------------
    # Top-level orchestration
    # -------------------------------------------------------------------

    def fit_night(
        self,
        df: pd.DataFrame,
        *,
        night_mask: pd.Series,
        temperature_column: str = "Ta",
        reco_column: str = "TER",
        radiation_column: str = "Rg",
        datetime_column: str = "DateTime",
    ) -> pd.DataFrame:
        """End-to-end night fit. Returns the full night dataframe with E0, rb columns.

        Pipeline: apply night_mask -> apply radiation filter -> group into
        windows -> filter windows -> Lloyd-Taylor fit -> merge back into
        original night frame -> forward/back fill E0, rb gaps.
        """
        night = df[night_mask].copy().reset_index(drop=True)
        if reco_column not in night.columns and "NEE" in night.columns:
            log.info("reco_column %r missing; falling back to 'NEE' (Reco=NEE at night)", reco_column)
            reco_column = "NEE"

        # Note: the notebook pipeline reassigns night_data to the radiation-filtered
        # subset (Rg < threshold) BEFORE windowing, and that same subset is what
        # gets returned. Preserve this so the row count matches existing artefacts.
        filtered = self._filter_low_radiation(night, radiation_column=radiation_column)
        windowed = self._group_into_windows(
            filtered,
            gap=pd.Timedelta(hours=self.night_window_gap_hours),
            datetime_column=datetime_column,
        )
        valid = self._select_valid_windows(
            windowed,
            temperature_column=temperature_column,
        )
        if valid.empty:
            raise RuntimeError("No valid windows after temperature/length filtering")

        fitted = self._fit_lloyd_taylor_per_window(
            valid,
            temperature_column=temperature_column,
            reco_column=reco_column,
        )

        # Merge fits back into the radiation-filtered subset, fill gaps
        common = list(set(filtered.columns) & set(fitted.columns) - {datetime_column})
        merged = pd.merge(
            filtered,
            fitted.drop(columns=common),
            how="left",
            on=datetime_column,
        )
        merged["E0"] = merged["E0"].ffill().bfill()
        merged["rb"] = merged["rb"].ffill().bfill()
        return merged

    def fit_day(
        self,
        df: pd.DataFrame,
        night_results: pd.DataFrame,
        *,
        day_mask: pd.Series,
        temperature_column: str = "Ta",
        par_column: str = "Rg",
        gpp_column: str = "GEP",
        ter_column: str = "TER",
        datetime_column: str = "DateTime",
        date_column: str = "Date",
    ) -> pd.DataFrame:
        """End-to-end day fit (alpha, beta) plus rb recovered from TER & E0.

        For each day's rows we pull E0 from `night_results` (one value per day),
        group consecutive day-time samples into windows, fit light-response,
        then back out rb from TER.
        """
        day = df[day_mask].copy().reset_index(drop=True)
        day = day.dropna(subset=[par_column]).reset_index(drop=True)

        if date_column not in day.columns:
            day[date_column] = pd.to_datetime(day[datetime_column]).dt.date
        if date_column not in night_results.columns:
            night_results = night_results.copy()
            night_results[date_column] = pd.to_datetime(night_results[datetime_column]).dt.date

        # Pull E0 per date
        e0_by_date = (
            night_results.dropna(subset=["E0"]).groupby(date_column)["E0"].first().to_dict()
        )
        day["E0"] = day[date_column].map(e0_by_date)

        windowed = self._group_into_windows(
            day,
            gap=pd.Timedelta(hours=self.day_window_gap_hours),
            datetime_column=datetime_column,
        )

        fitted = self._fit_light_response_per_window(
            windowed,
            par_column=par_column,
            gpp_column=gpp_column,
        )

        # Back out rb from TER and the now-known E0
        if ter_column in fitted.columns:
            fitted["rb"] = lloyd_taylor_invert_to_rb(
                fitted[ter_column].values,
                fitted[temperature_column].values,
                fitted["E0"].values,
            )
        return fitted

    # -------------------------------------------------------------------
    # Building blocks
    # -------------------------------------------------------------------

    def _filter_low_radiation(self, df: pd.DataFrame, *, radiation_column: str) -> pd.DataFrame:
        return df[df[radiation_column] < self.night_radiation_threshold].copy()

    @staticmethod
    def _group_into_windows(
        df: pd.DataFrame,
        *,
        gap: pd.Timedelta,
        datetime_column: str,
    ) -> pd.DataFrame:
        """Add a `window_id` column that increments at every gap > `gap`."""
        out = df.sort_values(datetime_column).copy()
        out["TimeDiff"] = out[datetime_column].diff()
        out["window_id"] = (out["TimeDiff"] > gap).cumsum()
        return out.drop(columns=["TimeDiff"]).reset_index(drop=True)

    def _select_valid_windows(self, df: pd.DataFrame, *, temperature_column: str) -> pd.DataFrame:
        kept = []
        for _, window in df.groupby("window_id"):
            temp_range = window[temperature_column].max() - window[temperature_column].min()
            if temp_range > self.temperature_range_threshold and len(window) >= self.min_data_points_per_window:
                kept.append(window)
        return pd.concat(kept).reset_index(drop=True) if kept else pd.DataFrame(columns=df.columns)

    def _fit_lloyd_taylor_per_window(
        self,
        df: pd.DataFrame,
        *,
        temperature_column: str,
        reco_column: str,
    ) -> pd.DataFrame:
        out = df.copy()
        out["E0"] = 0.0
        out["rb"] = 0.0
        for window_id, window in df.groupby("window_id"):
            try:
                (Reco_ref, E0), _ = curve_fit(
                    lloyd_taylor,
                    window[temperature_column],
                    window[reco_column],
                    bounds=self.lloyd_taylor_bounds,
                    maxfev=self.maxfev,
                )
            except (ValueError, RuntimeError) as exc:
                log.warning("Lloyd-Taylor fit failed for window %s: %s", window_id, exc)
                continue
            out.loc[out["window_id"] == window_id, "E0"] = E0
            out.loc[out["window_id"] == window_id, "rb"] = Reco_ref
        return out

    def _fit_light_response_per_window(
        self,
        df: pd.DataFrame,
        *,
        par_column: str,
        gpp_column: str,
    ) -> pd.DataFrame:
        out = df.copy()
        out["alpha"] = 0.0
        out["beta"] = 0.0
        if gpp_column not in df.columns:
            log.warning("gpp_column %r not in dataframe; skipping light-response fit", gpp_column)
            return out
        for window_id, window in df.groupby("window_id"):
            try:
                (alpha, beta), _ = curve_fit(
                    light_response_curve,
                    window[par_column],
                    window[gpp_column],
                    bounds=self.light_response_bounds,
                    maxfev=self.maxfev,
                )
            except (ValueError, RuntimeError) as exc:
                log.warning("Light-response fit failed for window %s: %s", window_id, exc)
                continue
            out.loc[out["window_id"] == window_id, "alpha"] = alpha
            out.loc[out["window_id"] == window_id, "beta"] = beta
        return out


# ---------------------------------------------------------------------------
# Convenience top-level entrypoint matching the notebook flow
# ---------------------------------------------------------------------------

def add_time_derivatives(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...] = ("NEE", "Ta", "Rg"),
    datetime_column: str = "DateTime",
) -> pd.DataFrame:
    """Add `dCOL`, `COL_next`, and a `dt` column.

    Forward differences are the change to the next timestamp divided by the
    minutes to that timestamp (a per-minute rate), and `dt` is that same minute
    count — so the Euler-Maruyama step can recover the per-step increment as
    `dCOL * dt`.

    Note on the minute count: the original notebook used
    `TimeDiff.dt.components.minutes`, which returns only the *minutes component*
    of the gap — it is 0 for whole-hour gaps (silent div-by-zero -> the row is
    later dropped) and wrong for multi-hour gaps (e.g. 90 min -> 30). We use
    `total_seconds() / 60`, the true minute count, which fixes both and lets the
    stored `dt` be used per-row downstream.
    """
    out = df.copy()
    out["TimeDiff"] = out[datetime_column].shift(-1) - out[datetime_column]
    dt_minutes = out["TimeDiff"].dt.total_seconds() / 60.0
    out["dt"] = dt_minutes
    for col in columns:
        if col not in out.columns:
            log.warning("add_time_derivatives: column %r missing; skipping", col)
            continue
        out[f"d{col}"] = (out[col].shift(-1) - out[col]) / dt_minutes
    if "NEE" in out.columns:
        out["NEE_next"] = out["NEE"].shift(-1)
    return out.iloc[1:].reset_index(drop=True)


def estimate_for_site(
    input_path: str | Path,
    night_output_path: str | Path,
    *,
    day_output_path: str | Path | None = None,
    site: str | None = None,
    night_mask_strategy: str = "column",
    night_mask_column: str = "Day/Night",
    reco_column: str = "TER",
    estimator: ParameterEstimator | None = None,
) -> pd.DataFrame:
    """Run the full notebook pipeline on a single site, end-to-end.

    Args:
        input_path: parquet file from `SitePreprocessor.process_file`.
        night_output_path: where to write final_night_data.parquet.
        day_output_path: optional; if set, also fit and write day parameters.
        site: required if `night_mask_strategy='astral'`.
        night_mask_strategy: 'column' (precomputed Day/Night boolean) or
            'astral' (computed from sunrise/sunset).
        night_mask_column: column name when strategy='column'.
        reco_column: respiration column to fit; falls back to NEE if missing.
        estimator: optional pre-configured ParameterEstimator. Default
            constructs one with `config.py` thresholds.

    Returns:
        The night-fit dataframe (with E0, rb).
    """
    estimator = estimator or ParameterEstimator()

    df = pd.read_parquet(input_path)
    df = df.replace(NAN_SENTINEL, np.nan)
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df["DateTime"]).dt.date
    df = add_time_derivatives(df)

    if night_mask_strategy == "column":
        night_mask = make_night_mask_from_column(df, column=night_mask_column)
    elif night_mask_strategy == "astral":
        if site is None:
            raise ValueError("`site` is required for night_mask_strategy='astral'")
        night_mask = make_night_mask_from_astral(df, site=site)
    else:
        raise ValueError(f"Unknown night_mask_strategy: {night_mask_strategy!r}")

    night_results = estimator.fit_night(
        df,
        night_mask=night_mask,
        reco_column=reco_column,
    )
    Path(night_output_path).parent.mkdir(parents=True, exist_ok=True)
    night_results.to_parquet(night_output_path)
    log.info("Wrote night results to %s (%d rows)", night_output_path, len(night_results))

    if day_output_path is not None:
        day_results = estimator.fit_day(
            df,
            night_results,
            day_mask=~night_mask,
        )
        Path(day_output_path).parent.mkdir(parents=True, exist_ok=True)
        day_results.to_parquet(day_output_path)
        log.info("Wrote day results to %s (%d rows)", day_output_path, len(day_results))

    return night_results
