"""Centralised constants for the data pipeline.

Every magic number or hardcoded list that used to live inside notebook cells
ends up here, so changing a threshold is a one-line edit reviewable in
git diff instead of a scavenger hunt.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Site geography (decimal degrees)
# Used by Parameter Estimation Other Sites for astral sunrise/sunset.
# Also used by the training notebook to build site_xyz unit-sphere coordinates.
# ---------------------------------------------------------------------------
SITES: dict[str, dict[str, float | str]] = {
    "rosedene":   {"lat": 52.533, "lon":  0.480, "timezone": "Europe/London", "country": "UK"},
    "redmere_1":  {"lat": 52.5,   "lon":  0.37,  "timezone": "Europe/London", "country": "UK"},
    "redmere_2":  {"lat": 52.5,   "lon":  0.37,  "timezone": "Europe/London", "country": "UK"},
    "great_fen":  {"lat": 52.484, "lon": -0.180, "timezone": "Europe/London", "country": "UK"},
    "wicken_fen": {"lat": 52.309, "lon":  0.299, "timezone": "Europe/London", "country": "UK"},
    "woodwalton": {"lat": 52.413, "lon": -0.2108421, "timezone": "Europe/London", "country": "UK"},
}

# ---------------------------------------------------------------------------
# FLUXNET2015 -> canonical column mapping for site preprocessing.
# Each canonical column has an ordered candidate list; the first column from
# the candidate list that exists in the raw data wins.
# ---------------------------------------------------------------------------
FLUXNET_COLUMN_MAPPING: dict[str, list[str]] = {
    "NEE":   ["NEE_VUT_REF", "NEE", "NEE_VUT_USTAR50", "NEE_VUT_USTAR05", "NEE_VUT_USTAR95"],
    "H":     ["H", "H_F_MDS"],
    "Tau":   ["TAU"],
    "LE":    ["LE", "LE_F_MDS"],
    "RH":    ["RH", "RH_F_MDS"],
    "VPD":   ["VPD", "VPD_F_MDS"],
    "Ustar": ["USTAR"],
    "Ta":    ["TA", "TA_F_MDS"],
}

# Radiation: derived from SW + LW components by default.
# Some sites only provide SWin; the preprocessing pipeline handles both modes.
RADIATION_COMPONENT_COLUMNS = ["SW_IN", "SW_OUT", "LW_IN", "LW_OUT"]

# FLUXNET timestamp format: YYYYMMDDHHMM
FLUXNET_TIMESTAMP_FORMAT = "%Y%m%d%H%M"

# Sentinel value used by FLUXNET to indicate missing data.
NAN_SENTINEL = -9999

# ---------------------------------------------------------------------------
# Lloyd-Taylor (Reichstein 2005) parameter fitting constants
# ---------------------------------------------------------------------------
# Reference temperature in degrees Celsius
LLOYD_TAYLOR_TREF = 10.0

# T0 offset. NOTE: the notebooks contain two conventions that look like
# typos for one another:
#
#   formula_A (used by `lloyd_taylor` in the parameter-estimation notebooks):
#       R = R_ref * exp(E0 * (1/(Tref + 46.02) - 1/(T + 46.02)))
#       => effectively T0 = -46.02  (standard Reichstein form, T in °C)
#
#   formula_B (used by `physics_nee` and by the night models in night/*.py):
#       NEE = rb * exp(E0 * (1/(Tref - T0) - 1/(T - T0)))   with T0 = 46.02
#       => the SIGN of T0 differs from formula_A
#
# These produce DIFFERENT respiration values for the same (E0, rb), so the
# fitted parameters do not faithfully reproduce the model's physics term.
# This is preserved here only to match existing behaviour pending an explicit
# decision (see audit notes). Both constants below; use only the appropriate
# one inside the formula you are reproducing.
LLOYD_TAYLOR_T0_PARAM_FIT = 46.02      # used as `+T0` in the partitioning notebooks
LLOYD_TAYLOR_T0_PHYSICS_MODEL = 46.02  # used as `-T0` in the night models

# Curve-fit bounds: scipy.optimize.curve_fit expects (lower_tuple, upper_tuple).
# Order of params is (Reco_ref, E0).
LLOYD_TAYLOR_BOUNDS = ((0.0, 50.0), (1100.0, 400.0))
LIGHT_RESPONSE_BOUNDS = ((0.0, 0.0), (0.22, 250.0))

# ---------------------------------------------------------------------------
# Window-grouping thresholds for parameter fitting
# ---------------------------------------------------------------------------
NIGHT_WINDOW_GAP_HOURS = 2
DAY_WINDOW_GAP_HOURS = 4
NIGHT_RADIATION_THRESHOLD = 10  # W/m^2; below this the timestamp counts as night
TEMPERATURE_RANGE_THRESHOLD = 5  # min spread °C within a window for fit to be valid
MIN_DATA_POINTS_PER_WINDOW = 6

# scipy.optimize.curve_fit's maxfev
CURVE_FIT_MAXFEV = 10000
