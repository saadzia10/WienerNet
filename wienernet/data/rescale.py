"""Rescale the training/eval targets to a coarser SDE step (a `k*30`-min horizon).

Each model is trained and evaluated at a single temporal step. For step `k` the
*input* row stays at time `t` (drivers, `E0`, `rb`, `bNEE` unchanged), but the
targets it predicts move to `t+k`:

    NEE_next -> NEE_{t+k}
    dNEE     -> (NEE_{t+k} - NEE_t) / (k*step)     # per-minute rate over the span
    dTa      -> (Ta_{t+k}  - Ta_t)  / (k*step)
    dt       -> k*step                              # Euler step size (minutes)

so the Euler-Maruyama step `bNEE + f*dt` reconstructs `NEE_{t+k}` exactly.

Night-crossover guard: after nighttime filtering the day gaps are removed, so a
`k`-step lookahead in row order could jump into the *next* night (~24 h later in
real time). We group each site into contiguous nights (a real-time gap greater
than `step` starts a new night) and keep a row only if its `t+k` sample lies in
the *same* night — within which consecutive rows are exactly `step` apart, so
`t+k` is genuinely `k*step` minutes ahead. Rows whose lookahead crosses a night
(or runs off the end) are dropped.
"""

from __future__ import annotations

import pandas as pd


def rescale_to_timestep(
    df: pd.DataFrame,
    k: int,
    *,
    step_minutes: int = 30,
    site_column: str = "site",
    datetime_column: str = "DateTime",
    nee_column: str = "NEE",
    temperature_column: str = "Ta",
    nee_target_column: str = "NEE_next",
    dtemp_column: str = "dTa",
    dnee_column: str = "dNEE",
    dt_column: str = "dt",
) -> pd.DataFrame:
    """Rebuild the increment targets at a `k`-step horizon within each night.

    Returns a new frame with `nee_target_column`, `dnee_column`, `dtemp_column`
    and `dt_column` overwritten and night-crossover / off-the-end rows dropped.
    `k=1` reproduces the native 30-min step but under the same within-night rule
    (so a scale sweep is consistent across `k`).
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"time step k must be >= 1, got {k}")
    span = float(k * step_minutes)

    if site_column not in df.columns:
        groups = [(None, df)]
    else:
        groups = list(df.groupby(site_column, sort=False))

    kept: list[pd.DataFrame] = []
    for _, g in groups:
        g = g.sort_values(datetime_column).reset_index(drop=True)
        t = pd.to_datetime(g[datetime_column])
        gap_prev = (t - t.shift(1)).dt.total_seconds() / 60.0
        # A real-time gap larger than one step starts a new contiguous night.
        night_id = (gap_prev > step_minutes).cumsum()
        # t and t+k share a night iff no boundary lies between them; within a
        # night all steps are `step` minutes, so t+k is exactly k*step ahead.
        same_night = night_id.eq(night_id.shift(-k))

        nee_ahead = g[nee_column].shift(-k)
        ta_ahead = g[temperature_column].shift(-k)
        g = g.copy()
        g[nee_target_column] = nee_ahead
        g[dnee_column] = (nee_ahead - g[nee_column]) / span
        g[dtemp_column] = (ta_ahead - g[temperature_column]) / span
        g[dt_column] = span

        kept.append(g[same_night.fillna(False).to_numpy()])

    return pd.concat(kept, ignore_index=True)
