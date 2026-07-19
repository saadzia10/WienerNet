"""Train/test splitting strategies.

The default in the existing training notebook is `split_data_by_site_fraction`,
which holds out a contiguous tail of each site for testing (temporal split,
preserves site distribution in both halves).
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger("wienernet.data.splits")


def split_data_by_site_fraction(
    df: pd.DataFrame,
    *,
    site_column: str = "site",
    test_frac: float = 0.3,
    shuffle: bool = False,
    random_state: int = 42,
    test_first: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-site fractional split. Each site contributes `test_frac` to test.

    Args:
        site_column: column identifying each site. Rows without a site are
            treated as one synthetic group.
        test_frac: fraction of each site's rows that go to test.
        shuffle: if True, shuffle rows within each site before splitting (use
            with random_state for reproducibility). If False, preserve order
            (typical for time-series splits).
        random_state: passed to numpy when shuffling.
        test_first: if True, the test set is the first `test_frac` of each
            site (used if a site's tail is too short/noisy and the early
            period is more representative). Default False = test is the tail.

    Returns:
        (train_df, test_df). Each is a reset-index copy.
    """
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    rng = np.random.default_rng(random_state)
    if site_column not in df.columns:
        log.warning(
            "site_column %r not in dataframe; using a single synthetic site", site_column
        )
        groups: Iterable[tuple[object, pd.DataFrame]] = [("__all__", df)]
    else:
        groups = df.groupby(site_column, sort=False)

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for site, group in groups:
        group = group.copy()
        if shuffle:
            order = rng.permutation(len(group))
            group = group.iloc[order]
        n_test = int(round(len(group) * test_frac))
        if n_test == 0 or n_test == len(group):
            log.warning("site %r split produced empty train or test (n=%d, n_test=%d)", site, len(group), n_test)
        if test_first:
            test_parts.append(group.iloc[:n_test])
            train_parts.append(group.iloc[n_test:])
        else:
            train_parts.append(group.iloc[:-n_test] if n_test > 0 else group)
            test_parts.append(group.iloc[-n_test:] if n_test > 0 else group.iloc[0:0])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def split_data_by_site_holdout(
    df: pd.DataFrame,
    *,
    holdout_site: str,
    site_column: str = "site",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-site-out: test = all rows of `holdout_site`, train = every other
    site. Tests extrapolation to a completely unseen site (the physics-vs-black-box
    generalisation experiment). No rows of the held-out site enter training.
    """
    if site_column not in df.columns:
        raise ValueError(f"site_holdout split needs a {site_column!r} column")
    is_test = df[site_column].astype(str) == str(holdout_site)
    train_df = df.loc[~is_test].reset_index(drop=True)
    test_df = df.loc[is_test].reset_index(drop=True)
    if len(test_df) == 0:
        raise ValueError(
            f"holdout_site {holdout_site!r} has no rows; available: "
            f"{sorted(df[site_column].astype(str).unique())}"
        )
    if len(train_df) == 0:
        raise ValueError(f"holdout_site {holdout_site!r} left no training rows")
    return train_df, test_df


def split_data_by_year(
    df: pd.DataFrame,
    *,
    datetime_column: str = "DateTime",
    test_years: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out specified calendar years for testing.

    Used when the analysis cares about year-level generalisation (e.g. the
    paper's main result: train 2012-2017, test 2018-2019).
    """
    dt = pd.to_datetime(df[datetime_column])
    test_years_set = set(int(y) for y in test_years)
    is_test = dt.dt.year.isin(test_years_set)
    train_df = df.loc[~is_test].reset_index(drop=True)
    test_df = df.loc[is_test].reset_index(drop=True)
    return train_df, test_df
