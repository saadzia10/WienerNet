"""End-to-end data loading: parquet on disk → train/test DataLoaders.

Replaces ~25 cells of the training notebook (cell 7 to cell 32) with a single
`build_dataloaders(cfg)` call.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data_pipeline.config import SITES

from .dataset import ClimateDataset
from .features import (
    add_site_vector_for,
    add_time_vars,
    assemble_features,
    compute_noise_statistics,
    physics_nee_numpy,
    set_season_tag,
)
from .splits import split_data_by_site_fraction, split_data_by_year

log = logging.getLogger("wienernet.data.loader")


# Default driver columns used by the published training pipeline
DEFAULT_DRIVERS: tuple[str, ...] = (
    "Ta", "H", "Tau", "RH", "VPD", "Rg", "Ustar", "Tsoil1",
)


# ---------------------------------------------------------------------------
# DataLoader bundle returned from build_dataloaders
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Everything build_dataloaders returns. Easier to evolve than a tuple."""

    train_loader: DataLoader
    test_loader: DataLoader
    train_dataset: ClimateDataset
    test_dataset: ClimateDataset
    scaler: StandardScaler
    noise_mu: float
    noise_std: float
    feature_columns: list[str]
    input_dim: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    # Train-only residual pool (NEE - NEE_phy) for the empirical MMD-noise prior.
    # Train-only (unlike the leaky combined noise_mu/noise_std) so no test signal
    # enters the training objective.
    noise_residuals: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Step 1: load parquet files per site
# ---------------------------------------------------------------------------

def load_site_parquets(
    site_paths: dict[str, str | Path],
    *,
    drop_unknown_sites: bool = False,
) -> dict[str, pd.DataFrame]:
    """Read each `<site_name>: parquet_path` mapping, return loaded DataFrames.

    Each DataFrame is augmented with a `site` column equal to the site name,
    enabling downstream per-site splitting and balanced sampling.
    """
    out: dict[str, pd.DataFrame] = {}
    for site_name, path in site_paths.items():
        if site_name not in SITES and not drop_unknown_sites:
            log.warning(
                "site %r not in SITES registry; site_xyz features will fail unless added", site_name
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found for site {site_name!r}: {path}")
        df = pd.read_parquet(path)
        df["site"] = site_name
        out[site_name] = df
        log.info("loaded site %r: %d rows from %s", site_name, len(df), path)
    return out


# ---------------------------------------------------------------------------
# Step 2: add features, concatenate sites, drop NaNs
# ---------------------------------------------------------------------------

def prepare_combined_frame(
    site_frames: dict[str, pd.DataFrame],
    *,
    drivers: Iterable[str] = DEFAULT_DRIVERS,
    include_sites: Iterable[str] | None = None,
    exclude_sites: Iterable[str] = (),
    add_features: bool = True,
    nighttime_only: bool = True,
    night_radiation_threshold: float = 20.0,
    require_columns: Iterable[str] = ("NEE", "dNEE", "Ta", "dTa", "E0", "rb"),
) -> pd.DataFrame:
    """Concatenate per-site frames, add features, drop rows missing required columns.

    Args:
        site_frames: as returned by `load_site_parquets`.
        drivers: meteorological feature columns required for the model.
        include_sites: if set, only these sites are concatenated (others dropped).
        exclude_sites: subset of sites to leave out (e.g. paper's data4, data5).
        add_features: if False, only concatenate (assumes features already present).
        nighttime_only: if True, filter to Rg < night_radiation_threshold. The
            existing pipeline pre-filters when producing final_night_data.parquet,
            so this defaults to True but is essentially a noop on that input.
        require_columns: rows missing any of these get dropped.
    """
    selected: dict[str, pd.DataFrame] = {}
    include = set(include_sites) if include_sites is not None else None
    excluded = set(exclude_sites)
    for name, frame in site_frames.items():
        if include is not None and name not in include:
            log.info("skipping site %r (not in include_sites)", name)
            continue
        if name in excluded:
            log.info("skipping site %r (in exclude_sites)", name)
            continue
        selected[name] = frame

    if not selected:
        raise ValueError("No sites left after include/exclude filtering")

    enriched_parts: list[pd.DataFrame] = []
    for name, frame in selected.items():
        f = frame.copy()
        if add_features:
            f = add_time_vars(f)
            f = set_season_tag(f)
            f = add_site_vector_for(f, name)
        enriched_parts.append(f)

    combined = pd.concat(enriched_parts, ignore_index=True)
    log.info("combined frame: %d rows across %d sites", len(combined), len(selected))

    # Filter to nighttime
    if nighttime_only and "Rg" in combined.columns:
        before = len(combined)
        combined = combined[combined["Rg"] < night_radiation_threshold].copy()
        log.info("nighttime filter (Rg < %g): %d -> %d rows",
                 night_radiation_threshold, before, len(combined))

    # Drop rows without required columns (typically dNEE NaN for first/last sample)
    missing_required = [c for c in require_columns if c not in combined.columns]
    if missing_required:
        raise KeyError(f"Required columns missing from combined frame: {missing_required}")
    before = len(combined)
    combined = combined.dropna(subset=list(require_columns)).reset_index(drop=True)
    log.info("dropna over %s: %d -> %d rows", list(require_columns), before, len(combined))

    return combined


# ---------------------------------------------------------------------------
# Step 3: produce X arrays + scaler
# ---------------------------------------------------------------------------

def fit_scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    drivers: Iterable[str] = DEFAULT_DRIVERS,
    include_season: bool = True,
    include_time_vars: bool = True,
    include_site_xyz: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """Fit StandardScaler on train, transform both, return (X_train, X_test, scaler, cols)."""
    X_train_raw, cols = assemble_features(
        train_df,
        drivers=drivers,
        include_season=include_season,
        include_time_vars=include_time_vars,
        include_site_xyz=include_site_xyz,
    )
    X_test_raw, _ = assemble_features(
        test_df,
        drivers=drivers,
        include_season=include_season,
        include_time_vars=include_time_vars,
        include_site_xyz=include_site_xyz,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, scaler, cols


def save_scaler(scaler: StandardScaler, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("saved scaler to %s", path)
    return path


def load_scaler(path: str | Path) -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def build_dataloaders(
    site_paths: dict[str, str | Path],
    *,
    drivers: Iterable[str] = DEFAULT_DRIVERS,
    include_sites: Iterable[str] | None = None,
    exclude_sites: Iterable[str] = (),
    nighttime_only: bool = True,
    night_radiation_threshold: float = 20.0,
    nee_target_column: str = "NEE_next",
    boundary_nee_column: str = "NEE",
    e0_column: str = "E0",
    rb_column: str = "rb",
    temperature_column: str = "Ta",
    dtemp_column: str = "dTa",
    dnee_column: str = "dNEE",
    split_strategy: str = "site_fraction",
    test_frac: float = 0.3,
    test_years: Iterable[int] = (),
    shuffle_split: bool = False,
    split_random_state: int = 42,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    save_scaler_path: str | Path | None = None,
    save_dataframes: bool = True,
) -> DataBundle:
    """Top-level data loading.

    Reads all sites, builds the combined frame, splits, scales, and returns
    DataLoaders ready for training. A single function call so train.py stays
    a clean Hydra entrypoint.
    """
    site_frames = load_site_parquets(site_paths)
    # Require ALL driver columns to be non-NaN, plus targets/conditioning.
    # The original notebook silently propagated NaN through StandardScaler,
    # producing NaN losses without dropping the bad rows.
    require_cols = tuple(drivers) + (
        "NEE", dnee_column, temperature_column, dtemp_column, e0_column, rb_column,
    )
    combined = prepare_combined_frame(
        site_frames,
        drivers=drivers,
        include_sites=include_sites,
        exclude_sites=exclude_sites,
        nighttime_only=nighttime_only,
        night_radiation_threshold=night_radiation_threshold,
        require_columns=require_cols,
    )

    # Add NEE_next if not already present
    if nee_target_column not in combined.columns:
        log.info("computing %r as NEE.shift(-1) per site", nee_target_column)
        combined[nee_target_column] = combined.groupby("site")["NEE"].shift(-1)
        combined = combined.dropna(subset=[nee_target_column]).reset_index(drop=True)

    # Train/test split
    if split_strategy == "site_fraction":
        train_df, test_df = split_data_by_site_fraction(
            combined, test_frac=test_frac, shuffle=shuffle_split, random_state=split_random_state,
        )
    elif split_strategy == "year":
        train_df, test_df = split_data_by_year(combined, test_years=test_years)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}")
    log.info("split sizes: train=%d  test=%d", len(train_df), len(test_df))

    # Feature scaling
    X_train, X_test, scaler, feature_cols = fit_scale_features(
        train_df, test_df, drivers=drivers,
    )
    log.info("feature matrix: train %s, test %s, cols=%d", X_train.shape, X_test.shape, len(feature_cols))

    if save_scaler_path is not None:
        save_scaler(scaler, save_scaler_path)

    # Noise statistics (from all combined data, matches the notebook)
    nee_phys = physics_nee_numpy(combined[e0_column].values,
                                 combined[rb_column].values,
                                 combined[temperature_column].values)
    noise_mu, noise_std = compute_noise_statistics(combined["NEE"].values, nee_phys)
    log.info("noise stats: mu=%.4f  std=%.4f", noise_mu, noise_std)

    # Train-only residual pool for the empirical MMD-noise prior (step 2). Kept
    # separate from the (combined) noise_mu/noise_std above so the empirical
    # target carries no test signal.
    nee_phys_train = physics_nee_numpy(train_df[e0_column].values,
                                       train_df[rb_column].values,
                                       train_df[temperature_column].values)
    noise_residuals = (train_df["NEE"].values - nee_phys_train).astype(np.float32)
    noise_residuals = noise_residuals[np.isfinite(noise_residuals)]

    # Build datasets
    train_dataset = ClimateDataset(
        X_train,
        train_df[[e0_column, rb_column]].values.astype(np.float32),
        train_df[temperature_column].values.astype(np.float32),
        train_df[dnee_column].values.astype(np.float32),
        train_df[boundary_nee_column].values.astype(np.float32),
        train_df[dtemp_column].values.astype(np.float32),
        train_df[nee_target_column].values.astype(np.float32),
        site_ids=train_df["site"].tolist() if "site" in train_df.columns else None,
    )
    test_dataset = ClimateDataset(
        X_test,
        test_df[[e0_column, rb_column]].values.astype(np.float32),
        test_df[temperature_column].values.astype(np.float32),
        test_df[dnee_column].values.astype(np.float32),
        test_df[boundary_nee_column].values.astype(np.float32),
        test_df[dtemp_column].values.astype(np.float32),
        test_df[nee_target_column].values.astype(np.float32),
        site_ids=test_df["site"].tolist() if "site" in test_df.columns else None,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # FIX: notebook used shuffle=True which broke plot alignment
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return DataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        scaler=scaler,
        noise_mu=noise_mu,
        noise_std=noise_std,
        feature_columns=feature_cols,
        input_dim=X_train.shape[1] + 1 + 2,  # X + bNEE + k
        train_df=train_df if save_dataframes else None,
        test_df=test_df if save_dataframes else None,
        noise_residuals=noise_residuals,
    )
