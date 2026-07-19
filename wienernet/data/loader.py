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
from .rescale import rescale_to_timestep
from .features import (
    add_site_vector_for,
    add_time_vars,
    assemble_features,
    compute_noise_statistics,
    physics_nee_numpy,
    set_season_tag,
)
from .splits import split_data_by_site_fraction, split_data_by_site_holdout, split_data_by_year

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
    # Width of the scaled feature matrix alone (X.shape[1], i.e. drivers + time +
    # site). The full encoder input adds the flagged GT parts (bNEE/k/dTa); use
    # `wienernet.models.encoder_input_dim(feature_dim, inputs_cfg)` to size the
    # encoder when the input flags differ from the legacy all-on default.
    feature_dim: int = 0
    # Train-only per-feature stats for standardising the optional GT encoder
    # inputs (k=(E0, rb) and dTa) when `inputs.scale_extra_inputs` is on. Fed to
    # the model via `WienerNetModel.set_input_norm_stats(...)`.
    k_mean: np.ndarray | None = None
    k_std: np.ndarray | None = None
    dtemp_mean: float | None = None
    dtemp_std: float | None = None
    # Train-only per-target stds for the loss's optional MSE normalisation
    # (compute_losses target_scales). Keyed by loss-term name. Anchors like E0
    # (~112±42) otherwise dwarf mse_nee / mse_drift; scaling each MSE by its
    # target variance puts every term on a comparable O(1) footing.
    target_scales: dict[str, float] | None = None
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


def ordered_site_names(
    site_paths: dict[str, object] | Iterable[str],
    include_sites: Iterable[str] | None = None,
    exclude_sites: Iterable[str] = (),
) -> tuple[str, ...]:
    """Deterministic, sorted list of the sites a run trains on, from the data
    config alone (site_paths keys minus exclude, intersected with include).

    Used to fix a checkpoint-stable per-site parameter layout (e.g. the per-site
    analytical baseline's diffusion vector) identically at train and eval time,
    without threading site info through the run artifacts.
    """
    excl = set(exclude_sites or ())
    incl = set(include_sites) if include_sites else None
    names = [s for s in site_paths if s not in excl and (incl is None or s in incl)]
    return tuple(sorted(names))


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
    dt_column: str = "dt",
    time_step_k: int | None = None,
    split_strategy: str = "site_fraction",
    test_frac: float = 0.3,
    test_years: Iterable[int] = (),
    holdout_site: str | None = None,
    train_subsample_frac: float | None = None,
    train_subsample_seed: int = 0,
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

    # Time-scale preprocessing: rebuild the targets (NEE_{t+k}, dNEE, dTa, dt) at a
    # k-step horizon within each contiguous night, dropping rows whose k-step-ahead
    # crosses a night boundary. time_step_k=None -> native 30-min columns unchanged.
    if time_step_k is not None:
        before = len(combined)
        combined = rescale_to_timestep(
            combined, int(time_step_k),
            nee_target_column=nee_target_column, dtemp_column=dtemp_column,
            dnee_column=dnee_column, dt_column=dt_column,
            temperature_column=temperature_column,
        )
        log.info("rescaled to k=%d (%d min step): %d -> %d rows",
                 int(time_step_k), int(time_step_k) * 30, before, len(combined))

    # Physics diurnal temperature tendency: the smooth, predictable part of dT/dt is
    # the diurnal cooling, estimated as the per-site (month, hour-of-day) climatology of
    # the observed tendency. The turbulent remainder (dTa - dTa_diurnal) is left to the
    # noise. Derived from measured temperature only (always available, incl. gaps), so
    # it is a physics/climatology feature, not target leakage. Used by the state-space
    # increment-SDE variant as the drift's dT/dt.
    if "DateTime" in combined.columns:
        _dtp = pd.to_datetime(combined["DateTime"])
        combined["_m"] = _dtp.dt.month
        combined["_h"] = _dtp.dt.hour + _dtp.dt.minute / 60.0
        combined["dTa_diurnal"] = (
            combined.groupby(["site", "_m", "_h"])[dtemp_column].transform("mean")
        ).astype(np.float32)
        combined = combined.drop(columns=["_m", "_h"])
    else:
        log.warning("no DateTime column; dTa_diurnal falls back to observed dTa")
        combined["dTa_diurnal"] = combined[dtemp_column].astype(np.float32)

    # Train/test split
    if split_strategy == "site_fraction":
        train_df, test_df = split_data_by_site_fraction(
            combined, test_frac=test_frac, shuffle=shuffle_split, random_state=split_random_state,
        )
    elif split_strategy == "year":
        train_df, test_df = split_data_by_year(combined, test_years=test_years)
    elif split_strategy == "site_holdout":
        if not holdout_site:
            raise ValueError("split_strategy='site_holdout' requires holdout_site")
        train_df, test_df = split_data_by_site_holdout(combined, holdout_site=holdout_site)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy!r}")
    log.info("split sizes: train=%d  test=%d", len(train_df), len(test_df))

    # Optional training-data subsampling for a data-efficiency / learning-curve
    # sweep. NESTED (fixed seed -> a smaller fraction is a subset of a larger one)
    # so the curve is monotone in data seen. The test set is never subsampled.
    if train_subsample_frac is not None and 0 < float(train_subsample_frac) < 1.0:
        n = len(train_df)
        order = np.random.default_rng(train_subsample_seed).permutation(n)
        m = max(batch_size + 1, int(round(n * float(train_subsample_frac))))
        m = min(m, n)
        train_df = train_df.iloc[np.sort(order[:m])].reset_index(drop=True)
        log.info("subsampled train to frac=%.4f: %d -> %d rows (seed=%d, nested)",
                 float(train_subsample_frac), n, m, train_subsample_seed)

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

    # Train-only stats for standardising the optional GT encoder inputs (k, dTa)
    # when a run turns on inputs.scale_extra_inputs. Kept train-only (like the
    # empirical noise prior) so no test signal enters the input normalisation.
    # std floored to avoid divide-by-zero on a degenerate column.
    k_train = train_df[[e0_column, rb_column]].values.astype(np.float32)
    k_mean = k_train.mean(axis=0)
    k_std = np.where(k_train.std(axis=0) > 1e-8, k_train.std(axis=0), 1.0).astype(np.float32)
    dtemp_train = train_df[dtemp_column].values.astype(np.float32)
    dtemp_mean = float(dtemp_train.mean())
    dtemp_std = float(dtemp_train.std()) if dtemp_train.std() > 1e-8 else 1.0

    # Per-target stds for the loss MSE normalisation. Only the k anchors (E0~112,
    # rb~4) live on the REddyProc-parameter scale, orders of magnitude off the
    # flux scale (~1-3) — their raw MSE (~1e4 for E0) dominates the loss and
    # starves everything else. The other MSE targets (NEE~O(1), and the naturally
    # SMALL dTa/dNEE) are on the working scale already; normalising them to unit
    # variance would wrongly force the weak, near-unpredictable 30-min physics
    # terms up to parity with mse_nee and hurt the NEE fit (verified: seed-0
    # full-normalisation gave raw mse_nee ~2.97 vs ~2.66 for E0/rb-only). So we
    # scale ONLY E0/rb; mse_nee/temp/drift keep their natural scale + config weight.
    target_scales = {
        "mse_E0": float(k_std[0]),
        "mse_rb": float(k_std[1]),
    }

    # Per-row dt (minutes to next timestamp) for the Euler step. Absent in older
    # parquets -> None, and the model falls back to its config dt.
    train_dt = (train_df[dt_column].values.astype(np.float32)
                if dt_column in train_df.columns else None)
    test_dt = (test_df[dt_column].values.astype(np.float32)
               if dt_column in test_df.columns else None)
    train_dtd = (train_df["dTa_diurnal"].values.astype(np.float32)
                 if "dTa_diurnal" in train_df.columns else None)
    test_dtd = (test_df["dTa_diurnal"].values.astype(np.float32)
                if "dTa_diurnal" in test_df.columns else None)

    # Build datasets
    train_dataset = ClimateDataset(
        X_train,
        train_df[[e0_column, rb_column]].values.astype(np.float32),
        train_df[temperature_column].values.astype(np.float32),
        train_df[dnee_column].values.astype(np.float32),
        train_df[boundary_nee_column].values.astype(np.float32),
        train_df[dtemp_column].values.astype(np.float32),
        train_df[nee_target_column].values.astype(np.float32),
        dt=train_dt,
        dT_diurnal=train_dtd,
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
        dt=test_dt,
        dT_diurnal=test_dtd,
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
        input_dim=X_train.shape[1] + 1 + 2,  # legacy default: X + bNEE + k
        feature_dim=X_train.shape[1],
        k_mean=k_mean,
        k_std=k_std,
        dtemp_mean=dtemp_mean,
        dtemp_std=dtemp_std,
        target_scales=target_scales,
        train_df=train_df if save_dataframes else None,
        test_df=test_df if save_dataframes else None,
        noise_residuals=noise_residuals,
    )
