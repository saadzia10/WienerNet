"""Data loading, feature engineering, splits."""

from .dataset import BalancedBatchSampler, ClimateDataset
from .features import (
    SITE_VECTOR_COLUMNS,
    TIME_VAR_COLUMNS,
    add_site_vector,
    add_site_vector_for,
    add_time_vars,
    assemble_features,
    compute_noise_statistics,
    lat_lon_to_unit_vector,
    physics_nee_numpy,
    set_season_tag,
)
from .loader import (
    DEFAULT_DRIVERS,
    DataBundle,
    build_dataloaders,
    fit_scale_features,
    load_scaler,
    load_site_parquets,
    ordered_site_names,
    prepare_combined_frame,
    save_scaler,
)
from .rescale import rescale_to_timestep
from .splits import split_data_by_site_fraction, split_data_by_site_holdout, split_data_by_year

__all__ = [
    # dataset
    "ClimateDataset",
    "BalancedBatchSampler",
    # features
    "add_time_vars",
    "set_season_tag",
    "add_site_vector",
    "add_site_vector_for",
    "lat_lon_to_unit_vector",
    "assemble_features",
    "compute_noise_statistics",
    "physics_nee_numpy",
    "TIME_VAR_COLUMNS",
    "SITE_VECTOR_COLUMNS",
    # splits
    "split_data_by_site_fraction",
    "split_data_by_site_holdout",
    "split_data_by_year",
    # loader
    "DataBundle",
    "build_dataloaders",
    "rescale_to_timestep",
    "load_site_parquets",
    "ordered_site_names",
    "prepare_combined_frame",
    "fit_scale_features",
    "save_scaler",
    "load_scaler",
    "DEFAULT_DRIVERS",
]
