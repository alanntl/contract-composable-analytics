"""
NYC Taxi Trip Duration - Contract-Composable Analytics Services
========================================
Competition: https://www.kaggle.com/c/nyc-taxi-trip-duration
Problem Type: Regression
Target: trip_duration (seconds)
Metric: RMSLE (Root Mean Squared Logarithmic Error)

Services derived from top-3 solution notebook analysis:
- engineer_taxi_features: datetime extraction, haversine distance, bearing, flag encoding
- filter_rows_by_range: generic outlier removal by column value bounds
- log_transform_column: in-place log1p transform for skewed targets

Reused services (imported):
- drop_columns (preprocessing_services)
- split_data (preprocessing_services)
- train_lightgbm_regressor (regression_services)
- predict_regressor (regression_services)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services from common modules
from services.preprocessing_services import drop_columns, split_data, create_submission
from services.regression_services import train_lightgbm_regressor, predict_regressor


# =============================================================================
# HELPERS: Geospatial calculations
# =============================================================================

def _haversine_distance(lat1, lon1, lat2, lon2):
    """Compute haversine distance in km between two sets of lat/lon coordinates."""
    AVG_EARTH_RADIUS = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2):
    """Compute bearing (direction) in degrees between two sets of lat/lon coordinates."""
    lon_delta_rad = np.radians(lon2 - lon1)
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    y = np.sin(lon_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon_delta_rad)
    return np.degrees(np.arctan2(y, x))


# =============================================================================
# SERVICE: Engineer Taxi Features
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract datetime + geospatial features for taxi/ride data",
    tags=["feature-engineering", "datetime", "geospatial", "taxi", "generic"],
    version="1.0.0",
)
def engineer_taxi_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    datetime_column: str = "pickup_datetime",
    pickup_lat_column: str = "pickup_latitude",
    pickup_lon_column: str = "pickup_longitude",
    dropoff_lat_column: str = "dropoff_latitude",
    dropoff_lon_column: str = "dropoff_longitude",
    flag_column: str = "store_and_fwd_flag",
    drop_datetime_columns: Optional[List[str]] = None,
) -> str:
    """
    Engineer features for taxi/ride-duration prediction.

    Derived from top-3 Kaggle solution notebooks for NYC Taxi Trip Duration.
    All three solutions use: datetime extraction, haversine distance, bearing.

    Works with: any taxi/ride data with pickup_datetime and lat/lon coordinates.

    Features created:
    - hour, dayofweek, month, day, is_weekend, minute_of_day (datetime)
    - distance (haversine km), bearing (direction degrees)
    - store_and_fwd_flag encoded as 0/1

    Parameters:
        datetime_column: Column with pickup datetime
        pickup_lat_column: Pickup latitude column
        pickup_lon_column: Pickup longitude column
        dropoff_lat_column: Dropoff latitude column
        dropoff_lon_column: Dropoff longitude column
        flag_column: Binary flag column to label-encode (e.g. store_and_fwd_flag)
        drop_datetime_columns: Datetime columns to drop after extraction
    """
    df = _load_data(inputs["data"])
    n_original = len(df.columns)

    # --- DateTime features ---
    if datetime_column in df.columns:
        dt = pd.to_datetime(df[datetime_column])
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['day'] = dt.dt.day
        df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        df['minute_of_day'] = dt.dt.hour * 60 + dt.dt.minute

    # --- Geospatial features ---
    geo_cols = [pickup_lat_column, pickup_lon_column, dropoff_lat_column, dropoff_lon_column]
    if all(c in df.columns for c in geo_cols):
        df['distance'] = _haversine_distance(
            df[pickup_lat_column].values, df[pickup_lon_column].values,
            df[dropoff_lat_column].values, df[dropoff_lon_column].values,
        )
        df['bearing'] = _bearing(
            df[pickup_lat_column].values, df[pickup_lon_column].values,
            df[dropoff_lat_column].values, df[dropoff_lon_column].values,
        )

    # --- Encode binary flag ---
    if flag_column in df.columns:
        df[flag_column] = df[flag_column].map({'N': 0, 'Y': 1}).fillna(0).astype(int)

    # --- Drop datetime columns ---
    if drop_datetime_columns is None:
        drop_datetime_columns = ["pickup_datetime", "dropoff_datetime"]
    to_drop = [c for c in drop_datetime_columns if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    _save_data(df, outputs["data"])
    n_new = len(df.columns) - n_original + len(to_drop)
    return f"engineer_taxi_features: {len(df)} rows, {n_new} new features, dropped {len(to_drop)} datetime cols"


# =============================================================================
# SERVICE: Filter Rows by Range (generic)
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Filter rows where column values fall within specified bounds",
    tags=["preprocessing", "filtering", "outlier-removal", "generic"],
    version="1.0.0",
)
def filter_rows_by_range(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    column: str = "target",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> str:
    """
    Filter rows by column value bounds.

    Removes rows where column value is outside [min_value, max_value].
    Useful for outlier removal before training.

    Works with: any tabular data needing outlier filtering.

    Parameters:
        column: Column to filter on
        min_value: Minimum allowed value (None = no lower bound)
        max_value: Maximum allowed value (None = no upper bound)
    """
    df = _load_data(inputs["data"])
    n_before = len(df)

    if column not in df.columns:
        _save_data(df, outputs["data"])
        return f"filter_rows_by_range: column '{column}' not found, no filtering"

    mask = pd.Series(True, index=df.index)
    if min_value is not None:
        mask &= df[column] >= min_value
    if max_value is not None:
        mask &= df[column] <= max_value

    df = df[mask]
    n_removed = n_before - len(df)
    _save_data(df, outputs["data"])
    return f"filter_rows_by_range: {n_removed} rows removed ({n_before} -> {len(df)})"


# =============================================================================
# SERVICE: Log Transform Column (generic)
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Apply log1p transform to a column in-place",
    tags=["preprocessing", "transformation", "log", "regression", "generic"],
    version="1.0.0",
)
def log_transform_column(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    column: str = "target",
) -> str:
    """
    Apply np.log1p to a column in-place (replaces original values).

    Essential for RMSLE-scored competitions where the target is right-skewed.
    Use predict_regressor(log_target=True) to reverse with expm1.

    Works with: any regression target with positive skew (trip_duration, price, count).

    Parameters:
        column: Column to transform
    """
    df = _load_data(inputs["data"])

    if column not in df.columns:
        _save_data(df, outputs["data"])
        return f"log_transform_column: column '{column}' not found"

    original_mean = df[column].mean()
    df[column] = np.log1p(df[column].clip(lower=0))
    new_mean = df[column].mean()

    _save_data(df, outputs["data"])
    return f"log_transform_column: '{column}' mean {original_mean:.2f} -> {new_mean:.4f}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific services
    "engineer_taxi_features": engineer_taxi_features,
    "filter_rows_by_range": filter_rows_by_range,
    "log_transform_column": log_transform_column,
    # Reused from common modules
    "drop_columns": drop_columns,
    "split_data": split_data,
    "create_submission": create_submission,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "predict_regressor": predict_regressor,
}