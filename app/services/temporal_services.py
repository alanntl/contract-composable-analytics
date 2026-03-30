"""
Contract-Composable Analytics Temporal Services - DateTime and Time-Series Features
============================================================
This module provides reusable services for temporal feature engineering.
Use these for any competition with datetime or time-series data.

Usage:
    from services.temporal_services import extract_datetime_features, create_lag_features
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from functools import wraps


def contract(inputs=None, outputs=None, params=None, description=None, tags=None, version="1.0.0"):
    """Service contract decorator for Contract-Composable Analytics services."""
    def decorator(func):
        func._contract = {
            'inputs': inputs or {},
            'outputs': outputs or {},
            'params': params or {},
            'description': description or func.__doc__,
            'tags': tags or [],
            'version': version
        }
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._contract = func._contract
        return wrapper
    return decorator


# =============================================================================
# DateTime Feature Extraction
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "datetime_column": "Column containing datetime values",
        "features": "List of features to extract",
        "add_cyclical": "Add sin/cos cyclical encoding",
        "drop_original": "Drop original datetime column"
    },
    description="Extract datetime components as features",
    tags=["temporal", "feature-engineering", "datetime", "generic"]
)
def extract_datetime_features(
    data: str,
    output: str,
    datetime_column: str = 'datetime',
    features: List[str] = None,
    add_cyclical: bool = False,
    drop_original: bool = False
) -> Dict[str, str]:
    """
    Extract temporal features from a datetime column.

    Works with: bike-sharing, store-sales, taxi duration, forecasting, etc.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        datetime_column: Name of datetime column
        features: List of features to extract. Options:
            - 'year', 'month', 'day', 'hour', 'minute', 'second'
            - 'dayofweek', 'dayofyear', 'weekofyear', 'quarter'
            - 'is_weekend', 'is_month_start', 'is_month_end'
        add_cyclical: Add sin/cos encoding for cyclical features (hour, day, month)
        drop_original: Drop the original datetime column after extraction

    Returns:
        Dict with output path
    """
    df = pd.read_csv(data)

    if datetime_column not in df.columns:
        df.to_csv(output, index=False)
        return {'data': output}

    # Convert to datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Default features
    if features is None:
        features = ['year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend']

    # Extract features
    dt = df[datetime_column]

    feature_map = {
        'year': lambda x: x.dt.year,
        'month': lambda x: x.dt.month,
        'day': lambda x: x.dt.day,
        'hour': lambda x: x.dt.hour,
        'minute': lambda x: x.dt.minute,
        'second': lambda x: x.dt.second,
        'dayofweek': lambda x: x.dt.dayofweek,
        'dayofyear': lambda x: x.dt.dayofyear,
        'weekofyear': lambda x: x.dt.isocalendar().week.astype(int),
        'quarter': lambda x: x.dt.quarter,
        'is_weekend': lambda x: (x.dt.dayofweek >= 5).astype(int),
        'is_month_start': lambda x: x.dt.is_month_start.astype(int),
        'is_month_end': lambda x: x.dt.is_month_end.astype(int),
    }

    for feat in features:
        if feat in feature_map:
            df[feat] = feature_map[feat](dt)

    # Add cyclical encoding
    if add_cyclical:
        if 'hour' in features:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        if 'dayofweek' in features:
            df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        if 'month' in features:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        if 'day' in features:
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    if drop_original:
        df = df.drop(columns=[datetime_column])

    df.to_csv(output, index=False)
    return {'data': output}


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "target_column": "Column to create lags for",
        "lags": "List of lag periods",
        "group_columns": "Columns to group by before creating lags"
    },
    description="Create lag features for time series",
    tags=["temporal", "feature-engineering", "time-series", "generic"]
)
def create_lag_features(
    data: str,
    output: str,
    target_column: str = 'target',
    lags: List[int] = None,
    group_columns: List[str] = None
) -> Dict[str, str]:
    """
    Create lag features for time series prediction.

    Works with: m5-forecasting, store-sales, demand-forecasting, etc.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        target_column: Column to create lags for
        lags: List of lag periods (e.g., [1, 7, 28])
        group_columns: Columns to group by (e.g., ['store_id', 'item_id'])
    """
    df = pd.read_csv(data)

    if target_column not in df.columns:
        df.to_csv(output, index=False)
        return {'data': output}

    if lags is None:
        lags = [1, 7, 14, 28]

    if group_columns:
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df.groupby(group_columns)[target_column].shift(lag)
    else:
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)

    df.to_csv(output, index=False)
    return {'data': output}


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "target_column": "Column to compute rolling features for",
        "windows": "List of rolling window sizes",
        "agg_funcs": "Aggregation functions to apply",
        "group_columns": "Columns to group by"
    },
    description="Create rolling window features",
    tags=["temporal", "feature-engineering", "time-series", "generic"]
)
def create_rolling_features(
    data: str,
    output: str,
    target_column: str = 'target',
    windows: List[int] = None,
    agg_funcs: List[str] = None,
    group_columns: List[str] = None
) -> Dict[str, str]:
    """
    Create rolling window aggregate features.

    Works with: m5-forecasting, store-sales, demand-forecasting, etc.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        target_column: Column to compute rolling features for
        windows: List of window sizes (e.g., [7, 14, 28])
        agg_funcs: Functions to apply (e.g., ['mean', 'std', 'min', 'max'])
        group_columns: Columns to group by
    """
    df = pd.read_csv(data)

    if target_column not in df.columns:
        df.to_csv(output, index=False)
        return {'data': output}

    if windows is None:
        windows = [7, 14, 28]

    if agg_funcs is None:
        agg_funcs = ['mean', 'std']

    for window in windows:
        for func in agg_funcs:
            col_name = f'{target_column}_roll_{window}_{func}'

            if group_columns:
                df[col_name] = df.groupby(group_columns)[target_column].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).agg(func)
                )
            else:
                df[col_name] = df[target_column].shift(1).rolling(window, min_periods=1).agg(func)

    df.to_csv(output, index=False)
    return {'data': output}


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={
        "train_data": {"format": "csv"},
        "valid_data": {"format": "csv"}
    },
    params={
        "datetime_column": "Column for temporal ordering",
        "split_date": "Date to split on",
        "test_size": "Fraction for validation if no split_date"
    },
    description="Temporal train/validation split",
    tags=["temporal", "data-splitting", "time-series", "generic"]
)
def temporal_train_valid_split(
    data: str,
    train_output: str,
    valid_output: str,
    datetime_column: str = 'date',
    split_date: str = None,
    test_size: float = 0.2
) -> Dict[str, str]:
    """
    Split data temporally for time series validation.

    Ensures validation data is always after training data (no data leakage).

    Args:
        data: Path to input CSV
        train_output: Path for training data
        valid_output: Path for validation data
        datetime_column: Column with datetime for ordering
        split_date: Specific date to split on (YYYY-MM-DD)
        test_size: Fraction for validation if no split_date specified
    """
    df = pd.read_csv(data)

    if datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.sort_values(datetime_column)

    if split_date:
        split_dt = pd.to_datetime(split_date)
        train_df = df[df[datetime_column] < split_dt]
        valid_df = df[df[datetime_column] >= split_dt]
    else:
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        valid_df = df.iloc[split_idx:]

    train_df.to_csv(train_output, index=False)
    valid_df.to_csv(valid_output, index=False)

    return {'train_data': train_output, 'valid_data': valid_output}


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "target_column": "Column to compute expanding features for",
        "agg_funcs": "Aggregation functions to apply",
        "group_columns": "Columns to group by"
    },
    description="Create expanding window features",
    tags=["temporal", "feature-engineering", "time-series", "generic"]
)
def create_expanding_features(
    data: str,
    output: str,
    target_column: str = 'target',
    agg_funcs: List[str] = None,
    group_columns: List[str] = None
) -> Dict[str, str]:
    """
    Create expanding (cumulative) window features.

    Useful for running totals, cumulative averages, etc.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        target_column: Column to compute features for
        agg_funcs: Functions to apply (e.g., ['mean', 'sum', 'count'])
        group_columns: Columns to group by
    """
    df = pd.read_csv(data)

    if target_column not in df.columns:
        df.to_csv(output, index=False)
        return {'data': output}

    if agg_funcs is None:
        agg_funcs = ['mean', 'sum']

    for func in agg_funcs:
        col_name = f'{target_column}_expanding_{func}'

        if group_columns:
            df[col_name] = df.groupby(group_columns)[target_column].transform(
                lambda x: x.shift(1).expanding(min_periods=1).agg(func)
            )
        else:
            df[col_name] = df[target_column].shift(1).expanding(min_periods=1).agg(func)

    df.to_csv(output, index=False)
    return {'data': output}


# =============================================================================
# Service Registry
# =============================================================================

SERVICE_REGISTRY = {
    "extract_datetime_features": extract_datetime_features,
    "create_lag_features": create_lag_features,
}
