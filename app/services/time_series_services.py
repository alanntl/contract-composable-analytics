"""
Time Series Services - SLEGO Common Module
============================================

Generic time series services for lag features, temporal splitting,
rolling features, and recursive prediction.

Services:
  Features: create_lag_features, create_rolling_features
  Splitting: split_temporal
  Prediction: predict_recursive
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract


# Import shared I/O utilities
from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {
                "type": "tabular",
                "allow_missing": True,
            }
        },
    },
    outputs={
        "data_with_lags": {
            "format": "parquet",
            "schema": {
                "type": "tabular",
                "allow_missing": True,
            }
        },
    },
    description="Create lag and rolling mean features grouped by entity",
    tags=["feature-engineering", "lag-features", "time-series"],
    version="1.0.0",
)
def create_lag_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    group_column: str = "id",
    time_column: str = "d",
    value_column: str = "demand",
    lags: List[int] = None,
    windows: List[int] = None,
    drop_na: bool = True,
) -> str:
    """
    Create lag and rolling mean features by group.

    For each entity (group_column), creates:
    - lag_t{L}: Value shifted by L time steps
    - rolling_mean_lag{L}_w{W}: Rolling mean of W periods on the lagged value

    Args:
        group_column: Column identifying each entity/series (default: "id")
        time_column: Column representing temporal ordering (default: "d")
        value_column: Column whose lags to compute (default: "demand")
        lags: List of lag offsets (default: [7, 28])
        windows: List of rolling window sizes (default: [7, 28])
        drop_na: Whether to drop rows with NaN in lag features (default: True)
    """
    lags = lags or [7, 28]
    windows = windows or [7, 28]

    # Auto-detect input format
    data_path = inputs["data"]
    df = _load_data(data_path)
    original_cols = df.shape[1]

    # Sort by group and time for proper shifting
    df = df.sort_values([group_column, time_column])

    # Create lag features
    for lag in lags:
        col_name = f"lag_t{lag}"
        df[col_name] = (
            df.groupby(group_column)[value_column]
            .shift(lag)
            .astype("float32")
        )

        # Rolling means on lagged values
        for w in windows:
            rolling_col = f"rolling_mean_lag{lag}_w{w}"
            df[rolling_col] = (
                df.groupby(group_column)[col_name]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
                .astype("float32")
            )

    # Drop rows where lag features are NaN (early rows per group)
    if drop_na:
        lag_cols = [f"lag_t{lag}" for lag in lags]
        df = df.dropna(subset=lag_cols)

    _save_data(df, outputs["data_with_lags"])

    new_cols = df.shape[1] - original_cols
    return (
        f"create_lag_features: {original_cols} -> {df.shape[1]} columns "
        f"(+{new_cols}), lags={lags}, windows={windows}, {len(df)} rows"
    )


@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {
                "type": "tabular",
                "allow_missing": True,
            }
        },
    },
    outputs={
        "data_with_rolling": {
            "format": "parquet",
            "schema": {
                "type": "tabular",
                "allow_missing": True,
            }
        },
    },
    description="Create rolling window statistics (mean, std, min, max) by group",
    tags=["feature-engineering", "rolling-features", "time-series"],
    version="1.0.0",
)
def create_rolling_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    group_column: str = "id",
    time_column: str = "d",
    value_column: str = "demand",
    windows: List[int] = None,
    functions: List[str] = None,
) -> str:
    """
    Create rolling window statistics by group.

    For each entity and window size, computes the specified aggregate
    functions over a trailing window. Columns are named as
    rolling_{func}_{value_column}_w{W}.

    Args:
        group_column: Column identifying each entity/series (default: "id")
        time_column: Column representing temporal ordering (default: "d")
        value_column: Column to compute rolling stats on (default: "demand")
        windows: List of rolling window sizes (default: [7, 14, 28])
        functions: List of aggregate functions (default: ["mean", "std", "min", "max"])
    """
    windows = windows or [7, 14, 28]
    functions = functions or ["mean", "std", "min", "max"]

    # Auto-detect input format
    data_path = inputs["data"]
    df = _load_data(data_path)
    original_cols = df.shape[1]

    # Sort by group and time for proper rolling
    df = df.sort_values([group_column, time_column])

    grouped = df.groupby(group_column)[value_column]

    for w in windows:
        for func in functions:
            col_name = f"rolling_{func}_{value_column}_w{w}"
            df[col_name] = (
                grouped
                .transform(lambda x: x.rolling(w, min_periods=1).agg(func))
                .astype("float32")
            )

    _save_data(df, outputs["data_with_rolling"])

    new_cols = df.shape[1] - original_cols
    return (
        f"create_rolling_features: {original_cols} -> {df.shape[1]} columns "
        f"(+{new_cols}), windows={windows}, functions={functions}, {len(df)} rows"
    )


# =============================================================================
# DATA SPLITTING
# =============================================================================

@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {
                "type": "tabular",
                "allow_missing": True,
            }
        },
    },
    outputs={
        "train_data": {
            "format": "parquet",
            "schema": {
                "type": "tabular",
                "allow_missing": False,
                "min_rows": 1,
            }
        },
        "valid_data": {
            "format": "parquet",
            "schema": {
                "type": "tabular",
                "allow_missing": False,
                "min_rows": 1,
            }
        },
        "test_data": {
            "format": "parquet",
            "schema": {
                "type": "tabular",
                "allow_missing": True,  # Future days may have NaN target
            }
        },
    },
    description="Split data temporally for time series validation",
    tags=["data-handling", "split", "time-series"],
    version="1.0.0",
)
def split_temporal(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    time_column: str = "d",
    first_pred_day: int = 1942,
    valid_size: float = 0.1,
    buffer_days: int = 56,
    random_state: int = 42,
) -> str:
    """
    Split data temporally for time series validation.

    For time series we must respect temporal ordering:
    - Test: days >= first_pred_day - buffer_days (includes lag buffer for
      recursive prediction so lag features can be computed for forecast days)
    - Train/Valid: historical days < first_pred_day, split randomly
      (random split within historical data is acceptable for tree models)

    Args:
        time_column: Column representing temporal ordering (default: "d")
        first_pred_day: First day of the forecast horizon (default: 1942)
        valid_size: Fraction of historical data to hold out for validation (default: 0.1)
        buffer_days: Number of days before first_pred_day to include in test
                     set so that lag features are available (default: 56)
        random_state: Random seed for train/valid split (default: 42)
    """
    from sklearn.model_selection import train_test_split

    # Auto-detect input format
    data_path = inputs["data"]
    df = _load_data(data_path)

    # Test set: future days plus buffer for lag features
    test = df[df[time_column] >= first_pred_day - buffer_days].copy()

    # Historical data for train/valid
    historical = df[df[time_column] < first_pred_day].copy()

    # Random split on historical (acceptable for tree models)
    train, valid = train_test_split(
        historical,
        test_size=valid_size,
        shuffle=True,
        random_state=random_state,
    )

    _save_data(train, outputs["train_data"])
    _save_data(valid, outputs["valid_data"])
    _save_data(test, outputs["test_data"])

    return (
        f"split_temporal: train={len(train)}, valid={len(valid)}, "
        f"test={len(test)} (buffer={buffer_days}d before day {first_pred_day})"
    )


# =============================================================================
# PREDICTION
# =============================================================================

@contract(
    inputs={
        "model": {
            "format": "pickle",
            "required": True,
            "schema": {
                "type": "artifact",
                "artifact_type": "any",
            }
        },
        "test_data": {
            "format": "parquet",
            "required": True,
            "schema": {
                "type": "tabular",
                "allow_missing": True,
            }
        },
    },
    outputs={
        "predictions": {
            "format": "parquet",
            "schema": {
                "type": "tabular",
                "allow_missing": True,
            }
        },
    },
    description="Generate recursive multi-step predictions with lag feature updates",
    tags=["inference", "prediction", "time-series", "recursive"],
    version="1.0.0",
)
def predict_recursive(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    forecast_horizon: int = 28,
    first_pred_day: int = 1942,
    id_column: str = "id",
    time_column: str = "d",
    value_column: str = "demand",
) -> str:
    """
    Generate recursive multi-step predictions with lag feature updates.

    For each day in the forecast horizon:
    1. Select the rows for the current day
    2. Predict using the model and current feature values
    3. Write predictions into the value column
    4. Update lag features for future days so that subsequent predictions
       can use today's predicted value as a lag input

    This is critical for time series where lag features depend on previous
    predictions (autoregressive / recursive forecasting).

    Args:
        forecast_horizon: Number of days to forecast (default: 28)
        first_pred_day: First day of the forecast horizon (default: 1942)
        id_column: Column identifying each entity/series (default: "id")
        time_column: Column representing temporal ordering (default: "d")
        value_column: Column to predict (default: "demand")
    """
    with open(inputs["model"], "rb") as f:
        model = pickle.load(f)

    test = _load_data(inputs["test_data"])

    # Determine feature columns (everything except id, time, and target)
    exclude_cols = {id_column, time_column, value_column}
    feature_cols = [c for c in test.columns if c not in exclude_cols]

    # Discover which lag columns exist and their offsets
    lag_cols = [
        c for c in feature_cols
        if c.startswith("lag_t") and "rolling" not in c
    ]
    lag_values = sorted(set(
        int(c.split("lag_t")[1].split("_")[0])
        for c in lag_cols
    ))

    # Recursive prediction: day by day
    for day in range(first_pred_day, first_pred_day + forecast_horizon):
        day_mask = test[time_column] == day

        if day_mask.sum() == 0:
            continue

        # Predict current day
        X_day = test.loc[day_mask, feature_cols]
        preds = model.predict(X_day).astype(np.float32)
        preds = np.maximum(preds, 0)  # Clamp negatives
        test.loc[day_mask, value_column] = preds

        # Update lag features for future days using today's predictions
        for future_offset in lag_values:
            future_day = day + future_offset
            future_mask = test[time_column] == future_day
            lag_col = f"lag_t{future_offset}"

            if future_mask.sum() == 0 or lag_col not in test.columns:
                continue

            # Map predictions by entity id
            id_to_pred = dict(zip(
                test.loc[day_mask, id_column].values,
                preds,
            ))
            future_ids = test.loc[future_mask, id_column].values
            for idx, fid in zip(test.index[future_mask], future_ids):
                if fid in id_to_pred:
                    test.at[idx, lag_col] = id_to_pred[fid]

    _save_data(test, outputs["predictions"])

    return (
        f"predict_recursive: {forecast_horizon} days predicted "
        f"(days {first_pred_day}-{first_pred_day + forecast_horizon - 1})"
    )


# =============================================================================
# MOON PHASE FEATURE (from M5 4th place solution)
# =============================================================================

@contract(
    inputs={
        "data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "required_columns": ["date"]}
        },
    },
    outputs={
        "data_with_moon": {
            "format": "csv",
            "schema": {"type": "tabular", "required_columns": ["date", "moon"]}
        },
    },
    description="Add moon phase feature (0-7) based on lunar cycle",
    tags=["feature-engineering", "calendar", "time-series"],
    version="1.0.0",
)
def add_moon_phase(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "date",
) -> str:
    """
    Add moon phase feature based on lunar cycle.

    Moon phase is a surprisingly useful feature for retail forecasting
    (from M5 4th place solution). Phase values:
    - 0: New moon
    - 4: Full moon
    - Other values represent intermediate phases

    Args:
        date_column: Name of the date column (default: "date")
    """
    import math
    import decimal
    from datetime import datetime

    dec = decimal.Decimal

    def get_moon_phase(date_str):
        """Calculate moon phase (0-7) for a given date."""
        try:
            d = datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            return 0
        diff = d - datetime(2001, 1, 1)
        days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
        lunations = dec("0.20439731") + (days * dec("0.03386319269"))
        phase_index = math.floor((lunations % dec(1) * dec(8)) + dec('0.5'))
        return int(phase_index) & 7

    df = pd.read_csv(inputs["data"])
    df['moon'] = df[date_column].apply(get_moon_phase).astype(np.int8)

    os.makedirs(os.path.dirname(outputs["data_with_moon"]) or ".", exist_ok=True)
    df.to_csv(outputs["data_with_moon"], index=False)

    return f"add_moon_phase: Added moon phase feature ({df['moon'].nunique()} unique values)"


# =============================================================================
# PRICE FEATURES (from M5 4th place solution)
# =============================================================================

@contract(
    inputs={
        "prices": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "required_columns": ["store_id", "item_id", "sell_price"]}
        },
    },
    outputs={
        "prices_with_features": {
            "format": "csv",
            "schema": {"type": "tabular"}
        },
    },
    description="Create price-based features (momentum, normalized, statistics)",
    tags=["feature-engineering", "price", "time-series"],
    version="1.0.0",
)
def create_price_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    group_columns: List[str] = None,
    price_column: str = "sell_price",
) -> str:
    """
    Create price-based features for retail forecasting.

    Features from M5 4th place solution:
    - price_max, price_min, price_std, price_mean: Per item-store statistics
    - price_norm: Normalized price (current / max)
    - price_nunique: Number of unique prices
    - price_momentum: Price change ratio vs previous period
    - price_cent: Fractional part of price (0.97, 0.99 psychological pricing)

    Args:
        group_columns: Columns to group by for statistics (default: ["store_id", "item_id"])
        price_column: Name of the price column (default: "sell_price")
    """
    import math

    group_columns = group_columns or ["store_id", "item_id"]

    df = pd.read_csv(inputs["prices"])
    original_cols = df.shape[1]

    # Group statistics
    grouped = df.groupby(group_columns)[price_column]
    df['price_max'] = grouped.transform('max')
    df['price_min'] = grouped.transform('min')
    df['price_std'] = grouped.transform('std')
    df['price_mean'] = grouped.transform('mean')

    # Normalized price
    df['price_norm'] = df[price_column] / df['price_max']

    # Number of unique prices per item-store
    df['price_nunique'] = grouped.transform('nunique')

    # Price momentum (current / previous)
    df['price_momentum'] = df[price_column] / grouped.shift(1)

    # Fractional cents (psychological pricing indicator)
    df['price_cent'] = df[price_column].apply(lambda x: math.modf(x)[0] if pd.notna(x) else 0)
    df['price_max_cent'] = df['price_max'].apply(lambda x: math.modf(x)[0] if pd.notna(x) else 0)

    # Optimize dtypes
    float_cols = ['price_max', 'price_min', 'price_std', 'price_mean',
                  'price_norm', 'price_momentum', 'price_cent', 'price_max_cent']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    os.makedirs(os.path.dirname(outputs["prices_with_features"]) or ".", exist_ok=True)
    df.to_csv(outputs["prices_with_features"], index=False)

    new_cols = df.shape[1] - original_cols
    return f"create_price_features: {original_cols} -> {df.shape[1]} columns (+{new_cols})"


# =============================================================================
# TARGET ENCODING (from M5 4th place solution)
# =============================================================================

@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular"}
        },
    },
    outputs={
        "data_with_encoding": {
            "format": "parquet",
            "schema": {"type": "tabular"}
        },
    },
    description="Create hierarchical target encoding features",
    tags=["feature-engineering", "target-encoding", "time-series"],
    version="1.0.0",
)
def create_target_encoding(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "sales",
    encoding_columns: List[List[str]] = None,
) -> str:
    """
    Create hierarchical target encoding features.

    From M5 4th place solution: Computes mean and std of target
    for various groupings (state, store, category, department, item).

    Args:
        target_column: Name of target column (default: "sales")
        encoding_columns: List of column groupings to encode.
            Default: [["state_id"], ["store_id"], ["cat_id"], ["dept_id"],
                      ["state_id", "cat_id"], ["store_id", "dept_id"], ["item_id"]]
    """
    encoding_columns = encoding_columns or [
        ["state_id"],
        ["store_id"],
        ["cat_id"],
        ["dept_id"],
        ["state_id", "cat_id"],
        ["state_id", "dept_id"],
        ["store_id", "cat_id"],
        ["store_id", "dept_id"],
        ["item_id"],
        ["item_id", "state_id"],
        ["item_id", "store_id"],
    ]

    df = _load_data(inputs["data"])
    original_cols = df.shape[1]

    for cols in encoding_columns:
        # Only process if columns exist
        if not all(c in df.columns for c in cols):
            continue

        col_name = '_' + '_'.join(cols) + '_'

        # Mean encoding
        mean_col = f'enc{col_name}mean'
        df[mean_col] = df.groupby(cols)[target_column].transform('mean').astype(np.float16)

        # Std encoding
        std_col = f'enc{col_name}std'
        df[std_col] = df.groupby(cols)[target_column].transform('std').astype(np.float16)

    _save_data(df, outputs["data_with_encoding"])

    new_cols = df.shape[1] - original_cols
    return f"create_target_encoding: {original_cols} -> {df.shape[1]} columns (+{new_cols})"


# =============================================================================
# CALENDAR FEATURES (from M5 4th place solution)
# =============================================================================

@contract(
    inputs={
        "calendar": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "required_columns": ["d", "date"]}
        },
    },
    outputs={
        "calendar_features": {
            "format": "csv",
            "schema": {"type": "tabular"}
        },
    },
    description="Create calendar features including day/week/month and events",
    tags=["feature-engineering", "calendar", "time-series"],
    version="1.0.0",
)
def create_calendar_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    add_moon: bool = True,
) -> str:
    """
    Create calendar features for time series forecasting.

    Features from M5 4th place solution:
    - tm_d: Day of month (1-31)
    - tm_w: Week of year (1-52)
    - tm_m: Month (1-12)
    - tm_y: Year (normalized to 0-based)
    - tm_wm: Week of month (1-5)
    - tm_dw: Day of week (0-6, Mon=0)
    - tm_w_end: Weekend flag (1 if Sat/Sun)
    - moon: Moon phase (0-7) if add_moon=True
    - Event encodings

    Args:
        add_moon: Whether to add moon phase feature (default: True)
    """
    import math
    import decimal
    from datetime import datetime
    from math import ceil
    from sklearn.preprocessing import OrdinalEncoder

    df = pd.read_csv(inputs["calendar"])
    original_cols = df.shape[1]

    # Convert d column to integer if it's in d_XXX format
    if df['d'].dtype == 'object' and df['d'].str.startswith('d_').any():
        df['d'] = df['d'].str[2:].astype(np.int16)

    # Parse date
    df['date'] = pd.to_datetime(df['date'])

    # Temporal features
    df['tm_d'] = df['date'].dt.day.astype(np.int8)
    df['tm_w'] = df['date'].dt.isocalendar().week.astype(np.int8)
    df['tm_m'] = df['date'].dt.month.astype(np.int8)
    df['tm_y'] = df['date'].dt.year
    df['tm_y'] = (df['tm_y'] - df['tm_y'].min()).astype(np.int8)
    df['tm_wm'] = df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8)
    df['tm_dw'] = df['date'].dt.dayofweek.astype(np.int8)
    df['tm_w_end'] = (df['tm_dw'] >= 5).astype(np.int8)

    # Moon phase
    if add_moon:
        dec = decimal.Decimal

        def get_moon_phase(date):
            diff = date - datetime(2001, 1, 1)
            days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
            lunations = dec("0.20439731") + (days * dec("0.03386319269"))
            phase_index = math.floor((lunations % dec(1) * dec(8)) + dec('0.5'))
            return int(phase_index) & 7

        df['moon'] = df['date'].apply(get_moon_phase).astype(np.int8)

    # Encode event columns
    event_cols = ['event_name_1', 'event_name_2']
    for col in event_cols:
        if col in df.columns:
            df[col] = df[col].fillna('_none_')
            encoder = OrdinalEncoder(dtype='int8', handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = encoder.fit_transform(df[[col]]).ravel() + 1

    # Ensure SNAP columns are int8
    snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
    for col in snap_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.int8)

    # Drop unnecessary columns
    drop_cols = ['weekday', 'event_type_1', 'event_type_2']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Convert date back to string for CSV storage
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    os.makedirs(os.path.dirname(outputs["calendar_features"]) or ".", exist_ok=True)
    df.to_csv(outputs["calendar_features"], index=False)

    new_cols = df.shape[1] - original_cols
    return f"create_calendar_features: {original_cols} -> {df.shape[1]} columns (+{new_cols})"


# =============================================================================
# MELT SALES DATA (M5-specific but generic pattern)
# =============================================================================

@contract(
    inputs={
        "sales": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular"}
        },
    },
    outputs={
        "sales_long": {
            "format": "parquet",
            "schema": {"type": "tabular"}
        },
    },
    description="Melt wide-format sales data to long format",
    tags=["data-handling", "reshape", "time-series"],
    version="1.0.0",
)
def melt_sales_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_columns: List[str] = None,
    drop_first_days: int = 0,
    target_name: str = "sales",
) -> str:
    """
    Melt wide-format time series data to long format.

    Converts data with columns like d_1, d_2, ..., d_N to long format
    with columns: id_columns + [d, target_name]

    Args:
        id_columns: Columns that identify each series (default: auto-detect)
        drop_first_days: Number of initial days to drop to save memory (default: 0)
        target_name: Name for the value column (default: "sales")
    """
    df = pd.read_csv(inputs["sales"])
    original_shape = df.shape

    # Auto-detect id columns (non d_* columns)
    if id_columns is None:
        id_columns = [c for c in df.columns if not c.startswith('d_')]

    # Drop first N days to save memory
    if drop_first_days > 0:
        cols_to_drop = [f'd_{i+1}' for i in range(drop_first_days)]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Clean ID column if present (remove _evaluation suffix)
    if 'id' in df.columns:
        df['id'] = df['id'].str.replace('_evaluation', '').str.replace('_validation', '')

    # Melt to long format
    df = df.melt(
        id_vars=id_columns,
        var_name='d',
        value_name=target_name
    )

    # Convert d to integer
    df['d'] = df['d'].str[2:].astype(np.int16)
    df[target_name] = df[target_name].astype(np.float32)

    _save_data(df, outputs["sales_long"])

    return f"melt_sales_data: {original_shape} -> {df.shape}, dropped first {drop_first_days} days"


# =============================================================================
# EVENT UPLIFT FEATURES (from Top 3% solution)
# =============================================================================

@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular"}
        },
    },
    outputs={
        "data_with_uplift": {
            "format": "parquet",
            "schema": {"type": "tabular"}
        },
    },
    description="Create event uplift features measuring sales impact around holidays",
    tags=["feature-engineering", "event", "time-series"],
    version="1.0.0",
)
def create_event_uplift(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "sales",
    event_column: str = "event_name_1",
    group_columns: List[str] = None,
    windows: List[int] = None,
) -> str:
    """
    Create event uplift features measuring sales impact around holidays.

    From Top 3% solution: Calculates mean sales around events vs normal days
    to capture holiday lift effect.

    Features created:
    - event_uplift_{group}: Ratio of sales on event days vs normal days
    - days_to_event: Days until next event
    - days_from_event: Days since last event

    Args:
        target_column: Name of target column (default: "sales")
        event_column: Name of event indicator column (default: "event_name_1")
        group_columns: Columns to group by (default: ["item_id", "store_id"])
        windows: Days before/after event to consider (default: [1, 3, 7])
    """
    group_columns = group_columns or ["item_id", "store_id"]
    windows = windows or [1, 3, 7]

    df = _load_data(inputs["data"])
    original_cols = df.shape[1]

    # Check if event column exists and has events
    if event_column not in df.columns:
        _save_data(df, outputs["data_with_uplift"])
        return f"create_event_uplift: No event column found, skipped"

    # Event indicator (non-zero/non-null means event day)
    is_event = (df[event_column] != 0) & (df[event_column].notna())

    # Calculate uplift ratio per group
    for cols in [group_columns, group_columns[:1]]:
        if not all(c in df.columns for c in cols):
            continue

        col_name = f"event_uplift_{'_'.join(cols)}"

        # Mean sales on event vs non-event days
        event_mean = df[is_event].groupby(cols)[target_column].transform('mean')
        normal_mean = df[~is_event].groupby(cols)[target_column].transform('mean')

        # Create lookup dict
        event_lookup = df[is_event].groupby(cols)[target_column].mean()
        normal_lookup = df[~is_event].groupby(cols)[target_column].mean()

        # Calculate uplift ratio
        def calc_uplift(row):
            key = tuple(row[c] for c in cols)
            e_mean = event_lookup.get(key, np.nan)
            n_mean = normal_lookup.get(key, np.nan)
            if pd.isna(n_mean) or n_mean == 0:
                return 1.0
            return e_mean / n_mean if pd.notna(e_mean) else 1.0

        df[col_name] = df.apply(calc_uplift, axis=1).astype(np.float32)

    _save_data(df, outputs["data_with_uplift"])

    new_cols = df.shape[1] - original_cols
    return f"create_event_uplift: {original_cols} -> {df.shape[1]} columns (+{new_cols})"


# =============================================================================
# CUMULATIVE STATISTICS (from Top 3% solution)
# =============================================================================

@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular"}
        },
    },
    outputs={
        "data_with_cumstats": {
            "format": "parquet",
            "schema": {"type": "tabular"}
        },
    },
    description="Create cumulative/expanding statistics by day of week",
    tags=["feature-engineering", "cumulative", "time-series"],
    version="1.0.0",
)
def create_cumulative_stats(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "sales",
    group_column: str = "id",
    time_column: str = "d",
    day_of_week_column: str = "tm_dw",
) -> str:
    """
    Create cumulative/expanding statistics by day of week.

    From Top 3% solution: Computes expanding mean and variance
    of sales for each day of week within each series.

    Features:
    - cum_mean_dow: Expanding mean of sales for this day of week
    - cum_var_dow: Expanding variance of sales for this day of week

    Args:
        target_column: Name of target column (default: "sales")
        group_column: Column identifying each series (default: "id")
        time_column: Column for temporal ordering (default: "d")
        day_of_week_column: Day of week column (default: "tm_dw")
    """
    df = _load_data(inputs["data"])
    original_cols = df.shape[1]

    if day_of_week_column not in df.columns:
        _save_data(df, outputs["data_with_cumstats"])
        return f"create_cumulative_stats: No day of week column, skipped"

    # Sort by group and time
    df = df.sort_values([group_column, time_column])

    # Group by id and day of week
    group_dow = df.groupby([group_column, day_of_week_column])[target_column]

    # Expanding mean (cumulative mean up to current point)
    df['cum_mean_dow'] = group_dow.transform(
        lambda x: x.expanding().mean().shift(1)
    ).astype(np.float32)

    # Expanding variance
    df['cum_var_dow'] = group_dow.transform(
        lambda x: x.expanding().var().shift(1)
    ).astype(np.float32)

    _save_data(df, outputs["data_with_cumstats"])

    new_cols = df.shape[1] - original_cols
    return f"create_cumulative_stats: {original_cols} -> {df.shape[1]} columns (+{new_cols})"


# =============================================================================
# MULTI-LEVEL LAG FEATURES (from Top 3% solution)
# =============================================================================

@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular"}
        },
    },
    outputs={
        "data_with_multilags": {
            "format": "parquet",
            "schema": {"type": "tabular"}
        },
    },
    description="Create lag features at multiple aggregation levels",
    tags=["feature-engineering", "lag", "hierarchical", "time-series"],
    version="1.1.0",
)
def create_multilevel_lags(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "sales",
    time_column: str = "d",
    level_configs: List[Dict] = None,
) -> str:
    """
    Create lag features at multiple aggregation levels (memory-optimized).

    From Top 3% solution: Creates lags at item, dept+store, and item+store levels
    to capture different granularities of demand patterns.

    Args:
        target_column: Name of target column (default: "sales")
        time_column: Column for temporal ordering (default: "d")
        level_configs: List of dicts with keys:
            - group_columns: List of columns to group by
            - lags: List of lag offsets
            - windows: Optional list of rolling windows (default: None)
    """
    import gc

    level_configs = level_configs or [
        {"group_columns": ["item_id"], "lags": [7, 28], "windows": [7]},
        {"group_columns": ["dept_id", "store_id"], "lags": [7, 28], "windows": None},
    ]

    df = _load_data(inputs["data"])
    original_cols = df.shape[1]

    for config in level_configs:
        group_cols = config["group_columns"]
        lags = config["lags"]
        windows = config.get("windows")

        # Check if columns exist
        if not all(c in df.columns for c in group_cols):
            continue

        level_name = '_'.join(group_cols)

        # First aggregate target by group and time (memory efficient)
        agg_df = df.groupby(group_cols + [time_column], as_index=False)[target_column].mean()
        agg_df = agg_df.rename(columns={target_column: f'agg_{level_name}'})
        agg_df = agg_df.sort_values(group_cols + [time_column])

        # Create lags on aggregated data
        for lag in lags:
            lag_col = f'lag_{level_name}_t{lag}'
            agg_df[lag_col] = (
                agg_df.groupby(group_cols)[f'agg_{level_name}']
                .shift(lag)
                .astype(np.float32)
            )

            # Rolling mean on lagged value (optional)
            if windows:
                for w in windows:
                    roll_col = f'roll_{level_name}_lag{lag}_w{w}'
                    agg_df[roll_col] = (
                        agg_df.groupby(group_cols)[lag_col]
                        .transform(lambda x: x.rolling(w, min_periods=1).mean())
                        .astype(np.float32)
                    )

        # Drop aggregation column before merge
        agg_df = agg_df.drop(columns=[f'agg_{level_name}'])

        # Merge back to main df
        df = df.merge(agg_df, on=group_cols + [time_column], how='left')

        # Free memory
        del agg_df
        gc.collect()

    _save_data(df, outputs["data_with_multilags"])

    new_cols = df.shape[1] - original_cols
    return f"create_multilevel_lags: {original_cols} -> {df.shape[1]} columns (+{new_cols})"


# =============================================================================
# FIRST SALE DAY TRACKING (from Top 3% solution)
# =============================================================================

@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular"}
        },
    },
    outputs={
        "data_with_firstsale": {
            "format": "parquet",
            "schema": {"type": "tabular"}
        },
    },
    description="Add days since first sale feature for each item-store",
    tags=["feature-engineering", "time-series"],
    version="1.0.0",
)
def create_first_sale_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "sales",
    group_column: str = "id",
    time_column: str = "d",
) -> str:
    """
    Add days since first sale feature for each item-store.

    From Top 3% solution: Tracks how long a product has been selling,
    useful for distinguishing new vs established products.

    Features:
    - first_sale_day: Day number of first sale for this series
    - days_since_first_sale: Days since the first sale

    Args:
        target_column: Name of target column (default: "sales")
        group_column: Column identifying each series (default: "id")
        time_column: Column for temporal ordering (default: "d")
    """
    df = _load_data(inputs["data"])
    original_cols = df.shape[1]

    # Find first day with positive sales for each group
    positive_sales = df[df[target_column] > 0]
    first_sale = positive_sales.groupby(group_column)[time_column].min()
    first_sale_df = first_sale.reset_index()
    first_sale_df.columns = [group_column, 'first_sale_day']

    # Merge back
    df = df.merge(first_sale_df, on=group_column, how='left')

    # Calculate days since first sale
    df['days_since_first_sale'] = (df[time_column] - df['first_sale_day']).astype(np.float32)
    df['days_since_first_sale'] = df['days_since_first_sale'].clip(lower=0)

    _save_data(df, outputs["data_with_firstsale"])

    new_cols = df.shape[1] - original_cols
    return f"create_first_sale_features: {original_cols} -> {df.shape[1]} columns (+{new_cols})"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "create_lag_features": create_lag_features,
    "split_temporal": split_temporal,
    "predict_recursive": predict_recursive,
}
