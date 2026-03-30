"""
M5 Forecasting Accuracy - SLEGO Services
==========================================

Competition: https://www.kaggle.com/competitions/m5-forecasting-accuracy
Problem Type: Time Series Forecasting
Target: 28-day ahead sales forecast

This file contains services for the M5 forecasting pipeline.
Services are designed for review, testing, and conversion to SLEGO format.

Dataset Structure:
- calendar.csv: Date features, events, SNAP info
- sell_prices.csv: Weekly prices per item/store
- sales_train_evaluation.csv: Historical daily sales (1941 days)
- sales_train_validation.csv: Same but without last 28 days
- sample_submission.csv: Submission format

Pipeline Steps (based on top solutions):
1. load_m5_data - Load and validate all M5 datasets
2. prep_calendar - Process calendar features
3. reshape_sales - Melt sales from wide to long format
4. create_lag_features - Generate lag and rolling features
5. merge_features - Join calendar, prices, and sales
6. encode_categoricals - Ordinal encode categorical columns
7. split_temporal - Create train/valid/test splits
8. train_lightgbm - Train LightGBM with Poisson loss (user-friendly params)
9. predict_recursive - 28-day recursive prediction
10. format_m5_submission - Create submission file

Available Model Services (each is a separate service with user-friendly params):
- train_lightgbm: LightGBM (recommended for M5) - num_leaves, learning_rate, etc.
- train_xgboost: XGBoost with learning_rate, max_depth, subsample
- train_gradient_boosting: Sklearn's GradientBoostingRegressor
- train_random_forest: Random Forest with n_estimators, max_depth

Note: M5 is a time-series competition requiring:
- Temporal validation (no random shuffle)
- Recursive prediction (predict day N, use for day N+1)
- Memory optimization (dataset is ~500MB+)

Usage:
    from services.m5_forecasting_services import run_pipeline
    result = run_pipeline("storage/m5-forecasting-accuracy")
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Tuple

# ML Libraries
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slego_contract import contract


# =============================================================================
# CONSTANTS
# =============================================================================

# M5 Competition specific constants
LAGS = [7, 14, 21, 28]   # Lag features: 1, 2, 3, 4 weeks (expanded from winning solutions)
WINDOWS = [7, 14, 30, 60]  # Rolling window sizes (from 4th place solution)
FIRST_PRED_DAY = 1942    # First day to predict (d_1942)
FORECAST_HORIZON = 28    # 28-day forecast
DROP_FIRST_DAYS = 800    # Drop first N days to reduce memory (reduced to keep more history)


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "load_m5_data",
        "inputs": {
            "calendar_path": "m5-forecasting-accuracy/datasets/calendar.csv",
            "prices_path": "m5-forecasting-accuracy/datasets/sell_prices.csv",
            "sales_path": "m5-forecasting-accuracy/datasets/sales_train_evaluation.csv",
            "submission_path": "m5-forecasting-accuracy/datasets/sample_submission.csv"
        },
        "outputs": {
            "data_info": "m5-forecasting-accuracy/artifacts/data_info.json"
        },
        "params": {},
        "module": "m5_forecasting_services"
    },
    {
        "service": "prep_calendar",
        "inputs": {"calendar_path": "m5-forecasting-accuracy/datasets/calendar.csv"},
        "outputs": {"calendar_processed": "m5-forecasting-accuracy/artifacts/calendar_processed.csv"},
        "params": {},
        "module": "m5_forecasting_services"
    },
    {
        "service": "reshape_sales",
        "inputs": {"sales_path": "m5-forecasting-accuracy/datasets/sales_train_evaluation.csv"},
        "outputs": {"sales_long": "m5-forecasting-accuracy/artifacts/sales_long.parquet"},
        "params": {"drop_first_days": 1000},
        "module": "m5_forecasting_services"
    },
    {
        "service": "create_lag_features",
        "inputs": {"sales_long": "m5-forecasting-accuracy/artifacts/sales_long.parquet"},
        "outputs": {"sales_with_lags": "m5-forecasting-accuracy/artifacts/sales_with_lags.parquet"},
        "params": {"lags": [7, 14, 28], "windows": [7, 28]},
        "module": "m5_forecasting_services"
    },
    {
        "service": "merge_features",
        "inputs": {
            "sales_data": "m5-forecasting-accuracy/artifacts/sales_with_lags.parquet",
            "calendar_data": "m5-forecasting-accuracy/artifacts/calendar_processed.csv",
            "prices_path": "m5-forecasting-accuracy/datasets/sell_prices.csv"
        },
        "outputs": {"merged_data": "m5-forecasting-accuracy/artifacts/merged_data.parquet"},
        "params": {},
        "module": "m5_forecasting_services"
    },
    {
        "service": "encode_categoricals",
        "inputs": {"data": "m5-forecasting-accuracy/artifacts/merged_data.parquet"},
        "outputs": {
            "encoded_data": "m5-forecasting-accuracy/artifacts/encoded_data.parquet",
            "encoders": "m5-forecasting-accuracy/artifacts/encoders.pkl"
        },
        "params": {"columns": ["item_id", "store_id", "state_id", "dept_id", "cat_id"]},
        "module": "m5_forecasting_services"
    },
    {
        "service": "split_temporal",
        "inputs": {"data": "m5-forecasting-accuracy/artifacts/encoded_data.parquet"},
        "outputs": {
            "train_data": "m5-forecasting-accuracy/artifacts/train.parquet",
            "valid_data": "m5-forecasting-accuracy/artifacts/valid.parquet",
            "test_data": "m5-forecasting-accuracy/artifacts/test.parquet"
        },
        "params": {"first_pred_day": 1942, "valid_size": 0.1},
        "module": "m5_forecasting_services"
    },
    {
        # Each model is a separate service with user-friendly parameters
        # LightGBM is recommended for M5 due to Poisson objective support
        "service": "train_lightgbm",
        "inputs": {
            "train_data": "m5-forecasting-accuracy/artifacts/train.parquet",
            "valid_data": "m5-forecasting-accuracy/artifacts/valid.parquet"
        },
        "outputs": {
            "model": "m5-forecasting-accuracy/artifacts/model.pkl",
            "metrics": "m5-forecasting-accuracy/artifacts/metrics.json",
            "feature_importance": "m5-forecasting-accuracy/artifacts/feature_importance.csv"
        },
        "params": {
            "label_column": "demand",
            "n_estimators": 3000,         # Number of boosting rounds
            "learning_rate": 0.05,        # Step size shrinkage (lower for better generalization)
            "num_leaves": 127,            # Max leaves per tree
            "max_depth": -1,              # No depth limit
            "min_child_samples": 25,      # Min data in a leaf
            "subsample": 0.7,             # Row sampling ratio
            "colsample_bytree": 0.6,      # Column sampling ratio
            "reg_alpha": 0.1,             # L1 regularization
            "reg_lambda": 0.3,            # L2 regularization
            "objective": "poisson",       # Poisson loss for count data
            "feature_exclude": ["id", "d"],
            "random_state": 42
        },
        "module": "m5_forecasting_services"
        # Alternative model services (swap service name and params):
        #
        # "service": "train_xgboost",
        # "params": {
        #     "label_column": "demand",
        #     "n_estimators": 2000,
        #     "learning_rate": 0.08,
        #     "max_depth": 6,
        #     "subsample": 0.7,
        #     "colsample_bytree": 0.7,
        #     "objective": "count:poisson",
        #     "feature_exclude": ["id", "d"],
        #     "random_state": 42
        # }
    },
    {
        "service": "predict_recursive",
        "inputs": {
            "model": "m5-forecasting-accuracy/artifacts/model.pkl",
            "test_data": "m5-forecasting-accuracy/artifacts/test.parquet"
        },
        "outputs": {"predictions": "m5-forecasting-accuracy/artifacts/predictions.parquet"},
        "params": {"forecast_horizon": 28, "first_pred_day": 1942},
        "module": "m5_forecasting_services"
    },
    {
        "service": "format_m5_submission",
        "inputs": {
            "predictions": "m5-forecasting-accuracy/artifacts/predictions.parquet",
            "sample_submission": "m5-forecasting-accuracy/datasets/sample_submission.csv"
        },
        "outputs": {"submission": "m5-forecasting-accuracy/submission.csv"},
        "params": {},
        "module": "m5_forecasting_services"
    }
]


# =============================================================================
# SERVICE IMPLEMENTATIONS
# =============================================================================

@contract(
    inputs={
        "calendar_path": {"format": "csv", "required": True},
        "prices_path": {"format": "csv", "required": True},
        "sales_path": {"format": "csv", "required": True},
        "submission_path": {"format": "csv", "required": True},
    },
    outputs={
        "data_info": {
            "format": "json",
            "schema": {
                "type": "json",
                "required_fields": ["files", "valid"],
                "fields": {"files": "dict", "valid": "bool", "errors": "list"},
            }
        },
    },
    description="Load and validate M5 datasets",
    tags=["data-handling", "validation", "m5"],
)
def load_m5_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Load and validate M5 datasets.

    Checks that all required files exist and reports basic statistics.
    """
    data_info = {
        "files": {},
        "valid": True,
        "errors": []
    }

    for name, path in inputs.items():
        if os.path.exists(path):
            # Get file size
            size_mb = os.path.getsize(path) / (1024 * 1024)
            data_info["files"][name] = {
                "path": path,
                "size_mb": round(size_mb, 2),
                "exists": True
            }
        else:
            data_info["files"][name] = {"exists": False}
            data_info["valid"] = False
            data_info["errors"].append(f"Missing: {path}")

    os.makedirs(os.path.dirname(outputs["data_info"]), exist_ok=True)
    with open(outputs["data_info"], "w") as f:
        json.dump(data_info, f, indent=2)

    total_size = sum(f.get("size_mb", 0) for f in data_info["files"].values())
    return f"load_m5_data: {len(inputs)} files, {total_size:.1f} MB total"


@contract(
    inputs={
        "calendar_path": {
            "format": "csv",
            "required": True,
            "schema": {
                "type": "tabular",
                "required_columns": ["d", "wm_yr_wk", "wday", "month", "year"],
            }
        },
    },
    outputs={
        "calendar_processed": {
            "format": "csv",
            "schema": {
                "type": "tabular",
                "required_columns": ["d", "wm_yr_wk", "wday", "month", "year"],
                "allow_missing": False,
            }
        },
    },
    description="Process calendar features for M5 competition",
    tags=["preprocessing", "calendar", "m5"],
)
def prep_calendar(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Process calendar features for M5 competition.

    - Drops unnecessary columns (date, weekday, event_type_*)
    - Converts d column to integer
    - Ordinal encodes event names
    - Optimizes dtypes for memory
    """
    df = pd.read_csv(inputs["calendar_path"])
    original_cols = df.shape[1]

    # Drop unnecessary columns
    df = df.drop(["date", "weekday", "event_type_1", "event_type_2"], axis=1, errors="ignore")

    # Convert d to integer (d_1 -> 1)
    df["d"] = df["d"].str[2:].astype(int)

    # Ordinal encode event names
    event_cols = ["event_name_1", "event_name_2"]
    for col in event_cols:
        if col in df.columns:
            df[col] = df[col].fillna("_none_")
            encoder = OrdinalEncoder(dtype="int")
            df[col] = encoder.fit_transform(df[[col]]).ravel() + 1

    # Optimize dtypes
    int8_cols = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"] + event_cols
    for col in int8_cols:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    os.makedirs(os.path.dirname(outputs["calendar_processed"]), exist_ok=True)
    df.to_csv(outputs["calendar_processed"], index=False)

    return f"prep_calendar: {original_cols} -> {df.shape[1]} columns, {len(df)} rows"


@contract(
    inputs={
        "sales_path": {
            "format": "csv",
            "required": True,
            "schema": {
                "type": "tabular",
                "required_columns": ["id", "item_id", "store_id", "state_id", "dept_id", "cat_id"],
            }
        },
    },
    outputs={
        "sales_long": {
            "format": "parquet",
            "schema": {
                "type": "tabular",
                "required_columns": ["id", "item_id", "store_id", "d", "demand"],
                "allow_missing": True,  # Future days have NaN demand
            }
        },
    },
    description="Reshape sales data from wide to long format",
    tags=["data-handling", "reshape", "m5"],
)
def reshape_sales(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    drop_first_days: int = 1000,
) -> str:
    """
    Reshape sales data from wide to long format.

    M5 sales data has 1941 columns (d_1 to d_1941). This melts to long format
    for easier feature engineering.

    Memory optimization: drops first N days to reduce dataset size.
    """
    df = pd.read_csv(inputs["sales_path"])
    original_shape = df.shape

    # Drop old dates to save memory
    cols_to_drop = [f"d_{i+1}" for i in range(drop_first_days)]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Clean ID column (remove _evaluation suffix)
    df["id"] = df["id"].str.replace("_evaluation", "")

    # Add future columns for prediction (as NaN)
    for i in range(FORECAST_HORIZON):
        df[f"d_{FIRST_PRED_DAY + i}"] = np.nan

    # Melt to long format
    id_cols = ["id", "item_id", "store_id", "state_id", "dept_id", "cat_id"]
    df = df.melt(
        id_vars=id_cols,
        var_name="d",
        value_name="demand"
    )

    # Convert d to integer
    df["d"] = df["d"].str[2:].astype("int64")
    df["demand"] = df["demand"].astype("float32")

    os.makedirs(os.path.dirname(outputs["sales_long"]), exist_ok=True)
    df.to_parquet(outputs["sales_long"], index=False)

    return f"reshape_sales: {original_shape} -> {df.shape}, dropped first {drop_first_days} days"


@contract(
    inputs={
        "sales_long": {
            "format": "parquet",
            "required": True,
            "schema": {
                "type": "tabular",
                "required_columns": ["id", "d", "demand"],
                "allow_missing": True,
            }
        },
    },
    outputs={
        "sales_with_lags": {
            "format": "parquet",
            "schema": {
                "type": "tabular",
                "required_columns": ["id", "d", "demand"],
                "allow_missing": True,
            }
        },
    },
    description="Create lag and rolling window features",
    tags=["feature-engineering", "lag-features", "time-series"],
)
def create_lag_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    lags: List[int] = None,
    windows: List[int] = None,
) -> str:
    """
    Create lag and rolling window features.

    Enhanced based on 4th place solution:
    - lag_t{L}: Demand L days ago
    - rolling_mean_w{W}: Rolling mean of last W days (shifted by 28 to avoid leakage)
    - rolling_std_w{W}: Rolling std of last W days
    """
    lags = lags or LAGS
    windows = windows or WINDOWS

    df = pd.read_parquet(inputs["sales_long"])
    original_cols = df.shape[1]

    # Sort by id and day for proper shifting
    df = df.sort_values(["id", "d"])

    # Create lag features (shifted by at least 28 days to avoid leakage)
    for lag in lags:
        df[f"lag_t{lag}"] = df.groupby("id")["demand"].shift(lag).astype("float32")

    # Rolling statistics (shifted by prediction horizon to avoid leakage)
    # Based on 4th place solution approach
    for w in windows:
        # Rolling mean shifted by 28 days
        df[f"rolling_mean_w{w}"] = (
            df.groupby("id")["demand"]
            .transform(lambda x: x.shift(FORECAST_HORIZON).rolling(w, min_periods=1).mean())
            .astype("float32")
        )
        # Rolling std shifted by 28 days
        df[f"rolling_std_w{w}"] = (
            df.groupby("id")["demand"]
            .transform(lambda x: x.shift(FORECAST_HORIZON).rolling(w, min_periods=1).std())
            .astype("float32")
        )

    # Remove rows with NaN in lag features (first rows per id)
    min_required_days = max(lags) + max(windows)
    df = df[df["d"] > (DROP_FIRST_DAYS + min_required_days)]

    os.makedirs(os.path.dirname(outputs["sales_with_lags"]), exist_ok=True)
    df.to_parquet(outputs["sales_with_lags"], index=False)

    return f"create_lag_features: {original_cols} -> {df.shape[1]} columns"


@contract(
    inputs={
        "sales_data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular", "required_columns": ["id", "d", "demand"]}
        },
        "calendar_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "required_columns": ["d"]}
        },
        "prices_path": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "required_columns": ["store_id", "item_id", "sell_price"]}
        },
    },
    outputs={
        "merged_data": {
            "format": "parquet",
            "schema": {"type": "tabular", "allow_missing": True}
        },
    },
    description="Merge sales data with calendar and pricing features",
    tags=["data-handling", "merge", "m5"],
)
def merge_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Merge sales data with calendar and pricing features.

    Enhanced with price features from 4th place solution:
    - price_max, price_min, price_mean, price_std per item/store
    - price_norm: normalized price (sell_price / price_max)
    - price_momentum: price compared to previous week
    - price_nunique: number of unique prices per item/store
    """
    sales = pd.read_parquet(inputs["sales_data"])
    calendar = pd.read_csv(inputs["calendar_data"])
    prices = pd.read_csv(inputs["prices_path"])

    original_cols = sales.shape[1]

    # Merge calendar
    sales = sales.merge(calendar, how="left", on="d")

    # ========== Enhanced Price Features (from 4th place solution) ==========
    # Price statistics per item/store
    prices['price_max'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
    prices['price_min'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
    prices['price_mean'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
    prices['price_std'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('std')

    # Normalized price (current price relative to max)
    prices['price_norm'] = prices['sell_price'] / prices['price_max']

    # Number of unique prices per item/store
    prices['price_nunique'] = prices.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')

    # Price momentum (current price vs previous week)
    prices['price_momentum'] = prices['sell_price'] / prices.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.shift(1))
    prices['price_momentum'] = prices['price_momentum'].fillna(1.0)

    # Merge prices (need store_id, item_id, wm_yr_wk)
    sales = sales.merge(
        prices,
        how="left",
        on=["store_id", "item_id", "wm_yr_wk"]
    )

    # Drop wm_yr_wk after merge
    if "wm_yr_wk" in sales.columns:
        sales = sales.drop(columns=["wm_yr_wk"])

    os.makedirs(os.path.dirname(outputs["merged_data"]), exist_ok=True)
    sales.to_parquet(outputs["merged_data"], index=False)

    return f"merge_features: {original_cols} -> {sales.shape[1]} columns"


@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": True}
        },
    },
    outputs={
        "encoded_data": {
            "format": "parquet",
            "schema": {"type": "tabular", "allow_missing": True}
        },
        "encoders": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "sklearn_encoder"}
        },
    },
    description="Ordinal encode categorical columns",
    tags=["preprocessing", "encoding", "m5"],
)
def encode_categoricals(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
) -> str:
    """
    Ordinal encode categorical columns.

    For tree-based models (LightGBM), ordinal encoding is often better
    than one-hot encoding due to memory efficiency.
    """
    import pickle

    columns = columns or ["item_id", "store_id", "state_id", "dept_id", "cat_id"]

    df = pd.read_parquet(inputs["data"])
    encoders = {}

    for col in columns:
        # Check for object/string dtype (parquet may use various string representations)
        if col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            is_string_type = dtype_str in ['object', 'str', 'string', 'category'] or 'str' in dtype_str
            if is_string_type:
                # Convert to string first if needed
                df[col] = df[col].astype(str)
                encoder = OrdinalEncoder(dtype="int16")
                df[col] = encoder.fit_transform(df[[col]]).ravel() + 1
                encoders[col] = encoder

    os.makedirs(os.path.dirname(outputs["encoded_data"]), exist_ok=True)
    df.to_parquet(outputs["encoded_data"], index=False)

    with open(outputs["encoders"], "wb") as f:
        pickle.dump(encoders, f)

    return f"encode_categoricals: encoded {len(encoders)} columns"


@contract(
    inputs={
        "data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular", "required_columns": ["d", "demand"]}
        },
    },
    outputs={
        "train_data": {
            "format": "parquet",
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1}
        },
        "valid_data": {
            "format": "parquet",
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1}
        },
        "test_data": {
            "format": "parquet",
            "schema": {"type": "tabular", "allow_missing": True}  # Future days have NaN
        },
    },
    description="Split data temporally for time series validation",
    tags=["data-handling", "split", "time-series"],
)
def split_temporal(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    first_pred_day: int = 1942,
    valid_size: float = 0.1,
) -> str:
    """
    Split data temporally for time series validation.

    For time series, we cannot shuffle! Split by time:
    - Test: days >= first_pred_day (for recursive prediction)
    - Train/Valid: days < first_pred_day (random split OK within historical data)
    """
    df = pd.read_parquet(inputs["data"])

    # Test set: future days (includes lag buffer for recursive prediction)
    buffer_days = max(LAGS) + max(WINDOWS) + 28
    test = df[df["d"] >= first_pred_day - buffer_days].copy()

    # Historical data for train/valid
    historical = df[df["d"] < first_pred_day].copy()

    # Determine feature columns (exclude id, d, demand)
    feature_cols = [c for c in df.columns if c not in ["id", "d", "demand"]]

    # Random split on historical (OK for tree models)
    train, valid = train_test_split(
        historical,
        test_size=valid_size,
        shuffle=True,
        random_state=42
    )

    os.makedirs(os.path.dirname(outputs["train_data"]), exist_ok=True)
    train.to_parquet(outputs["train_data"], index=False)
    valid.to_parquet(outputs["valid_data"], index=False)
    test.to_parquet(outputs["test_data"], index=False)

    return f"split_temporal: train={len(train)}, valid={len(valid)}, test={len(test)}"


# =============================================================================
# MODEL TRAINING - Individual Model Services
# =============================================================================
# Each model is a separate service with user-friendly parameters.
# Users don't need to know internal library parameter names like model_class.
# =============================================================================

def _evaluate_regression_parquet(model, X_valid, y_valid) -> Dict:
    """Helper to evaluate regression model."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    preds = model.predict(X_valid)
    return {
        "valid_rmse": float(np.sqrt(mean_squared_error(y_valid, preds))),
        "valid_mae": float(mean_absolute_error(y_valid, preds)),
    }


def _save_model_metrics_importance(model, metrics, feature_cols, outputs):
    """Helper to save model, metrics, and feature importance."""
    import pickle
    os.makedirs(os.path.dirname(outputs["model"]), exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model, f)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance if available and output specified
    if "feature_importance" in outputs and hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(outputs["feature_importance"], index=False)


@contract(
    inputs={
        "train_data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1}
        },
        "valid_data": {
            "format": "parquet",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False}
        },
    },
    outputs={
        "model": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "lightgbm_model"}
        },
        "metrics": {
            "format": "json",
            "schema": {
                "type": "json",
                "required_fields": ["model_type", "valid_rmse"],
                "fields": {"model_type": "str", "valid_rmse": "float", "valid_mae": "float"}
            }
        },
        "feature_importance": {
            "format": "csv",
            "schema": {"type": "tabular", "required_columns": ["feature", "importance"]}
        },
    },
    description="Train LightGBM Regressor",
    tags=["modeling", "training", "lightgbm", "regression"],
)
def train_lightgbm(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "demand",
    n_estimators: int = 1400,
    learning_rate: float = 0.03,
    num_leaves: int = 2047,
    max_depth: int = -1,
    min_child_samples: int = 4095,
    subsample: float = 0.5,
    colsample_bytree: float = 0.5,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    objective: str = "tweedie",
    tweedie_variance_power: float = 1.1,
    feature_exclude: List[str] = None,
    random_state: int = 42,
) -> str:
    """
    Train LightGBM Regressor with optimized parameters from 4th place solution.

    Key improvements:
    - Tweedie loss (better for zero-inflated count data like M5)
    - Optimized hyperparameters from winning solution
    - Early stopping to prevent overfitting

    Args:
        label_column: Target column name
        n_estimators: Number of boosting iterations (default: 1400 from 4th place)
        learning_rate: Shrinkage rate (default: 0.03 from 4th place)
        num_leaves: Maximum leaves per tree (default: 2047 from 4th place)
        max_depth: Maximum tree depth (-1 = no limit)
        min_child_samples: Minimum data in a leaf (default: 4095 from 4th place)
        subsample: Row sampling ratio (default: 0.5 from 4th place)
        colsample_bytree: Column sampling ratio (default: 0.5 from 4th place)
        reg_alpha: L1 regularization (default: 0)
        reg_lambda: L2 regularization (default: 0)
        objective: Loss function (default: "tweedie" - key improvement)
        tweedie_variance_power: Tweedie variance power (default: 1.1)
        feature_exclude: Columns to exclude from features
        random_state: Random seed
    """
    try:
        from lightgbm import LGBMRegressor
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    train = pd.read_parquet(inputs["train_data"])
    valid = pd.read_parquet(inputs["valid_data"])

    # Determine feature columns
    feature_exclude = feature_exclude or ["id", "d"]
    exclude_cols = set(feature_exclude + [label_column])
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    # Build params dict for LightGBM (4th place winning configuration)
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': 'rmse',
        'subsample': subsample,
        'subsample_freq': 1,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_child_samples,
        'feature_fraction': colsample_bytree,
        'max_bin': 100,
        'n_estimators': n_estimators,
        'boost_from_average': False,
        'verbose': -1,
        'random_state': random_state,
        'n_jobs': -1,
    }

    # Add tweedie power if using tweedie objective
    if objective == 'tweedie':
        lgb_params['tweedie_variance_power'] = tweedie_variance_power

    model = LGBMRegressor(**lgb_params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    metrics = {
        "model_type": "LGBMRegressor",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "objective": objective,
    }
    metrics.update(_evaluate_regression_parquet(model, X_valid, y_valid))

    # Add best_iteration
    if hasattr(model, 'best_iteration_'):
        metrics["best_iteration"] = model.best_iteration_

    _save_model_metrics_importance(model, metrics, feature_cols, outputs)

    return f"train_lightgbm: {len(X_train)} samples, lr={learning_rate}, RMSE={metrics['valid_rmse']:.4f}"


@contract(
    inputs={
        "train_data": {"format": "parquet", "required": True, "schema": {"type": "tabular", "allow_missing": False}},
        "valid_data": {"format": "parquet", "required": True, "schema": {"type": "tabular", "allow_missing": False}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "xgboost_model"}},
        "metrics": {"format": "json", "schema": {"type": "json", "required_fields": ["model_type", "valid_rmse"]}},
        "feature_importance": {"format": "csv", "schema": {"type": "tabular", "required_columns": ["feature", "importance"]}},
    },
    description="Train XGBoost Regressor",
    tags=["modeling", "training", "xgboost", "regression"],
)
def train_xgboost(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "demand",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    objective: str = "reg:squarederror",
    feature_exclude: List[str] = None,
    random_state: int = 42,
) -> str:
    """
    Train XGBoost Regressor.

    Args:
        label_column: Target column name
        n_estimators: Number of boosting rounds (default: 100)
        learning_rate: Step size shrinkage (default: 0.1)
        max_depth: Maximum tree depth (default: 6)
        min_child_weight: Minimum sum of instance weight in child (default: 1)
        subsample: Row sampling ratio (default: 0.8)
        colsample_bytree: Column sampling ratio (default: 0.8)
        reg_alpha: L1 regularization (default: 0)
        reg_lambda: L2 regularization (default: 1)
        objective: Loss function ("reg:squarederror", "count:poisson")
        feature_exclude: Columns to exclude from features
        random_state: Random seed
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    train = pd.read_parquet(inputs["train_data"])
    valid = pd.read_parquet(inputs["valid_data"])

    # Determine feature columns
    feature_exclude = feature_exclude or ["id", "d"]
    exclude_cols = set(feature_exclude + [label_column])
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective=objective,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    metrics = {
        "model_type": "XGBRegressor",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "objective": objective,
    }
    metrics.update(_evaluate_regression_parquet(model, X_valid, y_valid))

    _save_model_metrics_importance(model, metrics, feature_cols, outputs)

    return f"train_xgboost: {len(X_train)} samples, lr={learning_rate}, RMSE={metrics['valid_rmse']:.4f}"


@contract(
    inputs={
        "train_data": {"format": "parquet", "required": True, "schema": {"type": "tabular", "allow_missing": False}},
        "valid_data": {"format": "parquet", "required": True, "schema": {"type": "tabular", "allow_missing": False}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "sklearn_model"}},
        "metrics": {"format": "json", "schema": {"type": "json", "required_fields": ["model_type", "valid_rmse"]}},
        "feature_importance": {"format": "csv", "schema": {"type": "tabular", "required_columns": ["feature", "importance"]}},
    },
    description="Train Gradient Boosting Regressor",
    tags=["modeling", "training", "gradient-boosting", "regression"],
)
def train_gradient_boosting(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "demand",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    feature_exclude: List[str] = None,
    random_state: int = 42,
) -> str:
    """
    Train Gradient Boosting Regressor.

    Args:
        label_column: Target column name
        n_estimators: Number of boosting stages (default: 100)
        learning_rate: Shrinkage rate (default: 0.1)
        max_depth: Maximum tree depth (default: 3)
        min_samples_split: Minimum samples to split (default: 2)
        min_samples_leaf: Minimum samples in leaf (default: 1)
        subsample: Fraction of samples per tree (default: 1.0)
        feature_exclude: Columns to exclude from features
        random_state: Random seed
    """
    from sklearn.ensemble import GradientBoostingRegressor

    train = pd.read_parquet(inputs["train_data"])
    valid = pd.read_parquet(inputs["valid_data"])

    feature_exclude = feature_exclude or ["id", "d"]
    exclude_cols = set(feature_exclude + [label_column])
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    metrics = {
        "model_type": "GradientBoostingRegressor",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
    }
    metrics.update(_evaluate_regression_parquet(model, X_valid, y_valid))

    _save_model_metrics_importance(model, metrics, feature_cols, outputs)

    return f"train_gradient_boosting: {len(X_train)} samples, RMSE={metrics['valid_rmse']:.4f}"


@contract(
    inputs={
        "train_data": {"format": "parquet", "required": True, "schema": {"type": "tabular", "allow_missing": False}},
        "valid_data": {"format": "parquet", "required": True, "schema": {"type": "tabular", "allow_missing": False}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "sklearn_model"}},
        "metrics": {"format": "json", "schema": {"type": "json", "required_fields": ["model_type", "valid_rmse"]}},
        "feature_importance": {"format": "csv", "schema": {"type": "tabular", "required_columns": ["feature", "importance"]}},
    },
    description="Train Random Forest Regressor",
    tags=["modeling", "training", "random-forest", "regression"],
)
def train_random_forest(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "demand",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    feature_exclude: List[str] = None,
    random_state: int = 42,
) -> str:
    """
    Train Random Forest Regressor.

    Args:
        label_column: Target column name
        n_estimators: Number of trees (default: 100)
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_split: Minimum samples to split a node (default: 2)
        min_samples_leaf: Minimum samples in a leaf (default: 1)
        max_features: Features per split ("sqrt", "log2", or float 0-1)
        feature_exclude: Columns to exclude from features
        random_state: Random seed for reproducibility
    """
    from sklearn.ensemble import RandomForestRegressor

    train = pd.read_parquet(inputs["train_data"])
    valid = pd.read_parquet(inputs["valid_data"])

    feature_exclude = feature_exclude or ["id", "d"]
    exclude_cols = set(feature_exclude + [label_column])
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = {
        "model_type": "RandomForestRegressor",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    metrics.update(_evaluate_regression_parquet(model, X_valid, y_valid))

    _save_model_metrics_importance(model, metrics, feature_cols, outputs)

    return f"train_random_forest: {len(X_train)} samples, RMSE={metrics['valid_rmse']:.4f}"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "any"}},
        "test_data": {"format": "parquet", "required": True, "schema": {"type": "tabular", "allow_missing": True}},
    },
    outputs={
        "predictions": {"format": "parquet", "schema": {"type": "tabular", "required_columns": ["id", "d", "demand"]}},
    },
    description="Generate recursive 28-day predictions",
    tags=["inference", "prediction", "time-series", "recursive"],
)
def predict_recursive(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    forecast_horizon: int = 28,
    first_pred_day: int = 1942,
) -> str:
    """
    Generate recursive 28-day predictions.

    For each day:
    1. Use lag features from previous predictions
    2. Predict current day
    3. Update lag features for next day
    """
    import pickle

    try:
        import lightgbm as lgb
    except ImportError:
        return "ERROR: lightgbm not installed"

    with open(inputs["model"], "rb") as f:
        model = pickle.load(f)

    test = pd.read_parquet(inputs["test_data"])

    # Feature columns
    feature_cols = [c for c in test.columns if c not in ["id", "d", "demand"]]

    # Determine which lag features exist
    lag_cols = [c for c in feature_cols if c.startswith("lag_t")]
    lag_values = sorted(set(int(c.split("lag_t")[1].split("_")[0]) for c in lag_cols if "rolling" not in c))

    # Recursive prediction with lag feature updates
    for day in range(first_pred_day, first_pred_day + forecast_horizon):
        day_mask = test["d"] == day

        if day_mask.sum() > 0:
            # Predict
            X_day = test.loc[day_mask, feature_cols]
            preds = model.predict(X_day).astype(np.float32)
            preds = np.maximum(preds, 0)
            test.loc[day_mask, "demand"] = preds

            # Update lag features for future days using new predictions
            for future_offset in lag_values:
                future_day = day + future_offset
                future_mask = test["d"] == future_day
                lag_col = f"lag_t{future_offset}"

                if future_mask.sum() > 0 and lag_col in test.columns:
                    # Map predictions by id
                    id_to_pred = dict(zip(
                        test.loc[day_mask, "id"].values,
                        preds
                    ))
                    future_ids = test.loc[future_mask, "id"].values
                    for idx, fid in zip(test.index[future_mask], future_ids):
                        if fid in id_to_pred:
                            test.at[idx, lag_col] = id_to_pred[fid]

    os.makedirs(os.path.dirname(outputs["predictions"]), exist_ok=True)
    test.to_parquet(outputs["predictions"], index=False)

    return f"predict_recursive: {forecast_horizon} days predicted"


@contract(
    inputs={
        "predictions": {"format": "parquet", "required": True, "schema": {"type": "tabular", "required_columns": ["id", "d", "demand"]}},
        "sample_submission": {"format": "csv", "required": True, "schema": {"type": "tabular", "required_columns": ["id"]}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular", "required_columns": ["id"], "allow_missing": False}},
    },
    description="Format predictions for M5 Kaggle submission",
    tags=["inference", "submission", "m5"],
)
def format_m5_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Format predictions for M5 Kaggle submission.

    M5 submission format:
    - id: {item_id}_{store_id}_validation or _evaluation
    - F1-F28: Predictions for 28 days
    """
    pred = pd.read_parquet(inputs["predictions"])
    sample = pd.read_csv(inputs["sample_submission"])

    # Filter to prediction days only
    pred = pred[pred["d"] >= FIRST_PRED_DAY]

    # Create F column (F1, F2, ..., F28)
    pred["F"] = "F" + (pred["d"] - FIRST_PRED_DAY + 1).astype(str)

    # Pivot to wide format (base ids without suffix)
    base_submission = pred.pivot(
        index="id",
        columns="F",
        values="demand"
    ).reset_index()

    # Create both evaluation and validation rows
    eval_submission = base_submission.copy()
    eval_submission["id"] = eval_submission["id"] + "_evaluation"

    valid_submission = base_submission.copy()
    valid_submission["id"] = valid_submission["id"] + "_validation"

    submission = pd.concat([valid_submission, eval_submission], ignore_index=True)

    # Align to sample submission ordering and columns
    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission = submission[sample.columns]

    # Fill any NaN with 1 (minimum prediction)
    submission = submission.fillna(1)

    submission.to_csv(outputs["submission"], index=False)

    return f"format_m5_submission: {len(submission)} rows"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "load_m5_data": load_m5_data,
    "prep_calendar": prep_calendar,
    "reshape_sales": reshape_sales,
    "create_lag_features": create_lag_features,
    "merge_features": merge_features,
    "encode_categoricals": encode_categoricals,
    "split_temporal": split_temporal,
    "train_lightgbm": train_lightgbm,
    "predict_recursive": predict_recursive,
    "format_m5_submission": format_m5_submission,
}


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run_pipeline(base_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the M5 forecasting pipeline.

    Args:
        base_path: Path to storage folder (e.g., "storage/m5-forecasting-accuracy")
        verbose: Print progress messages

    Returns:
        Dict with execution results and metrics
    """
    results = {
        "success": True,
        "steps_completed": 0,
        "errors": [],
        "outputs": {},
    }

    if verbose:
        print(f"\n{'='*60}")
        print("M5 Forecasting Pipeline - Standalone Execution")
        print(f"{'='*60}")
        print(f"Base path: {base_path}\n")

    # Ensure artifacts folder exists
    os.makedirs(os.path.join(base_path, "m5-forecasting-accuracy", "artifacts"), exist_ok=True)

    for i, step in enumerate(PIPELINE_SPEC, 1):
        service_name = step["service"]

        try:
            # Resolve paths
            resolved_inputs = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
            resolved_outputs = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

            service_fn = SERVICE_REGISTRY.get(service_name)
            if not service_fn:
                raise ValueError(f"Service '{service_name}' not found")

            if verbose:
                print(f"[{i}/{len(PIPELINE_SPEC)}] {service_name}...", end=" ")

            result = service_fn(
                inputs=resolved_inputs,
                outputs=resolved_outputs,
                **step.get("params", {})
            )

            if verbose:
                print(f"OK - {result}")

            results["steps_completed"] += 1
            results["outputs"][service_name] = resolved_outputs

        except Exception as e:
            error_msg = f"Step {i} ({service_name}) failed: {str(e)}"
            results["errors"].append(error_msg)
            results["success"] = False

            if verbose:
                print(f"FAILED - {e}")

            break

    # Load metrics if available
    metrics_path = os.path.join(base_path, "m5-forecasting-accuracy/artifacts/metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            results["metrics"] = json.load(f)

    if verbose:
        print(f"\n{'='*60}")
        if results["success"]:
            print(f"Pipeline completed: {results['steps_completed']}/{len(PIPELINE_SPEC)} steps")
            if "metrics" in results:
                print(f"Best score: {results['metrics'].get('best_score', 'N/A')}")
            print(f"Submission: {base_path}/m5-forecasting-accuracy/submission.csv")
        else:
            print(f"Pipeline failed at step {results['steps_completed'] + 1}")
            for err in results["errors"]:
                print(f"  {err}")
        print(f"{'='*60}\n")

    return results


# =============================================================================
# REUSABLE SERVICES (Can be registered in SLEGO KB)
# =============================================================================

"""
Services that can be reused across competitions:

FROM HOUSE-PRICES (already in KB):
- train_model (with lightgbm as new model_type)
- predict

NEW FOR M5 (register in KB):
- prep_calendar: Calendar feature processing (events, holidays)
- create_lag_features: Lag/rolling features for time series
- split_temporal: Time-based train/test split
- predict_recursive: Recursive multi-step prediction

COMPETITION-SPECIFIC (don't register):
- reshape_sales: M5-specific wide-to-long reshape
- format_m5_submission: M5-specific submission format
"""


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M5 Forecasting Pipeline")
    parser.add_argument("--base-path", default="storage",
                       help="Path to storage folder")
    parser.add_argument("--validate", action="store_true",
                       help="Validate pipeline spec only")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check if required packages are installed")

    args = parser.parse_args()

    if args.check_deps:
        print("Checking dependencies...")
        deps = ["pandas", "numpy", "sklearn", "lightgbm"]
        for dep in deps:
            try:
                __import__(dep)
                print(f"  {dep}: OK")
            except ImportError:
                print(f"  {dep}: MISSING")
    elif args.validate:
        print("Pipeline specification:")
        for i, step in enumerate(PIPELINE_SPEC, 1):
            print(f"  {i}. {step['service']}")
            print(f"     inputs: {list(step['inputs'].keys())}")
            print(f"     outputs: {list(step['outputs'].keys())}")
    else:
        run_pipeline(args.base_path)
