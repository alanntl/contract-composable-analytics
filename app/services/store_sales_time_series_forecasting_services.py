"""
Store Sales Time Series Forecasting - SLEGO Services
=====================================================
Competition: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
Problem Type: Regression (time-series)
Target: sales
Metric: RMSLE
Submission: id, sales

Competition-specific services derived from top-scoring solution notebooks:
- prepare_store_sales_features: Merge train/test with external data (oil, stores,
  transactions, holidays), engineer temporal + lag + rolling features, temporal
  train/valid split, prepare test data for prediction.

Key insights from top-3 solution notebooks:
1. Solution #1 (xiewenwei29, score ~0.38): LightGBM + XGBoost ensemble with darts
   library. Multiple lag configs (7, 63, 365, 730). Oil price interpolation,
   transaction data, holiday processing.
2. Solution #3 (ivanlydkin, score ~0.53): Linear Regression + XGBoost hybrid with
   Fourier seasonality, day-of-week OHE, holiday indicators, volcano/earthquake
   lags, sales lags, onpromotion leads, rolling stats (7-day mean/std, 5-day EWM).
3. Common patterns: temporal features (dayofweek, month), lag features grouped by
   (store_nbr, family), rolling statistics, oil price as external regressor,
   payday indicator (wages paid 15th and last day of month).

For SLEGO pipeline: flat-table LightGBM approach with rich feature engineering
from all external datasets.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract


# ---------------------------------------------------------------------------
# Helper: load / save
# ---------------------------------------------------------------------------

def _load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {ext}")


def _save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)


# ===========================================================================
# PUBLIC SERVICE: prepare_store_sales_features
# ===========================================================================

@contract(
    inputs={
        "train": {"format": "csv", "required": True},
        "test": {"format": "csv", "required": True},
    },
    outputs={
        "train_data": {"format": "csv"},
        "valid_data": {"format": "csv"},
        "test_data": {"format": "csv"},
    },
    description="Merge store-sales data with external datasets, engineer temporal/lag/rolling features, temporal split, prepare test data",
    tags=["preprocessing", "feature-engineering", "time-series", "store-sales"],
    version="1.0.0",
)
def prepare_store_sales_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "sales",
    datetime_column: str = "date",
    id_column: str = "id",
    valid_days: int = 16,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> str:
    """
    End-to-end feature engineering for store-sales time series forecasting.

    Processes both train and test data in a single pass to ensure consistent
    encoding and correct lag/rolling feature computation across the train/test
    boundary.

    Steps:
      1. Load train and test CSVs, mark _is_test flag
      2. Merge external data (stores, oil, transactions, holidays)
      3. Create temporal features (dayofweek, month, day, year, is_weekend,
         is_payday, dayofyear, weekofyear)
      4. Create lag features grouped by (store_nbr, family)
      5. Create rolling mean/std features grouped by (store_nbr, family)
      6. Label-encode categorical columns (family, city, state, type)
      7. Split back into train and test
      8. Temporal-split train into train_split and valid_split
      9. Drop date column from all outputs

    Args:
        inputs:  "train" and "test" CSV paths
        outputs: "train_data", "valid_data", "test_data" CSV paths
        target_column: Target column name (default "sales")
        datetime_column: Date column name (default "date")
        id_column: ID column name (default "id")
        valid_days: Days to hold out for validation (default 16)
        lags: Lag periods for sales (default [1, 7, 14, 28])
        rolling_windows: Rolling window sizes (default [7, 14, 28])
    """
    lags = lags or [1, 7, 14, 28]
    rolling_windows = rolling_windows or [7, 14, 28]

    # ------------------------------------------------------------------
    # 1. Load and combine train + test
    # ------------------------------------------------------------------
    train = _load_data(inputs["train"])
    test = _load_data(inputs["test"])

    train[datetime_column] = pd.to_datetime(train[datetime_column])
    test[datetime_column] = pd.to_datetime(test[datetime_column])

    train["_is_test"] = 0
    test["_is_test"] = 1
    if target_column not in test.columns:
        test[target_column] = np.nan

    data = pd.concat([train, test], axis=0, ignore_index=True)
    data = data.sort_values([datetime_column, "store_nbr", "family"]).reset_index(drop=True)

    # Determine datasets directory from train path
    datasets_dir = os.path.dirname(inputs["train"])

    # ------------------------------------------------------------------
    # 2. Merge external data
    # ------------------------------------------------------------------
    # Stores
    stores_path = os.path.join(datasets_dir, "stores.csv")
    if os.path.exists(stores_path):
        stores = pd.read_csv(stores_path)
        data = data.merge(stores, on="store_nbr", how="left")

    # Oil prices (interpolated)
    oil_path = os.path.join(datasets_dir, "oil.csv")
    if os.path.exists(oil_path):
        oil = pd.read_csv(oil_path)
        oil["date"] = pd.to_datetime(oil["date"])
        oil = oil.rename(columns={"dcoilwtico": "oil_price"})
        full_dates = pd.DataFrame({
            "date": pd.date_range(data[datetime_column].min(), data[datetime_column].max())
        })
        oil = full_dates.merge(oil, on="date", how="left")
        oil["oil_price"] = oil["oil_price"].interpolate(method="linear", limit_direction="both")
        data = data.merge(oil, left_on=datetime_column, right_on="date", how="left", suffixes=("", "_oil"))
        if "date_oil" in data.columns:
            data = data.drop(columns=["date_oil"])

    # Transactions (interpolated per store)
    transactions_path = os.path.join(datasets_dir, "transactions.csv")
    if os.path.exists(transactions_path):
        txn = pd.read_csv(transactions_path)
        txn["date"] = pd.to_datetime(txn["date"])
        data = data.merge(txn, left_on=[datetime_column, "store_nbr"],
                          right_on=["date", "store_nbr"], how="left", suffixes=("", "_txn"))
        if "date_txn" in data.columns:
            data = data.drop(columns=["date_txn"])
        if "transactions" in data.columns:
            data["transactions"] = (
                data.groupby("store_nbr")["transactions"]
                .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            )
            data["transactions"] = data["transactions"].fillna(0)

    # Holidays (national, non-transferred)
    holidays_path = os.path.join(datasets_dir, "holidays_events.csv")
    if os.path.exists(holidays_path):
        holidays = pd.read_csv(holidays_path)
        holidays["date"] = pd.to_datetime(holidays["date"])
        national = holidays[
            (holidays["locale"] == "National") & (holidays["transferred"] == False)
        ][["date"]].drop_duplicates()
        national["is_holiday"] = 1
        data = data.merge(national, left_on=datetime_column, right_on="date",
                          how="left", suffixes=("", "_hol"))
        if "date_hol" in data.columns:
            data = data.drop(columns=["date_hol"])
        data["is_holiday"] = data["is_holiday"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 3. Temporal features
    # ------------------------------------------------------------------
    dt = data[datetime_column]
    data["dayofweek"] = dt.dt.dayofweek
    data["month"] = dt.dt.month
    data["day"] = dt.dt.day
    data["dayofyear"] = dt.dt.dayofyear
    data["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    data["year"] = dt.dt.year
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["is_payday"] = (
        (data["day"] == 15) | (data["day"] == dt.dt.days_in_month)
    ).astype(int)

    # ------------------------------------------------------------------
    # 4. Lag features (grouped by store_nbr + family)
    # ------------------------------------------------------------------
    group_cols = ["store_nbr", "family"]
    for lag in lags:
        data[f"lag_{lag}"] = data.groupby(group_cols)[target_column].shift(lag)

    # ------------------------------------------------------------------
    # 5. Rolling features (grouped, shifted by 1 to avoid leakage)
    # ------------------------------------------------------------------
    for window in rolling_windows:
        data[f"rolling_mean_{window}"] = (
            data.groupby(group_cols)[target_column]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        )
        data[f"rolling_std_{window}"] = (
            data.groupby(group_cols)[target_column]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).std())
        )

    # Fill NaN from lags/rolling
    lag_roll_cols = [c for c in data.columns if c.startswith(("lag_", "rolling_"))]
    data[lag_roll_cols] = data[lag_roll_cols].fillna(0)

    # ------------------------------------------------------------------
    # 6. Label-encode categoricals
    # ------------------------------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    for col in ["family", "city", "state", "type"]:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Fill any remaining NaN
    data["oil_price"] = data.get("oil_price", pd.Series(0, index=data.index)).fillna(0)

    # ------------------------------------------------------------------
    # 7. Split into train vs test
    # ------------------------------------------------------------------
    train_mask = data["_is_test"] == 0
    test_mask = data["_is_test"] == 1

    train_df = data.loc[train_mask].copy()
    test_df = data.loc[test_mask].copy()

    # Drop helper columns
    cols_to_drop = [datetime_column, "_is_test"]
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in cols_to_drop + [target_column] if c in test_df.columns])

    # ------------------------------------------------------------------
    # 8. Temporal split (last valid_days of training dates for validation)
    # ------------------------------------------------------------------
    # Re-derive date from temporal features for splitting
    # Since we already dropped date, use the original sorted order:
    # train_df is sorted by date, last valid_days * n_series rows are validation
    n_series = train[["store_nbr", "family"]].drop_duplicates().shape[0]
    valid_rows = valid_days * n_series
    if valid_rows < len(train_df):
        train_split = train_df.iloc[:-valid_rows].copy()
        valid_split = train_df.iloc[-valid_rows:].copy()
    else:
        # Fallback: 80/20 split
        split_idx = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:split_idx].copy()
        valid_split = train_df.iloc[split_idx:].copy()

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    _save_data(train_split, outputs["train_data"])
    _save_data(valid_split, outputs["valid_data"])
    _save_data(test_df, outputs["test_data"])

    n_features = len([c for c in train_split.columns if c not in [id_column, target_column]])
    return (
        f"prepare_store_sales_features: "
        f"train={len(train_split)}, valid={len(valid_split)}, test={len(test_df)}, "
        f"features={n_features}"
    )


# ===========================================================================
# PUBLIC SERVICE: prepare_store_sales_features_v2 (IMPROVED)
# ===========================================================================

@contract(
    inputs={
        "train": {"format": "csv", "required": True},
        "test": {"format": "csv", "required": True},
    },
    outputs={
        "train_data": {"format": "csv"},
        "valid_data": {"format": "csv"},
        "test_data": {"format": "csv"},
    },
    description="Improved feature engineering with log transform, earthquake feature, EWM, and comprehensive lags for RMSLE optimization",
    tags=["preprocessing", "feature-engineering", "time-series", "store-sales", "rmsle"],
    version="2.0.0",
)
def prepare_store_sales_features_v2(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "sales",
    datetime_column: str = "date",
    id_column: str = "id",
    valid_days: int = 16,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    use_log_transform: bool = True,
    add_earthquake_feature: bool = True,
    ewm_spans: Optional[List[int]] = None,
) -> str:
    """
    Enhanced feature engineering for store-sales with RMSLE optimization.

    Key improvements from top Kaggle solutions:
    1. Log1p transform of target for RMSLE metric optimization
    2. Earthquake feature (April 16, 2016) with 21-day lag effects
    3. More comprehensive lag features [1, 7, 14, 28, 63]
    4. Exponential weighted mean (EWM) features
    5. Promotion lag features

    Args:
        use_log_transform: Apply log1p to target (predictions need expm1 inverse)
        add_earthquake_feature: Add April 2016 earthquake indicator with lags
        ewm_spans: EWM span periods (default [7, 14])
    """
    lags = lags or [1, 7, 14, 28, 63]
    rolling_windows = rolling_windows or [7, 14, 28]
    ewm_spans = ewm_spans or [7, 14]

    # ------------------------------------------------------------------
    # 1. Load and combine train + test
    # ------------------------------------------------------------------
    train = _load_data(inputs["train"])
    test = _load_data(inputs["test"])

    train[datetime_column] = pd.to_datetime(train[datetime_column])
    test[datetime_column] = pd.to_datetime(test[datetime_column])

    train["_is_test"] = 0
    test["_is_test"] = 1
    if target_column not in test.columns:
        test[target_column] = np.nan

    data = pd.concat([train, test], axis=0, ignore_index=True)
    data = data.sort_values([datetime_column, "store_nbr", "family"]).reset_index(drop=True)

    # Determine datasets directory from train path
    datasets_dir = os.path.dirname(inputs["train"])

    # ------------------------------------------------------------------
    # 2. Apply log transform to target for RMSLE optimization
    # ------------------------------------------------------------------
    if use_log_transform:
        # Store original sales for test
        data["_original_sales"] = data[target_column]
        # Apply log1p (handles zeros gracefully)
        data[target_column] = np.log1p(data[target_column].clip(lower=0))

    # ------------------------------------------------------------------
    # 3. Merge external data
    # ------------------------------------------------------------------
    # Stores
    stores_path = os.path.join(datasets_dir, "stores.csv")
    if os.path.exists(stores_path):
        stores = pd.read_csv(stores_path)
        data = data.merge(stores, on="store_nbr", how="left")

    # Oil prices (interpolated)
    oil_path = os.path.join(datasets_dir, "oil.csv")
    if os.path.exists(oil_path):
        oil = pd.read_csv(oil_path)
        oil["date"] = pd.to_datetime(oil["date"])
        oil = oil.rename(columns={"dcoilwtico": "oil_price"})
        full_dates = pd.DataFrame({
            "date": pd.date_range(data[datetime_column].min(), data[datetime_column].max())
        })
        oil = full_dates.merge(oil, on="date", how="left")
        oil["oil_price"] = oil["oil_price"].interpolate(method="linear", limit_direction="both")
        data = data.merge(oil, left_on=datetime_column, right_on="date", how="left", suffixes=("", "_oil"))
        if "date_oil" in data.columns:
            data = data.drop(columns=["date_oil"])

    # Transactions (interpolated per store)
    transactions_path = os.path.join(datasets_dir, "transactions.csv")
    if os.path.exists(transactions_path):
        txn = pd.read_csv(transactions_path)
        txn["date"] = pd.to_datetime(txn["date"])
        data = data.merge(txn, left_on=[datetime_column, "store_nbr"],
                          right_on=["date", "store_nbr"], how="left", suffixes=("", "_txn"))
        if "date_txn" in data.columns:
            data = data.drop(columns=["date_txn"])
        if "transactions" in data.columns:
            data["transactions"] = (
                data.groupby("store_nbr")["transactions"]
                .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            )
            data["transactions"] = data["transactions"].fillna(0)

    # Holidays (national, non-transferred)
    holidays_path = os.path.join(datasets_dir, "holidays_events.csv")
    if os.path.exists(holidays_path):
        holidays = pd.read_csv(holidays_path)
        holidays["date"] = pd.to_datetime(holidays["date"])
        national = holidays[
            (holidays["locale"] == "National") & (holidays["transferred"] == False)
        ][["date"]].drop_duplicates()
        national["is_holiday"] = 1
        data = data.merge(national, left_on=datetime_column, right_on="date",
                          how="left", suffixes=("", "_hol"))
        if "date_hol" in data.columns:
            data = data.drop(columns=["date_hol"])
        data["is_holiday"] = data["is_holiday"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 4. Temporal features
    # ------------------------------------------------------------------
    dt = data[datetime_column]
    data["dayofweek"] = dt.dt.dayofweek
    data["month"] = dt.dt.month
    data["day"] = dt.dt.day
    data["dayofyear"] = dt.dt.dayofyear
    data["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    data["year"] = dt.dt.year
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["is_payday"] = (
        (data["day"] == 15) | (data["day"] == dt.dt.days_in_month)
    ).astype(int)

    # Quarter and is_month_start/end
    data["quarter"] = dt.dt.quarter
    data["is_month_start"] = dt.dt.is_month_start.astype(int)
    data["is_month_end"] = dt.dt.is_month_end.astype(int)

    # ------------------------------------------------------------------
    # 5. Earthquake feature (April 16, 2016 magnitude 7.8)
    # ------------------------------------------------------------------
    if add_earthquake_feature:
        earthquake_date = pd.Timestamp("2016-04-16")
        # Create indicator and 21-day lag effects (from solution #3)
        data["is_earthquake"] = (data[datetime_column] == earthquake_date).astype(int)
        for lag in range(1, 22):
            data[f"earthquake_lag_{lag}"] = (
                data[datetime_column] == earthquake_date + pd.Timedelta(days=lag)
            ).astype(int)

    # ------------------------------------------------------------------
    # 6. Lag features (grouped by store_nbr + family)
    # ------------------------------------------------------------------
    group_cols = ["store_nbr", "family"]
    for lag in lags:
        data[f"lag_{lag}"] = data.groupby(group_cols)[target_column].shift(lag)

    # ------------------------------------------------------------------
    # 7. Rolling features (grouped, shifted by 1 to avoid leakage)
    # ------------------------------------------------------------------
    for window in rolling_windows:
        data[f"rolling_mean_{window}"] = (
            data.groupby(group_cols)[target_column]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        )
        data[f"rolling_std_{window}"] = (
            data.groupby(group_cols)[target_column]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).std())
        )

    # ------------------------------------------------------------------
    # 8. EWM features (exponential weighted mean from solution #3)
    # ------------------------------------------------------------------
    for span in ewm_spans:
        data[f"ewm_{span}"] = (
            data.groupby(group_cols)[target_column]
            .transform(lambda s: s.shift(1).ewm(span=span, adjust=False).mean())
        )

    # ------------------------------------------------------------------
    # 9. Promotion features (leads from solution #3)
    # ------------------------------------------------------------------
    if "onpromotion" in data.columns:
        for lead in [1, 7, 14]:
            data[f"promo_lead_{lead}"] = data.groupby(group_cols)["onpromotion"].shift(-lead)
        data[[f"promo_lead_{l}" for l in [1, 7, 14]]] = (
            data[[f"promo_lead_{l}" for l in [1, 7, 14]]].fillna(0)
        )

    # Fill NaN from lags/rolling/ewm
    lag_roll_cols = [c for c in data.columns if c.startswith(("lag_", "rolling_", "ewm_"))]
    data[lag_roll_cols] = data[lag_roll_cols].fillna(0)

    # ------------------------------------------------------------------
    # 10. Label-encode categoricals
    # ------------------------------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    for col in ["family", "city", "state", "type"]:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Fill any remaining NaN
    data["oil_price"] = data.get("oil_price", pd.Series(0, index=data.index)).fillna(0)

    # ------------------------------------------------------------------
    # 11. Split into train vs test
    # ------------------------------------------------------------------
    train_mask = data["_is_test"] == 0
    test_mask = data["_is_test"] == 1

    train_df = data.loc[train_mask].copy()
    test_df = data.loc[test_mask].copy()

    # Drop helper columns
    cols_to_drop = [datetime_column, "_is_test", "_original_sales"]
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in cols_to_drop + [target_column] if c in test_df.columns])

    # ------------------------------------------------------------------
    # 12. Temporal split (last valid_days of training dates for validation)
    # ------------------------------------------------------------------
    n_series = train[["store_nbr", "family"]].drop_duplicates().shape[0]
    valid_rows = valid_days * n_series
    if valid_rows < len(train_df):
        train_split = train_df.iloc[:-valid_rows].copy()
        valid_split = train_df.iloc[-valid_rows:].copy()
    else:
        split_idx = int(len(train_df) * 0.8)
        train_split = train_df.iloc[:split_idx].copy()
        valid_split = train_df.iloc[split_idx:].copy()

    # ------------------------------------------------------------------
    # 13. Save outputs
    # ------------------------------------------------------------------
    _save_data(train_split, outputs["train_data"])
    _save_data(valid_split, outputs["valid_data"])
    _save_data(test_df, outputs["test_data"])

    n_features = len([c for c in train_split.columns if c not in [id_column, target_column]])
    return (
        f"prepare_store_sales_features_v2: "
        f"train={len(train_split)}, valid={len(valid_split)}, test={len(test_df)}, "
        f"features={n_features}, log_transform={use_log_transform}"
    )


# ===========================================================================
# PUBLIC SERVICE: predict_regressor_rmsle (with inverse log transform)
# ===========================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Predict with regressor and apply expm1 inverse transform for RMSLE submissions",
    tags=["prediction", "regression", "rmsle", "store-sales"],
    version="1.0.0",
)
def predict_regressor_rmsle(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "sales",
    apply_expm1: bool = True,
    clip_min: float = 0.0,
) -> str:
    """
    Make predictions with inverse log transform for RMSLE metric.

    If training was done on log1p(target), this applies expm1 to predictions.
    """
    import pickle

    with open(inputs["model"], "rb") as f:
        model = pickle.load(f)

    data = _load_data(inputs["data"])

    # Identify feature columns
    exclude_cols = {id_column, prediction_column}
    feature_cols = [c for c in data.columns if c not in exclude_cols]

    X = data[feature_cols]
    preds = model.predict(X)

    # Apply inverse transform if log was used
    if apply_expm1:
        preds = np.expm1(preds)

    # Clip to non-negative
    preds = np.clip(preds, clip_min, None)

    # Create submission
    result = pd.DataFrame({
        id_column: data[id_column],
        prediction_column: preds,
    })

    _save_data(result, outputs["predictions"])

    return f"predict_regressor_rmsle: {len(result)} predictions, expm1={apply_expm1}"


# ===========================================================================
# SERVICE REGISTRY
# ===========================================================================

SERVICE_REGISTRY = {
    "prepare_store_sales_features": prepare_store_sales_features,
    "prepare_store_sales_features_v2": prepare_store_sales_features_v2,
    "predict_regressor_rmsle": predict_regressor_rmsle,
}
