"""
Rossmann Store Sales - SLEGO Services
======================================
Competition: https://www.kaggle.com/competitions/rossmann-store-sales
Problem Type: Regression (RMSPE metric)
Target: Sales (daily store sales)
ID Column: Id (test row identifier)

Key insights from top solution notebooks:
- Merge train/test with store.csv for store metadata
- Filter training to Open==1 and Sales>0 only
- Log1p transform target (Sales) for better RMSPE
- Extract date features (year, month, day, dayofweek, weekofyear)
- Create competition duration and promo2 duration features
- Encode categoricals: StateHoliday, StoreType, Assortment, PromoInterval
- Set closed stores (Open==0) to Sales=0 in submission

Competition-specific services:
- prepare_rossmann_data: Full train preprocessing (merge, filter, features, log target)
- prepare_rossmann_test: Full test preprocessing (merge, features, keep Id)
- create_rossmann_submission: Handle expm1 reversal + closed stores
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract
from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services
from services.preprocessing_services import split_data, merge_dataframes
from services.regression_services import (
    train_lightgbm_regressor,
    train_xgboost_regressor,
    predict_regressor,
)


# =============================================================================
# HELPER: Common feature engineering for both train and test
# =============================================================================

def _engineer_rossmann_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Rossmann-specific feature engineering.
    Used by both prepare_rossmann_data and prepare_rossmann_test.

    Features created (from top solution notebooks):
    - Date components: Year, Month, Day, WeekOfYear
    - CompetitionOpen: months since competition opened (capped at 24)
    - Promo2Open: weeks since Promo2 started (capped at 25)
    - Encoded categoricals: StateHoliday, StoreType, Assortment, PromoInterval
    - Log-transformed CompetitionDistance
    """
    # --- Date Feature Extraction ---
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # --- Competition Duration (months since competitor opened, capped at 24) ---
    if "CompetitionOpenSinceYear" in df.columns and "CompetitionOpenSinceMonth" in df.columns:
        df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0).astype(int)
        df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0).astype(int)

        curr_year = df.get("Year", pd.Series(2015, index=df.index))
        curr_month = df.get("Month", pd.Series(6, index=df.index))

        df["CompetitionOpen"] = (
            12 * (curr_year - df["CompetitionOpenSinceYear"])
            + (curr_month - df["CompetitionOpenSinceMonth"])
        )
        df.loc[df["CompetitionOpenSinceYear"] == 0, "CompetitionOpen"] = 0
        df["CompetitionOpen"] = df["CompetitionOpen"].clip(lower=0, upper=24).fillna(0).astype(int)

    # --- Promo2 Duration (weeks since Promo2 started, capped at 25) ---
    if "Promo2SinceYear" in df.columns and "Promo2SinceWeek" in df.columns:
        df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0).astype(int)
        df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0).astype(int)

        curr_year = df.get("Year", pd.Series(2015, index=df.index))
        curr_week = df.get("WeekOfYear", pd.Series(26, index=df.index))

        df["Promo2Open"] = (
            (curr_year - df["Promo2SinceYear"]) * 52
            + (curr_week - df["Promo2SinceWeek"])
        )
        df.loc[df["Promo2SinceYear"] == 0, "Promo2Open"] = 0
        df.loc[df["Promo2"] == 0, "Promo2Open"] = 0
        df["Promo2Open"] = df["Promo2Open"].clip(lower=0, upper=25).fillna(0).astype(int)

    # --- Categorical Encoding (from solution notebooks) ---
    if "StateHoliday" in df.columns:
        holiday_map = {"0": 0, 0: 0, "a": 1, "b": 2, "c": 3, "d": 4}
        df["StateHoliday"] = df["StateHoliday"].map(holiday_map).fillna(0).astype(int)

    if "StoreType" in df.columns:
        type_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        df["StoreType"] = df["StoreType"].map(type_map).fillna(0).astype(int)

    if "Assortment" in df.columns:
        assort_map = {"a": 0, "b": 1, "c": 2}
        df["Assortment"] = df["Assortment"].map(assort_map).fillna(0).astype(int)

    if "PromoInterval" in df.columns:
        interval_map = {
            "Jan,Apr,Jul,Oct": 1,
            "Feb,May,Aug,Nov": 2,
            "Mar,Jun,Sept,Dec": 3,
        }
        df["PromoInterval"] = df["PromoInterval"].map(interval_map).fillna(0).astype(int)

    # --- Log-transform CompetitionDistance ---
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = np.log1p(df["CompetitionDistance"].fillna(0))

    # --- Fill remaining missing numeric values ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


# =============================================================================
# SERVICE 1: Prepare Training Data
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "store_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Prepare Rossmann training data: merge store, engineer features, filter, log target",
    tags=["preprocessing", "feature-engineering", "rossmann", "retail"],
    version="1.0.0",
)
def prepare_rossmann_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Sales",
    filter_open: bool = True,
    log_target: bool = True,
) -> str:
    """
    Full preprocessing pipeline for Rossmann training data.

    Steps (from top-scoring solution notebooks):
    1. Merge train with store metadata
    2. Filter to Open==1 and Sales>0 (critical for RMSPE metric)
    3. Engineer date and domain features
    4. Encode categoricals
    5. Apply log1p to target (Sales)
    6. Drop non-feature columns (Date, Customers, Open)

    Parameters:
        target_column: Target column name (default: Sales)
        filter_open: Filter to Open==1 and Sales>0 rows (default: True)
        log_target: Apply log1p to target (default: True)
    """
    train = _load_data(inputs["train_data"])
    store = _load_data(inputs["store_data"])

    # Step 1: Merge with store metadata
    df = train.merge(store, on="Store", how="left")
    n_before = len(df)

    # Step 2: Filter (from all 3 solution notebooks)
    if filter_open:
        df = df[(df["Open"] == 1) & (df[target_column] > 0)].copy()

    # Step 3-4: Feature engineering + encoding
    df = _engineer_rossmann_features(df)

    # Step 5: Log1p transform target
    if log_target and target_column in df.columns:
        df[target_column] = np.log1p(df[target_column])

    # Step 6: Drop non-feature columns
    drop_cols = ["Date", "Customers", "Open"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    _save_data(df, outputs["data"])

    return (
        f"prepare_rossmann_data: {n_before} -> {len(df)} rows "
        f"({len(df.columns)} cols, filtered={filter_open}, log={log_target})"
    )


# =============================================================================
# SERVICE 2: Prepare Test Data
# =============================================================================

@contract(
    inputs={
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "store_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Prepare Rossmann test data: merge store, engineer features, keep Id",
    tags=["preprocessing", "feature-engineering", "rossmann", "retail"],
    version="1.0.0",
)
def prepare_rossmann_test(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
) -> str:
    """
    Full preprocessing pipeline for Rossmann test data.

    Steps (matching training preprocessing):
    1. Merge test with store metadata
    2. Fill Open NaN with 1 (assumption: stores are open unless stated)
    3. Engineer same features as training
    4. Drop non-feature columns (Date, Open) but keep Id

    Parameters:
        id_column: ID column to preserve (default: Id)
    """
    test = _load_data(inputs["test_data"])
    store = _load_data(inputs["store_data"])

    # Step 1: Merge with store metadata
    df = test.merge(store, on="Store", how="left")

    # Step 2: Fill Open NaN with 1 (from solution notebook 1)
    if "Open" in df.columns:
        df["Open"] = df["Open"].fillna(1).astype(int)

    # Save Open status before dropping (for submission post-processing)
    if "Open" in df.columns:
        open_path = os.path.join(
            os.path.dirname(outputs["data"]), "test_open_status.csv"
        )
        _save_data(df[[id_column, "Open"]], open_path)

    # Step 3: Feature engineering (same as training)
    df = _engineer_rossmann_features(df)

    # Step 4: Drop non-feature columns (keep Id for prediction)
    drop_cols = ["Date", "Open"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    _save_data(df, outputs["data"])

    return f"prepare_rossmann_test: {len(df)} rows, {len(df.columns)} cols"


# =============================================================================
# SERVICE 3: Create Submission with Closed Store Handling
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Create Rossmann submission: handle closed stores (Sales=0)",
    tags=["inference", "submission", "rossmann", "retail"],
    version="1.0.0",
)
def create_rossmann_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "Sales",
) -> str:
    """
    Create Rossmann submission with closed store handling.

    From all 3 solution notebooks:
    - Predictions are already in original scale (expm1 applied by predict_regressor)
    - Closed stores (Open==0) must have Sales=0
    - Output format: Id, Sales

    Parameters:
        id_column: ID column name (default: Id)
        prediction_column: Prediction column name (default: Sales)
    """
    preds = _load_data(inputs["predictions"])
    test = _load_data(inputs["test_data"])

    # Get Open status from original test data
    if "Open" in test.columns:
        open_status = test[[id_column, "Open"]].copy()
        open_status["Open"] = open_status["Open"].fillna(1).astype(int)
    else:
        open_status = None

    # Build submission
    submission = preds[[id_column, prediction_column]].copy()

    # Set closed stores to 0 sales (from solution notebooks)
    n_closed = 0
    if open_status is not None:
        submission = submission.merge(open_status, on=id_column, how="left")
        closed_mask = submission["Open"] == 0
        n_closed = closed_mask.sum()
        submission.loc[closed_mask, prediction_column] = 0
        submission = submission[[id_column, prediction_column]]

    # Clip negative predictions and round
    submission[prediction_column] = submission[prediction_column].clip(lower=0)
    submission[id_column] = submission[id_column].astype(int)

    _save_data(submission, outputs["submission"])

    return (
        f"create_rossmann_submission: {len(submission)} rows, "
        f"{n_closed} closed stores set to 0"
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "prepare_rossmann_data": prepare_rossmann_data,
    "prepare_rossmann_test": prepare_rossmann_test,
    "create_rossmann_submission": create_rossmann_submission,
    "split_data": split_data,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "train_xgboost_regressor": train_xgboost_regressor,
    "predict_regressor": predict_regressor,
}
