"""
Allstate Claims Severity - Contract-Composable Analytics Services
==========================================

Competition: https://www.kaggle.com/competitions/allstate-claims-severity
Problem Type: Tabular Regression
Target: loss (continuous, positive, right-skewed)
Metric: MAE (Mean Absolute Error)

Approach (from top solutions):
  - Combine train+test for consistent label encoding
  - Label encode all categorical features (cat1-cat116) 
  - Log1p transform target for training
  - XGBoost + LightGBM ensemble
  - expm1 to reverse predictions

This module imports from GENERIC common services (maximum reusability).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Add parent directory for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract import contract

# =============================================================================
# I/O UTILS
# =============================================================================

def _load_data(path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(path)

def _save_data(df: pd.DataFrame, path: str) -> None:
    """Save data to CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# GENERIC SERVICES (local wrappers with proper contract signatures)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Combine train and test data for consistent preprocessing",
    tags=["preprocessing", "generic"],
    version="1.0.0",
)
def combine_train_test(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    marker_column: str = "is_train",
    target_column: str = "loss",
) -> str:
    """
    Combine train and test datasets with a marker column.
    
    The marker column allows splitting them back after preprocessing.
    """
    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])
    
    train[marker_column] = 1
    test[marker_column] = 0
    
    # Ensure target column exists in test (as NaN)
    if target_column not in test.columns:
        test[target_column] = np.nan
    
    combined = pd.concat([train, test], ignore_index=True)
    _save_data(combined, outputs["data"])
    
    return f"combine_train_test: combined {len(train)} + {len(test)} = {len(combined)} rows"


@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"},
    },
    description="Split combined data back into train and test",
    tags=["preprocessing", "generic"],
    version="1.0.0",
)
def split_train_test(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    marker_column: str = "is_train",
    target_column: str = "loss",
) -> str:
    """Split combined data back into train and test sets."""
    df = _load_data(inputs["data"])
    
    train = df[df[marker_column] == 1].drop(columns=[marker_column])
    test = df[df[marker_column] == 0].drop(columns=[marker_column])
    
    # Remove target column from test
    if target_column in test.columns:
        test = test.drop(columns=[target_column])
    
    _save_data(train, outputs["train_data"])
    _save_data(test, outputs["test_data"])
    
    return f"split_train_test: train={len(train)}, test={len(test)}"


# =============================================================================
# IMPORTS FROM GENERIC MODULES
# =============================================================================

from services.preprocessing_services import (
    label_encode_categorical,
    split_data,
    create_submission,
)
from services.regression_services import (
    train_ensemble_regressor,
    predict_ensemble_regressor,
)


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Local wrappers
    "combine_train_test": combine_train_test,
    "split_train_test": split_train_test,
    # Imported from preprocessing_services
    "label_encode_categorical": label_encode_categorical,
    "split_data": split_data,
    "create_submission": create_submission,
    # Imported from regression_services
    "train_ensemble_regressor": train_ensemble_regressor,
    "predict_ensemble_regressor": predict_ensemble_regressor,
}


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "combine_train_test",
        "inputs": {
            "train_data": "allstate-claims-severity/datasets/train.csv",
            "test_data": "allstate-claims-severity/datasets/test.csv",
        },
        "outputs": {
            "data": "allstate-claims-severity/artifacts/combined.csv",
        },
        "params": {
            "marker_column": "is_train",
            "target_column": "loss",
        },
        "module": "allstate_claims_severity_services",
    },
    {
        "service": "label_encode_categorical",
        "inputs": {
            "data": "allstate-claims-severity/artifacts/combined.csv",
        },
        "outputs": {
            "data": "allstate-claims-severity/artifacts/combined_encoded.csv",
            "encoder": "allstate-claims-severity/artifacts/label_encoder.pkl",
        },
        "params": {
            "include_target": False,
        },
        "module": "preprocessing_services",
    },
    {
        "service": "split_train_test",
        "inputs": {
            "data": "allstate-claims-severity/artifacts/combined_encoded.csv",
        },
        "outputs": {
            "train_data": "allstate-claims-severity/artifacts/train_encoded.csv",
            "test_data": "allstate-claims-severity/artifacts/test_encoded.csv",
        },
        "params": {
            "marker_column": "is_train",
            "target_column": "loss",
        },
        "module": "allstate_claims_severity_services",
    },
    {
        "service": "split_data",
        "inputs": {
            "data": "allstate-claims-severity/artifacts/train_encoded.csv",
        },
        "outputs": {
            "train_data": "allstate-claims-severity/artifacts/train_split.csv",
            "valid_data": "allstate-claims-severity/artifacts/valid_split.csv",
        },
        "params": {
            "test_size": 0.2,
            "random_state": 42,
        },
        "module": "preprocessing_services",
    },
    {
        "service": "train_ensemble_regressor",
        "inputs": {
            "train_data": "allstate-claims-severity/artifacts/train_split.csv",
            "valid_data": "allstate-claims-severity/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": "allstate-claims-severity/artifacts/model.pkl",
            "metrics": "allstate-claims-severity/artifacts/metrics.json",
            "feature_importance": "allstate-claims-severity/artifacts/feature_importance.csv",
        },
        "params": {
            "label_column": "loss",
            "id_column": "id",
            "model_types": ["xgboost", "lightgbm"],
            "weights": [0.5, 0.5],
            "log_target": True,
            "xgb_n_estimators": 1000,
            "xgb_learning_rate": 0.03,
            "xgb_max_depth": 7,
            "xgb_subsample": 0.7,
            "xgb_colsample_bytree": 0.7,
            "xgb_min_child_weight": 1,
            "lgbm_n_estimators": 1000,
            "lgbm_learning_rate": 0.03,
            "lgbm_num_leaves": 63,
            "lgbm_subsample": 0.7,
            "lgbm_colsample_bytree": 0.7,
            "lgbm_min_child_samples": 10,
        },
        "module": "regression_services",
    },
    {
        "service": "predict_ensemble_regressor",
        "inputs": {
            "model": "allstate-claims-severity/artifacts/model.pkl",
            "data": "allstate-claims-severity/artifacts/test_encoded.csv",
        },
        "outputs": {
            "predictions": "allstate-claims-severity/artifacts/predictions.csv",
        },
        "params": {
            "id_column": "id",
            "prediction_column": "loss",
        },
        "module": "regression_services",
    },
    {
        "service": "create_submission",
        "inputs": {
            "predictions": "allstate-claims-severity/artifacts/predictions.csv",
        },
        "outputs": {
            "submission": "allstate-claims-severity/submission.csv",
        },
        "params": {
            "id_column": "id",
            "prediction_column": "loss",
        },
        "module": "preprocessing_services",
    },
]
