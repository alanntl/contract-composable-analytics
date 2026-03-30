"""
Contract-Composable Analytics Services for dont-overfit-ii competition
Binary Classification - Target: target
High dimensional (300 features), small sample (250 rows) challenge - requires strong regularization.

Competition-specific services:
- select_top_variance_features: Feature selection by variance for high-dimensional data
- remove_highly_correlated: Remove correlated features to reduce overfitting
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services from common modules
from services.preprocessing_services import (
    fit_scaler,
    transform_scaler,
    split_data,
    create_submission,
    fit_rfe_selector,
    transform_rfe_selector,
)
from services.classification_services import (
    train_logistic_classifier,
    train_cv_logistic_classifier,
    predict_classifier,
)


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Select top features by variance for high-dimensional data",
    tags=["feature-selection", "variance", "high-dimensional", "generic"],
    version="1.0.0",
)
def select_top_variance_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_features: int = 50,
    target_column: str = "target",
    id_column: str = "id",
) -> str:
    """Select top features by variance for high-dimensional data.

    Useful when the number of features greatly exceeds the number of samples.
    Keeps only the most variable features plus id and target columns.
    """
    df = _load_data(inputs["data"])

    exclude_cols = set()
    if id_column and id_column in df.columns:
        exclude_cols.add(id_column)
    if target_column and target_column in df.columns:
        exclude_cols.add(target_column)

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    variances = df[feature_cols].var()
    top_features = variances.nlargest(n_features).index.tolist()

    keep_cols = [c for c in [id_column, target_column] if c and c in df.columns]
    keep_cols += top_features

    df = df[keep_cols]
    _save_data(df, outputs["data"])

    return f"select_top_variance_features: kept {len(top_features)} of {len(feature_cols)} features"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Remove highly correlated features to reduce overfitting",
    tags=["feature-selection", "correlation", "high-dimensional", "generic"],
    version="1.0.0",
)
def remove_highly_correlated(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    threshold: float = 0.95,
    target_column: str = "target",
    id_column: str = "id",
) -> str:
    """Remove highly correlated features to reduce multicollinearity.

    Computes pairwise correlation matrix and drops one of each pair
    exceeding the threshold.
    """
    df = _load_data(inputs["data"])

    exclude_cols = set()
    if id_column and id_column in df.columns:
        exclude_cols.add(id_column)
    if target_column and target_column in df.columns:
        exclude_cols.add(target_column)

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    n_before = len(feature_cols)
    if len(feature_cols) > 1:
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        df = df.drop(columns=to_drop)
        n_dropped = len(to_drop)
    else:
        n_dropped = 0

    _save_data(df, outputs["data"])

    return f"remove_highly_correlated: dropped {n_dropped} of {n_before} features (threshold={threshold})"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "fit_scaler": fit_scaler,
    "transform_scaler": transform_scaler,
    "fit_rfe_selector": fit_rfe_selector,
    "transform_rfe_selector": transform_rfe_selector,
    "split_data": split_data,
    "train_logistic_classifier": train_logistic_classifier,
    "train_cv_logistic_classifier": train_cv_logistic_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}

# Pipeline Specification (matches pipeline.json v3)
PIPELINE_SPEC = {
    "name": "dont-overfit-ii",
    "description": "Binary classification with RFE feature selection + CV-averaged L1 Logistic Regression for high-dimensional small-sample data",
    "version": "3.0.0",
    "problem_type": "binary",
    "target_column": "target",
    "id_column": "id",
    "steps": [
        {
            "service": "fit_scaler",
            "module": "preprocessing_services",
            "inputs": {"data": "dont-overfit-ii/datasets/train.csv"},
            "outputs": {"artifact": "dont-overfit-ii/artifacts/scaler.pkl"},
            "params": {"method": "standard", "exclude_columns": ["id", "target"]},
        },
        {
            "service": "transform_scaler",
            "module": "preprocessing_services",
            "inputs": {"data": "dont-overfit-ii/datasets/train.csv", "artifact": "dont-overfit-ii/artifacts/scaler.pkl"},
            "outputs": {"data": "dont-overfit-ii/artifacts/train_scaled.csv"},
        },
        {
            "service": "transform_scaler",
            "module": "preprocessing_services",
            "inputs": {"data": "dont-overfit-ii/datasets/test.csv", "artifact": "dont-overfit-ii/artifacts/scaler.pkl"},
            "outputs": {"data": "dont-overfit-ii/artifacts/test_scaled.csv"},
        },
        {
            "service": "fit_rfe_selector",
            "module": "preprocessing_services",
            "inputs": {"data": "dont-overfit-ii/artifacts/train_scaled.csv"},
            "outputs": {"artifact": "dont-overfit-ii/artifacts/rfe_selector.pkl"},
            "params": {"target_column": "target", "id_column": "id", "n_features_to_select": 25, "estimator_type": "logistic", "estimator_C": 0.1, "estimator_penalty": "l1", "random_state": 42},
        },
        {
            "service": "transform_rfe_selector",
            "module": "preprocessing_services",
            "inputs": {"data": "dont-overfit-ii/artifacts/train_scaled.csv", "artifact": "dont-overfit-ii/artifacts/rfe_selector.pkl"},
            "outputs": {"data": "dont-overfit-ii/artifacts/train_rfe.csv"},
            "params": {"target_column": "target", "id_column": "id"},
        },
        {
            "service": "transform_rfe_selector",
            "module": "preprocessing_services",
            "inputs": {"data": "dont-overfit-ii/artifacts/test_scaled.csv", "artifact": "dont-overfit-ii/artifacts/rfe_selector.pkl"},
            "outputs": {"data": "dont-overfit-ii/artifacts/test_rfe.csv"},
            "params": {"target_column": "target", "id_column": "id"},
        },
        {
            "service": "train_cv_logistic_classifier",
            "module": "classification_services",
            "inputs": {"train_data": "dont-overfit-ii/artifacts/train_rfe.csv", "test_data": "dont-overfit-ii/artifacts/test_rfe.csv"},
            "outputs": {"model": "dont-overfit-ii/artifacts/model.pkl", "predictions": "dont-overfit-ii/artifacts/predictions.csv", "metrics": "dont-overfit-ii/artifacts/metrics.json"},
            "params": {"label_column": "target", "id_column": "id", "prediction_column": "target", "penalty": "l1", "C": 0.1, "solver": "liblinear", "class_weight": "balanced", "max_iter": 1000, "n_splits": 25, "n_repeats": 10, "add_noise": True, "noise_std": 0.01, "random_state": 42},
        },
        {
            "service": "create_submission",
            "module": "preprocessing_services",
            "inputs": {"predictions": "dont-overfit-ii/artifacts/predictions.csv"},
            "outputs": {"submission": "dont-overfit-ii/submission.csv"},
            "params": {"id_column": "id", "prediction_column": "target"},
        },
    ],
}