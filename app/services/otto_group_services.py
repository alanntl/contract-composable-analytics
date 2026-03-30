"""
Contract-Composable Analytics Services for otto-group-product-classification-challenge
Multiclass Classification - Target: target (Class_1 to Class_9)
Anonymous features challenge - 93 numeric count-like features

Competition metric: Multi-class Log Loss
Submission format: id, Class_1, Class_2, ..., Class_9 (probabilities)
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from contract import contract


def _load_data(path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(path)


def _save_data(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# Import reusable generic services
# =============================================================================
from services.preprocessing_services import split_data, drop_columns, create_submission
from services.classification_services import train_lightgbm_classifier, train_xgboost_classifier, predict_classifier


# =============================================================================
# COMPETITION-SPECIFIC SERVICES (Layer 6)
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Encode Otto Group class target (Class_1 -> 0, Class_2 -> 1, etc.)",
    tags=["preprocessing", "encoding", "otto-group"],
    version="2.0.0",
)
def encode_class_target(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    prefix: str = "Class_",
) -> str:
    """Encode class target (Class_1 -> 0, Class_2 -> 1, etc.)

    G2 Compliance: Single responsibility - encode target labels.
    G4 Compliance: Column name and prefix injected via params.
    """
    df = _load_data(inputs["data"])

    if target_column in df.columns:
        df[target_column] = df[target_column].str.replace(prefix, '', regex=False).astype(int) - 1

    _save_data(df, outputs["data"])
    n_classes = df[target_column].nunique() if target_column in df.columns else 0
    return f"encode_class_target: {len(df)} rows, {n_classes} classes"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Apply log1p transformation to count-like numeric features",
    tags=["preprocessing", "transformation", "otto-group"],
    version="2.0.0",
)
def log1p_transform(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """Apply log1p transformation to count-like features.

    G2 Compliance: Single responsibility - apply log1p.
    G4 Compliance: Columns parameterized, auto-detects numerics if not specified.
    """
    df = _load_data(inputs["data"])

    if exclude_columns is None:
        exclude_columns = []

    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_columns]

    transformed = 0
    for col in columns:
        if col in df.columns and col not in exclude_columns:
            df[col] = np.log1p(df[col])
            transformed += 1

    _save_data(df, outputs["data"])
    return f"log1p_transform: {transformed} columns transformed"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Add row-wise statistics (sum, mean, std, max, min, nonzero count) for feature columns",
    tags=["feature-engineering", "row-statistics", "otto-group"],
    version="2.0.0",
)
def add_row_statistics(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    prefix: str = "feat_",
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """Add row-wise statistics for feature columns.

    G2 Compliance: Single responsibility - compute row statistics.
    G4 Compliance: Prefix and exclusions parameterized.
    """
    df = _load_data(inputs["data"])

    if exclude_columns is None:
        exclude_columns = ["id", "target"]

    feature_cols = [c for c in df.columns
                    if c.startswith(prefix) and c not in exclude_columns]

    added = 0
    if feature_cols:
        df["row_sum"] = df[feature_cols].sum(axis=1)
        df["row_mean"] = df[feature_cols].mean(axis=1)
        df["row_std"] = df[feature_cols].std(axis=1)
        df["row_max"] = df[feature_cols].max(axis=1)
        df["row_min"] = df[feature_cols].min(axis=1)
        df["row_nonzero"] = (df[feature_cols] > 0).sum(axis=1)
        added = 6

    _save_data(df, outputs["data"])
    return f"add_row_statistics: {added} row features from {len(feature_cols)} columns"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Predict multiclass probabilities and create submission CSV with per-class probability columns",
    tags=["prediction", "multiclass", "submission", "otto-group"],
    version="1.0.0",
)
def predict_multiclass_proba(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    class_prefix: str = "Class_",
    n_classes: int = 9,
) -> str:
    """Predict multiclass probabilities and output as submission CSV.

    Output format: id, Class_1, Class_2, ..., Class_N
    Handles both single-model and ensemble artifacts.

    G2 Compliance: Single responsibility - multiclass probability prediction.
    G4 Compliance: ID column, class prefix, and n_classes parameterized.
    """
    # Load model artifact
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    data_df = _load_data(inputs["data"])

    # Extract IDs
    if id_column in data_df.columns:
        ids = data_df[id_column].values
    else:
        ids = np.arange(1, len(data_df) + 1)

    # Prepare feature matrix
    feature_cols = artifact.get("feature_cols")
    if feature_cols:
        for col in feature_cols:
            if col not in data_df.columns:
                data_df[col] = 0
        X = data_df[feature_cols]
    else:
        drop_cols = [id_column] if id_column in data_df.columns else []
        X = data_df.drop(columns=drop_cols, errors="ignore")

    # Generate probability predictions
    if "models" in artifact and "weights" in artifact:
        # Ensemble artifact
        models = artifact["models"]
        weights = artifact["weights"]
        proba = np.zeros((len(X), n_classes))
        for m, w in zip(models, weights):
            proba += w * m.predict_proba(X)
    else:
        # Single model artifact
        model = artifact["model"]
        proba = model.predict_proba(X)

    # Create submission DataFrame: id, Class_1, Class_2, ..., Class_N
    class_columns = [f"{class_prefix}{i + 1}" for i in range(n_classes)]
    submission = pd.DataFrame(proba, columns=class_columns)
    submission.insert(0, id_column, ids)

    _save_data(submission, outputs["submission"])
    return f"predict_multiclass_proba: {len(submission)} predictions, {n_classes} classes"


# =============================================================================
# Service Registry
# =============================================================================
SERVICE_REGISTRY = {
    # Competition-specific (Layer 6)
    'encode_class_target': encode_class_target,
    'log1p_transform': log1p_transform,
    'add_row_statistics': add_row_statistics,
    'predict_multiclass_proba': predict_multiclass_proba,
    # Generic (imported from Layers 2-5)
    'split_data': split_data,
    'drop_columns': drop_columns,
    'create_submission': create_submission,
    'train_lightgbm_classifier': train_lightgbm_classifier,
    'train_xgboost_classifier': train_xgboost_classifier,
    'predict_classifier': predict_classifier,
}

# =============================================================================
# Pipeline Specification
# =============================================================================
PIPELINE_SPEC = {
    'name': 'otto-group-product-classification-challenge',
    'description': 'Multiclass classification with 93 anonymous count-like features. '
                   'Uses log1p transformation and row statistics for feature engineering, '
                   'LightGBM classifier optimized for multi-class log loss.',
    'version': '2.0.0',
    'problem_type': 'multiclass',
    'target_column': 'target',
    'id_column': 'id',
    'steps': [
        # === TRAIN PATH ===
        {
            'service': 'encode_class_target',
            'inputs': {'data': 'otto-group-product-classification-challenge/datasets/train.csv'},
            'outputs': {'data': 'otto-group-product-classification-challenge/artifacts/train_01_encoded.csv'},
            'params': {'target_column': 'target', 'prefix': 'Class_'},
            'module': 'otto_group_services',
        },
        {
            'service': 'log1p_transform',
            'inputs': {'data': 'otto-group-product-classification-challenge/artifacts/train_01_encoded.csv'},
            'outputs': {'data': 'otto-group-product-classification-challenge/artifacts/train_02_log.csv'},
            'params': {'exclude_columns': ['id', 'target']},
            'module': 'otto_group_services',
        },
        {
            'service': 'add_row_statistics',
            'inputs': {'data': 'otto-group-product-classification-challenge/artifacts/train_02_log.csv'},
            'outputs': {'data': 'otto-group-product-classification-challenge/artifacts/train_03_features.csv'},
            'params': {'prefix': 'feat_', 'exclude_columns': ['id', 'target']},
            'module': 'otto_group_services',
        },
        {
            'service': 'drop_columns',
            'inputs': {'data': 'otto-group-product-classification-challenge/artifacts/train_03_features.csv'},
            'outputs': {'data': 'otto-group-product-classification-challenge/artifacts/train_04_dropped.csv'},
            'params': {'columns': ['id']},
            'module': 'preprocessing_services',
        },
        {
            'service': 'split_data',
            'inputs': {'data': 'otto-group-product-classification-challenge/artifacts/train_04_dropped.csv'},
            'outputs': {
                'train_data': 'otto-group-product-classification-challenge/artifacts/train_split.csv',
                'valid_data': 'otto-group-product-classification-challenge/artifacts/valid_split.csv',
            },
            'params': {'stratify_column': 'target', 'test_size': 0.2, 'random_state': 42},
            'module': 'preprocessing_services',
        },
        {
            'service': 'train_lightgbm_classifier',
            'inputs': {
                'train_data': 'otto-group-product-classification-challenge/artifacts/train_split.csv',
                'valid_data': 'otto-group-product-classification-challenge/artifacts/valid_split.csv',
            },
            'outputs': {
                'model': 'otto-group-product-classification-challenge/artifacts/model.pkl',
                'metrics': 'otto-group-product-classification-challenge/artifacts/metrics.json',
            },
            'params': {
                'label_column': 'target',
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 127,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'early_stopping_rounds': 100,
            },
            'module': 'classification_services',
        },
        # === TEST PATH ===
        {
            'service': 'log1p_transform',
            'inputs': {'data': 'otto-group-product-classification-challenge/datasets/test.csv'},
            'outputs': {'data': 'otto-group-product-classification-challenge/artifacts/test_01_log.csv'},
            'params': {'exclude_columns': ['id']},
            'module': 'otto_group_services',
        },
        {
            'service': 'add_row_statistics',
            'inputs': {'data': 'otto-group-product-classification-challenge/artifacts/test_01_log.csv'},
            'outputs': {'data': 'otto-group-product-classification-challenge/artifacts/test_02_features.csv'},
            'params': {'prefix': 'feat_', 'exclude_columns': ['id']},
            'module': 'otto_group_services',
        },
        # === PREDICT + SUBMIT ===
        {
            'service': 'predict_multiclass_proba',
            'inputs': {
                'model': 'otto-group-product-classification-challenge/artifacts/model.pkl',
                'data': 'otto-group-product-classification-challenge/artifacts/test_02_features.csv',
            },
            'outputs': {
                'submission': 'otto-group-product-classification-challenge/submission.csv',
            },
            'params': {
                'id_column': 'id',
                'class_prefix': 'Class_',
                'n_classes': 9,
            },
            'module': 'otto_group_services',
        },
    ],
}