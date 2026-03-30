"""
Mercedes-Benz Greener Manufacturing - Contract-Composable Analytics Services
=====================================================

Competition: https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing
Problem Type: Regression
Target: y (time spent on the test bench)
Evaluation: R² (coefficient of determination)

Competition-specific services derived from top-scored solution notebooks:
- encode_train_test_labels: Joint label encoding of categoricals (solutions 2 & 3)
- remove_constant_columns: Remove zero-variance features (solution 2)
- add_decomposition_features: PCA + ICA feature extraction (solution 3)

Top solution insights:
1. Label encode (not drop!) categorical features X0-X8
2. Remove constant binary features (12 identified in solution 2)
3. Add PCA and ICA decomposition components (12 each, from solution 3)
4. Gradient boosting models (XGBoost/LightGBM) work well for this dataset
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# Import shared I/O utilities
from services.io_utils import load_data as _load_data, save_data as _save_data

# Import generic services for reuse
try:
    from .preprocessing_services import (
        split_data, create_submission, drop_columns,
        label_encode_categorical,
    )
    from .regression_services import (
        train_lightgbm_regressor, train_xgboost_regressor,
        predict_regressor,
        train_stacked_regressor, predict_stacked_regressor,
    )
except ImportError:
    from services.preprocessing_services import (
        split_data, create_submission, drop_columns,
        label_encode_categorical,
    )
    from services.regression_services import (
        train_lightgbm_regressor, train_xgboost_regressor,
        predict_regressor,
        train_stacked_regressor, predict_stacked_regressor,
    )


# =============================================================================
# COMPETITION-SPECIFIC SERVICES (designed for reuse across competitions)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular", "allow_missing": False}},
        "test_data": {"format": "csv", "schema": {"type": "tabular", "allow_missing": False}},
        "encoder": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "encoder"}},
    },
    description="Label encode categoricals using combined train+test unique values",
    tags=["preprocessing", "encoding", "label-encoding", "generic"],
    version="1.0.0",
)
def encode_train_test_labels(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    id_column: Optional[str] = None,
) -> str:
    """
    Label encode categorical columns using combined train+test unique values.

    Ensures consistent encoding between train and test sets by fitting
    the encoder on the union of values from both datasets.

    G1: Single task - joint label encoding
    G4: Column names parameterized, not hardcoded

    Parameters:
        columns: Specific columns to encode. If None, encodes all object columns.
        target_column: Target column to exclude from encoding.
        id_column: ID column to exclude from encoding.
    """
    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    # Determine columns to encode
    if columns is None:
        encode_cols = train.select_dtypes(include=['object']).columns.tolist()
    else:
        encode_cols = [c for c in columns if c in train.columns]

    # Exclude target and ID columns
    exclude = set()
    if target_column:
        exclude.add(target_column)
    if id_column:
        exclude.add(id_column)
    encode_cols = [c for c in encode_cols if c not in exclude]

    # Encode using combined unique values
    encodings = {}
    for col in encode_cols:
        if col in train.columns and col in test.columns:
            combined = pd.concat([train[col], test[col]], ignore_index=True)
            codes, uniques = pd.factorize(combined)
            mapping = {str(v): int(i) for i, v in enumerate(uniques)}
            encodings[col] = mapping

            train[col] = train[col].map(lambda x, m=mapping: m.get(str(x), -1))
            test[col] = test[col].map(lambda x, m=mapping: m.get(str(x), -1))

    _save_data(train, outputs["train_data"])
    _save_data(test, outputs["test_data"])

    os.makedirs(os.path.dirname(outputs["encoder"]) or ".", exist_ok=True)
    with open(outputs["encoder"], "wb") as f:
        pickle.dump(encodings, f)

    return f"encode_train_test_labels: encoded {len(encode_cols)} columns: {encode_cols}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
        "encoder": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "encoder"}},
    },
    description="Target-encode categoricals using k-fold smoothed means (leak-free)",
    tags=["preprocessing", "encoding", "target-encoding", "generic"],
    version="1.0.0",
)
def target_encode_train_test(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "y",
    id_column: Optional[str] = None,
    columns: Optional[List[str]] = None,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
) -> str:
    """
    Target-encode categorical columns with k-fold smoothing to avoid leakage.

    For each categorical column, replaces category values with the smoothed
    mean of the target for that category. Uses k-fold on training data to
    prevent target leakage. Test data uses global (full-train) means.

    Smoothing formula: (count * cat_mean + smoothing * global_mean) / (count + smoothing)

    G1: Single task - target encoding
    G3: Explicit random_state for reproducibility
    G4: Column names parameterized

    Parameters:
        target_column: Name of the target column.
        id_column: ID column to exclude from encoding.
        columns: Specific columns to encode. If None, encodes all object columns.
        n_folds: Number of folds for leak-free train encoding.
        smoothing: Smoothing factor (higher = more regularization toward global mean).
        random_state: Random seed for fold splitting.
    """
    from sklearn.model_selection import KFold

    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    if columns is None:
        encode_cols = train.select_dtypes(include=['object']).columns.tolist()
    else:
        encode_cols = [c for c in columns if c in train.columns]

    exclude = set()
    if target_column:
        exclude.add(target_column)
    if id_column:
        exclude.add(id_column)
    encode_cols = [c for c in encode_cols if c not in exclude]

    global_mean = train[target_column].mean()
    encoding_maps = {}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for col in encode_cols:
        # Compute global target encoding (for test data)
        stats = train.groupby(col)[target_column].agg(['mean', 'count'])
        smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        encoding_maps[col] = smoothed.to_dict()

        # K-fold target encoding for train data (leak-free)
        train[f'{col}_te'] = global_mean
        for train_idx, val_idx in kf.split(train):
            fold_train = train.iloc[train_idx]
            fold_stats = fold_train.groupby(col)[target_column].agg(['mean', 'count'])
            fold_smoothed = (fold_stats['count'] * fold_stats['mean'] + smoothing * global_mean) / (fold_stats['count'] + smoothing)
            train.loc[train.index[val_idx], f'{col}_te'] = train.iloc[val_idx][col].map(fold_smoothed).fillna(global_mean)

        # Apply global encoding to test data
        test[f'{col}_te'] = test[col].map(smoothed).fillna(global_mean)

        # Drop original categorical column (replaced by _te version)
        train = train.drop(columns=[col])
        test = test.drop(columns=[col])

    _save_data(train, outputs["train_data"])
    _save_data(test, outputs["test_data"])

    artifact = {"encoding_maps": encoding_maps, "global_mean": global_mean}
    os.makedirs(os.path.dirname(outputs["encoder"]) or ".", exist_ok=True)
    with open(outputs["encoder"], "wb") as f:
        pickle.dump(artifact, f)

    return f"target_encode_train_test: encoded {len(encode_cols)} columns: {encode_cols}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Remove zero-variance (constant) columns from train and test",
    tags=["preprocessing", "feature-selection", "variance-filter", "generic"],
    version="1.0.0",
)
def remove_constant_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: Optional[str] = None,
    id_column: Optional[str] = None,
    threshold: float = 0.0,
) -> str:
    """
    Remove constant (zero-variance) columns from train and test datasets.

    Identifies columns with variance <= threshold in the training data,
    then removes them from both train and test sets.

    G1: Single task - constant column removal
    G4: Columns identified dynamically, not hardcoded

    Parameters:
        target_column: Target column to preserve (never remove).
        id_column: ID column to preserve (never remove).
        threshold: Variance threshold. Columns with variance <= this are removed.
                   Default 0.0 means only truly constant columns.
    """
    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    preserve = set()
    if target_column and target_column in train.columns:
        preserve.add(target_column)
    if id_column and id_column in train.columns:
        preserve.add(id_column)

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    check_cols = [c for c in numeric_cols if c not in preserve]

    to_remove = []
    for col in check_cols:
        if train[col].var() <= threshold:
            to_remove.append(col)

    train = train.drop(columns=[c for c in to_remove if c in train.columns])
    test = test.drop(columns=[c for c in to_remove if c in test.columns])

    _save_data(train, outputs["train_data"])
    _save_data(test, outputs["test_data"])

    return f"remove_constant_columns: removed {len(to_remove)} columns: {to_remove[:10]}{'...' if len(to_remove) > 10 else ''}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
        "decomposition": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "decomposition"}},
    },
    description="Add PCA and ICA decomposition features to datasets",
    tags=["feature-engineering", "decomposition", "pca", "ica", "generic"],
    version="1.0.0",
)
def add_decomposition_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_components: int = 12,
    target_column: Optional[str] = None,
    id_column: Optional[str] = None,
    methods: Optional[List[str]] = None,
    random_state: int = 42,
) -> str:
    """
    Add PCA and ICA decomposition features to train and test datasets.

    Fits decomposition on training numeric features, transforms both
    train and test, and appends the components as new columns.

    G1: Single task - decomposition feature generation
    G3: Explicit random_state for reproducibility
    G4: Column names parameterized

    Parameters:
        n_components: Number of components per decomposition method.
        target_column: Target column to exclude from decomposition.
        id_column: ID column to exclude from decomposition.
        methods: List of decomposition methods. Default: ["pca", "ica"]
        random_state: Random seed for reproducibility.
    """
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import StandardScaler

    methods = methods or ["pca", "ica"]

    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    exclude = set()
    if target_column and target_column in train.columns:
        exclude.add(target_column)
    if id_column and id_column in train.columns:
        exclude.add(id_column)

    feature_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    # Scale features before decomposition
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[feature_cols].fillna(0))
    test_scaled = scaler.transform(test[feature_cols].fillna(0))

    decomposers = {}

    for method in methods:
        if method == "pca":
            decomposer = PCA(n_components=n_components, random_state=random_state)
        elif method == "ica":
            decomposer = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
        else:
            continue

        train_components = decomposer.fit_transform(train_scaled)
        test_components = decomposer.transform(test_scaled)
        decomposers[method] = decomposer

        for i in range(n_components):
            col_name = f"{method}_{i+1}"
            train[col_name] = train_components[:, i]
            test[col_name] = test_components[:, i]

    _save_data(train, outputs["train_data"])
    _save_data(test, outputs["test_data"])

    artifact = {"scaler": scaler, "decomposers": decomposers, "feature_cols": feature_cols}
    os.makedirs(os.path.dirname(outputs["decomposition"]) or ".", exist_ok=True)
    with open(outputs["decomposition"], "wb") as f:
        pickle.dump(artifact, f)

    total_new = len(methods) * n_components
    return f"add_decomposition_features: added {total_new} features ({methods}, {n_components} each)"


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

_P = "mercedes-benz-greener-manufacturing"

PIPELINE_SPEC = [
    # Step 1: Target encode categoricals (k-fold smoothed, leak-free)
    {
        "service": "target_encode_train_test",
        "inputs": {
            "train_data": f"{_P}/datasets/train.csv",
            "test_data": f"{_P}/datasets/test.csv",
        },
        "outputs": {
            "train_data": f"{_P}/artifacts/train_encoded.csv",
            "test_data": f"{_P}/artifacts/test_encoded.csv",
            "encoder": f"{_P}/artifacts/target_encoder.pkl",
        },
        "params": {
            "target_column": "y",
            "id_column": "ID",
            "n_folds": 10,
            "smoothing": 10.0,
            "random_state": 42,
        },
        "module": "mercedes_benz_greener_manufacturing_services",
    },
    # Step 2: Remove constant (zero-variance) features
    {
        "service": "remove_constant_columns",
        "inputs": {
            "train_data": f"{_P}/artifacts/train_encoded.csv",
            "test_data": f"{_P}/artifacts/test_encoded.csv",
        },
        "outputs": {
            "train_data": f"{_P}/artifacts/train_filtered.csv",
            "test_data": f"{_P}/artifacts/test_filtered.csv",
        },
        "params": {
            "target_column": "y",
            "id_column": "ID",
        },
        "module": "mercedes_benz_greener_manufacturing_services",
    },
    # Step 3: Add PCA + ICA decomposition features
    {
        "service": "add_decomposition_features",
        "inputs": {
            "train_data": f"{_P}/artifacts/train_filtered.csv",
            "test_data": f"{_P}/artifacts/test_filtered.csv",
        },
        "outputs": {
            "train_data": f"{_P}/artifacts/train_enriched.csv",
            "test_data": f"{_P}/artifacts/test_enriched.csv",
            "decomposition": f"{_P}/artifacts/decomposition.pkl",
        },
        "params": {
            "n_components": 20,
            "target_column": "y",
            "id_column": "ID",
            "methods": ["pca", "ica"],
            "random_state": 42,
        },
        "module": "mercedes_benz_greener_manufacturing_services",
    },
    # Step 4: Split train into train/validation
    {
        "service": "split_data",
        "inputs": {"data": f"{_P}/artifacts/train_enriched.csv"},
        "outputs": {
            "train_data": f"{_P}/artifacts/train_split.csv",
            "valid_data": f"{_P}/artifacts/valid_split.csv",
        },
        "params": {"test_size": 0.2, "random_state": 42},
        "module": "mercedes_benz_greener_manufacturing_services",
    },
    # Step 5: Train stacked ensemble regressor (10-fold CV)
    {
        "service": "train_stacked_regressor",
        "inputs": {
            "train_data": f"{_P}/artifacts/train_split.csv",
            "valid_data": f"{_P}/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": f"{_P}/artifacts/model.pkl",
            "metrics": f"{_P}/artifacts/metrics.json",
        },
        "params": {
            "label_column": "y",
            "id_column": "ID",
            "model_types": ["gradient_boosting", "lightgbm", "xgboost", "ridge", "elasticnet"],
            "n_folds": 10,
            "log_target": False,
            "random_state": 42,
            "gbr_n_estimators": 3000,
            "gbr_learning_rate": 0.005,
            "gbr_max_depth": 3,
            "gbr_min_samples_leaf": 10,
            "gbr_subsample": 0.8,
            "lgbm_n_estimators": 5000,
            "lgbm_learning_rate": 0.005,
            "lgbm_num_leaves": 31,
            "lgbm_max_depth": -1,
            "lgbm_min_child_samples": 15,
            "lgbm_subsample": 0.7,
            "lgbm_colsample_bytree": 0.6,
            "lgbm_reg_alpha": 0.05,
            "lgbm_reg_lambda": 0.05,
            "xgb_n_estimators": 5000,
            "xgb_learning_rate": 0.005,
            "xgb_max_depth": 3,
            "xgb_min_child_weight": 5,
            "xgb_subsample": 0.7,
            "xgb_colsample_bytree": 0.6,
            "xgb_reg_alpha": 0.05,
            "xgb_reg_lambda": 1.0,
            "ridge_alpha": 5.0,
            "enet_alpha": 0.001,
            "enet_l1_ratio": 0.3,
            "meta_alpha": 0.5,
        },
        "module": "mercedes_benz_greener_manufacturing_services",
    },
    # Step 6: Generate predictions on test data
    {
        "service": "predict_stacked_regressor",
        "inputs": {
            "model": f"{_P}/artifacts/model.pkl",
            "data": f"{_P}/artifacts/test_enriched.csv",
        },
        "outputs": {
            "predictions": f"{_P}/artifacts/predictions.csv",
        },
        "params": {
            "id_column": "ID",
            "prediction_column": "y",
        },
        "module": "mercedes_benz_greener_manufacturing_services",
    },
    # Step 7: Format submission
    {
        "service": "create_submission",
        "inputs": {
            "predictions": f"{_P}/artifacts/predictions.csv",
        },
        "outputs": {
            "submission": f"{_P}/submission.csv",
        },
        "params": {
            "id_column": "ID",
            "prediction_column": "y",
        },
        "module": "mercedes_benz_greener_manufacturing_services",
    },
]


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "encode_train_test_labels": encode_train_test_labels,
    "target_encode_train_test": target_encode_train_test,
    "remove_constant_columns": remove_constant_columns,
    "add_decomposition_features": add_decomposition_features,
    "split_data": split_data,
    "create_submission": create_submission,
    "drop_columns": drop_columns,
    "label_encode_categorical": label_encode_categorical,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "train_xgboost_regressor": train_xgboost_regressor,
    "predict_regressor": predict_regressor,
    "train_stacked_regressor": train_stacked_regressor,
    "predict_stacked_regressor": predict_stacked_regressor,
}