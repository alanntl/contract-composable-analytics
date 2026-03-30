"""
Preprocessing Services - SLEGO Common Module
=============================================

Generic preprocessing services reusable across any tabular competition.

All services follow G1-G6 design principles:
- G1: Each service does exactly ONE thing (fit != transform)
- G2: Explicit I/O contracts with @contract
- G3: Pure functions, no hidden state (random_state explicit)
- G4: No hardcoded column names (injected via params/config)
- G5: DAG pipeline structure
- G6: Semantic metadata via docstrings/tags

Services:
  Imputation: fit_imputer / transform_imputer
  Encoding: fit_encoder / transform_encoder
  Column Filter: fit_column_filter / transform_column_filter
  Scaling: fit_scaler / transform_scaler
  Feature Engineering: engineer_features (config-driven)
  Data Handling: split_data, create_submission
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


# =============================================================================
# HELPERS: Import from shared io_utils
# =============================================================================

from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# GENERIC SERVICES: IMPUTATION (fit / transform)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular", "allow_missing": True}},
    },
    outputs={
        "artifact": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "imputer"}},
    },
    description="Learn per-column fill values from training data (G1: fit only, no transform)",
    tags=["preprocessing", "imputation", "fit", "generic"],
    version="1.0.0",
)
def fit_imputer(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """
    Learn imputation statistics from data without transforming.

    G1 Compliance: Fit-only operation, produces reusable artifact.
    G4 Compliance: No hardcoded column names, uses exclude_columns param.

    Parameters:
        numeric_strategy: 'median', 'mean', or 'zero' for numeric columns
        categorical_strategy: 'most_frequent' or 'missing' for categorical columns
        exclude_columns: Columns to skip (e.g., target, id)
    """
    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    categorical_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]

    fill_values = {}

    for col in numeric_cols:
        if numeric_strategy == "median":
            fill_values[col] = float(df[col].median())
        elif numeric_strategy == "mean":
            fill_values[col] = float(df[col].mean())
        elif numeric_strategy == "zero":
            fill_values[col] = 0.0
        else:
            fill_values[col] = float(df[col].median())

    for col in categorical_cols:
        if categorical_strategy == "most_frequent":
            mode = df[col].mode()
            fill_values[col] = str(mode.iloc[0]) if len(mode) > 0 else "MISSING"
        else:
            fill_values[col] = "MISSING"

    artifact = {
        "fill_values": fill_values,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_strategy": numeric_strategy,
        "categorical_strategy": categorical_strategy,
    }

    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "wb") as f:
        pickle.dump(artifact, f)

    total_missing = df[numeric_cols + categorical_cols].isnull().sum().sum()
    return f"fit_imputer: learned {len(fill_values)} fill values, {total_missing} missing in source"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular", "allow_missing": True}},
        "artifact": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "imputer"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular", "allow_missing": False}},
    },
    description="Apply learned fill values to data (G1: transform only)",
    tags=["preprocessing", "imputation", "transform", "generic"],
    version="1.0.0",
)
def transform_imputer(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Apply fill values from fit_imputer artifact.

    G1 Compliance: Transform-only, uses pre-fitted artifact.
    G3 Compliance: Deterministic output from inputs + artifact.
    """
    df = _load_data(inputs["data"])

    with open(inputs["artifact"], "rb") as f:
        artifact = pickle.load(f)

    filled = 0
    for col, value in artifact["fill_values"].items():
        if col in df.columns:
            n = df[col].isnull().sum()
            if n > 0:
                df[col] = df[col].fillna(value)
                filled += n

    _save_data(df, outputs["data"])

    return f"transform_imputer: filled {filled} missing values"


# =============================================================================
# GENERIC SERVICES: ENCODING (fit / transform)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "artifact": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "encoder"}},
    },
    description="Learn categorical encoding mapping from training data",
    tags=["preprocessing", "encoding", "fit", "generic"],
    version="1.0.0",
)
def fit_encoder(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    method: str = "onehot",
    exclude_columns: Optional[List[str]] = None,
    max_categories: int = 30,
) -> str:
    """
    Learn encoding mapping without transforming.

    G1 Compliance: Fit-only, produces reusable encoder artifact.
    G4 Compliance: Column names injected via exclude_columns.

    Parameters:
        method: 'onehot', 'ordinal', or 'label'
        exclude_columns: Columns to skip (e.g., target, id)
        max_categories: Cap categories per column (onehot only)
    """
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])

    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]

    artifact = {"method": method, "categorical_columns": cat_cols, "max_categories": max_categories}

    if not cat_cols:
        artifact["encoder"] = None
    elif method == "onehot":
        for col in cat_cols:
            vc = df[col].value_counts()
            if len(vc) > max_categories:
                top = vc.head(max_categories).index.tolist()
                df[col] = df[col].apply(lambda x, t=top: x if x in t else "Other")

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(df[cat_cols])
        artifact["encoder"] = encoder
        artifact["feature_names"] = list(encoder.get_feature_names_out(cat_cols))

    elif method == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(df[cat_cols])
        artifact["encoder"] = encoder

    elif method == "label":
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            encoders[col] = le
        artifact["encoder"] = encoders

    else:
        raise ValueError(f"Unknown encoding method: {method}")

    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "wb") as f:
        pickle.dump(artifact, f)

    return f"fit_encoder ({method}): learned mapping for {len(cat_cols)} columns"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "artifact": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "encoder"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Apply learned encoding to data",
    tags=["preprocessing", "encoding", "transform", "generic"],
    version="1.0.0",
)
def transform_encoder(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Apply encoding from fit_encoder artifact.

    G1 Compliance: Transform-only, uses pre-fitted artifact.
    Handles unknown categories gracefully (ignored for onehot, -1 for ordinal).
    """
    df = _load_data(inputs["data"])

    with open(inputs["artifact"], "rb") as f:
        artifact = pickle.load(f)

    method = artifact["method"]
    cat_cols = [c for c in artifact["categorical_columns"] if c in df.columns]
    encoder = artifact["encoder"]
    original_shape = df.shape

    if not cat_cols or encoder is None:
        _save_data(df, outputs["data"])
        return "transform_encoder: no categorical columns to encode"

    if method == "onehot":
        max_categories = artifact.get("max_categories", 30)
        for col in cat_cols:
            idx = artifact["categorical_columns"].index(col)
            known = set(encoder.categories_[idx])
            df[col] = df[col].apply(lambda x, k=known: x if x in k else "Other")

        encoded = encoder.transform(df[cat_cols])
        feature_names = artifact.get("feature_names", list(encoder.get_feature_names_out(cat_cols)))
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded_df], axis=1)

    elif method == "ordinal":
        df[cat_cols] = encoder.transform(df[cat_cols])

    elif method == "label":
        for col in cat_cols:
            if col in encoder:
                le = encoder[col]
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x, k=known, l=le: l.transform([x])[0] if x in k else -1
                )

    _save_data(df, outputs["data"])

    return f"transform_encoder ({method}): {original_shape} -> {df.shape}"



@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "artifact": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "encoder"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Inverse transform encoded data (e.g. map predictions back to labels)",
    tags=["preprocessing", "encoding", "inverse_transform", "generic"],
    version="1.0.0",
)
def inverse_transform_encoder(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_columns: Optional[List[str]] = None,
) -> str:
    """
    Inverse transform specific columns using the encoder artifact.
    Useful for mapping predicted labels back to original class names.

    Parameters:
        target_columns: List of columns to inverse transform. If None, tries to use all categorical columns from artifact.
    """
    df = _load_data(inputs["data"])

    with open(inputs["artifact"], "rb") as f:
        artifact = pickle.load(f)

    method = artifact["method"]
    encoder = artifact["encoder"]
    
    # Determine columns to process
    cols_to_process = target_columns if target_columns else [c for c in artifact["categorical_columns"] if c in df.columns]

    if not cols_to_process or encoder is None:
        _save_data(df, outputs["data"])
        return "inverse_transform_encoder: no columns to process"

    if method == "label":
        count = 0
        for col in cols_to_process:
            # For label encoding, we expect 'encoder' to be a dict of LabelEncoders
            if isinstance(encoder, dict) and col in encoder:
                le = encoder[col]
                try:
                     # valid integers for inverse_transform
                    df[col] = le.inverse_transform(df[col].astype(int))
                    count += 1
                except Exception as e:
                    print(f"Warning: Could not inverse transform column {col}: {e}")
            else:
                print(f"Warning: No encoder found for column {col}")
        
    elif method == "ordinal":
        pass # Not fully supported for partial column inverse yet
    
    _save_data(df, outputs["data"])
    return f"inverse_transform_encoder ({method}): processed {len(cols_to_process)} columns"


# =============================================================================
# GENERIC SERVICES: COLUMN FILTER (fit / transform)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular", "allow_missing": True}},
    },
    outputs={
        "artifact": {"format": "json", "schema": {"type": "json"}},
    },
    description="Identify columns to keep/drop based on missing value threshold",
    tags=["preprocessing", "cleaning", "fit", "generic"],
    version="1.0.0",
)
def fit_column_filter(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    missing_threshold: float = 0.8,
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """
    Identify which columns exceed the missing-value threshold.

    G1 Compliance: Fit-only, produces column list artifact.
    G4 Compliance: No hardcoded columns, uses exclude_columns param.

    Parameters:
        missing_threshold: Drop columns with missing ratio > threshold (0.0 to 1.0)
        exclude_columns: Columns to always keep (e.g., target, id)
    """
    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])

    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = [c for c in missing_ratio[missing_ratio > missing_threshold].index if c not in exclude]
    cols_to_keep = [c for c in df.columns if c not in cols_to_drop]

    artifact = {
        "columns_to_keep": cols_to_keep,
        "columns_dropped": cols_to_drop,
        "missing_threshold": missing_threshold,
    }

    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "w") as f:
        json.dump(artifact, f, indent=2)

    return f"fit_column_filter: keeping {len(cols_to_keep)}, dropping {len(cols_to_drop)} (>{missing_threshold*100:.0f}% missing)"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "artifact": {"format": "json", "required": True, "schema": {"type": "json"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Keep only columns from fit_column_filter artifact",
    tags=["preprocessing", "cleaning", "transform", "generic"],
    version="1.0.0",
)
def transform_column_filter(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Keep only columns from the artifact.

    G1 Compliance: Transform-only, uses pre-fitted column list.
    Missing columns in new data are silently skipped.
    """
    df = _load_data(inputs["data"])

    with open(inputs["artifact"], "r") as f:
        artifact = json.load(f)

    cols_to_keep = [c for c in artifact["columns_to_keep"] if c in df.columns]
    original_cols = len(df.columns)
    df = df[cols_to_keep]

    _save_data(df, outputs["data"])

    return f"transform_column_filter: {original_cols} -> {len(cols_to_keep)} columns"


# =============================================================================
# GENERIC SERVICES: SCALING (fit / transform)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "artifact": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "scaler"}},
    },
    description="Learn scaler parameters from training data (G1: fit only, no transform)",
    tags=["preprocessing", "scaling", "fit", "generic"],
    version="1.0.0",
)
def fit_scaler(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    method: str = "standard",
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """
    Learn scaler statistics from data without transforming.

    G1 Compliance: Fit-only operation, produces reusable artifact.
    G4 Compliance: No hardcoded column names, uses exclude_columns param.

    Parameters:
        method: 'standard' (StandardScaler), 'robust' (RobustScaler),
                or 'minmax' (MinMaxScaler)
        exclude_columns: Columns to skip (e.g., target, id, categorical)
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard', 'robust', or 'minmax'.")

    if numeric_cols:
        scaler.fit(df[numeric_cols])

    artifact = {
        "scaler": scaler,
        "numeric_columns": numeric_cols,
        "method": method,
    }

    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "wb") as f:
        pickle.dump(artifact, f)

    return f"fit_scaler ({method}): learned scaling for {len(numeric_cols)} numeric columns"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "artifact": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "scaler"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Apply learned scaler to data (G1: transform only)",
    tags=["preprocessing", "scaling", "transform", "generic"],
    version="1.0.0",
)
def transform_scaler(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Apply scaler from fit_scaler artifact.

    G1 Compliance: Transform-only, uses pre-fitted artifact.
    G3 Compliance: Deterministic output from inputs + artifact.
    Only scales numeric columns that were present during fit.
    """
    df = _load_data(inputs["data"])

    with open(inputs["artifact"], "rb") as f:
        artifact = pickle.load(f)

    method = artifact["method"]
    numeric_cols = [c for c in artifact["numeric_columns"] if c in df.columns]
    scaler = artifact["scaler"]

    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    _save_data(df, outputs["data"])

    return f"transform_scaler ({method}): scaled {len(numeric_cols)} numeric columns"


# =============================================================================
# GENERIC SERVICES: DATA SPLITTING
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular", "min_rows": 10}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Split data into train and validation sets",
    tags=["data-handling", "splitting", "generic"],
    version="1.0.0",
)
def split_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: Optional[str] = None,
    shuffle: bool = True,
) -> str:
    """
    Split data into train and validation sets.

    G3 Compliance: Explicit random_state for reproducibility.
    G4 Compliance: Optional stratify_column injected via param.

    Parameters:
        test_size: Fraction of data for validation (0.0 to 1.0)
        random_state: Seed for reproducibility
        stratify_column: Column for stratified splitting (classification)
        shuffle: Whether to shuffle data before splitting (default True).
                 Set to False for sequential/time-series data.
    """
    from sklearn.model_selection import train_test_split

    df = _load_data(inputs["data"])

    stratify = None
    if stratify_column and stratify_column in df.columns and shuffle:
        stratify = df[stratify_column]

    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify, shuffle=shuffle
    )

    _save_data(train_df, outputs["train_data"])
    _save_data(valid_df, outputs["valid_data"])

    return f"split_data: train={len(train_df)}, valid={len(valid_df)}"


# =============================================================================
# GENERIC SERVICES: SUBMISSION FORMATTING
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Format predictions for Kaggle submission",
    tags=["inference", "submission", "kaggle", "generic"],
    version="1.0.0",
)
def create_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "target",
    **kwargs,
) -> str:
    """
    Format predictions for Kaggle submission.

    G4 Compliance: Column names injected via params, not hardcoded.

    Parameters:
        id_column: Name of the ID column
        prediction_column: Name of the prediction column

    Also accepts extra kwargs (e.g. ``predictions``, ``submission``) so that
    pipeline specs which pass file paths as params instead of inside the
    ``inputs``/``outputs`` dicts still work.
    """
    # Resolve predictions path: prefer inputs dict, fall back to kwarg / params
    pred_path = inputs.get("predictions") or kwargs.get("predictions")
    if not pred_path:
        raise ValueError("create_submission: no predictions path provided in inputs or params")

    # Resolve submission output path: prefer outputs dict, fall back to kwarg
    sub_path = outputs.get("submission") or kwargs.get("submission")
    if not sub_path:
        raise ValueError("create_submission: no submission path provided in outputs or params")

    pred_df = _load_data(pred_path)
    submission_df = pred_df[[id_column, prediction_column]].copy()

    _save_data(submission_df, sub_path)

    return f"create_submission: {len(submission_df)} rows"


# =============================================================================
# GENERIC SERVICES: FEATURE ENGINEERING (config-driven)
# =============================================================================

# Example feature config for house prices competition (can be overridden via params)
HOUSE_PRICES_FEATURE_CONFIG = {
    "sum_features": [
        {"name": "TotalSF", "columns": ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]},
        {"name": "TotalPorch", "columns": ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]},
    ],
    "add_features": [
        {"name": "TotalArea", "columns": ["GrLivArea", "TotalBsmtSF"]},
    ],
    "weighted_sum_features": [
        {"name": "TotalBath", "weights": {"FullBath": 1.0, "HalfBath": 0.5, "BsmtFullBath": 1.0, "BsmtHalfBath": 0.5}},
    ],
    "difference_features": [
        {"name": "HouseAge", "minuend": "YrSold", "subtrahend": "YearBuilt"},
        {"name": "RemodAge", "minuend": "YrSold", "subtrahend": "YearRemodAdd"},
    ],
    "product_features": [
        {"name": "OverallScore", "columns": ["OverallQual", "OverallCond"]},
        {"name": "QualArea", "columns": ["OverallQual", "GrLivArea"]},
    ],
    "binary_features": [
        {"name": "HasGarage", "column": "GarageArea", "threshold": 0},
        {"name": "HasBsmt", "column": "TotalBsmtSF", "threshold": 0},
        {"name": "HasPool", "column": "PoolArea", "threshold": 0},
        {"name": "Has2ndFloor", "column": "2ndFlrSF", "threshold": 0},
        {"name": "HasFireplace", "column": "Fireplaces", "threshold": 0},
    ],
}


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Create features based on configuration (sums, products, differences, binary flags)",
    tags=["feature-engineering", "generic"],
    version="1.0.0",
)
def engineer_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    feature_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create features based on a configuration dictionary.

    G1 Compliance: Single responsibility - create features from config.
    G4 Compliance: No hardcoded column names - all injected via feature_config.

    Parameters:
        feature_config: Dict defining features to create. Supports:
            - sum_features: [{"name": "TotalSF", "columns": ["col1", "col2"]}]
            - add_features: [{"name": "Total", "columns": ["col1", "col2"]}] (same as sum)
            - weighted_sum_features: [{"name": "TotalBath", "weights": {"col1": 1.0, "col2": 0.5}}]
            - difference_features: [{"name": "Age", "minuend": "col1", "subtrahend": "col2"}]
            - product_features: [{"name": "Score", "columns": ["col1", "col2"]}]
            - binary_features: [{"name": "HasX", "column": "col", "threshold": 0}]
    """
    df = _load_data(inputs["data"])

    # Resolve string config references (e.g. "HOUSE_PRICES_FEATURE_CONFIG_V2")
    # to the actual dict by looking up known config constants in this module
    # and the calling module's namespace.
    if isinstance(feature_config, str):
        import importlib
        # Search this module's globals first
        resolved = globals().get(feature_config)
        if resolved is None:
            # Search all loaded service modules for the constant
            import sys as _sys
            for mod_name, mod in _sys.modules.items():
                if 'services' in mod_name and hasattr(mod, feature_config):
                    resolved = getattr(mod, feature_config)
                    break
        if isinstance(resolved, dict):
            feature_config = resolved
        else:
            feature_config = {}

    config = feature_config or {}
    features_added = []

    # Sum features: sum of multiple columns
    for feat in config.get("sum_features", []) + config.get("add_features", []):
        name = feat["name"]
        cols = feat["columns"]
        total = pd.Series(0.0, index=df.index)
        for col in cols:
            if col in df.columns:
                total = total + df[col].fillna(0)
        df[name] = total
        features_added.append(name)

    # Weighted sum features: weighted sum of columns
    for feat in config.get("weighted_sum_features", []):
        name = feat["name"]
        weights = feat["weights"]
        total = pd.Series(0.0, index=df.index)
        for col, weight in weights.items():
            if col in df.columns:
                total = total + weight * df[col].fillna(0)
            else:
                # If column missing, use 0
                total = total + weight * df.get(col, pd.Series(0, index=df.index)).fillna(0)
        df[name] = total
        features_added.append(name)

    # Difference features: col1 - col2
    for feat in config.get("difference_features", []):
        name = feat["name"]
        minuend = feat["minuend"]
        subtrahend = feat["subtrahend"]
        if minuend in df.columns and subtrahend in df.columns:
            df[name] = df[minuend].fillna(0) - df[subtrahend].fillna(0)
            features_added.append(name)

    # Product features: multiply columns together
    for feat in config.get("product_features", []):
        name = feat["name"]
        cols = feat["columns"]
        if all(c in df.columns for c in cols):
            result = pd.Series(1.0, index=df.index)
            for col in cols:
                result = result * df[col].fillna(0)
            df[name] = result
            features_added.append(name)

    # Binary features: column > threshold
    for feat in config.get("binary_features", []):
        name = feat["name"]
        col = feat["column"]
        threshold = feat.get("threshold", 0)
        if col in df.columns:
            df[name] = (df[col].fillna(0) > threshold).astype(int)
            features_added.append(name)

    # Ratio features: numerator / denominator (safe division)
    for feat in config.get("ratio_features", []):
        name = feat["name"]
        num = feat["numerator"]
        den = feat["denominator"]
        if num in df.columns and den in df.columns:
            denominator = df[den].fillna(0).replace(0, np.nan)
            df[name] = df[num].fillna(0) / denominator
            df[name] = df[name].fillna(0)
            features_added.append(name)

    # Power features: column^power
    for feat in config.get("power_features", []):
        name = feat["name"]
        col = feat["column"]
        power = feat.get("power", 2)
        if col in df.columns:
            df[name] = df[col].fillna(0) ** power
            features_added.append(name)

    # Factorized interaction features: unique ID for each combination of columns
    # Key technique from 1st place solution: pd.factorize(col1 + '_' + col2)[0]
    for feat in config.get("factorized_features", []):
        name = feat["name"]
        cols = feat["columns"]
        if all(c in df.columns for c in cols):
            combined = df[cols[0]].astype(str)
            for col in cols[1:]:
                combined = combined + '_' + df[col].astype(str)
            df[name] = pd.factorize(combined)[0]
            features_added.append(name)

    _save_data(df, outputs["data"])

    return f"engineer_features: added {len(features_added)} features: {features_added}"


# =============================================================================
# LABEL ENCODING SERVICE
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
        "encoder": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Label encode categorical columns (including target) to numeric values",
    tags=["preprocessing", "encoding", "categorical", "generic"],
    version="1.0.0",
)
def label_encode_categorical(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[List[str]] = None,
    include_target: bool = True,
    target_column: Optional[str] = None,
) -> str:
    """
    Label encode specified categorical columns to numeric values.

    Uses pandas factorize for simple integer encoding (0, 1, 2, ...).
    Saves encoder mapping for later use on test data.

    G1 Compliance: Single responsibility - label encoding only
    G4 Compliance: Parameterized columns, no hardcoding

    Parameters:
        columns: List of columns to encode. If None, encodes all object columns.
        include_target: Whether to also encode target_column if specified.
        target_column: Target column name (for classification targets that need encoding).
    """
    df = _load_data(inputs["data"])

    # Determine columns to encode
    if columns is None:
        encode_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    else:
        encode_cols = [c for c in columns if c in df.columns]

    # Optionally include target
    if include_target and target_column and target_column in df.columns:
        if target_column not in encode_cols:
            encode_cols.append(target_column)

    # Store encodings for later use on test data
    encodings = {}

    for col in encode_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            encodings[col] = {str(v): int(i) for i, v in enumerate(uniques)}

    _save_data(df, outputs["data"])

    # Save encoder artifact
    os.makedirs(os.path.dirname(outputs["encoder"]) or ".", exist_ok=True)
    with open(outputs["encoder"], "wb") as f:
        pickle.dump(encodings, f)

    return f"label_encode_categorical: encoded {len(encode_cols)} columns: {encode_cols}"


# =============================================================================
# SIMPLE ONE-STEP UTILITIES (combines fit+transform)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Fill missing values with specified strategy",
    tags=["preprocessing", "imputation", "generic"],
    version="1.0.0",
)
def fill_missing(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    strategy: str = "median",
    fill_value: Optional[float] = None,
) -> str:
    """
    Simple one-step missing value imputation.

    G1 Compliance: Single responsibility - fill missing values.
    G4 Compliance: Strategy parameterized.

    Parameters:
        strategy: 'mean', 'median', 'most_frequent', 'constant', or 'zero'
        fill_value: Value to use when strategy='constant'
    """
    df = _load_data(inputs["data"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif strategy == "constant" and fill_value is not None:
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)
    elif strategy == "most_frequent":
        for col in numeric_cols:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])

    # Fill categorical with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val.iloc[0])

    _save_data(df, outputs["data"])

    n_filled = df.isnull().sum().sum()
    return f"fill_missing ({strategy}): filled missing in {len(numeric_cols)} numeric columns"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Drop specified columns from dataframe",
    tags=["preprocessing", "feature-selection", "generic"],
    version="1.0.0",
)
def drop_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = [],
) -> str:
    """
    Simple service to drop specified columns.

    G1 Compliance: Single responsibility - drop columns.
    G4 Compliance: Column names parameterized.
    """
    df = _load_data(inputs["data"])
    to_drop = [c for c in columns if c in df.columns]
    df = df.drop(columns=to_drop)
    _save_data(df, outputs["data"])
    return f"drop_columns: dropped {len(to_drop)} columns"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
        "label_mapping": {"format": "json", "required": False, "schema": {"type": "object"}},
    },
    description="Filter data to keep only top N most frequent classes",
    tags=["preprocessing", "filtering", "classification", "generic"],
    version="1.1.0",
)
def filter_top_classes(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    top_n: int = 50,
    min_samples: int = 2,
) -> str:
    """
    Filter dataset to keep only the top N most frequent classes.

    Useful for many-class classification problems where training on all classes
    is computationally prohibitive. Keeps only samples belonging to the top N
    most frequent classes.

    Parameters:
        label_column: Column containing class labels
        top_n: Number of top classes to keep
        min_samples: Minimum samples required for a class to be considered

    Outputs:
        data: Filtered dataset with labels re-encoded to 0..n-1
        label_mapping: JSON mapping from integer label to original class name (optional)
    """
    df = _load_data(inputs["data"])

    # Count class frequencies
    class_counts = df[label_column].value_counts()

    # Filter to classes with min_samples
    valid_classes = class_counts[class_counts >= min_samples]

    # Take top N classes
    top_classes = valid_classes.head(top_n).index.tolist()

    # Filter dataframe
    df_filtered = df[df[label_column].isin(top_classes)].copy()

    # Re-encode labels to be consecutive integers (0 to n-1)
    # Create bidirectional mapping: int->original and original->int
    label_to_int = {label: i for i, label in enumerate(top_classes)}
    int_to_label = {i: str(label) for i, label in enumerate(top_classes)}
    df_filtered[label_column] = df_filtered[label_column].map(label_to_int)

    _save_data(df_filtered, outputs["data"])

    # Save label mapping if output path provided
    if "label_mapping" in outputs and outputs["label_mapping"]:
        mapping_path = outputs["label_mapping"]
        os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
        with open(mapping_path, "w") as f:
            json.dump(int_to_label, f, indent=2)

    return f"filter_top_classes: kept {len(top_classes)} classes, {len(df_filtered)}/{len(df)} samples"


# =============================================================================
# DATA LOADING AND MERGING SERVICES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv.gz", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Read compressed .gz CSV file and output as regular CSV",
    tags=["io", "preprocessing", "generic"],
    version="1.0.0",
)
def read_compressed_csv(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    sample_size: Optional[int] = None,
    chunksize: int = 50000,
) -> str:
    """
    Read a gzip-compressed CSV file.

    Parameters:
        sample_size: If specified, only read first N rows (for large datasets)
        chunksize: Chunk size for reading (memory optimization)
    """
    import gzip

    input_path = inputs["data"]

    if sample_size:
        # Read in chunks up to sample_size
        chunks = []
        rows_read = 0
        with gzip.open(input_path, 'rt') as f:
            chunk_iter = pd.read_csv(f, chunksize=chunksize)
            for chunk in chunk_iter:
                remaining = sample_size - rows_read
                if remaining <= 0:
                    break
                chunks.append(chunk.head(remaining))
                rows_read += len(chunk)
                if rows_read >= sample_size:
                    break
        df = pd.concat(chunks, ignore_index=True)
    else:
        with gzip.open(input_path, 'rt') as f:
            df = pd.read_csv(f)

    _save_data(df, outputs["data"])
    return f"read_compressed_csv: loaded {len(df)} rows, {len(df.columns)} columns"


@contract(
    inputs={
        "left": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "right": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Merge two dataframes on a common column",
    tags=["preprocessing", "data-handling", "generic"],
    version="1.0.0",
)
def merge_dataframes(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    on: Any = "id",
    how: str = "inner",
) -> str:
    """
    Merge two dataframes on common column(s).

    Parameters:
        on: Column name or list of column names to merge on
        how: Merge type ('inner', 'left', 'right', 'outer')
    """
    left_df = _load_data(inputs["left"])
    right_df = _load_data(inputs["right"])

    # Handle list of columns for on parameter
    merge_on = on if isinstance(on, list) else on

    merged_df = left_df.merge(right_df, on=merge_on, how=how)

    _save_data(merged_df, outputs["data"])
    on_str = str(on) if isinstance(on, list) else on
    return f"merge_dataframes: merged on {on_str}, {len(left_df)} + {len(right_df)} -> {len(merged_df)} rows"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Encode all object columns as numeric codes",
    tags=["preprocessing", "encoding", "generic"],
    version="1.0.0",
)
def encode_all_categorical(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    exclude_columns: Optional[List[str]] = None,
    fill_missing: str = "MISSING",
) -> str:
    """
    Convert all object/categorical columns to numeric codes.

    Parameters:
        exclude_columns: List of columns to exclude from encoding
        fill_missing: String to use for missing values before encoding
    """
    df = _load_data(inputs["data"])

    exclude = exclude_columns or []
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclude]

    for col in categorical_cols:
        df[col] = df[col].fillna(fill_missing)
        codes, _ = pd.factorize(df[col])
        df[col] = codes

    # Fill remaining numeric missing values with 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    _save_data(df, outputs["data"])
    return f"encode_all_categorical: encoded {len(categorical_cols)} columns"


@contract(
    inputs={
        "data": {"format": "json", "required": True, "schema": {"type": "json"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Convert JSON data to CSV format",
    tags=["io", "preprocessing", "generic"],
    version="1.0.0",
)
def json_to_csv(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    flatten_tokens: bool = False,
) -> str:
    """
    Convert JSON data to CSV format.

    For NER/token data, can flatten tokens to document-level.

    Parameters:
        text_column: Column containing text to extract
        label_column: Column containing labels to extract
        flatten_tokens: If True, create document-level features from token data
    """
    import json as json_lib

    with open(inputs["data"]) as f:
        data = json_lib.load(f)

    if flatten_tokens and isinstance(data, list) and len(data) > 0:
        # Handle NER-style data - convert to document-level
        rows = []
        for doc in data:
            row = {}
            if 'document' in doc:
                row['document'] = doc['document']
            if 'full_text' in doc:
                row['text'] = doc['full_text']
            if 'tokens' in doc and 'labels' in doc:
                # Check if any non-O labels exist
                has_entity = any(l != 'O' for l in doc['labels'])
                row['has_pii'] = 1 if has_entity else 0
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(data)

    _save_data(df, outputs["data"])
    return f"json_to_csv: converted {len(df)} rows"


# =============================================================================
# OUTLIER REMOVAL SERVICE
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Remove rows matching configurable outlier conditions (generic)",
    tags=["preprocessing", "outliers", "generic"],
    version="1.0.0",
)
def remove_outliers(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    conditions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Remove outlier rows based on configurable conditions.

    G1 Compliance: Single responsibility - remove outliers.
    G4 Compliance: No hardcoded column names - all injected via conditions.

    Parameters:
        conditions: List of condition dicts, each with:
            - column: str (column name)
            - op: str (">", "<", ">=", "<=", "==", "!=")
            - value: numeric threshold
          Rows matching ALL conditions simultaneously are removed.
          Example: [{"column": "GrLivArea", "op": ">", "value": 4000}]
    """
    import operator as op_module

    df = _load_data(inputs["data"])
    n_before = len(df)

    if not conditions:
        _save_data(df, outputs["data"])
        return f"remove_outliers: no conditions specified, {n_before} rows unchanged"

    ops = {
        ">": op_module.gt,
        "<": op_module.lt,
        ">=": op_module.ge,
        "<=": op_module.le,
        "==": op_module.eq,
        "!=": op_module.ne,
    }

    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        col = cond["column"]
        op_str = cond["op"]
        val = cond["value"]
        if col not in df.columns:
            continue
        op_fn = ops.get(op_str)
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {op_str}. Use: {list(ops.keys())}")
        mask = mask & op_fn(df[col].fillna(0), val)

    df = df[~mask]
    n_removed = n_before - len(df)

    _save_data(df, outputs["data"])
    return f"remove_outliers: removed {n_removed} rows ({n_before} -> {len(df)})"


# =============================================================================
# SKEW CORRECTION SERVICES (fit/transform)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "artifact": {"format": "json", "schema": {"type": "json"}},
    },
    description="Identify highly skewed numeric columns for log1p correction",
    tags=["preprocessing", "skew", "fit", "generic"],
    version="1.0.0",
)
def fit_skew_corrector(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    skew_threshold: float = 0.75,
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """
    Identify numeric columns with absolute skewness above threshold.

    G1 Compliance: Single responsibility - detect skewed columns.
    G4 Compliance: Parameterized threshold and exclude_columns.

    Parameters:
        skew_threshold: Absolute skewness threshold (default 0.75)
        exclude_columns: Columns to exclude from skew analysis
    """
    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    skew_values = df[numeric_cols].skew()
    skewed_cols = skew_values[skew_values.abs() > skew_threshold].index.tolist()

    artifact = {
        "skewed_columns": skewed_cols,
        "skew_threshold": skew_threshold,
        "skew_values": {c: float(skew_values[c]) for c in skewed_cols},
    }

    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "w") as f:
        json.dump(artifact, f, indent=2)

    return f"fit_skew_corrector: {len(skewed_cols)} columns above threshold {skew_threshold}"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "artifact": {"format": "json", "required": True, "schema": {"type": "json"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Apply log1p transformation to skewed columns identified by fit_skew_corrector",
    tags=["preprocessing", "skew", "transform", "generic"],
    version="1.0.0",
)
def transform_skew_corrector(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Apply log1p to columns identified as skewed by fit_skew_corrector.

    G1 Compliance: Single responsibility - transform skewed columns.
    G5 Compliance: Uses artifact from fit_skew_corrector (no data leakage).
    """
    df = _load_data(inputs["data"])

    with open(inputs["artifact"]) as f:
        artifact = json.load(f)

    skewed_cols = artifact["skewed_columns"]
    applied = []
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(np.maximum(df[col].fillna(0), 0))
            applied.append(col)

    _save_data(df, outputs["data"])
    return f"transform_skew_corrector: applied log1p to {len(applied)} columns"


# =============================================================================
# GENERIC SERVICES: FEATURE SELECTION (RFE)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "artifact": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "rfe_selector"}},
    },
    description="Fit RFE feature selector to identify most important features (G1: fit only)",
    tags=["preprocessing", "feature-selection", "rfe", "generic"],
    version="1.0.0",
)
def fit_rfe_selector(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    id_column: Optional[str] = "id",
    n_features_to_select: int = 25,
    estimator_type: str = "logistic",
    estimator_C: float = 0.1,
    estimator_penalty: str = "l1",
    step: int = 1,
    random_state: int = 42,
) -> str:
    """Fit Recursive Feature Elimination to select the most important features.

    Uses a wrapped estimator (default: L1 LogisticRegression) to rank features
    and selects the top n_features_to_select. Saves the selector artifact for
    use with transform_rfe_selector.

    G1 Compliance: Fit-only, produces reusable artifact.
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    df = _load_data(inputs["data"])

    exclude = set()
    if id_column and id_column in df.columns:
        exclude.add(id_column)
    if target_column and target_column in df.columns:
        exclude.add(target_column)
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols]
    y = df[target_column]

    if estimator_type == "logistic":
        base = LogisticRegression(
            penalty=estimator_penalty, C=estimator_C,
            solver="liblinear", random_state=random_state, max_iter=1000,
        )
    elif estimator_type == "rf":
        base = RandomForestClassifier(
            n_estimators=50, max_depth=3, random_state=random_state, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown estimator_type: {estimator_type}")

    selector = RFE(base, n_features_to_select=n_features_to_select, step=step)
    selector.fit(X, y)

    selected = [c for c, s in zip(feature_cols, selector.support_) if s]

    artifact = {
        "selector": selector,
        "feature_cols": feature_cols,
        "selected_features": selected,
        "n_features_to_select": n_features_to_select,
    }

    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "wb") as f:
        pickle.dump(artifact, f)

    return f"fit_rfe_selector: selected {len(selected)} of {len(feature_cols)} features"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "artifact": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "rfe_selector"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Apply fitted RFE selector to keep only selected features (G1: transform only)",
    tags=["preprocessing", "feature-selection", "rfe", "transform", "generic"],
    version="1.0.0",
)
def transform_rfe_selector(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    id_column: Optional[str] = "id",
) -> str:
    """Apply RFE selector from fit_rfe_selector to filter features.

    Keeps only the features selected during fit, plus id and target columns.
    G1 Compliance: Transform-only, uses pre-fitted artifact.
    """
    df = _load_data(inputs["data"])

    with open(inputs["artifact"], "rb") as f:
        artifact = pickle.load(f)

    selected = artifact["selected_features"]

    keep = []
    if id_column and id_column in df.columns:
        keep.append(id_column)
    if target_column and target_column in df.columns:
        keep.append(target_column)
    keep += [c for c in selected if c in df.columns]

    df = df[keep]
    _save_data(df, outputs["data"])

    return f"transform_rfe_selector: kept {len(selected)} features"


# =============================================================================
# ONE-HOT TO MULTICLASS TARGET CONVERSION
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Convert one-hot encoded target columns to a single multiclass integer target",
    tags=["preprocessing", "target-encoding", "multiclass", "generic"],
    version="1.0.0",
)
def convert_onehot_to_multiclass(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    onehot_columns: List[str] = [],
    target_column: str = "target",
    drop_onehot: bool = True,
) -> str:
    """Convert one-hot encoded target columns to a single integer target column.

    For competitions where the target is represented as multiple binary columns
    (e.g., winner_model_a, winner_model_b, winner_tie), converts to a single
    integer column (0, 1, 2) based on which one-hot column is 1.

    G1 Compliance: Generic, works with any one-hot encoded target.
    G4 Compliance: All column names as parameters.

    Parameters:
        onehot_columns: List of one-hot encoded column names (order defines class index)
        target_column: Name for the output integer target column
        drop_onehot: Whether to drop the original one-hot columns
    """
    df = _load_data(inputs["data"])

    # Convert one-hot to integer: argmax across the specified columns
    onehot_df = df[onehot_columns]
    df[target_column] = onehot_df.values.argmax(axis=1)

    if drop_onehot:
        df = df.drop(columns=onehot_columns)

    _save_data(df, outputs["data"])

    n_classes = len(onehot_columns)
    class_dist = df[target_column].value_counts().to_dict()
    return f"convert_onehot_to_multiclass: {n_classes} classes, distribution={class_dist}"


# =============================================================================
# ROW-LEVEL STATISTICS SERVICE
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Add row-level statistics (missing count, std, var) as features",
    tags=["feature-engineering", "row-statistics", "generic"],
    version="1.0.0",
)
def add_row_statistics(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    exclude_columns: Optional[List[str]] = None,
    add_missing: bool = True,
    add_std: bool = True,
    add_var: bool = True,
    add_mean: bool = False,
    add_sum: bool = False,
    add_min: bool = False,
    add_max: bool = False,
) -> str:
    """
    Add row-level statistics as new features.

    These meta-features capture row-level patterns that can improve model performance.
    Inspired by top Kaggle solutions that use missing count, std, var, min, max per row.

    G1 Compliance: Single responsibility - add row statistics.
    G4 Compliance: Parameterized columns via exclude_columns.

    Parameters:
        exclude_columns: Columns to exclude from statistics (e.g., id, target)
        add_missing: Add count of missing values per row
        add_std: Add standard deviation per row
        add_var: Add variance per row
        add_mean: Add mean per row
        add_sum: Add sum per row
        add_min: Add min value per row
        add_max: Add max value per row
    """
    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])

    # Get numeric feature columns only
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    features_added = []

    if add_missing:
        df["row_missing"] = df[numeric_cols].isnull().sum(axis=1)
        features_added.append("row_missing")

    if add_std:
        df["row_std"] = df[numeric_cols].std(axis=1)
        features_added.append("row_std")

    if add_var:
        df["row_var"] = df[numeric_cols].var(axis=1)
        features_added.append("row_var")

    if add_mean:
        df["row_mean"] = df[numeric_cols].mean(axis=1)
        features_added.append("row_mean")

    if add_sum:
        df["row_sum"] = df[numeric_cols].sum(axis=1)
        features_added.append("row_sum")

    if add_min:
        df["row_min"] = df[numeric_cols].min(axis=1)
        features_added.append("row_min")

    if add_max:
        df["row_max"] = df[numeric_cols].max(axis=1)
        features_added.append("row_max")

    _save_data(df, outputs["data"])

    return f"add_row_statistics: added {len(features_added)} features: {features_added}"


# =============================================================================
# MAPPING SERVICE
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Map values in a column using a dictionary",
    tags=["preprocessing", "mapping", "generic"],
    version="1.0.0",
)
def map_column_values(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    column: str,
    mapping: Dict[str, Any],
    default: Any = None,
) -> str:
    """
    Map values in a column using a dictionary.

    G1 Compliance: Single responsibility - map values.
    G4 Compliance: Parameterized column and mapping.

    Parameters:
        column: Column to map
        mapping: Dictionary mapping old values to new values
        default: Value to use if key not found (if None, keeps original)
    """
    df = _load_data(inputs["data"])

    if column in df.columns:
        if default is not None:
             df[column] = df[column].map(mapping).fillna(default)
        else:
             # map only values present in mapping, keep others
             df[column] = df[column].map(lambda x: mapping.get(str(x), mapping.get(x, x)))
    
    _save_data(df, outputs["data"])
    return f"map_column_values: mapped values in '{column}' using {len(mapping)} keys"


# =============================================================================
# PROBABILITY TO LABELS SERVICE
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Convert probability columns to class labels using argmax",
    tags=["preprocessing", "classification", "probabilities", "generic"],
    version="1.0.0",
)
def proba_to_labels(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    proba_columns: List[str] = None,
    label_column: str = "label",
    keep_proba: bool = False,
    class_labels: List[Any] = None,
) -> str:
    """
    Convert probability columns to class labels using argmax.

    For multi-class classification predictions, converts per-class
    probability columns into a single label column by taking argmax.

    G1 Compliance: Single responsibility - convert probabilities to labels.
    G4 Compliance: Parameterized columns and labels.

    Parameters:
        proba_columns: List of probability column names (in class order).
                       If None, auto-detects columns excluding the first (ID) column.
        label_column: Output column name for labels. Default: "label".
        keep_proba: If True, keeps the probability columns. Default: False.
        class_labels: Custom class labels to use (0, 1, ... by default).
                      If provided, maps argmax indices to these labels.
    """
    import numpy as np
    df = _load_data(inputs["data"])

    # Auto-detect probability columns if not specified
    if proba_columns is None:
        # Assume first column is ID, rest are probabilities
        proba_columns = list(df.columns[1:])

    # Get probability values
    proba_values = df[proba_columns].values

    # Take argmax to get predicted class index
    predicted_indices = np.argmax(proba_values, axis=1)

    # Map to class labels if provided
    if class_labels is not None:
        predicted_labels = [class_labels[i] for i in predicted_indices]
    else:
        predicted_labels = predicted_indices

    # Add label column
    df[label_column] = predicted_labels

    # Remove probability columns if not keeping
    if not keep_proba:
        df = df.drop(columns=proba_columns)

    _save_data(df, outputs["data"])
    return f"proba_to_labels: converted {len(proba_columns)} probability columns to '{label_column}' using argmax"


# =============================================================================
# GENERIC SERVICES: DATA COMBINATION
# =============================================================================

@contract(
    inputs={
        "primary_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "secondary_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Concatenate two datasets vertically with optional column renaming",
    tags=["preprocessing", "data-combination", "generic"],
    version="1.0.0",
)
def combine_datasets(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    rename_primary: Optional[Dict[str, str]] = None,
    rename_secondary: Optional[Dict[str, str]] = None,
    drop_columns: Optional[List[str]] = None,
) -> str:
    """
    Concatenate two datasets vertically (row-wise).

    Useful for combining original external data with competition training data.
    Example: Concatenating UCI Abalone dataset with Kaggle Playground training data.

    G1 Compliance: Single-purpose concatenation.
    G4 Compliance: Parameterized column renaming and selection.

    Parameters:
        rename_primary: Dict mapping old->new column names for primary data
        rename_secondary: Dict mapping old->new column names for secondary data
        drop_columns: Columns to drop from both datasets after combining
    """
    primary = _load_data(inputs["primary_data"])
    secondary = _load_data(inputs["secondary_data"])

    # Rename columns if specified
    if rename_primary:
        primary = primary.rename(columns=rename_primary)
    if rename_secondary:
        secondary = secondary.rename(columns=rename_secondary)

    # Get common columns
    common_cols = list(set(primary.columns) & set(secondary.columns))

    # Concatenate on common columns
    combined = pd.concat([primary[common_cols], secondary[common_cols]], axis=0, ignore_index=True)

    # Drop columns if specified
    if drop_columns:
        cols_to_drop = [c for c in drop_columns if c in combined.columns]
        combined = combined.drop(columns=cols_to_drop)

    _save_data(combined, outputs["data"])

    return f"combine_datasets: {len(primary)} + {len(secondary)} = {len(combined)} rows, {len(combined.columns)} columns"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Rename columns in a dataset",
    tags=["preprocessing", "column-manipulation", "generic"],
    version="1.0.0",
)
def rename_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    rename_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Rename columns in a dataset.

    G1 Compliance: Single-purpose column renaming.
    G4 Compliance: Parameterized rename mapping.

    Parameters:
        rename_map: Dict mapping old column names to new column names
    """
    df = _load_data(inputs["data"])

    if rename_map:
        df = df.rename(columns=rename_map)

    _save_data(df, outputs["data"])

    renamed_count = len(rename_map) if rename_map else 0
    return f"rename_columns: renamed {renamed_count} columns"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "fit_imputer": fit_imputer,
    "fit_encoder": fit_encoder,
    "fit_column_filter": fit_column_filter,
    "fit_scaler": fit_scaler,
    "fit_skew_corrector": fit_skew_corrector,
    "fit_rfe_selector": fit_rfe_selector,
    "transform_imputer": transform_imputer,
    "transform_encoder": transform_encoder,
    "transform_column_filter": transform_column_filter,
    "transform_scaler": transform_scaler,
    "transform_skew_corrector": transform_skew_corrector,
    "transform_rfe_selector": transform_rfe_selector,
    "fill_missing": fill_missing,
    "drop_columns": drop_columns,
    "remove_outliers": remove_outliers,
    "split_data": split_data,
    "create_submission": create_submission,
    "engineer_features": engineer_features,
    "label_encode_categorical": label_encode_categorical,
    "filter_top_classes": filter_top_classes,
    "encode_all_categorical": encode_all_categorical,
    "convert_onehot_to_multiclass": convert_onehot_to_multiclass,
    "add_row_statistics": add_row_statistics,
    "map_column_values": map_column_values,
    "proba_to_labels": proba_to_labels,
    "combine_datasets": combine_datasets,
    "rename_columns": rename_columns,
    "inverse_transform_encoder": inverse_transform_encoder,
}
