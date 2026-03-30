"""
Contract-Composable Analytics Feature Engineering Services - Generic Feature Creation
==============================================================
This module provides reusable feature engineering services.
Use these for any competition requiring feature transformation.

Usage:
    from services.feature_engineering_services import create_polynomial_features, create_interaction_features
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
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
# Polynomial and Interaction Features
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "columns": "Columns to create polynomial features for",
        "degree": "Polynomial degree",
        "include_bias": "Include bias (constant) term"
    },
    description="Create polynomial features",
    tags=["feature-engineering", "polynomial", "generic"]
)
def create_polynomial_features(
    data: str,
    output: str,
    columns: List[str] = None,
    degree: int = 2,
    include_bias: bool = False
) -> Dict[str, str]:
    """
    Create polynomial features from numeric columns.

    Works with: regression, small datasets, non-linear relationships.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        columns: Columns to transform (None = all numeric)
        degree: Polynomial degree (2 = squared, 3 = cubed, etc.)
        include_bias: Include constant term
    """
    df = pd.read_csv(data)

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        for d in range(2, degree + 1):
            df[f'{col}_pow{d}'] = df[col] ** d

    if include_bias:
        df['bias'] = 1

    df.to_csv(output, index=False)
    return {'data': output}


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "column_pairs": "List of column pairs to create interactions for",
        "operations": "Operations to apply"
    },
    description="Create interaction features between column pairs",
    tags=["feature-engineering", "interaction", "generic"]
)
def create_interaction_features(
    data: str,
    output: str,
    column_pairs: List[Tuple[str, str]] = None,
    operations: List[str] = None
) -> Dict[str, str]:
    """
    Create interaction features between pairs of columns.

    Works with: any tabular data with related numeric features.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        column_pairs: List of (col1, col2) tuples
        operations: Operations to apply: 'multiply', 'add', 'subtract', 'divide', 'ratio'
    """
    df = pd.read_csv(data)

    if operations is None:
        operations = ['multiply', 'ratio']

    if column_pairs is None:
        # Auto-generate pairs from numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        column_pairs = [(numeric_cols[i], numeric_cols[j])
                        for i in range(len(numeric_cols))
                        for j in range(i+1, len(numeric_cols))]

    for col1, col2 in column_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            continue

        if 'multiply' in operations:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        if 'add' in operations:
            df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
        if 'subtract' in operations:
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
        if 'divide' in operations or 'ratio' in operations:
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)

    df.to_csv(output, index=False)
    return {'data': output}


# =============================================================================
# Binning and Discretization
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "columns": "Columns to bin",
        "n_bins": "Number of bins",
        "strategy": "Binning strategy: uniform, quantile, kmeans",
        "encode": "Encoding: ordinal, onehot"
    },
    description="Create binned features from continuous columns",
    tags=["feature-engineering", "binning", "discretization", "generic"]
)
def create_binned_features(
    data: str,
    output: str,
    columns: List[str] = None,
    n_bins: int = 5,
    strategy: str = 'quantile',
    encode: str = 'ordinal'
) -> Dict[str, str]:
    """
    Bin continuous features into discrete bins.

    Works with: any numeric data that benefits from discretization.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        columns: Columns to bin (None = all numeric)
        n_bins: Number of bins
        strategy: 'uniform' (equal width), 'quantile' (equal frequency)
        encode: 'ordinal' (integer labels) or 'onehot' (dummy variables)
    """
    df = pd.read_csv(data)

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        if strategy == 'quantile':
            df[f'{col}_binned'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        else:  # uniform
            df[f'{col}_binned'] = pd.cut(df[col], bins=n_bins, labels=False)

        if encode == 'onehot':
            dummies = pd.get_dummies(df[f'{col}_binned'], prefix=f'{col}_bin')
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[f'{col}_binned'])

    df.to_csv(output, index=False)
    return {'data': output}


# =============================================================================
# Aggregation Features
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "group_columns": "Columns to group by",
        "agg_columns": "Columns to aggregate",
        "agg_funcs": "Aggregation functions"
    },
    description="Create aggregated features by group",
    tags=["feature-engineering", "aggregation", "generic"]
)
def aggregate_by_group(
    data: str,
    output: str,
    group_columns: List[str],
    agg_columns: List[str],
    agg_funcs: List[str] = None
) -> Dict[str, str]:
    """
    Create group-level aggregate features and merge back.

    Works with: hierarchical data, customer/product data, etc.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        group_columns: Columns to group by
        agg_columns: Columns to aggregate
        agg_funcs: Functions: 'mean', 'sum', 'count', 'min', 'max', 'std'
    """
    df = pd.read_csv(data)

    if agg_funcs is None:
        agg_funcs = ['mean', 'sum', 'count']

    # Create aggregations
    agg_dict = {col: agg_funcs for col in agg_columns if col in df.columns}

    if agg_dict:
        agg_df = df.groupby(group_columns).agg(agg_dict)
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()

        # Merge back
        df = df.merge(agg_df, on=group_columns, how='left')

    df.to_csv(output, index=False)
    return {'data': output}


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={"exclude_columns": "Columns to exclude from statistics"},
    description="Add row-wise statistics as features",
    tags=["feature-engineering", "row-statistics", "generic"]
)
def add_row_statistics(
    data: str,
    output: str,
    exclude_columns: List[str] = None
) -> Dict[str, str]:
    """
    Add row-wise statistics (sum, mean, std, etc.) as features.

    Works with: wide datasets, anonymous features, image pixel data.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        exclude_columns: Columns to exclude (e.g., id, target)
    """
    df = pd.read_csv(data)

    if exclude_columns is None:
        exclude_columns = []

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_columns]

    if numeric_cols:
        df['row_sum'] = df[numeric_cols].sum(axis=1)
        df['row_mean'] = df[numeric_cols].mean(axis=1)
        df['row_std'] = df[numeric_cols].std(axis=1)
        df['row_max'] = df[numeric_cols].max(axis=1)
        df['row_min'] = df[numeric_cols].min(axis=1)
        df['row_nonzero'] = (df[numeric_cols] != 0).sum(axis=1)

    df.to_csv(output, index=False)
    return {'data': output}


# =============================================================================
# Transformation Features
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "columns": "Columns to transform",
        "transform": "Transformation: log1p, sqrt, square, reciprocal"
    },
    description="Apply mathematical transformations to columns",
    tags=["feature-engineering", "transformation", "generic"]
)
def apply_math_transform(
    data: str,
    output: str,
    columns: List[str] = None,
    transform: str = 'log1p'
) -> Dict[str, str]:
    """
    Apply mathematical transformation to numeric columns.

    Works with: skewed distributions, count data, etc.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        columns: Columns to transform (None = all numeric)
        transform: 'log1p', 'sqrt', 'square', 'reciprocal', 'boxcox'
    """
    df = pd.read_csv(data)

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    transform_funcs = {
        'log1p': np.log1p,
        'sqrt': np.sqrt,
        'square': np.square,
        'reciprocal': lambda x: 1 / (x + 1e-8),
        'cbrt': np.cbrt,
    }

    func = transform_funcs.get(transform, np.log1p)

    for col in columns:
        if col not in df.columns:
            continue
        df[f'{col}_{transform}'] = func(df[col].clip(lower=0))

    df.to_csv(output, index=False)
    return {'data': output}


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    params={
        "columns": "Columns to scale",
        "method": "Scaling method: standard, minmax, robust"
    },
    description="Scale numeric features",
    tags=["feature-engineering", "scaling", "normalization", "generic"]
)
def scale_features(
    data: str,
    output: str,
    columns: List[str] = None,
    method: str = 'standard'
) -> Dict[str, str]:
    """
    Scale numeric features using various methods.

    Args:
        data: Path to input CSV
        output: Path to output CSV
        columns: Columns to scale (None = all numeric)
        method: 'standard' (z-score), 'minmax' (0-1), 'robust' (IQR)
    """
    df = pd.read_csv(data)

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'standard':
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + 1e-8)
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
        elif method == 'robust':
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = (df[col] - median) / (iqr + 1e-8)

    df.to_csv(output, index=False)
    return {'data': output}


# =============================================================================
# PCA Dimensionality Reduction Services
# =============================================================================

import pickle


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"artifact": {"format": "pickle"}},
    params={
        "n_components": "Number of PCA components",
        "whiten": "Apply whitening normalization",
        "exclude_columns": "Columns to exclude from PCA"
    },
    description="Fit PCA on training data - effective for high-dimensional datasets",
    tags=["feature-engineering", "pca", "dimensionality-reduction", "generic"],
    version="1.0.0"
)
def fit_pca(
    inputs: Dict,
    outputs: Dict,
    n_components: int = 10,
    whiten: bool = True,
    exclude_columns: List[str] = None,
    random_state: int = 42
) -> str:
    """
    Fit PCA on training data for dimensionality reduction.

    Based on top Kaggle solution (siaa512) for tabular-playground-series-feb-2022.
    PCA with whitening is highly effective for reducing high-dimensional features.

    G1: Generic - works with any tabular dataset
    G3: Reproducible - fixed random_state
    G4: Parameterized - all key params exposed

    Args:
        inputs: Dict with 'data' key pointing to CSV path
        outputs: Dict with 'artifact' key for saving fitted PCA
        n_components: Number of principal components to keep
        whiten: If True, normalize each component to unit variance
        exclude_columns: Columns to exclude (e.g., id, target)
        random_state: Random seed for reproducibility

    Returns:
        Status message
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(inputs["data"])

    exclude_columns = exclude_columns or []
    feature_cols = [c for c in df.columns if c not in exclude_columns]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].values

    # Fit PCA
    pca = PCA(n_components=min(n_components, X.shape[1]), whiten=whiten, random_state=random_state)
    pca.fit(X)

    # Save artifact with metadata
    artifact = {
        "pca": pca,
        "feature_cols": numeric_cols,
        "n_components": pca.n_components_,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "wb") as f:
        pickle.dump(artifact, f)

    total_var = sum(pca.explained_variance_ratio_)
    return f"fit_pca: {len(numeric_cols)} features → {pca.n_components_} components, explained variance: {total_var:.4f}"


@contract(
    inputs={
        "data": {"format": "csv", "required": True},
        "artifact": {"format": "pickle", "required": True}
    },
    outputs={"data": {"format": "csv"}},
    description="Transform data using fitted PCA",
    tags=["feature-engineering", "pca", "dimensionality-reduction", "generic"],
    version="1.0.0"
)
def transform_pca(
    inputs: Dict,
    outputs: Dict,
    keep_original: bool = False,
    exclude_columns: List[str] = None
) -> str:
    """
    Apply fitted PCA transformation to data.

    G1: Generic - works with any dataset that matches fitted PCA
    G5: Loose coupling - input/output via files

    Args:
        inputs: Dict with 'data' and 'artifact' keys
        outputs: Dict with 'data' key for output CSV
        keep_original: If True, append PCA components to original data
        exclude_columns: Columns to pass through unchanged (e.g., id, target)

    Returns:
        Status message
    """
    df = pd.read_csv(inputs["data"])

    with open(inputs["artifact"], "rb") as f:
        artifact = pickle.load(f)

    pca = artifact["pca"]
    feature_cols = artifact["feature_cols"]

    # Get features that exist in this data
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].values

    # Transform
    X_pca = pca.transform(X)

    # Create output dataframe
    exclude_columns = exclude_columns or []
    pass_through_cols = [c for c in df.columns if c in exclude_columns or c not in feature_cols]

    if keep_original:
        result_df = df.copy()
        for i in range(X_pca.shape[1]):
            result_df[f"pca_{i}"] = X_pca[:, i]
    else:
        result_df = pd.DataFrame()
        # Keep pass-through columns
        for col in pass_through_cols:
            result_df[col] = df[col]
        # Add PCA columns
        for i in range(X_pca.shape[1]):
            result_df[f"pca_{i}"] = X_pca[:, i]

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    result_df.to_csv(outputs["data"], index=False)

    return f"transform_pca: {len(available_cols)} features → {X_pca.shape[1]} PCA components, {len(result_df)} rows"


# =============================================================================
# Service Registry
# =============================================================================

SERVICE_REGISTRY = {
    "create_polynomial_features": create_polynomial_features,
    "add_row_statistics": add_row_statistics,
    "fit_pca": fit_pca,
    "transform_pca": transform_pca,
}
