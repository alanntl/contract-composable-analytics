"""
Contract-Composable Analytics Services for cat-in-the-dat competition
==============================================
Competition: https://www.kaggle.com/competitions/cat-in-the-dat
Problem Type: Binary Classification
Target: target (0/1)
Metric: ROC AUC

All categorical features encoding challenge - binary, ordinal, nominal, cyclical.

Competition-specific services:
- encode_binary_mappings: Map T/F, Y/N string binaries to 0/1
- ordinal_encode_columns: Map ordered categorical values to integers
- create_cyclical_features: Sin/cos encoding for periodic features (day, month)

Top solution insights (0.80285 private LB):
- Combine train+test before encoding to handle unseen categories
- Ordinal encoding with StandardScaler for ord_0-5
- One-hot / label encoding for nom_0-9
- Cyclical sin/cos for day, month
- LogisticRegression / LightGBM performs well
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services from common modules
from services.preprocessing_services import (
    split_data,
    create_submission,
)
from services.classification_services import train_lightgbm_classifier, predict_classifier


# =============================================================================
# COMPETITION-SPECIFIC SERVICES (reusable across similar problems)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Encode binary string columns (T/F, Y/N) to numeric 0/1",
    tags=["preprocessing", "encoding", "binary", "categorical", "generic"],
    version="1.0.0",
)
def encode_binary_mappings(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[List[str]] = None,
    mappings: Optional[Dict] = None,
) -> str:
    """Encode binary columns with custom mappings (T/F, Y/N -> 0/1).

    G1 Compliance: Single task - binary string-to-numeric conversion.
    G4 Compliance: Column names and mappings injected via parameters.

    Parameters:
        columns: List of columns to encode. If None, auto-detects binary string columns.
        mappings: Additional value mappings to apply (merged with defaults).
    """
    df = _load_data(inputs["data"])

    default_mappings = {
        'T': 1, 'F': 0, 'Y': 1, 'N': 0,
        't': 1, 'f': 0, 'y': 1, 'n': 0,
        True: 1, False: 0,
    }

    if mappings:
        default_mappings.update(mappings)

    if columns is None:
        columns = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 2]

    encoded_count = 0
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(lambda x, m=default_mappings: m.get(x, x))
            df[col] = pd.to_numeric(df[col], errors='coerce')
            encoded_count += 1

    _save_data(df, outputs["data"])
    return f"encode_binary_mappings: encoded {encoded_count} columns"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Encode ordinal categorical columns with specified value order",
    tags=["preprocessing", "encoding", "ordinal", "categorical", "generic"],
    version="1.0.0",
)
def ordinal_encode_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[Dict] = None,
) -> str:
    """Encode ordinal categorical columns with specified order mapping.

    G1 Compliance: Single task - ordinal string-to-integer conversion.
    G4 Compliance: Column names and orderings injected via parameters.

    Parameters:
        columns: Dict mapping column name to ordered values list.
                 e.g. {'ord_1': ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']}
    """
    df = _load_data(inputs["data"])

    if columns is None:
        columns = {}

    encoded_count = 0
    for col, order in columns.items():
        if col in df.columns:
            mapping = {v: i for i, v in enumerate(order)}
            df[col] = df[col].map(mapping)
            df[col] = df[col].fillna(-1).astype(int)
            encoded_count += 1

    _save_data(df, outputs["data"])
    return f"ordinal_encode_columns: encoded {encoded_count} columns"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Create cyclical sin/cos features for periodic columns like day, month",
    tags=["preprocessing", "feature-engineering", "cyclical", "temporal", "generic"],
    version="1.0.0",
)
def create_cyclical_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[Dict] = None,
    drop_original: bool = False,
) -> str:
    """Create cyclical sin/cos features for periodic columns.

    G1 Compliance: Single task - cyclical feature engineering.
    G4 Compliance: Column names and max values injected via parameters.

    Parameters:
        columns: Dict mapping column name to max value for cyclical encoding.
                 e.g. {'day': 7, 'month': 12}
        drop_original: Whether to drop the original columns after encoding.
    """
    df = _load_data(inputs["data"])

    if columns is None:
        columns = {}

    created_count = 0
    for col, max_val in columns.items():
        if col in df.columns:
            df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
            created_count += 1

    if drop_original:
        drop_cols = [c for c in columns.keys() if c in df.columns]
        df = df.drop(columns=drop_cols)

    _save_data(df, outputs["data"])
    return f"create_cyclical_features: created {created_count * 2} sin/cos features"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
        "artifact": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "encoder"}},
    },
    description="Fit label encoder on training data and transform it (vectorized, fast)",
    tags=["preprocessing", "encoding", "label", "categorical", "generic"],
    version="1.0.0",
)
def fit_label_encode(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """Fit label encoder on specified columns and transform data.

    Uses vectorized pd.Categorical for fast encoding on large datasets.
    Saves the category mappings as an artifact for reuse on test data.

    G1 Compliance: Fit + transform on training data (produces reusable artifact).
    G4 Compliance: Column names injected via parameters.

    Parameters:
        columns: List of columns to encode. If None, auto-detects object columns.
        exclude_columns: Columns to skip (e.g., target, id).
    """
    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])

    if columns is None:
        columns = [c for c in df.columns if df[c].dtype == 'object' and c not in exclude]
    else:
        columns = [c for c in columns if c not in exclude]

    category_maps = {}
    for col in columns:
        if col in df.columns:
            cat = pd.Categorical(df[col])
            category_maps[col] = list(cat.categories)
            df[col] = cat.codes

    # Save encoder artifact
    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "wb") as f:
        pickle.dump({"columns": columns, "category_maps": category_maps}, f)

    _save_data(df, outputs["data"])
    return f"fit_label_encode: encoded {len(category_maps)} columns, saved encoder artifact"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "artifact": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "encoder"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Apply pre-fitted label encoder to new data (test set)",
    tags=["preprocessing", "encoding", "label", "categorical", "generic"],
    version="1.0.0",
)
def apply_label_encode(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """Apply pre-fitted label encoder to new data using saved category mappings.

    Categories not seen during fitting are encoded as -1.
    Uses vectorized pd.Categorical for fast encoding.

    G1 Compliance: Transform-only, uses pre-fitted artifact.
    """
    df = _load_data(inputs["data"])

    with open(inputs["artifact"], "rb") as f:
        artifact = pickle.load(f)

    category_maps = artifact["category_maps"]
    encoded_count = 0

    for col, categories in category_maps.items():
        if col in df.columns:
            cat = pd.Categorical(df[col], categories=categories)
            df[col] = cat.codes  # -1 for unseen categories
            encoded_count += 1

    _save_data(df, outputs["data"])
    return f"apply_label_encode: applied encoding to {encoded_count} columns"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific encoding services
    'encode_binary_mappings': encode_binary_mappings,
    'ordinal_encode_columns': ordinal_encode_columns,
    'create_cyclical_features': create_cyclical_features,
    'fit_label_encode': fit_label_encode,
    'apply_label_encode': apply_label_encode,
    # Reusable imported services
    'split_data': split_data,
    'train_lightgbm_classifier': train_lightgbm_classifier,
    'predict_classifier': predict_classifier,
    'create_submission': create_submission,
}
