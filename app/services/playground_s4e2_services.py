"""
Contract-Composable Analytics Services for playground-series-s4e2 competition
=====================================================
Competition: https://www.kaggle.com/competitions/playground-series-s4e2
Problem Type: Multiclass Classification
Target: NObeyesdad (7 obesity level categories)
Evaluation Metric: Accuracy

Competition-specific services:
- create_bmi_features: Calculate BMI and BMI category from Height/Weight
- create_lifestyle_features: Aggregate activity, eating, and hydration scores
- encode_obesity_categoricals: Label encode categorical columns via factorize
- encode_yes_no_columns: Deterministic binary encoding for yes/no columns
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services from common modules
from services.preprocessing_services import split_data, create_submission, drop_columns
from services.classification_services import train_lightgbm_classifier, predict_classifier


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
    description="Create BMI and body composition features from Height and Weight columns",
    tags=["feature-engineering", "obesity", "bmi", "health"],
    version="1.0.0",
)
def create_bmi_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """Create BMI and related body composition features.

    Calculates BMI = Weight / Height^2 and bins into WHO categories.
    Reusable for any dataset with Height and Weight columns.

    G1 Compliance: Single responsibility - BMI feature creation.
    G4 Compliance: Works with any dataset containing Height/Weight.
    """
    df = _load_data(inputs["data"])

    if "Height" in df.columns and "Weight" in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
        df["BMI_category"] = pd.cut(
            df["BMI"],
            bins=[0, 18.5, 25, 30, 35, 40, 100],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(float)
        df["BMI_category"] = df["BMI_category"].fillna(2)  # Default to normal

    _save_data(df, outputs["data"])
    return f"create_bmi_features: added BMI and BMI_category ({len(df)} rows)"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Create aggregated lifestyle features from activity, eating, and hydration columns",
    tags=["feature-engineering", "obesity", "lifestyle", "health"],
    version="1.0.0",
)
def create_lifestyle_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """Create aggregated lifestyle features.

    Combines multiple columns into activity_score, eating_score,
    and hydration_level. Reusable for health/obesity datasets.

    G1 Compliance: Single responsibility - lifestyle feature aggregation.
    G4 Compliance: Checks column existence before computation.
    """
    df = _load_data(inputs["data"])

    # Physical activity score
    if "FAF" in df.columns and "TUE" in df.columns:
        df["activity_score"] = df["FAF"] - df["TUE"] / 24

    # Eating habits score
    if "FCVC" in df.columns and "NCP" in df.columns:
        df["eating_score"] = df["FCVC"] + df["NCP"] / 3

    # Hydration score
    if "CH2O" in df.columns:
        df["hydration_level"] = pd.cut(
            df["CH2O"], bins=[0, 1, 2, 3], labels=[0, 1, 2]
        ).astype(float)
        df["hydration_level"] = df["hydration_level"].fillna(1)

    _save_data(df, outputs["data"])
    features = [f for f in ["activity_score", "eating_score", "hydration_level"] if f in df.columns]
    return f"create_lifestyle_features: added {len(features)} features ({len(df)} rows)"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Label encode specified categorical columns using pandas factorize",
    tags=["preprocessing", "encoding", "categorical", "obesity"],
    version="1.0.0",
)
def encode_obesity_categoricals(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[List[str]] = None,
) -> str:
    """Label encode categorical columns using pandas factorize.

    G1 Compliance: Single responsibility - label encoding.
    G4 Compliance: Column names as parameters, not hardcoded.

    Parameters:
        columns: List of columns to encode. Defaults to obesity dataset categoricals.
    """
    df = _load_data(inputs["data"])

    if columns is None:
        columns = ["Gender", "CAEC", "CALC", "MTRANS", "NObeyesdad"]

    encoded = []
    for col in columns:
        if col in df.columns:
            df[col] = pd.factorize(df[col])[0]
            encoded.append(col)

    _save_data(df, outputs["data"])
    return f"encode_obesity_categoricals: encoded {len(encoded)} columns: {encoded}"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Encode yes/no columns as binary (1/0) values",
    tags=["preprocessing", "encoding", "binary", "obesity"],
    version="1.0.0",
)
def encode_yes_no_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: Optional[List[str]] = None,
) -> str:
    """Encode yes/no columns as binary (1/0).

    Deterministic encoding: yes/Yes -> 1, no/No -> 0.
    Reusable for any dataset with binary yes/no columns.

    G1 Compliance: Single responsibility - binary encoding.
    G4 Compliance: Column names as parameters.

    Parameters:
        columns: List of columns to encode. Defaults to obesity dataset binary columns.
    """
    df = _load_data(inputs["data"])

    if columns is None:
        columns = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]

    encoded = []
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0, "Yes": 1, "No": 0}).fillna(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            encoded.append(col)

    _save_data(df, outputs["data"])
    return f"encode_yes_no_columns: encoded {len(encoded)} columns: {encoded}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific services
    "create_bmi_features": create_bmi_features,
    "create_lifestyle_features": create_lifestyle_features,
    "encode_obesity_categoricals": encode_obesity_categoricals,
    "encode_yes_no_columns": encode_yes_no_columns,
    # Imported reusable services
    "split_data": split_data,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "drop_columns": drop_columns,
    "create_submission": create_submission,
}