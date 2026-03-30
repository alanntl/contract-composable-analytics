"""
Playground Series S3E3 - Contract-Composable Analytics Services
========================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e3
Problem Type: Binary Classification (Employee Attrition: 0/1)
Target: Attrition (binary, AUC-scored → submit probabilities)
ID Column: id

Predict employee attrition from HR features. Dataset is synthetically
generated from the IBM HR Analytics Employee Attrition & Performance dataset.
Features include Age, BusinessTravel, Department, EducationField, Gender,
JobRole, MaritalStatus, OverTime, and various numeric HR metrics.

Solution Notebook Insights:
- Notebook 01 (tracyporter): OrdinalEncoder on all object columns, remove
  highly correlated features (>0.5), MinMax normalize, RandomForest
  (class_weight='balanced', max_depth=2). Submits predict_log_proba.
- Notebook 02 (ksqrt9): One-hot encoding via get_dummies, MinMaxScaler,
  Keras Neural Network (256→128→64→32→16→8→4→2→1, sigmoid).
  Early stopping + ReduceLROnPlateau.
- Notebook 03 (simranksaluja): Drop constant cols (EmployeeCount, Over18,
  StandardHours), get_dummies on string categoricals, StandardScaler,
  LogisticRegression + RandomForest. Submits predict_proba.

Key Techniques Across All Solutions:
1. Encode categoricals (ordinal or one-hot) - ALL solutions
2. Drop constant columns (EmployeeCount, Over18, StandardHours) - Solution 3
3. Feature scaling (MinMax/Standard) - ALL solutions
4. Multiple model types: RF, NN, LogReg (ensemble approach beneficial)
5. Submit probabilities (AUC-scored competition)

Competition-specific services:
- remove_correlated_features: Remove features with pairwise correlation above
  threshold (insight from Solution 1)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# =============================================================================
# IMPORTS FROM COMMON MODULES (G1: Reuse existing services)
# =============================================================================
try:
    from services.io_utils import load_data as _load_data, save_data as _save_data
    from services.classification_services import (
        train_ensemble_classifier, predict_classifier
    )
    from services.preprocessing_services import (
        split_data, drop_columns, fit_encoder, transform_encoder,
        create_submission
    )
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from classification_services import (
        train_ensemble_classifier, predict_classifier
    )
    from preprocessing_services import (
        split_data, drop_columns, fit_encoder, transform_encoder,
        create_submission
    )


# =============================================================================
# COMPETITION-SPECIFIC SERVICE: Remove Correlated Features
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
        "artifact": {"format": "json", "schema": {"type": "json"}},
    },
    description="Remove features with pairwise correlation above threshold (keeps first of each correlated pair)",
    tags=["preprocessing", "feature-selection", "correlation", "generic"],
    version="1.0.0",
)
def remove_correlated_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    correlation_threshold: float = 0.85,
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """
    Remove highly correlated features based on pairwise Pearson correlation.

    Inspired by Solution 1 (tracyporter) which removes features with
    correlation >= 0.5. Uses a configurable threshold to drop redundant
    features while preserving the first of each correlated pair.

    G1 Compliance: Generic, works with any numeric dataset.
    G4 Compliance: Parameterized threshold and exclude_columns.

    Parameters:
        correlation_threshold: Drop features with |correlation| >= threshold
        exclude_columns: Columns to never drop (e.g., target, id)
    """
    df = _load_data(inputs["data"])
    exclude = set(exclude_columns or [])

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns
                if any(upper[col] >= correlation_threshold)]

    dropped_info = {}
    for col in to_drop:
        correlated_with = [
            other for other in upper.index
            if upper.loc[other, col] >= correlation_threshold
        ]
        dropped_info[col] = correlated_with

    df_out = df.drop(columns=to_drop, errors="ignore")
    _save_data(df_out, outputs["data"])

    artifact = {
        "dropped_columns": to_drop,
        "correlation_threshold": correlation_threshold,
        "dropped_correlations": {k: v for k, v in dropped_info.items()},
        "kept_columns": list(df_out.columns),
    }
    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "w") as f:
        json.dump(artifact, f, indent=2)

    return (f"remove_correlated_features: dropped {len(to_drop)} columns "
            f"(threshold={correlation_threshold}), {df.shape[1]} -> {df_out.shape[1]}")


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "split_data": split_data,
    "drop_columns": drop_columns,
    "fit_encoder": fit_encoder,
    "transform_encoder": transform_encoder,
    "train_ensemble_classifier": train_ensemble_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
    # Competition-specific service
    "remove_correlated_features": remove_correlated_features,
}