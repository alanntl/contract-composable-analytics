"""
Playground Series S3E23 - Software Defect Prediction Services
==============================================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e23
Problem Type: Binary Classification
Target: defects (True/False → probability)
ID Column: id
Evaluation Metric: ROC AUC
Dataset: JM1 software defect dataset (synthetic) - predict defects in C programs
         from McCabe and Halstead software complexity metrics.

Solution Notebook Insights:
- Notebook 1 (khalidhabiburahman): SMOTE + StandardScaler + RandomForest pipeline
- Notebook 2 (daaadaaa): Original JM1 data merge + StandardScaler + outlier handling
  + StratifiedKFold + LightGBM(dart)/RF/LogReg ensemble
- Notebook 3 (vinitkp): Same as #2 with LightGBM(gbdt)

Key Insights Applied:
- Median imputation (more robust than zero for skewed software metrics)
- Log1p transforms on highly skewed features (e, v, n, t) to normalize distributions
- Ratio features between related Halstead/McCabe metrics
- Row-wise statistics across numeric features
- Weighted ensemble: LightGBM (0.5) + XGBoost (0.25) + RF (0.25)
- Note: StandardScaler removed as tree-based models are scale-invariant
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
    from services.classification_services import (
        train_ensemble_classifier, predict_classifier,
    )
    from services.preprocessing_services import (
        fill_missing, split_data,
    )
except ImportError:
    from classification_services import (
        train_ensemble_classifier, predict_classifier,
    )
    from preprocessing_services import (
        fill_missing, split_data,
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
    description="Engineer features from software complexity metrics (log transforms, ratios, row stats)",
    tags=["feature-engineering", "software-metrics", "defect-prediction", "generic"],
    version="1.0.0",
)
def engineer_software_defect_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    target_column: str = "defects",
    add_log_features: bool = True,
    add_ratio_features: bool = True,
    add_row_stats: bool = True,
) -> str:
    """
    Engineer features from software complexity metrics.

    Creates derived features inspired by solution notebooks and domain knowledge
    of McCabe and Halstead software complexity measures:
    - Log1p transforms on highly skewed metrics (e, v, n, t, loc)
    - Ratio features between related metrics (e.g., comment ratio, operator density)
    - Row-wise statistics across all numeric features

    G1 Compliance: Parameterized, works with any software metrics dataset.
    G2 Compliance: Single responsibility - feature engineering only.
    G4 Compliance: All column names and toggles parameterized.

    Parameters:
        id_column: Name of the ID column to exclude from engineering
        target_column: Name of the target column to exclude from engineering
        add_log_features: Add log1p transforms of skewed features
        add_ratio_features: Add ratio/interaction features
        add_row_stats: Add row-wise statistics
    """
    df = pd.read_csv(inputs["data"])
    exclude = {id_column, target_column}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]
    n_original = len(df.columns)

    # --- Log1p transforms for highly skewed features ---
    if add_log_features:
        skewed_cols = ["e", "v", "n", "t", "loc", "b", "i", "d",
                       "total_Op", "total_Opnd"]
        for col in skewed_cols:
            if col in df.columns:
                df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))

    # --- Ratio features from domain knowledge ---
    if add_ratio_features:
        # Comment ratio: proportion of comment lines
        if "lOComment" in df.columns and "loc" in df.columns:
            df["comment_ratio"] = df["lOComment"] / (df["loc"] + 1e-8)

        # Code density: code lines relative to total
        if "lOCode" in df.columns and "loc" in df.columns:
            df["code_density"] = df["lOCode"] / (df["loc"] + 1e-8)

        # Blank ratio
        if "lOBlank" in df.columns and "loc" in df.columns:
            df["blank_ratio"] = df["lOBlank"] / (df["loc"] + 1e-8)

        # Operator complexity: unique ops / total ops
        if "uniq_Op" in df.columns and "total_Op" in df.columns:
            df["op_uniqueness"] = df["uniq_Op"] / (df["total_Op"] + 1e-8)

        # Operand complexity: unique operands / total operands
        if "uniq_Opnd" in df.columns and "total_Opnd" in df.columns:
            df["opnd_uniqueness"] = df["uniq_Opnd"] / (df["total_Opnd"] + 1e-8)

        # Halstead volume per line
        if "v" in df.columns and "loc" in df.columns:
            df["volume_per_line"] = df["v"] / (df["loc"] + 1e-8)

        # Effort per line
        if "e" in df.columns and "loc" in df.columns:
            df["effort_per_line"] = df["e"] / (df["loc"] + 1e-8)

        # Cyclomatic complexity density
        if "v(g)" in df.columns and "loc" in df.columns:
            df["cyclomatic_density"] = df["v(g)"] / (df["loc"] + 1e-8)

        # Essential to cyclomatic ratio
        if "ev(g)" in df.columns and "v(g)" in df.columns:
            df["essential_ratio"] = df["ev(g)"] / (df["v(g)"] + 1e-8)

        # Design to cyclomatic ratio
        if "iv(g)" in df.columns and "v(g)" in df.columns:
            df["design_ratio"] = df["iv(g)"] / (df["v(g)"] + 1e-8)

    # --- Row-wise statistics ---
    if add_row_stats:
        feature_cols = [c for c in numeric_cols if c in df.columns]
        if feature_cols:
            df["row_sum"] = df[feature_cols].sum(axis=1)
            df["row_mean"] = df[feature_cols].mean(axis=1)
            df["row_std"] = df[feature_cols].std(axis=1)
            df["row_max"] = df[feature_cols].max(axis=1)
            df["row_min"] = df[feature_cols].min(axis=1)

    n_new = len(df.columns) - n_original
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    return f"engineer_software_defect_features: added {n_new} features ({len(df.columns)} total columns)"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "fill_missing": fill_missing,
    "split_data": split_data,
    "train_ensemble_classifier": train_ensemble_classifier,
    "predict_classifier": predict_classifier,
}


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = [
    # Step 1: Fill missing values in training data (median, more robust than zero)
    {
        "service": "fill_missing",
        "inputs": {"data": "playground-series-s3e23/datasets/train.csv"},
        "outputs": {"data": "playground-series-s3e23/artifacts/train_01_filled.csv"},
        "params": {"strategy": "median"},
        "module": "preprocessing_services",
    },
    # Step 2: Fill missing values in test data
    {
        "service": "fill_missing",
        "inputs": {"data": "playground-series-s3e23/datasets/test.csv"},
        "outputs": {"data": "playground-series-s3e23/artifacts/test_01_filled.csv"},
        "params": {"strategy": "median"},
        "module": "preprocessing_services",
    },
    # Step 3: Engineer features for training data
    {
        "service": "engineer_software_defect_features",
        "inputs": {"data": "playground-series-s3e23/artifacts/train_01_filled.csv"},
        "outputs": {"data": "playground-series-s3e23/artifacts/train_02_engineered.csv"},
        "params": {
            "id_column": "id",
            "target_column": "defects",
            "add_log_features": True,
            "add_ratio_features": True,
            "add_row_stats": True,
        },
        "module": "playground_series_s3e23_services",
    },
    # Step 4: Engineer features for test data
    {
        "service": "engineer_software_defect_features",
        "inputs": {"data": "playground-series-s3e23/artifacts/test_01_filled.csv"},
        "outputs": {"data": "playground-series-s3e23/artifacts/test_02_engineered.csv"},
        "params": {
            "id_column": "id",
            "target_column": "defects",
            "add_log_features": True,
            "add_ratio_features": True,
            "add_row_stats": True,
        },
        "module": "playground_series_s3e23_services",
    },
    # Step 5: Stratified train/validation split
    {
        "service": "split_data",
        "inputs": {"data": "playground-series-s3e23/artifacts/train_02_engineered.csv"},
        "outputs": {
            "train_data": "playground-series-s3e23/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e23/artifacts/valid_split.csv",
        },
        "params": {
            "stratify_column": "defects",
            "test_size": 0.2,
            "random_state": 42,
        },
        "module": "preprocessing_services",
    },
    # Step 6: Train ensemble classifier
    {
        "service": "train_ensemble_classifier",
        "inputs": {
            "train_data": "playground-series-s3e23/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e23/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": "playground-series-s3e23/artifacts/model.pkl",
            "metrics": "playground-series-s3e23/artifacts/metrics.json",
        },
        "params": {
            "label_column": "defects",
            "id_column": "id",
            "model_types": ["lightgbm", "xgboost", "random_forest"],
            "weights": [0.5, 0.25, 0.25],
        },
        "module": "classification_services",
    },
    # Step 7: Predict on test data
    {
        "service": "predict_classifier",
        "inputs": {
            "model": "playground-series-s3e23/artifacts/model.pkl",
            "data": "playground-series-s3e23/artifacts/test_02_engineered.csv",
        },
        "outputs": {
            "predictions": "playground-series-s3e23/submission.csv",
        },
        "params": {
            "id_column": "id",
            "prediction_column": "defects",
            "proba_as_prediction": True,
        },
        "module": "classification_services",
    },
]
