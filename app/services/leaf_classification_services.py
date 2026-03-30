"""
Leaf Classification - SLEGO Services
=====================================
Competition: https://www.kaggle.com/competitions/leaf-classification
Problem Type: Multiclass Classification (99 species)
Target: species (text labels for 99 leaf species)
Metric: Multi-class log loss (categorical cross-entropy)
Submission: id + 99 probability columns (one per species, sorted alphabetically)

Competition-specific services:
- predict_multiclass_proba_submission: Generic service to create multiclass probability
  submission CSVs (one column per class with probabilities), reusable for any competition
  requiring probability-based multiclass submissions.

Solution notebook insights (top 3 scored):
1. All use StandardScaler for feature normalization
2. All use Neural Networks with softmax output (99 classes)
3. All output probability matrix (id + 99 species probability columns)
4. Key: features are margin1-64, shape1-64, texture1-64 (192 total)
5. Solution 3 uses model averaging of 4 NNs for better generalization

Pipeline v3: Uses FLAML AutoML (train_automl_classifier) to automatically find
the best model and hyperparameters, optimizing directly for log_loss.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

# Import reusable services from common modules
from services.preprocessing_services import (
    label_encode_categorical,
    fit_scaler,
    transform_scaler,
    split_data,
    drop_columns,
)
from services.classification_services import (
    train_lightgbm_classifier,
    train_automl_classifier,
    train_keras_nn_classifier,
    predict_classifier,
    predict_multiclass_submission,
)
from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# COMPETITION-SPECIFIC SERVICE: Multiclass Probability Submission
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "encoder": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Predict multiclass probabilities and create submission with one column per class",
    tags=["inference", "prediction", "multiclass", "probability", "submission", "generic"],
    version="1.0.0",
)
def predict_multiclass_proba_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    target_column: str = "species",
) -> str:
    """Create a multiclass probability submission CSV with one column per class.

    Loads a trained classifier and label encoder, predicts probabilities for
    all classes on the test data, then creates a Kaggle submission with:
    - id column (from test data)
    - One column per class (sorted alphabetically) containing probabilities

    This format is required by Kaggle competitions scored with multi-class
    log loss (e.g., leaf-classification, otto-group, sf-crime).

    G1 Compliance: Single responsibility - predict + format submission.
    G4 Compliance: Column names parameterized via id_column, target_column.

    Parameters:
        id_column: Name of the ID column in test data
        target_column: Name of the target column (used to find encoder mapping)
    """
    # Load model artifact
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_cols = artifact.get("feature_cols", None)

    # Load test data
    data_df = _load_data(inputs["data"])

    # Load label encoder mapping: {col_name: {label_str: int_code, ...}}
    with open(inputs["encoder"], "rb") as f:
        encoder_mapping = pickle.load(f)

    # Get the target column mapping
    if target_column in encoder_mapping:
        label_to_int = encoder_mapping[target_column]
    elif len(encoder_mapping) == 1:
        # If only one column was encoded, use that
        label_to_int = list(encoder_mapping.values())[0]
    else:
        raise ValueError(
            f"Target column '{target_column}' not found in encoder mapping. "
            f"Available: {list(encoder_mapping.keys())}"
        )

    # Invert mapping: int_code -> label_str
    int_to_label = {v: k for k, v in label_to_int.items()}

    # Extract IDs
    ids = data_df[id_column].values if id_column in data_df.columns else np.arange(len(data_df))

    # Prepare feature matrix
    if feature_cols:
        for col in feature_cols:
            if col not in data_df.columns:
                data_df[col] = 0
        X = data_df[feature_cols]
    else:
        drop_cols = [id_column] if id_column in data_df.columns else []
        X = data_df.drop(columns=drop_cols, errors="ignore")

    # Predict probabilities
    proba = model.predict_proba(X)  # shape: (n_samples, n_classes)

    # Map model classes to original label names
    model_classes = model.classes_  # integer codes
    class_names = [int_to_label[int(c)] for c in model_classes]

    # Create DataFrame with probabilities
    proba_df = pd.DataFrame(proba, columns=class_names)

    # Sort columns alphabetically (Kaggle requirement)
    sorted_columns = sorted(proba_df.columns)
    proba_df = proba_df[sorted_columns]

    # Add id column
    proba_df.insert(0, id_column, ids)

    # Save submission
    _save_data(proba_df, outputs["submission"])

    return (
        f"predict_multiclass_proba_submission: {len(proba_df)} predictions, "
        f"{len(sorted_columns)} classes, mean_max_proba={proba.max(axis=1).mean():.4f}"
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific
    "predict_multiclass_proba_submission": predict_multiclass_proba_submission,
    # Imported generic services
    "label_encode_categorical": label_encode_categorical,
    "fit_scaler": fit_scaler,
    "transform_scaler": transform_scaler,
    "split_data": split_data,
    "drop_columns": drop_columns,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "train_automl_classifier": train_automl_classifier,
    "train_keras_nn_classifier": train_keras_nn_classifier,
    "predict_classifier": predict_classifier,
    "predict_multiclass_submission": predict_multiclass_submission,
}