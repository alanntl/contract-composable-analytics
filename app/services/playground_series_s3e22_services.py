"""
Playground Series S3E22 - Horse Health Prediction Services
==========================================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e22
Problem Type: Multiclass Classification (3 classes: died, euthanized, lived)
Target: outcome
ID Column: id
Evaluation Metric: Micro-averaged F1 Score

Dataset: Horse colic dataset - predict outcome from clinical examination features.
Features include rectal_temp, pulse, respiratory_rate, and various categorical
clinical indicators.

Solution Notebook Insights:
- Notebook 1 (aradhakkandhari): XGBoost + LabelEncoder + SMOTETomek, top 10 features by corr
- Notebook 2 (pasanpeiris): Custom ordinal encoding + outlier removal + XGBoost GridSearch
- Notebook 3 (catadanna): LGB best ~0.6955 (lr=0.01, n_est=500)

Key Insights Applied:
- Keep ALL features (original pipeline dropped 17 of 28 columns including the target!)
- fit_encoder/transform_encoder for consistent train/test label encoding
- fit_imputer/transform_imputer for consistent missing value handling
- LightGBM with tuned hyperparameters from solution notebooks
- Decode integer predictions back to string labels for submission
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
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.preprocessing_services import (
        fit_imputer, transform_imputer,
        fit_encoder, transform_encoder,
        split_data, create_submission,
    )
except ImportError:
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import (
        fit_imputer, transform_imputer,
        fit_encoder, transform_encoder,
        split_data, create_submission,
    )


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True},
        "encoder": {"format": "pkl", "required": True},
    },
    outputs={"submission": {"format": "csv"}},
    description="Decode integer class predictions back to original string labels using saved encoder",
    tags=["postprocessing", "classification", "multiclass", "submission"],
    version="1.0.0"
)
def decode_multiclass_predictions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    encoder_column: str = "outcome",
    id_column: str = "id",
    prediction_column: str = "outcome",
) -> str:
    """
    Decode integer predictions back to original string labels.

    Uses the saved LabelEncoder artifact from fit_encoder to inverse_transform
    integer class indices (0, 1, 2) back to string labels ("died", "euthanized", "lived").

    G1 Compliance: Generic, works with any multiclass prediction that was label-encoded.
    G4 Compliance: Column names parameterized.
    """
    pred_df = pd.read_csv(inputs["predictions"])

    with open(inputs["encoder"], "rb") as f:
        encoder_artifact = pickle.load(f)

    le = encoder_artifact["encoder"][encoder_column]
    pred_df[prediction_column] = le.inverse_transform(
        pred_df[prediction_column].astype(int)
    )

    # Output only id and prediction columns
    submission = pred_df[[id_column, prediction_column]]
    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    return f"decode_multiclass_predictions: decoded {len(submission)} predictions to string labels"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific
    "decode_multiclass_predictions": decode_multiclass_predictions,
    # Imported from common modules
    "fit_imputer": fit_imputer,
    "transform_imputer": transform_imputer,
    "fit_encoder": fit_encoder,
    "transform_encoder": transform_encoder,
    "split_data": split_data,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}


# =============================================================================
# PIPELINE SPECIFICATION (Training)
# =============================================================================

PIPELINE_SPEC = [
    # Step 1: Fit imputer on training data (learn fill values)
    {
        "service": "fit_imputer",
        "inputs": {"data": "playground-series-s3e22/datasets/train.csv"},
        "outputs": {"artifact": "playground-series-s3e22/artifacts/imputer.pkl"},
        "params": {
            "numeric_strategy": "median",
            "categorical_strategy": "most_frequent",
            "exclude_columns": ["id", "outcome"],
        },
        "module": "preprocessing_services",
    },
    # Step 2: Apply imputer to training data
    {
        "service": "transform_imputer",
        "inputs": {
            "data": "playground-series-s3e22/datasets/train.csv",
            "artifact": "playground-series-s3e22/artifacts/imputer.pkl",
        },
        "outputs": {"data": "playground-series-s3e22/artifacts/train_01_imputed.csv"},
        "params": {},
        "module": "preprocessing_services",
    },
    # Step 3: Fit label encoder on imputed training data
    {
        "service": "fit_encoder",
        "inputs": {"data": "playground-series-s3e22/artifacts/train_01_imputed.csv"},
        "outputs": {"artifact": "playground-series-s3e22/artifacts/encoder.pkl"},
        "params": {"method": "label", "exclude_columns": ["id"]},
        "module": "preprocessing_services",
    },
    # Step 4: Apply encoder to training data
    {
        "service": "transform_encoder",
        "inputs": {
            "data": "playground-series-s3e22/artifacts/train_01_imputed.csv",
            "artifact": "playground-series-s3e22/artifacts/encoder.pkl",
        },
        "outputs": {"data": "playground-series-s3e22/artifacts/train_02_encoded.csv"},
        "params": {},
        "module": "preprocessing_services",
    },
    # Step 5: Stratified train/validation split
    {
        "service": "split_data",
        "inputs": {"data": "playground-series-s3e22/artifacts/train_02_encoded.csv"},
        "outputs": {
            "train_data": "playground-series-s3e22/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e22/artifacts/valid_split.csv",
        },
        "params": {
            "stratify_column": "outcome",
            "test_size": 0.2,
            "random_state": 42,
        },
        "module": "preprocessing_services",
    },
    # Step 6: Train LightGBM classifier (hyperparams tuned from solution notebooks)
    {
        "service": "train_lightgbm_classifier",
        "inputs": {
            "train_data": "playground-series-s3e22/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e22/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": "playground-series-s3e22/artifacts/model.pkl",
            "metrics": "playground-series-s3e22/artifacts/metrics.json",
        },
        "params": {
            "label_column": "outcome",
            "id_column": "id",
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "num_leaves": 64,
            "max_depth": 8,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 100,
        },
        "module": "classification_services",
    },
]


# =============================================================================
# INFERENCE PIPELINE (for test set prediction + submission)
# =============================================================================

INFERENCE_SPEC = [
    # Step 1: Apply saved imputer to test data
    {
        "service": "transform_imputer",
        "inputs": {
            "data": "playground-series-s3e22/datasets/test.csv",
            "artifact": "playground-series-s3e22/artifacts/imputer.pkl",
        },
        "outputs": {"data": "playground-series-s3e22/artifacts/test_01_imputed.csv"},
        "params": {},
        "module": "preprocessing_services",
    },
    # Step 2: Apply saved encoder to test data
    {
        "service": "transform_encoder",
        "inputs": {
            "data": "playground-series-s3e22/artifacts/test_01_imputed.csv",
            "artifact": "playground-series-s3e22/artifacts/encoder.pkl",
        },
        "outputs": {"data": "playground-series-s3e22/artifacts/test_02_encoded.csv"},
        "params": {},
        "module": "preprocessing_services",
    },
    # Step 3: Predict with trained model
    {
        "service": "predict_classifier",
        "inputs": {
            "data": "playground-series-s3e22/artifacts/test_02_encoded.csv",
            "model": "playground-series-s3e22/artifacts/model.pkl",
        },
        "outputs": {
            "predictions": "playground-series-s3e22/artifacts/predictions.csv",
        },
        "params": {"id_column": "id", "prediction_column": "outcome"},
        "module": "classification_services",
    },
    # Step 4: Decode integer predictions back to string labels
    {
        "service": "decode_multiclass_predictions",
        "inputs": {
            "predictions": "playground-series-s3e22/artifacts/predictions.csv",
            "encoder": "playground-series-s3e22/artifacts/encoder.pkl",
        },
        "outputs": {
            "submission": "playground-series-s3e22/submission.csv",
        },
        "params": {
            "encoder_column": "outcome",
            "id_column": "id",
            "prediction_column": "outcome",
        },
        "module": "playground_series_s3e22_services",
    },
]


def run_pipeline(base_path: str, verbose: bool = True):
    """Run the training pipeline."""
    for i, step in enumerate(PIPELINE_SPEC, 1):
        service_name = step["service"]
        service_fn = SERVICE_REGISTRY.get(service_name)
        if not service_fn:
            print(f"Error: Service {service_name} not found")
            continue

        res_in = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
        res_out = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

        if verbose:
            print(f"[{i}/{len(PIPELINE_SPEC)}] {service_name}...", end=" ")

        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose:
                print(f"OK - {result}")
        except Exception as e:
            if verbose:
                print(f"FAILED - {e}")
            raise


def run_inference(base_path: str, verbose: bool = True):
    """Run the inference pipeline on test set."""
    for i, step in enumerate(INFERENCE_SPEC, 1):
        service_name = step["service"]
        service_fn = SERVICE_REGISTRY.get(service_name)
        if not service_fn:
            print(f"Error: Service {service_name} not found")
            continue

        res_in = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
        res_out = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

        if verbose:
            print(f"[{i}/{len(INFERENCE_SPEC)}] {service_name}...", end=" ")

        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose:
                print(f"OK - {result}")
        except Exception as e:
            if verbose:
                print(f"FAILED - {e}")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="storage", help="Base path for data")
    parser.add_argument("--inference", action="store_true", help="Run inference pipeline")
    args = parser.parse_args()

    if args.inference:
        run_inference(args.base_path)
    else:
        run_pipeline(args.base_path)
