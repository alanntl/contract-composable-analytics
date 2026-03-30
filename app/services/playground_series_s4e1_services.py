"""
Playground Series S4E1 - Bank Churn Prediction - Contract-Composable Analytics Services
===============================================================
Competition: https://www.kaggle.com/competitions/playground-series-s4e1
Problem Type: Binary Classification
Target: Exited (0=stayed, 1=churned)
ID Column: id

Bank customer churn prediction based on customer attributes.
Standard tabular binary classification with categorical and numeric features.
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

try:
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.preprocessing_services import split_data, create_submission
    from services.bike_sharing_services import drop_columns
    from services.spaceship_titanic_services import fill_missing_numeric, label_encode_columns
except ImportError:
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import split_data, create_submission
    from bike_sharing_services import drop_columns
    from spaceship_titanic_services import fill_missing_numeric, label_encode_columns


# =============================================================================
# GENERIC REUSABLE SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create customer behavior features from account data",
    tags=["preprocessing", "feature-engineering", "generic"],
    version="1.0.0"
)
def create_account_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    balance_column: str = "Balance",
    salary_column: str = "EstimatedSalary",
    tenure_column: str = "Tenure",
    num_products_column: str = "NumOfProducts",
) -> str:
    """
    Create customer behavior features from account information.

    Args:
        balance_column: Column with account balance
        salary_column: Column with estimated salary
        tenure_column: Column with years as customer
        num_products_column: Column with number of products
    """
    df = pd.read_csv(inputs["data"])

    # Balance to salary ratio
    if balance_column in df.columns and salary_column in df.columns:
        df['balance_salary_ratio'] = df[balance_column] / (df[salary_column] + 1)

    # Zero balance indicator
    if balance_column in df.columns:
        df['has_balance'] = (df[balance_column] > 0).astype(int)

    # Products per tenure
    if num_products_column in df.columns and tenure_column in df.columns:
        df['products_per_tenure'] = df[num_products_column] / (df[tenure_column] + 1)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"create_account_features: created account behavior features"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create age-based segment features",
    tags=["preprocessing", "feature-engineering", "generic"],
    version="1.0.0"
)
def create_age_segments(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    age_column: str = "Age",
    bins: List[int] = None,
    labels: List[str] = None,
) -> str:
    """
    Segment customers by age groups.

    Args:
        age_column: Column containing age
        bins: Age bin boundaries (default: [0, 25, 35, 45, 55, 65, 100])
        labels: Labels for age groups
    """
    df = pd.read_csv(inputs["data"])
    bins = bins or [0, 25, 35, 45, 55, 65, 100]
    labels = labels or ['young', 'young_adult', 'adult', 'middle_age', 'senior', 'elderly']

    if age_column in df.columns:
        df['age_segment'] = pd.cut(df[age_column], bins=bins, labels=labels)
        df['age_segment'] = df['age_segment'].astype(str)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"create_age_segments: created {len(labels)} age segments"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "create_account_features": create_account_features,
    "create_age_segments": create_age_segments,
    "fill_missing_numeric": fill_missing_numeric,
    "label_encode_columns": label_encode_columns,
    "drop_columns": drop_columns,
    "split_data": split_data,
    "create_submission": create_submission,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
}


PIPELINE_SPEC = [
    # === TRAIN DATA PREPROCESSING ===
    {
        "service": "fill_missing_numeric",
        "inputs": {"data": "playground-series-s4e1/datasets/train.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/train_01_fill.csv"},
        "params": {"strategy": "median"},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "create_account_features",
        "inputs": {"data": "playground-series-s4e1/artifacts/train_01_fill.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/train_02_account.csv"},
        "params": {},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "create_age_segments",
        "inputs": {"data": "playground-series-s4e1/artifacts/train_02_account.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/train_03_age.csv"},
        "params": {},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "label_encode_columns",
        "inputs": {"data": "playground-series-s4e1/artifacts/train_03_age.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/train_04_encoded.csv"},
        "params": {"columns": ["Geography", "Gender", "age_segment"]},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "drop_columns",
        "inputs": {"data": "playground-series-s4e1/artifacts/train_04_encoded.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/train_final.csv"},
        "params": {"columns": ["CustomerId", "Surname"]},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "split_data",
        "inputs": {"data": "playground-series-s4e1/artifacts/train_final.csv"},
        "outputs": {
            "train_data": "playground-series-s4e1/artifacts/train_split.csv",
            "valid_data": "playground-series-s4e1/artifacts/valid_split.csv"
        },
        "params": {"stratify_column": "Exited", "test_size": 0.2, "random_state": 42},
        "module": "playground_series_s4e1_services"
    },
    # === MODEL TRAINING ===
    {
        "service": "train_lightgbm_classifier",
        "inputs": {
            "train_data": "playground-series-s4e1/artifacts/train_split.csv",
            "valid_data": "playground-series-s4e1/artifacts/valid_split.csv"
        },
        "outputs": {
            "model": "playground-series-s4e1/artifacts/model.pkl",
            "metrics": "playground-series-s4e1/artifacts/metrics.json"
        },
        "params": {
            "label_column": "Exited",
            "id_column": "id",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 100
        },
        "module": "classification_services"
    },
    # === TEST DATA PREPROCESSING ===
    {
        "service": "fill_missing_numeric",
        "inputs": {"data": "playground-series-s4e1/datasets/test.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/test_01_fill.csv"},
        "params": {"strategy": "median"},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "create_account_features",
        "inputs": {"data": "playground-series-s4e1/artifacts/test_01_fill.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/test_02_account.csv"},
        "params": {},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "create_age_segments",
        "inputs": {"data": "playground-series-s4e1/artifacts/test_02_account.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/test_03_age.csv"},
        "params": {},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "label_encode_columns",
        "inputs": {"data": "playground-series-s4e1/artifacts/test_03_age.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/test_04_encoded.csv"},
        "params": {"columns": ["Geography", "Gender", "age_segment"]},
        "module": "playground_series_s4e1_services"
    },
    {
        "service": "drop_columns",
        "inputs": {"data": "playground-series-s4e1/artifacts/test_04_encoded.csv"},
        "outputs": {"data": "playground-series-s4e1/artifacts/test_final.csv"},
        "params": {"columns": ["CustomerId", "Surname"]},
        "module": "playground_series_s4e1_services"
    },
    # === PREDICTION & SUBMISSION ===
    {
        "service": "predict_classifier",
        "inputs": {
            "model": "playground-series-s4e1/artifacts/model.pkl",
            "data": "playground-series-s4e1/artifacts/test_final.csv"
        },
        "outputs": {
            "predictions": "playground-series-s4e1/submission.csv"
        },
        "params": {
            "id_column": "id",
            "prediction_column": "Exited",
            "proba_as_prediction": True
        },
        "module": "classification_services"
    }
]


def run_pipeline(base_path: str, verbose: bool = True):
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
            if verbose: print(f"OK - {result}")
        except Exception as e:
            if verbose: print(f"FAILED - {e}")
            break
