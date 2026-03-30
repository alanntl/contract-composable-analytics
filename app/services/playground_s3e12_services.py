"""
SLEGO Services for playground-series-s3e12 competition
Binary Classification - Target: target
Kidney Stone Prediction based on Urine Analysis

Domain-specific feature engineering inspired by top Kaggle solutions:
  - richeyjay (174 votes): ion_product, electrolyte_balance, osmo_density
  - tumpanjawat (141 votes): urine_volume, calcium_pH_interaction
  - tetsutani (101 votes): calcium_to_urea_ratio, osmolality_to_sg_ratio

Only domain-specific logic lives here; everything else
is imported from the generic SLEGO Layers 1-5.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract
from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable generic services (Layers 2-5)
from services.preprocessing_services import (
    split_data,
    engineer_features,
    create_submission,
    fill_missing,
)
from services.classification_services import (
    train_lightgbm_classifier,
    train_xgboost_classifier,
    train_ensemble_classifier,
    predict_classifier,
)


# =============================================================================
# DOMAIN-SPECIFIC SERVICE: Kidney Stone Feature Engineering
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Create domain-specific features for kidney stone prediction from urine analysis",
    tags=["feature-engineering", "medical", "kidney-stone", "domain-specific"],
    version="2.0.0",
)
def create_kidney_stone_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """Create domain-specific features for kidney stone prediction.

    Engineered from medical domain knowledge and top-scoring Kaggle solutions.
    Features capture relationships between urine concentration markers
    (gravity, osmolality), electrolyte properties (conductivity, pH),
    and stone-forming substances (calcium, urea).

    G1 Compliance: Single responsibility - kidney stone feature engineering.
    G4 Compliance: No hardcoded column names beyond known medical features.

    Parameters:
        exclude_columns: Columns to skip (e.g., id, target)
    """
    df = _load_data(inputs["data"])
    features_added = []

    # --- Product features (substance interactions) ---
    if "calc" in df.columns and "urea" in df.columns:
        df["ion_product"] = df["calc"] * df["urea"]
        features_added.append("ion_product")

    if "calc" in df.columns and "ph" in df.columns:
        df["calcium_pH_interaction"] = df["calc"] * df["ph"]
        features_added.append("calcium_pH_interaction")

    if "urea" in df.columns and "ph" in df.columns:
        df["urea_pH_interaction"] = df["urea"] * df["ph"]
        features_added.append("urea_pH_interaction")

    if "osmo" in df.columns and "calc" in df.columns:
        df["osmolarity_calcium_interaction"] = df["osmo"] * df["calc"]
        features_added.append("osmolarity_calcium_interaction")

    if "osmo" in df.columns and "gravity" in df.columns:
        df["osmo_density"] = df["osmo"] * df["gravity"]
        features_added.append("osmo_density")

    # --- Ratio features (concentration relationships) ---
    if "calc" in df.columns and "urea" in df.columns:
        df["calcium_to_urea_ratio"] = df["calc"] / (df["urea"] + 1e-8)
        features_added.append("calcium_to_urea_ratio")

    if "osmo" in df.columns and "gravity" in df.columns:
        df["osmolality_to_sg_ratio"] = df["osmo"] / (df["gravity"] + 1e-8)
        features_added.append("osmolality_to_sg_ratio")

    if "gravity" in df.columns and "calc" in df.columns:
        df["specific_gravity_calcium_ratio"] = df["gravity"] / (df["calc"] + 1e-8)
        features_added.append("specific_gravity_calcium_ratio")

    if "calc" in df.columns and "cond" in df.columns:
        df["calcium_conductivity_ratio"] = df["calc"] / (df["cond"] + 1e-8)
        features_added.append("calcium_conductivity_ratio")

    if "urea" in df.columns and "calc" in df.columns:
        df["urea_calc_ratio"] = df["urea"] / (df["calc"] + 1e-8)
        features_added.append("urea_calc_ratio")

    if "osmo" in df.columns and "cond" in df.columns:
        df["osmo_cond_ratio"] = df["osmo"] / (df["cond"] + 1e-8)
        features_added.append("osmo_cond_ratio")

    # --- Complex domain features ---
    if "cond" in df.columns and "ph" in df.columns:
        # Electrolyte balance: conductivity relative to hydrogen ion concentration
        df["electrolyte_balance"] = df["cond"] / (10 ** (-df["ph"]) + 1e-8)
        features_added.append("electrolyte_balance")

    if "gravity" in df.columns and "osmo" in df.columns:
        # Estimated urine volume proxy
        df["urine_volume"] = (1000 * df["gravity"] * df["osmo"]) / (18 * 1.001)
        features_added.append("urine_volume")

    _save_data(df, outputs["data"])

    return f"create_kidney_stone_features: added {len(features_added)} features: {features_added}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "split_data": split_data,
    "engineer_features": engineer_features,
    "create_submission": create_submission,
    "fill_missing": fill_missing,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "train_xgboost_classifier": train_xgboost_classifier,
    "train_ensemble_classifier": train_ensemble_classifier,
    "predict_classifier": predict_classifier,
}


# =============================================================================
# PIPELINE SPECIFICATION (mirrors pipeline.json)
# =============================================================================

PIPELINE_SPEC = {
    "name": "playground-series-s3e12",
    "description": "Binary classification for kidney stone prediction using ensemble of LightGBM + XGBoost + RandomForest with domain-specific urine analysis features",
    "version": "2.0.0",
    "problem_type": "binary",
    "target_column": "target",
    "id_column": "id",
    "steps": [
        {
            "service": "create_kidney_stone_features",
            "inputs": {"data": "playground-series-s3e12/datasets/train.csv"},
            "outputs": {"data": "playground-series-s3e12/artifacts/train_01_features.csv"},
            "module": "playground_s3e12_services",
        },
        {
            "service": "create_kidney_stone_features",
            "inputs": {"data": "playground-series-s3e12/datasets/test.csv"},
            "outputs": {"data": "playground-series-s3e12/artifacts/test_01_features.csv"},
            "module": "playground_s3e12_services",
        },
        {
            "service": "split_data",
            "inputs": {"data": "playground-series-s3e12/artifacts/train_01_features.csv"},
            "outputs": {
                "train_data": "playground-series-s3e12/artifacts/train_split.csv",
                "valid_data": "playground-series-s3e12/artifacts/valid_split.csv",
            },
            "params": {"stratify_column": "target", "test_size": 0.2, "random_state": 42},
            "module": "preprocessing_services",
        },
        {
            "service": "train_ensemble_classifier",
            "inputs": {
                "train_data": "playground-series-s3e12/artifacts/train_split.csv",
                "valid_data": "playground-series-s3e12/artifacts/valid_split.csv",
            },
            "outputs": {
                "model": "playground-series-s3e12/artifacts/model.pkl",
                "metrics": "playground-series-s3e12/artifacts/metrics.json",
            },
            "params": {
                "label_column": "target",
                "id_column": "id",
                "model_types": ["lightgbm", "xgboost", "random_forest"],
                "weights": [0.4, 0.35, 0.25],
                "random_state": 42,
            },
            "module": "classification_services",
        },
        {
            "service": "predict_classifier",
            "inputs": {
                "model": "playground-series-s3e12/artifacts/model.pkl",
                "data": "playground-series-s3e12/artifacts/test_01_features.csv",
            },
            "outputs": {
                "predictions": "playground-series-s3e12/artifacts/predictions.csv",
            },
            "params": {
                "id_column": "id",
                "prediction_column": "target",
                "proba_as_prediction": True,
            },
            "module": "classification_services",
        },
        {
            "service": "create_submission",
            "inputs": {
                "predictions": "playground-series-s3e12/artifacts/predictions.csv",
            },
            "outputs": {
                "submission": "playground-series-s3e12/artifacts/submission.csv",
            },
            "params": {
                "id_column": "id",
                "prediction_column": "target",
            },
            "module": "preprocessing_services",
        },
    ],
}
