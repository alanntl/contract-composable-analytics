"""
SIIM-ISIC Melanoma Classification - Contract-Composable Analytics Services
====================================================
Competition: https://www.kaggle.com/competitions/siim-isic-melanoma-classification
Problem Type: Binary Classification (image-based, metadata fallback)
Target: target (0=benign, 1=malignant)
ID Column: image_name
Metric: Area under ROC curve

Identify melanoma in skin lesion images. Real competition requires deep
learning (ResNet50/EfficientNet); this module provides a metadata-only
baseline using patient demographics (sex, age, anatomical site).

Key insights from solution notebooks:
- Top solutions use ResNet50 (timm) with 384x384 images
- All 3 scored solutions use Tez library + albumentations
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Dropout 0.5, AdamW optimizer, CosineAnnealingWarmRestarts scheduler
- Without images, metadata-only approach is a weak baseline

Competition-specific services:
- engineer_melanoma_metadata_features: Demographic risk features from patient metadata
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
    from services.classification_services import (
        train_lightgbm_classifier,
        train_ensemble_classifier,
        predict_classifier,
    )
    from services.preprocessing_services import (
        fill_missing,
        fit_encoder,
        transform_encoder,
        drop_columns,
        split_data,
        create_submission,
    )
except ImportError:
    from classification_services import (
        train_lightgbm_classifier,
        train_ensemble_classifier,
        predict_classifier,
    )
    from preprocessing_services import (
        fill_missing,
        fit_encoder,
        transform_encoder,
        drop_columns,
        split_data,
        create_submission,
    )


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Engineer demographic risk features for skin lesion classification",
    tags=["feature-engineering", "medical", "dermatology", "generic"],
    version="1.0.0",
)
def engineer_melanoma_metadata_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    age_column: str = "age_approx",
    sex_column: str = "sex",
    site_column: str = "anatom_site_general_challenge",
) -> str:
    """
    Engineer demographic risk features for skin lesion classification.

    Creates features from patient metadata that correlate with melanoma risk:
    - Age bins (melanoma risk increases with age)
    - Sex binary encoding
    - Anatomical site risk scores (trunk/extremities have higher risk)
    - Age-site interaction features

    Reusable for any dermatology/medical imaging competition with patient metadata.

    Args:
        age_column: Column with patient age
        sex_column: Column with patient sex (male/female)
        site_column: Column with anatomical site
    """
    df = pd.read_csv(inputs["data"])

    # Age bins (melanoma risk increases with age, especially 50+)
    if age_column in df.columns:
        df["age_bin"] = pd.cut(
            df[age_column].fillna(df[age_column].median()),
            bins=[0, 30, 40, 50, 60, 70, 100],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(float)
        df["age_over_50"] = (df[age_column].fillna(0) > 50).astype(int)
        df["age_over_70"] = (df[age_column].fillna(0) > 70).astype(int)

    # Sex binary encoding
    if sex_column in df.columns:
        df["is_male"] = (df[sex_column] == "male").astype(int)

    # Anatomical site risk score (based on melanoma epidemiology)
    # Trunk and lower extremities have higher melanoma incidence
    site_risk = {
        "torso": 3,
        "lower extremity": 2,
        "upper extremity": 2,
        "head/neck": 4,
        "palms/soles": 1,
        "oral/genital": 1,
    }
    if site_column in df.columns:
        df["site_risk"] = (
            df[site_column].map(site_risk).fillna(2).astype(int)
        )

    # Interaction: age * site_risk
    if age_column in df.columns and "site_risk" in df.columns:
        age_filled = df[age_column].fillna(df[age_column].median())
        df["age_site_interaction"] = age_filled * df["site_risk"]

    # Additional age features for melanoma risk stratification
    if age_column in df.columns:
        age_filled = df[age_column].fillna(df[age_column].median())
        df["age_normalized"] = age_filled / 100.0  # Normalized age
        df["age_squared"] = (age_filled / 100.0) ** 2  # Non-linear age effect

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    n_new_features = sum(
        1
        for c in ["age_bin", "age_over_50", "age_over_70", "is_male",
                   "site_risk", "age_site_interaction", "age_normalized", "age_squared"]
        if c in df.columns
    )
    return (
        f"engineer_melanoma_metadata_features: added {n_new_features} features, "
        f"{len(df)} rows"
    )


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Add patient-level aggregation features for multi-lesion patients",
    tags=["feature-engineering", "medical", "aggregation", "generic"],
    version="1.0.0",
)
def add_patient_level_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    patient_column: str = "patient_id",
    age_column: str = "age_approx",
    site_column: str = "anatom_site_general_challenge",
) -> str:
    """
    Add patient-level aggregation features for skin lesion classification.

    Creates features from patient-level statistics:
    - Number of lesions per patient
    - Patient has multiple lesion sites
    - Lesion count relative to patient's age

    Reusable for any medical competition with multiple samples per patient.
    """
    df = pd.read_csv(inputs["data"])

    # Count lesions per patient
    patient_counts = df[patient_column].value_counts()
    df["patient_lesion_count"] = df[patient_column].map(patient_counts)
    df["has_multiple_lesions"] = (df["patient_lesion_count"] > 1).astype(int)
    df["log_lesion_count"] = np.log1p(df["patient_lesion_count"])

    # Patient site diversity (how many different sites does this patient have lesions)
    if site_column in df.columns:
        site_counts = df.groupby(patient_column)[site_column].nunique()
        df["patient_site_diversity"] = df[patient_column].map(site_counts)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    return f"add_patient_level_features: added patient features, {len(df)} rows"


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = [
    # Step 1: Feature engineering on train metadata
    {
        "service": "engineer_melanoma_metadata_features",
        "inputs": {
            "data": "siim-isic-melanoma-classification/datasets/train.csv",
        },
        "outputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_01_features.csv",
        },
        "params": {
            "age_column": "age_approx",
            "sex_column": "sex",
            "site_column": "anatom_site_general_challenge",
        },
        "module": "siim_isic_melanoma_classification_services",
    },
    # Step 2: Feature engineering on test metadata
    {
        "service": "engineer_melanoma_metadata_features",
        "inputs": {
            "data": "siim-isic-melanoma-classification/datasets/test.csv",
        },
        "outputs": {
            "data": "siim-isic-melanoma-classification/artifacts/test_01_features.csv",
        },
        "params": {
            "age_column": "age_approx",
            "sex_column": "sex",
            "site_column": "anatom_site_general_challenge",
        },
        "module": "siim_isic_melanoma_classification_services",
    },
    # Step 3: Fill missing values in train
    {
        "service": "fill_missing",
        "inputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_01_features.csv",
        },
        "outputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_02_filled.csv",
        },
        "params": {"strategy": "median"},
        "module": "preprocessing_services",
    },
    # Step 4: Fill missing values in test
    {
        "service": "fill_missing",
        "inputs": {
            "data": "siim-isic-melanoma-classification/artifacts/test_01_features.csv",
        },
        "outputs": {
            "data": "siim-isic-melanoma-classification/artifacts/test_02_filled.csv",
        },
        "params": {"strategy": "median"},
        "module": "preprocessing_services",
    },
    # Step 5: Fit encoder on train
    {
        "service": "fit_encoder",
        "inputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_02_filled.csv",
        },
        "outputs": {
            "artifact": "siim-isic-melanoma-classification/artifacts/encoder.pkl",
        },
        "params": {
            "method": "ordinal",
            "exclude_columns": [
                "image_name",
                "patient_id",
                "target",
                "diagnosis",
                "benign_malignant",
            ],
        },
        "module": "preprocessing_services",
    },
    # Step 6: Transform encoder on train
    {
        "service": "transform_encoder",
        "inputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_02_filled.csv",
            "artifact": "siim-isic-melanoma-classification/artifacts/encoder.pkl",
        },
        "outputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_03_encoded.csv",
        },
        "params": {},
        "module": "preprocessing_services",
    },
    # Step 7: Transform encoder on test
    {
        "service": "transform_encoder",
        "inputs": {
            "data": "siim-isic-melanoma-classification/artifacts/test_02_filled.csv",
            "artifact": "siim-isic-melanoma-classification/artifacts/encoder.pkl",
        },
        "outputs": {
            "data": "siim-isic-melanoma-classification/artifacts/test_03_encoded.csv",
        },
        "params": {},
        "module": "preprocessing_services",
    },
    # Step 8: Drop non-feature columns from train
    {
        "service": "drop_columns",
        "inputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_03_encoded.csv",
        },
        "outputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_04_clean.csv",
        },
        "params": {
            "columns": ["patient_id", "diagnosis", "benign_malignant"],
        },
        "module": "preprocessing_services",
    },
    # Step 9: Drop non-feature columns from test
    {
        "service": "drop_columns",
        "inputs": {
            "data": "siim-isic-melanoma-classification/artifacts/test_03_encoded.csv",
        },
        "outputs": {
            "data": "siim-isic-melanoma-classification/artifacts/test_04_clean.csv",
        },
        "params": {
            "columns": ["patient_id"],
        },
        "module": "preprocessing_services",
    },
    # Step 10: Split train into train/valid
    {
        "service": "split_data",
        "inputs": {
            "data": "siim-isic-melanoma-classification/artifacts/train_04_clean.csv",
        },
        "outputs": {
            "train_data": "siim-isic-melanoma-classification/artifacts/train_split.csv",
            "valid_data": "siim-isic-melanoma-classification/artifacts/valid_split.csv",
        },
        "params": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify_column": "target",
        },
        "module": "preprocessing_services",
    },
    # Step 11: Train ensemble classifier (LightGBM + XGBoost)
    {
        "service": "train_ensemble_classifier",
        "inputs": {
            "train_data": "siim-isic-melanoma-classification/artifacts/train_split.csv",
            "valid_data": "siim-isic-melanoma-classification/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": "siim-isic-melanoma-classification/artifacts/model.pkl",
            "metrics": "siim-isic-melanoma-classification/artifacts/metrics.json",
        },
        "params": {
            "label_column": "target",
            "id_column": "image_name",
            "model_types": ["lightgbm", "xgboost"],
            "weights": [0.6, 0.4],
            "random_state": 42,
        },
        "module": "classification_services",
    },
    # Step 12: Predict probabilities on test
    {
        "service": "predict_classifier",
        "inputs": {
            "model": "siim-isic-melanoma-classification/artifacts/model.pkl",
            "data": "siim-isic-melanoma-classification/artifacts/test_04_clean.csv",
        },
        "outputs": {
            "predictions": "siim-isic-melanoma-classification/artifacts/predictions.csv",
        },
        "params": {
            "id_column": "image_name",
            "prediction_column": "target",
            "output_proba": True,
            "proba_as_prediction": True,
        },
        "module": "classification_services",
    },
    # Step 13: Format submission
    {
        "service": "create_submission",
        "inputs": {
            "predictions": "siim-isic-melanoma-classification/artifacts/predictions.csv",
        },
        "outputs": {
            "submission": "siim-isic-melanoma-classification/submission.csv",
        },
        "params": {
            "id_column": "image_name",
            "prediction_column": "target",
        },
        "module": "preprocessing_services",
    },
]


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "engineer_melanoma_metadata_features": engineer_melanoma_metadata_features,
    "fill_missing": fill_missing,
    "fit_encoder": fit_encoder,
    "transform_encoder": transform_encoder,
    "drop_columns": drop_columns,
    "split_data": split_data,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "train_ensemble_classifier": train_ensemble_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}


def run_pipeline():
    """Run the full SIIM-ISIC melanoma classification pipeline."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline_runner import PipelineRunner

    runner = PipelineRunner(
        "kb.sqlite",
        modules=[
            "siim_isic_melanoma_classification_services",
            "classification_services",
            "preprocessing_services",
        ],
    )
    result = runner.run(
        PIPELINE_SPEC,
        base_path="storage",
        pipeline_name="siim-isic-melanoma-classification",
    )
    return result
