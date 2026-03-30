"""
Playground Series S3E24 - Binary Prediction of Smoker Status
=============================================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e24
Problem Type: Binary Classification
Target: smoking (0 or 1)

Competition-specific services based on top solution notebooks:
- engineer_health_features: BMI, cholesterol ratios, blood pressure flags, log transforms
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# Import shared I/O
from services.io_utils import load_data as _load_data, save_data as _save_data

# =============================================================================
# HEALTH-SPECIFIC FEATURE ENGINEERING
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Engineer health-related features for smoker prediction (BMI, cholesterol ratio, blood pressure flags)",
    tags=["feature-engineering", "health", "biometrics", "generic"],
    version="1.0.0",
)
def engineer_health_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    height_column: str = "height(cm)",
    weight_column: str = "weight(kg)",
    systolic_column: str = "systolic",
    hdl_column: str = "HDL",
    ldl_column: str = "LDL",
    eyesight_left_column: str = "eyesight(left)",
    eyesight_right_column: str = "eyesight(right)",
    hearing_left_column: str = "hearing(left)",
    hearing_right_column: str = "hearing(right)",
    triglyceride_column: str = "triglyceride",
    creatinine_column: str = "serum creatinine",
    # Thresholds (parameterized for reuse)
    systolic_threshold: float = 130.0,
    eyesight_threshold: float = 0.5,
    hearing_threshold: float = 30.0,
) -> str:
    """
    Engineer health-related features for biometric prediction tasks.

    Based on top Kaggle solution notebooks for smoker status prediction.
    All column names and thresholds are parameterized for reuse.

    Features created:
    - BMI: Body Mass Index = weight / (height/100)^2
    - high_blood_pressure: systolic > threshold (binary)
    - cholesterol_ratio: HDL / (LDL + epsilon)
    - poor_eyesight: either eye below threshold (binary)
    - poor_hearing: either ear below threshold (binary)
    - log_triglyceride: log1p transformation
    - log_creatinine: log1p transformation
    """
    df = _load_data(inputs["data"])
    features_added = []

    # BMI = weight(kg) / (height(cm) / 100)^2
    if height_column in df.columns and weight_column in df.columns:
        height_m = df[height_column] / 100.0
        df["BMI"] = df[weight_column] / (height_m ** 2)
        features_added.append("BMI")

    # High blood pressure flag
    if systolic_column in df.columns:
        df["high_blood_pressure"] = (df[systolic_column] > systolic_threshold).astype(int)
        features_added.append("high_blood_pressure")

    # Cholesterol ratio (HDL / LDL)
    if hdl_column in df.columns and ldl_column in df.columns:
        df["cholesterol_ratio"] = df[hdl_column] / (df[ldl_column] + 1e-5)
        features_added.append("cholesterol_ratio")

    # Poor eyesight (either eye below threshold)
    if eyesight_left_column in df.columns and eyesight_right_column in df.columns:
        df["poor_eyesight"] = (
            (df[eyesight_left_column] < eyesight_threshold) |
            (df[eyesight_right_column] < eyesight_threshold)
        ).astype(int)
        features_added.append("poor_eyesight")

    # Poor hearing (either ear below threshold)
    if hearing_left_column in df.columns and hearing_right_column in df.columns:
        df["poor_hearing"] = (
            (df[hearing_left_column] < hearing_threshold) |
            (df[hearing_right_column] < hearing_threshold)
        ).astype(int)
        features_added.append("poor_hearing")

    # Log transformations for skewed features
    if triglyceride_column in df.columns:
        df["log_triglyceride"] = np.log1p(df[triglyceride_column])
        features_added.append("log_triglyceride")

    if creatinine_column in df.columns:
        df["log_creatinine"] = np.log1p(df[creatinine_column])
        features_added.append("log_creatinine")

    _save_data(df, outputs["data"])

    return f"engineer_health_features: added {len(features_added)} features: {features_added}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "engineer_health_features": engineer_health_features,
}
