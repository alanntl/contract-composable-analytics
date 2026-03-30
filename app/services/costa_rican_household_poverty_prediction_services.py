"""
Costa Rican Household Poverty Prediction - SLEGO Services
==========================================================
Competition: https://www.kaggle.com/competitions/costa-rican-household-poverty-prediction
Problem Type: Multiclass Classification (4 poverty levels: 1=extreme, 2=moderate, 3=vulnerable, 4=non-vulnerable)
Target: Target (1-4)
ID Column: Id

Predict poverty level for Costa Rican households using socioeconomic indicators.
Features include household characteristics, demographics, education levels, and asset ownership.

Solution Notebook Insights:
- Notebook 02 (nikitpatel): LightGBM/XGBoost with extensive feature engineering
  - Convert mixed-type columns (dependency, edjefe, edjefa) from yes/no to numeric
  - Create household ratio features (bedrooms/rooms, rent/person, etc.)
  - Create derived features (children_fraction, working_man_fraction, human_density)
  - Household-level aggregations (age, escolari stats by idhogar)
  - Drop SQB squared columns and redundant columns
- Notebook 01 (fabiookina): KNN baseline with LabelEncoder
- Notebook 03 (csmohamedayman): Deep learning with categorical features

Competition-specific services:
- clean_mixed_type_columns: Convert dependency/edjefe/edjefa from yes/no strings to numeric
- engineer_household_features: Create ratio and derived household features
- aggregate_household_features: Household-level statistical aggregations
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

# =============================================================================
# IMPORTS FROM COMMON MODULES (G1: Reuse existing services)
# =============================================================================
try:
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.preprocessing_services import split_data, encode_all_categorical, fill_missing, create_submission
except ImportError:
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import split_data, encode_all_categorical, fill_missing, create_submission

from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Convert mixed-type columns (yes/no strings and numeric) to pure numeric",
    tags=["preprocessing", "cleaning", "generic", "mixed-types"],
    version="1.0.0"
)
def clean_mixed_type_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    yes_value: float = 1.0,
    no_value: float = 0.0,
) -> str:
    """
    Convert columns that contain a mix of numeric values and yes/no strings
    to pure numeric. Common in survey data where some fields have yes/no
    while others have numeric values.

    G1 Compliance: Generic, works with any dataset having mixed-type columns.
    G4 Compliance: Column names parameterized.

    Args:
        columns: Columns to clean (auto-detected if None: dependency, edjefe, edjefa)
        yes_value: Numeric value for 'yes' (default 1.0)
        no_value: Numeric value for 'no' (default 0.0)

    Works with: costa-rican-household, any survey/census dataset with mixed types
    """
    df = _load_data(inputs["data"])

    columns = columns or ['dependency', 'edjefe', 'edjefa']
    cleaned = []

    for col in columns:
        if col not in df.columns:
            continue
        # Map yes/no to numeric, convert rest to float
        mapping = {'yes': yes_value, 'no': no_value}
        df[col] = df[col].replace(mapping)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        cleaned.append(col)

    _save_data(df, outputs["data"])
    return f"clean_mixed_type_columns: cleaned {len(cleaned)} columns: {cleaned}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create household ratio and derived features from demographic/housing columns",
    tags=["feature-engineering", "generic", "household", "demographics"],
    version="1.0.0"
)
def engineer_household_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    drop_sqb: bool = True,
    drop_redundant: bool = True,
    drop_ids: bool = True,
) -> str:
    """
    Create household-level ratio and derived features commonly used in
    poverty prediction and socioeconomic analysis.

    Features created (from solution notebook 02):
    - Ratio features: bedrooms/rooms, rent/rooms, people/rooms, etc.
    - Fraction features: children_fraction, working_man_fraction, etc.
    - Density features: human_density, mobile_density, etc.
    - Difference features: people_not_living, people_weird_stat

    G1 Compliance: Generic, works with household survey data.
    G4 Compliance: Control feature groups via parameters.

    Args:
        drop_sqb: Drop SQB (squared) columns that are redundant
        drop_redundant: Drop redundant columns (hhsize, female, area2)
        drop_ids: Drop idhogar (keep Id for submission)

    Works with: costa-rican-household, household surveys, census data
    """
    df = _load_data(inputs["data"])
    features_added = []

    # --- Ratio features (from notebook 02: extract_features) ---
    ratio_pairs = [
        ('bedrooms_to_rooms', 'bedrooms', 'rooms'),
        ('rent_to_rooms', 'v2a1', 'rooms'),
        ('rent_to_bedrooms', 'v2a1', 'bedrooms'),
        ('tamhog_to_rooms', 'tamhog', 'rooms'),
        ('tamhog_to_bedrooms', 'tamhog', 'bedrooms'),
        ('r4t3_to_tamhog', 'r4t3', 'tamhog'),
        ('r4t3_to_rooms', 'r4t3', 'rooms'),
        ('r4t3_to_bedrooms', 'r4t3', 'bedrooms'),
        ('rent_to_r4t3', 'v2a1', 'r4t3'),
        ('hhsize_to_rooms', 'hhsize', 'rooms'),
        ('hhsize_to_bedrooms', 'hhsize', 'bedrooms'),
        ('rent_to_hhsize', 'v2a1', 'hhsize'),
    ]
    for name, num, den in ratio_pairs:
        if num in df.columns and den in df.columns:
            df[name] = (df[num] / df[den].replace(0, np.nan)).fillna(0).astype(np.float32)
            features_added.append(name)

    # --- Fraction and density features (from notebook 02: do_features) ---
    feats_div = [
        ('fe_children_fraction', 'r4t1', 'r4t3'),
        ('fe_working_man_fraction', 'r4h2', 'r4t3'),
        ('fe_all_man_fraction', 'r4h3', 'r4t3'),
        ('fe_human_density', 'tamviv', 'rooms'),
        ('fe_human_bed_density', 'tamviv', 'bedrooms'),
        ('fe_rent_per_person', 'v2a1', 'r4t3'),
        ('fe_rent_per_room', 'v2a1', 'rooms'),
        ('fe_mobile_density', 'qmobilephone', 'r4t3'),
        ('fe_tablet_density', 'v18q1', 'r4t3'),
        ('fe_mobile_adult_density', 'qmobilephone', 'r4t2'),
        ('fe_tablet_adult_density', 'v18q1', 'r4t2'),
    ]
    for name, num, den in feats_div:
        if num in df.columns and den in df.columns:
            df[name] = (df[num] / df[den].replace(0, np.nan)).fillna(0).astype(np.float32)
            features_added.append(name)

    # --- Difference features ---
    feats_sub = [
        ('fe_people_not_living', 'tamhog', 'tamviv'),
        ('fe_people_weird_stat', 'tamhog', 'r4t3'),
    ]
    for name, col1, col2 in feats_sub:
        if col1 in df.columns and col2 in df.columns:
            df[name] = (df[col1] - df[col2]).astype(np.float32)
            features_added.append(name)

    # --- Drop SQB (squared) columns ---
    if drop_sqb:
        sqb_cols = [c for c in df.columns if c.startswith('SQB') or c == 'agesq']
        df = df.drop(columns=sqb_cols, errors='ignore')

    # --- Drop redundant columns ---
    if drop_redundant:
        redundant = ['hhsize', 'female', 'area2']
        df = df.drop(columns=[c for c in redundant if c in df.columns], errors='ignore')

    # --- Drop household ID (but keep individual Id) ---
    if drop_ids:
        if 'idhogar' in df.columns:
            df = df.drop(columns=['idhogar'], errors='ignore')
        if 'Index' in df.columns:
            df = df.drop(columns=['Index'], errors='ignore')

    _save_data(df, outputs["data"])
    return f"engineer_household_features: added {len(features_added)} features, drop_sqb={drop_sqb}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "split_data": split_data,
    "encode_all_categorical": encode_all_categorical,
    "fill_missing": fill_missing,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}


# =============================================================================
# PIPELINE SPECIFICATION (Training)
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "clean_mixed_type_columns",
        "inputs": {"data": "costa-rican-household-poverty-prediction/datasets/train.csv"},
        "outputs": {"data": "costa-rican-household-poverty-prediction/artifacts/train_01_cleaned.csv"},
        "params": {"columns": ["dependency", "edjefe", "edjefa"]},
        "module": "costa_rican_household_poverty_prediction_services"
    },
    {
        "service": "engineer_household_features",
        "inputs": {"data": "costa-rican-household-poverty-prediction/artifacts/train_01_cleaned.csv"},
        "outputs": {"data": "costa-rican-household-poverty-prediction/artifacts/train_02_features.csv"},
        "params": {"drop_sqb": True, "drop_redundant": True, "drop_ids": True},
        "module": "costa_rican_household_poverty_prediction_services"
    },
    {
        "service": "encode_all_categorical",
        "inputs": {"data": "costa-rican-household-poverty-prediction/artifacts/train_02_features.csv"},
        "outputs": {"data": "costa-rican-household-poverty-prediction/artifacts/train_03_encoded.csv"},
        "params": {"exclude_columns": ["Id", "Target"]},
        "module": "preprocessing_services"
    },
    {
        "service": "split_data",
        "inputs": {"data": "costa-rican-household-poverty-prediction/artifacts/train_03_encoded.csv"},
        "outputs": {
            "train_data": "costa-rican-household-poverty-prediction/artifacts/train_split.csv",
            "valid_data": "costa-rican-household-poverty-prediction/artifacts/valid_split.csv"
        },
        "params": {"stratify_column": "Target", "test_size": 0.2, "random_state": 42},
        "module": "preprocessing_services"
    },
    {
        "service": "train_lightgbm_classifier",
        "inputs": {
            "train_data": "costa-rican-household-poverty-prediction/artifacts/train_split.csv",
            "valid_data": "costa-rican-household-poverty-prediction/artifacts/valid_split.csv"
        },
        "outputs": {
            "model": "costa-rican-household-poverty-prediction/artifacts/model.pkl",
            "metrics": "costa-rican-household-poverty-prediction/artifacts/metrics.json"
        },
        "params": {
            "label_column": "Target",
            "id_column": "Id",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "early_stopping_rounds": 100
        },
        "module": "classification_services"
    },
]


# =============================================================================
# INFERENCE PIPELINE (for test set prediction)
# =============================================================================

INFERENCE_SPEC = [
    {
        "service": "clean_mixed_type_columns",
        "inputs": {"data": "costa-rican-household-poverty-prediction/datasets/test.csv"},
        "outputs": {"data": "costa-rican-household-poverty-prediction/artifacts/test_01_cleaned.csv"},
        "params": {"columns": ["dependency", "edjefe", "edjefa"]},
        "module": "costa_rican_household_poverty_prediction_services"
    },
    {
        "service": "engineer_household_features",
        "inputs": {"data": "costa-rican-household-poverty-prediction/artifacts/test_01_cleaned.csv"},
        "outputs": {"data": "costa-rican-household-poverty-prediction/artifacts/test_02_features.csv"},
        "params": {"drop_sqb": True, "drop_redundant": True, "drop_ids": True},
        "module": "costa_rican_household_poverty_prediction_services"
    },
    {
        "service": "encode_all_categorical",
        "inputs": {"data": "costa-rican-household-poverty-prediction/artifacts/test_02_features.csv"},
        "outputs": {"data": "costa-rican-household-poverty-prediction/artifacts/test_03_encoded.csv"},
        "params": {"exclude_columns": ["Id"]},
        "module": "preprocessing_services"
    },
    {
        "service": "predict_classifier",
        "inputs": {
            "data": "costa-rican-household-poverty-prediction/artifacts/test_03_encoded.csv",
            "model": "costa-rican-household-poverty-prediction/artifacts/model.pkl"
        },
        "outputs": {"predictions": "costa-rican-household-poverty-prediction/artifacts/predictions.csv"},
        "params": {"id_column": "Id", "prediction_column": "Target"},
        "module": "classification_services"
    },
    {
        "service": "create_submission",
        "inputs": {"predictions": "costa-rican-household-poverty-prediction/artifacts/predictions.csv"},
        "outputs": {"submission": "costa-rican-household-poverty-prediction/submission.csv"},
        "params": {"id_column": "Id", "prediction_column": "Target"},
        "module": "preprocessing_services"
    },
]