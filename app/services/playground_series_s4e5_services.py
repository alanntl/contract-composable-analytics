"""
Playground Series S4E5 - SLEGO Services
========================================
Competition: https://www.kaggle.com/competitions/playground-series-s4e5
Problem Type: Regression (continuous probability 0-1)
Target: FloodProbability (float)
ID Column: id

Predict flood probability from 20 risk factor features. All features are
integer-valued risk scores. Target is a continuous float probability.

Solution Notebook Insights:
- Notebook 01 (syntheticprogrammer): StackingRegressor with 5 base models
  (Linear, RF, GB, XGBoost n_estimators=3000/depth=5, LightGBM n_estimators=3000/depth=4).
  StandardScaler applied. Used all 20 features selected by correlation.
- Notebook 02 (rajendarkatravath): Ridge regression with RFECV feature selection,
  StandardScaler. All 20 features selected by recursive elimination.
- Notebook 03 (vinaykashyap52): Simple LinearRegression with StandardScaler.

Key Techniques Across All Solutions:
1. All features are numeric - no encoding needed
2. No missing values in dataset
3. Feature scaling (StandardScaler) used in all solutions
4. Tree-based models (LightGBM, XGBoost) perform best
5. All 20 features contribute positively (no dropping needed)
6. Interaction features from domain groupings can help

Competition-specific services:
- create_flood_risk_features: Create domain-meaningful aggregation features
  from risk factor groupings (infrastructure, environmental, water management)
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
    from services.io_utils import load_data as _load_data, save_data as _save_data
    from services.regression_services import (
        train_lightgbm_regressor, predict_regressor
    )
    from services.preprocessing_services import (
        split_data, create_submission
    )
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from regression_services import (
        train_lightgbm_regressor, predict_regressor
    )
    from preprocessing_services import (
        split_data, create_submission
    )


# =============================================================================
# COMPETITION-SPECIFIC SERVICE: Create Flood Risk Aggregation Features
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Create domain-meaningful aggregation features from risk factor groupings",
    tags=["feature-engineering", "regression", "generic"],
    version="1.0.0",
)
def create_flood_risk_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    feature_groups: Optional[Dict[str, List[str]]] = None,
    add_total_risk: bool = True,
    add_mean_risk: bool = True,
) -> str:
    """
    Create domain-meaningful aggregation features from risk factor columns.

    Inspired by solution notebooks that used all 20 features: grouping related
    risk factors into composite scores can capture domain structure.

    G1 Compliance: Generic, works with any dataset having numeric risk columns.
    G4 Compliance: All column names parameterized via feature_groups.

    Parameters:
        feature_groups: Dict mapping group name to list of column names.
            Default groups are based on flood risk domain knowledge.
        add_total_risk: Add a TotalRisk feature (sum of all risk columns)
        add_mean_risk: Add a MeanRisk feature (mean of all risk columns)
    """
    df = _load_data(inputs["data"])
    n_original_cols = len(df.columns)

    # Default feature groups based on flood domain knowledge
    if feature_groups is None:
        feature_groups = {
            "InfrastructureRisk": [
                "DeterioratingInfrastructure", "InadequatePlanning",
                "DrainageSystems", "DamsQuality"
            ],
            "EnvironmentalRisk": [
                "Deforestation", "WetlandLoss", "ClimateChange",
                "AgriculturalPractices"
            ],
            "WaterRisk": [
                "MonsoonIntensity", "RiverManagement", "Siltation",
                "CoastalVulnerability", "TopographyDrainage"
            ],
            "HumanRisk": [
                "Urbanization", "Encroachments", "PopulationScore",
                "PoliticalFactors", "IneffectiveDisasterPreparedness"
            ],
        }

    features_added = []

    # Create group aggregation features
    for group_name, columns in feature_groups.items():
        valid_cols = [c for c in columns if c in df.columns]
        if valid_cols:
            df[group_name] = df[valid_cols].sum(axis=1)
            features_added.append(group_name)

    # Total risk: sum of all numeric features (excluding id and target)
    risk_cols = [c for c in df.columns
                 if c not in ("id", "FloodProbability") and c not in features_added
                 and df[c].dtype in ("int64", "float64")]

    if add_total_risk and risk_cols:
        df["TotalRisk"] = df[risk_cols].sum(axis=1)
        features_added.append("TotalRisk")

    if add_mean_risk and risk_cols:
        df["MeanRisk"] = df[risk_cols].mean(axis=1)
        features_added.append("MeanRisk")

    _save_data(df, outputs["data"])
    return (
        f"create_flood_risk_features: added {len(features_added)} features "
        f"({n_original_cols} -> {len(df.columns)} columns): {features_added}"
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Reused from common modules
    "split_data": split_data,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "predict_regressor": predict_regressor,
    "create_submission": create_submission,
    # Competition-specific
    "create_flood_risk_features": create_flood_risk_features,
}