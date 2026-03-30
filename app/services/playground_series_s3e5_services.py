"""
Playground Series S3E5 - SLEGO Services
========================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e5
Problem Type: Multiclass Classification (wine quality: 3-9)
Target: quality (integer)
ID Column: Id

Predict wine quality from physicochemical properties. Features include
fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
free/total sulfur dioxide, density, pH, sulphates, and alcohol.

Solution Notebook Insights:
- Notebook 01 (hamdy17298): MinMaxScaler + RandomForest with GridSearchCV
  (best: max_depth=13, n_estimators=300). Combined original wine dataset.
- Notebook 02 (ashokkumargarain): StandardScaler + Neural Network (Keras)
  with early stopping.
- Notebook 03 (faelk8): RobustScaler + CatBoost + LGBM + XGB ensemble.
  Combined original wine dataset, dropped duplicates. CatBoost best single model.

Key Techniques Across All Solutions:
1. Feature scaling (MinMax/Standard/Robust) - ALL solutions use scaling
2. Combining with original wine quality dataset (solutions 1 & 3)
3. Stratified train/test split on quality column
4. Gradient boosting models (RF, CatBoost, LGBM, XGB)

Competition-specific services:
- combine_with_external_data: Combine competition data with external dataset
  and drop duplicates (insights from solutions 1 & 3)
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
    from services.classification_services import (
        train_lightgbm_classifier, predict_classifier
    )
    from services.preprocessing_services import (
        split_data, fit_scaler, transform_scaler, create_submission
    )
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from classification_services import (
        train_lightgbm_classifier, predict_classifier
    )
    from preprocessing_services import (
        split_data, fit_scaler, transform_scaler, create_submission
    )


# =============================================================================
# COMPETITION-SPECIFIC SERVICE: Combine with External Data
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "external_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Combine competition data with external dataset and drop duplicates",
    tags=["preprocessing", "data-augmentation", "generic"],
    version="1.0.0",
)
def combine_with_external_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    drop_duplicates: bool = True,
    drop_id_before_dedup: bool = True,
) -> str:
    """
    Combine competition data with external/original dataset.

    Many Kaggle competitions provide synthetic data generated from real-world
    datasets. Combining with the original dataset often improves performance.
    Solutions 1 and 3 both used this technique with the original wine quality
    dataset.

    G1 Compliance: Generic, works with any dataset pair.
    G4 Compliance: All column names parameterized.

    Parameters:
        id_column: ID column to drop from external data (if present)
        drop_duplicates: Whether to drop duplicate rows after combining
        drop_id_before_dedup: Drop ID column before dedup check (IDs differ between datasets)
    """
    df = _load_data(inputs["data"])
    n_original = len(df)

    # Check if external data exists
    ext_path = inputs.get("external_data")
    if ext_path and os.path.exists(ext_path):
        ext_df = _load_data(ext_path)

        # Drop ID column from external data if present
        if id_column in ext_df.columns:
            ext_df = ext_df.drop(columns=[id_column])

        # Drop ID column from main data for concat
        has_id = id_column in df.columns
        if has_id:
            ids = df[id_column].copy()
            df_no_id = df.drop(columns=[id_column])
        else:
            df_no_id = df

        # Combine
        combined = pd.concat([df_no_id, ext_df], ignore_index=True)

        if drop_duplicates:
            before_dedup = len(combined)
            combined = combined.drop_duplicates()
            n_deduped = before_dedup - len(combined)
        else:
            n_deduped = 0

        # Re-add IDs (sequential for new combined dataset)
        if has_id:
            combined.insert(0, id_column, range(len(combined)))

        _save_data(combined, outputs["data"])
        return (f"combine_with_external_data: {n_original} + {len(ext_df)} = {len(combined)} rows "
                f"(dropped {n_deduped} duplicates)")
    else:
        # No external data available, pass through
        _save_data(df, outputs["data"])
        return f"combine_with_external_data: no external data found, pass-through ({n_original} rows)"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

# Import RandomForest classifier for this pipeline
try:
    from services.classification_services import train_random_forest_classifier
except ImportError:
    from classification_services import train_random_forest_classifier

SERVICE_REGISTRY = {
    # Competition-specific services
    "combine_with_external_data": combine_with_external_data,
    # Imported services
    "split_data": split_data,
    "fit_scaler": fit_scaler,
    "transform_scaler": transform_scaler,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "train_random_forest_classifier": train_random_forest_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}