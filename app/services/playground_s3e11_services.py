"""
Playground Series S3E11 - SLEGO Services
=========================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e11
Problem Type: Regression (Media Campaign Cost Prediction)
Target: cost
ID Column: id
Evaluation Metric: RMSLE

Predict media campaign costs from store attributes, demographics, and amenities.

Solution Notebook Insights:
- Notebook 1 (ambrosm): Zoo of 18 models (RF, ET, HGB, CatBoost, LightGBM, XGBoost, Keras),
  log1p target, feature selection (8 features), salad_bar+prepared_food merge,
  store_sqft as categorical, duplicate grouping, original data augmentation
- Notebook 2 (iqbalsyahakbar): XGBoost+CatBoost stacking, target encoding for store_sqft,
  child_ratio, facilities score, independent_child, drop 7 unimportant features
- Notebook 3 (nikitagrec): Simple averaging of top kernel predictions

Key Insights Applied:
- Merge salad_bar + prepared_food (perfectly correlated)
- Create child_ratio, children_away, amenity_score, sales_per_unit features
- Drop unimportant features: low_fat, gross_weight, recyclable_package, units_per_case, prepared_food
- Stacked ensemble (LightGBM + XGBoost + GBR) with log_target for RMSLE
- predict_stacked_regressor auto-applies expm1 inverse transform

Competition-specific services:
- create_sales_features: Sales ratio and store efficiency (G1, G4)
- create_amenity_score: Facility aggregation + salad_bar/prepared_food merge (G1, G4)
- create_family_demographics: Family demographic interactions (G1, G4)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

# =============================================================================
# IMPORTS FROM COMMON MODULES (G1: Reuse existing services)
# =============================================================================
try:
    from services.io_utils import load_data as _load_data, save_data as _save_data
    from services.preprocessing_services import split_data, drop_columns, create_submission
    from services.regression_services import (
        train_stacked_regressor,
        predict_stacked_regressor,
    )
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from preprocessing_services import split_data, drop_columns, create_submission
    from regression_services import (
        train_stacked_regressor,
        predict_stacked_regressor,
    )


# =============================================================================
# DOMAIN-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create sales ratio and store efficiency features",
    tags=["feature-engineering", "tabular", "generic"],
    version="1.0.0",
)
def create_sales_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    sales_column: str = "store_sales(in millions)",
    units_column: str = "unit_sales(in millions)",
    sqft_column: str = "store_sqft",
) -> str:
    """
    Create sales ratio and store efficiency features.

    Generates:
    - sales_per_unit: ratio of store sales to unit sales
    - sales_per_sqft: store sales normalised by store area

    G1: Works with any dataset containing sales and store size columns.
    G4: All column names are parameterised.
    """
    df = _load_data(inputs["data"])

    if sales_column in df.columns and units_column in df.columns:
        df["sales_per_unit"] = df[sales_column] / (df[units_column] + 0.001)

    if sqft_column in df.columns and sales_column in df.columns:
        df["sales_per_sqft"] = df[sales_column] / (df[sqft_column] + 1)

    _save_data(df, outputs["data"])
    return f"create_sales_features: shape={df.shape}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create aggregate amenity/facility features with salad merge",
    tags=["feature-engineering", "tabular", "generic"],
    version="2.0.0",
)
def create_amenity_score(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    amenity_columns: Optional[List[str]] = None,
    merge_salad: bool = True,
) -> str:
    """
    Create aggregate amenity/facility features from store binary columns.

    Generates:
    - salad: merged (salad_bar + prepared_food) / 2 (perfectly correlated pair)
    - amenity_score: sum of facility binary flags

    Based on notebook insight: salad_bar and prepared_food have perfect
    correlation, so merging them reduces redundancy without losing signal.

    G1: Works with any dataset containing binary facility columns.
    G4: amenity_columns and merge behaviour are parameterised.
    """
    df = _load_data(inputs["data"])

    if merge_salad and "salad_bar" in df.columns and "prepared_food" in df.columns:
        df["salad"] = (df["salad_bar"] + df["prepared_food"]) / 2

    if amenity_columns is None:
        amenity_columns = ["coffee_bar", "video_store", "salad_bar", "florist"]
    existing_cols = [c for c in amenity_columns if c in df.columns]

    if existing_cols:
        df["amenity_score"] = df[existing_cols].sum(axis=1)

    _save_data(df, outputs["data"])
    return f"create_amenity_score: shape={df.shape}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create family demographic interaction features",
    tags=["feature-engineering", "tabular", "generic"],
    version="2.0.0",
)
def create_family_demographics(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    total_children_col: str = "total_children",
    home_children_col: str = "num_children_at_home",
) -> str:
    """
    Create family demographic interaction features.

    Generates:
    - children_away: total_children - num_children_at_home
    - children_home_ratio: proportion of children still at home
    - child_ratio: total / home with inf handling (notebook 2 insight)

    G1: Works with any dataset containing parent-child demographic columns.
    G4: Column names are parameterised.
    """
    df = _load_data(inputs["data"])

    if total_children_col in df.columns and home_children_col in df.columns:
        df["children_away"] = df[total_children_col] - df[home_children_col]
        df["children_home_ratio"] = (
            df[home_children_col] / (df[total_children_col] + 0.001)
        )
        df["child_ratio"] = df[total_children_col] / df[home_children_col]
        df.replace([np.inf, -np.inf], 10, inplace=True)
        df.fillna(0, inplace=True)

    _save_data(df, outputs["data"])
    return f"create_family_demographics: shape={df.shape}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific
    "create_sales_features": create_sales_features,
    "create_amenity_score": create_amenity_score,
    "create_family_demographics": create_family_demographics,
    # Imported from common modules
    "split_data": split_data,
    "drop_columns": drop_columns,
    "create_submission": create_submission,
    "train_stacked_regressor": train_stacked_regressor,
    "predict_stacked_regressor": predict_stacked_regressor,
}


# =============================================================================
# PIPELINE SPECIFICATION (Training)
# =============================================================================

PIPELINE_SPEC = [
    # Step 1: Sales features
    {
        "service": "create_sales_features",
        "inputs": {"data": "playground-series-s3e11/datasets/train.csv"},
        "outputs": {"data": "playground-series-s3e11/artifacts/train_01_sales.csv"},
        "module": "playground_s3e11_services",
    },
    # Step 2: Amenity score + salad merge
    {
        "service": "create_amenity_score",
        "inputs": {"data": "playground-series-s3e11/artifacts/train_01_sales.csv"},
        "outputs": {"data": "playground-series-s3e11/artifacts/train_02_amenity.csv"},
        "module": "playground_s3e11_services",
    },
    # Step 3: Family demographics
    {
        "service": "create_family_demographics",
        "inputs": {"data": "playground-series-s3e11/artifacts/train_02_amenity.csv"},
        "outputs": {"data": "playground-series-s3e11/artifacts/train_03_family.csv"},
        "module": "playground_s3e11_services",
    },
    # Step 4: Drop unimportant features (notebook insight)
    {
        "service": "drop_columns",
        "inputs": {"data": "playground-series-s3e11/artifacts/train_03_family.csv"},
        "outputs": {"data": "playground-series-s3e11/artifacts/train_04_cleaned.csv"},
        "params": {
            "columns": [
                "low_fat", "gross_weight", "recyclable_package",
                "units_per_case", "prepared_food",
            ]
        },
        "module": "preprocessing_services",
    },
    # Step 5: Train/validation split
    {
        "service": "split_data",
        "inputs": {"data": "playground-series-s3e11/artifacts/train_04_cleaned.csv"},
        "outputs": {
            "train_data": "playground-series-s3e11/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e11/artifacts/valid_split.csv",
        },
        "params": {"test_size": 0.2, "random_state": 42},
        "module": "preprocessing_services",
    },
    # Step 6: Train stacked ensemble (LightGBM + XGBoost + GBR) with log_target
    {
        "service": "train_stacked_regressor",
        "inputs": {
            "train_data": "playground-series-s3e11/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e11/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": "playground-series-s3e11/artifacts/model.pkl",
            "metrics": "playground-series-s3e11/artifacts/metrics.json",
        },
        "params": {
            "label_column": "cost",
            "id_column": "id",
            "model_types": ["lightgbm", "xgboost", "gradient_boosting"],
            "log_target": True,
            "n_folds": 5,
            "random_state": 42,
            "lgbm_n_estimators": 450,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_leaves": 100,
            "lgbm_min_child_samples": 1,
            "xgb_n_estimators": 280,
            "xgb_learning_rate": 0.05,
            "xgb_max_depth": 10,
            "gbr_n_estimators": 500,
            "gbr_learning_rate": 0.05,
            "gbr_max_depth": 5,
        },
        "module": "regression_services",
    },
]


# =============================================================================
# INFERENCE PIPELINE (for test set prediction + submission)
# =============================================================================

INFERENCE_SPEC = [
    # Step 1: Sales features on test
    {
        "service": "create_sales_features",
        "inputs": {"data": "playground-series-s3e11/datasets/test.csv"},
        "outputs": {"data": "playground-series-s3e11/artifacts/test_01_sales.csv"},
        "module": "playground_s3e11_services",
    },
    # Step 2: Amenity score on test
    {
        "service": "create_amenity_score",
        "inputs": {"data": "playground-series-s3e11/artifacts/test_01_sales.csv"},
        "outputs": {"data": "playground-series-s3e11/artifacts/test_02_amenity.csv"},
        "module": "playground_s3e11_services",
    },
    # Step 3: Family demographics on test
    {
        "service": "create_family_demographics",
        "inputs": {"data": "playground-series-s3e11/artifacts/test_02_amenity.csv"},
        "outputs": {"data": "playground-series-s3e11/artifacts/test_03_family.csv"},
        "module": "playground_s3e11_services",
    },
    # Step 4: Drop unimportant features from test
    {
        "service": "drop_columns",
        "inputs": {"data": "playground-series-s3e11/artifacts/test_03_family.csv"},
        "outputs": {"data": "playground-series-s3e11/artifacts/test_04_cleaned.csv"},
        "params": {
            "columns": [
                "low_fat", "gross_weight", "recyclable_package",
                "units_per_case", "prepared_food",
            ]
        },
        "module": "preprocessing_services",
    },
    # Step 5: Predict with stacked model (auto-applies expm1 if log_target was True)
    {
        "service": "predict_stacked_regressor",
        "inputs": {
            "model": "playground-series-s3e11/artifacts/model.pkl",
            "data": "playground-series-s3e11/artifacts/test_04_cleaned.csv",
        },
        "outputs": {
            "predictions": "playground-series-s3e11/artifacts/predictions.csv",
        },
        "params": {"id_column": "id", "prediction_column": "cost"},
        "module": "regression_services",
    },
    # Step 6: Create submission file
    {
        "service": "create_submission",
        "inputs": {
            "predictions": "playground-series-s3e11/artifacts/predictions.csv",
        },
        "outputs": {
            "submission": "playground-series-s3e11/submission.csv",
        },
        "params": {"id_column": "id", "prediction_column": "cost"},
        "module": "preprocessing_services",
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
