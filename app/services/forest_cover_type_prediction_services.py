"""
Forest Cover Type Prediction - SLEGO Services
==============================================
Competition: https://www.kaggle.com/competitions/forest-cover-type-prediction
Problem Type: Multiclass Classification (7 forest cover types)
Target: Cover_Type (1-7)
ID Column: Id

Predict forest cover type from cartographic variables.
Features include elevation, aspect, slope, distances, and binary wilderness area/soil type indicators.

Solution Notebook Insights (03_ashakadiyala):
- AdaBoost classifier with StratifiedKFold cross-validation
- StandardScaler preprocessing for all numeric features
- n_estimators=50 gave good results

Competition-specific services:
- create_distance_features: Combined distance metrics (euclidean to hydrology)
- create_hillshade_features: Aggregate hillshade statistics
- create_elevation_features: Binned elevation and interaction terms
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
    from services.classification_services import train_automl_classifier, train_adaboost_classifier, predict_classifier
    from services.preprocessing_services import split_data, fit_scaler, transform_scaler, create_submission
except ImportError:
    from classification_services import train_automl_classifier, train_adaboost_classifier, predict_classifier
    from preprocessing_services import split_data, fit_scaler, transform_scaler, create_submission


# =============================================================================
# GENERIC REUSABLE SERVICES (G1: Parameterized for reuse)
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create distance-based aggregate features from multiple distance columns",
    tags=["preprocessing", "feature-engineering", "generic", "distance"],
    version="1.0.0"
)
def create_distance_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    distance_columns: List[str] = None,
    output_prefix: str = "dist",
) -> str:
    """
    Create aggregate features from distance columns.

    G1 Compliance: Generic, works with any dataset having distance columns.
    G4 Compliance: Column names parameterized, not hardcoded.

    Args:
        distance_columns: Columns containing distance values (auto-detected if None)
        output_prefix: Prefix for new feature columns

    Works with: forest-cover, any geospatial dataset with distance features
    """
    df = pd.read_csv(inputs["data"])
    distance_columns = distance_columns or [
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points'
    ]

    existing = [c for c in distance_columns if c in df.columns]

    if existing:
        df[f'{output_prefix}_sum'] = df[existing].sum(axis=1)
        df[f'{output_prefix}_mean'] = df[existing].mean(axis=1)
        df[f'{output_prefix}_std'] = df[existing].std(axis=1)

        # Euclidean distance to hydrology (common GIS feature)
        if 'Horizontal_Distance_To_Hydrology' in df.columns and 'Vertical_Distance_To_Hydrology' in df.columns:
            df['hydro_euclidean'] = np.sqrt(
                df['Horizontal_Distance_To_Hydrology']**2 +
                df['Vertical_Distance_To_Hydrology']**2
            )

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"create_distance_features: created {len(existing) + 4} distance aggregate features"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create hillshade interaction features from solar illumination columns",
    tags=["preprocessing", "feature-engineering", "generic", "geospatial"],
    version="1.0.0"
)
def create_hillshade_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    hillshade_columns: List[str] = None,
) -> str:
    """
    Create aggregate and interaction features from hillshade columns.

    G1 Compliance: Generic, works with any dataset having hillshade/illumination columns.
    G4 Compliance: Column names parameterized.

    Args:
        hillshade_columns: Columns containing hillshade values (0-255 scale)

    Works with: forest-cover, terrain analysis, remote sensing datasets
    """
    df = pd.read_csv(inputs["data"])
    hillshade_columns = hillshade_columns or ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

    existing = [c for c in hillshade_columns if c in df.columns]

    if existing:
        df['hillshade_mean'] = df[existing].mean(axis=1)
        df['hillshade_std'] = df[existing].std(axis=1)
        df['hillshade_range'] = df[existing].max(axis=1) - df[existing].min(axis=1)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"create_hillshade_features: created 3 hillshade aggregate features"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create elevation-based features including binning and interactions",
    tags=["preprocessing", "feature-engineering", "generic", "geospatial"],
    version="1.0.0"
)
def create_elevation_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    elevation_column: str = "Elevation",
    vertical_dist_column: str = "Vertical_Distance_To_Hydrology",
    n_bins: int = 10,
) -> str:
    """
    Create features related to elevation including binning and interactions.

    G1 Compliance: Generic, works with any elevation/altitude data.
    G4 Compliance: Column names and bin count parameterized.

    Args:
        elevation_column: Column containing elevation values
        vertical_dist_column: Column with vertical distance (for interaction)
        n_bins: Number of bins for discretization

    Works with: forest-cover, terrain classification, environmental modeling
    """
    df = pd.read_csv(inputs["data"])

    if elevation_column in df.columns:
        df['elevation_binned'] = pd.cut(df[elevation_column], bins=n_bins, labels=False)

        if vertical_dist_column in df.columns:
            df['elev_minus_vdist'] = df[elevation_column] - df[vertical_dist_column]

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"create_elevation_features: created elevation binned and interaction features"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "create_distance_features": create_distance_features,
    "create_hillshade_features": create_hillshade_features,
    "create_elevation_features": create_elevation_features,
    "split_data": split_data,
    "fit_scaler": fit_scaler,
    "transform_scaler": transform_scaler,
    "train_automl_classifier": train_automl_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}


# =============================================================================
# PIPELINE SPECIFICATION (Matches solution notebook approach)
# =============================================================================

PIPELINE_SPEC = [
    # Step 1: Feature Engineering - Distance features
    {
        "service": "create_distance_features",
        "inputs": {"data": "forest-cover-type-prediction/datasets/train.csv"},
        "outputs": {"data": "forest-cover-type-prediction/artifacts/train_01_dist.csv"},
        "params": {},
        "module": "forest_cover_type_prediction_services"
    },
    # Step 2: Feature Engineering - Hillshade features
    {
        "service": "create_hillshade_features",
        "inputs": {"data": "forest-cover-type-prediction/artifacts/train_01_dist.csv"},
        "outputs": {"data": "forest-cover-type-prediction/artifacts/train_02_hill.csv"},
        "params": {},
        "module": "forest_cover_type_prediction_services"
    },
    # Step 3: Feature Engineering - Elevation features
    {
        "service": "create_elevation_features",
        "inputs": {"data": "forest-cover-type-prediction/artifacts/train_02_hill.csv"},
        "outputs": {"data": "forest-cover-type-prediction/artifacts/train_final.csv"},
        "params": {},
        "module": "forest_cover_type_prediction_services"
    },
    # Step 4: Train/Validation Split (stratified)
    {
        "service": "split_data",
        "inputs": {"data": "forest-cover-type-prediction/artifacts/train_final.csv"},
        "outputs": {
            "train_data": "forest-cover-type-prediction/artifacts/train_split.csv",
            "valid_data": "forest-cover-type-prediction/artifacts/valid_split.csv"
        },
        "params": {"stratify_column": "Cover_Type", "test_size": 0.2, "random_state": 42},
        "module": "preprocessing_services"
    },
    # Step 5: Fit scaler on training data (from notebook: StandardScaler)
    {
        "service": "fit_scaler",
        "inputs": {"data": "forest-cover-type-prediction/artifacts/train_split.csv"},
        "outputs": {"artifact": "forest-cover-type-prediction/artifacts/scaler.pkl"},
        "params": {"method": "standard", "exclude_columns": ["Id", "Cover_Type"]},
        "module": "preprocessing_services"
    },
    # Step 6: Apply scaler to training data
    {
        "service": "transform_scaler",
        "inputs": {
            "data": "forest-cover-type-prediction/artifacts/train_split.csv",
            "artifact": "forest-cover-type-prediction/artifacts/scaler.pkl"
        },
        "outputs": {"data": "forest-cover-type-prediction/artifacts/train_scaled.csv"},
        "params": {},
        "module": "preprocessing_services"
    },
    # Step 7: Apply scaler to validation data
    {
        "service": "transform_scaler",
        "inputs": {
            "data": "forest-cover-type-prediction/artifacts/valid_split.csv",
            "artifact": "forest-cover-type-prediction/artifacts/scaler.pkl"
        },
        "outputs": {"data": "forest-cover-type-prediction/artifacts/valid_scaled.csv"},
        "params": {},
        "module": "preprocessing_services"
    },
    # Step 8: Train AutoML classifier (FLAML - finds best model automatically)
    {
        "service": "train_automl_classifier",
        "inputs": {
            "train_data": "forest-cover-type-prediction/artifacts/train_scaled.csv",
            "valid_data": "forest-cover-type-prediction/artifacts/valid_scaled.csv"
        },
        "outputs": {
            "model": "forest-cover-type-prediction/artifacts/model.pkl",
            "metrics": "forest-cover-type-prediction/artifacts/metrics.json"
        },
        "params": {
            "target_column": "Cover_Type",
            "id_column": "Id",
            "time_budget": 120,
            "metric": "accuracy",
            "random_state": 42
        },
        "module": "classification_services"
    }
]


# =============================================================================
# INFERENCE PIPELINE (for test set prediction)
# =============================================================================

INFERENCE_SPEC = [
    # Apply same feature engineering to test set
    {
        "service": "create_distance_features",
        "inputs": {"data": "forest-cover-type-prediction/datasets/test.csv"},
        "outputs": {"data": "forest-cover-type-prediction/artifacts/test_01_dist.csv"},
        "params": {},
        "module": "forest_cover_type_prediction_services"
    },
    {
        "service": "create_hillshade_features",
        "inputs": {"data": "forest-cover-type-prediction/artifacts/test_01_dist.csv"},
        "outputs": {"data": "forest-cover-type-prediction/artifacts/test_02_hill.csv"},
        "params": {},
        "module": "forest_cover_type_prediction_services"
    },
    {
        "service": "create_elevation_features",
        "inputs": {"data": "forest-cover-type-prediction/artifacts/test_02_hill.csv"},
        "outputs": {"data": "forest-cover-type-prediction/artifacts/test_final.csv"},
        "params": {},
        "module": "forest_cover_type_prediction_services"
    },
    # Apply fitted scaler to test data
    {
        "service": "transform_scaler",
        "inputs": {
            "data": "forest-cover-type-prediction/artifacts/test_final.csv",
            "artifact": "forest-cover-type-prediction/artifacts/scaler.pkl"
        },
        "outputs": {"data": "forest-cover-type-prediction/artifacts/test_scaled.csv"},
        "params": {},
        "module": "preprocessing_services"
    },
    # Generate predictions
    {
        "service": "predict_classifier",
        "inputs": {
            "data": "forest-cover-type-prediction/artifacts/test_scaled.csv",
            "model": "forest-cover-type-prediction/artifacts/model.pkl"
        },
        "outputs": {"predictions": "forest-cover-type-prediction/artifacts/predictions.csv"},
        "params": {"id_column": "Id", "prediction_column": "Cover_Type"},
        "module": "classification_services"
    },
    # Format submission
    {
        "service": "create_submission",
        "inputs": {"predictions": "forest-cover-type-prediction/artifacts/predictions.csv"},
        "outputs": {"submission": "forest-cover-type-prediction/submission.csv"},
        "params": {"id_column": "Id", "prediction_column": "Cover_Type"},
        "module": "preprocessing_services"
    }
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
