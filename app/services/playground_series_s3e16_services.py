"""
Playground Series S3E16 - Crab Age Prediction Services
======================================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e16
Problem Type: Regression (MAE evaluation - NOT classification)
Target: Age (crab age in months)
ID Column: id
Evaluation Metric: MAE

Solution Notebook Insights:
- 01 (cozyhn): AutoGluon, Size=L*D, Surface Area, weight ratios, one-hot Sex,
  USES EXTERNAL DATA (original + synthetic + extended)
- 02 (tanujtaneja): CatBoost ensemble, outlier handling, MAE objective, EXTERNAL DATA
- 03 (oscarm524): Multi-model ensemble (GB/LGBM/XGB/CatBoost), LAD stacking,
  features: Meat Yield, Shell Ratio, Weight_to_Shucked, Viscera Ratio, EXTERNAL DATA

Key Insights Applied:
- This is REGRESSION with MAE, NOT multiclass classification
- USE EXTERNAL ORIGINAL DATA (CrabAgePrediction.csv) - ALL top solutions do this!
- Encode Sex (I/M/F -> 0/1/2), do NOT drop it
- Weight ratio features are critical (Shell Ratio, Viscera Ratio, Meat Yield)
- Size interaction features help (L*D, Surface Area, sqrt Weight)
- Volume & Density features help (solution 01)
- Round predictions to integers improves score
- Multi-model ensemble outperforms single model

Competition-specific services:
- create_crab_features_with_external: Enhanced features + external data augmentation
- round_predictions: Round regression output to integers
- train_ensemble_crab: Train multi-model ensemble (LGBM+XGB+GBR)
- predict_ensemble_crab: Predict with ensemble and average
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

try:
    from services.io_utils import load_data as _load_data, save_data as _save_data
    from services.preprocessing_services import split_data, create_submission
    from services.regression_services import train_lightgbm_regressor, predict_regressor
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from preprocessing_services import split_data, create_submission
    from regression_services import train_lightgbm_regressor, predict_regressor


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"},
    },
    description="Create ratio, size, and encoded features for crab/abalone data with optional external data",
    tags=["feature-engineering", "biology", "ratios", "playground-series-s3e16"],
    version="2.0.0",
)
def create_crab_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Age",
    id_column: str = "id",
    external_data_path: Optional[str] = None,
) -> str:
    """
    Engineer features for crab age prediction with optional external data augmentation.

    Combines train+test (and optionally external data) for consistent encoding,
    creates ratio, size, and geometric features derived from top solution notebooks.

    G1 Compliance: Single responsibility - feature engineering.
    G4 Compliance: Column names parameterized.

    Features created:
    - shell_ratio, viscera_ratio, meat_yield, weight_to_shucked (solution 03)
    - size (L*D), surface_area, sqrt_weight (solution 01)
    - volume, density, vs_ratio (solution 01 - geometric features)
    - Sex label-encoded to numeric (all solutions)

    External Data:
    - If external_data_path provided, augment training with original CrabAgePrediction.csv
    - This is CRITICAL for good performance (all top solutions use external data)
    """
    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    # Augment with external data if provided (CRITICAL for performance!)
    if external_data_path and os.path.exists(external_data_path):
        external_df = _load_data(external_data_path)
        # External data doesn't have 'id' column, add it
        if id_column not in external_df.columns:
            external_df[id_column] = range(100000, 100000 + len(external_df))
        # Deduplicate - remove rows that exist in train based on features
        train_features = set(train_df.drop(columns=[id_column, target_column], errors='ignore').apply(tuple, axis=1))
        external_mask = ~external_df.drop(columns=[id_column, target_column], errors='ignore').apply(tuple, axis=1).isin(train_features)
        external_unique = external_df[external_mask]
        print(f"External data: {len(external_df)} rows, {len(external_unique)} unique after dedup")
        train_df = pd.concat([train_df, external_unique], ignore_index=True)

    train_df["_is_train"] = 1
    test_df["_is_train"] = 0
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Handle Height=0 (invalid measurement) - use small value as in solution 01
    combined["Height"] = combined["Height"].replace(0, 0.020)

    # Label-encode Sex (I/M/F -> 0/1/2) - all solutions encode, not drop
    for col in combined.select_dtypes(include=["object"]).columns:
        combined[col] = pd.factorize(combined[col])[0]

    eps = 1e-8
    w = combined["Weight"]
    sw = combined["Shell Weight"]
    vw = combined["Viscera Weight"]
    skw = combined["Shucked Weight"]

    # Weight ratios (solutions 01, 03)
    combined["shell_ratio"] = sw / (w + eps)
    combined["viscera_ratio"] = vw / (w + eps)
    combined["meat_yield"] = skw / (w + sw + eps)
    combined["weight_to_shucked"] = w / (skw + eps)

    # Size / geometry (solution 01)
    combined["size"] = combined["Length"] * combined["Diameter"]
    combined["surface_area"] = 2 * (
        combined["Length"] * combined["Diameter"]
        + combined["Length"] * combined["Height"]
        + combined["Diameter"] * combined["Height"]
    )
    combined["sqrt_weight"] = np.sqrt(w.clip(lower=0))

    # Volume and Density (solution 01 - geometric features)
    combined["volume"] = combined["Length"] * combined["Diameter"] * combined["Height"]
    combined["density"] = w / (combined["volume"] + eps)
    combined["vs_ratio"] = combined["surface_area"] / (combined["volume"] + eps)

    # Split back
    train_out = combined[combined["_is_train"] == 1].drop(columns=["_is_train"])
    test_out = combined[combined["_is_train"] == 0].drop(columns=["_is_train"])
    if target_column in test_out.columns:
        test_out = test_out.drop(columns=[target_column])

    _save_data(train_out, outputs["train_data"])
    _save_data(test_out, outputs["test_data"])

    n_feat = len(train_out.columns) - 2  # exclude id + target
    return f"create_crab_features: train={len(train_out)}, test={len(test_out)}, {n_feat} features"


@contract(
    inputs={"predictions": {"format": "csv", "required": True}},
    outputs={"predictions": {"format": "csv"}},
    description="Round regression predictions to nearest integer",
    tags=["postprocessing", "regression", "generic"],
    version="1.0.0",
)
def round_predictions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    prediction_column: str = "Age",
    min_value: int = 1,
) -> str:
    """Round predictions to integers and clip to min_value. All top solutions do this."""
    df = _load_data(inputs["predictions"])
    if prediction_column in df.columns:
        df[prediction_column] = df[prediction_column].round().astype(int).clip(lower=min_value)
    _save_data(df, outputs["predictions"])
    return f"round_predictions: rounded {len(df)} predictions"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "valid_data": {"format": "csv", "required": True},
    },
    outputs={
        "model": {"format": "pkl"},
        "metrics": {"format": "json"},
    },
    description="Train 3-model ensemble (LGBM+XGB+GBR) for MAE regression",
    tags=["training", "ensemble", "regression", "playground-series-s3e16"],
    version="1.0.0",
)
def train_ensemble_crab(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "Age",
    id_column: str = "id",
    random_state: int = 42,
) -> str:
    """
    Train a 3-model ensemble (LightGBM + XGBoost + GradientBoosting) for MAE regression.

    Based on solution 03 which uses multiple models with averaging/stacking.
    Uses MAE objective for all models to optimize for competition metric.

    Models:
    - LightGBM: Fast, handles categorical well, MAE objective
    - XGBoost: Robust, regularized, pseudohubererror (MAE-like)
    - GradientBoosting: Sklearn, absolute_error loss

    Returns ensemble dict with models and weights (equal).
    """
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    train = _load_data(inputs["train_data"])
    valid = _load_data(inputs["valid_data"])

    exclude_cols = {label_column}
    if id_column and id_column in train.columns:
        exclude_cols.add(id_column)
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    # Model 1: LightGBM (from solution 03 params)
    lgb_model = LGBMRegressor(
        objective="mae",
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.01,
        num_leaves=105,
        reg_alpha=8,
        reg_lambda=3,
        subsample=0.6,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    lgb_pred = lgb_model.predict(X_valid)
    lgb_mae = mean_absolute_error(y_valid, lgb_pred)

    # Model 2: XGBoost (from solution 03 params - use absoluteerror for MAE)
    xgb_model = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        colsample_bytree=0.8,
        min_child_weight=20,
        subsample=0.7,
        reg_alpha=5,
        reg_lambda=3,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    xgb_pred = xgb_model.predict(X_valid)
    xgb_mae = mean_absolute_error(y_valid, xgb_pred)

    # Model 3: GradientBoosting (from solution 03 params)
    gbr_model = GradientBoostingRegressor(
        loss="absolute_error",
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        min_samples_split=10,
        min_samples_leaf=20,
        random_state=random_state,
    )
    gbr_model.fit(X_train, y_train)
    gbr_pred = gbr_model.predict(X_valid)
    gbr_mae = mean_absolute_error(y_valid, gbr_pred)

    # Ensemble prediction (simple average)
    ensemble_pred = (lgb_pred + xgb_pred + gbr_pred) / 3
    ensemble_mae = mean_absolute_error(y_valid, ensemble_pred)

    # Save ensemble as dict
    ensemble = {
        "models": {
            "lightgbm": lgb_model,
            "xgboost": xgb_model,
            "gradient_boosting": gbr_model,
        },
        "weights": {"lightgbm": 1/3, "xgboost": 1/3, "gradient_boosting": 1/3},
        "feature_cols": feature_cols,
    }

    os.makedirs(os.path.dirname(outputs["model"]), exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(ensemble, f)

    metrics = {
        "model_type": "Ensemble_LGBM_XGB_GBR",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "lgb_valid_mae": float(lgb_mae),
        "xgb_valid_mae": float(xgb_mae),
        "gbr_valid_mae": float(gbr_mae),
        "ensemble_valid_mae": float(ensemble_mae),
        "best_single_model": min([("lgb", lgb_mae), ("xgb", xgb_mae), ("gbr", gbr_mae)], key=lambda x: x[1])[0],
    }

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_ensemble_crab: ensemble MAE={ensemble_mae:.4f} (lgb={lgb_mae:.4f}, xgb={xgb_mae:.4f}, gbr={gbr_mae:.4f})"


@contract(
    inputs={
        "model": {"format": "pkl", "required": True},
        "data": {"format": "csv", "required": True},
    },
    outputs={"predictions": {"format": "csv"}},
    description="Predict with ensemble model (averaged predictions)",
    tags=["prediction", "ensemble", "regression", "playground-series-s3e16"],
    version="1.0.0",
)
def predict_ensemble_crab(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "Age",
) -> str:
    """
    Predict using the 3-model ensemble with weighted average.

    Loads ensemble dict with models and weights, predicts with each,
    and returns weighted average as final prediction.
    """
    with open(inputs["model"], "rb") as f:
        ensemble = pickle.load(f)

    data = _load_data(inputs["data"])
    feature_cols = ensemble["feature_cols"]
    X = data[feature_cols]

    # Predict with each model
    preds = {}
    for name, model in ensemble["models"].items():
        preds[name] = model.predict(X)

    # Weighted average
    weights = ensemble["weights"]
    final_pred = sum(preds[name] * weights[name] for name in preds)

    # Create output
    result = pd.DataFrame({id_column: data[id_column], prediction_column: final_pred})
    _save_data(result, outputs["predictions"])

    return f"predict_ensemble_crab: predicted {len(result)} samples"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific
    "create_crab_features": create_crab_features,
    "round_predictions": round_predictions,
    "train_ensemble_crab": train_ensemble_crab,
    "predict_ensemble_crab": predict_ensemble_crab,
    # Imported from common modules
    "split_data": split_data,
    "create_submission": create_submission,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "predict_regressor": predict_regressor,
}


# =============================================================================
# PIPELINE SPECIFICATION (Training)
# =============================================================================

PIPELINE_SPEC = [
    # Step 1: Feature engineering with EXTERNAL DATA (CRITICAL for performance!)
    {
        "service": "create_crab_features",
        "inputs": {
            "train_data": "playground-series-s3e16/datasets/train.csv",
            "test_data": "playground-series-s3e16/datasets/test.csv",
        },
        "outputs": {
            "train_data": "playground-series-s3e16/artifacts/train_featured.csv",
            "test_data": "playground-series-s3e16/artifacts/test_featured.csv",
        },
        "params": {
            "target_column": "Age",
            "id_column": "id",
            "external_data_path": "playground-series-s3e16/datasets/CrabAgePrediction.csv",
        },
        "module": "playground_series_s3e16_services",
    },
    # Step 2: Stratified train/validation split
    {
        "service": "split_data",
        "inputs": {"data": "playground-series-s3e16/artifacts/train_featured.csv"},
        "outputs": {
            "train_data": "playground-series-s3e16/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e16/artifacts/valid_split.csv",
        },
        "params": {"stratify_column": "Age", "test_size": 0.2, "random_state": 42},
        "module": "preprocessing_services",
    },
    # Step 3: Train 3-model ensemble (LGBM+XGB+GBR) - from solution 03
    {
        "service": "train_ensemble_crab",
        "inputs": {
            "train_data": "playground-series-s3e16/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e16/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": "playground-series-s3e16/artifacts/model.pkl",
            "metrics": "playground-series-s3e16/artifacts/metrics.json",
        },
        "params": {"label_column": "Age", "id_column": "id", "random_state": 42},
        "module": "playground_series_s3e16_services",
    },
]


# =============================================================================
# INFERENCE PIPELINE (for test set prediction + submission)
# =============================================================================

INFERENCE_SPEC = [
    # Step 1: Predict with ensemble model
    {
        "service": "predict_ensemble_crab",
        "inputs": {
            "model": "playground-series-s3e16/artifacts/model.pkl",
            "data": "playground-series-s3e16/artifacts/test_featured.csv",
        },
        "outputs": {"predictions": "playground-series-s3e16/artifacts/predictions.csv"},
        "params": {"id_column": "id", "prediction_column": "Age"},
        "module": "playground_series_s3e16_services",
    },
    # Step 2: Round predictions to integers
    {
        "service": "round_predictions",
        "inputs": {"predictions": "playground-series-s3e16/artifacts/predictions.csv"},
        "outputs": {"predictions": "playground-series-s3e16/artifacts/predictions_rounded.csv"},
        "params": {"prediction_column": "Age", "min_value": 1},
        "module": "playground_series_s3e16_services",
    },
    # Step 3: Format submission
    {
        "service": "create_submission",
        "inputs": {"predictions": "playground-series-s3e16/artifacts/predictions_rounded.csv"},
        "outputs": {"submission": "playground-series-s3e16/submission.csv"},
        "params": {"id_column": "id", "prediction_column": "Age"},
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
