"""
House Prices Advanced Regression Techniques - SLEGO Services
==============================================================

Competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
Problem Type: Regression
Target: SalePrice

This module contains ONLY domain-specific configuration data.
All services are imported from generic reusable modules:
  - preprocessing_services: feature engineering, imputation, encoding, etc.
  - regression_services: stacked regressor training and prediction
"""

import os
import sys
from typing import Dict

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import generic preprocessing services
try:
    from .preprocessing_services import (
        fit_imputer, transform_imputer,
        fit_encoder, transform_encoder,
        fit_column_filter, transform_column_filter,
        fit_skew_corrector, transform_skew_corrector,
        remove_outliers,
        split_data, create_submission, engineer_features,
        HOUSE_PRICES_FEATURE_CONFIG,
    )
except ImportError:
    from services.preprocessing_services import (
        fit_imputer, transform_imputer,
        fit_encoder, transform_encoder,
        fit_column_filter, transform_column_filter,
        fit_skew_corrector, transform_skew_corrector,
        remove_outliers,
        split_data, create_submission, engineer_features,
        HOUSE_PRICES_FEATURE_CONFIG,
    )

# Import generic regression services
try:
    from .regression_services import (
        train_stacked_regressor, predict_stacked_regressor,
    )
except ImportError:
    from services.regression_services import (
        train_stacked_regressor, predict_stacked_regressor,
    )


# =============================================================================
# DOMAIN-SPECIFIC CONFIGURATION (data only, no code)
# =============================================================================

# V2: Expanded feature config with ratio features and more interactions
HOUSE_PRICES_FEATURE_CONFIG_V2 = {
    "sum_features": [
        {"name": "TotalSF", "columns": ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]},
        {"name": "TotalPorch", "columns": ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]},
    ],
    "add_features": [
        {"name": "TotalArea", "columns": ["GrLivArea", "TotalBsmtSF"]},
    ],
    "weighted_sum_features": [
        {"name": "TotalBath", "weights": {"FullBath": 1.0, "HalfBath": 0.5, "BsmtFullBath": 1.0, "BsmtHalfBath": 0.5}},
    ],
    "difference_features": [
        {"name": "HouseAge", "minuend": "YrSold", "subtrahend": "YearBuilt"},
        {"name": "RemodAge", "minuend": "YrSold", "subtrahend": "YearRemodAdd"},
        {"name": "GarageAge", "minuend": "YrSold", "subtrahend": "GarageYrBlt"},
    ],
    "product_features": [
        {"name": "OverallScore", "columns": ["OverallQual", "OverallCond"]},
        {"name": "QualArea", "columns": ["OverallQual", "GrLivArea"]},
        {"name": "QualTotalSF", "columns": ["OverallQual", "TotalSF"]},
    ],
    "binary_features": [
        {"name": "HasGarage", "column": "GarageArea", "threshold": 0},
        {"name": "HasBsmt", "column": "TotalBsmtSF", "threshold": 0},
        {"name": "HasPool", "column": "PoolArea", "threshold": 0},
        {"name": "Has2ndFloor", "column": "2ndFlrSF", "threshold": 0},
        {"name": "HasFireplace", "column": "Fireplaces", "threshold": 0},
    ],
    "ratio_features": [
        {"name": "LivAreaPerRoom", "numerator": "GrLivArea", "denominator": "TotRmsAbvGrd"},
        {"name": "GaragePerCar", "numerator": "GarageArea", "denominator": "GarageCars"},
    ],
}

_P = "house-prices-advanced-regression-techniques"

# =============================================================================
# PIPELINE SPECIFICATION (all generic services, no competition-specific code)
# =============================================================================

PIPELINE_SPEC = [
    # --- Feature engineering ---
    {
        "service": "engineer_features",
        "inputs": {"data": f"{_P}/datasets/train.csv"},
        "outputs": {"data": f"{_P}/artifacts/train_engineered.csv"},
        "params": {"feature_config": "HOUSE_PRICES_FEATURE_CONFIG_V2"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "engineer_features",
        "inputs": {"data": f"{_P}/datasets/test.csv"},
        "outputs": {"data": f"{_P}/artifacts/test_engineered.csv"},
        "params": {"feature_config": "HOUSE_PRICES_FEATURE_CONFIG_V2"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    # --- Outlier removal (train only) ---
    {
        "service": "remove_outliers",
        "inputs": {"data": f"{_P}/artifacts/train_engineered.csv"},
        "outputs": {"data": f"{_P}/artifacts/train_cleaned.csv"},
        "params": {"conditions": [
            {"column": "GrLivArea", "op": ">", "value": 4000},
            {"column": "SalePrice", "op": "<", "value": 300000},
        ]},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    # --- Column filter (fit on train, transform both) ---
    {
        "service": "fit_column_filter",
        "inputs": {"data": f"{_P}/artifacts/train_cleaned.csv"},
        "outputs": {"artifact": f"{_P}/artifacts/columns_to_keep.json"},
        "params": {"missing_threshold": 0.4, "exclude_columns": ["SalePrice", "Id"]},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "transform_column_filter",
        "inputs": {"data": f"{_P}/artifacts/train_cleaned.csv", "artifact": f"{_P}/artifacts/columns_to_keep.json"},
        "outputs": {"data": f"{_P}/artifacts/train_filtered.csv"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "transform_column_filter",
        "inputs": {"data": f"{_P}/artifacts/test_engineered.csv", "artifact": f"{_P}/artifacts/columns_to_keep.json"},
        "outputs": {"data": f"{_P}/artifacts/test_filtered.csv"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    # --- Imputation (fit on train, transform both) ---
    {
        "service": "fit_imputer",
        "inputs": {"data": f"{_P}/artifacts/train_filtered.csv"},
        "outputs": {"artifact": f"{_P}/artifacts/imputer.pkl"},
        "params": {"exclude_columns": ["SalePrice", "Id"]},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "transform_imputer",
        "inputs": {"data": f"{_P}/artifacts/train_filtered.csv", "artifact": f"{_P}/artifacts/imputer.pkl"},
        "outputs": {"data": f"{_P}/artifacts/train_imputed.csv"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "transform_imputer",
        "inputs": {"data": f"{_P}/artifacts/test_filtered.csv", "artifact": f"{_P}/artifacts/imputer.pkl"},
        "outputs": {"data": f"{_P}/artifacts/test_imputed.csv"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    # --- Skew correction (fit on train, transform both) ---
    {
        "service": "fit_skew_corrector",
        "inputs": {"data": f"{_P}/artifacts/train_imputed.csv"},
        "outputs": {"artifact": f"{_P}/artifacts/skew_corrector.json"},
        "params": {"skew_threshold": 0.75, "exclude_columns": ["SalePrice", "Id"]},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "transform_skew_corrector",
        "inputs": {"data": f"{_P}/artifacts/train_imputed.csv", "artifact": f"{_P}/artifacts/skew_corrector.json"},
        "outputs": {"data": f"{_P}/artifacts/train_skew_fixed.csv"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "transform_skew_corrector",
        "inputs": {"data": f"{_P}/artifacts/test_imputed.csv", "artifact": f"{_P}/artifacts/skew_corrector.json"},
        "outputs": {"data": f"{_P}/artifacts/test_skew_fixed.csv"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    # --- Encoding (fit on train, transform both) ---
    {
        "service": "fit_encoder",
        "inputs": {"data": f"{_P}/artifacts/train_skew_fixed.csv"},
        "outputs": {"artifact": f"{_P}/artifacts/encoder.pkl"},
        "params": {"exclude_columns": ["SalePrice", "Id"], "max_categories": 30},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "transform_encoder",
        "inputs": {"data": f"{_P}/artifacts/train_skew_fixed.csv", "artifact": f"{_P}/artifacts/encoder.pkl"},
        "outputs": {"data": f"{_P}/artifacts/train_encoded.csv"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "transform_encoder",
        "inputs": {"data": f"{_P}/artifacts/test_skew_fixed.csv", "artifact": f"{_P}/artifacts/encoder.pkl"},
        "outputs": {"data": f"{_P}/artifacts/test_encoded.csv"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    # --- Split ---
    {
        "service": "split_data",
        "inputs": {"data": f"{_P}/artifacts/train_encoded.csv"},
        "outputs": {"train_data": f"{_P}/artifacts/train_split.csv", "valid_data": f"{_P}/artifacts/valid_split.csv"},
        "params": {"test_size": 0.2, "random_state": 42},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    # --- Train stacked regressor (generic, 7-model stacking) ---
    {
        "service": "train_stacked_regressor",
        "inputs": {"train_data": f"{_P}/artifacts/train_split.csv", "valid_data": f"{_P}/artifacts/valid_split.csv"},
        "outputs": {"model": f"{_P}/artifacts/model.pkl", "metrics": f"{_P}/artifacts/metrics.json"},
        "params": {
            "label_column": "SalePrice",
            "id_column": "Id",
            "model_types": ["gradient_boosting", "lightgbm", "xgboost", "ridge", "elasticnet"],
            "n_folds": 5,
            "log_target": True,
            "random_state": 42,
            "gbr_n_estimators": 800,
            "gbr_learning_rate": 0.02,
            "gbr_max_depth": 3,
            "gbr_min_samples_leaf": 15,
            "gbr_subsample": 0.8,
            "lgbm_n_estimators": 1000,
            "lgbm_learning_rate": 0.02,
            "lgbm_num_leaves": 31,
            "lgbm_max_depth": -1,
            "lgbm_min_child_samples": 20,
            "lgbm_subsample": 0.7,
            "lgbm_colsample_bytree": 0.7,
            "lgbm_reg_alpha": 0.1,
            "lgbm_reg_lambda": 0.1,
            "xgb_n_estimators": 1000,
            "xgb_learning_rate": 0.02,
            "xgb_max_depth": 3,
            "xgb_min_child_weight": 3,
            "xgb_subsample": 0.7,
            "xgb_colsample_bytree": 0.7,
            "xgb_reg_alpha": 0.1,
            "xgb_reg_lambda": 1.0,
            "ridge_alpha": 10.0,
            "enet_alpha": 0.005,
            "enet_l1_ratio": 0.5,
            "lasso_alpha": 0.0005,
            "meta_alpha": 1.0,
        },
        "module": "house_prices_advanced_regression_techniques_services",
    },
    # --- Predict + submission ---
    {
        "service": "predict_stacked_regressor",
        "inputs": {"model": f"{_P}/artifacts/model.pkl", "data": f"{_P}/artifacts/test_encoded.csv"},
        "outputs": {"predictions": f"{_P}/artifacts/predictions.csv"},
        "params": {"id_column": "Id", "prediction_column": "SalePrice"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
    {
        "service": "create_submission",
        "inputs": {"predictions": f"{_P}/artifacts/predictions.csv"},
        "outputs": {"submission": f"{_P}/submission.csv"},
        "params": {"id_column": "Id", "prediction_column": "SalePrice"},
        "module": "house_prices_advanced_regression_techniques_services",
    },
]


# =============================================================================
# SERVICE REGISTRY (all imported from generic modules, zero competition-specific code)
# =============================================================================

SERVICE_REGISTRY = {
    # Generic preprocessing (from preprocessing_services)
    "fit_imputer": fit_imputer,
    "fit_encoder": fit_encoder,
    "fit_column_filter": fit_column_filter,
    "fit_skew_corrector": fit_skew_corrector,
    "transform_imputer": transform_imputer,
    "transform_encoder": transform_encoder,
    "transform_column_filter": transform_column_filter,
    "transform_skew_corrector": transform_skew_corrector,
    "remove_outliers": remove_outliers,
    "split_data": split_data,
    "create_submission": create_submission,
    "engineer_features": engineer_features,
    # Generic regression (from regression_services)
    "train_stacked_regressor": train_stacked_regressor,
    "predict_stacked_regressor": predict_stacked_regressor,
}


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run_pipeline(base_path: str, verbose: bool = True) -> Dict:
    """Run the pipeline end-to-end."""
    results = {"success": True, "steps_completed": 0, "outputs": {}, "errors": []}

    # Config lookup for feature_config references
    config_lookup = {
        "HOUSE_PRICES_FEATURE_CONFIG": HOUSE_PRICES_FEATURE_CONFIG,
        "HOUSE_PRICES_FEATURE_CONFIG_V2": HOUSE_PRICES_FEATURE_CONFIG_V2,
    }

    for i, step in enumerate(PIPELINE_SPEC, 1):
        service_name = step["service"]
        try:
            resolved_inputs = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
            resolved_outputs = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

            # Resolve config references in params
            params = {}
            for k, v in step.get("params", {}).items():
                if isinstance(v, str) and v in config_lookup:
                    params[k] = config_lookup[v]
                else:
                    params[k] = v

            service_fn = SERVICE_REGISTRY.get(service_name)
            if not service_fn:
                raise ValueError(f"Service '{service_name}' not found")

            if verbose:
                print(f"[{i}/{len(PIPELINE_SPEC)}] {service_name}...", end=" ", flush=True)

            result = service_fn(inputs=resolved_inputs, outputs=resolved_outputs, **params)

            if verbose:
                print(f"OK - {result}")

            results["steps_completed"] += 1
            results["outputs"][service_name] = resolved_outputs

        except Exception as e:
            results["errors"].append(f"Step {i} ({service_name}) failed: {e}")
            results["success"] = False
            if verbose:
                print(f"FAILED - {e}")
            break

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="House Prices Pipeline")
    parser.add_argument("--base-path", default="storage")
    args = parser.parse_args()

    print(f"\nRunning House Prices Pipeline from {args.base_path}\n")
    result = run_pipeline(args.base_path)

    if result["success"]:
        print(f"\nPipeline completed: {result['steps_completed']} steps")
    else:
        print(f"\nPipeline failed")
        for err in result["errors"]:
            print(f"  {err}")
