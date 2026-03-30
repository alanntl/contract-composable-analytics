"""
Tabular Playground Series Jan 2022 - Contract-Composable Analytics Services (v3.0)
==========================================================

Competition: https://www.kaggle.com/competitions/tabular-playground-series-jan-2022
Problem Type: Regression (Time Series Retail Sales)
Target: num_sold (number of products sold)
Metric: SMAPE (Symmetric Mean Absolute Percentage Error)

Best Score: 5.65858 (SMAPE) - Rank ~567/1592 (Top 35.6%)

Insights from Top Solutions:
- Automated ensembling of multiple model predictions (Solution 01, 02, 03)
- Row-ID based time trend adjustment: num_sold * exp((row_id - 27200) / 2000000) (Solution 01)
- Data has time series structure: date, country, store, product
- Log transform helps with SMAPE metric
- CRITICAL: Do NOT use year as feature - causes time leakage (train=2015-2018, test=2019)
- Use only cyclical temporal features that generalize across years

Competition-specific services:
- prepare_train_test_features_v2: Extract cyclical date features + encode categoricals + interactions (no year)
- train_lightgbm_cv: Train LightGBM with 5-fold CV for robust predictions
- predict_with_adjustment: Apply the time trend adjustment from solution 01
"""

import os
import sys
import json
import hashlib
import inspect
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Setup path for internal imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# Import shared I/O utilities
from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# FEATURE ENGINEERING SERVICES (IMPROVED)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "train_features": {"format": "csv"},
        "test_features": {"format": "csv"},
    },
    description="Prepare combined features for train and test sets (improved)",
    tags=["preprocessing", "tps-jan-2022"],
    version="2.0.0"
)
def prepare_train_test_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "num_sold",
    id_column: str = "row_id",
    use_log_target: bool = True,
) -> str:
    """
    Prepare features for both train and test sets consistently (IMPROVED).

    Improvements:
    - Log transform for target (better for SMAPE)
    - Interaction features between categoricals
    - More temporal features
    - Lag features for time series

    Parameters:
        target_column: Target column name (default: "num_sold")
        id_column: ID column name (default: "row_id")
        use_log_target: Apply log transform to target (default: True)
    """
    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    # Mark train/test and combine
    train['is_train'] = 1
    test['is_train'] = 0
    if target_column not in test.columns:
        test[target_column] = np.nan

    combined = pd.concat([train, test], ignore_index=True)

    # Parse date
    combined['date'] = pd.to_datetime(combined['date'])
    dt = combined['date']

    # Basic temporal features
    combined['year'] = dt.dt.year.astype(np.int16)
    combined['month'] = dt.dt.month.astype(np.int8)
    combined['day'] = dt.dt.day.astype(np.int8)
    combined['dayofweek'] = dt.dt.dayofweek.astype(np.int8)
    combined['dayofyear'] = dt.dt.dayofyear.astype(np.int16)
    combined['weekofyear'] = dt.dt.isocalendar().week.astype(np.int8)
    combined['is_weekend'] = (dt.dt.dayofweek >= 5).astype(np.int8)
    combined['is_month_start'] = dt.dt.is_month_start.astype(np.int8)
    combined['is_month_end'] = dt.dt.is_month_end.astype(np.int8)
    combined['quarter'] = dt.dt.quarter.astype(np.int8)

    # Additional temporal features
    combined['days_in_month'] = dt.dt.days_in_month.astype(np.int8)
    combined['is_quarter_start'] = dt.dt.is_quarter_start.astype(np.int8)
    combined['is_quarter_end'] = dt.dt.is_quarter_end.astype(np.int8)
    combined['is_year_start'] = dt.dt.is_year_start.astype(np.int8)
    combined['is_year_end'] = dt.dt.is_year_end.astype(np.int8)

    # Cyclical encoding for periodic features
    combined['month_sin'] = np.sin(2 * np.pi * combined['month'] / 12).astype(np.float32)
    combined['month_cos'] = np.cos(2 * np.pi * combined['month'] / 12).astype(np.float32)
    combined['dayofweek_sin'] = np.sin(2 * np.pi * combined['dayofweek'] / 7).astype(np.float32)
    combined['dayofweek_cos'] = np.cos(2 * np.pi * combined['dayofweek'] / 7).astype(np.float32)
    combined['day_sin'] = np.sin(2 * np.pi * combined['day'] / 31).astype(np.float32)
    combined['day_cos'] = np.cos(2 * np.pi * combined['day'] / 31).astype(np.float32)

    # Encode categoricals
    cat_cols = ['country', 'store', 'product']
    for col in cat_cols:
        if col in combined.columns:
            codes, uniques = pd.factorize(combined[col])
            combined[f'{col}_encoded'] = codes.astype(np.int8)

    # Create interaction features
    combined['country_store'] = combined['country'].astype(str) + '_' + combined['store'].astype(str)
    combined['country_product'] = combined['country'].astype(str) + '_' + combined['product'].astype(str)
    combined['store_product'] = combined['store'].astype(str) + '_' + combined['product'].astype(str)
    combined['country_store_product'] = combined['country'].astype(str) + '_' + combined['store'].astype(str) + '_' + combined['product'].astype(str)

    # Encode interaction features
    for col in ['country_store', 'country_product', 'store_product', 'country_store_product']:
        codes, _ = pd.factorize(combined[col])
        combined[f'{col}_encoded'] = codes.astype(np.int16)
        combined = combined.drop(columns=[col])

    # Apply log transform to target if specified
    if use_log_target:
        train_mask = combined['is_train'] == 1
        combined.loc[train_mask, target_column] = np.log1p(combined.loc[train_mask, target_column])
        combined['log_target'] = 1
    else:
        combined['log_target'] = 0

    # Drop original categorical and date columns
    combined = combined.drop(columns=['date', 'country', 'store', 'product'], errors='ignore')

    # Split back
    train_out = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    test_out = combined[combined['is_train'] == 0].drop(columns=['is_train', target_column, 'log_target'])

    _save_data(train_out, outputs["train_features"])
    _save_data(test_out, outputs["test_features"])

    return f"prepare_train_test_features: train={len(train_out)}, test={len(test_out)} rows, log_target={use_log_target}"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Generate predictions and apply row-ID adjustment",
    tags=["inference", "tps-jan-2022"],
    version="2.0.0"
)
def predict_with_adjustment(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "row_id",
    target_column: str = "num_sold",
    apply_adjustment: bool = True,
    center_id: int = 27200,
    scale_factor: int = 2000000,
) -> str:
    """
    Generate predictions and optionally apply row-ID time trend adjustment.

    From Solution 01: The winning solutions discovered that predictions
    should be adjusted based on row_id to capture a time trend:

        adjusted = predicted * exp((row_id - center_id) / scale_factor)

    Parameters:
        id_column: ID column name (default: "row_id")
        target_column: Target column name (default: "num_sold")
        apply_adjustment: Whether to apply row-ID adjustment (default: True)
        center_id: Center point for adjustment (default: 27200)
        scale_factor: Scaling factor (default: 2000000)
    """
    # Load model
    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    # Handle both direct model and ensemble dict formats
    if isinstance(model_data, dict) and "models" in model_data:
        # Ensemble format
        models = model_data["models"]
        weights = model_data.get("weights", [1.0 / len(models)] * len(models))
        feature_cols = model_data.get("feature_cols", None)
        log_target = model_data.get("log_target", False)
    else:
        # Direct model
        models = {"single": model_data}
        weights = [1.0]
        feature_cols = None
        log_target = False

    # Load test data
    test = _load_data(inputs["test_data"])

    # Get row IDs
    if id_column in test.columns:
        ids = test[id_column].values
    else:
        ids = np.arange(len(test))

    # Determine feature columns
    if feature_cols:
        X = test[[c for c in feature_cols if c in test.columns]]
    else:
        X = test.drop(columns=[id_column], errors='ignore')

    # Make predictions
    predictions = np.zeros(len(X))
    for (model_name, model), weight in zip(models.items(), weights):
        predictions += weight * model.predict(X)

    # Reverse log transform if needed
    if log_target:
        predictions = np.expm1(predictions)

    # Apply row-ID adjustment
    if apply_adjustment:
        adjustment = np.exp((ids - center_id) / scale_factor)
        predictions = predictions * adjustment

    # Round and clip
    predictions = np.round(predictions).astype(np.int64)
    predictions = np.maximum(predictions, 0)

    # Create output dataframe
    result = pd.DataFrame({
        id_column: ids,
        target_column: predictions,
    })

    _save_data(result, outputs["predictions"])

    adj_str = "with row-ID adjustment" if apply_adjustment else "without adjustment"
    return f"predict_with_adjustment: {len(result)} predictions {adj_str}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "valid_data": {"format": "csv", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train LightGBM regressor optimized for SMAPE",
    tags=["modeling", "training", "lightgbm", "tps-jan-2022"],
    version="2.0.0"
)
def train_lightgbm_smape(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "num_sold",
    id_column: str = "row_id",
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    num_leaves: int = 63,
    max_depth: int = 10,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    random_state: int = 42,
) -> str:
    """
    Train LightGBM regressor with improved hyperparameters.

    Optimized for SMAPE metric with log-transformed target.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    train_df = _load_data(inputs["train_data"])
    valid_df = _load_data(inputs["valid_data"])

    # Check if log transform was applied
    log_target = bool('log_target' in train_df.columns and train_df['log_target'].iloc[0] == 1)

    # Exclude non-feature columns
    exclude_cols = [label_column, id_column, 'log_target']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[label_column]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[label_column]

    # Train model
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
    )

    # Evaluate
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    # If log transform, reverse for actual metric calculation
    if log_target:
        y_train_orig = np.expm1(y_train)
        y_valid_orig = np.expm1(y_valid)
        train_pred_orig = np.expm1(train_pred)
        valid_pred_orig = np.expm1(valid_pred)
    else:
        y_train_orig = y_train
        y_valid_orig = y_valid
        train_pred_orig = train_pred
        valid_pred_orig = valid_pred

    # Calculate SMAPE
    def smape(y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred))
        diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
        return 200 * np.mean(diff)

    train_smape = smape(y_train_orig, train_pred_orig)
    valid_smape = smape(y_valid_orig, valid_pred_orig)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    metrics = {
        "model_type": "LightGBM",
        "n_samples_train": len(X_train),
        "n_samples_valid": len(X_valid),
        "n_features": len(feature_cols),
        "log_target": log_target,
        "train_rmse": float(train_rmse),
        "valid_rmse": float(valid_rmse),
        "train_smape": float(train_smape),
        "valid_smape": float(valid_smape),
        "hyperparameters": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
        }
    }

    # Save model
    model_data = {
        "models": {"lightgbm": model},
        "weights": [1.0],
        "feature_cols": feature_cols,
        "log_target": log_target,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_lightgbm_smape: SMAPE={valid_smape:.4f}, RMSE={valid_rmse:.4f}"


# =============================================================================
# V3 IMPROVED SERVICES (NO YEAR FEATURE - AVOIDS TIME LEAKAGE)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "train_features": {"format": "csv"},
        "test_features": {"format": "csv"},
    },
    description="Prepare features WITHOUT year (avoids time leakage)",
    tags=["preprocessing", "tps-jan-2022", "v3"],
    version="3.0.0"
)
def prepare_train_test_features_v2(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "num_sold",
    id_column: str = "row_id",
    use_log_target: bool = True,
) -> str:
    """
    Prepare features for train/test WITHOUT year feature to avoid time leakage.

    Train data is from 2015-2018, test data is from 2019.
    Including year as a feature causes the model to fail on test data.
    """
    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    # Mark train/test and combine
    train['is_train'] = 1
    test['is_train'] = 0
    if target_column not in test.columns:
        test[target_column] = np.nan

    combined = pd.concat([train, test], ignore_index=True)

    # Parse date
    combined['date'] = pd.to_datetime(combined['date'])
    dt = combined['date']

    # Temporal features that generalize across years (NO YEAR!)
    combined['month'] = dt.dt.month.astype(np.int8)
    combined['day'] = dt.dt.day.astype(np.int8)
    combined['dayofweek'] = dt.dt.dayofweek.astype(np.int8)
    combined['dayofyear'] = dt.dt.dayofyear.astype(np.int16)
    combined['weekofyear'] = dt.dt.isocalendar().week.astype(np.int8)
    combined['quarter'] = dt.dt.quarter.astype(np.int8)

    # Binary temporal features
    combined['is_weekend'] = (dt.dt.dayofweek >= 5).astype(np.int8)
    combined['is_month_start'] = dt.dt.is_month_start.astype(np.int8)
    combined['is_month_end'] = dt.dt.is_month_end.astype(np.int8)

    # Cyclical encoding for periodic features
    combined['month_sin'] = np.sin(2 * np.pi * combined['month'] / 12).astype(np.float32)
    combined['month_cos'] = np.cos(2 * np.pi * combined['month'] / 12).astype(np.float32)
    combined['day_sin'] = np.sin(2 * np.pi * combined['day'] / 31).astype(np.float32)
    combined['day_cos'] = np.cos(2 * np.pi * combined['day'] / 31).astype(np.float32)
    combined['dayofweek_sin'] = np.sin(2 * np.pi * combined['dayofweek'] / 7).astype(np.float32)
    combined['dayofweek_cos'] = np.cos(2 * np.pi * combined['dayofweek'] / 7).astype(np.float32)

    # Encode categoricals
    for col in ['country', 'store', 'product']:
        codes, _ = pd.factorize(combined[col])
        combined[f'{col}_encoded'] = codes.astype(np.int8)

    # Create interaction features
    combined['country_store'] = combined['country'].astype(str) + '_' + combined['store'].astype(str)
    combined['country_product'] = combined['country'].astype(str) + '_' + combined['product'].astype(str)
    combined['store_product'] = combined['store'].astype(str) + '_' + combined['product'].astype(str)
    combined['csp'] = combined['country'].astype(str) + '_' + combined['store'].astype(str) + '_' + combined['product'].astype(str)

    # Encode interaction features
    for col in ['country_store', 'country_product', 'store_product', 'csp']:
        codes, _ = pd.factorize(combined[col])
        combined[f'{col}_encoded'] = codes.astype(np.int16)
        combined = combined.drop(columns=[col])

    # Apply log transform to target if specified
    if use_log_target:
        train_mask = combined['is_train'] == 1
        combined.loc[train_mask, target_column] = np.log1p(combined.loc[train_mask, target_column])
        combined['log_target'] = 1
    else:
        combined['log_target'] = 0

    # Drop original categorical and date columns
    combined = combined.drop(columns=['date', 'country', 'store', 'product'], errors='ignore')

    # Split back
    train_out = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    test_out = combined[combined['is_train'] == 0].drop(columns=['is_train', target_column, 'log_target'])

    _save_data(train_out, outputs["train_features"])
    _save_data(test_out, outputs["test_features"])

    return f"prepare_train_test_features_v2: train={len(train_out)}, test={len(test_out)}, no year feature"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "predictions": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Train LightGBM with 5-fold CV and generate predictions",
    tags=["modeling", "training", "lightgbm", "cv", "tps-jan-2022"],
    version="3.0.0"
)
def train_lightgbm_cv(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "num_sold",
    id_column: str = "row_id",
    n_folds: int = 5,
    n_estimators: int = 800,
    learning_rate: float = 0.03,
    num_leaves: int = 63,
    max_depth: int = 10,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    apply_adjustment: bool = True,
    center_id: int = 27200,
    scale_factor: int = 2000000,
    random_state: int = 42,
) -> str:
    """
    Train LightGBM with K-fold CV and generate averaged predictions.

    Includes row-ID adjustment from winning solutions.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("LightGBM not installed. pip install lightgbm")

    from sklearn.model_selection import KFold

    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    # Check if log transform was applied
    log_target = bool('log_target' in train_df.columns and train_df['log_target'].iloc[0] == 1)

    # Exclude non-feature columns
    exclude_cols = [label_column, id_column, 'log_target']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df[label_column].values
    X_test = test_df[feature_cols].values
    test_ids = test_df[id_column].values

    # SMAPE calculation
    def smape(y_true, y_pred):
        denom = np.abs(y_true) + np.abs(y_pred)
        diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
        return 200 * np.mean(diff)

    # Train with CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    test_preds = np.zeros(len(X_test))
    fold_smapes = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

        val_pred = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds

        # Calculate fold SMAPE on original scale
        if log_target:
            val_smape = smape(np.expm1(y_val), np.expm1(val_pred))
        else:
            val_smape = smape(y_val, val_pred)
        fold_smapes.append(val_smape)
        models.append(model)

    # Final predictions
    final_preds = np.expm1(test_preds) if log_target else test_preds

    # Apply row-ID adjustment
    if apply_adjustment:
        adjustment = np.exp((test_ids - center_id) / scale_factor)
        final_preds = final_preds * adjustment

    # Round and clip
    final_preds = np.round(final_preds).astype(np.int64)
    final_preds = np.maximum(final_preds, 0)

    # Save model
    model_data = {
        "models": {f"fold_{i}": m for i, m in enumerate(models)},
        "weights": [1.0 / n_folds] * n_folds,
        "feature_cols": feature_cols,
        "log_target": log_target,
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    # Save predictions
    result = pd.DataFrame({id_column: test_ids, label_column: final_preds})
    _save_data(result, outputs["predictions"])

    # Save metrics
    metrics = {
        "model_type": "LightGBM_CV",
        "n_folds": n_folds,
        "fold_smapes": fold_smapes,
        "mean_cv_smape": float(np.mean(fold_smapes)),
        "std_cv_smape": float(np.std(fold_smapes)),
        "n_features": len(feature_cols),
        "log_target": log_target,
        "row_id_adjustment": apply_adjustment,
    }
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_lightgbm_cv: CV SMAPE={np.mean(fold_smapes):.4f} ± {np.std(fold_smapes):.4f}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "prepare_train_test_features": prepare_train_test_features,
    "prepare_train_test_features_v2": prepare_train_test_features_v2,
    "predict_with_adjustment": predict_with_adjustment,
    "train_lightgbm_smape": train_lightgbm_smape,
    "train_lightgbm_cv": train_lightgbm_cv,
}
