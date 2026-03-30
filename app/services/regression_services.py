"""
Regression Model Services - SLEGO Common Module
=================================================

Generic, reusable regression model training and prediction services.
Extracted and generalized from M5 forecasting and House Prices services.

These services are competition-agnostic and can be plugged into any
regression pipeline via PIPELINE_SPEC configuration.

Services:
  Training (individual models):
    - train_random_forest: sklearn RandomForestRegressor
    - train_gradient_boosting: sklearn GradientBoostingRegressor
    - train_lightgbm_regressor: LightGBM LGBMRegressor (lazy import)
    - train_xgboost_regressor: XGBoost XGBRegressor (lazy import)
  Training (ensemble):
    - train_ensemble_regressor: Blend multiple regressors with configurable weights
  Prediction:
    - predict_regressor: Generic prediction from any pickled regressor
    - predict_ensemble_regressor: Predict with ensemble model dict (models + weights)

All services follow SLEGO design principles:
- G1: Each service does exactly ONE thing
- G2: Explicit I/O contracts via @contract
- G3: Pure functions with explicit random_state
- G4: No hardcoded column names (injected via params)
- G5: DAG pipeline structure
- G6: Semantic metadata via tags/description

Version: 1.0.0
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slego_contract import contract

# =============================================================================
# HELPERS: Import from shared io_utils
# =============================================================================
from services.io_utils import load_data as _load_data, save_data as _save_data


def _evaluate_regression(model, X_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[str, float]:
    """
    Evaluate a regression model on validation data.

    Parameters
    ----------
    model : object
        Fitted model with a .predict() method.
    X_valid : pd.DataFrame
        Validation features.
    y_valid : pd.Series
        Validation target.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'valid_rmse' and 'valid_mae'.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    preds = model.predict(X_valid)
    return {
        "valid_rmse": float(np.sqrt(mean_squared_error(y_valid, preds))),
        "valid_mae": float(mean_absolute_error(y_valid, preds)),
    }


def _save_model_metrics_importance(
    model: Any,
    metrics: Dict[str, Any],
    feature_cols: List[str],
    outputs: Dict[str, str],
) -> None:
    """
    Save model pickle, metrics JSON, and feature importance CSV.

    Parameters
    ----------
    model : object
        Fitted model object.
    metrics : dict
        Metrics dictionary to save as JSON.
    feature_cols : list of str
        Feature column names (for importance mapping).
    outputs : dict
        Output paths dict. Expected keys:
        - 'model': path for pickle file
        - 'metrics': path for JSON file
        - 'feature_importance' (optional): path for CSV file
    """
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)

    with open(outputs["model"], "wb") as f:
        pickle.dump(model, f)

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    if "feature_importance" in outputs and hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(outputs["feature_importance"], index=False)


# =============================================================================
# SERVICE 1: RANDOM FOREST REGRESSOR
# =============================================================================

@contract(
    inputs={
        "train_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1},
        },
        "valid_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False},
        },
    },
    outputs={
        "model": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "sklearn_model"},
        },
        "metrics": {
            "format": "json",
            "schema": {
                "type": "json",
                "required_fields": ["model_type", "valid_rmse"],
                "fields": {
                    "model_type": "str",
                    "valid_rmse": "float",
                    "valid_mae": "float",
                },
            },
        },
        "feature_importance": {
            "format": "csv",
            "schema": {"type": "tabular", "required_columns": ["feature", "importance"]},
        },
    },
    description="Train a Random Forest Regressor with configurable hyperparameters",
    tags=["modeling", "training", "random-forest", "regression", "generic"],
    version="1.0.0",
)
def train_random_forest(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    feature_exclude: Optional[List[str]] = None,
    random_state: int = 42,
) -> str:
    """
    Train a Random Forest Regressor.

    Args:
        label_column: Target column name
        n_estimators: Number of trees in the forest (default: 100)
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_split: Minimum samples required to split a node (default: 2)
        min_samples_leaf: Minimum samples in a leaf node (default: 1)
        max_features: Features considered per split ("sqrt", "log2", or float 0-1)
        feature_exclude: Column names to exclude from features
        random_state: Random seed for reproducibility
    """
    from sklearn.ensemble import RandomForestRegressor

    train = _load_data(inputs["train_data"])
    valid = _load_data(inputs["valid_data"])

    feature_exclude = feature_exclude or []
    exclude_cols = set(feature_exclude + [label_column])
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = {
        "model_type": "RandomForestRegressor",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
    }
    metrics.update(_evaluate_regression(model, X_valid, y_valid))

    _save_model_metrics_importance(model, metrics, feature_cols, outputs)

    return (
        f"train_random_forest: {len(X_train)} samples, "
        f"{len(feature_cols)} features, RMSE={metrics['valid_rmse']:.4f}"
    )


# =============================================================================
# SERVICE 2: GRADIENT BOOSTING REGRESSOR
# =============================================================================

@contract(
    inputs={
        "train_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1},
        },
        "valid_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False},
        },
    },
    outputs={
        "model": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "sklearn_model"},
        },
        "metrics": {
            "format": "json",
            "schema": {
                "type": "json",
                "required_fields": ["model_type", "valid_rmse"],
                "fields": {
                    "model_type": "str",
                    "valid_rmse": "float",
                    "valid_mae": "float",
                },
            },
        },
        "feature_importance": {
            "format": "csv",
            "schema": {"type": "tabular", "required_columns": ["feature", "importance"]},
        },
    },
    description="Train a Gradient Boosting Regressor with configurable hyperparameters",
    tags=["modeling", "training", "gradient-boosting", "regression", "generic"],
    version="1.0.0",
)
def train_gradient_boosting(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    feature_exclude: Optional[List[str]] = None,
    random_state: int = 42,
) -> str:
    """
    Train a Gradient Boosting Regressor (sklearn).

    Args:
        label_column: Target column name
        n_estimators: Number of boosting stages (default: 100)
        learning_rate: Shrinkage rate applied to each tree (default: 0.1)
        max_depth: Maximum tree depth (default: 3)
        min_samples_split: Minimum samples to split a node (default: 2)
        min_samples_leaf: Minimum samples in a leaf node (default: 1)
        subsample: Fraction of samples used per tree (default: 1.0, no subsampling)
        feature_exclude: Column names to exclude from features
        random_state: Random seed for reproducibility
    """
    from sklearn.ensemble import GradientBoostingRegressor

    train = _load_data(inputs["train_data"])
    valid = _load_data(inputs["valid_data"])

    feature_exclude = feature_exclude or []
    exclude_cols = set(feature_exclude + [label_column])
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    metrics = {
        "model_type": "GradientBoostingRegressor",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
    }
    metrics.update(_evaluate_regression(model, X_valid, y_valid))

    _save_model_metrics_importance(model, metrics, feature_cols, outputs)

    return (
        f"train_gradient_boosting: {len(X_train)} samples, "
        f"lr={learning_rate}, RMSE={metrics['valid_rmse']:.4f}"
    )


# =============================================================================
# SERVICE 3: LIGHTGBM REGRESSOR (lazy import)
# =============================================================================

@contract(
    inputs={
        "train_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1},
        },
        "valid_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False},
        },
    },
    outputs={
        "model": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "lightgbm_model"},
        },
        "metrics": {
            "format": "json",
            "schema": {
                "type": "json",
                "required_fields": ["model_type", "valid_rmse"],
                "fields": {
                    "model_type": "str",
                    "valid_rmse": "float",
                    "valid_mae": "float",
                },
            },
        },
        "feature_importance": {
            "format": "csv",
            "schema": {"type": "tabular", "required_columns": ["feature", "importance"]},
        },
    },
    description="Train a LightGBM Regressor with configurable hyperparameters",
    tags=["modeling", "training", "lightgbm", "regression", "generic"],
    version="1.0.0",
)
def train_lightgbm_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: Optional[str] = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    subsample_freq: int = 0,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    objective: str = "regression",
    tweedie_variance_power: float = 1.5,
    max_bin: int = 255,
    early_stopping_rounds: Optional[int] = None,
    feature_exclude: Optional[List[str]] = None,
    random_state: int = 42,
) -> str:
    """
    Train a LightGBM Regressor.

    LightGBM is lazily imported; install with: pip install lightgbm

    Args:
        label_column: Target column name
        id_column: ID column to exclude from features (standard param per G1-G6)
        n_estimators: Number of boosting iterations (default: 100)
        learning_rate: Shrinkage rate (default: 0.1)
        num_leaves: Maximum number of leaves per tree (default: 31)
        max_depth: Maximum tree depth (-1 = no limit)
        min_child_samples: Minimum data points in a leaf (default: 20)
        subsample: Row sampling ratio per iteration (default: 0.8)
        subsample_freq: Frequency for bagging (0 = disabled)
        colsample_bytree: Column sampling ratio per tree (default: 0.8)
        reg_alpha: L1 regularization term (default: 0.0)
        reg_lambda: L2 regularization term (default: 0.0)
        objective: Loss function ("regression", "poisson", "mae", "tweedie", etc.)
        tweedie_variance_power: Power for Tweedie objective (1.0-2.0, default: 1.5)
        max_bin: Max bins for histogram (default: 255)
        early_stopping_rounds: Stop if no improvement after N rounds (default: None = disabled)
        feature_exclude: Column names to exclude from features
        random_state: Random seed for reproducibility
    """
    try:
        from lightgbm import LGBMRegressor, early_stopping
    except ImportError:
        raise ImportError(
            "LightGBM is not installed. Install it with: pip install lightgbm"
        )

    train = _load_data(inputs["train_data"])
    valid = _load_data(inputs["valid_data"])

    feature_exclude = feature_exclude or []
    # Add id_column to exclusions if provided (standard param per G1-G6)
    if id_column and id_column in train.columns:
        feature_exclude = list(feature_exclude) + [id_column]
    exclude_cols = set(feature_exclude + [label_column])
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    # Build model params
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "subsample_freq": subsample_freq,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "objective": objective,
        "max_bin": max_bin,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    # Add Tweedie power if using tweedie objective
    if objective == "tweedie":
        model_params["tweedie_variance_power"] = tweedie_variance_power

    model = LGBMRegressor(**model_params)

    # Setup callbacks for early stopping if specified
    fit_params = {"eval_set": [(X_valid, y_valid)]}
    if early_stopping_rounds is not None:
        fit_params["callbacks"] = [early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]

    model.fit(X_train, y_train, **fit_params)

    metrics = {
        "model_type": "LGBMRegressor",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "objective": objective,
    }
    metrics.update(_evaluate_regression(model, X_valid, y_valid))

    if hasattr(model, "best_iteration_"):
        metrics["best_iteration"] = int(model.best_iteration_)

    _save_model_metrics_importance(model, metrics, feature_cols, outputs)

    return (
        f"train_lightgbm_regressor: {len(X_train)} samples, "
        f"lr={learning_rate}, leaves={num_leaves}, RMSE={metrics['valid_rmse']:.4f}"
    )


# =============================================================================
# SERVICE 4: XGBOOST REGRESSOR (lazy import)
# =============================================================================

@contract(
    inputs={
        "train_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1},
        },
        "valid_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False},
        },
    },
    outputs={
        "model": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "xgboost_model"},
        },
        "metrics": {
            "format": "json",
            "schema": {
                "type": "json",
                "required_fields": ["model_type", "valid_rmse"],
                "fields": {
                    "model_type": "str",
                    "valid_rmse": "float",
                    "valid_mae": "float",
                },
            },
        },
        "feature_importance": {
            "format": "csv",
            "schema": {"type": "tabular", "required_columns": ["feature", "importance"]},
        },
    },
    description="Train an XGBoost Regressor with configurable hyperparameters",
    tags=["modeling", "training", "xgboost", "regression", "generic"],
    version="1.0.0",
)
def train_xgboost_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    objective: str = "reg:squarederror",
    early_stopping_rounds: Optional[int] = None,
    feature_exclude: Optional[List[str]] = None,
    random_state: int = 42,
) -> str:
    """
    Train an XGBoost Regressor.

    XGBoost is lazily imported; install with: pip install xgboost

    Args:
        label_column: Target column name
        n_estimators: Number of boosting rounds (default: 100)
        learning_rate: Step size shrinkage (default: 0.1)
        max_depth: Maximum tree depth (default: 6)
        min_child_weight: Minimum sum of instance weight in a child (default: 1)
        subsample: Row sampling ratio per tree (default: 0.8)
        colsample_bytree: Column sampling ratio per tree (default: 0.8)
        reg_alpha: L1 regularization on weights (default: 0.0)
        reg_lambda: L2 regularization on weights (default: 1.0)
        objective: Loss function ("reg:squarederror", "count:poisson", etc.)
        early_stopping_rounds: Stop if no improvement after N rounds (default: None)
        feature_exclude: Column names to exclude from features
        random_state: Random seed for reproducibility
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        )

    train = _load_data(inputs["train_data"])
    valid = _load_data(inputs["valid_data"])

    feature_exclude = feature_exclude or []
    exclude_cols = set(feature_exclude + [label_column])
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    model_kwargs = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective=objective,
        random_state=random_state,
        n_jobs=-1,
    )
    if early_stopping_rounds is not None:
        model_kwargs["early_stopping_rounds"] = early_stopping_rounds

    model = XGBRegressor(**model_kwargs)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    metrics = {
        "model_type": "XGBRegressor",
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "objective": objective,
    }
    metrics.update(_evaluate_regression(model, X_valid, y_valid))

    if hasattr(model, "best_iteration"):
        metrics["best_iteration"] = int(model.best_iteration)

    _save_model_metrics_importance(model, metrics, feature_cols, outputs)

    return (
        f"train_xgboost_regressor: {len(X_train)} samples, "
        f"lr={learning_rate}, depth={max_depth}, RMSE={metrics['valid_rmse']:.4f}"
    )


# =============================================================================
# SERVICE 5: ENSEMBLE REGRESSOR
# =============================================================================

@contract(
    inputs={
        "train_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1},
        },
        "valid_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False},
        },
    },
    outputs={
        "model": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "ensemble_model"},
        },
        "metrics": {
            "format": "json",
            "schema": {
                "type": "json",
                "required_fields": ["model_type", "valid_rmse"],
                "fields": {
                    "model_type": "str",
                    "valid_rmse": "float",
                    "valid_mae": "float",
                },
            },
        },
        "feature_importance": {
            "format": "csv",
            "schema": {"type": "tabular", "required_columns": ["feature", "importance"]},
        },
    },
    description="Train an ensemble of multiple regressors and blend predictions with configurable weights",
    tags=["modeling", "training", "ensemble", "regression", "generic"],
    version="1.0.0",
)
def train_ensemble_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = "Id",
    model_types: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    log_target: bool = False,
    random_state: int = 42,
    # --- Gradient Boosting per-model params ---
    gbr_n_estimators: int = 100,
    gbr_learning_rate: float = 0.1,
    gbr_max_depth: int = 3,
    gbr_min_samples_split: int = 2,
    gbr_min_samples_leaf: int = 1,
    gbr_subsample: float = 1.0,
    # --- LightGBM per-model params ---
    lgbm_n_estimators: int = 100,
    lgbm_learning_rate: float = 0.1,
    lgbm_num_leaves: int = 31,
    lgbm_max_depth: int = -1,
    lgbm_min_child_samples: int = 20,
    lgbm_subsample: float = 0.8,
    lgbm_colsample_bytree: float = 0.8,
    lgbm_reg_alpha: float = 0.0,
    lgbm_reg_lambda: float = 0.0,
    # --- XGBoost per-model params ---
    xgb_n_estimators: int = 100,
    xgb_learning_rate: float = 0.1,
    xgb_max_depth: int = 6,
    xgb_min_child_weight: int = 1,
    xgb_subsample: float = 0.8,
    xgb_colsample_bytree: float = 0.8,
    xgb_reg_alpha: float = 0.0,
    xgb_reg_lambda: float = 1.0,
    # --- Random Forest per-model params ---
    rf_n_estimators: int = 100,
    rf_max_depth: Optional[int] = None,
    rf_min_samples_split: int = 2,
    rf_min_samples_leaf: int = 1,
    rf_max_features: str = "sqrt",
    # --- CatBoost per-model params ---
    cat_n_estimators: int = 100,
    cat_learning_rate: float = 0.1,
    cat_max_depth: int = 6,
    cat_l2_leaf_reg: float = 3.0,
    cat_subsample: float = 0.8,
) -> str:
    """
    Train an ensemble of multiple regressors and blend predictions.

    Supported model types: "gradient_boosting", "lightgbm", "xgboost", "random_forest", "catboost".
    Each model type has prefixed hyperparameters (e.g. gbr_n_estimators, lgbm_num_leaves).

    If log_target is True, the target is log1p-transformed before training and
    predictions are expm1-transformed back. This is common for targets like
    sale prices where RMSLE is the evaluation metric.

    Args:
        label_column: Target column name
        id_column: ID column to exclude from features
        model_types: List of model types to include (default: ["gradient_boosting", "lightgbm"])
        weights: Blending weights per model (must sum to ~1.0, default: [0.4, 0.6])
        log_target: Whether to apply log1p transform to target
        random_state: Random seed for reproducibility
        gbr_*: GradientBoostingRegressor parameters
        lgbm_*: LGBMRegressor parameters
        xgb_*: XGBRegressor parameters
        rf_*: RandomForestRegressor parameters
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    model_types = model_types or ["gradient_boosting", "lightgbm"]
    weights = weights or [0.4, 0.6]

    if len(weights) != len(model_types):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of model_types ({len(model_types)})"
        )

    train = _load_data(inputs["train_data"])
    valid = _load_data(inputs["valid_data"])

    # Determine feature columns
    exclude_cols = {label_column}
    if id_column and id_column in train.columns:
        exclude_cols.add(id_column)
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train_raw = train[label_column]
    X_valid = valid[feature_cols]
    y_valid_raw = valid[label_column]

    if log_target:
        y_train = np.log1p(y_train_raw)
        y_valid = np.log1p(y_valid_raw)
    else:
        y_train = y_train_raw
        y_valid = y_valid_raw

    # Train each model
    trained_models = {}
    individual_metrics = {}

    for model_type in model_types:
        if model_type == "gradient_boosting":
            mdl = GradientBoostingRegressor(
                n_estimators=gbr_n_estimators,
                learning_rate=gbr_learning_rate,
                max_depth=gbr_max_depth,
                min_samples_split=gbr_min_samples_split,
                min_samples_leaf=gbr_min_samples_leaf,
                subsample=gbr_subsample,
                random_state=random_state,
            )
            mdl.fit(X_train, y_train)

        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError:
                raise ImportError(
                    "LightGBM is not installed. Install it with: pip install lightgbm"
                )
            mdl = LGBMRegressor(
                n_estimators=lgbm_n_estimators,
                learning_rate=lgbm_learning_rate,
                num_leaves=lgbm_num_leaves,
                max_depth=lgbm_max_depth,
                min_child_samples=lgbm_min_child_samples,
                subsample=lgbm_subsample,
                colsample_bytree=lgbm_colsample_bytree,
                reg_alpha=lgbm_reg_alpha,
                reg_lambda=lgbm_reg_lambda,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
            mdl.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

        elif model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError:
                raise ImportError(
                    "XGBoost is not installed. Install it with: pip install xgboost"
                )
            mdl = XGBRegressor(
                n_estimators=xgb_n_estimators,
                learning_rate=xgb_learning_rate,
                max_depth=xgb_max_depth,
                min_child_weight=xgb_min_child_weight,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                reg_alpha=xgb_reg_alpha,
                reg_lambda=xgb_reg_lambda,
                random_state=random_state,
                n_jobs=-1,
            )
            mdl.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        elif model_type == "random_forest":
            mdl = RandomForestRegressor(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
                max_features=rf_max_features,
                random_state=random_state,
                n_jobs=-1,
            )
            mdl.fit(X_train, y_train)

        elif model_type == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError:
                raise ImportError(
                    "CatBoost is not installed. Install it with: pip install catboost"
                )
            mdl = CatBoostRegressor(
                n_estimators=cat_n_estimators,
                learning_rate=cat_learning_rate,
                max_depth=cat_max_depth,
                l2_leaf_reg=cat_l2_leaf_reg,
                subsample=cat_subsample,
                random_state=random_state,
                verbose=False,
            )
            mdl.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50, verbose=False)

        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported: gradient_boosting, lightgbm, xgboost, random_forest, catboost"
            )

        trained_models[model_type] = mdl

        # Individual validation RMSE (in transformed space if log_target)
        pred_i = mdl.predict(X_valid)
        rmse_i = float(np.sqrt(mean_squared_error(y_valid, pred_i)))
        individual_metrics[f"valid_rmse_{model_type}"] = rmse_i

    # Blended prediction
    blended_pred = np.zeros(len(X_valid))
    for (model_type, mdl), w in zip(trained_models.items(), weights):
        blended_pred += w * mdl.predict(X_valid)

    # Metrics on blended prediction
    blend_rmse = float(np.sqrt(mean_squared_error(y_valid, blended_pred)))
    blend_mae = float(mean_absolute_error(y_valid, blended_pred))

    metrics = {
        "model_type": "ensemble_regressor",
        "model_types": model_types,
        "weights": weights,
        "log_target": log_target,
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "valid_rmse": blend_rmse,
        "valid_mae": blend_mae,
    }
    metrics.update(individual_metrics)

    # If log_target, also compute RMSE on original scale
    if log_target:
        pred_original = np.expm1(blended_pred)
        pred_original = np.maximum(pred_original, 0)
        rmse_original = float(np.sqrt(mean_squared_error(y_valid_raw, pred_original)))
        mae_original = float(mean_absolute_error(y_valid_raw, pred_original))
        metrics["valid_rmse_original_scale"] = rmse_original
        metrics["valid_mae_original_scale"] = mae_original

    # Save ensemble as dict
    ensemble_data = {
        "models": trained_models,
        "weights": weights,
        "model_types": model_types,
        "feature_cols": feature_cols,
        "log_target": log_target,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(ensemble_data, f)

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Aggregate feature importance (weighted average across models that have it)
    if "feature_importance" in outputs:
        total_importance = np.zeros(len(feature_cols))
        total_weight = 0.0
        for (model_type, mdl), w in zip(trained_models.items(), weights):
            if hasattr(mdl, "feature_importances_"):
                total_importance += w * mdl.feature_importances_
                total_weight += w
        if total_weight > 0:
            total_importance /= total_weight
            importance = pd.DataFrame({
                "feature": feature_cols,
                "importance": total_importance,
            }).sort_values("importance", ascending=False)
            importance.to_csv(outputs["feature_importance"], index=False)

    model_list = "+".join(model_types)
    return (
        f"train_ensemble_regressor: [{model_list}], "
        f"{len(X_train)} samples, RMSE={blend_rmse:.4f}"
    )


# =============================================================================
# SERVICE 6: GENERIC PREDICT REGRESSOR
# =============================================================================

@contract(
    inputs={
        "model": {
            "format": "pickle",
            "required": True,
            "schema": {"type": "artifact", "artifact_type": "any"},
        },
        "data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular"},
        },
    },
    outputs={
        "predictions": {
            "format": "csv",
            "schema": {"type": "tabular"},
        },
    },
    description="Generate predictions from a pickled regression model",
    tags=["inference", "prediction", "regression", "generic"],
    version="1.0.0",
)
def predict_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "target",
    log_target: bool = False,
) -> str:
    """
    Generate predictions from a pickled regression model.

    Loads any model with a .predict() method from a pickle file,
    applies it to the input data, and saves an output CSV with
    id and prediction columns.

    If log_target is True, the model was trained on log1p(target),
    so predictions are expm1-transformed back to the original scale.

    Args:
        id_column: Name of the ID column in input data
        prediction_column: Name for the prediction column in output
        log_target: Whether to apply expm1 to reverse log1p transform
    """
    with open(inputs["model"], "rb") as f:
        model = pickle.load(f)

    df = _load_data(inputs["data"])

    # Extract IDs
    if id_column in df.columns:
        ids = df[id_column]
    else:
        ids = pd.RangeIndex(len(df))

    # Determine feature columns (everything except id_column)
    feature_cols = [c for c in df.columns if c != id_column]
    X = df[feature_cols]

    # Predict
    preds = model.predict(X)

    # Reverse log transform if needed
    if log_target:
        preds = np.expm1(preds)

    # Ensure non-negative predictions
    preds = np.maximum(preds, 0)

    # Build output
    pred_df = pd.DataFrame({
        id_column: ids,
        prediction_column: preds,
    })

    _save_data(pred_df, outputs["predictions"])

    return (
        f"predict_regressor: {len(preds)} predictions, "
        f"mean={preds.mean():.4f}, std={preds.std():.4f}"
    )


# =============================================================================
# SERVICE 7: PREDICT ENSEMBLE REGRESSOR
# =============================================================================

@contract(
    inputs={
        "model": {
            "format": "pickle",
            "required": True,
            "schema": {"type": "artifact", "artifact_type": "ensemble_model"},
        },
        "data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular"},
        },
    },
    outputs={
        "predictions": {
            "format": "csv",
            "schema": {"type": "tabular"},
        },
    },
    description="Generate predictions from an ensemble regression model (dict with models and weights)",
    tags=["inference", "prediction", "ensemble", "regression", "generic"],
    version="1.0.0",
)
def predict_ensemble_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "target",
) -> str:
    """
    Generate predictions from an ensemble regression model.

    Expects a pickle file containing a dict with:
    - 'models': dict of {model_type: fitted_model}
    - 'weights': list of floats
    - 'model_types': list of model type names
    - 'feature_cols': list of feature column names
    - 'log_target': bool (whether to reverse log1p transform)

    This is the companion prediction service for train_ensemble_regressor.

    Args:
        id_column: Name of the ID column in input data
        prediction_column: Name for the prediction column in output
    """
    with open(inputs["model"], "rb") as f:
        ensemble_data = pickle.load(f)

    df = _load_data(inputs["data"])

    # Extract IDs
    if id_column in df.columns:
        ids = df[id_column]
    else:
        ids = pd.RangeIndex(len(df))

    # Use feature columns from training
    feature_cols = ensemble_data["feature_cols"]
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()

    # Add missing columns as 0 (handles unseen features gracefully)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    # Blended prediction
    models = ensemble_data["models"]
    weights = ensemble_data["weights"]
    model_types = ensemble_data["model_types"]

    blended_pred = np.zeros(len(X))
    for model_type, w in zip(model_types, weights):
        mdl = models[model_type]
        blended_pred += w * mdl.predict(X)

    # Reverse log transform if applicable
    log_target = ensemble_data.get("log_target", False)
    if log_target:
        blended_pred = np.expm1(blended_pred)

    # Ensure non-negative
    blended_pred = np.maximum(blended_pred, 0)

    # Build output
    pred_df = pd.DataFrame({
        id_column: ids,
        prediction_column: blended_pred,
    })

    _save_data(pred_df, outputs["predictions"])

    model_list = "+".join(model_types)
    return (
        f"predict_ensemble_regressor: [{model_list}], "
        f"{len(blended_pred)} predictions, mean={blended_pred.mean():.4f}"
    )


# =============================================================================
# SERVICE 8: MULTI-OUTPUT REGRESSOR (for multi-target regression)
# =============================================================================

@contract(
    inputs={
        "train_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 1},
        },
        "valid_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False},
        },
    },
    outputs={
        "model": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "multi_output_model"},
        },
        "metrics": {
            "format": "json",
            "schema": {
                "type": "json",
                "required_fields": ["model_type", "valid_rmse"],
            },
        },
    },
    description="Train a multi-output regressor for multiple target columns",
    tags=["modeling", "training", "multi-output", "regression", "generic"],
    version="1.0.0",
)
def train_multi_output_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_columns: List[str] = None,
    feature_exclude: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> str:
    """
    Train a multi-output regressor for predicting multiple targets.

    Uses sklearn's MultiOutputRegressor wrapper around RandomForestRegressor.
    Suitable for tasks like facial keypoint detection with multiple coordinate outputs.

    Args:
        target_columns: List of target column names to predict
        feature_exclude: Additional columns to exclude from features
        n_estimators: Number of trees in the forest
        max_depth: Maximum tree depth (None = unlimited)
        random_state: Random seed for reproducibility
    """
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    train = _load_data(inputs["train_data"])
    valid = _load_data(inputs["valid_data"])

    # Auto-detect target columns if not provided (assume all numeric non-feature columns)
    if target_columns is None:
        raise ValueError("target_columns must be specified for multi-output regression")

    # Filter to only include targets that exist in data
    target_columns = [c for c in target_columns if c in train.columns]

    if len(target_columns) == 0:
        raise ValueError("No valid target columns found in training data")

    # Determine feature columns
    feature_exclude = feature_exclude or []
    exclude_cols = set(feature_exclude + target_columns)
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    # Handle missing values in targets by dropping rows with any NaN in targets
    train_clean = train.dropna(subset=target_columns)
    valid_clean = valid.dropna(subset=target_columns)

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_columns]
    X_valid = valid_clean[feature_cols]
    y_valid = valid_clean[target_columns]

    # Create and train multi-output model
    base_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_valid)
    rmse_per_target = {}
    for i, col in enumerate(target_columns):
        rmse = float(np.sqrt(mean_squared_error(y_valid[col], preds[:, i])))
        rmse_per_target[col] = rmse

    avg_rmse = float(np.mean(list(rmse_per_target.values())))

    metrics = {
        "model_type": "MultiOutputRegressor_RandomForest",
        "n_samples_train": int(len(X_train)),
        "n_samples_valid": int(len(X_valid)),
        "n_features": int(len(feature_cols)),
        "n_targets": int(len(target_columns)),
        "target_columns": target_columns,
        "valid_rmse": avg_rmse,
        "rmse_per_target": rmse_per_target,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    # Save model artifact
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_columns": target_columns,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return (
        f"train_multi_output_regressor: {len(X_train)} samples, "
        f"{len(feature_cols)} features, {len(target_columns)} targets, RMSE={avg_rmse:.4f}"
    )


@contract(
    inputs={
        "model": {
            "format": "pickle",
            "required": True,
            "schema": {"type": "artifact", "artifact_type": "multi_output_model"},
        },
        "data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular"},
        },
    },
    outputs={
        "predictions": {
            "format": "csv",
            "schema": {"type": "tabular"},
        },
    },
    description="Generate predictions from a multi-output regression model",
    tags=["inference", "prediction", "multi-output", "regression", "generic"],
    version="1.0.0",
)
def predict_multi_output_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: Optional[str] = None,
) -> str:
    """
    Generate predictions from a multi-output regression model.

    Args:
        id_column: Optional ID column to include in output
    """
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    target_columns = artifact["target_columns"]

    df = _load_data(inputs["data"])

    # Prepare features
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()

    # Add missing columns as 0
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    # Predict
    preds = model.predict(X)

    # Build output DataFrame
    pred_df = pd.DataFrame(preds, columns=target_columns)

    # Add ID column if specified
    if id_column and id_column in df.columns:
        pred_df.insert(0, id_column, df[id_column].values)

    _save_data(pred_df, outputs["predictions"])

    return f"predict_multi_output_regressor: {len(preds)} predictions for {len(target_columns)} targets"


# =============================================================================
# SERVICE 9: STACKED REGRESSOR (K-fold stacking with meta-learner)
# =============================================================================

@contract(
    inputs={
        "train_data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "allow_missing": False, "min_rows": 10},
        },
        "valid_data": {
            "format": "csv",
            "required": False,
            "schema": {"type": "tabular"},
        },
    },
    outputs={
        "model": {
            "format": "pickle",
            "schema": {"type": "artifact", "artifact_type": "model"},
        },
        "metrics": {
            "format": "json",
            "schema": {"type": "json"},
        },
    },
    description="Train a K-fold stacked ensemble: Level-1 diverse models -> OOF -> Level-2 Ridge meta-learner",
    tags=["modeling", "training", "stacking", "regression", "generic"],
    version="1.0.0",
)
def train_stacked_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = "Id",
    model_types: Optional[List[str]] = None,
    n_folds: int = 5,
    log_target: bool = False,
    random_state: int = 42,
    # --- Gradient Boosting per-model params ---
    gbr_n_estimators: int = 500,
    gbr_learning_rate: float = 0.05,
    gbr_max_depth: int = 3,
    gbr_min_samples_split: int = 2,
    gbr_min_samples_leaf: int = 1,
    gbr_subsample: float = 0.8,
    # --- LightGBM per-model params ---
    lgbm_n_estimators: int = 500,
    lgbm_learning_rate: float = 0.05,
    lgbm_num_leaves: int = 31,
    lgbm_max_depth: int = -1,
    lgbm_min_child_samples: int = 20,
    lgbm_subsample: float = 0.8,
    lgbm_colsample_bytree: float = 0.8,
    lgbm_reg_alpha: float = 0.0,
    lgbm_reg_lambda: float = 0.0,
    # --- XGBoost per-model params ---
    xgb_n_estimators: int = 500,
    xgb_learning_rate: float = 0.05,
    xgb_max_depth: int = 6,
    xgb_min_child_weight: int = 1,
    xgb_subsample: float = 0.8,
    xgb_colsample_bytree: float = 0.8,
    xgb_reg_alpha: float = 0.0,
    xgb_reg_lambda: float = 1.0,
    # --- Ridge per-model params ---
    ridge_alpha: float = 10.0,
    # --- ElasticNet per-model params ---
    enet_alpha: float = 0.01,
    enet_l1_ratio: float = 0.5,
    # --- Lasso per-model params ---
    lasso_alpha: float = 0.0005,
    # --- KernelRidge per-model params ---
    kr_alpha: float = 0.6,
    kr_kernel: str = "polynomial",
    kr_degree: int = 2,
    kr_coef0: float = 2.5,
    # --- Meta-learner params ---
    meta_alpha: float = 1.0,
) -> str:
    """
    Train a K-fold stacked ensemble regressor.

    Level-1: Train diverse models using K-fold cross-validation to generate
    out-of-fold (OOF) predictions. Retrain on full training data for inference.
    Level-2: Ridge meta-learner trained on OOF predictions.

    Supported Level-1 model types: "gradient_boosting", "lightgbm", "xgboost",
    "ridge", "elasticnet".

    If log_target is True, the target is log1p-transformed before training and
    predictions are expm1-transformed back (common for RMSLE optimization).

    Args:
        label_column: Target column name
        id_column: ID column to exclude from features
        model_types: List of Level-1 model types
        n_folds: Number of folds for cross-validation
        log_target: Whether to apply log1p transform to target
        random_state: Random seed for reproducibility
        gbr_*: GradientBoostingRegressor parameters
        lgbm_*: LGBMRegressor parameters
        xgb_*: XGBRegressor parameters
        ridge_alpha: Ridge regularization for Level-1 ridge model
        enet_*: ElasticNet parameters
        meta_alpha: Ridge alpha for Level-2 meta-learner
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge, ElasticNet, Lasso
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    model_types = model_types or ["gradient_boosting", "lightgbm", "xgboost", "ridge", "elasticnet"]

    train = _load_data(inputs["train_data"])

    exclude_cols = {label_column}
    if id_column and id_column in train.columns:
        exclude_cols.add(id_column)
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols].values
    y_train_raw = train[label_column].values

    if log_target:
        y_train = np.log1p(y_train_raw)
    else:
        y_train = y_train_raw.copy()

    def _make_model(model_type):
        if model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=gbr_n_estimators,
                learning_rate=gbr_learning_rate,
                max_depth=gbr_max_depth,
                min_samples_split=gbr_min_samples_split,
                min_samples_leaf=gbr_min_samples_leaf,
                subsample=gbr_subsample,
                random_state=random_state,
            )
        elif model_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=lgbm_n_estimators,
                learning_rate=lgbm_learning_rate,
                num_leaves=lgbm_num_leaves,
                max_depth=lgbm_max_depth,
                min_child_samples=lgbm_min_child_samples,
                subsample=lgbm_subsample,
                colsample_bytree=lgbm_colsample_bytree,
                reg_alpha=lgbm_reg_alpha,
                reg_lambda=lgbm_reg_lambda,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif model_type == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=xgb_n_estimators,
                learning_rate=xgb_learning_rate,
                max_depth=xgb_max_depth,
                min_child_weight=xgb_min_child_weight,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                reg_alpha=xgb_reg_alpha,
                reg_lambda=xgb_reg_lambda,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        elif model_type == "ridge":
            return Ridge(alpha=ridge_alpha, random_state=random_state)
        elif model_type == "elasticnet":
            return ElasticNet(
                alpha=enet_alpha,
                l1_ratio=enet_l1_ratio,
                random_state=random_state,
                max_iter=10000,
            )
        elif model_type == "lasso":
            return Lasso(
                alpha=lasso_alpha,
                random_state=random_state,
                max_iter=10000,
            )
        elif model_type == "kernel_ridge":
            return KernelRidge(
                alpha=kr_alpha,
                kernel=kr_kernel,
                degree=kr_degree,
                coef0=kr_coef0,
            )
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported: gradient_boosting, lightgbm, xgboost, ridge, elasticnet, lasso, kernel_ridge"
            )

    # --- K-fold OOF predictions ---
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros((len(X_train), len(model_types)))

    print(f"  Stacking: {n_folds}-fold CV with {len(model_types)} models...")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train[train_idx]

        for m_idx, model_type in enumerate(model_types):
            mdl = _make_model(model_type)
            mdl.fit(X_fold_train, y_fold_train)
            oof_predictions[val_idx, m_idx] = mdl.predict(X_fold_val)

        print(f"    Fold {fold_idx + 1}/{n_folds} done")

    # --- Level-2 meta-learner on OOF predictions ---
    meta_learner = Ridge(alpha=meta_alpha, random_state=random_state)
    meta_learner.fit(oof_predictions, y_train)

    # --- Retrain Level-1 models on full training data (for test prediction) ---
    level1_models = {}
    for model_type in model_types:
        mdl = _make_model(model_type)
        mdl.fit(X_train, y_train)
        level1_models[model_type] = mdl

    # --- OOF meta-predictions for evaluation ---
    meta_oof_pred = meta_learner.predict(oof_predictions)
    oof_rmse = float(np.sqrt(mean_squared_error(y_train, meta_oof_pred)))

    metrics = {
        "model_type": "stacked_regressor",
        "model_types": model_types,
        "n_folds": n_folds,
        "log_target": log_target,
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "oof_rmse": oof_rmse,
        "meta_learner_coefs": meta_learner.coef_.tolist(),
    }

    # --- Evaluate on validation set if provided ---
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid = _load_data(inputs["valid_data"])
        X_valid = valid[feature_cols].values
        y_valid_raw = valid[label_column].values

        if log_target:
            y_valid = np.log1p(y_valid_raw)
        else:
            y_valid = y_valid_raw.copy()

        valid_l1_preds = np.zeros((len(X_valid), len(model_types)))
        for m_idx, model_type in enumerate(model_types):
            valid_l1_preds[:, m_idx] = level1_models[model_type].predict(X_valid)

        valid_meta_pred = meta_learner.predict(valid_l1_preds)
        valid_rmse = float(np.sqrt(mean_squared_error(y_valid, valid_meta_pred)))
        metrics["valid_rmse"] = valid_rmse

        if log_target:
            pred_original = np.expm1(valid_meta_pred)
            pred_original = np.maximum(pred_original, 0)
            from sklearn.metrics import mean_squared_log_error, mean_absolute_error
            rmse_original = float(np.sqrt(mean_squared_error(y_valid_raw, pred_original)))
            rmsle = float(np.sqrt(mean_squared_log_error(
                np.maximum(y_valid_raw, 0), pred_original
            )))
            metrics["valid_rmse_original_scale"] = rmse_original
            metrics["valid_rmsle"] = rmsle

        for m_idx, model_type in enumerate(model_types):
            individual_rmse = float(np.sqrt(mean_squared_error(
                y_valid, valid_l1_preds[:, m_idx]
            )))
            metrics[f"valid_rmse_{model_type}"] = individual_rmse

    # --- Save model artifact ---
    stacked_data = {
        "level1_models": level1_models,
        "meta_learner": meta_learner,
        "model_types": model_types,
        "feature_cols": feature_cols,
        "log_target": log_target,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(stacked_data, f)

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    model_list = "+".join(model_types)
    rmsle_str = f", RMSLE={metrics['valid_rmsle']:.5f}" if "valid_rmsle" in metrics else ""
    return (
        f"train_stacked_regressor: [{model_list}], "
        f"{len(X_train)} samples, OOF-RMSE={oof_rmse:.4f}{rmsle_str}"
    )


# =============================================================================
# SERVICE 10: PREDICT STACKED REGRESSOR
# =============================================================================

@contract(
    inputs={
        "model": {
            "format": "pickle",
            "required": True,
            "schema": {"type": "artifact", "artifact_type": "model"},
        },
        "data": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular"},
        },
    },
    outputs={
        "predictions": {
            "format": "csv",
            "schema": {"type": "tabular"},
        },
    },
    description="Generate predictions from a stacked regression model (Level-1 -> Level-2 meta-learner)",
    tags=["inference", "prediction", "stacking", "regression", "generic"],
    version="1.0.0",
)
def predict_stacked_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "target",
) -> str:
    """
    Generate predictions from a stacked regression model.

    Expects a pickle file containing a dict with:
    - 'level1_models': dict of {model_type: fitted_model}
    - 'meta_learner': fitted Ridge meta-learner
    - 'model_types': list of model type names
    - 'feature_cols': list of feature column names
    - 'log_target': bool (whether to reverse log1p transform)

    This is the companion prediction service for train_stacked_regressor.

    Args:
        id_column: Name of the ID column in input data
        prediction_column: Name for the prediction column in output
    """
    with open(inputs["model"], "rb") as f:
        stacked_data = pickle.load(f)

    df = _load_data(inputs["data"])

    if id_column in df.columns:
        ids = df[id_column]
    else:
        ids = pd.RangeIndex(len(df))

    feature_cols = stacked_data["feature_cols"]
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()

    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols].values

    level1_models = stacked_data["level1_models"]
    meta_learner = stacked_data["meta_learner"]
    model_types = stacked_data["model_types"]
    log_target = stacked_data.get("log_target", False)

    # Level-1 predictions
    l1_preds = np.zeros((len(X), len(model_types)))
    for m_idx, model_type in enumerate(model_types):
        l1_preds[:, m_idx] = level1_models[model_type].predict(X)

    # Level-2 meta prediction
    predictions = meta_learner.predict(l1_preds)

    if log_target:
        predictions = np.expm1(predictions)
    predictions = np.maximum(predictions, 0)

    pred_df = pd.DataFrame({
        id_column: ids,
        prediction_column: predictions,
    })

    _save_data(pred_df, outputs["predictions"])

    model_list = "+".join(model_types)
    return (
        f"predict_stacked_regressor: [{model_list}], "
        f"{len(predictions)} predictions, mean={predictions.mean():.4f}"
    )


# =============================================================================
# AUTOML REGRESSOR (FLAML)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "valid_data": {"format": "csv", "required": False},
    },
    outputs={
        "model": {"format": "pkl"},
        "metrics": {"format": "json"},
    },
    description="Train a regressor using FLAML AutoML - automatically finds best model",
    tags=["modeling", "training", "automl", "flaml", "regression", "generic"],
    version="1.0.0",
)
def train_automl_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    time_budget: int = 120,
    metric: str = "rmse",
    estimator_list: List[str] = None,
    random_state: int = 42,
    exclude_columns: Optional[List[str]] = None,
    n_jobs: int = -1,
) -> str:
    """Train a regressor using FLAML AutoML.

    FLAML (Fast and Lightweight AutoML) automatically searches for the best
    model and hyperparameters within the given time budget.

    Parameters:
        label_column: Target column name
        id_column: ID column to exclude from features
        time_budget: Time budget in seconds for AutoML search (default 120)
        metric: Optimization metric - 'rmse', 'mae', 'mse', 'r2', 'mape'
        estimator_list: List of estimators to try. Default: ['lgbm', 'xgboost', 'rf', 'extra_tree']
        random_state: Random seed for reproducibility
        exclude_columns: Additional columns to exclude from features
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    try:
        from flaml import AutoML
    except ImportError:
        raise ImportError("FLAML is not installed. Install with: pip install flaml")

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)
    if exclude_columns:
        drop_cols.extend([c for c in exclude_columns if c in train_df.columns])

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[label_column]
    feature_cols = list(X_train.columns)

    # Default estimator list
    if estimator_list is None:
        estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree']

    # Create AutoML
    automl = AutoML()

    # Prepare validation data if available
    X_val, y_val = None, None
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_val = valid_df.drop(columns=drop_cols, errors="ignore")
        y_val = valid_df[label_column]

    # Run AutoML
    automl.fit(
        X_train, y_train,
        task="regression",
        time_budget=time_budget,
        metric=metric,
        estimator_list=estimator_list,
        seed=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    # Evaluate
    train_preds = automl.predict(X_train)
    train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))

    metrics = {
        "model_type": "automl_flaml",
        "best_estimator": automl.best_estimator,
        "best_config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                        for k, v in (automl.best_config or {}).items()},
        "time_budget": time_budget,
        "metric": metric,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "train_rmse": float(train_rmse),
        "best_loss": float(automl.best_loss) if automl.best_loss else None,
    }

    # Validation metrics
    if X_val is not None and y_val is not None:
        val_preds = automl.predict(X_val)
        val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
        val_mae = np.mean(np.abs(y_val - val_preds))
        metrics["valid_rmse"] = float(val_rmse)
        metrics["valid_mae"] = float(val_mae)

    # Save model
    model_artifact = {
        "model": automl,
        "feature_cols": feature_cols,
        "best_config": automl.best_config,
        "best_estimator": automl.best_estimator,
    }

    with open(outputs["model"], "wb") as f:
        pickle.dump(model_artifact, f)

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return (
        f"train_automl_regressor: best={automl.best_estimator}, "
        f"time={time_budget}s, RMSE={metrics.get('valid_rmse', train_rmse):.4f}"
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "train_xgboost_regressor": train_xgboost_regressor,
    "train_ensemble_regressor": train_ensemble_regressor,
    "train_stacked_regressor": train_stacked_regressor,
    "predict_regressor": predict_regressor,
    "predict_ensemble_regressor": predict_ensemble_regressor,
    "predict_stacked_regressor": predict_stacked_regressor,
}
