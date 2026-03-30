"""
Ensemble Services
=================
Generic ensemble training and prediction services for combining multiple models.

Supports:
- Weighted ensemble classifiers (LightGBM + XGBoost + RandomForest)
- Weighted ensemble regressors
- Stacking with meta-learner
- Post-hoc prediction blending

Based on top Kaggle solution patterns.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# =============================================================================
# IMPORTS - with fallback for different execution contexts
# =============================================================================
try:
    from .io_utils import load_data, save_data
except ImportError:
    from services.io_utils import load_data, save_data

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from contract import contract
except ImportError:
    try:
        from app.contract import contract
    except ImportError:
        # Fallback: no-op decorator if contract system unavailable
        def contract(**kwargs):
            def decorator(func):
                return func
            return decorator


# =============================================================================
# WEIGHTED ENSEMBLE CLASSIFIER
# =============================================================================
@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train a weighted ensemble of classifiers (LightGBM, XGBoost, RandomForest)",
    tags=["ensemble", "classification", "training"],
)
def train_weighted_ensemble_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = "id",
    base_models: List[str] = None,
    weight_method: str = "validation_score",
    # LightGBM params
    lgb_n_estimators: int = 1000,
    lgb_learning_rate: float = 0.05,
    lgb_num_leaves: int = 64,
    lgb_max_depth: int = -1,
    lgb_min_child_samples: int = 30,
    lgb_subsample: float = 0.8,
    lgb_colsample_bytree: float = 0.8,
    lgb_early_stopping_rounds: int = 100,
    # XGBoost params
    xgb_n_estimators: int = 300,
    xgb_learning_rate: float = 0.1,
    xgb_max_depth: int = 5,
    xgb_subsample: float = 0.9,
    xgb_colsample_bytree: float = 0.9,
    # RandomForest params
    rf_n_estimators: int = 300,
    rf_max_depth: int = 10,
    rf_min_samples_split: int = 5,
    rf_min_samples_leaf: int = 2,
    rf_class_weight: str = "balanced",
    # General
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a weighted ensemble of classifiers (LightGBM, XGBoost, RandomForest).
    
    Weights are computed based on validation performance (AUC or accuracy).
    
    Args:
        inputs: Dict with 'train_data' and 'valid_data' paths
        outputs: Dict with 'model' and 'metrics' paths
        label_column: Name of target column
        id_column: Name of ID column (excluded from features)
        base_models: List of models to include ['lightgbm', 'xgboost', 'random_forest']
        weight_method: 'validation_score' (AUC-weighted) or 'equal'
        *_params: Hyperparameters for each base model
        
    Returns:
        Dict with status and metrics summary
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    if base_models is None:
        base_models = ["lightgbm", "xgboost", "random_forest"]
    
    # Load data
    train_df = load_data(inputs["train_data"])
    valid_df = load_data(inputs["valid_data"])
    
    # Prepare features
    exclude_cols = {label_column, id_column} if id_column else {label_column}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[label_column]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[label_column]
    
    models = {}
    scores = {}
    
    # Train LightGBM
    if "lightgbm" in base_models:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=lgb_n_estimators,
            learning_rate=lgb_learning_rate,
            num_leaves=lgb_num_leaves,
            max_depth=lgb_max_depth,
            min_child_samples=lgb_min_child_samples,
            subsample=lgb_subsample,
            colsample_bytree=lgb_colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(lgb_early_stopping_rounds, verbose=False)]
        )
        lgb_pred = lgb_model.predict_proba(X_valid)[:, 1]
        lgb_auc = roc_auc_score(y_valid, lgb_pred)
        models["lightgbm"] = lgb_model
        scores["lightgbm"] = lgb_auc
    
    # Train XGBoost
    if "xgboost" in base_models:
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_n_estimators,
            learning_rate=xgb_learning_rate,
            max_depth=xgb_max_depth,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="auc",
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        xgb_pred = xgb_model.predict_proba(X_valid)[:, 1]
        xgb_auc = roc_auc_score(y_valid, xgb_pred)
        models["xgboost"] = xgb_model
        scores["xgboost"] = xgb_auc
    
    # Train RandomForest
    if "random_forest" in base_models:
        rf_model = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            max_features="sqrt",
            class_weight=rf_class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict_proba(X_valid)[:, 1]
        rf_auc = roc_auc_score(y_valid, rf_pred)
        models["random_forest"] = rf_model
        scores["random_forest"] = rf_auc
    
    # Compute weights
    if weight_method == "validation_score":
        total_score = sum(scores.values())
        weights = {k: v / total_score for k, v in scores.items()}
    else:
        weights = {k: 1.0 / len(scores) for k in scores}
    
    # Compute ensemble prediction
    ensemble_pred = np.zeros(len(X_valid))
    for model_name, weight in weights.items():
        if model_name == "lightgbm":
            ensemble_pred += weight * lgb_pred
        elif model_name == "xgboost":
            ensemble_pred += weight * xgb_pred
        elif model_name == "random_forest":
            ensemble_pred += weight * rf_pred
    
    ensemble_auc = roc_auc_score(y_valid, ensemble_pred)
    
    # Save ensemble model
    ensemble_data = {
        "models": models,
        "weights": weights,
        "scores": scores,
        "feature_cols": feature_cols,
        "label_column": label_column,
        "id_column": id_column,
        "model_type": "weighted_ensemble_classifier",
    }
    
    os.makedirs(os.path.dirname(outputs["model"]), exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(ensemble_data, f)
    
    # Save metrics
    metrics = {
        "model_type": "weighted_ensemble_classifier",
        "base_models": list(models.keys()),
        "individual_scores": {k: float(v) for k, v in scores.items()},
        "weights": {k: float(v) for k, v in weights.items()},
        "ensemble_auc": float(ensemble_auc),
        "valid_auc": float(ensemble_auc),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_valid": len(X_valid),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)
    
    return {
        "status": "success",
        "ensemble_auc": ensemble_auc,
        "individual_scores": scores,
        "weights": weights,
    }


# =============================================================================
# WEIGHTED ENSEMBLE REGRESSOR
# =============================================================================
@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train a weighted ensemble of regressors (LightGBM, XGBoost, RandomForest)",
    tags=["ensemble", "regression", "training"],
)
def train_weighted_ensemble_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = "id",
    base_models: List[str] = None,
    weight_method: str = "validation_score",
    log_transform: bool = False,
    # LightGBM params
    lgb_n_estimators: int = 2000,
    lgb_learning_rate: float = 0.02,
    lgb_num_leaves: int = 255,
    lgb_max_depth: int = 8,
    lgb_min_child_samples: int = 20,
    lgb_subsample: float = 0.7,
    lgb_colsample_bytree: float = 0.5,
    lgb_early_stopping_rounds: int = 100,
    # XGBoost params
    xgb_n_estimators: int = 1000,
    xgb_learning_rate: float = 0.03,
    xgb_max_depth: int = 10,
    xgb_subsample: float = 0.9,
    xgb_colsample_bytree: float = 0.7,
    # RandomForest params
    rf_n_estimators: int = 200,
    rf_max_depth: int = 15,
    rf_min_samples_split: int = 5,
    # General
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a weighted ensemble of regressors (LightGBM, XGBoost, RandomForest).
    
    Weights are computed based on validation RMSE (inverse weighting).
    
    Args:
        inputs: Dict with 'train_data' and 'valid_data' paths
        outputs: Dict with 'model' and 'metrics' paths
        label_column: Name of target column
        id_column: Name of ID column (excluded from features)
        base_models: List of models to include
        weight_method: 'validation_score' (RMSE-weighted) or 'equal'
        log_transform: If True, target is already log-transformed
        
    Returns:
        Dict with status and metrics summary
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    
    if base_models is None:
        base_models = ["lightgbm", "xgboost"]
    
    # Load data
    train_df = load_data(inputs["train_data"])
    valid_df = load_data(inputs["valid_data"])
    
    # Prepare features
    exclude_cols = {label_column, id_column} if id_column else {label_column}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[label_column]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[label_column]
    
    models = {}
    rmse_scores = {}
    predictions = {}
    
    # Train LightGBM
    if "lightgbm" in base_models:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=lgb_n_estimators,
            learning_rate=lgb_learning_rate,
            num_leaves=lgb_num_leaves,
            max_depth=lgb_max_depth,
            min_child_samples=lgb_min_child_samples,
            subsample=lgb_subsample,
            colsample_bytree=lgb_colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
        )
        lgb_pred = lgb_model.predict(X_valid)
        lgb_rmse = np.sqrt(mean_squared_error(y_valid, lgb_pred))
        models["lightgbm"] = lgb_model
        rmse_scores["lightgbm"] = lgb_rmse
        predictions["lightgbm"] = lgb_pred
    
    # Train XGBoost
    if "xgboost" in base_models:
        xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_n_estimators,
            learning_rate=xgb_learning_rate,
            max_depth=xgb_max_depth,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        xgb_pred = xgb_model.predict(X_valid)
        xgb_rmse = np.sqrt(mean_squared_error(y_valid, xgb_pred))
        models["xgboost"] = xgb_model
        rmse_scores["xgboost"] = xgb_rmse
        predictions["xgboost"] = xgb_pred
    
    # Train RandomForest
    if "random_forest" in base_models:
        rf_model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_valid)
        rf_rmse = np.sqrt(mean_squared_error(y_valid, rf_pred))
        models["random_forest"] = rf_model
        rmse_scores["random_forest"] = rf_rmse
        predictions["random_forest"] = rf_pred
    
    # Compute weights (inverse RMSE for regression)
    if weight_method == "validation_score":
        inv_rmse = {k: 1.0 / v for k, v in rmse_scores.items()}
        total_inv = sum(inv_rmse.values())
        weights = {k: v / total_inv for k, v in inv_rmse.items()}
    else:
        weights = {k: 1.0 / len(rmse_scores) for k in rmse_scores}
    
    # Compute ensemble prediction
    ensemble_pred = np.zeros(len(X_valid))
    for model_name, weight in weights.items():
        ensemble_pred += weight * predictions[model_name]
    
    ensemble_rmse = np.sqrt(mean_squared_error(y_valid, ensemble_pred))
    
    # Compute RMSPE if log-transformed
    rmspe = None
    if log_transform:
        y_valid_orig = np.expm1(y_valid)
        pred_orig = np.expm1(ensemble_pred)
        mask = y_valid_orig != 0
        rmspe = np.sqrt(np.mean(((y_valid_orig[mask] - pred_orig[mask]) / y_valid_orig[mask]) ** 2))
    
    # Save ensemble model
    ensemble_data = {
        "models": models,
        "weights": weights,
        "rmse_scores": rmse_scores,
        "feature_cols": feature_cols,
        "label_column": label_column,
        "id_column": id_column,
        "log_transform": log_transform,
        "model_type": "weighted_ensemble_regressor",
    }
    
    os.makedirs(os.path.dirname(outputs["model"]), exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(ensemble_data, f)
    
    # Save metrics
    metrics = {
        "model_type": "weighted_ensemble_regressor",
        "base_models": list(models.keys()),
        "individual_rmse": {k: float(v) for k, v in rmse_scores.items()},
        "weights": {k: float(v) for k, v in weights.items()},
        "ensemble_rmse": float(ensemble_rmse),
        "valid_rmse": float(ensemble_rmse),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_valid": len(X_valid),
        "timestamp": datetime.now().isoformat(),
    }
    if rmspe is not None:
        metrics["valid_rmspe"] = float(rmspe)
    
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)
    
    return {
        "status": "success",
        "ensemble_rmse": ensemble_rmse,
        "individual_rmse": rmse_scores,
        "weights": weights,
        "rmspe": rmspe,
    }


# =============================================================================
# PREDICT ENSEMBLE
# =============================================================================
@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Generate predictions using a trained ensemble model",
    tags=["ensemble", "prediction"],
)
def predict_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "prediction",
    output_proba: bool = True,
) -> Dict[str, Any]:
    """
    Generate predictions using a trained ensemble model.
    
    Args:
        inputs: Dict with 'model' and 'data' paths
        outputs: Dict with 'predictions' path
        id_column: Name of ID column
        prediction_column: Name for prediction column in output
        output_proba: If True (classifier), output probabilities; else class labels
        
    Returns:
        Dict with status and prediction stats
    """
    # Load model
    with open(inputs["model"], "rb") as f:
        ensemble_data = pickle.load(f)
    
    models = ensemble_data["models"]
    weights = ensemble_data["weights"]
    feature_cols = ensemble_data["feature_cols"]
    model_type = ensemble_data.get("model_type", "classifier")
    log_transform = ensemble_data.get("log_transform", False)
    
    # Load data
    data = load_data(inputs["data"])
    X = data[feature_cols]
    
    # Generate predictions
    ensemble_pred = np.zeros(len(X))
    
    for model_name, weight in weights.items():
        model = models[model_name]
        if "classifier" in model_type:
            if output_proba:
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
        else:
            pred = model.predict(X)
        ensemble_pred += weight * pred
    
    # Handle log transform for regressors
    if "regressor" in model_type and log_transform:
        ensemble_pred = np.expm1(ensemble_pred)
        ensemble_pred = np.maximum(ensemble_pred, 0)  # Clip negatives
    
    # Create output dataframe
    if id_column and id_column in data.columns:
        result_df = pd.DataFrame({
            id_column: data[id_column],
            prediction_column: ensemble_pred,
        })
    else:
        result_df = pd.DataFrame({
            prediction_column: ensemble_pred,
        })
    
    # Save predictions
    os.makedirs(os.path.dirname(outputs["predictions"]), exist_ok=True)
    save_data(result_df, outputs["predictions"])
    
    return {
        "status": "success",
        "n_predictions": len(result_df),
        "mean_prediction": float(ensemble_pred.mean()),
        "std_prediction": float(ensemble_pred.std()),
    }


# =============================================================================
# BLEND PREDICTIONS
# =============================================================================
@contract(
    inputs={
        "prediction_1": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "prediction_2": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "blended": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Blend multiple prediction files with specified weights",
    tags=["ensemble", "blending", "prediction"],
)
def blend_predictions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    prediction_files: List[str] = None,
    weights: List[float] = None,
    id_column: str = "id",
    prediction_column: str = "prediction",
    output_column: str = "prediction",
) -> Dict[str, Any]:
    """
    Blend multiple prediction files with specified weights.
    
    Args:
        inputs: Dict with 'prediction_1', 'prediction_2', etc. paths
        outputs: Dict with 'blended' path
        prediction_files: List of prediction file keys in inputs (optional)
        weights: List of weights for each file (defaults to equal)
        id_column: Name of ID column
        prediction_column: Name of prediction column in input files
        output_column: Name of prediction column in output
        
    Returns:
        Dict with status and blend stats
    """
    # Get prediction files from inputs
    if prediction_files is None:
        prediction_files = sorted([k for k in inputs.keys() if k.startswith("prediction")])
    
    if weights is None:
        weights = [1.0 / len(prediction_files)] * len(prediction_files)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Load and blend
    blended = None
    ids = None
    
    for i, file_key in enumerate(prediction_files):
        df = load_data(inputs[file_key])
        if blended is None:
            blended = np.zeros(len(df))
            if id_column in df.columns:
                ids = df[id_column]
        
        blended += weights[i] * df[prediction_column].values
    
    # Create output
    if ids is not None:
        result_df = pd.DataFrame({
            id_column: ids,
            output_column: blended,
        })
    else:
        result_df = pd.DataFrame({
            output_column: blended,
        })
    
    os.makedirs(os.path.dirname(outputs["blended"]), exist_ok=True)
    save_data(result_df, outputs["blended"])
    
    return {
        "status": "success",
        "n_files_blended": len(prediction_files),
        "weights": weights,
        "n_predictions": len(result_df),
    }


# =============================================================================
# FEATURE ENGINEERING HELPERS
# =============================================================================
@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Create interaction features between numeric columns",
    tags=["feature_engineering", "interactions"],
)
def create_interaction_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    numeric_pairs: List[tuple] = None,
    operations: List[str] = None,
) -> Dict[str, Any]:
    """
    Create interaction features between numeric columns.
    
    Args:
        inputs: Dict with 'data' path
        outputs: Dict with 'data' path
        numeric_pairs: List of (col1, col2) tuples to create interactions for
        operations: List of operations ['multiply', 'divide', 'add', 'subtract']
        
    Returns:
        Dict with status and new feature count
    """
    if operations is None:
        operations = ["multiply", "divide"]
    
    df = load_data(inputs["data"])
    new_features = 0
    
    if numeric_pairs:
        for col1, col2 in numeric_pairs:
            if col1 in df.columns and col2 in df.columns:
                if "multiply" in operations:
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                    new_features += 1
                if "divide" in operations:
                    df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)
                    new_features += 1
                if "add" in operations:
                    df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
                    new_features += 1
                if "subtract" in operations:
                    df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
                    new_features += 1
    
    os.makedirs(os.path.dirname(outputs["data"]), exist_ok=True)
    save_data(df, outputs["data"])
    
    return {"status": "success", "new_features": new_features}


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Create binary indicator features based on conditions",
    tags=["feature_engineering", "binary_indicators"],
)
def create_binary_indicators(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    conditions: Dict[str, Dict] = None,
) -> Dict[str, Any]:
    """
    Create binary indicator features based on conditions.
    
    Args:
        inputs: Dict with 'data' path
        outputs: Dict with 'data' path
        conditions: Dict mapping new_col_name -> {"column": col, "op": ">", "value": 50}
        
    Returns:
        Dict with status and new feature count
    """
    df = load_data(inputs["data"])
    new_features = 0
    
    if conditions:
        for new_col, cond in conditions.items():
            col = cond["column"]
            op = cond["op"]
            value = cond["value"]
            
            if col in df.columns:
                if op == ">":
                    df[new_col] = (df[col] > value).astype(int)
                elif op == "<":
                    df[new_col] = (df[col] < value).astype(int)
                elif op == ">=":
                    df[new_col] = (df[col] >= value).astype(int)
                elif op == "<=":
                    df[new_col] = (df[col] <= value).astype(int)
                elif op == "==":
                    df[new_col] = (df[col] == value).astype(int)
                elif op == "!=":
                    df[new_col] = (df[col] != value).astype(int)
                new_features += 1
    
    os.makedirs(os.path.dirname(outputs["data"]), exist_ok=True)
    save_data(df, outputs["data"])
    
    return {"status": "success", "new_features": new_features}


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="One-hot encode categorical columns",
    tags=["feature_engineering", "encoding"],
)
def onehot_encode(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    drop_first: bool = False,
) -> Dict[str, Any]:
    """
    One-hot encode categorical columns.
    
    Args:
        inputs: Dict with 'data' path
        outputs: Dict with 'data' path
        columns: List of columns to one-hot encode
        drop_first: If True, drop first category to avoid multicollinearity
        
    Returns:
        Dict with status and new column count
    """
    df = load_data(inputs["data"])
    original_cols = len(df.columns)
    
    if columns:
        existing_cols = [c for c in columns if c in df.columns]
        if existing_cols:
            df = pd.get_dummies(df, columns=existing_cols, drop_first=drop_first)
    
    new_cols = len(df.columns) - original_cols + len(columns) if columns else 0
    
    os.makedirs(os.path.dirname(outputs["data"]), exist_ok=True)
    save_data(df, outputs["data"])
    
    return {"status": "success", "new_columns": new_cols}


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"artifact": {"format": "pickle", "schema": {"type": "artifact"}}},
    description="Fit a StandardScaler on numeric columns",
    tags=["preprocessing", "scaling"],
)
def fit_standard_scaler(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    exclude_columns: List[str] = None,
) -> Dict[str, Any]:
    """
    Fit a StandardScaler on numeric columns.
    
    Args:
        inputs: Dict with 'data' path
        outputs: Dict with 'artifact' path
        columns: List of columns to scale (if None, all numeric)
        exclude_columns: Columns to exclude from scaling
        
    Returns:
        Dict with status and columns scaled
    """
    from sklearn.preprocessing import StandardScaler
    
    df = load_data(inputs["data"])
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude_columns:
        columns = [c for c in columns if c not in exclude_columns]
    
    scaler = StandardScaler()
    scaler.fit(df[columns])
    
    artifact = {
        "scaler": scaler,
        "columns": columns,
    }
    
    os.makedirs(os.path.dirname(outputs["artifact"]), exist_ok=True)
    with open(outputs["artifact"], "wb") as f:
        pickle.dump(artifact, f)
    
    return {"status": "success", "columns_fitted": columns}


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "artifact": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
    },
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Transform data using a fitted StandardScaler",
    tags=["preprocessing", "scaling"],
)
def transform_standard_scaler(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> Dict[str, Any]:
    """
    Transform data using a fitted StandardScaler.
    
    Args:
        inputs: Dict with 'data' and 'artifact' paths
        outputs: Dict with 'data' path
        
    Returns:
        Dict with status and columns transformed
    """
    df = load_data(inputs["data"])
    
    with open(inputs["artifact"], "rb") as f:
        artifact = pickle.load(f)
    
    scaler = artifact["scaler"]
    columns = artifact["columns"]
    
    # Only transform columns that exist
    existing_cols = [c for c in columns if c in df.columns]
    df[existing_cols] = scaler.transform(df[existing_cols])
    
    os.makedirs(os.path.dirname(outputs["data"]), exist_ok=True)
    save_data(df, outputs["data"])
    
    return {"status": "success", "columns_transformed": existing_cols}


# =============================================================================
# SERVICE REGISTRY
# =============================================================================
SERVICE_REGISTRY = {
    "train_weighted_ensemble_classifier": train_weighted_ensemble_classifier,
    "train_weighted_ensemble_regressor": train_weighted_ensemble_regressor,
    "predict_ensemble": predict_ensemble,
    "blend_predictions": blend_predictions,
    "create_interaction_features": create_interaction_features,
    "create_binary_indicators": create_binary_indicators,
    "onehot_encode": onehot_encode,
    "fit_standard_scaler": fit_standard_scaler,
    "transform_standard_scaler": transform_standard_scaler,
}
