"""
Playground Series S3E18 - SLEGO Services
=========================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e18
Problem Type: Multi-label binary classification
Targets: EC1, EC2 (enzyme substrate classification probabilities)
ID Column: id
Evaluation Metric: Mean column-wise ROC AUC = (AUC(EC1) + AUC(EC2)) / 2

Predict enzyme substrate class probabilities from molecular descriptors.
Features: 31 chemical descriptors (BertzCT, Chi1, ExactMolWt, etc.)
Train also has EC3-EC6 but submission only requires EC1, EC2.

Solution Notebook Insights:
- 1st (muhannadmansour): StandardScaler + AdaBoost per target, submit probabilities
- 2nd (sjagkoo7): Outlier capping + StandardScaler + model comparison (CatBoost/XGB/LGBM best)
  Trains separate models for EC1 and EC2, uses predict_proba
- 3rd (thomasmeiner): BlueCast + CatBoost with Optuna tuning, separate models per target,
  merges original dataset, 5-fold CV

Competition-specific services:
- train_multi_target_classifier: Train separate binary classifiers per target (generic, reusable)
- predict_multi_target_classifier: Generate probability predictions for each target (generic, reusable)
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
    from services.preprocessing_services import split_data, drop_columns, fit_scaler, transform_scaler
    from services.classification_services import train_lightgbm_classifier, predict_classifier
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from preprocessing_services import split_data, drop_columns, fit_scaler, transform_scaler
    from classification_services import train_lightgbm_classifier, predict_classifier


# =============================================================================
# MULTI-TARGET BINARY CLASSIFIER (Generic, Reusable)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train separate binary classifiers for multiple target columns (multi-label)",
    tags=["modeling", "training", "multi-label", "classification", "generic"],
    version="1.0.0",
)
def train_multi_target_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_columns: List[str] = None,
    id_column: str = "id",
    exclude_columns: Optional[List[str]] = None,
    model_type: str = "lightgbm",
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 64,
    max_depth: int = -1,
    min_child_samples: int = 30,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    early_stopping_rounds: int = 100,
    random_state: int = 42,
) -> str:
    """Train separate binary classifiers for each target column.

    For multi-label binary classification where each target is independent.
    Trains one model per target, excluding all other targets from features
    to prevent data leakage.

    G1 Compliance: Generic, works with any multi-label dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All column names and hyperparameters as parameters.

    Parameters:
        target_columns: List of target column names to train models for
        id_column: ID column to exclude from features
        exclude_columns: Additional columns to exclude from features
        model_type: 'lightgbm', 'xgboost', or 'catboost'
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate
        num_leaves: Number of leaves (LightGBM)
        max_depth: Max tree depth (-1 for no limit)
        early_stopping_rounds: Early stopping patience
    """
    if target_columns is None:
        target_columns = ["EC1", "EC2"]

    train_df = _load_data(inputs["train_data"])

    # Build exclusion set: all targets + id + any extra
    drop_cols = set(target_columns)
    if id_column and id_column in train_df.columns:
        drop_cols.add(id_column)
    if exclude_columns:
        drop_cols.update(c for c in exclude_columns if c in train_df.columns)

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    feature_cols = list(X_train.columns)

    X_valid, valid_df = None, None
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=[c for c in drop_cols if c in valid_df.columns])

    trained_models = {}
    per_target_metrics = {}
    aucs = []

    for target in target_columns:
        if target not in train_df.columns:
            continue

        y_train = train_df[target].astype(int)

        if model_type == "lightgbm":
            import lightgbm as lgb
            from sklearn.metrics import roc_auc_score

            model = lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                num_leaves=num_leaves, max_depth=max_depth,
                min_child_samples=min_child_samples, subsample=subsample,
                colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                reg_lambda=reg_lambda, random_state=random_state,
                n_jobs=-1, objective="binary", verbose=-1,
            )

            if X_valid is not None and target in valid_df.columns:
                y_valid = valid_df[target].astype(int)
                callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                          eval_metric="auc", callbacks=callbacks)
                preds = model.predict_proba(X_valid)[:, 1]
                auc = float(roc_auc_score(y_valid, preds))
                per_target_metrics[target] = {
                    "roc_auc": auc,
                    "best_iteration": int(getattr(model, "best_iteration_", n_estimators)),
                }
                aucs.append(auc)
            else:
                model.fit(X_train, y_train)

        elif model_type == "xgboost":
            import xgboost as xgb
            from sklearn.metrics import roc_auc_score

            model = xgb.XGBClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth if max_depth > 0 else 6,
                subsample=subsample, colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                random_state=random_state, n_jobs=-1,
                objective="binary:logistic", eval_metric="auc",
                use_label_encoder=False, verbosity=0,
            )

            if X_valid is not None and target in valid_df.columns:
                y_valid = valid_df[target].astype(int)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                preds = model.predict_proba(X_valid)[:, 1]
                auc = float(roc_auc_score(y_valid, preds))
                per_target_metrics[target] = {
                    "roc_auc": auc,
                    "best_iteration": int(getattr(model, "best_iteration", n_estimators)),
                }
                aucs.append(auc)
            else:
                model.fit(X_train, y_train)

        elif model_type == "catboost":
            from catboost import CatBoostClassifier
            from sklearn.metrics import roc_auc_score

            # CatBoost: fast, robust, handles missing values well
            model = CatBoostClassifier(
                iterations=n_estimators, learning_rate=learning_rate,
                depth=max_depth if max_depth > 0 else 8,
                l2_leaf_reg=reg_lambda if reg_lambda > 0 else 3.0,
                random_seed=random_state,
                verbose=0, thread_count=-1,
            )

            if X_valid is not None and target in valid_df.columns:
                y_valid = valid_df[target].astype(int)
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                          early_stopping_rounds=early_stopping_rounds, verbose=False)
                preds = model.predict_proba(X_valid)[:, 1]
                auc = float(roc_auc_score(y_valid, preds))
                per_target_metrics[target] = {
                    "roc_auc": auc,
                    "best_iteration": int(getattr(model, "best_iteration_", n_estimators)),
                }
                aucs.append(auc)
            else:
                model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'lightgbm', 'xgboost', or 'catboost'.")

        trained_models[target] = model

    mean_auc = float(np.mean(aucs)) if aucs else None

    artifact = {
        "models": trained_models, "feature_cols": feature_cols,
        "target_columns": target_columns, "model_type": model_type,
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    metrics = {
        "model_type": f"multi_target_{model_type}",
        "target_columns": target_columns,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "per_target": per_target_metrics,
        "mean_roc_auc": mean_auc,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }
    if X_valid is not None:
        metrics["n_valid_samples"] = len(X_valid)

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    target_aucs = ", ".join(f"{t}={m.get('roc_auc', 'N/A')}" for t, m in per_target_metrics.items())
    auc_str = f", mean AUC={mean_auc:.4f}" if mean_auc else ""
    return f"train_multi_target_classifier: {len(target_columns)} targets ({target_aucs}){auc_str}"


# =============================================================================
# MULTI-TARGET PREDICTION (Generic, Reusable)
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Generate probability predictions for multiple targets using trained multi-target model",
    tags=["modeling", "prediction", "multi-label", "inference", "generic"],
    version="1.0.0",
)
def predict_multi_target_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    clip_probabilities: bool = True,
    clip_min: float = 0.01,
    clip_max: float = 0.99,
) -> str:
    """Generate probability predictions for each target column.

    Loads multi-target model artifact and produces probability predictions
    suitable for log-loss or AUC-scored Kaggle competitions.

    For log-loss competitions, probability clipping is CRITICAL to avoid
    extreme penalties from overly confident wrong predictions.

    G1 Compliance: Generic, works with any multi-target model artifact.
    G4 Compliance: id_column and clipping parameters parameterized.

    Parameters:
        id_column: ID column name
        clip_probabilities: Whether to clip predictions (recommended for log loss)
        clip_min: Minimum probability value (default 0.01)
        clip_max: Maximum probability value (default 0.99)
    """
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    data_df = _load_data(inputs["data"])
    models = artifact["models"]
    feature_cols = artifact["feature_cols"]
    target_columns = artifact["target_columns"]

    for col in feature_cols:
        if col not in data_df.columns:
            data_df[col] = 0
    X = data_df[feature_cols]

    result = pd.DataFrame()
    if id_column and id_column in data_df.columns:
        result[id_column] = data_df[id_column]

    for target in target_columns:
        if target in models:
            probs = models[target].predict_proba(X)[:, 1]
            if clip_probabilities:
                probs = np.clip(probs, clip_min, clip_max)
            result[target] = probs

    _save_data(result, outputs["predictions"])

    clip_str = f" (clipped to [{clip_min}, {clip_max}])" if clip_probabilities else ""
    mean_probas = ", ".join(f"{t}={result[t].mean():.4f}" for t in target_columns if t in result.columns)
    return f"predict_multi_target_classifier: {len(result)} predictions{clip_str}, mean_proba=[{mean_probas}]"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific (generic, reusable for any multi-label)
    "train_multi_target_classifier": train_multi_target_classifier,
    "predict_multi_target_classifier": predict_multi_target_classifier,
    # Imported from common modules
    "split_data": split_data,
    "drop_columns": drop_columns,
    "fit_scaler": fit_scaler,
    "transform_scaler": transform_scaler,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
}


# =============================================================================
# PIPELINE SPECIFICATION (mirrors pipeline.json)
# =============================================================================

PIPELINE_SPEC = {
    "name": "playground-series-s3e18",
    "description": "Multi-label enzyme classification - EC1/EC2 probability prediction with LightGBM",
    "version": "2.0.0",
    "problem_type": "multi-label",
    "target_column": "EC1,EC2",
    "id_column": "id",
    "evaluation_metric": "mean_column_wise_roc_auc",
    "steps": [
        {
            "service": "drop_columns",
            "inputs": {"data": "playground-series-s3e18/datasets/train.csv"},
            "outputs": {"data": "playground-series-s3e18/artifacts/train_clean.csv"},
            "params": {"columns": ["EC3", "EC4", "EC5", "EC6"]},
            "module": "preprocessing_services",
        },
        {
            "service": "split_data",
            "inputs": {"data": "playground-series-s3e18/artifacts/train_clean.csv"},
            "outputs": {
                "train_data": "playground-series-s3e18/artifacts/train_split.csv",
                "valid_data": "playground-series-s3e18/artifacts/valid_split.csv",
            },
            "params": {"test_size": 0.2, "random_state": 42, "stratify_column": "EC1"},
            "module": "preprocessing_services",
        },
        {
            "service": "fit_scaler",
            "inputs": {"data": "playground-series-s3e18/artifacts/train_split.csv"},
            "outputs": {"artifact": "playground-series-s3e18/artifacts/scaler.pkl"},
            "params": {"method": "standard", "exclude_columns": ["id", "EC1", "EC2"]},
            "module": "preprocessing_services",
        },
        {
            "service": "transform_scaler",
            "inputs": {
                "data": "playground-series-s3e18/artifacts/train_split.csv",
                "artifact": "playground-series-s3e18/artifacts/scaler.pkl",
            },
            "outputs": {"data": "playground-series-s3e18/artifacts/train_scaled.csv"},
            "module": "preprocessing_services",
        },
        {
            "service": "transform_scaler",
            "inputs": {
                "data": "playground-series-s3e18/artifacts/valid_split.csv",
                "artifact": "playground-series-s3e18/artifacts/scaler.pkl",
            },
            "outputs": {"data": "playground-series-s3e18/artifacts/valid_scaled.csv"},
            "module": "preprocessing_services",
        },
        {
            "service": "transform_scaler",
            "inputs": {
                "data": "playground-series-s3e18/datasets/test.csv",
                "artifact": "playground-series-s3e18/artifacts/scaler.pkl",
            },
            "outputs": {"data": "playground-series-s3e18/artifacts/test_scaled.csv"},
            "module": "preprocessing_services",
        },
        {
            "service": "train_multi_target_classifier",
            "inputs": {
                "train_data": "playground-series-s3e18/artifacts/train_scaled.csv",
                "valid_data": "playground-series-s3e18/artifacts/valid_scaled.csv",
            },
            "outputs": {
                "model": "playground-series-s3e18/artifacts/model.pkl",
                "metrics": "playground-series-s3e18/artifacts/metrics.json",
            },
            "params": {
                "target_columns": ["EC1", "EC2"],
                "id_column": "id",
                "model_type": "lightgbm",
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 64,
                "max_depth": -1,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 100,
            },
            "module": "playground_s3e18_services",
        },
        {
            "service": "predict_multi_target_classifier",
            "inputs": {
                "model": "playground-series-s3e18/artifacts/model.pkl",
                "data": "playground-series-s3e18/artifacts/test_scaled.csv",
            },
            "outputs": {
                "predictions": "playground-series-s3e18/submission.csv",
            },
            "params": {"id_column": "id"},
            "module": "playground_s3e18_services",
        },
    ],
}
