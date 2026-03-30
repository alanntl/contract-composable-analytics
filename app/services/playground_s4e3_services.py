"""
Steel Plate Defect Prediction (Playground Series S4E3) - Contract-Composable Analytics Services
=======================================================================
Competition: https://www.kaggle.com/competitions/playground-series-s4e3
Problem Type: Multi-label binary classification
Targets: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults
Evaluation: Mean column-wise ROC AUC

Solution notebook insights:
- Solution 01 (CatBoost): Train separate binary classifiers per target,
  use predict_proba for probability outputs.
- Solution 02 (Neural Network): StandardScaler + oversampling + softmax NN.
- Solution 03 (Neural Network): Simple Keras model with softmax.

Competition-specific services:
- train_multilabel_classifier: Train separate binary classifiers per target (reusable)
- predict_multilabel_submission: Generate multi-column probability submission (reusable)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

try:
    from services.io_utils import load_data as _load_data, save_data as _save_data
    from services.preprocessing_services import split_data
    from services.classification_services import (
        train_lightgbm_classifier,
        predict_classifier,
    )
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from preprocessing_services import split_data
    from classification_services import (
        train_lightgbm_classifier,
        predict_classifier,
    )


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "valid_data": {"format": "csv", "required": False},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train separate binary classifiers for each target in a multi-label problem",
    tags=["modeling", "training", "multilabel", "classification", "generic"],
    version="1.0.0",
)
def train_multilabel_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_columns: List[str] = None,
    id_column: str = "id",
    model_type: str = "lightgbm",
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 64,
    max_depth: int = -1,
    random_state: int = 42,
    early_stopping_rounds: int = 100,
) -> str:
    """Train one binary classifier per target column for multi-label problems.

    Works with: steel defect detection, multi-label text classification,
    any problem requiring independent binary predictions per column.
    """
    if target_columns is None:
        target_columns = ["Pastry", "Z_Scratch", "K_Scatch", "Stains",
                          "Dirtiness", "Bumps", "Other_Faults"]

    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])

    drop_cols = list(target_columns)
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_train.columns)

    X_valid, valid_targets = None, {}
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        for tc in target_columns:
            if tc in valid_df.columns:
                valid_targets[tc] = valid_df[tc].astype(int)

    models = {}
    per_target_metrics = []

    for tc in target_columns:
        y_train = train_df[tc].astype(int)

        if model_type == "lightgbm":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                num_leaves=num_leaves, max_depth=max_depth,
                random_state=random_state, n_jobs=-1,
                objective="binary", verbose=-1,
            )
            if X_valid is not None and tc in valid_targets:
                callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, valid_targets[tc])],
                          eval_metric="auc", callbacks=callbacks)
            else:
                model.fit(X_train, y_train)

        elif model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth if max_depth > 0 else 6,
                random_state=random_state, n_jobs=-1,
                objective="binary:logistic", eval_metric="auc", verbosity=0,
            )
            if X_valid is not None and tc in valid_targets:
                model.fit(X_train, y_train,
                          eval_set=[(X_valid, valid_targets[tc])],
                          verbose=False)
            else:
                model.fit(X_train, y_train)

        elif model_type == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                iterations=n_estimators, learning_rate=learning_rate,
                depth=max_depth if max_depth > 0 else 6,
                random_state=random_state, verbose=0,
            )
            if X_valid is not None and tc in valid_targets:
                model.fit(X_train, y_train, eval_set=(X_valid, valid_targets[tc]))
            else:
                model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        models[tc] = model

        target_info = {"target": tc, "model_type": model_type}
        if X_valid is not None and tc in valid_targets:
            preds = model.predict_proba(X_valid)[:, 1]
            target_info["roc_auc"] = float(roc_auc_score(valid_targets[tc], preds))
        per_target_metrics.append(target_info)

    aucs = [m["roc_auc"] for m in per_target_metrics if "roc_auc" in m]
    mean_auc = float(np.mean(aucs)) if aucs else None

    metrics = {
        "model_type": f"multilabel_{model_type}",
        "n_targets": len(target_columns),
        "target_columns": target_columns,
        "per_target": per_target_metrics,
        "mean_roc_auc": mean_auc,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }
    if X_valid is not None:
        metrics["n_valid_samples"] = len(X_valid)

    artifact = {
        "models": models, "feature_cols": feature_cols,
        "target_columns": target_columns, "model_type": model_type,
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    auc_str = f", mean AUC={mean_auc:.4f}" if mean_auc else ""
    return f"train_multilabel_classifier: {len(target_columns)} targets, {len(X_train)} samples{auc_str}"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Generate multi-label probability submission for Kaggle competitions",
    tags=["prediction", "multilabel", "submission", "classification", "generic"],
    version="1.0.0",
)
def predict_multilabel_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
) -> str:
    """Generate submission with per-target probability columns for multi-label problems."""
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

    submission = pd.DataFrame()
    if id_column and id_column in data_df.columns:
        submission[id_column] = data_df[id_column]

    for tc in target_columns:
        submission[tc] = models[tc].predict_proba(X)[:, 1]

    _save_data(submission, outputs["submission"])
    return f"predict_multilabel_submission: {len(submission)} rows, {len(target_columns)} targets"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "valid_data": {"format": "csv", "required": False},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train ensemble of binary classifiers (CatBoost+LightGBM+XGBoost) for multi-label problems",
    tags=["modeling", "training", "multilabel", "ensemble", "classification", "generic"],
    version="1.0.0",
)
def train_ensemble_multilabel_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_columns: List[str] = None,
    id_column: str = "id",
    model_types: List[str] = None,
    weights: List[float] = None,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    random_state: int = 42,
    early_stopping_rounds: int = 100,
) -> str:
    """Train ensemble of models per target column for multi-label problems.

    Combines predictions from multiple model types (CatBoost, LightGBM, XGBoost)
    via weighted averaging for more robust probability estimates.

    Works with: steel defect detection, multi-label text classification,
    any problem requiring independent binary predictions per column.
    """
    if target_columns is None:
        target_columns = ["Pastry", "Z_Scratch", "K_Scatch", "Stains",
                          "Dirtiness", "Bumps", "Other_Faults"]

    if model_types is None:
        model_types = ["catboost", "lightgbm", "xgboost"]

    if weights is None:
        weights = [1.0 / len(model_types)] * len(model_types)

    # Normalize weights
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])

    drop_cols = list(target_columns)
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_train.columns)

    X_valid, valid_targets = None, {}
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        for tc in target_columns:
            if tc in valid_df.columns:
                valid_targets[tc] = valid_df[tc].astype(int)

    ensemble_models = {}  # {target: {model_type: model}}
    per_target_metrics = []

    for tc in target_columns:
        y_train = train_df[tc].astype(int)
        ensemble_models[tc] = {}
        target_aucs = []

        for mtype in model_types:
            if mtype == "catboost":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(
                    iterations=n_estimators, learning_rate=learning_rate,
                    depth=6, random_state=random_state, verbose=0,
                )
                if X_valid is not None and tc in valid_targets:
                    model.fit(X_train, y_train, eval_set=(X_valid, valid_targets[tc]))
                else:
                    model.fit(X_train, y_train)

            elif mtype == "lightgbm":
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    num_leaves=64, max_depth=-1,
                    random_state=random_state, n_jobs=-1,
                    objective="binary", verbose=-1,
                )
                if X_valid is not None and tc in valid_targets:
                    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                    model.fit(X_train, y_train,
                              eval_set=[(X_valid, valid_targets[tc])],
                              eval_metric="auc", callbacks=callbacks)
                else:
                    model.fit(X_train, y_train)

            elif mtype == "xgboost":
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    max_depth=6, random_state=random_state, n_jobs=-1,
                    objective="binary:logistic", eval_metric="auc", verbosity=0,
                )
                if X_valid is not None and tc in valid_targets:
                    model.fit(X_train, y_train,
                              eval_set=[(X_valid, valid_targets[tc])],
                              verbose=False)
                else:
                    model.fit(X_train, y_train)

            else:
                raise ValueError(f"Unsupported model_type: {mtype}")

            ensemble_models[tc][mtype] = model

            if X_valid is not None and tc in valid_targets:
                preds = model.predict_proba(X_valid)[:, 1]
                auc = float(roc_auc_score(valid_targets[tc], preds))
                target_aucs.append(auc)

        # Compute blended AUC for this target
        target_info = {"target": tc, "model_types": model_types}
        if X_valid is not None and tc in valid_targets and target_aucs:
            # Blend predictions
            blended = np.zeros(len(X_valid))
            for mtype, w in zip(model_types, weights):
                blended += w * ensemble_models[tc][mtype].predict_proba(X_valid)[:, 1]
            target_info["blended_roc_auc"] = float(roc_auc_score(valid_targets[tc], blended))
            target_info["per_model_auc"] = dict(zip(model_types, target_aucs))
        per_target_metrics.append(target_info)

    aucs = [m["blended_roc_auc"] for m in per_target_metrics if "blended_roc_auc" in m]
    mean_auc = float(np.mean(aucs)) if aucs else None

    metrics = {
        "model_type": "ensemble_multilabel",
        "model_types": model_types,
        "weights": weights,
        "n_targets": len(target_columns),
        "target_columns": target_columns,
        "per_target": per_target_metrics,
        "mean_blended_roc_auc": mean_auc,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }
    if X_valid is not None:
        metrics["n_valid_samples"] = len(X_valid)

    artifact = {
        "ensemble_models": ensemble_models,
        "feature_cols": feature_cols,
        "target_columns": target_columns,
        "model_types": model_types,
        "weights": weights,
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    auc_str = f", mean blended AUC={mean_auc:.4f}" if mean_auc else ""
    return f"train_ensemble_multilabel_classifier: {len(target_columns)} targets, {len(model_types)} models{auc_str}"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Generate blended multi-label probability submission from ensemble models",
    tags=["prediction", "multilabel", "ensemble", "submission", "classification", "generic"],
    version="1.0.0",
)
def predict_ensemble_multilabel_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
) -> str:
    """Generate submission with blended per-target probability columns from ensemble."""
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    data_df = _load_data(inputs["data"])
    ensemble_models = artifact["ensemble_models"]
    feature_cols = artifact["feature_cols"]
    target_columns = artifact["target_columns"]
    model_types = artifact["model_types"]
    weights = artifact["weights"]

    for col in feature_cols:
        if col not in data_df.columns:
            data_df[col] = 0
    X = data_df[feature_cols]

    submission = pd.DataFrame()
    if id_column and id_column in data_df.columns:
        submission[id_column] = data_df[id_column]

    for tc in target_columns:
        blended = np.zeros(len(X))
        for mtype, w in zip(model_types, weights):
            blended += w * ensemble_models[tc][mtype].predict_proba(X)[:, 1]
        submission[tc] = blended

    _save_data(submission, outputs["submission"])
    return f"predict_ensemble_multilabel_submission: {len(submission)} rows, {len(target_columns)} targets"


SERVICE_REGISTRY = {
    "split_data": split_data,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "train_multilabel_classifier": train_multilabel_classifier,
    "predict_multilabel_submission": predict_multilabel_submission,
    "train_ensemble_multilabel_classifier": train_ensemble_multilabel_classifier,
    "predict_ensemble_multilabel_submission": predict_ensemble_multilabel_submission,
}