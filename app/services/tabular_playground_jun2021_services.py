"""
Contract-Composable Analytics Services for tabular-playground-series-jun-2021
Multiclass Classification - Target: target (Class_1 to Class_9)
75 anonymous count-like features (feature_0 to feature_74)

Competition metric: Multi-class Log Loss
Submission format: id, Class_1, Class_2, ..., Class_9 (probabilities)

Based on winning solutions that use:
- Multiple model ensemble (LightGBM, XGBoost, CatBoost)
- Weighted blending with optimized weights
- Log1p transformation for count features
- Row-wise statistics
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from contract import contract


def _load_data(path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(path)


def _save_data(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# Import reusable generic services
# =============================================================================
from services.preprocessing_services import split_data
from services.classification_services import (
    train_lightgbm_classifier,
    train_xgboost_classifier,
    predict_classifier
)


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Encode TPS-Jun-2021 class target (Class_1 -> 0, Class_2 -> 1, etc.)",
    tags=["preprocessing", "encoding", "tps-jun-2021"],
    version="1.0.0",
)
def encode_class_target(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    prefix: str = "Class_",
) -> str:
    """Encode class target (Class_1 -> 0, Class_2 -> 1, etc.)"""
    df = _load_data(inputs["data"])

    if target_column in df.columns:
        df[target_column] = df[target_column].str.replace(prefix, '', regex=False).astype(int) - 1

    _save_data(df, outputs["data"])
    n_classes = df[target_column].nunique() if target_column in df.columns else 0
    return f"encode_class_target: {len(df)} rows, {n_classes} classes"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Apply log1p transformation and add row statistics for count-like features",
    tags=["preprocessing", "feature-engineering", "tps-jun-2021"],
    version="1.0.0",
)
def preprocess_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    feature_prefix: str = "feature_",
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """Apply log1p transformation and add row statistics.

    Based on winning solutions that transform count-like features.
    """
    df = _load_data(inputs["data"])

    if exclude_columns is None:
        exclude_columns = ["id", "target"]

    feature_cols = [c for c in df.columns
                    if c.startswith(feature_prefix) and c not in exclude_columns]

    # Apply log1p transformation
    for col in feature_cols:
        df[col] = np.log1p(df[col])

    # Add row-wise statistics
    if feature_cols:
        df["row_sum"] = df[feature_cols].sum(axis=1)
        df["row_mean"] = df[feature_cols].mean(axis=1)
        df["row_std"] = df[feature_cols].std(axis=1)
        df["row_max"] = df[feature_cols].max(axis=1)
        df["row_min"] = df[feature_cols].min(axis=1)
        df["row_nonzero"] = (df[feature_cols] > 0).sum(axis=1)

    _save_data(df, outputs["data"])
    return f"preprocess_features: log1p on {len(feature_cols)} columns, added 6 row statistics"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train multiclass ensemble (LightGBM + XGBoost + HistGradientBoosting) with optimized weighted blending",
    tags=["modeling", "training", "ensemble", "multiclass", "tps-jun-2021"],
    version="2.1.0",
)
def train_multiclass_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    id_column: str = "id",
    n_classes: int = 9,
    lgb_weight: float = 0.45,
    xgb_weight: float = 0.35,
    hgb_weight: float = 0.20,
    n_estimators: int = 1500,
    learning_rate: float = 0.02,
    random_state: int = 42,
) -> str:
    """Train ensemble of LightGBM, XGBoost, and HistGradientBoosting for multiclass classification.

    Based on winning solutions that blend multiple gradient boosting models.
    Optimized for multi-class log loss. Uses sklearn's HistGradientBoostingClassifier
    which is fast, reliable, and excellent for tabular data.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss

    train_df = _load_data(inputs["train_data"])

    # Prepare features
    drop_cols = [target_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[target_column].astype(int)
    feature_cols = list(X_train.columns)

    # Load validation data if available
    X_valid, y_valid = None, None
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        y_valid = valid_df[target_column].astype(int)

    models = []
    weights = [lgb_weight, xgb_weight, hgb_weight]
    weights = [w / sum(weights) for w in weights]  # Normalize

    metrics = {
        "model_type": "multiclass_ensemble_v2",
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_classes": n_classes,
        "weights": weights,
    }

    # 1. Train LightGBM with optimized params from winning solutions
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=128,
        max_depth=10,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=random_state,
        n_jobs=-1,
        objective="multiclass",
        num_class=n_classes,
        verbosity=-1,
    )

    if X_valid is not None:
        callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)]
        lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      eval_metric="multi_logloss", callbacks=callbacks)
    else:
        lgb_model.fit(X_train, y_train)

    models.append(lgb_model)

    # 2. Train XGBoost with optimized params
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=10,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=random_state,
        n_jobs=-1,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        verbosity=0,
        early_stopping_rounds=150 if X_valid is not None else None,
    )

    if X_valid is not None:
        xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      verbose=False)
    else:
        xgb_model.fit(X_train, y_train)

    models.append(xgb_model)

    # 3. Train HistGradientBoosting (sklearn's fast native implementation)
    print("Training HistGradientBoosting...")
    hgb_model = HistGradientBoostingClassifier(
        max_iter=n_estimators,
        learning_rate=learning_rate,
        max_depth=10,
        min_samples_leaf=15,
        l2_regularization=0.1,
        random_state=random_state,
        early_stopping=True if X_valid is not None else False,
        validation_fraction=0.1 if X_valid is None else None,
        n_iter_no_change=150,
        verbose=0,
    )

    if X_valid is not None:
        # HistGradientBoosting doesn't support eval_set directly, use validation_fraction
        hgb_model.set_params(early_stopping=True, validation_fraction=None)
        # Combine train and valid, then let it use internal validation
        X_combined = pd.concat([X_train, X_valid], ignore_index=True)
        y_combined = pd.concat([y_train, y_valid], ignore_index=True)
        hgb_model.set_params(validation_fraction=len(X_valid) / len(X_combined))
        hgb_model.fit(X_combined, y_combined)
    else:
        hgb_model.fit(X_train, y_train)

    models.append(hgb_model)

    # Compute blended validation score
    if X_valid is not None:
        blended = np.zeros((len(X_valid), n_classes))
        for m, w in zip(models, weights):
            blended += w * m.predict_proba(X_valid)

        val_logloss = log_loss(y_valid, blended)
        metrics["blended_logloss"] = float(val_logloss)
        metrics["n_valid_samples"] = len(X_valid)

        # Individual model scores
        for i, (name, m) in enumerate(zip(["lgb", "xgb", "hgb"], models)):
            pred = m.predict_proba(X_valid)
            metrics[f"{name}_logloss"] = float(log_loss(y_valid, pred))

    # Save ensemble artifact
    artifact = {
        "models": models,
        "weights": weights,
        "feature_cols": feature_cols,
        "model_types": ["lightgbm", "xgboost", "histgradientboosting"],
        "n_classes": n_classes,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    logloss_str = f", logloss={metrics.get('blended_logloss', 'N/A'):.4f}" if 'blended_logloss' in metrics else ""
    return f"train_multiclass_ensemble: 3 models (LGB+XGB+HGB), {len(X_train)} samples{logloss_str}"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Predict multiclass probabilities and create submission CSV",
    tags=["prediction", "multiclass", "submission", "tps-jun-2021"],
    version="1.0.0",
)
def predict_multiclass_proba(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    class_prefix: str = "Class_",
    n_classes: int = 9,
) -> str:
    """Predict multiclass probabilities and output as submission CSV.

    Output format: id, Class_1, Class_2, ..., Class_9
    Handles both single-model and ensemble artifacts.
    """
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    data_df = _load_data(inputs["data"])

    # Extract IDs
    if id_column in data_df.columns:
        ids = data_df[id_column].values
    else:
        ids = np.arange(1, len(data_df) + 1)

    # Prepare feature matrix
    feature_cols = artifact.get("feature_cols")
    if feature_cols:
        for col in feature_cols:
            if col not in data_df.columns:
                data_df[col] = 0
        X = data_df[feature_cols]
    else:
        drop_cols = [id_column] if id_column in data_df.columns else []
        X = data_df.drop(columns=drop_cols, errors="ignore")

    # Generate probability predictions
    if "models" in artifact and "weights" in artifact:
        models = artifact["models"]
        weights = artifact["weights"]
        proba = np.zeros((len(X), n_classes))
        for m, w in zip(models, weights):
            proba += w * m.predict_proba(X)
    else:
        model = artifact["model"]
        proba = model.predict_proba(X)

    # Create submission DataFrame: id, Class_1, Class_2, ..., Class_N
    class_columns = [f"{class_prefix}{i + 1}" for i in range(n_classes)]
    submission = pd.DataFrame(proba, columns=class_columns)
    submission.insert(0, id_column, ids)

    _save_data(submission, outputs["submission"])
    return f"predict_multiclass_proba: {len(submission)} predictions, {n_classes} classes"


# =============================================================================
# Service Registry
# =============================================================================
SERVICE_REGISTRY = {
    # Competition-specific
    'encode_class_target': encode_class_target,
    'preprocess_features': preprocess_features,
    'train_multiclass_ensemble': train_multiclass_ensemble,
    'predict_multiclass_proba': predict_multiclass_proba,
    # Generic (imported)
    'split_data': split_data,
    'train_lightgbm_classifier': train_lightgbm_classifier,
    'train_xgboost_classifier': train_xgboost_classifier,
    'predict_classifier': predict_classifier,
}
