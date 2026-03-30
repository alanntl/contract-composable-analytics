"""
Contract-Composable Analytics Training Services - Unified Model Training Interface
==========================================================
This module provides a unified interface for training ML models.
Instead of separate train_lightgbm, train_xgboost, etc., use train_model().

Usage:
    from services.training_services import train_model, predict
"""
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from functools import wraps

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract


def _save_artifact(obj: Any, path: str) -> None:
    """Save artifact with auto-detected format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=str)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


def _detect_problem_type(y: pd.Series) -> str:
    """Auto-detect problem type from target column."""
    n_unique = y.nunique()

    if y.dtype in ['float64', 'float32']:
        if n_unique > 20:
            return 'regression'

    if n_unique == 2:
        return 'binary'
    elif n_unique <= 20:
        return 'multiclass'
    else:
        return 'regression'


# =============================================================================
# Unified Training Interface
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "valid_data": {"format": "csv", "required": False}
    },
    outputs={
        "model": {"format": "pkl"},
        "metrics": {"format": "json"}
    },
    description="Unified model training interface",
    tags=["training", "model", "generic", "unified"]
)
def train_model(
    train_data: str,
    model_output: str,
    metrics_output: str,
    valid_data: str = None,
    model_type: str = 'lightgbm',
    problem_type: str = 'auto',
    target_column: str = 'target',
    id_column: str = None,
    exclude_columns: List[str] = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = None,
    random_state: int = 42,
    auto_encode: bool = False,
    **model_kwargs
) -> Dict[str, str]:
    """
    Unified model training service. Supports multiple model types and auto-detects problem type.

    Works with: any tabular ML competition.

    Args:
        train_data: Path to training data CSV
        model_output: Path to save model
        metrics_output: Path to save metrics
        valid_data: Optional path to validation data
        model_type: 'lightgbm', 'xgboost', 'random_forest', 'catboost', 'gradient_boosting'
        problem_type: 'auto' (detect), 'regression', 'binary', 'multiclass'
        target_column: Name of target column
        id_column: ID column to exclude
        exclude_columns: Additional columns to exclude from features
        n_estimators: Number of estimators
        learning_rate: Learning rate (for boosting models)
        max_depth: Max depth (None = no limit)
        random_state: Random seed
        **model_kwargs: Additional model-specific parameters

    Returns:
        Dict with model and metrics output paths
    """
    # Load data
    train_df = pd.read_csv(train_data)
    valid_df = pd.read_csv(valid_data) if valid_data and os.path.exists(valid_data) else None

    # Prepare features
    exclude = set([target_column])
    if id_column:
        exclude.add(id_column)
    if exclude_columns:
        exclude.update(exclude_columns)

    feature_cols = [c for c in train_df.columns if c not in exclude]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_column]

    # Handle non-numeric columns
    object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        if auto_encode:
            import warnings
            warnings.warn(
                f"auto_encode=True: Automatically encoding {len(object_cols)} categorical columns. "
                f"Consider using fit_encoder/transform_encoder for explicit encoding (G2 compliance).",
                UserWarning
            )
            for col in object_cols:
                X_train[col] = pd.factorize(X_train[col])[0]
        else:
            raise ValueError(
                f"Non-numeric columns found: {object_cols}. "
                f"Either: (1) Add encoding step to pipeline (recommended), or "
                f"(2) Set auto_encode=True (not recommended for production)."
            )

    if valid_df is not None:
        X_valid = valid_df[feature_cols].copy()
        y_valid = valid_df[target_column]
        valid_object_cols = X_valid.select_dtypes(include=['object']).columns.tolist()
        if valid_object_cols and auto_encode:
            for col in valid_object_cols:
                X_valid[col] = pd.factorize(X_valid[col])[0]
        elif valid_object_cols:
            raise ValueError(
                f"Non-numeric columns in validation data: {valid_object_cols}. "
                f"Apply encoding before training."
            )
    else:
        X_valid, y_valid = None, None

    # Detect problem type
    if problem_type == 'auto':
        problem_type = _detect_problem_type(y_train)

    # Get model
    model = _get_model(
        model_type=model_type,
        problem_type=problem_type,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **model_kwargs
    )

    # Train
    if model_type in ['lightgbm', 'xgboost', 'catboost'] and X_valid is not None:
        # Models with early stopping
        if model_type == 'lightgbm':
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[_lgb_early_stopping(50)]
            )
        elif model_type == 'xgboost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=50,
                verbose=False
            )
        elif model_type == 'catboost':
            model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=50,
                verbose=False
            )
    else:
        model.fit(X_train, y_train)

    # Compute metrics
    metrics = _compute_metrics(model, X_train, y_train, X_valid, y_valid, problem_type)

    # Save outputs
    _save_artifact(model, model_output)
    _save_artifact(metrics, metrics_output)

    return {'model': model_output, 'metrics': metrics_output}


def _get_model(
    model_type: str,
    problem_type: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    random_state: int,
    **kwargs
):
    """Get model instance based on type and problem."""

    if model_type == 'lightgbm':
        import lightgbm as lgb

        if problem_type == 'regression':
            return lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth or -1,
                random_state=random_state,
                verbosity=-1,
                **kwargs
            )
        else:
            return lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth or -1,
                random_state=random_state,
                verbosity=-1,
                **kwargs
            )

    elif model_type == 'xgboost':
        import xgboost as xgb

        if problem_type == 'regression':
            return xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth or 6,
                random_state=random_state,
                verbosity=0,
                **kwargs
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth or 6,
                random_state=random_state,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss' if problem_type == 'binary' else 'mlogloss',
                **kwargs
            )

    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        if problem_type == 'regression':
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        else:
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )

    elif model_type == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

        if problem_type == 'regression':
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth or 3,
                random_state=random_state,
                **kwargs
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth or 3,
                random_state=random_state,
                **kwargs
            )

    elif model_type == 'catboost':
        from catboost import CatBoostRegressor, CatBoostClassifier

        if problem_type == 'regression':
            return CatBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth or 6,
                random_state=random_state,
                verbose=False,
                **kwargs
            )
        else:
            return CatBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth or 6,
                random_state=random_state,
                verbose=False,
                **kwargs
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _lgb_early_stopping(stopping_rounds):
    """LightGBM early stopping callback."""
    import lightgbm as lgb
    return lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)


def _compute_metrics(model, X_train, y_train, X_valid, y_valid, problem_type):
    """Compute training and validation metrics."""
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, roc_auc_score, log_loss, f1_score
    )

    metrics = {'problem_type': problem_type}

    if problem_type == 'regression':
        train_pred = model.predict(X_train)
        metrics['train_rmse'] = float(np.sqrt(mean_squared_error(y_train, train_pred)))
        metrics['train_mae'] = float(mean_absolute_error(y_train, train_pred))
        metrics['train_r2'] = float(r2_score(y_train, train_pred))

        if X_valid is not None:
            valid_pred = model.predict(X_valid)
            metrics['valid_rmse'] = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
            metrics['valid_mae'] = float(mean_absolute_error(y_valid, valid_pred))
            metrics['valid_r2'] = float(r2_score(y_valid, valid_pred))

    else:  # classification
        train_pred = model.predict(X_train)
        metrics['train_accuracy'] = float(accuracy_score(y_train, train_pred))

        if hasattr(model, 'predict_proba'):
            train_proba = model.predict_proba(X_train)
            if problem_type == 'binary':
                metrics['train_auc'] = float(roc_auc_score(y_train, train_proba[:, 1]))
            metrics['train_logloss'] = float(log_loss(y_train, train_proba))

        if X_valid is not None:
            valid_pred = model.predict(X_valid)
            metrics['valid_accuracy'] = float(accuracy_score(y_valid, valid_pred))

            if hasattr(model, 'predict_proba'):
                valid_proba = model.predict_proba(X_valid)
                if problem_type == 'binary':
                    metrics['valid_auc'] = float(roc_auc_score(y_valid, valid_proba[:, 1]))
                metrics['valid_logloss'] = float(log_loss(y_valid, valid_proba))

    return metrics


# =============================================================================
# Prediction Interface
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pkl", "required": True},
        "data": {"format": "csv", "required": True}
    },
    outputs={"predictions": {"format": "csv"}},
    description="Generate predictions using trained model",
    tags=["prediction", "inference", "generic"]
)
def predict(
    model_path: str,
    data: str,
    output: str,
    id_column: str = None,
    target_column: str = None,
    prediction_column: str = 'prediction',
    predict_proba: bool = False,
    exclude_columns: List[str] = None,
    auto_encode: bool = False
) -> Dict[str, str]:
    """
    Generate predictions using a trained model.

    Args:
        model_path: Path to trained model
        data: Path to data for prediction
        output: Path for predictions output
        id_column: ID column to preserve in output
        target_column: Target column to exclude (if present)
        prediction_column: Name for prediction column
        predict_proba: Return probabilities (for classifiers)
        exclude_columns: Additional columns to exclude
        auto_encode: If True, auto-encode categorical columns (not recommended)
    """
    # Load model and data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(data)

    # Prepare features
    exclude = set()
    if id_column:
        exclude.add(id_column)
    if target_column:
        exclude.add(target_column)
    if exclude_columns:
        exclude.update(exclude_columns)

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    # Handle non-numeric
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        if auto_encode:
            import warnings
            warnings.warn(
                f"auto_encode=True in predict: Encoding {len(object_cols)} columns. "
                f"This may cause inconsistent encoding with training. Use explicit encoding.",
                UserWarning
            )
            for col in object_cols:
                X[col] = pd.factorize(X[col])[0]
        else:
            raise ValueError(
                f"Non-numeric columns found: {object_cols}. "
                f"Apply same encoding used during training."
            )

    # Predict
    if predict_proba and hasattr(model, 'predict_proba'):
        preds = model.predict_proba(X)
        if preds.shape[1] == 2:
            preds = preds[:, 1]  # Binary classification
    else:
        preds = model.predict(X)

    # Build output
    result = pd.DataFrame({prediction_column: preds})
    if id_column and id_column in df.columns:
        result.insert(0, id_column, df[id_column])

    result.to_csv(output, index=False)
    return {'predictions': output}


# =============================================================================
# Cross-Validation Training
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={
        "model": {"format": "pkl"},
        "metrics": {"format": "json"},
        "oof_predictions": {"format": "csv"}
    },
    description="Train model with cross-validation",
    tags=["training", "cross-validation", "generic"]
)
def train_model_cv(
    data: str,
    model_output: str,
    metrics_output: str,
    oof_output: str,
    model_type: str = 'lightgbm',
    problem_type: str = 'auto',
    n_folds: int = 5,
    target_column: str = 'target',
    id_column: str = None,
    exclude_columns: List[str] = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = None,
    random_state: int = 42,
    auto_encode: bool = False,
    **model_kwargs
) -> Dict[str, str]:
    """
    Train model with K-fold cross-validation.

    Returns out-of-fold predictions for stacking/blending.
    
    Args:
        auto_encode: If True, auto-encode categorical columns (not recommended)
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    df = pd.read_csv(data)

    # Prepare features
    exclude = set([target_column])
    if id_column:
        exclude.add(id_column)
    if exclude_columns:
        exclude.update(exclude_columns)

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    y = df[target_column]

    # Handle non-numeric
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        if auto_encode:
            import warnings
            warnings.warn(
                f"auto_encode=True: Encoding {len(object_cols)} columns. "
                f"Consider using fit_encoder/transform_encoder for G2 compliance.",
                UserWarning
            )
            for col in object_cols:
                X[col] = pd.factorize(X[col])[0]
        else:
            raise ValueError(
                f"Non-numeric columns found: {object_cols}. "
                f"Add encoding step to pipeline, or set auto_encode=True."
            )

    # Detect problem type
    if problem_type == 'auto':
        problem_type = _detect_problem_type(y)

    # Choose CV strategy
    if problem_type in ['binary', 'multiclass']:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Cross-validation
    oof_preds = np.zeros(len(X))
    models = []
    fold_metrics = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = _get_model(
            model_type=model_type,
            problem_type=problem_type,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state + fold,
            **model_kwargs
        )

        if model_type in ['lightgbm', 'xgboost']:
            if model_type == 'lightgbm':
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                          callbacks=[_lgb_early_stopping(50)])
            else:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                          early_stopping_rounds=50, verbose=False)
        else:
            model.fit(X_train, y_train)

        # OOF predictions
        if problem_type in ['binary', 'multiclass'] and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_valid)
            oof_preds[valid_idx] = proba[:, 1] if proba.shape[1] == 2 else proba.argmax(1)
        else:
            oof_preds[valid_idx] = model.predict(X_valid)

        fold_metric = _compute_metrics(model, X_train, y_train, X_valid, y_valid, problem_type)
        fold_metrics.append(fold_metric)
        models.append(model)

    # Aggregate metrics
    metrics = {
        'problem_type': problem_type,
        'n_folds': n_folds,
        'fold_metrics': fold_metrics
    }

    # Average metrics
    metric_keys = [k for k in fold_metrics[0].keys() if k.startswith('valid_')]
    for key in metric_keys:
        values = [fm[key] for fm in fold_metrics if key in fm]
        if values:
            metrics[f'cv_{key}'] = float(np.mean(values))
            metrics[f'cv_{key}_std'] = float(np.std(values))

    # Save outputs
    _save_artifact(models, model_output)  # Save all fold models
    _save_artifact(metrics, metrics_output)

    # Save OOF predictions
    oof_df = pd.DataFrame({'prediction': oof_preds})
    if id_column and id_column in df.columns:
        oof_df.insert(0, id_column, df[id_column])
    oof_df.to_csv(oof_output, index=False)

    return {'model': model_output, 'metrics': metrics_output, 'oof_predictions': oof_output}


# =============================================================================
# Service Registry
# =============================================================================

SERVICE_REGISTRY = {
    "train_model": train_model,
    "predict": predict,
    "train_model_cv": train_model_cv,
}
