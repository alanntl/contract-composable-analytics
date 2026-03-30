"""
Classification Services - SLEGO Common Module
===============================================

Generic classification model services for binary and multiclass problems.

Services:
  Training: train_lightgbm_classifier, train_random_forest_classifier,
            train_xgboost_classifier, train_ensemble_classifier
  Prediction: predict_classifier
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

# =============================================================================
# HELPERS: Import from shared io_utils
# =============================================================================
from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# TRAINING SERVICES
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
    description="Train LightGBM classifier for binary classification with optional early stopping",
    tags=["modeling", "training", "lightgbm", "classification", "binary"],
    version="1.0.0",
)
def train_lightgbm_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 64,
    max_depth: int = -1,
    min_child_samples: int = 30,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    class_weight: str = None,
    random_state: int = 42,
    early_stopping_rounds: int = 100,
    categorical_feature: Optional[str] = None,
) -> str:
    """Train a LightGBM binary classifier with optional validation-based early stopping.

    Loads training data, fits an LGBMClassifier with the specified hyperparameters,
    and optionally evaluates on a held-out validation set using ROC AUC.

    Outputs:
        model pickle  - dict with keys 'model' and 'feature_cols'
        metrics json  - training metadata and optional roc_auc score
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")

    # Handle string labels with LabelEncoder
    label_encoder = None
    y_raw = train_df[label_column]
    if not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_raw)
    else:
        y_train = y_raw.astype(int).values

    feature_cols = list(X_train.columns)

    # Auto-detect binary vs multiclass
    n_classes = len(np.unique(y_train))
    is_binary = n_classes <= 2
    objective = "binary" if is_binary else "multiclass"
    eval_metric = "auc" if is_binary else "multi_logloss"

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        objective=objective,
        num_class=n_classes if not is_binary else None,
    )

    metrics = {
        "model_type": "lightgbm_classifier",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
    }

    # Train with or without validation early stopping
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")

        # Apply same label encoding as training
        y_valid_raw = valid_df[label_column]
        if label_encoder is not None:
            y_valid = label_encoder.transform(y_valid_raw)
        else:
            y_valid = y_valid_raw.astype(int).values

        # Filter out validation samples with labels not seen in training
        # This handles edge cases where rare classes end up only in validation
        train_labels = set(np.unique(y_train))
        valid_mask = pd.Series(y_valid).isin(train_labels)
        if not valid_mask.all():
            n_filtered = (~valid_mask).sum()
            X_valid = X_valid[valid_mask.values]
            y_valid = y_valid[valid_mask.values]
            # Log filtered samples (will be visible in metrics)
            metrics["n_filtered_unseen_labels"] = int(n_filtered)

        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        # Prepare categorical feature parameter
        cat_feat = None
        if categorical_feature:
            if isinstance(categorical_feature, str):
                cat_feat = [categorical_feature] if categorical_feature in feature_cols else None
            else:
                cat_feat = [c for c in categorical_feature if c in feature_cols]
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=eval_metric,
            callbacks=callbacks,
            categorical_feature=cat_feat if cat_feat else 'auto',
        )

        if is_binary:
            preds = model.predict_proba(X_valid)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_valid, preds))
        else:
            preds = model.predict_proba(X_valid)
            # Use multiclass AUC (OvR)
            from sklearn.metrics import roc_auc_score
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_valid, preds, multi_class="ovr"))
            except:
                metrics["roc_auc"] = None
        metrics["best_iteration"] = int(getattr(model, "best_iteration_", n_estimators))
        metrics["n_valid_samples"] = len(X_valid)
        metrics["n_classes"] = n_classes
    else:
        # Prepare categorical feature parameter
        cat_feat = None
        if categorical_feature:
            if isinstance(categorical_feature, str):
                cat_feat = [categorical_feature] if categorical_feature in feature_cols else None
            else:
                cat_feat = [c for c in categorical_feature if c in feature_cols]
        model.fit(X_train, y_train, categorical_feature=cat_feat if cat_feat else 'auto')

    # Save model artifact (include label_encoder for string labels)
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    artifact = {"model": model, "feature_cols": feature_cols}
    if label_encoder is not None:
        artifact["label_encoder"] = label_encoder
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    auc_str = f", AUC={metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else ""
    return f"train_lightgbm_classifier: {len(X_train)} samples, {len(feature_cols)} features{auc_str}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train Random Forest classifier with optional validation evaluation",
    tags=["modeling", "training", "random-forest", "classification", "binary"],
    version="1.0.0",
)
def train_random_forest_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    class_weight: str = "balanced",
    feature_exclude: str = None,
    random_state: int = 42,
) -> str:
    """Train a scikit-learn RandomForestClassifier.

    Supports optional validation evaluation via ROC AUC. The feature_exclude
    parameter accepts a comma-separated string of additional column names to
    drop from features beyond label_column and id_column.

    Outputs:
        model pickle  - dict with keys 'model' and 'feature_cols'
        metrics json  - training metadata and optional roc_auc score
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)
    if feature_exclude:
        extra = [c.strip() for c in feature_exclude.split(",")]
        drop_cols.extend([c for c in extra if c in train_df.columns])

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[label_column].astype(int)
    feature_cols = list(X_train.columns)

    # Parse max_depth: allow None via string "None"
    parsed_max_depth = None if max_depth is None or str(max_depth).lower() == "none" else int(max_depth)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=parsed_max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    metrics = {
        "model_type": "random_forest_classifier",
        "n_estimators": n_estimators,
        "max_depth": parsed_max_depth,
        "class_weight": class_weight,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
    }

    # Evaluate on validation set if provided
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        y_valid = valid_df[label_column].astype(int)
        metrics["n_valid_samples"] = len(X_valid)

        n_classes = len(model.classes_)
        proba = model.predict_proba(X_valid)
        if n_classes <= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_valid, proba[:, 1]))
        else:
            from sklearn.metrics import accuracy_score
            y_pred = model.predict(X_valid)
            metrics["accuracy"] = float(accuracy_score(y_valid, y_pred))
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_valid, proba, multi_class="ovr"))
            except ValueError:
                pass

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    auc_str = f", AUC={metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else ""
    return f"train_random_forest_classifier: {len(X_train)} samples, {len(feature_cols)} features{auc_str}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train XGBoost classifier for binary or multiclass classification",
    tags=["modeling", "training", "xgboost", "classification", "binary"],
    version="1.0.0",
)
def train_xgboost_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    objective: str = "binary:logistic",
    feature_exclude: str = None,
    random_state: int = 42,
) -> str:
    """Train an XGBoost classifier (XGBClassifier) with optional validation evaluation.

    XGBoost is imported lazily so that the module can be loaded even when xgboost
    is not installed. The feature_exclude parameter accepts a comma-separated
    string of additional column names to drop from features.

    Outputs:
        model pickle  - dict with keys 'model' and 'feature_cols'
        metrics json  - training metadata and optional roc_auc score
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)
    if feature_exclude:
        extra = [c.strip() for c in feature_exclude.split(",")]
        drop_cols.extend([c for c in extra if c in train_df.columns])

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[label_column].astype(int)
    feature_cols = list(X_train.columns)

    # Determine eval_metric from objective
    is_binary = "binary" in objective
    eval_metric = "auc" if is_binary else "mlogloss"

    model = xgb.XGBClassifier(
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
        eval_metric=eval_metric,
        use_label_encoder=False,
    )

    metrics = {
        "model_type": "xgboost_classifier",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "objective": objective,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
    }

    # Train with or without validation
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        y_valid = valid_df[label_column].astype(int)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        if is_binary:
            preds = model.predict_proba(X_valid)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_valid, preds))
        else:
            preds = model.predict_proba(X_valid)
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_valid, preds, multi_class="ovr", average="weighted")
            )
        metrics["best_iteration"] = int(getattr(model, "best_iteration", n_estimators))
        metrics["n_valid_samples"] = len(X_valid)
    else:
        model.fit(X_train, y_train)

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    auc_key = "roc_auc" if "roc_auc" in metrics else "roc_auc_ovr"
    auc_str = f", AUC={metrics[auc_key]:.4f}" if auc_key in metrics else ""
    return f"train_xgboost_classifier: {len(X_train)} samples, {len(feature_cols)} features{auc_str}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train an ensemble of classifiers and blend their predictions via weighted averaging",
    tags=["modeling", "training", "ensemble", "classification", "binary"],
    version="1.0.0",
)
def train_ensemble_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    model_types: List[str] = None,
    weights: List[float] = None,
    random_state: int = 42,
) -> str:
    """Train multiple classifiers and save them as a weighted ensemble.

    Supported model_types: 'lightgbm', 'random_forest', 'xgboost'.
    Each sub-model is trained on the same training data with sensible defaults.
    The ensemble blends predict_proba outputs using the supplied weights.

    Outputs:
        model pickle  - dict with keys 'models', 'weights', 'feature_cols', 'model_types'
        metrics json  - per-model and blended metrics
    """
    from sklearn.metrics import roc_auc_score

    if model_types is None:
        model_types = ["lightgbm", "random_forest"]
    if weights is None:
        weights = [1.0 / len(model_types)] * len(model_types)

    # Normalise weights to sum to 1
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[label_column].astype(int)
    feature_cols = list(X_train.columns)

    # Optionally load validation data
    X_valid, y_valid = None, None
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        y_valid = valid_df[label_column].astype(int)

    trained_models = []
    per_model_metrics = []

    for mtype in model_types:
        if mtype == "lightgbm":
            import lightgbm as lgb
            m = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, num_leaves=64,
                max_depth=-1, min_child_samples=30, subsample=0.8,
                colsample_bytree=0.8, random_state=random_state,
                n_jobs=-1, objective="binary",
            )
            if X_valid is not None:
                callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
                m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      eval_metric="auc", callbacks=callbacks)
            else:
                m.fit(X_train, y_train)

        elif mtype == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(
                n_estimators=100, max_features="sqrt", class_weight="balanced",
                random_state=random_state, n_jobs=-1,
            )
            m.fit(X_train, y_train)

        elif mtype == "xgboost":
            import xgboost as xgb
            m = xgb.XGBClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=8,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                objective="binary:logistic", random_state=random_state,
                n_jobs=-1, eval_metric="auc", early_stopping_rounds=100,
            )
            if X_valid is not None:
                m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                m.fit(X_train, y_train)

        elif mtype == "catboost":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=10,
                random_state=random_state, verbose=0,
            )
            if X_valid is not None:
                m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                m.fit(X_train, y_train)

        else:
            raise ValueError(f"Unsupported model type: '{mtype}'. Use 'lightgbm', 'random_forest', 'xgboost', or 'catboost'.")

        model_info = {"model_type": mtype}
        if X_valid is not None:
            preds = m.predict_proba(X_valid)[:, 1]
            model_info["roc_auc"] = float(roc_auc_score(y_valid, preds))

        trained_models.append(m)
        per_model_metrics.append(model_info)

    # Compute blended validation score
    metrics = {
        "model_type": "ensemble_classifier",
        "sub_models": per_model_metrics,
        "weights": weights,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
    }

    if X_valid is not None:
        blended = np.zeros(len(X_valid))
        for m, w in zip(trained_models, weights):
            blended += w * m.predict_proba(X_valid)[:, 1]
        metrics["blended_roc_auc"] = float(roc_auc_score(y_valid, blended))
        metrics["n_valid_samples"] = len(X_valid)

    # Save ensemble artifact
    artifact = {
        "models": trained_models,
        "weights": weights,
        "feature_cols": feature_cols,
        "model_types": model_types,
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    auc_str = f", blended AUC={metrics['blended_roc_auc']:.4f}" if "blended_roc_auc" in metrics else ""
    return f"train_ensemble_classifier: {len(model_types)} models, {len(X_train)} samples{auc_str}"


# =============================================================================
# PREDICTION SERVICE
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Load a trained classifier and generate predictions with optional probability output",
    tags=["modeling", "prediction", "inference", "classification"],
    version="1.0.0",
)
def predict_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "target",
    output_proba: bool = True,
    positive_class: int = 1,
    proba_as_prediction: bool = False,
) -> str:
    """Load a trained classifier from pickle and produce predictions.

    Handles both single-model artifacts (dict with 'model' key) and ensemble
    artifacts (dict with 'models' and 'weights' keys). When output_proba is
    True, an additional column '{prediction_column}_proba' is included with
    the probability of the positive class.

    When proba_as_prediction is True, the prediction_column itself contains
    probabilities instead of class labels (useful for Kaggle submissions
    that require probability outputs like AUC-scored competitions).

    Outputs:
        predictions CSV with columns: id_column, prediction_column,
        and optionally {prediction_column}_proba
    """
    # Load model artifact
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    data_df = _load_data(inputs["data"])

    # Determine feature columns
    feature_cols = artifact.get("feature_cols")

    # Build result DataFrame starting with ID
    result = pd.DataFrame()
    if id_column and id_column in data_df.columns:
        result[id_column] = data_df[id_column]

    # Prepare feature matrix
    if feature_cols:
        # Use stored feature columns, fill missing with 0
        for col in feature_cols:
            if col not in data_df.columns:
                data_df[col] = 0
        X = data_df[feature_cols]
    else:
        # Fallback: drop id and any known non-feature columns
        drop_cols = []
        if id_column and id_column in data_df.columns:
            drop_cols.append(id_column)
        if prediction_column in data_df.columns:
            drop_cols.append(prediction_column)
        X = data_df.drop(columns=drop_cols, errors="ignore")

    # Generate predictions
    if "models" in artifact and "weights" in artifact:
        # Ensemble artifact
        models = artifact["models"]
        weights = artifact["weights"]

        # Blended probability
        n_classes = len(models[0].classes_)
        blended_proba = np.zeros((len(X), n_classes))
        for m, w in zip(models, weights):
            blended_proba += w * m.predict_proba(X)

        predicted_classes = np.argmax(blended_proba, axis=1)
        # Map back to original class labels
        class_labels = models[0].classes_
        predicted_labels = class_labels[predicted_classes]

        if n_classes == 2:
            positive_idx = list(class_labels).index(positive_class) if positive_class in class_labels else 1
            proba_positive = blended_proba[:, positive_idx]
        else:
            proba_positive = np.max(blended_proba, axis=1)
    else:
        # Single model artifact
        model = artifact["model"]

        predicted_labels = model.predict(X)

        if hasattr(model, "predict_proba"):
            proba_all = model.predict_proba(X)
            n_classes = proba_all.shape[1]
            class_labels = model.classes_

            if n_classes == 2:
                positive_idx = list(class_labels).index(positive_class) if positive_class in class_labels else 1
                proba_positive = proba_all[:, positive_idx]
            else:
                proba_positive = np.max(proba_all, axis=1)
        else:
            proba_positive = None

    if proba_as_prediction and proba_positive is not None:
        result[prediction_column] = proba_positive
    else:
        # Decode labels back to strings if label_encoder exists
        label_encoder = artifact.get("label_encoder")
        if label_encoder is not None:
            predicted_labels = label_encoder.inverse_transform(predicted_labels)
        result[prediction_column] = predicted_labels
        if output_proba and proba_positive is not None:
            result[f"{prediction_column}_proba"] = proba_positive

    _save_data(result, outputs["predictions"])

    n_preds = len(result)
    proba_str = f", mean_proba={proba_positive.mean():.4f}" if proba_positive is not None else ""
    return f"predict_classifier: {n_preds} predictions{proba_str}"


# =============================================================================
# ADABOOST CLASSIFIER (Reusable)
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
    description="Train AdaBoost classifier for binary or multiclass classification",
    tags=["modeling", "training", "adaboost", "classification", "ensemble", "generic"],
    version="1.0.0",
)
def train_adaboost_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    id_column: str = None,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    algorithm: str = "SAMME",
    random_state: int = 42,
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """Train an AdaBoost classifier with optional validation evaluation.

    AdaBoost is an ensemble method that combines multiple weak learners
    (typically decision stumps) to create a strong classifier. Works well
    for multiclass classification problems like forest cover type prediction.

    G1 Compliance: Generic, works with any classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All column names as parameters.

    Parameters:
        target_column: Column containing class labels
        id_column: ID column to exclude from features
        n_estimators: Number of weak learners (default 50)
        learning_rate: Learning rate shrinks contribution of each classifier
        algorithm: 'SAMME' or 'SAMME.R' (deprecated, using SAMME)
        exclude_columns: Additional columns to exclude from features

    Outputs:
        model pickle - dict with 'model' and 'feature_cols'
        metrics json - accuracy, n_features, training info
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [target_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)
    if exclude_columns:
        drop_cols.extend([c for c in exclude_columns if c in train_df.columns])

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[target_column]
    feature_cols = list(X_train.columns)

    # Auto-detect number of classes
    n_classes = len(y_train.unique())

    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # Training accuracy
    train_preds = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_preds)

    metrics = {
        "model_type": "adaboost_classifier",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "algorithm": algorithm,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_classes": n_classes,
        "train_accuracy": float(train_accuracy),
    }

    # Evaluate on validation set if provided
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        y_valid = valid_df[target_column]

        valid_preds = model.predict(X_valid)
        valid_accuracy = accuracy_score(y_valid, valid_preds)
        metrics["valid_accuracy"] = float(valid_accuracy)
        metrics["n_valid_samples"] = len(X_valid)

        # Compute multiclass AUC if possible
        if n_classes > 2:
            try:
                proba = model.predict_proba(X_valid)
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_valid, proba, multi_class="ovr"))
            except Exception:
                pass
        else:
            try:
                proba = model.predict_proba(X_valid)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_valid, proba))
            except Exception:
                pass

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    acc_str = f"train_acc={train_accuracy:.4f}"
    if "valid_accuracy" in metrics:
        acc_str += f", valid_acc={metrics['valid_accuracy']:.4f}"

    return f"train_adaboost_classifier: {len(X_train)} samples, {len(feature_cols)} features, {acc_str}"


# =============================================================================
# AUTOML CLASSIFIER (FLAML)
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
    description="Train classifier using FLAML AutoML - automatically finds best model and hyperparameters",
    tags=["modeling", "training", "automl", "flaml", "classification", "generic"],
    version="1.0.0",
)
def train_automl_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    id_column: str = None,
    time_budget: int = 60,
    metric: str = "accuracy",
    estimator_list: List[str] = None,
    random_state: int = 42,
    exclude_columns: Optional[List[str]] = None,
    n_jobs: int = -1,
) -> str:
    """Train a classifier using FLAML AutoML.

    FLAML (Fast and Lightweight AutoML) automatically searches for the best
    model and hyperparameters within the given time budget. It supports
    LightGBM, XGBoost, Random Forest, Extra Trees, and more.

    G1 Compliance: Generic, works with any classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All column names and settings as parameters.

    Parameters:
        target_column: Column containing class labels
        id_column: ID column to exclude from features
        time_budget: Time budget in seconds for AutoML search (default 60)
        metric: Optimization metric - 'accuracy', 'roc_auc', 'log_loss', 'f1', 'macro_f1', 'micro_f1'
        estimator_list: List of estimators to try. Default: ['lgbm', 'xgboost', 'rf', 'extra_tree']
        random_state: Random seed for reproducibility
        exclude_columns: Additional columns to exclude from features
        n_jobs: Number of parallel jobs (-1 for all cores)

    Outputs:
        model pickle - dict with 'model', 'feature_cols', and 'best_config'
        metrics json - best model info, accuracy, training metadata
    """
    from flaml import AutoML
    from sklearn.metrics import accuracy_score, roc_auc_score

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [target_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)
    if exclude_columns:
        drop_cols.extend([c for c in exclude_columns if c in train_df.columns])

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[target_column]
    feature_cols = list(X_train.columns)

    # Auto-detect number of classes
    n_classes = len(y_train.unique())

    # Default estimator list
    if estimator_list is None:
        estimator_list = ['lgbm', 'xgboost', 'rf', 'extra_tree']

    # Create and configure AutoML
    automl = AutoML()

    # Prepare validation data if available
    X_val, y_val = None, None
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_val = valid_df.drop(columns=drop_cols, errors="ignore")
        y_val = valid_df[target_column]

    # Run AutoML
    automl.fit(
        X_train, y_train,
        task="classification",
        time_budget=time_budget,
        metric=metric,
        estimator_list=estimator_list,
        seed=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    # Training accuracy
    train_preds = automl.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_preds)

    metrics = {
        "model_type": "automl_flaml",
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "time_budget": time_budget,
        "metric": metric,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_classes": n_classes,
        "train_accuracy": float(train_accuracy),
        "best_loss": float(automl.best_loss) if automl.best_loss else None,
    }

    # Evaluate on validation set
    if X_val is not None:
        valid_preds = automl.predict(X_val)
        valid_accuracy = accuracy_score(y_val, valid_preds)
        metrics["valid_accuracy"] = float(valid_accuracy)
        metrics["n_valid_samples"] = len(X_val)

        # Compute multiclass AUC if possible
        if n_classes > 2:
            try:
                proba = automl.predict_proba(X_val)
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_val, proba, multi_class="ovr"))
            except Exception:
                pass
        else:
            try:
                proba = automl.predict_proba(X_val)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_val, proba))
            except Exception:
                pass

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({
            "model": automl,
            "feature_cols": feature_cols,
            "best_estimator": automl.best_estimator,
            "best_config": automl.best_config,
        }, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    acc_str = f"train_acc={train_accuracy:.4f}"
    if "valid_accuracy" in metrics:
        acc_str += f", valid_acc={metrics['valid_accuracy']:.4f}"

    return f"train_automl_classifier: best={automl.best_estimator}, {len(X_train)} samples, {acc_str}"


# =============================================================================
# LOGISTIC REGRESSION CLASSIFIER (Reusable)
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
    description="Train Logistic Regression classifier with L1/L2 regularization for binary/multiclass problems",
    tags=["modeling", "training", "logistic-regression", "classification", "regularization"],
    version="1.0.0",
)
def train_logistic_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    penalty: str = "l1",
    C: float = 0.1,
    solver: str = "liblinear",
    class_weight: str = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
) -> str:
    """Train a Logistic Regression classifier with regularization.

    Ideal for high-dimensional, small-sample problems where tree-based models
    tend to overfit. L1 penalty provides built-in feature selection via sparsity.

    Outputs:
        model pickle  - dict with keys 'model' and 'feature_cols'
        metrics json  - training metadata and optional roc_auc score
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])

    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[label_column].astype(int)
    feature_cols = list(X_train.columns)

    n_classes = len(y_train.unique())

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )

    metrics = {
        "model_type": "logistic_regression",
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "class_weight": class_weight,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_classes": n_classes,
    }

    model.fit(X_train, y_train)

    # Report L1 sparsity (non-zero coefficients)
    if hasattr(model, 'coef_'):
        n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-6))
        metrics["n_nonzero_features"] = n_nonzero

    # Evaluate on validation set if available
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        y_valid = valid_df[label_column].astype(int)

        if n_classes <= 2:
            preds = model.predict_proba(X_valid)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_valid, preds))
        else:
            preds = model.predict_proba(X_valid)
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_valid, preds, multi_class="ovr"))
            except Exception:
                metrics["roc_auc"] = None
        metrics["n_valid_samples"] = len(X_valid)

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    auc_str = f", AUC={metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else ""
    return f"train_logistic_classifier: {len(X_train)} samples, {len(feature_cols)} features{auc_str}"


# =============================================================================
# CV-AVERAGED LOGISTIC REGRESSION (Reusable)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train Logistic Regression via RepeatedStratifiedKFold and produce CV-averaged test predictions",
    tags=["modeling", "training", "logistic-regression", "cv", "classification", "regularization"],
    version="1.0.0",
)
def train_cv_logistic_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    prediction_column: str = "target",
    penalty: str = "l1",
    C: float = 0.1,
    solver: str = "liblinear",
    class_weight: str = "balanced",
    max_iter: int = 1000,
    n_splits: int = 10,
    n_repeats: int = 10,
    add_noise: bool = False,
    noise_std: float = 0.01,
    random_state: int = 42,
) -> str:
    """Train Logistic Regression via RepeatedStratifiedKFold CV and average test predictions.

    Instead of training a single model, trains n_splits * n_repeats models
    and averages their test-set predictions. This produces more robust
    probability estimates, which is critical for small-sample problems.

    Optionally adds Gaussian noise to training folds to further reduce overfitting.

    Outputs:
        model pickle  - dict with 'models' list and 'feature_cols'
        predictions CSV - CV-averaged probability predictions on test_data
        metrics json  - CV AUC and metadata
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore").values
    y_train = train_df[label_column].astype(int).values
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_test = test_df.drop(columns=[id_column] if id_column and id_column in test_df.columns else [], errors="ignore")
    X_test = X_test[feature_cols].values if all(c in test_df.columns for c in feature_cols) else X_test.values

    # CV-averaged predictions
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    models = []
    n_total = 0

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for fold_train_idx, fold_val_idx in rskf.split(X_train, y_train):
        X_fold_train = X_train[fold_train_idx]
        y_fold_train = y_train[fold_train_idx]

        if add_noise:
            rng = np.random.RandomState(random_state + n_total)
            X_fold_train = X_fold_train + rng.normal(0, noise_std, X_fold_train.shape)

        clf = LogisticRegression(
            penalty=penalty, C=C, solver=solver, class_weight=class_weight,
            max_iter=max_iter, random_state=random_state,
        )
        clf.fit(X_fold_train, y_fold_train)

        oof[fold_val_idx] += clf.predict_proba(X_train[fold_val_idx])[:, 1]
        test_preds += clf.predict_proba(X_test)[:, 1]
        models.append(clf)
        n_total += 1

    # Average predictions
    # OOF predictions were accumulated n_repeats times per sample
    oof /= n_repeats
    test_preds /= n_total

    cv_auc = float(roc_auc_score(y_train, oof))

    # Build result predictions DataFrame
    result = pd.DataFrame()
    if id_column and id_column in test_df.columns:
        result[id_column] = test_df[id_column]
    result[prediction_column] = test_preds

    # Save outputs
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols, "weights": [1.0/n_total]*n_total}, f)

    _save_data(result, outputs["predictions"])

    metrics = {
        "model_type": "cv_logistic_regression",
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "class_weight": class_weight,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_models": n_total,
        "add_noise": add_noise,
        "cv_roc_auc": cv_auc,
        "mean_test_proba": float(test_preds.mean()),
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_cv_logistic_classifier: {n_total} models, CV AUC={cv_auc:.4f}, {len(X_test)} test predictions"


# =============================================================================
# MULTICLASS PROBABILITY SUBMISSION
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Generate multiclass probability submission with per-class columns",
    tags=["prediction", "multiclass", "submission", "classification", "generic"],
    version="1.0.0",
)
def predict_multiclass_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_prefix: str = "target",
    class_names: Optional[List[str]] = None,
) -> str:
    """Generate Kaggle submission with per-class probability columns.

    For multiclass competitions requiring probability outputs like
    Status_C, Status_CL, Status_D or similar per-class columns.

    Handles both single-model artifacts (dict with 'model' key) and ensemble
    artifacts (dict with 'models' and 'weights' keys).

    Probabilities are normalized to sum to 1 per row.

    Parameters:
        id_column: Column containing sample IDs
        prediction_prefix: Prefix for output columns (e.g., "Status" → Status_C, Status_CL)
        class_names: Ordered list of class name suffixes (e.g., ["C", "CL", "D"]).
                     Must match the sorted order of integer-encoded class labels.
                     If None, uses model.classes_ values as strings.
    """
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    data_df = _load_data(inputs["data"])
    feature_cols = artifact.get("feature_cols")

    # Extract IDs
    ids = data_df[id_column] if id_column in data_df.columns else pd.RangeIndex(len(data_df))

    # Prepare feature matrix
    if feature_cols:
        for col in feature_cols:
            if col not in data_df.columns:
                data_df[col] = 0
        X = data_df[feature_cols]
    else:
        drop_cols = [id_column] if id_column in data_df.columns else []
        X = data_df.drop(columns=drop_cols, errors="ignore")

    # Predict probabilities
    if "models" in artifact and "weights" in artifact:
        models = artifact["models"]
        weights = artifact["weights"]
        n_classes = len(models[0].classes_)
        proba = np.zeros((len(X), n_classes))
        for m, w in zip(models, weights):
            proba += w * m.predict_proba(X)
        model_classes = models[0].classes_
    else:
        model = artifact["model"]
        proba = model.predict_proba(X)
        model_classes = model.classes_

    # Determine class names for column headers
    if class_names is None:
        class_names = [str(c) for c in model_classes]

    # Normalize probabilities to sum to 1
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / row_sums

    # Build submission DataFrame
    submission = pd.DataFrame({id_column: ids})
    for i, cls in enumerate(class_names):
        col_name = f"{prediction_prefix}_{cls}" if prediction_prefix else cls
        submission[col_name] = proba[:, i]

    _save_data(submission, outputs["submission"])

    return f"predict_multiclass_submission: {len(submission)} predictions, {len(class_names)} classes"


# =============================================================================
# SKLEARN MLP NEURAL NETWORK CLASSIFIER
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
    description="Train sklearn MLPClassifier neural network for multiclass classification",
    tags=["modeling", "training", "neural-network", "mlp", "classification", "multiclass", "generic"],
    version="1.0.0",
)
def train_keras_nn_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    id_column: str = None,
    hidden_units: List[int] = None,
    dropout_rate: float = 0.3,
    activation: str = "relu",
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 500,
    patience: int = 50,
    random_state: int = 42,
    exclude_columns: Optional[List[str]] = None,
) -> str:
    """Train a sklearn MLPClassifier for multiclass classification.

    Uses scikit-learn's MLPClassifier which provides a simple feedforward
    neural network with early stopping. Inspired by top Kaggle solutions
    for leaf-classification that use dense networks with dropout-like
    regularization (achieved via alpha parameter in sklearn).

    G1 Compliance: Generic, works with any multiclass classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All architecture parameters exposed.

    Parameters:
        target_column: Column containing class labels
        id_column: ID column to exclude from features
        hidden_units: Tuple of neurons per hidden layer (default (512, 512))
        dropout_rate: Not directly used - regularization via alpha instead
        activation: Activation function ('relu', 'tanh', 'logistic')
        learning_rate: Initial learning rate (default 0.001)
        batch_size: Training batch size (default 32)
        epochs: Maximum training iterations (default 500)
        patience: Early stopping patience (n_iter_no_change, default 50)
        random_state: Random seed for reproducibility
        exclude_columns: Additional columns to exclude from features

    Outputs:
        model pickle - dict with 'model', 'feature_cols', 'label_encoder'
        metrics json - training/validation accuracy and loss
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import log_loss, accuracy_score

    # Default hidden units
    if hidden_units is None:
        hidden_units = (512, 512)
    elif isinstance(hidden_units, list):
        hidden_units = tuple(hidden_units)

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [target_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)
    if exclude_columns:
        drop_cols.extend([c for c in exclude_columns if c in train_df.columns])

    X_train = train_df.drop(columns=drop_cols, errors="ignore").values.astype(np.float64)
    y_raw = train_df[target_column]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Map activation names
    activation_map = {"relu": "relu", "tanh": "tanh", "sigmoid": "logistic", "logistic": "logistic"}
    sklearn_activation = activation_map.get(activation, "relu")

    # Create MLPClassifier with regularization (alpha ~ dropout effect)
    # Higher alpha = more regularization
    alpha = dropout_rate * 0.01  # Scale dropout_rate to alpha

    model = MLPClassifier(
        hidden_layer_sizes=hidden_units,
        activation=sklearn_activation,
        solver='adam',
        alpha=alpha,
        batch_size=min(batch_size, 200),  # sklearn default max
        learning_rate_init=learning_rate,
        max_iter=epochs,
        early_stopping=True,
        n_iter_no_change=patience,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
    )

    # Train
    model.fit(X_train, y_encoded)

    # Compute metrics
    train_proba = model.predict_proba(X_train)
    train_loss = float(log_loss(y_encoded, train_proba))
    train_preds = model.predict(X_train)
    train_accuracy = float(accuracy_score(y_encoded, train_preds))

    metrics = {
        "model_type": "mlp_classifier",
        "hidden_units": list(hidden_units),
        "alpha": alpha,
        "activation": sklearn_activation,
        "learning_rate": learning_rate,
        "n_features": X_train.shape[1],
        "n_train_samples": len(X_train),
        "n_classes": n_classes,
        "n_iter": model.n_iter_,
        "train_accuracy": train_accuracy,
        "train_log_loss": train_loss,
        "best_validation_score": float(model.best_validation_score_) if hasattr(model, 'best_validation_score_') else None,
    }

    # Evaluate on validation set if provided
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_val = valid_df.drop(columns=drop_cols, errors="ignore").values.astype(np.float64)
        y_val_raw = valid_df[target_column]
        y_val_encoded = label_encoder.transform(y_val_raw)

        val_proba = model.predict_proba(X_val)
        val_loss = float(log_loss(y_val_encoded, val_proba))
        val_preds = model.predict(X_val)
        val_accuracy = float(accuracy_score(y_val_encoded, val_preds))
        metrics["valid_accuracy"] = val_accuracy
        metrics["valid_log_loss"] = val_loss
        metrics["n_valid_samples"] = len(X_val)

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "label_encoder": label_encoder,
            "classes_": label_encoder.classes_,
        }, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    loss_str = f"train_loss={train_loss:.4f}"
    if "valid_log_loss" in metrics:
        loss_str += f", val_loss={metrics['valid_log_loss']:.4f}"

    return f"train_keras_nn_classifier: {len(X_train)} samples, {n_classes} classes, {model.n_iter_} iters, {loss_str}"


# =============================================================================
# SERVICE: TRAIN LINEAR SVC CLASSIFIER
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
    description="Train LinearSVC classifier for multiclass classification - fast and effective for text/NLP",
    tags=["modeling", "training", "svm", "classification", "multiclass", "text", "nlp"],
    version="1.0.0",
)
def train_linear_svc_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    C: float = 1.0,
    penalty: str = "l2",
    loss: str = "squared_hinge",
    dual: str = "auto",
    max_iter: int = 10000,
    feature_exclude: str = None,
    random_state: int = 42,
) -> str:
    """Train a LinearSVC classifier - highly effective for text classification with TF-IDF features.

    Based on top Kaggle solutions for text/NLP competitions (e.g., What's Cooking).
    LinearSVC is often the best choice for high-dimensional sparse data like TF-IDF.

    Parameters:
        label_column: Column containing target labels
        id_column: ID column to exclude from features
        C: Regularization parameter (higher = less regularization)
        penalty: 'l1' or 'l2' regularization
        loss: 'hinge' (standard SVM) or 'squared_hinge'
        dual: 'auto' selects based on n_samples vs n_features
        max_iter: Maximum iterations for solver
        feature_exclude: Comma-separated columns to exclude (e.g., "text_col,other_col")
        random_state: Random seed for reproducibility

    Outputs:
        model pickle  - dict with keys 'model', 'feature_cols', 'label_encoder'
        metrics json  - training metadata and accuracy
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)
    if feature_exclude:
        extra = [c.strip() for c in feature_exclude.split(",")]
        drop_cols.extend([c for c in extra if c in train_df.columns])

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_train.columns)

    # Label encode target
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[label_column])
    n_classes = len(label_encoder.classes_)

    # Build LinearSVC model
    model = LinearSVC(
        C=C,
        penalty=penalty,
        loss=loss,
        dual=dual,
        max_iter=max_iter,
        random_state=random_state,
    )

    metrics = {
        "model_type": "linear_svc_classifier",
        "C": C,
        "penalty": penalty,
        "loss": loss,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_classes": n_classes,
    }

    # Train the model
    model.fit(X_train, y_train)

    # Calculate training accuracy
    train_preds = model.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, train_preds))
    metrics["train_accuracy"] = train_accuracy

    # Evaluate on validation set if provided
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
        y_valid = label_encoder.transform(valid_df[label_column])

        valid_preds = model.predict(X_valid)
        valid_accuracy = float(accuracy_score(y_valid, valid_preds))
        metrics["valid_accuracy"] = valid_accuracy
        metrics["n_valid_samples"] = len(X_valid)

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "label_encoder": label_encoder,
            "classes_": label_encoder.classes_,
        }, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    acc_str = f"train_acc={train_accuracy:.4f}"
    if "valid_accuracy" in metrics:
        acc_str += f", val_acc={metrics['valid_accuracy']:.4f}"

    return f"train_linear_svc_classifier: {len(X_train)} samples, {n_classes} classes, {acc_str}"


# =============================================================================
# CATBOOST CLASSIFIER (Fast and Reliable)
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
    description="Train CatBoost classifier - fast, reliable, handles categorical features natively",
    tags=["modeling", "training", "catboost", "classification", "multiclass", "generic"],
    version="1.0.0",
)
def train_catboost_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    iterations: int = 1000,
    learning_rate: float = 0.1,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    early_stopping_rounds: int = 100,
    random_state: int = 42,
    verbose: bool = False,
) -> str:
    """Train a CatBoost classifier - fast, reliable, handles categorical features natively.

    CatBoost is particularly effective for:
    - Datasets with categorical features (no explicit encoding needed)
    - Multiclass classification problems
    - Fast training with good default hyperparameters

    G1 Compliance: Generic, works with any classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All parameters exposed.

    Parameters:
        label_column: Column containing class labels
        id_column: ID column to exclude from features
        iterations: Number of boosting iterations (trees)
        learning_rate: Learning rate for gradient descent
        depth: Depth of trees
        l2_leaf_reg: L2 regularization coefficient
        early_stopping_rounds: Stop if no improvement after N rounds
        random_state: Random seed for reproducibility
        verbose: Print training progress

    Outputs:
        model pickle - dict with 'model', 'feature_cols', 'label_encoder'
        metrics json - training/validation metrics
    """
    from catboost import CatBoostClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score, accuracy_score

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_train.columns)

    # Handle string labels with LabelEncoder
    label_encoder = None
    y_raw = train_df[label_column]
    if not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_raw)
    else:
        y_train = y_raw.astype(int).values

    n_classes = len(np.unique(y_train))

    # Identify categorical columns for CatBoost
    cat_features = [i for i, col in enumerate(feature_cols)
                    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category']

    # Create CatBoost model
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_state,
        verbose=verbose,
        cat_features=cat_features if cat_features else None,
        early_stopping_rounds=early_stopping_rounds if inputs.get("valid_data") else None,
    )

    metrics = {
        "model_type": "catboost_classifier",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_classes": n_classes,
        "n_cat_features": len(cat_features),
    }

    # Train with or without validation
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")

        y_valid_raw = valid_df[label_column]
        if label_encoder is not None:
            y_valid = label_encoder.transform(y_valid_raw)
        else:
            y_valid = y_valid_raw.astype(int).values

        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=verbose)

        # Compute validation metrics
        valid_preds = model.predict(X_valid)
        valid_accuracy = float(accuracy_score(y_valid, valid_preds))
        metrics["valid_accuracy"] = valid_accuracy

        if n_classes <= 2:
            proba = model.predict_proba(X_valid)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_valid, proba))
        else:
            proba = model.predict_proba(X_valid)
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_valid, proba, multi_class="ovr"))
            except Exception:
                pass

        metrics["best_iteration"] = int(model.best_iteration_) if hasattr(model, 'best_iteration_') else iterations
        metrics["n_valid_samples"] = len(X_valid)
    else:
        model.fit(X_train, y_train, verbose=verbose)

    # Training metrics
    train_preds = model.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, train_preds))
    metrics["train_accuracy"] = train_accuracy

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    artifact = {"model": model, "feature_cols": feature_cols}
    if label_encoder is not None:
        artifact["label_encoder"] = label_encoder
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    auc_str = f", AUC={metrics['roc_auc']:.4f}" if metrics.get("roc_auc") is not None else ""
    return f"train_catboost_classifier: {len(X_train)} samples, {len(feature_cols)} features{auc_str}"


# =============================================================================
# STACKING CLASSIFIER (XGB + CatBoost + LightGBM + LogisticRegression Meta)
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
    description="Train stacking classifier with XGB+CatBoost+LightGBM and LogisticRegression meta-learner",
    tags=["modeling", "training", "stacking", "ensemble", "classification", "generic"],
    version="1.0.0",
)
def train_stacking_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    random_state: int = 42,
    # XGBoost params (from top solution)
    xgb_n_estimators: int = 1000,
    xgb_max_depth: int = 13,
    xgb_learning_rate: float = 0.04,
    xgb_subsample: float = 0.60,
    xgb_colsample_bytree: float = 0.4,
    xgb_min_child_weight: int = 12,
    # CatBoost params
    cat_iterations: int = 500,
    cat_depth: int = 8,
    cat_learning_rate: float = 0.05,
    # LightGBM params
    lgb_n_estimators: int = 300,
    lgb_num_leaves: int = 63,
    lgb_max_depth: int = 10,
    lgb_learning_rate: float = 0.05,
) -> str:
    """Train a stacking classifier using XGBoost, CatBoost, LightGBM as base models
    and LogisticRegression as meta-learner.

    Based on top Kaggle solutions for playground-series-s4e8 (mushroom classification).
    This stacking approach typically achieves higher MCC scores than simple ensembles.

    G1 Compliance: Generic, works with any binary classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All hyperparameters exposed as parameters.

    Outputs:
        model pickle - dict with 'model' (StackingClassifier), 'feature_cols', 'label_encoder'
        metrics json - validation metrics including MCC, accuracy, ROC-AUC
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_train.columns)

    # Handle string labels with LabelEncoder
    label_encoder = None
    y_raw = train_df[label_column]
    if not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_raw)
    else:
        y_train = y_raw.astype(int).values

    # Build base estimators with tuned parameters from top solutions
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        min_child_weight=xgb_min_child_weight,
        reg_alpha=0.0002,
        gamma=5.6e-08,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )

    cat_model = CatBoostClassifier(
        iterations=cat_iterations,
        depth=cat_depth,
        learning_rate=cat_learning_rate,
        l2_leaf_reg=3,
        random_seed=random_state,
        verbose=0,
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=lgb_n_estimators,
        num_leaves=lgb_num_leaves,
        max_depth=lgb_max_depth,
        learning_rate=lgb_learning_rate,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    # Create stacking classifier with LogisticRegression as meta-learner
    stacking_clf = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('catboost', cat_model),
            ('lgbm', lgb_model),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=5,
        n_jobs=-1,
    )

    # Train stacking classifier
    stacking_clf.fit(X_train, y_train)

    metrics = {
        "model_type": "stacking_classifier",
        "base_models": ["xgboost", "catboost", "lightgbm"],
        "meta_learner": "logistic_regression",
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
    }

    # Evaluate on validation set if provided
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")

        y_valid_raw = valid_df[label_column]
        if label_encoder is not None:
            y_valid = label_encoder.transform(y_valid_raw)
        else:
            y_valid = y_valid_raw.astype(int).values

        # Predictions
        y_pred = stacking_clf.predict(X_valid)
        y_proba = stacking_clf.predict_proba(X_valid)[:, 1]

        # Metrics
        metrics["valid_accuracy"] = float(accuracy_score(y_valid, y_pred))
        metrics["roc_auc"] = float(roc_auc_score(y_valid, y_proba))
        metrics["mcc"] = float(matthews_corrcoef(y_valid, y_pred))
        metrics["n_valid_samples"] = len(X_valid)

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    artifact = {
        "model": stacking_clf,
        "feature_cols": feature_cols,
    }
    if label_encoder is not None:
        artifact["label_encoder"] = label_encoder
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    mcc_str = f", MCC={metrics['mcc']:.5f}" if metrics.get("mcc") is not None else ""
    return f"train_stacking_classifier: {len(X_train)} samples, {len(feature_cols)} features{mcc_str}"


# =============================================================================
# EXTRATREES CLASSIFIER (Fast, Reliable for Multiclass)
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
    description="Train ExtraTrees classifier - fast and effective for multiclass problems, especially with PCA",
    tags=["modeling", "training", "extratrees", "classification", "multiclass", "generic"],
    version="1.0.0",
)
def train_extratrees_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = None,
    n_estimators: int = 500,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    class_weight: str = None,
    random_state: int = 42,
) -> str:
    """Train an ExtraTrees classifier - extremely effective for multiclass classification.

    Based on top Kaggle solution for tabular-playground-series-feb-2022 (siaa512).
    ExtraTrees often outperforms Random Forest and gradient boosting on certain datasets,
    especially when combined with PCA dimensionality reduction.

    G1 Compliance: Generic, works with any classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All parameters exposed.

    Parameters:
        label_column: Column containing class labels
        id_column: ID column to exclude from features
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None for unlimited)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        max_features: Features to consider at each split ('sqrt', 'log2', None)
        class_weight: Class weights ('balanced' or None)
        random_state: Random seed for reproducibility

    Outputs:
        model pickle - dict with 'model', 'feature_cols', 'label_encoder'
        metrics json - training/validation metrics
    """
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score, accuracy_score

    train_df = _load_data(inputs["train_data"])

    # Determine columns to exclude from features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X_train.columns)

    # Handle string labels with LabelEncoder
    label_encoder = None
    y_raw = train_df[label_column]
    if not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_raw)
    else:
        y_train = y_raw.astype(int).values

    n_classes = len(np.unique(y_train))

    # Parse max_depth: allow None via string "None"
    parsed_max_depth = None if max_depth is None or str(max_depth).lower() == "none" else int(max_depth)

    # Create ExtraTrees model
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=parsed_max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Training metrics
    train_preds = model.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, train_preds))

    metrics = {
        "model_type": "extratrees_classifier",
        "n_estimators": n_estimators,
        "max_depth": parsed_max_depth,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_classes": n_classes,
        "train_accuracy": train_accuracy,
    }

    # Evaluate on validation set if provided
    if inputs.get("valid_data") and os.path.exists(inputs["valid_data"]):
        valid_df = _load_data(inputs["valid_data"])
        X_valid = valid_df.drop(columns=drop_cols, errors="ignore")

        y_valid_raw = valid_df[label_column]
        if label_encoder is not None:
            y_valid = label_encoder.transform(y_valid_raw)
        else:
            y_valid = y_valid_raw.astype(int).values

        # Predictions
        valid_preds = model.predict(X_valid)
        valid_accuracy = float(accuracy_score(y_valid, valid_preds))
        metrics["valid_accuracy"] = valid_accuracy

        if n_classes <= 2:
            proba = model.predict_proba(X_valid)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_valid, proba))
        else:
            proba = model.predict_proba(X_valid)
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_valid, proba, multi_class="ovr"))
            except Exception:
                pass

        metrics["n_valid_samples"] = len(X_valid)

    # Save model artifact
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    artifact = {"model": model, "feature_cols": feature_cols}
    if label_encoder is not None:
        artifact["label_encoder"] = label_encoder
    with open(outputs["model"], "wb") as f:
        pickle.dump(artifact, f)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    acc_str = f"train_acc={train_accuracy:.4f}"
    if "valid_accuracy" in metrics:
        acc_str += f", val_acc={metrics['valid_accuracy']:.4f}"

    return f"train_extratrees_classifier: {len(X_train)} samples, {len(feature_cols)} features, {acc_str}"


# =============================================================================
# K-FOLD LIGHTGBM CLASSIFIER WITH MULTI-SEED AVERAGING
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train LightGBM classifier via K-Fold CV with multi-seed averaging for robust predictions",
    tags=["modeling", "training", "lightgbm", "kfold", "classification", "binary", "generic"],
    version="1.0.0",
)
def train_kfold_lightgbm_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = "id",
    prediction_column: str = "target",
    n_folds: int = 5,
    n_seeds: int = 3,
    n_estimators: int = 2000,
    learning_rate: float = 0.02,
    num_leaves: int = 64,
    max_depth: int = 10,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    early_stopping_rounds: int = 100,
    random_state: int = 42,
) -> str:
    """Train LightGBM via K-Fold CV with multi-seed averaging for robust predictions.

    Instead of training a single model, trains n_folds * n_seeds models
    and averages their test-set predictions. This produces more robust
    probability estimates for binary classification.

    G1 Compliance: Generic, works with any binary classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All parameters exposed.

    Outputs:
        model pickle - dict with 'models' list and 'feature_cols'
        predictions CSV - CV-averaged probability predictions on test_data
        metrics json - CV AUC and metadata
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    # Drop rows with missing target
    train_df = train_df.dropna(subset=[label_column])

    # Prepare features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    X = train_df[feature_cols].values
    y = train_df[label_column].astype(int).values
    X_test = test_df[feature_cols].values

    # Handle missing values
    train_median = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = train_median[i]
        mask_test = np.isnan(X_test[:, i])
        X_test[mask_test, i] = train_median[i]

    # K-fold CV with multi-seed averaging
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []

    for seed_idx in range(n_seeds):
        seed = random_state + seed_idx * 100
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=seed + fold,
                n_jobs=-1,
                verbose=-1,
                objective='binary',
            )

            callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric='auc', callbacks=callbacks)

            # OOF predictions
            oof_preds[val_idx] += model.predict_proba(X_val)[:, 1] / n_seeds

            # Test predictions
            test_preds += model.predict_proba(X_test)[:, 1] / (n_folds * n_seeds)

            models.append(model)

    # Final OOF score
    cv_auc = float(roc_auc_score(y, oof_preds))

    # Build result predictions DataFrame
    result = pd.DataFrame()
    if id_column and id_column in test_df.columns:
        result[id_column] = test_df[id_column]
    result[prediction_column] = test_preds

    # Save outputs
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({
            "models": models,
            "feature_cols": feature_cols,
            "weights": [1.0 / len(models)] * len(models),
            "n_folds": n_folds,
            "n_seeds": n_seeds,
        }, f)

    _save_data(result, outputs["predictions"])

    metrics = {
        "model_type": "kfold_lightgbm_multi_seed",
        "n_folds": n_folds,
        "n_seeds": n_seeds,
        "n_models": len(models),
        "cv_roc_auc": cv_auc,
        "n_features": len(feature_cols),
        "n_train_samples": len(X),
        "n_test_samples": len(X_test),
        "mean_test_proba": float(test_preds.mean()),
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_kfold_lightgbm_classifier: {len(models)} models, CV AUC={cv_auc:.4f}, {len(X_test)} test predictions"


# =============================================================================
# KFOLD XGBOOST CLASSIFIER
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train XGBoost classifier via K-Fold CV - fast, robust, widely available",
    tags=["modeling", "training", "xgboost", "kfold", "classification", "binary", "generic"],
    version="1.0.0",
)
def train_kfold_xgboost_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = "id",
    prediction_column: str = "target",
    n_folds: int = 5,
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    max_depth: int = 9,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.2,
    reg_lambda: float = 0.1,
    early_stopping_rounds: int = 100,
    random_state: int = 42,
) -> str:
    """Train XGBoost via K-Fold CV for robust predictions.

    Based on top Kaggle solution insights for binary classification.

    G1 Compliance: Generic, works with any binary classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All parameters exposed.
    """
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    train_df = train_df.dropna(subset=[label_column])

    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    X = train_df[feature_cols].values
    y = train_df[label_column].astype(int).values
    X_test = test_df[feature_cols].values

    train_median = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = train_median[i]
        mask_test = np.isnan(X_test[:, i])
        X_test[mask_test, i] = train_median[i]

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    fold_aucs = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state + fold,
            n_jobs=-1,
            eval_metric='auc',
            early_stopping_rounds=early_stopping_rounds,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)

        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

        models.append(model)
        print(f"  Fold {fold+1}/{n_folds}: AUC={fold_auc:.5f}")

    cv_auc = float(roc_auc_score(y, oof_preds))
    print(f"  CV AUC: {cv_auc:.5f} (+/- {np.std(fold_aucs):.5f})")

    result = pd.DataFrame()
    if id_column and id_column in test_df.columns:
        result[id_column] = test_df[id_column]
    result[prediction_column] = test_preds

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols, "n_folds": n_folds}, f)

    _save_data(result, outputs["predictions"])

    metrics = {
        "model_type": "kfold_xgboost",
        "n_folds": n_folds,
        "n_models": len(models),
        "cv_roc_auc": cv_auc,
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
        "fold_aucs": fold_aucs,
        "n_features": len(feature_cols),
        "n_train_samples": len(X),
        "n_test_samples": len(X_test),
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_kfold_xgboost_classifier: {len(models)} models, CV AUC={cv_auc:.4f}"


# =============================================================================
# KFOLD CATBOOST CLASSIFIER
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train CatBoost classifier via K-Fold CV - fast, handles categorical features natively",
    tags=["modeling", "training", "catboost", "kfold", "classification", "binary", "generic"],
    version="1.0.0",
)
def train_kfold_catboost_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "target",
    id_column: str = "id",
    prediction_column: str = "target",
    n_folds: int = 5,
    iterations: int = 5000,
    learning_rate: float = 0.05,
    depth: int = 9,
    l2_leaf_reg: float = 0.5,
    random_strength: float = 0.0,
    early_stopping_rounds: int = 200,
    all_categorical: bool = True,
    random_state: int = 42,
) -> str:
    """Train CatBoost via K-Fold CV for robust predictions.

    Based on 1st place solution insights: CatBoost with all features as categorical
    tends to perform better on tabular classification problems.

    G1 Compliance: Generic, works with any binary classification dataset.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All parameters exposed.

    Parameters:
        label_column: Column containing class labels
        id_column: ID column to exclude from features
        prediction_column: Column name for predictions in output
        n_folds: Number of CV folds
        iterations: Number of boosting iterations
        learning_rate: Learning rate
        depth: Tree depth
        l2_leaf_reg: L2 regularization coefficient
        random_strength: Random strength for scoring splits
        early_stopping_rounds: Stop if no improvement
        all_categorical: Treat all features as categorical (1st place insight)
        random_state: Random seed

    Outputs:
        model pickle - dict with 'models' list and 'feature_cols'
        predictions CSV - CV-averaged probability predictions on test_data
        metrics json - CV AUC and metadata
    """
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    # Drop rows with missing target
    train_df = train_df.dropna(subset=[label_column])

    # Prepare features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    X = train_df[feature_cols]
    y = train_df[label_column].astype(int).values
    X_test = test_df[feature_cols]

    # Handle categorical features - 1st place insight: all features as categorical
    if all_categorical:
        cat_features = list(range(len(feature_cols)))
    else:
        cat_features = [i for i, col in enumerate(feature_cols)
                        if X[col].dtype == 'object' or X[col].dtype.name == 'category']

    # K-fold CV
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    fold_aucs = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            random_seed=random_state + fold,
            verbose=False,
            cat_features=cat_features if cat_features else None,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric='AUC',
            loss_function='Logloss',
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        # OOF predictions
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)

        # Test predictions
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

        models.append(model)
        print(f"  Fold {fold+1}/{n_folds}: AUC={fold_auc:.5f}")

    # Final OOF score
    cv_auc = float(roc_auc_score(y, oof_preds))
    print(f"  CV AUC: {cv_auc:.5f} (+/- {np.std(fold_aucs):.5f})")

    # Build result predictions DataFrame
    result = pd.DataFrame()
    if id_column and id_column in test_df.columns:
        result[id_column] = test_df[id_column]
    result[prediction_column] = test_preds

    # Save outputs
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({
            "models": models,
            "feature_cols": feature_cols,
            "cat_features": cat_features,
            "n_folds": n_folds,
        }, f)

    _save_data(result, outputs["predictions"])

    metrics = {
        "model_type": "kfold_catboost",
        "n_folds": n_folds,
        "n_models": len(models),
        "cv_roc_auc": cv_auc,
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
        "fold_aucs": fold_aucs,
        "n_features": len(feature_cols),
        "n_cat_features": len(cat_features) if cat_features else 0,
        "n_train_samples": len(X),
        "n_test_samples": len(X_test),
        "all_categorical": all_categorical,
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_kfold_catboost_classifier: {len(models)} models, CV AUC={cv_auc:.4f}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "train_kfold_xgboost_classifier": train_kfold_xgboost_classifier,
    "train_kfold_catboost_classifier": train_kfold_catboost_classifier,
    "train_kfold_lightgbm_classifier": train_kfold_lightgbm_classifier,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "train_extratrees_classifier": train_extratrees_classifier,
    "train_catboost_classifier": train_catboost_classifier,
    "train_stacking_classifier": train_stacking_classifier,
    "train_keras_nn_classifier": train_keras_nn_classifier,
    "train_random_forest_classifier": train_random_forest_classifier,
    "train_xgboost_classifier": train_xgboost_classifier,
    "train_automl_classifier": train_automl_classifier,
    "train_ensemble_classifier": train_ensemble_classifier,
    "train_logistic_classifier": train_logistic_classifier,
    "train_cv_logistic_classifier": train_cv_logistic_classifier,
    "train_linear_svc_classifier": train_linear_svc_classifier,
    "predict_classifier": predict_classifier,
    "predict_multiclass_submission": predict_multiclass_submission,
}
