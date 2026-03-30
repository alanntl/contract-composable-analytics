"""
Playground Series S3E25 - Mohs Hardness Prediction Services
============================================================

Competition: https://www.kaggle.com/competitions/playground-series-s3e25
Problem Type: Regression (MEDAE metric) - treated as 9-class classification
Target: Hardness (continuous, discretized to 9 values)

Key Insight from Top Solutions:
The winning approach treats this regression problem as a 9-class classification.
Target values are discretized to: [1.25, 2.25, 3.05, 4.05, 4.85, 5.75, 6.55, 7.75, 9.25]
This significantly outperforms standard regression approaches.

Services:
- discretize_hardness_target: Convert continuous Hardness to discrete class labels
- map_predictions_to_hardness: Map predicted class indices back to hardness values
- train_catboost_discretized: Train CatBoost on discretized targets with CV
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
from services.io_utils import load_data as _load_data, save_data as _save_data


# Discretized target values from winning solution
HARDNESS_VALUES = [1.25, 2.25, 3.05, 4.05, 4.85, 5.75, 6.55, 7.75, 9.25]


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Discretize continuous Hardness values to 9-class labels",
    tags=["preprocessing", "discretization", "mohs-hardness"],
    version="1.0.0",
)
def discretize_hardness_target(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Hardness",
    hardness_values: List[float] = None,
) -> str:
    """
    Discretize continuous Hardness values to discrete class labels.

    Maps each Hardness value to the nearest value in the predefined set.
    The class label is the index of that value (0-8).

    G1 Compliance: Single responsibility - discretize target column.
    G4 Compliance: Parameterized target column and hardness values.

    Parameters:
        target_column: Name of the target column (default: "Hardness")
        hardness_values: List of discrete hardness values to map to
    """
    df = _load_data(inputs["data"])
    values = hardness_values or HARDNESS_VALUES

    if target_column in df.columns:
        # Map each value to the nearest discrete value, then to class index
        df[target_column] = df[target_column].apply(
            lambda x: np.argmin([abs(x - v) for v in values])
        )

    _save_data(df, outputs["data"])
    return f"discretize_hardness_target: mapped {target_column} to {len(values)} classes"


@contract(
    inputs={
        "predictions": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Map predicted class indices back to Hardness values",
    tags=["postprocessing", "mapping", "mohs-hardness"],
    version="1.0.0",
)
def map_predictions_to_hardness(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    prediction_column: str = "Hardness",
    hardness_values: List[float] = None,
) -> str:
    """
    Map predicted class indices (0-8) back to Hardness values.

    G1 Compliance: Single responsibility - map predictions to values.
    G4 Compliance: Parameterized columns and values.

    Parameters:
        prediction_column: Name of the prediction column (default: "Hardness")
        hardness_values: List of discrete hardness values to map from
    """
    df = _load_data(inputs["predictions"])
    values = hardness_values or HARDNESS_VALUES

    if prediction_column in df.columns:
        df[prediction_column] = df[prediction_column].apply(
            lambda x: values[int(x)] if 0 <= int(x) < len(values) else values[0]
        )

    _save_data(df, outputs["predictions"])
    return f"map_predictions_to_hardness: mapped classes to {len(values)} hardness values"


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
    description="Train LightGBM classifier on discretized Hardness targets with K-fold CV",
    tags=["modeling", "training", "lightgbm", "classification", "mohs-hardness"],
    version="1.0.0",
)
def train_lightgbm_discretized(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Hardness",
    id_column: str = "id",
    n_folds: int = 5,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 8,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.5,
    reg_lambda: float = 0.5,
    random_state: int = 42,
    hardness_values: List[float] = None,
) -> str:
    """
    Train LightGBM classifier with K-fold CV on discretized Hardness targets.

    This implements the winning approach for Playground Series S3E25:
    1. Discretize continuous Hardness to 9 classes
    2. Train LightGBM multiclass classifier with K-fold CV
    3. Predict class probabilities and map back to hardness values

    G1 Compliance: Single responsibility - train and predict.
    G3 Compliance: Fixed random_state for reproducibility.
    G4 Compliance: All parameters exposed.

    Parameters:
        target_column: Target column name (default: "Hardness")
        id_column: ID column name (default: "id")
        n_folds: Number of CV folds (default: 5)
        n_estimators: Number of boosting iterations (default: 500)
        learning_rate: Learning rate (default: 0.05)
        num_leaves: Max number of leaves per tree (default: 31)
        max_depth: Max tree depth (default: 8)
        min_child_samples: Min samples per leaf (default: 20)
        subsample: Row sampling ratio (default: 0.8)
        colsample_bytree: Column sampling ratio (default: 0.8)
        reg_alpha: L1 regularization (default: 0.5)
        reg_lambda: L2 regularization (default: 0.5)
        random_state: Random seed (default: 42)
        hardness_values: Discrete hardness values for mapping
    """
    from lightgbm import LGBMClassifier, early_stopping
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])
    values = hardness_values or HARDNESS_VALUES

    # Discretize target
    y = train_df[target_column].apply(
        lambda x: np.argmin([abs(x - v) for v in values])
    ).values

    # Prepare features
    drop_cols = [target_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X = train_df.drop(columns=drop_cols, errors="ignore")
    feature_cols = list(X.columns)
    X_test = test_df.drop(columns=[id_column] if id_column in test_df.columns else [], errors="ignore")

    # Ensure test has same columns
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols]

    # K-fold CV training
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(X_test), len(values)))
    models = []

    print(f"  Training LightGBM with {n_folds}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state + fold,
            n_jobs=-1,
            verbose=-1,
            objective="multiclass",
            num_class=len(values),
        )

        callbacks = [early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        # OOF predictions
        oof_preds[val_idx] = model.predict(X_val)

        # Test predictions (probabilities)
        test_preds += model.predict_proba(X_test) / n_folds

        models.append(model)

        val_acc = accuracy_score(y_val, oof_preds[val_idx])
        print(f"    Fold {fold + 1}/{n_folds}: Accuracy={val_acc:.4f}")

    # Final test predictions: argmax of averaged probabilities
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_pred_values = np.array([values[int(c)] for c in test_pred_classes])

    # Calculate OOF MEDAE
    oof_values = np.array([values[int(c)] for c in oof_preds])
    true_values = train_df[target_column].values
    oof_medae = float(np.median(np.abs(true_values - oof_values)))
    oof_accuracy = float(accuracy_score(y, oof_preds))

    print(f"  OOF MEDAE: {oof_medae:.4f}")
    print(f"  OOF Accuracy: {oof_accuracy:.4f}")

    # Build predictions DataFrame
    result = pd.DataFrame()
    if id_column and id_column in test_df.columns:
        result[id_column] = test_df[id_column]
    result[target_column] = test_pred_values

    # Save outputs
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({
            "models": models,
            "feature_cols": feature_cols,
            "hardness_values": values,
            "n_folds": n_folds,
        }, f)

    _save_data(result, outputs["predictions"])

    metrics = {
        "model_type": "lightgbm_discretized",
        "n_folds": n_folds,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "n_features": len(feature_cols),
        "n_train_samples": len(X),
        "n_test_samples": len(X_test),
        "n_classes": len(values),
        "oof_medae": oof_medae,
        "oof_accuracy": oof_accuracy,
        "hardness_values": values,
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_lightgbm_discretized: {n_folds}-fold CV, OOF MEDAE={oof_medae:.4f}, Accuracy={oof_accuracy:.4f}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "discretize_hardness_target": discretize_hardness_target,
    "map_predictions_to_hardness": map_predictions_to_hardness,
    "train_lightgbm_discretized": train_lightgbm_discretized,
}
