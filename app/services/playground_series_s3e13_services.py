"""
Playground Series S3E13 - SLEGO Services
=========================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e13
Problem Type: Multiclass Classification (11 vector-borne diseases)
Target: prognosis (string labels)
ID Column: id

Classify vector-borne diseases from binary symptom indicators.
Features are 64 binary symptom columns (0.0/1.0).
Target has 11 disease classes. Submission requires top-3 predictions
as space-separated disease names per row.

Best Approach: Stacking ensemble with feature engineering (score: 0.39624)
- 6 base models: 2xXGBoost, 2xLightGBM, RF, ExtraTrees
- Logistic Regression meta-learner
- Feature engineering: symptom clusters (fever, hemorrhagic, neuro, pain, skin, eye, GI)
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
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.preprocessing_services import (
        split_data, fit_encoder, transform_encoder,
        encode_all_categorical, create_submission
    )
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import (
        split_data, fit_encoder, transform_encoder,
        encode_all_categorical, create_submission
    )


# =============================================================================
# FEATURE ENGINEERING HELPER
# =============================================================================

def _add_symptom_features(df):
    """Add symptom cluster features for disease classification."""
    feature_cols = [c for c in df.columns if c not in ['id', 'prognosis']]

    # Total symptom count
    df['total_symptoms'] = df[feature_cols].sum(axis=1)

    # Symptom clusters
    fever_cols = ['sudden_fever', 'hyperpyrexia', 'chills', 'rigor']
    df['fever_cluster'] = df[[c for c in fever_cols if c in df.columns]].sum(axis=1)

    bleed_cols = ['mouth_bleed', 'nose_bleed', 'gum_bleed', 'gastro_bleeding']
    df['hemorrhagic_cluster'] = df[[c for c in bleed_cols if c in df.columns]].sum(axis=1)

    neuro_cols = ['confusion', 'coma', 'convulsion', 'paralysis', 'stiff_neck', 'diziness']
    df['neuro_cluster'] = df[[c for c in neuro_cols if c in df.columns]].sum(axis=1)

    pain_cols = ['headache', 'muscle_pain', 'joint_pain', 'abdominal_pain', 'back_pain', 'neck_pain']
    df['pain_cluster'] = df[[c for c in pain_cols if c in df.columns]].sum(axis=1)

    skin_cols = ['rash', 'skin_lesions', 'yellow_skin', 'itchiness', 'bullseye_rash']
    df['skin_cluster'] = df[[c for c in skin_cols if c in df.columns]].sum(axis=1)

    eye_cols = ['red_eyes', 'yellow_eyes', 'light_sensitivity', 'orbital_pain']
    df['eye_cluster'] = df[[c for c in eye_cols if c in df.columns]].sum(axis=1)

    gi_cols = ['vomiting', 'diarrhea', 'nausea', 'stomach_pain', 'digestion_trouble', 'loss_of_appetite']
    df['gi_cluster'] = df[[c for c in gi_cols if c in df.columns]].sum(axis=1)

    return df


# =============================================================================
# STACKING ENSEMBLE SERVICE
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train stacking ensemble with feature engineering for multiclass classification",
    tags=["modeling", "training", "ensemble", "stacking", "multiclass", "generic"],
    version="2.0.0"
)
def train_stacking_multiclass_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "prognosis",
    id_column: str = "id",
    n_estimators: int = 500,
    n_folds: int = 5,
    random_state: int = 42,
) -> str:
    """
    Train a stacking ensemble with 6 base models and logistic regression meta-learner.

    Base models: 2xXGBoost, 2xLightGBM, RandomForest, ExtraTrees
    Meta-learner: Logistic Regression
    Feature engineering: Adds symptom cluster features

    G1 Compliance: Generic, works with any multiclass classification problem.
    G4 Compliance: All hyperparameters are configurable.
    """
    import lightgbm as lgb
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    np.random.seed(random_state)

    # Load and preprocess data
    train_df = _load_data(inputs["train_data"])
    train_df = _add_symptom_features(train_df)

    # Prepare features
    drop_cols = [label_column]
    if id_column and id_column in train_df.columns:
        drop_cols.append(id_column)

    X_train = train_df.drop(columns=drop_cols, errors="ignore").values
    y_train = train_df[label_column].values
    feature_cols = list(train_df.drop(columns=drop_cols, errors="ignore").columns)

    # Encode labels if string
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    # Define base models
    models = {
        'xgb1': XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=0.03,
                              subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
                              objective='multi:softprob', random_state=random_state, n_jobs=-1, eval_metric='mlogloss'),
        'xgb2': XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                              objective='multi:softprob', random_state=random_state+100, n_jobs=-1, eval_metric='mlogloss'),
        'lgb1': lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=0.03,
                                   subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
                                   random_state=random_state, n_jobs=-1, verbose=-1),
        'lgb2': lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=32, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                                   random_state=random_state+200, n_jobs=-1, verbose=-1),
        'rf': RandomForestClassifier(n_estimators=n_estimators, max_depth=10, min_samples_split=3,
                                     random_state=random_state, n_jobs=-1),
        'et': ExtraTreesClassifier(n_estimators=n_estimators, max_depth=10, min_samples_split=3,
                                   random_state=random_state, n_jobs=-1),
    }

    # K-fold stacking
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = {name: np.zeros((len(X_train), n_classes)) for name in models}
    trained_models = {name: [] for name in models}

    for name, model in models.items():
        for train_idx, val_idx in kfold.split(X_train, y_encoded):
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train[train_idx], y_encoded[train_idx])
            oof_preds[name][val_idx] = model_clone.predict_proba(X_train[val_idx])
            trained_models[name].append(model_clone)

    # Stack predictions and train meta-learner
    oof_stack = np.hstack([oof_preds[name] for name in models])
    meta_model = LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, n_jobs=-1)
    meta_model.fit(oof_stack, y_encoded)

    # Save model artifact
    model_artifact = {
        "trained_models": trained_models,
        "meta_model": meta_model,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "model_names": list(models.keys()),
        "n_folds": n_folds,
        "n_classes": n_classes,
        "model_type": "stacking_multiclass_ensemble"
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_artifact, f)

    # Save metrics
    metrics = {
        "model_type": "stacking_multiclass_ensemble",
        "n_estimators": n_estimators,
        "n_folds": n_folds,
        "n_base_models": len(models),
        "n_train_samples": len(X_train),
        "n_features": len(feature_cols),
        "n_classes": n_classes
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_stacking_multiclass_ensemble: {len(X_train)} samples, {len(models)} base models + meta-learner"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Generate top-K multiclass predictions using stacking ensemble",
    tags=["inference", "multiclass", "top-k", "submission", "stacking", "generic"],
    version="2.0.0"
)
def predict_stacking_top_k(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "prognosis",
    k: int = 3,
    separator: str = " ",
) -> str:
    """
    Generate top-K class predictions using stacking ensemble.

    Applies feature engineering, gets predictions from all base models,
    stacks them, and uses meta-learner for final predictions.
    """
    # Load model artifact
    with open(inputs["model"], "rb") as f:
        model_artifact = pickle.load(f)

    # Load and preprocess test data
    test_df = _load_data(inputs["data"])
    test_df = _add_symptom_features(test_df)

    # Extract components
    trained_models = model_artifact["trained_models"]
    meta_model = model_artifact["meta_model"]
    label_encoder = model_artifact["label_encoder"]
    feature_cols = model_artifact["feature_cols"]
    model_names = model_artifact["model_names"]
    n_folds = model_artifact["n_folds"]
    n_classes = model_artifact["n_classes"]

    # Prepare features
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    X_test = test_df[feature_cols].values

    # Get predictions from all base models (average across folds)
    test_preds = {name: np.zeros((len(X_test), n_classes)) for name in model_names}
    for name in model_names:
        for model in trained_models[name]:
            test_preds[name] += model.predict_proba(X_test) / n_folds

    # Stack predictions
    test_stack = np.hstack([test_preds[name] for name in model_names])

    # Meta-learner predictions
    meta_proba = meta_model.predict_proba(test_stack)

    # Also compute weighted average blend
    weights = {'xgb1': 0.15, 'xgb2': 0.15, 'lgb1': 0.15, 'lgb2': 0.15, 'rf': 0.2, 'et': 0.2}
    blend_proba = sum(test_preds[name] * weights.get(name, 1/6) for name in model_names)

    # Final blend of meta-learner and weighted average
    final_proba = meta_proba * 0.6 + blend_proba * 0.4

    # Get top-K predictions
    top_k_indices = np.argsort(final_proba, axis=1)[:, -k:][:, ::-1]

    # Decode labels
    top_k_labels = []
    for row in top_k_indices:
        str_labels = label_encoder.inverse_transform(row)
        top_k_labels.append(separator.join(str_labels))

    # Build submission
    submission = pd.DataFrame()
    if id_column in test_df.columns:
        submission[id_column] = test_df[id_column]
    submission[prediction_column] = top_k_labels

    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    return f"predict_stacking_top_k: {len(submission)} rows, top-{k} predictions"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific stacking services
    "train_stacking_multiclass_ensemble": train_stacking_multiclass_ensemble,
    "predict_stacking_top_k": predict_stacking_top_k,
    # Imported from common modules
    "split_data": split_data,
    "fit_encoder": fit_encoder,
    "transform_encoder": transform_encoder,
    "encode_all_categorical": encode_all_categorical,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}
