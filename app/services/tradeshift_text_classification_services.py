"""
Tradeshift Text Classification - Contract-Composable Analytics Services
=================================================
Competition: https://www.kaggle.com/competitions/tradeshift-text-classification
Problem Type: Multi-label Binary Classification (33 targets)
Metric: Multi-column LogLoss

Based on top solution notebooks:
- Solution 03 (rohanrao): Uses FTRL from datatable - fast for sparse data
- Solution 01 (tarunaryyan): Uses model-based imputation for missing values
- Submission format: id_label (e.g., "1700001_y1"), pred (probability)

REUSED SERVICES:
- datatable.models.Ftrl: Fast online learning for sparse data
"""

import os
import sys
import json
import pickle
import gzip
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract


@contract(
    inputs={
        "train_data": {"format": "csv.gz", "required": True},
        "train_labels": {"format": "csv.gz", "required": True},
        "test_data": {"format": "csv.gz", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Train optimized SGD multi-label classifier for all 33 targets (fast for sparse data)",
    tags=["modeling", "training", "multilabel", "classification", "tradeshift", "sgd"],
    version="2.0.0",
)
def train_multilabel_sgd(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    alpha: float = 0.0001,
    loss: str = "log_loss",
    penalty: str = "l2",
    max_iter: int = 1000,
    tol: float = 1e-4,
    sample_train: Optional[int] = None,
    random_state: int = 42,
) -> str:
    """
    Train SGDClassifier (Stochastic Gradient Descent) for all 33 targets.

    SGD is extremely fast for high-dimensional sparse data (similar to FTRL).
    Based on solution notebook insights using online learning.

    Parameters:
        alpha: Regularization parameter (default: 0.0001)
        loss: Loss function - 'log_loss' for logistic regression (default: 'log_loss')
        penalty: Regularization type - 'l1', 'l2', 'elasticnet' (default: 'l2')
        max_iter: Max iterations (default: 1000)
        tol: Tolerance for stopping criterion (default: 1e-4)
        sample_train: If set, use only this many training samples
        random_state: Random seed
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import LabelEncoder
    import time

    start_time = time.time()

    # Load data
    print("Loading training data...")
    with gzip.open(inputs["train_data"], 'rt') as f:
        train_df = pd.read_csv(f, nrows=sample_train)
    print(f"  Train shape: {train_df.shape}")

    print("Loading training labels...")
    with gzip.open(inputs["train_labels"], 'rt') as f:
        labels_df = pd.read_csv(f, nrows=sample_train) if sample_train else pd.read_csv(f)
    print(f"  Labels shape: {labels_df.shape}")

    print("Loading test data...")
    with gzip.open(inputs["test_data"], 'rt') as f:
        test_df = pd.read_csv(f)
    print(f"  Test shape: {test_df.shape}")

    # Get test IDs
    test_ids = test_df['id'].values

    # Prepare features - drop id column
    X_train = train_df.drop(columns=['id'], errors='ignore')
    X_test = test_df.drop(columns=['id'], errors='ignore')

    # Encode categorical features
    print(f"Encoding {len(X_train.columns)} features...")
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'str':
            le = LabelEncoder()
            train_vals = X_train[col].astype(str).fillna('__NA__')
            test_vals = X_test[col].astype(str).fillna('__NA__')
            all_vals = pd.concat([train_vals, test_vals], ignore_index=True)
            le.fit(all_vals)
            X_train[col] = le.transform(train_vals)
            X_test[col] = le.transform(test_vals)

    # Fill NaN and convert to float32 for memory efficiency
    X_train = X_train.fillna(-1).astype(np.float32)
    X_test = X_test.fillna(-1).astype(np.float32)

    # Get target columns
    target_cols = [c for c in labels_df.columns if c.startswith('y')]
    print(f"\nTraining SGD models for {len(target_cols)} targets...")

    all_submissions = []
    metrics = {
        "model_type": "sgd_multilabel",
        "n_targets": len(target_cols),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": len(X_train.columns),
        "per_target_metrics": {}
    }

    for i, target in enumerate(target_cols):
        target_start = time.time()
        print(f"  [{i+1}/{len(target_cols)}] {target}...", end=" ")

        y_train = labels_df[target].values
        pos_rate = y_train.mean()

        # Skip if all same value
        if len(np.unique(y_train)) < 2:
            print(f"skipped (single class)")
            preds = np.full(len(X_test), float(y_train[0]) if len(y_train) > 0 else 0.0)
        else:
            model = SGDClassifier(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]

            target_time = time.time() - target_start
            metrics["per_target_metrics"][target] = {
                "positive_rate": float(pos_rate),
                "train_time_seconds": float(target_time)
            }
            print(f"done (pos_rate={pos_rate:.4f}, time={target_time:.1f}s)")

        target_submission = pd.DataFrame({
            'id_label': [f"{tid}_{target}" for tid in test_ids],
            'pred': preds
        })
        all_submissions.append(target_submission)

    submission = pd.concat(all_submissions, ignore_index=True)
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s")

    # Save
    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    metrics["total_time_seconds"] = float(total_time)
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], 'w') as f:
        json.dump(metrics, f, indent=2)

    return f"train_multilabel_sgd: trained {len(target_cols)} SGD models in {total_time:.1f}s, submission shape {submission.shape}"


@contract(
    inputs={
        "train_data": {"format": "csv.gz", "required": True},
        "train_labels": {"format": "csv.gz", "required": True},
        "test_data": {"format": "csv.gz", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Train multi-label classifier for all 33 targets and generate submission",
    tags=["modeling", "training", "multilabel", "classification", "tradeshift"],
    version="1.0.0",
)
def train_multilabel_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    max_depth: int = -1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    sample_train: Optional[int] = None,
    random_state: int = 42,
) -> str:
    """
    Train LightGBM classifiers for all 33 target labels and generate submission.

    This handles the Tradeshift competition's multi-label format where each row
    in submission corresponds to one (test_id, target) pair.

    Parameters:
        n_estimators: Number of boosting rounds per target
        learning_rate: Learning rate for LightGBM
        num_leaves: Max leaves per tree
        max_depth: Max depth (-1 for unlimited)
        subsample: Subsample ratio of training data
        colsample_bytree: Subsample ratio of columns
        sample_train: If set, use only this many training samples (for faster testing)
        random_state: Random seed
    """
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    # Load data
    print("Loading training data...")
    with gzip.open(inputs["train_data"], 'rt') as f:
        train_df = pd.read_csv(f, nrows=sample_train)

    print("Loading training labels...")
    with gzip.open(inputs["train_labels"], 'rt') as f:
        if sample_train:
            labels_df = pd.read_csv(f, nrows=sample_train)
        else:
            labels_df = pd.read_csv(f)

    print("Loading test data...")
    with gzip.open(inputs["test_data"], 'rt') as f:
        test_df = pd.read_csv(f)

    # Get test IDs
    test_ids = test_df['id'].values

    # Prepare features - drop id column
    X_train = train_df.drop(columns=['id'], errors='ignore')
    X_test = test_df.drop(columns=['id'], errors='ignore')

    # Handle categorical/object columns - convert to numeric using label encoding
    from sklearn.preprocessing import LabelEncoder

    print(f"Encoding {len(X_train.columns)} features...")
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'str':
            le = LabelEncoder()
            # Combine train and test for consistent encoding
            train_vals = X_train[col].astype(str).fillna('__NA__')
            test_vals = X_test[col].astype(str).fillna('__NA__')
            all_vals = pd.concat([train_vals, test_vals], ignore_index=True)
            le.fit(all_vals)
            X_train[col] = le.transform(train_vals)
            X_test[col] = le.transform(test_vals)

    # Fill remaining NaN with -1 and ensure numeric types
    X_train = X_train.fillna(-1).astype(np.float32)
    X_test = X_test.fillna(-1).astype(np.float32)

    # Get target columns (y1 to y33)
    target_cols = [c for c in labels_df.columns if c.startswith('y')]
    print(f"Training models for {len(target_cols)} targets: {target_cols}")

    # Train model for each target and collect predictions
    all_submissions = []
    metrics = {
        "model_type": "lightgbm_multilabel",
        "n_targets": len(target_cols),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": len(X_train.columns),
        "per_target_metrics": {}
    }

    for target in target_cols:
        print(f"\nTraining model for {target}...")
        y_train = labels_df[target].values

        # Skip if all same value (no variation)
        if len(np.unique(y_train)) < 2:
            print(f"  Skipping {target} - only one class present")
            # Predict the constant value
            pred_value = float(y_train[0]) if len(y_train) > 0 else 0.0
            preds = np.full(len(X_test), pred_value)
        else:
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train, y_train)

            # Predict probabilities for positive class
            preds = model.predict_proba(X_test)[:, 1]

            # Calculate train log loss for monitoring
            train_preds = model.predict_proba(X_train)[:, 1]
            train_logloss = log_loss(y_train, train_preds)
            metrics["per_target_metrics"][target] = {
                "train_logloss": float(train_logloss),
                "positive_rate": float(y_train.mean())
            }
            print(f"  {target}: train_logloss={train_logloss:.4f}, pos_rate={y_train.mean():.4f}")

        # Create submission rows for this target
        target_submission = pd.DataFrame({
            'id_label': [f"{tid}_{target}" for tid in test_ids],
            'pred': preds
        })
        all_submissions.append(target_submission)

    # Combine all submissions
    submission = pd.concat(all_submissions, ignore_index=True)

    # Save submission
    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    # Save metrics
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], 'w') as f:
        json.dump(metrics, f, indent=2)

    return f"train_multilabel_classifier: trained {len(target_cols)} models, submission shape {submission.shape}"


@contract(
    inputs={
        "train_data": {"format": "csv.gz", "required": True},
        "train_labels": {"format": "csv.gz", "required": True},
        "test_data": {"format": "csv.gz", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Train optimized LightGBM multi-label classifier with feature engineering from top solutions",
    tags=["modeling", "training", "multilabel", "classification", "tradeshift", "lightgbm", "optimized"],
    version="3.0.0",
)
def train_multilabel_optimized(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    max_depth: int = 8,
    min_child_samples: int = 50,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    max_cardinality: int = 100,
    sample_train: Optional[int] = None,
    random_state: int = 42,
) -> str:
    """
    Train optimized LightGBM classifiers with feature engineering from top solutions.

    Key improvements based on solution notebooks:
    1. Remove high-cardinality categorical columns (>=max_cardinality unique values)
    2. One-hot encode low-cardinality categorical features
    3. Use tuned LightGBM hyperparameters
    4. Proper handling of missing values

    Parameters:
        n_estimators: Number of boosting rounds (default: 200)
        learning_rate: Learning rate (default: 0.05)
        num_leaves: Max leaves per tree (default: 63)
        max_depth: Max depth (default: 8)
        min_child_samples: Min samples in leaf (default: 50)
        subsample: Subsample ratio (default: 0.8)
        colsample_bytree: Column subsample ratio (default: 0.8)
        reg_alpha: L1 regularization (default: 0.1)
        reg_lambda: L2 regularization (default: 0.1)
        max_cardinality: Max unique values for categorical columns (default: 100)
        sample_train: If set, use only this many training samples
        random_state: Random seed
    """
    import lightgbm as lgb
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import LabelEncoder
    import time
    import gc

    start_time = time.time()

    # Load data
    print("Loading training data...")
    with gzip.open(inputs["train_data"], 'rt') as f:
        train_df = pd.read_csv(f, nrows=sample_train)
    print(f"  Train shape: {train_df.shape}")

    print("Loading training labels...")
    with gzip.open(inputs["train_labels"], 'rt') as f:
        labels_df = pd.read_csv(f, nrows=sample_train) if sample_train else pd.read_csv(f)
    print(f"  Labels shape: {labels_df.shape}")

    print("Loading test data...")
    with gzip.open(inputs["test_data"], 'rt') as f:
        test_df = pd.read_csv(f)
    print(f"  Test shape: {test_df.shape}")

    # Get test IDs
    test_ids = test_df['id'].values

    # Combine train and test for consistent preprocessing
    train_df['_is_train'] = 1
    test_df['_is_train'] = 0
    combined = pd.concat([train_df.drop(columns=['id']), test_df.drop(columns=['id'])], ignore_index=True)
    del train_df, test_df
    gc.collect()

    print(f"\nFeature Engineering (based on solution notebooks)...")

    # Step 1: Identify and remove high-cardinality categorical columns
    high_cardinality_cols = []
    low_cardinality_cat_cols = []
    numeric_cols = []

    for col in combined.columns:
        if col == '_is_train':
            continue
        # Check if column contains string values (object dtype or has any string)
        is_categorical = combined[col].dtype == 'object'
        if not is_categorical:
            # Check first non-null value for string type
            first_valid = combined[col].dropna().iloc[0] if combined[col].notna().any() else None
            is_categorical = isinstance(first_valid, str)

        if is_categorical:
            n_unique = combined[col].nunique()
            if n_unique >= max_cardinality:
                high_cardinality_cols.append(col)
            else:
                low_cardinality_cat_cols.append(col)
        else:
            numeric_cols.append(col)

    print(f"  High-cardinality columns to drop (>={max_cardinality} unique): {len(high_cardinality_cols)}")
    print(f"  Low-cardinality categorical columns: {len(low_cardinality_cat_cols)}")
    print(f"  Numeric columns: {len(numeric_cols)}")

    # Drop high-cardinality columns
    combined = combined.drop(columns=high_cardinality_cols)
    gc.collect()

    # Step 2: Label encode low-cardinality categorical columns (LightGBM handles them as categorical)
    print(f"  Label encoding {len(low_cardinality_cat_cols)} categorical columns...")
    for col in low_cardinality_cat_cols:
        le = LabelEncoder()
        combined[col] = combined[col].astype(str).fillna('__MISSING__')
        combined[col] = le.fit_transform(combined[col])

    # Step 3: Fill numeric missing values with -999 (distinct from real values)
    print(f"  Filling missing values...")
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(-999)

    # Split back to train and test
    X_train = combined[combined['_is_train'] == 1].drop(columns=['_is_train'])
    X_test = combined[combined['_is_train'] == 0].drop(columns=['_is_train'])
    del combined
    gc.collect()

    print(f"  Final feature count: {X_train.shape[1]}")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Convert to float32 for memory efficiency
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Get target columns
    target_cols = [c for c in labels_df.columns if c.startswith('y')]
    print(f"\nTraining optimized LightGBM models for {len(target_cols)} targets...")

    all_submissions = []
    metrics = {
        "model_type": "lightgbm_optimized",
        "n_targets": len(target_cols),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "dropped_high_cardinality": len(high_cardinality_cols),
        "per_target_metrics": {}
    }

    for i, target in enumerate(target_cols):
        target_start = time.time()
        print(f"  [{i+1}/{len(target_cols)}] {target}...", end=" ", flush=True)

        y_train = labels_df[target].values
        pos_rate = float(y_train.mean())

        # Skip if all same value
        if len(np.unique(y_train)) < 2:
            print(f"skipped (single class)")
            preds = np.full(len(X_test), float(y_train[0]) if len(y_train) > 0 else 0.0)
        else:
            # Calculate scale_pos_weight for imbalanced classes
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / max(n_pos, 1)

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
                scale_pos_weight=min(scale_pos_weight, 100),  # Cap at 100
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
                force_col_wise=True,
            )
            model.fit(X_train, y_train)

            # Predict probabilities
            preds = model.predict_proba(X_test)[:, 1]

            # Calculate train log loss
            train_preds = model.predict_proba(X_train)[:, 1]
            train_logloss = log_loss(y_train, train_preds)

            target_time = time.time() - target_start
            metrics["per_target_metrics"][target] = {
                "train_logloss": float(train_logloss),
                "positive_rate": pos_rate,
                "train_time_seconds": float(target_time)
            }
            print(f"done (logloss={train_logloss:.4f}, pos={pos_rate:.4f}, time={target_time:.1f}s)")

        target_submission = pd.DataFrame({
            'id_label': [f"{tid}_{target}" for tid in test_ids],
            'pred': preds
        })
        all_submissions.append(target_submission)

    submission = pd.concat(all_submissions, ignore_index=True)
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s")

    # Save
    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    metrics["total_time_seconds"] = float(total_time)
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], 'w') as f:
        json.dump(metrics, f, indent=2)

    return f"train_multilabel_optimized: trained {len(target_cols)} LightGBM models in {total_time:.1f}s, submission shape {submission.shape}"


SERVICE_REGISTRY = {
    "train_multilabel_optimized": train_multilabel_optimized,
    "train_multilabel_sgd": train_multilabel_sgd,
    "train_multilabel_classifier": train_multilabel_classifier,
}
