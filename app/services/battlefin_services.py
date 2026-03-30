"""
BattleFin Big Data Combine Forecasting Challenge - Contract-Composable Analytics Services
==================================================================
Competition: https://www.kaggle.com/competitions/battlefin-s-big-data-combine-forecasting-challenge
Problem Type: Multi-Output Regression (Time-Series Forecasting)
Target: 198 outputs (O1-O198) for each file
ID Column: FileId

Structure:
- 510 CSV files in data/ folder
- Files 1-200: Training data (labels in trainLabels.csv)
- Files 201-510: Test data (predictions needed)
- Each file has time-series with 198 outputs + 244 inputs
- Goal: Predict final values O1-O198 for test files

Key Insight:
- The best approach is using historical means of each output column
- For time-series forecasting, the mean of non-zero historical values
  provides an excellent baseline prediction
- This simple approach outperforms complex ML models for this competition

Services:
- generate_battlefin_submission: Direct mean-based prediction (fast, reliable)
- load_battlefin_data: Load and aggregate all time-series files
- train_battlefin_regressor: Train multi-output regressor (LightGBM)
- predict_battlefin: Generate predictions for test files

Reusable services imported from:
- io_services: load_data, save_data
- regression_services: train_lightgbm_regressor (for advanced modeling)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import reusable services
from services.io_services import load_data, save_data


def contract(inputs=None, outputs=None, params=None, description=None, tags=None, version="1.0.0"):
    """Service contract decorator for Contract-Composable Analytics services."""
    def decorator(func):
        func._contract = {
            'inputs': inputs or {},
            'outputs': outputs or {},
            'params': params or {},
            'description': description or func.__doc__,
            'tags': tags or [],
            'version': version
        }
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._contract = func._contract
        return wrapper
    return decorator


def _compute_historical_means(filepath: str) -> Dict[str, float]:
    """
    Compute historical means of output columns from a time-series file.

    Key insight: Use non-zero rows only for computing means, as zero rows
    often represent missing data in this competition.

    Args:
        filepath: Path to the CSV file

    Returns:
        Dictionary mapping output column names to their historical means
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return {}

        output_cols = [c for c in df.columns if c.startswith('O')]

        # Filter to non-zero rows (key insight for this competition!)
        nonzero_mask = df[output_cols].abs().sum(axis=1) > 0
        nonzero_df = df[nonzero_mask] if nonzero_mask.any() else df

        means = {}
        for col in output_cols:
            col_data = nonzero_df[col]
            means[col] = float(col_data.mean()) if len(col_data) > 0 else 0.0

        return means
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {}


def _compute_last_values(filepath: str) -> Dict[str, float]:
    """
    Get the last non-zero row values for each output column.

    This is often more predictive for financial time-series than means.
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return {}

        output_cols = [c for c in df.columns if c.startswith('O')]

        # Filter to non-zero rows and get the last one
        nonzero_mask = df[output_cols].abs().sum(axis=1) > 0
        nonzero_df = df[nonzero_mask] if nonzero_mask.any() else df

        last_values = {}
        if len(nonzero_df) > 0:
            last_row = nonzero_df.iloc[-1]
            for col in output_cols:
                last_values[col] = float(last_row.get(col, 0))
        else:
            for col in output_cols:
                last_values[col] = 0.0

        return last_values
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {}


def _compute_weighted_prediction(filepath: str, alpha: float = 0.5) -> Dict[str, float]:
    """
    Compute weighted combination of last value and historical mean.

    For financial time series, a blend often works better than either alone:
    prediction = alpha * last_value + (1 - alpha) * historical_mean

    Args:
        filepath: Path to the CSV file
        alpha: Weight for last value (0=pure mean, 1=pure last value)
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return {}

        output_cols = [c for c in df.columns if c.startswith('O')]

        # Filter to non-zero rows
        nonzero_mask = df[output_cols].abs().sum(axis=1) > 0
        nonzero_df = df[nonzero_mask] if nonzero_mask.any() else df

        predictions = {}
        if len(nonzero_df) > 0:
            last_row = nonzero_df.iloc[-1]
            for col in output_cols:
                mean_val = float(nonzero_df[col].mean())
                last_val = float(last_row.get(col, 0))
                predictions[col] = alpha * last_val + (1 - alpha) * mean_val
        else:
            for col in output_cols:
                predictions[col] = 0.0

        return predictions
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {}


def _compute_ewma_prediction(filepath: str, span: int = 5) -> Dict[str, float]:
    """
    Compute exponentially weighted moving average prediction.

    EWMA gives more weight to recent observations, which is often
    better for financial time-series than simple mean.

    Args:
        filepath: Path to the CSV file
        span: EWMA span parameter (smaller = more weight on recent)
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return {}

        output_cols = [c for c in df.columns if c.startswith('O')]

        # Filter to non-zero rows
        nonzero_mask = df[output_cols].abs().sum(axis=1) > 0
        nonzero_df = df[nonzero_mask] if nonzero_mask.any() else df

        predictions = {}
        if len(nonzero_df) > 0:
            for col in output_cols:
                # EWMA of the column, take the last value
                ewma = nonzero_df[col].ewm(span=span, adjust=False).mean()
                predictions[col] = float(ewma.iloc[-1])
        else:
            for col in output_cols:
                predictions[col] = 0.0

        return predictions
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {}


def _compute_recent_mean(filepath: str, n_rows: int = 3) -> Dict[str, float]:
    """
    Compute mean of last N non-zero rows.

    This balances between using just the last row and using all history.

    Args:
        filepath: Path to the CSV file
        n_rows: Number of recent rows to average
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return {}

        output_cols = [c for c in df.columns if c.startswith('O')]

        # Filter to non-zero rows
        nonzero_mask = df[output_cols].abs().sum(axis=1) > 0
        nonzero_df = df[nonzero_mask] if nonzero_mask.any() else df

        predictions = {}
        if len(nonzero_df) > 0:
            # Take last n_rows
            recent_df = nonzero_df.tail(min(n_rows, len(nonzero_df)))
            for col in output_cols:
                predictions[col] = float(recent_df[col].mean())
        else:
            for col in output_cols:
                predictions[col] = 0.0

        return predictions
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {}


def _extract_file_features(filepath: str) -> pd.Series:
    """
    Extract statistical features from a time-series file.
    Uses historical means as baseline features (best approach for this competition).
    """
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return pd.Series()

        # Get column names
        input_cols = [c for c in df.columns if c.startswith('I')]
        output_cols = [c for c in df.columns if c.startswith('O')]

        features = {}

        # Non-zero rows only for outputs (key insight!)
        nonzero_mask = df[output_cols].abs().sum(axis=1) > 0
        nonzero_df = df[nonzero_mask] if nonzero_mask.any() else df

        # Historical means of ALL outputs (KEY FEATURE for forecasting!)
        for col in output_cols:
            col_data = nonzero_df[col]
            features[f'hist_mean_{col}'] = float(col_data.mean()) if len(col_data) > 0 else 0.0

        # Last row values for outputs (recent values matter)
        last_row = df.iloc[-1]
        for col in output_cols:
            features[f'last_{col}'] = float(last_row.get(col, 0))

        # Meta features
        features['n_rows'] = float(len(df))
        features['n_nonzero_rows'] = float(nonzero_mask.sum())
        features['nonzero_ratio'] = features['n_nonzero_rows'] / max(len(df), 1)

        return pd.Series(features)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return pd.Series()


# =============================================================================
# SERVICE 1: DIRECT MEAN-BASED PREDICTION (FAST, RELIABLE - RECOMMENDED)
# =============================================================================

@contract(
    inputs={
        "data_dir": {"format": "directory", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Generate BattleFin predictions using historical means (simple, effective)",
    tags=["prediction", "submission", "time-series", "battlefin", "baseline"],
    version="2.0.0",
)
def generate_battlefin_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    test_file_ids: List[int] = None,
) -> str:
    """
    Generate BattleFin predictions using historical means directly.

    This is the recommended approach for this competition as it:
    1. Uses the historical mean of each output column as the prediction
    2. Is fast and reliable (no model training needed)
    3. Provides excellent baseline performance for time-series forecasting

    Args:
        inputs: Dict with 'data_dir' path and 'sample_submission' path
        outputs: Dict with 'submission' and 'metrics' paths
        test_file_ids: List of test file IDs (default: 201-510)
    """
    data_dir = inputs['data_dir']
    sample_sub = pd.read_csv(inputs['sample_submission'])

    test_file_ids = test_file_ids or list(range(201, 511))
    output_cols = [c for c in sample_sub.columns if c.startswith('O')]

    print(f"Generating predictions for {len(test_file_ids)} test files...")

    submissions = []
    for file_id in test_file_ids:
        filepath = os.path.join(data_dir, f'{file_id}.csv')
        if os.path.exists(filepath):
            means = _compute_historical_means(filepath)
            row = {'FileId': file_id}
            for col in output_cols:
                row[col] = means.get(col, 0.0)
            submissions.append(row)
        else:
            print(f"Warning: File {filepath} not found")

    submission_df = pd.DataFrame(submissions)

    # Ensure columns match sample submission format
    submission_df = submission_df[sample_sub.columns]

    # Save submission
    os.makedirs(os.path.dirname(outputs['submission']) or '.', exist_ok=True)
    submission_df.to_csv(outputs['submission'], index=False)

    # Save metrics
    metrics = {
        'method': 'historical_mean',
        'n_test_files': len(submissions),
        'n_output_cols': len(output_cols),
        'description': 'Direct prediction using historical means of each output column',
    }
    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Submission created: {submission_df.shape}")
    return outputs['submission']


# =============================================================================
# SERVICE 1B: LAST VALUE PREDICTION (OFTEN BETTER FOR FINANCIAL DATA)
# =============================================================================

@contract(
    inputs={
        "data_dir": {"format": "directory", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Generate BattleFin predictions using last row values",
    tags=["prediction", "submission", "time-series", "battlefin", "last-value"],
    version="2.0.0",
)
def generate_battlefin_last_value(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    test_file_ids: List[int] = None,
) -> str:
    """
    Generate BattleFin predictions using last row values.

    For financial time series, the last observed value is often more
    predictive than the historical mean.
    """
    data_dir = inputs['data_dir']
    sample_sub = pd.read_csv(inputs['sample_submission'])

    test_file_ids = test_file_ids or list(range(201, 511))
    output_cols = [c for c in sample_sub.columns if c.startswith('O')]

    print(f"Generating last-value predictions for {len(test_file_ids)} test files...")

    submissions = []
    for file_id in test_file_ids:
        filepath = os.path.join(data_dir, f'{file_id}.csv')
        if os.path.exists(filepath):
            last_vals = _compute_last_values(filepath)
            row = {'FileId': file_id}
            for col in output_cols:
                row[col] = last_vals.get(col, 0.0)
            submissions.append(row)

    submission_df = pd.DataFrame(submissions)
    submission_df = submission_df[sample_sub.columns]

    os.makedirs(os.path.dirname(outputs['submission']) or '.', exist_ok=True)
    submission_df.to_csv(outputs['submission'], index=False)

    metrics = {
        'method': 'last_value',
        'n_test_files': len(submissions),
        'n_output_cols': len(output_cols),
    }
    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Submission created: {submission_df.shape}")
    return outputs['submission']


# =============================================================================
# SERVICE 1C: WEIGHTED BLEND PREDICTION (BEST OF BOTH WORLDS)
# =============================================================================

@contract(
    inputs={
        "data_dir": {"format": "directory", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Generate BattleFin predictions using weighted blend of last value and mean",
    tags=["prediction", "submission", "time-series", "battlefin", "ensemble"],
    version="2.0.0",
)
def generate_battlefin_weighted(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    alpha: float = 0.5,
    test_file_ids: List[int] = None,
) -> str:
    """
    Generate BattleFin predictions using weighted combination.

    prediction = alpha * last_value + (1 - alpha) * historical_mean

    Args:
        alpha: Weight for last value (0=pure mean, 1=pure last value)
    """
    data_dir = inputs['data_dir']
    sample_sub = pd.read_csv(inputs['sample_submission'])

    test_file_ids = test_file_ids or list(range(201, 511))
    output_cols = [c for c in sample_sub.columns if c.startswith('O')]

    print(f"Generating weighted predictions (alpha={alpha}) for {len(test_file_ids)} test files...")

    submissions = []
    for file_id in test_file_ids:
        filepath = os.path.join(data_dir, f'{file_id}.csv')
        if os.path.exists(filepath):
            predictions = _compute_weighted_prediction(filepath, alpha=alpha)
            row = {'FileId': file_id}
            for col in output_cols:
                row[col] = predictions.get(col, 0.0)
            submissions.append(row)

    submission_df = pd.DataFrame(submissions)
    submission_df = submission_df[sample_sub.columns]

    os.makedirs(os.path.dirname(outputs['submission']) or '.', exist_ok=True)
    submission_df.to_csv(outputs['submission'], index=False)

    metrics = {
        'method': 'weighted_blend',
        'alpha': alpha,
        'n_test_files': len(submissions),
        'n_output_cols': len(output_cols),
    }
    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Submission created: {submission_df.shape}")
    return outputs['submission']


# =============================================================================
# SERVICE 1D: EWMA PREDICTION (EXPONENTIALLY WEIGHTED MOVING AVERAGE)
# =============================================================================

@contract(
    inputs={
        "data_dir": {"format": "directory", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Generate BattleFin predictions using EWMA",
    tags=["prediction", "submission", "time-series", "battlefin", "ewma"],
    version="2.0.0",
)
def generate_battlefin_ewma(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    span: int = 5,
    test_file_ids: List[int] = None,
) -> str:
    """
    Generate BattleFin predictions using Exponentially Weighted Moving Average.

    EWMA gives more weight to recent observations while still incorporating
    historical data - often optimal for financial time-series.

    Args:
        span: EWMA span (smaller = more weight on recent values)
    """
    data_dir = inputs['data_dir']
    sample_sub = pd.read_csv(inputs['sample_submission'])

    test_file_ids = test_file_ids or list(range(201, 511))
    output_cols = [c for c in sample_sub.columns if c.startswith('O')]

    print(f"Generating EWMA predictions (span={span}) for {len(test_file_ids)} test files...")

    submissions = []
    for file_id in test_file_ids:
        filepath = os.path.join(data_dir, f'{file_id}.csv')
        if os.path.exists(filepath):
            predictions = _compute_ewma_prediction(filepath, span=span)
            row = {'FileId': file_id}
            for col in output_cols:
                row[col] = predictions.get(col, 0.0)
            submissions.append(row)

    submission_df = pd.DataFrame(submissions)
    submission_df = submission_df[sample_sub.columns]

    os.makedirs(os.path.dirname(outputs['submission']) or '.', exist_ok=True)
    submission_df.to_csv(outputs['submission'], index=False)

    metrics = {
        'method': 'ewma',
        'span': span,
        'n_test_files': len(submissions),
        'n_output_cols': len(output_cols),
    }
    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Submission created: {submission_df.shape}")
    return outputs['submission']


# =============================================================================
# SERVICE 1E: RECENT MEAN PREDICTION
# =============================================================================

@contract(
    inputs={
        "data_dir": {"format": "directory", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Generate BattleFin predictions using mean of recent N rows",
    tags=["prediction", "submission", "time-series", "battlefin", "recent-mean"],
    version="2.0.0",
)
def generate_battlefin_recent_mean(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_rows: int = 3,
    test_file_ids: List[int] = None,
) -> str:
    """
    Generate BattleFin predictions using mean of last N non-zero rows.

    Balances between using just the last row and using all historical data.

    Args:
        n_rows: Number of recent rows to average
    """
    data_dir = inputs['data_dir']
    sample_sub = pd.read_csv(inputs['sample_submission'])

    test_file_ids = test_file_ids or list(range(201, 511))
    output_cols = [c for c in sample_sub.columns if c.startswith('O')]

    print(f"Generating recent-mean predictions (n_rows={n_rows}) for {len(test_file_ids)} test files...")

    submissions = []
    for file_id in test_file_ids:
        filepath = os.path.join(data_dir, f'{file_id}.csv')
        if os.path.exists(filepath):
            predictions = _compute_recent_mean(filepath, n_rows=n_rows)
            row = {'FileId': file_id}
            for col in output_cols:
                row[col] = predictions.get(col, 0.0)
            submissions.append(row)

    submission_df = pd.DataFrame(submissions)
    submission_df = submission_df[sample_sub.columns]

    os.makedirs(os.path.dirname(outputs['submission']) or '.', exist_ok=True)
    submission_df.to_csv(outputs['submission'], index=False)

    metrics = {
        'method': 'recent_mean',
        'n_rows': n_rows,
        'n_test_files': len(submissions),
        'n_output_cols': len(output_cols),
    }
    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Submission created: {submission_df.shape}")
    return outputs['submission']


# =============================================================================
# SERVICE 2: LOAD AND EXTRACT FEATURES (FOR ML-BASED APPROACH)
# =============================================================================

@contract(
    inputs={
        "data_dir": {"format": "directory", "required": True},
        "labels": {"format": "csv", "required": True},
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"},
        "labels_processed": {"format": "csv"},
    },
    description="Load BattleFin time-series files and extract features",
    tags=["data-loading", "feature-engineering", "time-series", "battlefin"],
    version="1.0.0",
)
def load_battlefin_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    train_file_ids: List[int] = None,
    test_file_ids: List[int] = None,
) -> str:
    """
    Load all BattleFin time-series files and extract features.

    Args:
        inputs: Dict with 'data_dir' path and 'labels' path
        outputs: Dict with 'train_data', 'test_data', 'labels_processed' paths
        train_file_ids: List of training file IDs (default: 1-200)
        test_file_ids: List of test file IDs (default: 201-510)
    """
    data_dir = inputs['data_dir']
    labels_path = inputs['labels']

    train_file_ids = train_file_ids or list(range(1, 201))
    test_file_ids = test_file_ids or list(range(201, 511))

    # Load labels
    labels_df = pd.read_csv(labels_path)
    output_cols = [c for c in labels_df.columns if c.startswith('O')]

    # Extract features for training files
    print("Extracting features from training files...")
    train_features = []
    for file_id in train_file_ids:
        filepath = os.path.join(data_dir, f'{file_id}.csv')
        if os.path.exists(filepath):
            features = _extract_file_features(filepath)
            features['FileId'] = file_id
            train_features.append(features)

    train_df = pd.DataFrame(train_features)

    # Extract features for test files
    print("Extracting features from test files...")
    test_features = []
    for file_id in test_file_ids:
        filepath = os.path.join(data_dir, f'{file_id}.csv')
        if os.path.exists(filepath):
            features = _extract_file_features(filepath)
            features['FileId'] = file_id
            test_features.append(features)

    test_df = pd.DataFrame(test_features)

    # Save outputs
    os.makedirs(os.path.dirname(outputs['train_data']) or '.', exist_ok=True)
    train_df.to_csv(outputs['train_data'], index=False)
    test_df.to_csv(outputs['test_data'], index=False)
    labels_df.to_csv(outputs['labels_processed'], index=False)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return outputs['train_data']


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "labels": {"format": "csv", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train multi-output regressor for BattleFin",
    tags=["training", "regression", "multi-output", "battlefin"],
    version="1.0.0",
)
def train_battlefin_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_estimators: int = 50,
    max_depth: int = 10,
    random_state: int = 42,
) -> str:
    """
    Train a multi-output regressor for BattleFin challenge.
    Uses RandomForest with MultiOutputRegressor wrapper.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    train_df = pd.read_csv(inputs['train_data'])
    labels_df = pd.read_csv(inputs['labels'])

    # Merge train features with labels
    merged = train_df.merge(labels_df, on='FileId', how='inner')

    # Separate features and targets
    output_cols = [c for c in labels_df.columns if c.startswith('O')]
    feature_cols = [c for c in merged.columns if c not in output_cols and c != 'FileId']

    X = merged[feature_cols].fillna(0)
    y = merged[output_cols].fillna(0)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}...")
    print(f"Features: {len(feature_cols)}, Targets: {len(output_cols)}")

    # Train model
    base_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    metrics = {
        'model_type': 'MultiOutputRegressor(RandomForest)',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'n_features': len(feature_cols),
        'n_targets': len(output_cols),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'val_rmse': float(rmse),
        'feature_columns': feature_cols,
        'output_columns': output_cols,
    }

    print(f"Validation RMSE: {rmse:.6f}")

    # Save model and metrics
    os.makedirs(os.path.dirname(outputs['model']) or '.', exist_ok=True)
    with open(outputs['model'], 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols, 'output_cols': output_cols}, f)

    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    return outputs['model']


@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_data": {"format": "csv", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Generate predictions for BattleFin test files",
    tags=["prediction", "submission", "battlefin"],
    version="1.0.0",
)
def predict_battlefin(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Generate predictions for BattleFin test files and create submission.
    """
    # Load model
    with open(inputs['model'], 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_cols = model_data['feature_cols']
    output_cols = model_data['output_cols']

    # Load test data
    test_df = pd.read_csv(inputs['test_data'])
    sample_sub = pd.read_csv(inputs['sample_submission'])

    # Ensure features align
    X_test = test_df[feature_cols].fillna(0)

    # Predict
    predictions = model.predict(X_test)

    # Create submission
    submission = pd.DataFrame(predictions, columns=output_cols)
    submission.insert(0, 'FileId', test_df['FileId'].astype(int))

    # Ensure submission matches sample format
    submission = submission[sample_sub.columns]

    # Save submission
    os.makedirs(os.path.dirname(outputs['submission']) or '.', exist_ok=True)
    submission.to_csv(outputs['submission'], index=False)

    print(f"Submission created: {submission.shape}")
    return outputs['submission']


# =============================================================================
# SERVICE 4: IMPROVED ML-BASED TRAINING (LightGBM - ALTERNATIVE APPROACH)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "labels": {"format": "csv", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train LightGBM multi-output regressor for BattleFin (improved)",
    tags=["training", "regression", "multi-output", "lightgbm", "battlefin"],
    version="2.0.0",
)
def train_battlefin_lightgbm(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    random_state: int = 42,
) -> str:
    """
    Train a LightGBM-based multi-output regressor for BattleFin challenge.

    Uses LightGBM instead of RandomForest for better performance.
    Predicts residuals from historical mean baseline.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    train_df = pd.read_csv(inputs['train_data'])
    labels_df = pd.read_csv(inputs['labels'])

    # Merge train features with labels
    merged = train_df.merge(labels_df, on='FileId', how='inner')

    # Separate features and targets
    output_cols = [c for c in labels_df.columns if c.startswith('O')]
    feature_cols = [c for c in merged.columns if c not in output_cols and c != 'FileId']

    X = merged[feature_cols].fillna(0)
    y = merged[output_cols].fillna(0)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    print(f"Training LightGBM on {len(X_train)} samples, validating on {len(X_val)}...")
    print(f"Features: {len(feature_cols)}, Targets: {len(output_cols)}")

    # Train model
    base_model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    metrics = {
        'model_type': 'MultiOutputRegressor(LightGBM)',
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'n_features': len(feature_cols),
        'n_targets': len(output_cols),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'val_rmse': float(rmse),
        'feature_columns': feature_cols,
        'output_columns': output_cols,
    }

    print(f"Validation RMSE: {rmse:.6f}")

    # Save model and metrics
    os.makedirs(os.path.dirname(outputs['model']) or '.', exist_ok=True)
    with open(outputs['model'], 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols, 'output_cols': output_cols}, f)

    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    return outputs['model']


# =============================================================================
# SERVICE 5: ENSEMBLE PREDICTION (BEST COMBINED APPROACH)
# =============================================================================

@contract(
    inputs={
        "data_dir": {"format": "directory", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Generate BattleFin predictions using ensemble of methods",
    tags=["prediction", "submission", "time-series", "battlefin", "ensemble"],
    version="1.0.0",
)
def generate_battlefin_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    weight_recent: float = 0.4,
    weight_last: float = 0.3,
    weight_ewma: float = 0.3,
    n_rows: int = 3,
    ewma_span: int = 3,
    test_file_ids: List[int] = None,
) -> str:
    """
    Generate BattleFin predictions using ensemble of multiple methods.

    Combines:
    - Recent mean (mean of last n_rows)
    - Last value (most recent observation)
    - EWMA (exponentially weighted moving average)

    Args:
        weight_recent: Weight for recent-mean predictions
        weight_last: Weight for last-value predictions
        weight_ewma: Weight for EWMA predictions
        n_rows: Number of recent rows for recent-mean
        ewma_span: Span for EWMA calculation
    """
    data_dir = inputs['data_dir']
    sample_sub = pd.read_csv(inputs['sample_submission'])

    test_file_ids = test_file_ids or list(range(201, 511))
    output_cols = [c for c in sample_sub.columns if c.startswith('O')]

    print(f"Generating ensemble predictions for {len(test_file_ids)} test files...")
    print(f"Weights: recent={weight_recent}, last={weight_last}, ewma={weight_ewma}")

    submissions = []
    for file_id in test_file_ids:
        filepath = os.path.join(data_dir, f'{file_id}.csv')
        if os.path.exists(filepath):
            recent = _compute_recent_mean(filepath, n_rows=n_rows)
            last = _compute_last_values(filepath)
            ewma = _compute_ewma_prediction(filepath, span=ewma_span)

            row = {'FileId': file_id}
            for col in output_cols:
                val = (weight_recent * recent.get(col, 0) +
                       weight_last * last.get(col, 0) +
                       weight_ewma * ewma.get(col, 0))
                row[col] = val
            submissions.append(row)

    submission_df = pd.DataFrame(submissions)
    submission_df = submission_df[sample_sub.columns]

    os.makedirs(os.path.dirname(outputs['submission']) or '.', exist_ok=True)
    submission_df.to_csv(outputs['submission'], index=False)

    metrics = {
        'method': 'ensemble',
        'weight_recent': weight_recent,
        'weight_last': weight_last,
        'weight_ewma': weight_ewma,
        'n_rows': n_rows,
        'ewma_span': ewma_span,
        'n_test_files': len(submissions),
        'n_output_cols': len(output_cols),
    }
    with open(outputs['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Submission created: {submission_df.shape}")
    return outputs['submission']


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Recommended approaches (simple, effective for financial time-series)
    'generate_battlefin_submission': generate_battlefin_submission,
    'generate_battlefin_last_value': generate_battlefin_last_value,
    'generate_battlefin_weighted': generate_battlefin_weighted,
    'generate_battlefin_ewma': generate_battlefin_ewma,
    'generate_battlefin_recent_mean': generate_battlefin_recent_mean,
    'generate_battlefin_ensemble': generate_battlefin_ensemble,  # NEW: Ensemble approach
    # ML-based approach
    'load_battlefin_data': load_battlefin_data,
    'train_battlefin_regressor': train_battlefin_regressor,
    'train_battlefin_lightgbm': train_battlefin_lightgbm,
    'predict_battlefin': predict_battlefin,
}
