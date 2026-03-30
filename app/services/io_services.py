"""
SLEGO I/O Services - Consolidated Data Loading and Saving
==========================================================
This module provides unified I/O operations for all SLEGO pipelines.
All data loading and saving should be imported from this module.

Usage:
    from services.io_services import load_data, save_data, load_artifact, save_artifact
"""
import os
import json
import pickle
from typing import Any, Dict, Optional
import pandas as pd
from functools import wraps


def contract(inputs=None, outputs=None, params=None, description=None, tags=None, version="1.0.0"):
    """Service contract decorator for SLEGO services."""
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


# =============================================================================
# Core Data Loading Functions
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Auto-detect and load data from CSV, Parquet, or JSON files.

    Args:
        path: Path to the data file

    Returns:
        pandas DataFrame

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext == ".json":
        return pd.read_json(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported data format: {ext}")


def save_data(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Save DataFrame with auto-detected format based on extension.

    Args:
        df: DataFrame to save
        path: Output file path
        index: Whether to include index (default False)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df.to_csv(path, index=index)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=index)
    elif ext == ".json":
        df.to_json(path, orient='records')
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=index)
    else:
        # Default to CSV
        df.to_csv(path, index=index)


def load_artifact(path: str) -> Any:
    """
    Load artifact from pickle or JSON file.

    Args:
        path: Path to artifact file

    Returns:
        Loaded artifact (model, scaler, encoder, etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pkl":
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif ext == ".json":
        with open(path, 'r') as f:
            return json.load(f)
    elif ext == ".joblib":
        import joblib
        return joblib.load(path)
    else:
        # Default to pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


def save_artifact(obj: Any, path: str) -> None:
    """
    Save artifact with auto-detected format based on extension.

    Args:
        obj: Object to save (model, scaler, encoder, dict, etc.)
        path: Output file path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pkl":
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    elif ext == ".json":
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=str)
    elif ext == ".joblib":
        import joblib
        joblib.dump(obj, path)
    else:
        # Default to pickle
        with open(path, 'wb') as f:
            pickle.dump(obj, f)


# =============================================================================
# SLEGO Service Wrappers (for pipeline integration)
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Load data from CSV file",
    tags=["io", "data-loading", "generic"]
)
def load_csv_data(data: str, output: str) -> Dict[str, str]:
    """
    Load CSV data and save to output path.

    This is a pass-through service useful for explicit data loading in pipelines.
    """
    df = load_data(data)
    save_data(df, output)
    return {'data': output}


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True}
    },
    outputs={"data": {"format": "csv"}},
    params={
        "train_marker_column": "Column name for train/test marker",
        "train_value": "Value indicating train data"
    },
    description="Combine train and test data with marker column",
    tags=["io", "data-combination", "generic"]
)
def combine_train_test(
    train_data: str,
    test_data: str,
    output: str,
    train_marker_column: str = 'is_train',
    train_value: int = 1
) -> Dict[str, str]:
    """
    Combine train and test DataFrames with a marker column.

    Useful for applying the same preprocessing to both datasets.

    Args:
        train_data: Path to training data
        test_data: Path to test data
        output: Path for combined output
        train_marker_column: Name of column to mark train vs test
        train_value: Value for train rows (test gets 0)
    """
    train_df = load_data(train_data)
    test_df = load_data(test_data)

    train_df[train_marker_column] = train_value
    test_df[train_marker_column] = 0

    combined = pd.concat([train_df, test_df], ignore_index=True)
    save_data(combined, output)

    return {'data': output}


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"}
    },
    params={
        "marker_column": "Column indicating train/test split",
        "train_value": "Value indicating training data"
    },
    description="Split combined data back into train and test",
    tags=["io", "data-splitting", "generic"]
)
def split_train_test(
    data: str,
    train_output: str,
    test_output: str,
    marker_column: str = 'is_train',
    train_value: int = 1
) -> Dict[str, str]:
    """
    Split combined DataFrame back into train and test sets.

    Args:
        data: Path to combined data
        train_output: Path for train data output
        test_output: Path for test data output
        marker_column: Column indicating train vs test
        train_value: Value indicating training data
    """
    df = load_data(data)

    train_df = df[df[marker_column] == train_value].drop(columns=[marker_column])
    test_df = df[df[marker_column] != train_value].drop(columns=[marker_column])

    save_data(train_df, train_output)
    save_data(test_df, test_output)

    return {'train_data': train_output, 'test_data': test_output}


@contract(
    inputs={
        "predictions": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": False}
    },
    outputs={"submission": {"format": "csv"}},
    params={
        "id_column": "Name of ID column in output",
        "target_column": "Name of target/prediction column",
        "prediction_column": "Name of prediction column in input"
    },
    description="Create Kaggle submission file",
    tags=["io", "submission", "kaggle", "generic"]
)
def create_submission(
    predictions: str,
    output: str,
    id_column: str = 'Id',
    target_column: str = 'target',
    prediction_column: str = 'prediction',
    test_data: str = None,
    id_start: int = None
) -> Dict[str, str]:
    """
    Create a Kaggle-format submission file.

    Args:
        predictions: Path to predictions CSV
        output: Path for submission output
        id_column: Name of ID column in output
        target_column: Name of target column in output
        prediction_column: Name of prediction column in input
        test_data: Optional path to test data (for ID extraction)
        id_start: Starting ID if generating IDs (e.g., for image competitions)
    """
    pred_df = load_data(predictions)

    # Get IDs
    if id_column in pred_df.columns:
        ids = pred_df[id_column]
    elif test_data and os.path.exists(test_data):
        test_df = load_data(test_data)
        if id_column in test_df.columns:
            ids = test_df[id_column]
        else:
            ids = range(id_start or 0, (id_start or 0) + len(pred_df))
    elif id_start is not None:
        ids = range(id_start, id_start + len(pred_df))
    else:
        ids = range(len(pred_df))

    # Get predictions
    if prediction_column in pred_df.columns:
        preds = pred_df[prediction_column]
    elif 'prediction' in pred_df.columns:
        preds = pred_df['prediction']
    else:
        # Assume last column is predictions
        preds = pred_df.iloc[:, -1]

    submission = pd.DataFrame({
        id_column: ids,
        target_column: preds
    })

    save_data(submission, output)
    return {'submission': output}


# =============================================================================
# SLEGO-Compatible Services (using inputs/outputs dict pattern)
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Create Kaggle submission file from predictions (SLEGO interface)",
    tags=["io", "submission", "kaggle", "generic"],
    version="1.0.0",
)
def format_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = 'Id',
    target_column: str = 'Predicted',
    prediction_column: str = 'prediction',
    source_id_column: str = 'id',
) -> str:
    """
    Create a Kaggle-format submission file using SLEGO interface.

    Takes predictions CSV with cluster/prediction column and formats
    it for Kaggle submission with proper column names.

    Parameters:
        id_column: Name of ID column in output (e.g., 'Id')
        target_column: Name of target column in output (e.g., 'Predicted', 'target')
        prediction_column: Name of prediction column in input (e.g., 'cluster', 'prediction')
        source_id_column: Name of ID column in input (e.g., 'id')
    """
    pred_df = load_data(inputs["predictions"])

    # Get IDs
    if source_id_column in pred_df.columns:
        ids = pred_df[source_id_column]
    elif id_column in pred_df.columns:
        ids = pred_df[id_column]
    else:
        ids = range(len(pred_df))

    # Get predictions
    if prediction_column in pred_df.columns:
        preds = pred_df[prediction_column]
    elif 'prediction' in pred_df.columns:
        preds = pred_df['prediction']
    elif 'cluster' in pred_df.columns:
        preds = pred_df['cluster']
    else:
        # Assume last column is predictions
        preds = pred_df.iloc[:, -1]

    submission = pd.DataFrame({
        id_column: ids,
        target_column: preds
    })

    save_data(submission, outputs["submission"])
    return f"format_submission: Created submission with {len(submission)} rows, columns {list(submission.columns)}"


# =============================================================================
# Service Registry
# =============================================================================

SERVICE_REGISTRY = {
    "combine_train_test": combine_train_test,
    "split_train_test": split_train_test,
    "create_submission": create_submission,
    "format_submission": format_submission,
}
