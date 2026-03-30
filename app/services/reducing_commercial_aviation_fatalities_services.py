"""
Reducing Commercial Aviation Fatalities - Contract-Composable Analytics Services
=========================================================
Competition: https://www.kaggle.com/competitions/reducing-commercial-aviation-fatalities
Problem Type: Multiclass Classification (4 classes)
Target: event (A=baseline, B=startle/surprise, C=channelized attention, D=diverted attention)
Submission: id, A, B, C, D (class probabilities)
Evaluation: Multi-class log loss

Competition-specific services derived from top Kaggle solution notebooks:
- preprocess_aviation_data: Encode experiment column consistently across train/test,
  map categorical target to integers, downcast floats to save memory on the large
  (multi-GB) dataset, fill missing values.

Key insights from top-3 solution notebooks:
1. LightGBM multiclass with early stopping is the dominant approach
2. Experiment column (CA/DA/SS/LOFT) is a useful categorical feature when encoded consistently
3. EEG, ECG, GSR, and respiration signals are the core predictive features
4. Memory optimisation (float32 downcasting) is essential for the ~10M-row dataset
5. Event target must be explicitly mapped A=0, B=1, C=2, D=3 for consistent submission columns
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract


# ---------------------------------------------------------------------------
# Helper: load / save
# ---------------------------------------------------------------------------

def _load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {ext}")


def _save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)


# ===========================================================================
# PUBLIC SERVICE: preprocess_aviation_data
# ===========================================================================

@contract(
    inputs={
        "train": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_processed": {"format": "csv", "schema": {"type": "tabular"}},
        "test_processed": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Preprocess aviation crew-state data: encode target and experiment, fill NaN, downcast dtypes",
    tags=["preprocessing", "multiclass", "eeg", "physiological", "aviation"],
    version="1.0.0",
)
def preprocess_aviation_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "event",
    id_column: str = "id",
    target_mapping: Optional[Dict[str, int]] = None,
    fill_value: float = 0.0,
) -> str:
    """
    Load and preprocess train + test data for the aviation fatalities competition.

    Processes both datasets in a single pass to ensure consistent encoding of
    the experiment column. Produces processed train/test CSVs ready for modelling.

    Steps:
      1. Read train and test CSVs
      2. Encode experiment column consistently (sorted alphabetical mapping)
      3. Map target labels to integers (A=0, B=1, C=2, D=3)
      4. Fill missing values
      5. Downcast float64 to float32 to reduce memory (~50% savings)
      6. Write processed train and test CSVs

    Args:
        inputs: Must contain keys "train" and "test"
        outputs: Must contain keys "train_processed" and "test_processed"
        target_column: Name of the target column (default "event")
        id_column: Name of the ID column in test (default "id")
        target_mapping: Dict mapping target labels to integers
                        (default: {"A": 0, "B": 1, "C": 2, "D": 3})
        fill_value: Value for filling NaN in numeric columns (default 0.0)
    """
    if target_mapping is None:
        target_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}

    # --- Read data (low_memory=False to avoid mixed-type warnings) ---
    train = pd.read_csv(inputs["train"], low_memory=False)
    test = pd.read_csv(inputs["test"], low_memory=False)
    n_train, n_test = len(train), len(test)

    # --- Coerce any mixed-type columns to numeric ---
    skip_cols = {target_column, id_column, "experiment"}
    for df in (train, test):
        for col in df.columns:
            if col not in skip_cols and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Encode experiment consistently across train + test ---
    if "experiment" in train.columns:
        all_experiments = sorted(
            set(train["experiment"].dropna().unique())
            | set(test["experiment"].dropna().unique())
        )
        exp_mapping = {v: i for i, v in enumerate(all_experiments)}
        train["experiment"] = train["experiment"].map(exp_mapping)
        test["experiment"] = test["experiment"].map(exp_mapping)

    # --- Map target column in train ---
    if target_column in train.columns:
        train[target_column] = train[target_column].map(target_mapping)

    # --- Fill missing values ---
    for df in (train, test):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)

    # --- Downcast float64 → float32 to save memory ---
    for df in (train, test):
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype(np.float32)

    # --- Save ---
    _save_data(train, outputs["train_processed"])
    _save_data(test, outputs["test_processed"])

    n_features = len([c for c in train.columns if c != target_column])
    return (
        f"preprocess_aviation_data: train={n_train} rows, "
        f"test={n_test} rows, features={n_features}"
    )


# ===========================================================================
# SERVICE REGISTRY
# ===========================================================================

SERVICE_REGISTRY = {
    "preprocess_aviation_data": preprocess_aviation_data,
}