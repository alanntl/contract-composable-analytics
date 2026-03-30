"""
Digit Recognizer (MNIST) - SLEGO Services
==========================================
Competition: https://www.kaggle.com/competitions/digit-recognizer
Problem Type: Multiclass Classification (10 classes: 0-9)
Target: label (digit 0-9)
ID Column: None (generated ImageId starting from 1)

Image classification with 28x28 grayscale images as 784 pixel values.
No train ID column - submission uses generated ImageId starting at 1.

Key techniques from solution notebooks:
- Pixel normalization (divide by 255)
- Simple models work well: Random Forest, Ridge Regression, LightGBM
- CNN approaches achieve 99%+ accuracy
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

try:
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.preprocessing_services import split_data
except ImportError:
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import split_data


# =============================================================================
# GENERIC IMAGE PREPROCESSING SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Normalize pixel values to 0-1 range by dividing by scale factor",
    tags=["preprocessing", "image", "normalization", "generic"],
    version="1.0.0"
)
def normalize_pixels(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    pixel_columns: Optional[List[str]] = None,
    scale: float = 255.0,
    exclude_columns: List[str] = None,
) -> str:
    """
    Normalize pixel values by dividing by a scale factor.

    Works with any image-as-pixels dataset (MNIST, CIFAR flattened, etc.).
    By default, normalizes all numeric columns except those in exclude_columns.

    Args:
        pixel_columns: Specific columns to normalize (if None, all numeric except excluded)
        scale: Divide pixel values by this (255 for 8-bit images)
        exclude_columns: Columns to skip (e.g., ['label', 'id'])
    """
    df = pd.read_csv(inputs["data"])
    exclude_columns = exclude_columns or ['label', 'Label', 'id', 'Id', 'ImageId']

    if pixel_columns is None:
        pixel_columns = [c for c in df.select_dtypes(include=[np.number]).columns
                         if c not in exclude_columns]

    df[pixel_columns] = df[pixel_columns] / scale

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"normalize_pixels: normalized {len(pixel_columns)} columns by {scale}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Add pixel statistics features (mean, std, nonzero count)",
    tags=["preprocessing", "feature-engineering", "image", "generic"],
    version="1.0.0"
)
def add_pixel_statistics(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    pixel_prefix: str = "pixel",
    exclude_columns: List[str] = None,
) -> str:
    """
    Add aggregate statistics from pixel columns.

    Creates features: pixel_mean, pixel_std, pixel_nonzero_count, pixel_nonzero_ratio.
    Useful for simple models on image data.

    Args:
        pixel_prefix: Prefix of pixel columns (e.g., 'pixel' for pixel0, pixel1, ...)
        exclude_columns: Additional columns to exclude
    """
    df = pd.read_csv(inputs["data"])
    exclude_columns = exclude_columns or ['label', 'Label', 'id', 'Id', 'ImageId']

    pixel_cols = [c for c in df.columns
                  if c.startswith(pixel_prefix) and c not in exclude_columns]

    if pixel_cols:
        pixel_data = df[pixel_cols]
        df['pixel_mean'] = pixel_data.mean(axis=1)
        df['pixel_std'] = pixel_data.std(axis=1)
        df['pixel_nonzero'] = (pixel_data > 0).sum(axis=1)
        df['pixel_nonzero_ratio'] = df['pixel_nonzero'] / len(pixel_cols)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"add_pixel_statistics: added 4 aggregate features from {len(pixel_cols)} pixel columns"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "data": {"format": "csv", "required": True}
    },
    outputs={"predictions": {"format": "csv"}},
    description="Generate predictions with auto-generated sequential IDs",
    tags=["inference", "prediction", "generic"],
    version="1.0.0"
)
def predict_with_generated_id(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "ImageId",
    prediction_column: str = "Label",
    id_start: int = 1,
    exclude_columns: List[str] = None,
) -> str:
    """
    Generate predictions with auto-generated sequential IDs.

    For datasets without ID columns in test data (like MNIST digit-recognizer).

    Args:
        id_column: Name for the ID column in output
        prediction_column: Name for the prediction column
        id_start: Starting value for generated IDs
        exclude_columns: Columns to exclude from features
    """
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    # Handle dict artifacts from train_lightgbm_classifier
    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_cols = artifact.get("feature_cols")
    else:
        model = artifact
        feature_cols = None

    df = pd.read_csv(inputs["data"])
    exclude_columns = exclude_columns or []

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude_columns]
    X = df[feature_cols]

    preds = model.predict(X)

    result = pd.DataFrame({
        id_column: range(id_start, id_start + len(preds)),
        prediction_column: preds.astype(int)
    })

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    result.to_csv(outputs["predictions"], index=False)
    return f"predict_with_generated_id: {len(preds)} predictions, IDs {id_start}-{id_start + len(preds) - 1}"


# =============================================================================
# ADDITIONAL FEATURE ENGINEERING SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Remove columns with zero or near-zero variance (e.g., border pixels)",
    tags=["preprocessing", "feature-selection", "image", "generic"],
    version="1.0.0"
)
def remove_zero_variance_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    threshold: float = 0.0,
    exclude_columns: List[str] = None,
) -> str:
    """
    Remove columns with variance at or below threshold.

    For image data, many border pixels are always 0. Removing them
    reduces noise and speeds up training.
    """
    df = pd.read_csv(inputs["data"])
    exclude_columns = exclude_columns or ['label', 'Label', 'id', 'Id', 'ImageId']

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_columns]
    variances = df[numeric_cols].var()
    low_var = variances[variances <= threshold].index.tolist()
    df = df.drop(columns=low_var)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"remove_zero_variance_columns: removed {len(low_var)} cols, {len(df.columns)} remaining"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Binarize pixel columns (value > threshold -> 1, else 0)",
    tags=["preprocessing", "feature-engineering", "image", "generic"],
    version="1.0.0"
)
def binarize_pixels(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    threshold: float = 0.0,
    pixel_prefix: str = "pixel",
    exclude_columns: List[str] = None,
) -> str:
    """
    Create binary versions of pixel columns.

    Inspired by KNN/RF solution: converting pixels to 0/1 removes
    intensity noise and focuses on shape structure.
    """
    df = pd.read_csv(inputs["data"])
    exclude_columns = exclude_columns or ['label', 'Label', 'id', 'Id', 'ImageId']

    pixel_cols = [c for c in df.columns
                  if c.startswith(pixel_prefix) and c not in exclude_columns]

    for col in pixel_cols:
        df[f"bin_{col}"] = (df[col] > threshold).astype(int)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"binarize_pixels: created {len(pixel_cols)} binary features"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "normalize_pixels": normalize_pixels,
    "add_pixel_statistics": add_pixel_statistics,
    "predict_with_generated_id": predict_with_generated_id,
    "remove_zero_variance_columns": remove_zero_variance_columns,
    "binarize_pixels": binarize_pixels,
    "split_data": split_data,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
}


# =============================================================================
# PIPELINE SPEC
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "normalize_pixels",
        "inputs": {"data": "digit-recognizer/datasets/train.csv"},
        "outputs": {"data": "digit-recognizer/artifacts/train_01_normalized.csv"},
        "params": {"scale": 255.0, "exclude_columns": ["label"]},
        "module": "digit_recognizer_services"
    },
    {
        "service": "binarize_pixels",
        "inputs": {"data": "digit-recognizer/artifacts/train_01_normalized.csv"},
        "outputs": {"data": "digit-recognizer/artifacts/train_02_binarized.csv"},
        "params": {"threshold": 0.0, "exclude_columns": ["label"]},
        "module": "digit_recognizer_services"
    },
    {
        "service": "add_pixel_statistics",
        "inputs": {"data": "digit-recognizer/artifacts/train_02_binarized.csv"},
        "outputs": {"data": "digit-recognizer/artifacts/train_03_stats.csv"},
        "params": {"exclude_columns": ["label"]},
        "module": "digit_recognizer_services"
    },
    {
        "service": "remove_zero_variance_columns",
        "inputs": {"data": "digit-recognizer/artifacts/train_03_stats.csv"},
        "outputs": {"data": "digit-recognizer/artifacts/train_04_filtered.csv"},
        "params": {"threshold": 0.0, "exclude_columns": ["label"]},
        "module": "digit_recognizer_services"
    },
    {
        "service": "split_data",
        "inputs": {"data": "digit-recognizer/artifacts/train_04_filtered.csv"},
        "outputs": {
            "train_data": "digit-recognizer/artifacts/train_split.csv",
            "valid_data": "digit-recognizer/artifacts/valid_split.csv"
        },
        "params": {"stratify_column": "label", "test_size": 0.1, "random_state": 42},
        "module": "digit_recognizer_services"
    },
    {
        "service": "train_lightgbm_classifier",
        "inputs": {
            "train_data": "digit-recognizer/artifacts/train_split.csv",
            "valid_data": "digit-recognizer/artifacts/valid_split.csv"
        },
        "outputs": {
            "model": "digit-recognizer/artifacts/model.pkl",
            "metrics": "digit-recognizer/artifacts/metrics.json"
        },
        "params": {
            "label_column": "label",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 127,
            "max_depth": -1
        },
        "module": "digit_recognizer_services"
    },
    # Test data processing
    {
        "service": "normalize_pixels",
        "inputs": {"data": "digit-recognizer/datasets/test.csv"},
        "outputs": {"data": "digit-recognizer/artifacts/test_01_normalized.csv"},
        "params": {"scale": 255.0},
        "module": "digit_recognizer_services"
    },
    {
        "service": "binarize_pixels",
        "inputs": {"data": "digit-recognizer/artifacts/test_01_normalized.csv"},
        "outputs": {"data": "digit-recognizer/artifacts/test_02_binarized.csv"},
        "params": {"threshold": 0.0},
        "module": "digit_recognizer_services"
    },
    {
        "service": "add_pixel_statistics",
        "inputs": {"data": "digit-recognizer/artifacts/test_02_binarized.csv"},
        "outputs": {"data": "digit-recognizer/artifacts/test_03_stats.csv"},
        "module": "digit_recognizer_services"
    },
    {
        "service": "predict_with_generated_id",
        "inputs": {
            "model": "digit-recognizer/artifacts/model.pkl",
            "data": "digit-recognizer/artifacts/test_03_stats.csv"
        },
        "outputs": {"predictions": "digit-recognizer/artifacts/predictions.csv"},
        "params": {"id_column": "ImageId", "prediction_column": "Label", "id_start": 1},
        "module": "digit_recognizer_services"
    }
]


def run_pipeline(base_path: str, verbose: bool = True):
    """Run the pipeline spec end-to-end."""
    for i, step in enumerate(PIPELINE_SPEC, 1):
        service_name = step["service"]
        service_fn = SERVICE_REGISTRY.get(service_name)
        if not service_fn:
            print(f"Error: Service {service_name} not found")
            continue

        res_in = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
        res_out = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

        if verbose:
            print(f"[{i}/{len(PIPELINE_SPEC)}] {service_name}...", end=" ")

        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose: print(f"OK - {result}")
        except Exception as e:
            if verbose: print(f"FAILED - {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="../storage")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    storage_path = os.path.join(os.path.dirname(script_dir), "storage")
    if not os.path.exists(storage_path):
        storage_path = os.path.abspath(args.base_path)

    print(f"\n--- Digit Recognizer Pipeline (Base: {storage_path}) ---")
    run_pipeline(storage_path)
