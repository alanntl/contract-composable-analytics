"""
Whale Categorization Playground - Contract-Composable Analytics Services
=================================================
Competition: https://www.kaggle.com/competitions/whale-categorization-playground
Problem Type: Multiclass image classification
Target: Predict top 5 whale IDs for each test image

Based on top-scoring solution notebooks:
1. PCA + Logistic Regression (kmader) - Grayscale compression + PCA + LR
2. CNN approach (nelsongomesneto) - Conv2D layers with 150x150 RGB
3. CNN with data augmentation (vassalo) - BatchNorm + Dropout + augmentation

Key insights from solutions:
- PCA dimensionality reduction is effective for traditional ML approach
- Image compression to uniform size (100x54 or 150x150) is necessary
- Submission requires top 5 predictions space-separated

Competition-specific services:
- predict_whale_top5_submission: Generate top-5 whale ID predictions
- extract_test_pixel_features: Extract pixel features from test images
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# =============================================================================
# HELPERS
# =============================================================================

def _load_data(path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(path)


def _save_data(df: pd.DataFrame, path: str) -> None:
    """Save data to CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def _load_image(path: str, target_size: tuple = (48, 48)) -> np.ndarray:
    """Load and resize image to grayscale array."""
    try:
        from PIL import Image
        img = Image.open(path).convert('L')  # Grayscale
        img = img.resize(target_size)
        return np.array(img).flatten() / 255.0
    except Exception:
        return np.zeros(target_size[0] * target_size[1])


# =============================================================================
# SERVICE 1: EXTRACT TEST PIXEL FEATURES
# =============================================================================

@contract(
    inputs={
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Extract pixel features from test whale images",
    tags=["feature-engineering", "image", "whale", "test"],
    version="1.0.0",
)
def extract_test_pixel_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_column: str = "Image",
    image_dir: str = "",
    target_size: int = 48,
    prefix: str = "px_",
) -> str:
    """
    Extract pixel features from test whale images.

    Reads the sample_submission.csv to get test image names, then loads
    and processes each test image to extract flattened grayscale pixel features.

    Parameters:
        image_column: Column containing image filenames
        image_dir: Directory containing test images
        target_size: Resize images to target_size x target_size
        prefix: Prefix for pixel feature columns
    """
    df = _load_data(inputs["sample_submission"])

    # Determine image directory
    data_dir = os.path.dirname(inputs["sample_submission"])
    storage_dir = os.path.dirname(data_dir)  # competition directory

    if not image_dir:
        possible_dirs = [
            os.path.join(storage_dir, 'test'),
            os.path.join(data_dir, '..', 'test'),
            os.path.join(data_dir, 'test'),
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                image_dir = d
                break
    else:
        if not os.path.isabs(image_dir):
            possible_dirs = [
                os.path.join(storage_dir, image_dir),  # e.g., storage/comp/test
                os.path.join(data_dir, '..', image_dir),  # e.g., storage/comp/test
                os.path.join(data_dir, image_dir),  # e.g., datasets/test
                image_dir  # relative to cwd
            ]
            for d in possible_dirs:
                if os.path.exists(d):
                    image_dir = d
                    break

    n_pixels = target_size * target_size
    pixel_features = np.zeros((len(df), n_pixels))

    loaded = 0
    for i, img_name in enumerate(df[image_column]):
        img_path = os.path.join(image_dir, str(img_name))
        if os.path.exists(img_path):
            pixel_features[i] = _load_image(img_path, (target_size, target_size))
            loaded += 1

    # Create pixel feature columns
    pixel_cols = [f"{prefix}{j}" for j in range(n_pixels)]
    pixel_df = pd.DataFrame(pixel_features, columns=pixel_cols, index=df.index)

    # Keep only Image column and pixel features
    df_out = pd.DataFrame({image_column: df[image_column]})
    df_out = pd.concat([df_out, pixel_df], axis=1)

    _save_data(df_out, outputs["data"])

    return f"extract_test_pixel_features: loaded {loaded}/{len(df)} images, {n_pixels} features"


# =============================================================================
# SERVICE 2: PREDICT WHALE TOP-5 SUBMISSION
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_data": {"format": "csv", "required": True},
        "label_mapping": {"format": "json", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Generate top-5 whale ID predictions for Kaggle submission",
    tags=["prediction", "submission", "whale", "multiclass"],
    version="1.0.0",
)
def predict_whale_top5_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Image",
    top_k: int = 5,
    default_whale: str = "new_whale",
) -> str:
    """
    Generate top-5 whale ID predictions and create Kaggle submission.

    Loads the trained model and label mapping, predicts class probabilities
    for each test image, selects top-5 predictions, and maps integer labels
    back to original whale IDs.

    Submission format: Image,Id where Id is space-separated top 5 whale IDs.

    Parameters:
        id_column: Column containing image filenames
        top_k: Number of top predictions to include (default 5)
        default_whale: Default whale ID for padding if fewer than top_k classes
    """
    # Load model
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    # Load label mapping (int -> whale_id)
    with open(inputs["label_mapping"], "r") as f:
        int_to_label = json.load(f)
    # Convert string keys to int
    int_to_label = {int(k): v for k, v in int_to_label.items()}

    # Load test data
    test_df = _load_data(inputs["test_data"])

    # Extract model and feature columns
    model = artifact["model"]
    feature_cols = artifact.get("feature_cols", None)

    # Prepare feature matrix
    if feature_cols:
        for col in feature_cols:
            if col not in test_df.columns:
                test_df[col] = 0
        X = test_df[feature_cols]
    else:
        drop_cols = [id_column] if id_column in test_df.columns else []
        X = test_df.drop(columns=drop_cols, errors="ignore")

    # Predict probabilities
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
    else:
        # Fallback for models without predict_proba
        preds = model.predict(X)
        n_classes = len(int_to_label)
        proba = np.zeros((len(X), n_classes))
        for i, p in enumerate(preds):
            proba[i, int(p)] = 1.0

    # Get top-k predictions for each sample
    n_classes = proba.shape[1]
    results = []

    for i in range(len(proba)):
        # Get top-k class indices
        top_indices = np.argsort(proba[i])[::-1][:top_k]

        # Map to whale IDs
        top_whale_ids = []
        for idx in top_indices:
            if idx in int_to_label:
                top_whale_ids.append(int_to_label[idx])
            else:
                top_whale_ids.append(default_whale)

        # Pad with default if necessary
        while len(top_whale_ids) < top_k:
            top_whale_ids.append(default_whale)

        results.append(" ".join(top_whale_ids))

    # Create submission DataFrame
    submission = pd.DataFrame({
        id_column: test_df[id_column],
        "Id": results
    })

    _save_data(submission, outputs["submission"])

    return f"predict_whale_top5_submission: {len(submission)} predictions with top-{top_k} format"


# =============================================================================
# SERVICE 3: EXTRACT IMAGE STATISTICS FOR WHALE IMAGES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Extract statistical features from whale images",
    tags=["feature-engineering", "image", "statistics", "whale"],
    version="1.0.0",
)
def extract_whale_image_statistics(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_column: str = "Image",
    image_dir: str = "",
    prefix: str = "img_",
) -> str:
    """
    Extract statistical features from whale images.

    Based on solution notebook insights: basic image statistics
    (mean, std, contrast) can help distinguish whale patterns.

    Features: mean brightness, std, min, max, range, contrast proxy.
    """
    df = _load_data(inputs["data"])

    # Determine image directory
    if not image_dir:
        data_dir = os.path.dirname(inputs["data"])
        storage_dir = os.path.dirname(os.path.dirname(data_dir))
        comp_name = "whale-categorization-playground"
        possible_dirs = [
            os.path.join(storage_dir, comp_name, 'train'),
            os.path.join(data_dir, '..', 'train'),
            os.path.join(data_dir, 'train'),
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                image_dir = d
                break
    else:
        if not os.path.isabs(image_dir):
            data_dir = os.path.dirname(inputs["data"])
            storage_dir = os.path.dirname(os.path.dirname(data_dir))
            possible_dirs = [
                os.path.join(storage_dir, image_dir),
                os.path.join(data_dir, image_dir),
                image_dir
            ]
            for d in possible_dirs:
                if os.path.exists(d):
                    image_dir = d
                    break

    stats = {
        f'{prefix}mean': [],
        f'{prefix}std': [],
        f'{prefix}min': [],
        f'{prefix}max': [],
        f'{prefix}range': [],
        f'{prefix}contrast': [],
    }

    for img_name in df[image_column]:
        img_path = os.path.join(image_dir, str(img_name))
        try:
            from PIL import Image
            img = Image.open(img_path).convert('L')
            arr = np.array(img) / 255.0

            stats[f'{prefix}mean'].append(float(arr.mean()))
            stats[f'{prefix}std'].append(float(arr.std()))
            stats[f'{prefix}min'].append(float(arr.min()))
            stats[f'{prefix}max'].append(float(arr.max()))
            stats[f'{prefix}range'].append(float(arr.max() - arr.min()))
            # Simple contrast measure: std / mean
            stats[f'{prefix}contrast'].append(float(arr.std() / (arr.mean() + 1e-8)))
        except Exception:
            for key in stats:
                stats[key].append(0.0)

    for key, values in stats.items():
        df[key] = values

    _save_data(df, outputs["data"])

    return f"extract_whale_image_statistics: extracted 6 features for {len(df)} images"


# =============================================================================
# SERVICE 4: FORMAT TOP-5 PREDICTIONS FROM PRETRAINED CLASSIFIER
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Format probability predictions from pretrained classifier to top-5 whale IDs",
    tags=["prediction", "submission", "whale", "multiclass", "top-k", "generic"],
    version="1.0.0",
)
def format_top5_whale_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Image",
    top_k: int = 5,
    default_whale: str = "new_whale",
) -> str:
    """
    Format probability predictions from pretrained classifier to top-5 whale submission.

    Takes the probability CSV output from predict_pretrained_image_classifier
    and converts it to the whale competition submission format with top-5
    space-separated whale IDs.

    Parameters:
        id_column: Column containing image filenames
        top_k: Number of top predictions to include (default 5)
        default_whale: Default whale ID for padding if fewer than top_k classes
    """
    # Load predictions (has id_column + probability columns for each class)
    pred_df = _load_data(inputs["predictions"])
    sample_df = _load_data(inputs["sample_submission"])

    # Get class columns (all columns except id_column)
    class_cols = [col for col in pred_df.columns if col != id_column]

    # Get image IDs from predictions
    if id_column in pred_df.columns:
        image_ids = pred_df[id_column].tolist()
    else:
        # Fall back to sample submission IDs
        image_ids = sample_df[id_column].tolist()

    # Extract probability matrix
    proba = pred_df[class_cols].values

    # Get top-k predictions for each sample
    results = []
    for i in range(len(proba)):
        # Get top-k class indices
        top_indices = np.argsort(proba[i])[::-1][:top_k]

        # Map to whale IDs (class columns are the whale IDs)
        top_whale_ids = [class_cols[idx] for idx in top_indices if idx < len(class_cols)]

        # Pad with default if necessary
        while len(top_whale_ids) < top_k:
            top_whale_ids.append(default_whale)

        results.append(" ".join(top_whale_ids[:top_k]))

    # Create submission DataFrame matching sample format
    submission = pd.DataFrame({
        id_column: image_ids[:len(results)],
        "Id": results
    })

    _save_data(submission, outputs["submission"])

    return f"format_top5_whale_submission: {len(submission)} predictions with top-{top_k} format"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "extract_test_pixel_features": extract_test_pixel_features,
    "predict_whale_top5_submission": predict_whale_top5_submission,
    "extract_whale_image_statistics": extract_whale_image_statistics,
    "format_top5_whale_submission": format_top5_whale_submission,
}
