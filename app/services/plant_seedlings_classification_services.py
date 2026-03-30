"""
Plant Seedlings Classification - SLEGO Services
=================================================
Competition: https://www.kaggle.com/competitions/plant-seedlings-classification
Problem Type: Multiclass Classification (12 plant species from images)
Target: species
Submission: file, species

Solution notebook insights:
- Solution 1: CNN with TF/Keras, 256x256, data augmentation, 200 epochs
- Solution 3: CNN with Keras, 64x64, data augmentation, 35 epochs
- Both use ImageDataGenerator for augmentation (rotation, flip, zoom)
- 12 classes: Black-grass, Charlock, Cleavers, Common Chickweed, Common wheat,
              Fat Hen, Loose Silky-bent, Maize, Scentless Mayweed, Shepherds Purse,
              Small-flowered Cranesbill, Sugar beet
- Train: 4750 images in class subdirectories, Test: 794 flat images

Competition-specific services:
- format_image_submission: Creates submission CSV with file+species columns from
  model predictions and test metadata
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

# Import reusable services from image_services
from services.image_services import (
    load_image_dataset,
    normalize_images,
    train_image_classifier,
)


# =============================================================================
# COMPETITION-SPECIFIC: FORMAT SUBMISSION
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_metadata": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Predict image classes and format submission with filename and species columns",
    tags=["image", "classification", "submission", "inference", "generic"],
    version="1.0.0",
)
def format_image_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_size: tuple = (64, 64),
    id_column: str = "file",
    prediction_column: str = "species",
) -> str:
    """
    Load test images, predict species using trained model, and format
    submission CSV with filename and predicted class columns.

    Works with any image classification competition that requires
    a submission with filename + predicted class.

    Parameters
    ----------
    inputs : dict
        model : str - Path to pickle file with model bundle (from train_image_classifier).
        test_metadata : str - Path to CSV with 'file_path' column from load_image_dataset.
    outputs : dict
        submission : str - Output CSV path with id_column and prediction_column.
    target_size : tuple of (int, int)
        Image size to match training size. Default: (64, 64).
    id_column : str
        Name of the ID column in output. Default: "file".
    prediction_column : str
        Name of the prediction column in output. Default: "species".
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required. Install with: pip install Pillow")

    # Load model bundle
    with open(inputs["model"], "rb") as f:
        model_bundle = pickle.load(f)

    clf = model_bundle["model"]
    le = model_bundle["label_encoder"]
    expected_shape = model_bundle.get("input_shape", None)

    # Load test metadata
    test_meta = pd.read_csv(inputs["test_metadata"])

    if "file_path" not in test_meta.columns:
        raise ValueError("test_metadata must contain 'file_path' column")

    target_w, target_h = target_size

    # Load and process test images
    images = []
    filenames = []
    errors = 0

    for idx, row in test_meta.iterrows():
        file_path = row["file_path"]
        fname = os.path.basename(file_path)

        try:
            with Image.open(file_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((target_w, target_h), Image.LANCZOS)
                arr = np.array(img, dtype=np.float64) / 255.0
                images.append(arr)
                filenames.append(fname)
        except Exception as e:
            print(f"  [WARN] Could not load {file_path}: {e}")
            errors += 1

    if not images:
        raise ValueError("No test images loaded")

    X_test = np.stack(images, axis=0)

    # Validate shape
    if expected_shape is not None and X_test.shape[1:] != expected_shape:
        raise ValueError(
            f"Test image shape {X_test.shape[1:]} != training shape {expected_shape}"
        )

    # Flatten and predict
    X_flat = X_test.reshape(len(X_test), -1)
    y_pred_encoded = clf.predict(X_flat)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    # Build submission
    submission = pd.DataFrame({
        id_column: filenames,
        prediction_column: y_pred_labels,
    })

    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    n_unique = len(set(y_pred_labels))
    return (
        f"format_image_submission: {len(filenames)} predictions, "
        f"{n_unique} unique classes, {errors} errors"
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Reused from image_services
    "load_image_dataset": load_image_dataset,
    "normalize_images": normalize_images,
    "train_image_classifier": train_image_classifier,
    # Competition-specific
    "format_image_submission": format_image_submission,
}
