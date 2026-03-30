"""
Histopathologic Cancer Detection - Contract-Composable Analytics Competition Module
============================================================
Competition: https://www.kaggle.com/competitions/histopathologic-cancer-detection
Problem Type: Binary Classification (image-based)
Target: label (0=non-cancerous, 1=cancerous)
ID Column: id (hash string)
Metric: AUC

96x96 pixel TIF images of lymph node sections.

This module defines only the PIPELINE_SPEC and SERVICE_REGISTRY.
All services are imported from generic reusable modules:
  - image_services: load_labeled_images, train_cnn_image_classifier,
                    predict_cnn_image_classifier, prepare_test_images,
                    format_submission
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.image_services import (
    load_labeled_images,
    prepare_test_images,
    train_cnn_image_classifier,
    predict_cnn_image_classifier,
    format_submission,
)


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "load_labeled_images",
        "inputs": {
            "labels_csv": "histopathologic-cancer-detection/datasets/train_labels.csv",
        },
        "outputs": {
            "X": "histopathologic-cancer-detection/artifacts/X_train.pkl",
            "y": "histopathologic-cancer-detection/artifacts/y_train.pkl",
            "metadata": "histopathologic-cancer-detection/artifacts/train_metadata.json",
        },
        "params": {
            "image_dir": "train",
            "id_column": "id",
            "label_column": "label",
            "image_extension": ".tif",
            "target_size": 64,
        },
        "module": "image_services",
    },
    {
        "service": "prepare_test_images",
        "inputs": {
            "test_csv": "histopathologic-cancer-detection/datasets/sample_submission.csv",
            "image_dir": "histopathologic-cancer-detection/datasets/test",
        },
        "outputs": {
            "X_test": "histopathologic-cancer-detection/artifacts/X_test.pkl",
            "test_ids": "histopathologic-cancer-detection/artifacts/test_ids.csv",
        },
        "params": {
            "id_column": "id",
            "target_size": [64, 64],
            "image_extension": ".tif",
        },
        "module": "image_services",
    },
    {
        "service": "train_cnn_image_classifier",
        "inputs": {
            "X": "histopathologic-cancer-detection/artifacts/X_train.pkl",
            "y": "histopathologic-cancer-detection/artifacts/y_train.pkl",
        },
        "outputs": {
            "model": "histopathologic-cancer-detection/artifacts/model.pkl",
            "metrics": "histopathologic-cancer-detection/artifacts/metrics.json",
        },
        "params": {
            "n_classes": 2,
            "n_epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.0001,
            "dropout": 0.3,
            "conv_channels": [32, 64, 128],
            "fc_size": 256,
            "validation_split": 0.2,
            "random_state": 42,
        },
        "module": "image_services",
    },
    {
        "service": "predict_cnn_image_classifier",
        "inputs": {
            "model": "histopathologic-cancer-detection/artifacts/model.pkl",
            "X_test": "histopathologic-cancer-detection/artifacts/X_test.pkl",
            "test_ids": "histopathologic-cancer-detection/artifacts/test_ids.csv",
        },
        "outputs": {
            "predictions": "histopathologic-cancer-detection/artifacts/predictions.csv",
        },
        "params": {
            "id_column": "id",
            "prediction_column": "label",
            "batch_size": 256,
        },
        "module": "image_services",
    },
    {
        "service": "format_submission",
        "inputs": {
            "predictions": "histopathologic-cancer-detection/artifacts/predictions.csv",
            "sample_submission": "histopathologic-cancer-detection/datasets/sample_submission.csv",
        },
        "outputs": {
            "submission": "histopathologic-cancer-detection/submission.csv",
        },
        "params": {
            "id_column": "id",
            "prediction_column": "label",
        },
        "module": "image_services",
    },
]


# =============================================================================
# SERVICE REGISTRY (re-exports from generic modules)
# =============================================================================

SERVICE_REGISTRY = {
    "load_labeled_images": load_labeled_images,
    "prepare_test_images": prepare_test_images,
    "train_cnn_image_classifier": train_cnn_image_classifier,
    "predict_cnn_image_classifier": predict_cnn_image_classifier,
    "format_submission": format_submission,
}


def run_pipeline():
    """Run the full histopathologic cancer detection pipeline."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline_runner import PipelineRunner

    runner = PipelineRunner(
        "kb.sqlite",
        modules=["image_services"],
    )
    result = runner.run(
        PIPELINE_SPEC,
        base_path="storage",
        pipeline_name="histopathologic-cancer-detection",
    )
    return result
