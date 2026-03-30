"""
Dont Call Me Turkey - SLEGO Services
=====================================
Competition: https://www.kaggle.com/competitions/dont-call-me-turkey
Problem Type: Binary Classification
Target: is_turkey (0 or 1)
ID Column: vid_id

Audio classification: detect turkey sounds from VGGish audio embeddings.
Each sample has audio_embedding (list of up to 10 frames, each 128-dim VGGish).

Key techniques from solution notebooks:
- Solution 1 (LSTM): Pad to 10 frames, Bidirectional LSTM on raw embeddings
- Solution 2 (CNN): Treat 10x128 embedding as 2D image, train CNN
- Solution 3 (Spectrogram+CNN): Convert to spectrogram, train deeper CNN
- Tabular approach: Flatten/aggregate embeddings into statistical features
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

from services.io_utils import save_data as _save_data

# Import reusable services
from services.classification_services import (
    train_lightgbm_classifier,
    predict_classifier,
)
from services.preprocessing_services import split_data, create_submission


# =============================================================================
# AUDIO EMBEDDING FEATURE EXTRACTION (Reusable for any VGGish-based dataset)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "json", "required": True, "schema": {"type": "json"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Extract tabular features from VGGish audio embeddings in JSON data",
    tags=["feature-engineering", "audio", "vggish", "generic"],
    version="1.0.0",
)
def extract_audio_embedding_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    embedding_column: str = "audio_embedding",
    id_column: str = "vid_id",
    target_column: Optional[str] = "is_turkey",
    max_frames: int = 10,
    frame_dim: int = 128,
    normalize_scale: float = 255.0,
    include_flattened: bool = True,
    include_stats: bool = True,
) -> str:
    """
    Extract tabular features from VGGish audio embedding JSON data.

    Converts variable-length audio embeddings (list of lists) into fixed-size
    feature vectors suitable for tabular ML models. Supports multiple
    feature extraction strategies:
    - Flattened: Pad/truncate to max_frames, flatten to max_frames * frame_dim
    - Statistics: Per-dimension mean, std, max, min across frames

    Works with any dataset containing VGGish-style embeddings
    (e.g., AudioSet, dont-call-me-turkey, etc.)

    G1 Compliance: Generic, works with any VGGish embedding dataset.
    G3 Compliance: Deterministic processing.
    G4 Compliance: All column names as parameters.

    Parameters:
        embedding_column: Column containing audio embedding lists
        id_column: ID column to preserve
        target_column: Target column to preserve (None for test data)
        max_frames: Maximum number of frames to pad/truncate to
        frame_dim: Dimensionality of each frame embedding
        normalize_scale: Divide embedding values by this (255 for uint8 VGGish)
        include_flattened: Include flattened frame features (max_frames * frame_dim)
        include_stats: Include per-dimension statistics (mean, std, max, min)
    """
    df = pd.read_json(inputs["data"])

    feature_rows = []
    for idx, row in df.iterrows():
        features = {}

        # Preserve ID column
        if id_column and id_column in df.columns:
            features[id_column] = row[id_column]

        # Preserve target column if present
        if target_column and target_column in df.columns:
            features[target_column] = row[target_column]

        # Extract embedding
        emb = np.array(row[embedding_column], dtype=np.float32)
        if normalize_scale > 0:
            emb = emb / normalize_scale

        n_frames = emb.shape[0]
        features["n_frames"] = n_frames

        # Preserve time features if present
        if "start_time_seconds_youtube_clip" in df.columns:
            features["start_time"] = row["start_time_seconds_youtube_clip"]
        if "end_time_seconds_youtube_clip" in df.columns:
            features["end_time"] = row["end_time_seconds_youtube_clip"]
            if "start_time_seconds_youtube_clip" in df.columns:
                features["clip_duration"] = (
                    row["end_time_seconds_youtube_clip"]
                    - row["start_time_seconds_youtube_clip"]
                )

        # Pad or truncate to fixed size
        if n_frames < max_frames:
            padded = np.zeros((max_frames, frame_dim), dtype=np.float32)
            padded[:n_frames] = emb[:, :frame_dim]
        else:
            padded = emb[:max_frames, :frame_dim]

        # Flattened features
        if include_flattened:
            flat = padded.flatten()
            for i, val in enumerate(flat):
                features[f"emb_{i}"] = float(val)

        # Statistical features across frames
        if include_stats:
            for d in range(frame_dim):
                col_vals = emb[:, d] if d < emb.shape[1] else np.zeros(n_frames)
                features[f"mean_{d}"] = float(np.mean(col_vals))
                features[f"std_{d}"] = float(np.std(col_vals))
                features[f"max_{d}"] = float(np.max(col_vals))
                features[f"min_{d}"] = float(np.min(col_vals))

        feature_rows.append(features)

    result_df = pd.DataFrame(feature_rows)
    _save_data(result_df, outputs["data"])

    n_features = len(result_df.columns)
    has_target = target_column and target_column in result_df.columns
    return (
        f"extract_audio_embedding_features: {len(result_df)} samples, "
        f"{n_features} features, target={'yes' if has_target else 'no'}"
    )


# =============================================================================
# AUDIO EMBEDDING CNN SERVICES (Based on top Kaggle solution notebook)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "json", "required": True, "schema": {"type": "json"}},
    },
    outputs={
        "X": {"format": "pickle", "schema": {"type": "array"}},
        "y": {"format": "pickle", "schema": {"type": "array"}},
        "ids": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Prepare audio embeddings as 2D images for CNN training",
    tags=["feature-engineering", "audio", "vggish", "cnn", "generic"],
    version="1.0.0",
)
def prepare_audio_embeddings_for_cnn(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    embedding_column: str = "audio_embedding",
    id_column: str = "vid_id",
    target_column: Optional[str] = "is_turkey",
    max_frames: int = 10,
    frame_dim: int = 128,
    normalize_scale: float = 255.0,
) -> str:
    """
    Prepare audio embeddings as 2D images for CNN classifier.

    Converts VGGish audio embeddings into 4D tensors (N, C, H, W) suitable
    for CNN training. Each embedding becomes a 1-channel 10x128 "image".

    Based on Kaggle solution notebook that achieved LB 0.90 by treating
    audio embeddings as images and using a CNN classifier.

    G1 Compliance: Generic, works with any VGGish embedding dataset.
    G3 Compliance: Deterministic processing.
    G4 Compliance: All column names as parameters.

    Parameters:
        embedding_column: Column containing audio embedding lists
        id_column: ID column to preserve
        target_column: Target column (None for test data)
        max_frames: Height of output image (pad/truncate frames to this)
        frame_dim: Width of output image (embedding dimension)
        normalize_scale: Divide values by this (255 for uint8 VGGish)
    """
    from scipy.ndimage import zoom

    df = pd.read_json(inputs["data"])

    X_list = []
    for idx, row in df.iterrows():
        emb = np.array(row[embedding_column], dtype=np.float32)
        if normalize_scale > 0:
            emb = emb / normalize_scale

        # Resize to fixed shape if needed
        if emb.shape != (max_frames, frame_dim):
            # Use scipy.ndimage.zoom for resizing
            zoom_factors = (max_frames / emb.shape[0], frame_dim / emb.shape[1])
            emb = zoom(emb, zoom_factors, order=1)

        X_list.append(emb.astype(np.float32))

    # Shape: (N, 1, H, W) - NCHW format for PyTorch
    X = np.array(X_list).reshape(-1, 1, max_frames, frame_dim)

    # Get labels if present
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column].values.astype(np.float32).reshape(-1, 1)

    # Get IDs
    ids_df = df[[id_column]].copy()

    # Save outputs
    for key in outputs:
        os.makedirs(os.path.dirname(outputs[key]) or ".", exist_ok=True)

    with open(outputs["X"], "wb") as f:
        pickle.dump(X, f, protocol=4)

    if y is not None:
        with open(outputs["y"], "wb") as f:
            pickle.dump(y, f, protocol=4)
    else:
        # Save empty array for test data
        with open(outputs["y"], "wb") as f:
            pickle.dump(np.array([]), f, protocol=4)

    ids_df.to_csv(outputs["ids"], index=False)

    return f"prepare_audio_embeddings_for_cnn: {len(X)} samples, shape={X.shape}"


@contract(
    inputs={
        "X_train": {"format": "pickle", "required": True},
        "y_train": {"format": "pickle", "required": True},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train a CNN classifier on audio embeddings (treated as 2D images)",
    tags=["modeling", "training", "cnn", "audio", "classification", "pytorch", "generic"],
    version="1.0.0",
)
def train_audio_cnn_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    random_state: int = 42,
) -> str:
    """
    Train a CNN classifier on audio embeddings.

    Architecture based on Kaggle solution notebook that achieved LB 0.90:
    - Conv2d(1, 3, 2) -> MaxPool(2,2)
    - Conv2d(3, 16, 2) -> MaxPool(2,2)
    - FC layers: 120 -> 84 -> 1
    - Sigmoid output for binary classification

    G1 Compliance: Generic CNN for any 2D audio embedding classification.
    G3 Compliance: Fixed random seeds for reproducibility.
    G4 Compliance: All hyperparameters as parameters.

    Parameters:
        n_epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Adam optimizer learning rate
        validation_split: Fraction of data for validation
        random_state: Random seed for reproducibility
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    with open(inputs["X_train"], "rb") as f:
        X = pickle.load(f)
    with open(inputs["y_train"], "rb") as f:
        y = pickle.load(f)

    # Flatten y if needed
    y = y.flatten()

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=random_state, stratify=y
    )

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    y_train_t = torch.FloatTensor(y_train)
    y_val_t = torch.FloatTensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # CNN Architecture (based on solution notebook 02)
    class AudioCNN(nn.Module):
        def __init__(self):
            super(AudioCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 3, kernel_size=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(3, 16, kernel_size=2)
            # After conv1: (10-2+1)x(128-2+1) = 9x127
            # After pool1: 4x63
            # After conv2: (4-2+1)x(63-2+1) = 3x62
            # After pool2: 1x31
            self.fc1 = nn.Linear(16 * 1 * 31, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 16 * 1 * 31)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Using device: {device}")

    model = AudioCNN().to(device)
    criterion = nn.MSELoss()  # Following the solution notebook
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_auc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            batch_preds = model(batch_X)
            loss = criterion(batch_preds, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device).unsqueeze(1)).item()
            val_probs = val_outputs.cpu().numpy().flatten()
            val_auc = roc_auc_score(y_val, val_probs)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model artifact
    model_data = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_type": "audio_cnn",
        "input_shape": tuple(X.shape[1:]),
    }

    # Create output directories
    for output_path in [outputs["model"], outputs["metrics"]]:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f, protocol=4)

    metrics = {
        "model_type": "audio_cnn_pytorch",
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_val_auc": best_auc,
        "device": str(device),
    }

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_audio_cnn_classifier: best_val_auc={best_auc:.4f}, {len(X_train)} train, {len(X_val)} val"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "X_test": {"format": "pickle", "required": True},
        "ids": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Predict using trained audio CNN classifier",
    tags=["modeling", "inference", "cnn", "audio", "classification", "pytorch", "generic"],
    version="1.0.0",
)
def predict_audio_cnn_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "vid_id",
    prediction_column: str = "is_turkey",
    threshold: float = 0.5,
    output_proba: bool = False,
) -> str:
    """
    Generate predictions using trained audio CNN classifier.

    G1 Compliance: Generic prediction service for audio CNN models.
    G3 Compliance: Deterministic inference.
    G4 Compliance: All parameters configurable.

    Parameters:
        id_column: Name of ID column in output
        prediction_column: Name of prediction column in output
        threshold: Classification threshold (default 0.5)
        output_proba: If True, output probabilities instead of class labels
    """
    import torch
    import torch.nn as nn

    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    with open(inputs["X_test"], "rb") as f:
        X_test = pickle.load(f)

    ids_df = pd.read_csv(inputs["ids"])

    # Rebuild model architecture
    class AudioCNN(nn.Module):
        def __init__(self):
            super(AudioCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 3, kernel_size=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(3, 16, kernel_size=2)
            self.fc1 = nn.Linear(16 * 1 * 31, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 16 * 1 * 31)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = AudioCNN()
    model.load_state_dict(model_data["state_dict"])
    model.to(device)
    model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        probs = model(X_test_t).cpu().numpy().flatten()

    if output_proba:
        preds = probs
    else:
        preds = (probs > threshold).astype(int)

    result_df = pd.DataFrame({
        id_column: ids_df[id_column].values,
        prediction_column: preds,
    })

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    result_df.to_csv(outputs["predictions"], index=False)

    return f"predict_audio_cnn_classifier: {len(result_df)} predictions"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Audio embedding processing (reusable for any VGGish dataset)
    "extract_audio_embedding_features": extract_audio_embedding_features,
    "prepare_audio_embeddings_for_cnn": prepare_audio_embeddings_for_cnn,
    "train_audio_cnn_classifier": train_audio_cnn_classifier,
    "predict_audio_cnn_classifier": predict_audio_cnn_classifier,
    # Reused from classification_services
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    # Reused from preprocessing_services
    "split_data": split_data,
    "create_submission": create_submission,
}
