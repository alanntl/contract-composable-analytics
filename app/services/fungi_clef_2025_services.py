"""
FungiCLEF 2025 - SLEGO Services
================================
Competition: https://www.kaggle.com/competitions/fungi-clef-2025
Problem Type: Multiclass Classification (Few-Shot Species Identification)
Target: category_id (species)

The competition requires predicting a ranked list of fungi species (category_id)
for each observation. Submission format: observationId, predictions (space-separated).

Approach based on top solution notebooks:
- Solutions 1 & 2 use BioCLIP/CLIP embeddings + kNN/Prototype classifiers
- We use CLIP from transformers to generate image embeddings
- Plus text embeddings from AI-generated captions
- Combined with metadata features and LightGBM for ranking

Services:
- extract_clip_embeddings: Extract CLIP image embeddings
- extract_caption_embeddings: Extract CLIP text embeddings from captions
- build_prototype_classifier: Build class prototypes from embeddings
- predict_with_prototypes: Predict using cosine similarity to prototypes
- predict_topk_classifier: Predict top-K species (LightGBM-based)
- predict_topk_ensemble: Ensemble metadata + embedding predictions
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# CLIP EMBEDDING EXTRACTION
# =============================================================================

@contract(
    inputs={
        "metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "embeddings": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Extract CLIP image embeddings for fungi images",
    tags=["embedding", "clip", "image", "fungi"],
    version="1.0.0",
)
def extract_clip_embeddings(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_base_dir: str = "fungi-clef-2025/images/FungiTastic-FewShot",
    split: str = "train",
    batch_size: int = 32,
    device: str = "auto",
) -> str:
    """Extract CLIP embeddings from fungi images.

    Uses OpenAI CLIP model from transformers to generate 512-dim embeddings.
    Processes images in batches for efficiency.
    """
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from tqdm import tqdm

    # Setup device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()

    # Load metadata
    df = _load_data(inputs["metadata"])

    # Resolve image directory
    storage_base = os.path.dirname(os.path.dirname(inputs["metadata"]))
    image_dir = os.path.join(storage_base, image_base_dir, split, "300p")

    embeddings = []
    filenames = []

    # Process in batches
    batch_images = []
    batch_names = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CLIP embeddings"):
        filename = row.get("filename", "")
        if not filename:
            continue

        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(storage_base, "fungi-clef-2025/images/FungiTastic-FewShot", split, "300p", filename),
                os.path.join(storage_base, f"fungi-clef-2025/images/FungiTastic-FewShot/{split}/300p", filename),
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    img_path = alt
                    break

        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            batch_images.append(image)
            batch_names.append(filename)

            if len(batch_images) >= batch_size:
                inputs_clip = processor(images=batch_images, return_tensors="pt", padding=True)
                inputs_clip = {k: v.to(device) for k, v in inputs_clip.items()}

                with torch.no_grad():
                    image_features = model.get_image_features(**inputs_clip)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                embeddings.extend(image_features.cpu().numpy())
                filenames.extend(batch_names)
                batch_images = []
                batch_names = []
        except Exception as e:
            continue

    # Process remaining batch
    if batch_images:
        inputs_clip = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs_clip = {k: v.to(device) for k, v in inputs_clip.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs_clip)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embeddings.extend(image_features.cpu().numpy())
        filenames.extend(batch_names)

    # Save embeddings
    result = {
        "embeddings": np.array(embeddings),
        "filenames": filenames,
        "model": "clip-vit-base-patch32",
        "dim": 512,
    }

    os.makedirs(os.path.dirname(outputs["embeddings"]) or ".", exist_ok=True)
    with open(outputs["embeddings"], "wb") as f:
        pickle.dump(result, f)

    return f"extract_clip_embeddings: {len(embeddings)} images, {result['dim']}-dim embeddings"


@contract(
    inputs={
        "metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "embeddings": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Extract CLIP text embeddings from AI-generated captions",
    tags=["embedding", "clip", "text", "captions", "fungi"],
    version="1.0.0",
)
def extract_caption_embeddings(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    captions_dir: str = "fungi-clef-2025/captions",
    split: str = "train",
    batch_size: int = 64,
    device: str = "auto",
) -> str:
    """Extract CLIP text embeddings from AI-generated image captions.

    Processes JSON caption files and extracts 512-dim CLIP text embeddings.
    """
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from tqdm import tqdm

    # Setup device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()

    # Load metadata
    df = _load_data(inputs["metadata"])

    # Resolve captions directory - metadata path is like storage/fungi-clef-2025/metadata/...
    # captions_dir param is relative to competition folder, e.g. "captions"
    metadata_path = inputs["metadata"]

    # Get competition base dir from metadata path
    # e.g., storage/fungi-clef-2025/metadata/... -> storage/fungi-clef-2025
    path_parts = metadata_path.replace("\\", "/").split("/")
    if "fungi-clef-2025" in path_parts:
        idx = path_parts.index("fungi-clef-2025")
        comp_base = "/".join(path_parts[:idx+1])
    else:
        comp_base = os.path.dirname(os.path.dirname(metadata_path))

    caps_dir = os.path.join(comp_base, captions_dir, split)

    # Debug print
    print(f"  Captions dir: {caps_dir}, exists: {os.path.exists(caps_dir)}")

    embeddings = []
    filenames = []

    batch_texts = []
    batch_names = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting caption embeddings"):
        filename = row.get("filename", "")
        if not filename:
            continue

        caption_file = os.path.join(caps_dir, f"{filename}.json")
        if not os.path.exists(caption_file):
            continue

        try:
            with open(caption_file, "r") as f:
                caption = json.load(f)

            if isinstance(caption, str):
                text = caption[:500]  # Truncate long captions
            else:
                text = str(caption)[:500]

            batch_texts.append(text)
            batch_names.append(filename)

            if len(batch_texts) >= batch_size:
                inputs_clip = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
                inputs_clip = {k: v.to(device) for k, v in inputs_clip.items()}

                with torch.no_grad():
                    text_features = model.get_text_features(**inputs_clip)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                embeddings.extend(text_features.cpu().numpy())
                filenames.extend(batch_names)
                batch_texts = []
                batch_names = []
        except Exception as e:
            continue

    # Process remaining batch
    if batch_texts:
        inputs_clip = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs_clip = {k: v.to(device) for k, v in inputs_clip.items()}

        with torch.no_grad():
            text_features = model.get_text_features(**inputs_clip)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        embeddings.extend(text_features.cpu().numpy())
        filenames.extend(batch_names)

    # Save embeddings
    result = {
        "embeddings": np.array(embeddings),
        "filenames": filenames,
        "model": "clip-vit-base-patch32",
        "dim": 512,
    }

    os.makedirs(os.path.dirname(outputs["embeddings"]) or ".", exist_ok=True)
    with open(outputs["embeddings"], "wb") as f:
        pickle.dump(result, f)

    return f"extract_caption_embeddings: {len(embeddings)} captions, {result['dim']}-dim embeddings"


# =============================================================================
# PROTOTYPE-BASED CLASSIFIER (following solution notebooks)
# =============================================================================

@contract(
    inputs={
        "embeddings": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "prototypes": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Build class prototypes by averaging embeddings per species",
    tags=["modeling", "prototype", "classification", "fungi"],
    version="1.0.0",
)
def build_prototype_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "category_id",
) -> str:
    """Build class prototypes following solution notebook approach.

    For each class, compute the centroid (mean) of all embeddings.
    These prototypes are used for nearest-centroid classification.
    """
    # Load embeddings
    with open(inputs["embeddings"], "rb") as f:
        emb_data = pickle.load(f)

    embeddings = emb_data["embeddings"]
    filenames = emb_data["filenames"]

    # Load metadata with labels
    df = _load_data(inputs["metadata"])

    # Create filename to embedding mapping
    filename_to_idx = {fn: i for i, fn in enumerate(filenames)}

    # Build prototypes
    class_embeddings = {}
    for _, row in df.iterrows():
        filename = row.get("filename", "")
        label = row.get(label_column)

        if filename in filename_to_idx and pd.notna(label):
            idx = filename_to_idx[filename]
            label = int(label)

            if label not in class_embeddings:
                class_embeddings[label] = []
            class_embeddings[label].append(embeddings[idx])

    # Compute prototypes (mean embedding per class)
    prototypes = {}
    for label, embs in class_embeddings.items():
        prototype = np.mean(embs, axis=0)
        prototype = prototype / np.linalg.norm(prototype)  # L2 normalize
        prototypes[label] = prototype

    # Save prototypes
    result = {
        "prototypes": prototypes,
        "n_classes": len(prototypes),
        "dim": embeddings.shape[1],
    }

    os.makedirs(os.path.dirname(outputs["prototypes"]) or ".", exist_ok=True)
    with open(outputs["prototypes"], "wb") as f:
        pickle.dump(result, f)

    return f"build_prototype_classifier: {len(prototypes)} class prototypes, {result['dim']}-dim"


@contract(
    inputs={
        "embeddings": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "prototypes": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Predict species using prototype cosine similarity (like solution notebooks)",
    tags=["modeling", "prediction", "prototype", "fungi"],
    version="1.0.0",
)
def predict_with_prototypes(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "observationID",
    submission_id_column: str = "observationId",
    prediction_column: str = "predictions",
    top_k: int = 10,
) -> str:
    """Predict using prototype classifier (following solution notebooks).

    For each test sample, compute cosine similarity to all class prototypes
    and return the top-K most similar classes.
    """
    # Load embeddings
    with open(inputs["embeddings"], "rb") as f:
        emb_data = pickle.load(f)

    embeddings = emb_data["embeddings"]
    filenames = emb_data["filenames"]

    # Load prototypes
    with open(inputs["prototypes"], "rb") as f:
        proto_data = pickle.load(f)

    prototypes = proto_data["prototypes"]

    # Load test metadata
    df = _load_data(inputs["metadata"])

    # Create prototype matrix for efficient similarity computation
    class_labels = sorted(prototypes.keys())
    proto_matrix = np.array([prototypes[c] for c in class_labels])  # (n_classes, dim)

    # Create filename to embedding mapping
    filename_to_idx = {fn: i for i, fn in enumerate(filenames)}

    # Predict for each observation
    results = {}
    for _, row in df.iterrows():
        obs_id = row.get(id_column)
        filename = row.get("filename", "")

        if filename not in filename_to_idx:
            continue

        idx = filename_to_idx[filename]
        emb = embeddings[idx].reshape(1, -1)  # (1, dim)

        # Cosine similarity (embeddings are already normalized)
        similarities = np.dot(emb, proto_matrix.T).squeeze()  # (n_classes,)

        # Get top-K
        top_indices = np.argsort(-similarities)[:top_k]
        top_classes = [class_labels[i] for i in top_indices]

        # Aggregate by observation (may have multiple images)
        if obs_id not in results:
            results[obs_id] = {}
        for rank, cls in enumerate(top_classes):
            if cls not in results[obs_id]:
                results[obs_id][cls] = 0
            results[obs_id][cls] += (top_k - rank)  # Higher weight for higher rank

    # Build submission
    submission_rows = []
    for obs_id, class_scores in results.items():
        sorted_classes = sorted(class_scores.keys(), key=lambda c: -class_scores[c])
        top_predictions = sorted_classes[:top_k]
        pred_str = " ".join(str(int(c)) for c in top_predictions)
        submission_rows.append({submission_id_column: obs_id, prediction_column: pred_str})

    submission_df = pd.DataFrame(submission_rows)
    _save_data(submission_df, outputs["submission"])

    return f"predict_with_prototypes: {len(submission_df)} observations, top-{top_k} predictions"


# =============================================================================
# ORIGINAL TOP-K CLASSIFIER (for metadata-only baseline)
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact", "artifact_type": "model"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Predict top-K species and format as space-separated submission",
    tags=["modeling", "prediction", "classification", "topk", "fungi", "generic"],
    version="1.0.0",
)
def predict_topk_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "observationID",
    prediction_column: str = "predictions",
    submission_id_column: str = "observationId",
    top_k: int = 10,
) -> str:
    """Predict top-K class labels for multiclass classification."""
    # Load model artifact
    with open(inputs["model"], "rb") as f:
        artifact = pickle.load(f)

    data_df = _load_data(inputs["data"])

    model = artifact["model"]
    feature_cols = artifact.get("feature_cols", [])

    if feature_cols:
        for col in feature_cols:
            if col not in data_df.columns:
                data_df[col] = 0
        X = data_df[feature_cols]
    else:
        drop_cols = [id_column]
        X = data_df.drop(columns=drop_cols, errors="ignore")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        class_labels = model.classes_

        top_k_actual = min(top_k, proba.shape[1])
        top_k_indices = np.argsort(-proba, axis=1)[:, :top_k_actual]

        predictions = []
        for row_indices in top_k_indices:
            pred_classes = class_labels[row_indices]
            predictions.append(" ".join(str(int(c)) for c in pred_classes))
    else:
        predicted = model.predict(X)
        predictions = [str(int(p)) for p in predicted]

    result = pd.DataFrame()
    if id_column in data_df.columns:
        result[submission_id_column] = data_df[id_column]
    else:
        result[submission_id_column] = range(len(data_df))

    result[prediction_column] = predictions
    result = result.drop_duplicates(subset=submission_id_column)

    _save_data(result, outputs["submission"])

    n_preds = len(result)
    return f"predict_topk_classifier: {n_preds} observations, top-{top_k_actual} predictions"


# =============================================================================
# ENSEMBLE PREDICTION
# =============================================================================

@contract(
    inputs={
        "metadata_submission": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "prototype_submission": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Ensemble metadata and prototype predictions using rank fusion",
    tags=["modeling", "ensemble", "prediction", "fungi"],
    version="1.0.0",
)
def ensemble_submissions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "observationId",
    prediction_column: str = "predictions",
    metadata_weight: float = 0.3,
    prototype_weight: float = 0.7,
    top_k: int = 10,
) -> str:
    """Ensemble two submission files using weighted rank fusion.

    Combines metadata-based and prototype-based predictions.
    """
    meta_df = _load_data(inputs["metadata_submission"])
    proto_df = _load_data(inputs["prototype_submission"])

    # Index by observation ID
    meta_preds = dict(zip(meta_df[id_column], meta_df[prediction_column]))
    proto_preds = dict(zip(proto_df[id_column], proto_df[prediction_column]))

    all_obs_ids = set(meta_preds.keys()) | set(proto_preds.keys())

    results = []
    for obs_id in all_obs_ids:
        scores = {}

        # Metadata predictions
        if obs_id in meta_preds:
            preds = meta_preds[obs_id].split()
            for rank, cls in enumerate(preds):
                cls = int(cls)
                if cls not in scores:
                    scores[cls] = 0
                scores[cls] += metadata_weight * (top_k - rank)

        # Prototype predictions
        if obs_id in proto_preds:
            preds = proto_preds[obs_id].split()
            for rank, cls in enumerate(preds):
                cls = int(cls)
                if cls not in scores:
                    scores[cls] = 0
                scores[cls] += prototype_weight * (top_k - rank)

        # Get top-K by score
        sorted_classes = sorted(scores.keys(), key=lambda c: -scores[c])[:top_k]
        pred_str = " ".join(str(c) for c in sorted_classes)
        results.append({id_column: obs_id, prediction_column: pred_str})

    result_df = pd.DataFrame(results)
    _save_data(result_df, outputs["submission"])

    return f"ensemble_submissions: {len(result_df)} observations, weights meta={metadata_weight}, proto={prototype_weight}"


# =============================================================================
# FAST TF-IDF BASED CAPTION FEATURES
# =============================================================================

@contract(
    inputs={
        "train_metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Train LightGBM on TF-IDF caption features and predict top-K species",
    tags=["modeling", "classification", "tfidf", "fungi"],
    version="1.0.0",
)
def train_and_predict_tfidf(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    captions_dir: str = "captions",
    label_column: str = "category_id",
    id_column: str = "observationID",
    submission_id_column: str = "observationId",
    prediction_column: str = "predictions",
    top_k: int = 10,
    max_features: int = 500,
    n_estimators: int = 200,
) -> str:
    """Train LightGBM on TF-IDF caption features and predict.

    Fast end-to-end pipeline:
    1. Load captions from JSON files
    2. Extract TF-IDF features
    3. Train LightGBM classifier
    4. Predict top-K species per observation
    """
    import lightgbm as lgb
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Resolve paths
    train_path = inputs["train_metadata"]
    path_parts = train_path.replace("\\", "/").split("/")
    if "fungi-clef-2025" in path_parts:
        idx = path_parts.index("fungi-clef-2025")
        comp_base = "/".join(path_parts[:idx+1])
    else:
        comp_base = os.path.dirname(os.path.dirname(train_path))

    # Load train metadata and captions
    train_df = _load_data(inputs["train_metadata"])
    train_caps_dir = os.path.join(comp_base, captions_dir, "train")
    print(f"  Train captions dir: {train_caps_dir}")

    train_texts = []
    train_labels = []
    train_filenames = []
    for _, row in train_df.iterrows():
        fn = row.get("filename", "")
        label = row.get(label_column)
        if not fn or pd.isna(label):
            continue

        caption_file = os.path.join(train_caps_dir, f"{fn}.json")
        if os.path.exists(caption_file):
            try:
                with open(caption_file) as f:
                    cap = json.load(f)
                text = cap if isinstance(cap, str) else str(cap)
            except:
                text = ""
        else:
            text = ""

        train_texts.append(text)
        train_labels.append(int(label))
        train_filenames.append(fn)

    print(f"  Loaded {len(train_texts)} train captions")

    # Load test metadata and captions
    test_df = _load_data(inputs["test_metadata"])
    test_caps_dir = os.path.join(comp_base, captions_dir, "test")
    print(f"  Test captions dir: {test_caps_dir}")

    test_texts = []
    test_obs_ids = []
    test_filenames = []
    for _, row in test_df.iterrows():
        fn = row.get("filename", "")
        obs_id = row.get(id_column)
        if not fn:
            continue

        caption_file = os.path.join(test_caps_dir, f"{fn}.json")
        if os.path.exists(caption_file):
            try:
                with open(caption_file) as f:
                    cap = json.load(f)
                text = cap if isinstance(cap, str) else str(cap)
            except:
                text = ""
        else:
            text = ""

        test_texts.append(text)
        test_obs_ids.append(obs_id)
        test_filenames.append(fn)

    print(f"  Loaded {len(test_texts)} test captions")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    y_train = np.array(train_labels)

    print(f"  TF-IDF features: {X_train.shape[1]}")

    # Train LightGBM
    n_classes = len(np.unique(y_train))
    print(f"  Training LightGBM with {n_classes} classes...")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        num_leaves=31,
        n_jobs=-1,
        objective="multiclass",
        num_class=n_classes,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Predict
    proba = model.predict_proba(X_test)
    class_labels = model.classes_

    # Aggregate by observation
    results = {}
    for i in range(len(test_texts)):
        obs_id = test_obs_ids[i]

        top_indices = np.argsort(-proba[i])[:top_k]
        top_classes = class_labels[top_indices]

        if obs_id not in results:
            results[obs_id] = {}
        for rank, cls in enumerate(top_classes):
            if cls not in results[obs_id]:
                results[obs_id][cls] = 0
            results[obs_id][cls] += (top_k - rank)

    # Build submission
    submission_rows = []
    for obs_id, scores in results.items():
        sorted_classes = sorted(scores.keys(), key=lambda c: -scores[c])[:top_k]
        pred_str = " ".join(str(int(c)) for c in sorted_classes)
        submission_rows.append({submission_id_column: obs_id, prediction_column: pred_str})

    submission_df = pd.DataFrame(submission_rows)
    _save_data(submission_df, outputs["submission"])

    return f"train_and_predict_tfidf: {len(X_train)} train, {len(submission_df)} predictions, {n_classes} classes"


# =============================================================================
# CONVNEXT V2 TRAINING AND PREDICTION (Following solution notebooks approach)
# =============================================================================

@contract(
    inputs={
        "train_metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "metrics"}},
    },
    description="Train ConvNeXt V2 small on fungi images with fine-tuning",
    tags=["modeling", "training", "convnextv2", "image", "fungi"],
    version="1.0.0",
)
def train_convnextv2_fungi(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_dir: str = "fungi-clef-2025/images/FungiTastic-FewShot/train/300p",
    label_column: str = "category_id",
    filename_column: str = "filename",
    n_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = False,
    validation_split: float = 0.1,
    random_state: int = 42,
    target_size: int = 224,
    sample_size: Optional[int] = None,
    use_mixup: bool = True,
) -> str:
    """Train ConvNeXt V2 small on fungi images.

    Uses timm's pretrained ConvNeXt V2 small model with ImageNet weights.
    Fine-tunes on the fungi dataset with mixup augmentation for better generalization.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from PIL import Image
    import torchvision.transforms as T
    import timm
    from tqdm import tqdm

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Load metadata
    train_df = _load_data(inputs["train_metadata"])

    # Resolve image directory
    metadata_path = inputs["train_metadata"]
    path_parts = metadata_path.replace("\\", "/").split("/")
    if "fungi-clef-2025" in path_parts:
        idx = path_parts.index("fungi-clef-2025")
        base_path = "/".join(path_parts[:idx])
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(metadata_path)))

    full_image_dir = os.path.join(base_path, image_dir)
    print(f"  Image directory: {full_image_dir}")
    print(f"  Training samples: {len(train_df)}")

    # Sample if needed
    if sample_size and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"  Sampled to: {len(train_df)}")

    # Encode labels
    label_encoder = LabelEncoder()
    train_df["_label_idx"] = label_encoder.fit_transform(train_df[label_column])
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)
    print(f"  Number of classes: {n_classes}")

    # Train/val split
    train_split, val_split = train_test_split(
        train_df, test_size=validation_split, random_state=random_state,
        stratify=train_df["_label_idx"]
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    # Transforms
    train_transform = T.Compose([
        T.Resize((target_size + 32, target_size + 32)),
        T.RandomCrop(target_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    class FungiDataset(Dataset):
        def __init__(self, df, img_dir, fn_col, transform):
            self.df = df
            self.img_dir = img_dir
            self.fn_col = fn_col
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            filename = row[self.fn_col]
            img_path = os.path.join(self.img_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                # Return a black image if file not found
                img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            img = self.transform(img)
            label = int(row["_label_idx"])
            return img, label

    train_dataset = FungiDataset(train_split, full_image_dir, filename_column, train_transform)
    val_dataset = FungiDataset(val_split, full_image_dir, filename_column, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # Model - ConvNeXt V2 small
    print("  Loading ConvNeXt V2 small pretrained model...")
    model = timm.create_model('convnextv2_small', pretrained=True, num_classes=n_classes)

    if freeze_backbone:
        # Freeze all layers except the classifier head
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    # Device
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_accs = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")):
            images, labels = images.to(device), labels.to(device)

            # Mixup augmentation
            if use_mixup and np.random.random() > 0.5:
                lam = np.random.beta(0.4, 0.4)
                perm_idx = torch.randperm(images.size(0))
                mixed_images = lam * images + (1 - lam) * images[perm_idx]
                logits = model(mixed_images)
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm_idx])
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_accs.append(val_acc)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model
    model_artifact = {
        "model_state": model.cpu().state_dict(),
        "label_encoder": label_encoder,
        "class_names": class_names,
        "n_classes": n_classes,
        "architecture": "convnextv2_small",
        "target_size": target_size,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_artifact, f)

    # Save metrics
    metrics = {
        "val_accuracy": best_val_acc,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "n_classes": n_classes,
        "n_train": len(train_split),
        "n_val": len(val_split),
    }
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_convnextv2_fungi: {n_classes} classes, val_acc={best_val_acc:.4f}, epochs={n_epochs}"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "test_metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Predict top-K species using trained ConvNeXt V2 model",
    tags=["modeling", "prediction", "convnextv2", "fungi"],
    version="1.0.0",
)
def predict_convnextv2_fungi(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_dir: str = "fungi-clef-2025/images/FungiTastic-FewShot/test/300p",
    filename_column: str = "filename",
    id_column: str = "observationID",
    submission_id_column: str = "observationId",
    prediction_column: str = "predictions",
    top_k: int = 10,
    batch_size: int = 32,
) -> str:
    """Predict top-K species for test images using trained ConvNeXt V2 model."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torchvision.transforms as T
    import timm
    from tqdm import tqdm

    # Load model
    with open(inputs["model"], "rb") as f:
        model_artifact = pickle.load(f)

    label_encoder = model_artifact["label_encoder"]
    n_classes = model_artifact["n_classes"]
    target_size = model_artifact.get("target_size", 224)

    # Recreate model
    model = timm.create_model('convnextv2_small', pretrained=False, num_classes=n_classes)
    model.load_state_dict(model_artifact["model_state"])

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)
    model.eval()

    # Load test metadata
    test_df = _load_data(inputs["test_metadata"])

    # Resolve image directory
    metadata_path = inputs["test_metadata"]
    path_parts = metadata_path.replace("\\", "/").split("/")
    if "fungi-clef-2025" in path_parts:
        idx = path_parts.index("fungi-clef-2025")
        base_path = "/".join(path_parts[:idx])
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(metadata_path)))

    full_image_dir = os.path.join(base_path, image_dir)
    print(f"  Image directory: {full_image_dir}")
    print(f"  Test samples: {len(test_df)}")

    # Transform
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    class TestDataset(Dataset):
        def __init__(self, df, img_dir, fn_col, transform):
            self.df = df
            self.img_dir = img_dir
            self.fn_col = fn_col
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            filename = row[self.fn_col]
            img_path = os.path.join(self.img_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            img = self.transform(img)
            return img, idx

    test_dataset = TestDataset(test_df, full_image_dir, filename_column, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Predict
    all_probs = []
    with torch.no_grad():
        for images, indices in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Aggregate predictions by observation
    class_labels = label_encoder.classes_
    results = {}

    for i in range(len(test_df)):
        obs_id = test_df.iloc[i][id_column]
        probs = all_probs[i]

        top_indices = np.argsort(-probs)[:top_k]
        top_classes = class_labels[top_indices]
        top_probs = probs[top_indices]

        if obs_id not in results:
            results[obs_id] = {}

        for cls, prob in zip(top_classes, top_probs):
            if cls not in results[obs_id]:
                results[obs_id][cls] = 0
            results[obs_id][cls] += prob

    # Build submission
    submission_rows = []
    for obs_id, class_scores in results.items():
        sorted_classes = sorted(class_scores.keys(), key=lambda c: -class_scores[c])[:top_k]
        pred_str = " ".join(str(int(c)) for c in sorted_classes)
        submission_rows.append({submission_id_column: obs_id, prediction_column: pred_str})

    submission_df = pd.DataFrame(submission_rows)
    _save_data(submission_df, outputs["submission"])

    return f"predict_convnextv2_fungi: {len(submission_df)} observations, top-{top_k} predictions"


# =============================================================================
# EFFICIENTNET V2-S TRAINING AND PREDICTION (Fast and reliable model)
# =============================================================================

@contract(
    inputs={
        "train_metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "metrics"}},
    },
    description="Train EfficientNetV2-S on fungi images with fine-tuning (fast and reliable)",
    tags=["modeling", "training", "efficientnetv2", "image", "fungi"],
    version="1.0.0",
)
def train_efficientnetv2_fungi(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_dir: str = "fungi-clef-2025/images/FungiTastic-FewShot/train/300p",
    label_column: str = "category_id",
    filename_column: str = "filename",
    n_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    freeze_backbone: bool = False,
    validation_split: float = 0.1,
    random_state: int = 42,
    target_size: int = 224,
    sample_size: Optional[int] = None,
    use_mixup: bool = True,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
) -> str:
    """Train EfficientNetV2-S on fungi images.

    Uses torchvision's pretrained EfficientNetV2-S model with ImageNet weights.
    Fine-tunes on the fungi dataset with mixup augmentation for better generalization.

    EfficientNetV2-S is fast and reliable, good balance between speed and accuracy.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models
    from tqdm import tqdm

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Load metadata
    train_df = _load_data(inputs["train_metadata"])

    # Resolve image directory
    metadata_path = inputs["train_metadata"]
    path_parts = metadata_path.replace("\\", "/").split("/")
    if "fungi-clef-2025" in path_parts:
        idx = path_parts.index("fungi-clef-2025")
        base_path = "/".join(path_parts[:idx])
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(metadata_path)))

    full_image_dir = os.path.join(base_path, image_dir)
    print(f"  Image directory: {full_image_dir}")
    print(f"  Training samples: {len(train_df)}")

    # Sample if needed
    if sample_size and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"  Sampled to: {len(train_df)}")

    # Encode labels
    label_encoder = LabelEncoder()
    train_df["_label_idx"] = label_encoder.fit_transform(train_df[label_column])
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)
    print(f"  Number of classes: {n_classes}")

    # Train/val split - use random split (not stratified) for few-shot datasets
    # Many classes have only 1 sample, so stratified splitting fails
    train_split, val_split = train_test_split(
        train_df, test_size=validation_split, random_state=random_state,
        stratify=None  # Random split for few-shot learning
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    # Transforms - EfficientNetV2-S optimal input size is 384, but 224 works well too
    train_transform = T.Compose([
        T.Resize((target_size + 32, target_size + 32)),
        T.RandomCrop(target_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    class FungiDataset(Dataset):
        def __init__(self, df, img_dir, fn_col, transform, target_sz):
            self.df = df
            self.img_dir = img_dir
            self.fn_col = fn_col
            self.transform = transform
            self.target_sz = target_sz

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            filename = row[self.fn_col]
            img_path = os.path.join(self.img_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                # Return a black image if file not found
                img = Image.new("RGB", (self.target_sz, self.target_sz), (0, 0, 0))
            img = self.transform(img)
            label = int(row["_label_idx"])
            return img, label

    train_dataset = FungiDataset(train_split, full_image_dir, filename_column, train_transform, target_size)
    val_dataset = FungiDataset(val_split, full_image_dir, filename_column, val_transform, target_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # Model - EfficientNetV2-S (torchvision)
    print("  Loading EfficientNetV2-S pretrained model...")
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)

    if freeze_backbone:
        # Freeze all layers except the classifier head
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # Device
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_accs = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")):
            images, labels = images.to(device), labels.to(device)

            # Mixup augmentation
            if use_mixup and np.random.random() > 0.5:
                lam = np.random.beta(0.4, 0.4)
                perm_idx = torch.randperm(images.size(0))
                mixed_images = lam * images + (1 - lam) * images[perm_idx]
                logits = model(mixed_images)
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm_idx])
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_accs.append(val_acc)
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model
    model_artifact = {
        "model_state": model.cpu().state_dict(),
        "label_encoder": label_encoder,
        "class_names": class_names,
        "n_classes": n_classes,
        "architecture": "efficientnet_v2_s",
        "target_size": target_size,
        "in_features": in_features,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_artifact, f)

    # Save metrics
    metrics = {
        "val_accuracy": best_val_acc,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "n_classes": n_classes,
        "n_train": len(train_split),
        "n_val": len(val_split),
        "architecture": "efficientnet_v2_s",
    }
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_efficientnetv2_fungi: {n_classes} classes, val_acc={best_val_acc:.4f}, epochs={n_epochs}"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "test_metadata": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Predict top-K species using trained EfficientNetV2-S model",
    tags=["modeling", "prediction", "efficientnetv2", "fungi"],
    version="1.0.0",
)
def predict_efficientnetv2_fungi(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_dir: str = "fungi-clef-2025/images/FungiTastic-FewShot/test/300p",
    filename_column: str = "filename",
    id_column: str = "observationID",
    submission_id_column: str = "observationId",
    prediction_column: str = "predictions",
    top_k: int = 10,
    batch_size: int = 32,
) -> str:
    """Predict top-K species for test images using trained EfficientNetV2-S model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models
    from tqdm import tqdm

    # Load model
    with open(inputs["model"], "rb") as f:
        model_artifact = pickle.load(f)

    label_encoder = model_artifact["label_encoder"]
    n_classes = model_artifact["n_classes"]
    target_size = model_artifact.get("target_size", 224)
    in_features = model_artifact.get("in_features", 1280)

    # Recreate model
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(in_features, n_classes)
    model.load_state_dict(model_artifact["model_state"])

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)
    model.eval()

    # Load test metadata
    test_df = _load_data(inputs["test_metadata"])

    # Resolve image directory
    metadata_path = inputs["test_metadata"]
    path_parts = metadata_path.replace("\\", "/").split("/")
    if "fungi-clef-2025" in path_parts:
        idx = path_parts.index("fungi-clef-2025")
        base_path = "/".join(path_parts[:idx])
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(metadata_path)))

    full_image_dir = os.path.join(base_path, image_dir)
    print(f"  Image directory: {full_image_dir}")
    print(f"  Test samples: {len(test_df)}")

    # Transform
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    class TestDataset(Dataset):
        def __init__(self, df, img_dir, fn_col, transform, target_sz):
            self.df = df
            self.img_dir = img_dir
            self.fn_col = fn_col
            self.transform = transform
            self.target_sz = target_sz

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            filename = row[self.fn_col]
            img_path = os.path.join(self.img_dir, filename)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (self.target_sz, self.target_sz), (0, 0, 0))
            img = self.transform(img)
            return img, idx

    test_dataset = TestDataset(test_df, full_image_dir, filename_column, transform, target_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Predict
    all_probs = []
    with torch.no_grad():
        for images, indices in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Aggregate predictions by observation
    class_labels = label_encoder.classes_
    results = {}

    for i in range(len(test_df)):
        obs_id = test_df.iloc[i][id_column]
        probs = all_probs[i]

        top_indices = np.argsort(-probs)[:top_k]
        top_classes = class_labels[top_indices]
        top_probs = probs[top_indices]

        if obs_id not in results:
            results[obs_id] = {}

        for cls, prob in zip(top_classes, top_probs):
            if cls not in results[obs_id]:
                results[obs_id][cls] = 0
            results[obs_id][cls] += prob

    # Build submission
    submission_rows = []
    for obs_id, class_scores in results.items():
        sorted_classes = sorted(class_scores.keys(), key=lambda c: -class_scores[c])[:top_k]
        pred_str = " ".join(str(int(c)) for c in sorted_classes)
        submission_rows.append({submission_id_column: obs_id, prediction_column: pred_str})

    submission_df = pd.DataFrame(submission_rows)
    _save_data(submission_df, outputs["submission"])

    return f"predict_efficientnetv2_fungi: {len(submission_df)} observations, top-{top_k} predictions"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "predict_topk_classifier": predict_topk_classifier,
    "train_convnextv2_fungi": train_convnextv2_fungi,
    "predict_convnextv2_fungi": predict_convnextv2_fungi,
    "train_efficientnetv2_fungi": train_efficientnetv2_fungi,
    "predict_efficientnetv2_fungi": predict_efficientnetv2_fungi,
}
