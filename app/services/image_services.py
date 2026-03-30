"""
Image Services - Common Module
======================================

Image processing and classification services for image competitions.
Uses PIL/Pillow and sklearn. Optional torch support for CNN training.

Services:
  Processing: load_image_dataset, resize_images, normalize_images, augment_images
  Loading: load_labeled_images, prepare_multilabel_images, prepare_test_images
  Modeling (sklearn): train_image_classifier, predict_image_classifier
  Modeling (CNN/PyTorch): train_cnn_image_classifier, predict_cnn_image_classifier
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract


# =============================================================================
# IMAGE DATASET LOADING
# =============================================================================

@contract(
    inputs={
        "data_dir": {"format": "csv", "required": True},
    },
    outputs={
        "metadata": {"format": "csv"},
    },
    description="Scan a directory for images and produce metadata CSV with file_path, label, width, height",
    tags=["image", "loading", "preprocessing"],
)
def load_image_dataset(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_extensions: List[str] = None,
    labels_csv: str = None,
    id_column: str = "filename",
    label_column: str = "label",
) -> str:
    """
    Scan a directory for images using class_name/image.jpg structure
    or flat directory with an optional labels CSV mapping filename to label.

    Parameters
    ----------
    inputs : dict
        data_dir : str - Path to the root image directory (passed as a path string,
                   the contract slot holds the directory path directly).
    outputs : dict
        metadata : str - Output CSV path with columns: file_path, label, width, height.
    image_extensions : list of str, optional
        File extensions to include. Default: [".jpg", ".jpeg", ".png", ".bmp"].
    labels_csv : str, optional
        Path to a CSV file mapping filename to label. Expected columns: filename, label.
        If None, labels are inferred from subdirectory names.
    id_column : str, optional
        Name of the column containing image IDs or filenames in labels_csv. Default: "filename".
    label_column : str, optional
        Name of the column containing labels in labels_csv. Default: "label".
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image services. Install with: pip install Pillow"
        )

    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    data_dir = inputs["data_dir"]
    image_extensions_lower = [ext.lower() for ext in image_extensions]

    # Load labels mapping if provided
    label_map = {}
    if labels_csv is not None and os.path.exists(labels_csv):
        labels_df = pd.read_csv(labels_csv)
        if id_column in labels_df.columns and label_column in labels_df.columns:
            # Convert IDs to string to ensure matching works
            labels_df[id_column] = labels_df[id_column].astype(str)
            label_map = dict(zip(labels_df[id_column], labels_df[label_column]))
        else:
            raise ValueError(
                f"labels_csv must contain '{id_column}' and '{label_column}' columns. "
                f"Found: {list(labels_df.columns)}"
            )

    records = []

    for root, dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in image_extensions_lower:
                continue

            file_path = os.path.join(root, fname)

            # Determine label
            if label_map:
                # Flat structure with labels CSV
                # Try exact match first, then match without extension
                fname_no_ext = os.path.splitext(fname)[0]
                if fname in label_map:
                    label = label_map[fname]
                elif fname_no_ext in label_map:
                    label = label_map[fname_no_ext]
                else:
                    label = "unknown"
            else:
                # Subdirectory structure: data_dir/class_name/image.ext
                rel_path = os.path.relpath(root, data_dir)
                if rel_path == ".":
                    label = "unknown"
                else:
                    label = rel_path.split(os.sep)[0]

            # Read image dimensions
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"  [WARN] Could not read {file_path}: {e}")
                width, height = 0, 0

            records.append({
                "file_path": file_path,
                "label": label,
                "width": width,
                "height": height,
            })

    if not records:
        raise ValueError(
            f"No images found in '{data_dir}' with extensions {image_extensions}"
        )

    metadata_df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(outputs["metadata"]) or ".", exist_ok=True)
    metadata_df.to_csv(outputs["metadata"], index=False)

    n_labels = metadata_df["label"].nunique()
    return (
        f"load_image_dataset: {len(records)} images found, "
        f"{n_labels} unique labels, "
        f"extensions={image_extensions}"
    )


# =============================================================================
# IMAGE RESIZING
# =============================================================================

@contract(
    inputs={
        "metadata": {"format": "csv", "required": True},
        "source_dir": {"format": "csv", "required": True},
    },
    outputs={
        "resized_dir": {"format": "csv"},
        "metadata": {"format": "csv"},
    },
    description="Resize all images listed in metadata CSV to a target size",
    tags=["image", "resizing", "preprocessing"],
)
def resize_images(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_size: tuple = (224, 224),
    keep_aspect_ratio: bool = False,
) -> str:
    """
    Resize all images listed in the metadata CSV to the target size.

    Parameters
    ----------
    inputs : dict
        metadata : str - Path to metadata CSV with 'file_path' column.
        source_dir : str - Root directory containing the source images.
    outputs : dict
        resized_dir : str - Path to directory where resized images will be saved
                      (passed as a path string via the contract slot).
        metadata : str - Updated metadata CSV with resized file paths and dimensions.
    target_size : tuple of (int, int)
        Target (width, height). Default: (224, 224).
    keep_aspect_ratio : bool
        If True, resize preserving aspect ratio and pad with black pixels.
        If False, stretch to exact target size. Default: False.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image services. Install with: pip install Pillow"
        )

    metadata_df = pd.read_csv(inputs["metadata"])

    if "file_path" not in metadata_df.columns:
        raise ValueError("Metadata CSV must contain a 'file_path' column")

    resized_dir = outputs["resized_dir"]
    os.makedirs(resized_dir, exist_ok=True)

    target_w, target_h = target_size
    resized_records = []
    resized_count = 0
    error_count = 0

    for idx, row in metadata_df.iterrows():
        src_path = row["file_path"]
        fname = os.path.basename(src_path)

        # Preserve subdirectory structure relative to source_dir
        source_dir = inputs["source_dir"]
        rel_path = os.path.relpath(os.path.dirname(src_path), source_dir)
        out_subdir = os.path.join(resized_dir, rel_path) if rel_path != "." else resized_dir
        os.makedirs(out_subdir, exist_ok=True)

        out_path = os.path.join(out_subdir, fname)

        try:
            with Image.open(src_path) as img:
                if keep_aspect_ratio:
                    # Resize preserving aspect ratio, pad remainder
                    img.thumbnail((target_w, target_h), Image.LANCZOS)
                    padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                    paste_x = (target_w - img.width) // 2
                    paste_y = (target_h - img.height) // 2
                    # Convert to RGB if necessary for consistent pasting
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    padded.paste(img, (paste_x, paste_y))
                    padded.save(out_path)
                else:
                    # Stretch to exact size
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    resized = img.resize((target_w, target_h), Image.LANCZOS)
                    resized.save(out_path)

                resized_records.append({
                    "file_path": out_path,
                    "label": row.get("label", "unknown"),
                    "width": target_w,
                    "height": target_h,
                })
                resized_count += 1

        except Exception as e:
            print(f"  [WARN] Could not resize {src_path}: {e}")
            error_count += 1

    if not resized_records:
        raise ValueError("No images were successfully resized")

    out_metadata = pd.DataFrame(resized_records)

    os.makedirs(os.path.dirname(outputs["metadata"]) or ".", exist_ok=True)
    out_metadata.to_csv(outputs["metadata"], index=False)

    return (
        f"resize_images: {resized_count} images resized to {target_size}, "
        f"{error_count} errors, keep_aspect_ratio={keep_aspect_ratio}"
    )


# =============================================================================
# IMAGE NORMALIZATION
# =============================================================================

@contract(
    inputs={
        "metadata": {"format": "csv", "required": True},
        "image_dir": {"format": "csv", "required": True},
    },
    outputs={
        "X": {"format": "pickle"},
        "y": {"format": "pickle"},
    },
    description="Load images as numpy arrays and normalize pixel values",
    tags=["image", "normalization", "preprocessing"],
)
def normalize_images(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    method: str = "scale",
    target_size: tuple = (224, 224),
) -> str:
    """
    Load images as numpy arrays and normalize pixel values.

    Parameters
    ----------
    inputs : dict
        metadata : str - Path to metadata CSV with 'file_path' and 'label' columns.
        image_dir : str - Root directory containing images (used for reference).
    outputs : dict
        X : str - Output pickle path for the numpy array of shape (N, H, W, 3).
        y : str - Output pickle path for the label array of shape (N,).
    method : str
        Normalization method. Default: "scale".
        - "scale": Divide by 255.0 to get values in [0, 1].
        - "standardize": Z-score normalization (subtract mean, divide by std).
    target_size : tuple of (int, int)
        Target (width, height) for loading. Images will be resized on load.
        Default: (224, 224).
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image services. Install with: pip install Pillow"
        )

    if method not in ("scale", "standardize"):
        raise ValueError(
            f"Unknown normalization method '{method}'. Supported: 'scale', 'standardize'"
        )

    metadata_df = pd.read_csv(inputs["metadata"])

    if "file_path" not in metadata_df.columns:
        raise ValueError("Metadata CSV must contain a 'file_path' column")

    target_w, target_h = target_size
    images = []
    labels = []
    load_errors = 0

    for idx, row in metadata_df.iterrows():
        file_path = row["file_path"]
        try:
            with Image.open(file_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((target_w, target_h), Image.LANCZOS)
                arr = np.array(img, dtype=np.float64)
                images.append(arr)
                labels.append(row.get("label", "unknown"))
        except Exception as e:
            print(f"  [WARN] Could not load {file_path}: {e}")
            load_errors += 1

    if not images:
        raise ValueError("No images were successfully loaded for normalization")

    X = np.stack(images, axis=0)  # Shape: (N, H, W, 3)

    # Normalize
    if method == "scale":
        X = X / 255.0
    elif method == "standardize":
        mean = X.mean()
        std = X.std()
        if std == 0:
            std = 1.0
        X = (X - mean) / std

    y = np.array(labels)

    os.makedirs(os.path.dirname(outputs["X"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["y"]) or ".", exist_ok=True)
    with open(outputs["X"], "wb") as f:
        pickle.dump(X, f)
    with open(outputs["y"], "wb") as f:
        pickle.dump(y, f)

    return (
        f"normalize_images: {X.shape[0]} images loaded as {X.shape}, "
        f"method='{method}', dtype={X.dtype}, "
        f"range=[{X.min():.4f}, {X.max():.4f}], "
        f"{load_errors} load errors"
    )


# =============================================================================
# IMAGE AUGMENTATION
# =============================================================================

@contract(
    inputs={
        "metadata": {"format": "csv", "required": True},
        "image_dir": {"format": "csv", "required": True},
    },
    outputs={
        "augmented_dir": {"format": "csv"},
        "metadata": {"format": "csv"},
    },
    description="Create augmented copies of images with flips, rotations, and brightness adjustments",
    tags=["image", "augmentation", "preprocessing"],
)
def augment_images(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    flip_horizontal: bool = True,
    flip_vertical: bool = False,
    rotate_degrees: List[int] = None,
    brightness_range: Optional[tuple] = None,
) -> str:
    """
    Create augmented copies of images.

    Parameters
    ----------
    inputs : dict
        metadata : str - Path to metadata CSV with 'file_path' and 'label' columns.
        image_dir : str - Root directory of source images.
    outputs : dict
        augmented_dir : str - Path to directory for augmented images.
        metadata : str - Updated metadata CSV including original and augmented images.
    flip_horizontal : bool
        Create horizontal flip copies. Default: True.
    flip_vertical : bool
        Create vertical flip copies. Default: False.
    rotate_degrees : list of int, optional
        Create rotated copies at specified degrees. Default: [90, 180, 270].
    brightness_range : tuple of (float, float), optional
        Create brightness-adjusted copies. Values are multipliers, e.g. (0.7, 1.3).
        If None, no brightness augmentation is applied.
    """
    try:
        from PIL import Image, ImageEnhance
    except ImportError:
        raise ImportError(
            "Pillow is required for image services. Install with: pip install Pillow"
        )

    if rotate_degrees is None:
        rotate_degrees = [90, 180, 270]

    metadata_df = pd.read_csv(inputs["metadata"])

    if "file_path" not in metadata_df.columns:
        raise ValueError("Metadata CSV must contain a 'file_path' column")

    augmented_dir = outputs["augmented_dir"]
    os.makedirs(augmented_dir, exist_ok=True)

    all_records = []
    augmented_count = 0
    error_count = 0

    for idx, row in metadata_df.iterrows():
        src_path = row["file_path"]
        label = row.get("label", "unknown")
        fname_base, fname_ext = os.path.splitext(os.path.basename(src_path))

        # Include the original image in the output records
        all_records.append({
            "file_path": src_path,
            "label": label,
            "width": row.get("width", 0),
            "height": row.get("height", 0),
            "augmentation": "original",
        })

        try:
            with Image.open(src_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                orig_w, orig_h = img.size

                # --- Horizontal flip ---
                if flip_horizontal:
                    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                    out_name = f"{fname_base}_hflip{fname_ext}"
                    out_path = os.path.join(augmented_dir, out_name)
                    flipped.save(out_path)
                    all_records.append({
                        "file_path": out_path,
                        "label": label,
                        "width": orig_w,
                        "height": orig_h,
                        "augmentation": "horizontal_flip",
                    })
                    augmented_count += 1

                # --- Vertical flip ---
                if flip_vertical:
                    flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
                    out_name = f"{fname_base}_vflip{fname_ext}"
                    out_path = os.path.join(augmented_dir, out_name)
                    flipped.save(out_path)
                    all_records.append({
                        "file_path": out_path,
                        "label": label,
                        "width": orig_w,
                        "height": orig_h,
                        "augmentation": "vertical_flip",
                    })
                    augmented_count += 1

                # --- Rotations ---
                for deg in rotate_degrees:
                    rotated = img.rotate(deg, expand=False)
                    out_name = f"{fname_base}_rot{deg}{fname_ext}"
                    out_path = os.path.join(augmented_dir, out_name)
                    rotated.save(out_path)
                    all_records.append({
                        "file_path": out_path,
                        "label": label,
                        "width": orig_w,
                        "height": orig_h,
                        "augmentation": f"rotate_{deg}",
                    })
                    augmented_count += 1

                # --- Brightness adjustments ---
                if brightness_range is not None:
                    low, high = brightness_range
                    enhancer = ImageEnhance.Brightness(img)

                    # Create two variants: one darker, one brighter
                    for factor, suffix in [(low, "dark"), (high, "bright")]:
                        adjusted = enhancer.enhance(factor)
                        out_name = f"{fname_base}_{suffix}{fname_ext}"
                        out_path = os.path.join(augmented_dir, out_name)
                        adjusted.save(out_path)
                        all_records.append({
                            "file_path": out_path,
                            "label": label,
                            "width": orig_w,
                            "height": orig_h,
                            "augmentation": f"brightness_{suffix}",
                        })
                        augmented_count += 1

        except Exception as e:
            print(f"  [WARN] Could not augment {src_path}: {e}")
            error_count += 1

    out_metadata = pd.DataFrame(all_records)

    os.makedirs(os.path.dirname(outputs["metadata"]) or ".", exist_ok=True)
    out_metadata.to_csv(outputs["metadata"], index=False)

    n_originals = len(metadata_df)
    return (
        f"augment_images: {n_originals} originals, "
        f"{augmented_count} augmented copies created, "
        f"{len(all_records)} total entries, "
        f"{error_count} errors"
    )


# =============================================================================
# IMAGE CLASSIFIER TRAINING
# =============================================================================

@contract(
    inputs={
        "X": {"format": "pickle", "required": True},
        "y": {"format": "pickle", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train an image classifier (SVM, Random Forest, or Logistic Regression) on flattened image features",
    tags=["image", "classification", "training", "modeling"],
)
def train_image_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    model_type: str = "svm",
    random_state: int = 42,
) -> str:
    """
    Train a classifier on image features (flattened pixel arrays).

    Parameters
    ----------
    inputs : dict
        X : str - Path to pickle file containing numpy array of shape (N, H, W, C).
        y : str - Path to pickle file containing label array of shape (N,).
    outputs : dict
        model : str - Output pickle path for the trained model.
        metrics : str - Output JSON path for evaluation metrics.
    model_type : str
        Classifier type. Default: "svm".
        - "svm": Support Vector Machine (LinearSVC).
        - "random_forest": Random Forest Classifier.
        - "logistic": Logistic Regression.
    random_state : int
        Random seed for reproducibility. Default: 42.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    # Load data
    with open(inputs["X"], "rb") as f:
        X = pickle.load(f)
    with open(inputs["y"], "rb") as f:
        y = pickle.load(f)

    # Flatten images: (N, H, W, C) -> (N, H*W*C)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_flat, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
    )

    # Select and train model
    if model_type == "svm":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(random_state=random_state, max_iter=2000)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            random_state=random_state, max_iter=1000, n_jobs=-1
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: 'svm', 'random_forest', 'logistic'"
        )

    clf.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = clf.predict(X_val)
    accuracy = float(accuracy_score(y_val, y_pred))
    report = classification_report(
        y_val, y_pred, target_names=le.classes_.tolist(), output_dict=True
    )

    metrics = {
        "model_type": model_type,
        "accuracy": accuracy,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X_flat.shape[1],
        "n_classes": len(le.classes_),
        "classes": le.classes_.tolist(),
        "classification_report": report,
    }

    # Bundle model with label encoder for prediction
    model_bundle = {
        "model": clf,
        "label_encoder": le,
        "model_type": model_type,
        "input_shape": X.shape[1:],
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_bundle, f)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return (
        f"train_image_classifier: model_type='{model_type}', "
        f"accuracy={accuracy:.4f}, "
        f"train={len(X_train)}, val={len(X_val)}, "
        f"features={X_flat.shape[1]}, classes={len(le.classes_)}"
    )


# =============================================================================
# IMAGE CLASSIFIER PREDICTION
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "X_test": {"format": "pickle", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Predict image classes using a trained classifier",
    tags=["image", "classification", "inference", "prediction"],
)
def predict_image_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "image_id",
    prediction_column: str = "label",
) -> str:
    """
    Predict classes for test images using a trained image classifier.

    Parameters
    ----------
    inputs : dict
        model : str - Path to pickle file containing the model bundle
                (model, label_encoder, model_type, input_shape).
        X_test : str - Path to pickle file containing test images as numpy array
                 of shape (N, H, W, C).
    outputs : dict
        predictions : str - Output CSV path with id and predicted label columns.
    id_column : str
        Name of the ID column in the output CSV. Default: "image_id".
    prediction_column : str
        Name of the prediction column in the output CSV. Default: "label".
    """
    # Load model bundle
    with open(inputs["model"], "rb") as f:
        model_bundle = pickle.load(f)

    clf = model_bundle["model"]
    le = model_bundle["label_encoder"]
    expected_shape = model_bundle.get("input_shape", None)

    # Load test data
    with open(inputs["X_test"], "rb") as f:
        X_test = pickle.load(f)

    # Validate shape compatibility
    if expected_shape is not None and X_test.shape[1:] != expected_shape:
        raise ValueError(
            f"Test image shape {X_test.shape[1:]} does not match "
            f"training shape {expected_shape}. "
            f"Ensure images are resized to the same dimensions."
        )

    # Flatten images: (N, H, W, C) -> (N, H*W*C)
    n_samples = X_test.shape[0]
    X_flat = X_test.reshape(n_samples, -1)

    # Predict
    y_pred_encoded = clf.predict(X_flat)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    # Build output DataFrame
    pred_df = pd.DataFrame({
        id_column: list(range(n_samples)),
        prediction_column: y_pred_labels,
    })

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    pred_df.to_csv(outputs["predictions"], index=False)

    n_unique = len(set(y_pred_labels))
    return (
        f"predict_image_classifier: {n_samples} predictions, "
        f"{n_unique} unique classes predicted"
    )


# =============================================================================
# PREPARE MULTILABEL IMAGE DATA
# =============================================================================

@contract(
    inputs={
        "labels_csv": {"format": "csv", "required": True},
        "image_dir": {"format": "csv", "required": True},
    },
    outputs={
        "X": {"format": "pickle"},
        "y": {"format": "pickle"},
    },
    description="Load images and multi-label targets into pickle files for training",
    tags=["image", "loading", "preprocessing", "multilabel"],
)
def prepare_multilabel_images(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "image_id",
    target_columns: List[str] = None,
    target_size: tuple = (64, 64),
    image_extension: str = ".jpg",
) -> str:
    """
    Load images and multi-label targets from CSV into pickle files.

    This service prepares data for train_multilabel_image_classifier.

    Parameters
    ----------
    inputs : dict
        labels_csv : str - Path to CSV with image IDs and one-hot label columns.
        image_dir : str - Path to directory containing images.
    outputs : dict
        X : str - Output pickle path for image array of shape (N, H, W, C).
        y : str - Output pickle path for label array of shape (N, num_classes).
    id_column : str
        Column name containing image IDs (without extension). Default: "image_id".
    target_columns : list of str, optional
        Column names for target classes. If None, uses all columns except id_column.
    target_size : tuple of (int, int)
        Resize images to (width, height). Default: (64, 64). Smaller = faster training.
    image_extension : str
        File extension for images. Default: ".jpg".
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image services. Install with: pip install Pillow"
        )

    # Load labels CSV
    labels_df = pd.read_csv(inputs["labels_csv"])
    image_dir = inputs["image_dir"]

    if id_column not in labels_df.columns:
        raise ValueError(f"id_column '{id_column}' not found. Columns: {list(labels_df.columns)}")

    # Determine target columns
    if target_columns is None:
        target_columns = [c for c in labels_df.columns if c != id_column]

    # Validate target columns exist
    missing = [c for c in target_columns if c not in labels_df.columns]
    if missing:
        raise ValueError(f"Target columns not found: {missing}")

    # Load images and labels
    X_list = []
    y_list = []
    loaded_count = 0
    error_count = 0

    for idx, row in labels_df.iterrows():
        image_id = str(row[id_column])
        image_path = os.path.join(image_dir, image_id + image_extension)

        if not os.path.exists(image_path):
            print(f"  [WARN] Image not found: {image_path}")
            error_count += 1
            continue

        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                X_list.append(img_array)

            # Extract label values
            label_vals = [float(row[c]) for c in target_columns]
            y_list.append(label_vals)
            loaded_count += 1

        except Exception as e:
            print(f"  [WARN] Could not process {image_path}: {e}")
            error_count += 1

    if loaded_count == 0:
        raise ValueError(f"No images loaded from {image_dir}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Save to pickle
    os.makedirs(os.path.dirname(outputs["X"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["y"]) or ".", exist_ok=True)

    with open(outputs["X"], "wb") as f:
        pickle.dump(X, f)
    with open(outputs["y"], "wb") as f:
        pickle.dump(y, f)

    return (
        f"prepare_multilabel_images: {loaded_count} images loaded, "
        f"shape X={X.shape}, y={y.shape}, "
        f"target_columns={target_columns}, "
        f"{error_count} errors"
    )


# =============================================================================
# LOAD TEST IMAGES
# =============================================================================

@contract(
    inputs={
        "test_csv": {"format": "csv", "required": True},
        "image_dir": {"format": "csv", "required": True},
    },
    outputs={
        "X_test": {"format": "pickle"},
        "test_ids": {"format": "csv"},
    },
    description="Load test images into pickle file for prediction",
    tags=["image", "loading", "preprocessing", "inference"],
)
def prepare_test_images(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "image_id",
    target_size: tuple = (64, 64),
    image_extension: str = ".jpg",
) -> str:
    """
    Load test images into pickle file for prediction.

    Parameters
    ----------
    inputs : dict
        test_csv : str - Path to CSV with test image IDs.
        image_dir : str - Path to directory containing images.
    outputs : dict
        X_test : str - Output pickle path for test image array.
        test_ids : str - Output CSV with test IDs (for prediction service).
    id_column : str
        Column name containing image IDs. Default: "image_id".
    target_size : tuple of (int, int)
        Resize images to (width, height). Must match training size.
    image_extension : str
        File extension for images. Default: ".jpg".
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image services. Install with: pip install Pillow"
        )

    # Load test CSV
    test_df = pd.read_csv(inputs["test_csv"])
    image_dir = inputs["image_dir"]

    if id_column not in test_df.columns:
        raise ValueError(f"id_column '{id_column}' not found. Columns: {list(test_df.columns)}")

    # Load images
    X_list = []
    valid_ids = []
    error_count = 0

    for idx, row in test_df.iterrows():
        image_id = str(row[id_column])
        image_path = os.path.join(image_dir, image_id + image_extension)

        if not os.path.exists(image_path):
            print(f"  [WARN] Image not found: {image_path}")
            error_count += 1
            continue

        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                X_list.append(img_array)
                valid_ids.append(image_id)

        except Exception as e:
            print(f"  [WARN] Could not process {image_path}: {e}")
            error_count += 1

    if len(X_list) == 0:
        raise ValueError(f"No test images loaded from {image_dir}")

    X_test = np.array(X_list, dtype=np.float32)

    # Save outputs
    os.makedirs(os.path.dirname(outputs["X_test"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["test_ids"]) or ".", exist_ok=True)

    with open(outputs["X_test"], "wb") as f:
        pickle.dump(X_test, f)

    ids_df = pd.DataFrame({id_column: valid_ids})
    ids_df.to_csv(outputs["test_ids"], index=False)

    return (
        f"prepare_test_images: {len(X_list)} images loaded, "
        f"shape={X_test.shape}, "
        f"{error_count} errors"
    )


# MULTI-LABEL IMAGE CLASSIFIER TRAINING
# =============================================================================

@contract(
    inputs={
        "X": {"format": "pickle", "required": True},
        "y": {"format": "pickle", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train a multi-label image classifier with probability outputs",
    tags=["image", "classification", "multilabel", "training", "modeling"],
)
def train_multilabel_image_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    model_type: str = "random_forest",
    class_names: List[str] = None,
    n_estimators: int = 100,
    random_state: int = 42,
) -> str:
    """
    Train a multi-output classifier for multi-label image classification.

    This service is designed for competitions like Plant Pathology where
    each image has 4 class probabilities that sum to 1.

    Parameters
    ----------
    inputs : dict
        X : str - Path to pickle file containing numpy array of shape (N, H, W, C).
        y : str - Path to pickle file containing label array of shape (N, num_classes).
            Labels should be one-hot encoded or multi-label binary matrix.
    outputs : dict
        model : str - Output pickle path for the trained model.
        metrics : str - Output JSON path for evaluation metrics.
    model_type : str
        Classifier type. Default: "random_forest".
        - "random_forest": Random Forest with predict_proba.
        - "logistic": Logistic Regression with predict_proba.
    class_names : list of str, optional
        Names of the classes for output columns. 
        Default: ["class_0", "class_1", ...].
    n_estimators : int
        Number of trees for random forest. Default: 100.
    random_state : int
        Random seed for reproducibility. Default: 42.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.multioutput import MultiOutputClassifier

    # Load data
    with open(inputs["X"], "rb") as f:
        X = pickle.load(f)
    with open(inputs["y"], "rb") as f:
        y = pickle.load(f)

    # Flatten images: (N, H, W, C) -> (N, H*W*C)
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Ensure y is 2D
    if y.ndim == 1:
        raise ValueError(
            "y must be 2D array of shape (N, num_classes) for multi-label classification"
        )

    n_classes = y.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    
    # For multi-class (one class per sample), convert one-hot to class index
    # Check if y is one-hot encoded (each row sums to 1 and has exactly one 1)
    is_one_hot = np.allclose(y.sum(axis=1), 1.0) and np.allclose(y.max(axis=1), 1.0)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_flat, y, test_size=0.2, random_state=random_state
    )

    # Select and train model
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        if is_one_hot:
            # Single classifier with predict_proba
            y_train_idx = np.argmax(y_train, axis=1)
            y_val_idx = np.argmax(y_val, axis=1)
            clf = RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state, n_jobs=-1
            )
            clf.fit(X_train, y_train_idx)
            y_pred_proba = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)
            accuracy = float(accuracy_score(y_val_idx, y_pred))
            # Compute ROC AUC
            try:
                roc_auc = float(roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='macro'))
            except:
                roc_auc = 0.0
        else:
            # Multi-output classifier for true multi-label
            base_clf = RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state, n_jobs=-1
            )
            clf = MultiOutputClassifier(base_clf)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            accuracy = float(accuracy_score(y_val, y_pred))
            roc_auc = 0.0
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        if is_one_hot:
            y_train_idx = np.argmax(y_train, axis=1)
            y_val_idx = np.argmax(y_val, axis=1)
            clf = LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1)
            clf.fit(X_train, y_train_idx)
            y_pred_proba = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)
            accuracy = float(accuracy_score(y_val_idx, y_pred))
            try:
                roc_auc = float(roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='macro'))
            except:
                roc_auc = 0.0
        else:
            base_clf = LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1)
            clf = MultiOutputClassifier(base_clf)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            accuracy = float(accuracy_score(y_val, y_pred))
            roc_auc = 0.0
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: 'random_forest', 'logistic'"
        )

    metrics = {
        "model_type": model_type,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": X_flat.shape[1],
        "n_classes": n_classes,
        "class_names": class_names,
        "is_one_hot": is_one_hot,
    }

    # Bundle model with metadata
    model_bundle = {
        "model": clf,
        "model_type": model_type,
        "input_shape": X.shape[1:],
        "class_names": class_names,
        "is_one_hot": is_one_hot,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_bundle, f)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return (
        f"train_multilabel_image_classifier: model_type='{model_type}', "
        f"accuracy={accuracy:.4f}, roc_auc={roc_auc:.4f}, "
        f"train={len(X_train)}, val={len(X_val)}, "
        f"classes={n_classes}"
    )


# =============================================================================
# MULTI-LABEL IMAGE PROBABILITY PREDICTION
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "X_test": {"format": "pickle", "required": True},
        "test_ids": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Predict class probabilities for multi-label image classification",
    tags=["image", "classification", "multilabel", "inference", "prediction", "probabilities"],
)
def predict_image_probabilities(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "image_id",
) -> str:
    """
    Predict class probabilities for test images using a trained multi-label classifier.

    Output format matches submissions requiring per-class probability columns.

    Parameters
    ----------
    inputs : dict
        model : str - Path to pickle file containing the model bundle.
        X_test : str - Path to pickle file containing test images as numpy array.
        test_ids : str - Path to CSV with test image IDs.
    outputs : dict
        predictions : str - Output CSV path with id and probability columns.
    id_column : str
        Name of the ID column in the output CSV. Default: "image_id".
    """
    # Load model bundle
    with open(inputs["model"], "rb") as f:
        model_bundle = pickle.load(f)

    clf = model_bundle["model"]
    expected_shape = model_bundle.get("input_shape", None)
    class_names = model_bundle.get("class_names", None)
    is_one_hot = model_bundle.get("is_one_hot", True)

    # Load test data
    with open(inputs["X_test"], "rb") as f:
        X_test = pickle.load(f)

    # Load test IDs
    test_ids_df = pd.read_csv(inputs["test_ids"])
    if id_column in test_ids_df.columns:
        test_ids = test_ids_df[id_column].tolist()
    else:
        # Try first column
        test_ids = test_ids_df.iloc[:, 0].tolist()

    # Validate shape compatibility
    if expected_shape is not None and X_test.shape[1:] != expected_shape:
        raise ValueError(
            f"Test image shape {X_test.shape[1:]} does not match "
            f"training shape {expected_shape}. "
            f"Ensure images are resized to the same dimensions."
        )

    # Flatten images: (N, H, W, C) -> (N, H*W*C)
    n_samples = X_test.shape[0]
    X_flat = X_test.reshape(n_samples, -1)

    # Predict probabilities
    if is_one_hot and hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_flat)
    else:
        # MultiOutputClassifier case
        if hasattr(clf, "estimators_"):
            # Get proba from each estimator
            proba_list = []
            for est in clf.estimators_:
                if hasattr(est, "predict_proba"):
                    proba_list.append(est.predict_proba(X_flat)[:, 1:2])
                else:
                    proba_list.append(est.predict(X_flat).reshape(-1, 1))
            proba = np.hstack(proba_list)
        else:
            # Fall back to predict
            proba = clf.predict(X_flat)

    # Build output DataFrame
    if class_names is None:
        class_names = [f"class_{i}" for i in range(proba.shape[1])]

    pred_df = pd.DataFrame(proba, columns=class_names)
    pred_df.insert(0, id_column, test_ids[:n_samples])

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    pred_df.to_csv(outputs["predictions"], index=False)

    return (
        f"predict_image_probabilities: {n_samples} predictions, "
        f"{len(class_names)} classes: {class_names}"
    )


# =============================================================================
# LOAD LABELED IMAGES (single-label, with sampling support)
# =============================================================================

@contract(
    inputs={
        "labels_csv": {"format": "csv", "required": True},
    },
    outputs={
        "X": {"format": "pickle"},
        "y": {"format": "pickle"},
        "metadata": {"format": "json"},
    },
    description="Load images with a single label column into numpy arrays, with optional sampling",
    tags=["image", "loading", "preprocessing", "generic"],
    version="1.0.0",
)
def load_labeled_images(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_dir: str = "",
    id_column: str = "id",
    label_column: str = "label",
    image_extension: str = ".tif",
    target_size: int = 64,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> str:
    """
    Load images and labels into numpy arrays for classification training.

    Reads image files from a directory, resizes to target_size x target_size,
    normalizes pixel values to [0,1], and saves as pickle files.
    Supports optional random sampling for faster iteration.

    Parameters:
        image_dir: Directory containing image files.  If relative or empty,
                   auto-discovered from the labels_csv parent directory.
        id_column: Column containing image IDs (filename without extension).
        label_column: Column containing labels.
        image_extension: File extension for images (e.g., .tif, .png, .jpg).
        target_size: Resize images to target_size x target_size pixels.
        sample_size: If set, randomly sample N images (for faster iteration).
        random_state: Random seed for sampling reproducibility.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required. Install with: pip install Pillow")

    labels_df = pd.read_csv(inputs["labels_csv"])

    if sample_size and sample_size < len(labels_df):
        labels_df = labels_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # Auto-discover image directory relative to labels_csv
    if not image_dir or not os.path.isabs(image_dir):
        data_dir = os.path.dirname(inputs["labels_csv"])
        if image_dir:
            candidate = os.path.join(data_dir, image_dir)
            if os.path.exists(candidate):
                image_dir = candidate
        else:
            for d in ["train", "images"]:
                candidate = os.path.join(data_dir, d)
                if os.path.exists(candidate):
                    image_dir = candidate
                    break

    X_list, y_list, ids_list = [], [], []
    errors = 0

    for _, row in labels_df.iterrows():
        img_id = str(row[id_column])
        img_path = os.path.join(image_dir, img_id + image_extension)

        if not os.path.exists(img_path):
            errors += 1
            continue

        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_resized = img.resize((target_size, target_size), Image.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                X_list.append(img_array)
                y_list.append(row[label_column])
                ids_list.append(img_id)
        except Exception:
            errors += 1

    if not X_list:
        raise ValueError(f"No images loaded from {image_dir}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    for key in outputs:
        os.makedirs(os.path.dirname(outputs[key]) or ".", exist_ok=True)

    with open(outputs["X"], "wb") as f:
        pickle.dump(X, f, protocol=4)
    with open(outputs["y"], "wb") as f:
        pickle.dump(y, f, protocol=4)
    with open(outputs["metadata"], "w") as f:
        json.dump({
            "n_samples": len(X_list),
            "shape": list(X.shape),
            "image_dir": image_dir,
            "target_size": target_size,
            "errors": errors,
            "label_column": label_column,
            "n_unique_labels": int(len(set(y_list))),
        }, f, indent=2)

    return (
        f"load_labeled_images: {len(X_list)} images, shape={X.shape}, "
        f"{errors} errors"
    )


# =============================================================================
# TRAIN CNN IMAGE CLASSIFIER (PyTorch, generic)
# =============================================================================

@contract(
    inputs={
        "X": {"format": "pickle", "required": True},
        "y": {"format": "pickle", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train a CNN image classifier using PyTorch (binary or multiclass)",
    tags=["image", "classification", "training", "modeling", "cnn", "pytorch", "generic"],
    version="1.0.0",
)
def train_cnn_image_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_classes: int = 2,
    n_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
    dropout: float = 0.3,
    conv_channels: List[int] = None,
    fc_size: int = 256,
    validation_split: float = 0.2,
    random_state: int = 42,
) -> str:
    """
    Train a CNN image classifier using PyTorch.

    Architecture: configurable Conv2D blocks with BatchNorm + ReLU + MaxPool,
    followed by Dropout + FC layers. Supports binary and multiclass.

    Parameters:
        n_classes: Number of output classes.  2 = binary (BCE loss with
                   sigmoid), >2 = multiclass (CE loss with softmax).
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam optimizer learning rate.
        dropout: Dropout rate after conv blocks and between FC layers.
        conv_channels: List of output channels per conv block.
                       Default: [32, 64, 128] (3 blocks).
        fc_size: Number of units in the hidden FC layer.
        validation_split: Fraction of data for validation.
        random_state: Random seed for reproducibility.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score, accuracy_score

    if conv_channels is None:
        conv_channels = [32, 64, 128]

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    with open(inputs["X"], "rb") as f:
        X = pickle.load(f)
    with open(inputs["y"], "rb") as f:
        y = pickle.load(f)

    # Normalize labels to contiguous class indices
    y_arr = np.array(y)
    if y_arr.ndim > 1:
        y_arr = np.argmax(y_arr, axis=1)

    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y_arr)
    inferred_classes = len(label_encoder.classes_)

    if n_classes is None or n_classes <= 0:
        n_classes = inferred_classes
    elif n_classes != inferred_classes:
        raise ValueError(
            f"n_classes={n_classes} does not match inferred classes={inferred_classes}"
        )

    is_binary = (n_classes == 2)

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_int, test_size=validation_split, random_state=random_state, stratify=y_int
    )

    # PyTorch tensors (NCHW)
    X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val_t = torch.FloatTensor(X_val).permute(0, 3, 1, 2)

    if is_binary:
        y_train_t = torch.FloatTensor(y_train.astype(np.float32))
        y_val_t = torch.FloatTensor(y_val.astype(np.float32))
    else:
        y_train_t = torch.LongTensor(y_train)
        y_val_t = torch.LongTensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )

    img_size = X_train.shape[1]
    in_channels = X_train.shape[3] if X_train.ndim == 4 else 1

    # --- Build CNN ---
    class GenericCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            ch_in = in_channels
            for ch_out in conv_channels:
                layers += [
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                ]
                ch_in = ch_out
            self.features = nn.Sequential(*layers)

            n_pools = len(conv_channels)
            flat_size = conv_channels[-1] * (img_size // (2 ** n_pools)) ** 2

            out_units = 1 if is_binary else n_classes
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(),
                nn.Linear(flat_size, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_size, out_units),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            if is_binary:
                x = x.squeeze(-1)
            return x

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = GenericCNN().to(device)

    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_score = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = criterion(val_logits, y_val_t.to(device)).item()

            if is_binary:
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_auc = float(roc_auc_score(y_val, val_probs))
                score = val_auc
                score_name = "auc"
            else:
                val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                val_acc = float(accuracy_score(y_val, val_preds))
                score = val_acc
                score_name = "accuracy"

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_{score_name}={score:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model_data = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "img_size": img_size,
        "in_channels": in_channels,
        "n_classes": n_classes,
        "conv_channels": conv_channels,
        "fc_size": fc_size,
        "dropout": dropout,
        "is_binary": is_binary,
        "model_type": "generic_cnn",
        "class_names": list(label_encoder.classes_),
    }

    for key in outputs:
        os.makedirs(os.path.dirname(outputs[key]) or ".", exist_ok=True)

    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f, protocol=4)

    metrics = {
        "model_type": "cnn_pytorch",
        "n_classes": n_classes,
        "is_binary": is_binary,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "conv_channels": conv_channels,
        "fc_size": fc_size,
        "dropout": dropout,
        "img_size": img_size,
        f"best_val_{score_name}": best_score,
        "device": str(device),
    }

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_cnn_image_classifier: best_val_{score_name}={best_score:.4f}, {len(X_train)} train, {len(X_val)} val"


# =============================================================================
# PREDICT WITH CNN IMAGE CLASSIFIER (PyTorch, generic)
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "X_test": {"format": "pickle", "required": True},
        "test_ids": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Generate predictions using a trained PyTorch CNN image classifier",
    tags=["image", "classification", "inference", "prediction", "cnn", "pytorch", "generic"],
    version="1.0.0",
)
def predict_cnn_image_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "label",
    batch_size: int = 256,
    output_probabilities: bool = False,
    class_names: List[str] = None,
) -> str:
    """
    Generate predictions using a trained CNN image classifier.

    For binary classification, outputs probabilities (for AUC-scored competitions).
    For multiclass, outputs predicted class indices by default. If
    output_probabilities=True, outputs per-class probabilities.

    Parameters:
        id_column: Name of the ID column in output.
        prediction_column: Name of the prediction column in output.
        batch_size: Batch size for inference.
        output_probabilities: If True and multiclass, output probability columns.
        class_names: Optional list of class names to order probability columns.
    """
    import torch
    import torch.nn as nn

    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    img_size = model_data["img_size"]
    in_channels = model_data.get("in_channels", 3)
    n_classes = model_data.get("n_classes", 2)
    conv_channels = model_data.get("conv_channels", [32, 64, 128])
    fc_size = model_data.get("fc_size", 256)
    dropout_rate = model_data.get("dropout", 0.3)
    is_binary = model_data.get("is_binary", n_classes == 2)
    model_class_names = model_data.get("class_names")

    # Rebuild architecture
    class GenericCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            ch_in = in_channels
            for ch_out in conv_channels:
                layers += [
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                ]
                ch_in = ch_out
            self.features = nn.Sequential(*layers)

            n_pools = len(conv_channels)
            flat_size = conv_channels[-1] * (img_size // (2 ** n_pools)) ** 2

            out_units = 1 if is_binary else n_classes
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Flatten(),
                nn.Linear(flat_size, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(fc_size, out_units),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            if is_binary:
                x = x.squeeze(-1)
            return x

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = GenericCNN().to(device)
    model.load_state_dict(model_data["state_dict"])
    model.eval()

    with open(inputs["X_test"], "rb") as f:
        X_test = pickle.load(f)

    test_ids_df = pd.read_csv(inputs["test_ids"])
    test_ids = test_ids_df[id_column].tolist() if id_column in test_ids_df.columns else test_ids_df.iloc[:, 0].tolist()

    X_test_t = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for start in range(0, len(X_test_t), batch_size):
            batch = X_test_t[start:start + batch_size].to(device)
            logits = model(batch)
            if is_binary:
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds.tolist())
            else:
                if output_probabilities:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)
                else:
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds.tolist())

    if is_binary or not output_probabilities:
        pred_df = pd.DataFrame({
            id_column: test_ids[:len(all_preds)],
            prediction_column: all_preds,
        })
    else:
        probs = np.vstack(all_probs) if all_probs else np.zeros((0, n_classes))
        column_names = class_names or model_class_names
        if column_names is None:
            column_names = [f"class_{i}" for i in range(probs.shape[1])]
        # Reorder probabilities if requested order differs from model order
        if model_class_names and column_names != model_class_names:
            try:
                idx_map = [model_class_names.index(name) for name in column_names]
            except ValueError as exc:
                raise ValueError(
                    "class_names must match model classes for reordering"
                ) from exc
            probs = probs[:, idx_map]
        pred_df = pd.DataFrame(probs, columns=column_names)
        pred_df.insert(0, id_column, test_ids[: len(probs)])

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    pred_df.to_csv(outputs["predictions"], index=False)

    return f"predict_cnn_image_classifier: {len(pred_df)} predictions"


# =============================================================================
# TRAIN PRETRAINED IMAGE CLASSIFIER (Transfer Learning, generic)
# =============================================================================

@contract(
    inputs={
        "labels_csv": {"format": "csv", "required": True},
        "image_dir": {"format": "csv", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train an image classifier via transfer learning on pretrained models (ViT, ResNet, EfficientNet)",
    tags=["image", "classification", "training", "transfer-learning", "pretrained", "generic"],
    version="1.0.0",
)
def train_pretrained_image_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    label_column: str = "label",
    image_extension: str = ".jpg",
    architecture: str = "vit_b_16",
    target_size: int = 224,
    n_epochs: int = 7,
    batch_size: int = 8,
    learning_rate: float = 0.009,
    momentum: float = 0.9,
    freeze_backbone: bool = True,
    validation_split: float = 0.1,
    random_state: int = 42,
    num_workers: int = 0,
    sample_size: Optional[int] = None,
    lr_step_size: int = 0,
    lr_gamma: float = 0.1,
    use_all_data: bool = False,
    augmentation: str = "basic",
) -> str:
    """
    Train an image classifier using transfer learning on a pretrained model.

    Loads images on-the-fly from a directory (memory-efficient). By default,
    freezes the backbone and trains only the classification head using SGD
    with momentum. Applies ImageNet normalization and optional augmentation.

    Based on winning Kaggle notebook patterns for fine-grained image
    classification (dog-breed, plant-pathology, etc.).

    Parameters:
        id_column: Column in labels_csv containing image filenames (no extension).
        label_column: Column in labels_csv containing class labels.
        image_extension: File extension for images (e.g., ".jpg", ".png").
        architecture: Pretrained model. Options: "vit_b_16", "vit_l_16",
                      "resnet50", "resnet101", "efficientnet_b0", "efficientnet_b3",
                      "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
                      "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
                      "convnextv2_atto", "convnextv2_femto", "convnextv2_pico", "convnextv2_nano",
                      "convnextv2_tiny", "convnextv2_base", "convnextv2_large", "convnextv2_huge".
        target_size: Resize images to target_size x target_size. Default: 224.
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: SGD learning rate.
        momentum: SGD momentum.
        freeze_backbone: If True, only train the classification head.
        validation_split: Fraction held out for validation.
        random_state: Random seed for reproducibility.
        num_workers: DataLoader workers. 0 = main process only.
        sample_size: If set, randomly sample N images for faster iteration.
        lr_step_size: StepLR step size (epochs). 0 = no scheduler.
        lr_gamma: StepLR gamma (multiply LR by this every step_size epochs).
        use_all_data: If True, train on all data without validation split (final submission).
        augmentation: "basic" (flip+rotation) or "advanced" (RandomResizedCrop+ColorJitter+flip).
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

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # --- Load labels ---
    labels_df = pd.read_csv(inputs["labels_csv"])
    image_dir = inputs["image_dir"]

    if sample_size and sample_size < len(labels_df):
        labels_df = labels_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_df["_label_idx"] = label_encoder.fit_transform(labels_df[label_column])
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    # Split
    if use_all_data:
        train_df = labels_df.reset_index(drop=True)
        val_df = None
    else:
        train_df, val_df = train_test_split(
            labels_df, test_size=validation_split, random_state=random_state,
            stratify=labels_df["_label_idx"]
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    # --- Transforms ---
    if augmentation == "advanced":
        train_transform = T.Compose([
            T.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(target_size),
            T.RandomHorizontalFlip(p=0.6),
            T.RandomRotation(degrees=30),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset ---
    class _ImageDataset(Dataset):
        def __init__(self, df, img_dir, id_col, ext, transform):
            self.df = df
            self.img_dir = img_dir
            self.id_col = id_col
            self.ext = ext
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_id = str(row[self.id_col])
            img_path = os.path.join(self.img_dir, img_id + self.ext)
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            label = int(row["_label_idx"])
            return img, label

    train_dataset = _ImageDataset(train_df, image_dir, id_column, image_extension, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if val_df is not None:
        val_dataset = _ImageDataset(val_df, image_dir, id_column, image_extension, val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)
    else:
        val_loader = None

    # --- Model ---
    # ConvNeXt V2 models use timm library (map user-friendly names to timm model names)
    TIMM_ARCHITECTURES = {
        "convnextv2_atto": "convnextv2_atto.fcmae_ft_in1k",
        "convnextv2_femto": "convnextv2_femto.fcmae_ft_in1k",
        "convnextv2_pico": "convnextv2_pico.fcmae_ft_in1k",
        "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
        "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in1k",
        "convnextv2_base": "convnextv2_base.fcmae_ft_in1k",
        "convnextv2_large": "convnextv2_large.fcmae_ft_in1k",
        "convnextv2_huge": "convnextv2_huge.fcmae_ft_in1k",
    }

    ARCH_MAP = {
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT, "heads.head"),
        "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights.DEFAULT, "heads.head"),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, "fc"),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier.1"),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT, "classifier.1"),
        "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT, "classifier.1"),
        "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.DEFAULT, "classifier.1"),
        "efficientnet_v2_l": (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.DEFAULT, "classifier.1"),
        # ConvNeXt models (V1)
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, "classifier.2"),
        "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT, "classifier.2"),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT, "classifier.2"),
        "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT, "classifier.2"),
    }

    # Handle timm-based ConvNeXt V2 models
    if architecture in TIMM_ARCHITECTURES:
        import timm
        timm_model_name = TIMM_ARCHITECTURES[architecture]
        print(f"  Loading pretrained {architecture} ({timm_model_name}) via timm...")
        model = timm.create_model(timm_model_name, pretrained=True, num_classes=n_classes)
        head_attr = "head.fc"  # timm ConvNeXt V2 head attribute
        in_features = model.head.fc.in_features
    elif architecture not in ARCH_MAP:
        all_options = list(ARCH_MAP.keys()) + list(TIMM_ARCHITECTURES.keys())
        raise ValueError(f"Unsupported architecture '{architecture}'. Options: {all_options}")
    else:
        model_fn, weights, head_attr = ARCH_MAP[architecture]
        print(f"  Loading pretrained {architecture}...")
        model = model_fn(weights=weights)

        # Replace classification head for torchvision models
        parts = head_attr.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old_head = getattr(parent, parts[-1])
        in_features = old_head.in_features
        new_head = nn.Linear(in_features, n_classes)
        setattr(parent, parts[-1], new_head)

    # Freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the classification head
        if architecture in TIMM_ARCHITECTURES:
            for param in model.head.fc.parameters():
                param.requires_grad = True
            new_head = model.head.fc
        else:
            for param in new_head.parameters():
                param.requires_grad = True
    else:
        # For timm models, get the head reference for optimizer
        if architecture in TIMM_ARCHITECTURES:
            new_head = model.head.fc

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)
    print(f"  Device: {device}, freeze_backbone={freeze_backbone}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, momentum=momentum,
    )

    scheduler = None
    if lr_step_size > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # --- Training ---
    best_val_loss = float("inf")
    best_state = None
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        model.train()
        running_loss, n_correct, n_total = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            n_correct += (torch.argmax(logits, dim=1) == batch_y).sum().item()
            n_total += len(batch_y)

        if scheduler:
            scheduler.step()

        train_loss = running_loss / n_total
        train_acc = n_correct / n_total

        if val_df is not None:
            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits = model(vx)
                    vloss = criterion(vlogits, vy)
                    val_loss += vloss.item() * len(vy)
                    val_correct += (torch.argmax(vlogits, dim=1) == vy).sum().item()
                    val_total += len(vy)

            val_loss /= val_total
            val_acc = val_correct / val_total

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            lr_str = f", lr={optimizer.param_groups[0]['lr']:.6f}" if scheduler else ""
            print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}{lr_str}")
        else:
            lr_str = f", lr={optimizer.param_groups[0]['lr']:.6f}" if scheduler else ""
            print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}{lr_str}")

    # Load best model (or use last if no validation)
    if best_state:
        model.load_state_dict(best_state)

    # --- Save ---
    model_data = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "architecture": architecture,
        "n_classes": n_classes,
        "class_names": class_names,
        "in_features": in_features,
        "head_attr": head_attr,
        "target_size": target_size,
        "model_type": "pretrained_transfer",
    }

    for key in outputs:
        os.makedirs(os.path.dirname(outputs[key]) or ".", exist_ok=True)

    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f, protocol=4)

    n_val = len(val_df) if val_df is not None else 0
    # Convert numpy types to native Python for JSON serialization
    class_names_json = [int(c) if isinstance(c, (np.integer, np.int64)) else str(c) for c in class_names]
    metrics = {
        "model_type": f"pretrained_{architecture}",
        "architecture": architecture,
        "n_classes": n_classes,
        "class_names": class_names_json,
        "n_train": len(train_df),
        "n_val": n_val,
        "n_epochs": n_epochs,
        "best_val_loss": float(best_val_loss) if best_val_loss < float("inf") else None,
        "best_val_acc": float(best_val_acc),
        "freeze_backbone": freeze_backbone,
        "use_all_data": use_all_data,
        "augmentation": augmentation,
        "device": str(device),
    }

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    if use_all_data:
        return (
            f"train_pretrained_image_classifier: arch={architecture}, "
            f"train={len(train_df)} (all data), classes={n_classes}"
        )
    return (
        f"train_pretrained_image_classifier: arch={architecture}, "
        f"best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc:.4f}, "
        f"train={len(train_df)}, val={n_val}, classes={n_classes}"
    )


# =============================================================================
# TRAIN PRETRAINED MULTILABEL IMAGE CLASSIFIER (Transfer Learning, generic)
# =============================================================================

@contract(
    inputs={
        "labels_csv": {"format": "csv", "required": True},
        "image_dir": {"format": "directory", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train a multilabel image classifier using transfer learning with pretrained ResNet",
    tags=["image", "classification", "multilabel", "training", "transfer-learning", "pretrained", "generic"],
    version="1.0.0",
)
def train_pretrained_multilabel_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "image_name",
    label_column: str = "tags",
    image_extension: str = ".jpg",
    architecture: str = "resnet18",
    target_size: int = 224,
    n_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
    validation_split: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
    lr_step_size: int = 4,
    lr_gamma: float = 0.1,
    dropout: float = 0.3,
) -> str:
    """
    Train a multilabel image classifier using transfer learning on pretrained ResNet.

    Designed for competitions like Planet Amazon where each image has multiple
    space-separated tags. Uses BCEWithLogitsLoss and optimizes per-class thresholds.

    Based on top Kaggle solutions achieving 0.93+ F2-score.

    Parameters:
        id_column: Column containing image filenames (no extension).
        label_column: Column containing space-separated class labels.
        image_extension: File extension for images (e.g., ".jpg", ".png").
        architecture: "resnet18" (fast) or "resnet50" (accurate).
        target_size: Resize images to target_size x target_size. Default: 224.
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        freeze_backbone: If True, only train the classification head.
        validation_split: Fraction held out for validation.
        random_state: Random seed for reproducibility.
        num_workers: DataLoader workers. 0 = main process only.
        lr_step_size: StepLR step size (epochs).
        lr_gamma: StepLR gamma (multiply LR by this every step_size epochs).
        dropout: Dropout rate for classification head.
    """
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import fbeta_score
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Device selection
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Using device: {device}")

    # --- Load labels ---
    labels_df = pd.read_csv(inputs["labels_csv"])
    image_dir = inputs["image_dir"]

    # Parse multilabel tags
    labels_df["_tags_list"] = labels_df[label_column].str.split(" ")

    # Encode multilabel
    encoder = MultiLabelBinarizer()
    labels_encoded = encoder.fit_transform(labels_df["_tags_list"])
    class_names = list(encoder.classes_)
    n_classes = len(class_names)

    print(f"  Classes ({n_classes}): {class_names}")

    # Split
    train_indices, val_indices = train_test_split(
        range(len(labels_df)),
        test_size=validation_split,
        random_state=random_state,
    )
    train_df = labels_df.iloc[train_indices].reset_index(drop=True)
    val_df = labels_df.iloc[val_indices].reset_index(drop=True)
    train_labels = labels_encoded[train_indices]
    val_labels = labels_encoded[val_indices]

    # --- Transforms ---
    train_transform = T.Compose([
        T.Resize(target_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset ---
    class _MultilabelDataset(Dataset):
        def __init__(self, df, labels, img_dir, id_col, ext, transform):
            self.df = df
            self.labels = labels
            self.img_dir = img_dir
            self.id_col = id_col
            self.ext = ext
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_id = str(row[self.id_col])
            img_path = os.path.join(self.img_dir, img_id + self.ext)
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            label = torch.FloatTensor(self.labels[idx])
            return img, label

    train_dataset = _MultilabelDataset(train_df, train_labels, image_dir, id_column, image_extension, train_transform)
    val_dataset = _MultilabelDataset(val_df, val_labels, image_dir, id_column, image_extension, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

    # --- Model ---
    if architecture == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = 2048
    else:
        model = models.resnet18(weights="IMAGENET1K_V1")
        in_features = 512

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace FC layer with multilabel head
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, n_classes),
        # No sigmoid - BCEWithLogitsLoss applies it
    )

    # Unfreeze FC layer
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # --- Training ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    best_val_f2 = 0.0
    best_state = None
    best_val_preds = None
    best_val_labels = None

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_preds_list = []
        val_labels_list = []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds_list.append(probs)
                val_labels_list.append(labels.cpu().numpy())
        val_loss /= len(val_loader)

        val_preds = np.vstack(val_preds_list)
        val_true = np.vstack(val_labels_list)
        val_binary = (val_preds > 0.2).astype(float)
        val_f2 = fbeta_score(val_true, val_binary, beta=2, average="samples")

        print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f2={val_f2:.4f}")

        if val_f2 > best_val_f2:
            best_val_f2 = val_f2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_preds = val_preds
            best_val_labels = val_true

        scheduler.step()

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Optimize thresholds
    thresholds = [0.2] * n_classes
    for class_idx in range(n_classes):
        best_thresh = 0.2
        best_f2 = 0.0
        for thresh in np.arange(0.1, 0.6, 0.05):
            test_thresh = thresholds.copy()
            test_thresh[class_idx] = thresh
            preds = (best_val_preds > test_thresh).astype(float)
            f2 = fbeta_score(best_val_labels, preds, beta=2, average="samples")
            if f2 > best_f2:
                best_f2 = f2
                best_thresh = thresh
        thresholds[class_idx] = best_thresh

    # Final F2 with optimized thresholds
    preds_optimal = (best_val_preds > thresholds).astype(float)
    f2_optimal = fbeta_score(best_val_labels, preds_optimal, beta=2, average="samples")
    print(f"  F2 with optimized thresholds: {f2_optimal:.4f}")

    # --- Save outputs ---
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)

    model_bundle = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "architecture": architecture,
        "n_classes": n_classes,
        "class_names": class_names,
        "in_features": in_features,
        "thresholds": thresholds,
        "target_size": target_size,
        "dropout": dropout,
    }
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_bundle, f)

    metrics = {
        "architecture": architecture,
        "n_classes": n_classes,
        "class_names": class_names,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_epochs": n_epochs,
        "best_val_f2": float(best_val_f2),
        "f2_with_optimal_thresholds": float(f2_optimal),
        "thresholds": thresholds,
        "freeze_backbone": freeze_backbone,
        "device": str(device),
    }
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return (
        f"train_pretrained_multilabel_classifier: arch={architecture}, "
        f"best_val_f2={best_val_f2:.4f}, f2_optimized={f2_optimal:.4f}, "
        f"train={len(train_df)}, val={len(val_df)}, classes={n_classes}"
    )


# =============================================================================
# PREDICT PRETRAINED MULTILABEL IMAGE CLASSIFIER (Transfer Learning, generic)
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_csv": {"format": "csv", "required": True},
        "image_dir": {"format": "directory", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Predict multilabel tags using a pretrained multilabel image classifier with TTA",
    tags=["image", "classification", "multilabel", "inference", "prediction", "transfer-learning", "pretrained", "generic"],
    version="1.0.0",
)
def predict_pretrained_multilabel_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "image_name",
    image_extension: str = ".jpg",
    batch_size: int = 64,
    use_tta: bool = True,
    n_tta: int = 6,
) -> str:
    """
    Predict multilabel tags using a trained multilabel classifier with TTA.

    Uses 6 TTA augmentations (rotations + flips) and averages predictions.
    Applies per-class thresholds from training to generate binary predictions.

    Parameters:
        id_column: Column containing image IDs.
        image_extension: File extension for test images.
        batch_size: Inference batch size.
        use_tta: Whether to use test-time augmentation.
        n_tta: Number of TTA augmentations (6 = all rotations + flips).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from sklearn.preprocessing import MultiLabelBinarizer
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models

    # Device selection
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # --- Load model ---
    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    architecture = model_data["architecture"]
    n_classes = model_data["n_classes"]
    class_names = model_data["class_names"]
    thresholds = model_data["thresholds"]
    target_size = model_data["target_size"]
    in_features = model_data["in_features"]
    dropout = model_data.get("dropout", 0.3)

    # Rebuild model
    if architecture == "resnet50":
        model = models.resnet50(weights=None)
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, n_classes),
    )
    model.load_state_dict(model_data["state_dict"])
    model = model.to(device)
    model.eval()

    # --- Load test data ---
    test_df = pd.read_csv(inputs["test_csv"])
    image_dir = inputs["image_dir"]

    # --- Transform ---
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Dataset with TTA ---
    class _TTADataset(Dataset):
        def __init__(self, df, img_dir, id_col, ext, transform, tta_idx=None):
            self.df = df
            self.img_dir = img_dir
            self.id_col = id_col
            self.ext = ext
            self.transform = transform
            self.tta_idx = tta_idx

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_id = str(row[self.id_col])
            img_path = os.path.join(self.img_dir, img_id + self.ext)
            img = Image.open(img_path).convert("RGB")

            # Apply TTA augmentation
            if self.tta_idx is not None:
                if self.tta_idx == 0:
                    img = img.rotate(90)
                elif self.tta_idx == 1:
                    img = img.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
                elif self.tta_idx == 2:
                    img = img.rotate(180)
                elif self.tta_idx == 3:
                    img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                elif self.tta_idx == 4:
                    img = img.rotate(270)
                elif self.tta_idx == 5:
                    img = img.rotate(270).transpose(Image.FLIP_LEFT_RIGHT)

            img = self.transform(img)
            return img, img_id

    # --- Prediction with TTA ---
    all_probs = []
    n_tta_actual = n_tta if use_tta else 1

    with torch.no_grad():
        for tta_idx in range(n_tta_actual):
            tta_val = tta_idx if use_tta else None
            dataset = _TTADataset(test_df, image_dir, id_column, image_extension, transform, tta_val)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            tta_probs = []
            for images, _ in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                tta_probs.append(probs)
            all_probs.append(np.vstack(tta_probs))
            print(f"  TTA {tta_idx+1}/{n_tta_actual} complete")

    # Average TTA predictions
    avg_probs = np.mean(all_probs, axis=0)

    # Apply thresholds
    preds_binary = (avg_probs > thresholds).astype(int)

    # Convert to tags
    encoder = MultiLabelBinarizer(classes=class_names)
    encoder.fit([class_names])
    predicted_tags = encoder.inverse_transform(preds_binary)
    tags_str = [" ".join(tags) if tags else "primary" for tags in predicted_tags]

    # Create output DataFrame
    result_df = pd.DataFrame({
        id_column: test_df[id_column],
        "tags": tags_str,
    })

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    result_df.to_csv(outputs["predictions"], index=False)

    return (
        f"predict_pretrained_multilabel_classifier: "
        f"n_images={len(test_df)}, tta={n_tta_actual}, classes={n_classes}"
    )


# =============================================================================
# PREDICT WITH PRETRAINED IMAGE CLASSIFIER (Transfer Learning, generic)
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_csv": {"format": "csv", "required": True},
        "image_dir": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Predict class probabilities using a pretrained image classifier",
    tags=["image", "classification", "inference", "prediction", "transfer-learning", "pretrained", "generic"],
    version="1.0.0",
)
def predict_pretrained_image_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    image_extension: str = ".jpg",
    target_size: int = 224,
    batch_size: int = 64,
) -> str:
    """
    Predict class probabilities using a trained pretrained image classifier.

    Loads test images on-the-fly from a directory and outputs per-class
    probability columns suitable for Kaggle submission.

    Parameters:
        id_column: Column in test_csv containing image IDs (no extension).
        image_extension: File extension for test images.
        target_size: Image resize target. Must match training size.
        batch_size: Inference batch size.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models

    # --- Load model ---
    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    architecture = model_data["architecture"]
    n_classes = model_data["n_classes"]
    class_names = model_data["class_names"]
    in_features = model_data["in_features"]
    head_attr = model_data["head_attr"]
    saved_target_size = model_data.get("target_size", target_size)

    # ConvNeXt V2 models use timm library
    TIMM_ARCHITECTURES = {
        "convnextv2_atto": "convnextv2_atto.fcmae_ft_in1k",
        "convnextv2_femto": "convnextv2_femto.fcmae_ft_in1k",
        "convnextv2_pico": "convnextv2_pico.fcmae_ft_in1k",
        "convnextv2_nano": "convnextv2_nano.fcmae_ft_in1k",
        "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in1k",
        "convnextv2_base": "convnextv2_base.fcmae_ft_in1k",
        "convnextv2_large": "convnextv2_large.fcmae_ft_in1k",
        "convnextv2_huge": "convnextv2_huge.fcmae_ft_in1k",
    }

    # Rebuild model
    ARCH_MAP = {
        "vit_b_16": models.vit_b_16,
        "vit_l_16": models.vit_l_16,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_v2_s": models.efficientnet_v2_s,
        "efficientnet_v2_m": models.efficientnet_v2_m,
        "efficientnet_v2_l": models.efficientnet_v2_l,
        # ConvNeXt models (V1)
        "convnext_tiny": models.convnext_tiny,
        "convnext_small": models.convnext_small,
        "convnext_base": models.convnext_base,
        "convnext_large": models.convnext_large,
    }

    # Handle timm-based ConvNeXt V2 models
    if architecture in TIMM_ARCHITECTURES:
        import timm
        timm_model_name = TIMM_ARCHITECTURES[architecture]
        model = timm.create_model(timm_model_name, pretrained=False, num_classes=n_classes)
    else:
        model = ARCH_MAP[architecture](weights=None)
        parts = head_attr.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        new_head = nn.Linear(in_features, n_classes)
        setattr(parent, parts[-1], new_head)

    model.load_state_dict(model_data["state_dict"])

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {architecture}, {n_classes} classes, device={device}")

    # --- Load test data ---
    test_df = pd.read_csv(inputs["test_csv"])
    image_dir = inputs["image_dir"]
    test_ids = test_df[id_column].tolist()

    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(saved_target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class _TestImageDataset(Dataset):
        def __init__(self, ids, img_dir, ext, transform):
            self.ids = ids
            self.img_dir = img_dir
            self.ext = ext
            self.transform = transform

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            img_id = str(self.ids[idx])
            img_path = os.path.join(self.img_dir, img_id + self.ext)
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img

    test_dataset = _TestImageDataset(test_ids, image_dir, image_extension, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Predict ---
    all_probs = []
    with torch.no_grad():
        for batch_X in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)

    # Build output DataFrame
    pred_df = pd.DataFrame(probs, columns=class_names)
    pred_df.insert(0, id_column, test_ids[:len(probs)])

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    pred_df.to_csv(outputs["predictions"], index=False)

    return (
        f"predict_pretrained_image_classifier: {len(pred_df)} predictions, "
        f"{n_classes} classes, arch={architecture}"
    )


# =============================================================================
# FORMAT SUBMISSION (generic, matches sample_submission format)
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True},
        "sample_submission": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Format predictions to match a Kaggle sample submission CSV",
    tags=["submission", "formatting", "kaggle", "generic"],
    version="1.0.0",
)
def format_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "label",
) -> str:
    """
    Format predictions to match the sample submission structure.

    Aligns column names and order to the sample submission template.
    Works for single- or multi-column prediction formats.

    Parameters:
        id_column: Name of the ID column in predictions.
        prediction_column: Name of the prediction column in predictions.
    """
    pred_df = pd.read_csv(inputs["predictions"])
    sample_df = pd.read_csv(inputs["sample_submission"], nrows=5)
    expected_cols = list(sample_df.columns)

    # If predictions already contain all expected columns, just reorder
    if all(col in pred_df.columns for col in expected_cols):
        submission = pred_df[expected_cols]
    else:
        rename_map = {}
        if expected_cols and expected_cols[0] != id_column and id_column in pred_df.columns:
            rename_map[id_column] = expected_cols[0]
        if len(expected_cols) > 1 and expected_cols[1] != prediction_column and prediction_column in pred_df.columns:
            rename_map[prediction_column] = expected_cols[1]

        if rename_map:
            pred_df = pred_df.rename(columns=rename_map)

        missing = [col for col in expected_cols if col not in pred_df.columns]
        if missing:
            raise ValueError(f"Predictions missing expected columns: {missing}")

        submission = pred_df[expected_cols]

    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    return f"format_submission: {len(submission)} rows, columns={expected_cols}"


# =============================================================================
# IMAGE RE-IDENTIFICATION SERVICES
# =============================================================================

@contract(
    inputs={
        "metadata": {"format": "csv", "required": True},
    },
    outputs={
        "database_metadata": {"format": "csv"},
        "query_metadata": {"format": "csv"},
    },
    description="Split metadata CSV into database and query sets for re-identification tasks",
    tags=["image", "reid", "preprocessing", "generic"],
    version="1.0.0",
)
def prepare_reid_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    split_column: str = "split",
    database_value: str = "database",
    query_value: str = "query",
) -> str:
    """
    Split a metadata CSV into separate database (reference) and query sets
    for image re-identification pipelines.

    Works with any re-identification dataset that has a split column
    (e.g., animal-clef, person-reid, vehicle-reid).

    Parameters:
        split_column: Column name indicating database vs query split.
        database_value: Value in split_column for reference/database images.
        query_value: Value in split_column for query/test images.
    """
    df = pd.read_csv(inputs["metadata"])

    db_df = df[df[split_column] == database_value].copy()
    query_df = df[df[split_column] == query_value].copy()

    os.makedirs(os.path.dirname(outputs["database_metadata"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["query_metadata"]) or ".", exist_ok=True)

    db_df.to_csv(outputs["database_metadata"], index=False)
    query_df.to_csv(outputs["query_metadata"], index=False)

    return (
        f"prepare_reid_data: {len(db_df)} database rows, "
        f"{len(query_df)} query rows"
    )


@contract(
    inputs={
        "metadata": {"format": "csv", "required": True},
        "root_dir": {"format": "directory", "required": True},
    },
    outputs={
        "embeddings": {"format": "pickle"},
    },
    description="Extract image embeddings using a pretrained CNN for re-identification or retrieval",
    tags=["image", "reid", "embeddings", "feature-extraction", "generic"],
    version="1.0.0",
)
def extract_image_embeddings(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    path_column: str = "path",
    id_column: str = "image_id",
    target_column: str = None,
    orientation_column: str = None,
    model_name: str = "resnet18",
    image_size: int = 224,
) -> str:
    """
    Extract image embeddings using a pretrained backbone (CNN or ViT).

    Loads each image from paths in the metadata CSV, runs through a pretrained
    model to get a feature vector, and saves all embeddings as a pickle bundle.

    Works with any image re-identification or retrieval task.

    Parameters:
        path_column: Column in metadata CSV containing image file paths (relative to root_dir).
        id_column: Column containing unique image identifiers.
        target_column: Optional column containing identity/class labels (for database images).
        orientation_column: Optional column containing orientation info (for orientation filtering).
        model_name: Pretrained model backbone. Options: "resnet18", "resnet50", "vit".
        image_size: Resize images to this square size before feature extraction.
    """
    import torch
    from torchvision import models, transforms
    from PIL import Image

    df = pd.read_csv(inputs["metadata"])
    root_dir = inputs["root_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model as feature extractor
    if model_name == "vit":
        # Use Vision Transformer from transformers library
        try:
            from transformers import ViTModel, ViTImageProcessor
            vit_model_name = "google/vit-base-patch16-224-in21k"
            processor = ViTImageProcessor.from_pretrained(vit_model_name)
            model = ViTModel.from_pretrained(vit_model_name).to(device)
            model.eval()
            use_vit = True
        except ImportError:
            print("transformers library not available, falling back to resnet18")
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.fc = torch.nn.Identity()
            model = model.to(device)
            model.eval()
            use_vit = False
    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Identity()
        model = model.to(device)
        model.eval()
        use_vit = False
    else:
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Identity()
        model = model.to(device)
        model.eval()
        use_vit = False

    if not use_vit:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    all_embeddings = []
    all_ids = []
    all_labels = []
    all_orientations = []
    skipped = 0

    for idx, row in df.iterrows():
        img_path = os.path.join(root_dir, row[path_column])
        try:
            img = Image.open(img_path).convert("RGB")

            if use_vit:
                # ViT processing
                inputs_vit = processor(images=img, return_tensors="pt")
                inputs_vit = {k: v.to(device) for k, v in inputs_vit.items()}
                with torch.no_grad():
                    vit_output = model(**inputs_vit)
                    # Use CLS token as embedding
                    feat = vit_output.last_hidden_state[:, 0]
                    # L2 normalize for cosine similarity
                    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                all_embeddings.append(feat.squeeze(0).cpu().numpy())
            else:
                img_t = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(img_t)
                all_embeddings.append(feat.squeeze(0).cpu().numpy())

            all_ids.append(row[id_column])
            if target_column and target_column in df.columns:
                all_labels.append(row[target_column])
            if orientation_column and orientation_column in df.columns:
                all_orientations.append(row[orientation_column])
        except Exception as e:
            skipped += 1
            continue

    result_bundle = {
        "embeddings": np.array(all_embeddings),
        "ids": all_ids,
        "labels": all_labels if all_labels else None,
        "orientations": all_orientations if all_orientations else None,
    }

    os.makedirs(os.path.dirname(outputs["embeddings"]) or ".", exist_ok=True)
    with open(outputs["embeddings"], "wb") as f:
        pickle.dump(result_bundle, f, protocol=4)

    dim = all_embeddings[0].shape[0] if all_embeddings else 0
    skip_msg = f", {skipped} skipped" if skipped else ""
    return (
        f"extract_image_embeddings: {len(all_embeddings)} images, "
        f"{dim}-dim features ({model_name}){skip_msg}"
    )


@contract(
    inputs={
        "database_embeddings": {"format": "pickle", "required": True},
        "query_embeddings": {"format": "pickle", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Match query images to database identities using cosine similarity for re-identification",
    tags=["image", "reid", "matching", "cosine-similarity", "generic"],
    version="1.0.0",
)
def match_reid_identities(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    similarity_threshold: float = 0.7,
    new_identity_label: str = "new_individual",
    id_column: str = "image_id",
    prediction_column: str = "identity",
    use_orientation_filter: bool = False,
) -> str:
    """
    Match query image embeddings to database embeddings using cosine similarity
    for re-identification tasks.

    For each query image, finds the most similar database image. If the
    similarity exceeds the threshold, assigns that identity; otherwise
    labels as a new/unknown individual.

    Works with any re-identification task (animals, persons, vehicles).

    Parameters:
        similarity_threshold: Minimum cosine similarity to assign a known identity.
        new_identity_label: Label for query images that don't match any database identity.
        id_column: Column name for image IDs in output.
        prediction_column: Column name for predicted identities in output.
        use_orientation_filter: If True, only match query images with database images
                                of the same orientation (requires orientation data in embeddings).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    with open(inputs["database_embeddings"], "rb") as f:
        db_data = pickle.load(f)
    with open(inputs["query_embeddings"], "rb") as f:
        query_data = pickle.load(f)

    db_embeddings = db_data["embeddings"]
    db_labels = db_data["labels"]
    db_orientations = db_data.get("orientations")
    query_embeddings = query_data["embeddings"]
    query_ids = query_data["ids"]
    query_orientations = query_data.get("orientations")

    results = []

    for i, query_id in enumerate(query_ids):
        query_emb = query_embeddings[i:i+1]

        # Determine which database embeddings to compare against
        if use_orientation_filter and query_orientations and db_orientations:
            query_orient = query_orientations[i]

            # Handle NaN orientations - use all database embeddings
            if pd.isna(query_orient):
                filtered_indices = list(range(len(db_embeddings)))
            else:
                # Filter to same orientation
                filtered_indices = [
                    j for j, orient in enumerate(db_orientations)
                    if orient == query_orient
                ]
                # Fallback to all if no matches
                if not filtered_indices:
                    filtered_indices = list(range(len(db_embeddings)))

            filtered_embeddings = db_embeddings[filtered_indices]
            filtered_labels = [db_labels[j] for j in filtered_indices] if db_labels else None
        else:
            filtered_embeddings = db_embeddings
            filtered_labels = db_labels
            filtered_indices = list(range(len(db_embeddings)))

        # Compute cosine similarity for this query
        sims = cosine_similarity(query_emb, filtered_embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= similarity_threshold and filtered_labels is not None:
            identity = filtered_labels[best_idx]
        else:
            identity = new_identity_label

        results.append({id_column: query_id, prediction_column: identity})

    pred_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    pred_df.to_csv(outputs["predictions"], index=False)

    n_known = sum(1 for r in results if r[prediction_column] != new_identity_label)
    n_new = len(results) - n_known
    orient_msg = ", orientation filter ON" if use_orientation_filter else ""
    return (
        f"match_reid_identities: {len(results)} predictions, "
        f"{n_known} known, {n_new} new (threshold={similarity_threshold}{orient_msg})"
    )


# =============================================================================
# TRAIN CNN KEYPOINT REGRESSOR (PyTorch)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True,
                       "schema": {"type": "tabular", "description": "CSV with pixel columns (px_*) and target columns"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "model"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train a CNN for keypoint regression on embedded pixel data with NaN-aware loss",
    tags=["image", "keypoint", "regression", "cnn", "pytorch", "generic"],
    version="1.0.0",
)
def train_cnn_keypoint_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_columns: List[str] = None,
    pixel_prefix: str = "px_",
    img_size: int = 96,
    n_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
    conv_channels: List[int] = None,
    fc_size: int = 256,
    validation_split: float = 0.2,
    random_state: int = 42,
    fill_nan_value: float = -1.0,
) -> str:
    """
    Train a CNN keypoint regressor following top Kaggle solutions.

    Key features (from facial-keypoints-detection winners):
    - Reshapes pixel columns back to images
    - Uses MSE loss with NaN masking (NaN targets don't contribute to loss)
    - Supports grayscale images from embedded pixel strings

    Parameters:
        target_columns: List of target column names (keypoint coordinates).
        pixel_prefix: Prefix for pixel columns (default: "px_").
        img_size: Image dimension (assumes square images).
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam optimizer learning rate.
        dropout: Dropout rate.
        conv_channels: List of output channels per conv block. Default: [32, 64, 128].
        fc_size: Number of units in hidden FC layer.
        validation_split: Fraction for validation.
        random_state: Random seed.
        fill_nan_value: Value to mark NaN targets (will be masked in loss).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    if conv_channels is None:
        conv_channels = [32, 64, 128]

    if target_columns is None:
        raise ValueError("target_columns must be specified")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Load data
    df = pd.read_csv(inputs["train_data"])

    # Extract pixel columns and reshape to images
    pixel_cols = sorted([c for c in df.columns if c.startswith(pixel_prefix)],
                        key=lambda x: int(x.replace(pixel_prefix, "")))
    n_pixels = len(pixel_cols)
    expected_pixels = img_size * img_size
    if n_pixels != expected_pixels:
        raise ValueError(f"Expected {expected_pixels} pixels but found {n_pixels}")

    X = df[pixel_cols].values.reshape(-1, img_size, img_size, 1).astype(np.float32)

    # Extract targets, fill NaN with marker value
    available_targets = [c for c in target_columns if c in df.columns]
    if len(available_targets) == 0:
        raise ValueError("No target columns found in data")
    y = df[available_targets].values.astype(np.float32)
    nan_mask = np.isnan(y)
    y[nan_mask] = fill_nan_value

    n_targets = len(available_targets)
    print(f"  Data: {len(X)} samples, {n_pixels} pixels, {n_targets} targets")
    print(f"  NaN values: {nan_mask.sum()} ({100*nan_mask.sum()/(y.size):.1f}%)")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=random_state
    )

    # Convert to PyTorch tensors (NCHW format)
    X_train_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_val_t = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    y_train_t = torch.FloatTensor(y_train)
    y_val_t = torch.FloatTensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )

    # Build CNN for regression
    class KeypointCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            ch_in = 1  # grayscale
            for ch_out in conv_channels:
                layers += [
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                ]
                ch_in = ch_out
            self.features = nn.Sequential(*layers)

            n_pools = len(conv_channels)
            flat_size = conv_channels[-1] * (img_size // (2 ** n_pools)) ** 2

            self.regressor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(),
                nn.Linear(flat_size, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_size, n_targets),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.regressor(x)
            return x

    # Device selection
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")

    model = KeypointCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # NaN-aware MSE loss
    def masked_mse_loss(pred, target, fill_value=-1.0):
        mask = (target != fill_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = (pred - target) ** 2
        return (diff * mask).sum() / mask.sum()

    best_rmse = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = masked_mse_loss(preds, batch_y, fill_nan_value)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device))
            val_loss = masked_mse_loss(val_preds, y_val_t.to(device), fill_nan_value).item()
            val_rmse = np.sqrt(val_loss)

        scheduler.step(val_loss)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_rmse={val_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    model_data = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "img_size": img_size,
        "n_targets": n_targets,
        "target_columns": available_targets,
        "conv_channels": conv_channels,
        "fc_size": fc_size,
        "dropout": dropout,
        "pixel_prefix": pixel_prefix,
        "fill_nan_value": fill_nan_value,
        "model_type": "keypoint_cnn",
    }

    for key in outputs:
        os.makedirs(os.path.dirname(outputs[key]) or ".", exist_ok=True)

    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f, protocol=4)

    metrics = {
        "model_type": "cnn_keypoint_regressor",
        "n_samples_train": len(X_train),
        "n_samples_valid": len(X_val),
        "n_features": n_pixels,
        "n_targets": n_targets,
        "target_columns": available_targets,
        "valid_rmse": float(best_rmse),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "conv_channels": conv_channels,
        "fc_size": fc_size,
        "dropout": dropout,
        "device": str(device),
    }

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_cnn_keypoint_regressor: best_val_rmse={best_rmse:.4f}, {len(X_train)} train, {len(X_val)} val"


# =============================================================================
# PREDICT CNN KEYPOINT REGRESSOR (PyTorch)
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Generate keypoint predictions using a trained CNN regressor",
    tags=["image", "keypoint", "regression", "inference", "cnn", "pytorch", "generic"],
    version="1.0.0",
)
def predict_cnn_keypoint_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    batch_size: int = 64,
    clip_min: float = 0.0,
    clip_max: float = 96.0,
) -> str:
    """
    Generate predictions from a trained CNN keypoint regressor.

    Clips predictions to valid coordinate range.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    img_size = model_data["img_size"]
    n_targets = model_data["n_targets"]
    target_columns = model_data["target_columns"]
    conv_channels = model_data["conv_channels"]
    fc_size = model_data["fc_size"]
    dropout = model_data["dropout"]
    pixel_prefix = model_data["pixel_prefix"]

    # Load test data
    df = pd.read_csv(inputs["data"])
    pixel_cols = sorted([c for c in df.columns if c.startswith(pixel_prefix)],
                        key=lambda x: int(x.replace(pixel_prefix, "")))
    X = df[pixel_cols].values.reshape(-1, img_size, img_size, 1).astype(np.float32)

    # Rebuild model architecture
    class KeypointCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            ch_in = 1
            for ch_out in conv_channels:
                layers += [
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                ]
                ch_in = ch_out
            self.features = nn.Sequential(*layers)
            n_pools = len(conv_channels)
            flat_size = conv_channels[-1] * (img_size // (2 ** n_pools)) ** 2
            self.regressor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(),
                nn.Linear(flat_size, fc_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fc_size, n_targets),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.regressor(x)
            return x

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = KeypointCNN().to(device)
    model.load_state_dict(model_data["state_dict"])
    model.eval()

    # Predict in batches
    X_t = torch.FloatTensor(X).permute(0, 3, 1, 2)
    test_loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (batch_X,) in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy()
            all_preds.append(preds)

    predictions = np.vstack(all_preds)
    predictions = np.clip(predictions, clip_min, clip_max)

    pred_df = pd.DataFrame(predictions, columns=target_columns)
    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    pred_df.to_csv(outputs["predictions"], index=False)

    return f"predict_cnn_keypoint_regressor: {len(predictions)} predictions for {n_targets} targets"


# =============================================================================
# TRAIN PRETRAINED KEYPOINT REGRESSOR (PyTorch + timm)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True,
                       "schema": {"type": "tabular", "description": "CSV with pixel columns (px_*) and target columns"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "model"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train a pretrained backbone (EfficientNetV2/ResNet/DenseNet) for keypoint regression with augmentation",
    tags=["image", "keypoint", "regression", "pretrained", "transfer-learning", "pytorch", "generic"],
    version="1.0.0",
)
def train_pretrained_keypoint_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_columns: List[str] = None,
    pixel_prefix: str = "px_",
    img_size: int = 96,
    architecture: str = "efficientnet_v2_s",
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    lr_step_size: int = 30,
    lr_gamma: float = 0.25,
    validation_split: float = 0.15,
    random_state: int = 42,
    fill_nan_value: float = -1.0,
    augmentation: str = "medium",
) -> str:
    """
    Train a pretrained keypoint regressor following top Kaggle solutions.

    Key features (from facial-keypoints-detection winners):
    - Uses pretrained backbones (EfficientNetV2-S, ResNet18, DenseNet121) via timm
    - Modifies first conv layer for 1-channel grayscale input
    - Data augmentation: rotation, blur, pixel dropout (from top solutions)
    - NaN-masked MSE loss
    - Learning rate scheduling (StepLR)

    Parameters:
        target_columns: List of target column names (keypoint coordinates).
        pixel_prefix: Prefix for pixel columns (default: "px_").
        img_size: Image dimension (assumes square images).
        architecture: Pretrained model. Options: "efficientnet_v2_s", "resnet18",
                     "resnet50", "densenet121", "mobilenetv3_large".
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate.
        weight_decay: AdamW weight decay.
        lr_step_size: StepLR step size.
        lr_gamma: StepLR gamma (decay factor).
        validation_split: Fraction for validation.
        random_state: Random seed.
        fill_nan_value: Value to mark NaN targets (will be masked in loss).
        augmentation: Augmentation level. Options: "none", "light", "medium", "heavy".
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.optim.lr_scheduler import StepLR
    from sklearn.model_selection import train_test_split

    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for pretrained models. Install with: pip install timm")

    if target_columns is None:
        raise ValueError("target_columns must be specified")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Architecture mapping
    ARCH_MAP = {
        "efficientnet_v2_s": "tf_efficientnetv2_s",
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "densenet121": "densenet121",
        "densenet201": "densenet201",
        "mobilenetv3_large": "mobilenetv3_large_100",
    }

    if architecture not in ARCH_MAP:
        raise ValueError(f"Unknown architecture: {architecture}. Options: {list(ARCH_MAP.keys())}")

    timm_model_name = ARCH_MAP[architecture]

    # Load data
    df = pd.read_csv(inputs["train_data"])

    # Extract pixel columns and reshape to images
    pixel_cols = sorted([c for c in df.columns if c.startswith(pixel_prefix)],
                        key=lambda x: int(x.replace(pixel_prefix, "")))
    n_pixels = len(pixel_cols)
    expected_pixels = img_size * img_size
    if n_pixels != expected_pixels:
        raise ValueError(f"Expected {expected_pixels} pixels but found {n_pixels}")

    X = df[pixel_cols].values.reshape(-1, img_size, img_size).astype(np.float32)

    # Extract targets, fill NaN with marker value
    available_targets = [c for c in target_columns if c in df.columns]
    if len(available_targets) == 0:
        raise ValueError("No target columns found in data")
    y = df[available_targets].values.astype(np.float32)
    nan_mask = np.isnan(y)
    y[nan_mask] = fill_nan_value

    n_targets = len(available_targets)
    print(f"  Data: {len(X)} samples, {n_pixels} pixels, {n_targets} targets")
    print(f"  NaN values: {nan_mask.sum()} ({100*nan_mask.sum()/(y.size):.1f}%)")
    print(f"  Architecture: {architecture} ({timm_model_name})")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=random_state
    )

    # Custom Dataset with augmentation
    class KeypointDataset(Dataset):
        def __init__(self, images, targets, augment_level="none", is_train=True):
            self.images = images
            self.targets = targets
            self.augment_level = augment_level
            self.is_train = is_train

        def __len__(self):
            return len(self.images)

        def _augment(self, image, target):
            """Apply augmentation (from top Kaggle solutions)."""
            import random
            import math

            if not self.is_train or self.augment_level == "none":
                return image, target

            target = target.copy()

            # Rotation (from solution 1)
            if random.random() > 0.5:
                angle = random.randint(-15, 15) if self.augment_level == "medium" else random.randint(-20, 20)
                # Rotate image
                from scipy.ndimage import rotate as scipy_rotate
                image = scipy_rotate(image, angle, reshape=False, mode='constant', cval=0)
                # Rotate keypoints
                angle_rad = math.radians(-angle)
                center = img_size / 2
                valid_mask = target != fill_nan_value
                x_coords = target[::2].copy()
                y_coords = target[1::2].copy()
                x_valid = valid_mask[::2]
                y_valid = valid_mask[1::2]

                new_x = center + (x_coords - center) * math.cos(angle_rad) - (y_coords - center) * math.sin(angle_rad)
                new_y = center + (x_coords - center) * math.sin(angle_rad) + (y_coords - center) * math.cos(angle_rad)

                target[::2] = np.where(x_valid, new_x, fill_nan_value)
                target[1::2] = np.where(y_valid, new_y, fill_nan_value)

            # Gaussian blur (from solution 1)
            if random.random() > 0.5:
                from scipy.ndimage import gaussian_filter
                sigma = 0.5 if self.augment_level == "light" else 1.0
                image = gaussian_filter(image, sigma=sigma)

            # Pixel dropout (from solution 1)
            if self.augment_level in ["medium", "heavy"] and random.random() > 0.5:
                dropout_rate = 0.03 if self.augment_level == "medium" else 0.05
                mask = np.random.random(image.shape) > dropout_rate
                image = image * mask

            return image, target

        def __getitem__(self, idx):
            image = self.images[idx].copy()
            target = self.targets[idx].copy()

            image, target = self._augment(image, target)

            # Convert to tensor (add channel dim for grayscale)
            image_tensor = torch.FloatTensor(image).unsqueeze(0)  # [1, H, W]
            target_tensor = torch.FloatTensor(target)

            return image_tensor, target_tensor

    train_dataset = KeypointDataset(X_train, y_train, augment_level=augmentation, is_train=True)
    val_dataset = KeypointDataset(X_val, y_val, augment_level="none", is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build pretrained model for keypoint regression
    class PretrainedKeypointModel(nn.Module):
        def __init__(self, model_name, n_outputs):
            super().__init__()
            # Load pretrained model
            self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1)

            # Get number of features from backbone
            if hasattr(self.backbone, 'num_features'):
                n_features = self.backbone.num_features
            elif hasattr(self.backbone, 'fc'):
                n_features = self.backbone.fc.in_features
            elif hasattr(self.backbone, 'classifier'):
                if hasattr(self.backbone.classifier, 'in_features'):
                    n_features = self.backbone.classifier.in_features
                else:
                    n_features = 1000  # Default classifier output
            else:
                n_features = 1000

            # Replace classifier with regression head
            if hasattr(self.backbone, 'fc'):
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, 'classifier'):
                self.backbone.classifier = nn.Identity()
            elif hasattr(self.backbone, 'head'):
                self.backbone.head = nn.Identity()

            # Regression head
            self.regressor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(n_features if n_features != 1000 else 1000, n_outputs)
            )

        def forward(self, x):
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.mean(dim=[2, 3])  # Global avg pool if needed
            return self.regressor(features)

    # Device selection
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")
    print(f"  Augmentation: {augmentation}")

    model = PretrainedKeypointModel(timm_model_name, n_targets).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # NaN-aware MSE loss
    def masked_mse_loss(pred, target, fill_value=-1.0):
        mask = (target != fill_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = (pred - target) ** 2
        return (diff * mask).sum() / mask.sum()

    best_rmse = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        # Training
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = masked_mse_loss(preds, batch_y, fill_nan_value)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                loss = masked_mse_loss(preds, batch_y, fill_nan_value)
                val_loss_sum += loss.item() * len(batch_X)
                val_count += len(batch_X)

        val_loss = val_loss_sum / max(val_count, 1)
        val_rmse = np.sqrt(val_loss)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_rmse={val_rmse:.4f}, lr={current_lr:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    model_data = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "img_size": img_size,
        "n_targets": n_targets,
        "target_columns": available_targets,
        "architecture": architecture,
        "timm_model_name": timm_model_name,
        "pixel_prefix": pixel_prefix,
        "fill_nan_value": fill_nan_value,
        "model_type": "pretrained_keypoint_regressor",
    }

    for key in outputs:
        os.makedirs(os.path.dirname(outputs[key]) or ".", exist_ok=True)

    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f, protocol=4)

    metrics = {
        "model_type": "pretrained_keypoint_regressor",
        "architecture": architecture,
        "n_samples_train": len(X_train),
        "n_samples_valid": len(X_val),
        "n_features": n_pixels,
        "n_targets": n_targets,
        "target_columns": available_targets,
        "valid_rmse": float(best_rmse),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_step_size": lr_step_size,
        "lr_gamma": lr_gamma,
        "augmentation": augmentation,
        "device": str(device),
    }

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_pretrained_keypoint_regressor: arch={architecture}, best_val_rmse={best_rmse:.4f}, {len(X_train)} train"


# =============================================================================
# PREDICT PRETRAINED KEYPOINT REGRESSOR (PyTorch + timm)
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Generate keypoint predictions using a trained pretrained regressor",
    tags=["image", "keypoint", "regression", "inference", "pretrained", "pytorch", "generic"],
    version="1.0.0",
)
def predict_pretrained_keypoint_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    batch_size: int = 64,
    clip_min: float = 0.0,
    clip_max: float = 96.0,
) -> str:
    """
    Generate predictions from a trained pretrained keypoint regressor.

    Clips predictions to valid coordinate range.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for pretrained models. Install with: pip install timm")

    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    img_size = model_data["img_size"]
    n_targets = model_data["n_targets"]
    target_columns = model_data["target_columns"]
    timm_model_name = model_data["timm_model_name"]
    pixel_prefix = model_data["pixel_prefix"]

    # Load test data
    df = pd.read_csv(inputs["data"])
    pixel_cols = sorted([c for c in df.columns if c.startswith(pixel_prefix)],
                        key=lambda x: int(x.replace(pixel_prefix, "")))
    X = df[pixel_cols].values.reshape(-1, 1, img_size, img_size).astype(np.float32)

    # Rebuild model architecture
    class PretrainedKeypointModel(nn.Module):
        def __init__(self, model_name, n_outputs):
            super().__init__()
            self.backbone = timm.create_model(model_name, pretrained=False, in_chans=1)

            if hasattr(self.backbone, 'num_features'):
                n_features = self.backbone.num_features
            elif hasattr(self.backbone, 'fc'):
                n_features = self.backbone.fc.in_features
            elif hasattr(self.backbone, 'classifier'):
                if hasattr(self.backbone.classifier, 'in_features'):
                    n_features = self.backbone.classifier.in_features
                else:
                    n_features = 1000
            else:
                n_features = 1000

            if hasattr(self.backbone, 'fc'):
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, 'classifier'):
                self.backbone.classifier = nn.Identity()
            elif hasattr(self.backbone, 'head'):
                self.backbone.head = nn.Identity()

            self.regressor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(n_features if n_features != 1000 else 1000, n_outputs)
            )

        def forward(self, x):
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.mean(dim=[2, 3])
            return self.regressor(features)

    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = PretrainedKeypointModel(timm_model_name, n_targets).to(device)
    model.load_state_dict(model_data["state_dict"])
    model.eval()

    # Predict in batches
    X_t = torch.FloatTensor(X)
    test_loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (batch_X,) in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy()
            all_preds.append(preds)

    predictions = np.vstack(all_preds)
    predictions = np.clip(predictions, clip_min, clip_max)

    pred_df = pd.DataFrame(predictions, columns=target_columns)
    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    pred_df.to_csv(outputs["predictions"], index=False)

    return f"predict_pretrained_keypoint_regressor: {len(predictions)} predictions for {n_targets} targets"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "load_image_dataset": load_image_dataset,
    "normalize_images": normalize_images,
    "train_image_classifier": train_image_classifier,
    "prepare_test_images": prepare_test_images,
    "load_labeled_images": load_labeled_images,
    "train_cnn_image_classifier": train_cnn_image_classifier,
    "predict_cnn_image_classifier": predict_cnn_image_classifier,
    "train_pretrained_image_classifier": train_pretrained_image_classifier,
    "predict_pretrained_image_classifier": predict_pretrained_image_classifier,
    "format_submission": format_submission,
    "prepare_reid_data": prepare_reid_data,
    "extract_image_embeddings": extract_image_embeddings,
    "match_reid_identities": match_reid_identities,
    "train_cnn_keypoint_regressor": train_cnn_keypoint_regressor,
    "predict_cnn_keypoint_regressor": predict_cnn_keypoint_regressor,
    "train_pretrained_keypoint_regressor": train_pretrained_keypoint_regressor,
    "predict_pretrained_keypoint_regressor": predict_pretrained_keypoint_regressor,
}
