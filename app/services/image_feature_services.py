"""
Image Feature Services - Common Module
=============================================

Generic image feature extraction services for tabular pipelines.
Based on analysis of top solutions from:
- Whale Categorization (PCA + grayscale compression)
- Cidaut Fake Scene (Transfer learning features)

These services extract numeric features from images for use with
standard tabular ML models (LightGBM, XGBoost, etc.).

All services follow G1-G6 design principles.

Services:
  Image Loading: load_images_to_features
  Feature Extraction: extract_image_statistics, extract_pca_features
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
# HELPERS: Import from shared io_utils
# =============================================================================
from services.io_utils import load_data as _load_data, save_data as _save_data


class ConstantPredictor:
    """Simple model that predicts a constant value for all inputs."""
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X) if hasattr(X, '__len__') else 1, self.value)


def _load_image(path: str, target_size: tuple = (64, 64)) -> np.ndarray:
    """Load and resize image to grayscale array."""
    try:
        from PIL import Image
        img = Image.open(path).convert('L')  # Grayscale
        img = img.resize(target_size)
        return np.array(img).flatten() / 255.0
    except Exception as e:
        # Return zeros if image can't be loaded
        return np.zeros(target_size[0] * target_size[1])


# =============================================================================
# SERVICE 1: LOAD IMAGES TO FEATURES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Load images and convert to flattened pixel features",
    tags=["feature-engineering", "image", "pixels", "generic"],
    version="1.0.0",
)
def load_images_to_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_column: str = "image",
    image_dir: str = "",
    target_size: int = 32,
    prefix: str = "px_",
) -> str:
    """
    Load images and convert to flattened grayscale pixel features.

    Parameters:
        image_column: Column containing image filenames
        image_dir: Directory containing images (relative to storage)
        target_size: Resize images to (target_size, target_size)
        prefix: Prefix for pixel feature columns
    """
    df = _load_data(inputs["data"])

    # Determine image directory
    if not image_dir:
        # Try to find images directory
        data_dir = os.path.dirname(inputs["data"])
        possible_dirs = [
            os.path.join(data_dir, 'train'),
            os.path.join(data_dir, 'images'),
            data_dir
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

    # Create DataFrame with pixel features
    pixel_cols = [f"{prefix}{j}" for j in range(n_pixels)]
    pixel_df = pd.DataFrame(pixel_features, columns=pixel_cols, index=df.index)

    df_out = pd.concat([df, pixel_df], axis=1)
    _save_data(df_out, outputs["data"])

    return f"load_images_to_features: loaded {loaded}/{len(df)} images, {n_pixels} features"


# =============================================================================
# SERVICE 2: EXTRACT IMAGE STATISTICS
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Extract statistical features from images (mean, std, edges, etc.)",
    tags=["feature-engineering", "image", "statistics", "generic"],
    version="1.0.0",
)
def extract_image_statistics(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_column: str = "image",
    image_dir: str = "",
    prefix: str = "img_",
) -> str:
    """
    Extract statistical features from images.

    Features: mean, std, min, max, median brightness,
    edge density, contrast, entropy approximation.
    """
    df = _load_data(inputs["data"])

    # Determine image directory
    if not image_dir:
        data_dir = os.path.dirname(inputs["data"])
        possible_dirs = [
            os.path.join(data_dir, 'train'),
            os.path.join(data_dir, 'images'),
            data_dir
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
        f'{prefix}median': [],
        f'{prefix}range': [],
    }

    for img_name in df[image_column]:
        img_path = os.path.join(image_dir, str(img_name))
        try:
            from PIL import Image
            img = Image.open(img_path).convert('L')
            arr = np.array(img) / 255.0

            stats[f'{prefix}mean'].append(arr.mean())
            stats[f'{prefix}std'].append(arr.std())
            stats[f'{prefix}min'].append(arr.min())
            stats[f'{prefix}max'].append(arr.max())
            stats[f'{prefix}median'].append(np.median(arr))
            stats[f'{prefix}range'].append(arr.max() - arr.min())
        except:
            for key in stats:
                stats[key].append(0.0)

    for key, values in stats.items():
        df[key] = values

    _save_data(df, outputs["data"])

    return f"extract_image_statistics: extracted 6 features for {len(df)} images"


# =============================================================================
# SERVICE 3: ENCODE IMAGE LABELS
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
        "encoder": {"format": "pickle"},
    },
    description="Label encode image classification labels",
    tags=["preprocessing", "image", "encoding", "generic"],
    version="1.0.0",
)
def encode_image_labels(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "label",
) -> str:
    """
    Label encode image classification labels to integers.
    """
    df = _load_data(inputs["data"])

    if df[label_column].dtype == 'object':
        codes, uniques = pd.factorize(df[label_column])
        df[label_column] = codes
        encoding = {str(v): int(i) for i, v in enumerate(uniques)}
    else:
        encoding = {}

    _save_data(df, outputs["data"])

    os.makedirs(os.path.dirname(outputs["encoder"]) or ".", exist_ok=True)
    with open(outputs["encoder"], "wb") as f:
        pickle.dump(encoding, f)

    return f"encode_image_labels: encoded {len(encoding)} classes in '{label_column}'"


# =============================================================================
# SERVICE 4: PARSE EMBEDDED PIXEL STRING
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Parse space-separated pixel strings into numeric columns",
    tags=["preprocessing", "image", "pixels", "generic"],
    version="1.0.0",
)
def parse_embedded_pixels(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    pixel_column: str = "Image",
    prefix: str = "px_",
    normalize: bool = True,
    sample_size: Optional[int] = None,
) -> str:
    """
    Parse space-separated pixel strings into numeric columns.

    For datasets like facial-keypoints-detection where images are stored
    as space-separated pixel values in a string column.

    Parameters:
        pixel_column: Column containing space-separated pixel string
        prefix: Prefix for pixel feature columns
        normalize: If True, divide pixel values by 255
        sample_size: If set, only process first N rows (for quick baseline)
    """
    df = _load_data(inputs["data"])

    # Optionally sample for faster processing
    if sample_size and sample_size < len(df):
        df = df.head(sample_size).copy()

    # Parse the pixel string column
    if pixel_column in df.columns:
        # Parse space-separated pixel values
        pixel_lists = df[pixel_column].str.split(' ')
        n_pixels = len(pixel_lists.iloc[0])

        # Create pixel array
        pixel_array = np.zeros((len(df), n_pixels))
        for i, pixels in enumerate(pixel_lists):
            try:
                pixel_array[i] = [float(p) for p in pixels]
            except:
                pass

        if normalize:
            pixel_array = pixel_array / 255.0

        # Create column names
        pixel_cols = [f"{prefix}{j}" for j in range(n_pixels)]
        pixel_df = pd.DataFrame(pixel_array, columns=pixel_cols, index=df.index)

        # Drop original pixel column and concat new features
        df = df.drop(columns=[pixel_column])
        df = pd.concat([df, pixel_df], axis=1)

    _save_data(df, outputs["data"])

    return f"parse_embedded_pixels: parsed {n_pixels} pixels for {len(df)} samples"


# =============================================================================
# SERVICE 5: EXTRACT PIXEL FEATURES (small sample baseline)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Extract pixel features from images for baseline model",
    tags=["feature-engineering", "image", "pixels", "generic"],
    version="1.0.0",
)
def extract_pixel_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_column: str = "id",
    image_dir: str = "",
    image_ext: str = ".jpg",
    target_size: int = 16,
    sample_size: Optional[int] = None,
    prefix: str = "px_",
) -> str:
    """
    Extract flattened pixel features from images for baseline model.

    This is a simplified version for creating quick baseline models.
    Uses small image size for faster processing.

    Parameters:
        image_column: Column containing image filenames/ids
        image_dir: Directory containing images
        image_ext: File extension for images (e.g., .jpg, .png)
        target_size: Resize images to target_size x target_size
        sample_size: Only process first N samples (for quick baseline)
        prefix: Prefix for pixel feature columns
    """
    df = _load_data(inputs["data"])

    # Sample for quick baseline
    if sample_size and sample_size < len(df):
        df = df.head(sample_size).copy()

    # Determine image directory if not specified
    if not image_dir:
        data_dir = os.path.dirname(inputs["data"])
        # Look for train directory relative to storage
        storage_dir = os.path.dirname(os.path.dirname(inputs["data"]))
        # Get competition name from path
        comp_name = os.path.basename(os.path.dirname(data_dir))
        possible_dirs = [
            os.path.join(storage_dir, comp_name, 'train'),
            os.path.join(storage_dir, 'train'),
            os.path.join(data_dir, '..', 'train'),
            os.path.join(data_dir, 'train'),
            data_dir
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                image_dir = d
                break
    else:
        # If image_dir is provided as relative path, resolve it
        if not os.path.isabs(image_dir):
            # Try relative to storage base path
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

    n_pixels = target_size * target_size
    pixel_features = np.zeros((len(df), n_pixels))

    loaded = 0
    for i, img_id in enumerate(df[image_column]):
        # Construct image path
        img_name = str(img_id)
        if not img_name.endswith(image_ext):
            img_name = img_name + image_ext
        img_path = os.path.join(image_dir, img_name)

        if os.path.exists(img_path):
            pixel_features[i] = _load_image(img_path, (target_size, target_size))
            loaded += 1

    # Create pixel feature columns
    pixel_cols = [f"{prefix}{j}" for j in range(n_pixels)]
    pixel_df = pd.DataFrame(pixel_features, columns=pixel_cols, index=df.index)

    df_out = pd.concat([df, pixel_df], axis=1)
    _save_data(df_out, outputs["data"])

    return f"extract_pixel_features: loaded {loaded}/{len(df)} images, {n_pixels} features"


# =============================================================================
# SERVICE 6: CREATE FACE PAIR TRAINING DATA
# =============================================================================

@contract(
    inputs={
        "relationships": {"format": "csv", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Create training data for face verification from relationship data",
    tags=["preprocessing", "face-verification", "generic"],
    version="1.0.0",
)
def create_face_pair_training(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    negative_ratio: float = 1.0,
    sample_size: Optional[int] = 2000,
    random_state: int = 42,
) -> str:
    """
    Create training data for face pair verification from relationship CSV.

    Takes a CSV with columns p1, p2 (related pairs) and creates a training
    dataset with both positive (related) and negative (unrelated) pairs.

    Features created are based on metadata:
    - same_family: whether pairs are from the same family
    - family_size features
    - member_count features

    Parameters:
        negative_ratio: Ratio of negative to positive samples
        sample_size: Maximum number of positive samples to use
        random_state: Random seed for reproducibility
    """
    np.random.seed(random_state)

    df = _load_data(inputs["relationships"])

    # Extract family and member info from p1, p2 columns
    # Format: F0002/MID1
    df['family1'] = df['p1'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'UNK')
    df['member1'] = df['p1'].apply(lambda x: x.split('/')[1] if '/' in str(x) else 'UNK')
    df['family2'] = df['p2'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'UNK')
    df['member2'] = df['p2'].apply(lambda x: x.split('/')[1] if '/' in str(x) else 'UNK')

    # Create positive samples (related pairs)
    positive_df = df.copy()
    positive_df['is_related'] = 1

    # Sample if needed
    if sample_size and len(positive_df) > sample_size:
        positive_df = positive_df.sample(n=sample_size, random_state=random_state)

    # Create negative samples (unrelated pairs from different families)
    all_families = df['family1'].unique().tolist()
    all_people = set()
    for _, row in df.iterrows():
        all_people.add(row['p1'])
        all_people.add(row['p2'])
    all_people = list(all_people)

    n_negative = int(len(positive_df) * negative_ratio)
    negative_samples = []

    attempts = 0
    max_attempts = n_negative * 10

    while len(negative_samples) < n_negative and attempts < max_attempts:
        p1 = np.random.choice(all_people)
        p2 = np.random.choice(all_people)

        family1 = p1.split('/')[0] if '/' in p1 else 'UNK'
        family2 = p2.split('/')[0] if '/' in p2 else 'UNK'

        # Only create negative if different families
        if family1 != family2:
            negative_samples.append({
                'p1': p1,
                'p2': p2,
                'family1': family1,
                'family2': family2,
                'member1': p1.split('/')[1] if '/' in p1 else 'UNK',
                'member2': p2.split('/')[1] if '/' in p2 else 'UNK',
                'is_related': 0
            })
        attempts += 1

    negative_df = pd.DataFrame(negative_samples)

    # Combine positive and negative samples
    combined_df = pd.concat([
        positive_df[['p1', 'p2', 'family1', 'member1', 'family2', 'member2', 'is_related']],
        negative_df
    ], ignore_index=True)

    # Create metadata-based features
    # Count members per family
    family_member_counts = {}
    for _, row in df.iterrows():
        fam = row['family1']
        if fam not in family_member_counts:
            family_member_counts[fam] = set()
        family_member_counts[fam].add(row['member1'])
        family_member_counts[fam].add(row['member2'])

    for fam, members in family_member_counts.items():
        family_member_counts[fam] = len(members)

    # Add features
    combined_df['same_family'] = (combined_df['family1'] == combined_df['family2']).astype(int)
    combined_df['family1_size'] = combined_df['family1'].map(lambda x: family_member_counts.get(x, 0))
    combined_df['family2_size'] = combined_df['family2'].map(lambda x: family_member_counts.get(x, 0))
    combined_df['family_size_diff'] = abs(combined_df['family1_size'] - combined_df['family2_size'])
    combined_df['family1_hash'] = combined_df['family1'].apply(lambda x: hash(x) % 1000)
    combined_df['family2_hash'] = combined_df['family2'].apply(lambda x: hash(x) % 1000)
    combined_df['member1_num'] = combined_df['member1'].apply(lambda x: int(x.replace('MID', '')) if 'MID' in str(x) else 0)
    combined_df['member2_num'] = combined_df['member2'].apply(lambda x: int(x.replace('MID', '')) if 'MID' in str(x) else 0)
    combined_df['member_num_diff'] = abs(combined_df['member1_num'] - combined_df['member2_num'])

    # Keep only numeric features for training
    feature_cols = ['same_family', 'family1_size', 'family2_size', 'family_size_diff',
                    'family1_hash', 'family2_hash', 'member1_num', 'member2_num',
                    'member_num_diff', 'is_related']
    output_df = combined_df[feature_cols]

    # Shuffle
    output_df = output_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    _save_data(output_df, outputs["data"])

    n_pos = (output_df['is_related'] == 1).sum()
    n_neg = (output_df['is_related'] == 0).sum()
    return f"create_face_pair_training: {n_pos} positive, {n_neg} negative samples, {len(feature_cols)-1} features"


# =============================================================================
# SERVICE 7: CREATE DENOISING BASELINE MODEL
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Create baseline model for image denoising that predicts constant or simple values",
    tags=["modeling", "denoising", "baseline", "generic"],
    version="1.0.0",
)
def create_denoising_baseline(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    default_value: float = 1.0,
    model_type: str = "constant",
) -> str:
    """
    Create a baseline model for image denoising tasks.

    This creates a simple model that can be used when actual training images
    are not available. For document denoising, predicting white (1.0) is
    a reasonable baseline since documents are mostly white background.

    Parameters:
        default_value: Default pixel value to predict (0-1 scale, 1=white, 0=black)
        model_type: Type of baseline ('constant' for single value, 'mean' for average)
    """
    df = _load_data(inputs["data"])

    # Create a simple baseline model using the module-level class
    model = ConstantPredictor(default_value)

    # Save model
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump({
            "model": model,
            "model_type": "denoising_baseline",
            "default_value": default_value,
            "feature_cols": ["row", "col"]  # Placeholder features
        }, f)

    # Save metrics
    metrics = {
        "model_type": f"denoising_baseline_{model_type}",
        "default_value": default_value,
        "n_train_images": len(df),
        "note": "Baseline model predicting constant value (deep learning required for proper denoising)"
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"create_denoising_baseline: created {model_type} model with default_value={default_value}"


# =============================================================================
# HELPERS: Window feature extraction for pixel-level denoising
# =============================================================================

def _get_padded(imgarray: np.ndarray, padding: int = 1) -> np.ndarray:
    """Pad image with mean border value for window feature extraction."""
    padval = int(round(imgarray.flatten().mean()))
    rows, cols = imgarray.shape
    xpad = np.full((rows, padding), padval, dtype=np.uint8)
    ypad = np.full((padding, cols + 2 * padding), padval, dtype=np.uint8)
    return np.vstack((ypad, np.hstack((xpad, imgarray, xpad)), ypad))


def _get_window_features(imgarray: np.ndarray, padding: int = 1) -> np.ndarray:
    """Extract window features for every pixel in an image.

    For each pixel, extracts a (2*padding+1) x (2*padding+1) neighborhood
    as a flattened feature vector.

    Returns array of shape (rows*cols, window_size^2).
    """
    rows, cols = imgarray.shape
    padded = _get_padded(imgarray, padding=padding)
    window_size = 2 * padding + 1
    n_features = window_size * window_size
    features = np.zeros((rows * cols, n_features), dtype=np.uint8)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            features[idx] = padded[i:i + window_size, j:j + window_size].flatten()
            idx += 1
    return features


# =============================================================================
# SERVICE 8: TRAIN DENOISING MODEL (Random Forest with window features)
# =============================================================================

@contract(
    inputs={
        "train_dir": {"format": "directory", "required": True},
        "clean_dir": {"format": "directory", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train a Random Forest model for image denoising using pixel window features",
    tags=["modeling", "denoising", "random-forest", "image", "generic"],
    version="1.0.0",
)
def train_denoising_model(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    window_size: int = 1,
    n_estimators: int = 32,
    max_samples_per_image: int = 0,
    n_jobs: int = -1,
    random_state: int = 42,
) -> str:
    """
    Train a Random Forest regressor for pixel-level image denoising.

    Based on the top-scoring Random Forest approach: for each pixel in a
    noisy image, extract a local window of surrounding pixel values as
    features, then predict the corresponding clean pixel value.

    Parameters:
        window_size: Padding around each pixel (1 = 3x3 window, 2 = 5x5)
        n_estimators: Number of trees in the Random Forest
        max_samples_per_image: Max pixel samples per image (0 = all pixels)
        n_jobs: Number of parallel jobs (-1 = all cores)
        random_state: Random seed for reproducibility
    """
    from PIL import Image
    from sklearn.ensemble import RandomForestRegressor

    train_dir = inputs["train_dir"]
    clean_dir = inputs["clean_dir"]

    # Get sorted list of training images
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.png')])
    n_images = len(train_files)

    # Collect features and targets from all training image pairs
    all_X = []
    all_y = []
    rng = np.random.RandomState(random_state)

    for fname in train_files:
        # Load noisy and clean images as grayscale uint8
        noisy_img = np.array(Image.open(os.path.join(train_dir, fname)).convert('L'))
        clean_img = np.array(Image.open(os.path.join(clean_dir, fname)).convert('L'))

        # Extract window features from noisy image
        X = _get_window_features(noisy_img, padding=window_size)
        y = clean_img.flatten().astype(np.float64) / 255.0

        # Subsample if requested
        if max_samples_per_image > 0 and len(y) > max_samples_per_image:
            indices = rng.choice(len(y), max_samples_per_image, replace=False)
            X = X[indices]
            y = y[indices]

        all_X.append(X)
        all_y.append(y)

    X_train = np.concatenate(all_X)
    y_train = np.concatenate(all_y)

    # Train Random Forest with incremental warm_start for memory efficiency
    chunk_size = 500000
    n_chunks = max(1, (len(y_train) + chunk_size - 1) // chunk_size)

    if len(y_train) <= chunk_size:
        # Small enough to train in one pass
        model = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=n_jobs,
            random_state=random_state
        )
        model.fit(X_train, y_train)
    else:
        # Use warm_start for large datasets: add trees incrementally per chunk
        trees_per_chunk = max(1, n_estimators // n_chunks)
        model = RandomForestRegressor(
            n_estimators=0, warm_start=True, n_jobs=n_jobs,
            random_state=random_state
        )
        indices = list(range(0, len(y_train), chunk_size))
        indices.append(len(y_train))
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            model.set_params(n_estimators=model.n_estimators + trees_per_chunk)
            model.fit(X_train[start:end], y_train[start:end])

    # Compute training RMSE on a sample
    sample_size = min(100000, len(y_train))
    sample_idx = rng.choice(len(y_train), sample_size, replace=False)
    y_pred_sample = model.predict(X_train[sample_idx])
    train_rmse = float(np.sqrt(np.mean((y_train[sample_idx] - y_pred_sample) ** 2)))

    # Save model
    model_data = {
        "model": model,
        "model_type": "random_forest_denoising",
        "window_size": window_size,
        "n_estimators": model.n_estimators,
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    # Save metrics
    metrics = {
        "model_type": "random_forest_denoising",
        "n_train_images": n_images,
        "total_train_pixels": int(len(y_train)),
        "window_size": window_size,
        "n_features": (2 * window_size + 1) ** 2,
        "n_estimators": model.n_estimators,
        "train_rmse": train_rmse,
    }
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return (f"train_denoising_model: trained RF ({model.n_estimators} trees) on "
            f"{n_images} images ({len(y_train)} pixels), train_rmse={train_rmse:.4f}")


# =============================================================================
# SERVICE 9: PREDICT DENOISING SUBMISSION
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_dir": {"format": "directory", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Generate pixel-level denoising predictions for Kaggle submission",
    tags=["prediction", "denoising", "submission", "image", "generic"],
    version="1.0.0",
)
def predict_denoising_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> str:
    """
    Generate pixel-level predictions for image denoising and create submission CSV.

    Loads the trained denoising model, processes each test image to extract
    window features, predicts clean pixel values, and writes submission in
    Kaggle format: id (imageId_row_col), value (predicted pixel intensity).

    Parameters:
        clip_min: Minimum value to clip predictions (0.0)
        clip_max: Maximum value to clip predictions (1.0)
    """
    from PIL import Image
    import csv

    # Load model
    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    window_size = model_data.get("window_size", 1)

    test_dir = inputs["test_dir"]
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])

    # Generate predictions and write submission
    submission_path = outputs["submission"]
    os.makedirs(os.path.dirname(submission_path) or ".", exist_ok=True)

    total_pixels = 0
    with open(submission_path, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(['id', 'value'])

        for fname in test_files:
            img_id = int(fname.replace('.png', ''))
            img = np.array(Image.open(os.path.join(test_dir, fname)).convert('L'))
            rows, cols = img.shape

            # Extract window features
            X_test = _get_window_features(img, padding=window_size)

            # Predict in chunks to manage memory
            chunk = 100000
            preds = np.zeros(rows * cols)
            for start in range(0, len(preds), chunk):
                end = min(start + chunk, len(preds))
                preds[start:end] = model.predict(X_test[start:end])

            # Clip predictions
            preds = np.clip(preds, clip_min, clip_max)

            # Write rows: id format is imageId_row_col (1-indexed)
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    writer.writerow([f"{img_id}_{r+1}_{c+1}", f"{preds[idx]:.6f}"])
                    idx += 1
            total_pixels += rows * cols

    return (f"predict_denoising_submission: predicted {len(test_files)} images, "
            f"{total_pixels} total pixels")


# =============================================================================
# SERVICE 10: FORMAT KEYPOINT SUBMISSION
# =============================================================================

@contract(
    inputs={
        "predictions": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "description": "Multi-output predictions with keypoint columns"},
        },
        "lookup_table": {
            "format": "csv",
            "required": True,
            "schema": {"type": "tabular", "description": "ID lookup table mapping RowId to ImageId+FeatureName"},
        },
    },
    outputs={
        "submission": {
            "format": "csv",
            "schema": {"type": "tabular", "columns": ["RowId", "Location"]},
        },
    },
    description="Format multi-output keypoint predictions into Kaggle submission using IdLookupTable",
    tags=["submission", "keypoint", "image", "formatting", "generic"],
    version="1.0.0",
)
def format_keypoint_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    image_id_column: str = "ImageId",
    feature_name_column: str = "FeatureName",
    clip_min: float = 0.0,
    clip_max: float = 96.0,
) -> str:
    """
    Format multi-output keypoint predictions into Kaggle submission format.

    Converts wide-format predictions (one column per keypoint) into the
    long-format submission required by keypoint detection competitions,
    using an IdLookupTable to map (ImageId, FeatureName) -> Location.

    Works with: facial-keypoints-detection, or any competition using
    an IdLookupTable to specify which predictions to include.
    """
    pred_df = _load_data(inputs["predictions"])
    lookup_df = _load_data(inputs["lookup_table"])

    # Build a mapping from (ImageId, FeatureName) -> predicted value
    # Predictions are indexed by row position (0-based), ImageId is 1-based
    locations = []
    for _, row in lookup_df.iterrows():
        image_idx = int(row[image_id_column]) - 1  # 1-based to 0-based
        feature_name = row[feature_name_column]
        if feature_name in pred_df.columns and image_idx < len(pred_df):
            value = float(pred_df.iloc[image_idx][feature_name])
        else:
            # Default to center of image if feature not available
            value = (clip_max - clip_min) / 2.0
        # Clip to valid coordinate range
        value = max(clip_min, min(clip_max, value))
        locations.append(value)

    submission = pd.DataFrame({
        "RowId": lookup_df["RowId"].values,
        "Location": locations,
    })

    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    return f"format_keypoint_submission: {len(submission)} rows formatted"


# =============================================================================
# HELPERS: Extract image statistics for face pair verification
# =============================================================================

def _extract_face_stats(img_path: str, target_size: tuple = (64, 64)) -> Optional[Dict[str, float]]:
    """Extract statistical features from a single face image.

    Returns a dict of numeric features or None if the image cannot be loaded.
    """
    try:
        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert('RGB')
        img = img.resize(target_size)
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

        gray = arr.mean(axis=2)  # (H, W)

        stats = {
            "brightness_mean": float(gray.mean()),
            "brightness_std": float(gray.std()),
            "brightness_min": float(gray.min()),
            "brightness_max": float(gray.max()),
            "brightness_median": float(np.median(gray)),
            "brightness_range": float(gray.max() - gray.min()),
            "r_mean": float(arr[:, :, 0].mean()),
            "g_mean": float(arr[:, :, 1].mean()),
            "b_mean": float(arr[:, :, 2].mean()),
            "r_std": float(arr[:, :, 0].std()),
            "g_std": float(arr[:, :, 1].std()),
            "b_std": float(arr[:, :, 2].std()),
        }

        # Edge density via simple gradient magnitude
        gy, gx = np.gradient(gray)
        edge_mag = np.sqrt(gx**2 + gy**2)
        stats["edge_density"] = float(edge_mag.mean())
        stats["edge_std"] = float(edge_mag.std())

        # Histogram-based entropy approximation
        hist, _ = np.histogram(gray, bins=32, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        stats["entropy"] = float(-np.sum(hist * np.log2(hist)))

        return stats
    except Exception:
        return None


def _compute_pair_features(stats1: Dict[str, float], stats2: Dict[str, float]) -> Dict[str, float]:
    """Compute pair-level features from two sets of image statistics."""
    features = {}
    for key in stats1:
        features[f"diff_{key}"] = abs(stats1[key] - stats2[key])
        features[f"sum_{key}"] = stats1[key] + stats2[key]
        if stats1[key] + stats2[key] > 0:
            features[f"ratio_{key}"] = min(stats1[key], stats2[key]) / (max(stats1[key], stats2[key]) + 1e-8)
        else:
            features[f"ratio_{key}"] = 0.0
    return features


# =============================================================================
# SERVICE 11: CREATE FACE PAIR IMAGE FEATURES (TRAINING)
# =============================================================================

@contract(
    inputs={
        "relationships": {"format": "csv", "required": True,
                          "schema": {"type": "tabular", "description": "CSV with p1, p2 columns (family/member paths)"}},
    },
    outputs={
        "data": {"format": "csv",
                 "schema": {"type": "tabular", "description": "Training CSV with image-based pair features and is_related label"}},
    },
    description="Create face pair training data with image statistical features extracted from face images",
    tags=["feature-engineering", "face-verification", "image", "generic"],
    version="1.0.0",
)
def create_face_pair_image_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    train_image_dir: str = "train-faces",
    negative_ratio: float = 1.0,
    sample_size: Optional[int] = 2000,
    random_state: int = 42,
    target_size: int = 64,
) -> str:
    """Create training data for face pair verification using actual image features.

    For each relationship pair, loads one face image per member, extracts
    statistical features (brightness, color, edges, entropy) from both faces,
    and computes pair-level distance features (absolute differences, ratios).

    Unlike create_face_pair_training which uses metadata features (family hash,
    same_family flag), this service extracts real image features that generalize
    to unseen test images.

    Parameters:
        train_image_dir: Directory containing training face images organized as Family/Member/image.jpg.
                         Can be absolute or relative to the relationships CSV directory.
        negative_ratio: Ratio of negative to positive samples
        sample_size: Maximum number of positive samples to use (None for all)
        random_state: Random seed for reproducibility
        target_size: Resize images to target_size x target_size before feature extraction
    """
    np.random.seed(random_state)

    df = _load_data(inputs["relationships"])

    # Resolve image directory
    base_dir = os.path.dirname(inputs["relationships"])
    if not os.path.isabs(train_image_dir):
        img_dir = os.path.join(base_dir, train_image_dir)
    else:
        img_dir = train_image_dir

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Train image directory not found: {img_dir}")

    # Parse family/member from p1, p2
    df['family1'] = df['p1'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'UNK')
    df['member1'] = df['p1'].apply(lambda x: x.split('/')[1] if '/' in str(x) else 'UNK')
    df['family2'] = df['p2'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'UNK')
    df['member2'] = df['p2'].apply(lambda x: x.split('/')[1] if '/' in str(x) else 'UNK')

    # Build cache: (family, member) -> first image path found
    member_image_cache = {}

    def _get_member_image(family, member):
        key = (family, member)
        if key not in member_image_cache:
            member_dir = os.path.join(img_dir, family, member)
            if os.path.isdir(member_dir):
                imgs = [f for f in os.listdir(member_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    member_image_cache[key] = os.path.join(member_dir, imgs[0])
                else:
                    member_image_cache[key] = None
            else:
                member_image_cache[key] = None
        return member_image_cache[key]

    # Create positive samples
    positive_df = df.copy()
    positive_df['is_related'] = 1
    if sample_size and len(positive_df) > sample_size:
        positive_df = positive_df.sample(n=sample_size, random_state=random_state)

    # Create negative samples (different families)
    all_people = set()
    for _, row in df.iterrows():
        all_people.add(row['p1'])
        all_people.add(row['p2'])
    all_people = list(all_people)

    n_negative = int(len(positive_df) * negative_ratio)
    negative_samples = []
    attempts = 0
    max_attempts = n_negative * 10
    while len(negative_samples) < n_negative and attempts < max_attempts:
        p1 = np.random.choice(all_people)
        p2 = np.random.choice(all_people)
        f1 = p1.split('/')[0] if '/' in p1 else 'UNK'
        f2 = p2.split('/')[0] if '/' in p2 else 'UNK'
        if f1 != f2:
            negative_samples.append({
                'p1': p1, 'p2': p2,
                'family1': f1, 'family2': f2,
                'member1': p1.split('/')[1] if '/' in p1 else 'UNK',
                'member2': p2.split('/')[1] if '/' in p2 else 'UNK',
                'is_related': 0,
            })
        attempts += 1

    negative_df = pd.DataFrame(negative_samples)

    combined_df = pd.concat([
        positive_df[['p1', 'p2', 'family1', 'member1', 'family2', 'member2', 'is_related']],
        negative_df,
    ], ignore_index=True)

    # Extract image features for each pair
    stats_cache = {}
    feature_rows = []
    skipped = 0

    for _, row in combined_df.iterrows():
        img1_path = _get_member_image(row['family1'], row['member1'])
        img2_path = _get_member_image(row['family2'], row['member2'])

        if img1_path is None or img2_path is None:
            skipped += 1
            continue

        # Cache stats per image
        if img1_path not in stats_cache:
            stats_cache[img1_path] = _extract_face_stats(img1_path, (target_size, target_size))
        if img2_path not in stats_cache:
            stats_cache[img2_path] = _extract_face_stats(img2_path, (target_size, target_size))

        s1 = stats_cache[img1_path]
        s2 = stats_cache[img2_path]

        if s1 is None or s2 is None:
            skipped += 1
            continue

        pair_feats = _compute_pair_features(s1, s2)
        pair_feats['is_related'] = row['is_related']
        feature_rows.append(pair_feats)

    output_df = pd.DataFrame(feature_rows)
    output_df = output_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    _save_data(output_df, outputs["data"])

    n_pos = (output_df['is_related'] == 1).sum()
    n_neg = (output_df['is_related'] == 0).sum()
    n_feats = len(output_df.columns) - 1
    return (f"create_face_pair_image_features: {n_pos} positive, {n_neg} negative, "
            f"{n_feats} features, {skipped} pairs skipped")


# =============================================================================
# SERVICE 12: CREATE TEST PAIR IMAGE FEATURES
# =============================================================================

@contract(
    inputs={
        "sample_submission": {"format": "csv", "required": True,
                              "schema": {"type": "tabular", "description": "CSV with img_pair and is_related columns"}},
    },
    outputs={
        "data": {"format": "csv",
                 "schema": {"type": "tabular", "description": "Test CSV with image-based pair features and img_pair ID column"}},
    },
    description="Extract image statistical features for test face pairs from sample submission",
    tags=["feature-engineering", "face-verification", "image", "generic"],
    version="1.0.0",
)
def create_test_pair_image_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    test_image_dir: str = "test",
    target_size: int = 64,
) -> str:
    """Extract image features for test face pairs referenced in sample_submission.csv.

    Parses the img_pair column (format: face05508.jpg-face01210.jpg), loads
    each face image from the test directory, extracts the same statistical
    features used by create_face_pair_image_features, and computes pair-level
    distance features.

    Parameters:
        test_image_dir: Directory containing test face images. Can be absolute
                        or relative to the sample_submission CSV directory.
        target_size: Resize images to target_size x target_size
    """
    sub_df = _load_data(inputs["sample_submission"])

    # Resolve image directory
    base_dir = os.path.dirname(inputs["sample_submission"])
    if not os.path.isabs(test_image_dir):
        img_dir = os.path.join(base_dir, test_image_dir)
    else:
        img_dir = test_image_dir

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Test image directory not found: {img_dir}")

    stats_cache = {}
    feature_rows = []
    skipped = 0

    for _, row in sub_df.iterrows():
        pair = str(row['img_pair'])
        parts = pair.split('-')
        if len(parts) != 2:
            skipped += 1
            continue

        img1_name, img2_name = parts[0], parts[1]
        img1_path = os.path.join(img_dir, img1_name)
        img2_path = os.path.join(img_dir, img2_name)

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            skipped += 1
            # Still add a row with zeros to maintain alignment
            pair_feats = {'img_pair': pair}
            feature_rows.append(pair_feats)
            continue

        if img1_path not in stats_cache:
            stats_cache[img1_path] = _extract_face_stats(img1_path, (target_size, target_size))
        if img2_path not in stats_cache:
            stats_cache[img2_path] = _extract_face_stats(img2_path, (target_size, target_size))

        s1 = stats_cache[img1_path]
        s2 = stats_cache[img2_path]

        if s1 is None or s2 is None:
            skipped += 1
            pair_feats = {'img_pair': pair}
            feature_rows.append(pair_feats)
            continue

        pair_feats = _compute_pair_features(s1, s2)
        pair_feats['img_pair'] = pair
        feature_rows.append(pair_feats)

    output_df = pd.DataFrame(feature_rows)

    # Fill missing features with 0 (for skipped pairs)
    output_df = output_df.fillna(0)

    _save_data(output_df, outputs["data"])

    return (f"create_test_pair_image_features: {len(output_df)} pairs, "
            f"{len(output_df.columns)-1} features, {skipped} skipped")


# =============================================================================
# SERVICE 13: TRAIN CNN AUTOENCODER FOR DENOISING
# =============================================================================

@contract(
    inputs={
        "train_dir": {"format": "directory", "required": True},
        "clean_dir": {"format": "directory", "required": True},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train a CNN autoencoder for image denoising (faster and more accurate than RF)",
    tags=["modeling", "denoising", "cnn", "autoencoder", "image", "generic"],
    version="1.0.0",
)
def train_cnn_autoencoder_denoising(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_height: int = 420,
    target_width: int = 540,
    n_epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    augment: bool = True,
    random_state: int = 42,
) -> str:
    """
    Train a CNN autoencoder for pixel-level image denoising.

    Based on top-scoring Kaggle solutions that use convolutional encoder-decoder
    architecture with image augmentation. This approach is significantly better
    than Random Forest for image denoising tasks.

    Parameters:
        target_height: Resize images to this height for training
        target_width: Resize images to this width for training
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        augment: Whether to use image augmentation (rotations, flips)
        random_state: Random seed for reproducibility
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from PIL import Image

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    train_dir = inputs["train_dir"]
    clean_dir = inputs["clean_dir"]

    # Load and preprocess training images
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.png')])
    n_images = len(train_files)

    noisy_images = []
    clean_images = []

    for fname in train_files:
        # Load noisy and clean images
        noisy_img = Image.open(os.path.join(train_dir, fname)).convert('L')
        clean_img = Image.open(os.path.join(clean_dir, fname)).convert('L')

        # Resize to target size
        noisy_img = noisy_img.resize((target_width, target_height))
        clean_img = clean_img.resize((target_width, target_height))

        # Convert to numpy and normalize to [0, 1]
        noisy_arr = np.array(noisy_img, dtype=np.float32) / 255.0
        clean_arr = np.array(clean_img, dtype=np.float32) / 255.0

        noisy_images.append(noisy_arr)
        clean_images.append(clean_arr)

    noisy_images = np.array(noisy_images)
    clean_images = np.array(clean_images)

    # Apply augmentation if enabled
    # Only use augmentations that preserve dimensions (no 90/270 degree rotations)
    if augment:
        augmented_noisy = [noisy_images]
        augmented_clean = [clean_images]

        # Rotate 180 degrees (preserves dimensions)
        augmented_noisy.append(np.rot90(noisy_images, k=2, axes=(1, 2)))
        augmented_clean.append(np.rot90(clean_images, k=2, axes=(1, 2)))

        # Horizontal flip (preserves dimensions)
        augmented_noisy.append(np.flip(noisy_images, axis=2).copy())
        augmented_clean.append(np.flip(clean_images, axis=2).copy())

        # Vertical flip (preserves dimensions)
        augmented_noisy.append(np.flip(noisy_images, axis=1).copy())
        augmented_clean.append(np.flip(clean_images, axis=1).copy())

        # Combine 180 rotation + horizontal flip
        rot180_noisy = np.rot90(noisy_images, k=2, axes=(1, 2))
        rot180_clean = np.rot90(clean_images, k=2, axes=(1, 2))
        augmented_noisy.append(np.flip(rot180_noisy, axis=2).copy())
        augmented_clean.append(np.flip(rot180_clean, axis=2).copy())

        noisy_images = np.concatenate(augmented_noisy, axis=0)
        clean_images = np.concatenate(augmented_clean, axis=0)

    # Add channel dimension
    noisy_images = noisy_images[:, np.newaxis, :, :]
    clean_images = clean_images[:, np.newaxis, :, :]

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(noisy_images.copy()).float()
    y_train = torch.from_numpy(clean_images.copy()).float()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define CNN Autoencoder architecture (similar to top solutions)
    class DenoisingAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(32),
            )
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(64),
            )
            self.enc3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
            )

            # Decoder
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(64),
            )
            self.dec2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            self.dec3 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
            )
            self.output = nn.Conv2d(32, 1, kernel_size=3, padding=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)

            # Decoder
            d1 = self.dec1(e3)
            d2 = self.dec2(d1)
            d3 = self.dec3(d2)
            out = self.sigmoid(self.output(d3))

            return out

    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop
    model.train()
    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_noisy, batch_clean in train_loader:
            batch_noisy = batch_noisy.to(device)
            batch_clean = batch_clean.to(device)

            optimizer.zero_grad()
            batch_pred = model(batch_noisy)
            loss = criterion(batch_pred, batch_clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_noisy.size(0)

        epoch_loss /= len(train_dataset)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Calculate training RMSE from best loss
    train_rmse = float(np.sqrt(best_loss))

    # Save model (convert to CPU for serialization)
    model = model.cpu()
    model_data = {
        "model_state_dict": model.state_dict(),
        "model_type": "cnn_autoencoder_denoising",
        "target_height": target_height,
        "target_width": target_width,
        "architecture": "DenoisingAutoencoder",
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    # Save metrics
    metrics = {
        "model_type": "cnn_autoencoder_denoising",
        "n_train_images": n_images,
        "augmented_samples": len(train_dataset),
        "target_height": target_height,
        "target_width": target_width,
        "n_epochs": n_epochs,
        "train_rmse": train_rmse,
        "best_loss": best_loss,
        "device": str(device),
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return (f"train_cnn_autoencoder_denoising: trained on {n_images} images "
            f"({len(train_dataset)} with augmentation), train_rmse={train_rmse:.6f}")


# =============================================================================
# SERVICE 14: PREDICT CNN AUTOENCODER DENOISING
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_dir": {"format": "directory", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
    },
    description="Generate pixel-level denoising predictions using trained CNN autoencoder",
    tags=["prediction", "denoising", "cnn", "submission", "image", "generic"],
    version="1.0.0",
)
def predict_cnn_autoencoder_denoising(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> str:
    """
    Generate pixel-level predictions for image denoising using CNN autoencoder.

    Loads the trained CNN autoencoder, processes each test image, predicts
    clean pixel values, and writes submission in Kaggle format.

    Parameters:
        clip_min: Minimum value to clip predictions (0.0)
        clip_max: Maximum value to clip predictions (1.0)
    """
    import torch
    import torch.nn as nn
    from PIL import Image
    import csv

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    target_height = model_data.get("target_height", 420)
    target_width = model_data.get("target_width", 540)

    # Define the same architecture
    class DenoisingAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(32),
            )
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(64),
            )
            self.enc3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
            )
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(64),
            )
            self.dec2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            self.dec3 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
            )
            self.output = nn.Conv2d(32, 1, kernel_size=3, padding=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            d1 = self.dec1(e3)
            d2 = self.dec2(d1)
            d3 = self.dec3(d2)
            out = self.sigmoid(self.output(d3))
            return out

    model = DenoisingAutoencoder()
    model.load_state_dict(model_data["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_dir = inputs["test_dir"]
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])

    submission_path = outputs["submission"]
    os.makedirs(os.path.dirname(submission_path) or ".", exist_ok=True)

    total_pixels = 0
    with open(submission_path, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(['id', 'value'])

        with torch.no_grad():
            for fname in test_files:
                img_id = int(fname.replace('.png', ''))

                # Load original image to get dimensions
                orig_img = Image.open(os.path.join(test_dir, fname)).convert('L')
                orig_height, orig_width = orig_img.size[1], orig_img.size[0]

                # Resize for model input
                resized_img = orig_img.resize((target_width, target_height))
                img_arr = np.array(resized_img, dtype=np.float32) / 255.0

                # Convert to tensor
                img_tensor = torch.from_numpy(img_arr[np.newaxis, np.newaxis, :, :]).float().to(device)

                # Predict
                pred_tensor = model(img_tensor)
                pred_arr = pred_tensor.cpu().numpy()[0, 0]

                # Resize back to original dimensions
                pred_img = Image.fromarray((pred_arr * 255).astype(np.uint8))
                pred_img = pred_img.resize((orig_width, orig_height), Image.BILINEAR)
                pred_final = np.array(pred_img, dtype=np.float32) / 255.0

                # Clip predictions
                pred_final = np.clip(pred_final, clip_min, clip_max)

                # Write rows: id format is imageId_row_col (1-indexed)
                for r in range(orig_height):
                    for c in range(orig_width):
                        writer.writerow([f"{img_id}_{r+1}_{c+1}", f"{pred_final[r, c]:.6f}"])

                total_pixels += orig_height * orig_width

    return (f"predict_cnn_autoencoder_denoising: predicted {len(test_files)} images, "
            f"{total_pixels} total pixels")


# =============================================================================
# SERVICE 15: EXTRACT FACE EMBEDDINGS (DEEP LEARNING)
# =============================================================================

def _load_face_embedding_model(device):
    """Load pretrained ResNet model for face embedding extraction."""
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms

    # Use ResNet50 pretrained on ImageNet as feature extractor
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove final classification layer to get embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    # Standard ImageNet transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, transform


def _extract_single_embedding(img_path: str, model, transform, device) -> Optional[np.ndarray]:
    """Extract embedding from a single face image."""
    import torch
    try:
        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(img_tensor)

        return embedding.cpu().numpy().flatten()
    except Exception:
        return None


def _compute_embedding_pair_features(emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
    """Compute similarity features between two face embeddings."""
    from scipy.spatial.distance import cosine, euclidean

    features = {}

    # Cosine similarity (key metric for face verification)
    features['cosine_similarity'] = float(1 - cosine(emb1, emb2))

    # Euclidean distance
    features['euclidean_distance'] = float(euclidean(emb1, emb2))

    # L1 (Manhattan) distance
    features['manhattan_distance'] = float(np.sum(np.abs(emb1 - emb2)))

    # Element-wise absolute difference stats
    diff = np.abs(emb1 - emb2)
    features['diff_mean'] = float(diff.mean())
    features['diff_std'] = float(diff.std())
    features['diff_max'] = float(diff.max())
    features['diff_min'] = float(diff.min())

    # Element-wise product stats (captures co-activation patterns)
    prod = emb1 * emb2
    features['prod_mean'] = float(prod.mean())
    features['prod_std'] = float(prod.std())
    features['prod_sum'] = float(prod.sum())

    # Correlation coefficient
    if emb1.std() > 0 and emb2.std() > 0:
        features['correlation'] = float(np.corrcoef(emb1, emb2)[0, 1])
    else:
        features['correlation'] = 0.0

    return features


@contract(
    inputs={
        "relationships": {"format": "csv", "required": True,
                          "schema": {"type": "tabular", "description": "CSV with p1, p2 columns"}},
    },
    outputs={
        "data": {"format": "csv",
                 "schema": {"type": "tabular", "description": "Training CSV with embedding features"}},
    },
    description="Extract deep face embedding features using pretrained ResNet for face pair verification",
    tags=["feature-engineering", "face-verification", "deep-learning", "embedding", "generic"],
    version="1.0.0",
)
def create_face_embedding_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    train_image_dir: str = "train-faces",
    negative_ratio: float = 1.0,
    sample_size: Optional[int] = 3000,
    random_state: int = 42,
) -> str:
    """Create training data for face pair verification using deep face embeddings.

    Uses a pretrained ResNet model to extract 2048-dimensional face embeddings
    and computes similarity features (cosine similarity, euclidean distance, etc.)
    between face pairs. This approach is based on top Kaggle solutions that use
    deep learning embeddings for kinship verification.

    Parameters:
        train_image_dir: Directory containing training face images (Family/Member/image.jpg)
        negative_ratio: Ratio of negative to positive samples
        sample_size: Maximum number of positive samples to use
        random_state: Random seed for reproducibility
    """
    import torch

    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    df = _load_data(inputs["relationships"])

    # Resolve image directory
    base_dir = os.path.dirname(inputs["relationships"])
    if not os.path.isabs(train_image_dir):
        img_dir = os.path.join(base_dir, train_image_dir)
    else:
        img_dir = train_image_dir

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Train image directory not found: {img_dir}")

    # Load embedding model
    print(f"  Loading ResNet model on {device}...")
    model, transform = _load_face_embedding_model(device)

    # Parse family/member
    df['family1'] = df['p1'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'UNK')
    df['member1'] = df['p1'].apply(lambda x: x.split('/')[1] if '/' in str(x) else 'UNK')
    df['family2'] = df['p2'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'UNK')
    df['member2'] = df['p2'].apply(lambda x: x.split('/')[1] if '/' in str(x) else 'UNK')

    # Build image path cache
    member_image_cache = {}
    def _get_member_image(family, member):
        key = (family, member)
        if key not in member_image_cache:
            member_dir = os.path.join(img_dir, family, member)
            if os.path.isdir(member_dir):
                imgs = [f for f in os.listdir(member_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    member_image_cache[key] = os.path.join(member_dir, imgs[0])
                else:
                    member_image_cache[key] = None
            else:
                member_image_cache[key] = None
        return member_image_cache[key]

    # Create positive samples
    positive_df = df.copy()
    positive_df['is_related'] = 1
    if sample_size and len(positive_df) > sample_size:
        positive_df = positive_df.sample(n=sample_size, random_state=random_state)

    # Create negative samples
    all_people = set()
    for _, row in df.iterrows():
        all_people.add(row['p1'])
        all_people.add(row['p2'])
    all_people = list(all_people)

    n_negative = int(len(positive_df) * negative_ratio)
    negative_samples = []
    attempts = 0
    max_attempts = n_negative * 10

    while len(negative_samples) < n_negative and attempts < max_attempts:
        p1 = np.random.choice(all_people)
        p2 = np.random.choice(all_people)
        f1 = p1.split('/')[0] if '/' in p1 else 'UNK'
        f2 = p2.split('/')[0] if '/' in p2 else 'UNK'
        if f1 != f2:
            negative_samples.append({
                'p1': p1, 'p2': p2,
                'family1': f1, 'family2': f2,
                'member1': p1.split('/')[1] if '/' in p1 else 'UNK',
                'member2': p2.split('/')[1] if '/' in p2 else 'UNK',
                'is_related': 0,
            })
        attempts += 1

    negative_df = pd.DataFrame(negative_samples)

    combined_df = pd.concat([
        positive_df[['p1', 'p2', 'family1', 'member1', 'family2', 'member2', 'is_related']],
        negative_df,
    ], ignore_index=True)

    # Extract embeddings and compute features
    embedding_cache = {}
    feature_rows = []
    skipped = 0

    print(f"  Extracting embeddings for {len(combined_df)} pairs...")
    for idx, row in combined_df.iterrows():
        if idx % 500 == 0:
            print(f"    Processing pair {idx}/{len(combined_df)}...")

        img1_path = _get_member_image(row['family1'], row['member1'])
        img2_path = _get_member_image(row['family2'], row['member2'])

        if img1_path is None or img2_path is None:
            skipped += 1
            continue

        # Cache embeddings
        if img1_path not in embedding_cache:
            embedding_cache[img1_path] = _extract_single_embedding(img1_path, model, transform, device)
        if img2_path not in embedding_cache:
            embedding_cache[img2_path] = _extract_single_embedding(img2_path, model, transform, device)

        emb1 = embedding_cache[img1_path]
        emb2 = embedding_cache[img2_path]

        if emb1 is None or emb2 is None:
            skipped += 1
            continue

        # Compute embedding similarity features
        pair_feats = _compute_embedding_pair_features(emb1, emb2)
        pair_feats['is_related'] = row['is_related']
        feature_rows.append(pair_feats)

    output_df = pd.DataFrame(feature_rows)
    output_df = output_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    _save_data(output_df, outputs["data"])

    n_pos = (output_df['is_related'] == 1).sum()
    n_neg = (output_df['is_related'] == 0).sum()
    n_feats = len(output_df.columns) - 1
    return (f"create_face_embedding_features: {n_pos} positive, {n_neg} negative, "
            f"{n_feats} embedding features, {skipped} pairs skipped")


# =============================================================================
# SERVICE 16: TEST FACE EMBEDDING FEATURES
# =============================================================================

@contract(
    inputs={
        "sample_submission": {"format": "csv", "required": True,
                              "schema": {"type": "tabular", "description": "CSV with img_pair column"}},
    },
    outputs={
        "data": {"format": "csv",
                 "schema": {"type": "tabular", "description": "Test CSV with embedding features"}},
    },
    description="Extract deep face embedding features for test pairs",
    tags=["feature-engineering", "face-verification", "deep-learning", "embedding", "generic"],
    version="1.0.0",
)
def create_test_face_embedding_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    test_image_dir: str = "test",
) -> str:
    """Extract face embedding features for test pairs from sample_submission.

    Uses the same pretrained ResNet model as create_face_embedding_features
    to extract embeddings and compute similarity features.

    Parameters:
        test_image_dir: Directory containing test face images
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    sub_df = _load_data(inputs["sample_submission"])

    # Resolve image directory
    base_dir = os.path.dirname(inputs["sample_submission"])
    if not os.path.isabs(test_image_dir):
        img_dir = os.path.join(base_dir, test_image_dir)
    else:
        img_dir = test_image_dir

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Test image directory not found: {img_dir}")

    # Load embedding model
    print(f"  Loading ResNet model on {device}...")
    model, transform = _load_face_embedding_model(device)

    embedding_cache = {}
    feature_rows = []
    skipped = 0

    print(f"  Extracting embeddings for {len(sub_df)} test pairs...")
    for idx, row in sub_df.iterrows():
        if idx % 1000 == 0:
            print(f"    Processing pair {idx}/{len(sub_df)}...")

        pair = str(row['img_pair'])
        parts = pair.split('-')
        if len(parts) != 2:
            skipped += 1
            # Add dummy features to maintain alignment
            pair_feats = {'img_pair': pair, 'cosine_similarity': 0.5}
            feature_rows.append(pair_feats)
            continue

        img1_name, img2_name = parts[0], parts[1]
        img1_path = os.path.join(img_dir, img1_name)
        img2_path = os.path.join(img_dir, img2_name)

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            skipped += 1
            pair_feats = {'img_pair': pair, 'cosine_similarity': 0.5}
            feature_rows.append(pair_feats)
            continue

        # Cache embeddings
        if img1_path not in embedding_cache:
            embedding_cache[img1_path] = _extract_single_embedding(img1_path, model, transform, device)
        if img2_path not in embedding_cache:
            embedding_cache[img2_path] = _extract_single_embedding(img2_path, model, transform, device)

        emb1 = embedding_cache[img1_path]
        emb2 = embedding_cache[img2_path]

        if emb1 is None or emb2 is None:
            skipped += 1
            pair_feats = {'img_pair': pair, 'cosine_similarity': 0.5}
            feature_rows.append(pair_feats)
            continue

        pair_feats = _compute_embedding_pair_features(emb1, emb2)
        pair_feats['img_pair'] = pair
        feature_rows.append(pair_feats)

    output_df = pd.DataFrame(feature_rows)
    output_df = output_df.fillna(0.5)  # Fill missing with neutral value

    _save_data(output_df, outputs["data"])

    return (f"create_test_face_embedding_features: {len(output_df)} pairs, "
            f"{len(output_df.columns)-1} features, {skipped} skipped")


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "load_images_to_features": load_images_to_features,
    "extract_image_statistics": extract_image_statistics,
    "parse_embedded_pixels": parse_embedded_pixels,
    "extract_pixel_features": extract_pixel_features,
    "create_face_pair_image_features": create_face_pair_image_features,
    "create_test_pair_image_features": create_test_pair_image_features,
    "train_denoising_model": train_denoising_model,
    "predict_denoising_submission": predict_denoising_submission,
    "format_keypoint_submission": format_keypoint_submission,
    "train_cnn_autoencoder_denoising": train_cnn_autoencoder_denoising,
    "predict_cnn_autoencoder_denoising": predict_cnn_autoencoder_denoising,
    "create_face_embedding_features": create_face_embedding_features,
    "create_test_face_embedding_features": create_test_face_embedding_features,
}
