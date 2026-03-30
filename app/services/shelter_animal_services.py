"""
Shelter Animal Outcomes - SLEGO Services
=========================================
Competition: https://www.kaggle.com/competitions/shelter-animal-outcomes
Problem Type: Multiclass Classification
Target: OutcomeType (Adoption, Died, Euthanasia, Return_to_owner, Transfer)
Metric: Multi-class Log Loss

Competition-specific services (derived from top solution notebooks):
- parse_age_to_days: Convert age strings like '2 years' to numeric days
- parse_sex_outcome: Parse sex/neutered status into binary features
- extract_name_features: Extract has_name and name_length
- extract_breed_features: Extract is_mix and breed cross count
- extract_color_features: Extract color count and multicolor flag
- extract_shelter_datetime: Extract datetime components (month, weekday, hour)
- encode_shelter_target: Alphabetically encode OutcomeType for consistent class ordering
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
from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services from common modules
from services.preprocessing_services import (
    split_data,
    drop_columns,
    encode_all_categorical,
)
from services.classification_services import (
    train_lightgbm_classifier,
    train_xgboost_classifier,
    predict_classifier,
    predict_multiclass_submission,
)


# =============================================================================
# DOMAIN-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Convert age strings like '2 years' to numeric days",
    tags=["preprocessing", "feature-engineering", "shelter", "generic"],
    version="1.0.0",
)
def parse_age_to_days(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    age_column: str = "AgeuponOutcome",
) -> str:
    """Convert age strings like '2 years' to numeric days.

    Handles year, month, week, day units. Missing values filled with median.
    Derived from top-scoring shelter-animal-outcomes solution notebooks.

    G4 Compliance: Column name parameterized.
    """
    df = _load_data(inputs["data"])

    def convert_age(age_str):
        if pd.isna(age_str):
            return np.nan
        parts = str(age_str).split()
        if len(parts) < 2:
            return np.nan
        try:
            num = int(parts[0])
        except ValueError:
            return np.nan
        unit = parts[1].lower()
        if "year" in unit:
            return num * 365
        elif "month" in unit:
            return num * 30
        elif "week" in unit:
            return num * 7
        elif "day" in unit:
            return num
        return np.nan

    if age_column in df.columns:
        df["age_days"] = df[age_column].apply(convert_age)
        df["age_days"] = df["age_days"].fillna(df["age_days"].median())

    _save_data(df, outputs["data"])
    return f"parse_age_to_days: converted {age_column} to age_days"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Parse sex/neutered column into binary gender and neutered status features",
    tags=["preprocessing", "feature-engineering", "shelter", "generic"],
    version="1.0.0",
)
def parse_sex_outcome(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    sex_column: str = "SexuponOutcome",
) -> str:
    """Parse sex column into gender and neutered status binary features.

    Creates: is_male, is_neutered, is_intact.
    Derived from top-scoring solution notebooks.

    G4 Compliance: Column name parameterized.
    """
    df = _load_data(inputs["data"])

    if sex_column in df.columns:
        df["is_male"] = df[sex_column].str.contains("Male", case=False, na=False).astype(int)
        df["is_neutered"] = df[sex_column].str.contains("Neutered|Spayed", case=False, na=False).astype(int)
        df["is_intact"] = df[sex_column].str.contains("Intact", case=False, na=False).astype(int)

    _save_data(df, outputs["data"])
    return f"parse_sex_outcome: created is_male, is_neutered, is_intact"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract features from animal name (has_name, name_length)",
    tags=["preprocessing", "feature-engineering", "shelter", "generic"],
    version="1.0.0",
)
def extract_name_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    name_column: str = "Name",
) -> str:
    """Extract features from animal name column.

    Creates: has_name (binary), name_length (integer).
    Derived from top-scoring solution notebooks.
    """
    df = _load_data(inputs["data"])

    if name_column in df.columns:
        df["has_name"] = df[name_column].notna().astype(int)
        df["name_length"] = df[name_column].fillna("").str.len()

    _save_data(df, outputs["data"])
    return f"extract_name_features: created has_name, name_length"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract breed features (is_mix, breed_count)",
    tags=["preprocessing", "feature-engineering", "shelter", "generic"],
    version="1.0.0",
)
def extract_breed_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    breed_column: str = "Breed",
) -> str:
    """Extract features from breed column.

    Creates: is_mix (binary for Mix or cross breeds), breed_count (number of breeds).
    Derived from top-scoring solution notebooks which use breed indicator variables.
    """
    df = _load_data(inputs["data"])

    if breed_column in df.columns:
        df["is_mix"] = df[breed_column].str.contains("Mix|/", case=False, na=False).astype(int)
        df["breed_count"] = df[breed_column].str.count("/").fillna(0).astype(int) + 1

    _save_data(df, outputs["data"])
    return f"extract_breed_features: created is_mix, breed_count"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"},
        "breed_vocab": {"format": "json"},
    },
    description="Create binary indicator variables for each breed (top solution technique)",
    tags=["preprocessing", "feature-engineering", "shelter", "generic"],
    version="1.0.0",
)
def create_breed_indicators(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    breed_column: str = "Breed",
    min_frequency: int = 10,
) -> str:
    """Create binary indicator columns for each unique breed word.

    This is the KEY technique from top-scoring solutions that dramatically improves
    accuracy by giving XGBoost explicit breed signals instead of just aggregate counts.

    Process:
    1. Extract all unique breed words from train+test (split by "/" and " Mix")
    2. Filter breeds appearing at least min_frequency times
    3. Create binary columns: 1 if breed word appears in animal's Breed, 0 otherwise
    4. Save breed vocabulary for reproducibility

    G4 Compliance: Column name parameterized.
    """
    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    if breed_column not in train_df.columns:
        _save_data(train_df, outputs["train_data"])
        _save_data(test_df, outputs["test_data"])
        return "create_breed_indicators: breed column not found"

    # Combine breeds to build vocabulary
    all_breeds = pd.concat([train_df[breed_column].fillna(""), test_df[breed_column].fillna("")])

    # Extract individual breed words (split by / and remove Mix suffix)
    import re
    breed_words = []
    for breed in all_breeds:
        words = re.split(r'/| Mix', str(breed))
        breed_words.extend([w.strip() for w in words if w.strip()])

    # Count frequencies and filter
    from collections import Counter
    breed_counts = Counter(breed_words)
    frequent_breeds = [b for b, c in breed_counts.items() if c >= min_frequency and b]

    # Create indicator columns
    for breed in frequent_breeds:
        col_name = f"breed_{breed.replace(' ', '_').replace('-', '_')}"
        train_df[col_name] = train_df[breed_column].str.contains(re.escape(breed), case=False, na=False).astype(int)
        test_df[col_name] = test_df[breed_column].str.contains(re.escape(breed), case=False, na=False).astype(int)

    # Save breed vocabulary
    vocab_path = outputs.get("breed_vocab")
    if vocab_path:
        os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump({"breeds": frequent_breeds, "counts": dict(breed_counts)}, f, indent=2)

    _save_data(train_df, outputs["train_data"])
    _save_data(test_df, outputs["test_data"])
    return f"create_breed_indicators: created {len(frequent_breeds)} breed indicator columns"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"},
        "color_vocab": {"format": "json"},
    },
    description="Create binary indicator variables for each color (top solution technique)",
    tags=["preprocessing", "feature-engineering", "shelter", "generic"],
    version="1.0.0",
)
def create_color_indicators(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    color_column: str = "Color",
    min_frequency: int = 10,
) -> str:
    """Create binary indicator columns for each unique color word.

    Similar to breed indicators, this technique from top solutions provides
    explicit color signals to the model.

    G4 Compliance: Column name parameterized.
    """
    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    if color_column not in train_df.columns:
        _save_data(train_df, outputs["train_data"])
        _save_data(test_df, outputs["test_data"])
        return "create_color_indicators: color column not found"

    # Combine colors to build vocabulary
    all_colors = pd.concat([train_df[color_column].fillna(""), test_df[color_column].fillna("")])

    # Extract individual color words (split by /)
    color_words = []
    for color in all_colors:
        words = str(color).split("/")
        color_words.extend([w.strip() for w in words if w.strip()])

    # Count frequencies and filter
    from collections import Counter
    color_counts = Counter(color_words)
    frequent_colors = [c for c, cnt in color_counts.items() if cnt >= min_frequency and c]

    # Create indicator columns
    import re
    for color in frequent_colors:
        col_name = f"color_{color.replace(' ', '_').replace('/', '_')}"
        train_df[col_name] = train_df[color_column].str.contains(re.escape(color), case=False, na=False).astype(int)
        test_df[col_name] = test_df[color_column].str.contains(re.escape(color), case=False, na=False).astype(int)

    # Save color vocabulary
    vocab_path = outputs.get("color_vocab")
    if vocab_path:
        os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump({"colors": frequent_colors, "counts": dict(color_counts)}, f, indent=2)

    _save_data(train_df, outputs["train_data"])
    _save_data(test_df, outputs["test_data"])
    return f"create_color_indicators: created {len(frequent_colors)} color indicator columns"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract color features (color_count, is_multicolor)",
    tags=["preprocessing", "feature-engineering", "shelter", "generic"],
    version="1.0.0",
)
def extract_color_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    color_column: str = "Color",
) -> str:
    """Extract features from color column.

    Creates: color_count (number of colors), is_multicolor (binary).
    Derived from top-scoring solution notebooks which create color indicator variables.

    G4 Compliance: Column name parameterized.
    """
    df = _load_data(inputs["data"])

    if color_column in df.columns:
        df["color_count"] = df[color_column].str.count("/").fillna(0).astype(int) + 1
        df["is_multicolor"] = (df["color_count"] > 1).astype(int)

    _save_data(df, outputs["data"])
    return f"extract_color_features: created color_count, is_multicolor"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract datetime features for shelter data (year, month, weekday, hour, is_weekend, datetime_numeric)",
    tags=["preprocessing", "feature-engineering", "temporal", "shelter", "generic"],
    version="2.0.0",
)
def extract_shelter_datetime(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    datetime_column: str = "DateTime",
) -> str:
    """Extract datetime components for shelter data.

    Creates: year, month, dayofweek, hour, is_weekend, datetime_numeric.
    All three top solution notebooks extract these same temporal features.
    Added: year and datetime_numeric (epoch time) based on top solution analysis.

    G4 Compliance: Column name parameterized.
    """
    df = _load_data(inputs["data"])

    if datetime_column in df.columns:
        dt = pd.to_datetime(df[datetime_column])
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["dayofweek"] = dt.dt.dayofweek
        df["hour"] = dt.dt.hour
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        # Convert datetime to numeric (epoch seconds) - used by top solutions
        df["datetime_numeric"] = dt.astype(np.int64) // 10**9

    _save_data(df, outputs["data"])
    return f"extract_shelter_datetime: created year, month, dayofweek, hour, is_weekend, datetime_numeric"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={
        "data": {"format": "csv"},
        "encoder": {"format": "json"},
    },
    description="Alphabetically encode target column for consistent multiclass ordering",
    tags=["preprocessing", "encoding", "classification", "generic"],
    version="1.0.0",
)
def encode_target_sorted(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "OutcomeType",
) -> str:
    """Alphabetically encode target column to ensure consistent class ordering.

    Maps sorted unique values to 0..N-1 (alphabetical order).
    Saves the class-to-integer mapping as a JSON encoder artifact.

    This ensures class order matches Kaggle submission column order
    (which is alphabetical: Adoption, Died, Euthanasia, Return_to_owner, Transfer).

    G4 Compliance: Column name parameterized.
    """
    df = _load_data(inputs["data"])

    if target_column in df.columns:
        classes = sorted(df[target_column].dropna().unique())
        class_map = {cls: i for i, cls in enumerate(classes)}
        df[target_column] = df[target_column].map(class_map)

        # Save encoder mapping
        encoder_path = outputs.get("encoder")
        if encoder_path:
            os.makedirs(os.path.dirname(encoder_path) or ".", exist_ok=True)
            with open(encoder_path, "w") as f:
                json.dump({"class_map": class_map, "classes": classes}, f, indent=2)

    _save_data(df, outputs["data"])
    return f"encode_target_sorted: encoded {target_column} with {len(classes)} classes: {classes}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Domain-specific services
    "parse_age_to_days": parse_age_to_days,
    "parse_sex_outcome": parse_sex_outcome,
    "extract_name_features": extract_name_features,
    "extract_breed_features": extract_breed_features,
    "extract_color_features": extract_color_features,
    "extract_shelter_datetime": extract_shelter_datetime,
    "encode_target_sorted": encode_target_sorted,
    # NEW: Top solution techniques - breed/color indicator variables
    "create_breed_indicators": create_breed_indicators,
    "create_color_indicators": create_color_indicators,
    # Reused from common modules
    "split_data": split_data,
    "drop_columns": drop_columns,
    "encode_all_categorical": encode_all_categorical,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "train_xgboost_classifier": train_xgboost_classifier,
    "predict_classifier": predict_classifier,
    "predict_multiclass_submission": predict_multiclass_submission,
}