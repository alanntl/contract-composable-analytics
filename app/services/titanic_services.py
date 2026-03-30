"""
Titanic: Machine Learning from Disaster - Contract-Composable Analytics Services
=========================================================
Competition: https://www.kaggle.com/competitions/titanic
Problem Type: Binary Classification
Target: Survived (0=No, 1=Yes)
ID Column: PassengerId

Classic beginner competition. Key insights from solution notebooks:
- Family features (FamilySize, IsAlone) improve predictions
- Title extraction from Name is valuable
- Age imputation using median by Pclass+Sex
- Cabin deck extraction (first letter)

Services follow reusability principles - all can be used across competitions.
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

# Import from common modules
try:
    from services.preprocessing_services import split_data, create_submission
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.bike_sharing_services import drop_columns
except ImportError:
    from preprocessing_services import split_data, create_submission
    from classification_services import train_lightgbm_classifier, predict_classifier
    from bike_sharing_services import drop_columns


# =============================================================================
# GENERIC REUSABLE SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract title from name column (Mr, Mrs, Miss, etc.)",
    tags=["preprocessing", "feature-engineering", "text", "generic"],
    version="1.0.0"
)
def extract_title_from_name(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    name_column: str = "Name",
    output_column: str = "Title",
    rare_threshold: int = 10,
) -> str:
    """
    Extract title (Mr, Mrs, Miss, Master, etc.) from name column.

    Works with any dataset that has names in format "LastName, Title. FirstName"
    Common rare titles are mapped to standard categories.

    Args:
        name_column: Column containing full names
        output_column: Name for the extracted title column
        rare_threshold: Titles with fewer occurrences become 'Rare'
    """
    df = pd.read_csv(inputs["data"])

    # Extract title between comma and period
    df[output_column] = df[name_column].str.extract(r',\s*([^.]+)\.', expand=False)

    # Map rare titles to common categories
    title_mapping = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Dona': 'Rare',
        'Dr': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Col': 'Rare',
        'Major': 'Rare', 'Capt': 'Rare'
    }
    df[output_column] = df[output_column].replace(title_mapping)

    # Mark remaining rare titles
    title_counts = df[output_column].value_counts()
    rare_titles = title_counts[title_counts < rare_threshold].index
    df.loc[df[output_column].isin(rare_titles), output_column] = 'Rare'

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"extract_title_from_name: extracted {df[output_column].nunique()} unique titles"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create family size and is_alone features from sibling/spouse and parent/child columns",
    tags=["preprocessing", "feature-engineering", "generic"],
    version="1.0.0"
)
def create_family_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    sibsp_column: str = "SibSp",
    parch_column: str = "Parch",
    family_size_column: str = "FamilySize",
    is_alone_column: str = "IsAlone",
) -> str:
    """
    Create family-related features from sibling/spouse and parent/child counts.

    Works with Titanic, Spaceship Titanic, or any dataset with family relationship columns.

    Args:
        sibsp_column: Column with sibling/spouse count
        parch_column: Column with parent/child count
        family_size_column: Output column for total family size
        is_alone_column: Output column for alone indicator (1=alone, 0=has family)
    """
    df = pd.read_csv(inputs["data"])

    df[family_size_column] = df[sibsp_column] + df[parch_column] + 1
    df[is_alone_column] = (df[family_size_column] == 1).astype(int)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"create_family_features: FamilySize range [{df[family_size_column].min()}-{df[family_size_column].max()}], {df[is_alone_column].sum()} alone"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract first character from a column (e.g., deck from cabin)",
    tags=["preprocessing", "feature-engineering", "generic"],
    version="1.0.0"
)
def extract_first_char(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    source_column: str = "Cabin",
    output_column: str = "Deck",
    fill_missing: str = "Unknown",
) -> str:
    """
    Extract first character from a string column.

    Common use: Extract deck letter from cabin number (e.g., 'C85' -> 'C').
    Works with any column where the first character is meaningful.

    Args:
        source_column: Column to extract from
        output_column: Name for the output column
        fill_missing: Value for missing/null entries
    """
    df = pd.read_csv(inputs["data"])

    df[output_column] = df[source_column].fillna(fill_missing).astype(str).str[0]
    df.loc[df[output_column] == '', output_column] = fill_missing

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"extract_first_char: {df[output_column].nunique()} unique values from {source_column}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Impute missing values using group median",
    tags=["preprocessing", "imputation", "generic"],
    version="1.0.0"
)
def impute_by_group_median(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Age",
    group_columns: List[str] = None,
    fallback_value: Optional[float] = None,
) -> str:
    """
    Impute missing values using median of specified groups.

    E.g., impute Age using median Age per (Pclass, Sex) combination.
    Falls back to global median if group median unavailable.

    Args:
        target_column: Column with missing values to impute
        group_columns: Columns to group by for calculating median
        fallback_value: Value to use if median calculation fails (default: global median)
    """
    df = pd.read_csv(inputs["data"])
    group_columns = group_columns or ["Pclass", "Sex"]

    missing_before = df[target_column].isna().sum()

    if missing_before > 0:
        global_median = df[target_column].median()
        fallback = fallback_value if fallback_value is not None else global_median

        # Calculate group medians
        group_medians = df.groupby(group_columns)[target_column].transform('median')

        # Fill missing with group median, then fallback
        df[target_column] = df[target_column].fillna(group_medians).fillna(fallback)

    missing_after = df[target_column].isna().sum()

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"impute_by_group_median: {missing_before} → {missing_after} missing in {target_column}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Fill missing categorical values with mode or specified value",
    tags=["preprocessing", "imputation", "generic"],
    version="1.0.0"
)
def fill_missing_categorical(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    fill_value: str = "mode",
) -> str:
    """
    Fill missing values in categorical columns.

    Args:
        columns: List of columns to fill (if None, fills all object columns)
        fill_value: "mode" to use most common value, or a specific string value
    """
    df = pd.read_csv(inputs["data"])

    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    filled = 0
    for col in columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                if fill_value == "mode":
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                else:
                    df[col] = df[col].fillna(fill_value)
                filled += missing

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"fill_missing_categorical: filled {filled} missing values in {len(columns)} columns"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="One-hot encode specified categorical columns",
    tags=["preprocessing", "encoding", "generic"],
    version="1.0.0"
)
def one_hot_encode(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    drop_first: bool = True,
    max_categories: int = 20,
) -> str:
    """
    One-hot encode categorical columns.

    Args:
        columns: Columns to encode (if None, encodes all object columns with < max_categories)
        drop_first: Drop first category to avoid multicollinearity
        max_categories: Skip columns with more unique values than this
    """
    df = pd.read_csv(inputs["data"])

    if columns is None:
        columns = [c for c in df.select_dtypes(include=['object']).columns
                   if df[c].nunique() <= max_categories]

    original_cols = len(df.columns)
    df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
    new_cols = len(df.columns)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"one_hot_encode: {original_cols} → {new_cols} columns"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True}
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"}
    },
    description="Combine train and test for consistent preprocessing, then split back",
    tags=["preprocessing", "generic"],
    version="1.0.0"
)
def combine_train_test(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Survived",
    marker_column: str = "_is_train",
) -> str:
    """
    Combine train and test datasets with a marker column.

    Useful for ensuring consistent preprocessing (encoding, scaling) across both sets.
    The marker column allows splitting them back after preprocessing.

    Args:
        target_column: Target column (will be NaN for test data)
        marker_column: Name for the train/test indicator column
    """
    train = pd.read_csv(inputs["train_data"])
    test = pd.read_csv(inputs["test_data"])

    train[marker_column] = 1
    test[marker_column] = 0

    if target_column not in test.columns:
        test[target_column] = np.nan

    combined = pd.concat([train, test], ignore_index=True)

    # Split back
    train_out = combined[combined[marker_column] == 1].drop(columns=[marker_column])
    test_out = combined[combined[marker_column] == 0].drop(columns=[marker_column, target_column])

    os.makedirs(os.path.dirname(outputs["train_data"]) or ".", exist_ok=True)
    train_out.to_csv(outputs["train_data"], index=False)
    test_out.to_csv(outputs["test_data"], index=False)

    return f"combine_train_test: train={len(train_out)}, test={len(test_out)}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True}
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"}
    },
    description="One-hot encode train and test together for consistent columns",
    tags=["preprocessing", "encoding", "generic"],
    version="1.0.0"
)
def one_hot_encode_combined(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Survived",
    columns: List[str] = None,
    drop_first: bool = True,
    max_categories: int = 20,
) -> str:
    """
    One-hot encode train and test together for consistent columns.

    This ensures both datasets have identical columns after encoding,
    preventing column mismatch issues during prediction.
    """
    train = pd.read_csv(inputs["train_data"])
    test = pd.read_csv(inputs["test_data"])

    # Mark train/test
    train["_is_train"] = 1
    test["_is_train"] = 0

    # Store target and remove from test (to avoid NaN issues)
    target_values = train[target_column].copy() if target_column in train.columns else None
    if target_column in test.columns:
        test = test.drop(columns=[target_column])
    if target_column in train.columns:
        train = train.drop(columns=[target_column])

    # Combine
    combined = pd.concat([train, test], ignore_index=True)

    # Determine columns to encode
    if columns is None:
        columns = [c for c in combined.select_dtypes(include=['object']).columns
                   if combined[c].nunique() <= max_categories and c != "_is_train"]

    original_cols = len(combined.columns)
    combined = pd.get_dummies(combined, columns=columns, drop_first=drop_first)
    new_cols = len(combined.columns)

    # Split back
    train_out = combined[combined["_is_train"] == 1].drop(columns=["_is_train"])
    test_out = combined[combined["_is_train"] == 0].drop(columns=["_is_train"])

    # Add target back to train
    if target_values is not None:
        train_out[target_column] = target_values.values

    os.makedirs(os.path.dirname(outputs["train_data"]) or ".", exist_ok=True)
    train_out.to_csv(outputs["train_data"], index=False)
    test_out.to_csv(outputs["test_data"], index=False)

    return f"one_hot_encode_combined: {original_cols} → {new_cols} columns, train={len(train_out)}, test={len(test_out)}"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True}
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"}
    },
    description="Fill missing Fare with median from combined data",
    tags=["preprocessing", "imputation", "generic"],
    version="1.0.0"
)
def fill_missing_fare(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    fare_column: str = "Fare",
) -> str:
    """
    Fill missing Fare values with median from combined train+test.
    """
    train = pd.read_csv(inputs["train_data"])
    test = pd.read_csv(inputs["test_data"])

    # Calculate median from combined data
    all_fare = pd.concat([train[fare_column], test[fare_column]])
    median_fare = all_fare.median()

    train_filled = train[fare_column].isna().sum()
    test_filled = test[fare_column].isna().sum()

    train[fare_column] = train[fare_column].fillna(median_fare)
    test[fare_column] = test[fare_column].fillna(median_fare)

    os.makedirs(os.path.dirname(outputs["train_data"]) or ".", exist_ok=True)
    train.to_csv(outputs["train_data"], index=False)
    test.to_csv(outputs["test_data"], index=False)

    return f"fill_missing_fare: filled {train_filled} in train, {test_filled} in test (median={median_fare:.2f})"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Generic reusable services
    "extract_title_from_name": extract_title_from_name,
    "create_family_features": create_family_features,
    "extract_first_char": extract_first_char,
    "impute_by_group_median": impute_by_group_median,
    "fill_missing_categorical": fill_missing_categorical,
    "one_hot_encode": one_hot_encode,
    "combine_train_test": combine_train_test,
    "one_hot_encode_combined": one_hot_encode_combined,
    "fill_missing_fare": fill_missing_fare,
    # Imported from common modules
    "drop_columns": drop_columns,
    "split_data": split_data,
    "create_submission": create_submission,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
}


# =============================================================================
# PIPELINE SPEC
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "extract_title_from_name",
        "inputs": {"data": "titanic/datasets/train.csv"},
        "outputs": {"data": "titanic/artifacts/train_01_title.csv"},
        "params": {"name_column": "Name"},
        "module": "titanic_services"
    },
    {
        "service": "create_family_features",
        "inputs": {"data": "titanic/artifacts/train_01_title.csv"},
        "outputs": {"data": "titanic/artifacts/train_02_family.csv"},
        "module": "titanic_services"
    },
    {
        "service": "extract_first_char",
        "inputs": {"data": "titanic/artifacts/train_02_family.csv"},
        "outputs": {"data": "titanic/artifacts/train_03_deck.csv"},
        "params": {"source_column": "Cabin", "output_column": "Deck"},
        "module": "titanic_services"
    },
    {
        "service": "impute_by_group_median",
        "inputs": {"data": "titanic/artifacts/train_03_deck.csv"},
        "outputs": {"data": "titanic/artifacts/train_04_age.csv"},
        "params": {"target_column": "Age", "group_columns": ["Pclass", "Sex"]},
        "module": "titanic_services"
    },
    {
        "service": "fill_missing_categorical",
        "inputs": {"data": "titanic/artifacts/train_04_age.csv"},
        "outputs": {"data": "titanic/artifacts/train_05_fill.csv"},
        "params": {"columns": ["Embarked"]},
        "module": "titanic_services"
    },
    {
        "service": "one_hot_encode",
        "inputs": {"data": "titanic/artifacts/train_05_fill.csv"},
        "outputs": {"data": "titanic/artifacts/train_06_encoded.csv"},
        "params": {"columns": ["Sex", "Embarked", "Title", "Deck"]},
        "module": "titanic_services"
    },
    {
        "service": "drop_columns",
        "inputs": {"data": "titanic/artifacts/train_06_encoded.csv"},
        "outputs": {"data": "titanic/artifacts/train_final.csv"},
        "params": {"columns": ["Name", "Ticket", "Cabin"]},
        "module": "titanic_services"
    },
    {
        "service": "split_data",
        "inputs": {"data": "titanic/artifacts/train_final.csv"},
        "outputs": {"train_data": "titanic/artifacts/train_split.csv", "valid_data": "titanic/artifacts/valid_split.csv"},
        "params": {"stratify_column": "Survived", "test_size": 0.2, "random_state": 42},
        "module": "titanic_services"
    },
    {
        "service": "train_lightgbm_classifier",
        "inputs": {"train_data": "titanic/artifacts/train_split.csv", "valid_data": "titanic/artifacts/valid_split.csv"},
        "outputs": {"model": "titanic/artifacts/model.pkl", "metrics": "titanic/artifacts/metrics.json"},
        "params": {
            "label_column": "Survived",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15
        },
        "module": "titanic_services"
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

    print(f"\n--- Titanic Pipeline (Base: {storage_path}) ---")
    run_pipeline(storage_path)
