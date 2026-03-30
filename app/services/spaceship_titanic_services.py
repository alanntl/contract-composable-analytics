"""
Spaceship Titanic - SLEGO Services
==================================
Competition: https://www.kaggle.com/competitions/spaceship-titanic
Problem Type: Binary Classification
Target: Transported (True/False)
ID Column: PassengerId

Predict which passengers were transported to another dimension.
Similar to Titanic but with spending features and cabin parsing.

Key insights from solution notebooks:
- Parse Cabin into Deck/Number/Side
- Parse PassengerId into Group/Number
- Spending features (RoomService, FoodCourt, etc.) are important
- CryoSleep passengers have 0 spending
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
    from services.preprocessing_services import split_data, create_submission
    from services.bike_sharing_services import drop_columns
except ImportError:
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import split_data, create_submission
    from bike_sharing_services import drop_columns


# =============================================================================
# GENERIC REUSABLE SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Parse delimited column into multiple columns",
    tags=["preprocessing", "feature-engineering", "parsing", "generic"],
    version="1.0.0"
)
def parse_delimited_column(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    source_column: str = "Cabin",
    delimiter: str = "/",
    output_columns: List[str] = None,
    fill_missing: str = "Unknown",
) -> str:
    """
    Parse a delimited column into multiple columns.

    E.g., Parse 'B/0/P' into Deck='B', CabinNum='0', Side='P'.

    Args:
        source_column: Column containing delimited values
        delimiter: Separator character
        output_columns: Names for the output columns (must match number of parts)
        fill_missing: Value for missing/null entries
    """
    df = pd.read_csv(inputs["data"])
    output_columns = output_columns or [f"{source_column}_part{i}" for i in range(3)]

    # Split the column
    split_data = df[source_column].fillna(f"{fill_missing}{delimiter}{fill_missing}{delimiter}{fill_missing}")
    split_df = split_data.str.split(delimiter, expand=True)

    # Handle case where split produces fewer columns than expected
    for i, col_name in enumerate(output_columns):
        if i < split_df.shape[1]:
            df[col_name] = split_df[i].replace('', fill_missing).fillna(fill_missing)
        else:
            df[col_name] = fill_missing

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"parse_delimited_column: {source_column} → {output_columns}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create aggregate features from multiple numeric columns",
    tags=["preprocessing", "feature-engineering", "aggregation", "generic"],
    version="1.0.0"
)
def aggregate_numeric_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    output_prefix: str = "spending",
    operations: List[str] = None,
) -> str:
    """
    Create aggregate features (sum, mean, etc.) from multiple numeric columns.

    E.g., Create TotalSpending, MeanSpending from RoomService, FoodCourt, etc.

    Args:
        columns: Columns to aggregate
        output_prefix: Prefix for output column names
        operations: List of operations ('sum', 'mean', 'max', 'min', 'std')
    """
    df = pd.read_csv(inputs["data"])
    columns = columns or ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    operations = operations or ['sum', 'mean']

    # Filter to existing columns
    existing_cols = [c for c in columns if c in df.columns]

    if existing_cols:
        for op in operations:
            if op == 'sum':
                df[f"{output_prefix}_total"] = df[existing_cols].sum(axis=1)
            elif op == 'mean':
                df[f"{output_prefix}_mean"] = df[existing_cols].mean(axis=1)
            elif op == 'max':
                df[f"{output_prefix}_max"] = df[existing_cols].max(axis=1)
            elif op == 'min':
                df[f"{output_prefix}_min"] = df[existing_cols].min(axis=1)
            elif op == 'std':
                df[f"{output_prefix}_std"] = df[existing_cols].std(axis=1)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"aggregate_numeric_columns: created {len(operations)} features from {len(existing_cols)} columns"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create group ID from composite ID column",
    tags=["preprocessing", "feature-engineering", "generic"],
    version="1.0.0"
)
def extract_group_from_id(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "PassengerId",
    delimiter: str = "_",
    group_column: str = "GroupId",
    position_column: str = "GroupPosition",
) -> str:
    """
    Extract group and position from composite ID.

    E.g., '0001_01' → GroupId='0001', GroupPosition='01'

    Args:
        id_column: Column containing composite IDs
        delimiter: Separator character
        group_column: Output column for group part
        position_column: Output column for position part
    """
    df = pd.read_csv(inputs["data"])

    split_id = df[id_column].astype(str).str.split(delimiter, expand=True)
    df[group_column] = split_id[0] if split_id.shape[1] > 0 else '0'
    df[position_column] = split_id[1].astype(int) if split_id.shape[1] > 1 else 0

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"extract_group_from_id: {df[group_column].nunique()} unique groups"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Convert boolean columns to integer (0/1)",
    tags=["preprocessing", "encoding", "generic"],
    version="1.0.0"
)
def encode_boolean_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    true_values: List[str] = None,
) -> str:
    """
    Convert boolean columns to integer 0/1.

    Args:
        columns: Columns to convert (if None, detects boolean-like columns)
        true_values: Values to treat as True (default: ['True', 'true', '1', 'Yes', 'yes'])
    """
    df = pd.read_csv(inputs["data"])
    true_values = true_values or ['True', 'true', '1', 'Yes', 'yes', True, 1]

    if columns is None:
        # Auto-detect boolean-like columns
        columns = [c for c in df.columns
                   if df[c].dropna().astype(str).isin(['True', 'False', 'true', 'false', '0', '1']).all()]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).isin([str(v) for v in true_values]).astype(int)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"encode_boolean_columns: encoded {len(columns)} columns"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Fill missing numeric values with specified strategy",
    tags=["preprocessing", "imputation", "generic"],
    version="1.0.0"
)
def fill_missing_numeric(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    strategy: str = "median",
    fill_value: float = 0,
) -> str:
    """
    Fill missing values in numeric columns.

    Args:
        columns: Columns to fill (if None, all numeric columns)
        strategy: 'median', 'mean', 'zero', or 'value'
        fill_value: Value to use if strategy is 'value'
    """
    df = pd.read_csv(inputs["data"])

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    filled = 0
    for col in columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "zero":
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(fill_value)
                filled += missing

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"fill_missing_numeric: filled {filled} values using {strategy}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Label encode categorical columns to integers",
    tags=["preprocessing", "encoding", "generic"],
    version="1.0.0"
)
def label_encode_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    max_categories: int = 50,
) -> str:
    """
    Label encode categorical columns to integers.

    Args:
        columns: Columns to encode (if None, all object columns)
        max_categories: Skip columns with more unique values
    """
    df = pd.read_csv(inputs["data"])

    if columns is None:
        columns = [c for c in df.select_dtypes(include=['object']).columns
                   if df[c].nunique() <= max_categories]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna('Missing').astype('category').cat.codes

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"label_encode_columns: encoded {len(columns)} columns"


# =============================================================================
# ADDITIONAL FEATURE ENGINEERING SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Add group size and solo traveler flag from a group column",
    tags=["feature-engineering", "grouping", "generic"],
    version="1.0.0"
)
def add_group_size(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    group_column: str = "GroupId",
    size_column: str = "GroupSize",
    solo_column: str = "IsSolo",
) -> str:
    """Add group size count and solo traveler flag."""
    df = pd.read_csv(inputs["data"])
    if group_column in df.columns:
        df[size_column] = df.groupby(group_column)[group_column].transform('count')
        df[solo_column] = (df[size_column] == 1).astype(int)
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    solo_count = df[solo_column].sum() if solo_column in df.columns else 0
    return f"add_group_size: max_group={df[size_column].max()}, solo={solo_count}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Add binary flag for zero total across specified columns",
    tags=["feature-engineering", "binary", "generic"],
    version="1.0.0"
)
def add_zero_spending_flag(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    output_column: str = "NoSpending",
) -> str:
    """Flag rows where all specified numeric columns are zero or NaN."""
    df = pd.read_csv(inputs["data"])
    columns = columns or ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    existing = [c for c in columns if c in df.columns]
    if existing:
        df[output_column] = (df[existing].fillna(0).sum(axis=1) == 0).astype(int)
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    count = df[output_column].sum() if output_column in df.columns else 0
    return f"add_zero_spending_flag: {count} zero-spending rows"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Apply log1p transform to specified numeric columns",
    tags=["feature-engineering", "transform", "generic"],
    version="1.0.0"
)
def log_transform_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = None,
    prefix: str = "log_",
) -> str:
    """Apply log1p(x) transform, creating new columns with prefix."""
    df = pd.read_csv(inputs["data"])
    columns = columns or ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    existing = [c for c in columns if c in df.columns]
    for col in existing:
        df[f"{prefix}{col}"] = np.log1p(df[col].fillna(0))
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"log_transform_columns: created {len(existing)} log features"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create binned age feature from continuous age column",
    tags=["feature-engineering", "binning", "generic"],
    version="1.0.0"
)
def bin_age_column(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    age_column: str = "Age",
    output_column: str = "AgeBin",
    bins: List[float] = None,
    labels: List[int] = None,
) -> str:
    """Bin age into categories (child, teen, young_adult, adult, senior)."""
    df = pd.read_csv(inputs["data"])
    bins = bins or [0, 12, 18, 25, 50, 100]
    labels = labels or [0, 1, 2, 3, 4]
    if age_column in df.columns:
        df[output_column] = pd.cut(
            df[age_column].fillna(df[age_column].median()),
            bins=bins, labels=labels, include_lowest=True,
        ).astype(int)
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"bin_age_column: {df[output_column].value_counts().to_dict()}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Generic reusable services
    "parse_delimited_column": parse_delimited_column,
    "aggregate_numeric_columns": aggregate_numeric_columns,
    "extract_group_from_id": extract_group_from_id,
    "encode_boolean_columns": encode_boolean_columns,
    "fill_missing_numeric": fill_missing_numeric,
    "label_encode_columns": label_encode_columns,
    # New feature engineering services
    "add_group_size": add_group_size,
    "add_zero_spending_flag": add_zero_spending_flag,
    "log_transform_columns": log_transform_columns,
    "bin_age_column": bin_age_column,
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
        "service": "parse_delimited_column",
        "inputs": {"data": "spaceship-titanic/datasets/train.csv"},
        "outputs": {"data": "spaceship-titanic/artifacts/train_01_cabin.csv"},
        "params": {
            "source_column": "Cabin",
            "delimiter": "/",
            "output_columns": ["Deck", "CabinNum", "Side"]
        },
        "module": "spaceship_titanic_services"
    },
    {
        "service": "extract_group_from_id",
        "inputs": {"data": "spaceship-titanic/artifacts/train_01_cabin.csv"},
        "outputs": {"data": "spaceship-titanic/artifacts/train_02_group.csv"},
        "params": {"id_column": "PassengerId"},
        "module": "spaceship_titanic_services"
    },
    {
        "service": "aggregate_numeric_columns",
        "inputs": {"data": "spaceship-titanic/artifacts/train_02_group.csv"},
        "outputs": {"data": "spaceship-titanic/artifacts/train_03_spending.csv"},
        "params": {
            "columns": ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"],
            "output_prefix": "spending",
            "operations": ["sum", "mean"]
        },
        "module": "spaceship_titanic_services"
    },
    {
        "service": "encode_boolean_columns",
        "inputs": {"data": "spaceship-titanic/artifacts/train_03_spending.csv"},
        "outputs": {"data": "spaceship-titanic/artifacts/train_04_bool.csv"},
        "params": {"columns": ["CryoSleep", "VIP", "Transported"]},
        "module": "spaceship_titanic_services"
    },
    {
        "service": "fill_missing_numeric",
        "inputs": {"data": "spaceship-titanic/artifacts/train_04_bool.csv"},
        "outputs": {"data": "spaceship-titanic/artifacts/train_05_fillnum.csv"},
        "params": {"strategy": "median"},
        "module": "spaceship_titanic_services"
    },
    {
        "service": "label_encode_columns",
        "inputs": {"data": "spaceship-titanic/artifacts/train_05_fillnum.csv"},
        "outputs": {"data": "spaceship-titanic/artifacts/train_06_encoded.csv"},
        "params": {"columns": ["HomePlanet", "Destination", "Deck", "Side"]},
        "module": "spaceship_titanic_services"
    },
    {
        "service": "drop_columns",
        "inputs": {"data": "spaceship-titanic/artifacts/train_06_encoded.csv"},
        "outputs": {"data": "spaceship-titanic/artifacts/train_final.csv"},
        "params": {"columns": ["PassengerId", "Cabin", "Name", "CabinNum", "GroupId"]},
        "module": "spaceship_titanic_services"
    },
    {
        "service": "split_data",
        "inputs": {"data": "spaceship-titanic/artifacts/train_final.csv"},
        "outputs": {
            "train_data": "spaceship-titanic/artifacts/train_split.csv",
            "valid_data": "spaceship-titanic/artifacts/valid_split.csv"
        },
        "params": {"stratify_column": "Transported", "test_size": 0.2, "random_state": 42},
        "module": "spaceship_titanic_services"
    },
    {
        "service": "train_lightgbm_classifier",
        "inputs": {
            "train_data": "spaceship-titanic/artifacts/train_split.csv",
            "valid_data": "spaceship-titanic/artifacts/valid_split.csv"
        },
        "outputs": {
            "model": "spaceship-titanic/artifacts/model.pkl",
            "metrics": "spaceship-titanic/artifacts/metrics.json"
        },
        "params": {
            "label_column": "Transported",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31
        },
        "module": "spaceship_titanic_services"
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

    print(f"\n--- Spaceship Titanic Pipeline (Base: {storage_path}) ---")
    run_pipeline(storage_path)
