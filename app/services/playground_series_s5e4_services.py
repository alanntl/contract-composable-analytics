"""
Playground Series S5E4 - Predict Podcast Listening Time - Contract-Composable Analytics Services
========================================================================
Competition: https://www.kaggle.com/competitions/playground-series-s5e4
Problem Type: Regression
Target: Listening_Time_minutes
ID Column: id

Predict podcast listening time based on episode and host attributes.
Standard tabular regression with categorical and numeric features.

Competition-specific services (from solution notebooks):
- create_podcast_features: Ratio and derived features from episode/host data
- extract_episode_number: Extract numeric episode number from Episode_Title
- combine_train_with_external: Combine train data with external podcast dataset
- apply_data_leak_fixes: Post-processing to exploit known data leaks (solution 02)
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

try:
    from services.regression_services import train_lightgbm_regressor, predict_regressor
    from services.preprocessing_services import split_data, fill_missing, drop_columns, engineer_features
    from services.spaceship_titanic_services import fill_missing_numeric, label_encode_columns
except ImportError:
    from regression_services import train_lightgbm_regressor, predict_regressor
    from preprocessing_services import split_data, fill_missing, drop_columns, engineer_features
    from spaceship_titanic_services import fill_missing_numeric, label_encode_columns


# =============================================================================
# COMPETITION-SPECIFIC SERVICES (from solution notebooks)
# =============================================================================

@contract(
    inputs={"train_data": {"format": "csv", "required": True}, "external_data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Combine train data with external podcast dataset for augmented training",
    tags=["preprocessing", "data-augmentation", "generic"],
    version="1.0.0"
)
def combine_train_with_external(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Listening_Time_minutes",
    id_column: str = "id",
) -> str:
    """
    Combine competition train data with external podcast dataset.

    Based on solution 02 (greysky) - uses podcast_dataset.csv for augmentation.
    Adds unique ID values for external data starting from 1,000,000.

    Args:
        target_column: Target column name
        id_column: ID column name
    """
    train = pd.read_csv(inputs["train_data"])
    external = pd.read_csv(inputs["external_data"])

    # Filter external data - remove rows without target
    external = external.dropna(subset=[target_column])

    # Assign unique IDs to external data
    external[id_column] = range(1000000, 1000000 + len(external))

    # Ensure same columns
    common_cols = [c for c in train.columns if c in external.columns]
    train = train[common_cols]
    external = external[common_cols]

    # Combine
    combined = pd.concat([train, external], ignore_index=True)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    combined.to_csv(outputs["data"], index=False)

    return f"combine_train_with_external: combined {len(train)} train + {len(external)} external = {len(combined)} total"


@contract(
    inputs={"predictions": {"format": "csv", "required": True}, "test_data": {"format": "csv", "required": True}, "train_data": {"format": "csv", "required": True}},
    outputs={"predictions": {"format": "csv"}},
    description="Apply data leak fixes based on known patterns from solution notebooks",
    tags=["post-processing", "prediction", "generic"],
    version="1.0.0"
)
def apply_data_leak_fixes(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    target_column: str = "Listening_Time_minutes",
    length_column: str = "Episode_Length_minutes",
    ads_column: str = "Number_of_Ads",
) -> str:
    """
    Apply data leak exploitation from solution 02 (greysky).

    Known data leaks:
    1. Episode_Length_minutes with >2 decimal digits → listening_time = length * 0.9607
    2. Number_of_Ads > 3 → listening_time = ads * 1.0588
    3. Feature matching on key columns with train data

    Args:
        id_column: ID column name
        target_column: Target/prediction column name
        length_column: Episode length column
        ads_column: Number of ads column
    """
    predictions = pd.read_csv(inputs["predictions"])
    test = pd.read_csv(inputs["test_data"])
    train = pd.read_csv(inputs["train_data"])

    fixes_applied = 0

    # Data Leak 1: More than 2 decimal digits in Episode_Length_minutes
    test_dec = test.copy()
    if length_column in test.columns:
        # Read as string to count decimal digits
        test_str = pd.read_csv(inputs["test_data"], dtype={length_column: str})
        def count_decimal_digits(s):
            if pd.isna(s):
                return 0
            s = str(s)
            if '.' in s:
                return len(s.split('.')[1])
            return 0

        test_str['decimal_digits'] = test_str[length_column].apply(count_decimal_digits)
        test_str[length_column] = pd.to_numeric(test_str[length_column], errors='coerce')

        mask_dec = test_str['decimal_digits'] > 2
        if mask_dec.sum() > 0:
            leak_ids = test_str.loc[mask_dec, id_column].values
            leak_values = test_str.loc[mask_dec, length_column].values * 0.9607
            for lid, lval in zip(leak_ids, leak_values):
                predictions.loc[predictions[id_column] == lid, target_column] = lval
                fixes_applied += 1

    # Data Leak 2: Abnormal Number_of_Ads values (> 3)
    if ads_column in test.columns:
        test_ads = test.copy()
        test_ads.loc[test_ads[ads_column] > 103.91, ads_column] = 0.0  # Cap extreme values
        mask_ads = test_ads[ads_column] > 3
        if mask_ads.sum() > 0:
            leak_ids = test_ads.loc[mask_ads, id_column].values
            leak_values = test_ads.loc[mask_ads, ads_column].values * 1.0588
            for lid, lval in zip(leak_ids, leak_values):
                predictions.loc[predictions[id_column] == lid, target_column] = lval
                fixes_applied += 1

    # Data Leak 3: Feature matching with train data
    # Create derived features for matching
    train_fe = train.copy()
    test_fe = test.copy()

    if length_column in train_fe.columns:
        train_fe['ELen_Int'] = np.floor(train_fe[length_column])
        test_fe['ELen_Int'] = np.floor(test_fe[length_column])

    if 'Host_Popularity_percentage' in train_fe.columns:
        train_fe['HPperc_Int'] = np.floor(train_fe['Host_Popularity_percentage'])
        test_fe['HPperc_Int'] = np.floor(test_fe['Host_Popularity_percentage'])

    # Match on key feature combinations
    cols_to_compare = ['Publication_Day', 'Guest_Popularity_percentage', 'ELen_Int', 'HPperc_Int']
    cols_available = [c for c in cols_to_compare if c in train_fe.columns and c in test_fe.columns]

    if len(cols_available) >= 2 and target_column in train_fe.columns:
        test_match = test_fe.dropna(subset=['Guest_Popularity_percentage', length_column] if 'Guest_Popularity_percentage' in test_fe.columns else [])

        if len(test_match) > 0:
            train_subset = train_fe[cols_available + [target_column]].drop_duplicates()
            merged = test_match.merge(train_subset, on=cols_available, how='inner')

            if len(merged) > 0:
                mean_values = merged.groupby(id_column)[target_column].mean()
                for lid in mean_values.index:
                    predictions.loc[predictions[id_column] == lid, target_column] = mean_values[lid]
                    fixes_applied += 1

    # Clip predictions to non-negative
    predictions[target_column] = predictions[target_column].clip(lower=0)

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    predictions.to_csv(outputs["predictions"], index=False)

    return f"apply_data_leak_fixes: applied {fixes_applied} leak-based fixes"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract numeric episode number from episode title column",
    tags=["preprocessing", "feature-engineering", "generic"],
    version="1.0.0"
)
def extract_episode_number(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    title_column: str = "Episode_Title",
    output_column: str = "Episode_Num",
) -> str:
    """
    Extract numeric episode number from a title like 'Episode 98'.

    Works with any column containing 'prefix number' format.
    From solution notebook 03 (masaishi).

    Args:
        title_column: Column containing episode titles
        output_column: Name for the extracted numeric column
    """
    df = pd.read_csv(inputs["data"])

    if title_column in df.columns:
        df[output_column] = df[title_column].str.extract(r'(\d+)').astype(float)
        df[output_column] = df[output_column].fillna(0).astype(int)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"extract_episode_number: extracted from {title_column} -> {output_column}"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create ratio and derived features for podcast/content data",
    tags=["feature-engineering", "generic", "regression"],
    version="1.0.0"
)
def create_podcast_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    length_column: str = "Episode_Length_minutes",
    ads_column: str = "Number_of_Ads",
    host_pop_column: str = "Host_Popularity_percentage",
    guest_pop_column: str = "Guest_Popularity_percentage",
) -> str:
    """
    Create ratio and derived features from podcast/content data.

    Based on top solution notebooks (masaishi, greysky):
    - Length per ad, host popularity, guest popularity ratios
    - Integer and decimal parts of episode length
    - Host popularity integer and decimal parts

    Args:
        length_column: Episode/content length column
        ads_column: Number of ads column
        host_pop_column: Host popularity percentage column
        guest_pop_column: Guest popularity percentage column
    """
    df = pd.read_csv(inputs["data"])

    # Ratio features (from solution 03 - masaishi)
    if length_column in df.columns and ads_column in df.columns:
        df['Length_per_Ads'] = (df[length_column] / (df[ads_column] + 1)).fillna(0)

    if length_column in df.columns and host_pop_column in df.columns:
        df['Length_per_Host'] = (df[length_column] / (df[host_pop_column] + 1)).fillna(0)

    if length_column in df.columns and guest_pop_column in df.columns:
        df['Length_per_Guest'] = (df[length_column] / (df[guest_pop_column] + 1)).fillna(0)

    # Integer/decimal split features (from solution 03 - masaishi)
    if length_column in df.columns:
        df['ELen_Int'] = np.floor(df[length_column]).fillna(0)
        df['ELen_Dec'] = (df[length_column] - df['ELen_Int']).fillna(0)

    if host_pop_column in df.columns:
        df['HPperc_Int'] = np.floor(df[host_pop_column]).fillna(0)
        df['HPperc_Dec'] = (df[host_pop_column] - df['HPperc_Int']).fillna(0)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    new_cols = ['Length_per_Ads', 'Length_per_Host', 'Length_per_Guest',
                'ELen_Int', 'ELen_Dec', 'HPperc_Int', 'HPperc_Dec']
    return f"create_podcast_features: created {len(new_cols)} features"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific services
    "extract_episode_number": extract_episode_number,
    "create_podcast_features": create_podcast_features,
    "combine_train_with_external": combine_train_with_external,
    "apply_data_leak_fixes": apply_data_leak_fixes,
    # Imported reusable services
    "fill_missing_numeric": fill_missing_numeric,
    "fill_missing": fill_missing,
    "label_encode_columns": label_encode_columns,
    "drop_columns": drop_columns,
    "split_data": split_data,
    "engineer_features": engineer_features,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "predict_regressor": predict_regressor,
}