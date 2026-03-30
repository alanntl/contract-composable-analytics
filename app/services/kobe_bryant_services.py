"""
SLEGO Services for kobe-bryant-shot-selection competition
=========================================================
Competition: https://www.kaggle.com/competitions/kobe-bryant-shot-selection
Problem Type: Binary Classification
Target: shot_made_flag
Metric: Log Loss

Competition-specific services:
- create_shot_features: Spatial distance/angle + time pressure features (from solutions)
- extract_game_features: Game date, season year, home/away extraction
- filter_missing_target: Remove rows with missing target (train/test split in data)
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

# Import reusable services from common modules
from services.preprocessing_services import (
    split_data,
    create_submission,
    drop_columns,
    fit_encoder,
    transform_encoder,
)
from services.classification_services import (
    train_lightgbm_classifier,
    predict_classifier,
)
from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create basketball shot spatial and temporal features from location/time data",
    tags=["feature-engineering", "spatial", "temporal", "basketball"],
    version="3.0.0",
)
def create_shot_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
):
    """Create basketball shot features from location and time data.

    Derived from top Kaggle solutions (vasa137, minzai0116):
    - shot_dist_calc: Euclidean distance from basket at (0,0)
    - shot_angle: Angle from basket using arcsin (degrees)
    - angle_bin: Binned angle (0-6) as in top solutions
    - time_remaining: Combined seconds remaining in period
    - is_last_minute: Binary flag for final minute
    - is_last_5sec: Binary flag for last 5 seconds
    - in_home: Binary flag for home game (from matchup)
    - pt_class: Shot point class (0=2PT, 1=3PT near, 2=3PT far)
    """
    import math
    df = _load_data(inputs["data"])

    # Distance from basket (at 0,0)
    if "loc_x" in df.columns and "loc_y" in df.columns:
        df["shot_dist_calc"] = np.sqrt(df["loc_x"] ** 2 + df["loc_y"] ** 2)
        # Angle in degrees from basket (as in vasa137 solution)
        df["shot_angle"] = df.apply(
            lambda row: 90 if row["loc_y"] == 0 else math.degrees(math.atan(row["loc_x"] / abs(row["loc_y"]))),
            axis=1
        )
        # Angle bins (from vasa137 solution)
        df["angle_bin"] = pd.cut(df["shot_angle"], 5, labels=range(5)).astype(float).fillna(5).astype(int)
        # Adjust for restricted area and long shots
        if "shot_zone_basic" in df.columns:
            df.loc[df["shot_zone_basic"] == "Restricted Area", "angle_bin"] = 5
        if "shot_distance" in df.columns:
            df.loc[df["shot_distance"] > 30, "angle_bin"] = 6

    # Time pressure features
    if "minutes_remaining" in df.columns and "seconds_remaining" in df.columns:
        df["time_remaining"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]
        df["is_last_minute"] = (df["minutes_remaining"] == 0).astype(int)
        df["is_last_5sec"] = (
            (df["minutes_remaining"] == 0) & (df["seconds_remaining"] <= 5)
        ).astype(int)
        # Hurry shot feature (from vasa137)
        df["hurry_shot"] = ((df["time_remaining"] <= 2) & (df["shot_distance"] < 30)).astype(int) if "shot_distance" in df.columns else 0

    # Home/away indicator from matchup (solutions: in_home feature)
    if "matchup" in df.columns:
        df["in_home"] = df["matchup"].apply(
            lambda x: 0 if "@" in str(x) else 1
        )

    # Point class (2PT/3PT classification from vasa137)
    if "shot_type" in df.columns:
        df["pt_class"] = 2  # Default: 3PT above 30ft
        df.loc[(df["shot_type"].str.contains("3", na=False)) & (df["shot_distance"] < 30), "pt_class"] = 1 if "shot_distance" in df.columns else 1
        df.loc[df["shot_type"].str.contains("2", na=False), "pt_class"] = 0

    _save_data(df, outputs["data"])
    return f"create_shot_features: {len(df)} rows, added spatial/temporal/home/angle_bin/pt_class features"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract game-level features from date and season columns",
    tags=["feature-engineering", "temporal", "basketball"],
    version="2.0.0",
)
def extract_game_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "game_date",
):
    """Extract temporal features from game date and season.

    Creates:
    - game_month: Month of the game
    - game_year: Year of the game
    - season_year: Numeric season start year (from 'YYYY-YY' format)
    """
    df = _load_data(inputs["data"])

    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df["game_month"] = df[date_column].dt.month
        df["game_year"] = df[date_column].dt.year

    # Season encoding (e.g., "2000-01" → 2000)
    if "season" in df.columns:
        df["season_year"] = df["season"].str.split("-").str[0].astype(int)

    _save_data(df, outputs["data"])
    return f"extract_game_features: {len(df)} rows, added month/year/season_year"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Remove rows with missing target values for training data preparation",
    tags=["preprocessing", "filtering", "basketball"],
    version="2.0.0",
)
def filter_missing_target(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "shot_made_flag",
):
    """Remove rows where the target column is NaN.

    In the Kobe dataset, test rows have NaN shot_made_flag.
    This service filters them out for training data preparation.
    """
    df = _load_data(inputs["data"])
    n_before = len(df)

    if target_column in df.columns:
        df = df[df[target_column].notna()]

    _save_data(df, outputs["data"])
    return f"filter_missing_target: {n_before} -> {len(df)} rows"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific services
    "create_shot_features": create_shot_features,
    "extract_game_features": extract_game_features,
    "filter_missing_target": filter_missing_target,
    # Imported reusable services
    "split_data": split_data,
    "drop_columns": drop_columns,
    "fit_encoder": fit_encoder,
    "transform_encoder": transform_encoder,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}