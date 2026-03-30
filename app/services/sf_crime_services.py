"""
SLEGO Services for sf-crime competition
========================================

Competition: https://www.kaggle.com/c/sf-crime
Problem Type: Multiclass Classification (39 crime categories)
Target: Category
Metric: Multi-class Log Loss

Competition-specific services for feature engineering based on
top-scoring solution notebooks:
  - extract_datetime_features: Temporal features from Dates column
  - create_geospatial_features: Geospatial features from X/Y coordinates
  - create_address_features: Text features from Address column

All other services (encoding, splitting, training, prediction) are
imported from common modules (preprocessing_services, classification_services).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract
from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services from common modules
from services.preprocessing_services import (
    split_data, drop_columns, fit_encoder, transform_encoder,
)
from services.classification_services import (
    train_lightgbm_classifier, predict_classifier,
    predict_multiclass_submission,
)


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Extract temporal features from datetime column",
    tags=["feature-engineering", "temporal", "sf-crime"],
    version="3.0.0",
)
def extract_datetime_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    datetime_column: str = "Dates",
) -> str:
    """Extract temporal features from a datetime column.

    Based on top solution notebooks (sjun4530, mohitsital):
    Creates year, month, day, hour, minute, dayofweek, is_weekend,
    time_of_day, n_days, special_time features.

    G1 Compliance: Single responsibility - temporal feature extraction.
    G4 Compliance: datetime_column parameterized.
    """
    df = _load_data(inputs["data"])

    if datetime_column in df.columns:
        dt = pd.to_datetime(df[datetime_column])
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["hour"] = dt.dt.hour
        df["minute"] = dt.dt.minute
        df["dayofweek"] = dt.dt.dayofweek
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["time_of_day"] = pd.cut(
            dt.dt.hour, bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]
        ).astype(int)
        df["n_days"] = (dt - dt.min()).dt.days
        # From solution 03: special time indicator (on the hour or half hour)
        df["special_time"] = df["minute"].isin([0, 30]).astype(int)

    _save_data(df, outputs["data"])
    return f"extract_datetime_features: added 10 temporal features from '{datetime_column}'"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Create geospatial features from coordinate columns",
    tags=["feature-engineering", "geospatial", "sf-crime"],
    version="4.0.0",
)
def create_geospatial_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    x_column: str = "X",
    y_column: str = "Y",
    center_x: float = -122.4194,
    center_y: float = 37.7749,
    gmm_components: int = 100,
) -> str:
    """Create geospatial features from coordinate columns.

    Based on top solution notebooks (sjun4530, mohitsital):
    - Fixes known outlier coordinates (X=-120.5, Y=90.0)
    - Creates dist_from_center, X_plus_Y, X_minus_Y, x_bin, y_bin
    - Adds rotational features (30 and 60 degree rotations)
    - Adds corner distance features
    - Adds GaussianMixture clustering (XYcluster) - KEY feature from mohitsital

    G1 Compliance: Single responsibility - geospatial feature creation.
    G4 Compliance: Column names and center coords parameterized.
    """
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    df = _load_data(inputs["data"])

    if x_column in df.columns and y_column in df.columns:
        # Fix known coordinate outliers (from solution 03 analysis)
        outlier_mask = (df[x_column] > -121.0) | (df[y_column] > 80.0)
        if outlier_mask.any():
            x_median = df.loc[~outlier_mask, x_column].median()
            y_median = df.loc[~outlier_mask, y_column].median()
            df.loc[outlier_mask, x_column] = x_median
            df.loc[outlier_mask, y_column] = y_median

        X = df[x_column].values
        Y = df[y_column].values

        # Distance from city center
        df["dist_from_center"] = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        # Coordinate interactions (from solution 02)
        df["X_plus_Y"] = X + Y
        df["X_minus_Y"] = X - Y

        # Rotational features at 30 and 60 degrees (from solution 03)
        df["XY30_1"] = X * np.cos(np.pi / 6) + Y * np.sin(np.pi / 6)
        df["XY30_2"] = Y * np.cos(np.pi / 6) - X * np.sin(np.pi / 6)
        df["XY60_1"] = X * np.cos(np.pi / 3) + Y * np.sin(np.pi / 3)
        df["XY60_2"] = Y * np.cos(np.pi / 3) - X * np.sin(np.pi / 3)

        # Corner distance features (from solution 03)
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        df["XY_corner1"] = (X - x_min) ** 2 + (Y - y_min) ** 2
        df["XY_corner2"] = (x_max - X) ** 2 + (Y - y_min) ** 2
        df["XY_corner3"] = (X - x_min) ** 2 + (y_max - Y) ** 2
        df["XY_corner4"] = (x_max - X) ** 2 + (y_max - Y) ** 2

        # PCA on coordinates (from solution 03)
        pca = PCA(n_components=2)
        XY_pca = pca.fit_transform(df[[x_column, y_column]])
        df["XY_pca1"] = XY_pca[:, 0]
        df["XY_pca2"] = XY_pca[:, 1]

        # Grid binning
        df["x_bin"] = pd.cut(df[x_column], bins=20, labels=False)
        df["y_bin"] = pd.cut(df[y_column], bins=20, labels=False)

        # GaussianMixture clustering (KEY feature from mohitsital's top solution)
        # n_components=100-150 found optimal in solution 03
        gmm = GaussianMixture(n_components=gmm_components, covariance_type="diag", random_state=42)
        gmm.fit(df[[x_column, y_column]])
        df["XYcluster"] = gmm.predict(df[[x_column, y_column]])

    _save_data(df, outputs["data"])
    return f"create_geospatial_features: added 18 geospatial features (incl. GMM cluster with {gmm_components} components)"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Extract text features from address column",
    tags=["feature-engineering", "text", "sf-crime"],
    version="2.0.0",
)
def create_address_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    address_column: str = "Address",
) -> str:
    """Extract simple text features from address column.

    Based on top solution notebooks (sjun4530, mohitsital):
    Creates block_flag (contains 'Block'), st_flag (intersection with '/'),
    and additional pattern-based features.

    G1 Compliance: Single responsibility - address feature extraction.
    G4 Compliance: address_column parameterized.
    """
    df = _load_data(inputs["data"])

    if address_column in df.columns:
        addr = df[address_column].fillna("")
        addr_upper = addr.str.upper()
        df["block_flag"] = addr_upper.str.contains("BLOCK").astype(int)
        df["st_flag"] = addr.str.contains("/").astype(int)
        # Additional patterns from top solutions
        df["st_type"] = addr_upper.str.contains(" ST$| ST ").astype(int)
        df["av_type"] = addr_upper.str.contains(" AV$| AV ").astype(int)
        # Count tokens in address
        df["addr_tokens"] = addr.str.split().str.len().fillna(0).astype(int)

    _save_data(df, outputs["data"])
    return "create_address_features: added 5 address features"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific feature engineering
    "extract_datetime_features": extract_datetime_features,
    "create_geospatial_features": create_geospatial_features,
    "create_address_features": create_address_features,
    # Common services (re-exported for convenience)
    "split_data": split_data,
    "drop_columns": drop_columns,
    "fit_encoder": fit_encoder,
    "transform_encoder": transform_encoder,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "predict_multiclass_submission": predict_multiclass_submission,
}
