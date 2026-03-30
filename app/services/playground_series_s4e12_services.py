"""
Playground Series S4E12 - Insurance Premium Prediction - Contract-Composable Analytics Services
========================================================================
Competition: https://www.kaggle.com/competitions/playground-series-s4e12
Problem Type: Regression (RMSLE metric)
Target: Premium Amount
ID: id

Solution notebook insights:
- 1st place (cdeotte): XGBoost with target encoding on column combinations,
  label-encoded categoricals, datetime features from Policy Start Date,
  log1p target transform, 20-fold CV. The secret: 120 features from
  target-encoding 20 powerful column combos (2-6 columns each).
- 2nd (masayakawamata): XGBoost L2 meta-model stacking on OOF predictions
- 3rd (nina2025): Weighted h-blend of multiple submissions

Competition-specific services:
- preprocess_insurance_data: Label encode categoricals + extract datetime features
  in one step (reusable for any tabular dataset with dates and categoricals)
- target_encode_features: K-fold target encoding on column combinations
  (the key feature engineering technique from 1st place solution)
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from contract import contract


# ============================================================================
# IMPORT REUSABLE SERVICES FROM EXISTING MODULES
# ============================================================================

from services.preprocessing_services import (
    split_data,
    drop_columns,
    fit_encoder,
    transform_encoder,
    create_submission,
)
from services.regression_services import (
    train_xgboost_regressor,
    predict_regressor,
)
from services.bike_sharing_services import (
    extract_datetime_features,
)


# ============================================================================
# COMPETITION-SPECIFIC SERVICES
# ============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
        "encoder": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Preprocess tabular data with datetime extraction and categorical encoding",
    tags=["preprocessing", "encoding", "datetime", "tabular", "generic"],
    version="1.0.0",
)
def preprocess_insurance_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    datetime_column: str = "Policy Start Date",
    datetime_features: Optional[List[str]] = None,
    target_column: str = "Premium Amount",
    id_column: str = "id",
    encoding_method: str = "ordinal",
    log_transform_target: bool = False,
) -> str:
    """
    Preprocess tabular data: extract datetime features + encode categoricals.

    Combines datetime feature extraction and categorical encoding in a single
    service to ensure consistent preprocessing of train and test data.
    Inspired by 1st place solution insights.

    Reusable for any tabular dataset with datetime and categorical columns.

    Parameters:
        datetime_column: Name of the datetime column to extract features from
        datetime_features: List of features to extract (year, month, day, dayofweek)
        target_column: Name of the target column (excluded from encoding)
        id_column: Name of the ID column (excluded from encoding)
        encoding_method: Encoding method for categoricals ('ordinal' or 'label')
        log_transform_target: If True, apply log1p to target column (for RMSLE optimization)
    """
    from sklearn.preprocessing import OrdinalEncoder

    datetime_features = datetime_features or ['year', 'month', 'day', 'dayofweek']

    train_df = pd.read_csv(inputs["train_data"])
    test_df = pd.read_csv(inputs["test_data"])

    # --- Extract datetime features ---
    for df in [train_df, test_df]:
        if datetime_column in df.columns:
            dt = pd.to_datetime(df[datetime_column])
            if 'year' in datetime_features:
                df['year'] = dt.dt.year
            if 'month' in datetime_features:
                df['month'] = dt.dt.month
            if 'day' in datetime_features:
                df['day'] = dt.dt.day
            if 'dayofweek' in datetime_features:
                df['dayofweek'] = dt.dt.dayofweek
            if 'seconds' in datetime_features:
                df['seconds'] = (dt.astype("int64") // 10**9)
            df.drop(columns=[datetime_column], inplace=True)

    # --- Log-transform target (for RMSLE optimization) ---
    if log_transform_target and target_column in train_df.columns:
        train_df[target_column] = np.log1p(train_df[target_column])

    # --- Encode categorical columns ---
    exclude = {id_column, target_column}
    cat_cols = [c for c in train_df.select_dtypes(include=["object", "category"]).columns
                if c not in exclude]

    encodings = {}
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])
        test_df[cat_cols] = encoder.transform(test_df[cat_cols])
        encodings = {"method": encoding_method, "categorical_columns": cat_cols, "encoder": encoder}

    # --- Save outputs ---
    for path in [outputs["train_data"], outputs["test_data"], outputs["encoder"]]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    train_df.to_csv(outputs["train_data"], index=False)
    test_df.to_csv(outputs["test_data"], index=False)

    with open(outputs["encoder"], "wb") as f:
        pickle.dump(encodings, f)

    log_msg = " + log1p target" if log_transform_target else ""
    return (
        f"preprocess_insurance_data: {len(train_df)} train rows, {len(test_df)} test rows, "
        f"extracted {len(datetime_features)} datetime features, encoded {len(cat_cols)} categoricals{log_msg}"
    )


# ============================================================================
# TARGET ENCODING SERVICE (from 1st place solution insights)
# ============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="K-fold target encoding + count encoding on individual columns and column combinations (1st place solution)",
    tags=["feature-engineering", "target-encoding", "count-encoding", "tabular", "generic"],
    version="2.0.0",
)
def target_encode_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Premium Amount",
    id_column: str = "id",
    column_combinations: Optional[List[List[str]]] = None,
    kfold: int = 10,
    smooth: int = 20,
    agg_types: Optional[List[str]] = None,
    high_cardinality_aggs: Optional[List[str]] = None,
    high_cardinality_threshold: int = 9,
    encode_individual_columns: bool = True,
    add_count_encoding: bool = True,
    target_already_log: bool = False,
) -> str:
    """
    Apply k-fold target encoding + count encoding on column combinations.

    Implements cdeotte's 1st place solution techniques:
    - K-fold target encoding (no leakage) with multiple aggregation types
    - Separate treatment for HIGH_CARDINALITY features (9+ unique): add min, max, nunique
    - Count encoding (CE) for all column combinations
    - Higher k-fold (10-20) for better generalization

    Parameters:
        target_column: Name of the target column
        id_column: Name of the ID column (excluded from encoding)
        column_combinations: List of column lists to target-encode together
        kfold: Number of folds for k-fold target encoding (default: 10)
        smooth: Smoothing factor for TE (higher = more regularization)
        agg_types: Base aggregation types for all columns ('mean', 'median')
        high_cardinality_aggs: Additional aggs for high-cardinality cols ('min', 'max', 'nunique')
        high_cardinality_threshold: Cardinality threshold (default: 9)
        encode_individual_columns: Also target-encode all individual feature columns
        add_count_encoding: Add count encoding (CE) features
        target_already_log: If True, target is already log-transformed
    """
    from sklearn.model_selection import KFold

    train_df = pd.read_csv(inputs["train_data"])
    test_df = pd.read_csv(inputs["test_data"])

    agg_types = agg_types or ["mean", "median"]
    high_cardinality_aggs = high_cardinality_aggs or ["min", "max", "nunique"]
    column_combinations = column_combinations or []

    # Combine train+test for count encoding
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Use log1p target for TE computation (better for RMSLE-scored competitions)
    y_col = "__te_target__"
    if target_already_log:
        train_df[y_col] = train_df[target_column]
    else:
        train_df[y_col] = np.log1p(train_df[target_column])

    exclude = {id_column, target_column, y_col}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    # Identify HIGH_CARDINALITY columns (9+ unique values)
    high_cardinality_cols = set()
    for c in feature_cols:
        if combined_df[c].nunique() >= high_cardinality_threshold:
            high_cardinality_cols.add(c)

    # Build list of column groups to encode
    col_groups = []
    if encode_individual_columns:
        col_groups.extend([[c] for c in feature_cols])
    col_groups.extend(column_combinations)

    n_te_features = 0
    n_ce_features = 0
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)

    for idx, cols in enumerate(col_groups):
        # Verify all columns exist
        if not all(c in train_df.columns for c in cols):
            continue

        col_name = '_'.join(cols)
        is_individual = len(cols) == 1
        is_high_cardinality = is_individual and cols[0] in high_cardinality_cols

        # Determine which aggregations to use
        # - All columns get mean and median
        # - High cardinality individual cols AND all combinations get min, max, nunique
        if is_high_cardinality or not is_individual:
            current_aggs = agg_types + high_cardinality_aggs
        else:
            current_aggs = agg_types

        for agg in current_aggs:
            feat_name = f"TE_{agg.upper()}_{col_name}"

            # --- Train: k-fold target encoding (no leakage) ---
            train_df[feat_name] = 0.0

            if agg == "nunique":
                global_stat = 0
            elif agg == "min":
                global_stat = train_df[y_col].min()
            elif agg == "max":
                global_stat = train_df[y_col].max()
            else:
                global_stat = getattr(train_df[y_col], agg)()

            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
                fold_train = train_df.iloc[tr_idx]

                if agg == "nunique":
                    grp = fold_train.groupby(cols)[y_col].agg(["count"]).reset_index()
                    grp["nunique"] = fold_train.groupby(cols)[y_col].nunique().values
                    grp.columns = list(cols) + ["count", "nunique"]
                    grp["te_val"] = grp["nunique"] / grp["count"]
                else:
                    grp = fold_train.groupby(cols)[y_col].agg([agg, "count"]).reset_index()
                    grp.columns = list(cols) + [agg, "count"]
                    # Apply smoothing (0 for min/max to preserve actual values)
                    smooth_factor = 0 if agg in ["min", "max"] else smooth
                    grp["te_val"] = (grp[agg] * grp["count"] + global_stat * smooth_factor) / (grp["count"] + smooth_factor)

                merged = train_df.iloc[val_idx][cols].merge(grp[list(cols) + ["te_val"]], on=cols, how="left")
                train_df.iloc[val_idx, train_df.columns.get_loc(feat_name)] = merged["te_val"].fillna(global_stat).values

            # --- Test: use full train aggregates ---
            if agg == "nunique":
                grp_full = train_df.groupby(cols)[y_col].agg(["count"]).reset_index()
                grp_full["nunique"] = train_df.groupby(cols)[y_col].nunique().values
                grp_full.columns = list(cols) + ["count", "nunique"]
                grp_full["te_val"] = grp_full["nunique"] / grp_full["count"]
            else:
                grp_full = train_df.groupby(cols)[y_col].agg([agg, "count"]).reset_index()
                grp_full.columns = list(cols) + [agg, "count"]
                smooth_factor = 0 if agg in ["min", "max"] else smooth
                grp_full["te_val"] = (grp_full[agg] * grp_full["count"] + global_stat * smooth_factor) / (grp_full["count"] + smooth_factor)

            merged_test = test_df[cols].merge(grp_full[list(cols) + ["te_val"]], on=cols, how="left")
            test_df[feat_name] = merged_test["te_val"].fillna(global_stat).values

            n_te_features += 1

        # --- Count Encoding (CE) for high cardinality / combinations ---
        if add_count_encoding and (is_high_cardinality or not is_individual):
            ce_name = f"CE_{col_name}"
            counts = combined_df.groupby(cols).size().reset_index(name="count")

            train_merged = train_df[cols].merge(counts, on=cols, how="left")
            train_df[ce_name] = train_merged["count"].fillna(0).astype("int32")

            test_merged = test_df[cols].merge(counts, on=cols, how="left")
            test_df[ce_name] = test_merged["count"].fillna(0).astype("int32")

            n_ce_features += 1

    # Drop temporary target column
    train_df.drop(columns=[y_col], inplace=True)

    # Save outputs
    for path in [outputs["train_data"], outputs["test_data"]]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    train_df.to_csv(outputs["train_data"], index=False)
    test_df.to_csv(outputs["test_data"], index=False)

    return (
        f"target_encode_features v2: {n_te_features} TE + {n_ce_features} CE features, "
        f"train={train_df.shape}, test={test_df.shape}, "
        f"high_card_cols={len(high_cardinality_cols)}"
    )


# ============================================================================
# SERVICE REGISTRY
# ============================================================================

SERVICE_REGISTRY = {
    # Competition-specific
    "preprocess_insurance_data": preprocess_insurance_data,
    "target_encode_features": target_encode_features,
    # Re-exported generic services
    "extract_datetime_features": extract_datetime_features,
    "drop_columns": drop_columns,
    "fit_encoder": fit_encoder,
    "transform_encoder": transform_encoder,
    "split_data": split_data,
    "train_xgboost_regressor": train_xgboost_regressor,
    "predict_regressor": predict_regressor,
    "create_submission": create_submission,
}