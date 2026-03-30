"""
PUBG Finish Placement Prediction - SLEGO Services
===================================================
Competition: https://www.kaggle.com/competitions/pubg-finish-placement-prediction
Problem Type: Regression
Target: winPlacePerc (normalized placement, 0-1)
ID Column: Id

Key insights from top solution notebooks:
- Group-level aggregations (kills, damage, distance) are critical features
- Match-type encoding (solo/duo/squad) provides useful signal
- Post-processing with rank-based adjustment and maxPlace grid alignment
  dramatically improves score (from solution #3 post-processor)
- killPlace is the single most important feature

Competition-specific services:
- create_pubg_features: Group-level feature engineering derived from top solutions
- postprocess_pubg_predictions: Rank-based adjustment and maxPlace alignment
"""

import os
import sys
import gc
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from common modules
try:
    from services.preprocessing_services import drop_columns, split_data
    from services.regression_services import train_lightgbm_regressor, predict_regressor
except ImportError:
    from preprocessing_services import drop_columns, split_data
    from regression_services import train_lightgbm_regressor, predict_regressor

try:
    from slego_contract import contract
except ImportError:
    try:
        from app.slego_contract import contract
    except ImportError:
        def contract(**kwargs):
            def decorator(func):
                return func
            return decorator


# =============================================================================
# SERVICE 1: CREATE PUBG FEATURES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Create PUBG-specific features from group/match aggregations",
    tags=["competition", "feature_engineering", "pubg"],
)
def create_pubg_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "winPlacePerc",
    id_column: str = "Id",
    group_column: str = "groupId",
    match_column: str = "matchId",
    match_type_column: str = "matchType",
) -> str:
    """
    Create PUBG-specific features derived from top solution notebooks.

    Features created:
    - Player-level: total_distance, kill_efficiency, headshot_rate, items_used
    - Group-level: group_size, total kills/damage/distance/boosts/heals per group
    - Match-normalized: percentile rank of killPlace, walkDistance, damageDealt
    - Match-type encoding: solo=0, duo=1, squad=2

    Works with any team-based game data with similar column structure.

    Args:
        target_column: Target column name (auto-detected whether present)
        id_column: Player ID column
        group_column: Team/group ID column
        match_column: Match ID column
        match_type_column: Match type column (solo/duo/squad variants)
    """
    data_path = inputs["data"]
    output_path = outputs["data"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = pd.read_csv(data_path)
    is_train = target_column in df.columns

    # Drop NaN target rows (known data issue in train_V2)
    if is_train:
        before = len(df)
        df.dropna(subset=[target_column], inplace=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with NaN {target_column}")

    # --- Player-level features ---
    df['total_distance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    df['kill_efficiency'] = df['kills'] / (df['damageDealt'] + 1)
    df['headshot_rate'] = df['headshotKills'] / (df['kills'] + 1)
    df['items_used'] = df['boosts'] + df['heals']
    df['kill_without_moving'] = ((df['kills'] > 0) & (df['total_distance'] == 0)).astype(np.int8)

    # --- Group-level aggregations ---
    group_cols = [match_column, group_column]
    group_agg = df.groupby(group_cols).agg(
        group_size=(id_column, 'count'),
        group_total_kills=('kills', 'sum'),
        group_max_kills=('kills', 'max'),
        group_total_damage=('damageDealt', 'sum'),
        group_total_distance=('total_distance', 'sum'),
        group_mean_walk=('walkDistance', 'mean'),
        group_total_boosts=('boosts', 'sum'),
        group_total_heals=('heals', 'sum'),
        group_total_weapons=('weaponsAcquired', 'sum'),
    ).reset_index()

    df = df.merge(group_agg, on=group_cols, how='left')
    del group_agg
    gc.collect()

    # --- Match-level normalized features (percentile ranks) ---
    df['killPlace_norm'] = df.groupby(match_column)['killPlace'].rank(pct=True)
    df['walkDistance_norm'] = df.groupby(match_column)['walkDistance'].rank(pct=True)
    df['damageDealt_norm'] = df.groupby(match_column)['damageDealt'].rank(pct=True)
    df['total_distance_norm'] = df.groupby(match_column)['total_distance'].rank(pct=True)

    # --- matchType encoding ---
    match_type_map = {}
    for mt in df[match_type_column].unique():
        mt_lower = mt.lower()
        if 'solo' in mt_lower:
            match_type_map[mt] = 0
        elif 'duo' in mt_lower:
            match_type_map[mt] = 1
        else:  # squad, crash, flare, normal
            match_type_map[mt] = 2
    df['match_mode'] = df[match_type_column].map(match_type_map).astype(np.int8)

    # --- Drop non-feature columns (used for aggregations only) ---
    drop_cols = [group_column, match_column, match_type_column]
    df.drop(columns=drop_cols, inplace=True)

    # Save
    df.to_csv(output_path, index=False)

    n_features = len([c for c in df.columns
                      if c not in [id_column, target_column]])
    return f"create_pubg_features: {n_features} features for {len(df)} rows"


# =============================================================================
# SERVICE 2: POSTPROCESS PUBG PREDICTIONS
# =============================================================================

@contract(
    inputs={"predictions": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"predictions": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Post-process PUBG predictions with rank-based adjustment",
    tags=["competition", "postprocessing", "pubg"],
)
def postprocess_pubg_predictions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "winPlacePerc",
    group_column: str = "groupId",
    match_column: str = "matchId",
) -> str:
    """
    Post-process PUBG predictions using rank-based adjustment and maxPlace alignment.

    Derived from top Kaggle solution notebooks (solution #3 post-processor):
    1. Groups predictions by match and group
    2. Ranks groups within each match by predicted winPlacePerc
    3. Adjusts percentile: (rank - 1) / (numGroups - 1)
    4. Aligns to maxPlace grid: round(perc / gap) * gap where gap = 1/(maxPlace-1)
    5. Handles edge cases (maxPlace=0, maxPlace=1, numGroups=1)

    Works with any placement prediction task with group/match structure.

    Args:
        id_column: Player ID column
        prediction_column: Predicted placement column
        group_column: Team/group ID column
        match_column: Match ID column
    """
    pred_path = inputs["predictions"]
    test_path = inputs["test_data"]
    output_path = outputs["submission"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    preds = pd.read_csv(pred_path)
    test = pd.read_csv(
        test_path,
        usecols=[id_column, match_column, group_column, 'maxPlace', 'numGroups']
    )

    # Merge predictions with test metadata
    df = test.merge(preds[[id_column, prediction_column]], on=id_column, how='left')
    del preds, test
    gc.collect()

    # Clip predictions to [0, 1]
    df[prediction_column] = df[prediction_column].clip(0, 1)

    # --- Rank-based adjustment per group ---
    group_df = df.groupby([match_column, group_column]).agg(
        pred_mean=(prediction_column, 'mean'),
        maxPlace=('maxPlace', 'first'),
        numGroups=('numGroups', 'first'),
    ).reset_index()

    # Rank within match
    group_df['rank'] = group_df.groupby(match_column)['pred_mean'].rank()

    # Adjusted percentile
    group_df['adjusted_perc'] = (group_df['rank'] - 1) / (group_df['numGroups'] - 1)
    group_df.loc[group_df['numGroups'] <= 1, 'adjusted_perc'] = 0

    # --- Align with maxPlace grid ---
    mask = group_df['maxPlace'] > 1
    gap = 1.0 / (group_df.loc[mask, 'maxPlace'] - 1)
    group_df.loc[mask, 'adjusted_perc'] = (
        np.around(group_df.loc[mask, 'adjusted_perc'].values / gap.values) * gap.values
    )

    # Edge cases
    group_df.loc[group_df['maxPlace'] == 0, 'adjusted_perc'] = 0
    group_df.loc[group_df['maxPlace'] == 1, 'adjusted_perc'] = 1
    group_df.loc[
        (group_df['maxPlace'] > 1) & (group_df['numGroups'] == 1),
        'adjusted_perc'
    ] = 0

    # Clip final values
    group_df['adjusted_perc'] = group_df['adjusted_perc'].clip(0, 1)

    # Merge back to all players
    df = df.merge(
        group_df[[match_column, group_column, 'adjusted_perc']],
        on=[match_column, group_column],
        how='left'
    )
    df[prediction_column] = df['adjusted_perc']
    del group_df
    gc.collect()

    # Save submission
    submission = df[[id_column, prediction_column]].copy()
    submission.to_csv(output_path, index=False)

    return f"postprocess_pubg_predictions: {len(submission)} predictions post-processed"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "drop_columns": drop_columns,
    "split_data": split_data,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "predict_regressor": predict_regressor,
}


# =============================================================================
# PIPELINE SPEC
# =============================================================================

PIPELINE_SPEC = [
    {
        "service": "create_pubg_features",
        "inputs": {"data": "pubg-finish-placement-prediction/datasets/train.csv"},
        "outputs": {"data": "pubg-finish-placement-prediction/artifacts/train_01_features.csv"},
        "params": {},
        "module": "pubg_finish_placement_prediction_services"
    },
    {
        "service": "create_pubg_features",
        "inputs": {"data": "pubg-finish-placement-prediction/datasets/test.csv"},
        "outputs": {"data": "pubg-finish-placement-prediction/artifacts/test_01_features.csv"},
        "params": {},
        "module": "pubg_finish_placement_prediction_services"
    },
    {
        "service": "split_data",
        "inputs": {"data": "pubg-finish-placement-prediction/artifacts/train_01_features.csv"},
        "outputs": {
            "train_data": "pubg-finish-placement-prediction/artifacts/train_split.csv",
            "valid_data": "pubg-finish-placement-prediction/artifacts/valid_split.csv"
        },
        "params": {"test_size": 0.2, "random_state": 42},
        "module": "pubg_finish_placement_prediction_services"
    },
    {
        "service": "train_lightgbm_regressor",
        "inputs": {
            "train_data": "pubg-finish-placement-prediction/artifacts/train_split.csv",
            "valid_data": "pubg-finish-placement-prediction/artifacts/valid_split.csv"
        },
        "outputs": {
            "model": "pubg-finish-placement-prediction/artifacts/model.pkl",
            "metrics": "pubg-finish-placement-prediction/artifacts/metrics.json"
        },
        "params": {
            "label_column": "winPlacePerc",
            "id_column": "Id",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "module": "pubg_finish_placement_prediction_services"
    },
    {
        "service": "predict_regressor",
        "inputs": {
            "model": "pubg-finish-placement-prediction/artifacts/model.pkl",
            "data": "pubg-finish-placement-prediction/artifacts/test_01_features.csv"
        },
        "outputs": {
            "predictions": "pubg-finish-placement-prediction/artifacts/predictions.csv"
        },
        "params": {
            "id_column": "Id",
            "prediction_column": "winPlacePerc"
        },
        "module": "pubg_finish_placement_prediction_services"
    },
    {
        "service": "postprocess_pubg_predictions",
        "inputs": {
            "predictions": "pubg-finish-placement-prediction/artifacts/predictions.csv",
            "test_data": "pubg-finish-placement-prediction/datasets/test.csv"
        },
        "outputs": {
            "submission": "pubg-finish-placement-prediction/submission.csv"
        },
        "params": {},
        "module": "pubg_finish_placement_prediction_services"
    },
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
            print(f"[{i}/{len(PIPELINE_SPEC)}] {service_name}...", end=" ", flush=True)

        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose:
                print(f"OK - {result}")
        except Exception as e:
            if verbose:
                print(f"FAILED - {e}")
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

    print(f"\n--- PUBG Pipeline (Base: {storage_path}) ---")
    run_pipeline(storage_path)
