"""
Tabular Playground Series Oct 2022 - Contract-Composable Analytics Services
=====================================================
Competition: https://www.kaggle.com/competitions/tabular-playground-series-oct-2022
Problem Type: Multi-label binary classification (two independent probabilities)
Targets: team_A_scoring_within_10sec, team_B_scoring_within_10sec
ID Column: id
Evaluation Metric: Mean Log Loss = (LogLoss(A) + LogLoss(B)) / 2

Rocket League game state prediction: predict probability of each team scoring
within 10 seconds given current ball/player positions, velocities, and boost states.

Features:
- Ball: pos_x/y/z, vel_x/y/z (6 features)
- Players p0-p5: pos_x/y/z, vel_x/y/z, boost (7 features each, 42 total)
- Boost timers: boost0_timer - boost5_timer (6 features)
- Total: 54 raw features (plus id)

Train-only columns to drop:
- game_num, event_id, event_time, player_scoring_next, team_scoring_next

Solution Notebook Insights (from top 3 solutions):
1. Host solution (dster): TensorFlow NN with heavy data augmentation
   - flip_x, flip_y, player reordering (144 permutations)
   - Multi-task learning for different time windows

2. Boltzmann Ensemble (jbomitchell): Weighted ensemble of 60+ models
   - Probability clipping (0.01 to 0.99) to avoid log loss penalties
   - Scaling factors for team A vs B asymmetry

3. FastAI with multistart and TTA (alexryzhkov):
   - Feature engineering: distance to goal, distance to ball, speeds
   - Data augmentation: mirror, flip_x, player shuffling
   - Multiple model restarts and Test-Time Augmentation

Contract-Composable Analytics approach: Multi-target LightGBM classifier with feature engineering
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

# =============================================================================
# IMPORTS FROM COMMON MODULES (G1: Reuse existing services)
# =============================================================================
try:
    from services.io_utils import load_data as _load_data, save_data as _save_data
    from services.preprocessing_services import split_data, drop_columns
    from services.playground_s3e18_services import (
        train_multi_target_classifier,
        predict_multi_target_classifier,
    )
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from preprocessing_services import split_data, drop_columns
    from playground_s3e18_services import (
        train_multi_target_classifier,
        predict_multi_target_classifier,
    )


# =============================================================================
# ROCKET LEAGUE FEATURE ENGINEERING (Generic, Reusable)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Engineer features for Rocket League game state prediction",
    tags=["preprocessing", "feature-engineering", "rocket-league", "game-state"],
    version="1.0.0",
)
def engineer_rocket_league_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    add_ball_distance: bool = True,
    add_player_distance: bool = True,
    add_speeds: bool = True,
    add_team_aggregates: bool = True,
) -> str:
    """Engineer features for Rocket League game state prediction.

    Inspired by top solution notebooks:
    - Distance of ball to each goal (goal A at y=120, goal B at y=-120)
    - Distance of each player to ball
    - Speed of ball and each player
    - Team aggregate features (mean position, total boost)

    G1 Compliance: Generic, works with any 3v3 game state data.
    G4 Compliance: All feature flags as parameters.
    """
    df = _load_data(inputs["data"])

    # Ball position/velocity columns
    ball_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z',
                 'ball_vel_x', 'ball_vel_y', 'ball_vel_z']

    # Player position columns (p0-p5)
    player_pos_cols = [[f'p{i}_pos_x', f'p{i}_pos_y', f'p{i}_pos_z'] for i in range(6)]
    player_vel_cols = [[f'p{i}_vel_x', f'p{i}_vel_y', f'p{i}_vel_z'] for i in range(6)]

    new_features = []

    if add_ball_distance:
        # Distance from ball to goal A (y=120) and goal B (y=-120)
        # Goal positions approximately at (0, +-120, 0)
        if all(c in df.columns for c in ['ball_pos_x', 'ball_pos_y', 'ball_pos_z']):
            df['ball_dist_goal_A'] = np.sqrt(
                df['ball_pos_x']**2 +
                (df['ball_pos_y'] - 120)**2 +
                df['ball_pos_z']**2
            )
            df['ball_dist_goal_B'] = np.sqrt(
                df['ball_pos_x']**2 +
                (df['ball_pos_y'] + 120)**2 +
                df['ball_pos_z']**2
            )
            new_features.extend(['ball_dist_goal_A', 'ball_dist_goal_B'])

    if add_player_distance:
        # Distance of each player to ball
        for i in range(6):
            pos_cols = player_pos_cols[i]
            if all(c in df.columns for c in pos_cols + ball_cols[:3]):
                df[f'p{i}_dist_ball'] = np.sqrt(
                    (df[pos_cols[0]] - df['ball_pos_x'])**2 +
                    (df[pos_cols[1]] - df['ball_pos_y'])**2 +
                    (df[pos_cols[2]] - df['ball_pos_z'])**2
                )
                new_features.append(f'p{i}_dist_ball')

    if add_speeds:
        # Ball speed
        if all(c in df.columns for c in ball_cols[3:6]):
            df['ball_speed'] = np.sqrt(
                df['ball_vel_x']**2 +
                df['ball_vel_y']**2 +
                df['ball_vel_z']**2
            )
            new_features.append('ball_speed')

        # Player speeds
        for i in range(6):
            vel_cols = player_vel_cols[i]
            if all(c in df.columns for c in vel_cols):
                df[f'p{i}_speed'] = np.sqrt(
                    df[vel_cols[0]]**2 +
                    df[vel_cols[1]]**2 +
                    df[vel_cols[2]]**2
                )
                new_features.append(f'p{i}_speed')

    if add_team_aggregates:
        # Team A: p0, p1, p2
        # Team B: p3, p4, p5
        for team, players in [('A', [0, 1, 2]), ('B', [3, 4, 5])]:
            # Mean position
            for axis in ['x', 'y', 'z']:
                cols = [f'p{i}_pos_{axis}' for i in players]
                if all(c in df.columns for c in cols):
                    df[f'team_{team}_mean_pos_{axis}'] = df[cols].mean(axis=1)
                    new_features.append(f'team_{team}_mean_pos_{axis}')

            # Total boost
            boost_cols = [f'p{i}_boost' for i in players]
            if all(c in df.columns for c in boost_cols):
                df[f'team_{team}_total_boost'] = df[boost_cols].sum(axis=1)
                new_features.append(f'team_{team}_total_boost')

    # Fill NaN in new features with 0
    for col in new_features:
        df[col] = df[col].fillna(0)

    _save_data(df, outputs["data"])

    return f"engineer_rocket_league_features: added {len(new_features)} features"


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Drop train-only columns from Rocket League dataset",
    tags=["preprocessing", "data-cleaning", "rocket-league"],
    version="1.0.0",
)
def clean_rocket_league_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    drop_train_only: bool = True,
    fill_na_method: str = "zero",
) -> str:
    """Clean Rocket League data by dropping train-only columns and handling NaN.

    Train-only columns: game_num, event_id, event_time, player_scoring_next, team_scoring_next

    G1 Compliance: Single responsibility - data cleaning only.
    G4 Compliance: Columns to drop parameterized.
    """
    df = _load_data(inputs["data"])

    dropped = []
    if drop_train_only:
        train_only_cols = ['game_num', 'event_id', 'event_time',
                           'player_scoring_next', 'team_scoring_next']
        for col in train_only_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
                dropped.append(col)

    # Fill NaN values (player positions when demoed)
    filled = 0
    if fill_na_method == "zero":
        filled = df.isnull().sum().sum()
        df = df.fillna(0)
    elif fill_na_method == "median":
        for col in df.select_dtypes(include=[np.number]).columns:
            n = df[col].isnull().sum()
            if n > 0:
                df[col] = df[col].fillna(df[col].median())
                filled += n

    _save_data(df, outputs["data"])

    return f"clean_rocket_league_data: dropped {len(dropped)} columns, filled {filled} NaN values"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific services
    "engineer_rocket_league_features": engineer_rocket_league_features,
    "clean_rocket_league_data": clean_rocket_league_data,
    # Imported from common modules
    "split_data": split_data,
    "drop_columns": drop_columns,
    "train_multi_target_classifier": train_multi_target_classifier,
    "predict_multi_target_classifier": predict_multi_target_classifier,
}


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = {
    "name": "tabular-playground-series-oct-2022",
    "description": "Rocket League game state prediction - multi-label probability prediction",
    "version": "2.0.0",
    "problem_type": "multi-label",
    "target_column": "team_A_scoring_within_10sec,team_B_scoring_within_10sec",
    "id_column": "id",
    "evaluation_metric": "mean_log_loss",
    "steps": [
        {
            "service": "clean_rocket_league_data",
            "inputs": {"data": "tabular-playground-series-oct-2022/datasets/train.csv"},
            "outputs": {"data": "tabular-playground-series-oct-2022/artifacts/train_clean.csv"},
            "params": {"drop_train_only": True, "fill_na_method": "zero"},
            "module": "tabular_playground_oct_2022_services",
        },
        {
            "service": "engineer_rocket_league_features",
            "inputs": {"data": "tabular-playground-series-oct-2022/artifacts/train_clean.csv"},
            "outputs": {"data": "tabular-playground-series-oct-2022/artifacts/train_fe.csv"},
            "params": {
                "add_ball_distance": True,
                "add_player_distance": True,
                "add_speeds": True,
                "add_team_aggregates": True,
            },
            "module": "tabular_playground_oct_2022_services",
        },
        {
            "service": "split_data",
            "inputs": {"data": "tabular-playground-series-oct-2022/artifacts/train_fe.csv"},
            "outputs": {
                "train_data": "tabular-playground-series-oct-2022/artifacts/train_split.csv",
                "valid_data": "tabular-playground-series-oct-2022/artifacts/valid_split.csv",
            },
            "params": {"test_size": 0.2, "random_state": 42},
            "module": "preprocessing_services",
        },
        {
            "service": "train_multi_target_classifier",
            "inputs": {
                "train_data": "tabular-playground-series-oct-2022/artifacts/train_split.csv",
                "valid_data": "tabular-playground-series-oct-2022/artifacts/valid_split.csv",
            },
            "outputs": {
                "model": "tabular-playground-series-oct-2022/artifacts/model.pkl",
                "metrics": "tabular-playground-series-oct-2022/artifacts/metrics.json",
            },
            "params": {
                "target_columns": ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"],
                "id_column": "id",
                "model_type": "lightgbm",
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 64,
                "max_depth": -1,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 100,
            },
            "module": "playground_s3e18_services",
        },
        {
            "service": "clean_rocket_league_data",
            "inputs": {"data": "tabular-playground-series-oct-2022/datasets/test.csv"},
            "outputs": {"data": "tabular-playground-series-oct-2022/artifacts/test_clean.csv"},
            "params": {"drop_train_only": True, "fill_na_method": "zero"},
            "module": "tabular_playground_oct_2022_services",
        },
        {
            "service": "engineer_rocket_league_features",
            "inputs": {"data": "tabular-playground-series-oct-2022/artifacts/test_clean.csv"},
            "outputs": {"data": "tabular-playground-series-oct-2022/artifacts/test_fe.csv"},
            "params": {
                "add_ball_distance": True,
                "add_player_distance": True,
                "add_speeds": True,
                "add_team_aggregates": True,
            },
            "module": "tabular_playground_oct_2022_services",
        },
        {
            "service": "predict_multi_target_classifier",
            "inputs": {
                "model": "tabular-playground-series-oct-2022/artifacts/model.pkl",
                "data": "tabular-playground-series-oct-2022/artifacts/test_fe.csv",
            },
            "outputs": {
                "predictions": "tabular-playground-series-oct-2022/submission.csv",
            },
            "params": {"id_column": "id"},
            "module": "playground_s3e18_services",
        },
    ],
}
