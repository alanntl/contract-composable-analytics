"""
March Machine Learning Mania 2017 - Contract-Composable Analytics Services
===================================================
Competition: https://www.kaggle.com/competitions/march-machine-learning-mania-2017
Problem Type: Binary Classification (probability)
Target: Pred (probability that team1 wins)
ID Column: Id (format: Season_Team1ID_Team2ID)
Evaluation: Log Loss

Predict NCAA tournament game outcomes as probabilities.
Uses original competition data files:
- RegularSeasonCompactResults.csv: historical regular season games
- TourneyCompactResults.csv: historical tournament outcomes (for training)
- TourneySeeds.csv: tournament seeding (1-16 per region, strong signal)
- SampleSubmission.csv: test matchups

Key insights from solution notebooks:
- Solution 01 (kaledata): Historical win rates between team pairs
- Solution 03 (aikinogard): Collaborative filtering with Keras embeddings (score ~0.56)
- Solution 05 (baeng72): Massey rating system (least squares per-team ratings)
- All solutions use regular season data for team strength, seeds for ranking

Competition-specific services:
- engineer_ncaa_features: Full feature engineering from raw competition files
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
    from services.io_utils import load_data as _load_data, save_data as _save_data
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.preprocessing_services import split_data, create_submission
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import split_data, create_submission


def _parse_seed(seed_str):
    """Parse seed string like 'W01' to integer 1, 'X16a' to 16."""
    # Remove region letter (W/X/Y/Z) and any play-in suffix (a/b)
    num_str = seed_str[1:]  # Remove first character (region)
    num_str = ''.join(c for c in num_str if c.isdigit())
    return int(num_str) if num_str else 16


def _compute_team_season_stats(regular_season_df):
    """Compute per-team per-season statistics from regular season results.

    Returns dict: (team_id, season) -> stats dict
    """
    stats = {}

    for _, row in regular_season_df.iterrows():
        season = row['Season']
        wteam = row['Wteam']
        lteam = row['Lteam']
        wscore = row['Wscore']
        lscore = row['Lscore']

        # Update winner stats
        key_w = (wteam, season)
        if key_w not in stats:
            stats[key_w] = {'wins': 0, 'losses': 0, 'points_for': 0,
                           'points_against': 0, 'games': 0}
        stats[key_w]['wins'] += 1
        stats[key_w]['games'] += 1
        stats[key_w]['points_for'] += wscore
        stats[key_w]['points_against'] += lscore

        # Update loser stats
        key_l = (lteam, season)
        if key_l not in stats:
            stats[key_l] = {'wins': 0, 'losses': 0, 'points_for': 0,
                           'points_against': 0, 'games': 0}
        stats[key_l]['losses'] += 1
        stats[key_l]['games'] += 1
        stats[key_l]['points_for'] += lscore
        stats[key_l]['points_against'] += wscore

    # Compute derived metrics
    result = {}
    for key, s in stats.items():
        games = max(s['games'], 1)
        result[key] = {
            'win_rate': s['wins'] / games,
            'avg_points_for': s['points_for'] / games,
            'avg_points_against': s['points_against'] / games,
            'avg_margin': (s['points_for'] - s['points_against']) / games,
            'games_played': s['games'],
        }

    return result


# =============================================================================
# COMPETITION-SPECIFIC SERVICE: NCAA Feature Engineering
# =============================================================================

@contract(
    inputs={
        "regular_season": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "tourney_results": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "tourney_seeds": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "sample_submission": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Engineer NCAA tournament prediction features from raw competition files",
    tags=["feature-engineering", "sports", "ncaa", "tournament-prediction"],
    version="3.0.0",
)
def engineer_ncaa_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "target",
    id_column: str = "Id",
    predict_season: int = 2017,
) -> str:
    """Engineer features for NCAA tournament prediction from raw data.

    Uses regular season data to build team strength metrics per season,
    tournament seeds for ranking features, and historical tournament results
    for training data with proper balanced targets.

    Training data: all historical tournament games (1985-2016) with features
    Test data: all possible 2017 matchups from sample submission

    Features per team:
    - seed: tournament seed number (1-16)
    - win_rate: regular season win percentage
    - avg_points_for, avg_points_against: scoring averages
    - avg_margin: average point differential

    Matchup features:
    - seed_diff: team1_seed - team2_seed
    - win_rate_diff, margin_diff, etc.

    G1 Compliance: Single-purpose feature engineering.
    G4 Compliance: Column names parameterized.
    """
    # Load raw data
    reg_season = _load_data(inputs["regular_season"])
    tourney = _load_data(inputs["tourney_results"])
    seeds = _load_data(inputs["tourney_seeds"])
    sample_sub = _load_data(inputs["sample_submission"])

    # --- Step 1: Compute per-team per-season stats ---
    team_season_stats = _compute_team_season_stats(reg_season)

    # --- Step 2: Build seed lookup: (team, season) -> seed_number ---
    seed_lookup = {}
    for _, row in seeds.iterrows():
        seed_lookup[(row['Team'], row['Season'])] = _parse_seed(row['Seed'])

    # --- Step 3: Build training data from historical tournament games ---
    # Each tournament game: Wteam beat Lteam
    # Create row with team1 = min(Wteam, Lteam), team2 = max
    # target = 1 if team1 won, 0 if team2 won
    train_rows = []
    for _, row in tourney.iterrows():
        season = row['Season']
        wteam = row['Wteam']
        lteam = row['Lteam']

        team1 = min(wteam, lteam)
        team2 = max(wteam, lteam)
        target = 1 if wteam == team1 else 0

        features = _build_matchup_features(
            team1, team2, season, team_season_stats, seed_lookup
        )
        features[id_column] = f"{season}_{team1}_{team2}"
        features[target_column] = target
        train_rows.append(features)

    train_df = pd.DataFrame(train_rows)

    # --- Step 4: Build test data from sample submission ---
    test_rows = []
    for _, row in sample_sub.iterrows():
        parts = str(row[id_column]).split('_')
        season = int(parts[0])
        team1 = int(parts[1])
        team2 = int(parts[2])

        features = _build_matchup_features(
            team1, team2, season, team_season_stats, seed_lookup
        )
        features[id_column] = row[id_column]
        test_rows.append(features)

    test_df = pd.DataFrame(test_rows)

    _save_data(train_df, outputs["train_data"])
    _save_data(test_df, outputs["test_data"])

    n_target_1 = (train_df[target_column] == 1).sum()
    n_target_0 = (train_df[target_column] == 0).sum()
    n_features = len([c for c in train_df.columns if c not in [id_column, target_column]])

    return (f"engineer_ncaa_features: "
            f"train={len(train_df)} (1={n_target_1}, 0={n_target_0}), "
            f"test={len(test_df)}, {n_features} features, "
            f"seasons {tourney['Season'].min()}-{tourney['Season'].max()}")


def _build_matchup_features(team1, team2, season, team_season_stats, seed_lookup):
    """Build feature dict for a matchup between team1 and team2 in a season."""
    s1 = team_season_stats.get((team1, season), {})
    s2 = team_season_stats.get((team2, season), {})

    seed1 = seed_lookup.get((team1, season), 16)  # Default to 16 (weakest)
    seed2 = seed_lookup.get((team2, season), 16)

    features = {
        # Team 1 features
        'team1_seed': seed1,
        'team1_win_rate': s1.get('win_rate', 0.5),
        'team1_avg_points_for': s1.get('avg_points_for', 70),
        'team1_avg_points_against': s1.get('avg_points_against', 70),
        'team1_avg_margin': s1.get('avg_margin', 0),
        'team1_games': s1.get('games_played', 0),

        # Team 2 features
        'team2_seed': seed2,
        'team2_win_rate': s2.get('win_rate', 0.5),
        'team2_avg_points_for': s2.get('avg_points_for', 70),
        'team2_avg_points_against': s2.get('avg_points_against', 70),
        'team2_avg_margin': s2.get('avg_margin', 0),
        'team2_games': s2.get('games_played', 0),

        # Difference features (most important for prediction)
        'seed_diff': seed1 - seed2,
        'win_rate_diff': s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5),
        'avg_margin_diff': s1.get('avg_margin', 0) - s2.get('avg_margin', 0),
        'avg_points_for_diff': s1.get('avg_points_for', 70) - s2.get('avg_points_for', 70),
        'avg_points_against_diff': s1.get('avg_points_against', 70) - s2.get('avg_points_against', 70),
    }

    return features


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific
    "engineer_ncaa_features": engineer_ncaa_features,
    # Re-exported from common modules
    "split_data": split_data,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
}
