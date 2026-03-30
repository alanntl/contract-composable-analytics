"""
Scrabble Player Rating - SLEGO Services (Enhanced v3.0.0)
=========================================================
Competition: https://www.kaggle.com/competitions/scrabble-player-rating
Problem Type: Regression
Target: rating (player's rating before the game)

Competition-specific services derived from top solution notebooks:
- prepare_scrabble_data_enhanced: Full feature engineering including CRITICAL
  cumulative features that dramatically improve predictions.

Key insights from ALL THREE top solution notebooks:
1. Players play against bots (BetterBot, STEEBot, HastyBot)
2. Turn-level features (points, move length, word difficulty) are highly predictive
3. Bot performance features and game metadata add significant value
4. **CRITICAL**: Cumulative player statistics over time (expanding window features)
5. **CRITICAL**: Cumulative features broken down by game type (rating_mode × lexicon)
6. **CRITICAL**: Cumulative bot features per player-bot pair
7. Player vs Bot difference features
8. GroupKFold by nickname is the proper CV strategy
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

# ---------------------------------------------------------------------------
# Helper: load / save
# ---------------------------------------------------------------------------

def _load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {ext}")


def _save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Internal helpers (feature engineering from top notebooks)
# ---------------------------------------------------------------------------

BOT_NAMES = ["BetterBot", "STEEBot", "HastyBot"]


def _create_cumm_player_features_overall(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: Get cumulative (expanding) player statistics over time.
    These features are key to achieving top scores according to solution notebooks.

    Features created:
    - cumm_avg_player_score: Running average of player scores
    - cumm_max_player_score: Running max score
    - cumm_min_player_score: Running min score
    - cumm_player_wins: Cumulative wins
    - cumm_avg_player_win_ratio: Running win ratio
    - cumm_avg_game_duration_seconds: Running avg game duration
    """
    df = df[["nickname", "created_at", "score", "winner", "game_duration_seconds"]].copy()
    df = df.sort_values(by="created_at")

    # Initialize columns
    for col in ["cumm_avg_player_score", "cumm_max_player_score", "cumm_min_player_score",
                "cumm_player_wins", "cumm_avg_player_win_ratio", "cumm_avg_game_duration_seconds"]:
        df[col] = 0.0

    for nickname in df["nickname"].unique():
        mask = df["nickname"] == nickname
        # IMPORTANT: Shift by 1 to avoid data leakage (current game's data shouldn't be used)
        df.loc[mask, "cumm_avg_player_score"] = np.append(
            0, df[mask]["score"].expanding(min_periods=1).mean().values[:-1]
        )
        df.loc[mask, "cumm_max_player_score"] = np.append(
            0, df[mask]["score"].expanding(min_periods=1).max().values[:-1]
        )
        df.loc[mask, "cumm_min_player_score"] = np.append(
            0, df[mask]["score"].expanding(min_periods=1).min().values[:-1]
        )
        df.loc[mask, "cumm_player_wins"] = np.append(
            0, df[mask]["winner"].expanding(min_periods=1).sum().values[:-1]
        )
        counts = np.append(0, df[mask]["winner"].expanding(min_periods=1).count().values[:-1])
        df.loc[mask, "cumm_avg_player_win_ratio"] = df.loc[mask, "cumm_player_wins"] / np.maximum(counts, 1)
        df.loc[mask, "cumm_avg_game_duration_seconds"] = np.append(
            0, df[mask]["game_duration_seconds"].expanding(min_periods=1).mean().values[:-1]
        )

    df = df.fillna(0)
    df = df.sort_index()

    return df[["cumm_avg_player_score", "cumm_max_player_score", "cumm_min_player_score",
               "cumm_player_wins", "cumm_avg_player_win_ratio", "cumm_avg_game_duration_seconds"]]


def _create_cumm_player_features_by_game_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: Cumulative player features broken down by rating_mode × lexicon.
    This captures how players perform in different game configurations over time.
    """
    df = df[["nickname", "created_at", "score", "winner", "rating_mode", "lexicon",
             "game_duration_seconds"]].copy()
    df = df.sort_values(by="created_at")

    # Create columns for each combination
    result_cols = []
    for rating_mode in df["rating_mode"].unique():
        for lexicon in df["lexicon"].unique():
            for metric in ["cumm_avg_player_score", "cumm_player_wins",
                          "cumm_avg_player_win_ratio", "cumm_avg_game_duration_seconds"]:
                col_name = f"{metric}_{rating_mode}_{lexicon}"
                df[col_name] = 0.0
                result_cols.append(col_name)

    for nickname in df["nickname"].unique():
        for rating_mode in df["rating_mode"].unique():
            for lexicon in df["lexicon"].unique():
                mask = (df["nickname"] == nickname) & (df["lexicon"] == lexicon) & (df["rating_mode"] == rating_mode)
                if mask.sum() == 0:
                    continue

                prefix = f"_{rating_mode}_{lexicon}"

                df.loc[mask, f"cumm_avg_player_score{prefix}"] = np.append(
                    0, df[mask]["score"].expanding(min_periods=1).mean().values[:-1]
                )
                df.loc[mask, f"cumm_player_wins{prefix}"] = np.append(
                    0, df[mask]["winner"].expanding(min_periods=1).sum().values[:-1]
                )
                counts = np.append(0, df[mask]["winner"].expanding(min_periods=1).count().values[:-1])
                df.loc[mask, f"cumm_avg_player_win_ratio{prefix}"] = (
                    df.loc[mask, f"cumm_player_wins{prefix}"] / np.maximum(counts, 1)
                )
                df.loc[mask, f"cumm_avg_game_duration_seconds{prefix}"] = np.append(
                    0, df[mask]["game_duration_seconds"].expanding(min_periods=1).mean().values[:-1]
                )

    df = df.fillna(0)
    df = df.sort_index()

    return df[result_cols]


def _create_cumm_bot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: Cumulative bot performance features per player-bot pair.
    Tracks how each player performs against specific bots over time.
    """
    df = df[["nickname", "created_at", "bot_name", "bot_score", "bot_rating", "winner"]].copy()
    df["score_rating_ratio"] = df["bot_score"] / df["bot_rating"].replace(0, 1)
    df = df.sort_values(by="created_at")

    result_cols = []
    for bot_name in df["bot_name"].unique():
        for metric in ["cumm_avg_bot_score", "cumm_avg_bot_rating", "cumm_avg_bot_wins",
                      "cumm_avg_bot_win_ratio", "cumm_avg_bot_score_rating_ratio"]:
            col_name = f"{metric}_{bot_name}"
            df[col_name] = 0.0
            result_cols.append(col_name)

    for nickname in df["nickname"].unique():
        for bot_name in df["bot_name"].unique():
            mask = (df["nickname"] == nickname) & (df["bot_name"] == bot_name)
            if mask.sum() == 0:
                continue

            suffix = f"_{bot_name}"

            df.loc[mask, f"cumm_avg_bot_score{suffix}"] = np.append(
                0, df[mask]["bot_score"].expanding(min_periods=1).mean().values[:-1]
            )
            # Bot rating is known before game, so no shift needed
            df.loc[mask, f"cumm_avg_bot_rating{suffix}"] = (
                df[mask]["bot_rating"].expanding(min_periods=1).mean().values
            )
            # Bot wins = when player loses (winner == 0)
            df.loc[mask, f"cumm_avg_bot_wins{suffix}"] = np.append(
                0, df[mask]["winner"].expanding(min_periods=1).apply(lambda x: (x == 0).sum()).values[:-1]
            )
            counts = np.append(0, df[mask]["winner"].expanding(min_periods=1).count().values[:-1])
            df.loc[mask, f"cumm_avg_bot_win_ratio{suffix}"] = (
                df.loc[mask, f"cumm_avg_bot_wins{suffix}"] / np.maximum(counts, 1)
            )
            df.loc[mask, f"cumm_avg_bot_score_rating_ratio{suffix}"] = np.append(
                0, df[mask]["score_rating_ratio"].expanding(min_periods=1).mean().values[:-1]
            )

    df = df.fillna(0)
    df = df.sort_index()

    return df[result_cols]


def _create_cumm_turns_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: Cumulative turn-based features including player-vs-bot differences.
    Tracks how player's turn performance evolves over time.
    """
    turn_features = ['turn_type_Play', 'turn_type_End', 'turn_type_Exchange', 'turn_type_Pass',
                     'turn_type_Timeout', 'turn_type_Challenge', 'turn_type_Six-Zero Rule',
                     'turn_type_None', 'points_mean', 'points_max', 'move_len_mean', 'move_len_max',
                     'difficult_letters_mean', 'difficult_letters_sum', 'points_per_letter_mean',
                     'direction_of_play_mean', 'rack_len_less_than_7_sum', 'turn_number_count']

    # Create difference features (player vs bot)
    df = df.copy()
    df['play_counts_diff'] = df.get('turn_type_Play', 0) - df.get('bot_turn_type_Play', 0)
    df['avg_points_diff'] = df.get('points_mean', 0) - df.get('bot_points_mean', 0)
    df['avg_move_len_diff'] = df.get('move_len_mean', 0) - df.get('bot_move_len_mean', 0)
    df['avg_points_per_letter_diff'] = df.get('points_per_letter_mean', 0) - df.get('bot_points_per_letter_mean', 0)
    df['difficult_letters_count_diff'] = df.get('difficult_letters_sum', 0) - df.get('bot_difficult_letters_sum', 0)

    diff_features = ['play_counts_diff', 'avg_points_diff', 'avg_move_len_diff',
                     'avg_points_per_letter_diff', 'difficult_letters_count_diff']

    # Filter to available columns
    available_turn_features = [c for c in turn_features if c in df.columns]

    df = df.sort_values(by="created_at")

    result_cols = diff_features.copy()
    for feat in available_turn_features:
        col_name = f"cumm_{feat}_average"
        df[col_name] = 0.0
        result_cols.append(col_name)

    for nickname in df["nickname"].unique():
        mask = df["nickname"] == nickname
        for feat in available_turn_features:
            df.loc[mask, f"cumm_{feat}_average"] = np.append(
                0, df[mask][feat].expanding(min_periods=1).mean().values[:-1]
            )

    df = df.fillna(0)
    df = df.sort_index()

    return df[result_cols]


def _create_turn_features(turns: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate turn-level data to game+player level.
    Based on the approach from all 3 top solution notebooks.
    """
    df = turns.copy()

    # Basic turn features
    df["rack_len"] = df["rack"].str.len()
    df["rack_len_less_than_7"] = (df["rack_len"] < 7).astype(int)
    df["move_len"] = df["move"].str.len()
    df["move"] = df["move"].fillna("None")

    # Rare/difficult letters
    rare_letters = set("ZQJXKVYWG")
    df["difficult_letters"] = df["move"].apply(
        lambda x: sum(1 for c in str(x) if c in rare_letters)
    )
    df["points_per_letter"] = df["points"] / df["move_len"].replace(0, np.nan)

    # Turn type dummies
    df["turn_type"] = df["turn_type"].fillna("None")
    turn_type_unique = df["turn_type"].unique()
    df = pd.get_dummies(df, columns=["turn_type"])
    dummy_features = [f"turn_type_{v}" for v in turn_type_unique]

    # Board position features
    df["direction_of_play"] = df["location"].apply(
        lambda x: 1 if str(x)[:1].isdigit() else 0
    )

    # Aggregate counts of turn types
    agg_counts = {f: "sum" for f in dummy_features if f in df.columns}
    grouped_counts = df.groupby(["game_id", "nickname"], as_index=False).agg(agg_counts)

    # Aggregate stats for Play turns only
    play_col = "turn_type_Play"
    agg_stats = {
        "points": ["mean", "max"],
        "move_len": ["mean", "max"],
        "difficult_letters": ["mean", "sum"],
        "points_per_letter": "mean",
        "direction_of_play": "mean",
        "rack_len_less_than_7": "sum",
        "turn_number": "count",
    }
    play_mask = df[play_col] == 1 if play_col in df.columns else pd.Series(True, index=df.index)
    grouped_stats = (
        df[play_mask]
        .groupby(["game_id", "nickname"], as_index=False)
        .agg(agg_stats)
    )
    # Flatten multi-level columns
    grouped_stats.columns = [
        "_".join(a) if a[0] not in ("game_id", "nickname") else a[0]
        for a in grouped_stats.columns.to_flat_index()
    ]

    # Merge
    grouped = grouped_counts.merge(grouped_stats, how="outer", on=["game_id", "nickname"])
    grouped = grouped.fillna(0)

    return grouped


def _build_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    turns: pd.DataFrame,
    games: pd.DataFrame,
    use_cumulative: bool = True,
) -> tuple:
    """
    Full feature engineering pipeline based on top solution notebooks.
    Returns (train_df, test_df) with numeric features ready for modeling.

    CRITICAL: When use_cumulative=True, adds the cumulative features that
    differentiate top solutions from average submissions.
    """
    # --- 1. Create turn features ---
    turn_features = _create_turn_features(turns)

    # --- 2. Combine train & test for consistent processing ---
    df = pd.concat([train, test], ignore_index=False)
    df = df.merge(turn_features, how="left", on=["game_id", "nickname"])

    # --- 3. Separate bot vs player data ---
    turn_cols = [c for c in turn_features.columns if c not in ("game_id", "nickname")]
    bot_df = df[["game_id", "nickname", "score", "rating"] + turn_cols].copy()
    bot_df["bot_name"] = bot_df["nickname"].apply(
        lambda x: x if x in BOT_NAMES else np.nan
    )
    bot_df = bot_df.dropna(subset=["bot_name"])
    bot_df = bot_df[["game_id", "bot_name", "score", "rating"] + turn_cols]
    bot_df.columns = (
        ["game_id", "bot_name", "bot_score", "bot_rating"]
        + ["bot_" + c for c in turn_cols]
    )

    # Keep only human players
    df = df[~df["nickname"].isin(BOT_NAMES)]
    df = df.merge(bot_df, on="game_id", how="left")

    # --- 4. Merge game metadata ---
    df = df.merge(games, on="game_id", how="left")
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
    if "first" in df.columns:
        df["first"] = df["first"].apply(
            lambda x: "bot" if x in BOT_NAMES else "player"
        )

    # --- 5. Simple aggregate player statistics (fast) ---
    player_stats = df.groupby("nickname").agg(
        player_mean_score=("score", "mean"),
        player_game_count=("score", "count"),
    ).reset_index()
    df = df.merge(player_stats, on="nickname", how="left")

    # --- 6. Score difference features (player vs bot) ---
    df["score_diff"] = df["score"] - df.get("bot_score", 0)
    if "points_mean" in df.columns and "bot_points_mean" in df.columns:
        df["avg_points_diff"] = df["points_mean"] - df["bot_points_mean"]
    if "move_len_mean" in df.columns and "bot_move_len_mean" in df.columns:
        df["avg_move_len_diff"] = df["move_len_mean"] - df["bot_move_len_mean"]

    # --- 7. CRITICAL: Add cumulative features from top solutions ---
    if use_cumulative and "created_at" in df.columns:
        # Set index for joining
        df = df.reset_index(drop=True)
        original_index = df.index.copy()

        # 7a. Overall cumulative player features
        cumm_overall = _create_cumm_player_features_overall(df)
        for col in cumm_overall.columns:
            df[col] = cumm_overall[col].values

        # 7b. Cumulative features by game type (rating_mode × lexicon)
        cumm_by_type = _create_cumm_player_features_by_game_type(df)
        for col in cumm_by_type.columns:
            df[col] = cumm_by_type[col].values

        # 7c. Cumulative bot features
        cumm_bot = _create_cumm_bot_features(df)
        for col in cumm_bot.columns:
            df[col] = cumm_bot[col].values

        # 7d. Cumulative turn features and difference features
        cumm_turns = _create_cumm_turns_features(df)
        for col in cumm_turns.columns:
            df[col] = cumm_turns[col].values

    # --- 8. Ordinal-encode all categoricals ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Remove target-related and identifier columns from encoding
    cat_cols = [c for c in cat_cols if c not in ("game_id",)]
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    # --- 9. Drop non-feature columns ---
    drop_cols = ["created_at"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # --- 10. Split back to train/test ---
    train_ids = set(train["game_id"].values)
    test_ids = set(test["game_id"].values)
    train_out = df[df["game_id"].isin(train_ids)].copy()
    test_out = df[df["game_id"].isin(test_ids)].copy()

    # Deduplicate: keep one row per game_id (first human player row)
    train_out = train_out.drop_duplicates(subset=["game_id"], keep="first")
    test_out = test_out.drop_duplicates(subset=["game_id"], keep="first")

    return train_out, test_out


# ===========================================================================
# PUBLIC SERVICE: prepare_scrabble_data
# ===========================================================================

def _reconstruct_missing_test_rows(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    turns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure test has rows for ALL game_ids in sample_submission.

    Two cases of missing human player data:
    1. game_ids in submission but not in test.csv at all (6,840 games)
    2. game_ids in test.csv with only bot entries (5,408 games) - the human
       player row is missing from test.csv

    For both cases, reconstruct the human player row from turns.csv using
    the last turn's cumulative score.
    """
    sub_ids = set(submission["game_id"])

    # Case 1: game_ids not in test at all
    extra_ids = sub_ids - set(test["game_id"])

    # Case 2: game_ids in test with only bot rows (no human entry)
    test_humans = test[~test["nickname"].isin(BOT_NAMES)]
    bot_only_ids = sub_ids - set(test_humans["game_id"]) - extra_ids

    missing_ids = extra_ids | bot_only_ids
    if not missing_ids:
        return test

    # Reconstruct from turns: final score per player
    missing_turns = turns[turns["game_id"].isin(missing_ids)]
    last_turns = (
        missing_turns.sort_values("turn_number")
        .groupby(["game_id", "nickname"])
        .last()
        .reset_index()
    )
    new_rows = last_turns[["game_id", "nickname", "score"]].copy()
    new_rows["rating"] = np.nan

    return pd.concat([test, new_rows], ignore_index=True)


@contract(
    inputs={
        "train": {"format": "csv", "required": True},
        "test": {"format": "csv", "required": True},
        "turns": {"format": "csv", "required": True},
        "games": {"format": "csv", "required": True},
        "submission": {"format": "csv", "required": True},
    },
    outputs={
        "train_processed": {"format": "csv"},
        "test_processed": {"format": "csv"},
    },
    description="Engineer features from turns/games data for Scrabble Player Rating prediction",
    tags=["preprocessing", "feature-engineering", "scrabble", "regression"],
    version="2.0.0",
)
def prepare_scrabble_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "rating",
    bot_names: Optional[List[str]] = None,
) -> str:
    """
    Load and engineer features for the Scrabble Player Rating competition.

    Reads train.csv, test.csv, turns.csv, games.csv, and sample_submission.csv.
    Produces processed train/test CSVs with rich features derived from all three
    top-scoring solution notebooks:
      - Turn-level statistics aggregated per game per player
      - Bot performance features (score, rating, turn stats)
      - Game metadata (lexicon, time control, winner, etc.)
      - Player aggregate statistics (mean score, game count)
      - Player-vs-bot difference features
      - Ordinal encoding of all categoricals

    Handles extra game_ids in sample_submission that are not in test.csv
    by reconstructing player data from turns.csv.

    Args:
        inputs: Must contain keys "train", "test", "turns", "games", "submission"
        outputs: Must contain keys "train_processed", "test_processed"
        target_column: Name of the target column (default "rating")
        bot_names: List of bot nicknames (default ["BetterBot", "STEEBot", "HastyBot"])
    """
    global BOT_NAMES
    if bot_names is not None:
        BOT_NAMES = bot_names

    train = _load_data(inputs["train"])
    test = _load_data(inputs["test"])
    turns = _load_data(inputs["turns"])
    games = _load_data(inputs["games"])
    submission = _load_data(inputs["submission"])

    # Reconstruct test rows for missing game_ids in submission
    test = _reconstruct_missing_test_rows(submission, test, turns)

    train_processed, test_processed = _build_feature_matrix(train, test, turns, games)

    # Fill NaN values that arise from missing merges
    train_processed = train_processed.fillna(0)
    test_processed = test_processed.fillna(0)

    # Filter test to only include game_ids from sample_submission
    sub_ids = set(submission["game_id"].values)
    test_processed = test_processed[test_processed["game_id"].isin(sub_ids)]

    # Drop target column from test (not available at inference time)
    if target_column in test_processed.columns:
        test_processed = test_processed.drop(columns=[target_column])

    _save_data(train_processed, outputs["train_processed"])
    _save_data(test_processed, outputs["test_processed"])

    n_features = len([c for c in train_processed.columns if c != target_column])
    return (
        f"prepare_scrabble_data: train={len(train_processed)} rows, "
        f"test={len(test_processed)} rows, features={n_features}"
    )


@contract(
    inputs={
        "train": {"format": "csv", "required": True},
        "test": {"format": "csv", "required": True},
        "turns": {"format": "csv", "required": True},
        "games": {"format": "csv", "required": True},
        "submission": {"format": "csv", "required": True},
    },
    outputs={
        "train_processed": {"format": "csv"},
        "test_processed": {"format": "csv"},
    },
    description="Enhanced feature engineering with CRITICAL cumulative features for top performance",
    tags=["preprocessing", "feature-engineering", "scrabble", "regression", "enhanced"],
    version="3.0.0",
)
def prepare_scrabble_data_enhanced(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "rating",
    bot_names: Optional[List[str]] = None,
    use_cumulative: bool = True,
) -> str:
    """
    ENHANCED: Load and engineer features for the Scrabble Player Rating competition.

    This version includes the CRITICAL cumulative features from top solution notebooks:
    - Cumulative player performance (overall expanding window features)
    - Cumulative player performance by game type (rating_mode × lexicon)
    - Cumulative bot performance per player-bot pair
    - Cumulative turn statistics
    - Player vs bot difference features

    These cumulative features typically improve CV score by 20-30+ points.

    Args:
        inputs: Must contain keys "train", "test", "turns", "games", "submission"
        outputs: Must contain keys "train_processed", "test_processed"
        target_column: Name of the target column (default "rating")
        bot_names: List of bot nicknames (default ["BetterBot", "STEEBot", "HastyBot"])
        use_cumulative: Whether to include cumulative features (default True)
    """
    global BOT_NAMES
    if bot_names is not None:
        BOT_NAMES = bot_names

    train = _load_data(inputs["train"])
    test = _load_data(inputs["test"])
    turns = _load_data(inputs["turns"])
    games = _load_data(inputs["games"])
    submission = _load_data(inputs["submission"])

    # Reconstruct test rows for missing game_ids in submission
    test = _reconstruct_missing_test_rows(submission, test, turns)

    train_processed, test_processed = _build_feature_matrix(
        train, test, turns, games, use_cumulative=use_cumulative
    )

    # Fill NaN values that arise from missing merges
    train_processed = train_processed.fillna(0)
    test_processed = test_processed.fillna(0)

    # Filter test to only include game_ids from sample_submission
    sub_ids = set(submission["game_id"].values)
    test_processed = test_processed[test_processed["game_id"].isin(sub_ids)]

    # Drop target column from test (not available at inference time)
    if target_column in test_processed.columns:
        test_processed = test_processed.drop(columns=[target_column])

    _save_data(train_processed, outputs["train_processed"])
    _save_data(test_processed, outputs["test_processed"])

    n_features = len([c for c in train_processed.columns if c != target_column])
    return (
        f"prepare_scrabble_data_enhanced: train={len(train_processed)} rows, "
        f"test={len(test_processed)} rows, features={n_features} (cumulative={use_cumulative})"
    )


# ===========================================================================
# SERVICE REGISTRY
# ===========================================================================

SERVICE_REGISTRY = {
    "prepare_scrabble_data": prepare_scrabble_data,
    "prepare_scrabble_data_enhanced": prepare_scrabble_data_enhanced,
}
