"""
TMDB Box Office Prediction - Contract-Composable Analytics Services
=============================================
Competition: https://www.kaggle.com/competitions/tmdb-box-office-prediction
Problem Type: Regression
Target: revenue
Evaluation: RMSLE (Root Mean Squared Logarithmic Error)

Competition-specific services derived from top solution notebooks:
- engineer_tmdb_features: Parse JSON columns, extract date features, create ratio/count/binary features
- prepare_tmdb_test: Apply same feature engineering to test data for prediction

Key insights from top solutions (notebooks 01-03):
1. JSON columns (genres, cast, crew, production_companies, Keywords, etc.) contain rich information
2. Log-transform of target (revenue) and budget dramatically improves performance
3. Release date features (month, year, dayofweek, quarter) are important
4. Ratio features (budget/runtime, budget/popularity) are strong predictors
5. Count features (num_cast, num_crew, num_keywords, num_genres) add signal
6. Binary features (has_homepage, belongs_to_collection, is_english) are useful
7. Ensemble of LightGBM + XGBoost gives best results
"""

import os
import sys
import json
import ast
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data


def _safe_parse_json_col(val):
    """Safely parse a JSON-like string column value into a Python object."""
    if pd.isna(val) or val == '' or val == 'nan':
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, (list, dict)) else []
    except (ValueError, SyntaxError):
        return []


def _extract_json_features(df, col, name_key='name'):
    """Extract count and top-item features from a JSON list column."""
    parsed = df[col].apply(_safe_parse_json_col)

    # Count of items
    df[f'{col}_count'] = parsed.apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Extract names for one-hot encoding of top categories
    all_names = []
    for items in parsed:
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and name_key in item:
                    all_names.append(item[name_key])

    # Get top categories (appearing >= 10 times)
    from collections import Counter
    name_counts = Counter(all_names)
    top_names = [name for name, count in name_counts.most_common(30) if count >= 10]

    # Create binary columns for top categories
    for name in top_names:
        safe_col = f'{col}_{name}'.replace(' ', '_').replace('-', '_').replace("'", "")
        df[safe_col] = parsed.apply(
            lambda x: 1 if isinstance(x, list) and any(
                isinstance(i, dict) and i.get(name_key) == name for i in x
            ) else 0
        )

    return df


@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Engineer features from TMDB movie data: parse JSON, extract dates, create ratios",
    tags=["feature-engineering", "tmdb", "movies", "json-parsing", "generic"],
    version="1.0.0",
)
def engineer_tmdb_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "revenue",
    id_column: str = "id",
    log_transform_target: bool = True,
    log_transform_budget: bool = True,
    json_columns: Optional[List[str]] = None,
    date_column: str = "release_date",
) -> str:
    """
    Engineer features from TMDB movie data.

    Parses JSON-embedded columns, extracts release date features, creates
    ratio/count/binary features. Designed based on top Kaggle solution notebooks.

    G1: Single responsibility - feature engineering for movie box office data
    G4: Parameterized columns and transforms
    G3: Deterministic output

    Parameters:
        target_column: Target column to log-transform (if present)
        id_column: ID column to preserve
        log_transform_target: Whether to apply log1p to target
        log_transform_budget: Whether to apply log1p to budget
        json_columns: JSON columns to parse (default: genres, cast, crew, etc.)
        date_column: Release date column name
    """
    df = _load_data(inputs["data"])
    initial_cols = len(df.columns)

    # =========================================================================
    # 1. Parse JSON columns and extract count + top category features
    # =========================================================================
    if json_columns is None:
        json_columns = ['genres', 'production_companies', 'production_countries',
                        'spoken_languages', 'Keywords', 'cast', 'crew']

    for col in json_columns:
        if col in df.columns:
            df = _extract_json_features(df, col)

    # =========================================================================
    # 2. belongs_to_collection: binary flag + collection name encoding
    # =========================================================================
    if 'belongs_to_collection' in df.columns:
        parsed_collection = df['belongs_to_collection'].apply(_safe_parse_json_col)
        df['has_collection'] = parsed_collection.apply(
            lambda x: 1 if isinstance(x, list) and len(x) > 0 else (
                1 if isinstance(x, dict) and len(x) > 0 else 0
            )
        )

    # =========================================================================
    # 3. Release date features
    # =========================================================================
    if date_column in df.columns:
        # Parse release_date (format: mm/dd/yy or similar)
        release_date = pd.to_datetime(df[date_column], format='mixed', errors='coerce')

        df['release_year'] = release_date.dt.year
        df['release_month'] = release_date.dt.month
        df['release_day'] = release_date.dt.day
        df['release_dayofweek'] = release_date.dt.dayofweek
        df['release_quarter'] = release_date.dt.quarter

        # Fix 2-digit year parsing (e.g., 21 -> 2021 vs 1921)
        mask = df['release_year'] > 2025
        df.loc[mask, 'release_year'] = df.loc[mask, 'release_year'] - 100

    # =========================================================================
    # 4. Binary indicator features
    # =========================================================================
    if 'homepage' in df.columns:
        df['has_homepage'] = df['homepage'].notna().astype(int)

    if 'tagline' in df.columns:
        df['has_tagline'] = df['tagline'].notna().astype(int)

    if 'original_language' in df.columns:
        df['is_english'] = (df['original_language'] == 'en').astype(int)

    if 'original_title' in df.columns and 'title' in df.columns:
        df['title_different'] = (df['original_title'] != df['title']).astype(int)

    if 'status' in df.columns:
        df['is_released'] = (df['status'] == 'Released').astype(int)

    # =========================================================================
    # 5. Text length features
    # =========================================================================
    if 'title' in df.columns:
        df['title_word_count'] = df['title'].fillna('').str.split().str.len()

    if 'overview' in df.columns:
        df['overview_word_count'] = df['overview'].fillna('').str.split().str.len()

    if 'tagline' in df.columns:
        df['tagline_word_count'] = df['tagline'].fillna('').str.split().str.len()

    # =========================================================================
    # 6. Log transforms
    # =========================================================================
    if log_transform_budget and 'budget' in df.columns:
        df['log_budget'] = np.log1p(df['budget'])

    if log_transform_target and target_column in df.columns:
        df[target_column] = np.log1p(df[target_column])

    # =========================================================================
    # 7. Ratio features (using log_budget if available, else budget)
    # =========================================================================
    budget_col = 'log_budget' if 'log_budget' in df.columns else 'budget'

    if budget_col in df.columns and 'runtime' in df.columns:
        runtime_safe = df['runtime'].replace(0, np.nan)
        df['budget_runtime_ratio'] = df[budget_col] / runtime_safe
        df['budget_runtime_ratio'] = df['budget_runtime_ratio'].fillna(0)

    if budget_col in df.columns and 'popularity' in df.columns:
        pop_safe = df['popularity'].replace(0, np.nan)
        df['budget_popularity_ratio'] = df[budget_col] / pop_safe
        df['budget_popularity_ratio'] = df['budget_popularity_ratio'].fillna(0)

    if 'release_year' in df.columns and budget_col in df.columns:
        year_safe = df['release_year'].replace(0, np.nan)
        df['budget_year_ratio'] = df[budget_col] / (year_safe * year_safe)
        df['budget_year_ratio'] = df['budget_year_ratio'].fillna(0)

    if 'release_year' in df.columns and 'popularity' in df.columns:
        df['year_popularity_ratio'] = df['popularity'] / df['release_year'].replace(0, np.nan)
        df['year_popularity_ratio'] = df['year_popularity_ratio'].fillna(0)

    # =========================================================================
    # 8. Mean aggregations by year
    # =========================================================================
    if 'release_year' in df.columns:
        if 'popularity' in df.columns:
            year_pop = df.groupby('release_year')['popularity'].transform('mean')
            df['popularity_mean_year'] = df['popularity'] / year_pop.replace(0, np.nan)
            df['popularity_mean_year'] = df['popularity_mean_year'].fillna(1)

        if 'runtime' in df.columns:
            df['mean_runtime_by_year'] = df.groupby('release_year')['runtime'].transform('mean')

        if budget_col in df.columns:
            df['mean_budget_by_year'] = df.groupby('release_year')[budget_col].transform('mean')

    # =========================================================================
    # 9. Drop original text/JSON/ID columns that can't be used by models
    # =========================================================================
    drop_cols = [
        'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview',
        'poster_path', 'production_companies', 'production_countries',
        'release_date', 'spoken_languages', 'status', 'title', 'Keywords',
        'cast', 'crew', 'original_language', 'original_title', 'tagline',
        'budget',  # keep log_budget instead
    ]
    to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=to_drop)

    # =========================================================================
    # 10. Fill remaining NaN with 0
    # =========================================================================
    df = df.fillna(0)

    # Ensure all columns are numeric
    for col in df.columns:
        if col not in [id_column, target_column] and df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    _save_data(df, outputs["data"])

    return (
        f"engineer_tmdb_features: {initial_cols} -> {len(df.columns)} columns, "
        f"{len(df)} rows"
    )


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Process train and test together to ensure consistent feature columns",
    tags=["feature-engineering", "tmdb", "movies", "preprocessing", "generic"],
    version="1.0.0",
)
def engineer_tmdb_features_combined(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "revenue",
    id_column: str = "id",
    log_transform_budget: bool = True,
    json_columns: Optional[List[str]] = None,
    date_column: str = "release_date",
) -> str:
    """
    Process train and test data together to ensure consistent one-hot encoded columns.

    Combines train+test, applies feature engineering, then splits back.
    Target is log1p transformed on train only.

    Parameters:
        target_column: Target column (only in train)
        id_column: ID column to preserve
        log_transform_budget: Whether to apply log1p to budget
        json_columns: JSON columns to parse
        date_column: Release date column name
    """
    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    n_train = len(train_df)

    # Save target before combining
    y_train = train_df[target_column].copy() if target_column in train_df.columns else None

    # Remove target from train for combined processing
    if target_column in train_df.columns:
        train_df = train_df.drop(columns=[target_column])

    # Remove revenue from test if present (NaN placeholder)
    if target_column in test_df.columns:
        test_df = test_df.drop(columns=[target_column])

    # Combine
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Parse JSON columns
    if json_columns is None:
        json_columns = ['genres', 'production_companies', 'production_countries',
                        'spoken_languages', 'Keywords', 'cast', 'crew']

    for col in json_columns:
        if col in combined.columns:
            combined = _extract_json_features(combined, col)

    # belongs_to_collection: binary flag
    if 'belongs_to_collection' in combined.columns:
        parsed_collection = combined['belongs_to_collection'].apply(_safe_parse_json_col)
        combined['has_collection'] = parsed_collection.apply(
            lambda x: 1 if isinstance(x, list) and len(x) > 0 else (
                1 if isinstance(x, dict) and len(x) > 0 else 0
            )
        )

    # Release date features
    if date_column in combined.columns:
        release_date = pd.to_datetime(combined[date_column], format='mixed', errors='coerce')
        combined['release_year'] = release_date.dt.year
        combined['release_month'] = release_date.dt.month
        combined['release_day'] = release_date.dt.day
        combined['release_dayofweek'] = release_date.dt.dayofweek
        combined['release_quarter'] = release_date.dt.quarter
        mask = combined['release_year'] > 2025
        combined.loc[mask, 'release_year'] = combined.loc[mask, 'release_year'] - 100

    # Binary indicators
    if 'homepage' in combined.columns:
        combined['has_homepage'] = combined['homepage'].notna().astype(int)
    if 'tagline' in combined.columns:
        combined['has_tagline'] = combined['tagline'].notna().astype(int)
    if 'original_language' in combined.columns:
        combined['is_english'] = (combined['original_language'] == 'en').astype(int)
    if 'original_title' in combined.columns and 'title' in combined.columns:
        combined['title_different'] = (combined['original_title'] != combined['title']).astype(int)
    if 'status' in combined.columns:
        combined['is_released'] = (combined['status'] == 'Released').astype(int)

    # Text length features
    if 'title' in combined.columns:
        combined['title_word_count'] = combined['title'].fillna('').str.split().str.len()
    if 'overview' in combined.columns:
        combined['overview_word_count'] = combined['overview'].fillna('').str.split().str.len()
    if 'tagline' in combined.columns:
        combined['tagline_word_count'] = combined['tagline'].fillna('').str.split().str.len()

    # Log budget
    if log_transform_budget and 'budget' in combined.columns:
        combined['log_budget'] = np.log1p(combined['budget'])

    # Ratio features
    budget_col = 'log_budget' if 'log_budget' in combined.columns else 'budget'
    if budget_col in combined.columns and 'runtime' in combined.columns:
        runtime_safe = combined['runtime'].replace(0, np.nan)
        combined['budget_runtime_ratio'] = combined[budget_col] / runtime_safe
        combined['budget_runtime_ratio'] = combined['budget_runtime_ratio'].fillna(0)
    if budget_col in combined.columns and 'popularity' in combined.columns:
        pop_safe = combined['popularity'].replace(0, np.nan)
        combined['budget_popularity_ratio'] = combined[budget_col] / pop_safe
        combined['budget_popularity_ratio'] = combined['budget_popularity_ratio'].fillna(0)
    if 'release_year' in combined.columns and budget_col in combined.columns:
        year_safe = combined['release_year'].replace(0, np.nan)
        combined['budget_year_ratio'] = combined[budget_col] / (year_safe * year_safe)
        combined['budget_year_ratio'] = combined['budget_year_ratio'].fillna(0)
    if 'release_year' in combined.columns and 'popularity' in combined.columns:
        combined['year_popularity_ratio'] = combined['popularity'] / combined['release_year'].replace(0, np.nan)
        combined['year_popularity_ratio'] = combined['year_popularity_ratio'].fillna(0)

    # Mean aggregations by year
    if 'release_year' in combined.columns:
        if 'popularity' in combined.columns:
            year_pop = combined.groupby('release_year')['popularity'].transform('mean')
            combined['popularity_mean_year'] = combined['popularity'] / year_pop.replace(0, np.nan)
            combined['popularity_mean_year'] = combined['popularity_mean_year'].fillna(1)
        if 'runtime' in combined.columns:
            combined['mean_runtime_by_year'] = combined.groupby('release_year')['runtime'].transform('mean')
        if budget_col in combined.columns:
            combined['mean_budget_by_year'] = combined.groupby('release_year')[budget_col].transform('mean')

    # Drop text/JSON columns
    drop_cols = [
        'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview',
        'poster_path', 'production_companies', 'production_countries',
        'release_date', 'spoken_languages', 'status', 'title', 'Keywords',
        'cast', 'crew', 'original_language', 'original_title', 'tagline',
        'budget',
    ]
    to_drop = [c for c in drop_cols if c in combined.columns]
    combined = combined.drop(columns=to_drop)

    # Fill NaN and ensure numeric
    combined = combined.fillna(0)
    for col in combined.columns:
        if col != id_column and combined[col].dtype == 'object':
            combined[col] = pd.factorize(combined[col])[0]

    # Split back
    train_out = combined.iloc[:n_train].copy()
    test_out = combined.iloc[n_train:].copy()

    # Add log1p target back to train
    if y_train is not None:
        train_out[target_column] = np.log1p(y_train.values)

    _save_data(train_out, outputs["train_data"])
    _save_data(test_out, outputs["test_data"])

    return (
        f"engineer_tmdb_features_combined: train={len(train_out)} rows x {len(train_out.columns)} cols, "
        f"test={len(test_out)} rows x {len(test_out.columns)} cols"
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "engineer_tmdb_features_combined": engineer_tmdb_features_combined,
}
