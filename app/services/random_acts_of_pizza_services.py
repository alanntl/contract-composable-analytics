"""
Random Acts of Pizza - SLEGO Services
======================================
Competition: https://www.kaggle.com/competitions/random-acts-of-pizza
Problem Type: Binary Classification
Target: requester_received_pizza (True/False -> 1/0)
ID Column: request_id
Evaluation: ROC AUC

Competition-specific services derived from top solution notebooks:
- prepare_pizza_data: Load train/test JSON, extract numeric/temporal/text/NLP
  features, fit TF-IDF on train text and transform both, produce ready-to-model
  train and test CSVs.

Key insights from top-3 solution notebooks (533, 12, 10 votes):
1. The JSON data is far richer than the CSV (30+ fields vs 7 numeric columns)
2. Text features (request_text_edit_aware, request_title) are the strongest predictors
3. TF-IDF on request text + title with bigrams is the dominant NLP approach
4. Temporal features (hour, day_of_week, month) from unix_timestamp are predictive
5. Reddit engagement features (comments, posts, subreddits, karma) matter
6. Account age and RAOP-specific activity provide strong signal
7. ~24% base rate of receiving pizza makes class imbalance relevant
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Import from common io_services module (following PIPELINE_CONVERSION_PLAN)
from services.io_services import load_data as _load_data, save_data as _save_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_features_from_json(records: list) -> pd.DataFrame:
    """
    Convert raw JSON records into a feature DataFrame.
    Extracts numeric, temporal, text-stat, and derived features.
    """
    df = pd.json_normalize(records)
    result = pd.DataFrame()

    # ID
    result["request_id"] = df["request_id"]

    # --- Numeric features ---
    numeric_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "requester_upvotes_minus_downvotes_at_request",
    ]
    for col in numeric_cols:
        if col in df.columns:
            result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Temporal features from unix timestamp ---
    if "unix_timestamp_of_request_utc" in df.columns:
        dt = pd.to_datetime(df["unix_timestamp_of_request_utc"], unit="s")
        result["request_hour"] = dt.dt.hour
        result["request_day_of_week"] = dt.dt.dayofweek
        result["request_month"] = dt.dt.month
        result["request_day_of_month"] = dt.dt.day

    # --- Derived features ---
    up_plus = result.get(
        "requester_upvotes_plus_downvotes_at_request",
        pd.Series(0, index=result.index),
    )
    up_minus = result.get(
        "requester_upvotes_minus_downvotes_at_request",
        pd.Series(0, index=result.index),
    )
    result["upvotes"] = (up_plus + up_minus) / 2
    result["downvotes"] = (up_plus - up_minus) / 2
    result["vote_ratio"] = result["upvotes"] / (result["upvotes"] + result["downvotes"] + 1)

    # Giver known
    if "giver_username_if_known" in df.columns:
        result["giver_known"] = (df["giver_username_if_known"] != "N/A").astype(int)

    # Subreddit list length
    if "requester_subreddits_at_request" in df.columns:
        result["subreddit_count"] = df["requester_subreddits_at_request"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    # --- Text statistics ---
    for col, prefix in [
        ("request_text_edit_aware", "text"),
        ("request_title", "title"),
    ]:
        if col in df.columns:
            text = df[col].fillna("")
            result[f"{prefix}_length"] = text.str.len()
            result[f"{prefix}_word_count"] = text.str.split().str.len().fillna(0).astype(int)

    # --- Keep raw text for TF-IDF (will be dropped later) ---
    for col in ("request_text_edit_aware", "request_title"):
        if col in df.columns:
            result[col] = df[col].fillna("")

    # --- Target (train only) ---
    if "requester_received_pizza" in df.columns:
        result["requester_received_pizza"] = df["requester_received_pizza"].astype(int)

    return result


# ===========================================================================
# PUBLIC SERVICE: prepare_pizza_data
# ===========================================================================

@contract(
    inputs={
        "train": {"format": "json", "required": True, "schema": {"type": "json"}},
        "test": {"format": "json", "required": True, "schema": {"type": "json"}},
    },
    outputs={
        "train_processed": {"format": "csv", "schema": {"type": "tabular"}},
        "test_processed": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Load and engineer features for Random Acts of Pizza competition",
    tags=["competition", "feature_engineering", "nlp"],
)
def prepare_pizza_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "requester_received_pizza",
    id_column: str = "request_id",
    max_tfidf_features: int = 500,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.95,
) -> str:
    """
    Load and engineer features for the Random Acts of Pizza competition.

    Reads train.json and test.json. Produces processed train/test CSVs with
    rich features derived from all three top-scoring solution notebooks:
      - Numeric Reddit engagement features (karma, comments, posts, subreddits)
      - Temporal features (hour, day_of_week, month from unix timestamp)
      - Text statistics (length, word count for request text and title)
      - TF-IDF features from request text and title (fit on train, transform both)
      - Derived features (vote ratio, giver_known, upvotes/downvotes)

    Args:
        inputs: Must contain keys "train" and "test" (paths to JSON files)
        outputs: Must contain keys "train_processed" and "test_processed"
        target_column: Name of the target column
        id_column: Name of the ID column
        max_tfidf_features: Max TF-IDF features per text field
        ngram_max: Maximum n-gram size for TF-IDF
        min_df: Minimum document frequency for TF-IDF terms
        max_df: Maximum document frequency for TF-IDF terms
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # --- Load JSON ---
    with open(inputs["train"], "r") as f:
        train_records = json.load(f)
    with open(inputs["test"], "r") as f:
        test_records = json.load(f)

    train_df = _extract_features_from_json(train_records)
    test_df = _extract_features_from_json(test_records)
    n_train, n_test = len(train_df), len(test_df)

    # --- TF-IDF on request text (fit on train, transform both) ---
    text_fields = ["request_text_edit_aware", "request_title"]
    for field in text_fields:
        if field not in train_df.columns:
            continue

        prefix = "tfidf_text" if "text" in field else "tfidf_title"
        vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, ngram_max),
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )

        train_text = train_df[field].fillna("")
        test_text = test_df[field].fillna("")

        train_tfidf = vectorizer.fit_transform(train_text)
        test_tfidf = vectorizer.transform(test_text)

        n_feats = train_tfidf.shape[1]
        feat_names = [f"{prefix}_{i}" for i in range(n_feats)]

        train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=feat_names, index=train_df.index)
        test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=feat_names, index=test_df.index)

        train_df = pd.concat([train_df, train_tfidf_df], axis=1)
        test_df = pd.concat([test_df, test_tfidf_df], axis=1)

    # --- Drop raw text columns ---
    for field in text_fields:
        if field in train_df.columns:
            train_df = train_df.drop(columns=[field])
        if field in test_df.columns:
            test_df = test_df.drop(columns=[field])

    # --- Save ---
    _save_data(train_df, outputs["train_processed"])
    _save_data(test_df, outputs["test_processed"])

    n_features = len([c for c in train_df.columns if c not in (target_column, id_column)])
    return (
        f"prepare_pizza_data: train={n_train} rows, "
        f"test={n_test} rows, features={n_features}"
    )


# ===========================================================================
# SERVICE REGISTRY
# ===========================================================================

SERVICE_REGISTRY = {
    "prepare_pizza_data": prepare_pizza_data,
}