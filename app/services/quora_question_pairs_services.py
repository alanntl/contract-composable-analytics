"""
Quora Question Pairs - Contract-Composable Analytics Services
======================================
Competition: https://www.kaggle.com/competitions/quora-question-pairs
Problem Type: Binary Classification (log loss)
Target: is_duplicate (1=duplicate, 0=not duplicate)
ID Column: id (train) / test_id (test)

Determine whether pairs of questions have the same meaning.

Competition-specific services:
- extract_text_pair_features: Extract similarity features between two text columns
  (word overlap, Jaccard similarity, length ratios, TF-IDF cosine similarity)
  Designed generically for any text pair comparison task.

Solution notebook insights:
- Solution 1 (XGBoost): Uses pre-computed text features, scale_pos_weight for
  train/test distribution mismatch, XGBoost with eta=0.02, max_depth=6
- Solution 2 (XGBoost): Similar feature-based XGBoost approach
- Solution 3 (LSTM+GloVe): Deep learning with leaky features (question frequency,
  intersection count), class re-weighting

Key insight: The core challenge is extracting meaningful features from question pairs.
Raw text columns (question1, question2) must be converted to numeric similarity
features before training any tree-based model.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data

try:
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.preprocessing_services import split_data, drop_columns
except ImportError:
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import split_data, drop_columns


# =============================================================================
# SERVICE 1: EXTRACT TEXT PAIR FEATURES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Extract similarity features between two text columns (word overlap, length ratios, TF-IDF cosine similarity)",
    tags=["feature-engineering", "text", "nlp", "similarity", "generic"],
    version="1.0.0",
)
def extract_text_pair_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column_1: str = "question1",
    text_column_2: str = "question2",
    prefix: str = "pair_",
    include_length_features: bool = True,
    include_word_overlap: bool = True,
    include_tfidf_cosine: bool = True,
    tfidf_max_features: int = 5000,
) -> str:
    """
    Extract similarity features between two text columns.

    Designed for text pair comparison tasks (duplicate detection, paraphrase
    identification, semantic similarity). All column names are parameterized
    for reuse across competitions.

    Features extracted:
    - Length features: char length, word count, differences, ratios
    - Word overlap: common words, Jaccard similarity, directional ratios
    - TF-IDF cosine similarity: sparse row-wise cosine between TF-IDF vectors

    Parameters:
        text_column_1: First text column name
        text_column_2: Second text column name
        prefix: Prefix for new feature column names
        include_length_features: Extract character/word length features
        include_word_overlap: Extract word overlap / Jaccard features
        include_tfidf_cosine: Compute TF-IDF cosine similarity (slower)
        tfidf_max_features: Max features for TF-IDF vectorizer
    """
    df = _load_data(inputs["data"])

    t1 = df[text_column_1].fillna("").astype(str)
    t2 = df[text_column_2].fillna("").astype(str)

    features_added = []

    # -----------------------------------------------------------------
    # LENGTH FEATURES
    # -----------------------------------------------------------------
    if include_length_features:
        df[f"{prefix}len1"] = t1.str.len()
        df[f"{prefix}len2"] = t2.str.len()
        df[f"{prefix}len_diff"] = (df[f"{prefix}len1"] - df[f"{prefix}len2"]).abs()
        max_len = pd.concat([df[f"{prefix}len1"], df[f"{prefix}len2"]], axis=1).max(axis=1).replace(0, 1)
        df[f"{prefix}len_ratio"] = pd.concat([df[f"{prefix}len1"], df[f"{prefix}len2"]], axis=1).min(axis=1) / max_len

        df[f"{prefix}wc1"] = t1.str.split().str.len().fillna(0).astype(int)
        df[f"{prefix}wc2"] = t2.str.split().str.len().fillna(0).astype(int)
        df[f"{prefix}wc_diff"] = (df[f"{prefix}wc1"] - df[f"{prefix}wc2"]).abs()
        max_wc = pd.concat([df[f"{prefix}wc1"], df[f"{prefix}wc2"]], axis=1).max(axis=1).replace(0, 1)
        df[f"{prefix}wc_ratio"] = pd.concat([df[f"{prefix}wc1"], df[f"{prefix}wc2"]], axis=1).min(axis=1) / max_wc

        features_added.extend([
            f"{prefix}len1", f"{prefix}len2", f"{prefix}len_diff", f"{prefix}len_ratio",
            f"{prefix}wc1", f"{prefix}wc2", f"{prefix}wc_diff", f"{prefix}wc_ratio",
        ])

    # -----------------------------------------------------------------
    # WORD OVERLAP FEATURES
    # -----------------------------------------------------------------
    if include_word_overlap:
        words1 = t1.str.lower().str.split()
        words2 = t2.str.lower().str.split()

        def _word_overlap_stats(row_w1, row_w2):
            """Compute word overlap stats for a pair of word lists."""
            if not row_w1 or not row_w2:
                return 0, 0.0, 0.0, 0.0
            s1 = set(row_w1)
            s2 = set(row_w2)
            common = s1 & s2
            n_common = len(common)
            union = s1 | s2
            jaccard = n_common / len(union) if union else 0.0
            ratio1 = n_common / len(s1) if s1 else 0.0
            ratio2 = n_common / len(s2) if s2 else 0.0
            return n_common, jaccard, ratio1, ratio2

        # Vectorized computation via apply (efficient enough for ~400k-2M rows)
        overlap = pd.DataFrame(
            [_word_overlap_stats(w1, w2) for w1, w2 in zip(words1, words2)],
            columns=[f"{prefix}common_words", f"{prefix}jaccard",
                     f"{prefix}common_ratio1", f"{prefix}common_ratio2"],
            index=df.index,
        )
        for col in overlap.columns:
            df[col] = overlap[col]

        features_added.extend(overlap.columns.tolist())

    # -----------------------------------------------------------------
    # TF-IDF COSINE SIMILARITY
    # -----------------------------------------------------------------
    if include_tfidf_cosine:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize as sklearn_normalize

        # Fit on all texts (both columns)
        all_text = pd.concat([t1, t2], ignore_index=True)
        vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
        )
        vectorizer.fit(all_text)

        # Transform each column separately (sparse matrices)
        tfidf1 = vectorizer.transform(t1)
        tfidf2 = vectorizer.transform(t2)

        # Row-wise cosine similarity using sparse ops
        # cos(a, b) = sum(a_norm * b_norm) for each row
        tfidf1_norm = sklearn_normalize(tfidf1, norm="l2")
        tfidf2_norm = sklearn_normalize(tfidf2, norm="l2")
        cosine_sim = np.array(tfidf1_norm.multiply(tfidf2_norm).sum(axis=1)).flatten()

        df[f"{prefix}tfidf_cosine"] = cosine_sim
        features_added.append(f"{prefix}tfidf_cosine")

    _save_data(df, outputs["data"])

    return f"extract_text_pair_features: added {len(features_added)} features to {len(df)} rows"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific (but generically designed)
    "extract_text_pair_features": extract_text_pair_features,
    # Re-exported from common modules
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
    "split_data": split_data,
    "drop_columns": drop_columns,
}
