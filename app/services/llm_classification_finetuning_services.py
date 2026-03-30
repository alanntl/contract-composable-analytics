"""
LLM Classification Finetuning - Contract-Composable Analytics Services
================================================
Competition: https://www.kaggle.com/competitions/llm-classification-finetuning
Problem Type: Multiclass classification (3 classes)
Target: winner_model_a, winner_model_b, winner_tie (one-hot -> single target)

This competition compares responses from two LLMs to a given prompt and predicts
which model's response is preferred. Top solutions use fine-tuned LLMs (Gemma2-9b,
Llama3-8b) with ensemble inference. This Contract-Composable Analytics pipeline uses text feature engineering
+ TF-IDF + LightGBM as a local baseline approach.

Competition-specific services:
- extract_llm_comparison_features: Extract text comparison features from prompt,
  response_a, response_b (response lengths, differences, ratios)

Reused services from common modules:
- combine_text_columns (text_services): Combine text columns for TF-IDF
- extract_text_features (text_services): Handcrafted text features
- vectorize_tfidf / transform_tfidf (text_services): TF-IDF vectorization
- drop_text_columns (text_services): Drop text columns after feature extraction
- convert_onehot_to_multiclass (preprocessing_services): Convert one-hot targets
- split_data (preprocessing_services): Train/validation split
- train_lightgbm_classifier (classification_services): LightGBM training
- predict_multiclass_submission (classification_services): Multiclass probability prediction
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# =============================================================================
# HELPERS: Import from shared io_utils
# =============================================================================
from services.io_utils import load_data as _load_data, save_data as _save_data

# =============================================================================
# REUSED SERVICES (imported for SERVICE_REGISTRY)
# =============================================================================
from services.text_services import (
    combine_text_columns,
    extract_text_features,
    vectorize_tfidf,
    transform_tfidf,
    drop_text_columns,
)
from services.preprocessing_services import (
    convert_onehot_to_multiclass,
    split_data,
    drop_columns,
)
from services.classification_services import (
    train_lightgbm_classifier,
    predict_multiclass_submission,
)


# =============================================================================
# COMPETITION-SPECIFIC SERVICE: EXTRACT LLM COMPARISON FEATURES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Extract text comparison features from multi-response LLM data",
    tags=["feature-engineering", "text", "nlp", "comparison", "generic"],
    version="1.0.0",
)
def extract_llm_comparison_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    prompt_column: str = "prompt",
    response_a_column: str = "response_a",
    response_b_column: str = "response_b",
    prefix: str = "cmp_",
) -> str:
    """Extract comparison features between two LLM responses and a prompt.

    Designed for competitions where two model responses are compared. Extracts
    per-response features (length, word count) and cross-response comparison
    features (length difference, ratio, word count difference).

    G1 Compliance: Generic, works with any multi-response text dataset.
    G4 Compliance: All column names parameterized.

    Parameters:
        prompt_column: Column containing the prompt text
        response_a_column: Column containing the first response
        response_b_column: Column containing the second response
        prefix: Prefix for generated feature columns
    """
    df = _load_data(inputs["data"])
    features_added = []

    # Helper to safely compute string length
    def _safe_len(series):
        return series.fillna("").astype(str).str.len()

    def _safe_word_count(series):
        return series.fillna("").astype(str).str.split().str.len().fillna(0).astype(int)

    # Per-column features
    for col, label in [(prompt_column, "prompt"), (response_a_column, "resp_a"), (response_b_column, "resp_b")]:
        if col in df.columns:
            df[f"{prefix}{label}_len"] = _safe_len(df[col])
            df[f"{prefix}{label}_words"] = _safe_word_count(df[col])
            features_added.extend([f"{prefix}{label}_len", f"{prefix}{label}_words"])

    # Comparison features between responses
    if response_a_column in df.columns and response_b_column in df.columns:
        len_a = _safe_len(df[response_a_column])
        len_b = _safe_len(df[response_b_column])
        words_a = _safe_word_count(df[response_a_column])
        words_b = _safe_word_count(df[response_b_column])

        # Length differences
        df[f"{prefix}len_diff"] = len_a - len_b
        df[f"{prefix}len_ratio"] = len_a / len_b.replace(0, 1)
        df[f"{prefix}words_diff"] = words_a - words_b
        df[f"{prefix}words_ratio"] = words_a / words_b.replace(0, 1)

        # Absolute differences
        df[f"{prefix}len_abs_diff"] = (len_a - len_b).abs()
        df[f"{prefix}words_abs_diff"] = (words_a - words_b).abs()

        # Which response is longer (binary indicator)
        df[f"{prefix}a_longer"] = (len_a > len_b).astype(int)

        features_added.extend([
            f"{prefix}len_diff", f"{prefix}len_ratio",
            f"{prefix}words_diff", f"{prefix}words_ratio",
            f"{prefix}len_abs_diff", f"{prefix}words_abs_diff",
            f"{prefix}a_longer",
        ])

    # Prompt-response relationships
    if prompt_column in df.columns:
        prompt_len = _safe_len(df[prompt_column])
        if response_a_column in df.columns:
            df[f"{prefix}resp_a_to_prompt_ratio"] = _safe_len(df[response_a_column]) / prompt_len.replace(0, 1)
            features_added.append(f"{prefix}resp_a_to_prompt_ratio")
        if response_b_column in df.columns:
            df[f"{prefix}resp_b_to_prompt_ratio"] = _safe_len(df[response_b_column]) / prompt_len.replace(0, 1)
            features_added.append(f"{prefix}resp_b_to_prompt_ratio")

    _save_data(df, outputs["data"])

    return f"extract_llm_comparison_features: added {len(features_added)} features"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific
    "extract_llm_comparison_features": extract_llm_comparison_features,
    # Reused from text_services
    "combine_text_columns": combine_text_columns,
    "extract_text_features": extract_text_features,
    "vectorize_tfidf": vectorize_tfidf,
    "transform_tfidf": transform_tfidf,
    "drop_text_columns": drop_text_columns,
    # Reused from preprocessing_services
    "convert_onehot_to_multiclass": convert_onehot_to_multiclass,
    "split_data": split_data,
    "drop_columns": drop_columns,
    # Reused from classification_services
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_multiclass_submission": predict_multiclass_submission,
}
