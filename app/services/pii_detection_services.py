"""
PII Detection and Removal from Educational Data - SLEGO Services
================================================================
Competition: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data
Problem Type: Token-level NER (Named Entity Recognition)
Target: BIO labels per token (B-NAME_STUDENT, I-EMAIL, etc.)
Metric: F-beta score (beta=5, micro average)

This is a TOKEN-LEVEL NER task, NOT document-level classification.
Each token in each document needs a BIO label. The submission contains
only non-O tokens with columns: row_id, document, token, label.

Competition-specific services derived from top solution analysis:
- flatten_pii_json_to_tokens: Convert nested JSON to token-level CSV rows
- extract_token_ner_features: Regex-based and positional features per token (enhanced with wider context)
- undersample_majority_class: Handle extreme O class imbalance (99.95% -> balanced ratio)
- encode_ner_labels: Map BIO string labels to integers with saved encoder
- format_pii_submission: Convert predictions back to BIO labels, filter non-O

Top solution insights:
- Solution 1 (steffking01): LSTM with custom vocabulary, token padding
- Solution 2 (jdonnelly0804): SpanMarker (BERT NER) with BIO scheme
- Solution 3 (guillermoch): Probabilistic baseline using label distributions
- Key pattern: Token-level features + context are critical for NER
"""

import os
import sys
import json
import pickle
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data

# =============================================================================
# PII LABEL DEFINITIONS
# =============================================================================

PII_LABELS = [
    "O",
    "B-NAME_STUDENT", "I-NAME_STUDENT",
    "B-EMAIL", "I-EMAIL",
    "B-USERNAME", "I-USERNAME",
    "B-ID_NUM", "I-ID_NUM",
    "B-PHONE_NUM", "I-PHONE_NUM",
    "B-URL_PERSONAL", "I-URL_PERSONAL",
    "B-STREET_ADDRESS", "I-STREET_ADDRESS",
]


# =============================================================================
# SERVICE 1: FLATTEN JSON TO TOKEN ROWS
# =============================================================================

@contract(
    inputs={
        "data": {"format": "json", "required": True, "schema": {"type": "nested_json"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Flatten nested NER JSON to token-level CSV rows with one row per token",
    tags=["preprocessing", "ner", "token-level", "json", "generic"],
    version="1.0.0",
)
def flatten_pii_json_to_tokens(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    has_labels: bool = True,
) -> str:
    """
    Flatten nested JSON (document -> tokens/labels) to token-level CSV.

    Each output row = one token with columns:
    - document: document ID
    - token_idx: position of token in document
    - token_text: the actual token string
    - trailing_ws: whether token has trailing whitespace
    - label: BIO label (only if has_labels=True)

    Parameters:
        has_labels: Whether the JSON contains labels (True for train, False for test)
    """
    with open(inputs["data"], "r") as f:
        data = json.load(f)

    rows = []
    for doc in data:
        doc_id = doc["document"]
        tokens = doc["tokens"]
        trailing_ws = doc.get("trailing_whitespace", [True] * len(tokens))
        labels = doc.get("labels", []) if has_labels else []

        for idx, token_text in enumerate(tokens):
            row = {
                "document": doc_id,
                "token_idx": idx,
                "token_text": token_text,
                "trailing_ws": int(trailing_ws[idx]) if idx < len(trailing_ws) else 1,
            }
            if has_labels and idx < len(labels):
                row["label"] = labels[idx]
            rows.append(row)

    df = pd.DataFrame(rows)
    _save_data(df, outputs["data"])

    n_non_o = 0
    if has_labels and "label" in df.columns:
        n_non_o = (df["label"] != "O").sum()

    return f"flatten_pii_json_to_tokens: {len(data)} docs -> {len(df)} tokens ({n_non_o} non-O)"


# =============================================================================
# SERVICE 2: EXTRACT TOKEN-LEVEL NER FEATURES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Extract token-level features for NER: regex patterns, capitalization, position",
    tags=["feature-engineering", "ner", "token-level", "regex", "generic"],
    version="1.0.0",
)
def extract_token_ner_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    token_column: str = "token_text",
    doc_column: str = "document",
    token_idx_column: str = "token_idx",
    prefix: str = "feat_",
) -> str:
    """
    Extract handcrafted token-level features for NER classification.

    Features include:
    - Capitalization: is_title, is_upper, is_lower, is_mixed
    - Character patterns: has_digit, has_at, has_dot, has_hyphen
    - Token properties: token_length, is_single_char
    - Regex patterns: is_email, is_url, is_phone, is_number
    - Positional: position_in_doc, is_first_token, is_second_token
    - Context: prev_token features, next_token features

    Parameters:
        token_column: Column with token text
        doc_column: Column with document ID
        token_idx_column: Column with token position index
        prefix: Prefix for feature column names
    """
    df = _load_data(inputs["data"])
    tok = df[token_column].fillna("").astype(str)

    # --- Capitalization features ---
    df[f"{prefix}is_title"] = tok.str.istitle().astype(int)
    df[f"{prefix}is_upper"] = tok.str.isupper().astype(int)
    df[f"{prefix}is_lower"] = tok.str.islower().astype(int)
    df[f"{prefix}is_mixed"] = (~tok.str.istitle() & ~tok.str.isupper() & ~tok.str.islower() & (tok.str.len() > 1)).astype(int)
    df[f"{prefix}first_char_upper"] = tok.str[0].str.isupper().fillna(False).astype(int)

    # --- Character pattern features ---
    df[f"{prefix}has_digit"] = tok.str.contains(r"\d", regex=True, na=False).astype(int)
    df[f"{prefix}has_at"] = tok.str.contains("@", na=False).astype(int)
    df[f"{prefix}has_dot"] = tok.str.contains(r"\.", regex=True, na=False).astype(int)
    df[f"{prefix}has_hyphen"] = tok.str.contains("-", na=False).astype(int)
    df[f"{prefix}has_slash"] = tok.str.contains("/", na=False).astype(int)
    df[f"{prefix}has_underscore"] = tok.str.contains("_", na=False).astype(int)

    # --- Token property features ---
    df[f"{prefix}token_length"] = tok.str.len()
    df[f"{prefix}is_single_char"] = (tok.str.len() == 1).astype(int)
    df[f"{prefix}is_alpha"] = tok.str.isalpha().fillna(False).astype(int)
    df[f"{prefix}is_digit"] = tok.str.isdigit().fillna(False).astype(int)
    df[f"{prefix}is_alnum"] = tok.str.isalnum().fillna(False).astype(int)
    df[f"{prefix}n_digits"] = tok.str.count(r"\d")
    df[f"{prefix}digit_ratio"] = df[f"{prefix}n_digits"] / df[f"{prefix}token_length"].replace(0, 1)

    # --- Regex pattern features ---
    df[f"{prefix}is_email"] = tok.str.contains(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", regex=True, na=False
    ).astype(int)
    df[f"{prefix}is_url"] = tok.str.contains(
        r"^https?://|^www\.", regex=True, na=False
    ).astype(int)
    df[f"{prefix}is_phone"] = tok.str.contains(
        r"^\+?\d[\d\-\.\s\(\)]{6,}$", regex=True, na=False
    ).astype(int)
    df[f"{prefix}looks_like_name"] = (
        tok.str.istitle() & tok.str.isalpha() & (tok.str.len() >= 2)
    ).astype(int)
    df[f"{prefix}is_punctuation"] = tok.str.match(
        r"^[^\w\s]+$", na=False
    ).astype(int)

    # --- Positional features ---
    df[f"{prefix}is_first_token"] = (df[token_idx_column] == 0).astype(int)
    df[f"{prefix}is_second_token"] = (df[token_idx_column] == 1).astype(int)

    # Position ratio within document
    doc_lengths = df.groupby(doc_column)[token_idx_column].transform("max") + 1
    df[f"{prefix}position_ratio"] = df[token_idx_column] / doc_lengths

    # --- Trailing whitespace feature ---
    if "trailing_ws" in df.columns:
        df[f"{prefix}trailing_ws"] = df["trailing_ws"].astype(int)

    # --- Context window features (wider: prev-1/2/3, next-1/2/3) ---
    # Solutions show sequence context is critical for NER (LSTM uses 800-token windows)
    for shift, label in [(-1, "prev1"), (-2, "prev2"), (-3, "prev3"),
                         (1, "next1"), (2, "next2"), (3, "next3")]:
        shifted = tok.shift(shift).fillna("")
        # Reset at document boundaries - check all intermediate positions
        if abs(shift) == 1:
            same_doc = df[doc_column] == df[doc_column].shift(shift)
        elif abs(shift) == 2:
            same_doc = (df[doc_column] == df[doc_column].shift(shift)) & \
                       (df[doc_column] == df[doc_column].shift(shift // abs(shift)))
        else:  # shift == 3 or -3
            sign = shift // abs(shift)
            same_doc = (df[doc_column] == df[doc_column].shift(shift)) & \
                       (df[doc_column] == df[doc_column].shift(sign)) & \
                       (df[doc_column] == df[doc_column].shift(2 * sign))
        shifted = shifted.where(same_doc, "")

        df[f"{prefix}{label}_is_title"] = shifted.str.istitle().astype(int)
        df[f"{prefix}{label}_is_upper"] = shifted.str.isupper().astype(int)
        df[f"{prefix}{label}_has_digit"] = shifted.str.contains(r"\d", regex=True, na=False).astype(int)
        df[f"{prefix}{label}_length"] = shifted.str.len()
        df[f"{prefix}{label}_is_alpha"] = shifted.str.isalpha().fillna(False).astype(int)

    # --- Additional name-detection features (inspired by top solutions) ---
    # Consecutive title-case tokens often indicate names
    df[f"{prefix}prev1_next1_both_title"] = (
        (df[f"{prefix}prev1_is_title"] == 1) & (df[f"{prefix}next1_is_title"] == 1)
    ).astype(int)

    # Token appears to be in a name sequence
    df[f"{prefix}in_name_sequence"] = (
        (df[f"{prefix}is_title"] == 1) &
        ((df[f"{prefix}prev1_is_title"] == 1) | (df[f"{prefix}next1_is_title"] == 1))
    ).astype(int)

    features_added = [c for c in df.columns if c.startswith(prefix)]
    _save_data(df, outputs["data"])

    return f"extract_token_ner_features: added {len(features_added)} features"


# =============================================================================
# SERVICE 3: UNDERSAMPLE MAJORITY CLASS (O)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Undersample majority class (O) to handle extreme NER class imbalance",
    tags=["preprocessing", "ner", "sampling", "imbalance", "generic"],
    version="1.0.0",
)
def undersample_majority_class(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "label",
    majority_class: str = "O",
    ratio: float = 20.0,
    random_state: int = 42,
) -> str:
    """
    Undersample the majority class to address extreme class imbalance in NER.

    The PII competition has 99.95% O tokens vs 0.05% PII tokens. This causes
    models to predict almost everything as O. By undersampling O to a ratio
    (e.g., 20:1), we force the model to learn PII patterns.

    Parameters:
        label_column: Column with class labels
        majority_class: The class to undersample (default "O")
        ratio: Target ratio of majority:minority samples (e.g., 20.0 = keep 20x minority count of majority)
        random_state: Random seed for reproducibility
    """
    df = _load_data(inputs["data"])

    # Split into majority and minority
    majority_mask = df[label_column] == majority_class
    minority_df = df[~majority_mask]
    majority_df = df[majority_mask]

    n_minority = len(minority_df)
    n_majority_target = int(n_minority * ratio)

    # Sample majority class down to target
    if len(majority_df) > n_majority_target:
        majority_sampled = majority_df.sample(n=n_majority_target, random_state=random_state)
    else:
        majority_sampled = majority_df

    # Combine and shuffle
    balanced = pd.concat([minority_df, majority_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    _save_data(balanced, outputs["data"])

    return f"undersample_majority_class: {len(df)} -> {len(balanced)} rows (minority={n_minority}, majority={len(majority_sampled)}, ratio={len(majority_sampled)/max(n_minority,1):.1f}:1)"


# =============================================================================
# SERVICE 4: ENCODE NER LABELS
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
        "label_encoder": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Encode BIO NER string labels to integers and save encoder mapping",
    tags=["preprocessing", "ner", "encoding", "generic"],
    version="1.0.0",
)
def encode_ner_labels(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "label",
    encoded_column: str = "label_encoded",
    labels: Optional[List[str]] = None,
) -> str:
    """
    Map BIO string labels to integer codes for classifier training.

    Saves a label encoder artifact (dict mapping label->int and int->label)
    so predictions can be decoded back.

    Parameters:
        label_column: Column with string labels
        encoded_column: Name for output integer column
        labels: Fixed label list (if None, derived from data)
    """
    df = _load_data(inputs["data"])

    if labels is None:
        labels = sorted(df[label_column].unique().tolist())

    label_to_int = {label: i for i, label in enumerate(labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    df[encoded_column] = df[label_column].map(label_to_int)

    # Handle any unseen labels (map to O=0 if O is in mapping)
    if "O" in label_to_int:
        df[encoded_column] = df[encoded_column].fillna(label_to_int["O"]).astype(int)
    else:
        df[encoded_column] = df[encoded_column].fillna(0).astype(int)

    encoder_artifact = {
        "label_to_int": label_to_int,
        "int_to_label": int_to_label,
        "labels": labels,
    }

    _save_data(df, outputs["data"])

    os.makedirs(os.path.dirname(outputs["label_encoder"]) or ".", exist_ok=True)
    with open(outputs["label_encoder"], "wb") as f:
        pickle.dump(encoder_artifact, f)

    n_classes = len(labels)
    dist = df[encoded_column].value_counts().to_dict()
    return f"encode_ner_labels: {n_classes} classes, distribution (top5): {dict(list(dist.items())[:5])}"


# =============================================================================
# SERVICE 5: FORMAT PII SUBMISSION
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_features": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "label_encoder": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Format NER predictions into Kaggle PII submission: filter non-O, create row_id",
    tags=["submission", "ner", "pii", "generic"],
    version="1.0.0",
)
def format_pii_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    prediction_column: str = "target",
    doc_column: str = "document",
    token_idx_column: str = "token_idx",
) -> str:
    """
    Convert numeric NER predictions to Kaggle PII submission format.

    Reads predictions (integer class labels), maps back to BIO string labels,
    filters out O-class tokens, and formats as:
    row_id, document, token, label

    Uses test_features to recover document and token_idx columns that
    predict_classifier does not include in its output.

    Parameters:
        prediction_column: Column with integer predictions
        doc_column: Column with document IDs
        token_idx_column: Column with token indices
    """
    pred_df = _load_data(inputs["predictions"])
    test_df = _load_data(inputs["test_features"])

    with open(inputs["label_encoder"], "rb") as f:
        encoder = pickle.load(f)

    int_to_label = encoder["int_to_label"]
    # Convert keys to int if they're strings (JSON serialization can cause this)
    int_to_label = {int(k): v for k, v in int_to_label.items()}

    # Merge predictions with test features to recover document/token_idx
    # predict_classifier outputs rows in the same order as input
    merged = test_df[[doc_column, token_idx_column]].copy()
    merged[prediction_column] = pred_df[prediction_column].values

    # Map predictions back to string labels
    merged["label_str"] = merged[prediction_column].map(int_to_label)
    merged["label_str"] = merged["label_str"].fillna("O")

    # Filter non-O tokens only
    non_o = merged[merged["label_str"] != "O"].copy()

    # Create submission format
    submission = pd.DataFrame({
        "row_id": range(len(non_o)),
        "document": non_o[doc_column].values,
        "token": non_o[token_idx_column].values,
        "label": non_o["label_str"].values,
    })

    _save_data(submission, outputs["submission"])

    # Metrics
    label_counts = submission["label"].value_counts().to_dict() if len(submission) > 0 else {}
    metrics = {
        "total_predictions": len(merged),
        "non_o_predictions": len(submission),
        "pii_ratio": len(submission) / max(len(merged), 1),
        "label_distribution": label_counts,
        "unique_documents": int(submission["document"].nunique()) if len(submission) > 0 else 0,
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"format_pii_submission: {len(submission)} non-O predictions from {len(merged)} tokens"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "flatten_pii_json_to_tokens": flatten_pii_json_to_tokens,
    "extract_token_ner_features": extract_token_ner_features,
    "undersample_majority_class": undersample_majority_class,
    "encode_ner_labels": encode_ner_labels,
    "format_pii_submission": format_pii_submission,
}
