"""
StumbleUpon Evergreen Classification - SLEGO Services
=====================================================
Competition: https://www.kaggle.com/competitions/stumbleupon
Problem Type: Binary Classification
Target: label (0=ephemeral, 1=evergreen)

Enhanced preprocessing based on top-6 solution notebooks:
1. stumbleupon-eda-and-model-baseline (52 votes) - URL feature engineering
2. stumble-upon-challenge-auc-private-lb-0-85 - Boilerplate text features
3. stumbleupon-challenge-using-bert - DistilBERT + BiLSTM on text

Key insights from top solutions:
- URL feature engineering (website, domain, website_type) improves scores significantly
- Boilerplate text statistics (title_len, body_len, word_count) capture text signals
- alchemy_category should be one-hot encoded, not dropped
- '?' values must be handled properly
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

from services.io_utils import save_data as _save_data


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_url_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from URL (from stumbleupon-eda-and-model-baseline notebook).

    Features extracted:
    - website: The website name (e.g., 'bloomberg', 'popsci')
    - domain: The domain extension (e.g., 'com', 'org', 'net')
    - website_type: First path segment (e.g., 'news', 'health', 'article')
    - url_len: Length of URL string
    """
    df = df.copy()

    # Extract website name from URL
    def get_website(url):
        try:
            # Remove http:// or https://
            url = re.sub(r'^https?://', '', str(url))
            # Remove www.
            url = re.sub(r'^www\.', '', url)
            # Get the first part before the domain
            parts = url.split('.')
            return parts[0] if parts else 'unknown'
        except:
            return 'unknown'

    # Extract domain extension
    def get_domain(url):
        try:
            url = re.sub(r'^https?://', '', str(url))
            url = re.sub(r'^www\.', '', url)
            parts = url.split('/')
            domain_part = parts[0] if parts else ''
            domain_parts = domain_part.split('.')
            if len(domain_parts) >= 2:
                return domain_parts[-1]  # e.g., 'com', 'org'
            return 'unknown'
        except:
            return 'unknown'

    # Extract first path segment (website_type)
    def get_website_type(url):
        try:
            url = re.sub(r'^https?://', '', str(url))
            parts = url.split('/')
            if len(parts) >= 2 and parts[1]:
                # Check if it's a year
                if re.match(r'^20\d{2}$', parts[1]):
                    return 'YEAR'
                return parts[1][:20]  # Limit length
            return 'none'
        except:
            return 'unknown'

    df['website'] = df['url'].apply(get_website)
    df['domain'] = df['url'].apply(get_domain)
    df['website_type'] = df['url'].apply(get_website_type)
    df['url_len'] = df['url'].str.len()

    return df


def extract_boilerplate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract text statistics from boilerplate JSON field.

    The boilerplate field contains JSON with 'title', 'body', and 'url' keys.

    Features extracted:
    - title_len: Length of title
    - body_len: Length of body text
    - title_word_count: Number of words in title
    - body_word_count: Number of words in body
    - has_title: Binary indicator if title exists
    - has_body: Binary indicator if body exists
    """
    df = df.copy()

    def parse_boilerplate(bp):
        """Parse boilerplate JSON and extract fields."""
        try:
            if pd.isna(bp):
                return {'title': '', 'body': '', 'url': ''}
            # Parse as JSON if possible
            if isinstance(bp, str):
                # Try to extract title and body using regex
                title_match = re.search(r'"title"\s*:\s*"([^"]*)"', bp)
                body_match = re.search(r'"body"\s*:\s*"([^"]*)"', bp)

                title = title_match.group(1) if title_match else ''
                body = body_match.group(1) if body_match else ''

                return {'title': title, 'body': body}
            return {'title': '', 'body': ''}
        except:
            return {'title': '', 'body': ''}

    # Parse boilerplate
    bp_parsed = df['boilerplate'].apply(parse_boilerplate)

    # Extract title features
    df['title_len'] = bp_parsed.apply(lambda x: len(x.get('title', '')))
    df['title_word_count'] = bp_parsed.apply(lambda x: len(x.get('title', '').split()))
    df['has_title'] = (df['title_len'] > 0).astype(int)

    # Extract body features
    df['body_len'] = bp_parsed.apply(lambda x: len(x.get('body', '')))
    df['body_word_count'] = bp_parsed.apply(lambda x: len(x.get('body', '').split()))
    df['has_body'] = (df['body_len'] > 0).astype(int)

    # Ratio features
    df['title_body_ratio'] = df['title_len'] / (df['body_len'] + 1)

    return df


# =============================================================================
# PREPROCESSING SERVICE (ENHANCED V2)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "tsv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "tsv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Enhanced StumbleUpon preprocessing with URL features and text statistics from top notebooks",
    tags=["preprocessing", "classification", "stumbleupon", "binary", "feature-engineering"],
    version="2.0.0",
)
def preprocess_stumbleupon_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "label",
    id_column: str = "urlid",
    use_url_features: bool = True,
    use_text_features: bool = True,
) -> str:
    """
    Enhanced StumbleUpon preprocessing based on top-6 solution notebooks.

    Improvements over v1.0:
    - URL feature engineering: website, domain, website_type (from 52-vote notebook)
    - Boilerplate text statistics: title_len, body_len, word counts
    - Keep alchemy_category one-hot encoded
    - Handle '?' values properly

    Args:
        inputs: train_data (TSV), test_data (TSV, optional)
        outputs: train_data (CSV), test_data (CSV, optional)
        target_column: Name of the target column (default: "label")
        id_column: Name of the ID column (default: "urlid")
        use_url_features: Extract features from URL (default: True)
        use_text_features: Extract text statistics from boilerplate (default: True)
    """
    # Load train data (TSV format)
    train_df = pd.read_csv(inputs["train_data"], sep="\t")

    # Load test data if provided
    test_df = None
    if "test_data" in inputs and inputs["test_data"]:
        test_path = inputs["test_data"]
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path, sep="\t")

    # =========================================================================
    # NEW: Extract URL features (from stumbleupon-eda-and-model-baseline)
    # =========================================================================
    if use_url_features and "url" in train_df.columns:
        train_df = extract_url_features(train_df)
        if test_df is not None:
            test_df = extract_url_features(test_df)

    # =========================================================================
    # NEW: Extract text statistics from boilerplate
    # =========================================================================
    if use_text_features and "boilerplate" in train_df.columns:
        train_df = extract_boilerplate_features(train_df)
        if test_df is not None:
            test_df = extract_boilerplate_features(test_df)

    # Drop raw text columns (url and boilerplate) after feature extraction
    drop_columns = ["url", "boilerplate"]
    train_df = train_df.drop(columns=[c for c in drop_columns if c in train_df.columns])
    if test_df is not None:
        test_df = test_df.drop(columns=[c for c in drop_columns if c in test_df.columns])

    # Replace all "?" with NaN
    train_df = train_df.replace("?", np.nan)
    if test_df is not None:
        test_df = test_df.replace("?", np.nan)

    # Convert alchemy_category_score to float
    if "alchemy_category_score" in train_df.columns:
        train_df["alchemy_category_score"] = pd.to_numeric(
            train_df["alchemy_category_score"], errors="coerce"
        )
        if test_df is not None:
            test_df["alchemy_category_score"] = pd.to_numeric(
                test_df["alchemy_category_score"], errors="coerce"
            )

    # Convert is_news and news_front_page to numeric
    for col in ["is_news", "news_front_page"]:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            if test_df is not None and col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    # =========================================================================
    # One-hot encode categorical columns
    # =========================================================================
    cat_columns = ["alchemy_category"]

    # Add URL-derived categorical columns if they exist
    if use_url_features:
        for col in ["website", "domain", "website_type"]:
            if col in train_df.columns:
                cat_columns.append(col)

    # Prepare for consistent one-hot encoding
    target = train_df[target_column].copy() if target_column in train_df.columns else None
    train_no_target = train_df.drop(columns=[target_column], errors="ignore")

    if test_df is not None:
        train_no_target["_split"] = "train"
        test_df_temp = test_df.copy()
        test_df_temp["_split"] = "test"

        combined = pd.concat([train_no_target, test_df_temp], axis=0, ignore_index=True)

        # Fill NaN in categorical columns with "unknown"
        for col in cat_columns:
            if col in combined.columns:
                combined[col] = combined[col].fillna("unknown")
                # Limit cardinality for high-cardinality columns
                if col in ["website", "website_type"]:
                    # Keep top 50 values, rest become "other"
                    top_values = combined[col].value_counts().head(50).index.tolist()
                    combined[col] = combined[col].apply(lambda x: x if x in top_values else "other")

        # One-hot encode all categorical columns
        combined = pd.get_dummies(combined, columns=[c for c in cat_columns if c in combined.columns])

        train_df = combined[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
        test_df = combined[combined["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

        if target is not None:
            train_df[target_column] = target.values
    else:
        for col in cat_columns:
            if col in train_df.columns:
                train_df[col] = train_df[col].fillna("unknown")
        train_df = pd.get_dummies(train_df, columns=[c for c in cat_columns if c in train_df.columns])

    # Fill remaining NaN with 0
    train_df = train_df.fillna(0)
    if test_df is not None:
        test_df = test_df.fillna(0)

    # Ensure all feature columns are numeric
    for col in train_df.columns:
        if col not in [id_column, target_column]:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)
    if test_df is not None:
        for col in test_df.columns:
            if col != id_column:
                test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0)

    # Save outputs
    train_output = outputs["train_data"]
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    train_df.to_csv(train_output, index=False)

    test_shape_str = "N/A"
    if test_df is not None and "test_data" in outputs:
        test_output = outputs["test_data"]
        os.makedirs(os.path.dirname(test_output), exist_ok=True)
        test_df.to_csv(test_output, index=False)
        test_shape_str = str(test_df.shape)

    n_features = len([c for c in train_df.columns if c not in [id_column, target_column]])
    return f"preprocess_stumbleupon_data v2: train={train_df.shape}, test={test_shape_str}, features={n_features}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "preprocess_stumbleupon_data": preprocess_stumbleupon_data,
}
