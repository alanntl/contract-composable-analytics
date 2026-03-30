"""
What's Cooking? - SLEGO Services
=================================
Competition: https://www.kaggle.com/competitions/whats-cooking
Problem Type: Multiclass Classification (20 cuisines)
Target: cuisine
ID Column: id

Classify recipes into cuisine types based on ingredient lists.
Data is JSON with ingredient arrays requiring text-based feature extraction.

Key insights from top solution notebooks:
- Ingredients lists need to be joined into text and vectorized (TF-IDF/DTM)
- Text cleaning: lowercase, replace hyphens, remove special chars
- XGBoost multiclass with TF-IDF features is the winning approach
- Feature: ingredient count per recipe adds signal

Competition-specific services:
- prepare_ingredients_data: Convert JSON ingredient lists to TF-IDF-ready text CSV
- map_predictions_to_labels: Map integer predictions back to cuisine strings
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable services from common modules
try:
    from services.text_services import vectorize_tfidf, transform_tfidf
    from services.classification_services import train_xgboost_classifier, train_linear_svc_classifier, train_logistic_classifier, predict_classifier
    from services.preprocessing_services import split_data, create_submission
except ImportError:
    from text_services import vectorize_tfidf, transform_tfidf
    from classification_services import train_xgboost_classifier, train_linear_svc_classifier, train_logistic_classifier, predict_classifier
    from preprocessing_services import split_data, create_submission


# =============================================================================
# SERVICE 1: PREPARE INGREDIENTS DATA (Competition-Specific)
# =============================================================================

@contract(
    inputs={
        "data": {"format": "json", "required": True, "schema": {"type": "json"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
        "label_map": {"format": "json", "required": False},
    },
    description="Convert JSON ingredient lists to text CSV with optional label encoding",
    tags=["preprocessing", "text", "nlp", "ingredients", "multiclass"],
    version="1.0.0",
)
def prepare_ingredients_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    ingredients_column: str = "ingredients",
    target_column: str = "cuisine",
    id_column: str = "id",
    is_train: bool = True,
) -> str:
    """
    Convert JSON data with ingredient lists to a TF-IDF-ready CSV.

    For each recipe:
    1. Joins ingredient list into a single text string
    2. Cleans text: lowercase, replace hyphens, remove special characters
    3. Adds ingredient count feature
    4. Label-encodes target column (train only) and saves mapping

    Parameters:
        ingredients_column: Column containing ingredient lists
        target_column: Column with cuisine labels (train only)
        id_column: ID column name
        is_train: If True, label-encode target and save mapping
    """
    with open(inputs["data"]) as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Clean and join ingredient lists to text strings
    # Based on top Kaggle solutions: lemmatization + cleaning improves accuracy
    try:
        from nltk.stem import WordNetLemmatizer
        import nltk
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        lemmatizer = WordNetLemmatizer()
        use_lemma = True
    except ImportError:
        use_lemma = False
        lemmatizer = None

    def clean_ingredients(ingredient_list):
        cleaned = []
        for ing in ingredient_list:
            # Lemmatize each word in the ingredient (top solution technique)
            if use_lemma and lemmatizer:
                ing = " ".join([lemmatizer.lemmatize(w) for w in ing.split()])
            # Clean: remove measurements, units, and special characters
            ing = re.sub(r'\(.*oz\.\)|\(.*\)|crushed|crumbles|ground|minced|powder|chopped|sliced', '', ing.lower())
            ing = re.sub(r'[^a-zA-Z ]', ' ', ing)
            ing = ing.strip()
            cleaned.append(ing)
        return " ".join(cleaned)

    df["ingredients_text"] = df[ingredients_column].apply(clean_ingredients)
    df["num_ingredients"] = df[ingredients_column].apply(len)

    # Drop original list column (cannot serialize lists to CSV cleanly)
    df = df.drop(columns=[ingredients_column])

    # Label-encode target for train data
    if is_train and target_column in df.columns:
        labels = sorted(df[target_column].unique().tolist())
        label_to_int = {label: i for i, label in enumerate(labels)}
        df[target_column] = df[target_column].map(label_to_int)

        # Save label mapping (int -> string) for later decoding
        if "label_map" in outputs and outputs["label_map"]:
            label_map = {str(i): label for label, i in label_to_int.items()}
            os.makedirs(os.path.dirname(outputs["label_map"]) or ".", exist_ok=True)
            with open(outputs["label_map"], "w") as f:
                json.dump(label_map, f, indent=2)

    _save_data(df, outputs["data"])

    n_classes = df[target_column].nunique() if is_train and target_column in df.columns else 0
    return f"prepare_ingredients_data: {len(df)} rows, {n_classes} classes, text column ready"


# =============================================================================
# SERVICE 2: MAP PREDICTIONS TO LABELS
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True},
        "label_map": {"format": "json", "required": True},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Map integer predictions back to original string labels for submission",
    tags=["postprocessing", "submission", "multiclass", "generic"],
    version="1.0.0",
)
def map_predictions_to_labels(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "cuisine",
) -> str:
    """
    Map integer predictions back to original string labels.

    Reads predictions CSV and a label mapping JSON, converts numeric
    predictions to their original string labels, and outputs a
    submission-ready CSV.

    Parameters:
        id_column: ID column name
        prediction_column: Column containing numeric predictions to map
    """
    pred_df = _load_data(inputs["predictions"])

    with open(inputs["label_map"]) as f:
        label_map = json.load(f)

    # Map integer predictions to original labels
    pred_df[prediction_column] = (
        pred_df[prediction_column]
        .astype(int)
        .astype(str)
        .map(label_map)
    )

    # Keep only id and prediction columns for submission
    submission = pred_df[[id_column, prediction_column]].copy()
    _save_data(submission, outputs["submission"])

    return f"map_predictions_to_labels: {len(submission)} predictions mapped to {submission[prediction_column].nunique()} classes"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific services
    "prepare_ingredients_data": prepare_ingredients_data,
    "map_predictions_to_labels": map_predictions_to_labels,
    # Imported reusable services
    "vectorize_tfidf": vectorize_tfidf,
    "transform_tfidf": transform_tfidf,
    "train_xgboost_classifier": train_xgboost_classifier,
    "train_linear_svc_classifier": train_linear_svc_classifier,
    "train_logistic_classifier": train_logistic_classifier,
    "predict_classifier": predict_classifier,
    "split_data": split_data,
    "create_submission": create_submission,
}
