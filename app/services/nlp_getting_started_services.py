"""
NLP Getting Started (Real or Not? Disaster Tweets) - SLEGO Services
====================================================================
Competition: https://www.kaggle.com/competitions/nlp-getting-started
Problem Type: Binary Classification
Target: target (1=real disaster, 0=not)
ID Column: id

Classify whether tweets are about real disasters.
Baseline: TF-IDF + Logistic Regression / LightGBM
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract

try:
    from services.classification_services import train_lightgbm_classifier, predict_classifier
    from services.preprocessing_services import split_data, create_submission
except ImportError:
    from classification_services import train_lightgbm_classifier, predict_classifier
    from preprocessing_services import split_data, create_submission


# =============================================================================
# TEXT PREPROCESSING SERVICES (Reusable for any NLP task)
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Clean text column by removing URLs, mentions, and special characters",
    tags=["preprocessing", "nlp", "text", "generic"],
    version="1.0.0"
)
def clean_text_column(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
    output_column: str = "text_clean",
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False,
    lowercase: bool = True,
) -> str:
    """
    Clean text by removing URLs, mentions, special characters.

    Args:
        text_column: Source text column
        output_column: Name for cleaned text column
        remove_urls: Remove http/https URLs
        remove_mentions: Remove @username mentions
        remove_hashtags: Remove #hashtag (keeps word, removes #)
        lowercase: Convert to lowercase
    """
    import re
    df = pd.read_csv(inputs["data"])

    text = df[text_column].fillna('').astype(str)

    if remove_urls:
        text = text.str.replace(r'http\S+|www\S+', '', regex=True)

    if remove_mentions:
        text = text.str.replace(r'@\w+', '', regex=True)

    if remove_hashtags:
        text = text.str.replace(r'#(\w+)', r'\1', regex=True)

    # Remove special characters but keep spaces
    text = text.str.replace(r'[^\w\s]', ' ', regex=True)

    # Remove extra whitespace
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()

    if lowercase:
        text = text.str.lower()

    df[output_column] = text

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"clean_text_column: cleaned {len(df)} texts"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={
        "data": {"format": "csv"},
        "vectorizer": {"format": "pickle"}
    },
    description="Convert text to TF-IDF features and save vectorizer",
    tags=["preprocessing", "nlp", "feature-engineering", "generic"],
    version="1.0.0"
)
def fit_tfidf_vectorizer(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text_clean",
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> str:
    """
    Fit TF-IDF vectorizer and transform text to features.

    Args:
        text_column: Column containing text to vectorize
        max_features: Maximum number of features
        ngram_range: Range of n-grams (1,2) means unigrams and bigrams
        min_df: Minimum document frequency
        max_df: Maximum document frequency (as proportion)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = pd.read_csv(inputs["data"])
    text = df[text_column].fillna('').astype(str)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )

    tfidf_matrix = vectorizer.fit_transform(text)
    feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Keep non-text columns
    non_text_cols = [c for c in df.columns if c != text_column]
    result = pd.concat([df[non_text_cols].reset_index(drop=True), tfidf_df], axis=1)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    result.to_csv(outputs["data"], index=False)

    with open(outputs["vectorizer"], "wb") as f:
        pickle.dump(vectorizer, f)

    return f"fit_tfidf_vectorizer: {tfidf_matrix.shape[1]} features from {len(df)} texts"


@contract(
    inputs={
        "data": {"format": "csv", "required": True},
        "vectorizer": {"format": "pickle", "required": True}
    },
    outputs={"data": {"format": "csv"}},
    description="Transform text using fitted TF-IDF vectorizer",
    tags=["preprocessing", "nlp", "feature-engineering", "generic"],
    version="1.0.0"
)
def transform_tfidf(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text_clean",
) -> str:
    """
    Transform text using a pre-fitted TF-IDF vectorizer.

    Args:
        text_column: Column containing text to transform
    """
    df = pd.read_csv(inputs["data"])

    with open(inputs["vectorizer"], "rb") as f:
        vectorizer = pickle.load(f)

    text = df[text_column].fillna('').astype(str)
    tfidf_matrix = vectorizer.transform(text)

    feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    non_text_cols = [c for c in df.columns if c != text_column]
    result = pd.concat([df[non_text_cols].reset_index(drop=True), tfidf_df], axis=1)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    result.to_csv(outputs["data"], index=False)

    return f"transform_tfidf: transformed {len(df)} texts"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Add text length and word count features",
    tags=["preprocessing", "nlp", "feature-engineering", "generic"],
    version="1.0.0"
)
def add_text_statistics(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
) -> str:
    """
    Add text length, word count, and other statistics.

    Args:
        text_column: Source text column
    """
    df = pd.read_csv(inputs["data"])
    text = df[text_column].fillna('').astype(str)

    df['text_len'] = text.str.len()
    df['word_count'] = text.str.split().str.len()
    df['avg_word_len'] = df['text_len'] / (df['word_count'] + 1)
    df['char_count'] = text.str.replace(' ', '').str.len()
    df['has_url'] = text.str.contains(r'http|www', regex=True).astype(int)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"add_text_statistics: added 5 text features"


# =============================================================================
# TEXT COMBINATION SERVICE (Reusable for any multi-text-field NLP task)
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Combine multiple text columns into a single text column",
    tags=["preprocessing", "nlp", "text", "generic"],
    version="1.0.0"
)
def combine_text_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_columns: List[str] = None,
    output_column: str = "combined_text",
    separator: str = " ",
) -> str:
    """
    Combine multiple text columns into a single text column.

    Useful for NLP tasks where multiple text fields (e.g., question + answer)
    need to be concatenated before vectorization.

    Args:
        text_columns: List of column names to combine
        output_column: Name for the combined text column
        separator: Separator between text fields
    """
    df = pd.read_csv(inputs["data"])

    if text_columns is None:
        raise ValueError("text_columns must be specified")

    existing = [c for c in text_columns if c in df.columns]
    if not existing:
        raise ValueError(f"None of {text_columns} found in data")

    df[output_column] = df[existing].fillna('').astype(str).agg(separator.join, axis=1)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"combine_text_columns: combined {len(existing)} columns into '{output_column}'"


# =============================================================================
# OPTIMIZED END-TO-END CLASSIFIER (Best performing: TF-IDF word+char + LogisticRegression)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
        "sample_submission": {"format": "csv", "required": True}
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"}
    },
    description="End-to-end disaster tweet classifier using TF-IDF + Logistic Regression",
    tags=["nlp", "classification", "tfidf", "logistic-regression", "generic"],
    version="2.0.0"
)
def train_disaster_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
    keyword_column: str = "keyword",
    target_column: str = "target",
    id_column: str = "id",
    max_word_features: int = 15000,
    max_char_features: int = 5000,
    C: float = 3.0,
) -> str:
    """
    Train and predict disaster tweets using optimized TF-IDF + LogisticRegression.

    This service achieved 0.80784 F1 score on Kaggle (top 38%).
    Uses word and character n-grams with sublinear TF scaling.

    Args:
        text_column: Column containing tweet text
        keyword_column: Column containing keyword
        target_column: Column with labels (0/1)
        id_column: ID column for submission
        max_word_features: Max TF-IDF word features
        max_char_features: Max TF-IDF char features
        C: Regularization parameter for LogisticRegression
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from scipy import sparse

    train = pd.read_csv(inputs["train_data"])
    test = pd.read_csv(inputs["test_data"])

    # Text cleaning
    def clean_text(text):
        if pd.isna(text):
            return ''
        text = str(text)
        text = re.sub(r'http\S+', 'httpurl', text)
        text = re.sub(r'@\w+', 'usermention', text)
        return text

    train['text_clean'] = train[text_column].apply(clean_text)
    test['text_clean'] = test[text_column].apply(clean_text)

    # Combine keyword + text
    train['keyword_clean'] = train[keyword_column].fillna('').str.replace('%20', ' ')
    test['keyword_clean'] = test[keyword_column].fillna('').str.replace('%20', ' ')
    train['full_text'] = train['keyword_clean'] + ' ' + train['text_clean']
    test['full_text'] = test['keyword_clean'] + ' ' + test['text_clean']

    # Word TF-IDF
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 3), max_features=max_word_features, min_df=2, sublinear_tf=True
    )
    X_train_word = tfidf_word.fit_transform(train['full_text'])
    X_test_word = tfidf_word.transform(test['full_text'])

    # Char TF-IDF
    tfidf_char = TfidfVectorizer(
        analyzer='char', ngram_range=(3, 5), max_features=max_char_features, sublinear_tf=True
    )
    X_train_char = tfidf_char.fit_transform(train['full_text'])
    X_test_char = tfidf_char.transform(test['full_text'])

    # Combine features
    X_train = sparse.hstack([X_train_word, X_train_char])
    X_test = sparse.hstack([X_test_word, X_test_char])
    y = train[target_column].values

    # Validation split for metrics
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify=y)

    # Train and validate
    model = LogisticRegression(C=C, max_iter=3000, solver='liblinear')
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)

    # Train on full data
    final_model = LogisticRegression(C=C, max_iter=3000, solver='liblinear')
    final_model.fit(X_train, y)

    # Predict
    test_pred = final_model.predict(X_test)

    # Create submission
    submission = pd.DataFrame({id_column: test[id_column], target_column: test_pred})
    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    # Save metrics
    metrics = {
        "method": "tfidf_word_char_logistic_regression",
        "validation_f1": round(val_f1, 5),
        "word_features": X_train_word.shape[1],
        "char_features": X_train_char.shape[1],
        "total_features": X_train.shape[1],
        "C": C
    }
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_disaster_classifier: val_f1={val_f1:.5f}, {X_train.shape[1]} features"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "clean_text_column": clean_text_column,
    "fit_tfidf_vectorizer": fit_tfidf_vectorizer,
    "transform_tfidf": transform_tfidf,
    "add_text_statistics": add_text_statistics,
    "combine_text_columns": combine_text_columns,
    "train_disaster_classifier": train_disaster_classifier,
    "split_data": split_data,
    "create_submission": create_submission,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
}


PIPELINE_SPEC = [
    {
        "service": "add_text_statistics",
        "inputs": {"data": "nlp-getting-started/datasets/train.csv"},
        "outputs": {"data": "nlp-getting-started/artifacts/train_01_stats.csv"},
        "params": {"text_column": "text"},
        "module": "nlp_getting_started_services"
    },
    {
        "service": "clean_text_column",
        "inputs": {"data": "nlp-getting-started/artifacts/train_01_stats.csv"},
        "outputs": {"data": "nlp-getting-started/artifacts/train_02_clean.csv"},
        "params": {"text_column": "text"},
        "module": "nlp_getting_started_services"
    },
    {
        "service": "fit_tfidf_vectorizer",
        "inputs": {"data": "nlp-getting-started/artifacts/train_02_clean.csv"},
        "outputs": {
            "data": "nlp-getting-started/artifacts/train_03_tfidf.csv",
            "vectorizer": "nlp-getting-started/artifacts/tfidf_vectorizer.pkl"
        },
        "params": {"text_column": "text_clean", "max_features": 3000},
        "module": "nlp_getting_started_services"
    },
    {
        "service": "split_data",
        "inputs": {"data": "nlp-getting-started/artifacts/train_03_tfidf.csv"},
        "outputs": {
            "train_data": "nlp-getting-started/artifacts/train_split.csv",
            "valid_data": "nlp-getting-started/artifacts/valid_split.csv"
        },
        "params": {"stratify_column": "target", "test_size": 0.2, "random_state": 42},
        "module": "nlp_getting_started_services"
    },
    {
        "service": "train_lightgbm_classifier",
        "inputs": {
            "train_data": "nlp-getting-started/artifacts/train_split.csv",
            "valid_data": "nlp-getting-started/artifacts/valid_split.csv"
        },
        "outputs": {
            "model": "nlp-getting-started/artifacts/model.pkl",
            "metrics": "nlp-getting-started/artifacts/metrics.json"
        },
        "params": {
            "label_column": "target",
            "id_column": "id",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31
        },
        "module": "nlp_getting_started_services"
    }
]


def run_pipeline(base_path: str, verbose: bool = True):
    for i, step in enumerate(PIPELINE_SPEC, 1):
        service_name = step["service"]
        service_fn = SERVICE_REGISTRY.get(service_name)
        if not service_fn:
            print(f"Error: Service {service_name} not found")
            continue

        res_in = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
        res_out = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

        if verbose:
            print(f"[{i}/{len(PIPELINE_SPEC)}] {service_name}...", end=" ")

        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose: print(f"OK - {result}")
        except Exception as e:
            if verbose: print(f"FAILED - {e}")
            import traceback
            traceback.print_exc()
            break
