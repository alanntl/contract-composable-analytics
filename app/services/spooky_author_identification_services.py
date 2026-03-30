"""
Spooky Author Identification - SLEGO Services
==============================================
Competition: https://www.kaggle.com/competitions/spooky-author-identification
Problem Type: Multiclass Classification (3 classes: EAP, HPL, MWS)
Target: author
Evaluation: Multi-class Log Loss

Based on top Kaggle solution notebooks:
- SRK's Simple Feature Engineering Notebook (0.32 LB score)
- Key insight: Naive Bayes on CountVectorizer + stacking beats TF-IDF alone

Competition-specific services:
- extract_author_meta_features: Extract text statistics (word count, char count, etc.)
- train_nb_stack_features: Create stacked Naive Bayes OOF predictions
- train_author_classifier: Train final XGBoost/CatBoost on stacked features
- predict_author_probabilities: Generate probability predictions for submission
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

# =============================================================================
# HELPERS
# =============================================================================
from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# SERVICE 1: EXTRACT AUTHOR META FEATURES
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Extract meta features from text for author identification",
    tags=["feature-engineering", "text", "nlp", "spooky-author"],
    version="1.0.0",
)
def extract_author_meta_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
    prefix: str = "",
) -> str:
    """
    Extract handcrafted meta features from text for author style identification.

    Features:
    - num_words: Word count
    - num_unique_words: Unique word count
    - num_chars: Character count
    - num_stopwords: Stopword count
    - num_punctuations: Punctuation count
    - num_words_upper: Uppercase word count
    - num_words_title: Title case word count
    - mean_word_len: Average word length
    """
    import string

    # Stopwords (minimal set to avoid NLTK dependency)
    eng_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                     'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                     'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                     'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                     'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                     'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                     'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                     'into', 'through', 'during', 'before', 'after', 'above', 'below',
                     'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once'}

    df = _load_data(inputs["data"])
    text = df[text_column].fillna('')

    df[f'{prefix}num_words'] = text.apply(lambda x: len(str(x).split()))
    df[f'{prefix}num_unique_words'] = text.apply(lambda x: len(set(str(x).split())))
    df[f'{prefix}num_chars'] = text.apply(lambda x: len(str(x)))
    df[f'{prefix}num_stopwords'] = text.apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    df[f'{prefix}num_punctuations'] = text.apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df[f'{prefix}num_words_upper'] = text.apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df[f'{prefix}num_words_title'] = text.apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df[f'{prefix}mean_word_len'] = text.apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)

    _save_data(df, outputs["data"])

    return f"extract_author_meta_features: added 8 meta features to {len(df)} rows"


# =============================================================================
# SERVICE 2: TRAIN NAIVE BAYES STACKING
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_features": {"format": "csv", "schema": {"type": "tabular"}},
        "test_features": {"format": "csv", "schema": {"type": "tabular"}},
        "vectorizers": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Create stacked Naive Bayes OOF predictions for author classification",
    tags=["feature-engineering", "stacking", "naive-bayes", "text", "spooky-author"],
    version="1.0.0",
)
def train_nb_stack_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
    target_column: str = "author",
    n_folds: int = 5,
    random_state: int = 42,
    max_features: int = 50000,
) -> str:
    """
    Create stacked Naive Bayes features using out-of-fold predictions.

    Based on SRK's top solution - key insight: NB on CountVectorizer significantly
    outperforms TF-IDF for this competition (0.45 vs 0.84 logloss).

    Creates features from:
    1. Word CountVectorizer + NB (best single model: ~0.45 logloss)
    2. Character CountVectorizer + NB
    3. Character TF-IDF + NB
    4. Word TF-IDF SVD (20 components)
    5. Char TF-IDF SVD (20 components)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import log_loss

    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    # Encode target
    le = LabelEncoder()
    train_y = le.fit_transform(train_df[target_column])
    n_classes = len(le.classes_)

    all_text = train_df[text_column].tolist() + test_df[text_column].tolist()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    vectorizers = {}

    # Initialize output DataFrames with original data
    train_out = train_df.copy()
    test_out = test_df.copy()

    cv_scores = []

    def get_nb_predictions(train_X, test_X, train_y, kf, name):
        """Get out-of-fold Naive Bayes predictions."""
        train_pred = np.zeros((len(train_y), n_classes))
        test_pred = np.zeros((test_X.shape[0], n_classes))
        fold_scores = []

        for fold_idx, (dev_idx, val_idx) in enumerate(kf.split(train_X)):
            dev_X = train_X[dev_idx]
            val_X = train_X[val_idx]
            dev_y = train_y[dev_idx]
            val_y = train_y[val_idx]

            model = MultinomialNB()
            model.fit(dev_X, dev_y)

            train_pred[val_idx] = model.predict_proba(val_X)
            test_pred += model.predict_proba(test_X) / n_folds

            fold_scores.append(log_loss(val_y, model.predict_proba(val_X)))

        print(f"  {name} CV logloss: {np.mean(fold_scores):.4f}")
        return train_pred, test_pred, np.mean(fold_scores)

    # 1. Word CountVectorizer + NB (best single model)
    print("Training Word CountVectorizer + NB...")
    count_word = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=max_features)
    count_word.fit(all_text)
    train_count_word = count_word.transform(train_df[text_column])
    test_count_word = count_word.transform(test_df[text_column])
    vectorizers['count_word'] = count_word

    train_nb_word, test_nb_word, score1 = get_nb_predictions(
        train_count_word, test_count_word, train_y, kf, "Word CountVec NB"
    )
    cv_scores.append(score1)

    for i, cls in enumerate(le.classes_):
        train_out[f'nb_cvec_{cls.lower()}'] = train_nb_word[:, i]
        test_out[f'nb_cvec_{cls.lower()}'] = test_nb_word[:, i]

    # 2. Character CountVectorizer + NB
    print("Training Char CountVectorizer + NB...")
    count_char = CountVectorizer(ngram_range=(1, 7), analyzer='char', max_features=max_features)
    count_char.fit(all_text)
    train_count_char = count_char.transform(train_df[text_column])
    test_count_char = count_char.transform(test_df[text_column])
    vectorizers['count_char'] = count_char

    train_nb_char, test_nb_char, score2 = get_nb_predictions(
        train_count_char, test_count_char, train_y, kf, "Char CountVec NB"
    )
    cv_scores.append(score2)

    for i, cls in enumerate(le.classes_):
        train_out[f'nb_cvec_char_{cls.lower()}'] = train_nb_char[:, i]
        test_out[f'nb_cvec_char_{cls.lower()}'] = test_nb_char[:, i]

    # 3. Character TF-IDF + NB
    print("Training Char TF-IDF + NB...")
    tfidf_char = TfidfVectorizer(ngram_range=(1, 5), analyzer='char', max_features=max_features)
    tfidf_char.fit(all_text)
    train_tfidf_char = tfidf_char.transform(train_df[text_column])
    test_tfidf_char = tfidf_char.transform(test_df[text_column])
    vectorizers['tfidf_char'] = tfidf_char

    train_nb_tfidf, test_nb_tfidf, score3 = get_nb_predictions(
        train_tfidf_char, test_tfidf_char, train_y, kf, "Char TF-IDF NB"
    )
    cv_scores.append(score3)

    for i, cls in enumerate(le.classes_):
        train_out[f'nb_tfidf_char_{cls.lower()}'] = train_nb_tfidf[:, i]
        test_out[f'nb_tfidf_char_{cls.lower()}'] = test_nb_tfidf[:, i]

    # 4. Word TF-IDF + SVD
    print("Creating Word TF-IDF + SVD features...")
    tfidf_word = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=max_features)
    tfidf_word.fit(all_text)
    train_tfidf_word = tfidf_word.transform(train_df[text_column])
    test_tfidf_word = tfidf_word.transform(test_df[text_column])
    vectorizers['tfidf_word'] = tfidf_word

    n_svd = 20
    svd_word = TruncatedSVD(n_components=n_svd, algorithm='arpack', random_state=random_state)
    svd_word.fit(tfidf_word.transform(all_text))
    vectorizers['svd_word'] = svd_word

    train_svd_word = svd_word.transform(train_tfidf_word)
    test_svd_word = svd_word.transform(test_tfidf_word)

    for i in range(n_svd):
        train_out[f'svd_word_{i}'] = train_svd_word[:, i]
        test_out[f'svd_word_{i}'] = test_svd_word[:, i]

    # 5. Char TF-IDF + SVD
    print("Creating Char TF-IDF + SVD features...")
    svd_char = TruncatedSVD(n_components=n_svd, algorithm='arpack', random_state=random_state)
    svd_char.fit(tfidf_char.transform(all_text))
    vectorizers['svd_char'] = svd_char

    train_svd_char = svd_char.transform(train_tfidf_char)
    test_svd_char = svd_char.transform(test_tfidf_char)

    for i in range(n_svd):
        train_out[f'svd_char_{i}'] = train_svd_char[:, i]
        test_out[f'svd_char_{i}'] = test_svd_char[:, i]

    # Store label encoder for later use
    vectorizers['label_encoder'] = le

    _save_data(train_out, outputs["train_features"])
    _save_data(test_out, outputs["test_features"])

    os.makedirs(os.path.dirname(outputs["vectorizers"]) or ".", exist_ok=True)
    with open(outputs["vectorizers"], "wb") as f:
        pickle.dump(vectorizers, f)

    return f"train_nb_stack_features: created {len(train_out.columns) - len(train_df.columns)} stacked features, best NB CV: {min(cv_scores):.4f}"


# =============================================================================
# SERVICE 3: TRAIN AUTHOR CLASSIFIER (XGBoost/CatBoost)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "vectorizers": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Train XGBoost classifier on stacked features for author identification",
    tags=["modeling", "training", "xgboost", "classification", "multiclass", "spooky-author"],
    version="1.0.0",
)
def train_author_classifier(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "author",
    id_column: str = "id",
    text_column: str = "text",
    n_folds: int = 5,
    n_estimators: int = 2000,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    colsample_bytree: float = 0.7,
    subsample: float = 0.8,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    use_catboost: bool = False,
) -> str:
    """
    Train XGBoost or CatBoost classifier on stacked Naive Bayes features.

    Uses K-fold cross-validation with early stopping for robust training.
    """
    import xgboost as xgb
    from sklearn.model_selection import KFold
    from sklearn.metrics import log_loss

    train_df = _load_data(inputs["train_data"])

    with open(inputs["vectorizers"], "rb") as f:
        vectorizers = pickle.load(f)

    le = vectorizers['label_encoder']
    train_y = le.transform(train_df[target_column])

    # Drop non-feature columns
    drop_cols = [target_column, id_column, text_column]
    drop_cols = [c for c in drop_cols if c in train_df.columns]
    train_X = train_df.drop(columns=drop_cols)

    feature_cols = list(train_X.columns)
    n_classes = len(le.classes_)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    if use_catboost:
        from catboost import CatBoostClassifier

        train_pred = np.zeros((len(train_y), n_classes))
        cv_scores = []
        models = []

        for fold_idx, (dev_idx, val_idx) in enumerate(kf.split(train_X)):
            dev_X = train_X.iloc[dev_idx]
            val_X = train_X.iloc[val_idx]
            dev_y = train_y[dev_idx]
            val_y = train_y[val_idx]

            model = CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=max_depth,
                loss_function='MultiClass',
                random_seed=random_state,
                verbose=False,
                early_stopping_rounds=early_stopping_rounds,
            )

            model.fit(dev_X, dev_y, eval_set=(val_X, val_y), verbose=False)

            train_pred[val_idx] = model.predict_proba(val_X)
            fold_score = log_loss(val_y, train_pred[val_idx])
            cv_scores.append(fold_score)
            models.append(model)
            print(f"  Fold {fold_idx + 1}: logloss = {fold_score:.4f}")

        final_model = {"models": models, "feature_cols": feature_cols, "label_encoder": le, "model_type": "catboost"}

    else:
        # XGBoost
        param = {
            'objective': 'multi:softprob',
            'eta': learning_rate,
            'max_depth': max_depth,
            'num_class': n_classes,
            'eval_metric': 'mlogloss',
            'min_child_weight': 1,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'seed': random_state,
        }

        train_pred = np.zeros((len(train_y), n_classes))
        cv_scores = []
        models = []

        for fold_idx, (dev_idx, val_idx) in enumerate(kf.split(train_X)):
            dev_X = train_X.iloc[dev_idx]
            val_X = train_X.iloc[val_idx]
            dev_y = train_y[dev_idx]
            val_y = train_y[val_idx]

            xgtrain = xgb.DMatrix(dev_X, label=dev_y)
            xgval = xgb.DMatrix(val_X, label=val_y)

            watchlist = [(xgtrain, 'train'), (xgval, 'val')]
            model = xgb.train(param, xgtrain, n_estimators, watchlist,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

            train_pred[val_idx] = model.predict(xgval, iteration_range=(0, model.best_iteration + 1))
            fold_score = log_loss(val_y, train_pred[val_idx])
            cv_scores.append(fold_score)
            models.append(model)
            print(f"  Fold {fold_idx + 1}: logloss = {fold_score:.4f}")

        final_model = {"models": models, "feature_cols": feature_cols, "label_encoder": le, "model_type": "xgboost"}

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"Mean CV logloss: {mean_cv:.4f} (+/- {std_cv:.4f})")

    # Save model
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(final_model, f)

    # Save metrics
    metrics = {
        "model_type": "catboost" if use_catboost else "xgboost",
        "cv_logloss": float(mean_cv),
        "cv_std": float(std_cv),
        "n_folds": n_folds,
        "n_features": len(feature_cols),
        "n_train_samples": len(train_X),
        "classes": list(le.classes_),
        "fold_scores": [float(s) for s in cv_scores],
    }

    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_author_classifier: CV logloss = {mean_cv:.4f} (+/- {std_cv:.4f})"


# =============================================================================
# SERVICE 4: PREDICT AUTHOR PROBABILITIES
# =============================================================================

@contract(
    inputs={
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Generate probability predictions for author submission",
    tags=["prediction", "submission", "multiclass", "spooky-author"],
    version="1.0.0",
)
def predict_author_probabilities(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    text_column: str = "text",
    target_column: str = "author",
) -> str:
    """
    Generate probability predictions for Kaggle submission.

    Output format: id, EAP, HPL, MWS (probability for each author)
    """
    import xgboost as xgb

    test_df = _load_data(inputs["test_data"])

    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    models = model_data["models"]
    feature_cols = model_data["feature_cols"]
    le = model_data["label_encoder"]
    model_type = model_data.get("model_type", "xgboost")

    test_ids = test_df[id_column].values

    # Prepare features
    drop_cols = [id_column, text_column, target_column]
    drop_cols = [c for c in drop_cols if c in test_df.columns]
    test_X = test_df.drop(columns=drop_cols)

    # Ensure feature columns match
    test_X = test_X[feature_cols]

    # Average predictions from all fold models
    n_models = len(models)
    n_classes = len(le.classes_)
    test_pred = np.zeros((len(test_X), n_classes))

    for model in models:
        if model_type == "catboost":
            test_pred += model.predict_proba(test_X) / n_models
        else:
            xgtest = xgb.DMatrix(test_X)
            test_pred += model.predict(xgtest, iteration_range=(0, model.best_iteration + 1)) / n_models

    # Create submission DataFrame
    submission = pd.DataFrame({'id': test_ids})
    for i, cls in enumerate(le.classes_):
        submission[cls] = test_pred[:, i]

    _save_data(submission, outputs["submission"])

    return f"predict_author_probabilities: generated predictions for {len(test_ids)} samples"


# =============================================================================
# SERVICE 5: RUN COMPLETE PIPELINE (All-in-one for convenience)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
    },
    description="Run complete spooky author identification pipeline end-to-end",
    tags=["pipeline", "end-to-end", "classification", "multiclass", "spooky-author"],
    version="1.0.0",
)
def run_spooky_author_pipeline(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    text_column: str = "text",
    target_column: str = "author",
    id_column: str = "id",
    n_folds: int = 5,
    n_estimators: int = 2000,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    max_features: int = 50000,
    random_state: int = 42,
) -> str:
    """
    Run the complete spooky author identification pipeline end-to-end.

    This service combines all steps:
    1. Extract meta features
    2. Create stacked Naive Bayes features
    3. Train XGBoost classifier
    4. Generate predictions

    Based on SRK's top Kaggle solution achieving ~0.32 public score.
    """
    import xgboost as xgb
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import log_loss
    import string
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("Spooky Author Identification Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # Encode target
    le = LabelEncoder()
    train_y = le.fit_transform(train_df[target_column])
    test_ids = test_df[id_column].values
    n_classes = len(le.classes_)

    all_text = train_df[text_column].tolist() + test_df[text_column].tolist()

    # Stopwords
    eng_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                     'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                     'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                     'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                     'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                     'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                     'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                     'into', 'through', 'during', 'before', 'after', 'above', 'below',
                     'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once'}

    # Extract meta features
    print("\n[2/5] Extracting meta features...")

    def extract_meta(df, text_col):
        text = df[text_col].fillna('')
        features = pd.DataFrame()
        features['num_words'] = text.apply(lambda x: len(str(x).split()))
        features['num_unique_words'] = text.apply(lambda x: len(set(str(x).split())))
        features['num_chars'] = text.apply(lambda x: len(str(x)))
        features['num_stopwords'] = text.apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
        features['num_punctuations'] = text.apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
        features['num_words_upper'] = text.apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        features['num_words_title'] = text.apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
        features['mean_word_len'] = text.apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
        return features

    train_meta = extract_meta(train_df, text_column)
    test_meta = extract_meta(test_df, text_column)
    print(f"  Meta features: {train_meta.shape[1]} columns")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2017)

    def get_nb_predictions(train_X, test_X, train_y, kf, name):
        train_pred = np.zeros((len(train_y), n_classes))
        test_pred = np.zeros((test_X.shape[0], n_classes))
        cv_scores = []

        for fold_idx, (dev_idx, val_idx) in enumerate(kf.split(train_X)):
            dev_X = train_X[dev_idx]
            val_X = train_X[val_idx]
            dev_y = train_y[dev_idx]
            val_y = train_y[val_idx]

            model = MultinomialNB()
            model.fit(dev_X, dev_y)

            train_pred[val_idx] = model.predict_proba(val_X)
            test_pred += model.predict_proba(test_X) / n_folds
            cv_scores.append(log_loss(val_y, model.predict_proba(val_X)))

        print(f"    {name} CV logloss: {np.mean(cv_scores):.4f}")
        return train_pred, test_pred

    # Create NB stacked features
    print("\n[3/5] Creating Naive Bayes stacked features...")

    # Word CountVectorizer
    count_word = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=max_features)
    count_word.fit(all_text)
    train_count_word = count_word.transform(train_df[text_column])
    test_count_word = count_word.transform(test_df[text_column])
    train_nb_cvec, test_nb_cvec = get_nb_predictions(train_count_word, test_count_word, train_y, kf, "Word CountVec NB")
    train_nb_cvec = pd.DataFrame(train_nb_cvec, columns=['nb_cvec_eap', 'nb_cvec_hpl', 'nb_cvec_mws'])
    test_nb_cvec = pd.DataFrame(test_nb_cvec, columns=['nb_cvec_eap', 'nb_cvec_hpl', 'nb_cvec_mws'])

    # Char CountVectorizer
    count_char = CountVectorizer(ngram_range=(1, 7), analyzer='char', max_features=max_features)
    count_char.fit(all_text)
    train_count_char = count_char.transform(train_df[text_column])
    test_count_char = count_char.transform(test_df[text_column])
    train_nb_cvec_char, test_nb_cvec_char = get_nb_predictions(train_count_char, test_count_char, train_y, kf, "Char CountVec NB")
    train_nb_cvec_char = pd.DataFrame(train_nb_cvec_char, columns=['nb_cvec_char_eap', 'nb_cvec_char_hpl', 'nb_cvec_char_mws'])
    test_nb_cvec_char = pd.DataFrame(test_nb_cvec_char, columns=['nb_cvec_char_eap', 'nb_cvec_char_hpl', 'nb_cvec_char_mws'])

    # Char TF-IDF
    tfidf_char = TfidfVectorizer(ngram_range=(1, 5), analyzer='char', max_features=max_features)
    tfidf_char.fit(all_text)
    train_tfidf_char = tfidf_char.transform(train_df[text_column])
    test_tfidf_char = tfidf_char.transform(test_df[text_column])
    train_nb_tfidf_char, test_nb_tfidf_char = get_nb_predictions(train_tfidf_char, test_tfidf_char, train_y, kf, "Char TF-IDF NB")
    train_nb_tfidf_char = pd.DataFrame(train_nb_tfidf_char, columns=['nb_tfidf_char_eap', 'nb_tfidf_char_hpl', 'nb_tfidf_char_mws'])
    test_nb_tfidf_char = pd.DataFrame(test_nb_tfidf_char, columns=['nb_tfidf_char_eap', 'nb_tfidf_char_hpl', 'nb_tfidf_char_mws'])

    # Word TF-IDF + SVD
    print("\n[4/5] Creating TF-IDF + SVD features...")
    tfidf_word = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=max_features)
    tfidf_word.fit(all_text)
    train_tfidf_word = tfidf_word.transform(train_df[text_column])
    test_tfidf_word = tfidf_word.transform(test_df[text_column])

    n_svd = 20
    svd_word = TruncatedSVD(n_components=n_svd, algorithm='arpack', random_state=random_state)
    svd_word.fit(tfidf_word.transform(all_text))
    train_svd_word = pd.DataFrame(svd_word.transform(train_tfidf_word), columns=[f'svd_word_{i}' for i in range(n_svd)])
    test_svd_word = pd.DataFrame(svd_word.transform(test_tfidf_word), columns=[f'svd_word_{i}' for i in range(n_svd)])

    # Char TF-IDF + SVD
    svd_char = TruncatedSVD(n_components=n_svd, algorithm='arpack', random_state=random_state)
    svd_char.fit(tfidf_char.transform(all_text))
    train_svd_char = pd.DataFrame(svd_char.transform(train_tfidf_char), columns=[f'svd_char_{i}' for i in range(n_svd)])
    test_svd_char = pd.DataFrame(svd_char.transform(test_tfidf_char), columns=[f'svd_char_{i}' for i in range(n_svd)])

    print(f"  Word SVD: {n_svd} components, Char SVD: {n_svd} components")

    # Combine all features
    train_X = pd.concat([
        train_meta.reset_index(drop=True),
        train_svd_word.reset_index(drop=True),
        train_svd_char.reset_index(drop=True),
        train_nb_cvec.reset_index(drop=True),
        train_nb_cvec_char.reset_index(drop=True),
        train_nb_tfidf_char.reset_index(drop=True),
    ], axis=1)

    test_X = pd.concat([
        test_meta.reset_index(drop=True),
        test_svd_word.reset_index(drop=True),
        test_svd_char.reset_index(drop=True),
        test_nb_cvec.reset_index(drop=True),
        test_nb_cvec_char.reset_index(drop=True),
        test_nb_tfidf_char.reset_index(drop=True),
    ], axis=1)

    print(f"  Combined features: {train_X.shape[1]} columns")

    # Train XGBoost
    print("\n[5/5] Training XGBoost classifier...")
    param = {
        'objective': 'multi:softprob',
        'eta': learning_rate,
        'max_depth': max_depth,
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'seed': random_state,
    }

    train_pred = np.zeros((len(train_y), n_classes))
    test_pred = np.zeros((len(test_X), n_classes))
    cv_scores = []
    models = []

    for fold_idx, (dev_idx, val_idx) in enumerate(kf.split(train_X)):
        dev_X_fold = train_X.iloc[dev_idx]
        val_X_fold = train_X.iloc[val_idx]
        dev_y = train_y[dev_idx]
        val_y = train_y[val_idx]

        xgtrain = xgb.DMatrix(dev_X_fold, label=dev_y)
        xgval = xgb.DMatrix(val_X_fold, label=val_y)
        xgtest = xgb.DMatrix(test_X)

        watchlist = [(xgtrain, 'train'), (xgval, 'val')]
        model = xgb.train(param, xgtrain, n_estimators, watchlist,
                          early_stopping_rounds=50, verbose_eval=False)

        train_pred[val_idx] = model.predict(xgval, iteration_range=(0, model.best_iteration + 1))
        test_pred += model.predict(xgtest, iteration_range=(0, model.best_iteration + 1)) / n_folds

        fold_score = log_loss(val_y, train_pred[val_idx])
        cv_scores.append(fold_score)
        models.append(model)
        print(f"    Fold {fold_idx + 1}: logloss = {fold_score:.4f}")

    mean_cv = np.mean(cv_scores)
    print(f"\n  Mean CV logloss: {mean_cv:.4f} (+/- {np.std(cv_scores):.4f})")

    # Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        'EAP': test_pred[:, 0],
        'HPL': test_pred[:, 1],
        'MWS': test_pred[:, 2],
    })

    _save_data(submission, outputs["submission"])

    # Save model
    model_artifact = {
        "models": models,
        "feature_cols": list(train_X.columns),
        "label_encoder": le,
        "model_type": "xgboost",
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_artifact, f)

    # Save metrics
    metrics = {
        "model_type": "xgboost_stacked",
        "cv_logloss": float(mean_cv),
        "cv_std": float(np.std(cv_scores)),
        "n_folds": n_folds,
        "n_features": int(train_X.shape[1]),
        "n_train_samples": int(len(train_y)),
        "n_test_samples": int(len(test_ids)),
        "stacked_models": ["nb_word_countvec", "nb_char_countvec", "nb_char_tfidf"],
        "meta_features": list(train_meta.columns),
        "fold_scores": [float(s) for s in cv_scores],
    }
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Pipeline complete! CV logloss: {mean_cv:.4f}")
    print("=" * 60)

    return f"run_spooky_author_pipeline: CV logloss = {mean_cv:.4f}, submission saved"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "extract_author_meta_features": extract_author_meta_features,
    "train_nb_stack_features": train_nb_stack_features,
    "train_author_classifier": train_author_classifier,
    "predict_author_probabilities": predict_author_probabilities,
    "run_spooky_author_pipeline": run_spooky_author_pipeline,
}
