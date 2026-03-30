"""
CS-4248 Fact Checking - Contract-Composable Analytics Services
=======================================
Competition: https://www.kaggle.com/competitions/cs-4248-fact-checking-2420
Problem Type: Multiclass Classification (Verdict: -1, 0, 1)
Target: Verdict
ID Column: Sentence_id

This is a fact-checking task where sentences need to be classified into
three categories based on their factual veracity:
- -1: False/Incorrect statement
-  0: Neutral/Unknown
-  1: True/Correct statement

The competition is part of NUS CS-4248 course (Natural Language Processing).

Services: Imports reusable text and classification services from common modules.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import reusable text services
from services.text_services import (
    clean_text,
    extract_text_features,
    vectorize_tfidf,
    transform_tfidf,
    drop_text_columns,
)

# Import reusable classification services
from services.classification_services import (
    train_lightgbm_classifier,
    train_random_forest_classifier,
    predict_classifier,
)

# Import preprocessing services
from services.preprocessing_services import (
    split_data,
)


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Text preprocessing
    "clean_text": clean_text,
    "extract_text_features": extract_text_features,
    "vectorize_tfidf": vectorize_tfidf,
    "transform_tfidf": transform_tfidf,
    "drop_text_columns": drop_text_columns,
    # Classification
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "train_random_forest_classifier": train_random_forest_classifier,
    "predict_classifier": predict_classifier,
    # Preprocessing
    "split_data": split_data,
}
