"""
Google QUEST Q&A Labeling - Contract-Composable Analytics Services
============================================
Competition: https://www.kaggle.com/competitions/google-quest-challenge
Problem Type: Multi-label Regression (30 continuous targets, 0-1 range)
Metric: Spearman's Rank Correlation Coefficient
Target: 30 columns (question_asker_intent_understanding ... answer_well_written)
ID Column: qa_id

Top solution insights (ALBERT/BERT transformer models):
- Text features (question_title + question_body + answer) are the primary input
- Multi-output regression with sigmoid activation (values clipped to [0, 1])
- K-fold cross-validation with GroupKFold on question_body
- Spearman correlation as evaluation metric

Contract-Composable Analytics approach: TF-IDF vectorization + Multi-output regression (RandomForest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from services.nlp_getting_started_services import (
        combine_text_columns,
        clean_text_column,
        fit_tfidf_vectorizer,
        transform_tfidf,
    )
    from services.preprocessing_services import drop_columns, split_data
    from services.regression_services import (
        train_multi_output_regressor,
        predict_multi_output_regressor,
    )
except ImportError:
    from nlp_getting_started_services import (
        combine_text_columns,
        clean_text_column,
        fit_tfidf_vectorizer,
        transform_tfidf,
    )
    from preprocessing_services import drop_columns, split_data
    from regression_services import (
        train_multi_output_regressor,
        predict_multi_output_regressor,
    )


# =============================================================================
# TARGET COLUMNS (all 30 targets for this competition)
# =============================================================================

TARGET_COLUMNS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]

# Columns to drop (non-feature text/metadata columns)
NON_FEATURE_COLUMNS = [
    "qa_id",
    "question_title",
    "question_body",
    "question_user_name",
    "question_user_page",
    "answer",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
    "combined_text",
    "text_clean",
]


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "combine_text_columns": combine_text_columns,
    "clean_text_column": clean_text_column,
    "fit_tfidf_vectorizer": fit_tfidf_vectorizer,
    "transform_tfidf": transform_tfidf,
    "drop_columns": drop_columns,
    "split_data": split_data,
}
