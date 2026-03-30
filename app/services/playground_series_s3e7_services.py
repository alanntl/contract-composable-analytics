"""
Playground Series S3E7 - SLEGO Services
========================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e7
Problem Type: Binary Classification (booking_status: 0=Not Canceled, 1=Canceled)
Target: booking_status (integer 0/1)
ID Column: id
Evaluation Metric: ROC AUC (probability output required)

Predict hotel booking cancellation from reservation features.
Features include no_of_adults, no_of_children, lead_time, arrival_year/month/date,
type_of_meal_plan, room_type_reserved, market_segment_type, avg_price_per_room,
no_of_special_requests, etc. All features are already numeric (pre-encoded).

Solution Notebook Insights:
- Notebook 01 (mliammm): Linear model from scratch using PyTorch,
  feature normalization by dividing by max. Accuracy ~0.70.
- Notebook 02 (hardikgarg03): StandardScaler + winsorization of outliers.
  XGBoost (n_estimators=100, max_depth=5, lr=0.1) and
  RandomForest (n_estimators=100, max_depth=10) both perform well.
  Logistic Regression also tried.
- Notebook 03 (shresthapundir): Same approach as Notebook 02 (duplicate).

Key Techniques Across All Solutions:
1. All features already numeric - no encoding needed
2. StandardScaler / normalization improves linear models but optional for tree models
3. XGBoost and RandomForest both effective
4. Submission requires probability output (ROC AUC evaluation)
5. lead_time and avg_price_per_room are most important features

Competition-specific services: None required - all features are pre-encoded numeric.
Pipeline uses generic common services only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# IMPORTS FROM COMMON MODULES (G1: Reuse existing services)
# =============================================================================
try:
    from services.classification_services import (
        train_lightgbm_classifier, train_xgboost_classifier,
        train_random_forest_classifier, predict_classifier
    )
    from services.preprocessing_services import (
        split_data, create_submission, fit_scaler, transform_scaler
    )
except ImportError:
    from classification_services import (
        train_lightgbm_classifier, train_xgboost_classifier,
        train_random_forest_classifier, predict_classifier
    )
    from preprocessing_services import (
        split_data, create_submission, fit_scaler, transform_scaler
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Reused from common modules
    "split_data": split_data,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "train_xgboost_classifier": train_xgboost_classifier,
    "train_random_forest_classifier": train_random_forest_classifier,
    "predict_classifier": predict_classifier,
    "create_submission": create_submission,
    "fit_scaler": fit_scaler,
    "transform_scaler": transform_scaler,
}