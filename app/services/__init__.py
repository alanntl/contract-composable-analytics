"""
SLEGO Services Package
======================

Common service modules (reusable across competitions):
- preprocessing_services: Imputation, encoding, column filtering, scaling, feature engineering
- regression_services: Random forest, gradient boosting, LightGBM, XGBoost, ensemble regressors
- classification_services: LightGBM, random forest, XGBoost, ensemble classifiers
- time_series_services: Lag features, rolling features, temporal splits, recursive prediction
- image_services: Image loading, resizing, normalization, augmentation, classification
- text_services: TF-IDF/count vectorization, text classification
- clustering_services: KMeans, DBSCAN, hierarchical clustering, evaluation

Competition-specific service modules:
- house_prices_advanced_regression_techniques_services
- m5_forecasting_services
- bike_sharing_services
- home_credit_default_risk_services
- stanford_covid_vaccine_services
- allstate_claims_severity_services
- dont_call_me_turkey_services
- histopathologic_cancer_detection_services
- whats_cooking_services
"""

# --- Common modules ---
from .preprocessing_services import SERVICE_REGISTRY as PREPROCESSING_SERVICES
from .regression_services import SERVICE_REGISTRY as REGRESSION_SERVICES
from .classification_services import SERVICE_REGISTRY as CLASSIFICATION_SERVICES
from .time_series_services import SERVICE_REGISTRY as TIME_SERIES_SERVICES

try:
    from .image_services import SERVICE_REGISTRY as IMAGE_SERVICES
except ImportError:
    IMAGE_SERVICES = None

try:
    from .text_services import SERVICE_REGISTRY as TEXT_SERVICES
except ImportError:
    TEXT_SERVICES = None

try:
    from .clustering_services import SERVICE_REGISTRY as CLUSTERING_SERVICES
except ImportError:
    CLUSTERING_SERVICES = None

# --- Competition-specific modules ---
from .house_prices_advanced_regression_techniques_services import (
    PIPELINE_SPEC as HOUSE_PRICES_PIPELINE,
    SERVICE_REGISTRY as HOUSE_PRICES_SERVICES,
    run_pipeline as run_house_prices_pipeline,
)

try:
    from .m5_forecasting_services import (
        PIPELINE_SPEC as M5_PIPELINE,
        SERVICE_REGISTRY as M5_SERVICES,
        run_pipeline as run_m5_pipeline,
    )
except ImportError:
    M5_PIPELINE = None
    M5_SERVICES = None
    run_m5_pipeline = None

from .bike_sharing_services import (
    PIPELINE_SPEC as BIKE_SHARING_PIPELINE,
    SERVICE_REGISTRY as BIKE_SHARING_SERVICES,
    run_pipeline as run_bike_sharing_pipeline,
)

try:
    from .home_credit_default_risk_services import (
        PIPELINE_SPEC as HOME_CREDIT_PIPELINE,
        SERVICE_REGISTRY as HOME_CREDIT_SERVICES,
        run_pipeline as run_home_credit_pipeline,
    )
except ImportError:
    HOME_CREDIT_PIPELINE = None
    HOME_CREDIT_SERVICES = None
    run_home_credit_pipeline = None

try:
    from .stanford_covid_vaccine_services import (
        PIPELINE_SPEC as STANFORD_COVID_PIPELINE,
        SERVICE_REGISTRY as STANFORD_COVID_SERVICES,
        run_pipeline as run_stanford_covid_pipeline,
    )
except ImportError:
    STANFORD_COVID_PIPELINE = None
    STANFORD_COVID_SERVICES = None
    run_stanford_covid_pipeline = None

try:
    from .allstate_claims_severity_services import (
        PIPELINE_SPEC as ALLSTATE_PIPELINE,
        SERVICE_REGISTRY as ALLSTATE_SERVICES,
        run_pipeline as run_allstate_pipeline,
    )
except ImportError:
    ALLSTATE_PIPELINE = None
    ALLSTATE_SERVICES = None
    run_allstate_pipeline = None

try:
    from .dont_call_me_turkey_services import (
        SERVICE_REGISTRY as TURKEY_SERVICES,
    )
except ImportError:
    TURKEY_SERVICES = None

try:
    from .histopathologic_cancer_detection_services import (
        PIPELINE_SPEC as HISTOPATHOLOGIC_PIPELINE,
        SERVICE_REGISTRY as HISTOPATHOLOGIC_SERVICES,
        run_pipeline as run_histopathologic_pipeline,
    )
except ImportError:
    HISTOPATHOLOGIC_PIPELINE = None
    HISTOPATHOLOGIC_SERVICES = None
    run_histopathologic_pipeline = None

try:
    from .whats_cooking_services import (
        SERVICE_REGISTRY as WHATS_COOKING_SERVICES,
    )
except ImportError:
    WHATS_COOKING_SERVICES = None


__all__ = [
    # Common modules
    "PREPROCESSING_SERVICES",
    "REGRESSION_SERVICES",
    "CLASSIFICATION_SERVICES",
    "TIME_SERIES_SERVICES",
    "IMAGE_SERVICES",
    "TEXT_SERVICES",
    "CLUSTERING_SERVICES",
    # Competition-specific
    "HOUSE_PRICES_PIPELINE",
    "HOUSE_PRICES_SERVICES",
    "run_house_prices_pipeline",
    "M5_PIPELINE",
    "M5_SERVICES",
    "run_m5_pipeline",
    "BIKE_SHARING_PIPELINE",
    "BIKE_SHARING_SERVICES",
    "run_bike_sharing_pipeline",
    "HOME_CREDIT_PIPELINE",
    "HOME_CREDIT_SERVICES",
    "run_home_credit_pipeline",
    "STANFORD_COVID_PIPELINE",
    "STANFORD_COVID_SERVICES",
    "run_stanford_covid_pipeline",
    "ALLSTATE_PIPELINE",
    "ALLSTATE_SERVICES",
    "run_allstate_pipeline",
    "TURKEY_SERVICES",
    "HISTOPATHOLOGIC_PIPELINE",
    "HISTOPATHOLOGIC_SERVICES",
    "run_histopathologic_pipeline",
    "WHATS_COOKING_SERVICES",
]
