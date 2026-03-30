"""
Recruit Restaurant Visitor Forecasting - SLEGO Services
========================================================
Competition: https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting
Problem Type: Regression (Time-series forecasting)
Target: visitors (log1p transformed)
Metric: RMSLE

Competition-specific services based on top solutions:
- prepare_recruit_data: Merge all data sources and create base features
- create_store_historical_features: Store-level statistics over time windows
- create_dow_features: Day-of-week specific features per store
- create_reservation_features: Aggregate reservation data
- create_genre_area_features: Genre and area level aggregations

Reused services from:
- regression_services: train_lightgbm_regressor, predict_regressor
- temporal_services: extract_datetime_features
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from functools import wraps
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder


def contract(inputs=None, outputs=None, params=None, description=None, tags=None, version="1.0.0"):
    """Service contract decorator for SLEGO services."""
    def decorator(func):
        func._contract = {
            'inputs': inputs or {},
            'outputs': outputs or {},
            'params': params or {},
            'description': description or func.__doc__,
            'tags': tags or [],
            'version': version
        }
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._contract = func._contract
        return wrapper
    return decorator


# Import reusable services
try:
    from services.regression_services import train_lightgbm_regressor, predict_regressor
except ImportError:
    from regression_services import train_lightgbm_regressor, predict_regressor


# =============================================================================
# Data Preparation Services
# =============================================================================

@contract(
    inputs={
        "air_visit": {"format": "csv", "required": True},
        "air_store": {"format": "csv", "required": True},
        "air_reserve": {"format": "csv", "required": True},
        "hpg_reserve": {"format": "csv", "required": True},
        "hpg_store": {"format": "csv", "required": True},
        "date_info": {"format": "csv", "required": True},
        "store_relation": {"format": "csv", "required": True},
        "submission": {"format": "csv", "required": True}
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"}
    },
    description="Prepare recruit restaurant data by merging all sources",
    tags=["data-preparation", "restaurant-forecasting", "time-series"]
)
def prepare_recruit_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    log_transform_target: bool = True
) -> str:
    """
    Merge all data sources for recruit restaurant forecasting.

    Based on 1st place solution approach:
    - Merge air_visit with store info
    - Map hpg stores to air stores
    - Add date info (holidays)
    - Create submission template for test data
    """
    # Load all data
    air_visit = pd.read_csv(inputs['air_visit'])
    air_store = pd.read_csv(inputs['air_store'])
    air_reserve = pd.read_csv(inputs['air_reserve'])
    hpg_reserve = pd.read_csv(inputs['hpg_reserve'])
    hpg_store = pd.read_csv(inputs['hpg_store'])
    date_info = pd.read_csv(inputs['date_info'])
    store_relation = pd.read_csv(inputs['store_relation'])
    submission = pd.read_csv(inputs['submission'])

    # Rename columns for consistency
    air_visit = air_visit.rename(columns={'air_store_id': 'store_id'})
    air_store = air_store.rename(columns={'air_store_id': 'store_id'})
    date_info = date_info.rename(columns={'calendar_date': 'visit_date'})

    # Map HPG stores to AIR stores
    store_map = store_relation.set_index('hpg_store_id')['air_store_id']

    # Process submission to create test data
    submission['visit_date'] = submission['id'].str[-10:]
    submission['store_id'] = submission['id'].str[:-11]

    # Log transform target
    if log_transform_target:
        air_visit['visitors'] = np.log1p(air_visit['visitors'])

    # Create combined data (train + test template)
    test_data = submission[['store_id', 'visit_date']].copy()
    test_data['visitors'] = 0  # placeholder
    test_data['is_train'] = 0

    air_visit['is_train'] = 1
    train_data = air_visit[['store_id', 'visit_date', 'visitors', 'is_train']].copy()

    # Add ID column for train
    train_data['id'] = train_data['store_id'] + '_' + train_data['visit_date']
    test_data['id'] = submission['id']

    # Add day of week
    train_data['dow'] = pd.to_datetime(train_data['visit_date']).dt.dayofweek
    test_data['dow'] = pd.to_datetime(test_data['visit_date']).dt.dayofweek

    # Merge with store info
    # Process air_store: extract area prefix, encode
    air_store['air_area_name0'] = air_store['air_area_name'].apply(lambda x: x.split(' ')[0])
    lbl = LabelEncoder()
    air_store['air_genre_name_enc'] = lbl.fit_transform(air_store['air_genre_name'])
    air_store['air_area_name0_enc'] = lbl.fit_transform(air_store['air_area_name0'])

    train_data = train_data.merge(
        air_store[['store_id', 'air_genre_name_enc', 'air_area_name0_enc', 'latitude', 'longitude']],
        on='store_id', how='left'
    )
    test_data = test_data.merge(
        air_store[['store_id', 'air_genre_name_enc', 'air_area_name0_enc', 'latitude', 'longitude']],
        on='store_id', how='left'
    )

    # Add holiday info
    date_info['holiday_flg2'] = pd.to_datetime(date_info['visit_date']).dt.dayofweek
    date_info['holiday_flg2'] = ((date_info['holiday_flg2'] > 4) | (date_info['holiday_flg'] == 1)).astype(int)

    train_data = train_data.merge(date_info[['visit_date', 'holiday_flg', 'holiday_flg2']], on='visit_date', how='left')
    test_data = test_data.merge(date_info[['visit_date', 'holiday_flg', 'holiday_flg2']], on='visit_date', how='left')

    # Fill NaN
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)

    # Save
    train_data.to_csv(outputs['train_data'], index=False)
    test_data.to_csv(outputs['test_data'], index=False)

    return f"prepare_recruit_data: train={len(train_data)}, test={len(test_data)}"


@contract(
    inputs={
        "data": {"format": "csv", "required": True},
        "historical_data": {"format": "csv", "required": True}
    },
    outputs={"data": {"format": "csv"}},
    description="Create store-level historical statistics",
    tags=["feature-engineering", "restaurant-forecasting", "time-series"]
)
def create_store_historical_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    windows: List[int] = None,
    target_column: str = 'visitors',
    store_column: str = 'store_id',
    date_column: str = 'visit_date'
) -> str:
    """
    Create historical aggregate features per store over different time windows.

    Features: mean, median, std, count, min, max per window.
    Based on 1st place solution approach with exponential weighting.
    """
    df = pd.read_csv(inputs['data'])
    hist = pd.read_csv(inputs['historical_data'])

    if windows is None:
        windows = [14, 28, 56, 140]

    # Ensure datetime
    df[date_column] = pd.to_datetime(df[date_column])
    hist[date_column] = pd.to_datetime(hist[date_column])

    # For each row, compute historical stats from data before that date
    # Group by store and compute rolling stats

    # First, get global stats per store (from historical data)
    store_stats = hist.groupby(store_column)[target_column].agg([
        'mean', 'median', 'std', 'count', 'min', 'max'
    ]).reset_index()
    store_stats.columns = [store_column] + [f'store_{stat}_all' for stat in ['mean', 'median', 'std', 'count', 'min', 'max']]

    df = df.merge(store_stats, on=store_column, how='left')

    # Add window-specific features using exponential decay (simplified version)
    for window in windows:
        # Filter historical data to recent window
        max_date = hist[date_column].max()
        window_start = max_date - timedelta(days=window)
        hist_window = hist[hist[date_column] > window_start]

        window_stats = hist_window.groupby(store_column)[target_column].agg([
            'mean', 'median', 'std', 'count', 'min', 'max'
        ]).reset_index()
        window_stats.columns = [store_column] + [f'store_{stat}_{window}' for stat in ['mean', 'median', 'std', 'count', 'min', 'max']]

        df = df.merge(window_stats, on=store_column, how='left')

    df = df.fillna(0)
    df.to_csv(outputs['data'], index=False)

    return f"create_store_historical_features: windows={windows}, features added"


@contract(
    inputs={
        "data": {"format": "csv", "required": True},
        "historical_data": {"format": "csv", "required": True}
    },
    outputs={"data": {"format": "csv"}},
    description="Create day-of-week specific features per store",
    tags=["feature-engineering", "restaurant-forecasting", "time-series"]
)
def create_dow_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = 'visitors',
    store_column: str = 'store_id',
    dow_column: str = 'dow'
) -> str:
    """
    Create day-of-week specific statistics per store.

    Visitors patterns differ significantly by day of week (weekday vs weekend).
    """
    df = pd.read_csv(inputs['data'])
    hist = pd.read_csv(inputs['historical_data'])

    # Day-of-week stats per store
    dow_stats = hist.groupby([store_column, dow_column])[target_column].agg([
        'mean', 'median', 'std', 'count', 'min', 'max'
    ]).reset_index()
    dow_stats.columns = [store_column, dow_column] + [f'store_dow_{stat}' for stat in ['mean', 'median', 'std', 'count', 'min', 'max']]

    df = df.merge(dow_stats, on=[store_column, dow_column], how='left')
    df = df.fillna(0)

    df.to_csv(outputs['data'], index=False)

    return f"create_dow_features: added dow-specific features"


@contract(
    inputs={
        "data": {"format": "csv", "required": True},
        "air_reserve": {"format": "csv", "required": True},
        "hpg_reserve": {"format": "csv", "required": True},
        "store_relation": {"format": "csv", "required": True}
    },
    outputs={"data": {"format": "csv"}},
    description="Create reservation-based features",
    tags=["feature-engineering", "restaurant-forecasting", "reservations"]
)
def create_reservation_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    store_column: str = 'store_id',
    date_column: str = 'visit_date'
) -> str:
    """
    Create features from reservation data.

    Key insight from winning solutions: reservation count and visitor count
    from reservations are strong predictors.
    """
    df = pd.read_csv(inputs['data'])
    air_reserve = pd.read_csv(inputs['air_reserve'])
    hpg_reserve = pd.read_csv(inputs['hpg_reserve'])
    store_relation = pd.read_csv(inputs['store_relation'])

    # Map hpg store ids to air store ids
    store_map = dict(zip(store_relation['hpg_store_id'], store_relation['air_store_id']))
    hpg_reserve['air_store_id'] = hpg_reserve['hpg_store_id'].map(store_map)
    hpg_reserve = hpg_reserve[hpg_reserve['air_store_id'].notna()]

    # Process air reservations
    air_reserve['visit_date'] = air_reserve['visit_datetime'].str[:10]
    air_reserve['reserve_date'] = air_reserve['reserve_datetime'].str[:10]
    air_reserve['days_ahead'] = (
        pd.to_datetime(air_reserve['visit_date']) -
        pd.to_datetime(air_reserve['reserve_date'])
    ).dt.days

    # Aggregate reservations by store and visit_date
    air_agg = air_reserve.groupby(['air_store_id', 'visit_date']).agg({
        'reserve_visitors': ['sum', 'mean', 'count'],
        'days_ahead': 'mean'
    }).reset_index()
    air_agg.columns = ['store_id', 'visit_date', 'air_reserve_visitors', 'air_reserve_mean', 'air_reserve_count', 'air_days_ahead']

    # Process hpg reservations
    hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].str[:10]
    hpg_reserve['reserve_date'] = hpg_reserve['reserve_datetime'].str[:10]
    hpg_reserve['days_ahead'] = (
        pd.to_datetime(hpg_reserve['visit_date']) -
        pd.to_datetime(hpg_reserve['reserve_date'])
    ).dt.days

    hpg_agg = hpg_reserve.groupby(['air_store_id', 'visit_date']).agg({
        'reserve_visitors': ['sum', 'mean', 'count'],
        'days_ahead': 'mean'
    }).reset_index()
    hpg_agg.columns = ['store_id', 'visit_date', 'hpg_reserve_visitors', 'hpg_reserve_mean', 'hpg_reserve_count', 'hpg_days_ahead']

    # Merge with main data
    df = df.merge(air_agg, on=[store_column, date_column], how='left')
    df = df.merge(hpg_agg, on=[store_column, date_column], how='left')

    # Fill missing reservation features with 0
    reserve_cols = [c for c in df.columns if 'reserve' in c or 'days_ahead' in c]
    df[reserve_cols] = df[reserve_cols].fillna(0)

    df.to_csv(outputs['data'], index=False)

    return f"create_reservation_features: added {len(reserve_cols)} reservation features"


@contract(
    inputs={
        "data": {"format": "csv", "required": True},
        "historical_data": {"format": "csv", "required": True}
    },
    outputs={"data": {"format": "csv"}},
    description="Create genre and area level aggregate features",
    tags=["feature-engineering", "restaurant-forecasting"]
)
def create_genre_area_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = 'visitors',
    genre_column: str = 'air_genre_name_enc',
    dow_column: str = 'dow'
) -> str:
    """
    Create genre and area-level statistics.

    Helps generalize across stores with similar characteristics.
    """
    df = pd.read_csv(inputs['data'])
    hist = pd.read_csv(inputs['historical_data'])

    if genre_column not in df.columns:
        df.to_csv(outputs['data'], index=False)
        return "create_genre_area_features: genre column not found, skipped"

    # Genre stats
    genre_stats = hist.groupby(genre_column)[target_column].agg(['mean', 'std']).reset_index()
    genre_stats.columns = [genre_column, 'genre_mean', 'genre_std']
    df = df.merge(genre_stats, on=genre_column, how='left')

    # Genre + dow stats
    if dow_column in hist.columns:
        genre_dow_stats = hist.groupby([genre_column, dow_column])[target_column].agg(['mean', 'std']).reset_index()
        genre_dow_stats.columns = [genre_column, dow_column, 'genre_dow_mean', 'genre_dow_std']
        df = df.merge(genre_dow_stats, on=[genre_column, dow_column], how='left')

    df = df.fillna(0)
    df.to_csv(outputs['data'], index=False)

    return "create_genre_area_features: added genre-level features"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True}
    },
    outputs={"submission": {"format": "csv"}},
    description="Create submission file with proper formatting",
    tags=["submission", "restaurant-forecasting"]
)
def create_recruit_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = 'id',
    prediction_column: str = 'visitors',
    reverse_log: bool = True
) -> str:
    """
    Create submission file with proper id and visitors columns.

    Applies expm1 to reverse log1p transformation.
    """
    test = pd.read_csv(inputs['test_data'])

    if prediction_column not in test.columns:
        test[prediction_column] = 0

    # Reverse log transform
    if reverse_log:
        test[prediction_column] = np.expm1(test[prediction_column])

    # Ensure non-negative
    test[prediction_column] = test[prediction_column].clip(lower=0)

    # Round to integer visitors
    test[prediction_column] = test[prediction_column].round().astype(int)

    # Create submission
    submission = test[[id_column, prediction_column]].copy()
    submission.to_csv(outputs['submission'], index=False)

    return f"create_recruit_submission: {len(submission)} rows"


# =============================================================================
# Full Pipeline Runner
# =============================================================================

def run_recruit_pipeline(base_path: str, n_estimators: int = 1000, learning_rate: float = 0.02) -> str:
    """
    Run the complete recruit restaurant forecasting pipeline.

    This is a convenience function for quick execution.
    """
    import os

    datasets = os.path.join(base_path, 'datasets')
    artifacts = os.path.join(base_path, 'artifacts')
    os.makedirs(artifacts, exist_ok=True)

    # Step 1: Prepare data
    print("Step 1: Preparing data...")
    prepare_recruit_data(
        inputs={
            'air_visit': os.path.join(datasets, 'air_visit_data.csv'),
            'air_store': os.path.join(datasets, 'air_store_info.csv'),
            'air_reserve': os.path.join(datasets, 'air_reserve.csv'),
            'hpg_reserve': os.path.join(datasets, 'hpg_reserve.csv'),
            'hpg_store': os.path.join(datasets, 'hpg_store_info.csv'),
            'date_info': os.path.join(datasets, 'date_info.csv'),
            'store_relation': os.path.join(datasets, 'store_id_relation.csv'),
            'submission': os.path.join(datasets, 'sample_submission.csv')
        },
        outputs={
            'train_data': os.path.join(artifacts, 'train_prepared.csv'),
            'test_data': os.path.join(artifacts, 'test_prepared.csv')
        }
    )

    # Step 2: Create store historical features
    print("Step 2: Creating store historical features...")
    create_store_historical_features(
        inputs={
            'data': os.path.join(artifacts, 'train_prepared.csv'),
            'historical_data': os.path.join(artifacts, 'train_prepared.csv')
        },
        outputs={'data': os.path.join(artifacts, 'train_01_store_features.csv')}
    )
    create_store_historical_features(
        inputs={
            'data': os.path.join(artifacts, 'test_prepared.csv'),
            'historical_data': os.path.join(artifacts, 'train_prepared.csv')
        },
        outputs={'data': os.path.join(artifacts, 'test_01_store_features.csv')}
    )

    # Step 3: Create dow features
    print("Step 3: Creating day-of-week features...")
    create_dow_features(
        inputs={
            'data': os.path.join(artifacts, 'train_01_store_features.csv'),
            'historical_data': os.path.join(artifacts, 'train_prepared.csv')
        },
        outputs={'data': os.path.join(artifacts, 'train_02_dow_features.csv')}
    )
    create_dow_features(
        inputs={
            'data': os.path.join(artifacts, 'test_01_store_features.csv'),
            'historical_data': os.path.join(artifacts, 'train_prepared.csv')
        },
        outputs={'data': os.path.join(artifacts, 'test_02_dow_features.csv')}
    )

    # Step 4: Create reservation features
    print("Step 4: Creating reservation features...")
    create_reservation_features(
        inputs={
            'data': os.path.join(artifacts, 'train_02_dow_features.csv'),
            'air_reserve': os.path.join(datasets, 'air_reserve.csv'),
            'hpg_reserve': os.path.join(datasets, 'hpg_reserve.csv'),
            'store_relation': os.path.join(datasets, 'store_id_relation.csv')
        },
        outputs={'data': os.path.join(artifacts, 'train_03_reserve_features.csv')}
    )
    create_reservation_features(
        inputs={
            'data': os.path.join(artifacts, 'test_02_dow_features.csv'),
            'air_reserve': os.path.join(datasets, 'air_reserve.csv'),
            'hpg_reserve': os.path.join(datasets, 'hpg_reserve.csv'),
            'store_relation': os.path.join(datasets, 'store_id_relation.csv')
        },
        outputs={'data': os.path.join(artifacts, 'test_03_reserve_features.csv')}
    )

    # Step 5: Create genre features
    print("Step 5: Creating genre/area features...")
    create_genre_area_features(
        inputs={
            'data': os.path.join(artifacts, 'train_03_reserve_features.csv'),
            'historical_data': os.path.join(artifacts, 'train_prepared.csv')
        },
        outputs={'data': os.path.join(artifacts, 'train_final.csv')}
    )
    create_genre_area_features(
        inputs={
            'data': os.path.join(artifacts, 'test_03_reserve_features.csv'),
            'historical_data': os.path.join(artifacts, 'train_prepared.csv')
        },
        outputs={'data': os.path.join(artifacts, 'test_final.csv')}
    )

    # Step 6: Train model
    print("Step 6: Training LightGBM model...")
    train_lightgbm_regressor(
        inputs={
            'train_data': os.path.join(artifacts, 'train_final.csv')
        },
        outputs={
            'model': os.path.join(artifacts, 'model.pkl'),
            'metrics': os.path.join(artifacts, 'metrics.json')
        },
        target_column='visitors',
        id_column='id',
        exclude_columns=['store_id', 'visit_date', 'is_train'],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=60,
        max_depth=-1,
        min_child_samples=100,
        random_state=42,
        verbose=-1
    )

    # Step 7: Predict
    print("Step 7: Generating predictions...")
    predict_regressor(
        inputs={
            'model': os.path.join(artifacts, 'model.pkl'),
            'test_data': os.path.join(artifacts, 'test_final.csv')
        },
        outputs={
            'predictions': os.path.join(artifacts, 'predictions.csv')
        },
        id_column='id',
        prediction_column='visitors'
    )

    # Step 8: Create submission
    print("Step 8: Creating submission...")
    pred_df = pd.read_csv(os.path.join(artifacts, 'predictions.csv'))
    pred_df['visitors'] = np.expm1(pred_df['visitors']).clip(lower=0).round().astype(int)
    pred_df[['id', 'visitors']].to_csv(os.path.join(base_path, 'submission.csv'), index=False)

    return f"Pipeline complete! Submission saved to {os.path.join(base_path, 'submission.csv')}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific services
    "prepare_recruit_data": prepare_recruit_data,
    "create_store_historical_features": create_store_historical_features,
    "create_dow_features": create_dow_features,
    "create_reservation_features": create_reservation_features,
    "create_genre_area_features": create_genre_area_features,
    "create_recruit_submission": create_recruit_submission,
    "run_recruit_pipeline": run_recruit_pipeline,
    # Reused services
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "predict_regressor": predict_regressor,
}
