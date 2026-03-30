"""
Two Sigma Connect: Rental Listing Inquiries - Contract-Composable Analytics Services
============================================================
Competition: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries
Problem Type: Multiclass classification (high, medium, low interest)
Target: interest_level
Metric: Multi-class log loss

This module combines winning features from top 3 solutions:
- Solution 1 (chriscc): Manager/building aggregations, geo-clustering, target encoding
- Solution 2 (slavik0505): Coordinate rotations, feature deduplication
- Solution 3 (chengzhan): XGBoost with density features, half bathrooms

Key Features Implemented:
1. Manager target encoding (5-fold CV to prevent leakage)
2. Building target encoding
3. Location density features
4. Distance to NYC landmarks
5. Price interaction features
6. Listing quality features
7. Text features using CountVectorizer
8. Coordinate rotations
"""

import os
import sys
import json
import pickle
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _load_json_data(path: str) -> pd.DataFrame:
    """Load JSON data from file or zip."""
    if path.endswith('.zip'):
        return pd.read_json(path, compression='zip')
    return pd.read_json(path)


def _create_manager_target_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                     n_folds: int = 5, seed: int = 42) -> tuple:
    """
    Create manager-level target encoding with CV to prevent leakage.
    Based on Solution 2 and 3 approach.
    """
    random.seed(seed)

    index = list(range(train_df.shape[0]))
    random.shuffle(index)

    a = [np.nan] * len(train_df)
    b = [np.nan] * len(train_df)
    c = [np.nan] * len(train_df)

    for i in range(n_folds):
        building_level = {}
        for j in train_df['manager_id'].values:
            building_level[j] = [0, 0, 0]

        test_index = index[int((i * train_df.shape[0]) / n_folds):int(((i + 1) * train_df.shape[0]) / n_folds)]
        train_index = list(set(index).difference(test_index))

        for j in train_index:
            temp = train_df.iloc[j]
            if temp['interest_level'] == 'low':
                building_level[temp['manager_id']][0] += 1
            elif temp['interest_level'] == 'medium':
                building_level[temp['manager_id']][1] += 1
            elif temp['interest_level'] == 'high':
                building_level[temp['manager_id']][2] += 1

        for j in test_index:
            temp = train_df.iloc[j]
            total = sum(building_level[temp['manager_id']])
            if total != 0:
                a[j] = building_level[temp['manager_id']][0] / total
                b[j] = building_level[temp['manager_id']][1] / total
                c[j] = building_level[temp['manager_id']][2] / total

    train_df = train_df.copy()
    train_df['manager_level_low'] = a
    train_df['manager_level_medium'] = b
    train_df['manager_level_high'] = c

    # For test set, use all training data
    building_level = {}
    for j in train_df['manager_id'].values:
        building_level[j] = [0, 0, 0]

    for j in range(len(train_df)):
        temp = train_df.iloc[j]
        if temp['interest_level'] == 'low':
            building_level[temp['manager_id']][0] += 1
        elif temp['interest_level'] == 'medium':
            building_level[temp['manager_id']][1] += 1
        elif temp['interest_level'] == 'high':
            building_level[temp['manager_id']][2] += 1

    test_a, test_b, test_c = [], [], []
    for mgr_id in test_df['manager_id'].values:
        if mgr_id not in building_level:
            test_a.append(np.nan)
            test_b.append(np.nan)
            test_c.append(np.nan)
        else:
            total = sum(building_level[mgr_id])
            test_a.append(building_level[mgr_id][0] / total)
            test_b.append(building_level[mgr_id][1] / total)
            test_c.append(building_level[mgr_id][2] / total)

    test_df = test_df.copy()
    test_df['manager_level_low'] = test_a
    test_df['manager_level_medium'] = test_b
    test_df['manager_level_high'] = test_c

    return train_df, test_df


def _add_coordinate_rotations(df: pd.DataFrame, angles: List[int] = [15, 30, 45, 60]) -> pd.DataFrame:
    """Add coordinate rotation features (from Solution 2)."""
    df = df.copy()
    for angle in angles:
        alpha = math.pi / (180 / angle)
        df[f'rot{angle}_x'] = df['latitude'] * math.cos(alpha) + df['longitude'] * math.sin(alpha)
        df[f'rot{angle}_y'] = df['longitude'] * math.cos(alpha) - df['latitude'] * math.sin(alpha)
    return df


# =============================================================================
# MAIN SERVICES
# =============================================================================

@contract(
    inputs={
        "train": {"format": "json", "required": True},
        "test": {"format": "json", "required": True},
    },
    outputs={
        "data": {"format": "csv"},
    },
    description="Load and combine train/test JSON data for Two Sigma rental prediction",
    tags=["io", "json", "two-sigma", "rental"],
    version="2.0.0",
)
def load_rental_json_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """Load train and test JSON files and combine them."""
    train_df = _load_json_data(inputs["train"])
    test_df = _load_json_data(inputs["test"])

    train_df['is_train'] = 1
    test_df['is_train'] = 0

    combined = pd.concat([train_df, test_df], ignore_index=True)

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    combined.to_csv(outputs["data"], index=False)

    return f"load_rental_json_data: {len(train_df)} train, {len(test_df)} test rows"


@contract(
    inputs={
        "train": {"format": "json", "required": True},
        "test": {"format": "json", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
        "model": {"format": "pickle"},
    },
    description="Full pipeline: feature engineering + LightGBM training for Two Sigma rental prediction",
    tags=["training", "lightgbm", "two-sigma", "multiclass", "feature-engineering"],
    version="3.0.0",
)
def train_two_sigma_lightgbm(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    max_depth: int = 15,
    subsample: float = 0.7,
    colsample_bytree: float = 0.7,
    min_child_samples: int = 30,
    random_state: int = 42,
    text_max_features: int = 200,
    do_cv: bool = True,
    n_folds: int = 5,
) -> str:
    """
    Complete Two Sigma pipeline combining winning features from top solutions.

    Uses LightGBM multiclass classifier with:
    - Manager/building target encoding (CV-based)
    - Price features and interactions
    - Datetime features
    - Location features (density, distance to landmarks, geo areas)
    - Listing quality features (from Solution 1)
    - Manager aggregation features (from Solution 1)
    - Text features (CountVectorizer on features list)
    - Coordinate rotations
    """
    import lightgbm as lgb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss

    # Load data
    print("Loading data...")
    train_df = _load_json_data(inputs["train"])
    test_df = _load_json_data(inputs["test"])

    train_size = len(train_df)
    print(f"Train: {train_size}, Test: {len(test_df)}")

    # Store IDs and target
    train_ids = train_df['listing_id'].values
    test_ids = test_df['listing_id'].values
    target = train_df['interest_level']
    target_map = {'low': 2, 'medium': 1, 'high': 0}
    y = target.map(target_map).values

    # Merge for feature engineering
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    full_data = pd.concat([train_df, test_df], ignore_index=True)

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    print("Engineering features...")

    # 1. Fix outliers (from Solution 3)
    mean_price = int(train_df['price'].mean())
    full_data.loc[full_data['price'] < 200, 'price'] = mean_price
    full_data['price'] = full_data['price'].clip(upper=13000)

    # 2. Price features
    full_data['logprice'] = np.log1p(full_data['price'])
    full_data['rooms'] = full_data['bedrooms'] + full_data['bathrooms']
    full_data['price_per_room'] = full_data['price'] / (full_data['rooms'] + 1)
    full_data['price_per_bed'] = full_data['price'] / (full_data['bedrooms'] + 1)
    full_data['price_per_bath'] = full_data['price'] / (full_data['bathrooms'] + 1)
    full_data['half_bathrooms'] = full_data['bathrooms'] - full_data['bathrooms'].astype(int)

    # 3. Datetime features
    full_data['created'] = pd.to_datetime(full_data['created'])
    full_data['created_year'] = full_data['created'].dt.year
    full_data['created_month'] = full_data['created'].dt.month
    full_data['created_day'] = full_data['created'].dt.day
    full_data['created_hour'] = full_data['created'].dt.hour
    full_data['created_weekday'] = full_data['created'].dt.weekday
    full_data['created_week'] = full_data['created'].dt.isocalendar().week.astype(int)
    full_data['created_dayofyear'] = full_data['created'].dt.dayofyear
    full_data['created_epoch'] = full_data['created'].astype(np.int64) // 10**9

    # 4. Text/count features
    full_data['num_photos'] = full_data['photos'].apply(len)
    full_data['num_features'] = full_data['features'].apply(len)
    full_data['num_desc_words'] = full_data['description'].apply(lambda x: len(str(x).split()))
    full_data['desc_len'] = full_data['description'].apply(lambda x: len(str(x)))

    # 5. Geo area features (from Solution 1)
    full_data['geo_area_50'] = full_data.apply(
        lambda x: (int(x['latitude'] * 50) % 50) * 50 + (int(-x['longitude'] * 50) % 50), axis=1)
    full_data['geo_area_100'] = full_data.apply(
        lambda x: (int(x['latitude'] * 100) % 100) * 100 + (int(-x['longitude'] * 100) % 100), axis=1)

    # 6. Location density (from Solution 3)
    full_data['pos'] = full_data['longitude'].round(3).astype(str) + '_' + full_data['latitude'].round(3).astype(str)
    pos_counts = full_data['pos'].value_counts().to_dict()
    full_data['density'] = full_data['pos'].map(pos_counts)

    # 7. Distance to NYC landmarks
    full_data['dist_fi'] = np.sqrt((full_data['latitude'] - 40.705628)**2 + (full_data['longitude'] + 74.010278)**2)
    full_data['dist_cp'] = np.sqrt((full_data['latitude'] - 40.785091)**2 + (full_data['longitude'] + 73.968285)**2)

    # 8. Polar coordinates (from Solution 2)
    full_data['rho'] = np.sqrt((full_data['latitude'] - 40.78222222)**2 + (full_data['longitude'] + 73.96527777)**2)
    full_data['phi'] = np.arctan2(full_data['latitude'] - 40.78222222, full_data['longitude'] + 73.96527777)

    # 9. Coordinate rotations (from Solution 2)
    for angle in [15, 30, 45, 60]:
        alpha = math.pi / (180 / angle)
        full_data[f'rot{angle}_x'] = full_data['latitude'] * math.cos(alpha) + full_data['longitude'] * math.sin(alpha)
        full_data[f'rot{angle}_y'] = full_data['longitude'] * math.cos(alpha) - full_data['latitude'] * math.sin(alpha)

    # 10. Listing quality features (from Solution 1)
    full_data['num_html_tags'] = full_data['description'].apply(lambda x: str(x).count('<'))
    full_data['num_exclaim'] = full_data['description'].apply(lambda x: str(x).count('!'))
    full_data['num_caps'] = full_data['description'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
    full_data['has_phone'] = full_data['description'].apply(
        lambda x: 1 if len([w for w in str(x).split() if w.isdigit() and len(w) == 10]) > 0 else 0)
    full_data['has_email'] = full_data['description'].apply(lambda x: 1 if '@' in str(x) else 0)
    full_data['building_is_zero'] = (full_data['building_id'] == '0').astype(int)

    # 11. Count features
    for col in ['manager_id', 'building_id', 'display_address', 'street_address']:
        counts = full_data[col].value_counts().to_dict()
        full_data[f'{col}_count'] = full_data[col].map(counts)

    # 12. Price interactions (from Solution 1)
    full_data['price_per_photo'] = full_data['price'] / (full_data['num_photos'] + 1)
    full_data['price_per_feature'] = full_data['price'] / (full_data['num_features'] + 1)
    full_data['photos_per_room'] = full_data['num_photos'] / (full_data['rooms'] + 1)
    full_data['features_per_room'] = full_data['num_features'] / (full_data['rooms'] + 1)

    # 13. Manager aggregations (from Solution 1) - price stats by manager
    mgr_price_stats = full_data.groupby('manager_id')['price'].agg(['mean', 'std', 'min', 'max']).reset_index()
    mgr_price_stats.columns = ['manager_id', 'mgr_price_mean', 'mgr_price_std', 'mgr_price_min', 'mgr_price_max']
    full_data = full_data.merge(mgr_price_stats, on='manager_id', how='left')
    full_data['price_vs_mgr_mean'] = full_data['price'] - full_data['mgr_price_mean']

    # 14. Building aggregations
    bld_price_stats = full_data.groupby('building_id')['price'].agg(['mean', 'std']).reset_index()
    bld_price_stats.columns = ['building_id', 'bld_price_mean', 'bld_price_std']
    full_data = full_data.merge(bld_price_stats, on='building_id', how='left')

    # 15. Label encode categoricals
    print("Encoding categoricals...")
    categoricals = ['display_address', 'manager_id', 'building_id', 'street_address']
    for col in categoricals:
        lbl = LabelEncoder()
        full_data[col + '_enc'] = lbl.fit_transform(full_data[col].astype(str))

    # Split back
    train_df = full_data[full_data['is_train'] == 1].copy()
    test_df = full_data[full_data['is_train'] == 0].copy()

    # 16. Manager target encoding (must be done separately to avoid leakage)
    print("Creating target encodings...")
    train_df, test_df = _create_manager_target_encoding(train_df, test_df, n_folds=5, seed=random_state)

    # 17. Text features using CountVectorizer
    print("Creating text features...")
    train_df['features_str'] = train_df['features'].apply(lambda x: " ".join(["_".join(str(i).split()) for i in x]))
    test_df['features_str'] = test_df['features'].apply(lambda x: " ".join(["_".join(str(i).split()) for i in x]))

    tfidf = CountVectorizer(stop_words='english', max_features=text_max_features)
    train_sparse = tfidf.fit_transform(train_df['features_str'])
    test_sparse = tfidf.transform(test_df['features_str'])

    # =========================================================================
    # PREPARE FINAL FEATURES
    # =========================================================================
    feature_cols = [
        'bathrooms', 'bedrooms', 'latitude', 'longitude', 'price',
        'logprice', 'rooms', 'price_per_room', 'price_per_bed', 'price_per_bath', 'half_bathrooms',
        'created_year', 'created_month', 'created_day', 'created_hour', 'created_weekday', 'created_week',
        'created_dayofyear', 'created_epoch',
        'num_photos', 'num_features', 'num_desc_words', 'desc_len',
        'geo_area_50', 'geo_area_100', 'density', 'dist_fi', 'dist_cp', 'rho', 'phi',
        'rot15_x', 'rot15_y', 'rot30_x', 'rot30_y', 'rot45_x', 'rot45_y', 'rot60_x', 'rot60_y',
        'num_html_tags', 'num_exclaim', 'num_caps', 'has_phone', 'has_email', 'building_is_zero',
        'manager_id_count', 'building_id_count', 'display_address_count', 'street_address_count',
        'price_per_photo', 'price_per_feature', 'photos_per_room', 'features_per_room',
        'mgr_price_mean', 'mgr_price_std', 'mgr_price_min', 'mgr_price_max', 'price_vs_mgr_mean',
        'bld_price_mean', 'bld_price_std',
        'manager_level_low', 'manager_level_medium', 'manager_level_high',
        'display_address_enc', 'manager_id_enc', 'building_id_enc', 'street_address_enc',
        'listing_id',
    ]

    # Fill NaN values
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(-1)
            test_df[col] = test_df[col].fillna(-1)

    X_train = sparse.hstack([train_df[feature_cols].values, train_sparse]).tocsr()
    X_test = sparse.hstack([test_df[feature_cols].values, test_sparse]).tocsr()

    print(f"Feature shape: train={X_train.shape}, test={X_test.shape}")

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("Training LightGBM...")

    lgb_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'seed': random_state,
        'verbose': -1,
    }

    cv_scores = []
    oof_preds = np.zeros((train_size, 3))

    if do_cv:
        print(f"Running {n_folds}-fold CV...")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr)
            val_set = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                lgb_params,
                train_set,
                num_boost_round=n_estimators,
                valid_sets=[train_set, val_set],
                valid_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )

            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds
            fold_score = log_loss(y_val, val_preds)
            cv_scores.append(fold_score)
            print(f"  Fold {fold+1}: logloss = {fold_score:.5f}, best_iter = {model.best_iteration}")

        cv_mean = np.mean(cv_scores)
        print(f"CV Mean: {cv_mean:.5f} (+/- {np.std(cv_scores):.5f})")

    # Train final model on all data
    print("Training final model...")
    train_set = lgb.Dataset(X_train, label=y)

    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=n_estimators,
    )

    # Predict
    predictions = model.predict(X_test)

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["metrics"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)

    # Create submission
    submission = pd.DataFrame({
        'listing_id': test_ids,
        'high': predictions[:, 0],
        'medium': predictions[:, 1],
        'low': predictions[:, 2],
    })
    submission.to_csv(outputs["submission"], index=False)

    # Save metrics
    metrics = {
        'model_type': 'lightgbm_multiclass',
        'cv_logloss': float(np.mean(cv_scores)) if cv_scores else None,
        'cv_std': float(np.std(cv_scores)) if cv_scores else None,
        'n_features': X_train.shape[1],
        'n_train': train_size,
        'n_test': len(test_df),
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
    }
    with open(outputs["metrics"], 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save model
    with open(outputs["model"], 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feature_cols, 'tfidf': tfidf}, f)

    cv_str = f", CV={np.mean(cv_scores):.5f}" if cv_scores else ""
    return f"train_two_sigma_lightgbm: {train_size} samples, {X_train.shape[1]} features{cv_str}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================
SERVICE_REGISTRY = {
    "load_rental_json_data": load_rental_json_data,
    "train_two_sigma_lightgbm": train_two_sigma_lightgbm,
}
