"""
BigQuery Geotab Intersection Congestion - Contract-Composable Analytics Services
=========================================================
Competition: https://www.kaggle.com/competitions/bigquery-geotab-intersection-congestion
Problem Type: Multi-target Regression
Targets: TotalTimeStopped_p20/50/80, DistanceToFirstStop_p20/50/80 (6 targets)

Competition-specific services:
- preprocess_geotab_data: Full feature engineering pipeline
- train_multi_target_lgbm: Train LightGBM for all 6 targets
- create_geotab_submission: Create stacked submission format

Key Insights from Top Solutions:
- Cyclical encoding for hours (sin/cos)
- Cardinal direction encoding (N=0, NE=1/4, etc.)
- Street type encoding (Road, Street, Avenue, etc.)
- Climate data by city-month (temperature, rainfall, snowfall, daylight)
- City-intersection feature cross
- Distance from city center
"""

import os
import sys
import json
import pickle
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract import contract

# =============================================================================
# HELPERS
# =============================================================================

def _load_data(path: str) -> pd.DataFrame:
    """Load data from CSV or Parquet."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV or Parquet."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

# Climate data mappings
MONTHLY_TEMPERATURE = {
    'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79, 'Atlanta8': 78,
    'Atlanta9': 73, 'Atlanta10': 62, 'Atlanta11': 53, 'Atlanta12': 45,
    'Boston1': 30, 'Boston5': 59, 'Boston6': 68, 'Boston7': 74, 'Boston8': 73,
    'Boston9': 66, 'Boston10': 55, 'Boston11': 45, 'Boston12': 35,
    'Chicago1': 27, 'Chicago5': 60, 'Chicago6': 70, 'Chicago7': 76, 'Chicago8': 76,
    'Chicago9': 68, 'Chicago10': 56, 'Chicago11': 45, 'Chicago12': 32,
    'Philadelphia1': 35, 'Philadelphia5': 66, 'Philadelphia6': 76, 'Philadelphia7': 81,
    'Philadelphia8': 79, 'Philadelphia9': 72, 'Philadelphia10': 60, 'Philadelphia11': 49, 'Philadelphia12': 40
}

MONTHLY_RAINFALL = {
    'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67,
    'Atlanta9': 4.09, 'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82,
    'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22, 'Boston7': 3.06, 'Boston8': 3.37,
    'Boston9': 3.47, 'Boston10': 3.79, 'Boston11': 3.98, 'Boston12': 3.73,
    'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62,
    'Chicago9': 3.27, 'Chicago10': 2.71, 'Chicago11': 3.01, 'Chicago12': 2.43,
    'Philadelphia1': 3.52, 'Philadelphia5': 3.88, 'Philadelphia6': 3.29, 'Philadelphia7': 4.39,
    'Philadelphia8': 3.82, 'Philadelphia9': 3.88, 'Philadelphia10': 2.75, 'Philadelphia11': 3.16, 'Philadelphia12': 3.31
}

MONTHLY_SNOWFALL = {
    'Atlanta1': 0.6, 'Atlanta5': 0, 'Atlanta6': 0, 'Atlanta7': 0, 'Atlanta8': 0,
    'Atlanta9': 0, 'Atlanta10': 0, 'Atlanta11': 0, 'Atlanta12': 0.2,
    'Boston1': 12.9, 'Boston5': 0, 'Boston6': 0, 'Boston7': 0, 'Boston8': 0,
    'Boston9': 0, 'Boston10': 0, 'Boston11': 1.3, 'Boston12': 9.0,
    'Chicago1': 11.5, 'Chicago5': 0, 'Chicago6': 0, 'Chicago7': 0, 'Chicago8': 0,
    'Chicago9': 0, 'Chicago10': 0, 'Chicago11': 1.3, 'Chicago12': 8.7,
    'Philadelphia1': 6.5, 'Philadelphia5': 0, 'Philadelphia6': 0, 'Philadelphia7': 0,
    'Philadelphia8': 0, 'Philadelphia9': 0, 'Philadelphia10': 0, 'Philadelphia11': 0.3, 'Philadelphia12': 3.4
}

MONTHLY_DAYLIGHT = {
    'Atlanta1': 10, 'Atlanta5': 14, 'Atlanta6': 14, 'Atlanta7': 14, 'Atlanta8': 13,
    'Atlanta9': 12, 'Atlanta10': 11, 'Atlanta11': 10, 'Atlanta12': 10,
    'Boston1': 9, 'Boston5': 15, 'Boston6': 15, 'Boston7': 15, 'Boston8': 14,
    'Boston9': 12, 'Boston10': 11, 'Boston11': 10, 'Boston12': 9,
    'Chicago1': 10, 'Chicago5': 15, 'Chicago6': 15, 'Chicago7': 15, 'Chicago8': 14,
    'Chicago9': 12, 'Chicago10': 11, 'Chicago11': 10, 'Chicago12': 9,
    'Philadelphia1': 10, 'Philadelphia5': 14, 'Philadelphia6': 15, 'Philadelphia7': 15,
    'Philadelphia8': 14, 'Philadelphia9': 12, 'Philadelphia10': 11, 'Philadelphia11': 10, 'Philadelphia12': 9
}

# Monthly sunshine hours (from top solutions)
MONTHLY_SUNSHINE = {
    'Atlanta1': 5.3, 'Atlanta5': 9.3, 'Atlanta6': 9.5, 'Atlanta7': 8.8, 'Atlanta8': 8.3,
    'Atlanta9': 7.6, 'Atlanta10': 7.7, 'Atlanta11': 6.2, 'Atlanta12': 5.3,
    'Boston1': 5.3, 'Boston5': 8.6, 'Boston6': 9.6, 'Boston7': 9.7, 'Boston8': 8.9,
    'Boston9': 7.9, 'Boston10': 6.7, 'Boston11': 4.8, 'Boston12': 4.6,
    'Chicago1': 4.4, 'Chicago5': 9.1, 'Chicago6': 10.4, 'Chicago7': 10.3, 'Chicago8': 9.1,
    'Chicago9': 7.6, 'Chicago10': 6.2, 'Chicago11': 3.6, 'Chicago12': 3.4,
    'Philadelphia1': 5.0, 'Philadelphia5': 7.9, 'Philadelphia6': 9.0, 'Philadelphia7': 8.9,
    'Philadelphia8': 8.4, 'Philadelphia9': 7.9, 'Philadelphia10': 6.6, 'Philadelphia11': 5.2, 'Philadelphia12': 4.4
}

# Direction encoding (theta/pi)
DIRECTIONS = {
    'N': 0, 'NE': 1/4, 'E': 1/2, 'SE': 3/4,
    'S': 1, 'SW': 5/4, 'W': 3/2, 'NW': 7/4
}

# Road type encoding
ROAD_ENCODING = {
    'Street': 0, 'St': 0, 'Avenue': 1, 'Ave': 1, 'Boulevard': 2,
    'Road': 3, 'Drive': 4, 'Lane': 5, 'Tunnel': 6, 'Highway': 7,
    'Way': 8, 'Parkway': 9, 'Parking': 10, 'Oval': 11, 'Square': 12,
    'Place': 13, 'Bridge': 14
}

# City centers for distance calculation
CITY_CENTERS = {
    'Atlanta': (33.753746, -84.386330),
    'Boston': (42.361145, -71.057083),
    'Chicago': (41.881832, -87.623177),
    'Philadelphia': (39.952583, -75.165222)
}

# Target columns
TARGET_COLUMNS = [
    'TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80',
    'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80'
]


def _encode_road_type(x):
    """Encode street type from street name."""
    if pd.isna(x):
        return 0
    for road, code in ROAD_ENCODING.items():
        if road in str(x):
            return code
    return 0


def _get_street_word(x, word_idx=0, freq_map=None, min_count=500):
    """Extract word from street name with frequency filtering."""
    if pd.isna(x):
        return 'Other'
    words = str(x).split()
    if len(words) > word_idx:
        word = words[word_idx]
        if freq_map is not None and freq_map.get(word, 0) <= min_count:
            return 'Other'
        return word
    return 'Other'


def _calculate_center_distance(row):
    """Calculate distance from city center."""
    city = row['City']
    if city not in CITY_CENTERS:
        return 0.0
    center_lat, center_lon = CITY_CENTERS[city]
    return math.sqrt((center_lat - row['Latitude'])**2 + (center_lon - row['Longitude'])**2)


# =============================================================================
# SERVICE 1: PREPROCESS GEOTAB DATA
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True}
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"}
    },
    description="Preprocess Geotab intersection congestion data with feature engineering",
    tags=["preprocessing", "feature-engineering", "geotab", "traffic"],
    version="1.0.0"
)
def preprocess_geotab_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    add_cyclical_hour: bool = True,
    add_climate_data: bool = True,
    add_center_distance: bool = True,
    scale_coordinates: bool = True,
    drop_unused_columns: bool = True,
    add_frequency_encoding: bool = True,
    add_aggregations: bool = True,
    add_feature_crosses: bool = True
) -> str:
    """
    Full preprocessing pipeline for Geotab intersection congestion data.

    Based on top-scoring solutions, includes:
    - Hour cyclical encoding (sin/cos)
    - Direction encoding (Entry/Exit heading as theta/pi)
    - Heading difference and same_heading flag
    - Street type encoding (multiple levels)
    - Same street flag
    - Intersection+City feature
    - Climate data (temperature, rainfall, snowfall, daylight, sunshine)
    - Distance from city center
    - City one-hot encoding
    - is_day, is_morning, is_night flags
    - Frequency encoding for categorical features
    - Group aggregations (mean, std by groups)
    - Feature crosses (Hour_Month, etc.)
    """
    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    # Store targets and RowId before processing
    targets = {}
    for col in TARGET_COLUMNS:
        if col in train.columns:
            targets[col] = train[col].copy()

    train_rowid = train['RowId'].copy() if 'RowId' in train.columns else None
    test_rowid = test['RowId'].copy() if 'RowId' in test.columns else None

    def process_dataframe(df, is_train=True):
        """Apply all preprocessing transformations to a dataframe."""
        # 1. Hour cyclical encoding
        if add_cyclical_hour and 'Hour' in df.columns:
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

        # 2. Time of day flags
        if 'Hour' in df.columns:
            df['is_day'] = ((df['Hour'] > 5) & (df['Hour'] < 20)).astype(int)
            df['is_morning'] = ((df['Hour'] > 6) & (df['Hour'] < 10)).astype(int)
            df['is_night'] = ((df['Hour'] > 17) & (df['Hour'] < 20)).astype(int)

        # 3. Direction encoding (theta/pi representation)
        if 'EntryHeading' in df.columns:
            df['EntryHeading'] = df['EntryHeading'].map(DIRECTIONS)
        if 'ExitHeading' in df.columns:
            df['ExitHeading'] = df['ExitHeading'].map(DIRECTIONS)

        # 4. Heading difference and same heading flag
        if 'EntryHeading' in df.columns and 'ExitHeading' in df.columns:
            df['diffHeading'] = df['EntryHeading'] - df['ExitHeading']
            df['same_heading'] = (df['EntryHeading'] == df['ExitHeading']).astype(int)

        # 5. Street type encoding (from top solutions)
        if 'EntryStreetName' in df.columns:
            df['EntryType'] = df['EntryStreetName'].apply(_encode_road_type)
        if 'ExitStreetName' in df.columns:
            df['ExitType'] = df['ExitStreetName'].apply(_encode_road_type)

        # 6. Same street flag
        if 'EntryStreetName' in df.columns and 'ExitStreetName' in df.columns:
            df['same_street'] = (df['EntryStreetName'] == df['ExitStreetName']).astype(int)

        # 7. Intersection + City feature
        if 'IntersectionId' in df.columns and 'City' in df.columns:
            df['Intersection'] = df['IntersectionId'].astype(str) + '_' + df['City'].astype(str)

        return df

    # NEW: Extract street name words (from forward feature selection notebook)
    def extract_street_words(train_df, test_df):
        """Extract first and second words from street names with frequency filtering."""
        for col in ['EntryStreetName', 'ExitStreetName']:
            if col not in train_df.columns:
                continue
            prefix = col.replace('StreetName', '')

            # Get word frequencies from combined data
            combined_words1 = pd.concat([
                train_df[col].apply(lambda x: _get_street_word(x, 0)),
                test_df[col].apply(lambda x: _get_street_word(x, 0))
            ])
            freq_map1 = combined_words1.value_counts().to_dict()

            combined_words2 = pd.concat([
                train_df[col].apply(lambda x: _get_street_word(x, 1)),
                test_df[col].apply(lambda x: _get_street_word(x, 1))
            ])
            freq_map2 = combined_words2.value_counts().to_dict()

            # Apply with frequency filtering
            train_df[f'{prefix}Type_1'] = train_df[col].apply(lambda x: _get_street_word(x, 0, freq_map1, 500))
            test_df[f'{prefix}Type_1'] = test_df[col].apply(lambda x: _get_street_word(x, 0, freq_map1, 500))
            train_df[f'{prefix}Type_2'] = train_df[col].apply(lambda x: _get_street_word(x, 1, freq_map2, 500))
            test_df[f'{prefix}Type_2'] = test_df[col].apply(lambda x: _get_street_word(x, 1, freq_map2, 500))

        return train_df, test_df

    # NEW: Bucketize lat/lon and create crosses (from top solutions)
    def add_geo_buckets(train_df, test_df, n_bins=30):
        """Bucketize latitude/longitude and create crosses."""
        combined = pd.concat([train_df[['Latitude', 'Longitude']], test_df[['Latitude', 'Longitude']]]).reset_index(drop=True)

        # Create bins
        lat_bins = pd.cut(combined['Latitude'], n_bins, labels=False)
        lon_bins = pd.cut(combined['Longitude'], n_bins, labels=False)

        # Assign to dataframes
        train_df['Latitude_B'] = lat_bins[:len(train_df)].values
        test_df['Latitude_B'] = lat_bins[len(train_df):].values
        train_df['Longitude_B'] = lon_bins[:len(train_df)].values
        test_df['Longitude_B'] = lon_bins[len(train_df):].values

        # Create cross
        train_df['LatLon_B'] = train_df['Latitude_B'].astype(str) + '_' + train_df['Longitude_B'].astype(str)
        test_df['LatLon_B'] = test_df['Latitude_B'].astype(str) + '_' + test_df['Longitude_B'].astype(str)

        return train_df, test_df

    # NEW: Add nunique aggregations (from forward feature selection)
    def add_nunique_aggregations(train_df, test_df):
        """Add nunique count aggregations."""
        agg_configs = [
            ('Intersection', 'Hour'),
            ('Intersection', 'Month'),
            ('LatLon_B', 'Hour'),
            ('LatLon_B', 'Month'),
        ]

        for main_col, group_col in agg_configs:
            if main_col not in train_df.columns or group_col not in train_df.columns:
                continue
            combined = pd.concat([train_df[[group_col, main_col]], test_df[[group_col, main_col]]])
            nunique_map = combined.groupby(group_col)[main_col].nunique().to_dict()
            col_name = f'{group_col}_{main_col}_nunique'
            train_df[col_name] = train_df[group_col].map(nunique_map).astype('float32')
            test_df[col_name] = test_df[group_col].map(nunique_map).astype('float32')

        return train_df, test_df

    train = process_dataframe(train, is_train=True)
    test = process_dataframe(test, is_train=False)

    # NEW: Extract street name words
    train, test = extract_street_words(train, test)

    # NEW: Add geo buckets
    train, test = add_geo_buckets(train, test, n_bins=30)

    # Label encode Intersection (needs to fit on both train and test)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    all_intersections = pd.concat([train['Intersection'], test['Intersection']]).unique()
    le.fit(all_intersections)
    train['Intersection'] = le.transform(train['Intersection'])
    test['Intersection'] = le.transform(test['Intersection'])

    # Label encode LatLon_B
    if 'LatLon_B' in train.columns:
        all_latlon = pd.concat([train['LatLon_B'], test['LatLon_B']]).unique()
        le_latlon = LabelEncoder()
        le_latlon.fit(all_latlon)
        train['LatLon_B'] = le_latlon.transform(train['LatLon_B'])
        test['LatLon_B'] = le_latlon.transform(test['LatLon_B'])

    # Label encode street word features
    for col in ['EntryType_1', 'EntryType_2', 'ExitType_1', 'ExitType_2']:
        if col in train.columns:
            all_vals = pd.concat([train[col], test[col]]).unique()
            le_col = LabelEncoder()
            le_col.fit(all_vals)
            train[col] = le_col.transform(train[col])
            test[col] = le_col.transform(test[col])

    # NEW: Add nunique aggregations before frequency encoding
    train, test = add_nunique_aggregations(train, test)

    # Add frequency encoding for categorical features (from top solutions)
    if add_frequency_encoding:
        freq_cols = ['Hour', 'Month', 'EntryType', 'ExitType', 'Intersection', 'IntersectionId',
                     'EntryType_1', 'EntryType_2', 'ExitType_1', 'ExitType_2', 'LatLon_B', 'City']
        for col in freq_cols:
            if col in train.columns and col in test.columns:
                # Compute frequency on combined data
                combined = pd.concat([train[col], test[col]])
                freq_map = combined.value_counts(normalize=True).to_dict()
                train[f'{col}_FE'] = train[col].map(freq_map).astype('float32')
                test[f'{col}_FE'] = test[col].map(freq_map).astype('float32')

    # Add feature crosses (from top solutions)
    if add_feature_crosses:
        if 'Hour' in train.columns and 'Month' in train.columns:
            train['Hour_Month'] = train['Hour'].astype(str) + '_' + train['Month'].astype(str)
            test['Hour_Month'] = test['Hour'].astype(str) + '_' + test['Month'].astype(str)
            # Label encode the cross
            all_hm = pd.concat([train['Hour_Month'], test['Hour_Month']]).unique()
            le_hm = LabelEncoder()
            le_hm.fit(all_hm)
            train['Hour_Month'] = le_hm.transform(train['Hour_Month'])
            test['Hour_Month'] = le_hm.transform(test['Hour_Month'])

        if 'Weekend' in train.columns:
            train['is_day_weekend'] = (train['is_day'] * train['Weekend']).astype(int)
            test['is_day_weekend'] = (test['is_day'] * test['Weekend']).astype(int)
            train['is_morning_weekend'] = (train['is_morning'] * train['Weekend']).astype(int)
            test['is_morning_weekend'] = (test['is_morning'] * test['Weekend']).astype(int)

    # Add group aggregations (from top solutions)
    if add_aggregations:
        agg_cols = ['Latitude', 'Longitude', 'EntryHeading', 'ExitHeading']
        group_cols = ['Intersection', 'Hour', 'Month']

        for agg_col in agg_cols:
            if agg_col not in train.columns:
                continue
            for group_col in group_cols:
                if group_col not in train.columns:
                    continue
                # Compute on combined data
                combined = pd.concat([train[[group_col, agg_col]], test[[group_col, agg_col]]])

                # Mean aggregation
                agg_mean = combined.groupby(group_col)[agg_col].mean().to_dict()
                train[f'{agg_col}_{group_col}_mean'] = train[group_col].map(agg_mean).astype('float32')
                test[f'{agg_col}_{group_col}_mean'] = test[group_col].map(agg_mean).astype('float32')

                # Std aggregation
                agg_std = combined.groupby(group_col)[agg_col].std().fillna(0).to_dict()
                train[f'{agg_col}_{group_col}_std'] = train[group_col].map(agg_std).astype('float32')
                test[f'{agg_col}_{group_col}_std'] = test[group_col].map(agg_std).astype('float32')

    def add_additional_features(df):
        """Add climate, distance, one-hot features."""
        # Climate data including sunshine (from top solutions)
        if add_climate_data and 'City' in df.columns and 'Month' in df.columns:
            df['city_month'] = df['City'] + df['Month'].astype(str)
            df['average_temp'] = df['city_month'].map(MONTHLY_TEMPERATURE)
            df['average_rainfall'] = df['city_month'].map(MONTHLY_RAINFALL)
            df['average_snowfall'] = df['city_month'].map(MONTHLY_SNOWFALL)
            df['average_daylight'] = df['city_month'].map(MONTHLY_DAYLIGHT)
            df['average_sunshine'] = df['city_month'].map(MONTHLY_SUNSHINE)
            df = df.drop('city_month', axis=1)

        # Distance from city center
        if add_center_distance and 'City' in df.columns and 'Latitude' in df.columns:
            df['CenterDistance'] = df.apply(_calculate_center_distance, axis=1)

        # City one-hot encoding
        if 'City' in df.columns:
            city_dummies = pd.get_dummies(df['City'], prefix='City')
            df = pd.concat([df, city_dummies], axis=1)

        return df

    train = add_additional_features(train)
    test = add_additional_features(test)

    # Scale coordinates using combined statistics (proper way to avoid leakage)
    if scale_coordinates:
        from sklearn.preprocessing import StandardScaler
        for col in ['Latitude', 'Longitude']:
            if col in train.columns and col in test.columns:
                scaler = StandardScaler()
                # Fit on combined data
                combined = pd.concat([train[[col]], test[[col]]])
                scaler.fit(combined)
                train[col] = scaler.transform(train[[col]])
                test[col] = scaler.transform(test[[col]])

    # Drop unused columns
    if drop_unused_columns:
        drop_cols = ['Path', 'EntryStreetName', 'ExitStreetName', 'City']
        for col in drop_cols:
            if col in train.columns:
                train = train.drop(col, axis=1)
            if col in test.columns:
                test = test.drop(col, axis=1)

    # Add targets back to train
    for col, values in targets.items():
        train[col] = values

    # Ensure RowId is preserved
    if train_rowid is not None and 'RowId' not in train.columns:
        train['RowId'] = train_rowid.values
    if test_rowid is not None and 'RowId' not in test.columns:
        test['RowId'] = test_rowid.values

    _save_data(train, outputs["train_data"])
    _save_data(test, outputs["test_data"])

    return f"preprocess_geotab_data: train={len(train)} rows with {len(train.columns)} features, test={len(test)} rows"


# =============================================================================
# SERVICE 2: TRAIN MULTI-TARGET LIGHTGBM
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True}
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"}
    },
    description="Train LightGBM models for all 6 Geotab targets",
    tags=["modeling", "training", "lightgbm", "multi-target", "regression"],
    version="1.0.0"
)
def train_multi_target_lgbm(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_columns: List[str] = None,
    id_column: str = "RowId",
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    num_leaves: int = 230,
    max_depth: int = 30,
    min_child_samples: int = 50,
    subsample: float = 0.7,
    colsample_bytree: float = 0.9,
    reg_alpha: float = 0.0,
    reg_lambda: float = 5.0,
    n_folds: int = 5,
    random_state: int = 42,
    early_stopping_rounds: int = 100
) -> str:
    """
    Train LightGBM models for all 6 target columns using K-Fold CV.

    Based on top solution approaches:
    - Trains separate model for each target
    - Uses K-Fold cross-validation
    - Returns ensemble of models
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM required. Install with: pip install lightgbm")

    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error

    train = _load_data(inputs["train_data"])

    if target_columns is None:
        target_columns = TARGET_COLUMNS

    # Identify feature columns
    exclude_cols = set(target_columns + [id_column])
    # Also exclude any other target-like columns
    for col in train.columns:
        if 'TotalTimeStopped' in col or 'TimeFromFirstStop' in col or 'DistanceToFirstStop' in col:
            exclude_cols.add(col)

    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X = train[feature_cols]

    models = {}
    metrics = {'model_type': 'LightGBM_MultiTarget', 'targets': {}}

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'verbose': -1,
        'seed': random_state
    }

    # Identify categorical features
    cat_cols = ['IntersectionId', 'Hour', 'Weekend', 'Month', 'Intersection',
                'is_day', 'is_morning', 'is_night', 'same_street', 'EntryType', 'ExitType']
    cat_features = [c for c in cat_cols if c in feature_cols]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for target_idx, target_col in enumerate(target_columns):
        if target_col not in train.columns:
            print(f"Target {target_col} not in data, skipping...")
            continue

        y = train[target_col]

        oof_preds = np.zeros(len(train))
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            )

            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            fold_models.append(model)

        cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))

        models[target_col] = fold_models
        metrics['targets'][target_col] = {
            'cv_rmse': float(cv_rmse),
            'n_folds': n_folds
        }

        print(f"Target {target_idx+1}/6 ({target_col}): CV RMSE = {cv_rmse:.4f}")

    # Save models and metrics
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)

    with open(outputs["model"], 'wb') as f:
        pickle.dump({'models': models, 'feature_cols': feature_cols, 'params': params}, f)

    with open(outputs["metrics"], 'w') as f:
        json.dump(metrics, f, indent=2)

    avg_rmse = np.mean([m['cv_rmse'] for m in metrics['targets'].values()])
    return f"train_multi_target_lgbm: 6 targets, avg CV RMSE={avg_rmse:.4f}"


# =============================================================================
# SERVICE 3: PREDICT AND CREATE SUBMISSION
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_data": {"format": "csv", "required": True},
        "sample_submission": {"format": "csv", "required": False}
    },
    outputs={
        "submission": {"format": "csv"}
    },
    description="Generate predictions and create Geotab submission file",
    tags=["inference", "prediction", "submission", "geotab"],
    version="1.0.0"
)
def create_geotab_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "RowId",
    target_mapping: Dict[int, str] = None
) -> str:
    """
    Generate predictions for all 6 targets and create stacked submission.

    Submission format: TargetId,Target
    where TargetId = "{RowId}_{target_index}"

    Target indices (0-5) map to:
    0: TotalTimeStopped_p20
    1: TotalTimeStopped_p50
    2: TotalTimeStopped_p80
    3: DistanceToFirstStop_p20
    4: DistanceToFirstStop_p50
    5: DistanceToFirstStop_p80
    """
    with open(inputs["model"], 'rb') as f:
        model_data = pickle.load(f)

    models = model_data['models']
    feature_cols = model_data['feature_cols']

    test = _load_data(inputs["test_data"])

    if target_mapping is None:
        target_mapping = {
            0: 'TotalTimeStopped_p20',
            1: 'TotalTimeStopped_p50',
            2: 'TotalTimeStopped_p80',
            3: 'DistanceToFirstStop_p20',
            4: 'DistanceToFirstStop_p50',
            5: 'DistanceToFirstStop_p80'
        }

    X_test = test[feature_cols]
    row_ids = test[id_column].values if id_column in test.columns else np.arange(len(test))

    # Generate predictions for each target
    all_preds = {}
    for target_idx, target_col in target_mapping.items():
        if target_col not in models:
            print(f"Warning: No model for {target_col}")
            all_preds[target_idx] = np.zeros(len(test))
            continue

        # Average predictions from all folds
        fold_preds = np.zeros(len(test))
        for model in models[target_col]:
            fold_preds += model.predict(X_test, num_iteration=model.best_iteration)
        all_preds[target_idx] = fold_preds / len(models[target_col])

    # Check if sample submission is provided to get valid RowIds
    valid_row_ids = None
    if "sample_submission" in inputs and inputs["sample_submission"]:
        sample_sub = _load_data(inputs["sample_submission"])
        # Extract unique RowIds from sample submission
        sample_sub['_RowId'] = sample_sub['TargetId'].apply(lambda x: int(x.split('_')[0]))
        valid_row_ids = set(sample_sub['_RowId'].unique())
        print(f"Using {len(valid_row_ids)} RowIds from sample submission")

    # Create stacked submission
    submission_data = []
    for i, row_id in enumerate(row_ids):
        # Skip RowIds not in sample submission
        if valid_row_ids is not None and int(row_id) not in valid_row_ids:
            continue
        for target_idx in sorted(target_mapping.keys()):
            target_id = f"{int(row_id)}_{target_idx}"
            target_value = all_preds[target_idx][i]
            submission_data.append({'TargetId': target_id, 'Target': target_value})

    submission = pd.DataFrame(submission_data)

    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)

    n_rows = len(valid_row_ids) if valid_row_ids else len(test)
    return f"create_geotab_submission: {len(submission)} predictions for {n_rows} test samples"


# =============================================================================
# SERVICE 4: SPLIT DATA
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True}
    },
    outputs={
        "train_data": {"format": "csv"},
        "valid_data": {"format": "csv"}
    },
    description="Split data into train and validation sets",
    tags=["preprocessing", "data-splitting", "generic"],
    version="1.0.0"
)
def split_geotab_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    test_size: float = 0.1,
    random_state: int = 42
) -> str:
    """Split data for validation."""
    from sklearn.model_selection import train_test_split

    df = _load_data(inputs["data"])

    train, valid = train_test_split(df, test_size=test_size, random_state=random_state)

    _save_data(train, outputs["train_data"])
    _save_data(valid, outputs["valid_data"])

    return f"split_geotab_data: train={len(train)}, valid={len(valid)}"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "preprocess_geotab_data": preprocess_geotab_data,
    "train_multi_target_lgbm": train_multi_target_lgbm,
    "create_geotab_submission": create_geotab_submission,
}
