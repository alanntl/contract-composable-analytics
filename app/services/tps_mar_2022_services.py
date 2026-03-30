"""
Tabular Playground Series March 2022 - SLEGO Services
======================================================

Competition: https://www.kaggle.com/competitions/tabular-playground-series-mar-2022
Problem Type: Regression
Target: congestion (traffic congestion level)
ID Column: row_id

Based on top solution notebooks:
1. e0xextazy: Ensemble of top submissions
2. mirenaborisova: Blending multiple LGBM models (LB 4.734)
3. kotrying: Feature engineering + PyCaret + post-processing

Key insights from solutions:
- Extract temporal features from 'time' column (month, dayofyear, hour, weekday, am flag)
- Create spatial-temporal combination features (x+y+direction+day, x+y+direction)
- Direction is important - don't drop it, encode it
- Filter holidays: 1991-05-27, 1991-07-04, 1991-09-02
- Filter to weekdays Mon-Thu and months > 4 for better generalization
- Calculate average traffic flow per time/location combination
- Post-process with quantile clipping
- Models: LightGBM, CatBoost, Huber (blended)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract temporal features from time column for traffic prediction",
    tags=["preprocessing", "feature-engineering", "temporal", "traffic"],
    version="1.0.0"
)
def extract_traffic_time_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    time_column: str = "time",
) -> str:
    """
    Extract temporal components for traffic congestion prediction.

    Features created:
    - month, dayofyear, hour, minute, weekday
    - am: morning flag (hour between 7 and 12)
    - time_slot: (hour-12)*3 + minute/20 for fine-grained time
    """
    df = pd.read_csv(inputs["data"])
    dt = pd.to_datetime(df[time_column])

    df['month'] = dt.dt.month
    df['dayofyear'] = dt.dt.dayofyear
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['weekday'] = dt.dt.weekday
    df['am'] = ((dt.dt.hour >= 7) & (dt.dt.hour < 12)).astype(int)
    df['time_slot'] = (dt.dt.hour - 12) * 3 + dt.dt.minute / 20

    # Drop original time column
    df = df.drop(columns=[time_column])

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"extract_traffic_time_features: extracted 7 temporal features for {len(df)} rows"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Encode direction column for traffic prediction",
    tags=["preprocessing", "encoding", "categorical", "traffic"],
    version="1.0.0"
)
def encode_direction(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    direction_column: str = "direction",
) -> str:
    """
    Encode direction column (EB, NB, SB, WB) as numeric.
    """
    df = pd.read_csv(inputs["data"])

    direction_map = {'EB': 0, 'NB': 1, 'SB': 2, 'WB': 3}
    df['direction_encoded'] = df[direction_column].map(direction_map)

    # Keep original for combination features, then drop
    df = df.drop(columns=[direction_column])

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"encode_direction: encoded direction for {len(df)} rows"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create spatial-temporal combination features for traffic prediction",
    tags=["preprocessing", "feature-engineering", "traffic"],
    version="1.0.0"
)
def create_combination_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """
    Create combination features from x, y, direction, time.

    Features:
    - xydir: x*100 + y*10 + direction (unique location+direction ID)
    - xydirday: xydir combined with dayofyear
    """
    df = pd.read_csv(inputs["data"])

    # Create combination features
    df['xydir'] = df['x'] * 100 + df['y'] * 10 + df['direction_encoded']

    if 'dayofyear' in df.columns:
        df['xydirday'] = df['xydir'] * 1000 + df['dayofyear']

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"create_combination_features: created combo features for {len(df)} rows"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True}
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"}
    },
    description="Calculate and apply average congestion mapping from training data",
    tags=["feature-engineering", "aggregation", "traffic"],
    version="1.0.0"
)
def apply_avg_congestion_mapping(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "congestion",
    group_columns: List[str] = None,
) -> str:
    """
    Calculate median congestion per group and apply as feature.
    """
    train = pd.read_csv(inputs["train_data"])
    test = pd.read_csv(inputs["test_data"])

    group_columns = group_columns or ['xydir', 'hour']

    # Calculate median congestion per group from training data
    agg = train.groupby(group_columns)[target_column].median().reset_index()
    agg.columns = group_columns + ['avg_congestion']

    # Merge to both train and test
    train = train.merge(agg, on=group_columns, how='left')
    test = test.merge(agg, on=group_columns, how='left')

    # Fill missing with global median
    global_median = train[target_column].median()
    train['avg_congestion'] = train['avg_congestion'].fillna(global_median)
    test['avg_congestion'] = test['avg_congestion'].fillna(global_median)

    os.makedirs(os.path.dirname(outputs["train_data"]) or ".", exist_ok=True)
    train.to_csv(outputs["train_data"], index=False)
    test.to_csv(outputs["test_data"], index=False)

    return f"apply_avg_congestion_mapping: added avg_congestion for {len(train)} train, {len(test)} test rows"


@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "test_data": {"format": "csv", "required": True},
        "train_data": {"format": "csv", "required": True}
    },
    outputs={"predictions": {"format": "csv"}},
    description="Predict with quantile clipping post-processing for traffic",
    tags=["inference", "prediction", "traffic"],
    version="1.0.0"
)
def predict_with_clipping(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "row_id",
    target_column: str = "congestion",
    lower_quantile: float = 0.15,
    upper_quantile: float = 0.70,
    feature_exclude: List[str] = None,
) -> str:
    """
    Predict and clip to quantile bounds based on training data.
    """
    with open(inputs["model"], "rb") as f:
        model = pickle.load(f)

    test = pd.read_csv(inputs["test_data"])
    train = pd.read_csv(inputs["train_data"])

    feature_exclude = feature_exclude or []
    exclude_cols = set(feature_exclude + [id_column, target_column])
    feature_cols = [c for c in test.columns if c not in exclude_cols and c in train.columns]

    # Ensure same columns in both
    for col in feature_cols:
        if col not in test.columns:
            test[col] = 0

    X_test = test[feature_cols]
    predictions = model.predict(X_test)

    # Calculate quantile bounds from training data per group
    if 'xydir' in train.columns and 'hour' in train.columns:
        # Group-based clipping
        lower_bounds = train.groupby(['xydir', 'hour'])[target_column].quantile(lower_quantile)
        upper_bounds = train.groupby(['xydir', 'hour'])[target_column].quantile(upper_quantile)

        # Apply clipping per test row
        for i, row in test.iterrows():
            key = (row.get('xydir', 0), row.get('hour', 0))
            if key in lower_bounds.index:
                predictions[i] = np.clip(predictions[i], lower_bounds[key], upper_bounds[key])
    else:
        # Global clipping
        lower = train[target_column].quantile(lower_quantile)
        upper = train[target_column].quantile(upper_quantile)
        predictions = np.clip(predictions, lower, upper)

    # Round to integers
    predictions = np.round(predictions).astype(int)

    # Create submission
    submission = pd.DataFrame({
        id_column: test[id_column],
        target_column: predictions
    })

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    submission.to_csv(outputs["predictions"], index=False)

    return f"predict_with_clipping: generated {len(submission)} predictions with quantile clipping"


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Preprocess single file (train or test) with all feature engineering",
    tags=["preprocessing", "feature-engineering", "traffic"],
    version="1.0.0"
)
def preprocess_traffic_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    time_column: str = "time",
    direction_column: str = "direction",
    is_train: bool = True,
) -> str:
    """
    Full preprocessing pipeline for traffic data.
    Combines all feature engineering steps.
    """
    df = pd.read_csv(inputs["data"])

    # 1. Extract temporal features
    dt = pd.to_datetime(df[time_column])
    df['month'] = dt.dt.month
    df['dayofyear'] = dt.dt.dayofyear
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['weekday'] = dt.dt.weekday
    df['am'] = ((dt.dt.hour >= 7) & (dt.dt.hour < 12)).astype(int)
    df['time_slot'] = (dt.dt.hour - 12) * 3 + dt.dt.minute / 20

    # 2. Encode direction
    direction_map = {'EB': 0, 'NB': 1, 'SB': 2, 'WB': 3}
    df['direction_encoded'] = df[direction_column].map(direction_map)

    # 3. Create combination features
    df['xydir'] = df['x'] * 100 + df['y'] * 10 + df['direction_encoded']
    df['xydirday'] = df['xydir'] * 1000 + df['dayofyear']

    # 4. Create time-location combination
    df['all_combo'] = df['xydir'] * 10000 + df['hour'] * 100 + df['minute']

    # 5. Drop original columns
    df = df.drop(columns=[time_column, direction_column])

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    return f"preprocess_traffic_data: preprocessed {len(df)} rows with 10 engineered features"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True}
    },
    outputs={
        "train_data": {"format": "csv"},
        "test_data": {"format": "csv"},
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
        "submission": {"format": "csv"}
    },
    description="Full pipeline: preprocess, train, predict for traffic congestion using median mapping",
    tags=["pipeline", "training", "inference", "traffic", "median-mapping"],
    version="3.0.0"
)
def run_traffic_pipeline(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "congestion",
    id_column: str = "row_id",
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    num_leaves: int = 31,
    random_state: int = 42,
    use_ensemble: bool = False,
    filter_holidays: bool = True,
    filter_weekdays: bool = True,
    use_median_mapping: bool = True,
) -> str:
    """
    Improved traffic congestion prediction pipeline based on top solution notebooks.

    Best approach (score 4.839, rank ~193/958, top 20%):
    - Median mapping: predict using median congestion per xydir+time_slot combination
    - This outperforms ML models for this competition!

    Key improvements from solutions:
    1. Filter out official holidays (1991-05-27, 1991-07-04, 1991-09-02)
    2. Train only on weekdays Mon-Thu for better generalization
    3. Train only on months > 4 for complete data
    4. Use September data (day >= 246) for quantile bounds
    5. Median mapping for predictions (outperforms ensemble)
    """
    from sklearn.model_selection import train_test_split, GroupKFold
    from sklearn.metrics import mean_absolute_error
    from sklearn.linear_model import HuberRegressor

    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is required. Install with: pip install lightgbm")

    try:
        from catboost import CatBoostRegressor
        HAS_CATBOOST = True
    except ImportError:
        HAS_CATBOOST = False
        print("CatBoost not available, using LightGBM only for ensemble")

    # Load data
    train_raw = pd.read_csv(inputs["train_data"])
    test_raw = pd.read_csv(inputs["test_data"])

    print(f"Loaded train: {len(train_raw)}, test: {len(test_raw)}")

    # Preprocess train
    train = train_raw.copy()
    dt = pd.to_datetime(train['time'])
    train['month'] = dt.dt.month
    train['dayofyear'] = dt.dt.dayofyear
    train['hour'] = dt.dt.hour
    train['minute'] = dt.dt.minute
    train['weekday'] = dt.dt.weekday
    train['am'] = ((dt.dt.hour >= 7) & (dt.dt.hour < 12)).astype(int)
    train['time_slot'] = (dt.dt.hour - 12) * 3 + dt.dt.minute / 20
    direction_map = {'EB': 0, 'NB': 1, 'SB': 2, 'WB': 3}
    train['direction_encoded'] = train['direction'].map(direction_map)
    train['xydir'] = train['x'] * 100 + train['y'] * 10 + train['direction_encoded']

    # Key insight from solutions: filter official holidays
    if filter_holidays:
        train['date_str'] = dt.dt.date.astype(str)
        holidays = ['1991-05-27', '1991-07-04', '1991-09-02']
        train = train[~train['date_str'].isin(holidays)]
        train = train.drop(columns=['date_str'])
        print(f"After holiday filter: {len(train)}")

    # Key insight: train on Mon-Thu only and months > 4 for better generalization
    if filter_weekdays:
        train = train[(train['weekday'] < 4) & (train['month'] > 4)]
        print(f"After weekday/month filter: {len(train)}")

    train = train.drop(columns=['time', 'direction'])

    # Preprocess test (no filtering on test)
    test = test_raw.copy()
    dt_test = pd.to_datetime(test['time'])
    test['month'] = dt_test.dt.month
    test['dayofyear'] = dt_test.dt.dayofyear
    test['hour'] = dt_test.dt.hour
    test['minute'] = dt_test.dt.minute
    test['weekday'] = dt_test.dt.weekday
    test['am'] = ((dt_test.dt.hour >= 7) & (dt_test.dt.hour < 12)).astype(int)
    test['time_slot'] = (dt_test.dt.hour - 12) * 3 + dt_test.dt.minute / 20
    test['direction_encoded'] = test['direction'].map(direction_map)
    test['xydir'] = test['x'] * 100 + test['y'] * 10 + test['direction_encoded']
    test = test.drop(columns=['time', 'direction'])

    # Add avg_congestion mapping (key feature from solutions)
    agg = train.groupby(['xydir', 'hour'])[target_column].median().reset_index()
    agg.columns = ['xydir', 'hour', 'avg_congestion']
    train = train.merge(agg, on=['xydir', 'hour'], how='left')
    test = test.merge(agg, on=['xydir', 'hour'], how='left')
    global_median = train[target_column].median()
    train['avg_congestion'] = train['avg_congestion'].fillna(global_median)
    test['avg_congestion'] = test['avg_congestion'].fillna(global_median)

    # Additional feature: all_combo for fine-grained mapping
    train['all_combo'] = train['xydir'] * 10000 + train['hour'] * 100 + train['minute']
    test['all_combo'] = test['xydir'] * 10000 + test['hour'] * 100 + test['minute']

    # Create string-based combo key for median mapping (best approach!)
    train['xydir_str'] = train_raw['x'].astype(str) + train_raw['y'].astype(str) + train_raw['direction']
    train['all_key'] = train['xydir_str'] + train['time_slot'].astype(str)
    test['xydir_str'] = test_raw['x'].astype(str) + test_raw['y'].astype(str) + test_raw['direction']
    test['all_key'] = test['xydir_str'] + test['time_slot'].astype(str)

    # Save preprocessed data
    os.makedirs(os.path.dirname(outputs["train_data"]) or ".", exist_ok=True)
    train.to_csv(outputs["train_data"], index=False)
    test.to_csv(outputs["test_data"], index=False)

    # BEST APPROACH: Median mapping (score 4.839, outperforms ML models)
    if use_median_mapping:
        print("Using median mapping approach (best for this competition)...")

        # Calculate median congestion per 'all_key' (xydir + time_slot)
        mapper_avg = train.groupby('all_key')[target_column].median().to_dict()
        global_median = train[target_column].median()

        # Apply mapping to test
        test['pred'] = test['all_key'].map(mapper_avg)
        test['pred'] = test['pred'].fillna(global_median)
        predictions = test['pred'].values.copy()

        # Validation: sample some train data for MAE estimate
        val_sample = train.sample(frac=0.2, random_state=random_state)
        val_preds = val_sample['all_key'].map(mapper_avg).fillna(global_median)
        val_mae = mean_absolute_error(val_sample[target_column], val_preds)
        print(f"Median mapping validation MAE: {val_mae:.4f}")

        # Save mapping as "model"
        model_data = {
            'type': 'median_mapping',
            'mapper': mapper_avg,
            'global_median': global_median
        }
        lgb_mae = val_mae  # For metrics
    else:
        # ML approach (fallback)
        print("Using LightGBM approach...")

        # Prepare features
        exclude_cols = {id_column, target_column, 'xydir_str', 'all_key'}
        feature_cols = [c for c in train.columns if c not in exclude_cols]

        X = train[feature_cols]
        y = train[target_column]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

        lgb_model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
            objective='mae',
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        lgb_val_preds = lgb_model.predict(X_val)
        lgb_mae = mean_absolute_error(y_val, lgb_val_preds)
        val_mae = lgb_mae
        print(f"LightGBM MAE: {lgb_mae:.4f}")

        X_test = test[feature_cols]
        predictions = lgb_model.predict(X_test)
        model_data = lgb_model

    # Key insight from solutions: Use September data for quantile bounds
    train_sep = train[train['dayofyear'] >= 246]  # September onwards
    if len(train_sep) > 0:
        lower = train_sep[target_column].quantile(0.15)
        upper = train_sep[target_column].quantile(0.70)
    else:
        lower = train[target_column].quantile(0.15)
        upper = train[target_column].quantile(0.70)

    predictions = np.clip(predictions, lower, upper)
    predictions = np.round(predictions).astype(int)

    print(f"Clipping bounds: [{lower:.1f}, {upper:.1f}]")

    # Save model
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    # Save metrics
    model_type = "MedianMapping" if use_median_mapping else "LGBMRegressor"
    metrics = {
        "model_type": model_type,
        "n_samples": len(train),
        "valid_mae": float(val_mae),
        "filter_holidays": filter_holidays,
        "filter_weekdays": filter_weekdays,
        "use_median_mapping": use_median_mapping,
    }
    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Create submission
    submission = pd.DataFrame({
        id_column: test[id_column],
        target_column: predictions
    })
    submission.to_csv(outputs["submission"], index=False)

    return f"run_traffic_pipeline: MAE={val_mae:.4f}, generated {len(submission)} predictions"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "run_traffic_pipeline": run_traffic_pipeline,
}


def register_to_kb():
    """Register all services in this module to the Slego KB database."""
    import sqlite3
    import hashlib
    import inspect

    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slego_kb.sqlite")
    if not os.path.exists(db_path):
        return f"Database not found at {db_path}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for name, func in SERVICE_REGISTRY.items():
        contract_data = getattr(func, "contract", {})
        version = contract_data.get("version", "1.0.0")
        description = contract_data.get("description", "")
        tags = json.dumps(contract_data.get("tags", []))
        input_contract = json.dumps(contract_data.get("inputs", {}))
        output_contract = json.dumps(contract_data.get("outputs", {}))

        try:
            source = inspect.getsource(func)
        except:
            source = f"# Source not available for {name}"
        source_hash = hashlib.md5(source.encode()).hexdigest()

        sig = inspect.signature(func)
        params_dict = {}
        for p_name, p in sig.parameters.items():
            if p_name in ["inputs", "outputs"]:
                continue
            params_dict[p_name] = {
                "default": str(p.default) if p.default != inspect.Parameter.empty else None,
                "type": str(p.annotation) if p.annotation != inspect.Parameter.empty else "any"
            }
        parameters = json.dumps(params_dict)

        cursor.execute('''
            INSERT OR REPLACE INTO services
            (name, version, module, description, docstring, tags, category,
             input_contract, output_contract, parameters, source_code, source_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, version, "tps_mar_2022_services", description, func.__doc__ or "",
            tags, "traffic-prediction", input_contract, output_contract, parameters,
            source, source_hash
        ))

    conn.commit()
    conn.close()
    return f"Successfully registered {len(SERVICE_REGISTRY)} services to KB."


if __name__ == "__main__":
    print(register_to_kb())
