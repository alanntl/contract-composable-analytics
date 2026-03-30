"""
Tabular Playground Series September 2022 - Contract-Composable Analytics Services
==========================================================

Competition: https://www.kaggle.com/competitions/tabular-playground-series-sep-2022
Problem Type: Regression (SMAPE metric)
Target: num_sold (daily product sales count)

Key Insight from Solution Notebooks:
- Aggregate sales by date first, then disaggregate using historical ratios
- Use datetime features (month, day, dayofweek), holidays, important dates
- Ridge/Lasso linear models outperform tree-based models
- Product ratios and store/country weights are key to disaggregation

Services:
- aggregate_sales_by_date: Aggregate raw sales to daily totals
- engineer_tps_features: Create datetime, holiday, and cyclical features
- train_ridge_regressor: Train Ridge regression with GroupKFold
- compute_disaggregation_ratios: Calculate product/store/country ratios
- disaggregate_predictions: Split daily totals back to product/store/country level
- create_tps_submission: Format final submission
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# SERVICE 1: AGGREGATE SALES BY DATE
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Aggregate sales data by date for total daily sales prediction",
    tags=["preprocessing", "aggregation", "time-series", "generic"],
    version="1.0.0"
)
def aggregate_sales_by_date(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "date",
    target_column: str = "num_sold",
    exclude_covid: bool = True,
    covid_start: str = "2020-03-01",
    covid_end: str = "2020-06-01",
) -> str:
    """
    Aggregate sales by date for total daily sales prediction.

    The key insight from solution notebooks: predict aggregate daily sales first,
    then disaggregate using historical ratios.

    Parameters:
        date_column: Column containing dates
        target_column: Column containing sales to aggregate
        exclude_covid: Whether to exclude COVID period (high variance)
        covid_start: Start of COVID exclusion period
        covid_end: End of COVID exclusion period
    """
    df = _load_data(inputs["data"])
    df[date_column] = pd.to_datetime(df[date_column])

    # Aggregate by date
    daily = df.groupby(date_column)[target_column].sum().reset_index()

    # Optionally exclude COVID period
    if exclude_covid:
        mask = ~((daily[date_column] >= covid_start) & (daily[date_column] < covid_end))
        daily = daily[mask]

    _save_data(daily, outputs["data"])
    return f"aggregate_sales_by_date: {len(df)} rows -> {len(daily)} daily records"


# =============================================================================
# SERVICE 2: ENGINEER TPS FEATURES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Engineer datetime, holiday, and cyclical features for TPS Sep 2022",
    tags=["feature-engineering", "temporal", "holidays", "generic"],
    version="1.0.0"
)
def engineer_tps_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "date",
    target_column: str = "num_sold",
    log_transform_target: bool = True,
    add_holidays: bool = True,
    add_cyclical: bool = True,
    add_important_dates: bool = True,
) -> str:
    """
    Engineer datetime features based on top Kaggle solutions.

    Features include:
    - Basic datetime: year, month, day, dayofweek, day_of_year
    - Cyclical: month_sin, month_cos, day_sin
    - Weekend flags: friday, saturday, sunday
    - Holiday indicators from multiple European countries
    - Important dates identified from data patterns

    Parameters:
        date_column: Column containing dates
        target_column: Target column (will be log transformed if specified)
        log_transform_target: Apply log transform to target
        add_holidays: Add holiday features
        add_cyclical: Add cyclical encodings
        add_important_dates: Add important date flags
    """
    df = _load_data(inputs["data"])
    df[date_column] = pd.to_datetime(df[date_column])

    # Basic datetime features
    df['year'] = df[date_column].dt.year - 2016  # Normalize year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['day_of_year'] = df[date_column].dt.dayofyear

    # Adjust day_of_year for leap year (2020)
    df['day_of_year'] = df.apply(
        lambda x: x['day_of_year'] - 1
        if (x[date_column] > pd.Timestamp("2020-02-29") and x[date_column] < pd.Timestamp("2021-01-01"))
        else x['day_of_year'], axis=1
    )

    # Weekend flags
    df['friday'] = (df[date_column].dt.weekday == 4).astype(int)
    df['saturday'] = (df[date_column].dt.weekday == 5).astype(int)
    df['sunday'] = (df[date_column].dt.weekday == 6).astype(int)

    # Cyclical encodings
    if add_cyclical:
        df['month_sin'] = np.sin(df['month'] * np.pi / 24)
        df['month_cos'] = np.cos(df['month'] * np.pi / 24)
        df['day_sin'] = np.sin(df['day'] * np.pi / 62)

    # Important dates from solution notebook analysis
    if add_important_dates:
        important_dates = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17,
            124, 125, 126, 127, 140, 141, 142,
            167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181,
            203, 230, 231, 232, 233, 234, 282, 289, 290,
            307, 308, 309, 310, 311, 312, 313, 317, 318, 319, 320,
            360, 361, 362, 363, 364, 365
        ]
        df['is_important_date'] = df['day_of_year'].isin(important_dates).astype(int)

    # Holiday features
    if add_holidays:
        try:
            import holidays
            years = [2017, 2018, 2019, 2020, 2021]

            # Combine holidays from multiple European countries
            holiday_dict = {}
            for country in ['BE', 'FR', 'DE', 'IT', 'PL', 'ES']:
                country_holidays = holidays.CountryHoliday(country, years=years)
                holiday_dict.update(country_holidays)

            df['is_holiday'] = df[date_column].dt.date.map(
                lambda x: 1 if x in holiday_dict else 0
            )
        except ImportError:
            df['is_holiday'] = 0

    # Easter features (important for retail)
    try:
        import dateutil.easter as easter
        easter_dates = df[date_column].apply(lambda d: pd.Timestamp(easter.easter(d.year)))
        for day in list(range(-5, 5)) + list(range(40, 48)):
            df[f'easter_{day}'] = ((df[date_column] - easter_dates).dt.days == day).astype(int)
    except ImportError:
        pass

    # December special days
    for day in range(24, 32):
        col_name = f'dec_{day}'
        df[col_name] = ((df[date_column].dt.day == day) & (df[date_column].dt.month == 12)).astype(int)

    # Log transform target
    if log_transform_target and target_column in df.columns:
        df[target_column] = np.log(df[target_column])

    # Drop original date column (not needed for modeling)
    df = df.drop(columns=[date_column])

    _save_data(df, outputs["data"])
    return f"engineer_tps_features: created {len(df.columns)} features for {len(df)} rows"


# =============================================================================
# SERVICE 3: TRAIN RIDGE REGRESSOR
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "valid_data": {"format": "csv", "required": False},
    },
    outputs={
        "model": {"format": "pickle"},
        "metrics": {"format": "json"},
    },
    description="Train Ridge regression with GroupKFold by year",
    tags=["modeling", "training", "ridge", "regression", "generic"],
    version="1.0.0"
)
def train_ridge_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "num_sold",
    group_column: str = "year",
    alpha: float = 0.1,
    n_splits: int = 4,
    random_state: int = 42,
) -> str:
    """
    Train Ridge regression with GroupKFold cross-validation by year.

    Based on top solution insight: Ridge/Lasso perform better than tree-based
    models for this time series forecasting problem.

    Parameters:
        target_column: Target column name
        group_column: Column to use for GroupKFold (typically year)
        alpha: Ridge regularization strength
        n_splits: Number of CV folds
        random_state: Random seed
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    train = _load_data(inputs["train_data"])

    # Prepare features and target
    exclude_cols = {target_column}
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X = train[feature_cols]
    y = train[target_column]

    # Use year for GroupKFold if available
    if group_column in X.columns:
        groups = X[group_column]
        X = X.drop(columns=[group_column])
        feature_cols = [c for c in feature_cols if c != group_column]
    else:
        groups = np.zeros(len(X))

    # Cross-validation with GroupKFold
    kf = GroupKFold(n_splits=n_splits)
    cv_scores = []

    for train_idx, val_idx in kf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # Calculate SMAPE
        smape = np.mean(np.abs(y_pred - y_val) / (np.abs(y_pred) + np.abs(y_val))) * 200
        cv_scores.append(smape)

    # Train final model on all data
    final_model = Ridge(alpha=alpha, random_state=random_state)
    final_model.fit(X, y)

    metrics = {
        "model_type": "Ridge",
        "alpha": alpha,
        "cv_smape_mean": float(np.mean(cv_scores)),
        "cv_smape_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores,
        "n_samples": len(X),
        "n_features": len(feature_cols),
    }

    # Save model artifact
    model_data = {
        "model": final_model,
        "feature_cols": feature_cols,
        "target_column": target_column,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(model_data, f)

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"train_ridge_regressor: CV SMAPE = {metrics['cv_smape_mean']:.4f} (+/- {metrics['cv_smape_std']:.4f})"


# =============================================================================
# SERVICE 4: PREDICT DAILY TOTALS
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Predict daily total sales from trained model",
    tags=["inference", "prediction", "regression", "generic"],
    version="1.0.0"
)
def predict_daily_totals(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "date",
    exp_transform: bool = True,
) -> str:
    """
    Predict daily total sales.

    Parameters:
        date_column: Original date column to preserve
        exp_transform: Apply exp to reverse log transform
    """
    with open(inputs["model"], "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    feature_cols = model_data["feature_cols"]

    df = _load_data(inputs["data"])

    # Keep track of dates if available for later joining
    original_df = df.copy()

    # Prepare features (exclude year if it was used for grouping)
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features]

    # Add missing columns as 0
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    # Predict
    predictions = model.predict(X)

    # Reverse log transform
    if exp_transform:
        predictions = np.exp(predictions)

    result = pd.DataFrame({
        'predicted_total': predictions
    })

    _save_data(result, outputs["predictions"])
    return f"predict_daily_totals: {len(predictions)} predictions, mean={predictions.mean():.2f}"


# =============================================================================
# SERVICE 5: COMPUTE DISAGGREGATION RATIOS
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"artifact": {"format": "pickle"}},
    description="Compute product/store/country ratios for disaggregation",
    tags=["preprocessing", "ratios", "disaggregation", "generic"],
    version="1.0.0"
)
def compute_disaggregation_ratios(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "date",
    target_column: str = "num_sold",
    product_column: str = "product",
    store_column: str = "store",
    country_column: str = "country",
    reference_years: List[int] = None,
) -> str:
    """
    Compute historical ratios for disaggregating daily totals.

    Based on solution notebook: use historical product ratios, store weights,
    and country weights to disaggregate daily total predictions.

    Parameters:
        reference_years: Years to use for computing ratios (default: 2017, 2019)
    """
    df = _load_data(inputs["data"])
    df[date_column] = pd.to_datetime(df[date_column])

    reference_years = reference_years or [2017, 2019]

    # Product ratios by day of year (average across reference years)
    df['mm_dd'] = df[date_column].dt.strftime('%m-%d')
    df['year'] = df[date_column].dt.year

    # Get reference data
    ref_data = df[df['year'].isin(reference_years)]

    # Product ratios: proportion of total daily sales for each product
    daily_product = ref_data.groupby([date_column, product_column])[target_column].sum().reset_index()
    daily_total = ref_data.groupby(date_column)[target_column].sum().reset_index()
    daily_total.columns = [date_column, 'daily_total']

    daily_product = daily_product.merge(daily_total, on=date_column)
    daily_product['ratio'] = daily_product[target_column] / daily_product['daily_total']
    daily_product['mm_dd'] = pd.to_datetime(daily_product[date_column]).dt.strftime('%m-%d')

    # Average ratios across years for each mm-dd and product
    product_ratios = daily_product.groupby(['mm_dd', product_column])['ratio'].mean().reset_index()
    product_ratios.columns = ['mm_dd', product_column, 'product_ratio']

    # Store weights (overall proportion)
    store_weights = df.groupby(store_column)[target_column].sum()
    store_weights = store_weights / store_weights.sum()
    store_weights = store_weights.to_dict()

    # Country weights (use equal weights as per solution)
    countries = df[country_column].unique()
    country_weights = {c: 1.0 / len(countries) for c in countries}

    artifact = {
        'product_ratios': product_ratios,
        'store_weights': store_weights,
        'country_weights': country_weights,
        'product_column': product_column,
        'store_column': store_column,
        'country_column': country_column,
    }

    os.makedirs(os.path.dirname(outputs["artifact"]) or ".", exist_ok=True)
    with open(outputs["artifact"], "wb") as f:
        pickle.dump(artifact, f)

    return f"compute_disaggregation_ratios: {len(product_ratios)} product-date ratios, {len(store_weights)} stores"


# =============================================================================
# SERVICE 6: DISAGGREGATE PREDICTIONS
# =============================================================================

@contract(
    inputs={
        "test_data": {"format": "csv", "required": True},
        "daily_predictions": {"format": "csv", "required": True},
        "ratios_artifact": {"format": "pickle", "required": True},
    },
    outputs={"predictions": {"format": "csv"}},
    description="Disaggregate daily total predictions to product/store/country level",
    tags=["inference", "disaggregation", "postprocessing", "generic"],
    version="1.0.0"
)
def disaggregate_predictions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "date",
    id_column: str = "row_id",
    target_column: str = "num_sold",
) -> str:
    """
    Disaggregate daily total predictions back to product/store/country level.

    Uses the ratios computed from historical data to split daily totals.
    """
    test = _load_data(inputs["test_data"])
    daily_preds = _load_data(inputs["daily_predictions"])

    with open(inputs["ratios_artifact"], "rb") as f:
        ratios = pickle.load(f)

    product_ratios = ratios['product_ratios']
    store_weights = ratios['store_weights']
    country_weights = ratios['country_weights']
    product_col = ratios['product_column']
    store_col = ratios['store_column']
    country_col = ratios['country_column']

    test[date_column] = pd.to_datetime(test[date_column])
    test['mm_dd'] = test[date_column].dt.strftime('%m-%d')

    # Add daily predictions to test data
    # Predictions are ordered by unique dates
    unique_dates = test[date_column].unique()
    unique_dates = sorted(unique_dates)
    date_to_pred = dict(zip(unique_dates, daily_preds['predicted_total'].values))
    test['daily_total'] = test[date_column].map(date_to_pred)

    # Merge product ratios
    test = test.merge(product_ratios, on=['mm_dd', product_col], how='left')
    # Fill missing ratios with equal distribution
    n_products = test[product_col].nunique()
    test['product_ratio'] = test['product_ratio'].fillna(1.0 / n_products)

    # Apply weights
    test['store_weight'] = test[store_col].map(store_weights)
    test['country_weight'] = test[country_col].map(country_weights)

    # Calculate final prediction
    test[target_column] = (
        test['daily_total'] *
        test['product_ratio'] *
        test['store_weight'] *
        test['country_weight']
    )

    # Round to integer (sales count)
    test[target_column] = test[target_column].round().astype(int)

    # Ensure non-negative
    test[target_column] = test[target_column].clip(lower=0)

    result = test[[id_column, target_column]]
    _save_data(result, outputs["predictions"])

    return f"disaggregate_predictions: {len(result)} predictions, mean={result[target_column].mean():.2f}"


# =============================================================================
# SERVICE 7: CREATE TPS SUBMISSION (Complete Pipeline)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "test_data": {"format": "csv", "required": True},
    },
    outputs={
        "submission": {"format": "csv"},
        "metrics": {"format": "json"},
    },
    description="Complete TPS Sep 2022 pipeline: aggregate, train, predict, disaggregate",
    tags=["pipeline", "end-to-end", "submission", "generic"],
    version="1.0.0"
)
def create_tps_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "date",
    target_column: str = "num_sold",
    id_column: str = "row_id",
    product_column: str = "product",
    store_column: str = "store",
    country_column: str = "country",
    model_type: str = "lasso",
    lasso_alpha: float = 0.00001,
    ridge_alpha: float = 0.1,
    n_splits: int = 4,
    exclude_covid: bool = True,
) -> str:
    """
    Complete TPS Sep 2022 pipeline in a single service.

    This implements the full strategy from top solutions:
    1. Aggregate training data by date
    2. Engineer features
    3. Train Lasso/Ridge regression (Lasso recommended)
    4. Compute disaggregation ratios
    5. Predict daily totals for test dates
    6. Disaggregate to product/store/country level

    Parameters:
        date_column: Column containing dates
        target_column: Target column (num_sold)
        id_column: ID column for submission
        product_column: Product column
        store_column: Store column
        country_column: Country column
        model_type: 'lasso' (recommended) or 'ridge'
        lasso_alpha: Lasso regularization (default 0.00001)
        ridge_alpha: Ridge regularization (default 0.1)
        n_splits: CV folds
        exclude_covid: Exclude COVID period from training
    """
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.model_selection import GroupKFold

    # Load data
    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    train[date_column] = pd.to_datetime(train[date_column])
    test[date_column] = pd.to_datetime(test[date_column])

    # Clean product names (as per solution)
    for df in [train, test]:
        df[product_column] = df[product_column].str.replace(' ', '_').str.replace(':', '_')

    # ========== STEP 1: Compute disaggregation ratios ==========
    reference_years = [2017, 2019]
    train['mm_dd'] = train[date_column].dt.strftime('%m-%d')
    train['year'] = train[date_column].dt.year

    ref_data = train[train['year'].isin(reference_years)]

    # Product ratios
    daily_product = ref_data.groupby([date_column, product_column])[target_column].sum().reset_index()
    daily_total_ref = ref_data.groupby(date_column)[target_column].sum().reset_index()
    daily_total_ref.columns = [date_column, 'daily_total']

    daily_product = daily_product.merge(daily_total_ref, on=date_column)
    daily_product['ratio'] = daily_product[target_column] / daily_product['daily_total']
    daily_product['mm_dd'] = pd.to_datetime(daily_product[date_column]).dt.strftime('%m-%d')

    product_ratios = daily_product.groupby(['mm_dd', product_column])['ratio'].mean().reset_index()
    product_ratios.columns = ['mm_dd', product_column, 'product_ratio']

    # Store weights
    store_weights = train.groupby(store_column)[target_column].sum()
    store_weights = store_weights / store_weights.sum()
    store_weights = store_weights.to_dict()

    # Country weights (equal)
    countries = train[country_column].unique()
    country_weights = {c: 1.0 / len(countries) for c in countries}

    # ========== STEP 2: Aggregate training data by date ==========
    daily_train = train.groupby(date_column)[target_column].sum().reset_index()

    # Exclude COVID period
    if exclude_covid:
        mask = ~((daily_train[date_column] >= "2020-03-01") & (daily_train[date_column] < "2020-06-01"))
        daily_train = daily_train[mask]

    # ========== STEP 3: Engineer features for training ==========
    def add_features(df, date_col):
        df = df.copy()
        df['year'] = df[date_col].dt.year - 2016
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear

        # Adjust for leap year
        df['day_of_year'] = df.apply(
            lambda x: x['day_of_year'] - 1
            if (x[date_col] > pd.Timestamp("2020-02-29") and x[date_col] < pd.Timestamp("2021-01-01"))
            else x['day_of_year'], axis=1
        )

        # Weekend flags
        df['friday'] = (df[date_col].dt.weekday == 4).astype(int)
        df['saturday'] = (df[date_col].dt.weekday == 5).astype(int)
        df['sunday'] = (df[date_col].dt.weekday == 6).astype(int)

        # Cyclical
        df['month_sin'] = np.sin(df['month'] * np.pi / 24)
        df['month_cos'] = np.cos(df['month'] * np.pi / 24)
        df['day_sin'] = np.sin(df['day'] * np.pi / 62)

        # Important dates
        important_dates = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17,
            124, 125, 126, 127, 140, 141, 142,
            167, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181,
            203, 230, 231, 232, 233, 234, 282, 289, 290,
            307, 308, 309, 310, 311, 312, 313, 317, 318, 319, 320,
            360, 361, 362, 363, 364, 365
        ]
        df['is_important_date'] = df['day_of_year'].isin(important_dates).astype(int)

        # Holiday features
        try:
            import holidays
            years = [2017, 2018, 2019, 2020, 2021]
            holiday_dict = {}
            for country in ['BE', 'FR', 'DE', 'IT', 'PL', 'ES']:
                country_holidays = holidays.CountryHoliday(country, years=years)
                holiday_dict.update(country_holidays)
            df['is_holiday'] = df[date_col].dt.date.map(lambda x: 1 if x in holiday_dict else 0)
        except:
            df['is_holiday'] = 0

        return df

    daily_train = add_features(daily_train, date_column)

    # Get unique test dates and add features
    test_dates = test.groupby(date_column)[id_column].first().reset_index().drop(columns=[id_column])
    test_dates = add_features(test_dates, date_column)

    # Log transform target
    daily_train[target_column] = np.log(daily_train[target_column])

    # ========== STEP 4: Train model (Lasso or Ridge) ==========
    feature_cols = [c for c in daily_train.columns if c not in [date_column, target_column]]

    X = daily_train[feature_cols]
    y = daily_train[target_column]

    # GroupKFold by year
    groups = X['year']
    X_no_year = X.drop(columns=['year'])
    feature_cols_no_year = [c for c in feature_cols if c != 'year']

    kf = GroupKFold(n_splits=n_splits)
    cv_scores = []
    preds_list = []

    for train_idx, val_idx in kf.split(X_no_year, y, groups=groups):
        X_train, X_val = X_no_year.iloc[train_idx], X_no_year.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type.lower() == 'lasso':
            model = Lasso(alpha=lasso_alpha, tol=1e-3, max_iter=1000000, random_state=42)
        else:
            model = Ridge(alpha=ridge_alpha, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        smape = np.mean(np.abs(y_pred - y_val) / (np.abs(y_pred) + np.abs(y_val))) * 200
        cv_scores.append(smape)

        # Collect predictions for averaging
        preds_list.append(model.predict(test_dates[feature_cols_no_year]))

    # Average predictions from all folds
    if model_type.lower() == 'lasso':
        final_model = Lasso(alpha=lasso_alpha, tol=1e-3, max_iter=1000000, random_state=42)
    else:
        final_model = Ridge(alpha=ridge_alpha, random_state=42)
    final_model.fit(X_no_year, y)

    # ========== STEP 5: Predict daily totals ==========
    X_test = test_dates[feature_cols_no_year]
    daily_preds = final_model.predict(X_test)
    daily_preds = np.exp(daily_preds)  # Reverse log transform

    # ========== STEP 6: Disaggregate predictions ==========
    test['mm_dd'] = test[date_column].dt.strftime('%m-%d')

    # Map daily predictions to test data
    unique_test_dates = sorted(test[date_column].unique())
    date_to_pred = dict(zip(unique_test_dates, daily_preds))
    test['daily_total'] = test[date_column].map(date_to_pred)

    # Merge product ratios
    test = test.merge(product_ratios, on=['mm_dd', product_column], how='left')
    n_products = test[product_column].nunique()
    test['product_ratio'] = test['product_ratio'].fillna(1.0 / n_products)

    # Apply weights
    test['store_weight'] = test[store_column].map(store_weights)
    test['country_weight'] = test[country_column].map(country_weights)

    # Calculate final prediction
    test[target_column] = (
        test['daily_total'] *
        test['product_ratio'] *
        test['store_weight'] *
        test['country_weight']
    )

    # Round and clip
    test[target_column] = test[target_column].round().astype(int).clip(lower=0)

    # Create submission
    submission = test[[id_column, target_column]].sort_values(id_column)
    _save_data(submission, outputs["submission"])

    # Save metrics
    metrics = {
        "model_type": model_type.capitalize(),
        "alpha": lasso_alpha if model_type.lower() == 'lasso' else ridge_alpha,
        "cv_smape_mean": float(np.mean(cv_scores)),
        "cv_smape_std": float(np.std(cv_scores)),
        "cv_scores": [float(s) for s in cv_scores],
        "n_train_days": len(daily_train),
        "n_test_rows": len(submission),
        "exclude_covid": exclude_covid,
        "prediction_mean": float(submission[target_column].mean()),
        "prediction_std": float(submission[target_column].std()),
    }

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    return f"create_tps_submission: CV SMAPE = {metrics['cv_smape_mean']:.4f}, {len(submission)} predictions"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "create_tps_submission": create_tps_submission,
    "aggregate_sales_by_date": aggregate_sales_by_date,
    "engineer_tps_features": engineer_tps_features,
    "train_ridge_regressor": train_ridge_regressor,
    "predict_daily_totals": predict_daily_totals,
    "compute_disaggregation_ratios": compute_disaggregation_ratios,
    "disaggregate_predictions": disaggregate_predictions,
}


def register_to_kb():
    """Register all services in this module to the KB database."""
    import sqlite3
    import hashlib
    import inspect

    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb.sqlite")
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
            name, version, "tps_sep_2022_services", description, func.__doc__ or "",
            tags, "tabular-playground", input_contract, output_contract, parameters,
            source, source_hash
        ))

    conn.commit()
    conn.close()
    return f"Successfully registered {len(SERVICE_REGISTRY)} services to KB."


if __name__ == "__main__":
    print("Registering TPS Sep 2022 services...")
    result = register_to_kb()
    print(result)
