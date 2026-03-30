"""
Web Traffic Time Series Forecasting - Contract-Composable Analytics Services
=====================================================
Competition: https://www.kaggle.com/competitions/web-traffic-time-series-forecasting
Problem Type: Time Series Forecasting (Regression)
Target: Visits (web page views)
Metric: SMAPE (Symmetric Mean Absolute Percentage Error)

Competition-specific services:
- melt_wide_to_long: Transform wide-format time series to long format
- forecast_median: Fast median-based forecasting (proven top performer)
- forecast_median_weekday: Weekend-aware median forecasting
- create_web_traffic_submission: Create properly formatted submission

Based on top solution analysis:
- Simple median of last 49-56 days achieves top 5% performance
- Weekend/weekday differentiation can improve scores further
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from contract import contract
except ImportError:
    def contract(**kwargs):
        def decorator(func):
            func._contract = kwargs
            return func
        return decorator

# Import shared I/O utilities
from services.io_services import load_data, save_data


# =============================================================================
# DATA TRANSFORMATION SERVICES
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
    },
    outputs={
        "train_long": {"format": "csv"},
    },
    description="Transform wide-format time series to long format for aggregation",
    tags=["preprocessing", "time-series", "web-traffic", "reshape"],
    version="1.0.0",
)
def melt_wide_to_long(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    page_column: str = "Page",
    last_n_days: int = 49,
) -> str:
    """
    Transform wide-format time series (pages as rows, dates as columns) to long format.

    Args:
        page_column: Column containing page identifiers
        last_n_days: Number of trailing days to include (default 49)

    Returns:
        Long-format DataFrame with columns: Page, date, Visits
    """
    df = load_data(inputs["train_data"])

    # Get date columns (all except page_column)
    date_cols = [c for c in df.columns if c != page_column]

    # Select last N days
    if last_n_days > 0:
        date_cols = date_cols[-last_n_days:]

    # Melt to long format
    df_long = pd.melt(
        df[[page_column] + date_cols],
        id_vars=page_column,
        var_name='date',
        value_name='Visits'
    )

    # Convert date column to datetime
    df_long['date'] = pd.to_datetime(df_long['date'])

    save_data(df_long, outputs["train_long"])

    return f"melt_wide_to_long: {len(df)} pages, {len(date_cols)} days -> {len(df_long)} rows"


# =============================================================================
# FORECASTING SERVICES
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "key_data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Fast median-based forecasting - uses median of last N days per page",
    tags=["forecasting", "time-series", "web-traffic", "median", "fast"],
    version="1.0.0",
)
def forecast_median(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    page_column: str = "Page",
    last_n_days: int = 49,
    fill_na_value: float = 0.0,
) -> str:
    """
    Forecast using median of last N days per page.

    This is a fast and effective approach that achieves top 5% on this competition.
    Based on top solution analysis:
    - Median is robust to outliers (better than mean for SMAPE)
    - Last 49-56 days provides optimal balance of recency and stability

    Args:
        page_column: Column containing page identifiers
        last_n_days: Number of trailing days to use for median (default 49)
        fill_na_value: Value to use for missing predictions (default 0.0)

    Returns:
        Submission DataFrame with Id and Visits columns
    """
    # Load data
    train_df = load_data(inputs["train_data"])
    key_df = load_data(inputs["key_data"])

    # Get date columns (all except page_column)
    date_cols = [c for c in train_df.columns if c != page_column]

    # Select last N days for median calculation
    if last_n_days > 0 and last_n_days < len(date_cols):
        selected_cols = date_cols[-last_n_days:]
    else:
        selected_cols = date_cols

    # Calculate median per page (ignore NaN values)
    medians = np.nanmedian(train_df[selected_cols].values, axis=1)
    medians = np.nan_to_num(medians, nan=fill_na_value)
    medians = np.round(medians)

    # Create page to median mapping
    page_to_median = dict(zip(train_df[page_column].values, medians))

    # Extract page name from key (remove date suffix)
    # Format: "PageName_YYYY-MM-DD" -> "PageName"
    key_df['PageName'] = key_df[page_column].apply(lambda x: x[:-11])

    # Map predictions
    key_df['Visits'] = key_df['PageName'].map(page_to_median).fillna(fill_na_value)
    key_df['Visits'] = key_df['Visits'].round().astype(int)

    # Create submission
    submission = key_df[['Id', 'Visits']].copy()

    save_data(submission, outputs["predictions"])

    n_pages = len(train_df)
    n_preds = len(submission)
    return f"forecast_median: {n_pages} pages, last {len(selected_cols)} days -> {n_preds} predictions"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "key_data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Weekend-aware median forecasting - separate medians for weekdays vs weekends",
    tags=["forecasting", "time-series", "web-traffic", "median", "weekend"],
    version="1.0.0",
)
def forecast_median_weekday(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    page_column: str = "Page",
    last_n_days: int = 49,
    fill_na_value: float = 0.0,
    wiggle_factor: float = 0.02,
) -> str:
    """
    Weekend-aware median forecasting.

    Computes separate medians for weekdays vs weekends, then applies predictions
    based on whether each forecast date is a weekend.

    Args:
        page_column: Column containing page identifiers
        last_n_days: Number of trailing days to use
        fill_na_value: Value for missing predictions
        wiggle_factor: Small adjustment factor (default 2%)

    Returns:
        Submission DataFrame with Id and Visits columns
    """
    # Load data
    train_df = load_data(inputs["train_data"])
    key_df = load_data(inputs["key_data"])

    # Get date columns
    date_cols = [c for c in train_df.columns if c != page_column]

    # Convert to datetime for weekday extraction
    dates = pd.to_datetime(date_cols)
    is_weekend = np.array(dates.dayofweek >= 5)  # Saturday=5, Sunday=6

    # Select last N days
    if last_n_days > 0 and last_n_days < len(date_cols):
        start_idx = len(date_cols) - last_n_days
        selected_cols = date_cols[start_idx:]
        is_weekend = is_weekend[start_idx:]
    else:
        selected_cols = date_cols

    # Get data matrix
    data_matrix = train_df[selected_cols].values

    # Calculate weekday medians (is_weekend = False)
    weekday_cols = ~is_weekend
    if weekday_cols.any():
        weekday_medians = np.nanmedian(data_matrix[:, weekday_cols], axis=1)
    else:
        weekday_medians = np.nanmedian(data_matrix, axis=1)

    # Calculate weekend medians (is_weekend = True)
    weekend_cols = is_weekend
    if weekend_cols.any():
        weekend_medians = np.nanmedian(data_matrix[:, weekend_cols], axis=1)
    else:
        weekend_medians = weekday_medians

    # Fill NaN
    weekday_medians = np.nan_to_num(weekday_medians, nan=fill_na_value)
    weekend_medians = np.nan_to_num(weekend_medians, nan=fill_na_value)

    # Create page mappings
    pages = train_df[page_column].values
    page_to_weekday = dict(zip(pages, weekday_medians))
    page_to_weekend = dict(zip(pages, weekend_medians))

    # Extract page name and date from key
    key_df['PageName'] = key_df[page_column].apply(lambda x: x[:-11])
    key_df['date'] = key_df[page_column].apply(lambda x: x[-10:])
    key_df['date'] = pd.to_datetime(key_df['date'])
    key_df['is_weekend'] = (key_df['date'].dt.dayofweek >= 5).astype(int)

    # Map predictions based on weekend flag
    def get_prediction(row):
        page = row['PageName']
        if row['is_weekend']:
            return page_to_weekend.get(page, fill_na_value)
        else:
            return page_to_weekday.get(page, fill_na_value)

    key_df['Visits'] = key_df.apply(get_prediction, axis=1)

    # Apply wiggle adjustment
    if wiggle_factor > 0:
        key_df['Visits'] = key_df['Visits'] * (1 + wiggle_factor)

    key_df['Visits'] = key_df['Visits'].round().astype(int)

    # Create submission
    submission = key_df[['Id', 'Visits']].copy()

    save_data(submission, outputs["predictions"])

    n_pages = len(train_df)
    n_preds = len(submission)
    return f"forecast_median_weekday: {n_pages} pages, weekday/weekend split -> {n_preds} predictions"


# =============================================================================
# OPTIMIZED FORECASTING SERVICE (Best Approach from Top Solutions)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True},
        "key_data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Optimized median forecasting with weekend split and wiggle adjustment",
    tags=["forecasting", "time-series", "web-traffic", "median", "optimized"],
    version="1.0.0",
)
def forecast_median_optimized(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    page_column: str = "Page",
    last_n_days: int = 49,
    fill_na_value: float = 0.0,
    wiggle_base: float = 0.02,
    wiggle_second_term: float = 0.04,
    second_term_start: str = "2017-10-13",
) -> str:
    """
    Optimized median forecasting combining best techniques from top solutions:

    1. Weekend/weekday median separation (captures weekly patterns)
    2. Wiggle adjustment (positive bias improves SMAPE)
    3. Different wiggle for second forecast term

    Based on:
    - "weekend-flag-median-with-wiggle.py" (top solution)
    - "one-line-solution-2nd-stage-final.py" (top solution)

    Args:
        page_column: Column containing page identifiers
        last_n_days: Number of trailing days for median (default 49)
        fill_na_value: Value for missing predictions
        wiggle_base: Base wiggle factor (2% default)
        wiggle_second_term: Wiggle for second forecast term (4% default)
        second_term_start: Date when second term starts
    """
    # Load data
    train_df = load_data(inputs["train_data"])
    key_df = load_data(inputs["key_data"])

    # Get date columns
    date_cols = [c for c in train_df.columns if c != page_column]

    # Convert to datetime for weekday extraction
    dates = pd.to_datetime(date_cols)
    is_weekend = np.array(dates.dayofweek >= 5)  # Saturday=5, Sunday=6

    # Select last N days
    if last_n_days > 0 and last_n_days < len(date_cols):
        start_idx = len(date_cols) - last_n_days
        selected_cols = date_cols[start_idx:]
        is_weekend_selected = is_weekend[start_idx:]
    else:
        selected_cols = date_cols
        is_weekend_selected = is_weekend

    # Get data matrix
    data_matrix = train_df[selected_cols].values

    # Calculate weekday medians (is_weekend = False)
    weekday_mask = ~is_weekend_selected
    if weekday_mask.any():
        weekday_medians = np.nanmedian(data_matrix[:, weekday_mask], axis=1)
    else:
        weekday_medians = np.nanmedian(data_matrix, axis=1)

    # Calculate weekend medians (is_weekend = True)
    weekend_mask = is_weekend_selected
    if weekend_mask.any():
        weekend_medians = np.nanmedian(data_matrix[:, weekend_mask], axis=1)
    else:
        weekend_medians = weekday_medians.copy()

    # Fill NaN with fill_na_value
    weekday_medians = np.nan_to_num(weekday_medians, nan=fill_na_value)
    weekend_medians = np.nan_to_num(weekend_medians, nan=fill_na_value)

    # Create page mappings
    pages = train_df[page_column].values
    page_to_weekday = dict(zip(pages, weekday_medians))
    page_to_weekend = dict(zip(pages, weekend_medians))

    # Extract page name and date from key
    key_df['PageName'] = key_df[page_column].apply(lambda x: x[:-11])
    key_df['date'] = key_df[page_column].apply(lambda x: x[-10:])
    key_df['date'] = pd.to_datetime(key_df['date'])
    key_df['is_weekend'] = (key_df['date'].dt.dayofweek >= 5).astype(int)
    key_df['is_second_term'] = (key_df['date'] > second_term_start).astype(int)

    # Vectorized prediction mapping
    visits = np.zeros(len(key_df))
    for i, (page, is_wknd) in enumerate(zip(key_df['PageName'].values, key_df['is_weekend'].values)):
        if is_wknd:
            visits[i] = page_to_weekend.get(page, fill_na_value)
        else:
            visits[i] = page_to_weekday.get(page, fill_na_value)

    # Apply wiggle adjustment (different for first vs second term)
    is_second = key_df['is_second_term'].values.astype(bool)
    wiggle = np.where(is_second, wiggle_second_term, wiggle_base)
    visits = visits * (1 + wiggle)

    # Round and convert to int
    key_df['Visits'] = np.round(visits).astype(int)

    # Ensure no negative values
    key_df.loc[key_df['Visits'] < 0, 'Visits'] = 0

    # Create submission
    submission = key_df[['Id', 'Visits']].copy()

    save_data(submission, outputs["predictions"])

    n_pages = len(train_df)
    n_preds = len(submission)
    return f"forecast_median_optimized: {n_pages} pages, weekend split + wiggle -> {n_preds} predictions"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "melt_wide_to_long": melt_wide_to_long,
    "forecast_median": forecast_median,
    "forecast_median_weekday": forecast_median_weekday,
    "forecast_median_optimized": forecast_median_optimized,
}
