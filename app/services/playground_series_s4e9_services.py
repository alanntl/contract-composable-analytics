"""
Playground Series S4E9 - Used Car Price Prediction - Contract-Composable Analytics Services
====================================================================
Competition: https://www.kaggle.com/competitions/playground-series-s4e9
Problem Type: Regression
Target: price
Metric: RMSE

Solution Insights (from top 3 notebooks):
- Feature engineering: Vehicle_Age, Mileage_per_Year, Is_Luxury_Brand
- Keep ALL categorical columns (brand, model, fuel_type, engine, etc.)
- Replace rare categories (<100 occurrences) with "noise"
- Fill NaN categoricals with "missing"
- LightGBM with MAE objective + 5-fold CV works best

Competition-specific services:
- preprocess_used_car_data: feature engineering + rare grouping + ordinal encode
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract
from services.io_utils import load_data as _load_data, save_data as _save_data


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Preprocess used car data: engineer features, group rare categories, ordinal encode",
    tags=["preprocessing", "feature-engineering", "used-car", "regression"],
    version="1.0.0",
)
def preprocess_used_car_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "price",
    id_column: str = "id",
    current_year: int = 2024,
    year_column: str = "model_year",
    mileage_column: str = "milage",
    brand_column: str = "brand",
    rare_threshold: int = 100,
    categorical_columns: Optional[List[str]] = None,
    rare_columns: Optional[List[str]] = None,
    luxury_brands: Optional[List[str]] = None,
) -> str:
    """
    Full preprocessing for used car price prediction data.

    Steps applied to both train and test:
    1. Engineer features: Vehicle_Age, Mileage_per_Year, Is_Luxury_Brand
    2. Group rare categories (< threshold) as 'noise' (fitted on train)
    3. Fill NaN categoricals with 'missing'
    4. Ordinal-encode all categoricals (fitted on combined train+test)
    5. Fill remaining numeric NaN with 0

    Parameters:
        target_column: Target column name
        id_column: ID column name
        current_year: Reference year for Vehicle_Age calculation
        year_column: Column with vehicle model year
        mileage_column: Column with mileage
        brand_column: Column with brand name
        rare_threshold: Min occurrences to keep a category
        categorical_columns: Explicit list of categorical columns
        rare_columns: Columns to apply rare grouping (auto if None)
        luxury_brands: List of luxury brand names
    """
    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    if luxury_brands is None:
        luxury_brands = [
            'Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land',
            'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini',
            'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach',
            'Bugatti',
        ]

    if categorical_columns is None:
        categorical_columns = [
            'brand', 'model', 'fuel_type', 'engine',
            'transmission', 'ext_col', 'int_col', 'accident', 'clean_title',
        ]

    if rare_columns is None:
        rare_columns = ['model', 'engine', 'transmission', 'ext_col', 'int_col']

    for df in [train, test]:
        if year_column in df.columns:
            df['Vehicle_Age'] = current_year - df[year_column]
            df['Vehicle_Age'] = df['Vehicle_Age'].clip(lower=0)
        if mileage_column in df.columns and 'Vehicle_Age' in df.columns:
            age_safe = df['Vehicle_Age'].replace(0, 1)
            df['Mileage_per_Year'] = df[mileage_column] / age_safe
        if brand_column in df.columns:
            df['Is_Luxury_Brand'] = df[brand_column].apply(
                lambda x: 1 if x in luxury_brands else 0
            )

    # Add aggregation features from top solutions (milage_with_age, Mileage_per_Year_with_age)
    # Compute on combined train+test to ensure consistency
    combined = pd.concat([train, test], ignore_index=True)
    if mileage_column in combined.columns and 'Vehicle_Age' in combined.columns:
        age_milage_mean = combined.groupby('Vehicle_Age')[mileage_column].transform('mean')
        age_mpy_mean = combined.groupby('Vehicle_Age')['Mileage_per_Year'].transform('mean')

        train_len = len(train)
        train['milage_with_age'] = age_milage_mean.iloc[:train_len].values
        test['milage_with_age'] = age_milage_mean.iloc[train_len:].values
        train['Mileage_per_Year_with_age'] = age_mpy_mean.iloc[:train_len].values
        test['Mileage_per_Year_with_age'] = age_mpy_mean.iloc[train_len:].values

    for col in rare_columns:
        if col in train.columns:
            vc = train[col].value_counts(dropna=False)
            rare_vals = set(vc[vc < rare_threshold].index)
            for df in [train, test]:
                if col in df.columns:
                    df.loc[df[col].isin(rare_vals), col] = "noise"

    for df in [train, test]:
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("missing")

    for col in categorical_columns:
        if col in train.columns:
            combined = pd.concat([train[col], test[col]], ignore_index=True)
            codes, uniques = pd.factorize(combined)
            mapping = {v: i for i, v in enumerate(uniques)}
            train[col] = train[col].map(mapping).fillna(-1).astype(int)
            test[col] = test[col].map(mapping).fillna(-1).astype(int)

    for df in [train, test]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

    _save_data(train, outputs["train_data"])
    _save_data(test, outputs["test_data"])

    return f"preprocess_used_car_data: {train.shape[1]} cols, encoded {len(categorical_columns)} cats"


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "original_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Enhanced preprocessing with original data augmentation (top solution pattern)",
    tags=["preprocessing", "feature-engineering", "used-car", "regression", "augmentation"],
    version="2.0.0",
)
def preprocess_used_car_data_enhanced(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "price",
    id_column: str = "id",
    current_year: int = 2024,
    year_column: str = "model_year",
    mileage_column: str = "milage",
    brand_column: str = "brand",
    rare_threshold: int = 100,
    categorical_columns: Optional[List[str]] = None,
    rare_columns: Optional[List[str]] = None,
    luxury_brands: Optional[List[str]] = None,
    filter_unseen_models: bool = True,
) -> str:
    """
    Enhanced preprocessing with original data augmentation (top solution pattern).

    This follows the approach from top 3 notebooks:
    1. Load original used_cars.csv data and clean milage/price columns
    2. Concatenate with train.csv for more training samples
    3. Filter out models not present in test set
    4. Apply feature engineering and encoding

    Parameters:
        filter_unseen_models: Remove rows with models not in test set (default True)
        Other params same as preprocess_used_car_data
    """
    import re

    train = _load_data(inputs["train_data"])
    test = _load_data(inputs["test_data"])

    # Load and clean original data if provided
    original = None
    if "original_data" in inputs and inputs.get("original_data"):
        original_path = inputs["original_data"]
        if os.path.exists(original_path):
            original = _load_data(original_path)

            # Clean milage and price columns (they are strings like "$45,000" and "45,000 mi.")
            def clean_numeric(x):
                if pd.isna(x):
                    return np.nan
                if isinstance(x, (int, float)):
                    return float(x)
                digits = ''.join(re.findall(r'\d+', str(x)))
                return float(digits) if digits else np.nan

            if mileage_column in original.columns:
                original[mileage_column] = original[mileage_column].apply(clean_numeric)
            if target_column in original.columns:
                original[target_column] = original[target_column].apply(clean_numeric)

            # Drop rows with missing target
            original = original.dropna(subset=[target_column])

            # Remove id column from original if present (will be regenerated)
            if id_column in original.columns:
                original = original.drop(columns=[id_column])

            # Concatenate with train (drop id column from train for concat)
            train_no_id = train.drop(columns=[id_column]) if id_column in train.columns else train
            train = pd.concat([train_no_id, original], ignore_index=True)

            # Add back id column
            train[id_column] = range(len(train))

    # Filter out models not in test set (top solution technique)
    if filter_unseen_models and 'model' in train.columns and 'model' in test.columns:
        test_models = set(test['model'].unique())
        train_models_not_in_test = set(train['model'].unique()) - test_models
        if train_models_not_in_test:
            before_len = len(train)
            train = train[~train['model'].isin(train_models_not_in_test)]
            print(f"  Filtered {before_len - len(train)} rows with unseen models")

    if luxury_brands is None:
        luxury_brands = [
            'Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land',
            'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini',
            'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach',
            'Bugatti',
        ]

    if categorical_columns is None:
        categorical_columns = [
            'brand', 'model', 'fuel_type', 'engine',
            'transmission', 'ext_col', 'int_col', 'accident', 'clean_title',
        ]

    if rare_columns is None:
        rare_columns = ['model', 'engine', 'transmission', 'ext_col', 'int_col']

    # Feature engineering
    for df in [train, test]:
        if year_column in df.columns:
            df['Vehicle_Age'] = current_year - df[year_column]
            df['Vehicle_Age'] = df['Vehicle_Age'].clip(lower=0)
        if mileage_column in df.columns and 'Vehicle_Age' in df.columns:
            age_safe = df['Vehicle_Age'].replace(0, 1)
            df['Mileage_per_Year'] = df[mileage_column] / age_safe
        if brand_column in df.columns:
            df['Is_Luxury_Brand'] = df[brand_column].apply(
                lambda x: 1 if x in luxury_brands else 0
            )

    # Aggregation features
    combined = pd.concat([train, test], ignore_index=True)
    if mileage_column in combined.columns and 'Vehicle_Age' in combined.columns:
        age_milage_mean = combined.groupby('Vehicle_Age')[mileage_column].transform('mean')
        age_mpy_mean = combined.groupby('Vehicle_Age')['Mileage_per_Year'].transform('mean')

        train_len = len(train)
        train['milage_with_age'] = age_milage_mean.iloc[:train_len].values
        test['milage_with_age'] = age_milage_mean.iloc[train_len:].values
        train['Mileage_per_Year_with_age'] = age_mpy_mean.iloc[:train_len].values
        test['Mileage_per_Year_with_age'] = age_mpy_mean.iloc[train_len:].values

    # Rare category grouping
    for col in rare_columns:
        if col in train.columns:
            vc = train[col].value_counts(dropna=False)
            rare_vals = set(vc[vc < rare_threshold].index)
            for df in [train, test]:
                if col in df.columns:
                    df.loc[df[col].isin(rare_vals), col] = "noise"

    # Fill missing categoricals
    for df in [train, test]:
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("missing")

    # Ordinal encode
    for col in categorical_columns:
        if col in train.columns:
            combined = pd.concat([train[col], test[col]], ignore_index=True)
            codes, uniques = pd.factorize(combined)
            mapping = {v: i for i, v in enumerate(uniques)}
            train[col] = train[col].map(mapping).fillna(-1).astype(int)
            test[col] = test[col].map(mapping).fillna(-1).astype(int)

    # Fill remaining numeric NaN
    for df in [train, test]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

    _save_data(train, outputs["train_data"])
    _save_data(test, outputs["test_data"])

    orig_info = f" + {len(original)} original rows" if original is not None else ""
    return f"preprocess_used_car_data_enhanced: {train.shape[0]} train rows{orig_info}, {train.shape[1]} cols"


SERVICE_REGISTRY = {
    "preprocess_used_car_data": preprocess_used_car_data,
    "preprocess_used_car_data_enhanced": preprocess_used_car_data_enhanced,
}
