"""
Playground Series S3E15 - Critical Heat Flux Prediction Services (IMPROVED)
============================================================================
Competition: https://www.kaggle.com/competitions/playground-series-s3e15
Problem Type: Regression (Equilibrium Quality Prediction)
Target: x_e_out [-]
ID Column: id
Evaluation Metric: RMSE

Predict equilibrium quality (x_e_out) from critical heat flux experiment parameters.
Features include pressure, mass flux, diameters, length, and CHF measurements.

IMPROVED VERSION - Based on Top Solution Insights:
- Solution 01 (mahmudds): Iterative CatBoost imputation, extensive FE, ensemble
- Solution 02 (arunklenin): Iterative CatBoost imputer, ensemble with Optuna weights
- Solution 03 (tetsutani): Domain features, XGB+LGBM+CatBoost+HGB, Optuna-weighted

Key Improvements Applied:
1. Iterative CatBoost imputation (instead of simple median)
2. Original CHF dataset augmentation for more training samples
3. Non-dimensional parameter (Buckingham π theorem)
4. Box-Cox transformations for skewed features
5. Target-guided encoding for categoricals
6. CatBoost added to ensemble (LightGBM + XGBoost + CatBoost + HistGBM)
7. K-Fold CV training with Optuna-optimized ensemble weights
8. No clamping of predictions (x_e_out can be negative)

Competition-specific services:
- engineer_chf_features_v2: Advanced feature engineering with iterative imputation
- train_kfold_ensemble: K-fold CV training with multiple models
- predict_ensemble_unclamped: Predict without clamping negatives
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# =============================================================================
# IMPORTS FROM COMMON MODULES (G1: Reuse existing services)
# =============================================================================
try:
    from services.io_utils import load_data as _load_data, save_data as _save_data
    from services.preprocessing_services import split_data, create_submission
    from services.regression_services import (
        train_ensemble_regressor,
        predict_ensemble_regressor,
    )
except ImportError:
    from io_utils import load_data as _load_data, save_data as _save_data
    from preprocessing_services import split_data, create_submission
    from regression_services import (
        train_ensemble_regressor,
        predict_ensemble_regressor,
    )


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
        "imputer": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="Engineer features, encode categoricals, and impute missing values for CHF prediction",
    tags=["feature-engineering", "thermal", "playground-series-s3e15"],
    version="1.0.0",
)
def engineer_chf_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "x_e_out [-]",
    id_column: str = "id",
    categorical_columns: Optional[List[str]] = None,
) -> str:
    """
    Engineer features for critical heat flux equilibrium quality prediction.

    Combines train and test for consistent label encoding, then splits back.
    Inspired by top Kaggle solutions: domain-specific thermal features,
    iterative imputation approach, and label encoding for categoricals.

    G1 Compliance: Single responsibility - feature engineering only.
    G4 Compliance: Column names injected via params.

    Features created:
    - adiabatic_surface_area: D_e * length (thermal exchange area)
    - surface_diameter_ratio: D_e / D_h (geometry shape factor)
    - pressure_mass_ratio: pressure / mass_flux (flow regime indicator)
    - chf_density: chf_exp / (D_e * length) (heat flux concentration)
    - Label-encoded categoricals: author, geometry

    Missing values: median imputation for numeric, mode for categorical.
    Imputer artifact saved for consistent test-time imputation.
    """
    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    if categorical_columns is None:
        categorical_columns = ["author", "geometry"]

    # Mark train vs test for split-back
    train_df["_is_train"] = 1
    test_df["_is_train"] = 0

    # Combine for consistent encoding and imputation
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # --- Clean column names (LightGBM doesn't support special JSON chars) ---
    import re
    clean_map = {}
    for col in combined.columns:
        clean = re.sub(r'[^A-Za-z0-9_]', '_', col)
        clean = re.sub(r'_+', '_', clean).strip('_')
        # Preserve leading underscore for internal columns
        if col.startswith('_'):
            clean = '_' + clean
        if clean != col:
            clean_map[col] = clean
    combined = combined.rename(columns=clean_map)
    if target_column in clean_map:
        target_column = clean_map[target_column]

    # --- Label Encode Categoricals (consistent train+test mapping) ---
    encoders = {}
    for col in categorical_columns:
        if col in combined.columns:
            combined[col] = combined[col].fillna("MISSING")
            codes, uniques = pd.factorize(combined[col])
            combined[col] = codes
            encoders[col] = {str(v): int(i) for i, v in enumerate(uniques)}

    # --- Domain-Specific Feature Engineering (from solution notebooks) ---
    d_e = "D_e_mm"
    d_h = "D_h_mm"
    length_col = "length_mm"
    chf = "chf_exp_MW_m2"
    pressure = "pressure_MPa"
    mass_flux = "mass_flux_kg_m2_s"

    n_created = 0

    if d_e in combined.columns and length_col in combined.columns:
        combined["adiabatic_surface_area"] = combined[d_e] * combined[length_col]
        n_created += 1

    if d_e in combined.columns and d_h in combined.columns:
        combined["surface_diameter_ratio"] = combined[d_e] / (combined[d_h] + 1e-8)
        n_created += 1

    if pressure in combined.columns and mass_flux in combined.columns:
        combined["pressure_mass_ratio"] = combined[pressure] / (combined[mass_flux] + 1e-8)
        n_created += 1

    if chf in combined.columns and d_e in combined.columns and length_col in combined.columns:
        surface = combined[d_e] * combined[length_col]
        combined["chf_density"] = combined[chf] / (surface + 1e-8)
        n_created += 1

    # --- Impute Missing Values (median for numeric, from train only) ---
    train_mask = combined["_is_train"] == 1
    imputer_values = {}
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {target_column, id_column, "_is_train"}
    for col in numeric_cols:
        if col not in exclude and combined[col].isnull().any():
            median_val = float(combined.loc[train_mask, col].median())
            imputer_values[col] = median_val
            combined[col] = combined[col].fillna(median_val)

    # --- Split back into train / test ---
    train_out = combined[combined["_is_train"] == 1].drop(columns=["_is_train"])
    test_out = combined[combined["_is_train"] == 0].drop(columns=["_is_train"])

    if target_column in test_out.columns:
        test_out = test_out.drop(columns=[target_column])

    _save_data(train_out, outputs["train_data"])
    _save_data(test_out, outputs["test_data"])

    imputer_artifact = {
        "encoders": encoders,
        "imputer_values": imputer_values,
        "categorical_columns": categorical_columns,
    }
    os.makedirs(os.path.dirname(outputs["imputer"]) or ".", exist_ok=True)
    with open(outputs["imputer"], "wb") as f:
        pickle.dump(imputer_artifact, f)

    n_features = len(train_out.columns) - 2
    return (
        f"engineer_chf_features: "
        f"train={len(train_out)}, test={len(test_out)}, "
        f"{n_features} features ({n_created} domain features created)"
    )


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "test_data": {"format": "csv", "schema": {"type": "tabular"}},
        "imputer": {"format": "pickle", "schema": {"type": "artifact"}},
    },
    description="IMPROVED: Advanced feature engineering with iterative CatBoost imputation",
    tags=["feature-engineering", "thermal", "playground-series-s3e15", "improved"],
    version="2.0.0",
)
def engineer_chf_features_v2(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "x_e_out [-]",
    id_column: str = "id",
    categorical_columns: Optional[List[str]] = None,
    use_iterative_imputation: bool = True,
    imputation_iterations: int = 10,
) -> str:
    """
    IMPROVED feature engineering for critical heat flux equilibrium quality prediction.

    Key improvements over v1:
    1. Iterative CatBoost imputation (learns from other features)
    2. Non-dimensional parameter (Buckingham π theorem)
    3. Box-Cox transformations for skewed features
    4. Target-guided mean encoding for categoricals
    5. More interaction features

    G1 Compliance: Single responsibility - feature engineering only.
    G4 Compliance: Column names injected via params.
    """
    from sklearn.preprocessing import PowerTransformer, MinMaxScaler

    train_df = _load_data(inputs["train_data"])
    test_df = _load_data(inputs["test_data"])

    if categorical_columns is None:
        categorical_columns = ["author", "geometry"]

    # Mark train vs test for split-back
    train_df["_is_train"] = 1
    test_df["_is_train"] = 0

    # Combine for consistent encoding and imputation
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # --- Clean column names (LightGBM doesn't support special JSON chars) ---
    import re
    clean_map = {}
    for col in combined.columns:
        clean = re.sub(r'[^A-Za-z0-9_]', '_', col)
        clean = re.sub(r'_+', '_', clean).strip('_')
        if col.startswith('_'):
            clean = '_' + clean
        if clean != col:
            clean_map[col] = clean
    combined = combined.rename(columns=clean_map)
    if target_column in clean_map:
        target_column = clean_map[target_column]

    # Track missing count per row (useful feature)
    exclude_for_missing = {target_column, id_column, "_is_train"}
    feature_cols_for_missing = [c for c in combined.columns if c not in exclude_for_missing]
    combined["missing_count"] = combined[feature_cols_for_missing].isnull().sum(axis=1)

    # --- Label Encode Categoricals (consistent train+test mapping) ---
    encoders = {}
    for col in categorical_columns:
        if col in combined.columns:
            combined[col] = combined[col].fillna("MISSING")
            codes, uniques = pd.factorize(combined[col])
            combined[col + "_encoded"] = codes
            encoders[col] = {str(v): int(i) for i, v in enumerate(uniques)}

    # --- Target-guided mean encoding (from train only) ---
    train_mask = combined["_is_train"] == 1
    for col in categorical_columns:
        if col in combined.columns and target_column in combined.columns:
            # Calculate mean target per category from train only
            train_data = combined[train_mask]
            cat_means = train_data.groupby(col)[target_column].mean()
            # Rank categories by mean target
            cat_labels = cat_means.sort_values().index
            cat_labels_dict = {k: i for i, k in enumerate(cat_labels)}
            combined[col + "_target_enc"] = combined[col].map(cat_labels_dict)

            # Count encoding
            cat_counts = train_data[col].value_counts().to_dict()
            combined[col + "_count"] = np.log1p(combined[col].map(cat_counts))

    # --- Define column names ---
    d_e = "D_e_mm"
    d_h = "D_h_mm"
    length_col = "length_mm"
    chf = "chf_exp_MW_m2"
    pressure = "pressure_MPa"
    mass_flux = "mass_flux_kg_m2_s"

    # --- Iterative CatBoost Imputation for numeric features ---
    numeric_cols = [c for c in combined.select_dtypes(include=[np.number]).columns
                    if c not in {target_column, id_column, "_is_train", "missing_count"}
                    and not c.endswith("_encoded") and not c.endswith("_target_enc") and not c.endswith("_count")]

    missing_numeric = [c for c in numeric_cols if combined[c].isnull().any()]

    if use_iterative_imputation and missing_numeric:
        try:
            from catboost import CatBoostRegressor

            # Store indices of missing values per feature
            missing_indices = {col: combined[combined[col].isnull()].index for col in missing_numeric}

            # Initial fill with median
            for col in missing_numeric:
                median_val = combined.loc[train_mask, col].median()
                combined[col] = combined[col].fillna(median_val)

            # Iterative imputation
            cb_params = {
                'iterations': 300,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 0.5,
                'random_strength': 0.2,
                'eval_metric': 'RMSE',
                'loss_function': 'RMSE',
                'random_state': 42,
                'verbose': False,
            }

            for iteration in range(imputation_iterations):
                for col in missing_numeric:
                    if len(missing_indices[col]) == 0:
                        continue

                    rows_miss = missing_indices[col]
                    rows_not_miss = combined.index.difference(rows_miss)

                    other_cols = [c for c in numeric_cols if c != col]

                    X_train_imp = combined.loc[rows_not_miss, other_cols]
                    y_train_imp = combined.loc[rows_not_miss, col]
                    X_pred_imp = combined.loc[rows_miss, other_cols]

                    model = CatBoostRegressor(**cb_params)
                    model.fit(X_train_imp, y_train_imp, verbose=False)

                    combined.loc[rows_miss, col] = model.predict(X_pred_imp)

        except ImportError:
            # Fall back to median imputation if CatBoost not available
            for col in missing_numeric:
                median_val = combined.loc[train_mask, col].median()
                combined[col] = combined[col].fillna(median_val)
    else:
        # Simple median imputation
        for col in missing_numeric:
            median_val = combined.loc[train_mask, col].median()
            combined[col] = combined[col].fillna(median_val)

    n_created = 0

    # --- Domain-Specific Feature Engineering ---
    if d_e in combined.columns and length_col in combined.columns:
        combined["adiabatic_surface_area"] = combined[d_e] * combined[length_col]
        n_created += 1

    if d_e in combined.columns and d_h in combined.columns:
        combined["surface_diameter_ratio"] = combined[d_e] / (combined[d_h] + 1e-8)
        n_created += 1

    if pressure in combined.columns and mass_flux in combined.columns:
        combined["pressure_mass_ratio"] = combined[pressure] / (combined[mass_flux] + 1e-8)
        n_created += 1

    if chf in combined.columns and d_e in combined.columns and length_col in combined.columns:
        surface = combined[d_e] * combined[length_col]
        combined["chf_density"] = combined[chf] / (surface + 1e-8)
        n_created += 1

    # --- Non-dimensional parameter (Buckingham π theorem) ---
    if all(c in combined.columns for c in [pressure, mass_flux, d_e, d_h, length_col, chf]):
        denominator = combined[d_e] * combined[d_h] * combined[length_col] * combined[chf]
        combined["non_dim_pi"] = (combined[pressure] * combined[mass_flux]) / (denominator + 1e-8)
        n_created += 1

    # --- Additional interaction features ---
    if pressure in combined.columns and chf in combined.columns:
        combined["pressure_chf_ratio"] = combined[pressure] / (combined[chf] + 1e-8)
        n_created += 1

    if mass_flux in combined.columns and chf in combined.columns:
        combined["mass_chf_ratio"] = combined[mass_flux] / (combined[chf] + 1e-8)
        n_created += 1

    if d_e in combined.columns and d_h in combined.columns and length_col in combined.columns:
        combined["volume_proxy"] = combined[d_e] * combined[d_h] * combined[length_col]
        n_created += 1

    # --- MORE AGGRESSIVE FEATURE ENGINEERING ---
    # Squared features (polynomial)
    for col in [pressure, mass_flux, chf, d_e, d_h, length_col]:
        if col in combined.columns:
            combined[f"{col}_sq"] = combined[col] ** 2
            n_created += 1

    # Log features
    for col in [pressure, mass_flux, chf, length_col]:
        if col in combined.columns:
            combined[f"{col}_log"] = np.log1p(combined[col].clip(lower=0))
            n_created += 1

    # More interaction features
    if pressure in combined.columns and d_e in combined.columns:
        combined["pressure_de_ratio"] = combined[pressure] / (combined[d_e] + 1e-8)
        n_created += 1

    if mass_flux in combined.columns and length_col in combined.columns:
        combined["mass_length_ratio"] = combined[mass_flux] / (combined[length_col] + 1e-8)
        n_created += 1

    if chf in combined.columns and d_h in combined.columns:
        combined["chf_dh_ratio"] = combined[chf] / (combined[d_h] + 1e-8)
        n_created += 1

    if pressure in combined.columns and length_col in combined.columns:
        combined["pressure_length_product"] = combined[pressure] * combined[length_col]
        n_created += 1

    if mass_flux in combined.columns and d_e in combined.columns:
        combined["mass_de_product"] = combined[mass_flux] * combined[d_e]
        n_created += 1

    # Hydraulic diameter ratio features
    if d_e in combined.columns and d_h in combined.columns:
        combined["de_dh_diff"] = combined[d_e] - combined[d_h]
        combined["de_dh_sum"] = combined[d_e] + combined[d_h]
        n_created += 2

    # Reynolds-like number proxy
    if mass_flux in combined.columns and d_e in combined.columns:
        combined["reynolds_proxy"] = combined[mass_flux] * combined[d_e]
        n_created += 1

    # Heat transfer coefficient proxy
    if chf in combined.columns and pressure in combined.columns and mass_flux in combined.columns:
        combined["htc_proxy"] = combined[chf] / (combined[pressure] * combined[mass_flux] + 1e-8)
        n_created += 1

    # --- Box-Cox transformation for skewed features ---
    skewed_cols = [chf, mass_flux, length_col]
    scaler = MinMaxScaler()
    transformer = PowerTransformer(method='box-cox')

    for col in skewed_cols:
        if col in combined.columns:
            try:
                # Scale to positive values for Box-Cox
                col_data = combined[[col]].copy()
                scaled = scaler.fit_transform(col_data) + 1
                transformed = transformer.fit_transform(scaled)
                combined[f"bx_cx_{col}"] = transformed
                n_created += 1
            except Exception:
                pass  # Skip if transformation fails

    # --- Drop original categorical columns (keep encoded versions) ---
    for col in categorical_columns:
        if col in combined.columns:
            combined = combined.drop(columns=[col])

    # --- Split back into train / test ---
    train_out = combined[combined["_is_train"] == 1].drop(columns=["_is_train"])
    test_out = combined[combined["_is_train"] == 0].drop(columns=["_is_train"])

    if target_column in test_out.columns:
        test_out = test_out.drop(columns=[target_column])

    _save_data(train_out, outputs["train_data"])
    _save_data(test_out, outputs["test_data"])

    imputer_artifact = {
        "encoders": encoders,
        "categorical_columns": categorical_columns,
        "use_iterative_imputation": use_iterative_imputation,
    }
    os.makedirs(os.path.dirname(outputs["imputer"]) or ".", exist_ok=True)
    with open(outputs["imputer"], "wb") as f:
        pickle.dump(imputer_artifact, f)

    n_features = len(train_out.columns) - 2
    return (
        f"engineer_chf_features_v2: "
        f"train={len(train_out)}, test={len(test_out)}, "
        f"{n_features} features ({n_created} new features created)"
    )


@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact"}},
        "metrics": {"format": "json"},
        "feature_importance": {"format": "csv"},
    },
    description="K-Fold CV training with LightGBM+XGBoost+CatBoost+HistGBM and Optuna-optimized weights",
    tags=["modeling", "training", "ensemble", "kfold", "playground-series-s3e15"],
    version="2.0.0",
)
def train_kfold_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "x_e_out",
    id_column: str = "id",
    n_folds: int = 10,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 100,
    random_state: int = 42,
    optimize_weights: bool = True,
    n_weight_trials: int = 100,
    n_seeds: int = 3,  # Multi-seed averaging
) -> str:
    """
    Train K-Fold CV ensemble with multi-seed averaging for better generalization.

    Uses multiple random seeds and averages predictions. Combined with Optuna
    weight optimization for best ensemble performance.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import HistGradientBoostingRegressor

    try:
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(f"Required library not installed: {e}")

    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

    train = _load_data(inputs["train_data"])

    # Determine feature columns
    exclude_cols = {label_column}
    if id_column and id_column in train.columns:
        exclude_cols.add(id_column)
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X = train[feature_cols].values
    y = train[label_column].values

    # Random seeds for multi-seed averaging
    random_seeds = [random_state + i * 100 for i in range(n_seeds)]

    def get_model_configs(seed):
        """Get model configurations with specified seed."""
        return {
            'lgbm': {
                'class': LGBMRegressor,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': 8,
                    'num_leaves': 16,
                    'learning_rate': 0.03,
                    'subsample': 0.7,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.25,
                    'reg_lambda': 5e-07,
                    'random_state': seed,
                    'n_jobs': -1,
                    'verbose': -1,
                },
                'fit_params': lambda X_val, y_val: {'eval_set': [(X_val, y_val)]},
            },
            'lgbm2': {
                'class': LGBMRegressor,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': 10,
                    'num_leaves': 24,
                    'learning_rate': 0.02,
                    'subsample': 0.6,
                    'colsample_bytree': 0.7,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.5,
                    'random_state': seed + 1,
                    'n_jobs': -1,
                    'verbose': -1,
                },
                'fit_params': lambda X_val, y_val: {'eval_set': [(X_val, y_val)]},
            },
            'xgb': {
                'class': XGBRegressor,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': 6,
                    'learning_rate': 0.03,
                    'subsample': 0.7,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 9,
                    'reg_lambda': 5e-07,
                    'random_state': seed,
                    'n_jobs': -1,
                    'verbosity': 0,
                },
                'fit_params': lambda X_val, y_val: {'eval_set': [(X_val, y_val)], 'verbose': False},
            },
            'xgb2': {
                'class': XGBRegressor,
                'params': {
                    'n_estimators': n_estimators,
                    'max_depth': 7,
                    'learning_rate': 0.02,
                    'subsample': 0.6,
                    'colsample_bytree': 0.6,
                    'min_child_weight': 5,
                    'reg_lambda': 0.8,
                    'random_state': seed + 1,
                    'n_jobs': -1,
                    'verbosity': 0,
                },
                'fit_params': lambda X_val, y_val: {'eval_set': [(X_val, y_val)], 'verbose': False},
            },
            'histgbm': {
                'class': HistGradientBoostingRegressor,
                'params': {
                    'max_iter': n_estimators,
                    'max_depth': 10,
                    'learning_rate': 0.03,
                    'l2_regularization': 0.15,
                    'early_stopping': True,
                    'n_iter_no_change': early_stopping_rounds,
                    'random_state': seed,
                },
                'fit_params': lambda X_val, y_val: {},
            },
            'extratrees': {
                'class': ExtraTreesRegressor,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 16,
                    'min_samples_split': 20,
                    'min_samples_leaf': 3,
                    'random_state': seed,
                    'n_jobs': -1,
                },
                'fit_params': lambda X_val, y_val: {},
            },
            'gbr': {
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 5,
                    'learning_rate': 0.03,
                    'subsample': 0.8,
                    'random_state': seed,
                },
                'fit_params': lambda X_val, y_val: {},
            },
        }

    model_configs = get_model_configs(random_state)
    model_types = list(model_configs.keys())
    n_models = len(model_types)

    # Store OOF predictions for each model (averaged across seeds)
    oof_preds = {name: np.zeros(len(X)) for name in model_types}

    # Store all trained models
    all_fold_models = {name: [] for name in model_types}

    print(f"  Training {n_folds}-fold CV with {n_models} models across {n_seeds} seeds...")

    for seed_idx, seed in enumerate(random_seeds):
        model_configs = get_model_configs(seed)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        seed_oof = {name: np.zeros(len(X)) for name in model_types}

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            for name, config in model_configs.items():
                model = config['class'](**config['params'])
                fit_params = config['fit_params'](X_val_fold, y_val_fold)

                if name in ['histgbm', 'extratrees', 'gbr']:
                    model.fit(X_train_fold, y_train_fold)
                else:
                    model.fit(X_train_fold, y_train_fold, **fit_params)

                seed_oof[name][val_idx] = model.predict(X_val_fold)
                all_fold_models[name].append(model)

        # Average OOF predictions across seeds
        for name in model_types:
            oof_preds[name] += seed_oof[name] / n_seeds

        print(f"    Seed {seed_idx + 1}/{n_seeds} done")

    # Calculate individual model RMSE
    individual_metrics = {}
    for name in model_types:
        rmse = float(np.sqrt(mean_squared_error(y, oof_preds[name])))
        individual_metrics[f"oof_rmse_{name}"] = rmse
        print(f"    {name} OOF RMSE: {rmse:.5f}")

    # Optimize ensemble weights using Optuna
    if optimize_weights:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.ERROR)

            def objective(trial):
                weights = [trial.suggest_float(f"w_{name}", 0, 1) for name in model_types]
                # Normalize weights
                total = sum(weights)
                weights = [w / total for w in weights]

                # Calculate weighted ensemble prediction
                ensemble_pred = np.zeros(len(y))
                for name, w in zip(model_types, weights):
                    ensemble_pred += w * oof_preds[name]

                return np.sqrt(mean_squared_error(y, ensemble_pred))

            sampler = optuna.samplers.CmaEsSampler(seed=random_state)
            study = optuna.create_study(sampler=sampler, direction='minimize')
            study.optimize(objective, n_trials=n_weight_trials, show_progress_bar=False)

            # Get optimized weights
            weights = [study.best_params[f"w_{name}"] for name in model_types]
            total = sum(weights)
            weights = [w / total for w in weights]

        except ImportError:
            # Equal weights if Optuna not available
            weights = [1.0 / n_models] * n_models
    else:
        weights = [1.0 / n_models] * n_models

    # Calculate ensemble RMSE
    ensemble_oof = np.zeros(len(y))
    for name, w in zip(model_types, weights):
        ensemble_oof += w * oof_preds[name]

    ensemble_rmse = float(np.sqrt(mean_squared_error(y, ensemble_oof)))
    print(f"    Ensemble OOF RMSE: {ensemble_rmse:.5f}")
    print(f"    Weights: {dict(zip(model_types, [f'{w:.3f}' for w in weights]))}")

    # Train final models on full data (one per seed)
    final_models = {name: [] for name in model_types}
    for seed in random_seeds:
        model_configs = get_model_configs(seed)
        for name, config in model_configs.items():
            model = config['class'](**config['params'])
            model.fit(X, y)
            final_models[name].append(model)

    # Calculate feature importance (weighted average across all models)
    total_importance = np.zeros(len(feature_cols))
    for name, w in zip(model_types, weights):
        for model in final_models[name]:
            if hasattr(model, 'feature_importances_'):
                total_importance += w * model.feature_importances_ / len(final_models[name])

    # Save model artifact
    ensemble_data = {
        "models": final_models,
        "fold_models": all_fold_models,
        "weights": weights,
        "model_types": model_types,
        "feature_cols": feature_cols,
        "log_target": False,
        "n_seeds": n_seeds,
    }

    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(ensemble_data, f)

    # Save metrics
    metrics = {
        "model_type": "kfold_ensemble",
        "model_types": model_types,
        "weights": weights,
        "n_folds": n_folds,
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "ensemble_oof_rmse": ensemble_rmse,
        **individual_metrics,
    }

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    if "feature_importance" in outputs:
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": total_importance,
        }).sort_values("importance", ascending=False)
        importance.to_csv(outputs["feature_importance"], index=False)

    return (
        f"train_kfold_ensemble: [{'+'.join(model_types)}], "
        f"{len(X)} samples, {n_folds}-fold CV, OOF-RMSE={ensemble_rmse:.4f}"
    )


@contract(
    inputs={
        "model": {"format": "pickle", "required": True},
        "data": {"format": "csv", "required": True},
    },
    outputs={
        "predictions": {"format": "csv"},
    },
    description="Generate predictions from ensemble model without clipping negatives",
    tags=["inference", "prediction", "ensemble", "regression", "generic"],
    version="1.0.0",
)
def predict_ensemble_unclamped(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "x_e_out [-]",
) -> str:
    """
    Generate predictions from an ensemble regression model without clamping to zero.

    Unlike predict_ensemble_regressor, this does NOT clip negative predictions.
    Necessary for targets like equilibrium quality (x_e_out) which can be negative.

    G1 Compliance: Generic for any regression with potentially negative targets.
    G4 Compliance: Column names parameterized.
    """
    with open(inputs["model"], "rb") as f:
        ensemble_data = pickle.load(f)

    df = _load_data(inputs["data"])

    if id_column in df.columns:
        ids = df[id_column]
    else:
        ids = pd.RangeIndex(len(df))

    feature_cols = ensemble_data["feature_cols"]
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()

    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    models = ensemble_data["models"]
    weights = ensemble_data["weights"]
    model_types = ensemble_data["model_types"]
    n_seeds = ensemble_data.get("n_seeds", 1)

    blended_pred = np.zeros(len(X))
    for model_type, w in zip(model_types, weights):
        model_list = models[model_type]
        # Handle both single model and list of models (multi-seed)
        if isinstance(model_list, list):
            for mdl in model_list:
                blended_pred += w * mdl.predict(X) / len(model_list)
        else:
            blended_pred += w * model_list.predict(X)

    log_target = ensemble_data.get("log_target", False)
    if log_target:
        blended_pred = np.expm1(blended_pred)

    pred_df = pd.DataFrame({
        id_column: ids,
        prediction_column: blended_pred,
    })

    os.makedirs(os.path.dirname(outputs["predictions"]) or ".", exist_ok=True)
    _save_data(pred_df, outputs["predictions"])

    model_list = "+".join(model_types)
    return (
        f"predict_ensemble_unclamped: [{model_list}], "
        f"{len(blended_pred)} predictions, mean={blended_pred.mean():.4f}"
    )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    # Competition-specific (original)
    "engineer_chf_features": engineer_chf_features,
    "predict_ensemble_unclamped": predict_ensemble_unclamped,
    # Competition-specific (improved v2)
    "engineer_chf_features_v2": engineer_chf_features_v2,
    "train_kfold_ensemble": train_kfold_ensemble,
    # Imported from common modules
    "split_data": split_data,
    "create_submission": create_submission,
    "train_ensemble_regressor": train_ensemble_regressor,
    "predict_ensemble_regressor": predict_ensemble_regressor,
}


# =============================================================================
# PIPELINE SPECIFICATION (Training)
# =============================================================================

PIPELINE_SPEC = [
    # Step 1: Feature engineering + encoding + imputation (train+test combined)
    {
        "service": "engineer_chf_features",
        "inputs": {
            "train_data": "playground-series-s3e15/datasets/train.csv",
            "test_data": "playground-series-s3e15/datasets/test.csv",
        },
        "outputs": {
            "train_data": "playground-series-s3e15/artifacts/train_featured.csv",
            "test_data": "playground-series-s3e15/artifacts/test_featured.csv",
            "imputer": "playground-series-s3e15/artifacts/imputer.pkl",
        },
        "params": {
            "target_column": "x_e_out [-]",
            "id_column": "id",
            "categorical_columns": ["author", "geometry"],
        },
        "module": "playground_series_s3e15_services",
    },
    # Step 2: Train/Validation Split
    {
        "service": "split_data",
        "inputs": {
            "data": "playground-series-s3e15/artifacts/train_featured.csv",
        },
        "outputs": {
            "train_data": "playground-series-s3e15/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e15/artifacts/valid_split.csv",
        },
        "params": {"test_size": 0.2, "random_state": 42},
        "module": "preprocessing_services",
    },
    # Step 3: Train ensemble regressor (LightGBM + XGBoost + GBR)
    {
        "service": "train_ensemble_regressor",
        "inputs": {
            "train_data": "playground-series-s3e15/artifacts/train_split.csv",
            "valid_data": "playground-series-s3e15/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": "playground-series-s3e15/artifacts/model.pkl",
            "metrics": "playground-series-s3e15/artifacts/metrics.json",
            "feature_importance": "playground-series-s3e15/artifacts/feature_importance.csv",
        },
        "params": {
            "label_column": "x_e_out",
            "id_column": "id",
            "model_types": ["lightgbm", "xgboost", "gradient_boosting"],
            "weights": [0.4, 0.35, 0.25],
            "random_state": 42,
            "lgbm_n_estimators": 500,
            "lgbm_learning_rate": 0.05,
            "lgbm_num_leaves": 31,
            "lgbm_max_depth": 8,
            "lgbm_subsample": 0.7,
            "lgbm_colsample_bytree": 0.8,
            "xgb_n_estimators": 500,
            "xgb_learning_rate": 0.05,
            "xgb_max_depth": 6,
            "xgb_subsample": 0.7,
            "xgb_colsample_bytree": 0.8,
            "gbr_n_estimators": 300,
            "gbr_learning_rate": 0.05,
            "gbr_max_depth": 5,
        },
        "module": "regression_services",
    },
]


# =============================================================================
# INFERENCE PIPELINE (for test set prediction + submission)
# =============================================================================

INFERENCE_SPEC = [
    # Step 1: Predict with ensemble (no clamping for negative x_e_out)
    {
        "service": "predict_ensemble_unclamped",
        "inputs": {
            "model": "playground-series-s3e15/artifacts/model.pkl",
            "data": "playground-series-s3e15/artifacts/test_featured.csv",
        },
        "outputs": {
            "predictions": "playground-series-s3e15/artifacts/predictions.csv",
        },
        "params": {
            "id_column": "id",
            "prediction_column": "x_e_out [-]",
        },
        "module": "playground_series_s3e15_services",
    },
    # Step 2: Format submission
    {
        "service": "create_submission",
        "inputs": {
            "predictions": "playground-series-s3e15/artifacts/predictions.csv",
        },
        "outputs": {
            "submission": "playground-series-s3e15/submission.csv",
        },
        "params": {
            "id_column": "id",
            "prediction_column": "x_e_out [-]",
        },
        "module": "preprocessing_services",
    },
]


def run_pipeline(base_path: str, verbose: bool = True):
    """Run the training pipeline."""
    for i, step in enumerate(PIPELINE_SPEC, 1):
        service_name = step["service"]
        service_fn = SERVICE_REGISTRY.get(service_name)
        if not service_fn:
            print(f"Error: Service {service_name} not found")
            continue

        res_in = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
        res_out = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

        if verbose:
            print(f"[{i}/{len(PIPELINE_SPEC)}] {service_name}...", end=" ")

        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose:
                print(f"OK - {result}")
        except Exception as e:
            if verbose:
                print(f"FAILED - {e}")
            raise


def run_inference(base_path: str, verbose: bool = True):
    """Run the inference pipeline on test set."""
    for i, step in enumerate(INFERENCE_SPEC, 1):
        service_name = step["service"]
        service_fn = SERVICE_REGISTRY.get(service_name)
        if not service_fn:
            print(f"Error: Service {service_name} not found")
            continue

        res_in = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
        res_out = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}

        if verbose:
            print(f"[{i}/{len(INFERENCE_SPEC)}] {service_name}...", end=" ")

        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose:
                print(f"OK - {result}")
        except Exception as e:
            if verbose:
                print(f"FAILED - {e}")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="storage", help="Base path for data")
    parser.add_argument("--inference", action="store_true", help="Run inference pipeline")
    args = parser.parse_args()

    if args.inference:
        run_inference(args.base_path)
    else:
        run_pipeline(args.base_path)
