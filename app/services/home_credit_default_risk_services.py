"""
Home Credit Default Risk - SLEGO Services
=========================================

Competition: https://www.kaggle.com/competitions/home-credit-default-risk
Problem Type: Binary Classification
Target: TARGET

Competition-specific services:
- replace_value_with_nan_and_flag: Replace anomalous DAYS_EMPLOYED values and add flag
- prepare_tabular_features: Core feature engineering for application data
- add_ratio_features: Create common ratio features from application data
- aggregate_table_with_child: Aggregate bureau + bureau_balance data
- aggregate_table_basic: Aggregate previous application data
- aggregate_table_delinquency: Aggregate POS_CASH balance data
- aggregate_table_payments: Aggregate installments payments data
- aggregate_table_revolving: Aggregate credit card balance data
- merge_feature_tables: Merge aggregated tables into base data

Imported from common modules:
- train_lightgbm_classifier (from classification_services)
- predict_classifier (from classification_services)
- fit_column_filter, transform_column_filter (from preprocessing_services)
- fit_imputer, transform_imputer (from preprocessing_services)
- fit_encoder, transform_encoder (from preprocessing_services)
- split_data, create_submission (from preprocessing_services)
"""

import os
import sys
import json
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slego_contract import contract

# Import generic services from common modules
try:
    from .preprocessing_services import (
        fit_column_filter, transform_column_filter,
        fit_imputer, transform_imputer,
        fit_encoder, transform_encoder,
        split_data, create_submission,
    )
    from .classification_services import (
        train_lightgbm_classifier, predict_classifier,
    )
except ImportError:
    from services.preprocessing_services import (
        fit_column_filter, transform_column_filter,
        fit_imputer, transform_imputer,
        fit_encoder, transform_encoder,
        split_data, create_submission,
    )
    from services.classification_services import (
        train_lightgbm_classifier, predict_classifier,
    )


# =============================================================================
# PIPELINE SPECIFICATION
# =============================================================================

PIPELINE_SPEC = [
    # Application data cleanup + feature engineering
    {
        "service": "replace_value_with_nan_and_flag",
        "inputs": {"data": "home-credit-default-risk/datasets/application_train.csv"},
        "outputs": {"data": "home-credit-default-risk/artifacts/train_fixed.csv"},
        "params": {
            "days_employed_col": "DAYS_EMPLOYED",
            "anomaly_value": 365243,
            "flag_column": "DAYS_EMPLOYED_ANOM",
        },
        "module": "home_credit_default_risk_services",
    },
    {
        "service": "replace_value_with_nan_and_flag",
        "inputs": {"data": "home-credit-default-risk/datasets/application_test.csv"},
        "outputs": {"data": "home-credit-default-risk/artifacts/test_fixed.csv"},
        "params": {
            "days_employed_col": "DAYS_EMPLOYED",
            "anomaly_value": 365243,
            "flag_column": "DAYS_EMPLOYED_ANOM",
        },
        "module": "home_credit_default_risk_services",
    },
    {
        "service": "prepare_tabular_features",
        "inputs": {"data": "home-credit-default-risk/artifacts/train_fixed.csv"},
        "outputs": {"data": "home-credit-default-risk/artifacts/train_app.csv"},
        "params": {
            "drop_invalid_gender": True,
            "invalid_gender_value": "XNA",
            "replace_zero_phone_change": True,
        },
        "module": "home_credit_default_risk_services",
    },
    {
        "service": "prepare_tabular_features",
        "inputs": {"data": "home-credit-default-risk/artifacts/test_fixed.csv"},
        "outputs": {"data": "home-credit-default-risk/artifacts/test_app.csv"},
        "params": {
            "drop_invalid_gender": True,
            "invalid_gender_value": "XNA",
            "replace_zero_phone_change": True,
        },
        "module": "home_credit_default_risk_services",
    },
    # Aggregate auxiliary tables
    {
        "service": "aggregate_table_with_child",
        "inputs": {
            "bureau_data": "home-credit-default-risk/datasets/bureau.csv",
            "bureau_balance_data": "home-credit-default-risk/datasets/bureau_balance.csv",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/bureau_agg.csv"},
        "params": {"include_balance": True},
        "module": "home_credit_default_risk_services",
    },
    {
        "service": "aggregate_table_basic",
        "inputs": {"data": "home-credit-default-risk/datasets/previous_application.csv"},
        "outputs": {"data": "home-credit-default-risk/artifacts/prev_agg.csv"},
        "params": {},
        "module": "home_credit_default_risk_services",
    },
    {
        "service": "aggregate_table_delinquency",
        "inputs": {"data": "home-credit-default-risk/datasets/POS_CASH_balance.csv"},
        "outputs": {"data": "home-credit-default-risk/artifacts/pos_agg.csv"},
        "params": {},
        "module": "home_credit_default_risk_services",
    },
    {
        "service": "aggregate_table_payments",
        "inputs": {"data": "home-credit-default-risk/datasets/installments_payments.csv"},
        "outputs": {"data": "home-credit-default-risk/artifacts/inst_agg.csv"},
        "params": {},
        "module": "home_credit_default_risk_services",
    },
    {
        "service": "aggregate_table_revolving",
        "inputs": {"data": "home-credit-default-risk/datasets/credit_card_balance.csv"},
        "outputs": {"data": "home-credit-default-risk/artifacts/cc_agg.csv"},
        "params": {},
        "module": "home_credit_default_risk_services",
    },
    # Merge aggregates
    {
        "service": "merge_feature_tables",
        "inputs": {
            "base_data": "home-credit-default-risk/artifacts/train_app.csv",
            "bureau_data": "home-credit-default-risk/artifacts/bureau_agg.csv",
            "prev_data": "home-credit-default-risk/artifacts/prev_agg.csv",
            "pos_data": "home-credit-default-risk/artifacts/pos_agg.csv",
            "installments_data": "home-credit-default-risk/artifacts/inst_agg.csv",
            "credit_card_data": "home-credit-default-risk/artifacts/cc_agg.csv",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/train_merged.csv"},
        "params": {},
        "module": "home_credit_default_risk_services",
    },
    {
        "service": "merge_feature_tables",
        "inputs": {
            "base_data": "home-credit-default-risk/artifacts/test_app.csv",
            "bureau_data": "home-credit-default-risk/artifacts/bureau_agg.csv",
            "prev_data": "home-credit-default-risk/artifacts/prev_agg.csv",
            "pos_data": "home-credit-default-risk/artifacts/pos_agg.csv",
            "installments_data": "home-credit-default-risk/artifacts/inst_agg.csv",
            "credit_card_data": "home-credit-default-risk/artifacts/cc_agg.csv",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/test_merged.csv"},
        "params": {},
        "module": "home_credit_default_risk_services",
    },
    # Preprocessing (from preprocessing_services)
    {
        "service": "fit_column_filter",
        "inputs": {"data": "home-credit-default-risk/artifacts/train_merged.csv"},
        "outputs": {"artifact": "home-credit-default-risk/artifacts/columns_to_keep.json"},
        "params": {"missing_threshold": 0.7, "exclude_columns": ["TARGET", "SK_ID_CURR"]},
        "module": "preprocessing_services",
    },
    {
        "service": "transform_column_filter",
        "inputs": {
            "data": "home-credit-default-risk/artifacts/train_merged.csv",
            "artifact": "home-credit-default-risk/artifacts/columns_to_keep.json",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/train_filtered.csv"},
        "module": "preprocessing_services",
    },
    {
        "service": "transform_column_filter",
        "inputs": {
            "data": "home-credit-default-risk/artifacts/test_merged.csv",
            "artifact": "home-credit-default-risk/artifacts/columns_to_keep.json",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/test_filtered.csv"},
        "module": "preprocessing_services",
    },
    {
        "service": "fit_imputer",
        "inputs": {"data": "home-credit-default-risk/artifacts/train_filtered.csv"},
        "outputs": {"artifact": "home-credit-default-risk/artifacts/imputer.pkl"},
        "params": {"exclude_columns": ["TARGET", "SK_ID_CURR"]},
        "module": "preprocessing_services",
    },
    {
        "service": "transform_imputer",
        "inputs": {
            "data": "home-credit-default-risk/artifacts/train_filtered.csv",
            "artifact": "home-credit-default-risk/artifacts/imputer.pkl",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/train_imputed.csv"},
        "module": "preprocessing_services",
    },
    {
        "service": "transform_imputer",
        "inputs": {
            "data": "home-credit-default-risk/artifacts/test_filtered.csv",
            "artifact": "home-credit-default-risk/artifacts/imputer.pkl",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/test_imputed.csv"},
        "module": "preprocessing_services",
    },
    {
        "service": "fit_encoder",
        "inputs": {"data": "home-credit-default-risk/artifacts/train_imputed.csv"},
        "outputs": {"artifact": "home-credit-default-risk/artifacts/encoder.pkl"},
        "params": {
            "method": "ordinal",
            "exclude_columns": ["TARGET", "SK_ID_CURR"],
            "max_categories": 30,
        },
        "module": "preprocessing_services",
    },
    {
        "service": "transform_encoder",
        "inputs": {
            "data": "home-credit-default-risk/artifacts/train_imputed.csv",
            "artifact": "home-credit-default-risk/artifacts/encoder.pkl",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/train_encoded.csv"},
        "module": "preprocessing_services",
    },
    {
        "service": "transform_encoder",
        "inputs": {
            "data": "home-credit-default-risk/artifacts/test_imputed.csv",
            "artifact": "home-credit-default-risk/artifacts/encoder.pkl",
        },
        "outputs": {"data": "home-credit-default-risk/artifacts/test_encoded.csv"},
        "module": "preprocessing_services",
    },
    {
        "service": "split_data",
        "inputs": {"data": "home-credit-default-risk/artifacts/train_encoded.csv"},
        "outputs": {
            "train_data": "home-credit-default-risk/artifacts/train_split.csv",
            "valid_data": "home-credit-default-risk/artifacts/valid_split.csv",
        },
        "params": {"test_size": 0.2, "random_state": 42},
        "module": "preprocessing_services",
    },
    # Training + prediction (from classification_services)
    {
        "service": "train_lightgbm_classifier",
        "inputs": {
            "train_data": "home-credit-default-risk/artifacts/train_split.csv",
            "valid_data": "home-credit-default-risk/artifacts/valid_split.csv",
        },
        "outputs": {
            "model": "home-credit-default-risk/artifacts/model.pkl",
            "metrics": "home-credit-default-risk/artifacts/metrics.json",
        },
        "params": {
            "label_column": "TARGET",
            "id_column": "SK_ID_CURR",
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "num_leaves": 58,
            "max_depth": 11,
            "min_child_samples": 100,
            "subsample": 0.708,
            "colsample_bytree": 0.613,
            "reg_alpha": 3.564,
            "reg_lambda": 4.930,
            "random_state": 42,
            "early_stopping_rounds": 300,
        },
        "module": "classification_services",
    },
    {
        "service": "predict_classifier",
        "inputs": {
            "model": "home-credit-default-risk/artifacts/model.pkl",
            "data": "home-credit-default-risk/artifacts/test_encoded.csv",
        },
        "outputs": {"predictions": "home-credit-default-risk/artifacts/predictions.csv"},
        "params": {
            "id_column": "SK_ID_CURR",
            "prediction_column": "TARGET",
            "positive_class": 1,
            "proba_as_prediction": True,
        },
        "module": "classification_services",
    },
    {
        "service": "create_submission",
        "inputs": {"predictions": "home-credit-default-risk/artifacts/predictions.csv"},
        "outputs": {"submission": "home-credit-default-risk/submission.csv"},
        "params": {"id_column": "SK_ID_CURR", "prediction_column": "TARGET"},
        "module": "preprocessing_services",
    },
]


# =============================================================================
# COMPETITION-SPECIFIC SERVICES
# =============================================================================

def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safe element-wise division returning NaN where denominator is zero/NaN."""
    numer = numer.astype(float)
    denom = denom.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numer / denom
    result[(denom == 0) | denom.isna()] = np.nan
    return result

@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Fix anomalous DAYS_EMPLOYED values and add a flag feature",
    tags=["preprocessing", "cleaning", "feature-engineering"],
)
def replace_value_with_nan_and_flag(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    days_employed_col: str = "DAYS_EMPLOYED",
    anomaly_value: int = 365243,
    flag_column: str = "DAYS_EMPLOYED_ANOM",
) -> str:
    """Replace anomalous DAYS_EMPLOYED values with NaN and add an indicator flag."""
    df = pd.read_csv(inputs["data"])

    if days_employed_col in df.columns:
        if flag_column:
            df[flag_column] = (df[days_employed_col] == anomaly_value).astype(int)
        df.loc[df[days_employed_col] == anomaly_value, days_employed_col] = np.nan

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    return f"replace_value_with_nan_and_flag: {days_employed_col} cleaned"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Prepare application data with core feature engineering",
    tags=["preprocessing", "feature-engineering", "tabular"],
)
def prepare_tabular_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    drop_invalid_gender: bool = True,
    invalid_gender_value: str = "XNA",
    replace_zero_phone_change: bool = True,
    add_document_stats: bool = True,
    add_ext_source_stats: bool = True,
    add_ratio_features: bool = True,
) -> str:
    """Apply common Home Credit feature engineering to application data."""
    df = pd.read_csv(inputs["data"])

    if drop_invalid_gender and "CODE_GENDER" in df.columns:
        df = df[df["CODE_GENDER"] != invalid_gender_value]

    if replace_zero_phone_change and "DAYS_LAST_PHONE_CHANGE" in df.columns:
        df.loc[df["DAYS_LAST_PHONE_CHANGE"] == 0, "DAYS_LAST_PHONE_CHANGE"] = np.nan

    if add_document_stats:
        doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        if doc_cols:
            df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
            df["DOCUMENT_KURTOSIS"] = df[doc_cols].kurtosis(axis=1)

    if add_ext_source_stats:
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
        if ext_cols:
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
            df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
            df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
            df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
            # Advanced EXT_SOURCE features from top solutions
            if {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(df.columns):
                df["EXT_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
                df["EXT_SOURCES_WEIGHTED"] = df["EXT_SOURCE_1"] * 2 + df["EXT_SOURCE_2"] * 1 + df["EXT_SOURCE_3"] * 3
                df["EXT_SOURCE_1_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
                df["EXT_SOURCE_1_3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
                df["EXT_SOURCE_2_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
            # EXT_SOURCE with DAYS_BIRTH interactions
            if "DAYS_BIRTH" in df.columns:
                for i in [1, 2, 3]:
                    src_col = f"EXT_SOURCE_{i}"
                    if src_col in df.columns:
                        df[f"EXT_SOURCE_{i}_TO_BIRTH_RATIO"] = _safe_div(df[src_col], df["DAYS_BIRTH"] / 365.25)

    # AGE_RANGE feature from top solutions
    if "DAYS_BIRTH" in df.columns:
        def get_age_label(days_birth):
            age_years = -days_birth / 365 if pd.notna(days_birth) else 0
            if age_years < 27: return 1
            elif age_years < 40: return 2
            elif age_years < 50: return 3
            elif age_years < 65: return 4
            elif age_years < 99: return 5
            else: return 0
        df["AGE_RANGE"] = df["DAYS_BIRTH"].apply(get_age_label)

    if add_ratio_features:
        if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
            df["DAYS_EMPLOYED_PERC"] = _safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
        if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df.columns):
            df["INCOME_CREDIT_PERC"] = _safe_div(df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"])
        if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
            df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
        if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["ANNUITY_INCOME_PERC"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
        if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
            df["PAYMENT_RATE"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
            df["CREDIT_TERM"] = _safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])
        if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
            df["CREDIT_GOODS_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
        if {"AMT_INCOME_TOTAL", "CNT_CHILDREN"}.issubset(df.columns):
            df["INCOME_PER_CHILD"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_CHILDREN"] + 1)
        # Additional ratio features from top solutions
        if {"AMT_INCOME_TOTAL", "DAYS_EMPLOYED"}.issubset(df.columns):
            df["INCOME_TO_EMPLOYED_RATIO"] = _safe_div(df["AMT_INCOME_TOTAL"], df["DAYS_EMPLOYED"])
        if {"AMT_INCOME_TOTAL", "DAYS_BIRTH"}.issubset(df.columns):
            df["INCOME_TO_BIRTH_RATIO"] = _safe_div(df["AMT_INCOME_TOTAL"], df["DAYS_BIRTH"])
        if {"DAYS_ID_PUBLISH", "DAYS_BIRTH"}.issubset(df.columns):
            df["ID_TO_BIRTH_RATIO"] = _safe_div(df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"])
        if {"OWN_CAR_AGE", "DAYS_BIRTH"}.issubset(df.columns):
            df["CAR_TO_BIRTH_RATIO"] = _safe_div(df["OWN_CAR_AGE"], df["DAYS_BIRTH"])
        if {"OWN_CAR_AGE", "DAYS_EMPLOYED"}.issubset(df.columns):
            df["CAR_TO_EMPLOYED_RATIO"] = _safe_div(df["OWN_CAR_AGE"], df["DAYS_EMPLOYED"])
        if {"DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH"}.issubset(df.columns):
            df["PHONE_TO_BIRTH_RATIO"] = _safe_div(df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"])
        if {"AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["GOODS_INCOME_RATIO"] = _safe_div(df["AMT_GOODS_PRICE"], df["AMT_INCOME_TOTAL"])
        if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
            df["EMPLOYED_BIRTH_DIFF"] = df["DAYS_EMPLOYED"] - df["DAYS_BIRTH"]
        if {"AMT_INCOME_TOTAL", "AMT_ANNUITY"}.issubset(df.columns):
            df["INCOME_12_ANNUITY_DIFF"] = df["AMT_INCOME_TOTAL"] / 12.0 - df["AMT_ANNUITY"]

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    return f"prepare_tabular_features: {df.shape[1]} features"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Create ratio features from application data",
    tags=["feature-engineering", "ratios", "tabular"],
)
def add_ratio_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    ratio_specs: List[Dict[str, str]] = None,
    min_denominator: float = 1e-6,
) -> str:
    """Add common credit application ratio features."""
    df = pd.read_csv(inputs["data"])

    if ratio_specs is None:
        ratio_specs = [
            {"numerator": "AMT_CREDIT", "denominator": "AMT_INCOME_TOTAL", "output": "CREDIT_INCOME_RATIO"},
            {"numerator": "AMT_ANNUITY", "denominator": "AMT_INCOME_TOTAL", "output": "ANNUITY_INCOME_RATIO"},
            {"numerator": "AMT_CREDIT", "denominator": "AMT_ANNUITY", "output": "CREDIT_ANNUITY_RATIO"},
            {"numerator": "AMT_GOODS_PRICE", "denominator": "AMT_CREDIT", "output": "GOODS_CREDIT_RATIO"},
        ]

    created = 0
    for spec in ratio_specs:
        num = spec.get("numerator")
        den = spec.get("denominator")
        out = spec.get("output")
        if not num or not den or not out:
            continue
        if num in df.columns and den in df.columns:
            denom = df[den].astype(float)
            numerator = df[num].astype(float)
            safe_den = denom.where(denom.abs() > min_denominator, np.nan)
            df[out] = numerator / safe_den
            created += 1

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    return f"add_ratio_features: created {created} features"


@contract(
    inputs={
        "bureau_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "bureau_balance_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Aggregate bureau and bureau_balance tables by SK_ID_CURR",
    tags=["feature-engineering", "aggregation", "tabular"],
)
def aggregate_table_with_child(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    include_balance: bool = True,
    agg_functions: List[str] = None,
) -> str:
    """Aggregate bureau data (optionally with bureau_balance) to SK_ID_CURR level."""
    bureau = pd.read_csv(inputs["bureau_data"])

    if include_balance and inputs.get("bureau_balance_data") and os.path.exists(inputs["bureau_balance_data"]):
        bb = pd.read_csv(inputs["bureau_balance_data"])
        if "STATUS" in bb.columns:
            status_dummies = pd.get_dummies(bb["STATUS"], prefix="STATUS", dummy_na=True)
            bb = pd.concat([bb.drop(columns=["STATUS"]), status_dummies], axis=1)
        bb_agg = {"MONTHS_BALANCE": ["min", "max", "mean", "size"]}
        for col in [c for c in bb.columns if c.startswith("STATUS_")]:
            bb_agg[col] = ["mean"]
        bb_grouped = bb.groupby("SK_ID_BUREAU").agg(bb_agg)
        bb_grouped.columns = [f"BB_{c[0]}_{c[1].upper()}" for c in bb_grouped.columns]
        bureau = bureau.merge(bb_grouped, on="SK_ID_BUREAU", how="left")

    if {"DAYS_CREDIT", "DAYS_CREDIT_ENDDATE"}.issubset(bureau.columns):
        bureau["CREDIT_DURATION"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    if {"AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"}.issubset(bureau.columns):
        bureau["DEBT_PERCENTAGE"] = _safe_div(bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"])
        bureau["DEBT_CREDIT_DIFF"] = bureau["AMT_CREDIT_SUM"] - bureau["AMT_CREDIT_SUM_DEBT"]
    if {"AMT_CREDIT_SUM", "AMT_ANNUITY"}.issubset(bureau.columns):
        bureau["CREDIT_TO_ANNUITY_RATIO"] = _safe_div(bureau["AMT_CREDIT_SUM"], bureau["AMT_ANNUITY"])
    if {"AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM"}.issubset(bureau.columns):
        bureau["OVERDUE_RATIO"] = _safe_div(bureau["AMT_CREDIT_SUM_OVERDUE"], bureau["AMT_CREDIT_SUM"])
    if "CREDIT_DAY_OVERDUE" in bureau.columns:
        bureau["BUREAU_IS_DPD"] = (bureau["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    agg_functions = agg_functions or ["min", "max", "mean", "sum"]
    exclude_cols = {"SK_ID_BUREAU", "SK_ID_CURR"}
    numeric_cols = [c for c in bureau.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(bureau[c])]

    bureau_agg = bureau.groupby("SK_ID_CURR")[numeric_cols].agg(agg_functions)
    bureau_agg.columns = [f"BURO_{c[0]}_{c[1].upper()}" for c in bureau_agg.columns]
    bureau_agg["BURO_COUNT"] = bureau.groupby("SK_ID_CURR").size()
    bureau_agg = bureau_agg.reset_index()

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    bureau_agg.to_csv(outputs["data"], index=False)

    return f"aggregate_table_with_child: {bureau_agg.shape[1]} features"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Aggregate previous_application table by SK_ID_CURR",
    tags=["feature-engineering", "aggregation", "tabular"],
)
def aggregate_table_basic(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    agg_functions: List[str] = None,
) -> str:
    """Aggregate previous application data to SK_ID_CURR level."""
    prev = pd.read_csv(inputs["data"])

    if {"AMT_APPLICATION", "AMT_CREDIT"}.issubset(prev.columns):
        prev["APP_CREDIT_PERC"] = _safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])
        prev["APPLICATION_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(prev.columns):
        prev["CREDIT_TO_ANNUITY_RATIO"] = _safe_div(prev["AMT_CREDIT"], prev["AMT_ANNUITY"])
    if {"AMT_DOWN_PAYMENT", "AMT_CREDIT"}.issubset(prev.columns):
        prev["DOWN_PAYMENT_TO_CREDIT"] = _safe_div(prev["AMT_DOWN_PAYMENT"], prev["AMT_CREDIT"])

    agg_functions = agg_functions or ["min", "max", "mean", "sum"]
    exclude_cols = {"SK_ID_PREV", "SK_ID_CURR"}
    numeric_cols = [c for c in prev.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(prev[c])]

    prev_agg = prev.groupby("SK_ID_CURR")[numeric_cols].agg(agg_functions)
    prev_agg.columns = [f"PREV_{c[0]}_{c[1].upper()}" for c in prev_agg.columns]
    prev_agg["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size()
    prev_agg = prev_agg.reset_index()

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    prev_agg.to_csv(outputs["data"], index=False)

    return f"aggregate_table_basic: {prev_agg.shape[1]} features"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Aggregate POS_CASH balance table by SK_ID_CURR",
    tags=["feature-engineering", "aggregation", "tabular"],
)
def aggregate_table_delinquency(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    agg_functions: List[str] = None,
) -> str:
    """Aggregate POS_CASH balance to SK_ID_CURR level."""
    pos = pd.read_csv(inputs["data"])

    if "SK_DPD" in pos.columns:
        pos["LATE_PAYMENT"] = (pos["SK_DPD"] > 0).astype(int)
        pos["POS_IS_DPD_OVER_120"] = (pos["SK_DPD"] >= 120).astype(int)

    agg_functions = agg_functions or ["min", "max", "mean", "sum"]
    exclude_cols = {"SK_ID_PREV", "SK_ID_CURR"}
    numeric_cols = [c for c in pos.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(pos[c])]

    pos_agg = pos.groupby("SK_ID_CURR")[numeric_cols].agg(agg_functions)
    pos_agg.columns = [f"POS_{c[0]}_{c[1].upper()}" for c in pos_agg.columns]
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    pos_agg = pos_agg.reset_index()

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    pos_agg.to_csv(outputs["data"], index=False)

    return f"aggregate_table_delinquency: {pos_agg.shape[1]} features"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Aggregate installments_payments table by SK_ID_CURR",
    tags=["feature-engineering", "aggregation", "tabular"],
)
def aggregate_table_payments(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    agg_functions: List[str] = None,
) -> str:
    """Aggregate installments payments to SK_ID_CURR level."""
    ins = pd.read_csv(inputs["data"])

    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(ins.columns):
        ins["PAYMENT_PERC"] = _safe_div(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
        ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(ins.columns):
        ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
        ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
        ins["LATE_PAYMENT"] = (ins["DPD"] > 0).astype(int)

    agg_functions = agg_functions or ["min", "max", "mean", "sum"]
    exclude_cols = {"SK_ID_PREV", "SK_ID_CURR"}
    numeric_cols = [c for c in ins.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(ins[c])]

    ins_agg = ins.groupby("SK_ID_CURR")[numeric_cols].agg(agg_functions)
    ins_agg.columns = [f"INST_{c[0]}_{c[1].upper()}" for c in ins_agg.columns]
    ins_agg["INST_COUNT"] = ins.groupby("SK_ID_CURR").size()
    ins_agg = ins_agg.reset_index()

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    ins_agg.to_csv(outputs["data"], index=False)

    return f"aggregate_table_payments: {ins_agg.shape[1]} features"


@contract(
    inputs={"data": {"format": "csv", "required": True, "schema": {"type": "tabular"}}},
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Aggregate credit_card_balance table by SK_ID_CURR",
    tags=["feature-engineering", "aggregation", "tabular"],
)
def aggregate_table_revolving(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    agg_functions: List[str] = None,
) -> str:
    """Aggregate credit card balance to SK_ID_CURR level."""
    cc = pd.read_csv(inputs["data"])

    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(cc.columns):
        cc["LIMIT_USE"] = _safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    if {"AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY"}.issubset(cc.columns):
        cc["PAYMENT_DIV_MIN"] = _safe_div(cc["AMT_PAYMENT_CURRENT"], cc["AMT_INST_MIN_REGULARITY"])
    if "SK_DPD" in cc.columns:
        cc["LATE_PAYMENT"] = (cc["SK_DPD"] > 0).astype(int)

    agg_functions = agg_functions or ["min", "max", "mean", "sum"]
    exclude_cols = {"SK_ID_PREV", "SK_ID_CURR"}
    numeric_cols = [c for c in cc.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(cc[c])]

    cc_agg = cc.groupby("SK_ID_CURR")[numeric_cols].agg(agg_functions)
    cc_agg.columns = [f"CC_{c[0]}_{c[1].upper()}" for c in cc_agg.columns]
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    cc_agg = cc_agg.reset_index()

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    cc_agg.to_csv(outputs["data"], index=False)

    return f"aggregate_table_revolving: {cc_agg.shape[1]} features"


@contract(
    inputs={
        "base_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "bureau_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
        "prev_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
        "pos_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
        "installments_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
        "credit_card_data": {"format": "csv", "required": False, "schema": {"type": "tabular"}},
    },
    outputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
    description="Merge aggregated tables into base application data",
    tags=["data-handling", "merge", "tabular"],
)
def merge_feature_tables(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "SK_ID_CURR",
) -> str:
    """Merge aggregated tables into the base application dataset."""
    df = pd.read_csv(inputs["base_data"])

    merge_keys = [
        ("bureau_data", "bureau"),
        ("prev_data", "prev"),
        ("pos_data", "pos"),
        ("installments_data", "installments"),
        ("credit_card_data", "credit_card"),
    ]

    for input_key, _ in merge_keys:
        path = inputs.get(input_key)
        if path and os.path.exists(path):
            add_df = pd.read_csv(path)
            df = df.merge(add_df, on=id_column, how="left")

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)

    return f"merge_feature_tables: {df.shape[1]} features"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "replace_value_with_nan_and_flag": replace_value_with_nan_and_flag,
    "prepare_tabular_features": prepare_tabular_features,
    "aggregate_table_with_child": aggregate_table_with_child,
    "aggregate_table_basic": aggregate_table_basic,
    "aggregate_table_delinquency": aggregate_table_delinquency,
    "aggregate_table_payments": aggregate_table_payments,
    "aggregate_table_revolving": aggregate_table_revolving,
    "merge_feature_tables": merge_feature_tables,
    "fit_column_filter": fit_column_filter,
    "transform_column_filter": transform_column_filter,
    "fit_imputer": fit_imputer,
    "transform_imputer": transform_imputer,
    "fit_encoder": fit_encoder,
    "transform_encoder": transform_encoder,
    "split_data": split_data,
    "create_submission": create_submission,
    "train_lightgbm_classifier": train_lightgbm_classifier,
    "predict_classifier": predict_classifier,
}


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run_pipeline(base_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Run the Home Credit pipeline via PipelineRunner."""
    from pipeline_runner import PipelineRunner

    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slego_kb.sqlite")
    runner = PipelineRunner(
        db_path=db_path,
        verbose=verbose,
        storage=base_path,
        modules=[
            "home_credit_default_risk_services",
            "preprocessing_services",
            "classification_services",
        ],
    )
    return runner.run(PIPELINE_SPEC, base_path=base_path, pipeline_name="home-credit-default-risk")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Home Credit Default Risk Pipeline")
    parser.add_argument("--base-path", default="storage")
    args = parser.parse_args()

    print(f"\nRunning Home Credit Default Risk Pipeline from {args.base_path}\n")
    result = run_pipeline(args.base_path)
    print(json.dumps(result, indent=2))
