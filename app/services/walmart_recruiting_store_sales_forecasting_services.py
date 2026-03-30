"""
Walmart Recruiting Store Sales Forecasting - SLEGO Services v4
================================================================
Competition: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
Problem Type: Regression (WMAE metric - Weighted Mean Absolute Error)
Target: Weekly_Sales
ID Column: Id (format: Store_Dept_Date)

v4 improvements from 5 top solution notebooks analysis:
- Mean encoding features: Store/Dept average sales by year (from XGB notebook)
- 3-model ensemble: RF + ExtraTrees + XGBoost blend (from XGB + top-3% notebooks)
- Temporal validation split: train on earlier weeks, validate on later (time series best practice)
- WMAE sample weights (5x for holidays)
- CPI/Unemployment median imputation
- Christmas week adjustment

Competition-specific services:
- prepare_walmart_data_v4: Preprocessing with mean encoding features
- prepare_walmart_test_v4: Test preprocessing with encoded features
- temporal_split_walmart: Time-based train/valid split
- train_walmart_triple_ensemble: RF + ExtraTrees + XGBoost with sample weights
- predict_walmart_triple_ensemble: Predict using 3-model blend
- create_walmart_submission: Format submission with Christmas adjustment
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slego_contract import contract
from services.io_utils import load_data as _load_data, save_data as _save_data


# =============================================================================
# HELPER: Feature engineering with mean encoding
# =============================================================================

def _engineer_walmart_features_v5(
    df: pd.DataFrame,
    stats: Dict = None,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    v5 feature engineering with top-3% solution features.

    Features added from top-3% solution notebook:
    - Day, Week, Month, Year from Date
    - Days_to_Thanksgiving, Days_to_Christmas (distance features)
    - SuperBowlWeek, LaborDay, Thanksgiving, Christmas (binary holiday features)
    - Easter holiday → IsHoliday
    - Type ordinal encoding (A=3, B=2, C=1)
    - Mean encoding: Store_avg_sales, Dept_avg_sales (from stats for test)
    - CPI/Unemployment median imputation
    """
    stats = stats or {}

    # --- Date Feature Extraction ---
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day

    # --- Holiday distance features (from top-3% solution) ---
    if "Year" in df.columns and "Date" in df.columns:
        # Days to Thanksgiving (Nov 24th approximation)
        df["Days_to_Thanksgiving"] = (
            pd.to_datetime(df["Year"].astype(str) + "-11-24") - df["Date"]
        ).dt.days.astype(int)
        # Days to Christmas (Dec 24th)
        df["Days_to_Christmas"] = (
            pd.to_datetime(df["Year"].astype(str) + "-12-24") - df["Date"]
        ).dt.days.astype(int)

    # --- Binary holiday features (from top-3% solution) ---
    if "Week" in df.columns:
        df["SuperBowlWeek"] = (df["Week"] == 6).astype(int)
        df["LaborDay"] = (df["Week"] == 36).astype(int)
        df["ThanksgivingWeek"] = (df["Week"] == 47).astype(int)
        df["ChristmasWeek"] = (df["Week"] == 52).astype(int)

    # --- Easter Holiday ---
    if "Week" in df.columns and "Year" in df.columns and "IsHoliday" in df.columns:
        easter_weeks = {2010: 13, 2011: 16, 2012: 14, 2013: 13}
        for year, week in easter_weeks.items():
            mask = (df["Year"] == year) & (df["Week"] == week)
            df.loc[mask, "IsHoliday"] = True

    # --- Ordinal Encode Type ---
    if "Type" in df.columns:
        type_map = {"A": 3, "B": 2, "C": 1}
        df["Type"] = df["Type"].map(type_map).fillna(0).astype(int)

    # --- Encode IsHoliday ---
    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(int)

    # --- Mean Encoding: Store average sales ---
    if is_train and "Weekly_Sales" in df.columns:
        store_avg = df.groupby("Store")["Weekly_Sales"].mean().to_dict()
        df["Store_avg_sales"] = df["Store"].map(store_avg)
    elif "store_avg_sales" in stats:
        df["Store_avg_sales"] = df["Store"].map(stats["store_avg_sales"])
        df["Store_avg_sales"] = df["Store_avg_sales"].fillna(stats.get("global_avg_sales", 0))

    # --- Mean Encoding: Dept average sales ---
    if is_train and "Weekly_Sales" in df.columns:
        dept_avg = df.groupby("Dept")["Weekly_Sales"].mean().to_dict()
        df["Dept_avg_sales"] = df["Dept"].map(dept_avg)
    elif "dept_avg_sales" in stats:
        df["Dept_avg_sales"] = df["Dept"].map(stats["dept_avg_sales"])
        df["Dept_avg_sales"] = df["Dept_avg_sales"].fillna(stats.get("global_avg_sales", 0))

    # --- Mean Encoding: Store-Dept combo average sales ---
    if is_train and "Weekly_Sales" in df.columns:
        store_dept_avg = df.groupby(["Store", "Dept"])["Weekly_Sales"].mean().to_dict()
        df["Store_Dept_avg"] = df.apply(lambda r: store_dept_avg.get((r["Store"], r["Dept"]), 0), axis=1)
    elif "store_dept_avg_sales" in stats:
        df["Store_Dept_avg"] = df.apply(
            lambda r: stats["store_dept_avg_sales"].get(f"{int(r['Store'])}_{int(r['Dept'])}",
                      stats.get("global_avg_sales", 0)), axis=1
        )

    # --- CPI/Unemployment median fill ---
    if "CPI" in df.columns:
        fill_val = stats.get("cpi_median", df["CPI"].median())
        df["CPI"] = df["CPI"].fillna(fill_val)
    if "Unemployment" in df.columns:
        fill_val = stats.get("unemployment_median", df["Unemployment"].median())
        df["Unemployment"] = df["Unemployment"].fillna(fill_val)

    # --- Fill remaining numeric NaN ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


# =============================================================================
# SERVICE 1: Prepare Training Data v4
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "features_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "stores_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
        "stats": {"format": "json", "schema": {"type": "json"}},
    },
    description="Prepare Walmart training data v4: merge, mean encoding features, sample weights",
    tags=["preprocessing", "feature-engineering", "walmart", "retail", "time-series"],
    version="4.0.0",
)
def prepare_walmart_data(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "Weekly_Sales",
) -> str:
    """
    v4 preprocessing with mean encoding features.

    New features from solution notebooks:
    - Store_avg_sales: Historical average sales per store
    - Dept_avg_sales: Historical average sales per department
    - Store_Dept_avg: Historical average per store-dept combo
    """
    train = _load_data(inputs["train_data"])
    features = _load_data(inputs["features_data"])
    stores = _load_data(inputs["stores_data"])

    # Merge
    feat_store = features.merge(stores, how="inner", on="Store")
    df = train.merge(
        feat_store, how="inner", on=["Store", "Date", "IsHoliday"]
    ).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

    n_before = len(df)

    # Compute stats for test data imputation
    cpi_median = float(df["CPI"].median())
    unemp_median = float(df["Unemployment"].median())
    global_avg_sales = float(df[target_column].mean())
    store_avg = df.groupby("Store")[target_column].mean().to_dict()
    dept_avg = df.groupby("Dept")[target_column].mean().to_dict()
    store_dept_avg = {f"{k[0]}_{k[1]}": v for k, v in
                      df.groupby(["Store", "Dept"])[target_column].mean().to_dict().items()}

    # Feature engineering
    df = _engineer_walmart_features_v5(df, is_train=True)

    # Add sample weights for WMAE
    df["sample_weight"] = df["IsHoliday"].apply(lambda x: 5.0 if x == 1 else 1.0)

    # Keep Date for temporal split (v5: added new features from top-3% solution)
    keep_cols = [
        target_column, "Date",
        "Store", "Dept", "IsHoliday", "Size", "Week", "Type", "Year",
        "Month", "Day",  # v5: added
        "Days_to_Thanksgiving", "Days_to_Christmas",  # v5: added
        "SuperBowlWeek", "LaborDay", "ThanksgivingWeek", "ChristmasWeek",  # v5: added
        "CPI", "Unemployment",
        "Store_avg_sales", "Dept_avg_sales", "Store_Dept_avg",
        "sample_weight",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    _save_data(df, outputs["data"])

    # Save stats for test
    stats = {
        "cpi_median": cpi_median,
        "unemployment_median": unemp_median,
        "global_avg_sales": global_avg_sales,
        "store_avg_sales": {str(k): v for k, v in store_avg.items()},
        "dept_avg_sales": {str(k): v for k, v in dept_avg.items()},
        "store_dept_avg_sales": store_dept_avg,
        "n_rows": len(df),
        "n_features": len(keep_cols) - 3,  # minus target, Date, sample_weight
    }
    os.makedirs(os.path.dirname(outputs["stats"]) or ".", exist_ok=True)
    with open(outputs["stats"], "w") as f:
        json.dump(stats, f)

    return f"prepare_walmart_data: {n_before} -> {len(df)} rows ({len(df.columns)} cols)"


# =============================================================================
# SERVICE 2: Temporal Split for Time Series
# =============================================================================

@contract(
    inputs={
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "train_data": {"format": "csv", "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Temporal train/valid split for time series: train on earlier weeks, validate on later",
    tags=["preprocessing", "split", "temporal", "time-series"],
    version="1.0.0",
)
def temporal_split_walmart(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    date_column: str = "Date",
    valid_weeks: int = 8,
) -> str:
    """
    Time-based split: train on all but last N weeks, validate on last N weeks.

    This prevents data leakage from future to past, which is critical for
    time series forecasting competitions.

    Parameters:
        date_column: Column containing dates
        valid_weeks: Number of weeks to hold out for validation (default: 8)
    """
    df = _load_data(inputs["data"])

    df[date_column] = pd.to_datetime(df[date_column])
    max_date = df[date_column].max()
    split_date = max_date - pd.Timedelta(weeks=valid_weeks)

    train_df = df[df[date_column] <= split_date].copy()
    valid_df = df[df[date_column] > split_date].copy()

    # Drop Date column after split (not needed for modeling)
    if date_column in train_df.columns:
        train_df = train_df.drop(columns=[date_column])
        valid_df = valid_df.drop(columns=[date_column])

    _save_data(train_df, outputs["train_data"])
    _save_data(valid_df, outputs["valid_data"])

    return f"temporal_split_walmart: train={len(train_df)}, valid={len(valid_df)} (last {valid_weeks} weeks)"


# =============================================================================
# SERVICE 3: Prepare Test Data v4
# =============================================================================

@contract(
    inputs={
        "test_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "features_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "stores_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "stats": {"format": "json", "required": True, "schema": {"type": "json"}},
    },
    outputs={
        "data": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Prepare Walmart test data v4: merge, apply mean encoding from train stats",
    tags=["preprocessing", "feature-engineering", "walmart", "retail", "time-series"],
    version="4.0.0",
)
def prepare_walmart_test(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """v4 test preprocessing using train statistics for mean encoding."""
    test = _load_data(inputs["test_data"])
    features = _load_data(inputs["features_data"])
    stores = _load_data(inputs["stores_data"])

    with open(inputs["stats"]) as f:
        stats = json.load(f)

    # Convert string keys back to int for store/dept lookups
    stats["store_avg_sales"] = {int(k): v for k, v in stats["store_avg_sales"].items()}
    stats["dept_avg_sales"] = {int(k): v for k, v in stats["dept_avg_sales"].items()}

    # Merge
    feat_store = features.merge(stores, how="inner", on="Store")
    df = test.merge(
        feat_store, how="inner", on=["Store", "Date", "IsHoliday"]
    ).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

    # Build Id
    df["Id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str) + "_" + df["Date"].astype(str)

    # Feature engineering with stats
    df = _engineer_walmart_features_v5(df, stats=stats, is_train=False)

    # v5: added new features from top-3% solution
    keep_cols = [
        "Id",
        "Store", "Dept", "IsHoliday", "Size", "Week", "Type", "Year",
        "Month", "Day",  # v5: added
        "Days_to_Thanksgiving", "Days_to_Christmas",  # v5: added
        "SuperBowlWeek", "LaborDay", "ThanksgivingWeek", "ChristmasWeek",  # v5: added
        "CPI", "Unemployment",
        "Store_avg_sales", "Dept_avg_sales", "Store_Dept_avg",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    _save_data(df, outputs["data"])
    return f"prepare_walmart_test: {len(df)} rows, {len(df.columns)} cols"


# =============================================================================
# SERVICE 4: Train Triple Ensemble (RF + ETR + XGB)
# =============================================================================

@contract(
    inputs={
        "train_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
        "valid_data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "model": {"format": "pickle", "schema": {"type": "artifact", "artifact_type": "ensemble_model"}},
        "metrics": {"format": "json", "schema": {"type": "json"}},
        "feature_importance": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Train RF + ExtraTrees + XGBoost ensemble with WMAE sample weights",
    tags=["modeling", "training", "ensemble", "walmart", "retail"],
    version="2.0.0",
)
def train_walmart_weighted_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    label_column: str = "Weekly_Sales",
    weight_column: str = "sample_weight",
    # RF params (from solution 1)
    rf_n_estimators: int = 100,
    rf_max_depth: int = 27,
    rf_max_features: int = 8,
    rf_min_samples_split: int = 3,
    rf_min_samples_leaf: int = 1,
    # ETR params (from top-3% solution)
    etr_n_estimators: int = 100,
    etr_max_depth: int = 25,
    etr_bootstrap: bool = True,
    # XGB params (from XGB notebook)
    xgb_n_estimators: int = 300,
    xgb_max_depth: int = 12,
    xgb_learning_rate: float = 0.1,
    # Blend weights
    blend_weight_rf: float = 0.35,
    blend_weight_etr: float = 0.35,
    blend_weight_xgb: float = 0.30,
    random_state: int = 42,
) -> str:
    """
    Train 3-model ensemble: RF + ExtraTrees + XGBoost.

    From solution notebook analysis:
    - RF and ETR provide good baseline (top-3% solution used this)
    - XGBoost adds diversity (XGB notebook achieved good WMAE)
    - 3-model blend reduces variance
    """
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    try:
        from xgboost import XGBRegressor
        use_xgb = True
    except ImportError:
        use_xgb = False
        # Redistribute weights
        blend_weight_rf = 0.5
        blend_weight_etr = 0.5
        blend_weight_xgb = 0.0

    train = _load_data(inputs["train_data"])
    valid = _load_data(inputs["valid_data"])

    exclude_cols = {label_column, weight_column}
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[label_column]
    X_valid = valid[feature_cols]
    y_valid = valid[label_column]

    w_train = train[weight_column].values if weight_column in train.columns else None
    w_valid = valid[weight_column].values if weight_column in valid.columns else None

    models = {}
    preds = {}

    # --- RandomForest ---
    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        max_features=rf_max_features,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=w_train)
    models["rf"] = rf
    preds["rf"] = rf.predict(X_valid)

    # --- ExtraTrees ---
    etr = ExtraTreesRegressor(
        n_estimators=etr_n_estimators,
        max_depth=etr_max_depth,
        bootstrap=etr_bootstrap,
        random_state=random_state,
        n_jobs=-1,
    )
    etr.fit(X_train, y_train, sample_weight=w_train)
    models["etr"] = etr
    preds["etr"] = etr.predict(X_valid)

    # --- XGBoost ---
    if use_xgb:
        xgb = XGBRegressor(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(X_train, y_train, sample_weight=w_train)
        models["xgb"] = xgb
        preds["xgb"] = xgb.predict(X_valid)

    # --- Blend ---
    blended = (
        blend_weight_rf * preds["rf"] +
        blend_weight_etr * preds["etr"] +
        (blend_weight_xgb * preds.get("xgb", np.zeros_like(preds["rf"])) if use_xgb else 0)
    )

    # --- Metrics ---
    rmse_rf = float(np.sqrt(mean_squared_error(y_valid, preds["rf"])))
    rmse_etr = float(np.sqrt(mean_squared_error(y_valid, preds["etr"])))
    rmse_xgb = float(np.sqrt(mean_squared_error(y_valid, preds["xgb"]))) if use_xgb else 0
    rmse_blend = float(np.sqrt(mean_squared_error(y_valid, blended)))

    if w_valid is not None:
        wmae = float(np.sum(w_valid * np.abs(y_valid.values - blended)) / np.sum(w_valid))
    else:
        wmae = float(mean_absolute_error(y_valid, blended))

    metrics = {
        "model_type": "walmart_triple_ensemble",
        "models": list(models.keys()),
        "blend_weights": {"rf": blend_weight_rf, "etr": blend_weight_etr, "xgb": blend_weight_xgb},
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "valid_rmse": rmse_blend,
        "valid_rmse_rf": rmse_rf,
        "valid_rmse_etr": rmse_etr,
        "valid_rmse_xgb": rmse_xgb,
        "valid_wmae": wmae,
    }

    # Save
    ensemble = {
        "models": models,
        "feature_cols": feature_cols,
        "blend_weights": {"rf": blend_weight_rf, "etr": blend_weight_etr, "xgb": blend_weight_xgb},
    }
    os.makedirs(os.path.dirname(outputs["model"]) or ".", exist_ok=True)
    with open(outputs["model"], "wb") as f:
        pickle.dump(ensemble, f)

    with open(outputs["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance (average)
    if "feature_importance" in outputs:
        importances = []
        weights = []
        for name, model in models.items():
            if hasattr(model, "feature_importances_"):
                importances.append(model.feature_importances_)
                weights.append(ensemble["blend_weights"][name])
        if importances:
            avg_imp = np.average(importances, axis=0, weights=weights)
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": avg_imp,
            }).sort_values("importance", ascending=False)
            importance_df.to_csv(outputs["feature_importance"], index=False)

    model_str = "+".join(models.keys()).upper()
    return f"train_walmart_ensemble: {model_str}, WMAE={wmae:.2f}, RMSE={rmse_blend:.2f}"


# =============================================================================
# SERVICE 5: Predict with Triple Ensemble
# =============================================================================

@contract(
    inputs={
        "model": {"format": "pickle", "required": True, "schema": {"type": "artifact"}},
        "data": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "predictions": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Generate predictions from Walmart RF+ETR+XGB ensemble",
    tags=["inference", "prediction", "walmart", "retail"],
    version="2.0.0",
)
def predict_walmart_ensemble(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "Weekly_Sales",
) -> str:
    """Predict using 3-model ensemble."""
    with open(inputs["model"], "rb") as f:
        ensemble = pickle.load(f)

    df = _load_data(inputs["data"])
    ids = df[id_column] if id_column in df.columns else pd.RangeIndex(len(df))

    feature_cols = ensemble["feature_cols"]
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]

    # Blend
    blended = np.zeros(len(X))
    for name, model in ensemble["models"].items():
        weight = ensemble["blend_weights"].get(name, 0)
        if weight > 0:
            blended += weight * model.predict(X)

    blended = np.maximum(blended, 0)

    pred_df = pd.DataFrame({id_column: ids, prediction_column: blended})
    _save_data(pred_df, outputs["predictions"])

    return f"predict_walmart_ensemble: {len(blended)} predictions, mean={blended.mean():.2f}"


# =============================================================================
# SERVICE 6: Create Submission with Christmas Adjustment
# =============================================================================

@contract(
    inputs={
        "predictions": {"format": "csv", "required": True, "schema": {"type": "tabular"}},
    },
    outputs={
        "submission": {"format": "csv", "schema": {"type": "tabular"}},
    },
    description="Create Walmart submission with Christmas week adjustment",
    tags=["inference", "submission", "walmart", "retail"],
    version="1.0.0",
)
def create_walmart_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "Id",
    prediction_column: str = "Weekly_Sales",
    apply_christmas_adjustment: bool = True,
) -> str:
    """Create submission with Christmas week adjustment."""
    preds = _load_data(inputs["predictions"])

    id_parts = preds[id_column].str.split("_", expand=True)
    preds["Store"] = id_parts[0].astype(int)
    preds["Dept"] = id_parts[1].astype(int)
    preds["Date"] = id_parts[2]
    preds["Week"] = pd.to_datetime(preds["Date"]).dt.isocalendar().week.astype(int)

    n_adjusted = 0
    if apply_christmas_adjustment:
        for (store, dept), _ in preds.groupby(["Store", "Dept"]):
            w51_mask = (preds["Store"] == store) & (preds["Dept"] == dept) & (preds["Week"] == 51)
            w52_mask = (preds["Store"] == store) & (preds["Dept"] == dept) & (preds["Week"] == 52)

            w51_sales = preds.loc[w51_mask, prediction_column]
            w52_sales = preds.loc[w52_mask, prediction_column]

            if len(w51_sales) > 0 and len(w52_sales) > 0:
                w51_val = w51_sales.values[0]
                w52_val = w52_sales.values[0]
                if w51_val > 2 * w52_val:
                    preds.loc[w52_mask, prediction_column] = w52_val + (2.5 / 7.0) * w51_val
                    n_adjusted += 1

    submission = preds[[id_column, prediction_column]].copy()
    _save_data(submission, outputs["submission"])

    return f"create_walmart_submission: {len(submission)} rows, {n_adjusted} Christmas adjustments"


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "prepare_walmart_data": prepare_walmart_data,
    "prepare_walmart_test": prepare_walmart_test,
    "temporal_split_walmart": temporal_split_walmart,
    "train_walmart_weighted_ensemble": train_walmart_weighted_ensemble,
    "predict_walmart_ensemble": predict_walmart_ensemble,
    "create_walmart_submission": create_walmart_submission,
}
