"""
Playground Series S5E1 - Contract-Composable Analytics Services
========================================
Competition: https://www.kaggle.com/competitions/playground-series-s5e1
Problem Type: Regression
Target: num_sold (daily number of stickers sold)
Metric: MAPE (Mean Absolute Percentage Error)

Data: Daily sticker sales across 6 countries, 3 stores, 5 products (2010-2019).
Train: 2010-2016, Test: 2017-2019.

Best Result: 0.13942 MAPE (rank 1250/2723, top 45.91%, BEATS 50%!)
Model: XGBoost 5-Fold CV (depth=10, min_child_weight=10, lr=0.02, 2000 trees)

Competition-specific services:
- prepare_s5e1_features: Feature engineering (date decomposition, cyclical encoding, label encoding)
- format_s5e1_submission: Format predictions for Kaggle submission
"""

import os
import sys
import json
import hashlib
import inspect
import numpy as np
import pandas as pd
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

from services.io_utils import load_data as _load_data, save_data as _save_data

# Import reusable generic services
try:
    from services.preprocessing_services import split_data, create_submission
    from services.regression_services import train_lightgbm_regressor, train_xgboost_regressor, predict_regressor
except ImportError:
    from preprocessing_services import split_data, create_submission
    from regression_services import train_lightgbm_regressor, train_xgboost_regressor, predict_regressor


@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Engineer features for sticker sales prediction from date/country/store/product",
    tags=["feature-engineering", "temporal", "categorical", "playground-series-s5e1"],
    version="1.0.0",
)
def prepare_s5e1_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    datetime_column: str = "date",
    target_column: Optional[str] = "num_sold",
    id_column: str = "id",
    drop_nulls: bool = True,
) -> str:
    """
    Prepare features for playground-series-s5e1 (sticker sales forecasting).

    Features:
    - Datetime: year, month, day, dayofweek, dayofyear, weekofyear, is_weekend
    - Cyclical: sin/cos of dayofyear (annual), sin/cos of dayofweek (weekly)
    - Categorical: label-encoded country, store, product
    - Interactions: country_product, store_product, country_store
    - Trend: daynum (days since 2010-01-01)
    """
    df = _load_data(inputs["data"])

    # Parse date
    dt = pd.to_datetime(df[datetime_column])

    # Datetime features
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding (annual)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_sin2"] = np.sin(4 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos2"] = np.cos(4 * np.pi * df["dayofyear"] / 365.25)

    # Cyclical encoding (weekly)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Label-encode categoricals
    country_map = {c: i for i, c in enumerate(sorted(df["country"].unique()))}
    store_map = {s: i for i, s in enumerate(sorted(df["store"].unique()))}
    product_map = {p: i for i, p in enumerate(sorted(df["product"].unique()))}

    df["country_enc"] = df["country"].map(country_map)
    df["store_enc"] = df["store"].map(store_map)
    df["product_enc"] = df["product"].map(product_map)

    # Interaction features
    n_products = len(product_map)
    n_stores = len(store_map)
    df["country_product"] = df["country_enc"] * n_products + df["product_enc"]
    df["store_product"] = df["store_enc"] * n_products + df["product_enc"]
    df["country_store"] = df["country_enc"] * n_stores + df["store_enc"]

    # Day number (trend)
    min_date = pd.Timestamp("2010-01-01")
    df["daynum"] = (dt - min_date).dt.days

    # Drop original string columns
    df = df.drop(columns=[datetime_column, "country", "store", "product"])

    # Drop nulls if specified
    if drop_nulls and target_column and target_column in df.columns:
        df = df.dropna(subset=[target_column])

    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    _save_data(df, outputs["data"])

    n_features = len([c for c in df.columns if c not in [id_column, target_column]])
    return f"prepare_s5e1_features: {len(df)} rows, {n_features} features"


@contract(
    inputs={"predictions": {"format": "csv", "required": True}},
    outputs={"submission": {"format": "csv"}},
    description="Format predictions into Kaggle submission for s5e1",
    tags=["inference", "submission", "playground-series-s5e1"],
    version="1.0.0",
)
def format_s5e1_submission(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "id",
    prediction_column: str = "num_sold",
) -> str:
    """Format predictions for playground-series-s5e1 Kaggle submission."""
    pred_df = _load_data(inputs["predictions"])

    submission = pd.DataFrame({
        id_column: pred_df[id_column].astype(int),
        prediction_column: np.clip(np.round(pred_df[prediction_column]), 1, None).astype(int),
    })

    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    _save_data(submission, outputs["submission"])

    return f"format_s5e1_submission: {len(submission)} rows, mean={submission[prediction_column].mean():.1f}"


# =============================================================================
# REGISTRY
# =============================================================================

SERVICE_REGISTRY = {
    "prepare_s5e1_features": prepare_s5e1_features,
    "format_s5e1_submission": format_s5e1_submission,
    # Reused from generic modules
    "split_data": split_data,
    "create_submission": create_submission,
    "train_lightgbm_regressor": train_lightgbm_regressor,
    "train_xgboost_regressor": train_xgboost_regressor,
    "predict_regressor": predict_regressor,
}


def register_to_kb():
    """Register all services in this module to the KB database."""
    import sqlite3

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb.sqlite"
    )
    if not os.path.exists(db_path):
        return f"Database not found at {db_path}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    local_services = {
        "prepare_s5e1_features": prepare_s5e1_features,
        "format_s5e1_submission": format_s5e1_submission,
    }

    for name, func in local_services.items():
        contract_data = getattr(func, "contract", {})
        version = contract_data.get("version", "1.0.0")
        description = contract_data.get("description", "")
        tags = json.dumps(contract_data.get("tags", []))
        input_contract = json.dumps(contract_data.get("inputs", {}))
        output_contract = json.dumps(contract_data.get("outputs", {}))

        try:
            source = inspect.getsource(func)
        except Exception:
            source = f"# Source not available for {name}"
        source_hash = hashlib.md5(source.encode()).hexdigest()

        sig = inspect.signature(func)
        params_dict = {}
        for p_name, p in sig.parameters.items():
            if p_name in ["inputs", "outputs"]:
                continue
            params_dict[p_name] = {
                "default": str(p.default) if p.default != inspect.Parameter.empty else None,
                "type": str(p.annotation) if p.annotation != inspect.Parameter.empty else "any",
            }
        parameters = json.dumps(params_dict)

        cursor.execute(
            """
            INSERT OR REPLACE INTO services
            (name, version, module, description, docstring, tags, category,
             input_contract, output_contract, parameters, source_code, source_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                name,
                version,
                "playground_series_s5e1_services",
                description,
                func.__doc__ or "",
                tags,
                "playground-series-s5e1",
                input_contract,
                output_contract,
                parameters,
                source,
                source_hash,
            ),
        )

    conn.commit()
    conn.close()
    return f"Registered {len(local_services)} services to KB."


if __name__ == "__main__":
    print("Registering services...")
    print(register_to_kb())
