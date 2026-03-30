"""
Bike Sharing Demand - Contract-Composable Analytics Services (Generalized)
==================================================

Competition: https://www.kaggle.com/competitions/bike-sharing-demand
Problem Type: Regression
Target: count (hourly bike rental count)

Refactored to leverage generic Contract-Composable Analytics services for modeling and preprocessing.
Follows G1-G6 principles:
- G1: Generic services (extract_datetime_features, impute_zeros_with_regressor)
- G2: Single responsibility (specific bike features separated from generic ones)
- G3: Reproducible (fixed random_state)
- G4: Parameterized (datetime columns and targets are configurable)
- G5: Loose coupling (file-based I/O)
- G6: Contract-based (@contract decorator)
"""

import os
import sys
import json
import hashlib
import inspect
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Setup path for internal imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import contract

# Import generic services for reuse
try:
    from services.preprocessing_services import split_data, create_submission
    from services.regression_services import train_ensemble_regressor, predict_ensemble_regressor
except ImportError:
    from preprocessing_services import split_data, create_submission
    from regression_services import train_ensemble_regressor, predict_ensemble_regressor


# =============================================================================
# GENERIC SERVICES (Extracted for reuse)
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Extract components from a datetime column (year, month, day, hour, etc.)",
    tags=["preprocessing", "feature-engineering", "temporal", "generic"],
    version="1.1.0"
)
def extract_datetime_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    datetime_column: str = "datetime",
    features: List[str] = None,
) -> str:
    """Extract temporal components from a datetime column."""
    df = pd.read_csv(inputs["data"])
    dt = pd.to_datetime(df[datetime_column])
    
    features = features or ['year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend']
    
    if 'year' in features: df['year'] = dt.dt.year
    if 'month' in features: df['month'] = dt.dt.month
    if 'day' in features: df['day'] = dt.dt.day
    if 'hour' in features: df['hour'] = dt.dt.hour
    if 'dayofweek' in features: df['dayofweek'] = dt.dt.dayofweek
    if 'is_weekend' in features: df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    
    # Store datetime as string for ID preservation in subsequent steps
    df[f"{datetime_column}_str"] = df[datetime_column].astype(str)
    
    # Drop original datetime column to avoid issues with models
    df = df.drop(columns=[datetime_column])
    
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"extract_datetime_features: extracted {len(features)} components for {len(df)} rows"

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Impute zero values in a column using a regressor based on other features",
    tags=["preprocessing", "imputation", "generic"],
    version="1.1.0"
)
def impute_zeros_with_regressor(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    target_column: str = "windspeed",
    feature_columns: List[str] = None,
) -> str:
    """Generic imputation for columns where 0 is likely a missing value (e.g. windspeed)."""
    from sklearn.ensemble import RandomForestRegressor
    df = pd.read_csv(inputs["data"])
    
    features = feature_columns or ['temp', 'atemp', 'humidity', 'month']
    features = [f for f in features if f in df.columns]
    
    df_known = df[df[target_column] != 0].copy()
    df_zero = df[df[target_column] == 0].copy()
    
    if len(df_zero) > 0 and len(df_known) > 0:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(df_known[features], df_known[target_column])
        df.loc[df[target_column] == 0, target_column] = rf.predict(df_zero[features])
        
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"impute_zeros_with_regressor: treated {len(df_zero)} zeros in {target_column}"

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Drop specific columns from a dataframe",
    tags=["preprocessing", "generic"],
    version="1.0.0"
)
def drop_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    columns: List[str] = [],
) -> str:
    """Generic service to drop specified columns."""
    df = pd.read_csv(inputs["data"])
    to_drop = [c for c in columns if c in df.columns]
    df = df.drop(columns=to_drop)
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"drop_columns: dropped {len(to_drop)} columns"

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Rename columns in a dataframe",
    tags=["preprocessing", "generic"],
    version="1.0.0"
)
def rename_columns(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    rename_dict: Dict[str, str] = {},
) -> str:
    """Generic service to rename columns."""
    df = pd.read_csv(inputs["data"])
    df = df.rename(columns=rename_dict)
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"rename_columns: renamed {len(rename_dict)} columns"

# =============================================================================
# BIKE-SHARING SPECIFIC SERVICES
# =============================================================================

@contract(
    inputs={"data": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Create peak-hour and weather features specific to bike demand",
    tags=["feature-engineering", "bike-sharing"],
    version="1.1.0"
)
def add_bike_domain_features(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
) -> str:
    """Bike-sharing specific features: peak hours, ideal/sticky weather, cyclicals."""
    df = pd.read_csv(inputs["data"])
    
    # Peak hours: Commute times on working days, midday on weekends
    df['is_peak'] = 0
    df.loc[(df['workingday'] == 1) & (df['hour'].isin([7, 8, 9, 17, 18, 19])), 'is_peak'] = 1
    df.loc[(df['workingday'] == 0) & (df['hour'].between(10, 16)), 'is_peak'] = 1
    
    # Weather comfort indices
    df['is_ideal'] = ((df['temp'].between(15, 30)) & (df['humidity'] < 60)).astype(int)
    df['is_sticky'] = ((df['humidity'] > 80) & (df['temp'] > 25)).astype(int)
    
    # Cyclical hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    df.to_csv(outputs["data"], index=False)
    return f"add_bike_domain_features: added domain specific features (is_peak, etc.) for {len(df)} rows"

@contract(
    inputs={"pred_a": {"format": "csv", "required": True}, "pred_b": {"format": "csv", "required": True}},
    outputs={"data": {"format": "csv"}},
    description="Combine two prediction columns (e.g. casual + registered)",
    tags=["inference", "bike-sharing"],
    version="1.1.0"
)
def sum_predictions(
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    id_column: str = "datetime_str",
    output_column: str = "count",
) -> str:
    """Sum two predictions. Specific to the 'casual + registered' model strategy."""
    df_a = pd.read_csv(inputs["pred_a"])
    df_b = pd.read_csv(inputs["pred_b"])
    
    # Ensure they match on ID
    df_b = df_b.set_index(id_column).reindex(df_a[id_column]).reset_index()
    
    combined = pd.DataFrame({
        id_column: df_a[id_column],
        output_column: np.round(df_a.iloc[:, 1] + df_b.iloc[:, 1]).astype(int)
    })
    
    os.makedirs(os.path.dirname(outputs["data"]) or ".", exist_ok=True)
    combined.to_csv(outputs["data"], index=False)
    return f"sum_predictions: combined into {output_column} for {len(combined)} rows"


# =============================================================================
# REGISTRY & PIPELINE
# =============================================================================

SERVICE_REGISTRY = {
    "extract_datetime_features": extract_datetime_features,
    "impute_zeros_with_regressor": impute_zeros_with_regressor,
    "drop_columns": drop_columns,
    "rename_columns": rename_columns,
    "add_bike_domain_features": add_bike_domain_features,
    "sum_predictions": sum_predictions,
    "split_data": split_data,
    "train_ensemble_regressor": train_ensemble_regressor,
    "predict_ensemble_regressor": predict_ensemble_regressor,
    "create_submission": create_submission,
}

# The pipeline below demonstrates the modular, generic-first approach
PIPELINE_SPEC = [
    {
        "service": "extract_datetime_features",
        "inputs": {"data": "bike-sharing-demand/datasets/train.csv"},
        "outputs": {"data": "bike-sharing-demand/artifacts/train_step1.csv"},
        "module": "bike_sharing_services"
    },
    {
        "service": "impute_zeros_with_regressor",
        "inputs": {"data": "bike-sharing-demand/artifacts/train_step1.csv"},
        "outputs": {"data": "bike-sharing-demand/artifacts/train_step2.csv"},
        "params": {"target_column": "windspeed"},
        "module": "bike_sharing_services"
    },
    {
        "service": "add_bike_domain_features",
        "inputs": {"data": "bike-sharing-demand/artifacts/train_step2.csv"},
        "outputs": {"data": "bike-sharing-demand/artifacts/train_final_raw.csv"},
        "module": "bike_sharing_services"
    },
    {
        "service": "drop_columns",
        "inputs": {"data": "bike-sharing-demand/artifacts/train_final_raw.csv"},
        "outputs": {"data": "bike-sharing-demand/artifacts/train_final.csv"},
        "params": {"columns": ["count", "registered"]}, # Drop other targets
        "module": "bike_sharing_services"
    },
    {
        "service": "train_ensemble_regressor",
        "inputs": {
            "train_data": "bike-sharing-demand/artifacts/train_final.csv", 
            "valid_data": "bike-sharing-demand/artifacts/train_final.csv"
        },
        "outputs": {
            "model": "bike-sharing-demand/artifacts/model_casual.pkl", 
            "metrics": "bike-sharing-demand/artifacts/metrics_casual.json"
        },
        "params": {
            "label_column": "casual", 
            "log_target": True, 
            "id_column": "datetime_str",
            "model_types": ["gradient_boosting", "random_forest"],
            "weights": [0.5, 0.5],
            "gbr_n_estimators": 50,
            "rf_n_estimators": 50
        },
        "module": "bike_sharing_services"
    }
]

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
        # Get metadata from contract decorator
        contract_data = getattr(func, "contract", {})
        
        # Extract metadata
        version = contract_data.get("version", "1.0.0")
        description = contract_data.get("description", "")
        tags = json.dumps(contract_data.get("tags", []))
        input_contract = json.dumps(contract_data.get("inputs", {}))
        output_contract = json.dumps(contract_data.get("outputs", {}))
        
        # Get source code
        try:
            source = inspect.getsource(func)
        except:
            source = f"# Source not available for {name}"
            
        source_hash = hashlib.md5(source.encode()).hexdigest()
        
        # Get parameters from signature (excluding inputs/outputs)
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
            name, version, "bike_sharing_services", description, func.__doc__ or "", 
            tags, "bike-sharing", input_contract, output_contract, parameters, 
            source, source_hash
        ))
    
    conn.commit()
    conn.close()
    return f"Successfully registered {len(SERVICE_REGISTRY)} services to KB."

def run_pipeline(base_path: str, verbose: bool = True):
    """Run the sample pipeline spec end-to-end for testing."""
    for i, step in enumerate(PIPELINE_SPEC, 1):
        service_name = step["service"]
        service_fn = SERVICE_REGISTRY.get(service_name)
        if not service_fn:
            print(f"Error: Service {service_name} not found in registry.")
            continue
            
        res_in = {k: os.path.join(base_path, v) for k, v in step["inputs"].items()}
        res_out = {k: os.path.join(base_path, v) for k, v in step["outputs"].items()}
        
        if verbose: 
            print(f"[{i}/{len(PIPELINE_SPEC)}] Running {service_name}...", end=" ")
            
        try:
            result = service_fn(inputs=res_in, outputs=res_out, **step.get("params", {}))
            if verbose: print(f"OK - {result}")
        except Exception as e:
            if verbose: print(f"FAILED - {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bike Sharing Demand Pipeline Test")
    parser.add_argument("--base-path", default="storage")
    parser.add_argument("--skip-register", action="store_true", help="Skip KB registration")
    args = parser.parse_args()

    # Determine absolute base path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If storage is in webapp/storage, and we are in webapp/services/
    potential_storage = os.path.join(os.path.dirname(script_dir), args.base_path)
    if os.path.exists(potential_storage):
        storage_path = potential_storage
    else:
        # Try relative to CWD
        storage_path = os.path.abspath(args.base_path)

    print(f"\n--- Testing Bike Sharing Pipeline (Base Path: {storage_path}) ---")
    run_pipeline(storage_path)
    
    if not args.skip_register:
        print(f"\n--- Syncing to Knowledge Base ---")
        status = register_to_kb()
        print(status)
