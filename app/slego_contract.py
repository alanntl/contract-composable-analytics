"""
SLEGO Data Contract System
==========================

A unified, machine-checkable schema system for SLEGO microservices.

SLEGO GUIDELINES IMPLEMENTED:
- G2: Explicit Data Interface - All I/O declared with format contracts
- G6: Semantic Metadata - Services self-register for AI discovery

FEATURES:
- IOManager: Universal format handler (CSV, Parquet, JSON, Pickle, etc.)
- @contract decorator: Code-first schema declaration
- ServiceRegistry: Global discovery of all services
- validate_pipeline(): Machine-check pipeline connections before execution

USAGE:
    from slego_contract import contract, IOManager, ServiceRegistry, validate_pipeline

    @contract(
        inputs={"data": {"format": "csv", "schema": {"type": "tabular"}}},
        outputs={"model": {"format": "pickle", "schema": {"type": "artifact"}}}
    )
    def my_service(inputs, outputs, **params):
        df = IOManager.load(inputs["data"], "csv")
        ...
        IOManager.save(model, outputs["model"], "pickle")

Author: SLEGO Framework
Version: 1.0.0
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional, Union, List, Tuple
from functools import wraps
from pathlib import Path
from abc import ABC, abstractmethod


# =============================================================================
# G2: STRUCTURAL SCHEMAS (Machine-Checkable Data Contracts)
# =============================================================================

class Schema(ABC):
    """
    Base class for all structural schemas.

    Each schema type defines:
    1. What structure the data must have
    2. How to validate actual data against the schema
    3. How to check compatibility with other schemas
    """

    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data against this schema. Returns (is_valid, errors)."""
        pass

    @abstractmethod
    def compatible_with(self, other: "Schema") -> Tuple[bool, str]:
        """Check if this schema's output can connect to another schema's input."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """Serialize schema to dict for storage/display."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict) -> "Schema":
        """Deserialize schema from dict."""
        pass


class TabularSchema(Schema):
    """
    Schema for tabular data (CSV, Parquet, Excel).

    Validates:
    - Required columns exist
    - Column types match (numeric, categorical, datetime)
    - Row count constraints
    - Missing value constraints

    Example:
        schema = TabularSchema(
            columns={"price": "numeric", "name": "categorical"},
            required_columns=["price"],
            allow_missing=False,
            min_rows=10
        )
    """

    def __init__(
        self,
        columns: Dict[str, str] = None,          # {col_name: type} where type = numeric|categorical|datetime|any
        required_columns: List[str] = None,       # Columns that MUST exist
        optional_columns: List[str] = None,       # Columns that MAY exist
        allow_extra_columns: bool = True,         # Allow columns not in schema?
        allow_missing: bool = True,               # Allow NaN values?
        min_rows: int = 0,
        max_rows: int = None,
        min_columns: int = 0,
        dynamic_columns: List[str] = None,        # G4: Columns injected via params
    ):
        self.columns = columns or {}
        self.required_columns = required_columns or list(self.columns.keys())
        self.optional_columns = optional_columns or []
        self.allow_extra_columns = allow_extra_columns
        self.allow_missing = allow_missing
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.min_columns = min_columns
        self.dynamic_columns = dynamic_columns or []

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate a DataFrame against this schema."""
        errors = []

        if not isinstance(data, pd.DataFrame):
            return False, ["Expected DataFrame, got " + type(data).__name__]

        df = data

        # Check row count
        if len(df) < self.min_rows:
            errors.append(f"Expected >= {self.min_rows} rows, got {len(df)}")
        if self.max_rows and len(df) > self.max_rows:
            errors.append(f"Expected <= {self.max_rows} rows, got {len(df)}")

        # Check column count
        if len(df.columns) < self.min_columns:
            errors.append(f"Expected >= {self.min_columns} columns, got {len(df.columns)}")

        # Check required columns exist
        for col in self.required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")

        # Check column types
        for col, expected_type in self.columns.items():
            if col not in df.columns:
                continue  # Already checked in required_columns

            actual_type = self._get_column_type(df[col])
            if expected_type != "any" and actual_type != expected_type:
                errors.append(f"Column '{col}': expected {expected_type}, got {actual_type}")

        # Check missing values
        if not self.allow_missing:
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                errors.append(f"Missing values not allowed, found in: {missing_cols}")

        # Check extra columns
        if not self.allow_extra_columns:
            allowed = set(self.columns.keys()) | set(self.optional_columns)
            extra = set(df.columns) - allowed
            if extra:
                errors.append(f"Unexpected columns: {extra}")

        return len(errors) == 0, errors

    def _get_column_type(self, series: pd.Series) -> str:
        """Infer column type from pandas Series."""
        import numpy as np
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        else:
            return "categorical"

    def compatible_with(self, other: "Schema") -> Tuple[bool, str]:
        """Check if this schema's output can connect to another TabularSchema input."""
        if not isinstance(other, TabularSchema):
            return False, f"Type mismatch: TabularSchema → {type(other).__name__}"

        # Rule 1: Output with missing → Input without missing = FAIL
        if not other.allow_missing and self.allow_missing:
            return False, "Input requires clean data (no missing), but output may have missing values"

        # Rule 2: Output must have all columns input requires
        missing_cols = set(other.required_columns) - set(self.columns.keys()) - set(self.dynamic_columns)
        # Filter out dynamic columns (G4: injected at runtime)
        missing_cols = missing_cols - set(other.dynamic_columns)
        if missing_cols:
            return False, f"Input requires columns not in output: {missing_cols}"

        # Rule 3: Column types must match
        for col, expected_type in other.columns.items():
            if col in self.columns and expected_type != "any":
                if self.columns[col] != expected_type and self.columns[col] != "any":
                    return False, f"Column '{col}' type mismatch: {self.columns[col]} → {expected_type}"

        # Rule 4: Row constraints
        if other.min_rows > 0 and self.max_rows and self.max_rows < other.min_rows:
            return False, f"Output max_rows ({self.max_rows}) < input min_rows ({other.min_rows})"

        return True, "Compatible"

    def to_dict(self) -> Dict:
        return {
            "type": "tabular",
            "columns": self.columns,
            "required_columns": self.required_columns,
            "optional_columns": self.optional_columns,
            "allow_extra_columns": self.allow_extra_columns,
            "allow_missing": self.allow_missing,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "min_columns": self.min_columns,
            "dynamic_columns": self.dynamic_columns,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TabularSchema":
        return cls(
            columns=d.get("columns", {}),
            required_columns=d.get("required_columns"),
            optional_columns=d.get("optional_columns"),
            allow_extra_columns=d.get("allow_extra_columns", True),
            allow_missing=d.get("allow_missing", True),
            min_rows=d.get("min_rows", 0),
            max_rows=d.get("max_rows"),
            min_columns=d.get("min_columns", 0),
            dynamic_columns=d.get("dynamic_columns"),
        )


class JSONSchema(Schema):
    """
    Schema for JSON/dict data.

    Validates:
    - Required fields exist
    - Field types match
    - Nested structure (recursive)

    Example:
        schema = JSONSchema(
            fields={"rmse": "float", "n_samples": "int"},
            required_fields=["rmse"],
            nested={"config": JSONSchema(fields={"lr": "float"})}
        )
    """

    TYPE_MAP = {
        "str": str,
        "int": int,
        "float": (int, float),
        "bool": bool,
        "list": list,
        "dict": dict,
        "any": object,
    }

    def __init__(
        self,
        fields: Dict[str, str] = None,           # {field_name: type}
        required_fields: List[str] = None,
        optional_fields: List[str] = None,
        allow_extra_fields: bool = True,
        nested: Dict[str, "JSONSchema"] = None,  # Nested schemas
    ):
        self.fields = fields or {}
        self.required_fields = required_fields or list(self.fields.keys())
        self.optional_fields = optional_fields or []
        self.allow_extra_fields = allow_extra_fields
        self.nested = nested or {}

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate a dict against this schema."""
        errors = []

        if not isinstance(data, dict):
            return False, ["Expected dict, got " + type(data).__name__]

        # Check required fields
        for field in self.required_fields:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")

        # Check field types
        for field, expected_type in self.fields.items():
            if field not in data:
                continue

            value = data[field]
            valid_types = self.TYPE_MAP.get(expected_type, object)

            if expected_type != "any" and not isinstance(value, valid_types):
                errors.append(f"Field '{field}': expected {expected_type}, got {type(value).__name__}")

        # Check nested schemas
        for field, nested_schema in self.nested.items():
            if field in data:
                nested_valid, nested_errors = nested_schema.validate(data[field])
                if not nested_valid:
                    errors.extend([f"{field}.{e}" for e in nested_errors])

        # Check extra fields
        if not self.allow_extra_fields:
            allowed = set(self.fields.keys()) | set(self.optional_fields) | set(self.nested.keys())
            extra = set(data.keys()) - allowed
            if extra:
                errors.append(f"Unexpected fields: {extra}")

        return len(errors) == 0, errors

    def compatible_with(self, other: "Schema") -> Tuple[bool, str]:
        """Check if this JSON schema can connect to another."""
        if not isinstance(other, JSONSchema):
            return False, f"Type mismatch: JSONSchema → {type(other).__name__}"

        # Output must provide all fields input requires
        missing = set(other.required_fields) - set(self.fields.keys()) - set(self.optional_fields)
        if missing:
            return False, f"Input requires fields not in output: {missing}"

        # Field types must be compatible
        for field, expected_type in other.fields.items():
            if field in self.fields and expected_type != "any":
                if self.fields[field] != expected_type and self.fields[field] != "any":
                    return False, f"Field '{field}' type mismatch: {self.fields[field]} → {expected_type}"

        return True, "Compatible"

    def to_dict(self) -> Dict:
        return {
            "type": "json",
            "fields": self.fields,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "allow_extra_fields": self.allow_extra_fields,
            "nested": {k: v.to_dict() for k, v in self.nested.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "JSONSchema":
        nested = {k: cls.from_dict(v) for k, v in d.get("nested", {}).items()}
        return cls(
            fields=d.get("fields", {}),
            required_fields=d.get("required_fields"),
            optional_fields=d.get("optional_fields"),
            allow_extra_fields=d.get("allow_extra_fields", True),
            nested=nested,
        )


class ArtifactSchema(Schema):
    """
    Schema for Python artifacts (models, transformers, etc.).

    Validates:
    - Object class matches expected
    - Required attributes/methods exist

    Example:
        schema = ArtifactSchema(
            artifact_type="model",
            expected_class="sklearn.ensemble.RandomForestRegressor",
            required_attrs=["predict", "fit", "n_estimators"]
        )
    """

    def __init__(
        self,
        artifact_type: str = "any",              # model, transformer, imputer, encoder, scaler
        expected_class: str = None,               # Full class path
        required_attrs: List[str] = None,         # Required attributes/methods
        required_methods: List[str] = None,       # Methods that must be callable
    ):
        self.artifact_type = artifact_type
        self.expected_class = expected_class
        self.required_attrs = required_attrs or []
        self.required_methods = required_methods or []

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate an artifact object against this schema."""
        errors = []

        # Check class
        if self.expected_class:
            full_class = f"{data.__class__.__module__}.{data.__class__.__name__}"
            if self.expected_class not in full_class:
                errors.append(f"Expected class '{self.expected_class}', got '{full_class}'")

        # Check required attributes
        for attr in self.required_attrs:
            if not hasattr(data, attr):
                errors.append(f"Missing required attribute: '{attr}'")

        # Check required methods
        for method in self.required_methods:
            if not hasattr(data, method) or not callable(getattr(data, method)):
                errors.append(f"Missing required method: '{method}'")

        return len(errors) == 0, errors

    def compatible_with(self, other: "Schema") -> Tuple[bool, str]:
        """Check if this artifact schema can connect to another."""
        if not isinstance(other, ArtifactSchema):
            return False, f"Type mismatch: ArtifactSchema → {type(other).__name__}"

        # Artifact type must match (unless 'any')
        if other.artifact_type != "any" and self.artifact_type != "any":
            if other.artifact_type != self.artifact_type:
                return False, f"Artifact type mismatch: {self.artifact_type} → {other.artifact_type}"

        # Class must be compatible
        if other.expected_class and self.expected_class:
            if other.expected_class not in self.expected_class:
                return False, f"Class mismatch: {self.expected_class} → {other.expected_class}"

        # Output must have all attrs/methods input requires
        missing_attrs = set(other.required_attrs) - set(self.required_attrs)
        if missing_attrs:
            return False, f"Input requires attributes not guaranteed by output: {missing_attrs}"

        return True, "Compatible"

    def to_dict(self) -> Dict:
        return {
            "type": "artifact",
            "artifact_type": self.artifact_type,
            "expected_class": self.expected_class,
            "required_attrs": self.required_attrs,
            "required_methods": self.required_methods,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ArtifactSchema":
        return cls(
            artifact_type=d.get("artifact_type", "any"),
            expected_class=d.get("expected_class"),
            required_attrs=d.get("required_attrs"),
            required_methods=d.get("required_methods"),
        )


class ImageSchema(Schema):
    """
    Schema for image data.

    Validates:
    - Image dimensions (width, height)
    - Number of channels
    - File extensions

    Example:
        schema = ImageSchema(
            min_width=224, min_height=224,
            channels=3,  # RGB
            extensions=[".png", ".jpg"]
        )
    """

    def __init__(
        self,
        min_width: int = None,
        max_width: int = None,
        min_height: int = None,
        max_height: int = None,
        channels: int = None,                     # 1=grayscale, 3=RGB, 4=RGBA
        extensions: List[str] = None,
        is_batch: bool = False,                   # True if directory of images
    ):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.channels = channels
        self.extensions = extensions or [".png", ".jpg", ".jpeg"]
        self.is_batch = is_batch

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate image data (numpy array or path)."""
        errors = []

        # If it's a path, just check extension
        if isinstance(data, (str, Path)):
            ext = Path(data).suffix.lower()
            if ext not in self.extensions:
                errors.append(f"Invalid extension '{ext}', expected one of {self.extensions}")
            return len(errors) == 0, errors

        # If it's an array, check dimensions
        if hasattr(data, 'shape'):
            shape = data.shape

            if len(shape) == 2:  # Grayscale
                h, w = shape
                c = 1
            elif len(shape) == 3:  # Color
                h, w, c = shape
            else:
                return False, [f"Invalid image shape: {shape}"]

            if self.min_width and w < self.min_width:
                errors.append(f"Width {w} < min_width {self.min_width}")
            if self.max_width and w > self.max_width:
                errors.append(f"Width {w} > max_width {self.max_width}")
            if self.min_height and h < self.min_height:
                errors.append(f"Height {h} < min_height {self.min_height}")
            if self.max_height and h > self.max_height:
                errors.append(f"Height {h} > max_height {self.max_height}")
            if self.channels and c != self.channels:
                errors.append(f"Channels {c} != expected {self.channels}")

        return len(errors) == 0, errors

    def compatible_with(self, other: "Schema") -> Tuple[bool, str]:
        """Check if this image schema can connect to another."""
        if not isinstance(other, ImageSchema):
            return False, f"Type mismatch: ImageSchema → {type(other).__name__}"

        # Check channel compatibility
        if other.channels and self.channels and other.channels != self.channels:
            return False, f"Channel mismatch: {self.channels} → {other.channels}"

        # Check size constraints
        if other.min_width and self.max_width and self.max_width < other.min_width:
            return False, f"Output max_width ({self.max_width}) < input min_width ({other.min_width})"
        if other.min_height and self.max_height and self.max_height < other.min_height:
            return False, f"Output max_height ({self.max_height}) < input min_height ({other.min_height})"

        return True, "Compatible"

    def to_dict(self) -> Dict:
        return {
            "type": "image",
            "min_width": self.min_width,
            "max_width": self.max_width,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "channels": self.channels,
            "extensions": self.extensions,
            "is_batch": self.is_batch,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ImageSchema":
        return cls(
            min_width=d.get("min_width"),
            max_width=d.get("max_width"),
            min_height=d.get("min_height"),
            max_height=d.get("max_height"),
            channels=d.get("channels"),
            extensions=d.get("extensions"),
            is_batch=d.get("is_batch", False),
        )


# Schema factory for deserialization
def schema_from_dict(d: Dict) -> Schema:
    """Create a Schema object from a dictionary."""
    if not d:
        return None

    schema_type = d.get("type", "any")

    if schema_type == "tabular":
        return TabularSchema.from_dict(d)
    elif schema_type == "json":
        return JSONSchema.from_dict(d)
    elif schema_type == "artifact":
        return ArtifactSchema.from_dict(d)
    elif schema_type == "image":
        return ImageSchema.from_dict(d)
    else:
        return None


def check_schema_compatibility(
    output_schema: Union[Schema, Dict],
    input_schema: Union[Schema, Dict]
) -> Tuple[bool, str]:
    """
    Check if an output schema is compatible with an input schema.

    Parameters
    ----------
    output_schema : Schema or Dict
        Schema of the producing service's output
    input_schema : Schema or Dict
        Schema of the consuming service's input

    Returns
    -------
    Tuple[bool, str]
        (is_compatible, reason)
    """
    # Convert dicts to Schema objects
    if isinstance(output_schema, dict):
        output_schema = schema_from_dict(output_schema)
    if isinstance(input_schema, dict):
        input_schema = schema_from_dict(input_schema)

    # If no schemas defined, assume compatible
    if output_schema is None or input_schema is None:
        return True, "No structural schema defined (format-only check)"

    # Use schema's compatibility method
    return output_schema.compatible_with(input_schema)


# =============================================================================
# G2: UNIVERSAL I/O MANAGER
# =============================================================================

class IOManager:
    """
    Central handler for all SLEGO Input/Output operations.

    Decouples service logic (G1) from storage formats.
    Each format defines: read, write, mime type, description, and output type.

    Supported Formats:
    - Tabular: csv, parquet, excel, json_table
    - Artifacts: pickle, joblib
    - Structured: json, metrics_json
    - Images: png, jpg (placeholder)
    - Text: txt, md

    Example:
        df = IOManager.load("data.csv", "csv")
        IOManager.save(df, "data.parquet", "parquet")

        # Check compatibility
        IOManager.compatible("csv", "parquet")  # True (both produce DataFrame)
    """

    _REGISTRY = {
        # =====================================================================
        # TABULAR DATA (all produce DataFrame)
        # =====================================================================
        "csv": {
            "read": lambda path: pd.read_csv(path),
            "write": lambda data, path: data.to_csv(path, index=False),
            "mime": "text/csv",
            "desc": "Comma Separated Values",
            "produces": "DataFrame",
            "extensions": [".csv"],
        },
        "parquet": {
            "read": lambda path: pd.read_parquet(path),
            "write": lambda data, path: data.to_parquet(path, index=False),
            "mime": "application/vnd.apache.parquet",
            "desc": "Apache Parquet (Binary Columnar)",
            "produces": "DataFrame",
            "extensions": [".parquet", ".pq"],
        },
        "excel": {
            "read": lambda path: pd.read_excel(path),
            "write": lambda data, path: data.to_excel(path, index=False),
            "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "desc": "Microsoft Excel",
            "produces": "DataFrame",
            "extensions": [".xlsx", ".xls"],
        },
        "json_table": {
            "read": lambda path: pd.read_json(path, orient="records"),
            "write": lambda data, path: data.to_json(path, orient="records", indent=2),
            "mime": "application/json",
            "desc": "JSON Array of Records (Tabular)",
            "produces": "DataFrame",
            "extensions": [".json"],
        },

        # =====================================================================
        # ARTIFACTS (produce Python objects)
        # =====================================================================
        "pickle": {
            "read": lambda path: joblib.load(path),
            "write": lambda data, path: joblib.dump(data, path),
            "mime": "application/octet-stream",
            "desc": "Python Serialized Object (joblib)",
            "produces": "Any",
            "extensions": [".pkl", ".joblib"],
        },

        # =====================================================================
        # STRUCTURED DATA (produce dict)
        # =====================================================================
        "json": {
            "read": lambda path: json.load(open(path, 'r')),
            "write": lambda data, path: json.dump(data, open(path, 'w'), indent=2),
            "mime": "application/json",
            "desc": "JSON Object/Dict",
            "produces": "dict",
            "extensions": [".json"],
        },
        "metrics_json": {
            "read": lambda path: json.load(open(path, 'r')),
            "write": lambda data, path: json.dump(data, open(path, 'w'), indent=2),
            "mime": "application/json+metrics",
            "desc": "Metrics/Evaluation JSON",
            "produces": "dict",
            "extensions": [".json"],
        },

        # =====================================================================
        # TEXT DATA (produce str)
        # =====================================================================
        "text": {
            "read": lambda path: Path(path).read_text(),
            "write": lambda data, path: Path(path).write_text(data),
            "mime": "text/plain",
            "desc": "Plain Text",
            "produces": "str",
            "extensions": [".txt", ".md", ".log"],
        },

        # =====================================================================
        # NUMPY ARRAYS
        # =====================================================================
        "npy": {
            "read": lambda path: np.load(path),
            "write": lambda data, path: np.save(path, data),
            "mime": "application/octet-stream",
            "desc": "NumPy Array",
            "produces": "ndarray",
            "extensions": [".npy"],
        },
        "npz": {
            "read": lambda path: np.load(path),
            "write": lambda data, path: np.savez(path, **data) if isinstance(data, dict) else np.savez(path, data=data),
            "mime": "application/octet-stream",
            "desc": "NumPy Compressed Archive",
            "produces": "NpzFile",
            "extensions": [".npz"],
        },
    }

    @classmethod
    def load(cls, path: str, fmt: str, verbose: bool = True) -> Any:
        """
        Load data from file using the declared format contract.

        Parameters
        ----------
        path : str
            File path to load
        fmt : str
            Format identifier (e.g., 'csv', 'parquet', 'pickle')
        verbose : bool
            Print loading message

        Returns
        -------
        Any
            Loaded data (DataFrame, dict, object, etc.)

        Raises
        ------
        ValueError
            If format is not registered
        FileNotFoundError
            If file doesn't exist
        """
        if fmt not in cls._REGISTRY:
            raise ValueError(
                f"Unknown format: '{fmt}'. "
                f"Supported: {list(cls._REGISTRY.keys())}"
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

        if verbose:
            print(f"  [IO-READ] Loading <{fmt}> from {os.path.basename(path)}")

        return cls._REGISTRY[fmt]["read"](path)

    @classmethod
    def save(cls, data: Any, path: str, fmt: str, verbose: bool = True) -> None:
        """
        Save data to file using the declared format contract.

        Parameters
        ----------
        data : Any
            Data to save
        path : str
            Output file path
        fmt : str
            Format identifier
        verbose : bool
            Print saving message
        """
        if fmt not in cls._REGISTRY:
            raise ValueError(
                f"Unknown format: '{fmt}'. "
                f"Supported: {list(cls._REGISTRY.keys())}"
            )

        # Ensure directory exists
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        if verbose:
            print(f"  [IO-WRITE] Saving <{fmt}> to {os.path.basename(path)}")

        cls._REGISTRY[fmt]["write"](data, path)

    @classmethod
    def compatible(cls, output_fmt: str, input_fmt: str) -> bool:
        """
        Check if output format can connect to input format.

        Compatibility Rules:
        - Same 'produces' type: compatible
        - Output produces 'Any': compatible with anything
        - Input accepts 'Any': compatible with anything

        Parameters
        ----------
        output_fmt : str
            Format of the producing service's output
        input_fmt : str
            Format of the consuming service's input

        Returns
        -------
        bool
            True if compatible
        """
        if output_fmt not in cls._REGISTRY or input_fmt not in cls._REGISTRY:
            return False

        out_type = cls._REGISTRY[output_fmt]["produces"]
        in_type = cls._REGISTRY[input_fmt]["produces"]

        return (
            out_type == in_type or
            out_type == "Any" or
            in_type == "Any"
        )

    @classmethod
    def get_format_info(cls, fmt: str) -> Dict[str, Any]:
        """Get full info about a format."""
        return cls._REGISTRY.get(fmt, {})

    @classmethod
    def infer_format(cls, path: str) -> Optional[str]:
        """Infer format from file extension."""
        ext = Path(path).suffix.lower()
        for fmt, info in cls._REGISTRY.items():
            if ext in info.get("extensions", []):
                return fmt
        return None

    @classmethod
    def list_formats(cls) -> None:
        """Print all supported formats."""
        print(f"\n{'FORMAT':<15} {'PRODUCES':<12} {'EXTENSIONS':<20} {'DESCRIPTION'}")
        print("=" * 80)
        for fmt, info in cls._REGISTRY.items():
            exts = ", ".join(info.get("extensions", []))
            print(f"{fmt:<15} {info['produces']:<12} {exts:<20} {info['desc']}")

    @classmethod
    def register_format(
        cls,
        name: str,
        read_fn: Callable,
        write_fn: Callable,
        produces: str,
        mime: str = "application/octet-stream",
        desc: str = "",
        extensions: List[str] = None
    ) -> None:
        """
        Register a custom format.

        Example:
            IOManager.register_format(
                name="hdf5",
                read_fn=lambda p: pd.read_hdf(p),
                write_fn=lambda d, p: d.to_hdf(p, key='data'),
                produces="DataFrame",
                extensions=[".h5", ".hdf5"]
            )
        """
        cls._REGISTRY[name] = {
            "read": read_fn,
            "write": write_fn,
            "mime": mime,
            "desc": desc,
            "produces": produces,
            "extensions": extensions or [],
        }


# =============================================================================
# G6: SERVICE REGISTRY (Discovery Interface)
# =============================================================================

class ServiceRegistry:
    """
    Global registry for microservice contracts.

    Enables AI-assisted discovery of available services,
    their input/output contracts, and compatibility checking.
    """
    _services: Dict[str, Dict] = {}

    @classmethod
    def register(cls, name: str, contract: Dict, func: Callable = None) -> None:
        """Register a service contract."""
        cls._services[name] = {
            **contract,
            "function": func,
        }

    @classmethod
    def get(cls, name: str) -> Optional[Dict]:
        """Get a service contract by name."""
        return cls._services.get(name)

    @classmethod
    def get_function(cls, name: str) -> Optional[Callable]:
        """Get the actual function for a service."""
        svc = cls._services.get(name)
        return svc.get("function") if svc else None

    @classmethod
    def list_all(cls) -> Dict[str, Dict]:
        """Get all registered services."""
        return cls._services

    @classmethod
    def list_names(cls) -> List[str]:
        """Get names of all registered services."""
        return list(cls._services.keys())

    @classmethod
    def find_by_tag(cls, tag: str) -> List[str]:
        """Find services by tag."""
        matches = []
        for name, contract in cls._services.items():
            if tag in contract.get("tags", []):
                matches.append(name)
        return matches

    @classmethod
    def find_by_input_format(cls, fmt: str) -> List[str]:
        """Find services that accept a specific input format."""
        matches = []
        for name, contract in cls._services.items():
            for slot_spec in contract.get("input", {}).values():
                if slot_spec.get("format") == fmt:
                    matches.append(name)
                    break
        return matches

    @classmethod
    def find_by_output_format(cls, fmt: str) -> List[str]:
        """Find services that produce a specific output format."""
        matches = []
        for name, contract in cls._services.items():
            for slot_spec in contract.get("output", {}).values():
                if slot_spec.get("format") == fmt:
                    matches.append(name)
                    break
        return matches

    @classmethod
    def describe(cls, name: str = None) -> None:
        """Print service contract(s) in readable format."""
        services = {name: cls._services[name]} if name else cls._services

        for svc_name, contract in services.items():
            print(f"\n{'='*60}")
            print(f"📦 {svc_name}")
            print(f"{'='*60}")

            if contract.get("description"):
                print(f"   {contract['description']}")

            if contract.get("tags"):
                print(f"   Tags: {', '.join(contract['tags'])}")

            print(f"\n   INPUTS:")
            for slot, spec in contract.get("input", {}).items():
                req = "required" if spec.get("required", True) else "optional"
                print(f"     • {slot}: <{spec.get('format', '?')}> ({req})")
                if spec.get("schema"):
                    print(f"       └─ schema: {spec['schema']}")

            print(f"\n   OUTPUTS:")
            for slot, spec in contract.get("output", {}).items():
                print(f"     • {slot}: <{spec.get('format', '?')}>")
                if spec.get("schema"):
                    print(f"       └─ schema: {spec['schema']}")


# =============================================================================
# G2: @CONTRACT DECORATOR
# =============================================================================

def contract(
    inputs: Dict[str, Dict],
    outputs: Dict[str, Dict],
    description: str = "",
    tags: List[str] = None,
    version: str = "1.0.0"
):
    """
    Decorator to declare a service's Data Contract (G2).

    Automatically:
    1. Registers the service in ServiceRegistry (G6)
    2. Attaches contract to function for introspection
    3. Validates inputs exist before execution
    4. Validates outputs created after execution

    Parameters
    ----------
    inputs : Dict[str, Dict]
        Input slot specifications. Each slot has:
        - format: str (e.g., 'csv', 'parquet', 'pickle')
        - schema: dict or str (optional structural schema)
        - required: bool (default True)

    outputs : Dict[str, Dict]
        Output slot specifications. Each slot has:
        - format: str
        - schema: dict or str (optional)

    description : str
        Human-readable description

    tags : List[str]
        Tags for discovery (e.g., ['preprocessing', 'imputation'])

    version : str
        Service version

    Example
    -------
    @contract(
        inputs={
            "data": {"format": "csv", "schema": {"type": "tabular", "allow_missing": True}},
        },
        outputs={
            "data": {"format": "csv", "schema": {"type": "tabular", "allow_missing": False}},
            "imputer": {"format": "pickle", "schema": {"type": "artifact"}},
        },
        description="Impute missing values",
        tags=["preprocessing", "imputation"],
    )
    def fill_missing(inputs, outputs, strategy="mean"):
        ...
    """
    def decorator(func: Callable) -> Callable:
        name = func.__name__

        contract_data = {
            "description": description,
            "tags": tags or [],
            "version": version,
            "input": inputs,
            "output": outputs,
        }

        # Register in global registry
        ServiceRegistry.register(name, contract_data, func)

        # Attach to function for introspection
        func.contract = contract_data

        @wraps(func)
        def wrapper(
            inputs: Dict[str, str],
            outputs: Dict[str, str],
            **kwargs
        ) -> Any:
            # Get the contract's input/output specs
            input_specs = contract_data["input"]
            output_specs = contract_data["output"]

            # ─────────────────────────────────────────────────────────────
            # PRE-EXECUTION: Validate inputs (existence + schema)
            # ─────────────────────────────────────────────────────────────
            for slot, spec in input_specs.items():
                path = inputs.get(slot)
                is_required = spec.get("required", True)

                if is_required and not path:
                    raise ValueError(
                        f"[{name}] Missing required input slot: '{slot}'"
                    )

                if path and is_required and not os.path.exists(path):
                    raise FileNotFoundError(
                        f"[{name}] Input '{slot}' not found: {path}"
                    )

                # --- Runtime schema validation on input data ---
                schema_dict = spec.get("schema")
                fmt = spec.get("format")
                if path and os.path.exists(path) and schema_dict and fmt:
                    schema_obj = schema_from_dict(schema_dict) if isinstance(schema_dict, dict) else None
                    if schema_obj is not None:
                        try:
                            data = IOManager.load(path, fmt, verbose=False)
                            is_valid, errors = schema_obj.validate(data)
                            if not is_valid:
                                raise ValueError(
                                    f"[{name}] Input '{slot}' failed schema validation: "
                                    + "; ".join(errors)
                                )
                        except (ValueError, FileNotFoundError):
                            raise
                        except Exception:
                            # If we can't load/validate (e.g. unsupported format
                            # for schema validation), skip gracefully
                            pass

            # ─────────────────────────────────────────────────────────────
            # EXECUTE
            # ─────────────────────────────────────────────────────────────
            result = func(inputs, outputs, **kwargs)

            # ─────────────────────────────────────────────────────────────
            # POST-EXECUTION: Validate outputs (existence + schema)
            # ─────────────────────────────────────────────────────────────
            for slot, spec in output_specs.items():
                path = outputs.get(slot)
                if path and not os.path.exists(path):
                    raise RuntimeError(
                        f"[{name}] Failed to produce output '{slot}': {path}"
                    )

                # --- Runtime schema validation on output data ---
                schema_dict = spec.get("schema")
                fmt = spec.get("format")
                if path and os.path.exists(path) and schema_dict and fmt:
                    schema_obj = schema_from_dict(schema_dict) if isinstance(schema_dict, dict) else None
                    if schema_obj is not None:
                        try:
                            data = IOManager.load(path, fmt, verbose=False)
                            is_valid, errors = schema_obj.validate(data)
                            if not is_valid:
                                raise RuntimeError(
                                    f"[{name}] Output '{slot}' failed schema validation: "
                                    + "; ".join(errors)
                                )
                        except (RuntimeError, ValueError):
                            raise
                        except Exception:
                            pass

            return result

        # Preserve contract on wrapper
        wrapper.contract = contract_data

        return wrapper

    return decorator


# =============================================================================
# PIPELINE VALIDATION (Machine-Checkable)
# =============================================================================

def validate_pipeline(
    steps: List[Dict],
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    Machine-check that all pipeline connections are type-safe.

    Validates:
    1. All services exist in registry
    2. All input slots are valid for each service
    3. Output formats are compatible with downstream input formats

    Parameters
    ----------
    steps : List[Dict]
        Pipeline definition. Each step has:
        - service: str (service name) or Callable (function with .contract)
        - inputs: Dict[str, str] (slot → path mapping)
        - outputs: Dict[str, str] (slot → path mapping)

    verbose : bool
        Print validation results

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)

    Example
    -------
    steps = [
        {"service": "clean_data", "inputs": {"data": "a.csv"}, "outputs": {"data": "b.parquet"}},
        {"service": "train_model", "inputs": {"train_data": "b.parquet"}, "outputs": {"model": "m.pkl"}},
    ]
    is_valid, errors = validate_pipeline(steps)
    """
    errors = []
    output_registry: Dict[str, str] = {}  # path → format

    for i, step in enumerate(steps):
        service = step.get("service")

        # Get service name and contract
        if callable(service):
            name = service.__name__
            contract = getattr(service, 'contract', None)
        else:
            name = service
            contract = ServiceRegistry.get(name)

        if not contract:
            errors.append(f"Step {i+1}: Unknown service '{name}'")
            continue

        step_inputs = step.get("inputs", {})
        step_outputs = step.get("outputs", {})

        # ─────────────────────────────────────────────────────────────────
        # Check: All input slots are valid
        # ─────────────────────────────────────────────────────────────────
        for slot in step_inputs.keys():
            if slot not in contract.get("input", {}):
                errors.append(
                    f"Step {i+1} ({name}): Unknown input slot '{slot}'"
                )

        # ─────────────────────────────────────────────────────────────────
        # Check: Required inputs are provided
        # ─────────────────────────────────────────────────────────────────
        for slot, spec in contract.get("input", {}).items():
            if spec.get("required", True) and slot not in step_inputs:
                errors.append(
                    f"Step {i+1} ({name}): Missing required input '{slot}'"
                )

        # ─────────────────────────────────────────────────────────────────
        # Check: Format compatibility for connected paths
        # ─────────────────────────────────────────────────────────────────
        for slot, path in step_inputs.items():
            if path in output_registry:
                # This path was produced by a previous step
                output_info = output_registry[path]
                output_fmt = output_info["format"]
                output_schema = output_info.get("schema")

                input_spec = contract.get("input", {}).get(slot, {})
                input_fmt = input_spec.get("format")
                input_schema = input_spec.get("schema")

                # Check format compatibility
                if input_fmt and not IOManager.compatible(output_fmt, input_fmt):
                    errors.append(
                        f"Step {i+1} ({name}.{slot}): "
                        f"Incompatible format - receives <{output_fmt}>, expects <{input_fmt}>"
                    )

                # Check structural schema compatibility
                if output_schema and input_schema:
                    schema_ok, schema_reason = check_schema_compatibility(output_schema, input_schema)
                    if not schema_ok:
                        errors.append(
                            f"Step {i+1} ({name}.{slot}): "
                            f"Incompatible structure - {schema_reason}"
                        )

        # ─────────────────────────────────────────────────────────────────
        # Register outputs for downstream checking
        # ─────────────────────────────────────────────────────────────────
        for slot, path in step_outputs.items():
            output_spec = contract.get("output", {}).get(slot, {})
            output_fmt = output_spec.get("format")
            output_schema = output_spec.get("schema")
            if output_fmt:
                output_registry[path] = {
                    "format": output_fmt,
                    "schema": output_schema,
                }

    is_valid = len(errors) == 0

    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE VALIDATION RESULT")
        print("=" * 60)
        if is_valid:
            print("✅ All connections are type-safe!")
        else:
            print(f"❌ Found {len(errors)} error(s):")
            for err in errors:
                print(f"   • {err}")

    return is_valid, errors


def validate_schema_compatibility(
    output_schema: Dict,
    input_schema: Dict
) -> Tuple[bool, str]:
    """
    Deep structural compatibility check between schemas.

    Parameters
    ----------
    output_schema : Dict
        Schema from producing service's output
    input_schema : Dict
        Schema from consuming service's input

    Returns
    -------
    Tuple[bool, str]
        (is_compatible, reason)
    """
    out_type = output_schema.get("type", "any")
    in_type = input_schema.get("type", "any")

    # Type must match
    if out_type != in_type and out_type != "any" and in_type != "any":
        return False, f"Type mismatch: {out_type} → {in_type}"

    # Tabular-specific checks
    if out_type == "tabular" and in_type == "tabular":
        out_missing = output_schema.get("allow_missing", True)
        in_missing = input_schema.get("allow_missing", True)

        if not in_missing and out_missing:
            return False, "Input requires clean data, but output may have missing values"

        out_min_rows = output_schema.get("min_rows", 0)
        in_min_rows = input_schema.get("min_rows", 0)

        if out_min_rows < in_min_rows:
            return False, f"Input requires min {in_min_rows} rows"

    # Artifact-specific checks
    if out_type == "artifact" and in_type == "artifact":
        out_class = output_schema.get("expected_class", "")
        in_class = input_schema.get("expected_class", "")

        if in_class and out_class and in_class not in out_class:
            return False, f"Class mismatch: {out_class} → {in_class}"

    return True, "Compatible"


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # 1. Define Services with @contract (including STRUCTURAL schemas)
    # =========================================================================

    @contract(
        inputs={
            "data": {
                "format": "csv",
                "required": True,
                "schema": TabularSchema(
                    allow_missing=True,  # Accepts data WITH missing values
                    min_rows=1,
                ).to_dict(),
            }
        },
        outputs={
            "data": {
                "format": "csv",
                "schema": TabularSchema(
                    allow_missing=True,  # May still have missing values
                ).to_dict(),
            }
        },
        description="Remove specified columns from dataset",
        tags=["preprocessing", "cleaning"],
    )
    def clean_data(inputs: Dict[str, str], outputs: Dict[str, str], drop_cols: List[str]):
        df = IOManager.load(inputs["data"], "csv")
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        IOManager.save(df, outputs["data"], "csv")
        return f"Cleaned: {df.shape}"


    @contract(
        inputs={
            "data": {
                "format": "csv",
                "required": True,
                "schema": TabularSchema(
                    allow_missing=True,  # Accepts missing values
                ).to_dict(),
            }
        },
        outputs={
            "data": {
                "format": "csv",
                "schema": TabularSchema(
                    allow_missing=False,  # GUARANTEES no missing values!
                ).to_dict(),
            },
            "imputer": {
                "format": "pickle",
                "schema": ArtifactSchema(
                    artifact_type="imputer",
                    required_methods=["transform", "fit_transform"],
                ).to_dict(),
            },
        },
        description="Fill missing values in dataset",
        tags=["preprocessing", "imputation"],
    )
    def fill_missing(inputs: Dict[str, str], outputs: Dict[str, str], strategy: str = "mean"):
        from sklearn.impute import SimpleImputer

        df = IOManager.load(inputs["data"], "csv")
        imputer = SimpleImputer(strategy=strategy)

        # Only impute numeric columns
        num_cols = df.select_dtypes("number").columns
        df[num_cols] = imputer.fit_transform(df[num_cols])

        IOManager.save(df, outputs["data"], "csv")
        IOManager.save(imputer, outputs["imputer"], "pickle")
        return f"Imputed {len(num_cols)} columns"


    @contract(
        inputs={
            "train_data": {
                "format": "csv",
                "required": True,
                "schema": TabularSchema(
                    allow_missing=False,  # REQUIRES clean data (no missing)!
                    min_rows=10,
                ).to_dict(),
            }
        },
        outputs={
            "model": {
                "format": "pickle",
                "schema": ArtifactSchema(
                    artifact_type="model",
                    expected_class="LinearRegression",
                    required_methods=["predict", "fit"],
                ).to_dict(),
            },
            "metrics": {
                "format": "metrics_json",
                "schema": JSONSchema(
                    fields={"r2": "float", "samples": "int"},
                    required_fields=["r2", "samples"],
                ).to_dict(),
            },
        },
        description="Train a model and output metrics",
        tags=["training", "model"],
    )
    def train_model(inputs: Dict[str, str], outputs: Dict[str, str], label_col: str = "target"):
        from sklearn.linear_model import LinearRegression

        df = IOManager.load(inputs["train_data"], "csv")
        X = df.select_dtypes("number")
        y = np.random.rand(len(X))

        model = LinearRegression().fit(X, y)
        metrics = {"r2": 0.85, "samples": len(X)}

        IOManager.save(model, outputs["model"], "pickle")
        IOManager.save(metrics, outputs["metrics"], "metrics_json")
        return f"Trained on {len(X)} samples"


    @contract(
        inputs={
            "model": {
                "format": "pickle",
                "required": True,
                "schema": ArtifactSchema(
                    artifact_type="model",
                    required_methods=["predict"],
                ).to_dict(),
            },
            "data": {
                "format": "csv",
                "required": True,
                "schema": TabularSchema(
                    allow_missing=False,
                ).to_dict(),
            },
        },
        outputs={
            "predictions": {
                "format": "csv",
                "schema": TabularSchema(
                    columns={"prediction": "numeric"},
                    required_columns=["prediction"],
                ).to_dict(),
            }
        },
        description="Generate predictions using trained model",
        tags=["inference", "prediction"],
    )
    def predict(inputs: Dict[str, str], outputs: Dict[str, str]):
        model = IOManager.load(inputs["model"], "pickle")
        df = IOManager.load(inputs["data"], "csv")

        X = df.select_dtypes("number")
        preds = model.predict(X)

        result = pd.DataFrame({"prediction": preds})
        IOManager.save(result, outputs["predictions"], "csv")
        return f"Generated {len(preds)} predictions"


    # =========================================================================
    # 2. Show Available Formats
    # =========================================================================

    print("\n" + "=" * 80)
    print("AVAILABLE I/O FORMATS")
    print("=" * 80)
    IOManager.list_formats()


    # =========================================================================
    # 3. Show Registered Services
    # =========================================================================

    print("\n" + "=" * 80)
    print("REGISTERED SERVICES")
    print("=" * 80)
    ServiceRegistry.describe()


    # =========================================================================
    # 4. Validate Pipeline (Machine-Checkable)
    # =========================================================================

    print("\n" + "=" * 80)
    print("PIPELINE VALIDATION")
    print("=" * 80)

    # =========================================================================
    # ✅ VALID PIPELINE: clean → fill_missing → train → predict
    # Structural flow: dirty data → clean data → model
    # =========================================================================
    valid_pipeline = [
        {
            "service": "clean_data",
            "inputs": {"data": "raw.csv"},
            "outputs": {"data": "cleaned.csv"},
        },
        {
            "service": "fill_missing",
            "inputs": {"data": "cleaned.csv"},
            "outputs": {"data": "imputed.csv", "imputer": "imputer.pkl"},
        },
        {
            "service": "train_model",
            "inputs": {"train_data": "imputed.csv"},  # ✓ fill_missing outputs allow_missing=False
            "outputs": {"model": "model.pkl", "metrics": "metrics.json"},
        },
        {
            "service": "predict",
            "inputs": {"model": "model.pkl", "data": "imputed.csv"},
            "outputs": {"predictions": "preds.csv"},
        },
    ]

    print("\n--- Valid Pipeline (with structural schemas) ---")
    validate_pipeline(valid_pipeline)

    # =========================================================================
    # ❌ INVALID PIPELINE 1: Structure mismatch (dirty data → clean-only input)
    # clean_data outputs allow_missing=True
    # train_model requires allow_missing=False
    # =========================================================================
    invalid_structure_pipeline = [
        {
            "service": "clean_data",
            "inputs": {"data": "raw.csv"},
            "outputs": {"data": "cleaned.csv"},  # Output may have missing values
        },
        {
            "service": "train_model",
            "inputs": {"train_data": "cleaned.csv"},  # ❌ Requires NO missing values!
            "outputs": {"model": "model.pkl", "metrics": "metrics.json"},
        },
    ]

    print("\n--- Invalid Pipeline (structural mismatch: dirty → clean-only) ---")
    validate_pipeline(invalid_structure_pipeline)

    # =========================================================================
    # ❌ INVALID PIPELINE 2: Format mismatch (JSON → CSV)
    # =========================================================================
    @contract(
        inputs={"data": {"format": "csv", "required": True}},
        outputs={"stats": {"format": "json"}},
        description="Calculate statistics",
        tags=["stats"],
    )
    def calc_stats(inputs: Dict[str, str], outputs: Dict[str, str]):
        df = IOManager.load(inputs["data"], "csv")
        stats = {"mean": df.select_dtypes("number").mean().to_dict()}
        IOManager.save(stats, outputs["stats"], "json")
        return "Stats calculated"

    invalid_format_pipeline = [
        {
            "service": "calc_stats",
            "inputs": {"data": "raw.csv"},
            "outputs": {"stats": "stats.json"},  # Outputs JSON (dict)
        },
        {
            "service": "train_model",
            "inputs": {"train_data": "stats.json"},  # ❌ Expects CSV (DataFrame)!
            "outputs": {"model": "model.pkl", "metrics": "metrics.json"},
        },
    ]

    print("\n--- Invalid Pipeline (format mismatch: json → csv) ---")
    validate_pipeline(invalid_format_pipeline)


    # =========================================================================
    # 5. Execute Valid Pipeline
    # =========================================================================

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION")
    print("=" * 80)

    # Create test data
    os.makedirs("storage", exist_ok=True)
    pd.DataFrame({
        "A": range(10),
        "B": range(10),
        "C": ["x"] * 10
    }).to_csv("storage/raw.csv", index=False)

    print("\n--- Executing Valid Pipeline ---")
    print(clean_data(
        inputs={"data": "storage/raw.csv"},
        outputs={"data": "storage/cleaned.csv"},
        drop_cols=["C"]
    ))

    print(fill_missing(
        inputs={"data": "storage/cleaned.csv"},
        outputs={"data": "storage/imputed.csv", "imputer": "storage/imputer.pkl"},
        strategy="mean"
    ))

    print(train_model(
        inputs={"train_data": "storage/imputed.csv"},
        outputs={"model": "storage/model.pkl", "metrics": "storage/metrics.json"},
    ))

    print(predict(
        inputs={"model": "storage/model.pkl", "data": "storage/imputed.csv"},
        outputs={"predictions": "storage/preds.csv"},
    ))

    print("\n✅ Pipeline completed successfully!")
