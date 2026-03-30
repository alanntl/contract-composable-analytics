"""
Tests for the Data Contract System
=========================================

Tests cover:
1. Schema validation (TabularSchema, JSONSchema, ArtifactSchema, ImageSchema)
2. Schema compatibility checking
3. IOManager load/save/compatibility
4. @contract decorator (runtime schema validation)
5. validate_pipeline() function
"""

import os
import sys
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Ensure the app directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract import (
    TabularSchema,
    JSONSchema,
    ArtifactSchema,
    ImageSchema,
    schema_from_dict,
    check_schema_compatibility,
    IOManager,
    ServiceRegistry,
    contract,
    validate_pipeline,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_df():
    """A simple DataFrame for testing."""
    return pd.DataFrame({
        "price": [100.0, 200.0, 300.0],
        "name": ["a", "b", "c"],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
    })


@pytest.fixture
def sample_csv(tmp_dir, sample_df):
    """Write a sample CSV and return its path."""
    path = os.path.join(tmp_dir, "sample.csv")
    sample_df.to_csv(path, index=False)
    return path


# =============================================================================
# 1. TABULAR SCHEMA TESTS
# =============================================================================

class TestTabularSchema:
    def test_valid_dataframe(self, sample_df):
        schema = TabularSchema(
            columns={"price": "numeric", "name": "categorical"},
            required_columns=["price"],
        )
        is_valid, errors = schema.validate(sample_df)
        assert is_valid
        assert errors == []

    def test_missing_required_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        schema = TabularSchema(
            columns={"a": "numeric", "b": "numeric"},
            required_columns=["a", "b"],
        )
        is_valid, errors = schema.validate(df)
        assert not is_valid
        assert any("Missing required column" in e for e in errors)

    def test_wrong_column_type(self):
        df = pd.DataFrame({"price": ["not", "numeric", "data"]})
        schema = TabularSchema(columns={"price": "numeric"})
        is_valid, errors = schema.validate(df)
        assert not is_valid
        assert any("expected numeric" in e for e in errors)

    def test_min_rows_constraint(self):
        df = pd.DataFrame({"x": [1]})
        schema = TabularSchema(min_rows=5)
        is_valid, errors = schema.validate(df)
        assert not is_valid
        assert any("rows" in e for e in errors)

    def test_max_rows_constraint(self):
        df = pd.DataFrame({"x": range(100)})
        schema = TabularSchema(max_rows=10)
        is_valid, errors = schema.validate(df)
        assert not is_valid

    def test_missing_values_not_allowed(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        schema = TabularSchema(allow_missing=False)
        is_valid, errors = schema.validate(df)
        assert not is_valid
        assert any("Missing values" in e for e in errors)

    def test_missing_values_allowed(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        schema = TabularSchema(allow_missing=True)
        is_valid, errors = schema.validate(df)
        assert is_valid

    def test_non_dataframe_input(self):
        schema = TabularSchema()
        is_valid, errors = schema.validate({"not": "a dataframe"})
        assert not is_valid
        assert any("Expected DataFrame" in e for e in errors)

    def test_roundtrip_serialization(self):
        schema = TabularSchema(
            columns={"price": "numeric"},
            required_columns=["price"],
            min_rows=5,
            allow_missing=False,
        )
        d = schema.to_dict()
        restored = TabularSchema.from_dict(d)
        assert restored.columns == schema.columns
        assert restored.required_columns == schema.required_columns
        assert restored.min_rows == schema.min_rows
        assert restored.allow_missing == schema.allow_missing

    def test_compatibility_pass(self):
        out_schema = TabularSchema(columns={"price": "numeric", "name": "categorical"})
        in_schema = TabularSchema(
            columns={"price": "numeric"},
            required_columns=["price"],
        )
        compat, msg = out_schema.compatible_with(in_schema)
        assert compat

    def test_compatibility_missing_column(self):
        out_schema = TabularSchema(columns={"price": "numeric"})
        in_schema = TabularSchema(
            columns={"price": "numeric", "volume": "numeric"},
            required_columns=["price", "volume"],
        )
        compat, msg = out_schema.compatible_with(in_schema)
        assert not compat
        assert "volume" in msg

    def test_compatibility_type_mismatch(self):
        out_schema = TabularSchema(columns={"price": "categorical"})
        in_schema = TabularSchema(columns={"price": "numeric"})
        compat, msg = out_schema.compatible_with(in_schema)
        assert not compat

    def test_extra_columns_not_allowed(self):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        schema = TabularSchema(
            columns={"x": "numeric"},
            allow_extra_columns=False,
        )
        is_valid, errors = schema.validate(df)
        assert not is_valid
        assert any("Unexpected columns" in e for e in errors)


# =============================================================================
# 2. JSON SCHEMA TESTS
# =============================================================================

class TestJSONSchema:
    def test_valid_dict(self):
        schema = JSONSchema(
            fields={"rmse": "float", "n_samples": "int"},
            required_fields=["rmse"],
        )
        is_valid, errors = schema.validate({"rmse": 0.5, "n_samples": 100})
        assert is_valid

    def test_missing_required_field(self):
        schema = JSONSchema(
            fields={"rmse": "float"},
            required_fields=["rmse"],
        )
        is_valid, errors = schema.validate({"accuracy": 0.9})
        assert not is_valid
        assert any("Missing required field" in e for e in errors)

    def test_wrong_field_type(self):
        schema = JSONSchema(fields={"count": "int"})
        is_valid, errors = schema.validate({"count": "not_an_int"})
        assert not is_valid

    def test_nested_schema(self):
        schema = JSONSchema(
            fields={"name": "str"},
            nested={
                "config": JSONSchema(
                    fields={"lr": "float"},
                    required_fields=["lr"],
                )
            },
        )
        is_valid, errors = schema.validate({
            "name": "test",
            "config": {"lr": 0.01},
        })
        assert is_valid

    def test_nested_schema_failure(self):
        schema = JSONSchema(
            nested={
                "config": JSONSchema(
                    fields={"lr": "float"},
                    required_fields=["lr"],
                )
            },
        )
        is_valid, errors = schema.validate({"config": {}})
        assert not is_valid
        assert any("config." in e for e in errors)

    def test_non_dict_input(self):
        schema = JSONSchema()
        is_valid, errors = schema.validate([1, 2, 3])
        assert not is_valid

    def test_roundtrip_serialization(self):
        schema = JSONSchema(
            fields={"rmse": "float"},
            required_fields=["rmse"],
        )
        d = schema.to_dict()
        restored = JSONSchema.from_dict(d)
        assert restored.fields == schema.fields
        assert restored.required_fields == schema.required_fields

    def test_compatibility(self):
        out_schema = JSONSchema(fields={"rmse": "float", "name": "str"})
        in_schema = JSONSchema(
            fields={"rmse": "float"},
            required_fields=["rmse"],
        )
        compat, msg = out_schema.compatible_with(in_schema)
        assert compat


# =============================================================================
# 3. ARTIFACT SCHEMA TESTS
# =============================================================================

class TestArtifactSchema:
    def test_valid_artifact(self):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit([[1], [2], [3]], [1, 2, 3])

        schema = ArtifactSchema(
            artifact_type="model",
            required_methods=["predict", "fit"],
        )
        is_valid, errors = schema.validate(model)
        assert is_valid

    def test_missing_method(self):
        class FakeModel:
            pass

        schema = ArtifactSchema(required_methods=["predict"])
        is_valid, errors = schema.validate(FakeModel())
        assert not is_valid
        assert any("predict" in e for e in errors)

    def test_missing_attribute(self):
        class FakeModel:
            pass

        schema = ArtifactSchema(required_attrs=["n_estimators"])
        is_valid, errors = schema.validate(FakeModel())
        assert not is_valid

    def test_roundtrip_serialization(self):
        schema = ArtifactSchema(
            artifact_type="model",
            expected_class="sklearn.ensemble.RandomForestRegressor",
            required_methods=["predict"],
        )
        d = schema.to_dict()
        restored = ArtifactSchema.from_dict(d)
        assert restored.artifact_type == schema.artifact_type
        assert restored.expected_class == schema.expected_class


# =============================================================================
# 4. IMAGE SCHEMA TESTS
# =============================================================================

class TestImageSchema:
    def test_valid_image_path(self, tmp_dir):
        schema = ImageSchema(extensions=[".png", ".jpg"])
        is_valid, errors = schema.validate("/fake/path/image.png")
        assert is_valid

    def test_invalid_extension(self):
        schema = ImageSchema(extensions=[".png"])
        is_valid, errors = schema.validate("/fake/path/image.bmp")
        assert not is_valid

    def test_valid_numpy_array(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        schema = ImageSchema(min_width=100, min_height=100, channels=3)
        is_valid, errors = schema.validate(img)
        assert is_valid

    def test_wrong_channels(self):
        img = np.zeros((224, 224, 1), dtype=np.uint8)
        schema = ImageSchema(channels=3)
        is_valid, errors = schema.validate(img)
        assert not is_valid

    def test_too_small_image(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        schema = ImageSchema(min_width=100, min_height=100)
        is_valid, errors = schema.validate(img)
        assert not is_valid
        assert len(errors) == 2  # Both width and height


# =============================================================================
# 5. SCHEMA FACTORY
# =============================================================================

class TestSchemaFactory:
    def test_tabular_from_dict(self):
        schema = schema_from_dict({"type": "tabular", "columns": {"x": "numeric"}})
        assert isinstance(schema, TabularSchema)

    def test_json_from_dict(self):
        schema = schema_from_dict({"type": "json", "fields": {"x": "str"}})
        assert isinstance(schema, JSONSchema)

    def test_artifact_from_dict(self):
        schema = schema_from_dict({"type": "artifact", "artifact_type": "model"})
        assert isinstance(schema, ArtifactSchema)

    def test_image_from_dict(self):
        schema = schema_from_dict({"type": "image", "channels": 3})
        assert isinstance(schema, ImageSchema)

    def test_unknown_type_returns_none(self):
        assert schema_from_dict({"type": "unknown"}) is None

    def test_empty_dict_returns_none(self):
        assert schema_from_dict({}) is None

    def test_none_returns_none(self):
        assert schema_from_dict(None) is None


# =============================================================================
# 6. CHECK SCHEMA COMPATIBILITY
# =============================================================================

class TestCheckSchemaCompatibility:
    def test_dict_inputs(self):
        out_dict = {"type": "tabular", "columns": {"x": "numeric"}}
        in_dict = {"type": "tabular", "columns": {"x": "numeric"}, "required_columns": ["x"]}
        compat, msg = check_schema_compatibility(out_dict, in_dict)
        assert compat

    def test_no_schema_is_compatible(self):
        compat, msg = check_schema_compatibility(None, None)
        assert compat

    def test_mixed_none_compatible(self):
        out_schema = TabularSchema(columns={"x": "numeric"})
        compat, msg = check_schema_compatibility(out_schema, None)
        assert compat


# =============================================================================
# 7. IO MANAGER TESTS
# =============================================================================

class TestIOManager:
    def test_csv_roundtrip(self, tmp_dir, sample_df):
        path = os.path.join(tmp_dir, "test.csv")
        IOManager.save(sample_df, path, "csv", verbose=False)
        loaded = IOManager.load(path, "csv", verbose=False)
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 3
        assert "price" in loaded.columns

    def test_parquet_roundtrip(self, tmp_dir, sample_df):
        pytest.importorskip("pyarrow")
        path = os.path.join(tmp_dir, "test.parquet")
        IOManager.save(sample_df, path, "parquet", verbose=False)
        loaded = IOManager.load(path, "parquet", verbose=False)
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 3

    def test_json_roundtrip(self, tmp_dir):
        data = {"rmse": 0.5, "n_samples": 100}
        path = os.path.join(tmp_dir, "metrics.json")
        IOManager.save(data, path, "json", verbose=False)
        loaded = IOManager.load(path, "json", verbose=False)
        assert loaded["rmse"] == 0.5
        assert loaded["n_samples"] == 100

    def test_pickle_roundtrip(self, tmp_dir):
        data = {"model": "fake_model", "params": [1, 2, 3]}
        path = os.path.join(tmp_dir, "artifact.pkl")
        IOManager.save(data, path, "pickle", verbose=False)
        loaded = IOManager.load(path, "pickle", verbose=False)
        assert loaded == data

    def test_npy_roundtrip(self, tmp_dir):
        arr = np.array([1.0, 2.0, 3.0])
        path = os.path.join(tmp_dir, "array.npy")
        IOManager.save(arr, path, "npy", verbose=False)
        loaded = IOManager.load(path, "npy", verbose=False)
        np.testing.assert_array_equal(loaded, arr)

    def test_text_roundtrip(self, tmp_dir):
        text = "Hello, world!"
        path = os.path.join(tmp_dir, "test.txt")
        IOManager.save(text, path, "text", verbose=False)
        loaded = IOManager.load(path, "text", verbose=False)
        assert loaded == text

    def test_unknown_format_raises(self, tmp_dir):
        with pytest.raises(ValueError, match="Unknown format"):
            IOManager.load("fake.xyz", "xyz_unknown")

    def test_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            IOManager.load(os.path.join(tmp_dir, "nope.csv"), "csv")

    def test_compatible_same_type(self):
        assert IOManager.compatible("csv", "parquet")  # Both produce DataFrame

    def test_incompatible_types(self):
        assert not IOManager.compatible("csv", "json")  # DataFrame vs dict

    def test_infer_format(self):
        assert IOManager.infer_format("data.csv") == "csv"
        assert IOManager.infer_format("model.pkl") == "pickle"
        assert IOManager.infer_format("data.parquet") == "parquet"

    def test_register_custom_format(self, tmp_dir):
        IOManager.register_format(
            name="test_custom",
            read_fn=lambda p: open(p).read(),
            write_fn=lambda d, p: open(p, "w").write(d),
            produces="str",
            extensions=[".custom"],
        )
        path = os.path.join(tmp_dir, "test.custom")
        IOManager.save("hello", path, "test_custom", verbose=False)
        loaded = IOManager.load(path, "test_custom", verbose=False)
        assert loaded == "hello"
        # Clean up custom format
        del IOManager._REGISTRY["test_custom"]


# =============================================================================
# 8. SERVICE REGISTRY TESTS
# =============================================================================

class TestServiceRegistry:
    def test_register_and_get(self):
        def dummy_service(inputs, outputs):
            pass

        ServiceRegistry.register("test_dummy", {
            "description": "A test service",
            "tags": ["test"],
            "input": {"data": {"format": "csv"}},
            "output": {"data": {"format": "csv"}},
        }, dummy_service)

        info = ServiceRegistry.get("test_dummy")
        assert info is not None
        assert info["description"] == "A test service"
        assert ServiceRegistry.get_function("test_dummy") is dummy_service

        # Clean up
        del ServiceRegistry._services["test_dummy"]

    def test_find_by_tag(self):
        ServiceRegistry.register("tagged_svc", {
            "tags": ["preprocessing", "imputation"],
            "input": {},
            "output": {},
        })
        results = ServiceRegistry.find_by_tag("preprocessing")
        assert "tagged_svc" in results
        del ServiceRegistry._services["tagged_svc"]

    def test_find_by_format(self):
        ServiceRegistry.register("csv_svc", {
            "tags": [],
            "input": {"data": {"format": "csv"}},
            "output": {"result": {"format": "parquet"}},
        })
        assert "csv_svc" in ServiceRegistry.find_by_input_format("csv")
        assert "csv_svc" in ServiceRegistry.find_by_output_format("parquet")
        del ServiceRegistry._services["csv_svc"]

    def test_get_nonexistent(self):
        assert ServiceRegistry.get("nonexistent_service_xyz") is None
        assert ServiceRegistry.get_function("nonexistent_service_xyz") is None


# =============================================================================
# 9. @CONTRACT DECORATOR TESTS
# =============================================================================

class TestContractDecorator:
    def test_basic_contract_execution(self, tmp_dir):
        """Service runs normally when inputs/outputs are valid."""
        # Create input file
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        in_path = os.path.join(tmp_dir, "input.csv")
        out_path = os.path.join(tmp_dir, "output.csv")
        df.to_csv(in_path, index=False)

        @contract(
            inputs={"data": {"format": "csv", "required": True}},
            outputs={"data": {"format": "csv"}},
            description="Test service",
        )
        def passthrough(inputs, outputs):
            data = pd.read_csv(inputs["data"])
            data.to_csv(outputs["data"], index=False)

        passthrough({"data": in_path}, {"data": out_path})
        assert os.path.exists(out_path)

        # Clean up registry
        del ServiceRegistry._services["passthrough"]

    def test_missing_required_input_raises(self, tmp_dir):
        @contract(
            inputs={"data": {"format": "csv", "required": True}},
            outputs={"data": {"format": "csv"}},
        )
        def needs_data(inputs, outputs):
            pass

        with pytest.raises(ValueError, match="Missing required input slot"):
            needs_data({}, {"data": "out.csv"})

        del ServiceRegistry._services["needs_data"]

    def test_missing_input_file_raises(self, tmp_dir):
        @contract(
            inputs={"data": {"format": "csv", "required": True}},
            outputs={"data": {"format": "csv"}},
        )
        def needs_file(inputs, outputs):
            pass

        with pytest.raises(FileNotFoundError):
            needs_file(
                {"data": os.path.join(tmp_dir, "nonexistent.csv")},
                {"data": os.path.join(tmp_dir, "out.csv")},
            )

        del ServiceRegistry._services["needs_file"]

    def test_missing_output_file_raises(self, tmp_dir):
        df = pd.DataFrame({"x": [1]})
        in_path = os.path.join(tmp_dir, "in.csv")
        df.to_csv(in_path, index=False)

        @contract(
            inputs={"data": {"format": "csv", "required": True}},
            outputs={"result": {"format": "csv"}},
        )
        def no_output(inputs, outputs):
            pass  # Deliberately doesn't create output

        with pytest.raises(RuntimeError, match="Failed to produce output"):
            no_output(
                {"data": in_path},
                {"result": os.path.join(tmp_dir, "missing_output.csv")},
            )

        del ServiceRegistry._services["no_output"]

    def test_runtime_schema_validation_on_input(self, tmp_dir):
        """The @contract decorator should validate input data against the schema."""
        # Create a CSV with string data where numeric is expected
        df = pd.DataFrame({"price": ["not", "numeric", "data"]})
        in_path = os.path.join(tmp_dir, "bad_input.csv")
        out_path = os.path.join(tmp_dir, "output.csv")
        df.to_csv(in_path, index=False)

        @contract(
            inputs={
                "data": {
                    "format": "csv",
                    "required": True,
                    "schema": {"type": "tabular", "columns": {"price": "numeric"}},
                }
            },
            outputs={"data": {"format": "csv"}},
        )
        def expects_numeric(inputs, outputs):
            data = pd.read_csv(inputs["data"])
            data.to_csv(outputs["data"], index=False)

        with pytest.raises(ValueError, match="failed schema validation"):
            expects_numeric({"data": in_path}, {"data": out_path})

        del ServiceRegistry._services["expects_numeric"]

    def test_runtime_schema_validation_on_output(self, tmp_dir):
        """The @contract decorator should validate output data against the schema."""
        df_in = pd.DataFrame({"x": [1, 2, 3]})
        in_path = os.path.join(tmp_dir, "input.csv")
        out_path = os.path.join(tmp_dir, "output.csv")
        df_in.to_csv(in_path, index=False)

        @contract(
            inputs={"data": {"format": "csv", "required": True}},
            outputs={
                "data": {
                    "format": "csv",
                    "schema": {"type": "tabular", "min_rows": 100},
                }
            },
        )
        def produces_too_few_rows(inputs, outputs):
            data = pd.read_csv(inputs["data"])
            data.to_csv(outputs["data"], index=False)

        with pytest.raises(RuntimeError, match="failed schema validation"):
            produces_too_few_rows({"data": in_path}, {"data": out_path})

        del ServiceRegistry._services["produces_too_few_rows"]

    def test_contract_attached_to_function(self):
        @contract(
            inputs={"data": {"format": "csv"}},
            outputs={"data": {"format": "csv"}},
            description="Test desc",
            tags=["test"],
            version="2.0.0",
        )
        def my_service(inputs, outputs):
            pass

        assert hasattr(my_service, "contract")
        assert my_service.contract["description"] == "Test desc"
        assert my_service.contract["version"] == "2.0.0"
        assert "test" in my_service.contract["tags"]

        del ServiceRegistry._services["my_service"]

    def test_schema_validation_passes_for_valid_data(self, tmp_dir):
        """Schema validation should pass silently for valid data."""
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})
        in_path = os.path.join(tmp_dir, "good_input.csv")
        out_path = os.path.join(tmp_dir, "output.csv")
        df.to_csv(in_path, index=False)

        @contract(
            inputs={
                "data": {
                    "format": "csv",
                    "required": True,
                    "schema": {"type": "tabular", "columns": {"price": "numeric"}},
                }
            },
            outputs={"data": {"format": "csv"}},
        )
        def valid_service(inputs, outputs):
            data = pd.read_csv(inputs["data"])
            data.to_csv(outputs["data"], index=False)

        # Should not raise
        valid_service({"data": in_path}, {"data": out_path})
        assert os.path.exists(out_path)

        del ServiceRegistry._services["valid_service"]


# =============================================================================
# 10. VALIDATE PIPELINE TESTS
# =============================================================================

class TestValidatePipeline:
    def test_valid_pipeline(self, tmp_dir):
        """A pipeline with compatible services should pass validation."""
        @contract(
            inputs={"data": {"format": "csv"}},
            outputs={"data": {"format": "csv"}},
        )
        def step_a(inputs, outputs):
            pass

        @contract(
            inputs={"data": {"format": "csv"}},
            outputs={"model": {"format": "pickle"}},
        )
        def step_b(inputs, outputs):
            pass

        steps = [
            {"service": step_a, "inputs": {"data": "a.csv"}, "outputs": {"data": "b.csv"}},
            {"service": step_b, "inputs": {"data": "b.csv"}, "outputs": {"model": "m.pkl"}},
        ]
        is_valid, errors = validate_pipeline(steps, verbose=False)
        assert is_valid
        assert errors == []

        del ServiceRegistry._services["step_a"]
        del ServiceRegistry._services["step_b"]

    def test_unknown_service(self):
        steps = [
            {"service": "nonexistent_service_xyz_456", "inputs": {}, "outputs": {}},
        ]
        is_valid, errors = validate_pipeline(steps, verbose=False)
        assert not is_valid
        assert any("Unknown service" in e for e in errors)

    def test_format_mismatch(self, tmp_dir):
        """Pipeline with incompatible formats should fail."""
        @contract(
            inputs={"data": {"format": "csv"}},
            outputs={"data": {"format": "json"}},
        )
        def produces_json(inputs, outputs):
            pass

        @contract(
            inputs={"data": {"format": "csv"}},
            outputs={"result": {"format": "csv"}},
        )
        def expects_csv(inputs, outputs):
            pass

        steps = [
            {"service": produces_json, "inputs": {"data": "a.csv"}, "outputs": {"data": "b.json"}},
            {"service": expects_csv, "inputs": {"data": "b.json"}, "outputs": {"result": "c.csv"}},
        ]
        is_valid, errors = validate_pipeline(steps, verbose=False)
        assert not is_valid

        del ServiceRegistry._services["produces_json"]
        del ServiceRegistry._services["expects_csv"]
