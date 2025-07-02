"""
Shared pytest configuration and fixtures for TPC ML Pipeline tests.
"""

import pytest
from pyspark.sql import SparkSession
import mlflow
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session")
def spark():
    """
    Create a SparkSession for testing.
    This is a session-scoped fixture to avoid creating multiple Spark sessions.
    """
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("tpc-ml-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    
    yield spark
    
    # Cleanup
    spark.stop()

@pytest.fixture(scope="function")
def spark_session(spark):
    """
    Function-scoped Spark session that clears cache between tests.
    """
    spark.catalog.clearCache()
    yield spark

@pytest.fixture(scope="session")
def mlflow_test_env(tmp_path_factory):
    """
    Set up MLflow for testing with a temporary directory.
    """
    # Create temporary directory for MLflow
    mlflow_dir = tmp_path_factory.mktemp("mlflow")
    tracking_uri = f"file://{mlflow_dir}"
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    yield tracking_uri
    
    # Cleanup is automatic with tmp_path_factory

@pytest.fixture(autouse=True)
def reset_mlflow_experiment(mlflow_test_env):
    """
    Reset MLflow experiment for each test.
    """
    mlflow.set_experiment("test-experiment")
    yield
    # End any active runs
    if mlflow.active_run():
        mlflow.end_run()

@pytest.fixture
def sample_config():
    """
    Sample configuration for testing.
    """
    return {
        "catalog": "test_catalog",
        "schema": "test_schema",
        "model_name": "test_tpc_model",
        "tables": {
            "policy": "test_policy_table",
            "claims": "test_claims_table",
            "svi": "test_svi_table"
        },
        "features": {
            "numeric": ["feature1", "feature2", "feature3"],
            "categorical": ["cat1", "cat2"]
        }
    }

@pytest.fixture
def mock_spark_table(spark_session):
    """
    Factory fixture to create mock Spark tables.
    """
    def _create_table(data, schema, table_name=None):
        df = spark_session.createDataFrame(data, schema)
        if table_name:
            df.createOrReplaceTempView(table_name)
        return df
    
    return _create_table

# Pytest hooks

def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    # Add custom markers documentation
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "spark: marks tests that require Spark"
    )
    config.addinivalue_line(
        "markers", "mlflow: marks tests that use MLflow"
    )

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on test names.
    """
    for item in items:
        # Add markers based on test file names
        if "test_datapreprocessing" in str(item.fspath):
            item.add_marker(pytest.mark.spark)
        elif "test_featureengineering" in str(item.fspath):
            item.add_marker(pytest.mark.spark)
        elif "test_training" in str(item.fspath):
            item.add_marker(pytest.mark.spark)
            item.add_marker(pytest.mark.mlflow)
        
        # Add unit marker to all tests by default
        if not any(mark.name in ["integration", "slow"] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

# Test utilities

class SparkTestCase:
    """
    Base class for Spark-based test cases.
    """
    @staticmethod
    def assert_dataframe_equal(df1, df2, check_order=True):
        """
        Assert two DataFrames are equal.
        """
        if check_order:
            assert df1.collect() == df2.collect()
        else:
            assert sorted(df1.collect()) == sorted(df2.collect())
    
    @staticmethod
    def assert_schema_equal(df1, df2):
        """
        Assert two DataFrames have the same schema.
        """
        assert df1.schema == df2.schema
    
    @staticmethod
    def assert_column_exists(df, column_name):
        """
        Assert a column exists in the DataFrame.
        """
        assert column_name in df.columns, f"Column '{column_name}' not found in DataFrame"

# Export utilities
__all__ = ["SparkTestCase"]