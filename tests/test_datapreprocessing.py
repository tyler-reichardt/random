import pytest
from pyspark.sql import SparkSession, Row, DataFrameReader # Added DataFrameReader
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, DateType, DecimalType, FloatType # Added FloatType
from pyspark.sql.functions import col, monotonically_increasing_id, when, row_number # Ensure all are imported
from unittest.mock import MagicMock, patch, PropertyMock # Added PropertyMock
from decimal import Decimal
import pandas as pd

from functions.data_processing import *

@pytest.fixture(scope="session")
def spark():
    """PySpark session fixture for tests."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("pytest-spark-integration-tests")
        .config("spark.sql.shuffle.partitions", "2") # Adjusted for local testing
        .config("spark.sql.legacy.createHiveTableByDefault", "false")
        .config("spark.ui.showConsoleProgress", "false") # Quieter console
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "2")
        .getOrCreate()
    )
    yield session
    session.stop()

@pytest.fixture
def sample_delta_df(spark):
    """Provides a sample DataFrame, mimicking what read_and_process_delta might get from a Delta read."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("value", IntegerType(), True),
        StructField("extra_col1", StringType(), True),
        StructField("extra_col2", StringType(), True),
        StructField("timestamp_col", DateType(), True), 
        StructField("is_active", BooleanType(), True), 
        StructField("amount", DecimalType(10, 2), True) 
    ])
    data = [
        Row(id=1, name="Alice", value=100, extra_col1="e1a", extra_col2="e2a", timestamp_col=pd.to_datetime("2023-01-15").date(), is_active=True, amount=Decimal("123.45")),
        Row(id=2, name="Bob", value=200, extra_col1="e1b", extra_col2="e2b", timestamp_col=pd.to_datetime("2023-01-20").date(), is_active=False, amount=Decimal("67.89")),
        Row(id=3, name="Charlie", value=300, extra_col1="e1c", extra_col2="e2c", timestamp_col=pd.to_datetime("2023-01-01").date(), is_active=True, amount=Decimal("0.50")),
        Row(id=4, name="David", value=400, extra_col1="e1d", extra_col2="e2d", timestamp_col=pd.to_datetime("2023-01-25").date(), is_active=False, amount=Decimal("1000.00")),
        Row(id=1, name="Alice_old", value=50, extra_col1="e1a_old", extra_col2="e2a_old", timestamp_col=pd.to_datetime("2023-01-10").date(), is_active=False, amount=Decimal("10.00")), 
    ]
    return spark.createDataFrame(data, schema)

# Common mock setup for DataFrameReader
def setup_mock_spark_read(mock_read_property, return_df):
    mock_df_reader = MagicMock(spec=DataFrameReader)
    mock_read_property.return_value = mock_df_reader
    mock_df_reader.format.return_value = mock_df_reader  # format() returns self
    mock_df_reader.table.return_value = return_df
    return mock_df_reader # Return for specific assertions if needed


def test_get_latest_incident(spark, sample_delta_df):
    """Test get_latest_incident function."""
    df_latest = get_latest_incident(sample_delta_df, "id", "timestamp_col")
    
    assert df_latest.count() == 4 
    
    alice_latest = df_latest.filter(col("id") == 1).first()
    assert alice_latest is not None
    assert alice_latest["name"] == "Alice_old" 
    assert alice_latest["timestamp_col"] == pd.to_datetime("2023-01-10").date()

    # Check other IDs are present and unique
    assert df_latest.filter(col("id") == 2).count() == 1
    assert df_latest.filter(col("id") == 3).count() == 1
    assert df_latest.filter(col("id") == 4).count() == 1
    assert "rn" not in df_latest.columns


def test_convert_boolean_columns(spark):
    """Test convert_boolean_columns function."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("is_feature_active", BooleanType(), True),
        StructField("name", StringType(), True),
        StructField("another_bool", BooleanType(), True)
    ])
    data = [
        Row(id=1, is_feature_active=True, name="A", another_bool=False),
        Row(id=2, is_feature_active=False, name="B", another_bool=True),
        Row(id=3, is_feature_active=None, name="C", another_bool=None), # Null booleans handled by otherwise(0)
    ]
    df = spark.createDataFrame(data, schema)
    
    df_converted = convert_boolean_columns(df)
    
    # Check data types
    assert isinstance(df_converted.schema["is_feature_active"].dataType, IntegerType)
    assert isinstance(df_converted.schema["another_bool"].dataType, IntegerType)
    assert isinstance(df_converted.schema["name"].dataType, StringType) # Ensure non-boolean untouched
    
    # Check values
    results_data = [(r.id, r.is_feature_active, r.name, r.another_bool) for r in df_converted.orderBy("id").collect()]
    expected_data = [
        (1, 1, "A", 0),
        (2, 0, "B", 1),
        (3, 0, "C", 0), # Nulls become 0
    ]
    assert results_data == expected_data


def test_convert_decimal_columns(spark):
    """Test convert_decimal_columns function."""
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("price", DecimalType(10, 2), True),
        StructField("quantity", IntegerType(), True),
        StructField("weight", DecimalType(10, 2), True) 
    ])
    data = [
        Row(id=1, price=Decimal("19.99"), quantity=100, weight=Decimal("1.250")),
        Row(id=2, price=Decimal("0.50"), quantity=20, weight=Decimal("2.075")),
        Row(id=3, price=None, quantity=30, weight=None), 
    ]
    df = spark.createDataFrame(data, schema)
    
    df_converted = convert_decimal_columns(df)
    
    # Check data types
    assert df_converted.schema["price"].dataType == FloatType() # Direct type comparison
    assert df_converted.schema["weight"].dataType == FloatType() # Direct type comparison
    assert isinstance(df_converted.schema["quantity"].dataType, IntegerType)
    
    # Check values
    results = df_converted.select("id", "price", "weight").orderBy("id").collect()

    assert results[0]["price"] == pytest.approx(19.99)
    assert results[0]["weight"] == pytest.approx(1.250)

    assert results[1]["price"] == pytest.approx(0.50)
    assert results[1]["weight"] == pytest.approx(2.08)
    
    assert results[2]["price"] is None
    assert results[2]["weight"] is None
