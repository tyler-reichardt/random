import pytest
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, BooleanType, DateType,
    DecimalType, FloatType, TimestampType, DoubleType
)
from pyspark.sql.functions import (
    col, regexp_replace, expr, row_number, when, datediff, year,
    dayofweek, dayofmonth, month, regexp_extract, lit, greatest,
    current_date, floor, udf, array, monotonically_increasing_id,
    sum as _sum, first, mean, count # Add mean and count for testing
)
from pyspark.sql.functions import col as spark_col # Use this alias as in user's code
from pyspark.sql.window import Window
import datetime
import pandas as pd
import numpy as np
from functools import reduce
from operator import add
from typing import List, Dict, Any # Added Any
import math # For pytest.approx if needed for float comparisons
from unittest.mock import MagicMock, patch, mock_open # For mocking

# Imports for the new functions being tested
from pyspark.ml.feature import VectorAssembler, VectorIndexer # StandardScaler is from pyspark.ml.feature
from pyspark.ml.pipeline import PipelineModel # For mocking pipeline
from pyspark.ml import Pipeline as PySparkPipeline # Alias to avoid clash with sklearn.pipeline
from pyspark.ml.evaluation import Evaluator as PySparkEvaluator # Base class for GiniEvaluator stub
from pyspark.ml.tuning import CrossValidatorModel # For mocking CV model

# Sklearn imports for custom transformers and stubs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error 
# Assuming gini_normalized_original is a custom function. If it's from a library, import it.
# For now, a stub will be created.

# MLflow imports
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature, Schema as MLflowSchema # Renamed to avoid clash
from mlflow.types import ColSpec

# XGBoost (assuming it's the Python version for train_and_evaluate_cv_original)
import xgboost as xgb


# --- Placeholder for user's Spark Functions ---
# Assume the functions provided by the user are in a module named 'spark_transformations.py'
# For testing, we will assume these functions are available in the scope.
# If they were in a file, e.g., "my_ml_functions.py", you'd use:
from functions.training import *
# --- For self-contained testing, the improved functions from previous turn are included here ---
# --- In a real project, these would be imported from their module ---

@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder \
        .appName("unit tests") \
        .master("local[*]") \
        .getOrCreate()
    return spark


@pytest.fixture(scope="module")
def mlflow_client():
    return MlflowClient()

# Sample data for testing
@pytest.fixture
def sample_data(spark):
    schema = StructType([
        StructField("column_name", DoubleType(), True),
        StructField("PPTV_RISK_PCODE", StringType(), True)
    ])
    data = [(1.0, "A"), (2.0, "B"), (5.0, "C"), (7.0, "D"), (10.0, "E")]
    return spark.createDataFrame(data, schema)

# Sample trials for testing getBestModelfromTrials
@pytest.fixture
def trials():
    return Trials()


#@pytest.mark.parametrize("model_name, latest_version", [
#    ("test_model", 1),  # Control test for retrieving latest version of model
#])
#def test_get_latest_model_version(mlflow_client, model_name, latest_version):
#    assert get_latest_model_version(mlflow_client, model_name) == latest_version, f"Expected {latest_version}, but got {get_latest_model_version(mlflow_client, model_name)}"


#def test_get_hyper_params():
#    # Test with valid hyperparameters
#    hyperparams = {
#        "name='learning_rate'": 0.01,
#        "name='batch_size'": 32,
#        "name='num_epochs'": 10
#    }
#    expected = {
#        'learning_rate': 0.01,
#        'batch_size': 32,
#        'num_epochs': 10
#    }
#    assert get_hyper_params(hyperparams) == expected
#
#    # Test with a single hyperparameter
#    hyperparams = {
#        "name='dropout_rate'": 0.5
#    }
#    expected = {
#        'dropout_rate': 0.5
#    }
#    assert get_hyper_params(hyperparams) == expected
#
#    # Test with empty input
#    hyperparams = {}
#    expected = {}
#    assert get_hyper_params(hyperparams) == expected
#
#    # Test with multiple hyperparameters with different data types
#    hyperparams = {
#        "name='momentum'": 0.9,
#        "name='activation_function'": 'relu',
#        "name='weight_decay'": 0.0001
#    }
#    expected = {
#        'momentum': 0.9,
#        'activation_function': 'relu',
#        'weight_decay': 0.0001
#    }
#    assert get_hyper_params(hyperparams) == expected


def test_create_mlflow_experiment_path():
    # Test with a standard notebook path
    notebk_path = "/Users/user/notebooks/project"
    project_name = "my_project"
    expected = "/Users/user/my_project"
    assert create_mlflow_experiment_path(notebk_path, project_name) == expected

    # Test with a relative notebook path
    notebk_path = "Users/user/notebooks/project"
    project_name = "another_project"
    expected = "Users/user/another_project"
    assert create_mlflow_experiment_path(notebk_path, project_name) == expected

    # Test with a leading slash and two parts
    notebk_path = "/Users/user/notebooks"
    project_name = "test_project"
    expected = "/Users/user/test_project"
    assert create_mlflow_experiment_path(notebk_path, project_name) == expected

    # Test with a notebook path that has less than two parts
    notebk_path = "/Users"
    project_name = "short_project"
    expected = "/Users/short_project"
    assert create_mlflow_experiment_path(notebk_path, project_name) == expected


#def test_split_dataset(spark):
#    # Create a sample DataFrame
#    data = [
#        Row(PPTV_RISK_PCODE="A", feature1=1),
#        Row(PPTV_RISK_PCODE="B", feature1=2),
#        Row(PPTV_RISK_PCODE="C", feature1=3),
#        Row(PPTV_RISK_PCODE="D", feature1=4),
#        Row(PPTV_RISK_PCODE="E", feature1=5)
#    ]
#    df = spark.createDataFrame(data)
#
#    # Test with a 40% test size
#    train_df, test_df = split_dataset(df, test_size=0.4, seed=42)
#
#    # Validate the number of records
#    total_count = df.count()
#    expected_train_count = int(total_count * 0.6)
#    expected_test_count = total_count - expected_train_count
#
#    assert train_df.count() == expected_train_count
#    assert test_df.count() == expected_test_count
#
#    # Validate that the split is reproducible with the same seed
#    train_df2, test_df2 = split_dataset(df, test_size=0.4, seed=42)
#    assert train_df.collect() == train_df2.collect()
#    assert test_df.collect() == test_df2.collect()

@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("pytest-pyspark-transformations")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.sources.default", "delta")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") 
        .getOrCreate()
    )
    yield session
    session.stop()

#class TestRecastDtype:
#    def test_recast_to_int_and_string(self, spark):
#        data = [("1", "2.0", "abc")]
#        schema = StructType([StructField("col_a", StringType()), StructField("col_b", StringType()), StructField("col_c", StringType())])
#        df = spark.createDataFrame(data, schema)
#        df_recast = recast_dtype(df, ["col_a", "col_b"], "integer")
#        df_recast = recast_dtype(df_recast, ["col_c"], "string")
#        assert df_recast.schema["col_a"].dataType == IntegerType()
#        assert df_recast.schema["col_b"].dataType == IntegerType()
#        assert df_recast.schema["col_c"].dataType == StringType()
#        row = df_recast.first()
#        assert row["col_a"] == 1
#        assert row["col_b"] == 2 
#        assert row["col_c"] == "abc"
#
#    def test_recast_to_date(self, spark):
#        data = [("2023-01-01",)]
#        schema = StructType([StructField("date_str", StringType())])
#        df = spark.createDataFrame(data, schema)
#        df_recast = recast_dtype(df, ["date_str"], "date")
#        assert df_recast.schema["date_str"].dataType == DateType()
#        assert df_recast.first()["date_str"] == datetime.date(2023, 1, 1)
#
#    def test_recast_non_existent_column(self, spark):
#        data = [("val1",)]
#        schema = StructType([StructField("col_x", StringType())])
#        df = spark.createDataFrame(data, schema)
#        df_recast = recast_dtype(df, ["non_existent_col"], "integer")
#        assert "non_existent_col" not in df_recast.columns
#        assert df_recast.schema == df.schema
#
#class TestAddAssessmentColumns:
#    def test_assessment_categories(self, spark):
#        data = [("DriveableRepair",), ("DriveableTotalLoss",), ("UnroadworthyTotalLoss",),
#                ("UnroadworthyRepair",), ("OtherCategory",), (None,)]
#        schema = StructType([StructField("assessment_category", StringType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = add_assessment_columns(df)
#        results = {r.assessment_category: (r.da_dr, r.da_dtl, r.da_utl, r.da_ur, r.da_o) for r in processed_df.collect()}
#        assert results["DriveableRepair"] == (1, 0, 0, 0, 0)
#        assert results["DriveableTotalLoss"] == (0, 1, 0, 0, 0)
#        assert results["UnroadworthyTotalLoss"] == (0, 0, 1, 0, 0)
#        assert results["UnroadworthyRepair"] == (0, 0, 0, 1, 0)
#        assert results["OtherCategory"] == (0, 0, 0, 0, 1)
#        assert results[None] == (0, 0, 0, 0, 1)
#
#    def test_missing_assessment_category_column(self, spark):
#        data = [(1,)]
#        schema = StructType([StructField("id", IntegerType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = add_assessment_columns(df)
#        expected_cols = ['id', 'da_dr', 'da_dtl', 'da_utl', 'da_ur', 'da_o']
#        assert all(col_name in processed_df.columns for col_name in expected_cols)
#        row = processed_df.first()
#        assert (row.da_dr, row.da_dtl, row.da_utl, row.da_ur, row.da_o) == (0,0,0,0,1)
#
#
#class TestConvertColumnsToInt:
#    def test_convert_single_and_multiple_columns(self, spark):
#        data = [("10", "20.5", 30, "text")]
#        schema = StructType([StructField("s_col", StringType()), StructField("f_col", StringType()), 
#                             StructField("i_col", IntegerType()), StructField("str_col", StringType())])
#        df = spark.createDataFrame(data, schema)
#        df_converted = convert_columns_to_int(df, ["s_col", "f_col", "str_col"]) 
#        assert df_converted.schema["s_col"].dataType == IntegerType()
#        assert df_converted.schema["f_col"].dataType == IntegerType()
#        assert df_converted.schema["i_col"].dataType == IntegerType() 
#        assert df_converted.schema["str_col"].dataType == IntegerType()
#        row = df_converted.first()
#        assert row["s_col"] == 10
#        assert row["f_col"] == 20 
#        assert row["i_col"] == 30
#        assert row["str_col"] is None 
#
#    def test_convert_non_existent_column_int(self, spark):
#        data = [(1,)]
#        schema = StructType([StructField("id", IntegerType())])
#        df = spark.createDataFrame(data, schema)
#        df_converted = convert_columns_to_int(df, ["non_existent"])
#        assert df_converted.schema == df.schema
#
#
#class TestConvertColumnsToTimestamp:
#    def test_convert_string_to_timestamp(self, spark):
#        data = [("2023-01-01 10:30:00", "NotADate")]
#        schema = StructType([StructField("ts_str", StringType()), StructField("other_str", StringType())])
#        df = spark.createDataFrame(data, schema)
#        df_converted = convert_columns_to_timestamp(df, ["ts_str", "other_str"])
#        assert df_converted.schema["ts_str"].dataType == TimestampType()
#        assert df_converted.schema["other_str"].dataType == TimestampType()
#        row = df_converted.first()
#        assert row["ts_str"] == datetime.datetime(2023, 1, 1, 10, 30, 0)
#        assert row["other_str"] is None
#
#    def test_convert_non_existent_column_timestamp(self, spark):
#        data = [(1,)]
#        schema = StructType([StructField("id", IntegerType())])
#        df = spark.createDataFrame(data, schema)
#        df_converted = convert_columns_to_timestamp(df, ["non_existent"])
#        assert df_converted.schema == df.schema
#
#class TestProcessImpactSpeed:
#    @pytest.fixture
#    def impact_speed_base_df(self, spark):
#        data = [
#            (1, "Sedan", "Stationary", "MPH"), (2, "SUV", "ZeroToSix", "KMH"),
#            (3, "Hatch", "SixToTen", "MPH"), (4, "Sedan", "SevenToFourteen", "KMH"),
#            (5, "Truck", "OverSeventy", "MPH"), (6, "Van", "TwentyOneToThirty", "MPH"),
#            (7, "Sedan", "ThirtyOneToFourty", "KMH"), (8, "SUV", "FourtyOneToFifty", "MPH"),
#            (9, None, "FiftyOneToSixty", "KMH"), (10, "Hatch", None, "MPH"),
#            (11, "Sedan", "UnknownRange", "MPH"), (12, "SUV", "ZeroToSix", None),
#            (13, "Hatch", "SixToTen", "MPH") 
#        ]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("body_key", StringType()),
#            StructField("impact_speed_range", StringType()), StructField("impact_speed_unit", StringType())
#        ])
#        return spark.createDataFrame(data, schema)
#
#    def test_full_processing_logic(self, spark, impact_speed_base_df):
#        df = process_impact_speed(impact_speed_base_df)
#        results = {r.id: (r.body_key, r.impact_speed) for r in df.orderBy("id").collect()}
#
#        assert df.columns == ["id", "body_key", "impact_speed_range", "impact_speed_unit", "impact_speed"]
#        
#        # Most common body_key: Sedan (from data, Hatch also 3, Sedan appears first in groupby if not ordered explicitly before first())
#        # Let's assume 'Sedan' for test consistency if tie-broken by name or original order.
#        # The function `process_impact_speed` uses `orderBy(col("count").desc()).first()[0]` for most_common_body_style
#        # If counts are tied, the one returned by `first()` can be arbitrary without further ordering.
#        # For `most_common_impact_speed`, the robust version in tests adds a tie-breaker.
#        # Let's assume for body_key, 'Hatch' or 'Sedan' could be chosen if counts are equal.
#        # impact_speed_base_df has Sedan:4, Hatch:3, SUV:3. So Sedan is most common.
#
#        assert results[1] == ("Sedan", 0)
#        assert results[2] == ("SUV", 0) 
#        assert results[3] == ("Hatch", 6)
#        assert results[4] == ("Sedan", 4) 
#        assert results[5] == ("Truck", 71)
#        assert results[6] == ("Van", 21)
#        assert results[7] == ("Sedan", 19) 
#        assert results[8] == ("SUV", 41)
#        assert results[9] == ("Sedan", 31) 
#        
#        # Impact speeds before final fillna: [0, 0, 6, 4, 71, 21, 19, 41, 31, -1, -1, 1]
#        # Counts: 0 (2), -1 (2), 1(1), 4(1), 6(1), 19(1), 21(1), 31(1), 41(1), 71(1)
#        # Most common speed with tie-breaking (value asc): -1
#        assert results[10] == ("Hatch", -1) 
#        assert results[11] == ("Sedan", -1) 
#        assert results[12] == ("SUV", 1) 
#
#    def test_missing_input_columns_impact(self, spark):
#        data = [(1, "SomeValue")]
#        df_missing_all = spark.createDataFrame(data, StructType([StructField("id", IntegerType()), StructField("other_col", StringType())]))
#        processed_missing_all = process_impact_speed(df_missing_all)
#        assert "impact_speed" in processed_missing_all.columns
#        assert processed_missing_all.first().impact_speed == -1 
#        assert processed_missing_all.first().body_key == "Unknown_Body_Style"
#
#        df_missing_range = spark.createDataFrame([(1, "Sedan", "KMH")], StructType([StructField("id", IntegerType()), StructField("body_key", StringType()), StructField("impact_speed_unit", StringType())]))
#        processed_missing_range = process_impact_speed(df_missing_range)
#        assert processed_missing_range.first().impact_speed == -1 
#        assert processed_missing_range.first().body_key == "Sedan"
#
#    def test_all_null_body_key(self, spark):
#        data = [(1, None, "Stationary", "MPH")]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("body_key", StringType()),
#            StructField("impact_speed_range", StringType()), StructField("impact_speed_unit", StringType())
#        ])
#        df = spark.createDataFrame(data, schema)
#        processed_df = process_impact_speed(df)
#        assert processed_df.first().body_key == "Unknown_Body_Style" 
#        assert processed_df.first().impact_speed == 0
#
#    def test_all_null_impact_speed_after_map(self, spark): 
#        data = [(1, "Sedan", "Unknown1", "MPH"), (2, "SUV", "Unknown2", "KMH")]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("body_key", StringType()),
#            StructField("impact_speed_range", StringType()), StructField("impact_speed_unit", StringType())
#        ])
#        df = spark.createDataFrame(data, schema)
#        processed_df = process_impact_speed(df)
#        results = {r.id: r.impact_speed for r in processed_df.collect()}
#        assert results[1] == -1 # Initially -1, most common is -1
#        assert results[2] == -1
#
#
#class TestProcessDeployedAirbags:
#    def test_airbag_mapping(self, spark):
#        data = [("None",), ("One",), ("Two",), ("Three",), ("Four",), ("All",), ("SomethingElse",), (None,)]
#        schema = StructType([StructField("deployed_airbags_str", StringType())])
#        df = spark.createDataFrame(data, schema).withColumnRenamed("deployed_airbags_str", "deployed_airbags")
#        processed_df = process_deployed_airbags(df)
#        results = [r.deployed_airbags for r in processed_df.collect()]
#        assert results == [0, 1, 2, 3, 4, 5, -1, -1] 
#
#    def test_missing_airbag_column(self, spark):
#        data = [(1,)]
#        schema = StructType([StructField("id", IntegerType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = process_deployed_airbags(df)
#        assert "deployed_airbags" in processed_df.columns
#        assert processed_df.first().deployed_airbags == -1
#
#class TestCountDamagedAreas:
#    def test_count_logic(self, spark):
#        data = [(1, "dmg", None, "dmg"), (2, None, None, None), (3, "dmg", "dmg", "dmg")]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("d1", StringType()),
#            StructField("d2", StringType()), StructField("d3", StringType())
#        ])
#        df = spark.createDataFrame(data, schema)
#        damage_cols = ["d1", "d2", "d3"]
#        processed_df = count_damaged_areas(df, damage_cols)
#        
#        results = {r.id: (r.damage_recorded, r.damage_assessed) for r in processed_df.collect()}
#        assert results[1] == (2, 1) 
#        assert results[2] == (0, 0) 
#        assert results[3] == (3, 1) 
#
#    def test_empty_damage_columns_list(self, spark):
#        data = [(1, "val")]
#        schema = StructType([StructField("id", IntegerType()), StructField("val_col", StringType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = count_damaged_areas(df, [])
#        row = processed_df.first()
#        assert row.damage_recorded == 0
#        assert row.damage_assessed == 0
#
#    def test_non_existent_damage_columns(self, spark):
#        data = [(1, "val")]
#        schema = StructType([StructField("id", IntegerType()), StructField("d1", StringType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = count_damaged_areas(df, ["d1", "non_existent"])
#        row = processed_df.first()
#        assert row.damage_recorded == 1 
#        assert row.damage_assessed == 1
#
#
#class TestProcessDamageSeverity:
#    @pytest.fixture
#    def severity_base_df(self, spark):
#        data = [
#            (1, "Minimal", 1), (2, "Medium", 1), (3, "Heavy", 1), (4, "Severe", 1),
#            (5, "Unknown", 1), (6, "OtherString", 1), (7, "Minimal", 0), (8, None, 1)
#        ]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("sev_col", StringType()),
#            StructField("damage_assessed", IntegerType()) 
#        ])
#        return spark.createDataFrame(data, schema)
#
#    def test_severity_mapping(self, spark, severity_base_df):
#        processed_df = process_damage_severity(severity_base_df, ["sev_col"])
#        results = {r.id: r.sev_col for r in processed_df.collect()}
#        assert results[1] == 1  
#        assert results[2] == 2  
#        assert results[3] == 3  
#        assert results[4] == 4  
#        assert results[5] == -1 
#        assert results[6] == 0  
#        assert results[7] == -1 
#        assert results[8] == -1 
#
#    def test_missing_damage_assessed_col(self, spark):
#        data = [(1, "Minimal")]
#        schema = StructType([StructField("id", IntegerType()), StructField("sev_col1", StringType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = process_damage_severity(df, ["sev_col1"])
#        assert "damage_assessed" in processed_df.columns
#        assert processed_df.first().damage_assessed == 0
#        assert processed_df.first().sev_col1 == -1 
#
#    def test_empty_severity_columns_list_damage(self, spark):
#        data = [(1, 1)]
#        schema = StructType([StructField("id", IntegerType()), StructField("damage_assessed", IntegerType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = process_damage_severity(df, [])
#        assert len(processed_df.columns) == 2 
#
#
#class TestTransformFuelAndBodyType:
#    def test_fuel_and_body_encoding(self, spark):
#        data = [
#            (1, "1", "5 Door Hatchback"), (2, "2", "5 Door Estate"),
#            (3, "3", "4 Door Saloon"), (4, None, "3 Door Hatchback"),
#            (5, "1", None), (6, "X", "Y")
#        ]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("fuel_type_code", StringType()),
#            StructField("body_key", StringType())
#        ])
#        df = spark.createDataFrame(data, schema)
#        processed_df = transform_fuel_and_body_type(df)
#        
#        r = {row.id: row for row in processed_df.collect()}
#        assert (r[1].fuel_type_01, r[1].fuel_type_02) == (1,0)
#        assert (r[1].body_key_01,r[1].body_key_02,r[1].body_key_03,r[1].body_key_04) == (1,0,0,0)
#        assert (r[2].fuel_type_01, r[2].fuel_type_02) == (0,1)
#        assert (r[2].body_key_01,r[2].body_key_02,r[2].body_key_03,r[2].body_key_04) == (0,1,0,0)
#        assert (r[3].fuel_type_01, r[3].fuel_type_02) == (0,0) 
#        assert (r[3].body_key_01,r[3].body_key_02,r[3].body_key_03,r[3].body_key_04) == (0,0,1,0)
#        assert (r[4].fuel_type_01, r[4].fuel_type_02) == (0,0) 
#        assert (r[4].body_key_01,r[4].body_key_02,r[4].body_key_03,r[4].body_key_04) == (0,0,0,1)
#        assert (r[5].fuel_type_01, r[5].fuel_type_02) == (1,0) 
#        assert (r[5].body_key_01,r[5].body_key_02,r[5].body_key_03,r[5].body_key_04) == (0,0,0,0) 
#        assert (r[6].fuel_type_01, r[6].fuel_type_02) == (0,0) 
#        assert (r[6].body_key_01,r[6].body_key_02,r[6].body_key_03,r[6].body_key_04) == (0,0,0,0) 
#
#    def test_missing_input_cols_fuel_body(self, spark):
#        df_empty = spark.createDataFrame([(1,)], StructType([StructField("id", IntegerType())]))
#        processed_df = transform_fuel_and_body_type(df_empty)
#        expected_cols = ["fuel_type_01", "fuel_type_02", "body_key_01", "body_key_02", "body_key_03", "body_key_04"]
#        for col_name in expected_cols:
#            assert col_name in processed_df.columns
#            assert processed_df.first()[col_name] == 0
#
#class TestCreateTimeToNotifyColumn:
#    def test_time_to_notify_logic(self, spark):
#        data = [
#            (1, datetime.date(2023,1,10), datetime.date(2023,1,1)),   
#            (2, datetime.date(2023,1,1), datetime.date(2023,1,10)),  
#            (3, datetime.date(2023,2,15), datetime.date(2023,1,1)),  
#            (4, None, datetime.date(2023,1,1)),                      
#            (5, datetime.date(2023,1,1), None),                      
#            (6, datetime.date(2023,1,15), datetime.date(2023,1,15)), 
#        ]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("notification_date", DateType()),
#            StructField("incident_date", DateType())
#        ])
#        df = spark.createDataFrame(data, schema)
#        processed_df = create_time_to_notify_column(df)
#        results = {r.id: r.time_to_notify for r in processed_df.collect()}
#        assert results[1] == 9
#        assert results[2] == 0
#        assert results[3] == 30
#        assert results[4] == 0 
#        assert results[5] == 0 
#        assert results[6] == 0
#
#    def test_missing_date_cols_notify(self, spark):
#        df_empty = spark.createDataFrame([(1,)], StructType([StructField("id", IntegerType())]))
#        processed_df = create_time_to_notify_column(df_empty)
#        assert "time_to_notify" in processed_df.columns
#        assert processed_df.first().time_to_notify == 0
#
#class TestCreateVehicleAgeColumn:
#    def test_vehicle_age_calculation(self, spark):
#        data = [
#            (1, datetime.date(2023,1,10), "2020"), 
#            (2, datetime.date(2023,1,1), "2023"),  
#            (3, datetime.date(2020,5,5), "2015"),  
#            (4, None, "2010"),                     
#            (5, datetime.date(2023,1,1), None),    
#        ]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("incident_date", DateType()),
#            StructField("year_of_manufacture", StringType())
#        ])
#        df = spark.createDataFrame(data, schema)
#        processed_df = create_vehicle_age_column(df)
#        
#        results = {r.id: r.vehicle_age for r in processed_df.collect()}
#        assert results[1] == 3
#        assert results[2] == 0
#        assert results[3] == 5
#        assert results[4] is None
#        assert results[5] is None
#
#    def test_missing_cols_vehicle_age(self, spark):
#        df_empty = spark.createDataFrame([(1,)], StructType([StructField("id", IntegerType())]))
#        processed_df = create_vehicle_age_column(df_empty)
#        assert "vehicle_age" in processed_df.columns
#        assert processed_df.first().vehicle_age is None
#
#class TestConvertRightHandDrive:
#    def test_rhd_conversion(self, spark):
#        data = [(1, "R"), (2, "L"), (3, None), (4, "r")]
#        schema = StructType([StructField("id", IntegerType()), StructField("rhd_str", StringType())])
#        df = spark.createDataFrame(data, schema).withColumnRenamed("rhd_str", "right_hand_drive")
#        processed_df = convert_right_hand_drive(df)
#        results = {r.id: r.right_hand_drive for r in processed_df.collect()}
#        assert results[1] == 1
#        assert results[2] == 0
#        assert results[3] == 0 
#        assert results[4] == 0 
#
#    def test_missing_rhd_column(self, spark):
#        df_empty = spark.createDataFrame([(1,)], StructType([StructField("id", IntegerType())]))
#        processed_df = convert_right_hand_drive(df_empty)
#        assert "right_hand_drive" in processed_df.columns
#        assert processed_df.first().right_hand_drive == 0
#
#class TestEncodeDamageColumns:
#    @pytest.fixture
#    def damage_encode_df(self, spark):
#        data = [
#            (1, "true", "Minimal", "One"), (2, "false", "Medium", "Two"),
#            (3, "null", "Heavy", "X"), (4, None, "Severe", None),
#            (5, "Unknown", "Unknown", "All")
#        ]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("d1", StringType()), 
#            StructField("d2", StringType()), 
#            StructField("s1", StringType())  
#        ])
#        return spark.createDataFrame(data, schema)
#
#    def test_damage_encoding_logic(self, spark, damage_encode_df):
#        damage_cols = ["d1", "d2"]
#        special_cols = ["s1"]
#        processed_df = encode_damage_columns(damage_encode_df, damage_cols, special_cols)
#        
#        r = {row.id: row for row in processed_df.collect()}
#        assert (r[1].d1, r[1].d2, r[1].s1) == (1, 1, 1)   
#        assert (r[2].d1, r[2].d2, r[2].s1) == (0, 2, 2)   
#        assert (r[3].d1, r[3].d2, r[3].s1) == (0, 3, -99) 
#        assert (r[4].d1, r[4].d2, r[4].s1) == (0, 4, 0)   
#        assert (r[5].d1, r[5].d2, r[5].s1) == (0, 0, 4)   
#
#    def test_non_existent_cols_encode_damage(self, spark):
#        df = spark.createDataFrame([(1, "true")], StructType([StructField("id", IntegerType()), StructField("d1", StringType())]))
#        processed_df = encode_damage_columns(df, ["d1", "non_exist_dmg"], ["non_exist_sp"])
#        assert processed_df.first().d1 == 1
#        assert "non_exist_dmg" not in processed_df.columns
#        assert "non_exist_sp" not in processed_df.columns
#
#
#class TestCreateDateFields:
#    @pytest.fixture
#    def date_fields_df(self, spark):
#        data = [
#            (1, datetime.date(2023,1,1), datetime.date(2023,1,8), "AB12 3CD", 2, "Ford"), 
#            (2, datetime.date(2023,3,15), datetime.date(2023,3,13), "D4 5EF", 35, None), 
#            (3, None, datetime.date(2022,12,25), "G67", 5, "BMW"),
#            (4, datetime.date(2024,2,29), None, "", 10, "Audi"), 
#        ]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("notification_date", DateType()),
#            StructField("incident_date", DateType()), StructField("vehicle_kept_at_postcode", StringType()),
#            StructField("vehicle_age", IntegerType()), StructField("manufacturer_description", StringType())
#        ])
#        return spark.createDataFrame(data, schema)
#
#    def test_date_field_creation_and_fills(self, spark, date_fields_df):
#        processed_df = create_date_fields(date_fields_df)
#        r = {row.id: row for row in processed_df.orderBy("id").collect()}
#
#        assert (r[1].notified_day_of_week, r[1].notified_day, r[1].notified_month, r[1].notified_year) == (7,1,1,2023)
#        assert (r[1].incident_day_of_week, r[1].incident_day) == (7,8)
#        assert r[1].postcode_area == "AB"
#        assert r[1].vehicle_age == 2 
#        assert r[1].manufacturer_description == "Ford"
#
#        assert (r[2].notified_day_of_week, r[2].notified_day, r[2].notified_month, r[2].notified_year) == (3,15,3,2023)
#        assert (r[2].incident_day_of_week, r[2].incident_day) == (1,13)
#        assert r[2].postcode_area == "D"
#        assert r[2].vehicle_age is None 
#        assert r[2].manufacturer_description == "zmissing" 
#
#        assert (r[3].notified_day_of_week, r[3].notified_day, r[3].notified_month, r[3].notified_year) == (None,None,None,None)
#        assert (r[3].incident_day_of_week, r[3].incident_day) == (7,25) 
#        assert r[3].postcode_area == "G"
#        assert r[3].vehicle_age == 5
#        assert r[3].manufacturer_description == "BMW"
#
#        assert (r[4].notified_day_of_week, r[4].notified_day, r[4].notified_month, r[4].notified_year) == (4,29,2,2024) 
#        assert (r[4].incident_day_of_week, r[4].incident_day) == (None,None)
#        assert r[4].postcode_area == "zz" 
#        assert r[4].vehicle_age == 10
#        assert r[4].manufacturer_description == "Audi"
#        
#    def test_missing_all_relevant_cols_date_fields(self, spark):
#        df = spark.createDataFrame([(1,)], StructType([StructField("id", IntegerType())]))
#        processed_df = create_date_fields(df)
#        r = processed_df.first()
#        assert (r.notified_day_of_week, r.notified_day, r.notified_month, r.notified_year) == (None,None,None,None)
#        assert (r.incident_day_of_week, r.incident_day) == (None,None)
#        assert r.postcode_area == "zz" 
#        assert "vehicle_age" not in r.asDict() # asDict() to check if column exists if it could be None
#        assert r.manufacturer_description == "zmissing" 
#
#class TestImputeMissingWithMedian:
#    @pytest.fixture
#    def median_df(self, spark):
#        data = [
#            (1, 10.0, 100, "A"), (2, 20.0, None, "B"), (3, None, 300, "C"),
#            (4, 40.0, 400, "D"), (5, 10.0, 500, "E"), (6, None, None, "F")
#        ] 
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("num_col1", DoubleType()),
#            StructField("num_col2", IntegerType()), StructField("str_col", StringType())
#        ])
#        return spark.createDataFrame(data, schema)
#
#    def test_median_imputation(self, spark, median_df):
#        median1 = median_df.approxQuantile("num_col1", [0.5], 0.001)[0] 
#        median2 = median_df.approxQuantile("num_col2", [0.5], 0.001)[0] 
#        
#        processed_df = impute_missing_with_median(median_df, ["num_col1", "num_col2"])
#        r = {row.id: row for row in processed_df.collect()}
#
#        assert r[1].num_col1 == 10.0
#        assert r[2].num_col1 == 20.0
#        assert pytest.approx(r[3].num_col1) == median1 
#        assert pytest.approx(r[6].num_col1) == median1
#
#        assert r[1].num_col2 == 100
#        assert r[2].num_col2 == median2
#        assert r[6].num_col2 == median2
#        
#    def test_median_non_numeric_and_all_null(self, spark):
#        data = [(1, None, "A"), (2, None, "B")]
#        schema = StructType([StructField("id", IntegerType()), StructField("all_null_num", IntegerType()), StructField("str_col", StringType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = impute_missing_with_median(df, ["all_null_num", "str_col"])
#        assert processed_df.first().all_null_num is None 
#        assert processed_df.first().str_col == "A" 
#
#class TestCalculateDamageSeverity:
#    @pytest.fixture
#    def damage_sev_df(self, spark):
#        data = [
#            (1, 1, 2, 3), (2, 0, 0, 0), (3, 4, None, 1), 
#            (4, None, None, None), (5, 0, 1, 0)
#        ]
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("sev1", IntegerType()),
#            StructField("sev2", IntegerType()), StructField("sev3", IntegerType())
#        ])
#        return spark.createDataFrame(data, schema)
#
#    def test_damage_severity_calcs(self, spark, damage_sev_df):
#        sev_cols = ["sev1", "sev2", "sev3"]
#        processed_df = calculate_damage_severity(damage_sev_df, sev_cols)
#        r = {row.id: row for row in processed_df.collect()}
#
#        assert (r[1].damage_sev_total, r[1].damage_sev_count, r[1].damage_sev_mean, r[1].damage_sev_max) == (6,3,2.0,3)
#        assert (r[2].damage_sev_total, r[2].damage_sev_count, r[2].damage_sev_mean, r[2].damage_sev_max) == (0,0,0.0,0)
#        assert (r[3].damage_sev_total, r[3].damage_sev_count, r[3].damage_sev_mean, r[3].damage_sev_max) == (5,2,2.5,4)
#        assert (r[4].damage_sev_total, r[4].damage_sev_count, r[4].damage_sev_mean, r[4].damage_sev_max) == (0,0,0.0,0)
#        assert (r[5].damage_sev_total, r[5].damage_sev_count, r[5].damage_sev_mean, r[5].damage_sev_max) == (1,1,1.0,1)
#
#    def test_empty_sev_damage_list_calc(self, spark):
#        df = spark.createDataFrame([(1,)], StructType([StructField("id", IntegerType())]))
#        processed_df = calculate_damage_severity(df, [])
#        r = processed_df.first()
#        assert (r.damage_sev_total, r.damage_sev_count, r.damage_sev_mean, r.damage_sev_max) == (0,0,0.0,0)
#
#class TestFillInsurerName:
#    def test_fill_insurer(self, spark):
#        data = [(1, "InsurerA"), (2, None), (3, "InsurerB")]
#        schema = StructType([StructField("id", IntegerType()), StructField("insurer_name", StringType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = fill_insurer_name(df)
#        r = {row.id: row.insurer_name for row in processed_df.collect()}
#        assert r[1] == "InsurerA"
#        assert r[2] == "Unknown" 
#        assert r[3] == "InsurerB"
#
#    def test_missing_insurer_col(self, spark):
#        df = spark.createDataFrame([(1,)], StructType([StructField("id", IntegerType())]))
#        processed_df = fill_insurer_name(df)
#        assert "insurer_name" in processed_df.columns
#        assert processed_df.first().insurer_name == "Unknown"
#
#class TestFillNaWithMostCommon:
#    @pytest.fixture
#    def most_common_df(self, spark):
#        data = [
#            (1, "A", 10), (2, "B", None), (3, "A", 30), (4, None, 10),
#            (5, "A", 10), (6, "C", None), (7, "B", 20)
#        ] 
#        schema = StructType([
#            StructField("id", IntegerType()), StructField("col1", StringType()),
#            StructField("col2", IntegerType())
#        ])
#        return spark.createDataFrame(data, schema)
#
#    def test_fill_most_common_logic(self, spark, most_common_df):
#        processed_df = fill_na_with_most_common(most_common_df, ["col1", "col2"])
#        r = {row.id: (row.col1, row.col2) for row in processed_df.orderBy("id").collect()}
#        assert r[1] == ("A", 10)
#        assert r[2] == ("B", 10) 
#        assert r[3] == ("A", 30)
#        assert r[4] == ("A", 10) 
#        assert r[5] == ("A", 10)
#        assert r[6] == ("C", 10) 
#
#    def test_all_null_col_most_common(self, spark):
#        data = [(1, None), (2, None)]
#        schema = StructType([StructField("id", IntegerType()), StructField("all_null_col", StringType())])
#        df = spark.createDataFrame(data, schema)
#        processed_df = fill_na_with_most_common(df, ["all_null_col"])
#        assert processed_df.filter(col("all_null_col").isNull()).count() == 2
#
## --- Stubs for dependencies of functions to be tested ---
#def gini_normalized_original(y_true, y_pred):
#    """Stub for Gini calculation."""
#    # A very basic stub. Real Gini is more complex.
#    # This ensures the test for train_and_evaluate_cv_original can run.
#    if isinstance(y_true, pd.Series): y_true = y_true.to_numpy()
#    if isinstance(y_pred, pd.Series): y_pred = y_pred.to_numpy()
#    
#    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred): return 0.0
#
#    # Example: return a simple correlation or a fixed value for testing flow
#    # For a more meaningful stub, you might implement a simplified Gini or use a known library's Gini.
#    # Here, just returning a dummy value.
#    return 0.5 
#
#class GiniEvaluator(PySparkEvaluator):
#    """Stub for Spark GiniEvaluator."""
#    def __init__(self, predictionCol="prediction", labelCol="label"):
#        super().__init__()
#        self._predictionCol = predictionCol
#        self._labelCol = labelCol
#
#    def _evaluate(self, dataset: DataFrame) -> float:
#        # This is a stub. A real Gini for Spark would require more complex logic.
#        # It might convert columns to RDDs or Pandas for calculation if no direct Spark Gini.
#        # Returning a dummy value for testing flow.
#        return 0.6 
#    
#    # Required for PySpark >= 3.0 if not implementing isLargerBetter
#    def isLargerBetter(self) -> bool:
#        return True
#
#
#def create_param_grid(estimator, params_dict):
#    """Stub for creating a parameter grid for Spark CrossValidator."""
#    # This is a simplified stub. Real ParamGridBuilder is more flexible.
#    from pyspark.ml.tuning import ParamGridBuilder
#    if not params_dict:
#        return ParamGridBuilder().build() # Empty grid
#    
#    builder = ParamGridBuilder()
#    for param_name, values in params_dict.items():
#        # Assuming estimator has a param with param_name
#        if hasattr(estimator, param_name):
#            param_obj = getattr(estimator, param_name)
#            if isinstance(values, list):
#                builder = builder.addGrid(param_obj, values)
#            else: # Single value
#                builder = builder.addGrid(param_obj, [values])
#        else:
#            print(f"Warning (create_param_grid_stub): Estimator {estimator} has no param {param_name}")
#    return builder.build()
#
## --- Tests for new ML/Pandas/Sklearn functions ---
#
#class TestTrainAndEvaluateCvOriginal:
#    @pytest.fixture
#    def sample_data_pandas(self):
#        X_train = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})
#        y_train = pd.Series(np.random.rand(100) * 10)
#        X_test = pd.DataFrame({'feature1': np.random.rand(50), 'feature2': np.random.rand(50)})
#        y_test = pd.Series(np.random.rand(50) * 10)
#        weights = pd.Series(np.ones(100))
#        return X_train, y_train, X_test, y_test, weights
#
#    @patch('mlflow.start_run')
#    @patch('mlflow.log_params')
#    @patch('mlflow.log_metric')
#    @patch('mlflow.xgboost.log_model')
#    @patch('xgboost.train')
#    @patch('xgboost.DMatrix')
#    # Patching the user's own get_latest_model_version as it's also a function to be tested separately
#    @patch(__name__ + '.get_latest_model_version' if __name__ == '__main__' else 'YOUR_MODULE_NAME.get_latest_model_version') # Adjust YOUR_MODULE_NAME
#    def test_train_evaluate_cv_original_flow(self, mock_get_latest_version, mock_dmatrix, mock_xgb_train, 
#                                             mock_log_model, mock_log_metric, mock_log_params, 
#                                             mock_start_run, sample_data_pandas, mocker):
#        X_train, y_train, X_test, y_test, weights = sample_data_pandas
#        
#        mock_bst = MagicMock()
#        mock_bst.predict.return_value = np.random.rand(len(y_test))
#        mock_xgb_train.return_value = mock_bst
#        mock_dmatrix.side_effect = lambda data, label, weight=None: MagicMock() # Simple mock for DMatrix
#        
#        mock_mlflow_client = MagicMock(spec=MlflowClient)
#        mock_get_latest_version.return_value = 1 # Mock the dependency
#
#        params = {'eta': 0.1, 'max_depth': 3}
#        registered_model_name = "test_xgb_model"
#        label = "test_label"
#
#        # Call the function under test
#        # Assuming gini_normalized_original is globally available or imported
#        # And get_latest_model_version is also available/imported or mocked
#        train_and_evaluate_cv_original(
#            X_train, y_train, X_test, y_test, weights, params, 
#            registered_model_name, mock_mlflow_client, label, num_round=10
#        )
#
#        mock_start_run.assert_called_once()
#        mock_dmatrix.assert_any_call(X_train, label=y_train, weight=weights)
#        mock_dmatrix.assert_any_call(X_test, label=y_test)
#        mock_xgb_train.assert_called_once_with(params, mock_dmatrix.return_value, 10)
#        mock_bst.predict.assert_called_once()
#        
#        assert mock_log_params.call_count == 1
#        assert mock_log_metric.call_count == 2 # rmse, gini_score
#        mock_log_model.assert_called_once()
#        mock_get_latest_version.assert_called_once_with(mock_mlflow_client, registered_model_name)
#        mock_mlflow_client.set_registered_model_alias.assert_called_once_with(registered_model_name, 'champion', 1)
#

class TestCreateMlflowExperimentPath:
    def test_path_creation(self):
        assert create_mlflow_experiment_path("/Users/testuser/project", "my_exp") == "/Users/testuser/my_exp"
        assert create_mlflow_experiment_path("Users/testuser/project/sub", "exp2") == "Users/testuser/exp2"
        assert create_mlflow_experiment_path("/some/path/here", "proj1") == "/some/path/proj1"
        assert create_mlflow_experiment_path("nix/path/here", "proj2") == "nix/path/proj2" # handles no leading slash


#class TestSparkTrainAndEvaluate:
#    @pytest.fixture
#    def sample_spark_df(self, spark):
#        data = [(1.0, 2.0, 3.0, 1), (0.0, 1.5, 2.5, 0), (2.0, 0.5, 1.0, 1)]
#        schema = StructType([
#            StructField("f1", DoubleType()), StructField("f2", DoubleType()),
#            StructField("f3", DoubleType()), StructField("label_col", IntegerType())
#        ])
#        return spark.createDataFrame(data, schema)
#
#    @patch('pyspark.ml.tuning.CrossValidator.fit') # Mock the fit method of CV
#    def test_train_evaluate_cv_pipeline_creation(self, mock_cv_fit, spark, sample_spark_df, mocker):
#        # Mock the SparkContext 'sc' if your function uses it directly e.g. sc.defaultParallelism
#        # If SparkXGBRegressor or other parts need 'sc', ensure it's available.
#        # For testing, we can patch where 'sc' is accessed or ensure spark.sparkContext is used.
#        # The function uses sc.defaultParallelism. We can mock 'sc' globally for this test.
#        mock_sc = MagicMock()
#        mock_sc.defaultParallelism = 2
#        mocker.patch('__main__.sc', mock_sc, create=True) # Or 'your_module.sc'
#        
#        # Mock the CrossValidator's fit method to return a dummy model
#        mock_cv_model = MagicMock(spec=CrossValidatorModel)
#        mock_cv_fit.return_value = mock_cv_model
#
#        params = {"max_depth": [3, 5], "n_estimators": [10, 20]} # Example params for create_param_grid
#        
#        pipeline_cv = train_and_evaluate(sample_spark_df, "label_col", params=params, tuning_type='cv', k=2)
#        assert isinstance(pipeline_cv, PySparkPipeline)
#        assert len(pipeline_cv.getStages()) == 4 # Assembler, Scaler, Indexer, CV
#        assert isinstance(pipeline_cv.getStages()[3], PySparkPipeline) # CV itself is a stage, but it wraps the estimator. The user code puts CV as the last stage.
#
#        pipeline_no_cv = train_and_evaluate(sample_spark_df, "label_col", tuning_type='none') # 'none' or any other string
#        assert isinstance(pipeline_no_cv, PySparkPipeline)
#        assert len(pipeline_no_cv.getStages()) == 4 # Assembler, Scaler, Indexer, XGBRegressor
#
#        pipeline_both = train_and_evaluate(sample_spark_df, "label_col", tuning_type='both', k=2)
#        assert isinstance(pipeline_both, PySparkPipeline)
#        assert len(pipeline_both.getStages()) == 4
#
#
#class TestLogBestModel:
#    @patch('mlflow.set_registry_uri')
#    @patch('mlflow.tracking.MlflowClient')
#    @patch('mlflow.start_run')
#    @patch('mlflow.log_params')
#    @patch('mlflow.log_metric')
#    @patch('mlflow.spark.log_model')
#    @patch(__name__ + '.get_latest_model_version' if __name__ == '__main__' else 'YOUR_MODULE_NAME.get_latest_model_version') # Adjust
#    def test_log_best_model_flow(self, mock_get_latest, mock_spark_log_model, mock_log_metric, 
#                                 mock_log_params, mock_start_run, mock_mlflow_client_constructor, 
#                                 mock_set_registry, spark, mocker):
#        mock_client_instance = MagicMock(spec=MlflowClient)
#        mock_mlflow_client_constructor.return_value = mock_client_instance
#        mock_get_latest.return_value = 2 # Test "Challenger" alias
#
#        # Sample Spark DataFrame for test_df and transformed_test_data
#        data = [(1, [0.1, 0.2], 1.0), (2, [0.3, 0.4], 0.0)]
#        schema_test = StructType([StructField("id", IntegerType()), 
#                                  StructField("features", ArrayType(DoubleType())), # Assuming features col after transformation
#                                  StructField("label", DoubleType())])
#        test_df_spark = spark.createDataFrame(data, schema_test)
#        
#        # transformed_test_data is also a Spark DF, used for infer_signature's X part
#        transformed_test_data_spark = test_df_spark.select("features")
#
#
#        # Mock pipeline and its transform method
#        mock_pipeline = MagicMock(spec=PipelineModel) # Spark ML PipelineModel
#        predictions_data = [(1, [0.1,0.2], 1.0, 0.8), (2, [0.3,0.4], 0.0, 0.2)]
#        predictions_schema = StructType(schema_test.fields + [StructField("prediction", DoubleType())])
#        predictions_df_spark = spark.createDataFrame(predictions_data, predictions_schema)
#        mock_pipeline.transform.return_value = predictions_df_spark
#        
#        mock_gini_evaluator = MagicMock(spec=GiniEvaluator) # Use our stub or a MagicMock
#        mock_gini_evaluator.evaluate.return_value = 0.75
#
#        best_hyperparams = {"max_depth": 5, "n_estimators": 100}
#        registered_model_name = "my_best_spark_model"
#
#        log_best_model(
#            test_df_spark, transformed_test_data_spark, mock_pipeline, 
#            best_hyperparams, mock_gini_evaluator, registered_model_name
#        )
#
#        mock_set_registry.assert_called_once_with("databricks-uc")
#        mock_pipeline.transform.assert_called_once_with(test_df_spark)
#        mock_gini_evaluator.evaluate.assert_called_once_with(predictions_df_spark)
#        
#        mock_start_run.assert_called_once()
#        mock_log_params.assert_called_once_with(best_hyperparams)
#        mock_log_metric.assert_called_once_with("gini_score", 0.75)
#        
#        # Check signature inference (this part is tricky to assert precisely without deeper inspection)
#        # For now, just check log_model was called.
#        mock_spark_log_model.assert_called_once()
#        args, kwargs = mock_spark_log_model.call_args
#        assert kwargs['registered_model_name'] == registered_model_name
#        assert isinstance(kwargs['signature'], ModelSignature)
#
#        mock_get_latest.assert_called_once_with(mock_client_instance, registered_model_name)
#        mock_client_instance.set_registered_model_alias.assert_called_once_with(registered_model_name, "Challenger", 2)
#
#class TestGetLatestModelVersion:
#    def test_get_latest_version(self, mocker):
#        mock_client = MagicMock(spec=MlflowClient)
#        mock_version_1 = MagicMock()
#        mock_version_1.version = "1"
#        mock_version_3 = MagicMock()
#        mock_version_3.version = "3"
#        mock_version_2 = MagicMock()
#        mock_version_2.version = "2"
#        
#        mock_client.search_model_versions.return_value = [mock_version_1, mock_version_3, mock_version_2]
#        
#        latest = get_latest_model_version(mock_client, "test_model")
#        assert latest == 3
#        mock_client.search_model_versions.assert_called_once_with("name='test_model'")
#
#    def test_get_latest_no_versions(self, mocker):
#        mock_client = MagicMock(spec=MlflowClient)
#        mock_client.search_model_versions.return_value = []
#        latest = get_latest_model_version(mock_client, "empty_model")
#        assert latest == 1 # Returns 1 by default as per function
#
#class TestGetModelAliases:
#    def test_get_aliases(self, mocker):
#        mock_client = MagicMock(spec=MlflowClient)
#        
#        # Mock search_model_versions
#        mv1 = MagicMock(); mv1.version = "1"
#        mv2 = MagicMock(); mv2.version = "2"
#        mv3 = MagicMock(); mv3.version = "3"
#        mock_client.search_model_versions.return_value = [mv1, mv2, mv3]
#
#        # Mock get_model_version
#        mvm_v1 = MagicMock(); mvm_v1.aliases = ["champion"]
#        mvm_v2 = MagicMock(); mvm_v2.aliases = ["challenger"]
#        mvm_v3 = MagicMock(); mvm_v3.aliases = [] # No alias
#
#        def get_model_version_side_effect(name, version):
#            if version == "1": return mvm_v1
#            if version == "2": return mvm_v2
#            if version == "3": return mvm_v3
#            return MagicMock()
#        mock_client.get_model_version.side_effect = get_model_version_side_effect
#        
#        aliases = get_model_aliases(mock_client, "model_with_aliases")
#        assert sorted(aliases) == sorted(["champion", "challenger"])
#        assert mock_client.search_model_versions.call_count == 1
#        assert mock_client.get_model_version.call_count == 3
#
#
#class TestProcessDataFramePandas:
#    def test_process_dataframe(self):
#        data = {
#            'OldColA': [1, 2], 'OldColB': [3, 4],
#            'body_key_01': [1, 0], 'body_key_02': [0, 1], 
#            'body_key_03': [0, 0], 'body_key_04': [0, 0],
#            'some_val': [10, 20], 'first_party_confirmed': [1, 0]
#        }
#        df = pd.DataFrame(data)
#        rename_map = {'OldColA': 'NewColA', 'OldColB': 'NewColB'}
#        fp_tp_cols = ['some_val']
#
#        processed_df = process_dataframe(df, rename_map, fp_tp_cols)
#
#        assert 'NewColA' in processed_df.columns
#        assert 'NewColB' in processed_df.columns
#        assert 'OldColA' not in processed_df.columns
#        assert 'body_key' in processed_df.columns
#        assert processed_df['body_key'].tolist() == [1, 2]
#        assert 'fp_some_val' in processed_df.columns
#        assert 'tp_some_val' in processed_df.columns
#        assert 'some_val' not in processed_df.columns
#        assert processed_df['fp_some_val'].tolist() == [10, 0]
#        assert processed_df['tp_some_val'].tolist() == [0, 20]

class TestStandardizePandasSchema:
    def test_standardize_schema(self):
        data = {'int_col': [1, 2], 'float_col': [1.1, 2.2], 'str_col': ['a', 'b'], 'bool_col': [True, False]}
        df = pd.DataFrame(data)
        df['dt_col'] = pd.to_datetime(['2023-01-01', '2023-01-02'])
        
        standardized_df = standardize_pandas_schema(df)
        
        assert standardized_df['int_col'].dtype == np.float64
        assert standardized_df['float_col'].dtype == np.float64
        assert standardized_df['str_col'].dtype == object # Pandas uses 'object' for strings
        assert standardized_df['bool_col'].dtype == np.float64 # Booleans are numeric, become float
        assert standardized_df['dt_col'].dtype == object # Datetimes become strings


# --- Tests for Sklearn Custom Transformers ---

class TestSplatmapExtractor:
    def test_splatmap_extraction(self):
        data = {
            'FP Splatmap Data': [{'Front': 1, 'Rear': 2}, {'Front': 3}],
            'TP Splatmap Data': [{'Left': 4}, {'Left': 5, 'Right': 6}]
        }
        df = pd.DataFrame(data)
        custom_splatmap_cols = ["Front", "Left"] # Test with a subset
        extractor = SplatmapExtractor(splatmap_cols=custom_splatmap_cols)
        transformed_df = extractor.fit_transform(df)

        assert 'FP_Front' in transformed_df.columns
        assert 'TP_Left' in transformed_df.columns
        assert 'FP Splatmap Data' not in transformed_df.columns
        assert transformed_df['FP_Front'].tolist() == [1.0, 3.0]
        assert transformed_df['TP_Left'].tolist() == [4.0, 5.0]
        # Check that unspecified splatmap cols (e.g., 'Rear') are not created if not in custom_splatmap_cols
        assert 'FP_Rear' not in transformed_df.columns 
        assert 'TP_Right' not in transformed_df.columns

    def test_splatmap_missing_main_cols(self):
        df = pd.DataFrame({'id': [1,2]}) # Missing 'FP Splatmap Data' and 'TP Splatmap Data'
        extractor = SplatmapExtractor()
        transformed_df = extractor.fit_transform(df)
        # Should not error and should not add FP_ or TP_ columns
        assert 'FP_Front' not in transformed_df.columns
        assert transformed_df.equals(df) # DataFrame should be unchanged


class TestRenameColumnsTransformer:
    def test_rename_columns(self):
        data = {'OldName1': [1, 2], 'OldName2': ['a', 'b']}
        df = pd.DataFrame(data)
        rename_map = {'OldName1': 'NewName1', 'NonExistent': 'WontAppear'}
        # Use a partial map for testing, not the full default_map from the class
        transformer = RenameColumnsTransformer(rename_map=rename_map)
        transformed_df = transformer.fit_transform(df)
        
        assert 'NewName1' in transformed_df.columns
        assert 'OldName1' not in transformed_df.columns
        assert 'OldName2' in transformed_df.columns # Unchanged
        assert transformed_df['NewName1'].tolist() == [1, 2]

#class TestGeneralPurposeImputer:
#    def test_imputation(self):
#        data = {'num1': [1.0, np.nan, 3.0], 'cat1': ['a', 'b', np.nan], 'num2': [np.nan, np.nan, np.nan]}
#        df = pd.DataFrame(data)
#        # Standardize schema first as the imputer expects float64 for numeric
#        df['num1'] = df['num1'].astype(np.float64)
#        df['num2'] = df['num2'].astype(np.float64) # All NaN numeric column
#        df['cat1'] = df['cat1'].astype(object)
#
#        imputer = GeneralPurposeImputer()
#        imputer.fit(df) # Fit on data that includes an all-NaN numeric column
#        transformed_df = imputer.transform(df)
#
#        # num1: median is 2.0. NaN becomes 2.0
#        assert transformed_df['num1'].isnull().sum() == 0
#        assert transformed_df['num1'].tolist() == [1.0, 2.0, 3.0] 
#        # cat1: most frequent is 'a' or 'b'. NaN becomes one of them.
#        assert transformed_df['cat1'].isnull().sum() == 0
#        assert transformed_df['cat1'].mode()[0] in ['a', 'b'] # Check if NaN filled with mode
#        # num2: all NaNs. Median is NaN. SimpleImputer with median strategy on all NaNs will result in NaNs.
#        # The behavior of SimpleImputer on all-NaN columns needs to be confirmed or handled.
#        # If strategy is 'median' and all are NaN, it might raise error or fill with 0 if 'add_indicator=True' (not used here).
#        # Let's check if it remains NaN or gets filled with something (often 0 by default for some versions/setups if not careful)
#        # The current GeneralPurposeImputer doesn't change all-NaN columns if SimpleImputer returns NaNs for median.
#        # If SimpleImputer fills all-NaN numeric with 0 (or its fit fails), test needs adjustment.
#        # Assuming SimpleImputer leaves it as NaN or fills with 0 if it can't compute median.
#        # Let's test if it's filled (e.g. with 0 if that's SimpleImputer's behavior for all-NaN median)
#        # For a column of all NaNs, `SimpleImputer(strategy='median')` will store `np.nan` as the statistic.
#        # Then, `transform` will fill NaNs with this stored `np.nan`, so they remain NaN.
#        assert transformed_df['num2'].isnull().all() # Should remain all NaNs
#
## ... (Continue with other Sklearn Transformer tests) ...
#
#class TestImpactSpeedTransformer:
#    def test_impact_speed_transform(self):
#        data = {
#            'impact_speed_range': ['Stationary', 'ZeroToSix', 'OverSeventy', None, 'SevenToFourteen'],
#            'impact_speed_unit': ['MPH', 'KMH', 'MPH', 'KMH', None] # None unit should default to MPH
#        }
#        df = pd.DataFrame(data)
#        transformer = ImpactSpeedTransformer()
#        transformed_df = transformer.fit_transform(df)
#
#        assert 'impact_speed' in transformed_df.columns
#        assert 'impact_speed_range' not in transformed_df.columns
#        # Expected:
#        # Stationary MPH -> 0 -> bucket 0
#        # ZeroToSix KMH (1) -> 1/1.61 = 0.62 -> bucket 1
#        # OverSeventy MPH -> 71 -> bucket 71
#        # None range -> -1 -> bucket -1
#        # SevenToFourteen (7), unit None (MPH) -> 7 -> bucket 7
#        expected_speeds = [0, 1, 71, -1, 7]
#        assert transformed_df['impact_speed'].tolist() == expected_speeds

class TestAirbagCountTransformer:
    def test_airbag_count_transform(self):
        data = {
            'tp_deployed_airbags': ['None', '2', 'All', 'Invalid', None],
            'fp_deployed_airbags': ['1', '3', 'None', None, 'All']
        }
        df = pd.DataFrame(data)
        transformer = AirbagCountTransformer()
        transformed_df = transformer.fit_transform(df)
        
        assert transformed_df['tp_deployed_airbags'].tolist() == [0, 2, 5, -1, -1]
        assert transformed_df['fp_deployed_airbags'].tolist() == [1, 3, 0, -1, 5]

# Note: TestCarCharacteristicTransformer would be very verbose due to many columns.
# It would involve setting up various 'yes'/'no' and assessment category strings.
# For brevity, I'll skip its full implementation, but the pattern is similar:
# Create input DataFrame -> transform -> assert output columns are 0/1 based on input strings.

#class TestDateFeatureTransformer:
#    def test_date_features(self):
#        data = {
#            'notification_date': ['2023-01-10', '2023-01-01', '2023-02-15', None],
#            'incident_date': ['2023-01-01', '2023-01-10', '2023-01-01', '2023-01-01']
#        }
#        df = pd.DataFrame(data)
#        transformer = DateFeatureTransformer()
#        transformed_df = transformer.fit_transform(df)
#
#        assert transformed_df['time_to_notify'].tolist() == [9, 0, 30, 0] # 45 capped at 30; None dates lead to 0
#        assert transformed_df['notified_day_of_week'].tolist() == [2, 7, 3, np.nan] # 2=Tue, 7=Sun, 3=Wed, NaN from NaT
#        assert transformed_df['notified_month'].tolist() == [1, 1, 2, np.nan]


class TestVehicleAgeTransformer:
    def test_vehicle_age(self):
        data = {
            'incident_date': pd.to_datetime(['2023-01-10', '2023-01-01', '2000-01-01']),
            'year_of_manufacture': ['2020', '2023', '1950'], # Str type as per user's code before cast
            'notification_date': pd.to_datetime(['2023-01-10', '2023-01-01', '2000-01-01']) # For drop check
        }
        df = pd.DataFrame(data)
        transformer = VehicleAgeTransformer()
        transformed_df = transformer.fit_transform(df)

        assert transformed_df['vehicle_age'].tolist() == [3, 0, 0] # 50 capped to 0 (as per where <=30, else 0)
        assert 'notification_date' not in transformed_df.columns
        assert 'incident_date' not in transformed_df.columns


class TestRHDTransformer:
    def test_rhd_transform(self):
        data = {'tp_right_hand_drive': ['R', 'L', None, 'r']}
        df = pd.DataFrame(data)
        transformer = RHDTransformer()
        transformed_df = transformer.fit_transform(df)
        assert transformed_df['tp_right_hand_drive'].tolist() == [1, 0, 0, 0]

class TestBodyKeyEncoder:
    def test_body_key_encode(self):
        data = {'tp_body_key': ['5 Door Hatchback', '4 Door Saloon', 'Unknown', None]}
        df = pd.DataFrame(data)
        transformer = BodyKeyEncoder() # Uses default keys
        transformed_df = transformer.fit_transform(df)
        
        assert transformed_df['tp_body_key_1'].tolist() == [1,0,0,0] # 5 Door Hatchback
        assert transformed_df['tp_body_key_3'].tolist() == [0,1,0,0] # 4 Door Saloon
        assert 'tp_body_key' in transformed_df.columns # Original column is kept by this transformer

class TestPostcodeAreaExtractor:
    def test_postcode_extract(self):
        data = {'postcode_area': ['AB12 3CD', 'D4 EF5', 'G67H', None, '']}
        df = pd.DataFrame(data)
        transformer = PostcodeAreaExtractor()
        transformed_df = transformer.fit_transform(df)
        assert transformed_df['postcode_area'].tolist() == ['AB', 'D', 'G', 'ZZ', 'ZZ']

class TestDropUnusedColumns:
    def test_drop_columns(self):
        data = {'colA': [1], 'colB': [2], 'colC': [3]}
        df = pd.DataFrame(data)
        transformer = DropUnusedColumns(to_drop=['colB', 'colD_non_existent'])
        transformed_df = transformer.fit_transform(df)
        assert 'colA' in transformed_df.columns
        assert 'colC' in transformed_df.columns
        assert 'colB' not in transformed_df.columns

# TestDamageTransformer and TestDamageSeverityCalculator are more involved
# due to the long list of severity columns. Will add a simplified version.

#class TestDamageTransformer:
#    def test_damage_transform_simple(self):
#        # Using a small subset of severity columns for testability
#        sev_cols_subset = ['tp_front_severity', 'tp_rear_severity']
#        data = {
#            'tp_front_severity': ['minimal', None, 'heavy'],
#            'tp_rear_severity':  [None, 'medium', 'severe'],
#            'other_col': [1,2,3]
#        }
#        df = pd.DataFrame(data)
#        # Ensure severity columns are lower case as per UDF logic
#        df['tp_front_severity'] = df['tp_front_severity'].str.lower()
#        df['tp_rear_severity'] = df['tp_rear_severity'].str.lower()
#
#
#        transformer = DamageTransformer(severity_columns=sev_cols_subset)
#        transformed_df = transformer.fit_transform(df)
#
#        assert 'damage_recorded' in transformed_df.columns
#        assert 'damage_assessed' in transformed_df.columns
#        assert transformed_df['damage_recorded'].tolist() == [1, 1, 2] # Count of non-nulls in subset
#        assert transformed_df['damage_assessed'].tolist() == [1, 1, 1]
#        
#        # Expected severity: minimal->1, medium->2, heavy->3, severe->4. None->-1 (if assessed=1)
#        assert transformed_df['tp_front_severity'].tolist() == [1, -1, 3] # row2 tp_front is None, assessed=1 -> -1
#        assert transformed_df['tp_rear_severity'].tolist()  == [-1, 2, 4]  # row1 tp_rear is None, assessed=1 -> -1


class TestDamageSeverityCalculator:
    def test_damage_severity_calc_simple(self):
        sev_cols_subset = ['tp_front_severity', 'tp_rear_severity']
        data = { # Assume these are already numerically encoded by DamageTransformer
            'tp_front_severity': [1, 0, 3], # 0 could mean not damaged or unassessed (after damage_scale)
            'tp_rear_severity':  [-1, 2, 4], # -1 could mean unknown or unassessed
        }
        df = pd.DataFrame(data)
        transformer = DamageSeverityCalculator(sev_damage=sev_cols_subset)
        transformed_df = transformer.fit_transform(df)
        
        # total: (1-1)=0, (0+2)=2, (3+4)=7
        # count (gt 0): (1), (1), (2)
        # mean: 0/1=0, 2/1=2, 7/2=3.5
        # max: 1, 2, 4
        assert transformed_df['damage_sev_total'].tolist() == [0, 2, 7]
        assert transformed_df['damage_sev_count'].tolist() == [1, 1, 2]
        assert transformed_df['damage_sev_mean'].tolist() == [0.0, 2.0, 3.5]
        assert transformed_df['damage_sev_max'].tolist() == [1, 2, 4]

# DaysInRepairPredictor and CaptureBenefitModel are highly dependent on MLflow models
# and artifacts. Testing them thoroughly requires significant mocking of mlflow.pyfunc.load_model,
# the loaded model's predict method, and context.artifacts.

@patch('mlflow.pyfunc.load_model')
class TestDaysInRepairPredictor:
    def test_days_in_repair_predictor(self, mock_load_model, spark):
        # Mock the loaded MLflow pyfunc model
        mock_pyfunc_model = MagicMock()
        mock_pyfunc_model.predict.return_value = pd.Series([5, 10]) # Example predictions
        
        # Mock the model's metadata and input schema
        mock_metadata = MagicMock()
        # Define a simple schema: one numeric 'feature1', one string 'feature2'
        input_schema_dict = [
            {'name': 'feature1', 'type': 'double'}, 
            {'name': 'feature2', 'type': 'string'}
        ]
        mock_input_schema = MagicMock()
        mock_input_schema.input_names.return_value = ['feature1', 'feature2']
        mock_input_schema.to_dict.return_value = input_schema_dict # Corrected: to_dict() returns the list of dicts
        mock_metadata.get_input_schema.return_value = mock_input_schema
        mock_pyfunc_model.metadata = mock_metadata
        
        mock_load_model.return_value = mock_pyfunc_model

        transformer = DaysInRepairPredictor(model_uri="runs:/dummy_run_id/model")
        
        # Test data
        input_data = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'feature2': ['a', 'b'],
            'other_col': [100, 200] # Extra column not in schema
        })
        
        # Test with missing column that is in schema
        input_data_missing_col = pd.DataFrame({'feature1': [3.0, 4.0], 'other_col': [300,400]})


        transformed_df = transformer.transform(input_data.copy()) # Pass copy
        assert "days_in_repair" in transformed_df.columns
        pd.testing.assert_series_equal(transformed_df["days_in_repair"], pd.Series([5, 10], name="days_in_repair"))
        mock_pyfunc_model.predict.assert_called_once()
        # Check that X_for_model passed to predict has correct columns and order
        call_args_df = mock_pyfunc_model.predict.call_args[0][0]
        assert list(call_args_df.columns) == ['feature1', 'feature2']


        # Test with missing column
        transformed_df_missing = transformer.transform(input_data_missing_col.copy())
        assert "days_in_repair" in transformed_df_missing.columns
        # Predict would be called again, reset mock or check call_count
        # Check that the missing 'feature2' was defaulted (e.g., to "" for string)
        call_args_df_missing = mock_pyfunc_model.predict.call_args[0][0]
        assert call_args_df_missing['feature2'].tolist() == ["", ""] # Default for missing string


#@patch('mlflow.sklearn.load_model') # For preprocessor and regressor
#@patch('pandas.read_csv')
#class TestCaptureBenefitModel:
#    def test_capture_benefit_model_predict(self, mock_read_csv, mock_load_sklearn_model, mocker):
#        # Mock context
#        mock_context = MagicMock()
#        mock_context.artifacts = {
#            "preprocessor_model_path": "path/to/preprocessor",
#            "regressor_model_path": "path/to/regressor",
#            "quantile_benefit_table_csv_path": "path/to/quantile_table.csv"
#        }
#
#        # Mock preprocessor
#        mock_preprocessor = MagicMock(spec=BaseEstimator) # Or a mock Pipeline
#        # It needs a transform method that returns a DataFrame or NumPy array
#        # Let's assume it returns a NumPy array of features
#        processed_features_cap = np.array([[0.1, 0.2], [0.3, 0.4]])
#        processed_features_nocap = np.array([[0.15, 0.25], [0.35, 0.45]])
#        
#        # Side effect for preprocessor.transform
#        def preprocessor_transform_side_effect(df_input):
#            if df_input['CaptureSuccess_AdrienneVersion'].iloc[0] == 1.0:
#                return processed_features_cap
#            else:
#                return processed_features_nocap
#        mock_preprocessor.transform.side_effect = preprocessor_transform_side_effect
#        
#        # Mock regressor
#        mock_regressor = MagicMock(spec=BaseEstimator) # Or a specific regressor type
#        # It needs a predict method
#        # tppd_pred_capture = [100, 120]
#        # tppd_pred_no_capture = [150, 180]
#        def regressor_predict_side_effect(features_input):
#            if np.array_equal(features_input, processed_features_cap):
#                return np.array([100, 120])
#            elif np.array_equal(features_input, processed_features_nocap):
#                return np.array([150, 180])
#            return np.array([0,0]) # Default
#        mock_regressor.predict.side_effect = regressor_predict_side_effect
#        
#        mock_load_sklearn_model.side_effect = [mock_preprocessor, mock_regressor] # Load preprocessor then regressor
#
#        # Mock quantile benefit table
#        quantile_data = {
#            'capture_benefit_lower_bound': [0, 50, 100],
#            'capture_benefit_upper_bound': [50, 100, np.inf],
#            'quant_benefit_multiplier': [1.0, 1.2, 1.5]
#        }
#        mock_quantile_df = pd.DataFrame(quantile_data)
#        mock_read_csv.return_value = mock_quantile_df
#
#        # Instantiate the model
#        cb_model = CaptureBenefitModel()
#        cb_model.load_context(mock_context) # Load mocks
#
#        # Prepare input data for predict
#        model_input_data = pd.DataFrame({
#            'some_feature': [10, 20], 
#            'another_feature': ['catA', 'catB'],
#            # CaptureSuccess_AdrienneVersion will be added/overwritten by the model
#        })
#        
#        # Call predict
#        predictions = cb_model.predict(mock_context, model_input_data)
#
#        # Assertions
#        # raw_capture_benefit = [150-100, 180-120] = [50, 60]
#        # Multipliers: for 50 -> 1.2 (0 <= 50 < 50 is false, 50 <= 50 < 100 is true)
#        #              for 60 -> 1.2 (50 <= 60 < 100 is true)
#        # Adjusted benefit: [50*1.2, 60*1.2] = [60, 72]
#        expected_predictions = pd.Series([60.0, 72.0], name="Capture_Benefit")
#        pd.testing.assert_series_equal(predictions, expected_predictions)
#        
#        assert mock_load_sklearn_model.call_count == 2
#        mock_read_csv.assert_called_once_with("path/to/quantile_table.csv")
#        assert mock_preprocessor.transform.call_count == 2
#        assert mock_regressor.predict.call_count == 2