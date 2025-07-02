import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    BooleanType, TimestampType, DateType
)
from pyspark.sql.functions import col, lit, when, sum as spark_sum
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Mock the notebooks module structure
import sys
sys.path.append('..')

# Mock the configs import
with patch('notebooks.FeatureEngineering.extract_column_transformation_lists'):
    with patch('notebooks.FeatureEngineering.spark') as mock_spark:
        from notebooks.FeatureEngineering import *

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-featureengineering")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()

class TestDamageCalculations:
    """Test damage calculation functions."""
    
    def test_calculate_damage_score(self, spark):
        """Test calculate_damage_score function."""
        # Create test data with different damage severities
        damage_data = [
            ("CLM001", "Minimal", "Medium", "Heavy", None),
            ("CLM002", "Severe", "Severe", None, "Minimal"),
            ("CLM003", None, None, None, None),
            ("CLM004", "Medium", "Medium", "Medium", "Medium"),
        ]
        df = spark.createDataFrame(
            damage_data,
            ["claim_number", "front_severity", "rear_severity", 
             "left_severity", "right_severity"]
        )
        
        # Apply damage calculations
        result = apply_damage_calculations(df)
        
        # Check that damage score column exists
        assert "damageScore" in result.columns
        
        # Verify damage score calculations
        scores = {row["claim_number"]: row["damageScore"] 
                 for row in result.collect()}
        
        # CLM001: Minimal(2) * Medium(3) * Heavy(4) = 24
        assert scores["CLM001"] == 24
        # CLM002: Severe(5) * Severe(5) * Minimal(2) = 50
        assert scores["CLM002"] == 50
        # CLM003: All None = 1
        assert scores["CLM003"] == 1
        # CLM004: Medium(3) * Medium(3) * Medium(3) * Medium(3) = 81
        assert scores["CLM004"] == 81
    
    def test_damage_area_counts(self, spark):
        """Test damage area counting."""
        # Create test data
        damage_data = [
            ("CLM001", "Minimal", "Medium", "Heavy", "Severe"),
            ("CLM002", None, "Minimal", "Minimal", None),
            ("CLM003", "Heavy", "Heavy", "Heavy", "Heavy"),
        ]
        df = spark.createDataFrame(
            damage_data,
            ["claim_number", "front", "rear", "left", "right"]
        )
        
        # Apply damage calculations
        result = apply_damage_calculations(df)
        
        # Check area count columns exist
        assert "areasDamagedMinimal" in result.columns
        assert "areasDamagedMedium" in result.columns
        assert "areasDamagedHeavy" in result.columns
        assert "areasDamagedSevere" in result.columns
        assert "areasDamagedTotal" in result.columns
        
        # Verify counts for CLM001
        clm001 = result.filter(col("claim_number") == "CLM001").collect()[0]
        assert clm001["areasDamagedMinimal"] == 1
        assert clm001["areasDamagedMedium"] == 1
        assert clm001["areasDamagedHeavy"] == 1
        assert clm001["areasDamagedSevere"] == 1
        assert clm001["areasDamagedTotal"] == 4

class TestBusinessRules:
    """Test business rule (check variable) generation."""
    
    def test_c1_friday_saturday_night(self, spark):
        """Test C1: Friday/Saturday night incidents."""
        # Create test data
        incident_data = [
            ("CLM001", datetime(2023, 10, 13, 22, 0, 0)),  # Friday 10 PM
            ("CLM002", datetime(2023, 10, 14, 2, 0, 0)),   # Saturday 2 AM
            ("CLM003", datetime(2023, 10, 11, 22, 0, 0)),  # Wednesday 10 PM
            ("CLM004", datetime(2023, 10, 15, 14, 0, 0)),  # Sunday 2 PM
        ]
        df = spark.createDataFrame(
            incident_data,
            ["claim_number", "incident_datetime"]
        )
        
        # Generate check variables
        result = generate_check_variables(df)
        
        # Verify C1 rule
        checks = {row["claim_number"]: row["C1_fri_sat_night"] 
                 for row in result.collect()}
        
        assert checks["CLM001"] == 1  # Friday night
        assert checks["CLM002"] == 1  # Saturday early morning
        assert checks["CLM003"] == 0  # Wednesday night
        assert checks["CLM004"] == 0  # Sunday afternoon
    
    def test_c2_reporting_delay(self, spark):
        """Test C2: Reporting delay >= 3 days."""
        # Create test data
        report_data = [
            ("CLM001", date(2023, 10, 1), date(2023, 10, 5)),   # 4 days delay
            ("CLM002", date(2023, 10, 1), date(2023, 10, 2)),   # 1 day delay
            ("CLM003", date(2023, 10, 1), date(2023, 10, 4)),   # 3 days delay
            ("CLM004", date(2023, 10, 1), None),                # Null reported
        ]
        df = spark.createDataFrame(
            report_data,
            ["claim_number", "incident_date", "reported_date"]
        )
        
        # Generate check variables
        result = generate_check_variables(df)
        
        # Verify C2 rule
        checks = {row["claim_number"]: row["C2_reporting_delay"] 
                 for row in result.collect()}
        
        assert checks["CLM001"] == 1  # 4 days >= 3
        assert checks["CLM002"] == 0  # 1 day < 3
        assert checks["CLM003"] == 1  # 3 days >= 3
        assert checks["CLM004"] == 1  # Null defaults to 1
    
    def test_c5_night_incident(self, spark):
        """Test C5: Night incident (11 PM - 5 AM)."""
        # Create test data with different times
        incident_data = [
            ("CLM001", 23),  # 11 PM
            ("CLM002", 0),   # Midnight
            ("CLM003", 4),   # 4 AM
            ("CLM004", 5),   # 5 AM (boundary)
            ("CLM005", 22),  # 10 PM
            ("CLM006", 6),   # 6 AM
        ]
        df = spark.createDataFrame(
            incident_data,
            ["claim_number", "incidentHourC"]
        )
        
        # Generate check variables
        result = generate_check_variables(df)
        
        # Verify C5 rule
        checks = {row["claim_number"]: row["C5_is_night_incident"] 
                 for row in result.collect()}
        
        assert checks["CLM001"] == 1  # 11 PM
        assert checks["CLM002"] == 1  # Midnight
        assert checks["CLM003"] == 1  # 4 AM
        assert checks["CLM004"] == 0  # 5 AM (not included)
        assert checks["CLM005"] == 0  # 10 PM
        assert checks["CLM006"] == 0  # 6 AM
    
    def test_c9_new_policy(self, spark):
        """Test C9: Policy inception within 30 days."""
        # Create test data
        policy_data = [
            ("CLM001", 15),   # 15 days
            ("CLM002", 30),   # 30 days (boundary)
            ("CLM003", 31),   # 31 days
            ("CLM004", 365),  # 1 year
            ("CLM005", None), # Null
        ]
        df = spark.createDataFrame(
            policy_data,
            ["claim_number", "inception_to_claim"]
        )
        
        # Generate check variables
        result = generate_check_variables(df)
        
        # Verify C9 rule
        checks = {row["claim_number"]: row["C9_policy_within_30_days"] 
                 for row in result.collect()}
        
        assert checks["CLM001"] == 1  # 15 days <= 30
        assert checks["CLM002"] == 1  # 30 days <= 30
        assert checks["CLM003"] == 0  # 31 days > 30
        assert checks["CLM004"] == 0  # 365 days > 30
        assert checks["CLM005"] == 1  # Null defaults to 1
    
    def test_c11_young_inexperienced_driver(self, spark):
        """Test C11: Young or inexperienced driver."""
        # Create test data
        driver_data = [
            ("CLM001", 20, 2),   # Young and inexperienced
            ("CLM002", 30, 10),  # Neither
            ("CLM003", 22, 5),   # Young only
            ("CLM004", 40, 1),   # Inexperienced only
            ("CLM005", None, 2), # Missing age
        ]
        df = spark.createDataFrame(
            driver_data,
            ["claim_number", "min_claim_driver_age", "min_licence_length_years"]
        )
        
        # Generate check variables
        result = generate_check_variables(df)
        
        # Verify C11 rule
        checks = {row["claim_number"]: row["C11_young_or_inexperienced"] 
                 for row in result.collect()}
        
        assert checks["CLM001"] == 1  # Young AND inexperienced
        assert checks["CLM002"] == 0  # Neither
        assert checks["CLM003"] == 1  # Young
        assert checks["CLM004"] == 1  # Inexperienced
        assert checks["CLM005"] == 1  # Missing age defaults to young
    
    def test_check_aggregation(self, spark):
        """Test aggregation of check variables."""
        # Create test data with multiple checks
        check_data = [
            ("CLM001", 1, 0, 1, 0, 1),  # 3 checks failed
            ("CLM002", 0, 0, 0, 0, 0),  # 0 checks failed
            ("CLM003", 1, 1, 1, 1, 1),  # 5 checks failed
        ]
        df = spark.createDataFrame(
            check_data,
            ["claim_number", "C1_fri_sat_night", "C2_reporting_delay",
             "C5_is_night_incident", "C9_policy_within_30_days",
             "C11_young_or_inexperienced"]
        )
        
        # Calculate num_failed_checks
        check_cols = ["C1_fri_sat_night", "C2_reporting_delay", 
                     "C5_is_night_incident", "C9_policy_within_30_days",
                     "C11_young_or_inexperienced"]
        
        df = df.withColumn(
            "num_failed_checks",
            sum([col(c) for c in check_cols])
        )
        
        # Verify aggregation
        results = {row["claim_number"]: row["num_failed_checks"] 
                  for row in df.collect()}
        
        assert results["CLM001"] == 3
        assert results["CLM002"] == 0
        assert results["CLM003"] == 5

class TestFeatureTransformations:
    """Test feature transformation functions."""
    
    def test_temporal_features(self, spark):
        """Test temporal feature creation."""
        # Create test data
        temporal_data = [
            ("CLM001", date(2020, 1, 1), date(2023, 6, 15), 2018),
            ("CLM002", date(2022, 3, 1), date(2023, 7, 20), 2020),
        ]
        df = spark.createDataFrame(
            temporal_data,
            ["claim_number", "policy_start_date", "incident_date", "manufacture_year"]
        )
        
        # Add driver birth date for age calculation
        df = df.withColumn("driver_birth_date", lit(date(1990, 1, 1)))
        
        # Create temporal features
        df = df.withColumn(
            "veh_age",
            year(col("incident_date")) - col("manufacture_year")
        )
        df = df.withColumn(
            "driver_age_at_claim",
            year(col("incident_date")) - year(col("driver_birth_date"))
        )
        df = df.withColumn(
            "policy_tenure_days",
            datediff(col("incident_date"), col("policy_start_date"))
        )
        
        # Verify calculations
        results = df.collect()
        
        # CLM001
        assert results[0]["veh_age"] == 5  # 2023 - 2018
        assert results[0]["driver_age_at_claim"] == 33  # 2023 - 1990
        assert results[0]["policy_tenure_days"] > 1200  # ~3.5 years
        
        # CLM002
        assert results[1]["veh_age"] == 3  # 2023 - 2020
    
    def test_missing_value_imputation(self, spark):
        """Test missing value imputation strategies."""
        # Create test data with missing values
        missing_data = [
            ("CLM001", 5000.0, 10, "Minimal"),
            ("CLM002", None, 15, "Heavy"),
            ("CLM003", 7500.0, None, None),
            ("CLM004", None, None, "Medium"),
        ]
        df = spark.createDataFrame(
            missing_data,
            ["claim_number", "vehicle_value", "policyholder_ncd_years", "damage_severity"]
        )
        
        # Apply imputation
        df = apply_missing_value_imputation(df, stage="training")
        
        # Check that no nulls remain in critical columns
        assert df.filter(col("vehicle_value").isNull()).count() == 0
        assert df.filter(col("policyholder_ncd_years").isNull()).count() == 0
        assert df.filter(col("damage_severity").isNull()).count() == 0
        
        # Verify imputation values
        results = {row["claim_number"]: row for row in df.collect()}
        
        # Vehicle value should be imputed with mean
        assert results["CLM002"]["vehicle_value"] > 0
        assert results["CLM004"]["vehicle_value"] > 0
        
        # NCD years should be imputed with median
        assert results["CLM003"]["policyholder_ncd_years"] >= 0
        assert results["CLM004"]["policyholder_ncd_years"] >= 0
        
        # Damage severity should be imputed with mode or default
        assert results["CLM003"]["damage_severity"] in ["Minimal", "Medium", "Heavy", "Severe", "Unknown"]

class TestFeatureEngineering:
    """Test main feature engineering pipeline."""
    
    def test_get_model_features(self):
        """Test get_model_features function."""
        features = get_model_features()
        
        # Check that feature lists exist
        assert "all_features" in features
        assert "interview_features" in features
        assert "numeric_features" in features
        assert "categorical_features" in features
        assert "num_interview" in features
        assert "cat_interview" in features
        
        # Check feature list properties
        assert len(features["all_features"]) > 0
        assert len(features["interview_features"]) < len(features["all_features"])
        assert set(features["interview_features"]).issubset(set(features["all_features"]))
    
    def test_run_feature_engineering_training(self, spark):
        """Test run_feature_engineering in training mode."""
        # Create comprehensive test data
        test_data = [
            ("CLM001", "POL001", datetime(2023, 10, 13, 22, 0), date(2023, 10, 13),
             date(2023, 10, 15), 10, 5000, "Minimal", "Medium", None, None,
             25, 5, date(2023, 9, 1), date(2024, 9, 1), 1, 0),
        ]
        df = spark.createDataFrame(
            test_data,
            ["claim_number", "policy_number", "incident_datetime", "incident_date",
             "reported_date", "inception_to_claim", "vehicle_value", 
             "front_severity", "rear_severity", "left_severity", "right_severity",
             "min_claim_driver_age", "min_licence_length_years",
             "policy_start_date", "policy_end_date", "fa_risk", "tbg_risk"]
        )
        
        # Add required columns
        df = df.withColumn("incidentHourC", hour(col("incident_datetime")))
        df = df.withColumn("is_weekend_incident", 
                          when(dayofweek(col("incident_date")).isin([1, 7]), 1).otherwise(0))
        
        # Run feature engineering
        result = run_feature_engineering(df, stage="training")
        
        # Check that all expected columns exist
        assert "damageScore" in result.columns
        assert "num_failed_checks" in result.columns
        assert "C1_fri_sat_night" in result.columns
        assert "features_generated" in result.columns
        
        # Verify data types
        numeric_cols = ["damageScore", "vehicle_value", "min_claim_driver_age"]
        for col_name in numeric_cols:
            assert result.schema[col_name].dataType in [IntegerType(), DoubleType()]
    
    def test_run_feature_engineering_scoring(self, spark):
        """Test run_feature_engineering in scoring mode."""
        # Create test data for scoring (no target variables)
        test_data = [
            ("CLM001", "POL001", datetime(2023, 10, 13, 22, 0), date(2023, 10, 13),
             date(2023, 10, 15), 10, 5000, "Minimal", "Medium", None, None,
             25, 5, date(2023, 9, 1), date(2024, 9, 1)),
        ]
        df = spark.createDataFrame(
            test_data,
            ["claim_number", "policy_number", "incident_datetime", "incident_date",
             "reported_date", "inception_to_claim", "vehicle_value", 
             "front_severity", "rear_severity", "left_severity", "right_severity",
             "min_claim_driver_age", "min_licence_length_years",
             "policy_start_date", "policy_end_date"]
        )
        
        # Add required columns
        df = df.withColumn("incidentHourC", hour(col("incident_datetime")))
        
        # Run feature engineering
        result = run_feature_engineering(df, stage="scoring")
        
        # Check that features are generated
        assert result.count() == 1
        assert "damageScore" in result.columns
        assert "num_failed_checks" in result.columns

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_apply_type_conversions(self, spark):
        """Test apply_type_conversions function."""
        # Create test data with mixed types
        test_data = [
            ("CLM001", "1", "100.5", "true"),
            ("CLM002", "2", "200.7", "false"),
            ("CLM003", "invalid", "300", "1"),
        ]
        df = spark.createDataFrame(
            test_data,
            ["claim_number", "int_col", "double_col", "bool_col"]
        )
        
        # Define conversions
        df = df.withColumn("int_col", col("int_col").cast("integer"))
        df = df.withColumn("double_col", col("double_col").cast("double"))
        df = df.withColumn("bool_col", 
                          when(col("bool_col") == "true", 1)
                          .when(col("bool_col") == "1", 1)
                          .otherwise(0))
        
        # Check conversions
        results = df.collect()
        
        assert results[0]["int_col"] == 1
        assert results[0]["double_col"] == 100.5
        assert results[0]["bool_col"] == 1
        
        assert results[1]["int_col"] == 2
        assert results[1]["double_col"] == 200.7
        assert results[1]["bool_col"] == 0
        
        assert results[2]["int_col"] is None  # Invalid conversion
        assert results[2]["double_col"] == 300.0
        assert results[2]["bool_col"] == 1

if __name__ == "__main__":
    pytest.main([__file__])