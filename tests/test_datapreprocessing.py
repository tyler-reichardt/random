import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    DoubleType, TimestampType, DateType
)
from pyspark.sql.functions import col, lit, when
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd

# Mock the notebooks module structure
import sys
sys.path.append('..')

# Mock the configs import
with patch('notebooks.DataPreprocessing.extract_column_transformation_lists'):
    with patch('notebooks.DataPreprocessing.spark') as mock_spark:
        from notebooks.DataPreprocessing import *

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-datapreprocessing")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()

@pytest.fixture
def mock_spark_with_table(spark):
    """Mock spark.table to return test data."""
    with patch('notebooks.DataPreprocessing.spark') as mock_spark:
        mock_spark.sql = spark.sql
        mock_spark.createDataFrame = spark.createDataFrame
        yield mock_spark

class TestPolicyDataReading:
    """Test policy-related data reading functions."""
    
    def test_read_policy_data(self, spark, mock_spark_with_table):
        """Test read_policy_data function."""
        # Create mock policy data
        policy_data = [
            ("POL001", "P001", 1, datetime(2023, 1, 1), datetime(2024, 1, 1), "ACTIVE", "AUTO"),
            ("POL002", "P002", 2, datetime(2023, 6, 1), datetime(2024, 6, 1), "ACTIVE", "AUTO"),
        ]
        policy_df = spark.createDataFrame(
            policy_data,
            ["policy_number", "policy_transaction_id", "policyholder_ncd_years", 
             "inception_date", "renewal_date", "status", "product_type"]
        )
        
        mock_spark_with_table.table = MagicMock(return_value=policy_df)
        
        # Test the function
        result = read_policy_data()
        
        assert result.count() == 2
        assert "policy_number" in result.columns
        assert "policyholder_ncd_years" in result.columns
    
    def test_read_vehicle_data(self, spark, mock_spark_with_table):
        """Test read_vehicle_data function."""
        # Create mock vehicle data
        vehicle_data = [
            ("VEH001", "POL001", "SEDAN", 2020, "London", "SW1A"),
            ("VEH002", "POL002", "SUV", 2021, "Manchester", "M1"),
        ]
        vehicle_df = spark.createDataFrame(
            vehicle_data,
            ["vehicle_id", "policy_number", "body_type", "manufacture_year", 
             "location_name", "postcode"]
        )
        
        mock_spark_with_table.table = MagicMock(return_value=vehicle_df)
        
        # Test the function
        result = read_vehicle_data()
        
        assert result.count() == 2
        assert "vehicle_id" in result.columns
        assert "location_name" in result.columns
    
    def test_read_driver_data(self, spark, mock_spark_with_table):
        """Test read_driver_data function."""
        # Create mock driver data
        driver_data = [
            ("DRV001", "POL001", date(1980, 1, 1), 20, 15, "EMPLOYED"),
            ("DRV002", "POL001", date(1990, 5, 15), 10, 8, "SELF_EMPLOYED"),
            ("DRV003", "POL002", date(1975, 3, 20), 25, 20, "EMPLOYED"),
        ]
        driver_df = spark.createDataFrame(
            driver_data,
            ["driver_id", "policy_number", "date_of_birth", "licence_years", 
             "years_in_uk", "employment_type"]
        )
        
        mock_spark_with_table.table = MagicMock(return_value=driver_df)
        
        # Test the function
        result = read_driver_data()
        
        # Should aggregate by policy
        assert result.count() == 2
        assert "min_licence_years" in result.columns
        assert "max_licence_years" in result.columns
    
    def test_read_customer_data(self, spark, mock_spark_with_table):
        """Test read_customer_data function."""
        # Create mock customer data
        customer_data = [
            ("CUST001", "John", "Doe", "john@email.com", "1234567890"),
            ("CUST002", "Jane", "Smith", "jane@email.com", "0987654321"),
        ]
        customer_df = spark.createDataFrame(
            customer_data,
            ["customer_id", "first_name", "last_name", "email", "phone"]
        )
        
        mock_spark_with_table.table = MagicMock(return_value=customer_df)
        
        # Test the function
        result = read_customer_data()
        
        assert result.count() == 2
        assert "customer_id" in result.columns

class TestClaimDataReading:
    """Test claim-related data reading functions."""
    
    def test_read_claim_data(self, spark, mock_spark_with_table):
        """Test read_claim_data function."""
        # Create mock claim data
        claim_data = [
            ("CLM001", "POL001", date(2023, 6, 15), "COLLISION", "OPEN", 1),
            ("CLM002", "POL002", date(2023, 7, 20), "THEFT", "CLOSED", 0),
        ]
        claim_df = spark.createDataFrame(
            claim_data,
            ["claim_number", "policy_number", "incident_date", "incident_type", 
             "status", "is_svi"]
        )
        
        mock_spark_with_table.table = MagicMock(return_value=claim_df)
        
        # Test with date filter
        result = read_claim_data(start_date="2023-06-01", incident_date="2023-06-15")
        
        assert result is not None
        assert "claim_number" in result.columns
    
    def test_get_latest_claim_version(self, spark):
        """Test get_latest_claim_version function."""
        # Create mock claim version data
        claim_data = [
            ("CLM001", 1, datetime(2023, 6, 15), "Initial"),
            ("CLM001", 2, datetime(2023, 6, 20), "Updated"),
            ("CLM002", 1, datetime(2023, 7, 20), "Initial"),
        ]
        claim_df = spark.createDataFrame(
            claim_data,
            ["claim_number", "version", "created_date", "status"]
        )
        
        result = get_latest_claim_version(claim_df)
        
        assert result.count() == 2
        # Check that we got the latest versions
        clm001 = result.filter(col("claim_number") == "CLM001").collect()[0]
        assert clm001["version"] == 2
        assert clm001["status"] == "Updated"

class TestTargetVariableCreation:
    """Test target variable creation functions."""
    
    def test_get_referral_vertices(self, spark, mock_spark_with_table):
        """Test get_referral_vertices function."""
        # Create mock referral data
        referral_data = [
            ("CLM001", "FRAUD", 1, date(2023, 6, 20)),
            ("CLM002", "NOT_FRAUD", 0, date(2023, 7, 25)),
            ("CLM003", "PENDING", -1, date(2023, 8, 1)),
        ]
        referral_df = spark.createDataFrame(
            referral_data,
            ["claim_number", "outcome", "risk_score", "decision_date"]
        )
        
        mock_spark_with_table.table = MagicMock(return_value=referral_df)
        
        # Test the function
        result = get_referral_vertices()
        
        assert result.count() == 3
        assert "risk_score" in result.columns
    
    def test_create_target_variable(self, spark, mock_spark_with_table):
        """Test create_target_variable function."""
        # Create mock SVI data
        svi_data = [
            ("CLM001", 1, 0),  # FA risk only
            ("CLM002", 0, 1),  # TBG risk only
            ("CLM003", 1, 1),  # Both risks
            ("CLM004", 0, 0),  # No risk
        ]
        svi_df = spark.createDataFrame(
            svi_data,
            ["claim_number", "fa_risk", "tbg_risk"]
        )
        
        mock_spark_with_table.table = MagicMock(return_value=svi_df)
        
        # Test the function
        result = create_target_variable(start_date="2023-01-01")
        
        assert result.count() == 4
        assert "svi_risk" in result.columns
        
        # Check risk calculations
        risks = {row["claim_number"]: row["svi_risk"] for row in result.collect()}
        assert risks["CLM001"] == 1  # max(1, 0) = 1
        assert risks["CLM002"] == 1  # max(0, 1) = 1
        assert risks["CLM003"] == 1  # max(1, 1) = 1
        assert risks["CLM004"] == 0  # max(0, 0) = 0

class TestPolicyFeaturePreparation:
    """Test policy feature preparation functions."""
    
    def test_prepare_policy_features(self, spark, mock_spark_with_table):
        """Test prepare_policy_features function."""
        # Mock the individual data reading functions
        with patch('notebooks.DataPreprocessing.read_policy_data') as mock_policy:
            with patch('notebooks.DataPreprocessing.read_vehicle_data') as mock_vehicle:
                with patch('notebooks.DataPreprocessing.read_driver_data') as mock_driver:
                    with patch('notebooks.DataPreprocessing.read_customer_data') as mock_customer:
                        # Create mock dataframes
                        policy_df = spark.createDataFrame(
                            [("POL001", "P001", 5)],
                            ["policy_number", "policy_transaction_id", "ncd_years"]
                        )
                        vehicle_df = spark.createDataFrame(
                            [("POL001", "SEDAN", 2020)],
                            ["policy_number", "body_type", "year"]
                        )
                        driver_df = spark.createDataFrame(
                            [("POL001", 10, 20)],
                            ["policy_number", "min_licence_years", "max_licence_years"]
                        )
                        customer_df = spark.createDataFrame(
                            [("CUST001", "John Doe")],
                            ["customer_id", "name"]
                        )
                        
                        mock_policy.return_value = policy_df
                        mock_vehicle.return_value = vehicle_df
                        mock_driver.return_value = driver_df
                        mock_customer.return_value = customer_df
                        
                        # Test the function
                        result = prepare_policy_features()
                        
                        assert result is not None
                        assert "policy_number" in result.columns

class TestTrainingDataPreparation:
    """Test training data preparation functions."""
    
    def test_prepare_training_data(self, spark, mock_spark_with_table):
        """Test prepare_training_data function."""
        # Create mock combined data
        train_data = [
            ("CLM001", "POL001", 1, date(2023, 6, 15), "train"),
            ("CLM002", "POL002", 0, date(2023, 7, 20), "train"),
            ("CLM003", "POL003", 1, date(2023, 8, 10), "test"),
        ]
        train_df = spark.createDataFrame(
            train_data,
            ["claim_number", "policy_number", "svi_risk", "incident_date", "dataset"]
        )
        
        # Mock the target variable creation
        with patch('notebooks.DataPreprocessing.create_target_variable') as mock_target:
            mock_target.return_value = train_df
            
            # Test the function
            result = prepare_training_data(start_date="2023-01-01")
            
            assert result is not None
            assert "dataset" in result.columns
            
            # Check train/test split
            train_count = result.filter(col("dataset") == "train").count()
            test_count = result.filter(col("dataset") == "test").count()
            assert train_count == 2
            assert test_count == 1

class TestPreprocessingPipelines:
    """Test main preprocessing pipeline functions."""
    
    def test_run_daily_preprocessing(self, spark, mock_spark_with_table):
        """Test run_daily_preprocessing function."""
        # Mock the required functions
        with patch('notebooks.DataPreprocessing.read_claim_data') as mock_claims:
            with patch('notebooks.DataPreprocessing.prepare_policy_features') as mock_policies:
                # Create mock data
                claims_df = spark.createDataFrame(
                    [("CLM001", "POL001", date(2023, 10, 15))],
                    ["claim_number", "policy_number", "incident_date"]
                )
                policies_df = spark.createDataFrame(
                    [("POL001", "P001", 5)],
                    ["policy_number", "policy_transaction_id", "ncd_years"]
                )
                
                mock_claims.return_value = claims_df
                mock_policies.return_value = policies_df
                
                # Mock spark.sql for writing
                mock_spark_with_table.sql = MagicMock()
                
                # Test the function
                result = run_daily_preprocessing(
                    execution_date="2023-10-15",
                    lookback_days=1
                )
                
                assert result is not None
                assert "daily_policies" in result
                assert "daily_claims" in result
    
    def test_run_training_preprocessing(self, spark, mock_spark_with_table):
        """Test run_training_preprocessing function."""
        with patch('notebooks.DataPreprocessing.prepare_training_data') as mock_training:
            # Create mock training data
            training_df = spark.createDataFrame(
                [
                    ("CLM001", "POL001", 1, "train"),
                    ("CLM002", "POL002", 0, "test"),
                ],
                ["claim_number", "policy_number", "svi_risk", "dataset"]
            )
            
            mock_training.return_value = training_df
            
            # Test the function
            result = run_training_preprocessing(start_date="2023-01-01")
            
            assert result is not None
            assert result.count() == 2

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_validate_data_quality(self, spark):
        """Test validate_data_quality function."""
        # Create test data with some nulls and duplicates
        test_data = [
            ("ID001", "Value1", None),
            ("ID002", "Value2", 100),
            ("ID002", "Value2", 100),  # Duplicate
            ("ID003", None, 200),
        ]
        test_df = spark.createDataFrame(
            test_data,
            ["id", "col1", "col2"]
        )
        
        # Test the function
        metrics = validate_data_quality(test_df, "test_table")
        
        assert metrics["row_count"] == 4
        assert metrics["duplicate_count"] == 1
        assert metrics["null_counts"]["col1"] == 1
        assert metrics["null_counts"]["col2"] == 1
        assert metrics["null_percentages"]["col1"] == 25.0
        assert metrics["null_percentages"]["col2"] == 25.0

if __name__ == "__main__":
    pytest.main([__file__])