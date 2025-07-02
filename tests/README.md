# Unit Tests for TPC ML Pipeline

This directory contains comprehensive unit tests for the Third Party Capture (TPC) ML pipeline modules.

## Test Structure

- `test_datapreprocessing.py` - Tests for data extraction and preprocessing functions
- `test_featureengineering.py` - Tests for feature engineering and business rule generation
- `test_training.py` - Tests for model training, evaluation, and MLflow integration

## Test Coverage

### DataPreprocessing Tests
- **Policy Data Reading**: Tests for reading policy, vehicle, driver, and customer data
- **Claim Data Reading**: Tests for claim data extraction and version handling
- **Target Variable Creation**: Tests for fraud risk indicator generation
- **Pipeline Functions**: Tests for daily and training preprocessing pipelines
- **Utility Functions**: Tests for data quality validation

### FeatureEngineering Tests
- **Damage Calculations**: Tests for damage score calculation and area counting
- **Business Rules**: Tests for all 11 check variables (C1-C14)
- **Feature Transformations**: Tests for temporal features and missing value imputation
- **Main Pipeline**: Tests for complete feature engineering pipeline

### Model Training Tests
- **Data Preparation**: Tests for training data preparation
- **Pipeline Creation**: Tests for preprocessing and model pipeline creation
- **Model Training**: Tests for hyperparameter tuning and evaluation
- **MLflow Integration**: Tests for experiment tracking and model registry
- **Combined Models**: Tests for two-stage model evaluation
- **Production Helpers**: Tests for model deployment utilities

## Running the Tests

### Prerequisites
```bash
pip install pytest pytest-cov
pip install pyspark pandas numpy scikit-learn lightgbm mlflow
```

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Module
```bash
pytest tests/test_datapreprocessing.py
pytest tests/test_featureengineering.py
pytest tests/test_training.py
```

### Run with Coverage
```bash
pytest tests/ --cov=notebooks --cov-report=html
```

### Run Specific Test Class
```bash
pytest tests/test_datapreprocessing.py::TestPolicyDataReading
pytest tests/test_featureengineering.py::TestBusinessRules
pytest tests/test_training.py::TestMLflowIntegration
```

### Run Specific Test Function
```bash
pytest tests/test_datapreprocessing.py::TestPolicyDataReading::test_read_policy_data
```

## Test Configuration

The tests use mocking extensively to avoid dependencies on:
- Spark cluster/tables
- MLflow tracking server
- External data sources
- Configuration files

This allows tests to run quickly and independently.

## Writing New Tests

When adding new functionality to the notebooks, follow these patterns:

1. **Mock External Dependencies**
   ```python
   with patch('notebooks.ModuleName.external_function') as mock:
       mock.return_value = expected_data
       # Test your function
   ```

2. **Use Fixtures for Common Data**
   ```python
   @pytest.fixture
   def sample_data(spark):
       return spark.createDataFrame([...])
   ```

3. **Test Both Success and Failure Cases**
   ```python
   def test_function_success(self):
       # Test normal operation
   
   def test_function_with_nulls(self):
       # Test edge cases
   ```

4. **Verify Data Transformations**
   ```python
   # Check output schema
   assert "expected_column" in result.columns
   
   # Verify calculations
   assert result.filter(condition).count() == expected_count
   ```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install -r requirements.txt
    pytest tests/ --junitxml=test-results.xml --cov=notebooks --cov-report=xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the notebooks directory is in PYTHONPATH
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

2. **Spark Session Conflicts**: Tests create local Spark sessions. If you see conflicts:
   ```python
   # In conftest.py or test file
   spark.stop()  # Stop any existing sessions
   ```

3. **Memory Issues**: For large test datasets, increase driver memory:
   ```python
   spark = SparkSession.builder \
       .config("spark.driver.memory", "2g") \
       .getOrCreate()
   ```

## Best Practices

1. **Keep Tests Fast**: Use small datasets and mock external calls
2. **Test One Thing**: Each test should verify a single behavior
3. **Use Descriptive Names**: Test names should explain what they verify
4. **Clean Up Resources**: Always stop Spark sessions in teardown
5. **Avoid Test Interdependence**: Tests should run in any order

## Contributing

When contributing new tests:
1. Follow existing patterns and naming conventions
2. Add docstrings explaining what the test verifies
3. Include both positive and negative test cases
4. Update this README if adding new test categories