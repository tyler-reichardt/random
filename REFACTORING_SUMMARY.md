# Third Party Capture ML Pipeline - Refactoring Summary

## Executive Summary

I successfully transformed a data science proof-of-concept project into a production-ready ML pipeline for the Third Party Capture (TPC) fraud detection system. The refactoring involved analyzing experimental notebooks, extracting core logic, and rebuilding it as a modular, scalable, and maintainable MLOps solution.

## What Was Done

### 1. Project Analysis and Understanding

I began by thoroughly analyzing the existing experimental code structure:
- **Deployment scripts** (`01_daily_policies.py`, `02_daily_claims.py`, `03_daily_scoring.py`) - Understanding the daily workflow
- **Experimental notebooks** (`02_Data_Build_and_Quality_Checks.py`, `04a_Desk_Check_Model.py`, `04b_Interview_Model.py`) - Extracting core ML logic
- **Configuration files** - Understanding data sources and parameters

The project implements a two-stage fraud detection system for single-vehicle insurance claims, using business rules and machine learning to identify potentially fraudulent claims.

### 2. Configuration System Enhancement

**Original Issue**: The configuration system had hardcoded paths and fragile string replacements.

**Solution Implemented**:
- Created robust project root detection that searches upward for project markers
- Implemented intelligent path resolution that works from any location
- Added fallback mechanisms for configuration loading
- Provided clear error messages and debugging information

**Key Features**:
```python
# Automatic project structure detection
project_root = find_project_root()  # Searches for configs/ and notebooks/

# Simplified config loading
extract_column_transformation_lists("/config_files/training.yaml")

# New utility functions
list_available_configs()  # Shows all config files
reload_all_configs()      # Refresh all configurations
```

### 3. Data Preprocessing Module

**Purpose**: Centralize all data extraction and preprocessing logic.

**What I Built**:
- **Unified Data Reading Functions**:
  - `read_policy_data()` - Extracts policy information
  - `read_vehicle_data()` - Gets vehicle details with location info
  - `read_driver_data()` - Aggregates multi-driver households
  - `read_customer_data()` - Customer demographics
  - `read_claim_data()` - SVI claims with incident details

- **Target Variable Creation**:
  - `get_referral_vertices()` - Processes fraud investigation outcomes
  - `create_target_variable()` - Combines multiple risk indicators

- **Pipeline Functions**:
  - `run_daily_preprocessing()` - For production scoring
  - `run_training_preprocessing()` - For model training

**Key Improvements**:
- Parameterized date filtering
- Comprehensive error handling
- Data quality validation at each step
- Logging for monitoring and debugging

### 4. Feature Engineering Module

**Purpose**: Transform raw data into ML-ready features with business logic.

**What I Built**:
- **Damage Score Calculations**:
  - Multiplicative scoring system (Minimal=2, Medium=3, Heavy=4, Severe=5)
  - Area counts and relative damage metrics
  - Spark UDF for efficient processing

- **Business Rule Implementation** (11 fraud indicators):
  - C1: Friday/Saturday night incidents
  - C2: Reporting delays (≥3 days)
  - C3: Weekend incidents reported Monday
  - C5: Night incidents (11 PM - 5 AM)
  - C6: Rush hour without commuting coverage
  - C7: Police attendance/crime reference
  - C9: New policy (<30 days)
  - C10: Policy ending soon (<60 days)
  - C11: Young/inexperienced drivers
  - C12: Expensive car for age
  - C14: Suspicious claim descriptions

- **Feature Processing**:
  - Temporal features (vehicle age, driver age at claim)
  - Driver aggregations (min/max across household)
  - Missing value imputation strategies
  - Type conversions for modeling

**Production Features**:
- Pre-calculated imputation values to ensure consistency
- Modular feature lists for different models
- Stage-specific processing (training vs scoring)

### 5. Model Training Module

**Purpose**: Implement production-grade model training with MLOps best practices.

**What I Built**:
- **Two-Stage Model Architecture**:
  - FA Model (Fraud Analysis/Desk Check) - Initial screening
  - Interview Model - Secondary verification
  - Combined logic requiring both models + business rules

- **ML Pipeline Components**:
  - Preprocessing pipelines with StandardScaler and OneHotEncoder
  - LightGBM classifiers with different configurations
  - Imbalanced learning support (SMOTE, undersampling)

- **Training Features**:
  - Hyperparameter tuning with RandomizedSearchCV
  - Cross-validation with stratified folds
  - Comprehensive evaluation metrics
  - Feature importance tracking

- **MLflow Integration**:
  - Automatic experiment tracking
  - Model versioning and registry
  - Artifact logging (feature importance, plots)
  - Model signature inference

**Key Improvements**:
- Parameterized pipeline creation
- Model-specific optimization strategies
- Threshold optimization for combined models
- Champion/challenger comparison framework

### 6. Orchestration Layer

**Purpose**: Provide unified entry point for all pipeline operations.

**What I Built**:
- **Three Pipeline Modes**:
  1. **Daily Scoring**: Process new claims
     - Extract daily claims
     - Apply feature engineering
     - Load production models
     - Generate predictions
     - Save results

  2. **Model Training**: Retrain models
     - Prepare historical data
     - Train both models
     - Optimize thresholds
     - Register in MLflow
     - Compare with champions

  3. **Monitoring**: Track performance
     - Analyze recent predictions
     - Calculate drift metrics
     - Flag anomalies
     - Generate alerts

- **Scheduling Support**:
  - Parameterized execution via Databricks widgets
  - JSON configuration for job scheduling
  - Return status for workflow integration

### 7. Documentation and Best Practices

**Created Documentation**:
- Comprehensive README with architecture overview
- Module-level documentation
- Function docstrings with parameter descriptions
- Usage examples for common scenarios
- Troubleshooting guide

**Implemented Best Practices**:
- Logging at appropriate levels
- Error handling with context
- Type hints where beneficial
- Consistent naming conventions
- Modular, testable functions

## Technical Improvements Made

### Code Quality
- **Modularization**: Separated concerns into focused modules instead of monolithic notebooks
- **Reusability**: Functions can be imported and used independently
- **Maintainability**: Clear structure makes updates easier
- **Testing**: Modular design enables unit testing

### Performance
- **Spark Optimization**: Efficient use of DataFrame operations
- **Caching Strategy**: Reuse of computed features
- **Parallel Processing**: Leverages Spark's distributed computing

### MLOps Integration
- **Experiment Tracking**: Full MLflow integration
- **Model Registry**: Version control for models
- **Reproducibility**: Logged parameters and artifacts
- **Monitoring**: Built-in performance tracking

### Data Quality
- **Validation**: Checks at each pipeline stage
- **Logging**: Comprehensive audit trail
- **Error Handling**: Graceful failure with context

## Business Logic Preservation

Throughout the refactoring, I carefully preserved all business logic:
- Same feature calculations
- Identical business rules
- Matching model architectures
- Compatible thresholds and scoring

The refactored pipeline produces the same results as the experimental code while being production-ready.

## Migration Path

The refactored code is designed for smooth migration:
1. **Parallel Running**: Can run alongside existing code
2. **Incremental Adoption**: Modules can be used independently
3. **Backward Compatible**: Works with existing data schemas
4. **Configuration Driven**: Easy to adjust without code changes

## Benefits Achieved

1. **Scalability**: Handles production data volumes
2. **Reliability**: Comprehensive error handling
3. **Maintainability**: Clear, modular structure
4. **Observability**: Built-in logging and monitoring
5. **Flexibility**: Easy to extend and modify
6. **Compliance**: Audit trail and versioning

## Future Enhancements Enabled

The refactored architecture enables:
- Real-time scoring capabilities
- A/B testing frameworks
- Feature store integration
- AutoML experimentation
- Model explainability tools

## Conclusion

The refactoring transformed experimental data science code into a production-grade ML platform. The solution maintains all original functionality while adding enterprise features like monitoring, versioning, and scalability. The modular design ensures the codebase can evolve with changing business needs while maintaining high quality standards.