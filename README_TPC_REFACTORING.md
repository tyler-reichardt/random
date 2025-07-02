# Third Party Capture (TPC) ML Pipeline - Production Refactoring

## Overview

This document describes the refactored production-ready ML pipeline for the Third Party Capture (TPC) project. The refactoring transforms the experimental data science code into a modular, scalable, and maintainable MLOps solution.

## Architecture

The TPC pipeline uses a two-stage model architecture:
1. **FA Model (Fraud Analysis/Desk Check)** - Initial screening model
2. **Interview Model** - Secondary verification model

Claims are flagged as high-risk only if both models predict positive AND business rules are triggered.

## Project Structure

```
.
├── configs/
│   ├── configs.py                      # Configuration management module
│   └── config_files/
│       ├── configs.yaml                # Main configuration
│       ├── data_preprocessing.yaml     # Data preprocessing settings
│       ├── feature_engineering.yaml    # Feature engineering settings
│       ├── training.yaml              # Model training configuration
│       ├── scoring.yaml               # Scoring configuration
│       └── serving.yaml               # Model serving configuration
│
├── notebooks/
│   ├── DataPreprocessing.py          # Data extraction and preprocessing
│   ├── FeatureEngineering.py         # Feature engineering and business rules
│   ├── TPCModelTraining.py           # Model training and evaluation
│   ├── TPCModelServing.py            # Model serving endpoint management
│   └── TPCOrchestration.py           # Pipeline orchestration
│
├── experiments/                       # Original data science code (reference)
│   ├── deployment/                    # Daily job scripts
│   └── notebooks/                     # Exploratory notebooks
│
└── functions/                         # Utility functions (if needed)
```

## Module Descriptions

### 1. DataPreprocessing.py

Handles all data extraction and preprocessing:
- **Policy Data**: Extracts policy, vehicle, driver, and customer information
- **Claims Data**: Processes Single Vehicle Incident (SVI) claims
- **Target Variables**: Creates fraud risk indicators from referral logs
- **Data Quality**: Validates and reports data quality metrics

Key Functions:
- `run_daily_preprocessing()` - Daily pipeline for new claims
- `run_training_preprocessing()` - Historical data for model training
- `prepare_policy_features()` - Consolidates all policy-related features
- `create_target_variable()` - Creates training labels from SVI performance data

### 2. FeatureEngineering.py

Transforms raw data into model-ready features:
- **Damage Scores**: Calculates vehicle damage severity metrics
- **Business Rules**: Implements 11 check variables (C1-C14) for fraud detection
- **Temporal Features**: Creates time-based features
- **Missing Values**: Applies domain-specific imputation strategies

Key Functions:
- `run_feature_engineering()` - Main feature engineering pipeline
- `generate_check_variables()` - Creates business rule indicators
- `apply_damage_calculations()` - Computes damage severity scores
- `get_model_features()` - Returns feature lists for each model

### 3. TPCModelTraining.py

Manages model training and MLflow integration:
- **Pipeline Creation**: Builds preprocessing and model pipelines
- **Hyperparameter Tuning**: Uses RandomizedSearchCV for optimization
- **Model Evaluation**: Comprehensive metrics and visualizations
- **MLflow Tracking**: Logs experiments, models, and artifacts
- **Model Registry**: Manages model versioning and staging

Key Functions:
- `run_training_pipeline()` - Complete training workflow
- `train_and_log_model()` - Trains individual model with MLflow
- `evaluate_combined_models()` - Evaluates two-stage model performance
- `optimize_thresholds()` - Finds optimal classification thresholds

### 4. TPCOrchestration.py

Orchestrates the complete ML pipeline:
- **Daily Scoring**: Processes new claims
- **Model Training**: Retrains models on schedule
- **Monitoring**: Tracks model performance and drift

Pipeline Modes:
- `daily_scoring` - Score new claims
- `model_training` - Retrain models
- `monitoring` - Performance monitoring

### 5. configs/configs.py

Centralized configuration management:
- Automatically detects project structure
- Loads YAML configurations
- Sets global variables for easy access
- Provides utility functions for config management

## Usage Examples

### Daily Scoring Pipeline

```python
# Run from TPCOrchestration notebook
%run ./TPCOrchestration

# Parameters:
# - pipeline_mode: "daily_scoring"
# - execution_date: "2025-01-15"
# - lookback_days: "1"
```

### Model Training

```python
# Run training pipeline
from TPCModelTraining import run_training_pipeline

results = run_training_pipeline(
    start_date="2023-01-01",
    register_models=True
)
```

### Feature Engineering Only

```python
# Process features for a dataset
from FeatureEngineering import run_feature_engineering

features_df = run_feature_engineering(
    preprocessed_df,
    stage="training"  # or "scoring"
)
```

## Business Rules (Check Variables)

The pipeline implements 11 business rules to flag suspicious claims:

1. **C1**: Friday/Saturday night incidents (8 PM - 4 AM)
2. **C2**: Reporting delay ≥ 3 days
3. **C3**: Weekend incident reported on Monday
4. **C5**: Night incident (11 PM - 5 AM)
5. **C6**: No commuting coverage but rush hour incident
6. **C7**: Police attendance or crime reference
7. **C9**: Policy inception within 30 days of incident
8. **C10**: Policy ends within 60 days of incident
9. **C11**: Young (<25) or inexperienced driver (≤3 years)
10. **C12**: Expensive vehicle for driver age
11. **C14**: Suspicious keywords in claim description

## Model Details

### FA Model (Desk Check)
- **Algorithm**: LightGBM
- **Features**: All engineered features (~120)
- **Target**: `fa_risk` (internal review outcome)
- **Optimization**: Recall (to minimize false negatives)

### Interview Model
- **Algorithm**: LightGBM with undersampling
- **Features**: Subset of features (~30)
- **Target**: `tbg_risk` (external review outcome)
- **Optimization**: F1-score (balance precision/recall)

### Combined Logic
```python
high_risk = (
    (fa_model_score >= 0.5) AND
    (interview_model_score >= 0.5) AND
    (num_failed_checks >= 1)
)
```

## Scheduling Recommendations

### Daily Jobs
- **Time**: 6:00 AM UTC
- **Purpose**: Score previous day's claims
- **Notebook**: `TPCOrchestration` with `pipeline_mode="daily_scoring"`

### Weekly Jobs
- **Time**: Mondays 8:00 AM UTC
- **Purpose**: Model monitoring and drift detection
- **Notebook**: `TPCOrchestration` with `pipeline_mode="monitoring"`

### Monthly Jobs
- **Time**: 1st of month 2:00 AM UTC
- **Purpose**: Model retraining
- **Notebook**: `TPCOrchestration` with `pipeline_mode="model_training"`

## Configuration Management

### Adding New Features
1. Update feature lists in `FeatureEngineering.get_model_features()`
2. Add transformation logic in appropriate function
3. Update imputation strategy if needed

### Modifying Business Rules
1. Edit rule logic in `FeatureEngineering.generate_check_variables()`
2. Add to check_cols list for aggregation
3. Update documentation

### Changing Model Parameters
1. Update param_grid in `TPCModelTraining.train_and_log_model()`
2. Modify pipeline creation parameters
3. Adjust scoring metrics as needed

## Data Dependencies

### Input Tables
- `prod_adp_certified.policy_motor.*` - Policy data
- `prod_adp_certified.claim.*` - Claims data
- `prod_adp_certified.customer_360.single_customer_view` - Customer data
- `prod_dsexp_auxiliarydata.single_vehicle_incident_checks.*` - SVI data

### Output Tables
- `prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_policy_svi`
- `prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_claims_svi`
- `prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions`
- `prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi`

## Monitoring and Alerts

The pipeline includes built-in monitoring:
- **Flag Rate**: Should be between 1-10% of claims
- **Model Drift**: Tracks average scores over time
- **Data Quality**: Validates nulls and duplicates

Alert conditions:
- Flag rate outside expected range
- Significant score distribution shift
- Data quality issues exceed thresholds

## Migration from Experimental Code

### Key Improvements
1. **Modularization**: Separated concerns into focused modules
2. **Configuration**: Centralized configuration management
3. **Error Handling**: Comprehensive logging and error handling
4. **MLflow Integration**: Full experiment tracking and model registry
5. **Data Quality**: Built-in validation and monitoring
6. **Scalability**: Optimized for large-scale processing
7. **Documentation**: Comprehensive inline and module documentation

### Backward Compatibility
- Core business logic preserved from original notebooks
- Same feature engineering and model architectures
- Compatible with existing data schemas
- Results match experimental outputs

## Troubleshooting

### Common Issues
1. **Config not found**: Check project root detection in configs.py
2. **Feature mismatch**: Verify feature lists match between training/scoring
3. **Memory issues**: Adjust Spark configurations for large datasets
4. **Model not found**: Ensure models are registered in MLflow

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Real-time Scoring**: Integrate with streaming pipeline
2. **Feature Store**: Centralize feature definitions
3. **A/B Testing**: Built-in experiment framework
4. **AutoML**: Automated hyperparameter optimization
5. **Model Explainability**: SHAP/LIME integration

## Contact

For questions or issues:
- Create an issue in the repository
- Contact the MLOps team
- Review the experimental notebooks for additional context

---

Last Updated: January 2025
Version: 1.0