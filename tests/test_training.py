import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, 
    BooleanType
)
from pyspark.sql.functions import col, lit, when
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
import mlflow
from mlflow.models import infer_signature

# Mock the notebooks module structure
import sys
sys.path.append('..')

# Mock the configs and dependencies
with patch('notebooks.TPCModelTraining.extract_column_transformation_lists'):
    with patch('notebooks.TPCModelTraining.spark') as mock_spark:
        with patch('notebooks.TPCModelTraining.run_training_preprocessing'):
            with patch('notebooks.TPCModelTraining.run_feature_engineering'):
                from notebooks.TPCModelTraining import *

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-training")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()

@pytest.fixture
def sample_training_data():
    """Create sample training data for model testing."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(20)]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    return {
        "X_train": X_train_df,
        "X_test": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "features": feature_names
    }

@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('notebooks.TPCModelTraining.mlflow') as mock:
        mock.set_experiment = MagicMock()
        mock.start_run = MagicMock()
        mock.log_param = MagicMock()
        mock.log_metric = MagicMock()
        mock.sklearn = MagicMock()
        mock.sklearn.log_model = MagicMock()
        yield mock

class TestDataPreparation:
    """Test data preparation functions."""
    
    def test_prepare_training_data(self, spark):
        """Test prepare_training_data function."""
        # Mock the preprocessing and feature engineering
        with patch('notebooks.TPCModelTraining.run_training_preprocessing') as mock_preprocess:
            with patch('notebooks.TPCModelTraining.run_feature_engineering') as mock_features:
                with patch('notebooks.TPCModelTraining.get_model_features') as mock_get_features:
                    # Create mock data
                    mock_df = spark.createDataFrame(
                        pd.DataFrame({
                            'claim_number': ['CLM001', 'CLM002', 'CLM003', 'CLM004'],
                            'fa_risk': [1, 0, 1, 0],
                            'tbg_risk': [0, 1, 1, 0],
                            'feature_1': [1.0, 2.0, 3.0, 4.0],
                            'feature_2': [5.0, 6.0, 7.0, 8.0]
                        })
                    )
                    
                    mock_preprocess.return_value = mock_df
                    mock_features.return_value = mock_df
                    mock_get_features.return_value = {
                        'all_features': ['feature_1', 'feature_2'],
                        'interview_features': ['feature_1']
                    }
                    
                    # Test the function
                    result = prepare_training_data(start_date="2023-01-01")
                    
                    assert result is not None
                    assert "fa_model" in result
                    assert "interview_model" in result
                    assert "feature_lists" in result
                    
                    # Check FA model data
                    assert len(result["fa_model"]["X_train"]) > 0
                    assert len(result["fa_model"]["X_test"]) > 0
                    assert result["fa_model"]["features"] == ['feature_1', 'feature_2']
                    
                    # Check Interview model data
                    assert len(result["interview_model"]["X_train"]) > 0
                    assert result["interview_model"]["features"] == ['feature_1']

class TestPipelineCreation:
    """Test model pipeline creation functions."""
    
    def test_create_preprocessing_pipeline(self):
        """Test create_preprocessing_pipeline function."""
        numeric_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']
        
        pipeline = create_preprocessing_pipeline(numeric_features, categorical_features)
        
        assert isinstance(pipeline, ColumnTransformer)
        assert len(pipeline.transformers) == 2
        assert pipeline.transformers[0][0] == 'num'
        assert pipeline.transformers[1][0] == 'cat'
    
    def test_create_lgbm_pipeline(self):
        """Test create_lgbm_pipeline function."""
        numeric_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']
        
        # Test without sampling
        pipeline = create_lgbm_pipeline(
            numeric_features, categorical_features,
            use_smote=False, use_undersampling=False
        )
        
        assert isinstance(pipeline, Pipeline)
        assert 'preprocessor' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps
        assert isinstance(pipeline.named_steps['classifier'], LGBMClassifier)
        
        # Test with SMOTE
        pipeline_smote = create_lgbm_pipeline(
            numeric_features, categorical_features,
            use_smote=True, use_undersampling=False
        )
        
        # Should be ImbPipeline when using SMOTE
        assert 'smote' in str(type(pipeline_smote))
        
        # Test with undersampling
        pipeline_under = create_lgbm_pipeline(
            numeric_features, categorical_features,
            use_smote=False, use_undersampling=True
        )
        
        assert 'undersampler' in str(type(pipeline_under))

class TestModelTraining:
    """Test model training functions."""
    
    def test_tune_model(self, sample_training_data):
        """Test tune_model function."""
        # Create a simple pipeline
        numeric_features = [f for f in sample_training_data['features']]
        categorical_features = []
        
        pipeline = create_lgbm_pipeline(
            numeric_features, categorical_features,
            use_smote=False, use_undersampling=False
        )
        
        # Simple parameter grid for testing
        param_grid = {
            'classifier__n_estimators': [10, 20],
            'classifier__max_depth': [3, 5]
        }
        
        # Test tuning with small dataset
        best_model = tune_model(
            pipeline,
            sample_training_data['X_train'],
            sample_training_data['y_train'],
            param_grid=param_grid,
            cv=2,  # Small CV for testing
            n_iter=4  # Small iterations
        )
        
        assert best_model is not None
        assert hasattr(best_model, 'predict')
        assert hasattr(best_model, 'predict_proba')
    
    def test_evaluate_model(self, sample_training_data):
        """Test evaluate_model function."""
        # Train a simple model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LGBMClassifier(n_estimators=10, random_state=42))
        ])
        model.fit(sample_training_data['X_train'], sample_training_data['y_train'])
        
        # Evaluate
        results = evaluate_model(
            model,
            sample_training_data['X_test'],
            sample_training_data['y_test'],
            model_name="Test Model"
        )
        
        assert 'metrics' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results
        assert 'predictions' in results
        assert 'probabilities' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'avg_precision' in metrics
        
        # All metrics should be between 0 and 1
        for metric_value in metrics.values():
            assert 0 <= metric_value <= 1

class TestMLflowIntegration:
    """Test MLflow integration functions."""
    
    def test_train_and_log_model(self, sample_training_data, mock_mlflow):
        """Test train_and_log_model function."""
        # Mock get_model_features
        with patch('notebooks.TPCModelTraining.get_model_features') as mock_features:
            mock_features.return_value = {
                'all_features': sample_training_data['features'],
                'interview_features': sample_training_data['features'][:10],
                'numeric_features': sample_training_data['features'],
                'categorical_features': [],
                'num_interview': sample_training_data['features'][:10],
                'cat_interview': []
            }
            
            # Prepare model data
            model_data = {
                'X_train': sample_training_data['X_train'],
                'X_test': sample_training_data['X_test'],
                'y_train': sample_training_data['y_train'],
                'y_test': sample_training_data['y_test'],
                'features': sample_training_data['features']
            }
            
            # Mock MLflow run
            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_id"
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
            
            # Test training without tuning
            model, run_id, eval_results = train_and_log_model(
                model_data,
                model_type="fa",
                experiment_name="test_experiment",
                tune_hyperparameters=False
            )
            
            assert model is not None
            assert run_id == "test_run_id"
            assert eval_results is not None
            assert 'metrics' in eval_results
            
            # Check MLflow calls
            mock_mlflow.set_experiment.assert_called_once()
            mock_mlflow.log_param.assert_called()
            mock_mlflow.log_metric.assert_called()
    
    def test_register_model(self, mock_mlflow):
        """Test register_model function."""
        # Mock MLflow client
        mock_client = MagicMock()
        with patch('notebooks.TPCModelTraining.mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Mock model version
            mock_version = MagicMock()
            mock_version.version = "1"
            mock_client.create_model_version.return_value = mock_version
            
            # Test registration
            version = register_model("test_run_id", "test_model", "fa")
            
            assert version is not None
            assert version.version == "1"
            
            # Check client calls
            mock_client.create_model_version.assert_called_once()
    
    def test_transition_model_stage(self, mock_mlflow):
        """Test transition_model_stage function."""
        # Mock MLflow client
        mock_client = MagicMock()
        with patch('notebooks.TPCModelTraining.mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Test transition
            transition_model_stage("test_model", "1", "Production")
            
            # Check client calls
            mock_client.transition_model_version_stage.assert_called_once_with(
                name="test_model",
                version="1",
                stage="Production"
            )

class TestCombinedModels:
    """Test combined model evaluation functions."""
    
    def test_evaluate_combined_models(self, sample_training_data):
        """Test evaluate_combined_models function."""
        # Create two simple models
        fa_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LGBMClassifier(n_estimators=10, random_state=42))
        ])
        interview_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LGBMClassifier(n_estimators=10, random_state=43))
        ])
        
        # Train models
        fa_model.fit(sample_training_data['X_train'], sample_training_data['y_train'])
        interview_model.fit(sample_training_data['X_train'][:, :10], sample_training_data['y_train'])
        
        # Create test dataframe
        test_df = sample_training_data['X_test'].copy()
        test_df['svi_risk'] = sample_training_data['y_test']
        test_df['num_failed_checks'] = np.random.randint(0, 3, size=len(test_df))
        
        # Mock get_model_features
        with patch('notebooks.TPCModelTraining.get_model_features') as mock_features:
            mock_features.return_value = {
                'all_features': sample_training_data['features'],
                'interview_features': sample_training_data['features'][:10]
            }
            
            # Evaluate combined models
            metrics = evaluate_combined_models(
                test_df, fa_model, interview_model,
                fa_threshold=0.5, interview_threshold=0.5
            )
            
            assert metrics is not None
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'confusion_matrix' in metrics
    
    def test_optimize_thresholds(self, sample_training_data):
        """Test optimize_thresholds function."""
        # Create two simple models
        fa_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LGBMClassifier(n_estimators=10, random_state=42))
        ])
        interview_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LGBMClassifier(n_estimators=10, random_state=43))
        ])
        
        # Train models
        fa_model.fit(sample_training_data['X_train'], sample_training_data['y_train'])
        interview_model.fit(sample_training_data['X_train'][:, :10], sample_training_data['y_train'])
        
        # Create test dataframe
        test_df = sample_training_data['X_test'].copy()
        test_df['svi_risk'] = sample_training_data['y_test']
        test_df['num_failed_checks'] = 1  # All have at least one failed check
        
        # Mock get_model_features
        with patch('notebooks.TPCModelTraining.get_model_features') as mock_features:
            mock_features.return_value = {
                'all_features': sample_training_data['features'],
                'interview_features': sample_training_data['features'][:10]
            }
            
            # Test threshold optimization with small range
            thresholds_range = np.arange(0.3, 0.8, 0.1)
            results_df, best_thresholds = optimize_thresholds(
                test_df, fa_model, interview_model,
                thresholds_range=thresholds_range
            )
            
            assert results_df is not None
            assert len(results_df) > 0
            assert 'fa_threshold' in results_df.columns
            assert 'interview_threshold' in results_df.columns
            assert 'f1' in results_df.columns
            
            assert best_thresholds is not None
            assert 'fa_threshold' in best_thresholds
            assert 'interview_threshold' in best_thresholds
            assert 'f1' in best_thresholds

class TestTrainingPipeline:
    """Test main training pipeline."""
    
    def test_run_training_pipeline(self, sample_training_data, mock_mlflow):
        """Test run_training_pipeline function."""
        # Mock all dependencies
        with patch('notebooks.TPCModelTraining.prepare_training_data') as mock_prepare:
            with patch('notebooks.TPCModelTraining.train_and_log_model') as mock_train:
                with patch('notebooks.TPCModelTraining.register_model') as mock_register:
                    with patch('notebooks.TPCModelTraining.optimize_thresholds') as mock_optimize:
                        # Setup mocks
                        mock_prepare.return_value = {
                            'fa_model': {
                                'X_train': sample_training_data['X_train'],
                                'X_test': sample_training_data['X_test'],
                                'y_train': sample_training_data['y_train'],
                                'y_test': sample_training_data['y_test'],
                                'features': sample_training_data['features']
                            },
                            'interview_model': {
                                'X_train': sample_training_data['X_train'],
                                'X_test': sample_training_data['X_test'],
                                'y_train': sample_training_data['y_train'],
                                'y_test': sample_training_data['y_test'],
                                'features': sample_training_data['features'][:10]
                            },
                            'feature_lists': {}
                        }
                        
                        # Mock model training results
                        mock_model = MagicMock()
                        mock_train.return_value = (
                            mock_model,
                            "test_run_id",
                            {'metrics': {'roc_auc': 0.85}}
                        )
                        
                        # Mock model registration
                        mock_version = MagicMock()
                        mock_version.version = "1"
                        mock_register.return_value = mock_version
                        
                        # Mock threshold optimization
                        mock_optimize.return_value = (
                            pd.DataFrame({'fa_threshold': [0.5], 'interview_threshold': [0.5], 'f1': [0.75]}),
                            {'fa_threshold': 0.5, 'interview_threshold': 0.5}
                        )
                        
                        # Test pipeline
                        results = run_training_pipeline(
                            start_date="2023-01-01",
                            register_models=True
                        )
                        
                        assert results is not None
                        assert 'fa_model' in results
                        assert 'interview_model' in results
                        
                        # Check calls
                        mock_prepare.assert_called_once()
                        assert mock_train.call_count == 2  # FA and Interview models
                        assert mock_register.call_count == 2  # Both models registered

class TestModelComparison:
    """Test model comparison functions."""
    
    def test_compare_models(self):
        """Test compare_models function."""
        # Current results
        current_results = {
            'fa_model': {
                'results': {
                    'metrics': {
                        'roc_auc': 0.90,
                        'avg_precision': 0.85,
                        'f1': 0.80
                    }
                }
            },
            'interview_model': {
                'results': {
                    'metrics': {
                        'roc_auc': 0.88,
                        'avg_precision': 0.82,
                        'f1': 0.78
                    }
                }
            }
        }
        
        # Champion metrics
        champion_metrics = {
            'fa_model': {
                'roc_auc': 0.85,
                'avg_precision': 0.80,
                'f1': 0.75
            },
            'interview_model': {
                'roc_auc': 0.86,
                'avg_precision': 0.81,
                'f1': 0.77
            }
        }
        
        # Compare
        comparison = compare_models(current_results, champion_metrics)
        
        assert comparison is not None
        assert 'fa_model' in comparison
        assert 'interview_model' in comparison
        
        # Check FA model comparison
        fa_comp = comparison['fa_model']
        assert fa_comp['current'] == current_results['fa_model']['results']['metrics']
        assert fa_comp['champion'] == champion_metrics['fa_model']
        assert fa_comp['improvement']['roc_auc'] > 0  # Current is better
        assert fa_comp['recommend_promotion'] is True
        
        # Check Interview model comparison
        int_comp = comparison['interview_model']
        assert int_comp['improvement']['roc_auc'] > 0  # Current is better

class TestProductionHelpers:
    """Test production deployment helper functions."""
    
    def test_get_production_models(self, mock_mlflow):
        """Test get_production_models function."""
        # Mock MLflow client
        mock_client = MagicMock()
        with patch('notebooks.TPCModelTraining.mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Mock model versions
            mock_fa_version = MagicMock()
            mock_fa_version.version = "2"
            mock_int_version = MagicMock()
            mock_int_version.version = "3"
            
            mock_client.get_latest_versions.side_effect = [
                [mock_fa_version],
                [mock_int_version]
            ]
            
            # Mock model loading
            mock_fa_model = MagicMock()
            mock_int_model = MagicMock()
            
            with patch('notebooks.TPCModelTraining.mlflow.sklearn.load_model') as mock_load:
                mock_load.side_effect = [mock_fa_model, mock_int_model]
                
                # Test loading
                models = get_production_models()
                
                assert models is not None
                assert 'fa_model' in models
                assert 'interview_model' in models
                assert models['fa_version'] == "2"
                assert models['interview_version'] == "3"

if __name__ == "__main__":
    pytest.main([__file__])