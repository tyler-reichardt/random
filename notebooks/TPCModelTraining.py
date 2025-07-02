# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training Module
# MAGIC 
# MAGIC Production-ready model training pipeline for Third Party Capture (TPC) project.
# MAGIC This module handles:
# MAGIC - Training two-stage model architecture (FA Model + Interview Model)
# MAGIC - Hyperparameter tuning with cross-validation
# MAGIC - Model evaluation and metrics tracking
# MAGIC - MLflow experiment tracking and model registry
# MAGIC - Model validation and champion/challenger comparison

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

# MAGIC %run ./DataPreprocessing

# COMMAND ----------

# MAGIC %run ./FeatureEngineering

# COMMAND ----------

# Import required libraries
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Loading

# COMMAND ----------

# Load training configuration
extract_column_transformation_lists("/config_files/training.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

# MLflow setup
mlflow.set_registry_uri("databricks-uc")
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
catalog = workspaces.get(workspace_url, "") + catalog_prefix

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation Functions

# COMMAND ----------

def prepare_training_data(start_date="2023-01-01", test_size=0.2, random_state=42):
    """
    Prepare data for model training with preprocessing and feature engineering.
    
    Parameters:
    start_date: Start date for training data
    test_size: Proportion of data for testing
    random_state: Random seed for reproducibility
    
    Returns:
    Tuple of (X_train, X_test, y_train, y_test, feature_lists)
    """
    logger.info("Preparing training data")
    
    # Run preprocessing
    training_data = run_training_preprocessing(start_date)
    
    # Run feature engineering
    features_df = run_feature_engineering(training_data, stage="training")
    
    # Convert to Pandas for model training
    features_pd = features_df.toPandas()
    
    # Get feature lists
    feature_lists = get_model_features()
    
    # Prepare target variables
    # FA model uses fa_risk as target
    # Interview model uses tbg_risk as target
    
    # Filter for valid target values
    fa_data = features_pd[features_pd['fa_risk'].isin([0, 1])].copy()
    interview_data = features_pd[features_pd['tbg_risk'].isin([0, 1])].copy()
    
    # Split data for FA model
    X_fa = fa_data[feature_lists['all_features']]
    y_fa = fa_data['fa_risk']
    
    X_train_fa, X_test_fa, y_train_fa, y_test_fa = train_test_split(
        X_fa, y_fa, test_size=test_size, random_state=random_state, stratify=y_fa
    )
    
    # Split data for Interview model
    X_interview = interview_data[feature_lists['interview_features']]
    y_interview = interview_data['tbg_risk']
    
    X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
        X_interview, y_interview, test_size=test_size, random_state=random_state, stratify=y_interview
    )
    
    logger.info(f"FA Model - Train: {len(X_train_fa)}, Test: {len(X_test_fa)}")
    logger.info(f"Interview Model - Train: {len(X_train_int)}, Test: {len(X_test_int)}")
    
    return {
        "fa_model": {
            "X_train": X_train_fa,
            "X_test": X_test_fa,
            "y_train": y_train_fa,
            "y_test": y_test_fa,
            "features": feature_lists['all_features']
        },
        "interview_model": {
            "X_train": X_train_int,
            "X_test": X_test_int,
            "y_train": y_train_int,
            "y_test": y_test_int,
            "features": feature_lists['interview_features']
        },
        "feature_lists": feature_lists
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Pipeline Functions

# COMMAND ----------

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Create preprocessing pipeline for features.
    
    Parameters:
    numeric_features: List of numeric feature names
    categorical_features: List of categorical feature names
    
    Returns:
    ColumnTransformer for preprocessing
    """
    # Numeric features - StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical features - OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def create_lgbm_pipeline(numeric_features, categorical_features, use_smote=False, use_undersampling=False):
    """
    Create LightGBM pipeline with preprocessing.
    
    Parameters:
    numeric_features: List of numeric feature names
    categorical_features: List of categorical feature names
    use_smote: Whether to use SMOTE for class imbalance
    use_undersampling: Whether to use undersampling for class imbalance
    
    Returns:
    Pipeline with preprocessing and LightGBM
    """
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # LightGBM classifier with default parameters
    lgbm_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    lgbm_clf = LGBMClassifier(**lgbm_params)
    
    # Create pipeline based on sampling strategy
    if use_smote:
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', lgbm_clf)
        ])
    elif use_undersampling:
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('undersampler', RandomUnderSampler(random_state=42)),
            ('classifier', lgbm_clf)
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', lgbm_clf)
        ])
    
    return pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Tuning

# COMMAND ----------

def tune_model(pipeline, X_train, y_train, param_grid=None, cv=5, scoring='roc_auc', n_iter=50):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Parameters:
    pipeline: Model pipeline
    X_train: Training features
    y_train: Training labels
    param_grid: Parameter grid for search
    cv: Number of cross-validation folds
    scoring: Scoring metric
    n_iter: Number of parameter settings sampled
    
    Returns:
    Best model from search
    """
    logger.info(f"Starting hyperparameter tuning with {n_iter} iterations")
    
    if param_grid is None:
        # Default parameter grid for LightGBM
        param_grid = {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__max_depth': [3, 5, 7, 10, -1],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__num_leaves': [15, 31, 63, 127],
            'classifier__min_child_samples': [10, 20, 30, 50],
            'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__reg_alpha': [0, 0.1, 0.5, 1.0],
            'classifier__reg_lambda': [0, 0.1, 0.5, 1.0]
        }
    
    # Create cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform randomized search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit the search
    random_search.fit(X_train, y_train)
    
    logger.info(f"Best {scoring} score: {random_search.best_score_:.4f}")
    logger.info(f"Best parameters: {random_search.best_params_}")
    
    return random_search.best_estimator_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance and return metrics.
    
    Parameters:
    model: Trained model
    X_test: Test features
    y_test: Test labels
    model_name: Name for logging
    
    Returns:
    Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Log metrics
    for metric, value in metrics.items():
        logger.info(f"{model_name} - {metric}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_model_performance(eval_results, model_name="Model"):
    """
    Plot model performance visualizations.
    
    Parameters:
    eval_results: Dictionary from evaluate_model
    model_name: Name for plot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion Matrix
    sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'{model_name} - Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(
        eval_results['y_test'], 
        eval_results['probabilities']
    )
    axes[0, 1].plot(recall, precision)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title(f'{model_name} - Precision-Recall Curve')
    axes[0, 1].grid(True)
    
    # Feature Importance (if available)
    if hasattr(eval_results.get('model'), 'classifier'):
        feature_importance = eval_results['model'].classifier.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-20:]
        top_features = feature_importance[top_features_idx]
        
        axes[1, 0].barh(range(len(top_features)), top_features)
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title(f'{model_name} - Top 20 Features')
    
    # Metrics Bar Chart
    metrics = eval_results['metrics']
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    axes[1, 1].bar(metric_names, metric_values)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title(f'{model_name} - Performance Metrics')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Integration

# COMMAND ----------

def train_and_log_model(model_data, model_type="fa", experiment_name=None, tune_hyperparameters=True):
    """
    Train model and log to MLflow.
    
    Parameters:
    model_data: Dictionary with train/test data and features
    model_type: Either "fa" or "interview"
    experiment_name: MLflow experiment name
    tune_hyperparameters: Whether to perform hyperparameter tuning
    
    Returns:
    Tuple of (trained_model, run_id, evaluation_results)
    """
    if experiment_name is None:
        experiment_name = f"{catalog}.{schema}.tpc_{model_type}_model_experiment"
    
    mlflow.set_experiment(experiment_name)
    
    # Get feature lists
    feature_lists = get_model_features()
    
    if model_type == "fa":
        numeric_features = feature_lists['numeric_features']
        categorical_features = feature_lists['categorical_features']
        model_name = "FA Model (Desk Check)"
        use_undersampling = False
    else:
        numeric_features = feature_lists['num_interview']
        categorical_features = feature_lists['cat_interview']
        model_name = "Interview Model"
        use_undersampling = True
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("tune_hyperparameters", tune_hyperparameters)
        mlflow.log_param("n_train_samples", len(model_data['X_train']))
        mlflow.log_param("n_test_samples", len(model_data['X_test']))
        mlflow.log_param("n_features", len(model_data['features']))
        mlflow.log_param("use_undersampling", use_undersampling)
        
        # Create pipeline
        pipeline = create_lgbm_pipeline(
            numeric_features, 
            categorical_features,
            use_undersampling=use_undersampling
        )
        
        # Train or tune model
        if tune_hyperparameters:
            # Custom param grids for each model type
            if model_type == "fa":
                param_grid = {
                    'classifier__n_estimators': [10, 20, 30, 50, 100],
                    'classifier__max_depth': [3, 4, 5, 6, 7],
                    'classifier__learning_rate': [0.05, 0.1, 0.15],
                    'classifier__num_leaves': [5, 10, 15, 31],
                    'classifier__min_child_weight': [0.1, 0.5, 1.0],
                    'classifier__scale_pos_weight': [1, 5, 12]
                }
                scoring = 'recall'
                n_iter = 50
            else:
                param_grid = {
                    'classifier__n_estimators': [10, 20, 30, 50],
                    'classifier__max_depth': [3, 4, 5],
                    'classifier__learning_rate': [0.1],
                    'classifier__num_leaves': [5, 10, 15, 31, 61],
                    'classifier__min_child_weight': [0.1, 0.5],
                    'classifier__scale_pos_weight': [1]
                }
                scoring = 'f1'
                n_iter = 30
            
            model = tune_model(
                pipeline, 
                model_data['X_train'], 
                model_data['y_train'],
                param_grid=param_grid,
                scoring=scoring,
                n_iter=n_iter
            )
        else:
            model = pipeline
            model.fit(model_data['X_train'], model_data['y_train'])
        
        # Evaluate model
        eval_results = evaluate_model(
            model, 
            model_data['X_test'], 
            model_data['y_test'], 
            model_name
        )
        eval_results['model'] = model
        eval_results['y_test'] = model_data['y_test']
        
        # Log metrics
        for metric, value in eval_results['metrics'].items():
            mlflow.log_metric(metric, value)
        
        # Log model
        signature = infer_signature(model_data['X_train'], model.predict(model_data['X_train']))
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            pyfunc_predict_fn="predict_proba"
        )
        
        # Log feature importance
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
        else:
            classifier = model
            
        if hasattr(classifier, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': model_data['features'],
                'importance': classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save as artifact
            feature_importance.to_csv('/tmp/feature_importance.csv', index=False)
            mlflow.log_artifact('/tmp/feature_importance.csv')
        
        # Plot and log performance visualizations
        plot_model_performance(eval_results, model_name)
        
        logger.info(f"Model logged with run_id: {run.info.run_id}")
        
        return model, run.info.run_id, eval_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registration

# COMMAND ----------

def register_model(run_id, model_name, model_type="fa"):
    """
    Register model in MLflow Model Registry.
    
    Parameters:
    run_id: MLflow run ID
    model_name: Registered model name
    model_type: Model type for description
    
    Returns:
    Model version
    """
    client = mlflow.tracking.MlflowClient()
    
    # Create registered model if it doesn't exist
    try:
        client.create_registered_model(
            name=model_name,
            description=f"Third Party Capture {model_type.upper()} Model"
        )
    except Exception as e:
        logger.info(f"Model {model_name} already exists")
    
    # Register the model version
    model_uri = f"runs:/{run_id}/model"
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=f"Version created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    logger.info(f"Model registered: {model_name} version {model_version.version}")
    
    return model_version

def transition_model_stage(model_name, version, stage="Staging"):
    """
    Transition model version to a new stage.
    
    Parameters:
    model_name: Registered model name
    version: Model version
    stage: Target stage (Staging, Production, Archived)
    """
    client = mlflow.tracking.MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    logger.info(f"Model {model_name} version {version} transitioned to {stage}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combined Model Evaluation

# COMMAND ----------

def evaluate_combined_models(data_df, fa_model, interview_model, fa_threshold=0.5, interview_threshold=0.5):
    """
    Evaluate the combined performance of FA and Interview models.
    
    Parameters:
    data_df: DataFrame with features and targets
    fa_model: Trained FA model
    interview_model: Trained Interview model
    fa_threshold: Classification threshold for FA model
    interview_threshold: Classification threshold for Interview model
    
    Returns:
    Dictionary with combined evaluation metrics
    """
    logger.info("Evaluating combined model performance")
    
    # Get feature lists
    feature_lists = get_model_features()
    
    # Make predictions with both models
    fa_features = data_df[feature_lists['all_features']]
    fa_pred_proba = fa_model.predict_proba(fa_features)[:, 1]
    
    interview_features = data_df[feature_lists['interview_features']]
    interview_pred_proba = interview_model.predict_proba(interview_features)[:, 1]
    
    # Apply thresholds
    fa_pred = (fa_pred_proba >= fa_threshold).astype(int)
    interview_pred = (interview_pred_proba >= interview_threshold).astype(int)
    
    # Combined prediction logic (both models must flag as positive)
    combined_pred = ((fa_pred == 1) & (interview_pred == 1)).astype(int)
    
    # Add check requirements if available
    if 'num_failed_checks' in data_df.columns:
        combined_pred = ((combined_pred == 1) & (data_df['num_failed_checks'] >= 1)).astype(int)
    
    # Calculate metrics against true labels
    if 'svi_risk' in data_df.columns:
        y_true = data_df['svi_risk'].replace({-1: 0})
    elif 'tbg_risk' in data_df.columns:
        y_true = data_df['tbg_risk']
    else:
        logger.warning("No target variable found for evaluation")
        return None
    
    metrics = {
        'accuracy': accuracy_score(y_true, combined_pred),
        'precision': precision_score(y_true, combined_pred, zero_division=0),
        'recall': recall_score(y_true, combined_pred),
        'f1': f1_score(y_true, combined_pred),
        'confusion_matrix': confusion_matrix(y_true, combined_pred)
    }
    
    logger.info(f"Combined Model - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    return metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Threshold Optimization

# COMMAND ----------

def optimize_thresholds(data_df, fa_model, interview_model, thresholds_range=np.arange(0.1, 1.0, 0.05)):
    """
    Find optimal thresholds for combined model performance.
    
    Parameters:
    data_df: DataFrame with features and targets
    fa_model: Trained FA model
    interview_model: Trained Interview model
    thresholds_range: Range of thresholds to test
    
    Returns:
    DataFrame with performance metrics for each threshold combination
    """
    logger.info("Optimizing thresholds for combined model")
    
    results = []
    
    for fa_threshold in thresholds_range:
        for interview_threshold in thresholds_range:
            metrics = evaluate_combined_models(
                data_df, fa_model, interview_model, 
                fa_threshold, interview_threshold
            )
            
            if metrics:
                results.append({
                    'fa_threshold': fa_threshold,
                    'interview_threshold': interview_threshold,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1']
                })
    
    results_df = pd.DataFrame(results)
    
    # Find best thresholds based on F1 score
    best_idx = results_df['f1'].idxmax()
    best_thresholds = results_df.iloc[best_idx]
    
    logger.info(f"Best thresholds - FA: {best_thresholds['fa_threshold']:.2f}, "
                f"Interview: {best_thresholds['interview_threshold']:.2f}, "
                f"F1: {best_thresholds['f1']:.4f}")
    
    return results_df, best_thresholds

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Training Pipeline

# COMMAND ----------

def run_training_pipeline(start_date="2023-01-01", register_models=True):
    """
    Run complete training pipeline for both FA and Interview models.
    
    Parameters:
    start_date: Start date for training data
    register_models: Whether to register models in MLflow
    
    Returns:
    Dictionary with trained models and results
    """
    logger.info("Starting training pipeline")
    
    try:
        # Prepare data
        data = prepare_training_data(start_date)
        
        # Train FA Model
        logger.info("Training FA Model")
        fa_model, fa_run_id, fa_results = train_and_log_model(
            data['fa_model'],
            model_type="fa",
            tune_hyperparameters=True
        )
        
        # Train Interview Model
        logger.info("Training Interview Model")
        interview_model, interview_run_id, interview_results = train_and_log_model(
            data['interview_model'],
            model_type="interview",
            tune_hyperparameters=True
        )
        
        # Register models if requested
        if register_models:
            fa_model_name = f"{catalog}.{schema}.{model_name}_fa"
            interview_model_name = f"{catalog}.{schema}.{model_name}_interview"
            
            fa_version = register_model(fa_run_id, fa_model_name, "fa")
            interview_version = register_model(interview_run_id, interview_model_name, "interview")
            
            # Optionally transition to staging
            # transition_model_stage(fa_model_name, fa_version.version, "Staging")
            # transition_model_stage(interview_model_name, interview_version.version, "Staging")
        
        results = {
            'fa_model': {
                'model': fa_model,
                'run_id': fa_run_id,
                'results': fa_results,
                'version': fa_version.version if register_models else None
            },
            'interview_model': {
                'model': interview_model,
                'run_id': interview_run_id,
                'results': interview_results,
                'version': interview_version.version if register_models else None
            }
        }
        
        logger.info("Training pipeline completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison and Champion Selection

# COMMAND ----------

def compare_models(current_results, champion_metrics=None):
    """
    Compare current model with champion model metrics.
    
    Parameters:
    current_results: Results from current training
    champion_metrics: Metrics from champion model
    
    Returns:
    Comparison summary and recommendation
    """
    comparison = {}
    
    for model_type in ['fa_model', 'interview_model']:
        current_metrics = current_results[model_type]['results']['metrics']
        
        if champion_metrics and model_type in champion_metrics:
            champion = champion_metrics[model_type]
            
            comparison[model_type] = {
                'current': current_metrics,
                'champion': champion,
                'improvement': {
                    metric: current_metrics[metric] - champion[metric] 
                    for metric in current_metrics
                }
            }
            
            # Determine if current model is better
            key_metrics = ['roc_auc', 'avg_precision', 'f1']
            improvements = [comparison[model_type]['improvement'][m] for m in key_metrics]
            
            comparison[model_type]['recommend_promotion'] = sum(improvements) > 0
        else:
            comparison[model_type] = {
                'current': current_metrics,
                'champion': None,
                'improvement': None,
                'recommend_promotion': True  # No champion to compare against
            }
    
    return comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Model Deployment Helper

# COMMAND ----------

def get_production_models():
    """
    Load the latest production models from MLflow registry.
    
    Returns:
    Dictionary with loaded models
    """
    client = mlflow.tracking.MlflowClient()
    
    fa_model_name = f"{catalog}.{schema}.{model_name}_fa"
    interview_model_name = f"{catalog}.{schema}.{model_name}_interview"
    
    try:
        # Load FA model
        fa_model_version = client.get_latest_versions(fa_model_name, stages=["Production"])[0]
        fa_model = mlflow.sklearn.load_model(f"models:/{fa_model_name}/{fa_model_version.version}")
        
        # Load Interview model
        interview_model_version = client.get_latest_versions(interview_model_name, stages=["Production"])[0]
        interview_model = mlflow.sklearn.load_model(f"models:/{interview_model_name}/{interview_model_version.version}")
        
        return {
            'fa_model': fa_model,
            'interview_model': interview_model,
            'fa_version': fa_model_version.version,
            'interview_version': interview_model_version.version
        }
    except Exception as e:
        logger.error(f"Error loading production models: {str(e)}")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Usage

# COMMAND ----------

# Example: Run full training pipeline
# results = run_training_pipeline(start_date="2023-01-01", register_models=True)

# Example: Train individual model
# data = prepare_training_data()
# fa_model, fa_run_id, fa_results = train_and_log_model(
#     data['fa_model'],
#     model_type="fa",
#     tune_hyperparameters=True
# )

# Example: Compare with champion
# champion_metrics = {
#     'fa_model': {'roc_auc': 0.85, 'avg_precision': 0.80, 'f1': 0.75},
#     'interview_model': {'roc_auc': 0.88, 'avg_precision': 0.83, 'f1': 0.78}
# }
# comparison = compare_models(results, champion_metrics)

# Example: Optimize thresholds
# test_data = prepare_training_data()
# optimal_thresholds_df, best_thresholds = optimize_thresholds(
#     test_data['fa_model']['X_test'], 
#     results['fa_model']['model'],
#     results['interview_model']['model']
# )