# Databricks notebook source
# MAGIC %md
# MAGIC # TPC Pipeline Orchestration
# MAGIC 
# MAGIC This notebook orchestrates the complete Third Party Capture (TPC) ML pipeline.
# MAGIC It provides entry points for:
# MAGIC - Daily scoring pipeline for new claims
# MAGIC - Weekly/Monthly model retraining
# MAGIC - Model monitoring and performance tracking
# MAGIC - A/B testing and champion/challenger evaluation

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

# MAGIC %run ./DataPreprocessing

# COMMAND ----------

# MAGIC %run ./FeatureEngineering

# COMMAND ----------

# MAGIC %run ./TPCModelTraining

# COMMAND ----------

# Import required libraries
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from datetime import datetime, timedelta
import mlflow
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Load orchestration configuration
extract_column_transformation_lists("/config_files/configs.yaml")

# Set up parameters
dbutils.widgets.text("execution_date", datetime.now().strftime("%Y-%m-%d"), "Execution Date")
dbutils.widgets.dropdown("pipeline_mode", "daily_scoring", ["daily_scoring", "model_training", "monitoring"], "Pipeline Mode")
dbutils.widgets.text("lookback_days", "1", "Lookback Days for Daily Processing")
dbutils.widgets.dropdown("register_models", "false", ["true", "false"], "Register Models (Training Only)")

# Get widget values
execution_date = dbutils.widgets.get("execution_date")
pipeline_mode = dbutils.widgets.get("pipeline_mode")
lookback_days = int(dbutils.widgets.get("lookback_days"))
register_models = dbutils.widgets.get("register_models") == "true"

logger.info(f"Starting TPC Pipeline - Mode: {pipeline_mode}, Date: {execution_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily Scoring Pipeline

# COMMAND ----------

def run_daily_scoring_pipeline(processing_date, lookback_days=1):
    """
    Run daily scoring pipeline for new claims.
    
    Parameters:
    processing_date: Date to process claims for
    lookback_days: Number of days to look back for claims
    
    Returns:
    DataFrame with scored claims
    """
    logger.info(f"Running daily scoring for {processing_date}")
    
    try:
        # Step 1: Extract policy features for the date
        logger.info("Step 1: Extracting policy features")
        
        # Calculate date range
        end_date = datetime.strptime(processing_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get SVI claims for the date range
        svi_claims = read_claim_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            incident_date=processing_date
        )
        
        # Get latest claim versions
        svi_claims = get_latest_claim_version(svi_claims)
        
        # Prepare policy features
        policy_features = prepare_policy_features()
        
        # Join claims with policies
        daily_data = svi_claims.join(
            policy_features, 
            ["policy_number"], 
            "left"
        ).filter(col("policy_transaction_id").isNotNull())
        
        # Step 2: Extract claim features
        logger.info("Step 2: Extracting claim features")
        
        # Get claim details
        claims_data = prepare_claims_data_for_date(processing_date)
        
        # Join with policy data
        full_data = claims_data.join(
            policy_features,
            ["policy_number"],
            "left"
        ).filter(col("policy_transaction_id").isNotNull())
        
        # Step 3: Apply feature engineering
        logger.info("Step 3: Applying feature engineering")
        features_df = run_feature_engineering(full_data, stage="scoring")
        
        # Step 4: Load production models
        logger.info("Step 4: Loading production models")
        models = get_production_models()
        
        if not models:
            raise Exception("No production models found")
        
        # Step 5: Score claims
        logger.info("Step 5: Scoring claims")
        scored_df = score_claims(features_df, models)
        
        # Step 6: Save results
        output_table = f"prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions"
        
        scored_df.write \
            .mode("append") \
            .format("delta") \
            .option("mergeSchema", "true") \
            .saveAsTable(output_table)
        
        # Log metrics
        total_claims = scored_df.count()
        high_risk_claims = scored_df.filter(col("y_cmb") == 1).count()
        
        logger.info(f"Daily scoring completed - Total: {total_claims}, High Risk: {high_risk_claims}")
        
        return scored_df
        
    except Exception as e:
        logger.error(f"Error in daily scoring pipeline: {str(e)}")
        raise

def prepare_claims_data_for_date(processing_date):
    """
    Prepare claims data for a specific date.
    Similar to daily_claims.py logic but integrated.
    """
    # This would include the full claims preparation logic
    # For now, using a simplified version
    claims_query = f"""
        SELECT *
        FROM latest_claim_version lcv
        INNER JOIN claim.claim_version cv ON lcv.claim_id = cv.claim_id
        WHERE DATE(reported_date) = '{processing_date}'
    """
    
    return spark.sql(claims_query)

def score_claims(features_df, models):
    """
    Score claims using FA and Interview models.
    
    Parameters:
    features_df: DataFrame with engineered features
    models: Dictionary with FA and Interview models
    
    Returns:
    DataFrame with predictions
    """
    # Convert to pandas for scoring
    features_pd = features_df.toPandas()
    
    # Get feature lists
    feature_lists = get_model_features()
    
    # Score with FA model
    fa_features = features_pd[feature_lists['all_features']]
    features_pd['fa_pred'] = models['fa_model'].predict_proba(fa_features)[:, 1]
    
    # Score with Interview model
    interview_features = features_pd[feature_lists['interview_features']]
    features_pd['y_prob2'] = models['interview_model'].predict_proba(interview_features)[:, 1]
    
    # Apply thresholds
    fa_threshold = 0.5
    interview_threshold = 0.5
    
    features_pd['y_pred'] = (features_pd['fa_pred'] >= fa_threshold).astype(int)
    features_pd['y_pred2'] = (features_pd['y_prob2'] >= interview_threshold).astype(int)
    
    # Combined prediction
    features_pd['y_cmb'] = (
        (features_pd['y_pred'] == 1) & 
        (features_pd['y_pred2'] == 1) & 
        (features_pd['num_failed_checks'] >= 1)
    ).astype(int)
    
    # Add metadata
    features_pd['score_date'] = processing_date
    features_pd['fa_model_version'] = models.get('fa_version', 'unknown')
    features_pd['interview_model_version'] = models.get('interview_version', 'unknown')
    
    # Convert back to Spark DataFrame
    return spark.createDataFrame(features_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training Pipeline

# COMMAND ----------

def run_model_training_pipeline(start_date="2023-01-01", register=True):
    """
    Run model training pipeline.
    
    Parameters:
    start_date: Start date for training data
    register: Whether to register models in MLflow
    
    Returns:
    Training results
    """
    logger.info("Running model training pipeline")
    
    try:
        # Run training
        results = run_training_pipeline(start_date, register_models=register)
        
        # Evaluate combined performance
        if results:
            logger.info("Evaluating combined model performance")
            
            # Get test data
            test_data = prepare_training_data(start_date)
            
            # Optimize thresholds
            _, best_thresholds = optimize_thresholds(
                test_data['fa_model']['X_test'],
                results['fa_model']['model'],
                results['interview_model']['model']
            )
            
            # Log best thresholds
            logger.info(f"Optimal thresholds - FA: {best_thresholds['fa_threshold']:.2f}, "
                       f"Interview: {best_thresholds['interview_threshold']:.2f}")
            
            # Save threshold configuration
            threshold_config = {
                'fa_threshold': float(best_thresholds['fa_threshold']),
                'interview_threshold': float(best_thresholds['interview_threshold']),
                'updated_date': datetime.now().isoformat(),
                'fa_model_version': results['fa_model']['version'],
                'interview_model_version': results['interview_model']['version']
            }
            
            # Save to a configuration table or file
            spark.createDataFrame([threshold_config]).write \
                .mode("overwrite") \
                .format("delta") \
                .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.model_thresholds")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Monitoring

# COMMAND ----------

def run_monitoring_pipeline(lookback_days=7):
    """
    Monitor model performance over recent predictions.
    
    Parameters:
    lookback_days: Number of days to analyze
    
    Returns:
    Monitoring metrics
    """
    logger.info(f"Running model monitoring for last {lookback_days} days")
    
    try:
        # Load recent predictions
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        predictions_df = spark.table("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_svi_predictions") \
            .filter(col("score_date").between(start_date, end_date))
        
        # Calculate metrics
        metrics = predictions_df.groupBy("score_date").agg(
            count("*").alias("total_claims"),
            sum(when(col("y_cmb") == 1, 1).otherwise(0)).alias("flagged_claims"),
            avg("fa_pred").alias("avg_fa_score"),
            avg("y_prob2").alias("avg_interview_score")
        ).orderBy("score_date")
        
        # Check for drift
        overall_metrics = predictions_df.agg(
            count("*").alias("total_claims"),
            sum(when(col("y_cmb") == 1, 1).otherwise(0)).alias("total_flagged"),
            avg("fa_pred").alias("avg_fa_score"),
            avg("y_prob2").alias("avg_interview_score")
        ).collect()[0]
        
        flag_rate = overall_metrics["total_flagged"] / overall_metrics["total_claims"]
        
        logger.info(f"Monitoring Summary - Total Claims: {overall_metrics['total_claims']}, "
                   f"Flag Rate: {flag_rate:.2%}")
        
        # Check if retraining is needed
        if flag_rate < 0.01 or flag_rate > 0.10:
            logger.warning(f"Flag rate {flag_rate:.2%} is outside expected range [1%, 10%]. "
                          "Consider investigating or retraining models.")
        
        # Save monitoring results
        monitoring_results = {
            'monitoring_date': datetime.now().isoformat(),
            'lookback_days': lookback_days,
            'total_claims': overall_metrics["total_claims"],
            'total_flagged': overall_metrics["total_flagged"],
            'flag_rate': flag_rate,
            'avg_fa_score': overall_metrics["avg_fa_score"],
            'avg_interview_score': overall_metrics["avg_interview_score"]
        }
        
        spark.createDataFrame([monitoring_results]).write \
            .mode("append") \
            .format("delta") \
            .saveAsTable("prod_dsexp_auxiliarydata.single_vehicle_incident_checks.model_monitoring")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in monitoring pipeline: {str(e)}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Execution

# COMMAND ----------

# Main execution based on pipeline mode
if pipeline_mode == "daily_scoring":
    logger.info("Executing daily scoring pipeline")
    scored_claims = run_daily_scoring_pipeline(execution_date, lookback_days)
    
    # Display summary
    print(f"Daily scoring completed for {execution_date}")
    print(f"Total claims processed: {scored_claims.count()}")
    print(f"High-risk claims flagged: {scored_claims.filter(col('y_cmb') == 1).count()}")
    
elif pipeline_mode == "model_training":
    logger.info("Executing model training pipeline")
    training_results = run_model_training_pipeline(register=register_models)
    
    # Display results
    if training_results:
        print("Model training completed successfully")
        print(f"FA Model ROC-AUC: {training_results['fa_model']['results']['metrics']['roc_auc']:.4f}")
        print(f"Interview Model ROC-AUC: {training_results['interview_model']['results']['metrics']['roc_auc']:.4f}")
    
elif pipeline_mode == "monitoring":
    logger.info("Executing monitoring pipeline")
    monitoring_metrics = run_monitoring_pipeline(lookback_days)
    
    # Display monitoring results
    print("Model monitoring completed")
    display(monitoring_metrics)
    
else:
    raise ValueError(f"Unknown pipeline mode: {pipeline_mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scheduling Recommendations
# MAGIC 
# MAGIC ### Daily Jobs:
# MAGIC 1. **Daily Scoring Pipeline** - Run at 6 AM daily
# MAGIC    - Process previous day's claims
# MAGIC    - Score and flag high-risk claims
# MAGIC    - Save results for review
# MAGIC 
# MAGIC ### Weekly Jobs:
# MAGIC 1. **Model Monitoring** - Run every Monday at 8 AM
# MAGIC    - Analyze last 7 days of predictions
# MAGIC    - Check for model drift
# MAGIC    - Generate performance reports
# MAGIC 
# MAGIC ### Monthly Jobs:
# MAGIC 1. **Model Retraining** - Run on 1st of each month at 2 AM
# MAGIC    - Retrain models with latest data
# MAGIC    - Compare with current champion
# MAGIC    - Register new models if improved
# MAGIC 
# MAGIC ### Databricks Job Configuration:
# MAGIC ```json
# MAGIC {
# MAGIC   "name": "TPC_Daily_Scoring",
# MAGIC   "tasks": [{
# MAGIC     "task_key": "daily_scoring",
# MAGIC     "notebook_task": {
# MAGIC       "notebook_path": "/notebooks/TPCOrchestration",
# MAGIC       "base_parameters": {
# MAGIC         "pipeline_mode": "daily_scoring",
# MAGIC         "execution_date": "{{job.start_time | date: '%Y-%m-%d'}}",
# MAGIC         "lookback_days": "1"
# MAGIC       }
# MAGIC     }
# MAGIC   }],
# MAGIC   "schedule": {
# MAGIC     "quartz_cron_expression": "0 0 6 * * ?",
# MAGIC     "timezone_id": "UTC"
# MAGIC   }
# MAGIC }
# MAGIC ```

# COMMAND ----------

# Return success status
dbutils.notebook.exit(json.dumps({
    "status": "success",
    "pipeline_mode": pipeline_mode,
    "execution_date": execution_date,
    "timestamp": datetime.now().isoformat()
}))