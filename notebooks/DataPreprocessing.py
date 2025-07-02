# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing Module
# MAGIC 
# MAGIC Production-ready data preprocessing pipeline for Third Party Capture (TPC) project.
# MAGIC This module handles:
# MAGIC - Policy data extraction from multiple sources
# MAGIC - Claims data processing with damage calculations
# MAGIC - Fraud risk indicators from referral logs
# MAGIC - Target variable creation for model training
# MAGIC - Data quality checks and validation

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

# Import required libraries
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql import Window
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Loading

# COMMAND ----------

# Load data preprocessing configuration
extract_column_transformation_lists("/config_files/data_preprocessing.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

# Get catalog information
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
catalog = workspaces.get(workspace_url, "") + catalog_prefix

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Reading Functions

# COMMAND ----------

def get_referral_vertices(df):
    """
    Process claim referral log data to create fraud risk indicators.
    
    Parameters:
    df: Spark DataFrame containing claim referral log data
    
    Returns:
    DataFrame with processed referral vertices including fraud risk indicators
    """
    logger.info("Processing claim referral log data")
    
    # Process claim referral log
    df = df.withColumn("Claim Ref", regexp_replace("Claim Ref", "\\*", "")) \
        .withColumn("siu_investigated", when(col("Source of referral") == "SIU", 1)
                                     .otherwise(0))
    
    # Create indicator fraud investigation risk
    risk_cols = {
        "Final Outcome of Claim": [
            "Withdrawn whilst investigation ongoing", 
            "Repudiated – Litigated – Claim then discontinued", 
            "Repudiated – Litigated – Success at trial", 
            "Repudiated – Not challenged"
        ],
        "Outcome of Referral": ["Accepted"],
        "Outcome of investigation": [
            "Repudiated", "Repudiated in part", "Under Investigation", 
            "Withdrawn whilst investigation ongoing"
        ]
    }
    
    # Create risk indicators for each column
    for this_col in risk_cols:
        df = df.withColumn(
            f'{this_col}_risk', 
            col(this_col).isin(*risk_cols[this_col]).cast('integer')
        ) 

    # Fill nulls with 0 for risk columns
    df = df.fillna({
        "Final Outcome of Claim_risk": 0, 
        "Outcome of Referral_risk": 0, 
        "Outcome of investigation_risk": 0
    })
    
    # Create overall fraud risk indicator
    df = df.withColumn(
        "fraud_risk", 
        greatest(
            "Final Outcome of Claim_risk", 
            "Outcome of Referral_risk", 
            "Outcome of investigation_risk"
        )
    )

    # Select and rename columns for output
    referral_vertices = df.select(
        col("Claim Ref").alias("id"), 
        "siu_investigated", 
        "fraud_risk", 
        "Final Outcome of Claim_risk", 
        "Outcome of Referral_risk", 
        "Outcome of investigation_risk",
        col("Concerns").alias("referral_concerns"),
        col("Date received").alias("transact_time"),
        col("Date received").alias("referral_log_date"),
        col("Date of Outcome").alias("referral_outcome_date")
    )
    
    return referral_vertices

def read_policy_data(policy_date=None):
    """
    Read policy data from certified tables.
    
    Parameters:
    policy_date: Optional date filter for policies
    
    Returns:
    DataFrame with policy information
    """
    logger.info("Reading policy data")
    
    # Read policy transaction data
    policy_transaction = spark.sql("""
        SELECT 
            pt.policy_transaction_id,
            pt.sales_channel, 
            pt.quote_session_id,
            pt.customer_focus_id,
            pt.customer_focus_version_id,
            pt.policy_number
        FROM prod_adp_certified.policy_motor.policy_transaction pt
    """)
    
    # Read main policy data
    policy = spark.sql(""" 
        SELECT
            p.policy_number,
            p.policy_start_date,
            p.policy_renewal_date,
            p.policy_type,
            p.policyholder_ncd_years,
            p.ncd_protected_flag,
            p.policy_number 
        FROM prod_adp_certified.policy_motor.policy p
    """)
    
    if policy_date:
        policy = policy.filter(col("policy_start_date") <= policy_date)
    
    return policy_transaction, policy

def read_vehicle_data():
    """
    Read vehicle data with overnight location information.
    
    Returns:
    DataFrame with vehicle information
    """
    logger.info("Reading vehicle data")
    
    vehicle = spark.sql(""" 
        SELECT 
            v.policy_transaction_id,
            v.vehicle_overnight_location_code as overnight_location_abi_code,
            vo.vehicle_overnight_location_id, 
            vo.vehicle_overnight_location_name, 
            v.business_mileage, 
            v.annual_mileage, 
            v.year_of_manufacture, 
            v.registration_date, 
            v.car_group, 
            v.vehicle_value, 
            v.vehicle_registration,
            v.purchase_date 
        FROM prod_adp_certified.policy_motor.vehicle v 
        LEFT JOIN prod_adp_certified.reference_motor.vehicle_overnight_location vo 
            ON v.vehicle_overnight_location_code = vo.vehicle_overnight_location_code
    """)
    
    return vehicle

def read_driver_data():
    """
    Read and transform driver data with aggregation by policy.
    
    Returns:
    DataFrame with driver information aggregated by policy_transaction_id
    """
    logger.info("Reading driver data")
    
    # Read driver data with occupation and marital status
    driver = spark.sql(""" 
        SELECT
            pd.policy_transaction_id,
            pd.first_name,
            pd.last_name, 
            pd.date_of_birth,
            pd.additional_vehicles_owned, 
            pd.age_at_policy_start_date, 
            pd.cars_in_household, 
            pd.licence_length_years, 
            pd.years_resident_in_uk,
            do.occupation_code as employment_type_abi_code,
            ms.marital_status_code,
            ms.marital_status_name
        FROM prod_adp_certified.policy_motor.driver pd
        LEFT JOIN prod_adp_certified.policy_motor.driver_occupation do
            ON pd.policy_transaction_id = do.policy_transaction_id
            AND pd.driver_index = do.driver_index
        LEFT JOIN prod_adp_certified.reference_motor.marital_status ms 
            ON pd.marital_status_code = ms.marital_status_id 
        WHERE do.occupation_index = 1
        ORDER BY pd.policy_transaction_id, pd.driver_index
    """).dropDuplicates()
    
    # Aggregate driver data by policy
    driver_transformed = driver.groupBy("policy_transaction_id").agg(
        F.collect_list("first_name").alias("first_name"),
        F.collect_list("last_name").alias("last_name"),
        F.collect_list("date_of_birth").alias("date_of_birth"),
        F.collect_list("marital_status_code").alias("marital_status_code"),
        F.collect_list("marital_status_name").alias("marital_status_name"),
        F.collect_list("additional_vehicles_owned").alias("additional_vehicles_owned"),
        F.collect_list("age_at_policy_start_date").alias("age_at_policy_start_date"),
        F.collect_list("cars_in_household").alias("cars_in_household"),
        F.collect_list("licence_length_years").alias("licence_length_years"),
        F.collect_list("years_resident_in_uk").alias("years_resident_in_uk"),
        F.collect_list("employment_type_abi_code").alias("employment_type_abi_code")
    )
    
    # Get max list size for driver arrays
    max_list_size = driver_transformed.select(
        *[F.size(F.col(col)).alias(col) for col in driver_transformed.columns if col != "policy_transaction_id"]
    ).agg(
        F.max(F.greatest(*[F.col(col) for col in driver_transformed.columns if col != "policy_transaction_id"]))
    ).collect()[0][0]
    
    # Explode lists into individual columns (support up to 5 drivers)
    columns_to_explode = [col for col in driver_transformed.columns if col != "policy_transaction_id"]
    for col in columns_to_explode:
        for i in range(min(max_list_size, 5)):
            driver_transformed = driver_transformed.withColumn(
                f"{col}_{i+1}",
                F.col(col)[i]
            )
    
    # Drop the original list columns
    driver_transformed = driver_transformed.drop(*columns_to_explode)
    
    return driver_transformed

def read_customer_data():
    """
    Read customer data from single customer view.
    
    Returns:
    DataFrame with customer information
    """
    logger.info("Reading customer data")
    
    customer = spark.sql(""" 
        SELECT 
            c.customer_focus_id,
            c.customer_focus_version_id,
            c.home_email, 
            c.postcode 
        FROM prod_adp_certified.customer_360.single_customer_view c
    """)
    
    return customer

def read_claim_data(start_date="2023-01-01", incident_date=None):
    """
    Read and process claim data for SVI (Single Vehicle Incidents).
    
    Parameters:
    start_date: Minimum date for claims
    incident_date: Optional specific incident date filter
    
    Returns:
    DataFrame with SVI claims
    """
    logger.info(f"Reading claim data from {start_date}")
    
    # Define SVI incident causes
    svi_causes = [
        'Animal', 'Attempted To Avoid Collision', 'Debris/Object', 
        'Immobile Object', 'Lost Control - No Third Party Involved'
    ]
    
    # Read SVI claims
    svi_claims_query = f"""
        SELECT 
            c.claim_id,
            c.policy_number, 
            i.reported_date,
            i.start_date as incident_date
        FROM prod_adp_certified.claim.claim_version cv
        LEFT JOIN prod_adp_certified.claim.incident i
            ON i.event_identity = cv.event_identity
        LEFT JOIN prod_adp_certified.claim.claim c
            ON cv.claim_id = c.claim_id
        WHERE incident_cause IN {tuple(svi_causes)}
        AND i.start_date >= '{start_date}'
    """
    
    svi_claims = spark.sql(svi_claims_query)
    
    if incident_date:
        svi_claims = svi_claims.filter(col("incident_date") == incident_date)
    
    return svi_claims

def get_latest_claim_version(claims_df):
    """
    Get the latest version for each claim.
    
    Parameters:
    claims_df: DataFrame with claim_id column
    
    Returns:
    DataFrame with latest claim_version_id for each claim
    """
    logger.info("Getting latest claim versions")
    
    latest_claims = spark.sql("""
        SELECT 
            MAX(claim_version_id) as claim_version_id,
            claim_id
        FROM prod_adp_certified.claim.claim_version
        GROUP BY claim_id
    """)
    
    return claims_df.join(latest_claims, ["claim_id"], "left")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Variable Creation

def create_target_variable(start_date="2023-01-01"):
    """
    Create target variable from SVI performance and claim referral log.
    
    Parameters:
    start_date: Minimum date for filtering
    
    Returns:
    DataFrame with target variables for modeling
    """
    logger.info("Creating target variable")
    
    # Read claim referral log
    clm_log_df = spark.sql(f"""
        SELECT DISTINCT * 
        FROM prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claim_referral_log
        WHERE `Date received` >= '{start_date}'
    """)
    
    # Process referral log
    clm_log = get_referral_vertices(clm_log_df).filter(
        lower(col("id")).contains("fc/")
    ).select("id", "fraud_risk")
    
    clm_log.createOrReplaceTempView("clm_log")
    
    # Read SVI performance data and create risk indicators
    target_df = spark.sql(f"""
        SELECT DISTINCT
            svi.`Claim Number` as claim_number, 
            svi.`Result of Outsourcing` as TBG_Outcome, 
            svi.`FA Outcome` as FA_Outcome,
            log.fraud_risk,
            CASE 
                WHEN lower(svi.`Result of Outsourcing`) = 'settled' THEN 0 
                WHEN lower(svi.`Result of Outsourcing`) IN ('withdrawn', 'repudiated', 'managed away', 'cancelled') THEN 1
            END AS tbg_risk,
            CASE 
                WHEN lower(svi.`FA Outcome`) IN ('claim closed', 'claim to review', 'not comprehensive cover') THEN 1 
                ELSE 0 
            END AS fa_risk
        FROM prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_performance svi
        LEFT JOIN clm_log log
            ON lower(svi.`Claim Number`) = lower(log.id)
        WHERE svi.`Notification Date` >= '{start_date}'
        AND (lower(svi.`Result of Outsourcing`) IS NULL 
             OR lower(svi.`Result of Outsourcing`) NOT IN ('ongoing - client', 'ongoing - tbg', 'pending closure'))
        AND lower(svi.`FA Outcome`) != 'not comprehensive cover'
    """)
    
    # Create combined risk indicator
    target_df = target_df.withColumn(
        "svi_risk", 
        greatest(col("fraud_risk"), col("tbg_risk"))
    ).fillna({"svi_risk": -1})
    
    return target_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Data Pipeline Functions

def prepare_policy_features(incident_date=None):
    """
    Prepare all policy-related features for a specific date.
    
    Parameters:
    incident_date: Date to filter policies (optional)
    
    Returns:
    DataFrame with consolidated policy features
    """
    logger.info("Preparing policy features")
    
    # Read all policy-related data
    policy_transaction, policy = read_policy_data(incident_date)
    vehicle = read_vehicle_data()
    driver_transformed = read_driver_data()
    customer = read_customer_data()
    
    # Read excess information
    excess = spark.sql(""" 
        SELECT 
            policy_transaction_id,
            voluntary_amount
        FROM prod_adp_certified.policy_motor.excess 
        WHERE excess_index = 0
    """)
    
    # Join all policy data
    policy_features = (
        policy
        .join(policy_transaction, ["policy_number"], "left")
        .join(vehicle, ["policy_transaction_id"], "left")
        .join(excess, ["policy_transaction_id"], "left")
        .join(driver_transformed, ["policy_transaction_id"], "left")
        .join(customer, ["customer_focus_id", "customer_focus_version_id"], "left")
    )
    
    # Add quote information for vehicle use
    quote_iteration_df = spark.table("prod_adp_certified.quote_motor.quote_iteration")
    vehicle_quote_df = spark.table("prod_adp_certified.quote_motor.vehicle").selectExpr(
        "quote_iteration_id", 
        "vehicle_use_code AS vehicle_use_quote"
    )
    
    policy_features = policy_features.join(
        quote_iteration_df.select("session_id", "quote_iteration_id"),
        policy_features.quote_session_id == quote_iteration_df.session_id, 
        "left"
    ).join(
        vehicle_quote_df, 
        "quote_iteration_id", 
        "left"
    )
    
    # Get latest transaction per policy
    window_spec = Window.partitionBy("policy_number").orderBy(col("policy_transaction_id").desc())
    policy_features = policy_features.withColumn(
        "row_num", 
        row_number().over(window_spec)
    ).filter(col("row_num") == 1).drop("row_num")
    
    # Calculate min/max features for driver columns
    driver_cols = [
        'additional_vehicles_owned', 'age_at_policy_start_date', 
        'cars_in_household', 'licence_length_years', 'years_resident_in_uk'
    ]
    
    for col_name in driver_cols:
        # Ensure all 5 driver columns exist
        for i in range(1, 6):
            if f"{col_name}_{i}" not in policy_features.columns:
                policy_features = policy_features.withColumn(f"{col_name}_{i}", lit(None))
        
        # Calculate max and min across drivers
        policy_features = policy_features.withColumn(
            f"max_{col_name}", 
            greatest(*[col(f"{col_name}_{i}") for i in range(1, 6)])
        ).withColumn(
            f"min_{col_name}", 
            least(*[col(f"{col_name}_{i}") for i in range(1, 6)])
        )
    
    return policy_features

def prepare_claims_data(target_df, start_date="2023-01-01"):
    """
    Prepare claims data with all features.
    
    Parameters:
    target_df: DataFrame with target variables
    start_date: Minimum date for claims
    
    Returns:
    DataFrame with all claim features
    """
    logger.info("Preparing claims data")
    
    # Get latest claim versions
    target_df.createOrReplaceTempView("target_df")
    
    latest_claim_version = spark.sql(f"""
        SELECT DISTINCT
            MAX(cv.claim_number) AS claim_number,
            MAX(svi.svi_risk) AS svi_risk, 
            MAX(svi.tbg_risk) AS tbg_risk, 
            MAX(svi.FA_Outcome) AS FA_Outcome, 
            MAX(svi.fa_risk) AS fa_risk,
            MAX(svi.fraud_risk) AS fraud_risk, 
            MAX(cv.claim_version_id) AS claim_version_id,
            cv.claim_id,
            MAX(cv.event_enqueued_utc_time) AS latest_event_time
        FROM target_df svi
        LEFT JOIN prod_adp_certified.claim.claim_version cv
            ON LOWER(cv.claim_number) = LOWER(svi.claim_number)
        GROUP BY cv.claim_id
        HAVING claim_number IS NOT NULL
    """)
    
    latest_claim_version.createOrReplaceTempView("latest_claim_version")
    
    # Get all claim details
    claims_df = spark.sql(f"""
        SELECT DISTINCT 
            claim.claim_version.claim_number,
            claim.claim_version.policy_number, 
            claim.claim_version.claim_version_id,
            claim.claim_version_item.claim_version_item_index, 
            claim.claim_version.policy_cover_type,
            claim.claim_version.position_status,
            claim.claim_version_item.claim_item_type, 
            claim.claim_version_item.not_on_mid, 
            claim.claim_version_item.vehicle_unattended,
            claim.claim_version_item.excesses_applied,
            claim.claim_version_item.total_loss_date, 
            claim.claim_version_item.total_loss_flag,
            claim.claim_version_item.first_party as cvi_first_party,
            claim.claimant.is_first_party,
            incident.event_identity as incident_event_identity,
            lcv.latest_event_time,
            claim.incident.start_date,
            claim.incident.reported_date,
            claim.incident.multiple_parties_involved,
            claim.incident.notification_method,
            claim.incident.impact_speed,
            claim.incident.impact_speed_unit,
            claim.incident.impact_speed_range,
            hour(claim.incident.start_date) as incidentHourC,
            dayofweek(claim.incident.start_date) as incidentDayOfWeekC,
            month(claim.incident.start_date) as incidentMonthC,
            claim.incident.incident_location_longitude,
            claim.incident.incident_type,
            claim.incident.incident_cause,
            claim.incident.incident_sub_cause,
            claim.incident.circumstances, 
            claim.vehicle.year_of_manufacture as manufacture_yr_claim,
            claim.vehicle.outstanding_finance_amount,
            claim.driver.driver_id,
            claim.driver.date_of_birth as claim_driver_dob,
            claim.claim.first_party_confirmed_tp_notified_claim,
            claim.claim_version.claim_id,
            claim.emergency_services.is_air_ambulance_attendance, 
            claim.emergency_services.is_ambulance_attendance, 
            claim.emergency_services.is_crime_reference_provided, 
            claim.emergency_services.is_fire_service_attendance, 
            claim.emergency_services.is_police_attendance,  
            claim.emergency_services.police_considering_actions, 
            claim.damage_details.assessment_category,
            claim.damage_details.boot_opens,
            claim.damage_details.doors_open,
            claim.damage_details.engine_damage,
            claim.damage_details.front_severity, 
            claim.damage_details.front_bonnet_severity, 
            claim.damage_details.front_left_severity, 
            claim.damage_details.front_right_severity, 
            claim.damage_details.left_severity, 
            claim.damage_details.left_back_seat_severity, 
            claim.damage_details.left_front_wheel_severity, 
            claim.damage_details.left_mirror_severity, 
            claim.damage_details.left_rear_wheel_severity, 
            claim.damage_details.left_underside_severity, 
            claim.damage_details.rear_severity, 
            claim.damage_details.rear_left_severity, 
            claim.damage_details.rear_right_severity, 
            claim.damage_details.rear_window_damage_severity, 
            claim.damage_details.right_severity, 
            claim.damage_details.right_back_seat_severity, 
            claim.damage_details.right_front_wheel_severity, 
            claim.damage_details.right_mirror_severity, 
            claim.damage_details.right_rear_wheel_severity, 
            claim.damage_details.right_roof_severity, 
            claim.damage_details.right_underside_severity, 
            claim.damage_details.roof_damage_severity, 
            claim.damage_details.underbody_damage_severity, 
            claim.damage_details.windscreen_damage_severity,
            lcv.tbg_risk, 
            lcv.fraud_risk, 
            lcv.svi_risk, 
            lcv.FA_Outcome, 
            lcv.fa_risk
        FROM latest_claim_version lcv
        INNER JOIN claim.claim_version
            ON lcv.claim_number = claim_version.claim_number 
        INNER JOIN claim.claim_version_item
            ON lcv.claim_version_id = claim_version.claim_version_id
            AND claim_version.claim_version_id = claim_version_item.claim_version_id
            AND lcv.claim_id = claim_version_item.claim_id
        INNER JOIN claim.claim
            ON claim.claim_id = claim_version.claim_id
            AND claim.claim_id = claim_version_item.claim_id
        LEFT JOIN claim.damage_details
            ON damage_details.event_identity = claim_version.event_identity
            AND damage_details.claim_version_item_index = claim_version_item.claim_version_item_index
        LEFT JOIN claim.incident
            ON claim_version.event_identity = incident.event_identity
        LEFT JOIN claim.vehicle
            ON claim_version.event_identity = vehicle.event_identity
            AND claim_version_item.claim_version_item_index = vehicle.claim_version_item_index
        LEFT JOIN claim.claimant
            ON claimant.claim_version_id = claim_version_item.claim_version_id
            AND claimant.claim_version_item_index = claim_version_item.claim_version_item_index
            AND claimant.event_identity = claim_version_item.event_identity
        LEFT JOIN claim.emergency_services
            ON claim.claim_version.event_identity = emergency_services.event_identity
        LEFT JOIN claim.driver
            ON claim.driver.claim_version_item_index = claim_version_item.claim_version_item_index
            AND claim.driver.event_identity = claim_version_item.event_identity
            AND claim_version.event_identity = claim.driver.event_identity
        WHERE claim_version.claim_number IS NOT NULL
        AND claim.claimant.is_first_party = true
        AND claim_version_item.claim_item_type = 'CarMotorVehicleClaimItem'
        AND claim_version_item.claim_version_item_index = 0
        AND year(incident.start_date) >= {start_date[:4]}
    """)
    
    return claims_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Functions

def validate_data_quality(df, stage="preprocessing"):
    """
    Perform data quality checks on the dataset.
    
    Parameters:
    df: DataFrame to validate
    stage: Stage name for logging
    
    Returns:
    Dict with validation results
    """
    logger.info(f"Running data quality checks for {stage}")
    
    validation_results = {
        "stage": stage,
        "total_records": df.count(),
        "columns": len(df.columns),
        "null_counts": {},
        "duplicate_count": 0
    }
    
    # Check for nulls in critical columns
    critical_columns = [
        "claim_number", "policy_number", "start_date", "reported_date"
    ]
    
    for col in critical_columns:
        if col in df.columns:
            null_count = df.filter(col(col).isNull()).count()
            validation_results["null_counts"][col] = null_count
            if null_count > 0:
                logger.warning(f"Found {null_count} null values in {col}")
    
    # Check for duplicates
    if "claim_number" in df.columns:
        duplicate_count = df.groupBy("claim_number").count().filter(col("count") > 1).count()
        validation_results["duplicate_count"] = duplicate_count
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate claim numbers")
    
    logger.info(f"Data quality check completed: {validation_results}")
    return validation_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Execution Functions

def run_daily_preprocessing(processing_date=None):
    """
    Run daily preprocessing for claims and policies.
    
    Parameters:
    processing_date: Date to process (defaults to today)
    
    Returns:
    Tuple of (policy_df, claims_df)
    """
    if processing_date is None:
        processing_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Starting daily preprocessing for {processing_date}")
    
    try:
        # Read SVI claims for the date
        svi_claims = read_claim_data(incident_date=processing_date)
        
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
        
        # Validate data quality
        validation_results = validate_data_quality(daily_data, "daily_preprocessing")
        
        # Save to table
        output_table = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.daily_policy_svi"
        daily_data.write.mode("overwrite").format("delta").option("mergeSchema", "true").saveAsTable(output_table)
        
        logger.info(f"Daily preprocessing completed. Saved {validation_results['total_records']} records to {output_table}")
        
        return daily_data
        
    except Exception as e:
        logger.error(f"Error in daily preprocessing: {str(e)}")
        raise

def run_training_preprocessing(start_date="2023-01-01"):
    """
    Run preprocessing for model training with historical data.
    
    Parameters:
    start_date: Start date for training data
    
    Returns:
    DataFrame with all features and target variables
    """
    logger.info(f"Starting training preprocessing from {start_date}")
    
    try:
        # Create target variable
        target_df = create_target_variable(start_date)
        
        # Prepare claims data
        claims_df = prepare_claims_data(target_df, start_date)
        
        # Prepare policy features  
        policy_features = prepare_policy_features()
        
        # Join claims with policies
        training_data = claims_df.join(
            policy_features,
            ["policy_number"],
            "left"
        ).filter(col("policy_transaction_id").isNotNull())
        
        # Validate data quality
        validation_results = validate_data_quality(training_data, "training_preprocessing")
        
        # Save to table
        output_table = "prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claims_pol_svi"
        training_data.write.mode("overwrite").format("delta").option("mergeSchema", "true").saveAsTable(output_table)
        
        logger.info(f"Training preprocessing completed. Saved {validation_results['total_records']} records to {output_table}")
        
        return training_data
        
    except Exception as e:
        logger.error(f"Error in training preprocessing: {str(e)}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Usage

# COMMAND ----------

# Example: Run daily preprocessing
# daily_data = run_daily_preprocessing("2025-01-15")

# Example: Run training preprocessing
# training_data = run_training_preprocessing("2023-01-01")
            "Repudiated in part", 
            "Under Investigation", 
            "Withdrawn whilst investigation ongoing"
        ]
    }
    
    for this_col in risk_cols:
        df = df.withColumn(f'{this_col}_risk', 
                          col(this_col).isin(*risk_cols[this_col]).cast('integer'))
    
    df = df.fillna({
        "Final Outcome of Claim_risk": 0, 
        "Outcome of Referral_risk": 0, 
        "Outcome of investigation_risk": 0
    })
    
    df = df.withColumn("fraud_risk", greatest(
        "Final Outcome of Claim_risk", 
        "Outcome of Referral_risk", 
        "Outcome of investigation_risk"
    ))
    
    referral_vertices = df.select(
        col("Claim Ref").alias("id"), 
        "siu_investigated", 
        "fraud_risk", 
        "Final Outcome of Claim_risk", 
        "Outcome of Referral_risk", 
        "Outcome of investigation_risk",
        col("Concerns").alias("referral_concerns"),
        col("Date received").alias("transact_time"),
        col("Date received").alias("referral_log_date"),
        col("Date of Outcome").alias("referral_outcome_date")
    )
    
    return referral_vertices

# COMMAND ----------

def get_claim_referral_log_data(spark, start_date="2023-01-01"):
    """
    Read and process claim referral log data.
    
    Parameters:
    spark: SparkSession object
    start_date: String date to filter claims from
    
    Returns:
    DataFrame with processed claim referral log
    """
    clm_log_df = spark.sql("""
        SELECT DISTINCT * 
        FROM prod_dsexp_auxiliarydata.single_vehicle_incident_checks.claim_referral_log
    """).filter(col("Date received") >= start_date)
    
    clm_log = get_referral_vertices(clm_log_df).filter(
        lower(col("id")).contains("fc/")
    ).select("id", "fraud_risk")
    
    return clm_log

# COMMAND ----------

def get_target_variable(spark):
    """
    Create target variable from SVI performance and claim referral log data.
    
    Parameters:
    spark: SparkSession object
    
    Returns:
    DataFrame with target variables (fa_risk, tbg_risk, svi_risk)
    """
    # Get data from SVI performance table
    df = spark.sql("""
        SELECT DISTINCT
        svi.`Claim Number` as claim_number, 
        svi.`Result of Outsourcing` as TBG_Outcome, 
        svi.`FA Outcome` as FA_Outcome,
        log.fraud_risk,
        
        CASE WHEN lower(svi.`Result of Outsourcing`) = 'settled' THEN 0 
            WHEN lower(svi.`Result of Outsourcing`) IN ('withdrawn', 'repudiated', 'managed away', 'cancelled') THEN 1
        END AS tbg_risk,
        CASE WHEN lower(svi.`FA Outcome`) IN ('claim closed', "claim to review", 'not comprehensive cover') THEN 1 ELSE 0 
        END AS fa_risk
        FROM prod_dsexp_auxiliarydata.single_vehicle_incident_checks.svi_performance svi
        LEFT JOIN clm_log log
        ON lower(svi.`Claim Number`) = lower(log.id)
        WHERE svi.`Notification Date` >= '2023-01-01'
        AND (lower(svi.`Result of Outsourcing`) IS NULL OR lower(svi.`Result of Outsourcing`) NOT IN ('ongoing - client', 'ongoing - tbg', 'pending closure'))
        AND lower(svi.`FA Outcome`) != 'not comprehensive cover'
    """)
    
    # Claim is high risk if flagged at either stages
    target_df = df.withColumn(
        "svi_risk", greatest(col("fraud_risk"), col("tbg_risk"))
    ).fillna({"svi_risk": -1})
    
    return target_df

# COMMAND ----------

def get_latest_claim_version(spark, target_df):
    """
    Get the latest claim version for each claim.
    
    Parameters:
    spark: SparkSession object
    target_df: DataFrame with target variables
    
    Returns:
    DataFrame with latest claim version information
    """
    spark.sql('USE CATALOG prod_adp_certified')
    target_df.createOrReplaceTempView("target_df")
    
    latest_claim_version = spark.sql("""
        SELECT DISTINCT
            MAX(cv.claim_number) AS claim_number,
            MAX(svi.svi_risk) AS svi_risk, 
            MAX(svi.tbg_risk) AS tbg_risk, 
            MAX(svi.FA_Outcome) AS FA_Outcome, 
            MAX(svi.fa_risk) AS fa_risk,
            MAX(svi.fraud_risk) AS fraud_risk, 
            MAX(cv.claim_version_id) AS claim_version_id,
            cv.claim_id,
            MAX(cv.event_enqueued_utc_time) AS latest_event_time
        FROM target_df svi
        LEFT JOIN prod_adp_certified.claim.claim_version cv
        ON LOWER(cv.claim_number) = LOWER(svi.claim_number)
        GROUP BY cv.claim_id
        HAVING claim_number IS NOT NULL
    """)
    
    return latest_claim_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preprocessing Functions

# COMMAND ----------

def do_fills_pd(raw_df, mean_dict, mean_fills, damage_cols, bool_cols, one_fills, string_cols):
    """
    Fill missing values in pandas DataFrame based on column types.
    
    Parameters:
    raw_df: Pandas DataFrame to process
    mean_dict: Dictionary of mean values for numeric columns
    mean_fills: List of columns to fill with means
    damage_cols: List of damage columns to fill with -1
    bool_cols: List of boolean columns to fill with -1
    one_fills: List of columns to fill with 1
    string_cols: List of string columns to fill with 'missing'
    
    Returns:
    Processed pandas DataFrame
    """
    # Define column groups
    cols_groups = {
        "float": mean_fills + damage_cols,
        "string": string_cols,
        "bool": bool_cols + one_fills 
    }
    
    # Create fill dictionaries
    neg_fills_dict = {x: -1 for x in bool_cols + damage_cols}
    one_fills_dict = {x: 1 for x in one_fills}
    string_fills_dict = {x: 'missing' for x in string_cols}
    combined_fills = {**one_fills_dict, **neg_fills_dict, **string_fills_dict, **mean_dict}
    
    # Fill NA values
    raw_df = raw_df.fillna(combined_fills)
    
    # Convert data types
    for dtype, column_list in cols_groups.items():
        if dtype == "float":
            raw_df[column_list] = raw_df[column_list].astype(float)
        elif dtype == "integer":
            raw_df[column_list] = raw_df[column_list].astype(int)
        elif dtype == "string":
            raw_df[column_list] = raw_df[column_list].astype('str')        
        elif dtype == "bool":
            raw_df[column_list] = raw_df[column_list].astype(int).astype('str')
    
    return raw_df

# COMMAND ----------

def split_train_test_data(df, test_size=0.3, random_state=42):
    """
    Split data into train and test sets with stratification.
    
    Parameters:
    df: Spark DataFrame with data
    test_size: Proportion of data for test set
    random_state: Random seed for reproducibility
    
    Returns:
    Spark DataFrame with 'dataset' column indicating train/test
    """
    # Convert to pandas for splitting
    df_pd = df.coalesce(1).toPandas()
    
    # Split data
    train_df, test_df = train_test_split(
        df_pd, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df_pd.svi_risk
    )
    
    # Add dataset indicator
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    
    # Combine and convert back to Spark
    combined_df_pd = pd.concat([test_df, train_df])
    
    return spark.createDataFrame(combined_df_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Functions

# COMMAND ----------

def get_info(df):
    """
    Get comprehensive information about a pandas DataFrame.
    
    Parameters:
    df: Pandas DataFrame
    
    Returns:
    DataFrame with column statistics
    """
    return pd.DataFrame({
        "name": df.columns,
        "% nulls": 100 * df.isnull().sum().values / len(df),
        "type": df.dtypes.astype(str),
        "nulls": df.isnull().sum().values,     
        "non-nulls": len(df) - df.isnull().sum().values,
        "min": df.apply(pd.to_numeric, args=['coerce']).min(axis=0).astype(float),
        "max": df.apply(pd.to_numeric, args=['coerce']).max(axis=0).astype(float),
        "median": df.apply(pd.to_numeric, args=['coerce']).median(axis=0),        
        "mean": df.apply(pd.to_numeric, args=['coerce']).mean(axis=0),        
        "std": df.apply(pd.to_numeric, args=['coerce']).std(axis=0),        
        "mode": df.mode(axis=0).iloc[0,:].astype(str),  
        "unique": df.nunique(axis=0)      
    })

# COMMAND ----------

def check_null_percentages(df):
    """
    Calculate percentage of nulls and empty values for each column.
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    Pandas DataFrame with null percentages
    """
    null_counts = df.select(
        [(count(when(col(c).isNull() | (col(c) == ""), c)) / count("*")).alias(c) 
         for c in df.columns]
    )
    
    null_counts_pd = null_counts.toPandas() * 100
    return null_counts_pd.T.reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Column Lists for Processing

# COMMAND ----------

# Define column lists for different processing types
def get_column_lists():
    """
    Get predefined column lists for different processing types.
    
    Returns:
    Dictionary containing various column lists
    """
    return {
        "mean_fills": [
            "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", 
            "veh_age", "business_mileage", "annual_mileage", "incidentHourC", 
            "additional_vehicles_owned_1", "age_at_policy_start_date_1", 
            "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", 
            "max_additional_vehicles_owned", "min_additional_vehicles_owned", 
            "max_age_at_policy_start_date", "min_age_at_policy_start_date", 
            "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", 
            "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", 
            "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", 
            "outstanding_finance_amount", "claim_to_policy_end", "incidentDayOfWeekC"
        ],
        
        "damage_cols": [
            "damageScore", "areasDamagedMinimal", "areasDamagedMedium", 
            "areasDamagedHeavy", "areasDamagedSevere", "areasDamagedTotal"
        ],
        
        "bool_cols": [
            "vehicle_unattended", "excesses_applied", "is_first_party", 
            "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", 
            "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance", 
            "veh_age_more_than_10", "police_considering_actions", "is_crime_reference_provided", 
            "ncd_protected_flag", "boot_opens", "doors_open", "multiple_parties_involved", 
            "is_incident_weekend", "is_reported_monday", "driver_age_low_1", 
            "claim_driver_age_low", "licence_low_1"
        ],
        
        "one_fills": [
            "C1_fri_sat_night", "C2_reporting_delay", "C3_weekend_incident_reported_monday", 
            "C5_is_night_incident", "C6_no_commuting_but_rush_hour", 
            "C7_police_attended_or_crime_reference", "C9_policy_within_30_days", 
            "C10_claim_to_policy_end", "C11_young_or_inexperienced", 
            "C12_expensive_for_driver_age", "C14_contains_watchwords"
        ],
        
        "string_cols": [
            'car_group', 'vehicle_overnight_location_id', 'incidentMonthC', 
            'employment_type_abi_code_5', 'employment_type_abi_code_4', 
            'employment_type_abi_code_3', 'employment_type_abi_code_2', 
            'policy_type', 'postcode', 'assessment_category', 'engine_damage', 
            'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_name', 
            'policy_cover_type', 'notification_method', 'impact_speed_unit', 
            'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause', 
            'front_severity', 'front_bonnet_severity', 'front_left_severity', 
            'front_right_severity', 'left_severity', 'left_back_seat_severity', 
            'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
            'left_underside_severity', 'rear_severity', 'rear_left_severity', 
            'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 
            'right_back_seat_severity', 'right_front_wheel_severity', 'right_mirror_severity', 
            'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 
            'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity', 
            'employment_type_abi_code_1', 'incident_day_of_week', 'reported_day_of_week'
        ],
        
        "damage_columns": [
            'front_severity', 'front_bonnet_severity', 'front_left_severity', 
            'front_right_severity', 'left_severity', 'left_back_seat_severity', 
            'left_front_wheel_severity', 'left_mirror_severity', 'left_rear_wheel_severity', 
            'left_underside_severity', 'rear_severity', 'rear_left_severity', 
            'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 
            'right_back_seat_severity', 'right_front_wheel_severity', 'right_mirror_severity', 
            'right_rear_wheel_severity', 'right_roof_severity', 'right_underside_severity', 
            'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity'
        ]
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## The End