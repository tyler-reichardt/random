# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Module
# MAGIC 
# MAGIC Production-ready feature engineering pipeline for Third Party Capture (TPC) project.
# MAGIC This module handles:
# MAGIC - Damage score calculations and severity aggregations
# MAGIC - Business rule check variables (C1-C14) for fraud detection
# MAGIC - Time-based and demographic feature engineering
# MAGIC - Driver feature aggregations
# MAGIC - Missing value imputation strategies

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

# Import required libraries
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, FloatType
from pyspark.sql import Window
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Loading

# COMMAND ----------

# Load feature engineering configuration
extract_column_transformation_lists("/config_files/feature_engineering.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Damage Score Calculation

# COMMAND ----------

def calculate_damage_score(*args):
    """
    Calculate damage score based on severity levels.
    Uses multiplicative scoring: Minimal=2, Medium=3, Heavy=4, Severe=5
    
    Parameters:
    *args: Variable number of damage severity values
    
    Returns:
    Tuple containing (damageScore, areasDamagedMinimal, areasDamagedMedium, 
                      areasDamagedHeavy, areasDamagedSevere)
    """
    damageScore = 1
    areasDamagedMinimal = 0
    areasDamagedMedium = 0
    areasDamagedHeavy = 0
    areasDamagedSevere = 0
    
    for damage in args:
        if damage == 'Minimal':
            damageScore *= 2
            areasDamagedMinimal += 1
        elif damage == 'Medium':
            damageScore *= 3
            areasDamagedMedium += 1
        elif damage == 'Heavy':
            damageScore *= 4
            areasDamagedHeavy += 1
        elif damage == 'Severe':
            damageScore *= 5
            areasDamagedSevere += 1
    
    return damageScore, areasDamagedMinimal, areasDamagedMedium, areasDamagedHeavy, areasDamagedSevere

# Register the UDF for Spark
calculate_damage_score_udf = udf(calculate_damage_score, StructType([
    StructField("damageScore", IntegerType(), False),
    StructField("areasDamagedMinimal", IntegerType(), False),
    StructField("areasDamagedMedium", IntegerType(), False),
    StructField("areasDamagedHeavy", IntegerType(), False),
    StructField("areasDamagedSevere", IntegerType(), False)
]))

def apply_damage_calculations(df):
    """
    Apply damage score calculations to claims data.
    
    Parameters:
    df: Spark DataFrame with damage severity columns
    
    Returns:
    DataFrame with damage scores and area counts added
    """
    logger.info("Calculating damage scores")
    
    # List of damage columns
    damage_columns = [
        'front_severity', 'front_bonnet_severity', 'front_left_severity', 'front_right_severity', 
        'left_severity', 'left_back_seat_severity', 'left_front_wheel_severity', 'left_mirror_severity', 
        'left_rear_wheel_severity', 'left_underside_severity', 'rear_severity', 'rear_left_severity', 
        'rear_right_severity', 'rear_window_damage_severity', 'right_severity', 'right_back_seat_severity', 
        'right_front_wheel_severity', 'right_mirror_severity', 'right_rear_wheel_severity', 'right_roof_severity', 
        'right_underside_severity', 'roof_damage_severity', 'underbody_damage_severity', 'windscreen_damage_severity'
    ]
    
    # Apply the UDF to calculate damage scores
    df = df.withColumn(
        "damage_scores",
        calculate_damage_score_udf(*[df[col] for col in damage_columns])
    )
    
    # Split the struct column into separate columns
    df = df.select(
        "*",
        "damage_scores.damageScore",
        "damage_scores.areasDamagedMinimal",
        "damage_scores.areasDamagedMedium",
        "damage_scores.areasDamagedHeavy",
        "damage_scores.areasDamagedSevere"
    ).drop("damage_scores")
    
    # Add total damaged areas and relative damage score
    df = df.withColumn(
        "areasDamagedTotal", 
        col("areasDamagedMinimal") + col("areasDamagedMedium") + 
        col("areasDamagedSevere") + col("areasDamagedHeavy")
    ).withColumn(
        "areasDamagedRelative",
        col("areasDamagedMinimal") + 2*col("areasDamagedMedium") + 
        3*col("areasDamagedSevere") + 4*col("areasDamagedHeavy")
    )
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Variables Generation (Business Rules)

# COMMAND ----------

def generate_check_variables(df):
    """
    Generate fraud detection check variables based on business rules.
    These checks flag potentially suspicious patterns in claims.
    
    Parameters:
    df: Spark DataFrame with claim and policy data
    
    Returns:
    DataFrame with check variables (C1-C14) added
    """
    logger.info("Generating check variables")
    
    # C1: Was the incident on a Friday/Saturday NIGHT?
    df = df.withColumn("incident_day_of_week", date_format(col("start_date"), "E"))
    
    fri_sat_night = (
        (col("incident_day_of_week").isin("Fri", "Sat") & (hour(col("start_date")).between(20, 23))) | 
        (col("incident_day_of_week").isin("Sat", "Sun") & (hour(col("start_date")).between(0, 4)))
    )
    
    df = df.withColumn(
        "C1_fri_sat_night",
        when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0)
    )
    
    # C2: Was there a delay in notifying us of the incident?
    df = df.withColumn("delay_in_reporting", datediff(col("reported_date"), col("start_date")))
    df = df.withColumn(
        "C2_reporting_delay", 
        when(col("delay_in_reporting") >= 3, 1)
        .when(col("delay_in_reporting").isNull(), 1)
        .otherwise(0)
    )
    
    # Weekend/Monday reporting pattern
    df = df.withColumn(
        "is_incident_weekend",
        when(date_format(col("start_date"), "E").isin("Fri", "Sat", "Sun"), 1).otherwise(0)
    )
    df = df.withColumn("reported_day_of_week", date_format(col("reported_date"), "E"))
    df = df.withColumn(
        "is_reported_monday",
        when(col("reported_day_of_week") == "Mon", 1).otherwise(0)
    )
    
    # C3: Cases taking place over weekend but reported on Monday
    df = df.withColumn(
        "C3_weekend_incident_reported_monday",
        when((col("is_incident_weekend") == 1) & (col("is_reported_monday") == 1), 1).otherwise(0)
    )
    
    # C5: Incident between 11pm and 5am
    df = df.withColumn(
        "C5_is_night_incident",
        when((hour(col("start_date")) >= 23) | (hour(col("start_date")) <= 5) | 
             (hour(col("start_date"))).isNull(), 1).otherwise(0)
    )
    
    # C6: No commuting on policy but traveling during rush hours
    not_commuting_rush = (
        (col("vehicle_use_quote") == 1) & 
        ((hour(col("start_date")).between(6, 10)) | (hour(col("start_date")).between(15, 18)))
    )
    
    df = df.withColumn(
        "C6_no_commuting_but_rush_hour",
        when(not_commuting_rush, 1)
        .when(not_commuting_rush.isNull(), 1)
        .otherwise(0)
    )
    
    # C7: Police attendance or crime reference provided
    df = df.withColumn(
        "C7_police_attended_or_crime_reference",
        when((col("is_police_attendance") == True) | 
             (col("is_crime_reference_provided") == True), 1).otherwise(0)
    )
    
    # C9: Policy incepted within 30 days of incident
    df = df.withColumn(
        "inception_to_claim", 
        datediff(to_date(col("start_date")), to_date(col("policy_start_date")))
    )
    
    df = df.withColumn(
        "C9_policy_within_30_days",
        when(col("inception_to_claim").between(0, 30), 1)
        .when(col("inception_to_claim").isNull(), 1)
        .otherwise(0)
    )
    
    # C10: Policy ends within 2 months of incident
    df = df.withColumn(
        "claim_to_policy_end", 
        datediff(to_date(col("policy_renewal_date")), to_date(col("start_date")))
    )
    
    df = df.withColumn(
        "C10_claim_to_policy_end",
        when(col("claim_to_policy_end") < 60, 1)
        .when(col("claim_to_policy_end").isNull(), 1)
        .otherwise(0)
    )
    
    # Age and experience checks
    df = df.withColumn(
        "driver_age_low_1", 
        when(col("age_at_policy_start_date_1") < 25, 1)
        .when(col("age_at_policy_start_date_1").isNull(), 1)
        .otherwise(0)
    )
    
    df = df.withColumn(
        "claim_driver_age_low", 
        when(col("min_claim_driver_age") < 25, 1)
        .when(col("min_claim_driver_age").isNull(), 1)
        .otherwise(0)
    )
    
    df = df.withColumn(
        "licence_low_1", 
        when(col("licence_length_years_1") <= 3, 1).otherwise(0)
    )
    
    # C11: Young or inexperienced driver
    condition_inexperienced = (col("driver_age_low_1") == 1) | (col("licence_low_1") == 1)
    df = df.withColumn(
        "C11_young_or_inexperienced", 
        when(condition_inexperienced, 1)
        .when(condition_inexperienced.isNull(), 1)
        .otherwise(0)
    )
    
    # C12: Expensive car for driver age
    condition_expensive_car = (
        ((col("age_at_policy_start_date_1") < 25) & (col("vehicle_value") >= 20000)) | 
        ((col("age_at_policy_start_date_1") >= 25) & (col("vehicle_value") >= 30000))
    )
    
    df = df.withColumn(
        "C12_expensive_for_driver_age", 
        when(condition_expensive_car, 1)
        .when(condition_expensive_car.isNull(), 1)
        .otherwise(0)
    )
    
    # C14: Circumstances contain watch words
    watch_words = "|".join([
        "commut", "deliver", "parcel", "drink", "police", "custody", "arrest", 
        "alcohol", "drug", "station", "custody"
    ])
    
    df = df.withColumn(
        "C14_contains_watchwords",
        when(lower(col("Circumstances")).rlike(watch_words), 1)
        .when(col("Circumstances").isNull(), 1)
        .otherwise(0)
    )
    
    # Aggregate check information
    check_cols = [
        'C1_fri_sat_night', 'C2_reporting_delay', 'C3_weekend_incident_reported_monday',
        'C5_is_night_incident', 'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference',
        'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced',
        'C12_expensive_for_driver_age', 'C14_contains_watchwords'
    ]
    
    # Create list of failed checks
    df = df.withColumn(
        "checks_list",
        array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in check_cols])
    )
    
    df = df.withColumn(
        "checks_list",
        expr("filter(checks_list, x -> x is not null)")
    ).withColumn(
        "num_failed_checks", 
        size(col("checks_list"))
    ).withColumn(
        "checks_max",
        greatest(*[col(c) for c in check_cols])
    )
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Feature Engineering

# COMMAND ----------

def engineer_temporal_features(df):
    """
    Create time-based features from dates and timestamps.
    
    Parameters:
    df: Spark DataFrame with date columns
    
    Returns:
    DataFrame with temporal features added
    """
    logger.info("Engineering temporal features")
    
    # Vehicle age calculation
    df = df.withColumn(
        "veh_age", 
        round(
            datediff(col("start_date"), to_date(concat(col("manufacture_yr_claim"), lit('-01-01')))) / 365.25, 
            0
        )
    )
    
    df = df.withColumn(
        "veh_age_more_than_10", 
        (col("veh_age") > 10).cast("int")
    )
    
    # Driver age at claim
    df = df.withColumn(
        "claim_driver_age",
        round(datediff(col("start_date"), to_date(col("claim_driver_dob"))) / 365.25)
    )
    
    return df

def aggregate_driver_features(df):
    """
    Handle driver-level aggregations and deduplication.
    
    Parameters:
    df: Spark DataFrame with driver information
    
    Returns:
    DataFrame with deduplicated driver features
    """
    logger.info("Aggregating driver features")
    
    # Get minimum driver age for each claim
    if "claim_driver_age" in df.columns:
        min_drv_age = df.groupBy("claim_number").agg(
            min(col("claim_driver_age")).alias("min_claim_driver_age")
        )
        
        # Join back and remove duplicates
        df = df.drop("claim_driver_age").join(
            min_drv_age, 
            on="claim_number", 
            how="left"
        ).drop("driver_id", "claim_driver_dob").dropDuplicates()
    
    return df

def prepare_feature_types(df):
    """
    Convert columns to appropriate data types for modeling.
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    DataFrame with corrected data types
    """
    logger.info("Preparing feature data types")
    
    # Boolean columns to convert to integer
    boolean_columns = [
        "vehicle_unattended", "excesses_applied", "is_first_party", 
        "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", 
        "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance"
    ]
    
    for col_name in boolean_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("integer"))
    
    # Fix decimal type columns
    decimal_cols = ['outstanding_finance_amount', 'vehicle_value', 'voluntary_amount']
    for col_name in decimal_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("float"))
    
    # Fix specific boolean type issues
    bool_fix_cols = ['police_considering_actions', 'is_crime_reference_provided', 
                     'multiple_parties_involved', 'total_loss_flag']
    for col_name in bool_fix_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast('boolean'))
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing Value Handling

# COMMAND ----------

def get_imputation_strategies():
    """
    Define imputation strategies for different feature types.
    
    Returns:
    Dict with imputation strategies by feature type
    """
    # Features to fill with mean
    mean_fills = [
        "policyholder_ncd_years", "inception_to_claim", "min_claim_driver_age", 
        "veh_age", "business_mileage", "annual_mileage", "incidentHourC", 
        "additional_vehicles_owned_1", "age_at_policy_start_date_1", 
        "cars_in_household_1", "licence_length_years_1", "years_resident_in_uk_1", 
        "max_additional_vehicles_owned", "min_additional_vehicles_owned", 
        "max_age_at_policy_start_date", "min_age_at_policy_start_date", 
        "max_cars_in_household", "min_cars_in_household", "max_licence_length_years", 
        "min_licence_length_years", "max_years_resident_in_uk", "min_years_resident_in_uk", 
        "impact_speed", "voluntary_amount", "vehicle_value", "manufacture_yr_claim", 
        "outstanding_finance_amount", "claim_to_policy_end", "incidentDayOfWeekC", 
        "num_failed_checks"
    ]
    
    # Boolean or damage columns with -1 fills
    neg_fills = [
        "vehicle_unattended", "excesses_applied", "is_first_party", 
        "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", 
        "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance", 
        "veh_age_more_than_10", "damageScore", "areasDamagedMinimal", "areasDamagedMedium", 
        "areasDamagedHeavy", "areasDamagedSevere", "areasDamagedTotal", 
        "police_considering_actions", "is_crime_reference_provided", "ncd_protected_flag", 
        "boot_opens", "doors_open", "multiple_parties_involved", "is_incident_weekend", 
        "is_reported_monday", "driver_age_low_1", "claim_driver_age_low", "licence_low_1", 
        "total_loss_flag"
    ]
    
    # Check variables - fill with 1 (to trigger manual review)
    one_fills = [
        "C1_fri_sat_night", "C2_reporting_delay", "C3_weekend_incident_reported_monday", 
        "C5_is_night_incident", "C6_no_commuting_but_rush_hour", 
        "C7_police_attended_or_crime_reference", "C9_policy_within_30_days", 
        "C10_claim_to_policy_end", "C11_young_or_inexperienced", 
        "C12_expensive_for_driver_age", "C14_contains_watchwords"
    ]
    
    # Categorical columns - fill with 'missing'
    string_fills = [
        'car_group', 'vehicle_overnight_location_id', 'incidentMonthC', 
        'policy_type', 'postcode', 'assessment_category', 'engine_damage', 
        'sales_channel', 'overnight_location_abi_code', 'vehicle_overnight_location_name', 
        'policy_cover_type', 'notification_method', 'impact_speed_unit', 
        'impact_speed_range', 'incident_type', 'incident_cause', 'incident_sub_cause',
        'incident_day_of_week', 'reported_day_of_week'
    ] + [
        f'{severity}_severity' for severity in [
            'front', 'front_bonnet', 'front_left', 'front_right', 'left', 
            'left_back_seat', 'left_front_wheel', 'left_mirror', 'left_rear_wheel', 
            'left_underside', 'rear', 'rear_left', 'rear_right', 'rear_window_damage', 
            'right', 'right_back_seat', 'right_front_wheel', 'right_mirror', 
            'right_rear_wheel', 'right_roof', 'right_underside', 'roof_damage', 
            'underbody_damage', 'windscreen_damage'
        ]
    ]
    
    # Pre-calculated means for production use
    mean_dict = {
        'policyholder_ncd_years': 6.7899, 'inception_to_claim': 141.2893, 
        'min_claim_driver_age': 37.5581, 'veh_age': 11.3038, 
        'business_mileage': 306.2093, 'annual_mileage': 7372.2649, 
        'incidentHourC': 12.8702, 'additional_vehicles_owned_1': 0.0022, 
        'age_at_policy_start_date_1': 39.4507, 'cars_in_household_1': 1.8289, 
        'licence_length_years_1': 15.3764, 'years_resident_in_uk_1': 34.6192, 
        'max_additional_vehicles_owned': 0.003, 'min_additional_vehicles_owned': 0.0013, 
        'max_age_at_policy_start_date': 43.1786, 'min_age_at_policy_start_date': 35.4692, 
        'max_cars_in_household': 1.8861, 'min_cars_in_household': 1.7626, 
        'max_licence_length_years': 18.3106, 'min_licence_length_years': 12.2208, 
        'max_years_resident_in_uk': 38.5058, 'min_years_resident_in_uk': 30.5888, 
        'impact_speed': 27.1128, 'voluntary_amount': 241.3595, 
        'vehicle_value': 7861.6867, 'manufacture_yr_claim': 2011.9375, 
        'outstanding_finance_amount': 0.0, 'claim_to_policy_end': 83.4337, 
        'incidentDayOfWeekC': 4.0115, 'num_failed_checks': 0.0
    }
    
    return {
        "mean_fills": mean_fills,
        "neg_fills": neg_fills,
        "one_fills": one_fills,
        "string_fills": string_fills,
        "mean_dict": mean_dict
    }

def apply_missing_value_imputation(df):
    """
    Apply missing value imputation based on feature type.
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    DataFrame with missing values imputed
    """
    logger.info("Applying missing value imputation")
    
    strategies = get_imputation_strategies()
    
    # Apply mean imputation
    for col_name, mean_val in strategies["mean_dict"].items():
        if col_name in df.columns:
            df = df.fillna({col_name: mean_val})
    
    # Apply -1 imputation for boolean/damage columns
    neg_fill_dict = {col: -1 for col in strategies["neg_fills"] if col in df.columns}
    df = df.fillna(neg_fill_dict)
    
    # Apply 1 imputation for check columns
    one_fill_dict = {col: 1 for col in strategies["one_fills"] if col in df.columns}
    df = df.fillna(one_fill_dict)
    
    # Apply 'missing' imputation for string columns
    string_fill_dict = {col: 'missing' for col in strategies["string_fills"] if col in df.columns}
    df = df.fillna(string_fill_dict)
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Feature Engineering Pipeline

# COMMAND ----------

def run_feature_engineering(df, stage="training"):
    """
    Run complete feature engineering pipeline.
    
    Parameters:
    df: Spark DataFrame with preprocessed data
    stage: Either "training" or "scoring" to adjust processing
    
    Returns:
    DataFrame with all engineered features
    """
    logger.info(f"Running feature engineering for {stage}")
    
    try:
        # Apply damage calculations
        df = apply_damage_calculations(df)
        
        # Generate check variables
        df = generate_check_variables(df)
        
        # Engineer temporal features
        df = engineer_temporal_features(df)
        
        # Aggregate driver features
        df = aggregate_driver_features(df)
        
        # Prepare data types
        df = prepare_feature_types(df)
        
        # Apply missing value imputation
        df = apply_missing_value_imputation(df)
        
        # Add dataset indicator for training
        if stage == "training":
            df = df.withColumn("dataset", lit("train"))
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def get_model_features():
    """
    Get lists of features for different models.
    
    Returns:
    Dict with feature lists for FA and Interview models
    """
    # Numeric features for all models
    numeric_features = [
        'policyholder_ncd_years', 'inception_to_claim', 'min_claim_driver_age', 
        'veh_age', 'business_mileage', 'annual_mileage', 'incidentHourC', 
        'additional_vehicles_owned_1', 'age_at_policy_start_date_1', 
        'cars_in_household_1', 'licence_length_years_1', 'years_resident_in_uk_1', 
        'max_additional_vehicles_owned', 'min_additional_vehicles_owned', 
        'max_age_at_policy_start_date', 'min_age_at_policy_start_date', 
        'max_cars_in_household', 'min_cars_in_household', 'max_licence_length_years', 
        'min_licence_length_years', 'max_years_resident_in_uk', 'min_years_resident_in_uk', 
        'impact_speed', 'voluntary_amount', 'vehicle_value', 'manufacture_yr_claim', 
        'outstanding_finance_amount', 'claim_to_policy_end', 'incidentDayOfWeekC', 
        'damageScore', 'areasDamagedMinimal', 'areasDamagedMedium', 
        'areasDamagedHeavy', 'areasDamagedSevere', 'areasDamagedTotal'
    ]
    
    # Categorical features for all models
    categorical_features = [
        'vehicle_unattended', 'excesses_applied', 'is_first_party', 
        'first_party_confirmed_tp_notified_claim', 'is_air_ambulance_attendance', 
        'is_ambulance_attendance', 'is_fire_service_attendance', 'is_police_attendance', 
        'veh_age_more_than_10', 'police_considering_actions', 'is_crime_reference_provided', 
        'ncd_protected_flag', 'boot_opens', 'doors_open', 'multiple_parties_involved', 
        'is_incident_weekend', 'is_reported_monday', 'driver_age_low_1', 
        'claim_driver_age_low', 'licence_low_1', 'C1_fri_sat_night', 
        'C2_reporting_delay', 'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 
        'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 
        'C9_policy_within_30_days', 'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 
        'C12_expensive_for_driver_age', 'C14_contains_watchwords', 
        'vehicle_overnight_location_id', 'incidentMonthC', 'policy_type', 
        'assessment_category', 'engine_damage', 'sales_channel', 
        'overnight_location_abi_code', 'vehicle_overnight_location_name', 
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
        'incident_day_of_week', 'reported_day_of_week', 'checks_max'
    ]
    
    # Interview model specific features
    num_interview = [
        'voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 
        'annual_mileage', 'min_claim_driver_age', 'incidentHourC', 
        'areasDamagedHeavy', 'impact_speed', 'years_resident_in_uk_1', 
        'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 
        'claim_to_policy_end', 'veh_age', 'licence_length_years_1', 
        'num_failed_checks', 'areasDamagedMedium', 'min_years_resident_in_uk', 
        'incidentDayOfWeekC', 'age_at_policy_start_date_1', 'max_age_at_policy_start_date'
    ]
    
    cat_interview = [
        'assessment_category', 'left_severity', 'C9_policy_within_30_days',
        'incident_sub_cause', 'rear_right_severity', 'front_left_severity',
        'rear_window_damage_severity', 'incident_cause', 'checks_max',
        'total_loss_flag'
    ]
    
    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": numeric_features + categorical_features,
        "num_interview": num_interview,
        "cat_interview": cat_interview,
        "interview_features": num_interview + cat_interview
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Usage

# COMMAND ----------

# Example: Run feature engineering on preprocessed data
# from DataPreprocessing import run_daily_preprocessing
# 
# # Get preprocessed data
# preprocessed_df = run_daily_preprocessing("2025-01-15")
# 
# # Apply feature engineering
# features_df = run_feature_engineering(preprocessed_df, stage="scoring")
# 
# # Get feature lists for modeling
# feature_lists = get_model_features()
        elif damage == 'Severe':
            damageScore *= 5
            areasDamagedSevere += 1
    
    return damageScore, areasDamagedMinimal, areasDamagedMedium, areasDamagedHeavy, areasDamagedSevere

# Register the UDF
calculate_damage_score_udf = udf(calculate_damage_score, StructType([
    StructField("damageScore", IntegerType(), False),
    StructField("areasDamagedMinimal", IntegerType(), False),
    StructField("areasDamagedMedium", IntegerType(), False),
    StructField("areasDamagedHeavy", IntegerType(), False),
    StructField("areasDamagedSevere", IntegerType(), False)
]))

# COMMAND ----------

def apply_damage_scores(df, damage_columns):
    """
    Apply damage score calculation to dataframe.
    
    Parameters:
    df: Spark DataFrame
    damage_columns: List of damage column names
    
    Returns:
    DataFrame with damage scores added
    """
    # Apply the UDF to the DataFrame
    df = df.withColumn(
        "damage_scores",
        calculate_damage_score_udf(*[df[col] for col in damage_columns])
    )
    
    # Split the struct column into separate columns
    df = df.select(
        "*",
        "damage_scores.damageScore",
        "damage_scores.areasDamagedMinimal",
        "damage_scores.areasDamagedMedium",
        "damage_scores.areasDamagedHeavy",
        "damage_scores.areasDamagedSevere"
    ).withColumn(
        "areasDamagedTotal", 
        col("areasDamagedMinimal") + col("areasDamagedMedium") + 
        col("areasDamagedSevere") + col("areasDamagedHeavy")
    ).drop("damage_scores")
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Variables Generation

# COMMAND ----------

def generate_check_variables(df):
    """
    Generate all check variables (C1-C14) for fraud detection.
    
    Parameters:
    df: Spark DataFrame with claim data
    
    Returns:
    DataFrame with check variables added
    """
    # C1: Was the incident on a Friday/Saturday NIGHT?
    df = df.withColumn("incident_day_of_week", date_format(col("latest_event_time"), "E"))
    
    fri_sat_night = (
        (col("incident_day_of_week").isin("Fri", "Sat") & 
         (hour(col("start_date")).between(20, 23))) | 
        (col("incident_day_of_week").isin("Sat", "Sun") & 
         (hour(col("start_date")).between(0, 4)))
    )
    
    df = df.withColumn(
        "C1_fri_sat_night",
        when(fri_sat_night, 1).when(fri_sat_night.isNull(), 1).otherwise(0)
    )
    
    df = df.withColumn("reported_day_of_week", date_format(col("latest_event_time"), "E"))
    
    # C2: Was there a delay in notifying us of the incident without reason?
    df = df.withColumn("delay_in_reporting", datediff(col("reported_date"), col("start_date")))
    df = df.withColumn(
        "C2_reporting_delay", 
        when(col("delay_in_reporting") >= 3, 1)
        .when(col("delay_in_reporting").isNull(), 1)
        .otherwise(0)
    )
    
    # Add weekend and Monday indicators
    df = df.withColumn(
        "is_incident_weekend",
        when(date_format(col("start_date"), "E").isin("Fri", "Sat", "Sun"), 1).otherwise(0)
    )
    
    df = df.withColumn(
        "is_reported_monday",
        when(date_format(col("reported_date"), "E") == "Mon", 1).otherwise(0)
    )
    
    # C3: Cases taking place over a weekend but not being reported until Monday
    df = df.withColumn(
        "C3_weekend_incident_reported_monday",
        when((col("is_incident_weekend") == True) & (col("is_reported_monday") == True), 1)
        .otherwise(0)
    )
    
    # C5: Incident between 11pm and 5am
    df = df.withColumn(
        "C5_is_night_incident",
        when((hour(col("start_date")) >= 23) | 
             (hour(col("start_date")) <= 5) | 
             (hour(col("start_date"))).isNull(), 1)
        .otherwise(0)
    )
    
    # C6: No commuting on policy and customer travelling between rush hours
    not_commuting_rush = (
        (lower(col("vehicle_use_quote")) == 1) & 
        ((hour(col("start_date")).between(6, 10)) | 
         (hour(col("start_date")).between(15, 18)))
    )
    
    df = df.withColumn(
        "C6_no_commuting_but_rush_hour",
        when(not_commuting_rush, 1)
        .when(not_commuting_rush.isNull(), 1)
        .otherwise(0)
    )
    
    # C7: Police attendance or crime reference provided
    df = df.withColumn(
        "C7_police_attended_or_crime_reference",
        when((col("is_police_attendance") == True) | 
             (col("is_crime_reference_provided") == True), 1)
        .otherwise(0)
    )
    
    # C9: Was the policy incepted within 30 days of the incident date?
    df = df.withColumn(
        "inception_to_claim", 
        datediff(to_date(col("start_date")), to_date(col("policy_start_date")))
    )
    
    df = df.withColumn(
        "C9_policy_within_30_days",
        when(col("inception_to_claim").between(0, 30), 1)
        .when(col("inception_to_claim").isNull(), 1)
        .otherwise(0)
    )
    
    # C10: Does the policy end within 1 or 2 months of the incident date?
    df = df.withColumn(
        "claim_to_policy_end", 
        datediff(to_date(col("policy_renewal_date")), to_date(col("start_date")))
    )
    
    df = df.withColumn(
        "C10_claim_to_policy_end",
        when(col("claim_to_policy_end") < 60, 1)
        .when(col("claim_to_policy_end").isNull(), 1)
        .otherwise(0)
    )
    
    # Driver age indicators
    df = df.withColumn(
        "driver_age_low_1", 
        when(col("age_at_policy_start_date_1") < 25, 1)
        .when(col("age_at_policy_start_date_1").isNull(), 1)
        .otherwise(0)
    )
    
    df = df.withColumn(
        "claim_driver_age_low", 
        when(col("min_claim_driver_age") < 25, 1)
        .when(col("min_claim_driver_age").isNull(), 1)
        .otherwise(0)
    )
    
    # Licence check
    df = df.withColumn(
        "licence_low_1", 
        when(col("licence_length_years_1") <= 3, 1).otherwise(0)
    )
    
    # C11: Young or inexperienced driver
    condition_inexperienced = (col("driver_age_low_1") == 1) | (col("licence_low_1") == 1)
    df = df.withColumn(
        "C11_young_or_inexperienced", 
        when(condition_inexperienced, 1)
        .when(condition_inexperienced.isNull(), 1)
        .otherwise(0)
    )
    
    # C12: Expensive car for driver age
    condition_expensive_car = (
        ((col("age_at_policy_start_date_1") < 25) & (col("vehicle_value") >= 20000)) | 
        ((col("age_at_policy_start_date_1") >= 25) & (col("vehicle_value") >= 30000))
    )
    
    df = df.withColumn(
        "C12_expensive_for_driver_age", 
        when(condition_expensive_car, 1)
        .when(condition_expensive_car.isNull(), 1)
        .otherwise(0)
    )
    
    # C14: Contains watch words
    watch_words = "|".join([
        "commut", "deliver", "parcel", "drink", "police", "custody", "arrest", 
        "alcohol", "drug", "station", "custody"
    ])
    
    df = df.withColumn(
        "C14_contains_watchwords",
        when(lower(col("Circumstances")).rlike(watch_words), 1)
        .when(col("Circumstances").isNull(), 1)
        .otherwise(0)
    )
    
    return df

# COMMAND ----------

def aggregate_check_columns(df, check_cols):
    """
    Aggregate check columns to create overall risk indicator.
    
    Parameters:
    df: Spark DataFrame
    check_cols: List of check column names
    
    Returns:
    DataFrame with checks_max column added
    """
    df = df.withColumn('checks_max', greatest(*[col(c) for c in check_cols]))
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vehicle and Driver Features

# COMMAND ----------

def add_vehicle_age_features(df):
    """
    Add vehicle age related features.
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    DataFrame with vehicle age features
    """
    df = df.withColumn(
        "veh_age", 
        round(datediff(col("start_date"), 
                      to_date(concat(col("manufacture_yr_claim"), lit('-01-01')))) / 365.25, 0)
    )
    
    df = df.withColumn("veh_age_more_than_10", (col("veh_age") > 10).cast("int"))
    
    return df

# COMMAND ----------

def add_driver_age_features(df):
    """
    Add driver age related features.
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    DataFrame with driver age features
    """
    df = df.withColumn(
        "claim_driver_age",
        round(datediff(col("start_date"), to_date(col("claim_driver_dob"))) / 365.25)
    )
    
    return df

# COMMAND ----------

def process_driver_features(df):
    """
    Process driver-related features by taking min/max values across drivers.
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    DataFrame with processed driver features
    """
    driver_cols = [
        'additional_vehicles_owned', 
        'age_at_policy_start_date', 
        'cars_in_household', 
        'licence_length_years', 
        'years_resident_in_uk'
    ]
    
    for col_name in driver_cols:
        df = df.withColumn(
            f"max_{col_name}", 
            greatest(
                col(f"{col_name}_1"), 
                col(f"{col_name}_2"), 
                col(f"{col_name}_3"), 
                col(f"{col_name}_4"), 
                col(f"{col_name}_5")
            )
        )    
        
        df = df.withColumn(
            f"min_{col_name}", 
            least(
                col(f"{col_name}_1"), 
                col(f"{col_name}_2"), 
                col(f"{col_name}_3"), 
                col(f"{col_name}_4"), 
                col(f"{col_name}_5")
            )
        )
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Type Conversions

# COMMAND ----------

def convert_boolean_columns(df):
    """
    Convert boolean columns to integer (0/1).
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    DataFrame with boolean columns converted to integers
    """
    boolean_columns = [
        "vehicle_unattended", "excesses_applied", "is_first_party",
        "first_party_confirmed_tp_notified_claim", "is_air_ambulance_attendance", 
        "is_ambulance_attendance", "is_fire_service_attendance", "is_police_attendance"
    ]
    
    for col_name in boolean_columns:
        df = df.withColumn(col_name, col(col_name).cast("integer"))
    
    return df

# COMMAND ----------

def convert_decimal_columns(df):
    """
    Convert decimal columns to float.
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    DataFrame with decimal columns converted to float
    """
    decimal_cols = ['outstanding_finance_amount', 'vehicle_value', 'voluntary_amount']
    
    for col_name in decimal_cols:
        df = df.withColumn(col_name, col(col_name).cast("float"))
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Lists and Configuration

# COMMAND ----------

def get_feature_lists():
    """
    Get all feature lists used in the models.
    
    Returns:
    Dictionary containing various feature lists
    """
    return {
        "check_cols": [
            'C10_claim_to_policy_end', 'C11_young_or_inexperienced', 
            'C12_expensive_for_driver_age', 'C14_contains_watchwords', 
            'C1_fri_sat_night', 'C2_reporting_delay',
            'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 
            'C6_no_commuting_but_rush_hour', 'C7_police_attended_or_crime_reference', 
            'C9_policy_within_30_days'
        ],
        
        "final_features_desk_check": [
            'C10_claim_to_policy_end', 'C2_reporting_delay', 
            'C3_weekend_incident_reported_monday', 'C5_is_night_incident', 
            'C9_policy_within_30_days', 'annual_mileage', 'areasDamagedHeavy', 
            'areasDamagedMedium', 'areasDamagedMinimal', 'areasDamagedSevere', 
            'areasDamagedTotal', 'assessment_category', 'business_mileage', 
            'checks_max', 'claim_to_policy_end', 'damageScore', 'doors_open', 
            'first_party_confirmed_tp_notified_claim', 'front_bonnet_severity', 
            'front_severity', 'impact_speed', 'impact_speed_range', 
            'inception_to_claim', 'incidentDayOfWeekC', 'incidentHourC', 
            'incidentMonthC', 'incident_cause', 'incident_day_of_week', 
            'incident_sub_cause', 'is_crime_reference_provided', 'is_police_attendance', 
            'is_reported_monday', 'manufacture_yr_claim', 'max_cars_in_household', 
            'min_claim_driver_age', 'min_licence_length_years', 'min_years_resident_in_uk', 
            'ncd_protected_flag', 'notification_method', 'policy_cover_type', 
            'policy_type', 'policyholder_ncd_years', 'right_rear_wheel_severity', 
            'veh_age', 'vehicle_overnight_location_id', 'vehicle_value', 
            'voluntary_amount', 'C11_young_or_inexperienced', 'C12_expensive_for_driver_age', 
            'C14_contains_watchwords', 'C1_fri_sat_night', 'C6_no_commuting_but_rush_hour', 
            'C7_police_attended_or_crime_reference'
        ],
        
        "final_features_interview": [
            'voluntary_amount', 'policyholder_ncd_years', 'max_years_resident_in_uk', 
            'annual_mileage', 'min_claim_driver_age', 'incidentHourC', 
            'assessment_category', 'left_severity', 'C9_policy_within_30_days', 
            'areasDamagedHeavy', 'impact_speed', 'incident_sub_cause', 
            'vehicle_value', 'areasDamagedTotal', 'manufacture_yr_claim', 
            'rear_right_severity', 'claim_to_policy_end', 'veh_age', 
            'licence_length_years_1', 'num_failed_checks', 'front_left_severity', 
            'areasDamagedMedium', 'rear_window_damage_severity', 'incident_cause', 
            'incidentDayOfWeekC', 'age_at_policy_start_date_1', 'checks_max', 
            'total_loss_flag'
        ]
    }

# COMMAND ----------

def create_num_failed_checks(df, check_cols):
    """
    Create num_failed_checks feature showing count of failed checks.
    
    Parameters:
    df: Spark DataFrame
    check_cols: List of check column names
    
    Returns:
    DataFrame with num_failed_checks column
    """
    df = df.withColumn(
        "checks_list",
        array(*[when(col(c) == 1, lit(c)).otherwise(lit(None)) for c in check_cols])
    )
    
    df = df.withColumn(
        "checks_list",
        expr("filter(checks_list, x -> x is not null)")
    ).withColumn("num_failed_checks", size(col("checks_list")))
    
    return df

# COMMAND ----------

def create_areas_damaged_relative(df):
    """
    Create relative damage score feature.
    
    Parameters:
    df: Spark DataFrame
    
    Returns:
    DataFrame with areasDamagedRelative column
    """
    df = df.withColumn(
        "areasDamagedRelative", 
        col("areasDamagedMinimal") + 
        2 * col("areasDamagedMedium") + 
        3 * col("areasDamagedSevere") + 
        4 * col("areasDamagedHeavy")
    )
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## The End