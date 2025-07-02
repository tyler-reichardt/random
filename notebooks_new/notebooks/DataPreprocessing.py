# Databricks notebook source
# MAGIC %run ../configs/configs

# COMMAND ----------

#Import modules
import math
from pyspark.sql import SparkSession
import sys, os
import re
import random
import datetime
import json
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, regexp_replace, max, expr, row_number, when, datediff, rank, array_contains, array, create_map, lit
from pyspark.sql import Window

notebk_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys_path = functions_path(notebk_path)

sys.path.append(sys_path)
from functions.data_processing import *

with open(f'{config_path}', 'r') as file:
    config = yaml.safe_load(file)

# Extract the congig lists
extract_column_transformation_lists("/config_files/data_preprocessing.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
catalog = config['workspaces'].get(workspace_url) + catalog_prefix

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Extraction

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import SparkSession
# If you want to hint broadcast, import it:
from pyspark.sql.functions import broadcast

# Load claim_version_item as early as possible
claim_version_item = spark.table("prod_adp_certified.claim.claim_version_item") \
    .select(
        "event_identity", "claim_id", "insurance_id",
        "vehicle_unattended", "first_party", 'is_party_at_fault', "not_on_mid", "claim_version_item_index", "claim_item_type"
    )

# Load claim data with only required columns
claim = spark.table("prod_adp_certified.claim.claim") \
    .select("claim_id", "policy_number", "claim_number", "notified_date", "first_party_confirmed") \
    .withColumnRenamed("notified_date", "notification_date")

# Base join for claim details
intermin_df = claim_version_item.join(claim, on="claim_id", how="left") # inner

# Load claim data with only required columns
claim_version = spark.table("prod_adp_certified.claim.claim_version") \
    .select("claim_id", "liability_status_description")

# Base join for claim details
intermin_df = intermin_df.join(claim_version, on="claim_id", how="left") # inner

# Join with insurance_details
insurance_details = spark.table("prod_adp_certified.claim.insurance_details") \
    .select("insurance_id", "name") \
    .withColumnRenamed("name", "insurer_name")

intermin_df = intermin_df.join(insurance_details, on="insurance_id", how="left")

# Join with vehicle data (from claim) on event_identity
vehicle_claim = spark.table("prod_adp_certified.claim.vehicle") \
    .select(
        "event_identity", "vehicle_registration", 
        "vehicle_kept_at_postcode", "year_of_manufacture", "manufacturer_description", "market_value"
    ).withColumnRenamed("market_value", "vehicle_value")
intermin_df = intermin_df.join(vehicle_claim, on="event_identity", how="left")

# Join with incident data and rename start_date
incident = spark.table("prod_adp_certified.claim.incident") \
    .select(
        "event_identity", "impact_speed_range", "impact_speed_unit", "start_date",
        "incident_cause", "incident_sub_cause", "incident_location_uk_country",
        "incident_type", "incident_weather_conditions", "reported_date",
        "notification_method", "road_conditions"
    ).withColumnRenamed(
        "incident_weather_conditions", "weather_conditions"
    ).withColumnRenamed(
        "incident_cause", "incident_cause_description"
    ).withColumnRenamed(
        "incident_sub_cause", "incident_sub_cause_description"
    )
    
intermin_df = intermin_df.join(incident, on="event_identity", how="left") \
    .withColumnRenamed("start_date", "incident_date")

# Join with damage_details
damage_details = spark.table("prod_adp_certified.claim.damage_details") \
    .select(
        "event_identity", 
        "claim_version_item_index",
        "boot_opens", 
        "deployed_airbags", 
        "doors_open", 
        "driveable", 
        "engine_damage",
        "exhaust_damaged",
        "assessment_category", 
        "front_bonnet_severity", 
        "front_left_severity", 
        "front_right_severity", 
        "front_severity",
        "left_back_seat_severity", 
        "left_front_wheel_severity", 
        "left_mirror_severity", 
        "left_rear_wheel_severity", 
        "left_roof_severity", 
        "left_severity", 
        "left_underside_severity", 
        "lights_damaged", 
        "load_area_severity", 
        "panel_gaps", 
        "radiator_damaged", 
        "rear_left_severity", 
        "rear_right_severity", 
        "rear_severity", 
        "rear_window_damage_severity", 
        "right_back_seat_severity", 
        "right_front_wheel_severity", 
        "right_mirror_severity", 
        "right_rear_wheel_severity", 
        "right_roof_severity", 
        "right_severity", 
        "right_underside_severity", 
        "roof_damage_severity", 
        "sharp_edges", 
        "underbody_damage_severity", 
        "wheels_damaged", 
        "windscreen_damage_severity" 
    ).withColumnRenamed(
        "driveable", "driveable_damage_assessment"
    )

# Define the window specification
window_spec = Window.partitionBy("event_identity").orderBy(col("claim_version_item_index").desc())

# Apply the window function and filter to keep only the latest version
damage_details = damage_details.withColumn(
    "row_num", row_number().over(window_spec)
).filter(col("row_num") == 1).drop("row_num")

intermin_df = intermin_df.join(damage_details, on="event_identity", how="left")

# Finally join with tp_intervention for intervention_outcome; result in claim_df
tp_intervention = spark.table("prod_adp_certified.claim.tp_intervention") \
    .select("event_identity", "intervention_outcome")
    
intermin_df = intermin_df.join(tp_intervention, on="event_identity", how="left").drop('claim_version_item_index')

payment_component  = spark.table("prod_adp_certified.claim.payment_component").select('event_identity', 'payment_category').dropDuplicates(['event_identity'])

claim_df = intermin_df.join(payment_component, on="event_identity", how="left")

# COMMAND ----------

# Load quote iteration and vehicle-related tables
quote_iteration = spark.table("prod_adp_certified.quote_motor.quote_iteration").select(
    "quote_iteration_id", "policy_reference"
)

# Load quote iteration and vehicle-related tables
cue_crif_claim_motor = spark.table("prod_adp_certified.quote_motor.cue_crif_claim_motor").select(
    "quote_iteration_id", "claim_nr"
)

vehicle_quote = spark.table("prod_adp_certified.quote_motor.vehicle").select(
    "quote_iteration_id", "vehicle_registration", "abi_model_code", "fuel_type_code",
    "right_hand_drive"
).withColumnRenamed(
    "vehicle_registration", "vehicle_reg"
)

vehicle_experian_profile = spark.table("prod_adp_certified.quote_motor.vehicle_experian_profile").select(
    "quote_iteration_id", "expv_colour", "expv_engine_capacity"
).withColumnRenamed(
    "expv_engine_capacity", "engine_capacity"
).withColumnRenamed(
    "expv_colour", "colour"
)

vehicle_mileage = spark.table("prod_adp_certified.quote_motor.vehicle_mileage").select(
    "quote_iteration_id", "mileage_count", "mileage_index"
).withColumnRenamed(
    "mileage_count", "mileage"
)

# Define window specification
window_spec = Window.partitionBy("quote_iteration_id").orderBy(col("mileage_index").desc())

# Add row number and filter to keep only the latest mileage_index
vehicle_mileage = vehicle_mileage.withColumn(
    "row_num", row_number().over(window_spec)
).filter(col("row_num") == 1).drop("row_num")

# Load reference tables
abi_model = spark.table("prod_adp_certified.reference.abi_model").select(
    "t_abi_code", "t_body_style", "t_number_of_doors", "t_number_of_seats"
).withColumnRenamed(
    "t_number_of_seats", "number_of_seats"
).withColumnRenamed(
    "t_number_of_doors", "number_of_doors"
).withColumnRenamed(
    "t_body_style", "body_key"
)

# Join vehicle with abi_model and fuel_type; if these are small dimensions, consider broadcasting them
vehicle_joined = vehicle_quote.join(
    abi_model, vehicle_quote["abi_model_code"] == abi_model["t_abi_code"], "left"
)

# Chain joins with the quote_motor tables using quote_iteration_id
full_joined_df = quote_iteration.join(
    vehicle_joined, on="quote_iteration_id", how="left"
).join(
    vehicle_experian_profile, on="quote_iteration_id", how="left"
).join(
    vehicle_mileage, on="quote_iteration_id", how="left"
).join(
    cue_crif_claim_motor, on="quote_iteration_id", how="left"
)


# Final selection and renaming for quote_motor
quote_motor = full_joined_df.select(
    "policy_reference", "claim_nr", "vehicle_reg", "abi_model_code",
    "fuel_type_code", "right_hand_drive", "body_key",
    "number_of_doors", "number_of_seats", "colour",
    "engine_capacity", "mileage"
)

quote_motor = quote_motor.dropDuplicates()

# COMMAND ----------

table_name = "prod_dsexp_auxiliarydata.third_party_capture.sas_claim_summaries_tyler_extract_20250616"
df_inc_tot_tppd = spark.read.table(table_name).select("CLM_NUMBER", "INC_TOT_TPPD", "STATUS", "FAULT_IND", "PAID_TOT_TPPD", "RESERVE_TPPD", "RESERVE_REC_TPPD", "PAID_REC_TPPD").withColumnRenamed('INC_TOT_TPPD', 'inc_tot_tppd') \
    .withColumnRenamed('STATUS', 'status') \
    .withColumnRenamed('FAULT_IND', 'fault_ind') \
    .withColumnRenamed('PAID_TOT_TPPD', 'paid_tot_tppd') \
    .withColumnRenamed('RESERVE_TPPD', 'reserve_tppd') \
    .withColumnRenamed('RESERVE_REC_TPPD', 'reserve_rec_tppd') \
    .withColumnRenamed('PAID_REC_TPPD', 'paid_rec_tppd')

# COMMAND ----------

claim_df = get_latest_incident(claim_df, 'claim_id', 'notification_date')

# COMMAND ----------

# Join df with final_df on policy_reference and vehicle_registration
df = claim_df.join(quote_motor, (claim_df.claim_number == quote_motor.claim_nr) & (claim_df.vehicle_registration == quote_motor.vehicle_reg), how="inner")

# COMMAND ----------

# Category to parent mapping
category_to_parent = {
    # TP_VEHICLE
    "TP Intervention (Mobility)": "TP_VEHICLE",
    "TP Intervention (Vehicle)": "TP_VEHICLE",
    "TP Credit Hire (Vehicle)": "TP_VEHICLE",
    "TP Authorised Hire (Vehicle)": "TP_VEHICLE",
    "TP Fees (Vehicle)": "TP_VEHICLE",
    "Ex-Gratia (Vehicle)": "TP_VEHICLE",
    "TP Authorized Hire Vehicle": "TP_VEHICLE",
    # TP_PROPERTY
    "Medical Expenses": "TP_PROPERTY",
    "Legal Expenses": "TP_PROPERTY",
    "TP Fees (Property)": "TP_PROPERTY",
    "TP Credit Hire (Property)": "TP_PROPERTY",
    "TP Authorised Hire (Property)": "TP_PROPERTY",
    "TP Intervention (Property)": "TP_PROPERTY",
    "Unknown": "TP_PROPERTY",
    "AD Ex-Gratia (Property)": "TP_PROPERTY",
    "Fire Ex-Gratia (Property)": "TP_PROPERTY",
    "OD Reinsurance": "TP_PROPERTY",
    "Theft Ex-Gratia (Property)": "TP_PROPERTY",
    "TP Damage (Property)": "TP_PROPERTY",
    "TP Authorized Hire Property": "TP_PROPERTY",
    "TP Intervention (Uninsured Loss)": "TP_PROPERTY",
    "TP Intervention (Fees)": "TP_PROPERTY",
    "TP Damage (Vehicle)": "TP_PROPERTY",
}
mapping_expr = create_map([lit(x) for i in category_to_parent.items() for x in i])
df = df.withColumn("skyfire_parent", mapping_expr[col("payment_category")])

# COMMAND ----------

df = df.join(df_inc_tot_tppd, df.claim_number == df_inc_tot_tppd.CLM_NUMBER, how='inner')

# COMMAND ----------

# Convert boolean columns from True/False to 1/0
df = convert_boolean_columns(df)

# Convert decimal(10,2) columns to float
df = convert_decimal_columns(df)

# COMMAND ----------

# Create the schema if it does not exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.third_party_capture")

# Write the DataFrame to a Delta table
df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(f"{catalog}.third_party_capture.tpc_raw_data_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ##The End