# Databricks notebook source
# MAGIC %run ../configs/configs

# COMMAND ----------

#Import modules
import math
import sys, os
import re
import random
import datetime
import json
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window, DataFrame
from pyspark.sql.functions import (
    col, regexp_replace, expr, row_number, when, datediff, year, 
    dayofweek, dayofmonth, month, regexp_extract, lit, greatest, 
    current_date, floor, udf, array
)
from pyspark.sql.types import IntegerType
from functools import reduce
from operator import add

# Append notebook path to Sys Path
notebk_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys_path = functions_path(notebk_path)
sys.path.append(sys_path)

# Import functions and configurations
from functions.feature_engineering import *

with open(f'{config_path}', 'r') as file:
    config = yaml.safe_load(file)

# Extract the congig lists
extract_column_transformation_lists("/config_files/feature_engineering.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
catalog = config['workspaces'].get(workspace_url) + catalog_prefix

# COMMAND ----------

# MAGIC %md
# MAGIC ### Functions for data_preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ### The End

# COMMAND ----------

df = spark.read.table(f"{catalog}.third_party_capture.tpc_raw_data_table").withColumnRenamed('INC_TOT_TPPD', 'inc_tot_tppd') \
    .withColumnRenamed('STATUS', 'status') \
    .withColumnRenamed('FAULT_IND', 'fault_ind') \
    .withColumnRenamed('PAID_TOT_TPPD', 'paid_tot_tppd') \
    .withColumnRenamed('RESERVE_TPPD', 'reserve_tppd') \
    .withColumnRenamed('RESERVE_REC_TPPD', 'reserve_rec_tppd') \
    .withColumnRenamed('PAID_REC_TPPD', 'paid_rec_tppd')

# COMMAND ----------

df = add_assessment_columns(df)

# Apply the function to the dataframe
df = process_impact_speed(df)

# Apply the function to the dataframe
df = process_deployed_airbags(df)

# COMMAND ----------

# Count the number of damaged areas on the vehicle.
df = count_damaged_areas(df, severity_columns)

# COMMAND ----------

# Apply the function to the dataframe
df = process_damage_severity(df, severity_columns)

df = transform_fuel_and_body_type(df)

# Apply the functions to the dataframe
df = convert_columns_to_int(df, 'vehicle_value')
df = convert_columns_to_timestamp(df, 'incident_date')

# Create column for time to notify
df = create_time_to_notify_column(df)

# Create vehicle_age variable
df = create_vehicle_age_column(df)

# COMMAND ----------

# Convert numeric columns to int
convert_to_int = ['number_of_doors', 
                  'engine_capacity', 
                  'number_of_seats', 
                  'body_key_01', 
                  'body_key_02', 
                  'body_key_03', 
                  'body_key_04', 
                  'fuel_type_01', 
                  'fuel_type_02']

# Convert numeric columns to int              
df = convert_columns_to_int(df, convert_to_int)

# COMMAND ----------

df = convert_right_hand_drive(df)

df = encode_damage_columns(df, severity_columns, 'right_hand_drive')

df = create_date_fields(df)

df = calculate_damage_severity(df, severity_columns)

df = fill_insurer_name(df)

# COMMAND ----------

all_columns = severity_columns + tp_fp_columns

for column in all_columns:
    df = df.withColumn(f'fp_{column}', when(col('first_party_confirmed') == 1, col(column)).otherwise(None)) \
           .withColumn(f'tp_{column}', when(col('first_party_confirmed') == 0, col(column)).otherwise(None))

# COMMAND ----------

# Fill NA for expv_colour with the most common colour
# Fill NA for t_number_of_doors with the most common number of doors
# Fill NA for t_number_of_seats with the most common number of seats

columns_to_fill = ["colour", 
                   "number_of_doors", 
                   "number_of_seats",
                   "engine_capacity"]

df = fill_na_with_most_common(df, columns_to_fill)

# COMMAND ----------

df = df.dropDuplicates()

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

# pivot and agg so have 2 cols for every sub TPPD divison of 'HOD_DESC' (_PAID AND _RESERVE)
df_sas_transactions_tppd_pivot = (
    df.groupBy(col("CLM_NUMBER"))
    .pivot("payment_category")
    .agg(
        sum(col("paid_tot_tppd")).alias("paid"),
        sum(col("reserve_tppd")).alias("reserve")
    )
    .na.fill(value=0)
).withColumn(
    "Has_Paid_Intervention",
    when(
        (col("TP Intervention (Vehicle)_paid") > 0) |
        (col("TP Intervention (Mobility)_paid") > 0) |
        (col("TP Intervention (Uninsured Loss)_paid") > 0),
        True
    ).otherwise(False)
)


# COMMAND ----------

df_joined = df.join(df_sas_transactions_tppd_pivot.select('CLM_NUMBER', 'Has_Paid_Intervention'), on="CLM_NUMBER", how="inner")

# COMMAND ----------

df_joined = df_joined.withColumn(
        "CaptureSuccess_AdrienneVersion",
        when(
            (
                (col("intervention_outcome") == "Captured")
                | (col("Has_Paid_Intervention") == True)
            ),
            1,
        ).otherwise(0),
    )

# COMMAND ----------

df_test = df_joined#.filter(col("inc_tot_tppd") > 0)

df_value_counts = df_test.groupBy("CaptureSuccess_AdrienneVersion").count()

display(df_value_counts)

# COMMAND ----------

df = df.withColumn(
        "CaptureSuccess_AdrienneVersion",
        when(
            (
                (col("payment_category") == "TP Intervention (Vehicle)") | (col("payment_category") == "TP Intervention (Uninsured Loss)") | (col("payment_category") == "TP Intervention (Mobility)")
                #| (col("paid_tot_tppd") >= 0)
                | (col("intervention_outcome") == 'Captured')
            ),
            1,
        ).otherwise(0),
    )

df = df.withColumn(
    "CaptureSuccess_AdrienneVersion",
    when(
        (
            ((col("payment_category") == "TP Intervention (Vehicle)") & (col("paid_tot_tppd") >= 0)) |
            ((col("payment_category") == "TP Intervention (Uninsured Loss)") & (col("paid_tot_tppd") >= 0)) |
            ((col("payment_category") == "TP Intervention (Mobility)") & (col("paid_tot_tppd") >= 0)) |
            (col("intervention_outcome") == 'Captured')
        ),
        1,
    ).otherwise(0),
)

# COMMAND ----------

df_test = df#.filter(col("inc_tot_tppd") > 0)

df_value_counts = df_test.groupBy("CaptureSuccess_AdrienneVersion").count()

display(df_value_counts)

# COMMAND ----------

from pyspark.sql import functions as F
df = df_joined.filter(F.col("liability_status_description").isin(
            "Fault",
    ))

# COMMAND ----------

# categorise claim items
motorVehicleList = [
    "LorryMotorVehicleClaimItem",
    "UnknownVehicleClaimItem",
    "MinibusMotorVehicleClaimItem",
    "VanMotorVehicleClaimItem",
    "BusCoachMotorVehicleClaimItem",
    "HeavyPlantMotorVehicleClaimItem",
    "CarMotorVehicleClaimItem",
    "MotorhomeClaimItem",
    "MotorcycleMotorVehicleClaimItem",
]
allowedVehicleList = [
    "VanMotorVehicleClaimItem",
    "CarMotorVehicleClaimItem",
    "MotorcycleMotorVehicleClaimItem",
]
personalInjuryItemList = ["PedestrianPersonalInjuryClaimItem"]
otherItemList = [
    "ReinsuranceClaimItem",
    "AnimalInjuryClaimItem",
    "Unknown",
    "KeyClaimItem",
    "PropertyClaimItem",
    "BicycleNonMotorVehicleClaimItem",
    "StreetFurnitureClaimItem",
]

# claim item indicators
df = (
    df.withColumn(
        "MotorVehicleFP",
        when(
            (col("first_party") == 1)
            & (col("claim_item_type").isin(motorVehicleList)),
            1,
        ).otherwise(0),
    )
    .withColumn(
        "MotorVehicleTP",
        when(
            (col("first_party") == 0)
            & (col("claim_item_type").isin(motorVehicleList)),
            1,
        ).otherwise(0),
    )
    .withColumn(
        "PIItems",
        when(
            col("claim_item_type") == "PedestrianPersonalInjuryClaimItem", 1
        ).otherwise(0),
    )
    .withColumn(
        "OtherItems", when(col("claim_item_type").isin(otherItemList), 1).otherwise(0)
    )
    .withColumn(
        "AllowedVehicle",
        when(
            (col("first_party") == 0)
            & (col("claim_item_type").isin(allowedVehicleList)),
            1,
        ).otherwise(0),
    )
)

# COMMAND ----------

# Remove columns with constant values
df = df.drop(*constant_cols)

# COMMAND ----------

from pyspark.sql.functions import desc, row_number
from pyspark.sql.window import Window

window_spec = Window.partitionBy("claim_number").orderBy(desc("claim_number"))
df = df.withColumn("row_num", row_number().over(window_spec)).filter("row_num = 1").drop("row_num")

# COMMAND ----------

from pyspark.sql.functions import col, sum as F_sum

# get item counts for each claim to help with filtering
df_grouped = df.groupBy("claim_number").agg(
    F_sum("MotorVehicleTP").alias("TPMotorVehicle_Count"),
    F_sum("PIItems").alias("PIItem_Count"),
    F_sum("OtherItems").alias("OtherItem_Count"),
    F_sum("AllowedVehicle").alias("AllowedVehicle_Count"),
)

df = df.join(df_grouped, on="claim_number", how="left")
df_grouped.unpersist()

# filter so only looking at TP allowed vehicles
df = df.where(col("MotorVehicleTP") == 1)

# COMMAND ----------

df = df.where(
        (col("TPMotorVehicle_Count") == 1) & (col("AllowedVehicle_Count") == 1)
    )

# where we have made a TPPD payment, and where reserve is closed
df = df.where(
    (col("RESERVE_TPPD") == 0)
    & (col("RESERVE_REC_TPPD") == 0)
    & (col("PAID_TOT_TPPD") > 0)
)

df.count() 

# COMMAND ----------

df_test = df#.filter(col("inc_tot_tppd") > 0)

df_value_counts = df_test.groupBy("CaptureSuccess_AdrienneVersion").count()

display(df_value_counts)

# COMMAND ----------

# Remove columns with constant values
df = df.drop(*constant_cols)

# COMMAND ----------

df = df.select(ordered_columns)

# COMMAND ----------

from pyspark.sql.functions import desc, row_number
from pyspark.sql.window import Window

window_spec = Window.partitionBy("claim_number").orderBy(desc("claim_number"))
df = df.withColumn("row_num", row_number().over(window_spec)).filter("row_num = 1").drop("row_num")

# COMMAND ----------

df = df.dropDuplicates() 

# COMMAND ----------

df = df.withColumnRenamed(
        "incident_cause", "incident_cause_description"
    ).withColumnRenamed(
        "incident_sub_cause", "incident_sub_cause_description"
    ).withColumnRenamed(
        "notified_dow", "notified_day_of_week"
    ).withColumnRenamed(
        "incident_dow", "incident_day_of_week"
    ).withColumnRenamed(
        "incident_location_uk_country", "incident_uk_country"
    ).withColumnRenamed(
        "postcode_area", "incident_postcode"
    ).withColumn(
        "year_of_manufacture", df["year_of_manufacture"].cast("int")
    ).withColumnRenamed(
        "vehicle_kept_at_postcode", "postcode_area"
    )

# COMMAND ----------

df_test = df.filter(col("inc_tot_tppd") > 0)

df_value_counts = df_test.groupBy("CaptureSuccess_AdrienneVersion").count()

display(df_value_counts)

# COMMAND ----------

# Create the schema if it does not exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.third_party_capture")

# Write the DataFrame to a Delta table
df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(f"{catalog}.third_party_capture.tpc_feature_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ##The End