import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace, udf, expr, col, when, lit, substring, length, col, col as spark_col, when, monotonically_increasing_id, lit, sum as _sum, first, col as spark_col, row_number
from pyspark.sql.functions import (
    col, regexp_replace, expr, row_number, when, datediff, year, 
    dayofweek, dayofmonth, month, regexp_extract, lit, greatest, 
    current_date, floor, udf, array, isnan
)
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
import re
from typing import List
from pyspark.sql.types import IntegerType, DoubleType, DateType, StringType
from pyspark.sql.window import Window
from functools import reduce
from operator import add

def read_and_process_delta(spark, file_path, useful_columns=None, drop_columns=None, skiprows=0):
    """Read a Delta table, select useful columns or drop unnecessary columns, and return the DataFrame."""
    print(file_path)
    df = spark.read.format("delta").table(file_path)
    
    # Select useful columns or drop unnecessary columns
    if useful_columns:
        df = df.select(*useful_columns)
    if drop_columns:
        df = df.drop(*drop_columns)
    
    # Skip rows if needed (in Spark this will be equivalent to filtering rows)
    if skiprows > 0:
        df = df.withColumn("row_index", row_number().over(Window.orderBy(monotonically_increasing_id())))
        df = df.filter(df.row_index > skiprows).drop("row_index")
    
    return df

def recast_dtype(raw_df: DataFrame, column_list: List[str], dtype: str) -> DataFrame:
    """
    Recast the data type of specified columns in a DataFrame.

    Parameters:
    raw_df (DataFrame): The current Spark DataFrame.
    column_list (List[str]): List of columns to cast to the given data type.
    dtype (str): The target data type to cast the columns to.

    Returns:
    DataFrame: A DataFrame with the specified columns cast to the given data type.
    """
    for column_name in column_list:
        raw_df = raw_df.withColumn(column_name, col(column_name).cast(dtype))
    
    return raw_df

def add_assessment_columns(df):
    df = df.withColumn('da_dr', when(col('assessment_category') == 'DriveableRepair', 1).otherwise(0))
    df = df.withColumn('da_dtl', when(col('assessment_category') == 'DriveableTotalLoss', 1).otherwise(0))
    df = df.withColumn('da_utl', when(col('assessment_category') == 'UnroadworthyTotalLoss', 1).otherwise(0))
    df = df.withColumn('da_ur', when(col('assessment_category') == 'UnroadworthyRepair', 1).otherwise(0))
    df = df.withColumn('da_o', when(
        (~col('assessment_category').isin([
            'DriveableRepair', 
            'DriveableTotalLoss', 
            'UnroadworthyTotalLoss', 
            'UnroadworthyRepair'
        ])) | col('assessment_category').isNull() | isnan(col('assessment_category')), 1).otherwise(0))
    return df

def convert_columns_to_int(df: DataFrame, columns) -> DataFrame:
    """
    Casts the specified columns of the DataFrame to integer data type.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list or str): List of column names or a single column name to be cast to integer.

    Returns:
    DataFrame: The DataFrame with specified columns cast to integer data type.
    """
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        df = df.withColumn(column, col(column).cast('int'))
    return df

def convert_columns_to_timestamp(df: DataFrame, columns) -> DataFrame:
    """
    Casts the specified columns of the DataFrame to timestamp data type.

    Parameters:
    df (DataFrame): The input DataFrame.
    columns (list or str): List of column names or a single column name to be cast to timestamp.

    Returns:
    DataFrame: The DataFrame with specified columns cast to timestamp data type.
    """
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        df = df.withColumn(column, col(column).cast('timestamp'))
    return df

def process_impact_speed(df):

    from pyspark.sql.functions import  floor

    # Fill NA for body_key with the most common body style
    most_common_body_style = df.groupBy("body_key").count().orderBy(col("count").desc()).first()[0]
    df = df.fillna({"body_key": most_common_body_style})

    # Map impact_speed_range to base impact_speed in MPH
    df = df.withColumn('impact_speed', when(df.impact_speed_range == 'Stationary', 0)
        .when(df.impact_speed_range == 'ZeroToSix', 1)
        .when(df.impact_speed_range == 'SixToTen', 6)
        .when(df.impact_speed_range == 'SevenToFourteen', 7)
        .when(df.impact_speed_range == 'FifteenToTwenty', 15)
        .when(df.impact_speed_range == 'TwentyOneToThirty', 21)
        .when(df.impact_speed_range == 'ThirtyOneToFourty', 31)
        .when(df.impact_speed_range == 'FourtyOneToFifty', 41)
        .when(df.impact_speed_range == 'FiftyOneToSixty', 51)
        .when(df.impact_speed_range == 'SixtyOneToSeventy', 61)
        .when(df.impact_speed_range == 'OverSeventy', 71)
        .otherwise(-1).cast('int')
    )

    # Convert KMH to MPH
    df = df.withColumn('impact_speed',
        when(col("impact_speed_unit") == 'KMH', floor(col("impact_speed") / 1.6))
        .otherwise(col("impact_speed"))
    )

    # Standardize impact speed ranges
    df = df.withColumn('impact_speed', when((col("impact_speed") >= 21) & (col("impact_speed") <= 30), 21)
        .when((col("impact_speed") >= 31) & (col("impact_speed") <= 40), 31)
        .when((col("impact_speed") >= 41) & (col("impact_speed") <= 50), 41)
        .when((col("impact_speed") >= 51) & (col("impact_speed") <= 60), 51)
        .when((col("impact_speed") >= 61) & (col("impact_speed") <= 70), 61)
        .otherwise(col("impact_speed"))
    )

    # Fill missing impact_speed with most common value
    most_common_impact_speed = df.groupBy("impact_speed").count().orderBy(col("count").desc()).first()[0]
    df = df.fillna({"impact_speed": most_common_impact_speed})

    return df


def process_deployed_airbags(df):
    df = df.withColumn(
        'deployed_airbags', 
        when(df.deployed_airbags == "None", 0)
        .when(df.deployed_airbags == "One", 1)
        .when(df.deployed_airbags == "Two", 2)
        .when(df.deployed_airbags == "Three", 3)
        .when(df.deployed_airbags == "Four", 4)
        .when(df.deployed_airbags == "All", 5)
        .otherwise(-1)
    )
    return df

def count_damaged_areas(df, damage_columns):
    # Create a new column 'damage_recorded' by counting non-null values in the specified columns
    df = df.withColumn(
        'damage_recorded', 
        sum([when(col(c).isNotNull(), 1).otherwise(0) for c in damage_columns])
    )

    # Create a new column 'damage_assessed' based on 'damage_recorded'
    df = df.withColumn(
        'damage_assessed', 
        when(col('damage_recorded') > 0, 1).otherwise(0)
    )
    
    return df

def process_damage_severity(df, severity_columns):
    def damage_scale(row_area, row_damageflag):
        if row_damageflag == 1:
            if row_area == 'Minimal':
                scale = 1
            elif row_area == 'Medium':
                scale = 2
            elif row_area == 'Heavy':
                scale = 3
            elif row_area == 'Severe':
                scale = 4
            elif row_area == 'Unknown':
                scale = -1
            else:
                scale = 0
        else:
            scale = -1
        return scale

    damage_scale_udf = udf(damage_scale, IntegerType())

    for col_name in severity_columns:
        df = df.withColumn(col_name, damage_scale_udf(col(col_name), col('damage_assessed')).cast('int'))

    return df

def transform_fuel_and_body_type(df):
    # Create a new column for each fuel type
    df = df.withColumn("fuel_type_01", when(col("fuel_type_code") == "1", 1).otherwise(0)) # Diesel
    df = df.withColumn("fuel_type_02", when(col("fuel_type_code") == "2", 1).otherwise(0)) # Petrol
    
    # Create a new column for each body type
    df = df.withColumn("body_key_01", when(col("body_key") == "5 Door Hatchback", 1).otherwise(0)) # 5 Door Hatchback
    df = df.withColumn("body_key_02", when(col("body_key") == "5 Door Estate", 1).otherwise(0)) # 5 Door Estate
    df = df.withColumn("body_key_03", when(col("body_key") == "4 Door Saloon", 1).otherwise(0)) # 4 Door Saloon
    df = df.withColumn("body_key_04", when(col("body_key") == "3 Door Hatchback", 1).otherwise(0)) # 3 Door Hatchback
    
    return df

def create_time_to_notify_column(df: DataFrame) -> DataFrame:
    """
    Creates a 'time_to_notify' column in the DataFrame based on the difference between 'notified_date' and 'incident_start_date'.
    The values are capped between 0 and 30, and any null values are filled with 0.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The DataFrame with the 'time_to_notify' column added.
    """
    df = (df.withColumn('time_to_notify', datediff(col('notification_date'), col('incident_date')))
            .withColumn('time_to_notify', when(col('time_to_notify') < 0, 0).otherwise(col('time_to_notify')))
            .withColumn('time_to_notify', when(col('time_to_notify') > 30, 30).otherwise(col('time_to_notify')))
            .fillna(0, ['time_to_notify']))
    return df

def create_vehicle_age_column(df: DataFrame) -> DataFrame:
    """
    Creates a 'vehicle_age' column in the DataFrame based on the difference between the year of 'incident_date' and 'year_of_manufacture'.
    Drops the 'year_of_manufacture' column after creating 'vehicle_age'.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The DataFrame with the 'vehicle_age' column added and 'year_of_manufacture' column dropped.
    """
    df = df.withColumn('vehicle_age', year(col('incident_date')) - col('year_of_manufacture').cast('int'))
    return df

def convert_right_hand_drive(df):
    df = df.withColumn('right_hand_drive', when(col('right_hand_drive') == 'R', 1).otherwise(0))
    return df

def encode_damage_columns(df, damage_cols, special_car_cols):

    mapper = {'false': 0, 
              'true': 1, 
              'null': 0, 
              None: 0, 
              'None': 0, 
              'Unknown': 0, 
              'Minimal': 1, 
              'Medium': 2, 
              'Heavy': 3, 
              'Severe': 4, 
              'One': 1, 
              'Two': 2, 
              'Three': 3, 
              'Four': 4, 
              'All': 4}
    
    def convert_damage(d):
        try:
            return mapper[d]
        except:
            return d

    damage_udf = udf(convert_damage, IntegerType())

    if isinstance(special_car_cols, str):
        special_car_cols = [special_car_cols]
    
    for c in damage_cols + special_car_cols:
        df = df.withColumn(c, damage_udf(col(c)).cast('int'))
    
    return df


def create_date_fields(df):
    df = (df.withColumn('notified_day_of_week', when(dayofweek('notification_date') == 1, 7).otherwise(dayofweek('notification_date') - 1))
            .withColumn('notified_day', dayofmonth('notification_date'))
            .withColumn('notified_month', month('notification_date'))
            .withColumn('notified_year', year('notification_date'))
            .withColumn('incident_day_of_week', when(dayofweek('incident_date') == 1, 7).otherwise(dayofweek('incident_date') - 1))
            .withColumn('incident_day', dayofmonth('incident_date'))
            #.drop('notification_date', 'incident_start_date')
            .withColumn('postcode_area', regexp_extract('vehicle_kept_at_postcode', r'^\D+', 0))
            #.drop('vehicle_kept_at_postcode')
            .withColumn('vehicle_age', when(col('vehicle_age') <= 30, col('vehicle_age')).otherwise(None))
            .fillna('zz', ['postcode_area'])
            .fillna('zmissing', ['manufacturer_description']))
    return df

def impute_missing_with_median(df, columns):
    for c in columns:
        median_value = df.approxQuantile(c, [0.5], 0.25)[0]
        df = df.withColumn(c, when(col(c).isNotNull(), col(c)).otherwise(median_value))
    return df

def calculate_damage_severity(df, sev_damage):
    df = df.withColumn("damage_sev_total", reduce(add, [col(c) for c in sev_damage]))
    df = df.withColumn("damage_sev_count", reduce(add, [when(col(c) > 0, 1).otherwise(0) for c in sev_damage]))
    df = df.withColumn("damage_sev_mean", when(col('damage_sev_count') > 0, (col('damage_sev_total') / col('damage_sev_count'))).otherwise(lit(0)))
    df = df.withColumn("damage_sev_max", greatest(*sev_damage))
    return df

def fill_insurer_name(df):
    return df.fillna({"insurer_name": "Unknown"})

def fill_na_with_most_common(df, columns):
    for column in columns:
        most_common_value = df.groupBy(column).count().orderBy(col("count").desc()).first()[0]
        df = df.fillna({column: most_common_value})
    return df