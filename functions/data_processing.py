import os
import random
import pandas as pd
import math
from math import floor, ceil
import re
from typing import Tuple, List, Dict
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, row_number, when

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


def get_latest_incident(df, column_id, date_column):
    # Define the window specification
    wp1 = Window.partitionBy(column_id).orderBy(col(date_column).asc())
    
    # Apply the window function and filter to keep only the first available version
    df_latest_incident = df.withColumn("rn", row_number().over(wp1)).filter(col("rn") == 1).drop("rn")
    
    return df_latest_incident

def convert_boolean_columns(df):
    # List of boolean columns to convert
    boolean_cols = [c for c, t in df.dtypes if t == 'boolean']
    
    # Convert boolean columns from True/False to 1/0
    for c in boolean_cols:
        df = df.withColumn(c, when(col(c) == True, 1).otherwise(0))
    
    return df

def convert_decimal_columns(df):
    # List of decimal(10,2) columns to convert
    decimal_cols = [c for c, t in df.dtypes if t == 'decimal(10,2)']
    
    # Convert decimal(10,2) columns to float
    for c in decimal_cols:
        df = df.withColumn(c, col(c).cast('float'))
    
    return df