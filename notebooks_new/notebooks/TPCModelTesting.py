# Databricks notebook source
# MAGIC %md
# MAGIC ## Configuration Import Block
# MAGIC
# MAGIC The block below is used to import training configurations for our machine learning project:
# MAGIC
# MAGIC ```python
# MAGIC %run ../configurations/configs
# MAGIC ```

# COMMAND ----------

# MAGIC %run ../configs/configs

# COMMAND ----------

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports and Configuration Setup
# MAGIC
# MAGIC This section of the notebook focuses on setting up the necessary environment for model training by performing the following actions:

# COMMAND ----------

import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.types import Schema, ColSpec
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, LongType

import re

# Append notebook path to Sys Path
notebk_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
sys_path = functions_path(notebk_path)
sys.path.append(sys_path)

# Import functions and configurations
from functions.training import *

with open(f'{training_config_path}', 'r') as file:
    config = yaml.safe_load(file)

extract_column_transformation_lists("/config_files/configs.yaml")

workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
catalog = config['workspaces'].get(workspace_url) + catalog_prefix

# COMMAND ----------

# MAGIC %%script echo skipping
# MAGIC
# MAGIC model_pickle = f'/Volumes/{catalog}/{schema}/{volume}/{model_path}'
# MAGIC
# MAGIC registered_model_name = f"{mlstore}.{schema}.{schema}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions Start

# COMMAND ----------

## CLASS FUNCTION BUILDER
 



import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline

# ─── 1. Splatmap Extraction ─────────────────────────────────────────────────────

class SplatmapExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, splatmap_cols=None):
        if splatmap_cols is None:
            splatmap_cols = [
                "Front", 
                "FrontBonnet", 
                "FrontLeft", 
                "FrontRight", 
                "Left", 
                "LeftBackseat",
                "LeftFrontWheel", 
                "LeftMirror", 
                "LeftRearWheel", 
                "LeftRoof", 
                "LeftUnderside",
                "LoadArea", 
                "Rear", 
                "RearLeft", 
                "RearRight", 
                "RearWindowDamage", 
                "Right",
                "RightBackseat", 
                "RightFrontWheel", 
                "RightMirror", 
                "RightRearWheel", 
                "RightRoof",
                "RightUnderside", 
                "RoofDamage", 
                "UnderbodyDamage", 
                "WindscreenDamage"
            ]
        self.splatmap_cols = splatmap_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'FP Splatmap Data' in df.columns and 'TP Splatmap Data' in df.columns:
            for col in self.splatmap_cols:
                df[f'FP_{col}'] = df['FP Splatmap Data'].apply(
                    lambda x: x.get(col) if isinstance(x, dict) else np.nan)
                df[f'TP_{col}'] = df['TP Splatmap Data'].apply(
                    lambda x: x.get(col) if isinstance(x, dict) else np.nan)
            df = df.drop(columns=['FP Splatmap Data', 'TP Splatmap Data'], errors='ignore')
        return df
    

# ─── 2. Column Renaming ─────────────────────────────────────────────────────────

class RenameColumnsTransformer(BaseEstimator, TransformerMixin):
    # allow injection of a custom map if desired
    default_map  = {
                'TP Engine Capacity':                       'tp_engine_capacity',
                'TP Damage Assessment':                     'tp_damage_assessment', # In TTR DATA but need to one hot encode
                'Notification Date':                        'notification_date',
                'IncidentCauseDescription':                 'incident_cause_description',
                'Incident Date':                            'incident_date',
                'Incident Postcode':                        'incident_postcode',
                'TP Body Key':                              'tp_body_key',
                'TP Doors':                                 'tp_doors',
                'FP Deployed Airbags':                      'fp_deployed_airbags',
                'FP Value':                                 'fp_value',
                'FP Engine Capacity':                       'fp_engine_capacity',
                'IncidentSubCauseDescription':              'incident_sub_cause_description',
                'FP Body Key':                              'fp_body_key', 
                'FP Radiator Damaged':                      'fp_radiator_damaged',
                'TP Boot Opens':                            'tp_boot_opens',
                'FP Doors':                                 'fp_doors', 
                'FP Seats':                                 'fp_seats',
                'FP Wheels Damaged':                        'fp_wheels_damaged',
                'FP Lights Damaged':                        'fp_lights_damaged',
                'TP Driveable / TP Damage Assessment?':     'tp_driveable_damage_assessment',
                'FP Doors Open':                            'fp_doors_open',
                'TP Doors Open':                            'tp_doors_open',
                'FP Panel Gaps':                            'fp_panel_gaps',
                'TP NotOnMID':                              'tp_not_on_mid',
                'FP Boot Opens':                            'fp_boot_opens',
                'TP Wheels Damaged':                        'tp_wheels_damaged',
                'FP Sharp Edges':                           'fp_sharp_edges',
                'TP Deployed Airbags':                      'tp_deployed_airbags',
                'TP Lights Damaged':                        'tp_lights_damaged',
                'FP Driveable / FP Damage Assessment':      'fp_driveable_fp_damage_assessment', # In TTR DATA but need to one hot encode
                'FP Registration':                          'fp_registration',
                'Claim Reference Number':                   'claim_reference_number',
                'FP_Front':                                 'fp_front_severity',
                'FP_FrontBonnet':                           'fp_front_bonnet_severity',
                'FP_FrontLeft':                             'fp_front_left_severity',
                'FP_FrontRight':                            'fp_front_right_severity',
                'FP_Left':                                  'fp_left_severity',
                'FP_LeftBackseat':                          'fp_left_back_seat_severity',
                'FP_LeftFrontWheel':                        'fp_left_front_wheel_severity',
                'FP_LeftMirror':                            'fp_left_mirror_severity',
                'FP_LeftRearWheel':                         'fp_left_rear_wheel_severity',
                'FP_LeftUnderside':                         'fp_left_underside_severity',
                'FP_Rear':                                  'fp_rear_severity',
                'FP_RearLeft':                              'fp_rear_left_severity',
                'FP_RearRight':                             'fp_rear_right_severity',
                'FP_RearWindowDamage':                      'fp_rear_window_damage_severity',
                'FP_Right':                                 'fp_right_severity',
                'FP_RightBackseat':                         'fp_right_back_seat_severity',
                'FP_RightFrontWheel':                       'fp_right_front_wheel_severity',
                'FP_RightMirror':                           'fp_right_mirror_severity',
                'FP_RightRearWheel':                        'fp_right_rear_wheel_severity',
                'FP_RightRoof':                             'fp_right_roof_severity',
                'FP_RightUnderside':                        'fp_right_underside_severity',
                'FP_RoofDamage':                            'fp_roof_damage_severity',
                'FP_UnderbodyDamage':                       'fp_underbody_damage_severity',
                'FP_WindscreenDamage':                      'fp_windscreen_damage_severity',
                'TP_Front':                                 'tp_front_severity',
                'TP_FrontBonnet':                           'tp_front_bonnet_severity',
                'TP_FrontLeft':                             'tp_front_left_severity',
                'TP_FrontRight':                            'tp_front_right_severity',
                'TP_Left':                                  'tp_left_severity',
                'TP_LeftBackseat':                          'tp_left_back_seat_severity',
                'TP_LeftFrontWheel':                        'tp_left_front_wheel_severity',
                'TP_LeftMirror':                            'tp_left_mirror_severity',
                'TP_LeftRearWheel':                         'tp_left_rear_wheel_severity',
                'TP_LeftUnderside':                         'tp_left_underside_severity',
                'TP_Rear':                                  'tp_rear_severity',
                'TP_RearLeft':                              'tp_rear_left_severity',
                'TP_RearRight':                             'tp_rear_right_severity',
                'TP_RearWindowDamage':                      'tp_rear_window_damage_severity',
                'TP_Right':                                 'tp_right_severity',
                'TP_RightBackseat':                         'tp_right_back_seat_severity',
                'TP_RightFrontWheel':                       'tp_right_front_wheel_severity',
                'TP_RightMirror':                           'tp_right_mirror_severity',
                'TP_RightRearWheel':                        'tp_right_rear_wheel_severity',
                'TP_RightRoof':                             'tp_right_roof_severity',
                'TP_RightUnderside':                        'tp_right_underside_severity',
                'TP_RoofDamage':                            'tp_roof_damage_severity',
                'TP_UnderbodyDamage':                       'tp_underbody_damage_severity',
                'TP_WindscreenDamage':                      'tp_windscreen_damage_severity'
            }
        
    def __init__(self, rename_map=None):
        self.rename_map = rename_map or self.default_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df.rename(columns=self.rename_map)
        renamed_columns = list(RenameColumnsTransformer().default_map.values())
        df = df[renamed_columns]
        return df

def standardize_pandas_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all numeric columns to float64 and all other columns to string.

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        pd.DataFrame: DataFrame with consistent float64 and string dtypes
    """
    df_standardized = df.copy()

    for col in df_standardized.columns:
        col_dtype = df_standardized[col].dtype

        if pd.api.types.is_numeric_dtype(col_dtype):
            df_standardized[col] = df_standardized[col].astype(np.float64)
        else:
            df_standardized[col] = df_standardized[col].astype(str)

    return df_standardized

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions End

# COMMAND ----------

feature_table_name = "prod_dsexp_auxiliarydata.third_party_capture.ttr_df"
df = spark.table(feature_table_name).drop('ExhaustDamaged', 'HailDamage', 'ManufacturerDescription').toPandas()

# COMMAND ----------

rename_map = {
    'DaysInRepair':                   'days_in_repair',
    'IncidentCauseDescription':       'incident_cause_description',
    'BootOpens':                      'boot_opens',
    'DeployedAirbags':                'deployed_airbags',
    'Front':                          'front_severity',
    'FrontBonnet':                    'front_bonnet_severity',
    'FrontLeft':                      'front_left_severity',
    'FrontRight':                     'front_right_severity',
    'Left':                           'left_severity',
    'LeftBackseat':                   'left_back_seat_severity',
    'LeftFrontWheel':                 'left_front_wheel_severity',
    'LeftMirror':                     'left_mirror_severity',
    'LeftRearWheel':                  'left_rear_wheel_severity',
    'LeftUnderside':                  'left_underside_severity',
    'Rear':                           'rear_severity',
    'RearLeft':                       'rear_left_severity',
    'RearRight':                      'rear_right_severity',
    'RearWindowDamage':               'rear_window_damage_severity',
    'Right':                          'right_severity',
    'RightBackseat':                  'right_back_seat_severity',
    'RightFrontWheel':                'right_front_wheel_severity',
    'RightMirror':                    'right_mirror_severity',
    'RightRearWheel':                 'right_rear_wheel_severity',
    'RightRoof':                      'right_roof_severity',
    'RightUnderside':                 'right_underside_severity',
    'RoofDamage':                     'roof_damage_severity',
    'UnderbodyDamage':                'underbody_damage_severity',
    'WindscreenDamage':               'windscreen_damage_severity',
    'DoorsOpen':                      'doors_open',
    'Driveable':                      'driveable_damage_assessment',
#    'EngineDamage':                   'engine_damage',
    'LightsDamaged':                  'lights_damaged',
    'PanelGaps':                      'panel_gaps',
#    'RadiatorDamaged':                'radiator_damaged',
    'SharpEdges':                     'sharp_edges',
    'WheelsDamaged':                  'wheels_damaged',
#    'WingMirrorDamaged':              'wing_mirror_damaged',
    'Doors':                          'number_of_doors',
    'EngineCapacity':                 'engine_capacity',
    'Seats':                          'number_of_seats',
    'Value':                          'vehicle_value',
    'DA_DR':                          'da_dr',
    'DA_DTL':                         'da_dtl',
    'DA_UTL':                         'da_utl',
    'DA_UR':                          'da_ur',
    'DA_O':                           'da_o',
    'TimeToNotify':                   'time_to_notify',
    'VehicleAge':                     'vehicle_age',
    'BodyKey_01':                     'body_key_01',
    'BodyKey_02':                     'body_key_02',
    'BodyKey_03':                     'body_key_03',
    'BodyKey_04':                     'body_key_04',
#    'FuelKey_01':                     'fuel_key_01',
#    'FuelKey_02':                     'fuel_key_02',
    'Nature_PH':                      'first_party_confirmed',
    'Notified_DOW':                   'notified_day_of_week',
    'Notified_Day':                   'notified_day',
    'Notified_Month':                 'notified_month',
    'Notified_Year':                  'notified_year',
    'Incident_DOW':                   'incident_day_of_week',
    'Incident_Day':                   'incident_day',
    'PostcodeArea':                   'postcode_area',
    'DamageSev_Total':                'damage_sev_total',
    'DamageSev_Count':                'damage_sev_count',
    'DamageSev_Mean':                 'damage_sev_mean',
    'DamageSev_Max':                  'damage_sev_max',
}

for old_col, new_col in rename_map.items(): 
    df = df.rename(columns={old_col: new_col})
    renamed_columns = list(rename_map.values())
df_1 = df[renamed_columns]

df_1['body_key'] = df_1[['body_key_01', 'body_key_02', 'body_key_03', 'body_key_04']].idxmax(axis=1)
df_1['body_key'] = df_1['body_key'].apply(lambda x: x.split('_0')[-1]).astype(int)

columns = [
    'boot_opens',
    'deployed_airbags',
    'front_severity',
    'front_bonnet_severity',
    'front_left_severity',
    'front_right_severity',
    'left_severity',
    'left_back_seat_severity',
    'left_front_wheel_severity',
    'left_mirror_severity',
    'left_rear_wheel_severity',
    'left_underside_severity',
    'rear_severity',
    'rear_left_severity',
    'rear_right_severity',
    'rear_window_damage_severity',
    'right_severity',
    'right_back_seat_severity',
    'right_front_wheel_severity',
    'right_mirror_severity',
    'right_rear_wheel_severity',
    'right_roof_severity',
    'right_underside_severity',
    'roof_damage_severity',
    'underbody_damage_severity',
    'windscreen_damage_severity',
    'doors_open',
    'driveable_damage_assessment',
    'lights_damaged',
    'panel_gaps',
    'sharp_edges',
    'wheels_damaged',
    'number_of_doors',
    'engine_capacity',
    'number_of_seats',
    'vehicle_value',
    'vehicle_age',
    'body_key_01',
    'body_key_02',
    'body_key_03',
    'body_key_04',
]

for column in columns:
    df_1.loc[:, f'fp_{column}'] = df_1.apply(lambda row: row[column] if row['first_party_confirmed'] == 1 else 0, axis=1)
    df_1.loc[:, f'tp_{column}'] = df_1.apply(lambda row: row[column] if row['first_party_confirmed'] == 0 else 0, axis=1)
    df_1 = df_1.drop(column, axis=1)

df_1 = df_1.drop(['first_party_confirmed', 'tp_panel_gaps', 'tp_sharp_edges', 'tp_number_of_seats', 'tp_vehicle_value', 'body_key'], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training & Tracking
# MAGIC
# MAGIC This section of the notebook loops through each of the columns to be modelled, creates a model for each and tracks them via `MLFlow`.

# COMMAND ----------

ttr_model_name = "prod_dsexp_mlstore.third_party_capture.LGBMRegressor"
ttr_model_uri  = f"models:/{ttr_model_name}@champion"

# COMMAND ----------

# ─── 8. FIND THE LATEST VERSION ────────────────────────────────────────────────

client = MlflowClient()

version = get_latest_model_version(client, ttr_model_name)

# ─── 9. SET THE ALIAS “champion” ──────────────────────────────────────────────

client.set_registered_model_alias(ttr_model_name, "Champion", version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start of TPC

# COMMAND ----------

## API DUMMY DATA LOAD


import pandas as pd
import json

# Create JSON data
sample_data = [
    {
        "FP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP VehicleAge": 2015,
        "TP Engine Capacity": 2000,
        "TP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "InsurerName1": "Insurer A",
        "Notification Date": "2023-01-01",
        "IncidentCauseDescription": "Collision",
        "Incident Date": "2023-01-01",
        "Incident Postcode": "12345",
        "FP Kept at postcode": "54321",
        "TP Body Key": "Sedan",
        "TP Doors": 4,
        "Notification Method": "Telephone",
        "FP Deployed Airbags": "2",
        "FP Value": 15000,
        "FP Engine Capacity": 1800,
        "IncidentSubCauseDescription": "Rear-end",
        "Impact Speed Range": "TwentyOneToThirty",
        "Impact Speed Unit": "MPH",
        "TP Colour": "Red",
        "TP Mileage": 50000,
        "FP Body Key": "SUV",
        "FP Radiator Damaged": "yes",
        "TP Boot Opens": "yes",
        "FP Doors": 4,
        "IncidentUKCountry": "England",
        "FP Seats": 5,
        "FP Wheels Damaged": "yes",
        "FP Lights Damaged": "yes",
        "TP Driveable / TP Damage Assessment": "DriveableRepair",
        "TP Is Right Hand Drive": "Yes",
        "FP Doors Open": "yes",
        "TP Doors Open": "yes",
        "FP Panel Gaps": "yes",
        "TP NotOnMID": "Yes",
        "TP Vehicle Unattended": "yes",
        "Road Conditions": "Wet",
        "FP Boot Opens": "yes",
        "TP Wheels Damaged": "yes",
        "FP Sharp Edges": "yes",
        "TP Deployed Airbags": "2",
        "TP Lights Damaged": "yes",
        "FP Driveable / FP Damage Assessment": "DriveableRepair",
        "Weather Conditions": "Clear",
        "FP Registration": "ABC123",
        "Claim Reference Number": "CRN12345"
    },
    {
        "FP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP VehicleAge": 2015,
        "TP Engine Capacity": 2000,
        "TP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "InsurerName1": "Insurer A",
        "Notification Date": "2023-01-01",
        "IncidentCauseDescription": "Collision",
        "Incident Date": "2023-01-01",
        "Incident Postcode": "12345",
        "FP Kept at postcode": "54321",
        "TP Body Key": "Sedan",
        "TP Doors": 4,
        "Notification Method": "Telephone",
        "FP Deployed Airbags": "2",
        "FP Value": 15000,
        "FP Engine Capacity": 1800,
        "IncidentSubCauseDescription": "Rear-end",
        "Impact Speed Range": "TwentyOneToThirty",
        "Impact Speed Unit": "KMH",
        "TP Colour": "Red",
        "TP Mileage": 50000,
        "FP Body Key": "SUV",
        "FP Radiator Damaged": "yes",
        "TP Boot Opens": "yes",
        "FP Doors": 4,
        "IncidentUKCountry": "England",
        "FP Seats": 5,
        "FP Wheels Damaged": "yes",
        "FP Lights Damaged": "yes",
        "TP Driveable / TP Damage Assessment": "DriveableRepair",
        "TP Is Right Hand Drive": "Yes",
        "FP Doors Open": "yes",
        "TP Doors Open": "yes",
        "FP Panel Gaps": "yes",
        "TP NotOnMID": "Yes",
        "TP Vehicle Unattended": "yes",
        "Road Conditions": "Dry",
        "FP Boot Opens": "yes",
        "TP Wheels Damaged": "yes",
        "FP Sharp Edges": "yes",
        "TP Deployed Airbags": "2",
        "TP Lights Damaged": "yes",
        "FP Driveable / FP Damage Assessment": "DriveableRepair",
        "Weather Conditions": "Clear",
        "FP Registration": "ABC123",
        "Claim Reference Number": "CRN12345"
    },
    {
        "FP Splatmap Data": {
            "Front": "medium",
            "FrontBonnet": "heavy",
            "FrontLeft": "severe",
            "FrontRight": "minimal",
            "LeftUnderside": "medium",
            "LoadArea": "heavy",
            "Left": "severe",
            "LeftBackseat": "minimal",
            "LeftFrontWheel": "medium",
            "LeftMirror": "heavy",
            "LeftRearWheel": "severe",
            "LeftRoof": "minimal",
            "RightRoof": "medium",
            "Right": "heavy",
            "RightBackseat": "severe",
            "RightFrontWheel": "minimal",
            "RightMirror": "medium",
            "RightRearWheel": "heavy",
            "Rear": "severe",
            "RearLeft": "minimal",
            "RearRight": "medium",
            "RearWindowDamage": "heavy",
            "RightUnderside": "severe",
            "RoofDamage": "minimal",
            "WindscreenDamage": "medium",
            "UnderbodyDamage": "heavy"
        },
        "TP VehicleAge": 2018,
        "TP Engine Capacity": 2200,
        "TP Splatmap Data": {
            "Front": "medium",
            "FrontBonnet": "heavy",
            "FrontLeft": "severe",
            "FrontRight": "minimal",
            "LeftUnderside": "medium",
            "LoadArea": "heavy",
            "Left": "severe",
            "LeftBackseat": "minimal",
            "LeftFrontWheel": "medium",
            "LeftMirror": "heavy",
            "LeftRearWheel": "severe",
            "LeftRoof": "minimal",
            "RightRoof": "medium",
            "Right": "heavy",
            "RightBackseat": "severe",
            "RightFrontWheel": "minimal",
            "RightMirror": "medium",
            "RightRearWheel": "heavy",
            "Rear": "severe",
            "RearLeft": "minimal",
            "RearRight": "medium",
            "RearWindowDamage": "heavy",
            "RightUnderside": "severe",
            "RoofDamage": "minimal",
            "WindscreenDamage": "medium",
            "UnderbodyDamage": "heavy"
        },
        "InsurerName1": "Insurer B",
        "Notification Date": "2023-02-01",
        "IncidentCauseDescription": "Theft",
        "Incident Date": "2023-02-01",
        "Incident Postcode": "67890",
        "FP Kept at postcode": "09876",
        "TP Body Key": "Coupe",
        "TP Doors": 2,
        "Notification Method": "Claims Portal Web",
        "FP Deployed Airbags": "4",
        "FP Value": 20000,
        "FP Engine Capacity": 2500,
        "IncidentSubCauseDescription": "Break-in",
        "Impact Speed Range": "ThirtyOneToForty",
        "TP Colour": "Blue",
        "TP Mileage": 30000,
        "FP Body Key": "Convertible",
        "FP Radiator Damaged": "no",
        "TP Boot Opens": "no",
        "FP Doors": 2,
        "IncidentUKCountry": "Scotland",
        "FP Seats": 2,
        "FP Wheels Damaged": "no",
        "FP Lights Damaged": "no",
        "TP Driveable / TP Damage Assessment": "UnroadworthyRepair",
        "TP Is Right Hand Drive": "No",
        "FP Doors Open": "no",
        "TP Doors Open": "no",
        "FP Panel Gaps": "no",
        "TP NotOnMID": "No",
        "TP Vehicle Unattended": "no",
        "Road Conditions": "Wet",
        "FP Boot Opens": "no",
        "TP Wheels Damaged": "no",
        "FP Sharp Edges": "no",
        "TP Deployed Airbags": "4",
        "TP Lights Damaged": "no",
        "FP Driveable / FP Damage Assessment": "UnroadworthyRepair",
        "Weather Conditions": "Rain",
        "FP Registration": "XYZ789",
        "Claim Reference Number": "CRN67890"
    },
    {
        "FP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP VehicleAge": 2015,
        "TP Engine Capacity": 2000,
        "TP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "InsurerName1": "Insurer A",
        "Notification Date": "2023-01-01",
        "IncidentCauseDescription": "Collision",
        "Incident Date": "2023-01-01",
        "Incident Postcode": "12345",
        "FP Kept at postcode": "54321",
        "TP Body Key": "Sedan",
        "TP Doors": 4,
        "Notification Method": "Telephone",
        "FP Deployed Airbags": "2",
        "FP Value": 15000,
        "FP Engine Capacity": 1800,
        "IncidentSubCauseDescription": "Rear-end",
        "Impact Speed Range": "Stationary",
        "Impact Speed Unit": "KMH",
        "TP Colour": "Red",
        "TP Mileage": 50000,
        "FP Body Key": "SUV",
        "FP Radiator Damaged": "yes",
        "TP Boot Opens": "yes",
        "FP Doors": 4,
        "IncidentUKCountry": "England",
        "FP Seats": 5,
        "FP Wheels Damaged": "yes",
        "FP Lights Damaged": "yes",
        "TP Driveable / TP Damage Assessment": "DriveableRepair",
        "TP Is Right Hand Drive": "Yes",
        "FP Doors Open": "yes",
        "TP Doors Open": "yes",
        "FP Panel Gaps": "yes",
        "TP NotOnMID": "Yes",
        "TP Vehicle Unattended": "yes",
        "Road Conditions": "Dry",
        "FP Boot Opens": "yes",
        "TP Wheels Damaged": "yes",
        "FP Sharp Edges": "yes",
        "TP Deployed Airbags": "2",
        "TP Lights Damaged": "yes",
        "FP Driveable / FP Damage Assessment": "DriveableRepair",
        "Weather Conditions": "Clear",
        "FP Registration": "ABC123",
        "Claim Reference Number": "CRN12345"
    },
    {
        "FP Splatmap Data": {
            "Front": "medium",
            "FrontBonnet": "heavy",
            "FrontLeft": "severe",
            "FrontRight": "minimal",
            "LeftUnderside": "medium",
            "LoadArea": "heavy",
            "Left": "severe",
            "LeftBackseat": "minimal",
            "LeftFrontWheel": "medium",
            "LeftMirror": "heavy",
            "LeftRearWheel": "severe",
            "LeftRoof": "minimal",
            "RightRoof": "medium",
            "Right": "heavy",
            "RightBackseat": "severe",
            "RightFrontWheel": "minimal",
            "RightMirror": "medium",
            "RightRearWheel": "heavy",
            "Rear": "severe",
            "RearLeft": "minimal",
            "RearRight": "medium",
            "RearWindowDamage": "heavy",
            "RightUnderside": "severe",
            "RoofDamage": "minimal",
            "WindscreenDamage": "medium",
            "UnderbodyDamage": "heavy"
        },
        "TP VehicleAge": 2018,
        "TP Engine Capacity": 2200,
        "TP Splatmap Data": {
            "Front": "medium",
            "FrontBonnet": "heavy",
            "FrontLeft": "severe",
            "FrontRight": "minimal",
            "LeftUnderside": "medium",
            "LoadArea": "heavy",
            "Left": "severe",
            "LeftBackseat": "minimal",
            "LeftFrontWheel": "medium",
            "LeftMirror": "heavy",
            "LeftRearWheel": "severe",
            "LeftRoof": "minimal",
            "RightRoof": "medium",
            "Right": "heavy",
            "RightBackseat": "severe",
            "RightFrontWheel": "minimal",
            "RightMirror": "medium",
            "RightRearWheel": "heavy",
            "Rear": "severe",
            "RearLeft": "minimal",
            "RearRight": "medium",
            "RearWindowDamage": "heavy",
            "RightUnderside": "severe",
            "RoofDamage": "minimal",
            "WindscreenDamage": "medium",
            "UnderbodyDamage": "heavy"
        },
        "InsurerName1": "Insurer B",
        "Notification Date": "2023-02-01",
        "IncidentCauseDescription": "Theft",
        "Incident Date": "2023-02-01",
        "Incident Postcode": "67890",
        "FP Kept at postcode": "09876",
        "TP Body Key": "Coupe",
        "TP Doors": 2,
        "Notification Method": "Claims Portal Web",
        "FP Deployed Airbags": "4",
        "FP Value": 20000,
        "FP Engine Capacity": 2500,
        "IncidentSubCauseDescription": "Break-in",
        "Impact Speed Range": "ThirtyOneToFourty",
        "Impact Speed Unit": "MPH",
        "TP Colour": "Blue",
        "TP Mileage": 30000,
        "FP Body Key": "Convertible",
        "FP Radiator Damaged": "no",
        "TP Boot Opens": "no",
        "FP Doors": 2,
        "IncidentUKCountry": "Scotland",
        "FP Seats": 2,
        "FP Wheels Damaged": "no",
        "FP Lights Damaged": "no",
        "TP Driveable / TP Damage Assessment": "UnroadworthyRepair",
        "TP Is Right Hand Drive": "No",
        "FP Doors Open": "no",
        "TP Doors Open": "no",
        "FP Panel Gaps": "no",
        "TP NotOnMID": "No",
        "TP Vehicle Unattended": "no",
        "Road Conditions": "Wet",
        "FP Boot Opens": "no",
        "TP Wheels Damaged": "no",
        "FP Sharp Edges": "no",
        "TP Deployed Airbags": "4",
        "TP Lights Damaged": "no",
        "FP Driveable / FP Damage Assessment": "UnroadworthyRepair",
        "Weather Conditions": "Rain",
        "FP Registration": "XYZ789",
        "Claim Reference Number": "CRN67890"
    },
    {
        "FP Splatmap Data": {
            "Front": "heavy",
            "FrontBonnet": "severe",
            "FrontLeft": "minimal",
            "FrontRight": "medium",
            "LeftUnderside": "heavy",
            "LoadArea": "severe",
            "Left": "minimal",
            "LeftBackseat": "medium",
            "LeftFrontWheel": "heavy",
            "LeftMirror": "severe",
            "LeftRearWheel": "minimal",
            "LeftRoof": "medium",
            "RightRoof": "heavy",
            "Right": "severe",
            "RightBackseat": "minimal",
            "RightFrontWheel": "medium",
            "RightMirror": "heavy",
            "RightRearWheel": "severe",
            "Rear": "minimal",
            "RearLeft": "medium",
            "RearRight": "heavy",
            "RearWindowDamage": "severe",
            "RightUnderside": "minimal",
            "RoofDamage": "medium",
            "WindscreenDamage": "heavy",
            "UnderbodyDamage": "severe"
        },
        "TP VehicleAge": 2017,
        "TP Engine Capacity": 2100,
        "TP Splatmap Data": {
            "Front": "heavy",
            "FrontBonnet": "severe",
            "FrontLeft": "minimal",
            "FrontRight": "medium",
            "LeftUnderside": "heavy",
            "LoadArea": "severe",
            "Left": "minimal",
            "LeftBackseat": "medium",
            "LeftFrontWheel": "heavy",
            "LeftMirror": "severe",
            "LeftRearWheel": "minimal",
            "LeftRoof": "medium",
            "RightRoof": "heavy",
            "Right": "severe",
            "RightBackseat": "minimal",
            "RightFrontWheel": "medium",
            "RightMirror": "heavy",
            "RightRearWheel": "severe",
            "Rear": "minimal",
            "RearLeft": "medium",
            "RearRight": "heavy",
            "RearWindowDamage": "severe",
            "RightUnderside": "minimal",
            "RoofDamage": "medium",
            "WindscreenDamage": "heavy",
            "UnderbodyDamage": "severe"
        },
        "InsurerName1": "Insurer C",
        "Notification Date": "2023-03-01",
        "IncidentCauseDescription": "Fire",
        "Incident Date": "2023-03-01",
        "Incident Postcode": "11223",
        "FP Kept at postcode": "33211",
        "TP Body Key": "Hatchback",
        "TP Doors": 5,
        "Notification Method": "Crash Detector Notification",
        "FP Deployed Airbags": "3",
        "FP Value": 18000,
        "FP Engine Capacity": 2000,
        "IncidentSubCauseDescription": "Engine Fire",
        "Impact Speed Range": "FourtyOneToFifty",
        "Impact Speed Unit": "MPH",
        "TP Colour": "Green",
        "TP Mileage": 40000,
        "FP Body Key": "Truck",
        "FP Radiator Damaged": "yes",
        "TP Boot Opens": "yes",
        "FP Doors": 2,
        "IncidentUKCountry": "Wales",
        "FP Seats": 3,
        "FP Wheels Damaged": "yes",
        "FP Lights Damaged": "yes",
        "TP Driveable / TP Damage Assessment": "DriveableTotalLoss",
        "TP Is Right Hand Drive": "Yes",
        "FP Doors Open": "yes",
        "TP Doors Open": "yes",
        "FP Panel Gaps": "yes",
        "TP NotOnMID": "Yes",
        "TP Vehicle Unattended": "yes",
        "Road Conditions": "Muddy",
        "FP Boot Opens": "yes",
        "TP Wheels Damaged": "yes",
        "FP Sharp Edges": "yes",
        "TP Deployed Airbags": "3",
        "TP Lights Damaged": "yes",
        "FP Driveable / FP Damage Assessment": "DriveableTotalLoss",
        "Weather Conditions": "Fog",
        "FP Registration": "LMN456",
        "Claim Reference Number": "CRN11223"
    },
    {
        "FP Splatmap Data": {
            "Front": "severe",
            "FrontBonnet": "minimal",
            "FrontLeft": "medium",
            "FrontRight": "heavy",
            "LeftUnderside": "severe",
            "LoadArea": "minimal",
            "Left": "medium",
            "LeftBackseat": "heavy",
            "LeftFrontWheel": "severe",
            "LeftMirror": "minimal",
            "LeftRearWheel": "medium",
            "LeftRoof": "heavy",
            "RightRoof": "severe",
            "Right": "minimal",
            "RightBackseat": "medium",
            "RightFrontWheel": "heavy",
            "RightMirror": "severe",
            "RightRearWheel": "minimal",
            "Rear": "medium",
            "RearLeft": "heavy",
            "RearRight": "severe",
            "RearWindowDamage": "minimal",
            "RightUnderside": "medium",
            "RoofDamage": "heavy",
            "WindscreenDamage": "severe",
            "UnderbodyDamage": "minimal"
        },
        "TP VehicleAge": 2016,
        "TP Engine Capacity": 2300,
        "TP Splatmap Data": {
            "Front": "severe",
            "FrontBonnet": "minimal",
            "FrontLeft": "medium",
            "FrontRight": "heavy",
            "LeftUnderside": "severe",
            "LoadArea": "minimal",
            "Left": "medium",
            "LeftBackseat": "heavy",
            "LeftFrontWheel": "severe",
            "LeftMirror": "minimal",
            "LeftRearWheel": "medium",
            "LeftRoof": "heavy",
            "RightRoof": "severe",
            "Right": "minimal",
            "RightBackseat": "medium",
            "RightFrontWheel": "heavy",
            "RightMirror": "severe",
            "RightRearWheel": "minimal",
            "Rear": "medium",
            "RearLeft": "heavy",
            "RearRight": "severe",
            "RearWindowDamage": "minimal",
            "RightUnderside": "medium",
            "RoofDamage": "heavy",
            "WindscreenDamage": "severe",
            "UnderbodyDamage": "minimal"
        },
        "InsurerName1": "Insurer D",
        "Notification Date": "2023-04-01",
        "IncidentCauseDescription": "Flood",
        "Incident Date": "2023-04-01",
        "Incident Postcode": "44556",
        "FP Kept at postcode": "66554",
        "TP Body Key": "SUV",
        "TP Doors": 4,
        "Notification Method": "Inbound Correspondence",
        "FP Deployed Airbags": "All",
        "FP Value": 25000,
        "FP Engine Capacity": 3000,
        "IncidentSubCauseDescription": "Water Damage",
        "Impact Speed Range": "FiftyOneToFifty",
        "Impact Speed Unit": "MPH",
        "TP Colour": "Black",
        "TP Mileage": 60000,
        "FP Body Key": "Van",
        "FP Radiator Damaged": "no",
        "TP Boot Opens": "no",
        "FP Doors": 4,
        "IncidentUKCountry": "Northern Ireland",
        "FP Seats": 4,
        "FP Wheels Damaged": "no",
        "FP Lights Damaged": "no",
        "TP Driveable / TP Damage Assessment": "UnroadworthyTotalLoss",
        "TP Is Right Hand Drive": "No",
        "FP Doors Open": "no",
        "TP Doors Open": "no",
        "FP Panel Gaps": "no",
        "TP NotOnMID": "No",
        "TP Vehicle Unattended": "no",
        "Road Conditions": "Icy/Snowy",
        "FP Boot Opens": "no",
        "TP Wheels Damaged": "no",
        "FP Sharp Edges": "no",
        "TP Deployed Airbags": "All",
        "TP Lights Damaged": "no",
        "FP Driveable / FP Damage Assessment": "UnroadworthyTotalLoss",
        "Weather Conditions": "Snow",
        "FP Registration": "OPQ789",
        "Claim Reference Number": "CRN44556"
    }
]

# Convert JSON data to pandas DataFrame
df = pd.DataFrame(sample_data)

display(df)

# COMMAND ----------

numerical_features = [n for n in df.select_dtypes(include=['int64']).columns.tolist() if n != 'days_in_repair']
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Define input schema manually from Pandas DataFrame
input_schema = Schema(
    [ColSpec("double", col) for col in df[numerical_features].columns] +
    [ColSpec("string", col) for col in df[categorical_features].columns]
)

# Define output schema (assume regression output)
output_schema = Schema([
    ColSpec("double", 'capture_benefit')
])

# COMMAND ----------

## CLASS FUNCTION BUILD


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline

def standardize_pandas_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all numeric columns to float64 and all other columns to string.

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        pd.DataFrame: DataFrame with consistent float64 and string dtypes
    """
    df_standardized = df.copy()

    for col in df_standardized.columns:
        col_dtype = df_standardized[col].dtype

        if pd.api.types.is_numeric_dtype(col_dtype):
            df_standardized[col] = df_standardized[col].astype(np.float64)
        else:
            df_standardized[col] = df_standardized[col].astype(str)

    return df_standardized

# ─── 1. Splatmap Extraction ─────────────────────────────────────────────────────

class SplatmapExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, splatmap_cols=None):
        if splatmap_cols is None:
            splatmap_cols = [
                "Front", 
                "FrontBonnet", 
                "FrontLeft", 
                "FrontRight", 
                "Left", 
                "LeftBackseat",
                "LeftFrontWheel", 
                "LeftMirror", 
                "LeftRearWheel", 
                "LeftRoof", 
                "LeftUnderside",
                "LoadArea", 
                "Rear", 
                "RearLeft", 
                "RearRight", 
                "RearWindowDamage", 
                "Right",
                "RightBackseat", 
                "RightFrontWheel", 
                "RightMirror", 
                "RightRearWheel", 
                "RightRoof",
                "RightUnderside", 
                "RoofDamage", 
                "UnderbodyDamage", 
                "WindscreenDamage"
            ]
        self.splatmap_cols = splatmap_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'FP Splatmap Data' in df.columns and 'TP Splatmap Data' in df.columns:
            for col in self.splatmap_cols:
                df[f'FP_{col}'] = df['FP Splatmap Data'].apply(
                    lambda x: x.get(col) if isinstance(x, dict) else np.nan)
                df[f'TP_{col}'] = df['TP Splatmap Data'].apply(
                    lambda x: x.get(col) if isinstance(x, dict) else np.nan)
            df = df.drop(columns=['FP Splatmap Data', 'TP Splatmap Data'], errors='ignore')
        return df
    

# ─── 2. Column Renaming ─────────────────────────────────────────────────────────

class RenameColumnsTransformer(BaseEstimator, TransformerMixin):
    # allow injection of a custom map if desired
    default_map  = {
                'TP VehicleAge': 'year_of_manufacture',
                'TP Engine Capacity': 'tp_engine_capacity',
                'InsurerName1': 'insurer_name',
                'Notification Date': 'notification_date',
                'IncidentCauseDescription': 'incident_cause_description',
                'Incident Date': 'incident_date',
                'Incident Postcode': 'incident_postcode',
                'FP Kept at postcode': 'vehicle_kept_at_postcode',
                'TP Body Key': 'tp_body_key',
                'TP Doors': 'tp_number_of_doors',
                'Notification Method': 'notification_method',
                'FP Deployed Airbags': 'fp_deployed_airbags',
                'FP Value': 'fp_vehicle_value',
                'FP Engine Capacity': 'fp_engine_capacity',
                'IncidentSubCauseDescription': 'incident_sub_cause_description',
                'Impact Speed Range': 'impact_speed_range',
                'Impact Speed Unit': 'impact_speed_unit',
                'TP Colour': 'tp_colour',
                'TP Mileage': 'tp_mileage',
                'FP Body Key': 'fp_body_key',
                'FP Radiator Damaged': 'fp_radiator_damaged',
                'TP Boot Opens': 'tp_boot_opens',
                'FP Doors': 'fp_number_of_doors',
                'IncidentUKCountry': 'incident_uk_country',
                'FP Seats': 'fp_number_of_seats',
                'FP Wheels Damaged': 'fp_wheels_damaged',
                'FP Lights Damaged': 'fp_lights_damaged',
                'TP Driveable / TP Damage Assessment?': 'tp_driveable_damage_assessment',
                'TP Is Right Hand Drive': 'tp_right_hand_drive',
                'FP Doors Open': 'fp_doors_open',
                'TP Doors Open': 'tp_doors_open',
                'FP Panel Gaps': 'fp_panel_gaps',
                'TP NotOnMID': 'tp_not_on_mid',
                'TP Vehicle Unattended': 'tp_vehicle_unattended',
                'Road Conditions': 'road_conditions',
                'FP Boot Opens': 'fp_boot_opens',
                'TP Wheels Damaged': 'tp_wheels_damaged',
                'FP Sharp Edges': 'fp_sharp_edges',
                'TP Deployed Airbags': 'tp_deployed_airbags',
                'TP Lights Damaged': 'tp_lights_damaged',
                'FP Driveable / FP Damage Assessment': 'fp_driveable_damage_assessment',
                'Weather Conditions': 'weather_conditions',
                'FP Registration': 'fp_registration',
                'Claim Reference Number': 'claim_reference_number',
                # Adding all FP severity renames from the original user prompt
                'FP_Front': 'fp_front_severity',
                'FP_FrontBonnet': 'fp_front_bonnet_severity',
                'FP_FrontLeft': 'fp_front_left_severity',
                'FP_FrontRight': 'fp_front_right_severity',
                'FP_Left': 'fp_left_severity',
                'FP_LeftBackseat': 'fp_left_back_seat_severity',
                'FP_LeftFrontWheel': 'fp_left_front_wheel_severity',
                'FP_LeftMirror': 'fp_left_mirror_severity',
                'FP_LeftRearWheel': 'fp_left_rear_wheel_severity',
                'FP_LeftRoof': 'fp_left_roof_severity',
                'FP_LeftUnderside': 'fp_left_underside_severity',
                'FP_LoadArea': 'fp_load_area_severity',
                'FP_Rear': 'fp_rear_severity',
                'FP_RearLeft': 'fp_rear_left_severity',
                'FP_RearRight': 'fp_rear_right_severity',
                'FP_RearWindowDamage': 'fp_rear_window_damage_severity',
                'FP_Right': 'fp_right_severity',
                'FP_RightBackseat': 'fp_right_back_seat_severity',
                'FP_RightFrontWheel': 'fp_right_front_wheel_severity',
                'FP_RightMirror': 'fp_right_mirror_severity',
                'FP_RightRearWheel': 'fp_right_rear_wheel_severity',
                'FP_RightRoof': 'fp_right_roof_severity',
                'FP_RightUnderside': 'fp_right_underside_severity',
                'FP_RoofDamage': 'fp_roof_damage_severity',
                'FP_UnderbodyDamage': 'fp_underbody_damage_severity',
                'FP_WindscreenDamage': 'fp_windscreen_damage_severity',
                # Adding all TP severity renames from the original user prompt
                'TP_Front': 'tp_front_severity',
                'TP_FrontBonnet': 'tp_front_bonnet_severity',
                'TP_FrontLeft': 'tp_front_left_severity',
                'TP_FrontRight': 'tp_front_right_severity',
                'TP_Left': 'tp_left_severity',
                'TP_LeftBackseat': 'tp_left_back_seat_severity',
                'TP_LeftFrontWheel': 'tp_left_front_wheel_severity',
                'TP_LeftMirror': 'tp_left_mirror_severity',
                'TP_LeftRearWheel': 'tp_left_rear_wheel_severity',
                'TP_LeftRoof': 'tp_left_roof_severity',
                'TP_LeftUnderside': 'tp_left_underside_severity',
                'TP_LoadArea': 'tp_load_area_severity',
                'TP_Rear': 'tp_rear_severity',
                'TP_RearLeft': 'tp_rear_left_severity',
                'TP_RearRight': 'tp_rear_right_severity',
                'TP_RearWindowDamage': 'tp_rear_window_damage_severity',
                'TP_Right': 'tp_right_severity',
                'TP_RightBackseat': 'tp_right_back_seat_severity',
                'TP_RightFrontWheel': 'tp_right_front_wheel_severity',
                'TP_RightMirror': 'tp_right_mirror_severity',
                'TP_RightRearWheel': 'tp_right_rear_wheel_severity',
                'TP_RightRoof': 'tp_right_roof_severity',
                'TP_RightUnderside': 'tp_right_underside_severity',
                'TP_RoofDamage': 'tp_roof_damage_severity',
                'TP_UnderbodyDamage': 'tp_underbody_damage_severity',
                'TP_WindscreenDamage': 'tp_windscreen_damage_severity'
            }
        
    def __init__(self, rename_map=None):
        self.rename_map = rename_map or self.default_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df.rename(columns=self.rename_map)
        renamed_columns = list(RenameColumnsTransformer().default_map.values())
        return df


# ─── 3. Mode & Median Imputation ─────────────────────────────────────────────────

class ModeMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.modes_ = {}
        self.medians_ = {}

    def fit(self, X, y=None):
        self.modes_['tp_body_key'] = X['tp_body_key'].mode(dropna=True)[0]
        self.modes_['impact_speed_range'] = X['impact_speed_range'].mode(dropna=True)[0]
        self.modes_['insurer_name'] = X['insurer_name'].fillna('Unknown').mode(dropna=True)[0]
        self.medians_['tp_engine_capacity'] = X['tp_engine_capacity'].median(dropna=True)
        return self

    def transform(self, X):
        df = X.copy()
        df['tp_body_key']       = df['tp_body_key'].fillna(self.modes_['tp_body_key'])
        df['impact_speed_range'] = df['impact_speed_range'].fillna(self.modes_['impact_speed_range'])
        df['insurer_name']       = df['insurer_name'].fillna(self.modes_['insurer_name'])
        df['tp_engine_capacity'] = df['tp_engine_capacity'].fillna(self.medians_['tp_engine_capacity'])
        return df


class GeneralPurposeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_cols_ = None
        self.categorical_cols_ = None
        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.fitted_columns_ = None # To store column names and order from fitting

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        self.fitted_columns_ = X.columns.tolist()

        # Identify numerical columns (float64 as per your standardize_pandas_schema)
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        
        # Identify categorical columns (object/string as per your standardize_pandas_schema)
        self.categorical_cols_ = X.select_dtypes(include='object').columns.tolist()

        if self.numerical_cols_:
            self.num_imputer_ = SimpleImputer(strategy='median')
            self.num_imputer_.fit(X[self.numerical_cols_])
        
        if self.categorical_cols_:
            self.cat_imputer_ = SimpleImputer(strategy='most_frequent')
            self.cat_imputer_.fit(X[self.categorical_cols_])
            
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        if self.fitted_columns_ is None:
            raise RuntimeError("Imputer has not been fitted yet. Call fit() first.")

        df = X.copy()

        # Impute numerical columns
        if self.num_imputer_ and self.numerical_cols_:
            # Filter for numerical columns that are actually present in X
            num_cols_in_X = [col for col in self.numerical_cols_ if col in df.columns]
            if num_cols_in_X:
                df[num_cols_in_X] = self.num_imputer_.transform(df[num_cols_in_X])
        
        # Impute categorical columns
        if self.cat_imputer_ and self.categorical_cols_:
            # Filter for categorical columns that are actually present in X
            cat_cols_in_X = [col for col in self.categorical_cols_ if col in df.columns]
            if cat_cols_in_X:
                df[cat_cols_in_X] = self.cat_imputer_.transform(df[cat_cols_in_X])
                
        return df


# ─── 4. Impact Speed Conversion & Bucketing ──────────────────────────────────────

class ImpactSpeedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # map buckets to representative speeds
        self.speed_map = {
            'Stationary': 0, 'ZeroToSix': 1, 'SevenToFourteen': 7,
            'FifteenToTwenty': 15, 'TwentyOneToThirty': 21, 'ThirtyOneToFourty': 31,
            'FourtyOneToFifty': 41, 'FiftyOneToSixty': 51, 'SixtyOneToSeventy': 61,
            'OverSeventy': 71
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['impact_speed_range'] = df['impact_speed_range'].map(self.speed_map).fillna(-1)
        df['impact_speed_unit'] = df['impact_speed_unit'].fillna('MPH')
        # convert KMH → MPH
        df['impact_speed_range'] = np.where(
            df['impact_speed_unit']=='KMH',
            df['impact_speed']/1.61,
            df['impact_speed']
        )
        # re-bucket numeric speed back into original bins
        def bucket(x):
            if   x==0:   return 0
            elif x<=6:  return 1
            elif x<15:  return 7
            elif x<21:  return 15
            elif x<31:  return 21
            elif x<41:  return 31
            elif x<51:  return 41
            elif x<61:  return 51
            elif x<71:  return 61
            elif x>=71: return 71
            else:       return -1
        df['impact_speed_range'] = df['impact_speed_range'].apply(bucket)
        return df.drop(columns=['impact_speed_range', 'impact_speed_unit'])


# ─── 5. Airbag Mapping ───────────────────────────────────────────────────────────

class AirbagCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.airbag_map = {'None': 0, '1':1, '2':2, '3':3, '4':4, 'All':5}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['tp_deployed_airbags'] = df['tp_deployed_airbags'].map(self.airbag_map).fillna(-1)
        df['fp_deployed_airbags'] = df['fp_deployed_airbags'].map(self.airbag_map).fillna(-1)
        return df


# ─── 6. Car Characteristics Mapping ───────────────────────────────────────────────────────────	

class CarCharacteristicTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df['tp_engine_capacity'] = (df['tp_engine_capacity']=='yes').astype(int)
        df['tp_number_of_doors'] = (df['tp_number_of_doors']=='yes').astype(int)
        df['tp_boot_opens'] = (df['tp_boot_opens']=='yes').astype(int)
        df['tp_doors_open'] = (df['tp_doors_open']=='yes').astype(int)
        df['tp_not_on_mid'] = (df['tp_not_on_mid']=='yes').astype(int)
        df['tp_vehicle_unattended'] = (df['tp_vehicle_unattended']=='yes').astype(int)
        df['tp_wheels_damaged'] = (df['tp_wheels_damaged']=='yes').astype(int)
        df['tp_lights_damaged'] = (df['tp_lights_damaged']=='yes').astype(int)

        df['da_dr'] = (df['tp_driveable_damage_assessment']=='DriveableRepair').astype(int)
        df['da_dtl'] = (df['tp_driveable_damage_assessment']=='DriveableTotalLoss').astype(int)
        df['da_utl'] = (df['tp_driveable_damage_assessment']=='UnroadworthyTotalLoss').astype(int)
        df['da_ur'] = (df['tp_driveable_damage_assessment']=='UnroadworthyRepair').astype(int)
        df['da_o'] = ~df['tp_driveable_damage_assessment'].isin([
                                                    'DriveableRepair', 
                                                    'DriveableTotalLoss', 
                                                    'UnroadworthyTotalLoss', 
                                                    'UnroadworthyRepair'
                                                ]).astype(int)

        return df

# ─── 7. Date Features ───────────────────────────────────────────────────────────

class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['time_to_notify'] = (df['notification_date'] - df['incident_date']).dt.days
        df['time_to_notify'] = df['time_to_notify'].clip(0, 30).fillna(0)
        
        df['notified_day_of_week'] = df['notification_date'].dt.dayofweek.apply(lambda x: 7 if x == 0 else x)
        df['notified_day'] = df['notification_date'].dt.day
        df['notified_month'] = df['notification_date'].dt.month
        df['notified_year'] = df['notification_date'].dt.year
        
        df['incident_day_of_week'] = df['incident_date'].dt.dayofweek.apply(lambda x: 7 if x == 0 else x)
        df['incident_day'] = df['incident_date'].dt.day
        
        return df


# ─── 8. Vehicle Age ─────────────────────────────────────────────────────────────

class VehicleAgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['vehicle_age'] = df['incident_date'].dt.year - df['year_of_manufacture'].astype(int)
        df['vehicle_age'] = df['vehicle_age'].where(df['vehicle_age']<=30, 0).fillna(0)
        return df.drop(['notification_date', 'incident_date'], axis = 1)


# ─── 9. Right-Hand Drive Flag ──────────────────────────────────────────────────

class RHDTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['tp_right_hand_drive'] = (df['tp_right_hand_drive']=='R').astype(int)
        return df


# ─── 10. Body Key One-Hot Encoding ──────────────────────────────────────────────

class BodyKeyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, keys=None):
        # only encode a fixed set of styles as in your example
        self.keys = keys or [
            '5 Door Hatchback','5 Door Estate','4 Door Saloon','3 Door Hatchback'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for i, style in enumerate(self.keys, start=1):
            df[f'tp_body_key_{i}'] = (df['tp_body_key']==style).astype(int)
        return df


# ─── 11. Postcode Area Extraction ─────────────────────────────────────────────

class PostcodeAreaExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['postcode_area'] = df['postcode_area']\
            .str.extract(r'^([A-Za-z]+)')[0]\
            .fillna('zz')
        return df


# ─── 12. Drop Any Leftovers ─────────────────────────────────────────────────────

class DropUnusedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop=None):
        self.to_drop = to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.to_drop, errors='ignore')


# ─── 13. Damage Recorded and Assessment ─────────────────────────────────────────────────────

class DamageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, severity_columns=None):
        self.severity_columns = severity_columns or [
            'tp_front_severity', 
            'tp_front_bonnet_severity', 
            'tp_front_left_severity', 
            'tp_front_right_severity', 
            'tp_left_severity', 
            'tp_left_back_seat_severity', 
            'tp_left_front_wheel_severity', 
            'tp_left_mirror_severity', 
            'tp_left_rear_wheel_severity', 
            'tp_left_roof_severity', 
            'tp_left_underside_severity', 
            'tp_load_area_severity', 
            'tp_rear_severity', 
            'tp_rear_left_severity', 
            'tp_rear_right_severity', 
            'tp_rear_window_damage_severity', 
            'tp_right_severity', 
            'tp_right_back_seat_severity', 
            'tp_right_front_wheel_severity', 
            'tp_right_mirror_severity', 
            'tp_right_rear_wheel_severity', 
            'tp_right_roof_severity', 
            'tp_right_underside_severity', 
            'tp_roof_damage_severity', 
            'tp_underbody_damage_severity', 
            'tp_windscreen_damage_severity'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Calculate damage recorded and assessed
        df['damage_recorded'] = df[self.severity_columns].notnull().sum(axis=1)
        df['damage_assessed'] = df['damage_recorded'].apply(lambda x: 1 if x > 0 else 0)
        
        # Scale damage severity
        def damage_scale(row_area, row_damageflag):
            if row_damageflag == 1:
                if row_area == 'minimal':
                    scale = 1
                elif row_area == 'medium':
                    scale = 2
                elif row_area == 'heavy':
                    scale = 3
                elif row_area == 'severe':
                    scale = 4
                elif row_area == 'unknown':
                    scale = -1
                else:
                    scale = 0
            else:
                scale = -1
            return scale

        for col_name in self.severity_columns:
            df[col_name] = df.apply(lambda row: damage_scale(row[col_name], row['damage_assessed']), axis=1).astype(int)
        
        return df


# ─── 14. Damage Severity ─────────────────────────────────────────────────────

class DamageSeverityCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, sev_damage=None):
        self.sev_damage = sev_damage or [
            'tp_front_severity', 
            'tp_front_bonnet_severity', 
            'tp_front_left_severity', 
            'tp_front_right_severity', 
            'tp_left_severity', 
            'tp_left_back_seat_severity', 
            'tp_left_front_wheel_severity', 
            'tp_left_mirror_severity', 
            'tp_left_rear_wheel_severity', 
            'tp_left_roof_severity', 
            'tp_left_underside_severity', 
            'tp_load_area_severity', 
            'tp_rear_severity', 
            'tp_rear_left_severity', 
            'tp_rear_right_severity', 
            'tp_rear_window_damage_severity', 
            'tp_right_severity', 
            'tp_right_back_seat_severity', 
            'tp_right_front_wheel_severity', 
            'tp_right_mirror_severity', 
            'tp_right_rear_wheel_severity', 
            'tp_right_roof_severity', 
            'tp_right_underside_severity', 
            'tp_roof_damage_severity', 
            'tp_underbody_damage_severity', 
            'tp_windscreen_damage_severity'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        df['damage_sev_total'] = df[self.sev_damage].sum(axis=1)
        df['damage_sev_count'] = df[self.sev_damage].gt(0).sum(axis=1)
        df['damage_sev_mean'] = df.apply(lambda row: row['damage_sev_total'] / row['damage_sev_count'] if row['damage_sev_count'] > 0 else 0, axis=1)
        df['damage_sev_max'] = df[self.sev_damage].max(axis=1)
        
        return df
    

class DaysInRepairPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_uri):
        self.model_uri = model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.input_cols = self.model.metadata.get_input_schema().input_names()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Coerce only numerical columns in input schema to float64
        for col in self.input_cols:
            if col in X_copy.columns and pd.api.types.is_numeric_dtype(X_copy[col]):
                X_copy[col] = X_copy[col].astype(np.float64)

        # Model prediction
        X_copy["days_in_repair"] = self.model.predict(X_copy[self.input_cols])
        return X_copy












class CaptureBenefitModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):

        print("Loading model context...")
        # Load the preprocessor pipeline (which includes all your transformers)
        self.preprocessor = mlflow.sklearn.load_model(context.artifacts["preprocessor_model_path"])
        print(f"Preprocessor loaded from: {context.artifacts['preprocessor_model_path']}")

        # Load the trained regressor model
        self.regressor = mlflow.sklearn.load_model(context.artifacts["regressor_model_path"])
        print(f"Regressor loaded from: {context.artifacts['regressor_model_path']}")

        # Load the quantile benefit multiplier table (expected as a CSV)
        quantile_table_path = context.artifacts["quantile_benefit_table_csv_path"]
        print(f"Attempting to load quantile benefit table from: {quantile_table_path}")
        self.quantile_benefit_df = pd.read_csv(quantile_table_path)

    def _get_quantile_benefit_multiplier(self, raw_capture_benefit_value: float) -> float:

        if self.quantile_benefit_df is None:
            print("Quantile benefit table not loaded. Returning default multiplier of 1.0.")
            return 1.0 # Should not happen if load_context is successful

        for _, row in self.quantile_benefit_df.iterrows():
            if row['capture_benefit_lower_bound'] <= raw_capture_benefit_value < row['capture_benefit_upper_bound']:
                return row['quant_benefit_multiplier']
        
        # If no range matches (e.g., value is exactly on the max upper_bound of the last interval,
        # or table doesn't cover all possibilities), define a fallback.
        # This might indicate an issue with the table's range definitions.
        # Consider logging this event.
        # print(f"Warning: No multiplier range found for raw_capture_benefit_value: {raw_capture_benefit_value}. Using default of 1.0.")
        # Check if it matches the last upper bound exactly (if it's inclusive)
        last_row = self.quantile_benefit_df.iloc[-1]
        if raw_capture_benefit_value == last_row['capture_benefit_upper_bound'] and last_row['capture_benefit_upper_bound'] == np.inf : # Special case for last interval potentially being inclusive of its start
             return last_row['quant_benefit_multiplier']

        return 1.0 # Default multiplier if no specific range is found or other conditions not met


    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:

        print(f"Received input with {model_input.shape[0]} rows and {model_input.shape[1]} columns.")
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # --- Prediction with CaptureSuccess_AdrienneVersion = 1 ---
        df_cap = input_df_standardized.copy()
        # Ensure the column name matches exactly what the preprocessor expects.
        # The value should be float if the column is treated as numeric by the scaler.
        df_cap['CaptureSuccess_AdrienneVersion'] = 1.0 

        # --- Prediction with CaptureSuccess_AdrienneVersion = 0 ---
        df_nocap = input_df_standardized.copy()
        df_nocap['CaptureSuccess_AdrienneVersion'] = 0.0

        # Preprocess both DataFrames using the loaded preprocessor pipeline
        # The preprocessor should handle all transformations (Splatmap, renaming, imputation,
        # speed, airbags, dates, OHE, scaling, DaysInRepairPredictor, etc.)
        print("Preprocessing data for 'capture' scenario...")
        try:
            processed_cap_features = self.preprocessor.transform(df_cap)
        except Exception as e:
            print(f"Error transforming df_cap: {e}")
            # Consider which columns are present in df_cap before transform
            # print(f"df_cap columns: {df_cap.columns.tolist()}")
            raise RuntimeError(f"Error during preprocessing 'capture' data: {e}")
        
        print("Preprocessing data for 'no capture' scenario...")
        try:
            processed_nocap_features = self.preprocessor.transform(df_nocap)
        except Exception as e:
            print(f"Error transforming df_nocap: {e}")
            raise RuntimeError(f"Error during preprocessing 'no capture' data: {e}")

        # Make predictions using the loaded regressor model
        print("Making predictions...")
        tppd_pred_capture = self.regressor.predict(processed_cap_features)
        tppd_pred_no_capture = self.regressor.predict(processed_nocap_features)

        # Calculate raw Capture_Benefit
        # (TPPD_Pred_No_Capture - TPPD_Pred_Capture)
        raw_capture_benefit = tppd_pred_no_capture - tppd_pred_capture
        print(f"Raw capture benefit calculated. Min: {np.min(raw_capture_benefit)}, Max: {np.max(raw_capture_benefit)}")


        # Adjust Capture_Benefit using the lookup table and multiplier
        # Apply the lookup for each value if raw_capture_benefit is an array (multiple input rows)
        if isinstance(raw_capture_benefit, np.ndarray):
            capture_benefit_adjusted = np.array([
                cb_raw * self._get_quantile_benefit_multiplier(cb_raw) for cb_raw in raw_capture_benefit
            ])
        elif isinstance(raw_capture_benefit, (int, float)): # Single prediction value
            capture_benefit_adjusted = raw_capture_benefit * self._get_quantile_benefit_multiplier(raw_capture_benefit)
        else:
            raise TypeError(f"Unexpected type for raw_capture_benefit: {type(raw_capture_benefit)}")
        
        print(f"Adjusted capture benefit calculated. Min: {np.min(capture_benefit_adjusted)}, Max: {np.max(capture_benefit_adjusted)}")

        # Return the final adjusted benefit as a pandas Series.
        # The problem states "returns the value in a variable called Capture_Benefit".
        # So, the output Series should be named 'Capture_Benefit'.
        output_series = pd.Series(capture_benefit_adjusted, name="Capture_Benefit")
        print("Prediction process complete.")
        return output_series

# COMMAND ----------

feature_table_name = "prod_dsexp_mlstore.third_party_capture.tpc_feature_table"

df = standardize_pandas_schema(spark.table(feature_table_name).toPandas())

# COMMAND ----------

# ─── 8. FIND THE LATEST VERSION ────────────────────────────────────────────────
tpc_model_name = "prod_dsexp_mlstore.third_party_capture.RandomForestRegressor" 
model_uri = f"models:/{tpc_model_name}@champion"

client = MlflowClient()

version = get_latest_model_version(client, tpc_model_name)

# ─── 9. SET THE ALIAS “champion” ──────────────────────────────────────────────

client.set_registered_model_alias(tpc_model_name, "Champion", version)

# COMMAND ----------

# ─── 10. LOAD THE MODEL BY ALIAS ──────────────────────────────────────────────

model_uri = f"models:/{tpc_model_name}@champion"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Display the model's output vs the actual for the first 10 rows
preds_first_10 = loaded_model.predict(X_test[:10])
actual_first_10 = y_test[:10]

results_df = pd.DataFrame({
    "Actual": actual_first_10,
    "Predicted": preds_first_10
})

display(results_df)

# COMMAND ----------

# Actual	Predicted
# 1974.56	4288.663137918081
# 9607.08	4286.565044483879
# 4565.13	4392.635943980873
# 9775.25	4392.635943980873
# 2339	4387.991892291505
# 3489.51	4787.024241272956
# 6925.38	4390.5378505466715
# 4528.32	4392.635943980873
# 3909.18	4392.635943980873
# 6600	4286.565044483879

# COMMAND ----------

## API DUMMY DATA LOAD


import pandas as pd
import json

# Create JSON data
sample_data = [
    {
        "FP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP VehicleAge": 2015,
        "TP Engine Capacity": 2000,
        "TP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP Damage Assessment": "DriveableRepair",
        "InsurerName1": "Insurer A",
        "Notification Date": "2023-01-01",
        "IncidentCauseDescription": "Collision",
        "Incident Date": "2023-01-01",
        "Incident Postcode": "12345",
        "FP Kept at postcode": "54321",
        "TP Body Key": "Sedan",
        "TP Doors": 4,
        "Notification Method": "Telephone",
        "FP Deployed Airbags": "2",
        "FP Value": 15000,
        "FP Engine Capacity": 1800,
        "IncidentSubCauseDescription": "Rear-end",
        "Impact Speed Range": "TwentyOneToThirty",
        "Impact Speed Unit": "MPH",
        "TP Colour": "Red",
        "TP Mileage": 50000,
        "FP Body Key": "SUV",
        "FP Radiator Damaged": "yes",
        "TP Boot Opens": "yes",
        "FP Doors": 4,
        "IncidentUKCountry": "England",
        "FP Seats": 5,
        "FP Wheels Damaged": "yes",
        "FP Lights Damaged": "yes",
        "TP Driveable / TP Damage Assessment?": "DriveableRepair",
        "TP Is Right Hand Drive": "Yes",
        "FP Doors Open": "yes",
        "TP Doors Open": "yes",
        "FP Panel Gaps": "yes",
        "TP NotOnMID": "Yes",
        "TP Vehicle Unattended": "yes",
        "Road Conditions": "Wet",
        "FP Boot Opens": "yes",
        "TP Wheels Damaged": "yes",
        "FP Sharp Edges": "yes",
        "TP Deployed Airbags": "2",
        "TP Lights Damaged": "yes",
        "FP Driveable / FP Damage Assessment": "DriveableRepair",
        "Weather Conditions": "Clear",
        "FP Registration": "ABC123",
        "Claim Reference Number": "CRN12345"
    },
    {
        "FP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP VehicleAge": 2015,
        "TP Engine Capacity": 2000,
        "TP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP Damage Assessment": "DriveableRepair",
        "InsurerName1": "Insurer A",
        "Notification Date": "2023-01-01",
        "IncidentCauseDescription": "Collision",
        "Incident Date": "2023-01-01",
        "Incident Postcode": "12345",
        "FP Kept at postcode": "54321",
        "TP Body Key": "Sedan",
        "TP Doors": 4,
        "Notification Method": "Telephone",
        "FP Deployed Airbags": "2",
        "FP Value": 15000,
        "FP Engine Capacity": 1800,
        "IncidentSubCauseDescription": "Rear-end",
        "Impact Speed Range": "TwentyOneToThirty",
        "Impact Speed Unit": "KMH",
        "TP Colour": "Red",
        "TP Mileage": 50000,
        "FP Body Key": "SUV",
        "FP Radiator Damaged": "yes",
        "TP Boot Opens": "yes",
        "FP Doors": 4,
        "IncidentUKCountry": "England",
        "FP Seats": 5,
        "FP Wheels Damaged": "yes",
        "FP Lights Damaged": "yes",
        "TP Driveable / TP Damage Assessment?": "DriveableRepair",
        "TP Is Right Hand Drive": "Yes",
        "FP Doors Open": "yes",
        "TP Doors Open": "yes",
        "FP Panel Gaps": "yes",
        "TP NotOnMID": "Yes",
        "TP Vehicle Unattended": "yes",
        "Road Conditions": "Dry",
        "FP Boot Opens": "yes",
        "TP Wheels Damaged": "yes",
        "FP Sharp Edges": "yes",
        "TP Deployed Airbags": "2",
        "TP Lights Damaged": "yes",
        "FP Driveable / FP Damage Assessment": "DriveableRepair",
        "Weather Conditions": "Clear",
        "FP Registration": "ABC123",
        "Claim Reference Number": "CRN12345"
    },
    {
        "FP Splatmap Data": {
            "Front": "medium",
            "FrontBonnet": "heavy",
            "FrontLeft": "severe",
            "FrontRight": "minimal",
            "LeftUnderside": "medium",
            "LoadArea": "heavy",
            "Left": "severe",
            "LeftBackseat": "minimal",
            "LeftFrontWheel": "medium",
            "LeftMirror": "heavy",
            "LeftRearWheel": "severe",
            "LeftRoof": "minimal",
            "RightRoof": "medium",
            "Right": "heavy",
            "RightBackseat": "severe",
            "RightFrontWheel": "minimal",
            "RightMirror": "medium",
            "RightRearWheel": "heavy",
            "Rear": "severe",
            "RearLeft": "minimal",
            "RearRight": "medium",
            "RearWindowDamage": "heavy",
            "RightUnderside": "severe",
            "RoofDamage": "minimal",
            "WindscreenDamage": "medium",
            "UnderbodyDamage": "heavy"
        },
        "TP VehicleAge": 2018,
        "TP Engine Capacity": 2200,
        "TP Splatmap Data": {
            "Front": "medium",
            "FrontBonnet": "heavy",
            "FrontLeft": "severe",
            "FrontRight": "minimal",
            "LeftUnderside": "medium",
            "LoadArea": "heavy",
            "Left": "severe",
            "LeftBackseat": "minimal",
            "LeftFrontWheel": "medium",
            "LeftMirror": "heavy",
            "LeftRearWheel": "severe",
            "LeftRoof": "minimal",
            "RightRoof": "medium",
            "Right": "heavy",
            "RightBackseat": "severe",
            "RightFrontWheel": "minimal",
            "RightMirror": "medium",
            "RightRearWheel": "heavy",
            "Rear": "severe",
            "RearLeft": "minimal",
            "RearRight": "medium",
            "RearWindowDamage": "heavy",
            "RightUnderside": "severe",
            "RoofDamage": "minimal",
            "WindscreenDamage": "medium",
            "UnderbodyDamage": "heavy"
        },
        "TP Damage Assessment": "UnroadworthyRepair",
        "InsurerName1": "Insurer B",
        "Notification Date": "2023-02-01",
        "IncidentCauseDescription": "Theft",
        "Incident Date": "2023-02-01",
        "Incident Postcode": "67890",
        "FP Kept at postcode": "09876",
        "TP Body Key": "Coupe",
        "TP Doors": 2,
        "Notification Method": "Claims Portal Web",
        "FP Deployed Airbags": "4",
        "FP Value": 20000,
        "FP Engine Capacity": 2500,
        "IncidentSubCauseDescription": "Break-in",
        "Impact Speed Range": "ThirtyOneToForty",
        "TP Colour": "Blue",
        "TP Mileage": 30000,
        "FP Body Key": "Convertible",
        "FP Radiator Damaged": "no",
        "TP Boot Opens": "no",
        "FP Doors": 2,
        "IncidentUKCountry": "Scotland",
        "FP Seats": 2,
        "FP Wheels Damaged": "no",
        "FP Lights Damaged": "no",
        "TP Driveable / TP Damage Assessment?": "UnroadworthyRepair",
        "TP Is Right Hand Drive": "No",
        "FP Doors Open": "no",
        "TP Doors Open": "no",
        "FP Panel Gaps": "no",
        "TP NotOnMID": "No",
        "TP Vehicle Unattended": "no",
        "Road Conditions": "Wet",
        "FP Boot Opens": "no",
        "TP Wheels Damaged": "no",
        "FP Sharp Edges": "no",
        "TP Deployed Airbags": "4",
        "TP Lights Damaged": "no",
        "FP Driveable / FP Damage Assessment": "UnroadworthyRepair",
        "Weather Conditions": "Rain",
        "FP Registration": "XYZ789",
        "Claim Reference Number": "CRN67890"
    },
    {
        "FP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP VehicleAge": 2015,
        "TP Engine Capacity": 2000,
        "TP Splatmap Data": {
            "Front": "minimal",
            "FrontBonnet": "medium",
            "FrontLeft": "heavy",
            "FrontRight": "severe",
            "LeftUnderside": "minimal",
            "LoadArea": "medium",
            "Left": "heavy",
            "LeftBackseat": "severe",
            "LeftFrontWheel": "minimal",
            "LeftMirror": "medium",
            "LeftRearWheel": "heavy",
            "LeftRoof": "severe",
            "RightRoof": "minimal",
            "Right": "medium",
            "RightBackseat": "heavy",
            "RightFrontWheel": "severe",
            "RightMirror": "minimal",
            "RightRearWheel": "medium",
            "Rear": "heavy",
            "RearLeft": "severe",
            "RearRight": "minimal",
            "RearWindowDamage": "medium",
            "RightUnderside": "heavy",
            "RoofDamage": "severe",
            "WindscreenDamage": "minimal",
            "UnderbodyDamage": "medium"
        },
        "TP Damage Assessment": "DriveableRepair",
        "InsurerName1": "Insurer A",
        "Notification Date": "2023-01-01",
        "IncidentCauseDescription": "Collision",
        "Incident Date": "2023-01-01",
        "Incident Postcode": "12345",
        "FP Kept at postcode": "54321",
        "TP Body Key": "Sedan",
        "TP Doors": 4,
        "Notification Method": "Telephone",
        "FP Deployed Airbags": "2",
        "FP Value": 15000,
        "FP Engine Capacity": 1800,
        "IncidentSubCauseDescription": "Rear-end",
        "Impact Speed Range": "Stationary",
        "Impact Speed Unit": "KMH",
        "TP Colour": "Red",
        "TP Mileage": 50000,
        "FP Body Key": "SUV",
        "FP Radiator Damaged": "yes",
        "TP Boot Opens": "yes",
        "FP Doors": 4,
        "IncidentUKCountry": "England",
        "FP Seats": 5,
        "FP Wheels Damaged": "yes",
        "FP Lights Damaged": "yes",
        "TP Driveable / TP Damage Assessment?": "DriveableRepair",
        "TP Is Right Hand Drive": "Yes",
        "FP Doors Open": "yes",
        "TP Doors Open": "yes",
        "FP Panel Gaps": "yes",
        "TP NotOnMID": "Yes",
        "TP Vehicle Unattended": "yes",
        "Road Conditions": "Dry",
        "FP Boot Opens": "yes",
        "TP Wheels Damaged": "yes",
        "FP Sharp Edges": "yes",
        "TP Deployed Airbags": "2",
        "TP Lights Damaged": "yes",
        "FP Driveable / FP Damage Assessment": "DriveableRepair",
        "Weather Conditions": "Clear",
        "FP Registration": "ABC123",
        "Claim Reference Number": "CRN12345"
    },
    {
        "FP Splatmap Data": {
            "Front": "medium",
            "FrontBonnet": "heavy",
            "FrontLeft": "severe",
            "FrontRight": "minimal",
            "LeftUnderside": "medium",
            "LoadArea": "heavy",
            "Left": "severe",
            "LeftBackseat": "minimal",
            "LeftFrontWheel": "medium",
            "LeftMirror": "heavy",
            "LeftRearWheel": "severe",
            "LeftRoof": "minimal",
            "RightRoof": "medium",
            "Right": "heavy",
            "RightBackseat": "severe",
            "RightFrontWheel": "minimal",
            "RightMirror": "medium",
            "RightRearWheel": "heavy",
            "Rear": "severe",
            "RearLeft": "minimal",
            "RearRight": "medium",
            "RearWindowDamage": "heavy",
            "RightUnderside": "severe",
            "RoofDamage": "minimal",
            "WindscreenDamage": "medium",
            "UnderbodyDamage": "heavy"
        },
        "TP VehicleAge": 2018,
        "TP Engine Capacity": 2200,
        "TP Splatmap Data": {
            "Front": "medium",
            "FrontBonnet": "heavy",
            "FrontLeft": "severe",
            "FrontRight": "minimal",
            "LeftUnderside": "medium",
            "LoadArea": "heavy",
            "Left": "severe",
            "LeftBackseat": "minimal",
            "LeftFrontWheel": "medium",
            "LeftMirror": "heavy",
            "LeftRearWheel": "severe",
            "LeftRoof": "minimal",
            "RightRoof": "medium",
            "Right": "heavy",
            "RightBackseat": "severe",
            "RightFrontWheel": "minimal",
            "RightMirror": "medium",
            "RightRearWheel": "heavy",
            "Rear": "severe",
            "RearLeft": "minimal",
            "RearRight": "medium",
            "RearWindowDamage": "heavy",
            "RightUnderside": "severe",
            "RoofDamage": "minimal",
            "WindscreenDamage": "medium",
            "UnderbodyDamage": "heavy"
        },
        "TP Damage Assessment": "UnroadworthyRepair",
        "InsurerName1": "Insurer B",
        "Notification Date": "2023-02-01",
        "IncidentCauseDescription": "Theft",
        "Incident Date": "2023-02-01",
        "Incident Postcode": "67890",
        "FP Kept at postcode": "09876",
        "TP Body Key": "Coupe",
        "TP Doors": 2,
        "Notification Method": "Claims Portal Web",
        "FP Deployed Airbags": "4",
        "FP Value": 20000,
        "FP Engine Capacity": 2500,
        "IncidentSubCauseDescription": "Break-in",
        "Impact Speed Range": "ThirtyOneToFourty",
        "Impact Speed Unit": "MPH",
        "TP Colour": "Blue",
        "TP Mileage": 30000,
        "FP Body Key": "Convertible",
        "FP Radiator Damaged": "no",
        "TP Boot Opens": "no",
        "FP Doors": 2,
        "IncidentUKCountry": "Scotland",
        "FP Seats": 2,
        "FP Wheels Damaged": "no",
        "FP Lights Damaged": "no",
        "TP Driveable / TP Damage Assessment?": "UnroadworthyRepair",
        "TP Is Right Hand Drive": "No",
        "FP Doors Open": "no",
        "TP Doors Open": "no",
        "FP Panel Gaps": "no",
        "TP NotOnMID": "No",
        "TP Vehicle Unattended": "no",
        "Road Conditions": "Wet",
        "FP Boot Opens": "no",
        "TP Wheels Damaged": "no",
        "FP Sharp Edges": "no",
        "TP Deployed Airbags": "4",
        "TP Lights Damaged": "no",
        "FP Driveable / FP Damage Assessment": "UnroadworthyRepair",
        "Weather Conditions": "Rain",
        "FP Registration": "XYZ789",
        "Claim Reference Number": "CRN67890"
    },
    {
        "FP Splatmap Data": {
            "Front": "heavy",
            "FrontBonnet": "severe",
            "FrontLeft": "minimal",
            "FrontRight": "medium",
            "LeftUnderside": "heavy",
            "LoadArea": "severe",
            "Left": "minimal",
            "LeftBackseat": "medium",
            "LeftFrontWheel": "heavy",
            "LeftMirror": "severe",
            "LeftRearWheel": "minimal",
            "LeftRoof": "medium",
            "RightRoof": "heavy",
            "Right": "severe",
            "RightBackseat": "minimal",
            "RightFrontWheel": "medium",
            "RightMirror": "heavy",
            "RightRearWheel": "severe",
            "Rear": "minimal",
            "RearLeft": "medium",
            "RearRight": "heavy",
            "RearWindowDamage": "severe",
            "RightUnderside": "minimal",
            "RoofDamage": "medium",
            "WindscreenDamage": "heavy",
            "UnderbodyDamage": "severe"
        },
        "TP VehicleAge": 2017,
        "TP Engine Capacity": 2100,
        "TP Splatmap Data": {
            "Front": "heavy",
            "FrontBonnet": "severe",
            "FrontLeft": "minimal",
            "FrontRight": "medium",
            "LeftUnderside": "heavy",
            "LoadArea": "severe",
            "Left": "minimal",
            "LeftBackseat": "medium",
            "LeftFrontWheel": "heavy",
            "LeftMirror": "severe",
            "LeftRearWheel": "minimal",
            "LeftRoof": "medium",
            "RightRoof": "heavy",
            "Right": "severe",
            "RightBackseat": "minimal",
            "RightFrontWheel": "medium",
            "RightMirror": "heavy",
            "RightRearWheel": "severe",
            "Rear": "minimal",
            "RearLeft": "medium",
            "RearRight": "heavy",
            "RearWindowDamage": "severe",
            "RightUnderside": "minimal",
            "RoofDamage": "medium",
            "WindscreenDamage": "heavy",
            "UnderbodyDamage": "severe"
        },
        "TP Damage Assessment": "DriveableTotalLoss",
        "InsurerName1": "Insurer C",
        "Notification Date": "2023-03-01",
        "IncidentCauseDescription": "Fire",
        "Incident Date": "2023-03-01",
        "Incident Postcode": "11223",
        "FP Kept at postcode": "33211",
        "TP Body Key": "Hatchback",
        "TP Doors": 5,
        "Notification Method": "Crash Detector Notification",
        "FP Deployed Airbags": "3",
        "FP Value": 18000,
        "FP Engine Capacity": 2000,
        "IncidentSubCauseDescription": "Engine Fire",
        "Impact Speed Range": "FourtyOneToFifty",
        "Impact Speed Unit": "MPH",
        "TP Colour": "Green",
        "TP Mileage": 40000,
        "FP Body Key": "Truck",
        "FP Radiator Damaged": "yes",
        "TP Boot Opens": "yes",
        "FP Doors": 2,
        "IncidentUKCountry": "Wales",
        "FP Seats": 3,
        "FP Wheels Damaged": "yes",
        "FP Lights Damaged": "yes",
        "TP Driveable / TP Damage Assessment?": "DriveableTotalLoss",
        "TP Is Right Hand Drive": "Yes",
        "FP Doors Open": "yes",
        "TP Doors Open": "yes",
        "FP Panel Gaps": "yes",
        "TP NotOnMID": "Yes",
        "TP Vehicle Unattended": "yes",
        "Road Conditions": "Muddy",
        "FP Boot Opens": "yes",
        "TP Wheels Damaged": "yes",
        "FP Sharp Edges": "yes",
        "TP Deployed Airbags": "3",
        "TP Lights Damaged": "yes",
        "FP Driveable / FP Damage Assessment": "DriveableTotalLoss",
        "Weather Conditions": "Fog",
        "FP Registration": "LMN456",
        "Claim Reference Number": "CRN11223"
    },
    {
        "FP Splatmap Data": {
            "Front": "severe",
            "FrontBonnet": "minimal",
            "FrontLeft": "medium",
            "FrontRight": "heavy",
            "LeftUnderside": "severe",
            "LoadArea": "minimal",
            "Left": "medium",
            "LeftBackseat": "heavy",
            "LeftFrontWheel": "severe",
            "LeftMirror": "minimal",
            "LeftRearWheel": "medium",
            "LeftRoof": "heavy",
            "RightRoof": "severe",
            "Right": "minimal",
            "RightBackseat": "medium",
            "RightFrontWheel": "heavy",
            "RightMirror": "severe",
            "RightRearWheel": "minimal",
            "Rear": "medium",
            "RearLeft": "heavy",
            "RearRight": "severe",
            "RearWindowDamage": "minimal",
            "RightUnderside": "medium",
            "RoofDamage": "heavy",
            "WindscreenDamage": "severe",
            "UnderbodyDamage": "minimal"
        },
        "TP VehicleAge": 2016,
        "TP Engine Capacity": 2300,
        "TP Splatmap Data": {
            "Front": "severe",
            "FrontBonnet": "minimal",
            "FrontLeft": "medium",
            "FrontRight": "heavy",
            "LeftUnderside": "severe",
            "LoadArea": "minimal",
            "Left": "medium",
            "LeftBackseat": "heavy",
            "LeftFrontWheel": "severe",
            "LeftMirror": "minimal",
            "LeftRearWheel": "medium",
            "LeftRoof": "heavy",
            "RightRoof": "severe",
            "Right": "minimal",
            "RightBackseat": "medium",
            "RightFrontWheel": "heavy",
            "RightMirror": "severe",
            "RightRearWheel": "minimal",
            "Rear": "medium",
            "RearLeft": "heavy",
            "RearRight": "severe",
            "RearWindowDamage": "minimal",
            "RightUnderside": "medium",
            "RoofDamage": "heavy",
            "WindscreenDamage": "severe",
            "UnderbodyDamage": "minimal"
        },
        "TP Damage Assessment": "UnroadworthyTotalLoss",
        "InsurerName1": "Insurer D",
        "Notification Date": "2023-04-01",
        "IncidentCauseDescription": "Flood",
        "Incident Date": "2023-04-01",
        "Incident Postcode": "44556",
        "FP Kept at postcode": "66554",
        "TP Body Key": "SUV",
        "TP Doors": 4,
        "Notification Method": "Inbound Correspondence",
        "FP Deployed Airbags": "All",
        "FP Value": 25000,
        "FP Engine Capacity": 3000,
        "IncidentSubCauseDescription": "Water Damage",
        "Impact Speed Range": "FiftyOneToFifty",
        "Impact Speed Unit": "MPH",
        "TP Colour": "Black",
        "TP Mileage": 60000,
        "FP Body Key": "Van",
        "FP Radiator Damaged": "no",
        "TP Boot Opens": "no",
        "FP Doors": 4,
        "IncidentUKCountry": "Northern Ireland",
        "FP Seats": 4,
        "FP Wheels Damaged": "no",
        "FP Lights Damaged": "no",
        "TP Driveable / TP Damage Assessment?": "UnroadworthyTotalLoss",
        "TP Is Right Hand Drive": "No",
        "FP Doors Open": "no",
        "TP Doors Open": "no",
        "FP Panel Gaps": "no",
        "TP NotOnMID": "No",
        "TP Vehicle Unattended": "no",
        "Road Conditions": "Icy/Snowy",
        "FP Boot Opens": "no",
        "TP Wheels Damaged": "no",
        "FP Sharp Edges": "no",
        "TP Deployed Airbags": "All",
        "TP Lights Damaged": "no",
        "FP Driveable / FP Damage Assessment": "UnroadworthyTotalLoss",
        "Weather Conditions": "Snow",
        "FP Registration": "OPQ789",
        "Claim Reference Number": "CRN44556"
    }
]

# Convert JSON data to pandas DataFrame
df = pd.DataFrame(sample_data)

display(df)

# COMMAND ----------

df = df.astype({col: 'float' for col in df.select_dtypes(include=['int64']).columns})
display(df)

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://adb-6646029803360125.5.azuredatabricks.net/serving-endpoints/third_party_capture_endpoint/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

pred = score_model(df)

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql import Window, DataFrame
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.sql.types import *
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from xgboost.spark import SparkXGBRegressor
import xgboost as xgb

from mlflow.models.signature import ModelSignature, Schema, infer_signature
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec

import pandas as pd
import numpy as np
import re, sys, os, yaml, time, mlflow

from typing import Tuple, List, Dict, Any

def train_and_evaluate_cv_original(X_train: pd.DataFrame,
                          y_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_test: pd.DataFrame,
                          weights: pd.DataFrame,
                          params: Dict[str, Any],
                          registered_model_name: str,
                          client: MlflowClient,
                          label: str,
                          num_round: int = 100,
                          n_splits: int = 5,
                          ) -> Tuple[float, float]:
    """
    Train and evaluate the model with cross-validation.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.DataFrame): Target DataFrame.
        params (Dict[str, Any]): Parameters for the XGBoost model.
        num_round (int, optional): Number of boosting rounds. Defaults to 100.
        n_splits (int, optional): Number of splits for cross-validation.

    Returns:
        Tuple[float, float]: Mean RMSE and mean normalized Gini coefficient.
    """
    with mlflow.start_run(run_name=f'{label}_XGB'):

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
        bst = xgb.train(params, dtrain, num_round)
        
        dtest = xgb.DMatrix(X_test, label=y_test)
        preds = bst.predict(dtest)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        gini_score = gini_normalized_original(y_test, preds)

        preds_df = pd.DataFrame(preds)
        signature = infer_signature(X_test.head(), preds_df.head())

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("gini_score", gini_score)
        mlflow.xgboost.log_model(bst, "model", registered_model_name=registered_model_name, signature=signature,)

        version = get_latest_model_version(client, registered_model_name)
        client.set_registered_model_alias(registered_model_name, 'champion', version)

        
def create_mlflow_experiment_path(notebk_path,
                                  project_name,
                                  ) -> str:
    """
    Function to create a path for the mlflow experiment (as you can't point to a Git Folder)

    Parameters:


    Return
    """
    parts = notebk_path.split("/")
    if parts[0] == "":
        experiment_path= "/" + "/".join(parts[1:3]) + f"/{project_name}"
    else:
        experiment_path="/".join(parts[:2]) + f"/{project_name}"

    return experiment_path
    

def train_and_evaluate(
    train_df: DataFrame,
    column_name: str,
    params: Dict = None,
    tuning_type:str = 'cv',
    k: int = 3,
) -> Tuple[float, float]:
    """
    Train an XGBoost model using SparkXGBRegressor and evaluate using RMSE and Gini coefficient.
    """
    featuresCols = train_df.columns
    featuresCols.remove(column_name)

    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="vector_cols")

    scaler = StandardScaler()\
        .setInputCol("vector_cols")\
        .setOutputCol("scaled_features")\
        .setWithMean(True)\
        .setWithStd(True)

    vectorIndexer = VectorIndexer(inputCol="scaled_features", outputCol="features", maxCategories=4, handleInvalid='keep')

    xgb_regressor = SparkXGBRegressor(
        num_workers=sc.defaultParallelism,
        label_col=column_name,
        missing=0.0
    )

    gini_evaluator = GiniEvaluator(predictionCol="prediction", labelCol=column_name)

    if tuning_type == 'cv':
        paramGrid = create_param_grid(xgb_regressor, params)
        cv = CrossValidator(estimator=xgb_regressor, evaluator=gini_evaluator, estimatorParamMaps=paramGrid, numFolds=k)
        pipeline = Pipeline(stages=[vectorAssembler, scaler, vectorIndexer, cv])

    elif tuning_type == 'both':
        paramGrid = ParamGridBuilder().build()
        cv = CrossValidator(estimator=xgb_regressor, evaluator=gini_evaluator, estimatorParamMaps=paramGrid, numFolds=k)
        pipeline = Pipeline(stages=[vectorAssembler, scaler, vectorIndexer, cv])

    else:
        pipeline = Pipeline(stages=[vectorAssembler, scaler, vectorIndexer, xgb_regressor])

    return pipeline



def log_best_model(test_df: DataFrame,
                   transformed_test_data: DataFrame,
                   best_pipeline: Pipeline,
                   best_hyperparams: Dict,
                   gini_evaluator: Evaluator,
                   registered_model_name: str,
                   ):
    """
    Use the best hyper-parameters from the tuning to train the final model, and us CV to prevent over-fitting
    """

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    predictions_df = best_pipeline.transform(test_df)

    gini_score = gini_evaluator.evaluate(predictions_df)

    X_test_pandas = transformed_test_data.select("features").toPandas()
    predictions_pandas = predictions_df.select("prediction").toPandas()
    signature = infer_signature(X_test_pandas, predictions_pandas)

    with mlflow.start_run():

        mlflow.log_params(best_hyperparams)
        mlflow.log_metric("gini_score", gini_score)
        mlflow.spark.log_model(best_pipeline, "model", registered_model_name=registered_model_name, signature=signature)

        version = get_latest_model_version(client, registered_model_name)
        alias = "Champion" if version == 1 else "Challenger"
        client.set_registered_model_alias(registered_model_name, alias, version)

def get_latest_model_version(client: MlflowClient, 
                             model_name: str,
                             ) -> int:
    """
    Get the latest version of a registered model from MLFlow.

    Args:
        client (MlflowClient): MLFlow client.
        model_name (str): Registered model name.

    Returns:
        int: Latest model version.

    It could also be achieved through:
        max([int(mv.version) for mv in client.search_model_versions(f"name='{model_name}'")])
    """
    latest_version = 1
    for mv in client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def get_model_aliases(client: MlflowClient,
                      model_name: str,
                      ) -> list:
    """
    Retrieve the aliases of all versions of a given model.

    Args:
        model_name (str): The name of the model.

    Returns:
        list: A list of aliases for the model versions.
    """
    model_versions = [int(mv.version) for mv in client.search_model_versions(f"name='{model_name}'")]
    aliases = []
    for model_version in model_versions:
        alias = client.get_model_version(name=model_name, version=model_version).aliases
        if alias:
            aliases.append(alias[0])
    return aliases

def process_dataframe(df, rename_map, fp_tp_columns):
    # Rename columns based on the provided mapping
    for old_col, new_col in rename_map.items():
        df = df.rename(columns={old_col: new_col})
    renamed_columns = list(rename_map.values())
    df = df[renamed_columns]

    # Determine the 'body_key' column
    df.loc[:, 'body_key'] = df[
        ['body_key_01', 'body_key_02', 'body_key_03', 'body_key_04']
    ].idxmax(axis=1)
    df.loc[:, 'body_key'] = df['body_key'].apply(lambda x: x.split('_0')[-1]).astype(int)

    # Process 'fp_tp_columns' to create 'fp_' and 'tp_' prefixed columns
    for column in fp_tp_columns:
        df.loc[:, f'fp_{column}'] = df.apply(
            lambda row: row[column] if row['first_party_confirmed'] == 1 else 0, axis=1
        )
        df.loc[:, f'tp_{column}'] = df.apply(
            lambda row: row[column] if row['first_party_confirmed'] == 0 else 0, axis=1
        )
        df = df.drop(column, axis=1)

    return df

## TPC CLASS FUNCTION BUILD

class StringNoneToNaNConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        for col in df.columns:
            # Apply only to object/string columns for efficiency
            if df[col].dtype == 'object':
                df[col] = df[col].replace('None', np.nan)
        return df

def standardize_pandas_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the DataFrame by:
    1. Cleaning and normalizing column names (lowercase, snake_case, single underscores).
    2. Converting common string representations of missing values to np.nan across all cells.
    3. Coercing columns to appropriate data types (e.g., numerical columns to numeric types),
       converting any unconvertible values encountered during type coercion to np.nan.

    Args:
        df (pd.DataFrame): The input DataFrame to standardize.

    Returns:
        pd.DataFrame: A new DataFrame with standardized column names and cleaned cell values.
    """
    df_copy = df.copy()

    # --- 1. Standardize column names ---
    new_columns = []
    for col in df_copy.columns:
        # Ensure column name is string, then convert to lowercase
        new_col = str(col).lower()
        # Replace sequences of non-alphanumeric characters (excluding underscore) with a single underscore
        new_col = re.sub(r'[^a-z0-9_]+', '_', new_col)
        # Collapse multiple underscores into a single one
        new_col = re.sub(r'_+', '_', new_col)
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        new_columns.append(new_col)
    df_copy.columns = new_columns

    # --- 2. Convert common string representations of missing values to np.nan ---
    # Define a comprehensive list of string values that should be treated as NaN,
    # converting them to lowercase for robust comparison.
    missing_value_strings = [
        'none', 'null', 'nan', '', ' ',
        'na', 'n/a', 'not available', 'unknown', '-',
        '#n/a', '#n/a n/a', 'n/a (not applicable)', 'unspecified' # Add more common missing indicators
    ]

    # Convert all object type columns to string first to ensure .lower() works reliably for comparison.
    # Apply this cleaning to the entire DataFrame to catch all instances.
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            # Create a mask for values that, when lowercased, match any of the missing_value_strings
            mask = df_copy[col].astype(str).str.lower().isin(missing_value_strings)
            df_copy.loc[mask, col] = np.nan

    # --- 3. Coerce data types, converting unconvertible values to np.nan ---
    # Iterate through columns to attempt type conversion.
    for col in df_copy.columns:
        # Attempt to convert to numeric, coercing errors to NaN
        converted_numeric = pd.to_numeric(df_copy[col], errors='coerce')
        if not converted_numeric.isnull().all() and pd.api.types.is_numeric_dtype(converted_numeric):
            # If conversion resulted in a numeric series and it's not all NaNs,
            # then assume it should be numeric.
            df_copy[col] = converted_numeric
        else:
            # If not numeric, or if it became all NaNs upon numeric coercion,
            # then check for datetime conversion.
            converted_datetime = pd.to_datetime(df_copy[col], errors='coerce')
            if not converted_datetime.isnull().all() and pd.api.types.is_datetime64_any_dtype(converted_datetime):
                df_copy[col] = converted_datetime
            # If neither numeric nor datetime, it remains as its original (cleaned) type,
            # likely 'object', with any string 'None' etc. already converted to np.nan.

    return df_copy

# ─── 1. Splatmap Extraction ─────────────────────────────────────────────────────

class SplatmapExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, splatmap_cols=None):
        if splatmap_cols is None:
            splatmap_cols = [
                "Front", 
                "FrontBonnet", 
                "FrontLeft", 
                "FrontRight", 
                "Left", 
                "LeftBackseat",
                "LeftFrontWheel", 
                "LeftMirror", 
                "LeftRearWheel", 
                "LeftRoof", 
                "LeftUnderside",
                "LoadArea", 
                "Rear", 
                "RearLeft", 
                "RearRight", 
                "RearWindowDamage", 
                "Right",
                "RightBackseat", 
                "RightFrontWheel", 
                "RightMirror", 
                "RightRearWheel", 
                "RightRoof",
                "RightUnderside", 
                "RoofDamage", 
                "UnderbodyDamage", 
                "WindscreenDamage"
            ]
        self.splatmap_cols = splatmap_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'FP Splatmap Data' in df.columns and 'TP Splatmap Data' in df.columns:
            for col in self.splatmap_cols:
                df[f'FP_{col}'] = df['FP Splatmap Data'].apply(
                    lambda x: x.get(col) if isinstance(x, dict) else np.nan)
                df[f'TP_{col}'] = df['TP Splatmap Data'].apply(
                    lambda x: x.get(col) if isinstance(x, dict) else np.nan)
            df = df.drop(columns=['FP Splatmap Data', 'TP Splatmap Data'], errors='ignore')
        return df
    

# ─── 2. Column Renaming ─────────────────────────────────────────────────────────

class RenameColumnsTransformer(BaseEstimator, TransformerMixin):
    # allow injection of a custom map if desired
    default_map  = {
                'TP VehicleAge': 'year_of_manufacture',
                'TP Engine Capacity': 'tp_engine_capacity',
                'InsurerName1': 'insurer_name',
                'Notification Date': 'notification_date',
                'IncidentCauseDescription': 'incident_cause_description',
                'Incident Date': 'incident_date',
                'Incident Postcode': 'postcode_area',
                'FP Kept at postcode': 'vehicle_kept_at_postcode',
                'TP Body Key': 'tp_body_key',
                'TP Doors': 'tp_number_of_doors',
                'Notification Method': 'notification_method',
                'FP Deployed Airbags': 'fp_deployed_airbags',
                'FP Value': 'fp_vehicle_value',
                'FP Engine Capacity': 'fp_engine_capacity',
                'IncidentSubCauseDescription': 'incident_sub_cause_description',
                'Impact Speed Range': 'impact_speed_range',
                'Impact Speed Unit': 'impact_speed_unit',
                'TP Colour': 'tp_colour',
                'TP Mileage': 'tp_mileage',
                'FP Body Key': 'fp_body_key',
                'FP Radiator Damaged': 'fp_radiator_damaged',
                'TP Boot Opens': 'tp_boot_opens',
                'FP Doors': 'fp_number_of_doors',
                'IncidentUKCountry': 'incident_uk_country',
                'FP Seats': 'fp_number_of_seats',
                'FP Wheels Damaged': 'fp_wheels_damaged',
                'FP Lights Damaged': 'fp_lights_damaged',
                'TP Driveable / TP Damage Assessment': 'tp_driveable_damage_assessment',
                'TP Is Right Hand Drive': 'tp_right_hand_drive',
                'FP Doors Open': 'fp_doors_open',
                'TP Doors Open': 'tp_doors_open',
                'FP Panel Gaps': 'fp_panel_gaps',
                'TP NotOnMID': 'tp_not_on_mid',
                'TP Vehicle Unattended': 'tp_vehicle_unattended',
                'Road Conditions': 'road_conditions',
                'FP Boot Opens': 'fp_boot_opens',
                'TP Wheels Damaged': 'tp_wheels_damaged',
                'FP Sharp Edges': 'fp_sharp_edges',
                'TP Deployed Airbags': 'tp_deployed_airbags',
                'TP Lights Damaged': 'tp_lights_damaged',
                'FP Driveable / FP Damage Assessment': 'fp_driveable_damage_assessment',
                'Weather Conditions': 'weather_conditions',
                'FP Registration': 'fp_registration',
                'Claim Reference Number': 'claim_reference_number',
                # Adding all FP severity renames from the original user prompt
                'FP_Front': 'fp_front_severity',
                'FP_FrontBonnet': 'fp_front_bonnet_severity',
                'FP_FrontLeft': 'fp_front_left_severity',
                'FP_FrontRight': 'fp_front_right_severity',
                'FP_Left': 'fp_left_severity',
                'FP_LeftBackseat': 'fp_left_back_seat_severity',
                'FP_LeftFrontWheel': 'fp_left_front_wheel_severity',
                'FP_LeftMirror': 'fp_left_mirror_severity',
                'FP_LeftRearWheel': 'fp_left_rear_wheel_severity',
                'FP_LeftRoof': 'fp_left_roof_severity',
                'FP_LeftUnderside': 'fp_left_underside_severity',
                'FP_LoadArea': 'fp_load_area_severity',
                'FP_Rear': 'fp_rear_severity',
                'FP_RearLeft': 'fp_rear_left_severity',
                'FP_RearRight': 'fp_rear_right_severity',
                'FP_RearWindowDamage': 'fp_rear_window_damage_severity',
                'FP_Right': 'fp_right_severity',
                'FP_RightBackseat': 'fp_right_back_seat_severity',
                'FP_RightFrontWheel': 'fp_right_front_wheel_severity',
                'FP_RightMirror': 'fp_right_mirror_severity',
                'FP_RightRearWheel': 'fp_right_rear_wheel_severity',
                'FP_RightRoof': 'fp_right_roof_severity',
                'FP_RightUnderside': 'fp_right_underside_severity',
                'FP_RoofDamage': 'fp_roof_damage_severity',
                'FP_UnderbodyDamage': 'fp_underbody_damage_severity',
                'FP_WindscreenDamage': 'fp_windscreen_damage_severity',
                # Adding all TP severity renames from the original user prompt
                'TP_Front': 'tp_front_severity',
                'TP_FrontBonnet': 'tp_front_bonnet_severity',
                'TP_FrontLeft': 'tp_front_left_severity',
                'TP_FrontRight': 'tp_front_right_severity',
                'TP_Left': 'tp_left_severity',
                'TP_LeftBackseat': 'tp_left_back_seat_severity',
                'TP_LeftFrontWheel': 'tp_left_front_wheel_severity',
                'TP_LeftMirror': 'tp_left_mirror_severity',
                'TP_LeftRearWheel': 'tp_left_rear_wheel_severity',
                'TP_LeftRoof': 'tp_left_roof_severity',
                'TP_LeftUnderside': 'tp_left_underside_severity',
                'TP_LoadArea': 'tp_load_area_severity',
                'TP_Rear': 'tp_rear_severity',
                'TP_RearLeft': 'tp_rear_left_severity',
                'TP_RearRight': 'tp_rear_right_severity',
                'TP_RearWindowDamage': 'tp_rear_window_damage_severity',
                'TP_Right': 'tp_right_severity',
                'TP_RightBackseat': 'tp_right_back_seat_severity',
                'TP_RightFrontWheel': 'tp_right_front_wheel_severity',
                'TP_RightMirror': 'tp_right_mirror_severity',
                'TP_RightRearWheel': 'tp_right_rear_wheel_severity',
                'TP_RightRoof': 'tp_right_roof_severity',
                'TP_RightUnderside': 'tp_right_underside_severity',
                'TP_RoofDamage': 'tp_roof_damage_severity',
                'TP_UnderbodyDamage': 'tp_underbody_damage_severity',
                'TP_WindscreenDamage': 'tp_windscreen_damage_severity'
            }
        
    def __init__(self, rename_map=None):
        self.rename_map = rename_map or self.default_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df.rename(columns=self.rename_map)
        renamed_columns = list(RenameColumnsTransformer().default_map.values())
        return df


# ─── 3. Mode & Median Imputation ─────────────────────────────────────────────────

class GeneralPurposeImputer(BaseEstimator, TransformerMixin):
    """
    A robust scikit-learn compatible transformer for imputing missing values
    in both numerical and categorical columns.

    Numerical columns are imputed using the median strategy.
    Categorical columns are imputed using the most frequent strategy.

    The imputer can be initialized with explicit lists of numerical and
    categorical features. If not provided, it infers them from the DataFrame's
    dtypes during the fit stage.

    Attributes:
        numerical_features (list): Optional. List of column names to treat
                                   as numerical for median imputation.
        categorical_features (list): Optional. List of column names to treat
                                     as categorical for most frequent imputation.
        num_imputer_ (SimpleImputer): Fitted imputer for numerical columns.
        cat_imputer_ (SimpleImputer): Fitted imputer for categorical columns.
        numerical_cols_ (list): List of numerical column names identified during fit.
        categorical_cols_ (list): List of categorical column names identified during fit.
    """
    def __init__(self, numerical_features: list = None, categorical_features: list = None):
        """
        Initializes the imputer.

        Args:
            numerical_features (list, optional): List of column names to treat as numerical
                                                 for median imputation. If None, all np.number
                                                 dtypes will be considered.
            categorical_features (list, optional): List of column names to treat as categorical
                                                   for most frequent imputation. If None, all 'object'
                                                   dtypes will be considered.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.numerical_cols_ = None
        self.categorical_cols_ = None

    def fit(self, X, y=None):
        """
        Fits the imputer on the provided data.
        It identifies numerical and categorical columns and fits separate
        SimpleImputer instances for each type.

        Args:
            X (pd.DataFrame): The input DataFrame to fit the imputer on.
            y (array-like, optional): Ignored. Present for scikit-learn API consistency.

        Returns:
            self: The fitted imputer instance.

        Raises:
            ValueError: If X is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        # Determine numerical columns based on initialization or dtypes
        if self.numerical_features is not None:
            self.numerical_cols_ = [col for col in self.numerical_features if col in X.columns]
        else:
            self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        # Determine categorical columns based on initialization or dtypes
        if self.categorical_features is not None:
            self.categorical_cols_ = [col for col in self.categorical_features if col in X.columns]
        else:
            self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Fit numerical imputer if numerical columns exist
        if self.numerical_cols_:
            self.num_imputer_ = SimpleImputer(strategy='median')
            # Fit only on the subset of X that contains the identified numerical columns
            self.num_imputer_.fit(X[self.numerical_cols_])

        # Fit categorical imputer if categorical columns exist
        if self.categorical_cols_:
            self.cat_imputer_ = SimpleImputer(strategy='most_frequent')
            # Fit only on the subset of X that contains the identified categorical columns
            self.cat_imputer_.fit(X[self.categorical_cols_])

        return self

    def transform(self, X):
        """
        Transforms the input data by imputing missing values.
        Missing numerical values are imputed using the median learned during fit.
        Missing categorical values are imputed using the most frequent value learned during fit.

        Args:
            X (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: A new DataFrame with missing values imputed.

        Raises:
            ValueError: If X is not a pandas DataFrame.
            RuntimeError: If the imputer has not been fitted before calling transform.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        if self.num_imputer_ is None and self.cat_imputer_ is None:
            raise RuntimeError("Imputer has not been fitted yet. Call fit() first.")

        df = X.copy()

        # Impute numerical columns
        if self.num_imputer_ and self.numerical_cols_:
            # Select only the numerical columns that were used for fitting AND are present in the current DataFrame
            cols_to_transform_num = [col for col in self.numerical_cols_ if col in df.columns]
            if cols_to_transform_num:
                # Perform transformation and convert the resulting NumPy array back to a DataFrame
                # ensuring column names and index are preserved.
                imputed_numerical_data = self.num_imputer_.transform(df[cols_to_transform_num])
                df[cols_to_transform_num] = pd.DataFrame(
                    imputed_numerical_data,
                    columns=cols_to_transform_num, # Explicitly set column names
                    index=df.index                 # Explicitly set index to match df
                )

        # Impute categorical columns
        if self.cat_imputer_ and self.categorical_cols_:
            # Select only the categorical columns that were used for fitting AND are present in the current DataFrame
            cols_to_transform_cat = [col for col in self.categorical_cols_ if col in df.columns]
            if cols_to_transform_cat:
                # Perform transformation and convert the resulting NumPy array back to a DataFrame
                # ensuring column names and index are preserved.
                imputed_categorical_data = self.cat_imputer_.transform(df[cols_to_transform_cat])
                df[cols_to_transform_cat] = pd.DataFrame(
                    imputed_categorical_data,
                    columns=cols_to_transform_cat, # Explicitly set column names
                    index=df.index                 # Explicitly set index to match df
                )

        return df

# ─── 4. Impact Speed Conversion & Bucketing ──────────────────────────────────────

class ImpactSpeedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # map buckets to representative speeds
        self.speed_map = {
            'Stationary': 0, 'ZeroToSix': 1, 'SevenToFourteen': 7,
            'FifteenToTwenty': 15, 'TwentyOneToThirty': 21, 'ThirtyOneToFourty': 31,
            'FourtyOneToFifty': 41, 'FiftyOneToSixty': 51, 'SixtyOneToSeventy': 61,
            'OverSeventy': 71
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['impact_speed_range'] = df['impact_speed_range'].map(self.speed_map).fillna(-1)
        df['impact_speed_unit'] = df['impact_speed_unit'].fillna('MPH')
        # convert KMH → MPH
        df['impact_speed'] = np.where(
            df['impact_speed_unit']=='KMH',
            df['impact_speed_range']/1.61,
            df['impact_speed_range']
        )
        # re-bucket numeric speed back into original bins
        def bucket(x):
            if   x==0:   return 0
            elif x<=6:  return 1
            elif x<15:  return 7
            elif x<21:  return 15
            elif x<31:  return 21
            elif x<41:  return 31
            elif x<51:  return 41
            elif x<61:  return 51
            elif x<71:  return 61
            elif x>=71: return 71
            else:       return -1
        df['impact_speed'] = df['impact_speed'].apply(bucket)
        return df.drop(columns=['impact_speed_range', 'impact_speed_unit'])


# ─── 5. Airbag Mapping ───────────────────────────────────────────────────────────

class AirbagCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.airbag_map = {'None': 0, '1':1, '2':2, '3':3, '4':4, 'All':5}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['tp_deployed_airbags'] = df['tp_deployed_airbags'].map(self.airbag_map).fillna(-1)
        df['fp_deployed_airbags'] = df['fp_deployed_airbags'].map(self.airbag_map).fillna(-1)
        return df


# ─── 6. Car Characteristics Mapping ───────────────────────────────────────────────────────────	

class CarCharacteristicTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df['fp_boot_opens'] = (df['fp_boot_opens']=='yes').astype(int)
        df['fp_doors_open'] = (df['fp_doors_open']=='yes').astype(int)
        df['fp_wheels_damaged'] = (df['fp_wheels_damaged']=='yes').astype(int)
        df['fp_lights_damaged'] = (df['fp_lights_damaged']=='yes').astype(int)

        df['tp_engine_capacity'] = (df['tp_engine_capacity']=='yes').astype(int)
        df['tp_number_of_doors'] = (df['tp_number_of_doors']=='yes').astype(int)
        df['tp_boot_opens'] = (df['tp_boot_opens']=='yes').astype(int)
        df['tp_doors_open'] = (df['tp_doors_open']=='yes').astype(int)
        df['tp_not_on_mid'] = (df['tp_not_on_mid']=='yes').astype(int)
        df['tp_vehicle_unattended'] = (df['tp_vehicle_unattended']=='yes').astype(int)
        df['tp_wheels_damaged'] = (df['tp_wheels_damaged']=='yes').astype(int)
        df['tp_lights_damaged'] = (df['tp_lights_damaged']=='yes').astype(int)

        df['da_dr'] = (df['tp_driveable_damage_assessment']=='DriveableRepair').astype(int)
        df['da_dtl'] = (df['tp_driveable_damage_assessment']=='DriveableTotalLoss').astype(int)
        df['da_utl'] = (df['tp_driveable_damage_assessment']=='UnroadworthyTotalLoss').astype(int)
        df['da_ur'] = (df['tp_driveable_damage_assessment']=='UnroadworthyRepair').astype(int)
        df['da_o'] = ~df['tp_driveable_damage_assessment'].isin([
                                                    'DriveableRepair', 
                                                    'DriveableTotalLoss', 
                                                    'UnroadworthyTotalLoss', 
                                                    'UnroadworthyRepair'
                                                ]).astype(int)

        return df.drop(['tp_driveable_damage_assessment'], axis = 1)

# ─── 7. Date Features ───────────────────────────────────────────────────────────

class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['time_to_notify'] = (df['notification_date'] - df['incident_date']).dt.days
        df['time_to_notify'] = df['time_to_notify'].clip(0, 30).fillna(0)
        
        df['notified_day_of_week'] = df['notification_date'].dt.dayofweek.apply(lambda x: 7 if x == 0 else x)
        df['notified_day'] = df['notification_date'].dt.day
        df['notified_month'] = df['notification_date'].dt.month
        df['notified_year'] = df['notification_date'].dt.year
        
        df['incident_day_of_week'] = df['incident_date'].dt.dayofweek.apply(lambda x: 7 if x == 0 else x)
        df['incident_day'] = df['incident_date'].dt.day
        
        return df


# ─── 8. Vehicle Age ─────────────────────────────────────────────────────────────

class VehicleAgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['fp_vehicle_age'] = df['incident_date'].dt.year - df['year_of_manufacture'].astype(int)
        df['fp_vehicle_age'] = df['fp_vehicle_age'].where(df['fp_vehicle_age']<=30, 0).fillna(0)
        return df.drop(['notification_date', 'incident_date'], axis = 1)


# ─── 9. Right-Hand Drive Flag ──────────────────────────────────────────────────

class RHDTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['tp_right_hand_drive'] = (df['tp_right_hand_drive']=='R').astype(int)
        return df


# ─── 10. Body Key One-Hot Encoding ──────────────────────────────────────────────

class BodyKeyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, keys=None):
        # only encode a fixed set of styles as in your example
        self.keys = keys or [
            '5 Door Hatchback','5 Door Estate','4 Door Saloon','3 Door Hatchback'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for i, style in enumerate(self.keys, start=1):
            df[f'fp_body_key_0{i}'] = (df['fp_body_key']==style).astype(int)
            df[f'tp_body_key_0{i}'] = (df['tp_body_key']==style).astype(int)
        return df.drop('fp_body_key', axis=1).drop('tp_body_key', axis=1)


# ─── 11. Postcode Area Extraction ─────────────────────────────────────────────

class PostcodeAreaExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['postcode_area'] = df['postcode_area']\
            .str.extract(r'^([A-Za-z]+)')[0]\
            .fillna('ZZ')
        return df


# ─── 12. Drop Any Leftovers ─────────────────────────────────────────────────────

class DropUnusedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop=None):
        self.to_drop = to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.to_drop, errors='ignore')


# ─── 13. Damage Recorded and Assessment ─────────────────────────────────────────────────────

class DamageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, severity_columns=None, fp_severity_columns=None):
        self.severity_columns = severity_columns or [
            'tp_front_severity', 
            'tp_front_bonnet_severity', 
            'tp_front_left_severity', 
            'tp_front_right_severity', 
            'tp_left_severity', 
            'tp_left_back_seat_severity', 
            'tp_left_front_wheel_severity', 
            'tp_left_mirror_severity', 
            'tp_left_rear_wheel_severity', 
            'tp_left_roof_severity', 
            'tp_left_underside_severity', 
            'tp_load_area_severity', 
            'tp_rear_severity', 
            'tp_rear_left_severity', 
            'tp_rear_right_severity', 
            'tp_rear_window_damage_severity', 
            'tp_right_severity', 
            'tp_right_back_seat_severity', 
            'tp_right_front_wheel_severity', 
            'tp_right_mirror_severity', 
            'tp_right_rear_wheel_severity', 
            'tp_right_roof_severity', 
            'tp_right_underside_severity', 
            'tp_roof_damage_severity', 
            'tp_underbody_damage_severity', 
            'tp_windscreen_damage_severity'
        ]

        self.fp_severity_columns = fp_severity_columns or [
            'fp_front_severity', 
            'fp_front_bonnet_severity', 
            'fp_front_left_severity', 
            'fp_front_right_severity', 
            'fp_left_severity', 
            'fp_left_back_seat_severity', 
            'fp_left_front_wheel_severity', 
            'fp_left_mirror_severity', 
            'fp_left_rear_wheel_severity', 
            'fp_left_roof_severity', 
            'fp_left_underside_severity', 
            'fp_load_area_severity', 
            'fp_rear_severity', 
            'fp_rear_left_severity', 
            'fp_rear_right_severity', 
            'fp_rear_window_damage_severity', 
            'fp_right_severity', 
            'fp_right_back_seat_severity', 
            'fp_right_front_wheel_severity', 
            'fp_right_mirror_severity', 
            'fp_right_rear_wheel_severity', 
            'fp_right_roof_severity', 
            'fp_right_underside_severity', 
            'fp_roof_damage_severity', 
            'fp_underbody_damage_severity', 
            'fp_windscreen_damage_severity',
            'fp_panel_gaps',
            'fp_radiator_damaged',
            'fp_sharp_edges']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Calculate damage recorded and assessed
        df['damage_recorded'] = df[self.severity_columns].notnull().sum(axis=1)
        df['tp_damage_assessed'] = df['damage_recorded'].apply(lambda x: 1 if x > 0 else 0)
        
        # Scale damage severity
        def damage_scale(row_area, row_damageflag):
            if row_damageflag == 1:
                if row_area == 'minimal':
                    scale = 1
                elif row_area == 'medium':
                    scale = 2
                elif row_area == 'heavy':
                    scale = 3
                elif row_area == 'severe':
                    scale = 4
                elif row_area == 'unknown':
                    scale = -1
                else:
                    scale = 0
            else:
                scale = -1
            return scale

        for col_name in self.severity_columns:
            df[col_name] = df.apply(lambda row: damage_scale(row[col_name], row['tp_damage_assessed']), axis=1).astype(int)
        
        for col_name in self.fp_severity_columns:
            df[col_name] = df.apply(lambda row: damage_scale(row[col_name], row['tp_damage_assessed']), axis=1).astype(int)

        return df


# ─── 14. Damage Severity ─────────────────────────────────────────────────────

class DamageSeverityCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, sev_damage=None):
        self.sev_damage = sev_damage or [
            'tp_front_severity', 
            'tp_front_bonnet_severity', 
            'tp_front_left_severity', 
            'tp_front_right_severity', 
            'tp_left_severity', 
            'tp_left_back_seat_severity', 
            'tp_left_front_wheel_severity', 
            'tp_left_mirror_severity', 
            'tp_left_rear_wheel_severity', 
            'tp_left_roof_severity', 
            'tp_left_underside_severity', 
            'tp_load_area_severity', 
            'tp_rear_severity', 
            'tp_rear_left_severity', 
            'tp_rear_right_severity', 
            'tp_rear_window_damage_severity', 
            'tp_right_severity', 
            'tp_right_back_seat_severity', 
            'tp_right_front_wheel_severity', 
            'tp_right_mirror_severity', 
            'tp_right_rear_wheel_severity', 
            'tp_right_roof_severity', 
            'tp_right_underside_severity', 
            'tp_roof_damage_severity', 
            'tp_underbody_damage_severity', 
            'tp_windscreen_damage_severity'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        df['damage_sev_total'] = df[self.sev_damage].sum(axis=1)
        df['damage_sev_count'] = df[self.sev_damage].gt(0).sum(axis=1)
        df['damage_sev_mean'] = df.apply(lambda row: row['damage_sev_total'] / row['damage_sev_count'] if row['damage_sev_count'] > 0 else 0, axis=1)
        df['damage_sev_max'] = df[self.sev_damage].max(axis=1)
        
        return df
    

class DaysInRepairPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_uri):
        self.model_uri = model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.input_cols_from_schema = list(self.model.metadata.get_input_schema().input_names())
        # Store the schema itself to check types
        self.model_schema = self.model.metadata.get_input_schema().to_dict()
        self.col_to_type_map = {spec['name']: spec['type'] for spec in self.model_schema}


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_original_for_output = X.copy()
        X_for_model = pd.DataFrame(index=X.index)

        for col_name in self.input_cols_from_schema:
            expected_type_str = self.col_to_type_map.get(col_name)

            if col_name in X.columns:
                X_for_model[col_name] = X[col_name]
            else:
                # Column is missing, fill with a sensible default based on expected type
                if expected_type_str in ['string', 'object', 'bytes']:
                    X_for_model[col_name] = ""  # Default for missing string
                elif expected_type_str in ['integer', 'long', 'float', 'double']:
                    X_for_model[col_name] = 0   # Default for missing numeric/OHE
                else:
                    X_for_model[col_name] = np.nan # Or a more generic default
                print(f"Warning: Column '{col_name}' was missing from input and defaulted.")


        # Coerce data types based on the model's schema
        for col_name in self.input_cols_from_schema:
            if col_name not in X_for_model.columns: # Should not happen if above loop is correct
                print(f"Critical Error: Column '{col_name}' still missing before type coercion.")
                continue

            actual_dtype = X_for_model[col_name].dtype
            expected_type_str = self.col_to_type_map.get(col_name)
            
            try:
                if expected_type_str == 'string':
                    if not pd.api.types.is_string_dtype(actual_dtype):
                        X_for_model[col_name] = X_for_model[col_name].astype(str)
                elif expected_type_str == 'double' or expected_type_str == 'float':
                    if not pd.api.types.is_float_dtype(actual_dtype):
                        X_for_model[col_name] = X_for_model[col_name].astype(np.float64)
                elif expected_type_str == 'integer' or expected_type_str == 'long':
                     # Handle potential floats from previous steps if an int is expected
                    if pd.api.types.is_float_dtype(actual_dtype):
                        # If it could have NaNs that can't be int, fill them first or handle
                        if X_for_model[col_name].isnull().any():
                             X_for_model[col_name] = X_for_model[col_name].fillna(0) # Or another appropriate int fill
                        X_for_model[col_name] = X_for_model[col_name].astype(np.int64) # Or np.int32
                    elif not pd.api.types.is_integer_dtype(actual_dtype):
                        X_for_model[col_name] = X_for_model[col_name].astype(np.int64)
                # Add other type coercions if needed (boolean, datetime etc.)
            except Exception as e:
                print(f"Error coercing column '{col_name}' to {expected_type_str}. Current dtype: {actual_dtype}. Error: {e}")
                # Optionally, re-raise or handle more gracefully

        # Ensure the column order is exactly as in input_cols_from_schema
        X_for_model = X_for_model[self.input_cols_from_schema]

        predictions = self.model.predict(X_for_model)
        X_original_for_output["days_in_repair"] = predictions
        
        return X_original_for_output

class CaptureBenefitModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):

        print("Loading model context...")
        # Load the preprocessor pipeline (which includes all your transformers)
        self.preprocessor = mlflow.sklearn.load_model(context.artifacts["preprocessor_model_path"])
        print(f"Preprocessor loaded from: {context.artifacts['preprocessor_model_path']}")

        # Load the trained regressor model
        self.regressor = mlflow.sklearn.load_model(context.artifacts["regressor_model_path"])
        print(f"Regressor loaded from: {context.artifacts['regressor_model_path']}")

        # Load the quantile benefit multiplier table (expected as a CSV)
        quantile_table_path = context.artifacts["quantile_benefit_table_csv_path"]
        print(f"Attempting to load quantile benefit table from: {quantile_table_path}")
        self.quantile_benefit_df = pd.read_csv(quantile_table_path)

    def _get_quantile_benefit_multiplier(self, raw_capture_benefit_value: float) -> float:

        if self.quantile_benefit_df is None:
            print("Quantile benefit table not loaded. Returning default multiplier of 1.0.")
            return 1.0 # Should not happen if load_context is successful

        for _, row in self.quantile_benefit_df.iterrows():
            if row['capture_benefit_lower_bound'] <= raw_capture_benefit_value < row['capture_benefit_upper_bound']:
                return row['quant_benefit_multiplier']
        
        # If no range matches (e.g., value is exactly on the max upper_bound of the last interval,
        # or table doesn't cover all possibilities), define a fallback.
        # This might indicate an issue with the table's range definitions.
        # Consider logging this event.
        # print(f"Warning: No multiplier range found for raw_capture_benefit_value: {raw_capture_benefit_value}. Using default of 1.0.")
        # Check if it matches the last upper bound exactly (if it's inclusive)
        last_row = self.quantile_benefit_df.iloc[-1]
        if raw_capture_benefit_value == last_row['capture_benefit_upper_bound'] and last_row['capture_benefit_upper_bound'] == np.inf : # Special case for last interval potentially being inclusive of its start
             return last_row['quant_benefit_multiplier']

        return 1.0 # Default multiplier if no specific range is found or other conditions not met
    
    def standardize_pandas_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all numeric columns to float64 and all other columns to string.

        Args:
            df (pd.DataFrame): The input DataFrame

        Returns:
            pd.DataFrame: DataFrame with consistent float64 and string dtypes
        """
        df_standardized = df.copy()

        for col in df_standardized.columns:
            col_dtype = df_standardized[col].dtype

            if pd.api.types.is_numeric_dtype(col_dtype):
                df_standardized[col] = df_standardized[col].astype(np.float64)
            else:
                df_standardized[col] = df_standardized[col].astype(str)

        return df_standardized

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:

        print(f"Received input with {model_input.shape[0]} rows and {model_input.shape[1]} columns.")
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # --- Prediction with CaptureSuccess_AdrienneVersion = 1 ---
        df_cap = model_input.copy()
        # Ensure the column name matches exactly what the preprocessor expects.
        # The value should be float if the column is treated as numeric by the scaler.
        df_cap['CaptureSuccess_AdrienneVersion'] = 1.0 

        # --- Prediction with CaptureSuccess_AdrienneVersion = 0 ---
        df_nocap = model_input.copy()
        df_nocap['CaptureSuccess_AdrienneVersion'] = 0.0

        # Preprocess both DataFrames using the loaded preprocessor pipeline
        # The preprocessor should handle all transformations (Splatmap, renaming, imputation,
        # speed, airbags, dates, OHE, scaling, DaysInRepairPredictor, etc.)
        print("Preprocessing data for 'capture' scenario...")
        try:
            processed_cap_features = self.preprocessor.transform(df_cap)
        except Exception as e:
            print(f"Error transforming df_cap: {e}")
            # Consider which columns are present in df_cap before transform
            # print(f"df_cap columns: {df_cap.columns.tolist()}")
            raise RuntimeError(f"Error during preprocessing 'capture' data: {e}")
        
        print("Preprocessing data for 'no capture' scenario...")
        try:
            processed_nocap_features = self.preprocessor.transform(df_nocap)
        except Exception as e:
            print(f"Error transforming df_nocap: {e}")
            raise RuntimeError(f"Error during preprocessing 'no capture' data: {e}")

        # Make predictions using the loaded regressor model
        print("Making predictions...")
        tppd_pred_capture = self.regressor.predict(processed_cap_features)
        tppd_pred_no_capture = self.regressor.predict(processed_nocap_features)

        # Calculate raw Capture_Benefit
        # (TPPD_Pred_No_Capture - TPPD_Pred_Capture)
        raw_capture_benefit = tppd_pred_no_capture - tppd_pred_capture
        print(f"Raw capture benefit calculated. Min: {np.min(raw_capture_benefit)}, Max: {np.max(raw_capture_benefit)}")


        # Adjust Capture_Benefit using the lookup table and multiplier
        # Apply the lookup for each value if raw_capture_benefit is an array (multiple input rows)
        if isinstance(raw_capture_benefit, np.ndarray):
            capture_benefit_adjusted = np.array([
                cb_raw * self._get_quantile_benefit_multiplier(cb_raw) for cb_raw in raw_capture_benefit
            ])
        elif isinstance(raw_capture_benefit, (int, float)): # Single prediction value
            capture_benefit_adjusted = raw_capture_benefit * self._get_quantile_benefit_multiplier(raw_capture_benefit)
        else:
            raise TypeError(f"Unexpected type for raw_capture_benefit: {type(raw_capture_benefit)}")
        
        print(f"Adjusted capture benefit calculated. Min: {np.min(capture_benefit_adjusted)}, Max: {np.max(capture_benefit_adjusted)}")

        # Return the final adjusted benefit as a pandas Series.
        # The problem states "returns the value in a variable called Capture_Benefit".
        # So, the output Series should be named 'Capture_Benefit'.
        output_series = pd.Series(capture_benefit_adjusted, name="Capture_Benefit")
        print("Prediction process complete.")
        return output_series

# COMMAND ----------

input_data_json = {
  "instances": [
    {
      "Claim Reference Number": "CRN12345",
      "FP Body Key": "SUV",
      "FP Boot Opens": "yes",
      "FP Deployed Airbags": "2",
      "FP Doors": 4,
      "FP Doors Open": "yes",
      "FP Driveable / FP Damage Assessment": "DriveableRepair",
      "FP Engine Capacity": 1800,
      "FP Kept at postcode": "54321",
      "FP Lights Damaged": "yes",
      "FP Panel Gaps": "yes",
      "FP Radiator Damaged": "yes",
      "FP Registration": "ABC123",
      "FP Seats": 5,
      "FP Sharp Edges": "yes",
      "FP Splatmap Data": {
        "Front": "minimal",
        "FrontBonnet": "medium",
        "FrontLeft": "heavy",
        "FrontRight": "severe",
        "Left": "heavy",
        "LeftBackseat": "severe",
        "LeftFrontWheel": "minimal",
        "LeftMirror": "medium",
        "LeftRearWheel": "heavy",
        "LeftRoof": "severe",
        "LeftUnderside": "minimal",
        "LoadArea": "medium",
        "Rear": "heavy",
        "RearLeft": "severe",
        "RearRight": "minimal",
        "RearWindowDamage": "medium",
        "Right": "medium",
        "RightBackseat": "heavy",
        "RightFrontWheel": "severe",
        "RightMirror": "minimal",
        "RightRearWheel": "medium",
        "RightRoof": "minimal",
        "RightUnderside": "heavy",
        "RoofDamage": "severe",
        "UnderbodyDamage": "medium",
        "WindscreenDamage": "minimal"
      },
      "FP Value": 15000,
      "FP Wheels Damaged": "yes",
      "Impact Speed Range": "TwentyOneToThirty",
      "Impact Speed Unit": "MPH",
      "Incident Date": "2023-01-01",
      "Incident Postcode": "12345",
      "IncidentSubCauseDescription": "Rear-end",
      "IncidentUKCountry": "England",
      "IncidentCauseDescription": "Collision",
      "InsurerName1": "Insurer A",
      "Notification Date": "2023-01-01",
      "Notification Method": "Telephone",
      "Road Conditions": "Wet",
      "TP Body Key": "Sedan",
      "TP Boot Opens": "yes",
      "TP Colour": "Red",
      "TP Deployed Airbags": "2",
      "TP Doors": 4,
      "TP Doors Open": "yes",
      "TP Driveable / TP Damage Assessment": "DriveableRepair",
      "TP Engine Capacity": 2000,
      "TP Is Right Hand Drive": "Yes",
      "TP Lights Damaged": "yes",
      "TP Mileage": 50000,
      "TP NotOnMID": "Yes",
      "TP Splatmap Data": {
        "Front": "minimal",
        "FrontBonnet": "medium",
        "FrontLeft": "heavy",
        "FrontRight": "severe",
        "Left": "heavy",
        "LeftBackseat": "severe",
        "LeftFrontWheel": "minimal",
        "LeftMirror": "medium",
        "LeftRearWheel": "heavy",
        "LeftRoof": "severe",
        "LeftUnderside": "minimal",
        "LoadArea": "medium",
        "Rear": "heavy",
        "RearLeft": "severe",
        "RearRight": "minimal",
        "RearWindowDamage": "medium",
        "Right": "medium",
        "RightBackseat": "heavy",
        "RightFrontWheel": "severe",
        "RightMirror": "minimal",
        "RightRearWheel": "medium",
        "RightRoof": "minimal",
        "RightUnderside": "heavy",
        "RoofDamage": "severe",
        "UnderbodyDamage": "medium",
        "WindscreenDamage": "minimal"
      },
      "TP Vehicle Unattended": "yes",
      "TP VehicleAge": 2015,
      "TP Wheels Damaged": "yes",
      "Weather Conditions": "Clear"
    }
  ]
}

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Step 0: Standardize Pandas Schema (as called in your predict method) ---
# Assuming standardize_pandas_schema is a top-level function
#feature_table_name = "prod_dsexp_mlstore.third_party_capture.tpc_feature_table"
#df_tpc = standardize_pandas_schema(spark.table(feature_table_name).toPandas())
#df_tpc = df_tpc[features_list + date_columns + [tpc_target_column]]
#current_df = df_tpc.copy()
input_df = pd.DataFrame(input_data_json['instances'])
current_df = input_df # standardize_pandas_schema(input_df).copy()
print("--- After standardize_pandas_schema ---")
print(current_df.head())
print(current_df.dtypes)
print("\n")

# --- Initialize your individual transformers ---
# Note: Some transformers like DaysInRepairPredictor might need a valid model_uri
# For simulation, you might need a dummy URI or mock the model if not focusing on that.
# Assuming ttr_model_uri exists or is mocked for DaysInRepairPredictor
# For a full run, DaysInRepairPredictor's internal model.predict would need a valid model URI and schema.
# For this example, we'll comment it out if it causes issues without a real model.
#ttr_model_uri = "models:/your_ttr_model_name/Production" # Replace with actual URI or mock

splatmap_ext = SplatmapExtractor()
rename_cols = RenameColumnsTransformer()
impact_speed_trans = ImpactSpeedTransformer()
airbags_trans = AirbagCountTransformer()
car_chars_trans = CarCharacteristicTransformer()
date_features_trans = DateFeatureTransformer()
vehicle_age_trans = VehicleAgeTransformer()
rhd_trans = RHDTransformer()
body_key_enc = BodyKeyEncoder() # Assuming updated BodyKeyEncoder
postcode_ext = PostcodeAreaExtractor()
damage_trans = DamageTransformer()
damage_severity_calc = DamageSeverityCalculator()
general_imputer = GeneralPurposeImputer()
#days_in_repair_pred = DaysInRepairPredictor(model_uri=ttr_model_uri) # Potentially problematic without a real model

# --- Define the correct order of transformations based on your pipeline logic ---

# COMMAND ----------

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# COMMAND ----------

input_df.head()

# COMMAND ----------

# 1. Splatmap Extraction
current_df = splatmap_ext.transform(current_df)
print("--- After SplatmapExtractor ---")
current_df.head()

# COMMAND ----------

# 2. Column Renaming
current_df = rename_cols.transform(current_df)
print("--- After RenameColumnsTransformer ---")
current_df.head()

# COMMAND ----------

# 3. Impact Speed, Airbags, Car Chars, RHD (these often convert strings to numbers)
current_df = impact_speed_trans.transform(current_df)
current_df = airbags_trans.transform(current_df)
current_df = car_chars_trans.transform(current_df)
current_df = rhd_trans.transform(current_df)
print("--- After ImpactSpeed, Airbags, CarChars, RHD Transformers ---")
current_df.head()

# COMMAND ----------

# 4. Date Features and Vehicle Age Calculation (order matters here)
current_df = date_features_trans.transform(current_df)
current_df = vehicle_age_trans.transform(current_df) # TP vehicle age
print("--- After Date & Vehicle Age Transformers ---")
current_df.head()

# COMMAND ----------

# 5. Postcode Area Extraction
current_df = postcode_ext.transform(current_df)
print("--- After PostcodeAreaExtractor ---")
current_df.head()

# COMMAND ----------

# 6. Body Key Encoding (creates OHE columns and drops originals)
current_df = body_key_enc.transform(current_df)
print("--- After BodyKeyEncoder ---")
current_df.head()

# COMMAND ----------

# 7. Damage Recorded and Assessment
current_df = damage_trans.transform(current_df)
current_df = damage_severity_calc.transform(current_df)
print("--- After Damage Transformers ---")
current_df.head()

# COMMAND ----------

# 8. Imputation (should handle any remaining NaNs after feature creation)
current_df = general_imputer.fit_transform(current_df)
print("--- After GeneralPurposeImputer ---")
current_df.head()

# COMMAND ----------

nan_rows_df = current_df[current_df.isna().any(axis=1)]
nan_rows_df

# COMMAND ----------

#current_df = standardize_pandas_schema(current_df)

# COMMAND ----------

# --- Final ColumnTransformer (requires defining cat_ohe and num_ohe based on current_df state) ---
# To properly simulate, you'd need the actual cat_ohe and num_ohe lists used during training.
# For this example, let's infer them from the current_df, but ensure consistency with training.

# Infer cat_ohe and num_ohe after all custom transformers and imputation
# Adjust these lists based on your actual training setup and the columns present
# after all your custom transformers run, but BEFORE the final ColumnTransformer.
cat_ohe = current_df.select_dtypes(include=['object', 'category']).columns.tolist()
num_ohe = current_df.select_dtypes(include=np.number).columns.tolist()

# Exclude specific columns if they are not meant for OHE/scaling or are target/ID columns
# Example: If 'Claim Reference Number' or 'FP Registration' should just pass through or be dropped later
columns_to_exclude = ['claim_reference_number', 'fp_registration'] # Add any other non-feature columns

cat_ohe = [col for col in cat_ohe if col not in columns_to_exclude]
num_ohe = [col for col in num_ohe if col not in columns_to_exclude]

print(f"Categorical OHE columns: {cat_ohe}")
print(f"Numerical OHE columns: {num_ohe}")


feature_pp_transformer = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_ohe),
    ("num", StandardScaler(), num_ohe),
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")


# For a full simulation, you'd ideally use a fitted ColumnTransformer.
# Here, we'll fit it to the single row for demonstration.
# In a real scenario, the preprocessor pipeline (including ColumnTransformer) is fitted on X_train.
processed_features = feature_pp_transformer.fit_transform(current_df)
print("--- After ColumnTransformer (feature_pp) ---")
print(processed_features.head())
print(processed_features.dtypes)
print("\n")

# 9. DaysInRepairPredictor (if uncommented and functional)
# This step would predict days_in_repair and add it to the DataFrame.
# processed_features = days_in_repair_pred.transform(processed_features)
# print("--- After DaysInRepairPredictor ---")
# print(processed_features.head())
# print("\n")

print("\n--- Final Processed DataFrame Ready for Model Prediction ---")
print(processed_features.shape)
print(processed_features.columns.tolist())

# COMMAND ----------


print(f"Categorical OHE columns: {cat_ohe}")
print(f"Numerical OHE columns: {num_ohe}")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

def delete_model_from_uc_registry(model_name: str):
    """
    Deletes all versions of a specific model from the Databricks Unity Catalog
    Model Registry and then deletes the registered model itself.

    Args:
        model_name (str): The full name of the model in the Unity Catalog,
                          e.g., 'prod_dsexp_mlstore.third_party_capture.tpc_capturebenefitmodel'.
    """
    client = MlflowClient()

    print(f"Attempting to delete all versions of model: '{model_name}' from Unity Catalog.")

    try:
        # Get all versions of the model
        model_versions = client.search_model_versions(f"name='{model_name}'")

        if not model_versions:
            print(f"No versions found for model '{model_name}'.")
        else:
            # Iterate through all versions and delete them
            for mv in model_versions:
                version = mv.version
                print(f"Deleting version {version} of model '{model_name}'...")
                
                try:
                    # Transition to 'Archived' stage if necessary (though delete_model_version handles it)
                    # This step is often implicit with delete_model_version, but explicitly showing the flow.
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version,
                        stage="Archived" # Recommended to archive before deleting to ensure clean state
                    )
                    print(f"Version {version} of '{model_name}' transitioned to 'Archived'.")
                except MlflowException as e:
                    if "Cannot transition model version to stage 'Archived' from 'Archived'" in str(e):
                        print(f"Version {version} of '{model_name}' is already 'Archived'.")
                    else:
                        print(f"Warning: Could not transition version {version} to Archived: {e}")
                
                client.delete_model_version(name=model_name, version=version)
                print(f"Successfully deleted version {version} of model '{model_name}'.")

        # After deleting all versions, delete the registered model itself
        print(f"Deleting registered model '{model_name}'...")
        client.delete_registered_model(name=model_name)
        print(f"Successfully deleted registered model '{model_name}'.")

    except MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST: Registered model" in str(e) and "not found" in str(e):
            print(f"Registered model '{model_name}' does not exist.")
        elif "PERMISSION_DENIED" in str(e):
            print(f"Permission denied. Ensure you have DELETE permissions for model '{model_name}'. Error: {e}")
        else:
            print(f"An MLflow error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Specify the model name to delete
model_to_delete = 'prod_dsexp_mlstore.third_party_capture.lgbmregressor'

# Execute the deletion function
delete_model_from_uc_registry(model_to_delete)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The End