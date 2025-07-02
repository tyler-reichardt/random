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

# MAGIC %md
# MAGIC ## Library Imports and Configuration Setup
# MAGIC
# MAGIC This section of the notebook focuses on setting up the necessary environment for model training by performing the following actions:

# COMMAND ----------

import sys
import re
import os
import yaml
import time
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import shutil

from pyspark.sql.functions import *
from pyspark.sql import Window, DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.sql.types import *
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

from scipy.stats import randint, uniform

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from xgboost.spark import SparkXGBRegressor
import xgboost as xgb

from lightgbm import LGBMRegressor

from mlflow.models.signature import ModelSignature, Schema, infer_signature
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, ColSpec

from typing import Tuple, List, Dict, Any

import joblib
import warnings
warnings.filterwarnings("ignore")

with open(f'{config_path}', 'r') as file:
    config = yaml.safe_load(file)

from functions.training import *

# Extract the congig lists
extract_column_transformation_lists("/config_files/training.yaml")
extract_column_transformation_lists("/config_files/configs.yaml")

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
catalog = workspaces.get(workspace_url) + catalog_prefix

# Define model name and URI for TTR and TPC
ttr_model_name = f"{catalog}.third_party_capture.time_to_repair_lgbm"
ttr_model_uri  = f"models:/{ttr_model_name}@champion"

tpc_model_name = f"{catalog}.third_party_capture.third_party_capture_lgbm" 
tpc_model_uri  = f"models:/{tpc_model_name}@champion" 

# COMMAND ----------

import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import mlflow # Keeping relevant MLflow imports for context of classes
from sklearn.pipeline import Pipeline # Explicitly import Pipeline to ensure it's sklearn's
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Keep these as they are used in ColumnTransformer
from sklearn.compose import ColumnTransformer # Keep this as it's used in your preprocessor pipeline


# Note: The Spark-related imports and functions (e.g., from pyspark.sql, pyspark.ml, SparkXGBRegressor,
# train_and_evaluate_cv_original, train_and_evaluate, log_best_model, create_mlflow_experiment_path,
# GiniEvaluator, etc.) are outside this specific code block. They would need to be removed or adapted
# if the intention is a pure scikit-learn/pandas-based workflow with LGBMRegressor.
# This response focuses only on the provided snippet.

def standardize_pandas_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes a DataFrame by cleaning column names, handling missing values,
    and coercing data types, while explicitly ignoring columns that are already
    in a datetime format.

    Args:
        df (pd.DataFrame): The input DataFrame to standardize.

    Returns:
        pd.DataFrame: A new DataFrame with standardized schema.
    """
    df_copy = df.copy()

    # --- 1. Standardize column names ---
    # This step applies to all columns, including datetime columns.
    new_columns = []
    for col in df_copy.columns:
        new_col = str(col).lower()
        new_col = re.sub(r'[^a-z0-9_]+', '_', new_col)
        new_col = re.sub(r'_+', '_', new_col)
        new_col = new_col.strip('_')
        new_columns.append(new_col)
    df_copy.columns = new_columns

    # --- 2. Convert common string representations of missing values to np.nan ---
    # This loop correctly targets only 'object' type columns, so it will
    # naturally skip any pre-existing datetime columns.
    missing_value_strings = [
        'none', 'null', 'nan', '', ' ', 'na', 'n/a', 'not available', 
        'unknown', '-', '#n/a', '#n/a n/a', 'n/a (not applicable)', 'unspecified'
    ]
    
    for col in df_copy.select_dtypes(include=['object']).columns:
        # Using .loc to ensure modification happens on the DataFrame copy
        mask = df_copy[col].astype(str).str.lower().isin(missing_value_strings)
        df_copy.loc[mask, col] = np.nan

    # --- 3. Coerce data types for non-datetime columns ---
    # Iterate through columns to attempt type conversion.
    for col in df_copy.columns:
        # --- FIX ---
        # First, check if the column is already a datetime type. If so, skip it
        # to prevent any modification to its values.
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            continue

        # Attempt to convert to numeric, coercing errors to NaN
        # This block will only run for non-datetime columns.
        converted_numeric = pd.to_numeric(df_copy[col], errors='coerce')
        if not converted_numeric.isnull().all():
            df_copy[col] = converted_numeric
        else:
            # If not cleanly numeric, attempt to convert to datetime.
            # This is useful for object columns that contain date strings.
            converted_datetime = pd.to_datetime(df_copy[col], errors='coerce')
            if not converted_datetime.isnull().all():
                df_copy[col] = converted_datetime
            # If neither numeric nor datetime, it remains as its original (cleaned) type,
            # likely 'object', with string NaNs already handled.

    return df_copy

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
        new_cols = []
        if 'FP Splatmap Data' in df.columns and 'TP Splatmap Data' in df.columns:
            for col in self.splatmap_cols:
                df[f'FP_{col}'] = df['FP Splatmap Data'].apply(
                    lambda x: x.get(col) if isinstance(x, dict) else np.nan)
                df[f'TP_{col}'] = df['TP Splatmap Data'].apply(
                    lambda x: x.get(col) if isinstance(x, dict) else np.nan)
                new_cols.extend([f'FP_{col}', f'TP_{col}'])
            df = df.drop(columns=['FP Splatmap Data', 'TP Splatmap Data'], errors='ignore')
        # Return new DataFrame with old columns + new columns, dropping used ones
        # Need to re-order columns if 'get_feature_names_out' is to be truly accurate by position.
        # For simplicity here, just ensuring old columns are kept.
        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            # If no input_features are provided, assume original column names before this transformer
            # This is a simplification; a more robust solution might track them during fit.
            # For a pipeline, input_features will typically be passed from the previous step.
            raise ValueError("input_features must be provided to get_feature_names_out for SplatmapExtractor.")

        output_features = list(input_features)
        
        # Remove columns that will be dropped
        if 'FP Splatmap Data' in output_features:
            output_features.remove('FP Splatmap Data')
        if 'TP Splatmap Data' in output_features:
            output_features.remove('TP Splatmap Data')

        # Add new columns
        for col in self.splatmap_cols:
            output_features.append(f'FP_{col}')
            output_features.append(f'TP_{col}')
        return np.array(output_features)


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
        # No need to filter columns here unless explicitly dropping non-renamed ones.
        # The pipeline handles passing all columns, and subsequent steps will select as needed.
        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for RenameColumnsTransformer.")
        
        output_features = []
        for feature in input_features:
            output_features.append(self.rename_map.get(feature, feature)) # Use .get to return original if not in map
        return np.array(output_features)


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
            cols_to_transform_num = [col for col in self.numerical_cols_ if col in df.columns]
            if cols_to_transform_num:
                imputed_numerical_data = self.num_imputer_.transform(df[cols_to_transform_num])
                df[cols_to_transform_num] = pd.DataFrame(
                    imputed_numerical_data,
                    columns=cols_to_transform_num,
                    index=df.index
                )

        # Impute categorical columns
        if self.cat_imputer_ and self.categorical_cols_:
            cols_to_transform_cat = [col for col in self.categorical_cols_ if col in df.columns]
            if cols_to_transform_cat:
                imputed_categorical_data = self.cat_imputer_.transform(df[cols_to_transform_cat])
                df[cols_to_transform_cat] = pd.DataFrame(
                    imputed_categorical_data,
                    columns=cols_to_transform_cat,
                    index=df.index
                )

        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for GeneralPurposeImputer.")
        return np.array(list(input_features)) # Imputer does not change column names or add new ones


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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for ImpactSpeedTransformer.")
        
        output_features = [f for f in input_features if f not in ['impact_speed_range', 'impact_speed_unit']]
        if 'impact_speed' not in output_features: # Ensure 'impact_speed' is present
            output_features.append('impact_speed')
        return np.array(output_features)


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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for AirbagCountTransformer.")
        # This transformer modifies existing columns, no new columns are added.
        return np.array(list(input_features))


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

        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for CarCharacteristicTransformer.")
        
        output_features = list(input_features)
        
        # Add new columns created
        new_cols = ['da_dr', 'da_dtl', 'da_utl', 'da_ur', 'da_o']
        for col in new_cols:
            if col not in output_features:
                output_features.append(col)
        
#        # Remove dropped columns
#        if 'tp_driveable_damage_assessment' in output_features:
#            output_features.remove('tp_driveable_damage_assessment')
#        if 'fp_driveable_damage_assessment' in output_features:
#            output_features.remove('fp_driveable_damage_assessment')
        
        # Note: Columns converted from 'yes'/'no' to int like 'fp_boot_opens'
        # retain their names but change dtype. This method only tracks names.
        return np.array(output_features)


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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for DateFeatureTransformer.")
        
        output_features = list(input_features)
        new_cols = [
            'time_to_notify', 'notified_day_of_week', 'notified_day',
            'notified_month', 'notified_year', 'incident_day_of_week', 'incident_day'
        ]
        for col in new_cols:
            if col not in output_features:
                output_features.append(col)
        # This transformer does not drop original date columns, VehicleAgeTransformer does.
        return np.array(output_features)


# ─── 8. Vehicle Age ─────────────────────────────────────────────────────────────

class VehicleAgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['tp_vehicle_age'] = df['incident_date'].dt.year - df['year_of_manufacture'].astype(int)
        df['tp_vehicle_age'] = df['tp_vehicle_age'].where(df['tp_vehicle_age']<=30, 0).fillna(0)
        return df.drop(['notification_date', 'incident_date'], axis = 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for VehicleAgeTransformer.")
        
        output_features = [f for f in input_features if f not in ['notification_date', 'incident_date']]
        if 'tp_vehicle_age' not in output_features:
            output_features.append('tp_vehicle_age')
        return np.array(output_features)


# ─── 9. Right-Hand Drive Flag ──────────────────────────────────────────────────

class RHDTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['tp_right_hand_drive'] = (df['tp_right_hand_drive']=='R').astype(int)
        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for RHDTransformer.")
        return np.array(list(input_features)) # Modifies in place, no new or dropped columns by name


# ─── 10. Body Key One-Hot Encoding ──────────────────────────────────────────────

class BodyKeyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, keys=None):
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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for BodyKeyEncoder.")
        
        output_features = [f for f in input_features if f not in ['fp_body_key', 'tp_body_key']]
        for i, _ in enumerate(self.keys, start=1):
            output_features.append(f'fp_body_key_0{i}')
            output_features.append(f'tp_body_key_0{i}')
        return np.array(output_features)


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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for PostcodeAreaExtractor.")
        return np.array(list(input_features)) # Modifies in place, no new or dropped columns by name


# ─── 12. Drop Any Leftovers ─────────────────────────────────────────────────────

class DropUnusedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop=None):
        self.to_drop = to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.to_drop, errors='ignore')
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for DropUnusedColumns.")
        # Filter out columns that will be dropped
        output_features = [f for f in input_features if f not in self.to_drop]
        return np.array(output_features)


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
        df['damage_recorded'] = df[self.severity_columns].notnull().sum(axis=1) # Counts non-null severity columns
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
                else: # Catch any other non-defined values that are not null
                    scale = 0
            else: # If tp_damage_assessed is 0 (no damage recorded/assessed)
                scale = -1 # Or a more appropriate default for no damage
            return scale

        for col_name in self.severity_columns:
            df[col_name] = df.apply(lambda row: damage_scale(row[col_name], row['tp_damage_assessed']), axis=1).astype(int)
        
        for col_name in self.fp_severity_columns:
            df[col_name] = df.apply(lambda row: damage_scale(row[col_name], row['tp_damage_assessed']), axis=1).astype(int)

        return df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for DamageTransformer.")
        
        output_features = list(input_features)
        new_cols = ['damage_recorded', 'tp_damage_assessed']
        for col in new_cols:
            if col not in output_features:
                output_features.append(col)
        # This transformer modifies existing severity columns in place,
        # but does not drop them.
        return np.array(output_features)


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
        # Ensure division by zero is handled for damage_sev_mean
        df['damage_sev_mean'] = df.apply(lambda row: row['damage_sev_total'] / row['damage_sev_count'] if row['damage_sev_count'] > 0 else 0, axis=1)
        df['damage_sev_max'] = df[self.sev_damage].max(axis=1)
        
        return df
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for DamageSeverityCalculator.")
        
        output_features = list(input_features)
        new_cols = ['damage_sev_total', 'damage_sev_count', 'damage_sev_mean', 'damage_sev_max']
        for col in new_cols:
            if col not in output_features:
                output_features.append(col)
        return np.array(output_features)


class DaysInRepairPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model_uri):
        self.model_uri = model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.input_cols_from_schema = list(self.model.metadata.get_input_schema().input_names())
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
                if expected_type_str in ['string', 'object', 'bytes']:
                    X_for_model[col_name] = ""
                elif expected_type_str in ['integer', 'long', 'float', 'double']:
                    X_for_model[col_name] = 0
                else:
                    X_for_model[col_name] = np.nan
                print(f"Warning: Column '{col_name}' was missing from input and defaulted.")

        for col_name in self.input_cols_from_schema:
            if col_name not in X_for_model.columns:
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
                    if pd.api.types.is_float_dtype(actual_dtype):
                        if X_for_model[col_name].isnull().any():
                            X_for_model[col_name] = X_for_model[col_name].fillna(0)
                        X_for_model[col_name] = X_for_model[col_name].astype(np.int64)
                    elif not pd.api.types.is_integer_dtype(actual_dtype):
                        X_for_model[col_name] = X_for_model[col_name].astype(np.int64)
            except Exception as e:
                print(f"Error coercing column '{col_name}' to {expected_type_str}. Current dtype: {actual_dtype}. Error: {e}")

        X_for_model = X_for_model[self.input_cols_from_schema]

        predictions = self.model.predict(X_for_model)
        X_original_for_output["days_in_repair"] = predictions
        
        return X_original_for_output

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out for DaysInRepairPredictor.")
        
        output_features = list(input_features)
        if 'days_in_repair' not in output_features:
            output_features.append('days_in_repair')
        return np.array(output_features)


class CaptureBenefitModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        print("Loading model context...")
        self.preprocessor = mlflow.sklearn.load_model(context.artifacts["preprocessor_model_path"])
        print(f"Preprocessor loaded from: {context.artifacts['preprocessor_model_path']}")

        self.regressor = mlflow.sklearn.load_model(context.artifacts["regressor_model_path"])
        print(f"Regressor loaded from: {context.artifacts['regressor_model_path']}")

        quantile_table_path = context.artifacts["quantile_benefit_table_csv_path"]
        print(f"Attempting to load quantile benefit table from: {quantile_table_path}")
        self.quantile_benefit_df = pd.read_csv(quantile_table_path)

        # Correctly initialize and load quantile_threshold_df
        self.quantile_threshold_df = None # Initialize to None first for robustness
        quantile_threshold_table_path = context.artifacts.get("quantile_threshold_table_csv_path") 
        
        if quantile_threshold_table_path: 
            if os.path.exists(quantile_threshold_table_path):
                print(f"Attempting to load quantile threshold table from: {quantile_threshold_table_path}")
                self.quantile_threshold_df = pd.read_csv(quantile_threshold_table_path)
                # IMPORTANT: Ensure thresholds are sorted ascending for correct sequential checking
                self.quantile_threshold_df = self.quantile_threshold_df.sort_values(by='quantile_threshold').reset_index(drop=True)
                print("Quantile threshold table loaded.")
            else:
                print(f"Warning: Quantile threshold table CSV not found at '{quantile_threshold_table_path}'. Benefit priority categorization will use a default.")
        else:
            print("Warning: 'quantile_threshold_table_csv_path' artifact not provided. Benefit priority categorization will use a default.")


    def _get_quantile_benefit_multiplier(self, raw_capture_benefit_value: float) -> float:
        if self.quantile_benefit_df is None:
            print("Quantile benefit table not loaded. Returning default multiplier of 1.0.")
            return 1.0 

        for _, row in self.quantile_benefit_df.iterrows():
            if row['capture_benefit_lower_bound'] <= raw_capture_benefit_value < row['capture_benefit_upper_bound']:
                return row['quant_benefit_multiplier']
        
        last_row = self.quantile_benefit_df.iloc[-1]
        if raw_capture_benefit_value == last_row['capture_benefit_upper_bound'] and last_row['capture_benefit_upper_bound'] == np.inf :
             return last_row['quant_benefit_multiplier']

        return 1.0
    
    def _get_benefit_priority_category(self, adjusted_capture_benefit_value: float) -> str:
        """
        Determines the 'Bronze', 'Silver', 'Gold' category based on adjusted capture benefit.
        """
        # FIX: Check if thresholds DataFrame is properly loaded
        if self.quantile_threshold_df is None or self.quantile_threshold_df.empty:
            return "Bronze" # Default if thresholds are not loaded or empty
        
        # Based on your previous logic, values below q1 were "Low".
        category = "Bronze" # Default for values below all defined tiers

        # Iterate through the thresholds to find the correct bin.
        # Iterating in ascending order allows for simple if-checks for upper bounds.
        # We process 'bins' by their lower thresholds.
        
        # Make sure to handle cases where a 'quantile_threshold_int' might be missing.
        # Convert thresholds to a dict for easier lookup, with sensible defaults (like 0 or inf)
        threshold_map = self.quantile_threshold_df.set_index('quantile_threshold_int')['quantile_threshold'].to_dict()

        # Get the specific thresholds, using -inf or inf for robustness if a key is missing
        t_bronze = threshold_map.get(1.00, np.inf) 
        t_silver = threshold_map.get(2.00, np.inf)
        t_gold = threshold_map.get(3.00, np.inf)

        # Assign category based on strict ranges
        if adjusted_capture_benefit_value >= t_gold:
            category = "Gold"
        elif adjusted_capture_benefit_value >= t_silver:
            category = "Silver"
        elif adjusted_capture_benefit_value >= t_bronze:
            category = "Bronze"

        return category


    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame: # Changed return type to DataFrame

        print(f"Received input with {model_input.shape[0]} rows and {model_input.shape[1]} columns.")
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # --- Prediction with capturesuccess_adrienneversion = 1 ---
        df_cap = model_input.copy()
        df_cap['capturesuccess_adrienneversion'] = 1.0 

        # --- Prediction with capturesuccess_adrienneversion = 0 ---
        df_nocap = model_input.copy()
        df_nocap['capturesuccess_adrienneversion'] = 0.0

        print("Preprocessing data for 'capture' scenario...")
        try:
            processed_cap_features = self.preprocessor.transform(df_cap)
        except Exception as e:
            print(f"Error transforming df_cap: {e}")
            raise RuntimeError(f"Error during preprocessing 'capture' data: {e}")
        
        print("Preprocessing data for 'no capture' scenario...")
        try:
            processed_nocap_features = self.preprocessor.transform(df_nocap)
        except Exception as e:
            print(f"Error transforming df_nocap: {e}")
            raise RuntimeError(f"Error during preprocessing 'no capture' data: {e}")

        print("Making predictions...")
        tppd_pred_capture = self.regressor.predict(processed_cap_features)
        tppd_pred_no_capture = self.regressor.predict(processed_nocap_features)

        # Calculate raw Capture_Benefit
        # This assumes tppd_pred_capture is greater than tppd_pred_no_capture for a positive benefit.
        raw_capture_benefit = tppd_pred_capture - tppd_pred_no_capture 
        
        print(f"Raw capture benefit calculated. Min: {np.min(raw_capture_benefit)}, Max: {np.max(raw_capture_benefit)}")

        # Create a DataFrame to hold results, making it easier to add new columns
        results_df = pd.DataFrame(index=model_input.index)
        results_df['TPPD_Pred_Capture'] = tppd_pred_capture
        results_df['TPPD_Pred_No_Capture'] = tppd_pred_no_capture
        results_df['Raw_Capture_Benefit'] = raw_capture_benefit

        # Apply adjustment multiplier
        results_df['Capture_Benefit_Adjusted'] = results_df['Raw_Capture_Benefit'] * results_df['Raw_Capture_Benefit'].apply(self._get_quantile_benefit_multiplier)
        
        print(f"Adjusted capture benefit calculated. Min: {np.min(results_df['Capture_Benefit_Adjusted'])}, Max: {np.max(results_df['Capture_Benefit_Adjusted'])}")

        # --- Apply Benefit Priority Logic ---
        results_df['Capture_Benefit_Priority'] = results_df['Capture_Benefit_Adjusted'].apply(self._get_benefit_priority_category)
        print("Benefit Priority assigned to results_df.")

        results_df.drop(['TPPD_Pred_Capture', 'TPPD_Pred_No_Capture', 'Raw_Capture_Benefit'], axis = 1, inplace=True)
        
        print("Prediction process complete.")
        return results_df

class DataCleaner(BaseEstimator, TransformerMixin):
    """
    A multi-purpose cleaner for DataFrames that standardizes data and metadata.
    It can be configured to perform one of two actions based on the 'mode' parameter:

    1. 'normalize_values':
       - Normalizes the text data within specified columns by converting all values
         to lowercase and stripping leading/trailing whitespace.
       - This should be used BEFORE one-hot encoding to prevent duplicate categories
         that only differ by case (e.g., 'Hit In Rear' vs. 'hit in rear').

    2. 'standardize_columns':
       - Standardizes all column names in the DataFrame by converting them to lowercase,
         replacing any non-alphanumeric characters with a single underscore, and
         ensuring uniqueness by appending a suffix to any resulting duplicates.
       - This should be used AFTER one-hot encoding to make feature names
         compatible with ML models like LightGBM.
    """
    def __init__(self, mode='standardize_columns', columns_to_normalize=None):
        """
        Initializes the DataCleaner.

        Args:
            mode (str): The operation to perform. Must be either 'standardize_columns' or 'normalize_values'.
            columns_to_normalize (list, optional): A list of column names whose text values should be normalized.
                                                    This parameter is required and used only for 'normalize_values' mode.
        """
        if mode not in ['standardize_columns', 'normalize_values']:
            raise ValueError("mode must be either 'standardize_columns' or 'normalize_values'")
        if mode == 'normalize_values' and columns_to_normalize is None:
            raise ValueError("The 'columns_to_normalize' parameter must be provided when mode is 'normalize_values'.")
        
        self.mode = mode
        self.columns_to_normalize = columns_to_normalize

    def fit(self, X, y=None):
        """This transformer does not learn from the data, so the fit method does nothing."""
        return self

    def transform(self, X):
        """Applies the configured cleaning transformation based on the initialized mode."""
        if not isinstance(X, pd.DataFrame):
            return X # Pass through if not a DataFrame

        if self.mode == 'normalize_values':
            return self._normalize_values(X)
        elif self.mode == 'standardize_columns':
            return self._standardize_columns(X)

    def _normalize_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Private method to normalize text values in specified columns."""
        X_copy = X.copy()
        for col in self.columns_to_normalize:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(str).str.lower().str.strip()
        return X_copy

    def _standardize_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Private method to standardize all DataFrame column names."""
        X_copy = X.copy()
        
        # Standardize names: convert to lower, then replace invalid characters.
        standardized_cols = [
            re.sub(r'[^a-z0-9_]+', '_', col.lower()) for col in X_copy.columns
        ]
        
        # Handle potential duplicates
        final_cols = []
        counts = {}
        for col in standardized_cols:
            counts[col] = counts.get(col, 0) + 1
            if counts[col] > 1:
                final_cols.append(f"{col}_{counts[col]-1}")
            else:
                final_cols.append(col)
        
        X_copy.columns = final_cols
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Generates output feature names for scikit-learn pipeline compatibility."""
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out.")
        
        if self.mode == 'normalize_values':
            # This mode does not change column names.
            return np.array(input_features)
        
        elif self.mode == 'standardize_columns':
            standardized_cols = [re.sub(r'[^a-z0-9_]+', '_', col.lower()) for col in input_features]
            final_cols = []
            counts = {}
            for col in standardized_cols:
                counts[col] = counts.get(col, 0) + 1
                if counts[col] > 1:
                    final_cols.append(f"{col}_{counts[col]-1}")
                else:
                    final_cols.append(col)
            return np.array(final_cols)
        
class DropUnusedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop=None):
        self.to_drop = to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.to_drop, errors='ignore')
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided.")
        return np.array([f for f in input_features if f not in self.to_drop])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dummy API Request Data for Configuration of Schema

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
tpc_api_df = pd.DataFrame(sample_data)

display(tpc_api_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training & Tracking for TTR
# MAGIC
# MAGIC This section of the notebook loops through each of the columns to be modelled, creates a model for each and tracks them via `MLFlow`.

# COMMAND ----------

# Load data to pandas
df = standardize_pandas_schema(spark.table(f"{workspaces.get(workspace_url)}auxiliarydata.third_party_capture.ttr_df").toPandas())

# Map FP_ and TP_ columns to match API request
df = process_dataframe(df, rename_map, fp_tp_columns)

# Drop specified columns
df = df.drop(
    [
        'first_party_confirmed',
        'tp_panel_gaps',
        'tp_sharp_edges',
        'tp_number_of_seats',
        'tp_vehicle_value',
        'body_key'
    ],
    axis=1
)

# Strip postcode_area of spaces and make all capitals
df['postcode_area'] = df['postcode_area'].str.strip().str.upper()
df = df.drop('fp_vehicle_age', axis = 1)

# COMMAND ----------

# Create categorical and numerical feature lists
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove 'days_in_repair' from numerical_features
numerical_features = [item for item in numerical_features if item not in ['days_in_repair', 'fp_driveable_damage_assessment', 'tp_driveable_damage_assessment']]

# Create full listr of features to be modelled
ttr_features_list = categorical_features + numerical_features

# COMMAND ----------

# Filter for features and dropna()
df = df.dropna(subset=numerical_features + categorical_features + [ttr_target_column])

# Separate features and target
X_raw_features = df[numerical_features + categorical_features]
y_series = df[ttr_target_column]

# --- FIX: Normalize categorical values BEFORE one-hot encoding ---
# This loop ensures that variations like 'Hit In Rear' and 'Hit in rear'
# are treated as the same category by the OneHotEncoder.
for col in categorical_features:
    if col in X_raw_features.columns:
        X_raw_features[col] = X_raw_features[col].astype(str).str.lower().str.strip()


# --- (Scaler and OneHotEncoder logic remains here) ---
scaler = StandardScaler()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Scale Numerical Features
scaled_numerical_data = scaler.fit_transform(X_raw_features[numerical_features])
df_scaled_numerical = pd.DataFrame(scaled_numerical_data, columns=numerical_features, index=X_raw_features.index)

# OneHotEncode the now-normalized Categorical Features
encoded_categorical_data = ohe.fit_transform(X_raw_features[categorical_features])
ohe_feature_names = ohe.get_feature_names_out(categorical_features)
df_encoded_categorical = pd.DataFrame(encoded_categorical_data, columns=ohe_feature_names, index=X_raw_features.index)


# Concatenate processed feature DataFrames
X_processed = pd.concat([df_scaled_numerical, df_encoded_categorical], axis=1)

# Standardize the COLUMN NAMES created by the steps above.
name_standardizer = DataCleaner()
X_processed = name_standardizer.transform(X_processed)

# The rest of the script continues...
# No duplicates with '_1' will be generated now.
ttr_features_list = X_processed.columns.tolist()
X = X_processed

y = y_series

# Ensure y is a 1D array for train_test_split and model fitting
y_ravel = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, test_size=0.2, random_state=42)

# Regressor instance
lgbm_estimator = LGBMRegressor(
    learning_rate=0.1,
    importance_type="gain",
    objective="gamma",
    random_state=42,
    verbose=-1
)

# Hyperparameter search
fixed_params_for_lgbm = {
    'max_depth': 3,
    'min_child_weight': 0.06268251410514163,
    'subsample': 1.0,
    'n_estimators': 664
}
param_distributions_for_search = {key: [value] for key, value in fixed_params_for_lgbm.items()}

random_search = RandomizedSearchCV(
    estimator=lgbm_estimator,
    param_distributions=param_distributions_for_search,
    n_iter=1,
    cv=3,
    random_state=100,
    n_jobs=-1,
    scoring="r2",
    refit=True,
)

# MLflow
client = MlflowClient()

# Define schemas for MLflow logging
input_cols = [ColSpec(type="double", name=col_name) for col_name in ttr_features_list]
output_cols = [ColSpec(type="double", name=ttr_target_column)]
input_schema = Schema(input_cols)
output_schema = Schema(output_cols)

with mlflow.start_run(run_name="time_to_repair_lgbm_direct_training") as run:
    random_search.fit(X_train, y_train)
    best_lgbm_model = random_search.best_estimator_
    mlflow.log_params(best_lgbm_model.get_params())

    raw_preds = best_lgbm_model.predict(X_test)
    preds_rounded_up = np.ceil(raw_preds)

    r2 = r2_score(y_test, preds_rounded_up)
    rmse = mean_squared_error(y_test, preds_rounded_up, squared=False)
    mae = mean_absolute_error(y_test, preds_rounded_up)

    mlflow.log_metric("r2_rounded", r2)
    mlflow.log_metric("rmse_rounded", rmse)
    mlflow.log_metric("mae_rounded", mae)

    class SklearnCompatibleRoundedUpModelWrapper:
        def __init__(self, model):
            self.model = model
            if hasattr(model, '_estimator_type'):
                self._estimator_type = model._estimator_type
            if hasattr(model, 'feature_name_'):
                self.feature_name_ = model.feature_name_
            if hasattr(model, 'n_features_in_'):
                self.n_features_in_ = model.n_features_in_

        def predict(self, X_input):
            return np.ceil(self.model.predict(X_input))

    sklearn_compatible_wrapped_model = SklearnCompatibleRoundedUpModelWrapper(best_lgbm_model)

    mlflow.sklearn.log_model(
        sk_model=sklearn_compatible_wrapped_model,
        artifact_path="model",
        signature=ModelSignature(inputs=input_schema, outputs=output_schema),
        registered_model_name=ttr_model_name
    )

    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Best Params Used: {random_search.best_params_}")
    print(f"R² (rounded): {r2:.4f}, RMSE (rounded): {rmse:.4f}, MAE (rounded): {mae:.4f}")

# COMMAND ----------

version = get_latest_model_version(client, ttr_model_name)
alias = "champion"
client.set_registered_model_alias(ttr_model_name, alias, version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training & Tracking for TPC
# MAGIC
# MAGIC This section of the notebook loops through each of the columns to be modelled, creates a model for each and tracks them via `MLFlow`.

# COMMAND ----------

# Define numerical and categorical features for TPC model based on API request
numerical_features = tpc_api_df.select_dtypes(include=['int64']).columns.tolist()
categorical_features = tpc_api_df.select_dtypes(include=['object']).columns.tolist()

# Define input schema manually from Pandas DataFrame
input_schema = Schema(
    [ColSpec("double", col) for col in tpc_api_df[numerical_features].columns] +
    [ColSpec("string", col) for col in tpc_api_df[categorical_features].columns]
)

# Define output schema (assume regression output)
output_schema = Schema([
    ColSpec("string", 'Capture_Benefit_Priority')] + [ColSpec("double", 'Capture_Benefit')])

# COMMAND ----------

import pandas as pd
import numpy as np

feature_table_name = f"{catalog}.third_party_capture.tpc_feature_table"
df_tpc = standardize_pandas_schema(spark.table(feature_table_name).toPandas())

# COMMAND ----------

cat_features = ['insurer_name',
 'incident_cause_description',
 'postcode_area',
 'tp_body_key',
 'fp_body_key',
 'notification_method',
 'incident_sub_cause_description',
 'impact_speed_range',
 'impact_speed_unit',
 'tp_colour',
 'incident_uk_country',
 'road_conditions',
 'weather_conditions']

num_features = ['capturesuccess_adrienneversion', 'year_of_manufacture', 'tp_engine_capacity', 'tp_number_of_doors', 'tp_boot_opens', 'tp_right_hand_drive', 'tp_doors_open', 'tp_not_on_mid', 'tp_vehicle_unattended', 'tp_wheels_damaged', 'fp_deployed_airbags', 'tp_deployed_airbags', 'tp_lights_damaged', 'tp_front_severity', 'tp_front_bonnet_severity', 'tp_front_left_severity', 'tp_front_right_severity', 'tp_left_severity', 'tp_left_back_seat_severity', 'tp_left_front_wheel_severity', 'tp_left_mirror_severity', 'tp_left_rear_wheel_severity', 'tp_left_roof_severity', 'tp_left_underside_severity', 'tp_load_area_severity', 'tp_rear_severity', 'tp_rear_left_severity', 'tp_rear_right_severity', 'tp_rear_window_damage_severity', 'tp_right_severity', 'tp_right_back_seat_severity', 'tp_right_front_wheel_severity', 'tp_right_mirror_severity', 'tp_right_rear_wheel_severity', 'tp_right_roof_severity', 'tp_right_underside_severity', 'tp_roof_damage_severity', 'tp_underbody_damage_severity', 'tp_windscreen_damage_severity', 'impact_speed', 'time_to_notify', 'notified_day_of_week', 'notified_day', 'notified_month', 'notified_year', 'incident_day_of_week', 'incident_day', 'tp_vehicle_age', 'damage_recorded', 'tp_damage_assessed', 'damage_sev_total', 'damage_sev_count', 'damage_sev_mean', 'damage_sev_max', 'fp_boot_opens', 'fp_doors_open', 'fp_engine_capacity', 'fp_front_bonnet_severity', 'fp_front_left_severity', 'fp_front_right_severity', 'fp_front_severity', 'fp_left_back_seat_severity', 'fp_left_front_wheel_severity', 'fp_left_mirror_severity', 'fp_left_rear_wheel_severity', 'fp_left_roof_severity', 'fp_left_severity', 'fp_left_underside_severity', 'fp_lights_damaged', 'fp_load_area_severity', 'fp_number_of_doors', 'fp_number_of_seats', 'fp_panel_gaps', 'fp_radiator_damaged', 'fp_rear_left_severity', 'fp_rear_right_severity', 'fp_rear_severity', 'fp_rear_window_damage_severity', 'fp_right_back_seat_severity', 'fp_right_front_wheel_severity', 'fp_right_mirror_severity', 'fp_right_rear_wheel_severity', 'fp_right_roof_severity', 'fp_right_severity', 'fp_right_underside_severity', 'fp_roof_damage_severity', 'fp_sharp_edges', 'fp_underbody_damage_severity', 'fp_vehicle_value', 'fp_wheels_damaged', 'fp_windscreen_damage_severity', 'tp_mileage']

# COMMAND ----------

cat_ohe = ['insurer_name',
'incident_cause_description',
'postcode_area',
'notification_method',
'incident_sub_cause_description',
'tp_colour',
'incident_uk_country',
'road_conditions',
'weather_conditions']

num_std = ['fp_number_of_doors', 'fp_engine_capacity', 'fp_number_of_seats', 'tp_mileage', 'year_of_manufacture', 'fp_right_roof_severity', 'impact_speed', 'time_to_notify', 'notified_day_of_week', 'notified_day', 'notified_month', 'notified_year', 'incident_day_of_week', 'incident_day', 'tp_vehicle_age', 'tp_mileage']

# COMMAND ----------

cat_impute = ['postcode_area', 
              'incident_sub_cause_description', 
              'incident_uk_country',
              'incident_cause_description', 
              'insurer_name', 
              'notification_method', 
              'road_conditions', 
              'tp_colour', 
              'weather_conditions']
num_impute = ['fp_boot_opens', 'fp_deployed_airbags', 'fp_number_of_doors', 'fp_doors_open', 'fp_engine_capacity', 'vehicle_kept_at_postcode', 'fp_lights_damaged', 'fp_number_of_seats', 'fp_vehicle_value', 'fp_wheels_damaged', 'tp_boot_opens', 'tp_deployed_airbags', 'tp_number_of_doors', 'tp_doors_open', 'tp_engine_capacity', 'tp_right_hand_drive', 'tp_lights_damaged', 'tp_mileage', 'tp_not_on_mid', 'tp_vehicle_unattended', 'year_of_manufacture', 'tp_wheels_damaged', 'tp_front_severity', 'tp_front_bonnet_severity', 'tp_front_left_severity', 'tp_front_right_severity', 'tp_left_severity', 'tp_left_back_seat_severity', 'tp_left_front_wheel_severity', 'tp_left_mirror_severity', 'tp_left_rear_wheel_severity', 'tp_left_roof_severity', 'tp_left_underside_severity', 'tp_load_area_severity', 'tp_rear_severity', 'tp_rear_left_severity', 'tp_rear_right_severity', 'tp_rear_window_damage_severity', 'tp_right_severity', 'tp_right_back_seat_severity', 'tp_right_front_wheel_severity', 'tp_right_mirror_severity', 'tp_right_rear_wheel_severity', 'tp_right_roof_severity', 'tp_right_underside_severity', 'tp_roof_damage_severity', 'tp_underbody_damage_severity', 'tp_windscreen_damage_severity', 'impact_speed', 'da_dr', 'da_dtl', 'da_utl', 'da_ur', 'da_o', 'time_to_notify', 'notified_day_of_week', 'notified_day', 'notified_month', 'notified_year', 'incident_day_of_week', 'incident_day', 'tp_vehicle_age', 'fp_body_key_01', 'tp_body_key_01', 'fp_body_key_02', 'tp_body_key_02', 'fp_body_key_03', 'tp_body_key_03', 'fp_body_key_04', 'tp_body_key_04', 'damage_recorded', 'tp_damage_assessed', 'damage_sev_total', 'damage_sev_count', 'damage_sev_mean', 'damage_sev_max']

# COMMAND ----------

features_list = cat_features + num_features

# COMMAND ----------

preprocessor = Pipeline([
    ('splatmap_extractor', SplatmapExtractor()),
    ('rename_columns', RenameColumnsTransformer()),
    ('none_to_nan', StringNoneToNaNConverter()),
    ('impact_speed', ImpactSpeedTransformer()),
    ('airbags', AirbagCountTransformer()),
    ('car_chars', CarCharacteristicTransformer()),
    ('date_features', DateFeatureTransformer()),
    ('vehicle_age', VehicleAgeTransformer()),
    ('rhd_transform', RHDTransformer()),
    ('body_key_ohe', BodyKeyEncoder()),
    ('postcode_extract', PostcodeAreaExtractor()), 
    ('damage_transform', DamageTransformer()),
    ('damage_severity_calc', DamageSeverityCalculator()),
    ('imputer', GeneralPurposeImputer(numerical_features=num_impute, categorical_features=cat_impute)), 
    ("feature_pp", ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_ohe),
        ("num", StandardScaler(), num_std),
    ], remainder="passthrough",
       verbose_feature_names_out=False).set_output(transform="pandas")),
    ('name_cleaner', DataCleaner ()),
    ('days_in_repair_pred', DaysInRepairPredictor(model_uri=ttr_model_uri))
])

# --- LOAD & SPLIT DATA (as in your script) ---
df_tpc = df_tpc[features_list + date_columns + [tpc_target_column]]
df_tpc["inc_tot_tppd"] = df_tpc["inc_tot_tppd"].fillna(0)
df_tpc = df_tpc[(df_tpc['inc_tot_tppd'] > 0) & (df_tpc['inc_tot_tppd'] < 20000)]
X = df_tpc.drop("inc_tot_tppd", axis=1)
y = df_tpc["inc_tot_tppd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MLflow SETUP ---

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()


with mlflow.start_run(run_name="third_party_capture_with_benefit_logic") as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    
    full_training_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LGBMRegressor(
                learning_rate=0.1,
                importance_type="gain",
                objective="gamma",
                random_state=100,
                verbose=-1,
            )) 
    ])

    param_distributions = {'regressor__max_depth': [3], 
                            'regressor__min_child_weight': [0.19852844748059362], 
                            'regressor__min_split_gain': [0.0], 
                            'regressor__subsample': [0.8], 
                            'regressor__n_estimators': [950]}

    random_search = RandomizedSearchCV(
        full_training_pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,  
        random_state=100,
        n_jobs=-1,
        scoring="r2", # Only one metric
        refit=True, # Back to True, since we have one scoring metric
    )

    print("Starting model training (RandomizedSearchCV)...")
    random_search.fit(X_train, y_train)
    best_overall_model = random_search.best_estimator_
    print("Model training complete.")

    # --- Apply monotonicity constraint ---
    # Use the fitted preprocessor from the best_overall_model to get the actual feature names
    fitted_preprocessor_component_from_best_model = best_overall_model.named_steps["preprocessor"]
    
    # Transform a small sample of X_train through the *entire* fitted preprocessor
    X_train_transformed_for_names = fitted_preprocessor_component_from_best_model.transform(X_train.head(10))
    all_feat_names_transformed = X_train_transformed_for_names.columns.tolist()

    monotonicity_dict = {
        # IMPORTANT: Verify this name against `all_feat_names_transformed` printout.
        # It's highly probable it will be 'num__capturecuccess_adrienneversion'
        "capturecuccess_adrienneversion": -1 
    }

    monotonicity_list = []
    for feat_name in all_feat_names_transformed:
        if feat_name in monotonicity_dict:
            monotonicity_list.append(monotonicity_dict[feat_name])
        else:
            base_feat_name = feat_name.split('__')[-1] # Fallback to checking by base name
            if base_feat_name in monotonicity_dict:
                monotonicity_list.append(monotonicity_dict[base_feat_name])
            else:
                monotonicity_list.append(0)

    best_overall_model.named_steps["regressor"].set_params(monotone_constraints=monotonicity_list)
    
    print("Re-fitting best model with monotonicity constraints...")
    best_overall_model.fit(X_train, y_train) # Re-fit to apply constraint
    print("Model re-fitting complete with monotonicity constraints.")

    fitted_preprocessor_component = best_overall_model.named_steps['preprocessor']
    fitted_regressor_component = best_overall_model.named_steps['regressor']

    best_params = {k.replace("regressor__", ""): v for k, v in random_search.best_params_.items()}
    best_params['monotone_constraints'] = str(monotonicity_list) # Convert to string for logging
    mlflow.log_params(best_params)

    preds_test = best_overall_model.predict(X_test)
    r2 = r2_score(y_test, preds_test)
    rmse = mean_squared_error(y_test, preds_test, squared=False)
    mae = mean_absolute_error(y_test, preds_test)
    mlflow.log_metric("r2_base", r2)
    mlflow.log_metric("rmse_base", rmse)
    mlflow.log_metric("mae_base", mae)

    preprocessor_artifact_path = "pipeline_artifacts/fitted_preprocessor_pipeline"
    regressor_artifact_path = "pipeline_artifacts/fitted_regressor_model"
    quantile_benefit_csv_local_path = "pipeline_artifacts/quantile_benefit_lookup.csv"
    quantile_threshold_csv_local_path = "pipeline_artifacts/quantile_threshold_lookup.csv"

    for path in [preprocessor_artifact_path, regressor_artifact_path, quantile_benefit_csv_local_path, quantile_threshold_csv_local_path]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    mlflow.sklearn.save_model(fitted_preprocessor_component, preprocessor_artifact_path)
    print(f"Fitted preprocessor saved to: {preprocessor_artifact_path}")

    mlflow.sklearn.save_model(fitted_regressor_component, regressor_artifact_path)
    print(f"Fitted regressor saved to: {regressor_artifact_path}")
    
    quantile_benefit_pd_df = spark.read.table('prod_dsexp_auxiliarydata.third_party_capture.quant_benefit_adjustments_dedupe_fix').toPandas()
    
    quantile_benefit_pd_df[['capture_benefit_lower_bound', 'capture_benefit_upper_bound']] = quantile_benefit_pd_df['inc_tot_tppd_qcut'].str.extract(r'\(([^,]+), ([^\]]+)\]')
    quantile_benefit_pd_df['capture_benefit_lower_bound'] = quantile_benefit_pd_df['capture_benefit_lower_bound'].astype(float)
    quantile_benefit_pd_df['capture_benefit_upper_bound'] = quantile_benefit_pd_df['capture_benefit_upper_bound'].astype(float)

    quantile_benefit_pd_df.to_csv(quantile_benefit_csv_local_path, index=False)
    print(f"Quantile benefit lookup table saved to: {quantile_benefit_csv_local_path}")

    quantile_threshold_pd_df = spark.read.table('prod_dsexp_auxiliarydata.third_party_capture.capture_benefit_prioritisation_thresholds_dedupe_fix').toPandas()
    quantile_threshold_pd_df.to_csv(quantile_threshold_csv_local_path, index=False)
    print(f"Quantile threshold lookup table saved to: {quantile_threshold_csv_local_path}")

    artifacts_for_pyfunc = {
        "preprocessor_model_path": preprocessor_artifact_path, 
        "regressor_model_path": regressor_artifact_path, 
        "quantile_benefit_table_csv_path": quantile_benefit_csv_local_path,
        "quantile_threshold_table_csv_path": quantile_threshold_csv_local_path
    }

    temp_model_instance = CaptureBenefitModel()
    class TempContext:
        def __init__(self, artifacts):
            self.artifacts = artifacts
    temp_context = TempContext(artifacts_for_pyfunc)
    temp_model_instance.load_context(temp_context)

    print("Logging custom CaptureBenefitModel to MLflow...")
    mlflow.pyfunc.log_model(
        artifact_path="capture_benefit_pyfunc_model", 
        python_model=CaptureBenefitModel(),
        artifacts=artifacts_for_pyfunc,
        signature=ModelSignature(inputs=input_schema, outputs=output_schema),
        registered_model_name=tpc_model_name,
    )
    print(f"Custom model '{tpc_model_name}' logged successfully.")

print("MLflow run finished.")

# COMMAND ----------

version = get_latest_model_version(client, tpc_model_name)
alias = "champion"
client.set_registered_model_alias(tpc_model_name, alias, version)

# COMMAND ----------

print(f"Base Model Metrics: R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The End

# COMMAND ----------

tpc_api_df

# COMMAND ----------

transformers = [

    ('splatmap_extractor', SplatmapExtractor()),
    ('rename_columns', RenameColumnsTransformer()),
    ('none_to_nan', StringNoneToNaNConverter()),
    ('impact_speed', ImpactSpeedTransformer()),
    ('airbags', AirbagCountTransformer()),
    ('car_chars', CarCharacteristicTransformer()),
    ('date_features', DateFeatureTransformer()),
    ('vehicle_age', VehicleAgeTransformer()),
    ('rhd_transform', RHDTransformer()),
    ('body_key_ohe', BodyKeyEncoder()),
    ('postcode_extract', PostcodeAreaExtractor()), 
    ('damage_transform', DamageTransformer()),
    ('damage_severity_calc', DamageSeverityCalculator()),
    ('imputer', GeneralPurposeImputer(numerical_features=num_impute, categorical_features=cat_impute)), 
    ('value_normalizer', DataCleaner(mode='normalize_values', columns_to_normalize=cat_ohe)),
    ("feature_pp", ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_ohe),
        ("num", StandardScaler(), num_std),
    ], remainder="passthrough",
       verbose_feature_names_out=False).set_output(transform="pandas")),
    ('name_cleaner', DataCleaner ()),
    ('days_in_repair_pred', DaysInRepairPredictor(model_uri=ttr_model_uri)),
]
tpc_api_df_new = tpc_api_df
for name, transformer in transformers:
    print(transformer)
    tpc_api_df_new = transformer.fit_transform(tpc_api_df_new)
    display(tpc_api_df_new)

# COMMAND ----------

