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

        input_df_standardized = standardize_pandas_schema(model_input.copy())

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