from mlflow.spark import load_model
import mlflow
from pyspark.sql import SparkSession, DataFrame, functions as F
import pandas as pd
from datetime import datetime
from mlflow.tracking import MlflowClient
from typing import List, Tuple
from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank, col, when, struct, lit, expr, udf, ceil
from builtins import abs as python_abs
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import Evaluator
import numpy as np
import xgboost as xgb

 
spark = SparkSession.builder.getOrCreate()


def optimized_collar_reduction(raw_df: DataFrame, column_name: str) -> DataFrame:
    """
    Optimized function to remove long tails in prediction distributions using collars,
    while retaining granularity for 200 buckets.
 
    Inputs:
    raw_df = input PySpark dataframe
    column_name = name of the column to reduce tails
 
    Outputs:
    raw_df_capped = output PySpark dataframe with new column 'column_name_capped'
    """
 
    # Define the quantiles for lower and upper collars (0.1% tails)
    quantiles = [0.001, 0.999]
 
    # Calculate the approximate quantiles for the specified column
    lower_collar, upper_collar = raw_df.approxQuantile(column_name, quantiles, 0.01)
 
    # Apply the capped values to the dataframe using a single transformation
    raw_df_capped = raw_df.withColumn(
        f'{column_name}_capped',
        F.when(F.col(column_name) > upper_collar, upper_collar)
         .when(F.col(column_name) < lower_collar, lower_collar)
         .otherwise(F.col(column_name))
    )
 
    return raw_df_capped

    
def iterative_collar_reduction(raw_df: DataFrame, column_name: str) -> DataFrame:
    """
    Optimized function to remove long tails in prediction distributions using collars, 
    while retaining granularity for 200 buckets.

    Inputs:
    raw_df = input PySpark dataframe
    column_name = name of the column to reduce tails

    Outputs:
    raw_df_capped = output PySpark dataframe with new column 'column_name_capped'
    """

    # Calculate total count of values in the given column once (no need to repeat this in every iteration)
    total_count = raw_df.count()

    # Determine initial upper and lower bounds (the min and max value in column) using a single aggregation call
    bounds = raw_df.agg(
        F.min(column_name).alias("min_val"), 
        F.max(column_name).alias("max_val")
    ).collect()[0]
    
    min_val = bounds["min_val"]
    max_val = bounds["max_val"]

    # Define the target frequency (0.1% of the total count)
    target_frequency = total_count * 0.001

    # Set initial collar values
    upper_collar = max_val
    lower_collar = min_val

    # Store initial reference bounds
    upper_collar_ref = max_val
    lower_collar_ref = min_val

    # Define helper function to calculate frequency in a more efficient manner
    def get_frequency(df: DataFrame, col_name: str, value: float, comparison_op: str) -> int:
        """
        Optimized function to get the frequency of values above or below a given value.
        The comparison_op is either 'ge' (>=) or 'le' (<=).
        """
        if comparison_op == 'ge':
            return df.filter(F.col(col_name) >= value).count()
        elif comparison_op == 'le':
            return df.filter(F.col(col_name) <= value).count()
        return 0

    # Initialize frequencies
    upper_frequency = get_frequency(raw_df, column_name, upper_collar, 'ge')
    lower_frequency = get_frequency(raw_df, column_name, lower_collar, 'le')

    # Start iterative adjustment of the collars
    while upper_frequency < target_frequency or lower_frequency < target_frequency:
        # Adjust upper collar if frequency is less than the target
        if upper_frequency < target_frequency:
            if upper_frequency >= target_frequency * 0.5:
                upper_collar -= abs(0.01 * upper_collar_ref)
            else:
                upper_collar -= abs(0.05 * upper_collar)

            # Recalculate frequency only when the upper collar is changed
            upper_frequency = get_frequency(raw_df, column_name, upper_collar, 'ge')

        # Adjust lower collar if frequency is less than the target
        if lower_frequency < target_frequency:
            if lower_frequency >= target_frequency * 0.5:
                lower_collar += abs(0.01 * lower_collar_ref)
            else:
                lower_collar += abs(0.05 * lower_collar)

            # Recalculate frequency only when the lower collar is changed
            lower_frequency = get_frequency(raw_df, column_name, lower_collar, 'le')

    # Apply the capped values to the dataframe using a single transformation
    raw_df_capped = raw_df.withColumn(
        f'{column_name}_capped',
        F.when(F.col(column_name) > upper_collar, upper_collar)
         .when(F.col(column_name) < lower_collar, lower_collar)
         .otherwise(F.col(column_name))
    )

    return raw_df_capped


def get_latest_model_version(client: MlflowClient,
                             model_name: str,
                             ) -> int:
    """
    Get the latest version of a model.

    Parameters:
    client (mlflow.tracking.MlflowClient): The MLflow client.
    model_name (str): The name of the model.

    Returns:
    int: The latest version of the model.
    """
    latest_version = 1
    for mv in client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def get_model_uri(registered_model_name: str,
                  alias: str,
                  ) -> Tuple:
    """
    Get the model URI and the loaded model.

    Parameters:
    model_name (str): The name of the model.
    alias (str): The alias of the model.

    Returns:
    Tuple[Any, str]: The loaded model and the model URI.
    """
    model_version_uri = f"models:/{registered_model_name}@{alias}"
    return model_version_uri


def batch_inference(registered_model_name: str,
                    alias: str,
                    data: DataFrame,
                    ) -> DataFrame:
    """
    Perform batch inference using a PyFunc model.

    Parameters:
    model_uri (str): The URI of the model.
    df (pd.DataFrame): The input DataFrame.
    cols (List[str]): The columns to use for inference.

    Returns:
    pd.DataFrame: The DataFrame with predictions.
    """
    model_version_uri = get_model_uri(registered_model_name, alias)
    model = load_model(model_version_uri)
    predictions_df = model.transform(data)

    return predictions_df


def batch_inference_xgb(registered_model_name: str,
                    alias: str,
                    data: xgb.DMatrix,
                    ) -> np.ndarray:
    """
    Perform batch inference using a PyFunc model.

    Parameters:
    model_uri (str): The URI of the model.
    df (pd.DataFrame): The input DataFrame.
    cols (List[str]): The columns to use for inference.

    Returns:
    pd.DataFrame: The DataFrame with predictions.
    """
    model_version_uri = get_model_uri(registered_model_name, alias)
    model = mlflow.xgboost.load_model(model_version_uri)
    predictions = model.predict(data)

    return predictions


def swap_alias(client: MlflowClient,
               model_name: str,
               ) -> None:
    """
    Swap the aliases of the model versions.

    Parameters:
    model_name (str): The name of the model.
    """
    champion_version = int(client.get_model_version_by_alias(model_name, "Champion").version)
    challenger_version = int(client.get_model_version_by_alias(model_name, "Challenger").version)

    client.delete_registered_model_alias(model_name,
                                         "Champion",
                                         )
    client.delete_registered_model_alias(model_name,
                                         "Challenger",
                                         )

    client.set_registered_model_alias(model_name,
                                      "Champion",
                                      challenger_version,
                                      )
    client.set_registered_model_alias(model_name,
                                      "Challenger",
                                      champion_version,
                                      )


def promote_challenger(client: MlflowClient,
                       model_name: str,
                       ) -> None:
    """
    Promote the challenger model to champion.

    Parameters:
    model_name (str): The name of the model.
    """
    challenger_version = int(client.get_model_version_by_alias(model_name, "Challenger").version)

    client.delete_registered_model_alias(model_name,
                                         "Champion",
                                         )
    client.delete_registered_model_alias(model_name,
                                         "Challenger",
                                         )

    client.set_registered_model_alias(model_name,
                                      "Champion",
                                      challenger_version,
                                      )
    



def prepare_data_for_scoring(
    data_path: str,
    unwanted_columns: List,
    data_loader_cols: List,
    model_names: List = [],
    ) -> DataFrame:
    """
    This aims to pre-process the data by adding or removing columns to stay congruent with the schema used for training

    Parameters:
    data_path: Path to dataset used for inference.
    unwanted_columns: List of columns to be dropped from the dataset.
    data_loader_cols: List of columns to be dropped from the dataset.
    model_names: List of columns to be dropped from the dataset.

    Returns:
    DataFrame with the predictions returned
    """ 
    data = spark.read.table(data_path)\
        .drop(*unwanted_columns)\
        .drop(*data_loader_cols)\
        .drop(*model_names)

    unwanted_updated_cols = [c for c in data.columns if c.endswith('_updated')]

    data = data.drop(*unwanted_updated_cols)

    data=data.fillna(0)

    return data



def create_uniform_index(raw_df, col_name):
    '''Function to create uniform index by ordering rows by value in given column, then defining percentile rank position and converting this to 200 bands [0.5-100]

    Inputs:
    raw_df = input pyspark dataframe
    col_name = name of column from which we create the uniform index

    Output:
    raw_df = output pyspark dataframe
    '''

    # Order rows by the value in given column
    windowSpec = Window.orderBy(col_name)

    # Calculate percentile rank and from this generate new banded_uniform_... column
    raw_df = raw_df.withColumn("percentile_rank", percent_rank().over(windowSpec))
    raw_df = raw_df.withColumn(f"banded_uniform_{col_name}", (col("percentile_rank") * 200).cast("int") * 0.5 + 0.5)

    # Ensure values do not exceed 100 and are not less than 0.5
    raw_df = raw_df.withColumn(f"banded_uniform_{col_name}", 
                               when(col(f"banded_uniform_{col_name}") > 100, 100)
                               .when(col(f"banded_uniform_{col_name}") < 0.5, 0.5)
                               .otherwise(col(f"banded_uniform_{col_name}")))
                               
    # Drop intermediate column
    raw_df = raw_df.drop('percentile_rank')

    return raw_df


# #Function to reduce tails in distribution of raw predictions
# def iterative_collar_reduction(raw_df, column_name):
#     '''This function aims to remove the very long tails in our prediction distributions. This is particularly important because we will be creating 200 buckets for the indices, so we want the smallest possible range whilst retaining as much information as possible to ensure maximum granularity in final indices. This is achieved by iteratively moving an upper and lower collar value until the frequency of values at the upper and lower collars is equal to 0.1% of the total count of values.

#     Inputs:
#     raw_df = input pyspark dataframe
#     column_name = name of column to reduce tails

#     Outputs:
#     raw_df_capped = output pyspark dataframe with new column 'column_name_capped'
#     '''
#     #Calculate total count of values in given column
#     total_count = raw_df.count()

#     #Determine initial upper and lower bounds (the min and max value in column)
#     max_val = raw_df.agg({column_name: "max"}).collect()[0][0]
#     min_val = raw_df.agg({column_name: "min"}).collect()[0][0]

#     #Define the target frequency (0.1% of the total count) - we will stop moving the collars once we reach this
#     target_frequency = total_count * 0.001

#     #Define initial collars as the min and max values from the column
#     upper_collar = max_val
#     lower_collar = min_val

#     #Save a version of the min & max as a reference
#     upper_collar_ref = max_val
#     lower_collar_ref = min_val

#     #Function to get the frequency of values outside of a given collar
#     def get_frequency(raw_df, col_name, value, comparison_op):
#         '''Function to get the frequency of values which fall above or below the given value

#         Inputs:
#         raw_df = input pyspark dataframe
#         col_name = name of column to get frequency from
#         value = threshold value to check frequency outside of
#         comparison_op = >= or <= showing whether we want to get the frequency >= value or <= value

#         Outputs:
#         Count of the number of values oustide of the threshold value. This defaults to 0 if the comparison_op is incorrect.
#         '''
#         if comparison_op == '>= ':
#             return raw_df.filter(col(col_name) >= value).count()
#         elif comparison_op == '<= ':
#             return raw_df.filter(col(col_name) <= value).count()
#         else:
#             return 0
    
#     #Iterate through this loop until upper_frequency >= target_frequency and lower_frequency >= target_frequency condition is met
#     while True:
#         #Calculate frequency of values above and below the current collars
#         upper_frequency = get_frequency(raw_df, column_name, upper_collar, '>= ')
#         lower_frequency = get_frequency(raw_df, column_name, lower_collar, '<= ')

#         #Check if both are above the target frequency (defined as 0.1% of total frequency)
#         if upper_frequency >= target_frequency and lower_frequency >= target_frequency:
#             break
        
#         #If upper frequency is below the target frequency, reduce upper_collar
#         if upper_frequency < target_frequency:
#             # Switch to smaller steps when close to the target frequency
#             #When upper_frequency <50% of target_frequency, reduce by 5% of current upper_collar value
#             #When upper_frequency >=50% of target_frequency, reduce by 1% of initial upper_collar value (done to ensure that steps aren't vanishingly small)
#             if upper_frequency >= target_frequency * 0.5:
#                 upper_collar -= python_abs(0.01 * upper_collar_ref)
#             else:
#                 upper_collar -= python_abs(0.05 * upper_collar)
        
#         if lower_frequency < target_frequency:
#             # Switch to smaller steps when close to the target frequency
#             #When lower_frequency <50% of target_frequency, increase by 5% of current lower_collar value
#             #When lower_frequency >=50% of target_frequency, increase by 1% of initial lower_collar value (done to ensure that steps aren't vanishingly small)
#             if lower_frequency >= target_frequency * 0.5:
#                 lower_collar += python_abs(0.01 * lower_collar_ref)
#             else:
#                 lower_collar += python_abs(0.05 * lower_collar)

#     #Apply the found collar values to the DataFrame
#     raw_df_capped = raw_df.withColumn(f'{column_name}_capped', 
#                                       when(col(column_name) > upper_collar, upper_collar)
#                                       .when(col(column_name) < lower_collar, lower_collar)
#                                       .otherwise(col(column_name)))

#     return raw_df_capped



def scale_and_band_columns(raw_df, columns_to_scale):
    '''Function to scale values to [0.5-100] and create 200 bands, by rounding values up to the nearest 0.5

        Inputs:
        raw_df = input pyspark dataframe
        columns_to_scale = list of columns to scale & band

        Outputs:
        raw_df = output pyspark_dataframe
    '''
    #Function to convert vector to double
    vector_to_double_udf = udf(lambda vector: float(vector[0]), FloatType())
    
    #For each column
    for col_name in columns_to_scale:
        #Assemble column into vector
        assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vec")
        raw_df = assembler.transform(raw_df)
        
        #Apply MinMaxScaler, with min 0.5 and max 100
        scaler = MinMaxScaler(inputCol=f"{col_name}_vec", outputCol=f"scaled_{col_name}_vec")
        scaler.setMin(0.5).setMax(100)
        model = scaler.fit(raw_df)
        raw_df = model.transform(raw_df)
        
        #Convert vector to scalar
        raw_df = raw_df.withColumn(f"scaled_{col_name}", vector_to_double_udf(col(f"scaled_{col_name}_vec")))
        
    #Drop intermediate vector columns
    columns_to_retain = ['pcd'] + [f"scaled_{col}" for col in columns_to_scale]
    raw_df = raw_df.select(*columns_to_retain)
    
    #Round values up to nearest 0.5, creating 200 bands
    for col_name in columns_to_scale:
        raw_df = raw_df.withColumn(f"banded_scaled_{col_name}", ceil(col(f"scaled_{col_name}") * 2) / 2)
    
    return raw_df



def rename_uniform_index_columns(old_name: str) -> str:
    if "prediction" in old_name:
        base_name = old_name.split("_prediction")[0]

        if "total" in base_name:
            if "burn_cost" in base_name:
                middle_string = "Total_BC"
            else:
                middle_string = "Total"
        elif "inc_tot" in base_name:
            middle_string = base_name.split("_inc_tot_")[1]
        elif "num_clm" in base_name:
            middle_string = base_name.split("_num_clm_")[1]
        else:
            middle_string = ""
        
        if "_inc_tot" in base_name:
            suffix = "Severity"
        elif "_num_clm" in base_name:
            suffix = "Frequency"
        else:
            suffix =  ""

    else:
        base_name = ""
        middle_string = ""
        suffix = ""

    if middle_string in ['ad', 'tpi', 'tppd', 'ws']:
        middle_string = middle_string.upper()
    elif "Total" in middle_string:
        middle_string = middle_string
    else:
        middle_string = middle_string.title()

    if base_name == "":
        new_name = old_name
    else:
        if "burn_cost" in base_name:
           new_name = f"Uniform_{middle_string}"
        else: 
            new_name = f"Uniform_{middle_string}_{suffix}"
    
    return new_name



def rename_relative_index_columns(old_name: str) -> str:
    if "prediction_capped" in old_name:
        base_name = old_name.split("_prediction_capped")[0]

        if "total" in base_name:
            if "burn_cost" in base_name:
                middle_string = "Total_BC"
            else:
                middle_string = "Total"
        elif "inc_tot" in base_name:
            middle_string = base_name.split("_inc_tot_")[1]
        elif "num_clm" in base_name:
            middle_string = base_name.split("_num_clm_")[1]
        else:
            middle_string = ""
        
        if "_inc_tot" in base_name:
            suffix = "Severity"
        elif "_num_clm" in base_name:
            suffix = "Frequency"
        else:
            suffix =  ""

    else:
        base_name = ""
        middle_string = ""
        suffix = ""

    if middle_string in ['ad', 'tpi', 'tppd', 'ws']:
        middle_string = middle_string.upper()
    elif "Total" in middle_string:
        middle_string = middle_string
    else:
        middle_string = middle_string.title()

    if base_name == "":
        new_name = old_name
    else:
        if "burn_cost" in base_name:
           new_name = f"Relative_{middle_string}"
        else: 
            new_name = f"Relative_{middle_string}_{suffix}"
    
    return new_name


class GiniEvaluator(Evaluator):
    """
    Custom evaluator that calculates the Gini coefficient for model evaluation.
    
    Attributes:
        predictionCol (str): The name of the column containing predicted values.
        labelCol (str): The name of the column containing actual labels.
    """
    def __init__(self, predictionCol="prediction", labelCol="label"):
        self.predictionCol = predictionCol
        self.labelCol = labelCol

    def _evaluate(self, dataset):
        # Convert the label and prediction columns to numpy arrays
        actual = np.array(dataset.select(col(self.labelCol)).rdd.flatMap(lambda x: x).collect())
        pred = np.array(dataset.select(col(self.predictionCol)).rdd.flatMap(lambda x: x).collect())
        
        # Calculate the normalized Gini coefficient
        gini_score = self.gini_normalized(actual, pred)
        return gini_score

    def gini(self, actual, pred):
        """Calculate the Gini coefficient."""
        assert len(actual) == len(pred), "Length of actual values and predicted values must be equal"
        all_data = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float64)
        all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
        total_losses = all_data[:, 0].sum()

        if total_losses == 0:
            return 0.0

        gini_sum = all_data[:, 0].cumsum().sum() / total_losses
        gini_sum -= (len(actual) + 1) / 2.

        return gini_sum / len(actual)