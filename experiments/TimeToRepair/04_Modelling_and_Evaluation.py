# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_gamma_deviance,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from scipy.stats import uniform, randint
from lightgbm import LGBMRegressor
import sys

sys.path.append(
    "/Workspace/Users/harry.bjarnason@first-central.com/CodingToolkit/coding_toolkit"
)
from interpret_ai.importance_cleaner import importance_cleaner
from model_evaluate.regression_metrics import regression_metrics
from model_evaluate.lift_chart_plots import plot_lift_chart_regression

# COMMAND ----------

mlflow.sklearn.autolog(max_tuning_runs=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Read

# COMMAND ----------

df_train = spark.read.parquet('/mnt/datalake/users/HarryBjarnason/TPC/TimeToRepair/train').toPandas()
df_test = spark.read.parquet('/mnt/datalake/users/HarryBjarnason/TPC/TimeToRepair/test').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing Pipeline

# COMMAND ----------

cat_features = ["ManufacturerDescription", "IncidentCauseDescription", "PostcodeArea"]
num_features = [
    "BootOpens",
    "DeployedAirbags",
    "Front",
    "FrontBonnet",
    "FrontLeft",
    "FrontRight",
    "Left",
    "LeftBackseat",
    "LeftFrontWheel",
    "LeftMirror",
    "LeftRearWheel",
    "LeftUnderside",
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
    "WindscreenDamage",
    "DoorsOpen",
    "Driveable",
    "EngineDamage",
    "ExhaustDamaged",
    "HailDamage",
    "LightsDamaged",
    "PanelGaps",
    "PassengerAreaSubmerged",
    "RadiatorDamaged",
    "SharpEdges",
    "WheelsDamaged",
    "WingMirrorDamaged",
    "Doors",
    "EngineCapacity",
    "Seats",
    "Value",
    "DA_DR",
    "DA_DTL",
    "DA_UTL",
    "DA_UR",
    "DA_O",
    "TimeToNotify",
    "VehicleAge",
    "BodyKey_01",
    "BodyKey_02",
    "BodyKey_03",
    "BodyKey_04",
    "FuelKey_01",
    "FuelKey_02",
    "Nature_PH",
    "Nature_TP",
    "Notified_DOW",
    "Notified_Day",
    "Notified_Month",
    "Notified_Year",
    "Incident_DOW",
    "Incident_Day",
    "DamageSev_Total",
    "DamageSev_Count",
    "DamageSev_Mean",
    "DamageSev_Max",
]

predictive_features = cat_features + num_features

# Pipeline for categorical features (with OneHotEncoding)
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

# create preprocessor stages
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_pipeline, cat_features),
        ("num", 'passthrough', num_features),
    ]
)

# create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            LGBMRegressor(
                learning_rate=0.1,
                importance_type="gain",
                objective="gamma",
                random_state=100,
                verbose=-1,
            ),
        ),
    ]
)

# COMMAND ----------

# split into X and y
train_X = df_train[predictive_features]
train_y = df_train["DaysInRepair"]
test_X = df_test[predictive_features]
test_y = df_test["DaysInRepair"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### HP Tuning

# COMMAND ----------

param_distributions = {
    "regressor__n_estimators": randint(350, 1001),
    "regressor__max_depth": [2, 3, 4, 5],
    "regressor__min_child_weight": uniform(loc=0, scale=0.2),
    "regressor__subsample": [0.8, 0.9, 1.0],
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=100,
    cv=5,
    verbose=0,
    random_state=100,
    n_jobs=-1,
    scoring=[
        "r2",
        "neg_root_mean_squared_error",
        "neg_mean_absolute_error",
    ],
    refit=False,
)

# Assuming X and y are your features and target variables
random_search.fit(train_X, train_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Model
# MAGIC

# COMMAND ----------

# fit final model on full training set
best_run = mlflow.get_run("76482a8d55f64864915249678f0193bf")

# double check
print(best_run.data.metrics)
print(best_run.data.params["regressor__max_depth"])

# fetch best params
best_params = {
    "regressor__max_depth": best_run.data.params["regressor__max_depth"],
    "regressor__min_child_weight": best_run.data.params["regressor__min_child_weight"],
    "regressor__subsample": best_run.data.params["regressor__subsample"],
    "regressor__n_estimators": int(best_run.data.params["regressor__n_estimators"]),
}

print("\n\n", best_params)

# COMMAND ----------

# fit model on full train data
pipeline.set_params(**best_params)
pipeline.fit(train_X, train_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation

# COMMAND ----------

final_model = mlflow.sklearn.load_model("runs:/0827ba5e08b94d5ea6741236c35f2791/model")
df_train["preds"] = final_model.predict(train_X)
df_train["actuals"] = train_y

df_test["preds"] = final_model.predict(test_X)
df_test["actuals"] = test_y

# COMMAND ----------

print(df_train['preds'].mean())
print(df_train['actuals'].mean())
print(df_test['preds'].mean())
print(df_test['actuals'].mean())

# COMMAND ----------

# regression metrics
regression_metrics(preds=df_test.preds, y_test=df_test.actuals)

# COMMAND ----------

# performance by yr
yr_results = {}
for yr in df_test.Notified_Year.unique():
    df_yr = df_test[df_test.Notified_Year == yr]
    yr_results[str(yr)] = regression_metrics(preds=df_yr.preds, y_test=df_yr.actuals)

yr_results = pd.DataFrame(yr_results)[["2021", "2022", "2023"]].transpose()
yr_results["rmse"].plot(marker="x")
yr_results["mae"].plot(marker="x")
plt.legend()
plt.show()
yr_results["r2"].plot(marker="x")
plt.legend()
plt.show()

# COMMAND ----------

#histogram of errors
df_test['error'] = df_test['preds'] - df_test['actuals']
df_test['error'].hist(bins=25)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpertation

# COMMAND ----------

# feature importance

# get feature importance:
feature_importances = final_model.named_steps["regressor"].feature_importances_
feature_names = final_model[:-1].get_feature_names_out()

# Create importance dict
importances = dict(map(lambda i, j: (i, j), feature_names, feature_importances))

# get cat and num names lists
names_series = pd.Series(['cat__' + x for x in cat_features] + ['num__' + x for x in num_features])
cat_feats = names_series[(names_series.str.startswith("cat"))]
num_feats = names_series[(names_series.str.startswith("num"))]

# clean and join importances
cleaned_importances = importance_cleaner(importances, cat_feats, num_feats)
# remove pipeline stage names from the feature list
cleaned_importances.index = cleaned_importances.index.str.extract(r"__(.*)")[0]

# plot top 50
cleaned_importances.iloc[:50].sort_values().plot(
    kind="barh",
    figsize=(8, 10),
    title="Top 50 Feature Importance Share",
    color="deepskyblue",
    ylabel="Feature",
    grid=True,
)

plt.show()