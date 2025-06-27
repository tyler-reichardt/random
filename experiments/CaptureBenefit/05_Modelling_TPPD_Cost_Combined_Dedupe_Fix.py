# Databricks notebook source
# MAGIC %md
# MAGIC ###### Author: Harry Bjarnason
# MAGIC ###### Notebook Purpose: Create modelling pipeline, perform hyperaprameter tuning, fit final model and model evaluation

# COMMAND ----------

pip show scikit-learn

# COMMAND ----------

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
    "/Workspace/Users/harry.bjarnason@first-central.com/Projects/TPC Third Party Capture/2024_07_TPC_Capture_Benefit/CodingToolkit/"
)
from interpret_ai.importance_cleaner import importance_cleaner
from model_evaluate.regression_metrics import regression_metrics
from model_evaluate.lift_chart_plots import plot_lift_chart_regression

# COMMAND ----------

mlflow.sklearn.autolog(max_tuning_runs=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset Creation

# COMMAND ----------

# read data
df = pd.read_csv(
    "/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2_cleaned/capture_benefit_df_20240806_dupe_clm_no_fix.csv"
)
df["NotificationDate"] = pd.to_datetime(df["NotificationDate"])

# cap at 20k
df = df[df.INC_TOT_TPPD < 20000]

print(df.shape)

# COMMAND ----------

# gonna use capture flag as a modelling feature so fillna with 0 for it
df["CaptureSuccess_AdrienneVersion"] = df["CaptureSuccess_AdrienneVersion"].fillna(0)

# COMMAND ----------

# keep date range 1/1/21 - 1/4/24 only
df = df[(df.NotificationDate > "2021-01-01") & (df.NotificationDate < "2024-04-01")]
print(df.shape)

# COMMAND ----------

# train test split (85-15)
df_train, df_test = train_test_split(
    df, test_size=0.15, train_size=0.85, random_state=100
)
print(df_train.shape)
print(df_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Preprocessing Pipeline

# COMMAND ----------

# feature lists - all predictive features
predictive_features = [
    "CaptureSuccess_AdrienneVersion",
    "ColourDescription",
    "CurrentMileage",
    "DamageAssessment",
    "FP_BodyKey",
    "FP_DamageAssessedInd",
    "FP_DamageSev_Count",
    "FP_DamageSev_Max",
    "FP_DamageSev_Mean",
    "FP_DamageSev_Total",
    "FP_Doors",
    "FP_Driveable",
    "FP_EngineCapacity",
    "FP_Front",
    "FP_FrontBonnet",
    "FP_FrontLeft",
    "FP_FrontRight",
    "FP_Left",
    "FP_LeftBackseat",
    "FP_LeftFrontWheel",
    "FP_LeftMirror",
    "FP_LeftRearWheel",
    "FP_LeftUnderside",
    "FP_Rear",
    "FP_RearLeft",
    "FP_RearRight",
    "FP_RearWindowDamage",
    "FP_Right",
    "FP_RightBackseat",
    "FP_RightFrontWheel",
    "FP_RightMirror",
    "FP_RightRearWheel",
    "FP_RightRoof",
    "FP_RightUnderside",
    "FP_RoofDamage",
    "FP_Seats",
    "FP_UnderbodyDamage",
    "FP_WindscreenDamage",
    "ImpactSpeedRange_int",
    "Incident_Day",
    "Incident_DOW",
    "IncidentCauseDescription",
    "IncidentSubCauseDescription",
    "IncidentUKCountry",
    "InsurerName1",
    "IsRightHandDrive",
    "NotificationMethod",
    "Notified_Day",
    "Notified_DOW",
    "Notified_Month",
    "Notified_Year",
    "NotOnMID",
    "PostcodeArea",
    "prediction",
    "WetRoad_Ind",
    "TimeToNotify",
    "TP_BodyKey",
    "TP_DamageAssessedInd",
    "TP_DamageSev_Count",
    "TP_DamageSev_Max",
    "TP_DamageSev_Mean",
    "TP_DamageSev_Total",
    "TP_DeployedAirbags",
    "FP_DeployedAirbags",
    "TP_Doors",
    "TP_Driveable",
    "TP_EngineCapacity",
    "TP_Front",
    "TP_FrontBonnet",
    "TP_FrontLeft",
    "TP_FrontRight",
    "TP_FuelKey",
    "TP_Left",
    "TP_LeftBackseat",
    "TP_LeftFrontWheel",
    "TP_LeftMirror",
    "TP_LeftRearWheel",
    "TP_LeftUnderside",
    "TP_ManufacturerDescription",
    "TP_Rear",
    "TP_RearLeft",
    "TP_RearRight",
    "TP_RearWindowDamage",
    "TP_Right",
    "TP_RightBackseat",
    "TP_RightFrontWheel",
    "TP_RightMirror",
    "TP_RightRearWheel",
    "TP_RightRoof",
    "TP_RightUnderside",
    "TP_RoofDamage",
    "TP_Seats",
    "TP_UnderbodyDamage",
    "TP_VehicleAge",
    "TP_WindscreenDamage",
    "VehicleUnattended",
    "RainyWeather_Ind",
    "FP_Value",
    "FP_DoorsOpen",
    "TP_DoorsOpen",
    "FP_BootOpens",
    "TP_BootOpens",
    "FP_RadiatorDamaged",
    "FP_LightsDamaged",
    "TP_LightsDamaged",
    "FP_PanelGaps",
    "FP_SharpEdges",
    "TP_WheelsDamaged",
    "FP_WheelsDamaged",
]


# now split into imputation methods and dtypees
categorical_features_mode = [
    "IncidentUKCountry",
    "TP_BodyKey",
    "TP_FuelKey",
    "FP_BodyKey",
    "ColourDescription",
    "NotificationMethod",
]

categorical_features_fixed = [
    "DamageAssessment",
    "TP_ManufacturerDescription",
    "InsurerName1",
    "PostcodeArea",
    "IncidentCauseDescription",
    "IncidentSubCauseDescription",
]

mode_impute_features = [
    "FP_Doors",
    "FP_Seats",
    "TP_Seats",
    "TP_DamageSev_Total",
    "TP_Doors",
    "TP_Driveable",
    "TP_EngineCapacity",
    "TP_Front",
    "TP_FrontBonnet",
    "TP_FrontLeft",
    "TP_FrontRight",
    "TP_Left",
    "TP_LeftBackseat",
    "TP_LeftFrontWheel",
    "TP_LeftMirror",
    "TP_LeftRearWheel",
    "TP_LeftUnderside",
    "TP_Rear",
    "TP_RearLeft",
    "TP_RearRight",
    "TP_RearWindowDamage",
    "TP_Right",
    "TP_RightBackseat",
    "TP_RightFrontWheel",
    "TP_RightMirror",
    "TP_RightRearWheel",
    "TP_RightRoof",
    "TP_RightUnderside",
    "TP_RoofDamage",
    "TP_UnderbodyDamage",
    "TP_WindscreenDamage",
    "VehicleUnattended",
    "RainyWeather_Ind",
    "FP_RadiatorDamaged",
    "FP_LightsDamaged",
    "TP_LightsDamaged",
    "FP_PanelGaps",
    "FP_SharpEdges",
    "TP_WheelsDamaged",
    "FP_WheelsDamaged",
    "FP_DamageSev_Count",
    "FP_DamageSev_Max",
    "FP_DamageSev_Total",
    "FP_EngineCapacity",
    "WetRoad_Ind",
    "TimeToNotify",
    "TP_DamageAssessedInd",
    "TP_DamageSev_Count",
    "TP_DamageSev_Max",
    "Notified_Day",
    "Notified_DOW",
    "Notified_Month",
    "Notified_Year",
    "NotOnMID",
    "FP_DamageAssessedInd",
    "FP_Driveable",
    "FP_Front",
    "FP_FrontBonnet",
    "FP_FrontLeft",
    "FP_FrontRight",
    "FP_Left",
    "FP_LeftBackseat",
    "FP_LeftFrontWheel",
    "FP_LeftMirror",
    "FP_LeftRearWheel",
    "FP_LeftUnderside",
    "FP_Rear",
    "FP_RearLeft",
    "FP_RearRight",
    "FP_RearWindowDamage",
    "FP_Right",
    "FP_RightBackseat",
    "FP_RightFrontWheel",
    "FP_RightMirror",
    "FP_RightRearWheel",
    "FP_RightRoof",
    "FP_RightUnderside",
    "FP_RoofDamage",
    "FP_UnderbodyDamage",
    "FP_WindscreenDamage",
    "ImpactSpeedRange_int",
    "Incident_Day",
    "Incident_DOW",
    "IsRightHandDrive",
    "TP_BootOpens",
    "FP_BootOpens",
    "TP_DoorsOpen",
    "FP_DoorsOpen",
    "TP_DeployedAirbags",
    "FP_DeployedAirbags",
]

mean_impute_features = [
    "prediction",
    "FP_Value",
    "CurrentMileage",
    "TP_VehicleAge",
    "FP_DamageSev_Mean",
    "TP_DamageSev_Mean",
    "CaptureSuccess_AdrienneVersion",
]

# Define the fixed values for categorical fixed value imputation:
fixed_values_dict = {
    "DamageAssessment": "Unknown",
    "TP_ManufacturerDescription": "Other",
    "InsurerName1": "Other",
    "PostcodeArea": "ZZ",
    "IncidentCauseDescription": "Unknown",
    "IncidentSubCauseDescription": "Unknown",
}

# COMMAND ----------

# Pipelines for all the fixed categorical value imputations - here we create a list of pipeline, one pipeline for each feature
categorical_fixed_pipelines = [
    (
        col + "_constant",  # name of the pipeline
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=fixed_val)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        ),  # the pipeline itself
        [col], # the column that will use this pipeline
    )  
    for col, fixed_val in fixed_values_dict.items()
]  # list comprehension looping through keys and values from the dictionary


# Pipeline for categorical features (with OneHotEncoding)
categorical_mode_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Pipeline for mode imputation
mode_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

# Pipeline for mean imputation
mean_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])


# create preprocessor stages
preprocessor = ColumnTransformer(
    transformers=categorical_fixed_pipelines
    + [
        ("cat_mode", categorical_mode_pipeline, categorical_features_mode),
        ("mode", mode_pipeline, mode_impute_features),
        ("mean", mean_pipeline, mean_impute_features),
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
train_y = df_train["INC_TOT_TPPD"]
test_X = df_test[predictive_features]
test_y = df_test["INC_TOT_TPPD"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### HP Tuning

# COMMAND ----------

param_distributions = {
    "regressor__n_estimators": [
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950,
        1000,
    ],
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
        "neg_mean_gamma_deviance",
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
# MAGIC ###Final Model Fit

# COMMAND ----------

# monotonicity constraints for final model:
monotonicity_dict = {"prediction": 1, "CaptureSuccess_AdrienneVersion": -1}

# get feature names from the pipeline, removing pipeline labels
all_feat_names = pd.Series(
    pipeline.named_steps["preprocessor"].get_feature_names_out()
).str.extract(r"__(.*)")[0]

monotonicity_list = [
    monotonicity_dict[x] if x in monotonicity_dict.keys() else 0 for x in all_feat_names
]

# COMMAND ----------

# fit final model on full training set
best_run = mlflow.get_run("0b1d19a93bff4dea97a52950248c78b8")

# double check
print(best_run.data.metrics)
print(best_run.data.params["regressor__max_depth"])

# fetch best params
best_params = {
    "regressor__max_depth": best_run.data.params["regressor__max_depth"],
    "regressor__min_child_weight": best_run.data.params["regressor__min_child_weight"],
    "regressor__min_split_gain": best_run.data.params["regressor__min_split_gain"],
    "regressor__subsample": best_run.data.params["regressor__subsample"],
    "regressor__n_estimators": int(best_run.data.params["regressor__n_estimators"]),
}

print("\n\n", best_params)

# COMMAND ----------

# fit model on full train data
pipeline.set_params(**best_params)
pipeline.set_params(regressor__monotone_constraints=monotonicity_list)
pipeline["regressor"]
pipeline.fit(train_X, train_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Model Evaluation

# COMMAND ----------

final_model = mlflow.pyfunc.load_model("runs:/d551491340ab4a1aa64ad182174b21ba/model")
df_train["preds"] = final_model.predict(train_X)
df_train["actuals"] = train_y

df_test["preds"] = final_model.predict(test_X)
df_test["actuals"] = test_y

# COMMAND ----------

df_test['preds'].hist(bins=50)

# COMMAND ----------

print(df_train['preds'].mean())
print(df_train['actuals'].mean())
print(df_test['preds'].mean())
print(df_test['actuals'].mean())

# COMMAND ----------

# regression metrics
print(regression_metrics(preds=df_test.preds, y_test=df_test.actuals))

# COMMAND ----------

# capture cases only
print(
    regression_metrics(
        preds=df_test[df_test.CaptureSuccess_AdrienneVersion == 1].preds,
        y_test=df_test[df_test.CaptureSuccess_AdrienneVersion == 1].actuals,
    )
)

# COMMAND ----------

# non captured cases only
print(
    regression_metrics(
        preds=df_test[df_test.CaptureSuccess_AdrienneVersion != 1].preds,
        y_test=df_test[df_test.CaptureSuccess_AdrienneVersion != 1].actuals,
    )
)

# COMMAND ----------

# performance by yr
yr_results = {}
for yr in df_test.Notified_Year.unique():
    df_yr = df_test[df_test.Notified_Year == yr]
    yr_results[str(yr)] = regression_metrics(preds=df_yr.preds, y_test=df_yr.actuals)

yr_results = pd.DataFrame(yr_results)[["2021", "2022", "2023", "2024"]].transpose()
yr_results["rmse"].plot(marker="x")
yr_results["mae"].plot(marker="x")
plt.legend()
plt.show()
yr_results["r2"].plot(marker="x")
yr_results["gamma_deviance"].plot(marker="x")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Interpertation

# COMMAND ----------

final_model = mlflow.sklearn.load_model('runs:/d551491340ab4a1aa64ad182174b21ba/model')

# COMMAND ----------

# feature importance

# get feature importance:
feature_importances = final_model.named_steps["regressor"].feature_importances_
feature_names = final_model[:-1].get_feature_names_out()

# Create importance dict
importances = dict(map(lambda i, j: (i, j), feature_names, feature_importances))

# get cat and num names lists
names_series = pd.Series(final_model[:-1].get_feature_names_out())
num_feats = names_series[
    (names_series.str.startswith("mean")) | (names_series.str.startswith("mode"))
]

# populate cat features names manually as too much effort to pull out of pipeline names
cat_feats = [
    "DamageAssessment_constant__DamageAssessment",
    "TP_ManufacturerDescription_constant__TP_ManufacturerDescription",
    "InsurerName1_constant__InsurerName1",
    "PostcodeArea_constant__PostcodeArea",
    "IncidentCauseDescription_constant__IncidentCauseDescription",
    "IncidentSubCauseDescription_constant__IncidentSubCauseDescription",
    "cat_mode__IncidentUKCountry",
    "cat_mode__TP_BodyKey",
    "cat_mode__TP_FuelKey",
    "cat_mode__FP_BodyKey",
    "cat_mode__ColourDescription",
    "cat_mode__NotificationMethod",
]

cleaned_importances = importance_cleaner(
    importances, cat_feats, num_feats
)  # clean and join importances
cleaned_importances.index = cleaned_importances.index.str.extract(r"__(.*)")[
    0
]  # remove pipeline stage names from the feature list
cleaned_importances.iloc[:50].sort_values().plot(
    kind="barh",
    figsize=(8, 10),
    title="Top 50 Feature Importance Share",
    color="deepskyblue",
    ylabel="Feature",
    grid=True,
)
plt.show()

# COMMAND ----------

# need to create boolean array indicating whether modelling features are categorical for use in PDP function
cat_input_feats = pd.Series(cat_feats).str.extract(r"__(.*)")[0].values
input_feats = final_model.feature_names_in_
categorical_indicator_array = [
    True if x in cat_input_feats else False for x in input_feats
]

# COMMAND ----------

# might want to play around with number of factors inspected so define here to easily change if need be
n = 30

# the pdp function can't handle missing values - this is an issue as the predictor pipeline handles missing values
# so need to manually do imputation for cols with some missing values
cols_w_missing = train_X.columns[train_X.isna().sum() > 0]

# check to see which of top 30 factors are missing that need to be manually imputed for the PDP plots
missings = [x for x in cleaned_importances.iloc[:n].index if x in cols_w_missing]
missings

# COMMAND ----------

impute_dict = {
    "prediction": train_X.prediction.mean(),
    "TP_ManufacturerDescription": "Other",
    "PostcodeArea": "ZZ",
    "TP_BodyKey": train_X.TP_BodyKey.mode().values[0],
    "DamageAssessment": "Unknown",
    "ColourDescription": train_X.ColourDescription.mode().values[0],
    "TP_FuelKey": train_X.TP_FuelKey.mode().values[0],
}

print(impute_dict)

train_X_imptued = train_X.copy()
for k, v in impute_dict.items():
    train_X_imptued[k] = train_X_imptued[k].fillna(v)

# check no missings now:
train_X_imptued[impute_dict.keys()].isna().sum()

# COMMAND ----------

fig, ax = plt.subplots()
fig.set_figheight(75)
fig.set_figwidth(12)
# PDP - numerics
PartialDependenceDisplay.from_estimator(
    estimator=final_model,
    X=train_X_imptued,
    features=[
        x for x in cleaned_importances.iloc[:n].index if x not in cat_input_feats
    ],
    categorical_features=categorical_indicator_array,
    percentiles=(0, 1),
    n_jobs=5,
    subsample=1000,
    random_state=100,
    n_cols=2,
    ax=ax,
)

# COMMAND ----------

# pdp for insurer and pc area and manufacturter
for col in [x for x in cleaned_importances.iloc[:n].index if x in cat_input_feats]:
    pd_results = partial_dependence(
        estimator=final_model,
        X=train_X_imptued.sample(1000),
        features=[col],
        kind="average",
        categorical_features=categorical_indicator_array,
    )
    pd.Series(
        index=pd_results["grid_values"][0], data=pd_results["average"][0]
    ).sort_values().plot(kind="barh", figsize=(9, 9), grid=True, title=col)
    plt.show()

# COMMAND ----------

#just check postcode area with larger plot
pd_results = partial_dependence(estimator=final_model, X=train_X_imptued.sample(1000), features=['PostcodeArea'], kind="average", categorical_features=categorical_indicator_array)
pd.Series(index=pd_results['grid_values'][0], data=pd_results['average'][0]).sort_values().plot(kind='barh', figsize=(8,18), grid=True, title='PostcodeArea')
plt.show()

# COMMAND ----------

# AvsE
def feature_impact(v, df):
    df_cut = df.copy()
    if len(df_cut[v].unique()) > 50 and df_cut[v].dtype != object:
        df_cut[v] = pd.cut(df_cut[v], 20)
    gbp1 = (
        df_cut.groupby(v)
        .agg({"preds": "mean", "actuals": "mean", "FP_Seats": "count"})
        .reset_index()
    )
    # Create a figure and axis objects
    fig, ax1 = plt.subplots()
    # Plotting bar chart on the first axis
    gbp1.plot(x=v, y="FP_Seats", kind="bar", ax=ax1, color="purple", legend=False)
    # Create a second axis sharing the same x-axis
    ax2 = ax1.twinx()
    # Plotting line chart on the second axis
    gbp1.plot(y="preds", kind="line", marker="o", ax=ax2, color="red", legend=True)
    gbp1.plot(
        y="actuals", kind="line", marker="o", ax=ax2, color="deepskyblue", legend=True
    )

    # adding labels to chart
    ax1.set_xlabel(v)
    ax1.set_ylabel("Count", color="purple")
    ax2.set_ylabel("Average prediction", color="red")
    # plt.title(v + ' impact on DaysInRepair Prediction')
    plt.show()


for col in cleaned_importances.iloc[:30].index:
    feature_impact(col, df_test)