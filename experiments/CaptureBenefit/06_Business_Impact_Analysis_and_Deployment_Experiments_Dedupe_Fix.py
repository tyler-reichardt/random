# Databricks notebook source
# MAGIC %md
# MAGIC ###### Author: Harry Bjarnason
# MAGIC ###### Notebook Purpose: Create PDPs for the benefit predictions, run some initial impact simulations and exploratory analysis, and perform the quantile analysis to see how predictions hold up against our crude view of 'actuals'

# COMMAND ----------

pip show scikit-learn

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import sys
sys.path.append("/Workspace/Users/harry.bjarnason@first-central.com/Projects/TPC Third Party Capture/2024_07_TPC_Capture_Benefit/CodingToolkit/")
from interpret_ai.importance_cleaner import importance_cleaner
from model_evaluate.regression_metrics import regression_metrics


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Prep

# COMMAND ----------

#read data
df = pd.read_csv("/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2_cleaned/capture_benefit_df_20240806_dupe_clm_no_fix.csv")
df['NotificationDate'] = pd.to_datetime(df['NotificationDate'])

#keep attempts only as these are the cases relevant to deployment 
df = df[df.CaptureCallAttempted==1]

print(df.shape)


#keep date range 1/1/21 - 1/4/24 only
df = df[(df.NotificationDate > '2021-01-01') & (df.NotificationDate < '2024-04-01')]
print(df.shape)


# COMMAND ----------

#load model
model = mlflow.pyfunc.load_model('runs:/d551491340ab4a1aa64ad182174b21ba/model')

df_cap = df.copy()
df_cap['CaptureSuccess_AdrienneVersion'] = 1

df_nocap = df.copy()
df_nocap['CaptureSuccess_AdrienneVersion'] = 0


df['TPPD_Pred_Capture'] = model.predict(df_cap)
df['TPPD_Pred_No_Capture'] = model.predict(df_nocap)

#predicted value
df['Capture_Benefit'] = df.TPPD_Pred_No_Capture - df.TPPD_Pred_Capture

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quantile Analysis

# COMMAND ----------

#create quantile variable to easily play around with these graphs:
q=50

# COMMAND ----------

df_cap = df[df.CaptureCallSuccess==1]
df_cap['INC_TOT_TPPD_QCUT'] = pd.qcut(df_cap['INC_TOT_TPPD'],q=q)
mean_capped_q_inc = df_cap.groupby('INC_TOT_TPPD_QCUT')['INC_TOT_TPPD'].mean()

df_nocap = df[df.CaptureCallSuccess!=1]
df_nocap['INC_TOT_TPPD_QCUT'] = pd.qcut(df_nocap['INC_TOT_TPPD'],q=q)

mean_nocapped_q_inc = df_nocap.groupby('INC_TOT_TPPD_QCUT')['INC_TOT_TPPD'].mean()
mean_nocapped_q_inc_ben = df_nocap.groupby('INC_TOT_TPPD_QCUT')['INC_TOT_TPPD', 'Capture_Benefit', 'TPPD_Pred_No_Capture'].mean()
mean_nocapped_q_inc_ben['Perc_Benefit_Pred'] =  mean_nocapped_q_inc_ben['INC_TOT_TPPD'] / (mean_nocapped_q_inc_ben['INC_TOT_TPPD'] - mean_nocapped_q_inc_ben['Capture_Benefit'])

# Create a new figure and axes for the plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Plot on the primary y-axis
ax1.plot(mean_nocapped_q_inc.values / mean_capped_q_inc.values, marker='x', color='g', label='Average No Cap vs Cap INC_TOT_TPPD')
ax1.plot(mean_nocapped_q_inc_ben.Perc_Benefit_Pred.values, marker='x', color='r', label='Average No Cap vs No Cap - Predicted Benefit INC_TOT_TPPD')
# Plot on the secondary y-axis
ax2.bar(x=np.arange(q), height=mean_nocapped_q_inc, alpha=0.6)

ax1.set_xlabel('Quantile')
ax1.set_ylabel('Cap Cost v No Cap Ratio', color='g')
ax2.set_ylabel('Average No Cap Cost', color='b')
plt.grid(True)
ax1.set_ylim(1,2)
plt.title('%age Benefit in INC_TOT_TPPD Quantiles')
ax1.legend()
plt.show()

# COMMAND ----------

#create a lookup for the actual saving ratio by quantile: 
q_ratio_lookup = pd.DataFrame(mean_nocapped_q_inc.values / mean_capped_q_inc.values, columns=['Ratio_Saving'])

#create column which will be used as the join key
q_ratio_lookup['INC_TOT_TPPD_QCUT_INT'] = q_ratio_lookup.index
q_ratio_lookup.head()


# COMMAND ----------

# MAGIC %md
# MAGIC We now want to see how the actual benefit amount in each quanitle compares with the predicted benefit amount. We will look at non capture cost only for simplicity. Requires some manipulation to get actual benefit amount in terms of ratio and non capture cost:
# MAGIC >     Benefit = non capture cost - capture cost
# MAGIC >
# MAGIC >     Ratio = non capture cost / capture cost
# MAGIC >     -> Capture cost = Non capture cost / ratio
# MAGIC >
# MAGIC >     Benefit = non capture cost - (Non capture cost / ratio)
# MAGIC               = non capture cost*(1-(1/ratio))
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

#redo the qcut but just give the integer label of the quantile for the lookup
df_nocap['INC_TOT_TPPD_QCUT_INT'] =  pd.qcut(df_nocap['INC_TOT_TPPD'], q=q, labels=np.arange(q))

#now can calculate our best view of an actual by joining on the saving ratio 
#benefit = non_cap_cost*(1-(1/ratio)) as shown above
df_nocap = df_nocap.merge(q_ratio_lookup, on='INC_TOT_TPPD_QCUT_INT', how='inner') 
df_nocap['Actual_Benefit_Ratio_View'] = df_nocap.INC_TOT_TPPD*(1-(1/df_nocap.Ratio_Saving))

#now redo groupby and create plot to look at predicted benefit amount and actual benefit amount by quantile 
mean_nocapped_q_inc_ben = df_nocap.groupby('INC_TOT_TPPD_QCUT')['INC_TOT_TPPD', 'Capture_Benefit', 'Actual_Benefit_Ratio_View'].mean()

# Create a new figure and axis for the bar plot
fig, ax1 = plt.subplots()
# Create secondary y-axis
ax2 = ax1.twinx()

# Plot on the primary y-axis
ax1.plot(mean_nocapped_q_inc_ben.Capture_Benefit.values, marker='x', color='g', label='Predicted Benefit')
ax1.plot(mean_nocapped_q_inc_ben.Actual_Benefit_Ratio_View.values, marker='x', color='r', label='Actual_Benefit_Ratio_View')
# Plot on the secondary y-axis
ax2.bar(x=np.arange(q), height=mean_nocapped_q_inc_ben.INC_TOT_TPPD, alpha=0.6)

ax1.set_xlabel('Quantile')
plt.grid(True)

ax1.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Quantile Adjustments
# MAGIC We want to adjust our benefit predictions based on predicted TPPD cost quantile using the ratio of actuals / predictions in the above chart. Here we calculate those ratios and save off. The adjustments and analysis will be performed in the next notebook

# COMMAND ----------

#we want to pull out the ratio of actual benefit / predicted benefit for each quantile to make the adjustments to our predicted benefit values 
quant_benefit_adjustments = pd.DataFrame(mean_nocapped_q_inc_ben.Actual_Benefit_Ratio_View / mean_nocapped_q_inc_ben.Capture_Benefit,columns=['quant_benefit_multiplier']).reset_index()
quant_benefit_adjustments['INC_TOT_TPPD_QCUT_INT'] = quant_benefit_adjustments.index

print(quant_benefit_adjustments.head())
print(quant_benefit_adjustments.tail())

# COMMAND ----------

#save off benefit adjustments lookup
quant_benefit_adjustments.to_csv("/Workspace/Users/harry.bjarnason@first-central.com/Projects/TPC Third Party Capture/2024_07_TPC_Capture_Benefit/artefacts/quant_benefit_adjustments_dedupe_fix.csv")

# COMMAND ----------

quant_benefit_adjustments['quant_benefit_multiplier'].plot(grid=True)