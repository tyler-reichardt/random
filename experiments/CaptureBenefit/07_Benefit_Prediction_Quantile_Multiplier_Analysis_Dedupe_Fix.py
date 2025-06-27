# Databricks notebook source
# MAGIC %md
# MAGIC ###### Author: Harry Bjarnason
# MAGIC ###### Notebook Purpose: Make quantile adjustments to benefit predictions and do an analysis on the result. Save off any lookups required to make this adjustment in deployment, and redo benefit analysis using these adjusted benefit predictions

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import sys

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Read

# COMMAND ----------

#read data
df = pd.read_csv("/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2_cleaned/capture_benefit_df_20240806_dupe_clm_no_fix.csv")
df['NotificationDate'] = pd.to_datetime(df['NotificationDate'])

#keep attempts only
df = df[df.CaptureCallAttempted==1]

print(df.shape)

#keep date range 1/1/21 - 1/4/24 only
df = df[(df.NotificationDate > '2021-01-01') & (df.NotificationDate < '2024-04-01')]
print(df.shape)

#read in benefit multiplier (created in 06_Business_Impact_Analysis_and_Deployment_Experiments)
quantile_benefit_multiplier = pd.read_csv("/Workspace/Users/harry.bjarnason@first-central.com/Projects/TPC Third Party Capture/2024_07_TPC_Capture_Benefit/artefacts/quant_benefit_adjustments_dedupe_fix.csv")



# COMMAND ----------

quantile_benefit_multiplier.head()

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
# MAGIC ### Data Processing

# COMMAND ----------

#get quantiles for TPPD_Pred_No_Cap and also save off - these will be used alongside quantile_benefit_multiplier to make adjustments to predicted benefit value
quantiles, bin_edges = pd.qcut(df['TPPD_Pred_No_Capture'], q=50, retbins=True)

#save off the bin edges as will be used in deployment
pd.Series(bin_edges).to_csv("/Workspace/Users/harry.bjarnason@first-central.com/Projects/TPC Third Party Capture/2024_07_TPC_Capture_Benefit/artefacts/TPPD_Pred_No_Cap_Quantile_Bin_Edges_Dedupe_Fix.csv")

bin_edges


# COMMAND ----------

#now use bin edges to get quantile for each case - we could have just used the qcut directly for this but wanted to illustrate how this would work in deployment
df['TPPD_Pred_No_Capture_Quantile'] = pd.cut(df['TPPD_Pred_No_Capture'], bins=bin_edges, labels=False, include_lowest=True)

#now join on quantile multiplier to the bin edge quantile
df = df.merge(quantile_benefit_multiplier, left_on='TPPD_Pred_No_Capture_Quantile', right_on='INC_TOT_TPPD_QCUT_INT', how='inner')
print(df.shape)

# COMMAND ----------

#now multiply benefit pred by multiplier
df['Capture_Benefit_Adjusted'] = df.Capture_Benefit * df.quant_benefit_multiplier
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quantile Analysis Charts

# COMMAND ----------

#now look at doing qcut for cap and no cap and look at ratio for an estimate of %age benefit saving by INC_TOT_TPPD
#note need to split df into cap and no cap to get separate quantiles for INC_TOT_TPPD

df_cap = df[df.CaptureCallSuccess==1]
df_cap['INC_TOT_TPPD_QCUT'] = pd.qcut(df_cap['INC_TOT_TPPD'],q=50)
mean_capped_q_inc = df_cap.groupby('INC_TOT_TPPD_QCUT')['INC_TOT_TPPD'].mean()

df_nocap = df[df.CaptureCallSuccess!=1]
df_nocap['INC_TOT_TPPD_QCUT'] = pd.qcut(df_nocap['INC_TOT_TPPD'],q=50)
mean_nocapped_q_inc = df_nocap.groupby('INC_TOT_TPPD_QCUT')['INC_TOT_TPPD'].mean()

# Create a new figure and axis for the bar plot
fig, ax1 = plt.subplots()
# Create secondary y-axis
ax2 = ax1.twinx()

# Plot on the primary y-axis
ax1.plot(mean_nocapped_q_inc.values / mean_capped_q_inc.values, marker='x', color='g')
# Plot on the secondary y-axis
ax2.bar(x=np.arange(50), height=mean_nocapped_q_inc, alpha=0.6)

ax1.set_xlabel('Quantile')
ax1.set_ylabel('Cap Cost v No Cap Ratio', color='g')
ax2.set_ylabel('Average No Cap Cost', color='b')
plt.title('Average Cap vs Non Cap Ratio in INC_TOT_TPPD quantiles')
plt.grid(True)
plt.show()


# COMMAND ----------

#create a lookup for the actual saving ratio by quantile that we are going to use to calculate a best view of 'actual' benefit amount 
q_ratio_lookup = pd.DataFrame(mean_nocapped_q_inc.values / mean_capped_q_inc.values, columns=['Ratio_Saving'])

#create column which will be used as the join key
q_ratio_lookup['INC_TOT_TPPD_QCUT_INT'] = q_ratio_lookup.index
q_ratio_lookup.head()

#redo the qcut but just give the integer label of the quantile for the lookup
df_nocap['INC_TOT_TPPD_QCUT_INT'] =  pd.qcut(df_nocap['INC_TOT_TPPD'], q=50, labels=np.arange(50))

#now can calculate our best view of an actual by joining on the saving ratio 
#benefit = non_cap_cost*(1-(1/ratio))
df_nocap = df_nocap.merge(q_ratio_lookup, on='INC_TOT_TPPD_QCUT_INT', how='inner') 
df_nocap['Actual_Benefit_Ratio_View'] = df_nocap.INC_TOT_TPPD*(1-(1/df_nocap.Ratio_Saving))

#now redo groupby and create plot to look at predicted benefit amount and actual benefit amount by quantile 
mean_nocapped_q_inc_ben = df_nocap.groupby('INC_TOT_TPPD_QCUT')['INC_TOT_TPPD', 'Capture_Benefit_Adjusted', 'Actual_Benefit_Ratio_View'].mean()

# Create a new figure and axis for the bar plot
fig, ax1 = plt.subplots()
# Create secondary y-axis
ax2 = ax1.twinx()

# Plot on the primary y-axis
ax1.plot(mean_nocapped_q_inc_ben.Capture_Benefit_Adjusted.values, marker='x', color='purple', label='Predicted Benefit')
ax1.plot(mean_nocapped_q_inc_ben.Actual_Benefit_Ratio_View.values, marker='x', color='deepskyblue', label='Actual Benefit (Quantile View)')
# Plot on the secondary y-axis
ax2.bar(x=np.arange(50), height=mean_nocapped_q_inc_ben.INC_TOT_TPPD, alpha=0.8, color='grey')

ax1.set_xlabel('Quantile')
plt.grid(True)

ax1.legend()
plt.show()

# COMMAND ----------

df.Capture_Benefit_Adjusted.hist(bins=50, color='deepskyblue')
plt.title('Capture Benefit Distribution')
plt.xlim(0,15000)
#plt.axline(xy1=(1440, 0), xy2=(1440, 1), color='purple', linestyle='--', label='mean')
plt.xlabel('Benefit (£)')
plt.ylabel('Count')
#plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Redo Benefit Analysis
# MAGIC Making these adjustments based off the predicted No Cap Cost quantile will shift prioritisations around a bit.
# MAGIC
# MAGIC Redo benefit analysis to ensure hasn't wrecked the prioritisation performance
# MAGIC
# MAGIC Will create the quantiles based on 2023 only 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

#keep only 2023
df = df[df.Notified_Year==2023]
df_nocap = df_nocap[df_nocap.Notified_Year==2023]
df_cap = df[(df.Notified_Year==2023) & (df.CaptureSuccess_AdrienneVersion==1)]

# COMMAND ----------

q1 = df['Capture_Benefit_Adjusted'].quantile(q=1/4)
q2 = df['Capture_Benefit_Adjusted'].quantile(q=2/4)
q3 = df['Capture_Benefit_Adjusted'].quantile(q=3/4)

df['Capture_Benefit_Adjusted'].hist(bins=50, color='deepskyblue')
plt.xlim(0, 10000)
plt.axline(xy1=(q1, 1), xy2=(q1, 0), color='purple', linestyle='--')
plt.axline(xy1=(q3, 1), xy2=(q3, 0), color='purple', linestyle='--')
plt.xlabel('Benefit (£)')
plt.ylabel('Count')
plt.title('Capture Benefit Distribution')
plt.show()

# COMMAND ----------

print(q1,q2,q3)

#save off priority thresholds 
pd.Series([0, q1, q3, 99999]).to_csv("/Workspace/Users/harry.bjarnason@first-central.com/Projects/TPC Third Party Capture/2024_07_TPC_Capture_Benefit/artefacts/Capture_Benefit_Prioritisation_Thresholds_Dedupe_Fix.csv")

# COMMAND ----------

#scenario testing
df['Benefit_Priority'] = 'Med'
df['Benefit_Priority'][df.Capture_Benefit_Adjusted>q3] = 'High'
df['Benefit_Priority'][df.Capture_Benefit_Adjusted<q1] = 'Low'

#also apply to df nocap as this is where we have our 'actual benefit ratio view' - closest we can get to actuals (although still far from it)
df_nocap['Benefit_Priority'] = 'Med'
df_nocap['Benefit_Priority'][df_nocap.Capture_Benefit_Adjusted>q3] = 'High'
df_nocap['Benefit_Priority'][df_nocap.Capture_Benefit_Adjusted<q1] = 'Low'

print(df.groupby('Benefit_Priority').size())
print(df.groupby('Benefit_Priority')['Capture_Benefit_Adjusted'].mean().reindex(['Low', 'Med', 'High']))


# COMMAND ----------

#impact calculation using our adjusted benefit predictions - all groupbys on whole df (which is 2023 only)

savings_dict_23 = df.groupby('Benefit_Priority')['Capture_Benefit_Adjusted'].mean().reindex(['Low', 'Med', 'High']).to_dict()
volume_dict_23 = df.groupby('Benefit_Priority').size().reindex(['Low', 'Med', 'High']).to_dict()

impact_dict = {'Low': -0.2,
               'Med': 0.0325,
               'High': 0.125}


print(volume_dict_23)
print(savings_dict_23)

total_impact = 0
for k in impact_dict.keys():
  impact = impact_dict[k]*volume_dict_23[k]*savings_dict_23[k]
  print(k, impact)
  total_impact += impact


print('TOTAL IMPACT: £',total_impact)