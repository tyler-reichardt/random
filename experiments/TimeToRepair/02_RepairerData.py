# Databricks notebook source
pip install openpyxl

# COMMAND ----------

import databricks.koalas as ks

# COMMAND ----------

from pyspark.sql.functions import col,to_date,upper,regexp_replace,when,datediff,lit

# COMMAND ----------

# DBTITLE 1,Read WNS data
dfWNS = ks.read_excel('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/FC Repair Data 05062023.xlsx', engine='openpyxl')
dfWNS = dfWNS.to_spark()
#dfWNS.count()

# COMMAND ----------

# one column name is duplicated so we'll change this
df_cols = dfWNS.columns
df_cols[-2] = "Repair Status2"
dfWNS = dfWNS.toDF(*df_cols)

# COMMAND ----------

dfWNS2 = dfWNS.withColumn("Insurer Reference",upper(col("Insurer Reference")))\
  .withColumn("Vehicle Registration",upper(regexp_replace(col("Vehicle Registration")," ","")))\
  .withColumn("Declared Total Loss Date",to_date(col("Declared Total Loss Date"),"dd/MM/yyyy"))\
  .withColumn("Repair Booking Date",to_date(col("Repair Booking Date"),"dd/MM/yyyy"))\
  .withColumn("Completed Date",to_date(col("Completed Date"),"dd/MM/yyyy"))\
  .withColumn("Vehicle Returned to Client",to_date(col("Vehicle Returned to Client"),"dd/MM/yyyy"))\

dfWNS2 = dfWNS2.withColumnRenamed("Insurer Reference","ClaimRef")\
  .withColumnRenamed("Vehicle Registration","VehReg")\
  .withColumn("IncidentDate",to_date(col("Incident Date")))\
  .withColumn("RepairOngoingFlag",when(col("Repair Status2")=="Closed",0).otherwise(1))\
  .withColumn("NotNetworkRepairedFlag",when((col("Repair Status2")=="Closed") & (col("Repair Status")=="Property returned to Client"),0).otherwise(1))\
  .withColumn("TotalLossFlag",when(col("Declared Total Loss Date").isNull(),0).otherwise(1))\
  .withColumnRenamed("Repair Booking Date","StartDate")\
  .withColumn("EndDate",when(col("Vehicle Returned to Client").isNotNull(),col("Vehicle Returned to Client")).otherwise(col("Completed Date")))\
  .withColumn("MissingDataFlag",when((col("RepairOngoingFlag") == 0) & (col("TotalLossFlag") == 0) & (col("NotNetworkRepairedFlag") == 0) & (col("EndDate").isNull()),1).otherwise(0))\
  .withColumn("DaysInRepair",datediff(col("EndDate"),col("StartDate")))\
  .withColumn("System",lit("WNS"))\
  .select(["ClaimRef","VehReg","IncidentDate","RepairOngoingFlag","NotNetworkRepairedFlag","TotalLossFlag","MissingDataFlag","StartDate","EndDate","DaysInRepair","System"])
  
#display(dfWNS2)

# COMMAND ----------

display(dfWNS2.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC below shows there are multiple repair jobs for the same claim and vehicle. Therefore we should join to the claim to reduce jobs to the right one

# COMMAND ----------

#do you get multiple repairs for the same vehicle and claim
from pyspark.sql.functions import countDistinct
print(dfWNS2.agg(countDistinct(dfWNS2.ClaimRef, dfWNS2.VehReg).alias('c')).collect())
print(dfWNS2.count())

print(dfWNS2.filter((col("MissingDataFlag") == 0) & (col("NotNetworkRepairedFlag") == 0)).agg(countDistinct(dfWNS2.ClaimRef, dfWNS2.VehReg).alias('c')).collect())
print(dfWNS2.filter((col("MissingDataFlag") == 0) & (col("NotNetworkRepairedFlag") == 0)).count())

# COMMAND ----------

# DBTITLE 1,Read WIP data
dfWIP = ks.read_excel('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/AM WIP Report FCI Interest Codes 050623.xlsx', engine='openpyxl')
dfWIP = dfWIP.to_spark()
#dfWIP.count()

# COMMAND ----------

# Convert french incident date to english

from pyspark.sql import functions as F
from pyspark.sql import Window

columns = ["French","English"]
data = [("janv.", "January"), ("févr.", "February"), ("mars", "March"),("avr.","April"),("mai","May"),("juin","June"),("juil.","July"),("août","August"),("sept.","September"),("oct.","October"),("nov.","November"),("déc.","December")]
dfMonthConv = spark.createDataFrame(data).toDF(*columns)


replacement_map = {}
for row in dfMonthConv.collect():
    replacement_map[row.French]=row.English

@F.udf()
def find_and_replace(column_value):
    for colfind in replacement_map:
      try:
        column_value = column_value.replace(colfind,replacement_map[colfind])
      except:
        pass
    return column_value
  
dfWIP = dfWIP.withColumn("Incident_date",to_date(find_and_replace(F.col("Incident_date")),"d MMMM yyyy"))

# COMMAND ----------

#some feature engineering, column renaming and column selection

dfWIP2 = dfWIP.withColumn("claim_reference",upper(col("claim_reference")))\
  .withColumn("Reg_number",upper(regexp_replace(col("Reg_number")," ","")))\
  .withColumn("Date_in_Date",to_date(col("Date_in_Date")))\
  .withColumn("Left_site",to_date(col("Left_site")))

dfWIP2 = dfWIP2.withColumnRenamed("claim_reference","ClaimRef")\
  .withColumnRenamed("Reg_number","VehReg")\
  .withColumnRenamed("Incident_date","IncidentDate")\
  .withColumn("RepairOngoingFlag",when((col("Job_status")=="CLOSED") | (col("Job_status")=="VEHICLE DELIVERED"),0).otherwise(1))\
  .withColumn("NotNetworkRepairedFlag",when(col("Cancelled_Date").isNotNull(),1).otherwise(0))\
  .withColumn("TotalLossFlag",when(col("Total_Loss").isNull(),0).otherwise(1))\
  .withColumnRenamed("Date_in_Date","StartDate")\
  .withColumnRenamed("Left_site","EndDate")\
  .withColumn("MissingDataFlag",when((col("RepairOngoingFlag")==0) & (col("Cancelled_date").isNull()) & (col("EndDate").isNull()) & (col("Total_Loss").isNull()),1).otherwise(0))\
  .withColumn("DaysInRepair",datediff(col("EndDate"),col("StartDate")))\
  .withColumn("System",lit("WIP"))\
  .select(["ClaimRef","VehReg","IncidentDate","RepairOngoingFlag","NotNetworkRepairedFlag","TotalLossFlag","MissingDataFlag","StartDate","EndDate","DaysInRepair","System"])


# COMMAND ----------

# DBTITLE 1,Combine WNS and WIP data
dfR = dfWNS2.union(dfWIP2)

# COMMAND ----------

#save off / read back in

#dfR.coalesce(1).write.mode("overwrite").csv('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/Exports/combinedRepairerData',header=True)
dfR = spark.read.csv('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/Exports/combinedRepairerData',header = True)

# COMMAND ----------

# DBTITLE 1,Add the claim vehicle data 
#read claim veh data
dfInitialClaimVehicle = spark.read.parquet('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/Exports/InitialClaimVehicleData')

# join the vehicle with repairer data
dfJ = dfR.withColumnRenamed("IncidentDate","RepairerIncidentDate").join(dfInitialClaimVehicle,how="inner",on=[dfR.VehReg == dfInitialClaimVehicle.Registration, dfR.ClaimRef == dfInitialClaimVehicle.ClaimNumber])

# COMMAND ----------

# review the column data types
dfJ = dfJ\
  .withColumn('RepairerIncidentDate', dfJ.RepairerIncidentDate.cast('timestamp'))\
  .withColumn('RepairOngoingFlag', dfJ.RepairOngoingFlag.cast('int'))\
  .withColumn('NotNetworkRepairedFlag', dfJ.NotNetworkRepairedFlag.cast('int'))\
  .withColumn('TotalLossFlag', dfJ.TotalLossFlag.cast('int'))\
  .withColumn('MissingDataFlag', dfJ.MissingDataFlag.cast('int'))\
  .withColumn('StartDate', dfJ.StartDate.cast('timestamp'))\
  .withColumn('EndDate', dfJ.EndDate.cast('timestamp'))\
  .withColumn('DaysInRepair',dfJ.DaysInRepair.cast('int'))\
  .withColumn('ImpactSpeed',dfJ.ImpactSpeed.cast('int'))

# COMMAND ----------

# exclude repairs where the dates do not make sense.
dfJ = dfJ.filter(col('StartDate') >= col('RepairerIncidentDate')).\
  filter(col('StartDate')<=col('EndDate'))

# only keep WNS repairs due to low quality data in WIP - this will need to be considered again later for current repair timings
dfJ = dfJ.filter(col('System')=='WNS')

# remove all with NotNetworkRepairedFlag, TotalLossFlag & MissingDataFlag
dfJ = dfJ.filter((col("NotNetworkRepairedFlag") == 0 ) & (col("TotalLossFlag") == 0 ) & (col("MissingDataFlag") == 0 ))

# COMMAND ----------

# cap days in repair to 100 days
dfJ1=dfJ.withColumn('DaysInRepairCapped',when(col('DaysInRepair')>100,100).otherwise(col('DaysInRepair')))

# COMMAND ----------

# count the number of damaged areas on the vehicle.
from pyspark.sql.functions import udf, array

def join_columns(row_list):
    return len([cell_val for cell_val in row_list if cell_val is not None])

join_udf = udf(join_columns)

dfJ1 = dfJ1.withColumn('DamageRecorded', join_udf(array(dfJ.Front,dfJ.FrontBonnet,dfJ.FrontLeft,dfJ.FrontRight,dfJ.Left,dfJ.LeftBackseat,dfJ.LeftFrontWheel,dfJ.LeftMirror,dfJ.LeftRearWheel,dfJ.LeftUnderside,dfJ.Rear,dfJ.RearLeft,dfJ.RearRight,dfJ.RearWindowDamage,dfJ.Right,dfJ.RightBackseat,dfJ.RightFrontWheel,dfJ.RightMirror,dfJ.RightRearWheel,dfJ.RightRoof,dfJ.RightUnderside,dfJ.RoofDamage,dfJ.UnderbodyDamage,dfJ.WindscreenDamage)).cast('int'))

# as long as there were some damaged areas recorded, we assume an assessment was done
dfJ1 = dfJ1.withColumn('DamageAssessed', when(col('DamageRecorded') > 0,1).otherwise(0))
display(dfJ1)

# COMMAND ----------

# convert field with a categoric damage scale to numeric

def damageScale(row_area,row_damageflag):
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

damageScale_udf = udf(damageScale)

dfJ2 = dfJ1.withColumn('Front', damageScale_udf(dfJ1.Front,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('FrontBonnet', damageScale_udf(dfJ1.FrontBonnet,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('FrontLeft', damageScale_udf(dfJ1.FrontLeft,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('FrontRight', damageScale_udf(dfJ1.FrontRight,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('Left', damageScale_udf(dfJ1.Left,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('LeftBackseat', damageScale_udf(dfJ1.LeftBackseat,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('LeftFrontWheel', damageScale_udf(dfJ1.LeftFrontWheel,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('LeftMirror', damageScale_udf(dfJ1.LeftMirror,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('LeftRearWheel', damageScale_udf(dfJ1.LeftRearWheel,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('LeftUnderside', damageScale_udf(dfJ1.LeftUnderside,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('Rear', damageScale_udf(dfJ1.Rear,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RearLeft', damageScale_udf(dfJ1.RearLeft,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RearRight', damageScale_udf(dfJ1.RearRight,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RearWindowDamage', damageScale_udf(dfJ1.RearWindowDamage,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('Right', damageScale_udf(dfJ1.Right,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RightBackseat', damageScale_udf(dfJ1.RightBackseat,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RightFrontWheel', damageScale_udf(dfJ1.RightFrontWheel,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RightMirror', damageScale_udf(dfJ1.RightMirror,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RightRearWheel', damageScale_udf(dfJ1.RightRearWheel,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RightRoof', damageScale_udf(dfJ1.RightRoof,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RightUnderside', damageScale_udf(dfJ1.RightUnderside,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('RoofDamage', damageScale_udf(dfJ1.RoofDamage,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('UnderbodyDamage', damageScale_udf(dfJ1.UnderbodyDamage,dfJ1.DamageAssessed).cast('int'))\
  .withColumn('WindscreenDamage', damageScale_udf(dfJ1.WindscreenDamage,dfJ1.DamageAssessed).cast('int'))


# COMMAND ----------

dfJ2 = dfJ2.withColumn('ImpactSpeedRange',when(dfJ2.ImpactSpeedRange == 'Stationary',0)\
  .when(dfJ2.ImpactSpeedRange == 'ZeroToSix',1)\
  .when(dfJ2.ImpactSpeedRange == 'SevenToFourteen',7)\
  .when(dfJ2.ImpactSpeedRange == 'FifteenToTwenty',15)\
  .when(dfJ2.ImpactSpeedRange == 'TwentyOneToThirty',21)\
  .when(dfJ2.ImpactSpeedRange == 'ThirtyOneToFourty',31)\
  .when(dfJ2.ImpactSpeedRange == 'FourtyOneToFifty',41)\
  .when(dfJ2.ImpactSpeedRange == 'FiftyOneToSixty',51)\
  .when(dfJ2.ImpactSpeedRange == 'SixtyOneToSeventy',61)\
  .when(dfJ2.ImpactSpeedRange == 'OverSeventy',71)\
  .otherwise(-1).cast('int'))

# COMMAND ----------

dfJ2 = dfJ2.withColumn('DeployedAirbags',when(dfJ2.DeployedAirbags == "None",0)\
  .when(dfJ2.DeployedAirbags == "One",1)\
  .when(dfJ2.DeployedAirbags == "Two",2)\
  .when(dfJ2.DeployedAirbags == "Three",3)\
  .when(dfJ2.DeployedAirbags == "Four",3)\
  .when(dfJ2.DeployedAirbags == "All",3)\
  .otherwise(-1))

# COMMAND ----------

dfJ.write.mode("overwrite").parquet('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/Exports/ClaimVehicleRepairerData')

# COMMAND ----------

dfJ = spark.read.parquet('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/Exports/ClaimVehicleRepairerData')
dfJ = dfJ.withColumn('StartDate', dfJ.StartDate.cast('timestamp'))\
  .withColumn('EndDate', dfJ.EndDate.cast('timestamp'))\
  .withColumn('IncidentDate', dfJ.IncidentDate.cast('timestamp'))\
  .withColumn('DaysInRepair',dfJ.DaysInRepair.cast('int'))

# COMMAND ----------

dfJ.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Plots
# MAGIC Not needed for deployment but left in for context

# COMMAND ----------

pdf = dfJ.toPandas()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
pdf['StartMonthYear'] = pdf['StartDateTime'].dt.to_period('M')
pdf['EndMonthYear'] = pdf['EndDateTime'].dt.to_period('M')
pdf['IncidentMonthYear'] = pdf['IncidentDateTime'].dt.to_period('M')

# COMMAND ----------

# plot the incident dates
pdf.groupby('IncidentMonthYear').VehReg.count().plot()

# COMMAND ----------

# filter the data to just 2021 onwards
pdf2021 = pdf[pdf['IncidentDateTime'].dt.year >= 2021]
pdf2021.groupby('IncidentMonthYear').VehReg.count().plot()
plt.show()

# COMMAND ----------

# split WNS and WIP
pdf2021.groupby(["IncidentMonthYear","System"]).VehReg.count().unstack().plot()
plt.show()

# COMMAND ----------

# complete repairs only
pdf2021[(pdf2021['RepairOngoingFlag'] == 0) & (pdf2021['NotNetworkRepairedFlag'] == 0) & (pdf2021['TotalLossFlag'] == 0) & (pdf2021['MissingDataFlag'] == 0)].groupby(["IncidentMonthYear","System"]).VehReg.count().unstack().plot()
plt.show()

# COMMAND ----------

# repair count (removing cancelled and total loss) (include multiple repair jobs for a single vehicle)
pdf2021[(pdf2021['NotNetworkRepairedFlag'] == 0) & (pdf2021['TotalLossFlag'] == 0)].groupby(["IncidentMonthYear","System"]).VehReg.count().unstack().plot()
plt.show()

# COMMAND ----------

# count repair ongoing vs closed
pdf2021[(pdf2021['NotNetworkRepairedFlag'] == 0) & (pdf2021['TotalLossFlag'] == 0)].groupby(["IncidentMonthYear","RepairOngoingFlag"])["VehReg"].count().unstack().plot()
plt.show()

# COMMAND ----------

# count repair finifhed but missing start and/or end dates
pdf2021[(pdf2021['RepairOngoingFlag'] == 0) & (pdf2021['NotNetworkRepairedFlag'] == 0) & (pdf2021['TotalLossFlag'] == 0)].groupby(["IncidentMonthYear","MissingDataFlag"])["VehReg"].count().unstack().plot()
plt.show()

# COMMAND ----------

# plot the incident dates
dfl1c = pd.crosstab(pdf[pdf['IncidentDateTime'].dt.year >= 2022].IncidentMonthYear, pdf[pdf['IncidentDateTime'].dt.year >= 2022].RepairOngoingFlag, normalize='index').mul(100).round(1)
dfl1c.plot(kind='area', ylabel='Percentage of tasks (%)', stacked=True, rot=0,color=["#743A84","#AFAFAF"])
plt.title("Percentage of repair jobs where repair is complete")
plt.show()

# COMMAND ----------

# percentage complete
dfl1c = pd.crosstab(pdf[pdf['StartDateTime'].dt.year >= 2022].StartMonthYear, pdf[pdf['StartDateTime'].dt.year >= 2022].RepairOngoingFlag, normalize='index').mul(100).round(1)
dfl1c.plot(kind='area', ylabel='Percentage of tasks (%)', stacked=True, rot=0,color=["#743A84","#AFAFAF"])
plt.title("Percentage of repair jobs where repair is complete")
plt.show()

# COMMAND ----------

# average days to repair by month
pdf[(pdf['DaysInRepair'].notnull()) & (pdf['StartDateTime'].dt.year >= 2020)].groupby('StartMonthYear').DaysInRepair.mean().plot()

# COMMAND ----------

pdf[(pdf['DaysInRepair'].notnull()) & (pdf['StartDateTime'].dt.year >= 2020) & (pdf['System'] == 'WNS')].groupby('StartMonthYear').DaysInRepair.mean().plot()

# COMMAND ----------

pdf[(pdf['DaysInRepair'].notnull()) & (pdf['StartDateTime'].dt.year >= 2020) & (pdf['System'] == 'WIP')].groupby('StartMonthYear').DaysInRepair.mean().plot()

# COMMAND ----------

pdf[(pdf['DaysInRepair'].notnull()) & (pdf['StartDateTime'].dt.year >= 2020) & (pdf['System'] == 'WIP')].groupby('StartMonthYear').DaysInRepair.median().plot()

# COMMAND ----------

pdf = pdf[pdf['StartDateTime'].dt.year >= 2020]

# COMMAND ----------

dfl1c = pd.crosstab(pdf.StartMonthYear, pdf.MissingDataFlag, normalize='index').mul(100).round(1)
dfl1c.plot(kind='area', ylabel='Percentage of repairs (%)', stacked=True, rot=0,color=["#743A84","#AFAFAF"])
plt.title("Percentage of repairs with identified missing data")
plt.show()

# COMMAND ----------

dfl2 = pdf.groupby(["StartMonthYear","MissingDataFlag"])["VehReg"].count()
dfl2.unstack().plot()
plt.show()

# COMMAND ----------

pdf[pdf['RepairOngoingFlag']==False].groupby('MissingDataFlag', dropna=False)['VehReg'].count().plot(kind='bar')
plt.title("Repair complete with missing data")
plt.show()