# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import *
from functools import reduce
from operator import add

# COMMAND ----------

# Data period runs from mid-19 to mid-23, includes only WNS data as WIP was incomplete at the time

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Read and Joins

# COMMAND ----------

#read in repairer data
dfJ = spark.read.parquet('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/Exports/ClaimVehicleRepairerData')

#join in the vehicle value
dfV = spark.read.parquet('/mnt/datalake/users/LukeRowland/claimextracts/ClaimItemDetail/*').select('ClaimNumber','Registration','Value')
dfV = dfV.groupby('ClaimNumber','Registration').agg(max("Value").alias("Value"))
dfV = dfV.withColumnRenamed("ClaimNumber", "ClaimNumber5").withColumnRenamed("Registration", "Registration5")
dfJ = dfJ.join(dfV, (dfJ.ClaimNumber == dfV.ClaimNumber5) & (dfJ.Registration == dfV.Registration5), how='left').drop('ClaimNumber5','Registration5')

#write
dfJ.write.mode("overwrite").parquet('/mnt/datalake/users/HarryBjarnason/TPC/TimeToRepair/dataPrep1')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

dfJ = spark.read.parquet('/mnt/datalake/users/HarryBjarnason/TPC/TimeToRepair/dataPrep1')

# format columns
dfJ = dfJ.withColumn('StartDate', dfJ.StartDate.cast('timestamp'))\
  .withColumn('EndDate', dfJ.EndDate.cast('timestamp'))\
  .withColumn('IncidentDate', dfJ.IncidentDate.cast('timestamp'))\
  .withColumn('DaysInRepair',dfJ.DaysInRepair.cast('int'))\
  .withColumn('Value',dfJ.Value.cast('int'))

# filter for only rows with realistic timeframes
dfJ = dfJ.filter((col('DaysInRepair')>0) & (col('DaysInRepair')<=60))

# filter for only car repairs - vast majority of cases
dfJ = dfJ.filter(col('ClaimItemType')=='CarMotorVehicleClaimItem').drop('ClaimItemType')

# encode damage assessment - group other/missing/unknown etc.
dfJ = dfJ.withColumn('DA_DR',col('DamageAssessment')=='DriveableRepair').withColumn('DA_DTL',col('DamageAssessment')=='DriveableTotalLoss').withColumn('DA_UTL',col('DamageAssessment')=='UnroadworthyTotalLoss').withColumn('DA_UR',col('DamageAssessment')=='UnroadworthyRepair').withColumn("DA_O",~(col("DA_DR") | col("DA_DTL") | col("DA_UTL") | col("DA_UR"))).drop(col('DamageAssessment'))

# create column for time to notify
dfJ = dfJ.withColumn("TimeToNotify",datediff(col('NotificationDate'),col('IncidentDate')))
dfJ = dfJ.withColumn("TimeToNotify",when(col('TimeToNotify')<0,0).otherwise(col('TimeToNotify')))
dfJ = dfJ.withColumn("TimeToNotify",when(col('TimeToNotify')>30,30).otherwise(col('TimeToNotify')))
dfJ = dfJ.fillna(0,['TimeToNotify'])

# create vehicle_age variable
dfJ = dfJ.withColumn('VehicleAge',year(col('IncidentDate')) - col('YearOfManufacture').cast('int')).drop('YearOfManufacture')

# create flags for categorical columns
dfJ = dfJ.withColumn("BodyKey_01",col('BodyKey')=='01').withColumn("BodyKey_02",col('BodyKey')=='02').withColumn("BodyKey_03",col('BodyKey')=='03').withColumn("BodyKey_04",col('BodyKey')=='04').drop('BodyKey')
dfJ = dfJ.withColumn("FuelKey_01",col('FuelKey')=='001').withColumn("FuelKey_02",col('FuelKey')=='002').drop('FuelKey')
dfJ = dfJ.withColumn("Nature_PH",col('Nature')=='CLAIM_PARTY').withColumn("Nature_TP",col('Nature')=='THIRD_PARTY').drop('Nature')

#convert numeric columns to int
convert_to_int = ['Doors','EngineCapacity','Seats','DA_DR','DA_DTL','DA_UTL','DA_UR','DA_O','BodyKey_01','BodyKey_02','BodyKey_03','BodyKey_04','FuelKey_01','FuelKey_02','Nature_PH','Nature_TP']
for c in convert_to_int:
  dfJ = dfJ.withColumn(c,col(c).cast('int'))

# drop columns not needed in modelling
to_drop = ['ClaimRef','VehReg','RepairOngoingFlag','NotNetworkRepairedFlag','TotalLossFlag','MissingDataFlag','StartDate','EndDate','System','ClaimNumber','PMCClaimVersionKey','EventTimestamp','ClaimVersion','IncidentType','ImpactSpeedRange','RepairerIncidentDate','ClaimItemID','RecoveryStatus','ColourDescription','ColourKey','DashCam','Lights','ManualModelDescription','RegistrationDate','RegistrationUnknown','InsurerName1','ManufacturerKey','ModelKey','CategoryDescription','KeptAtCounty','Registration','ModelDescription','IncidentSubCauseDescription','SeatingConfiguration','EmergencyServicesFire','EmergencyServicesPolice','ImpactSpeed']
for c in to_drop:
  dfJ = dfJ.drop(col(c))

# encode damage columns to numerics
mapper = {'false':0,
          'true':1,
          'null':0,
          None:0,
          'None':0,
          'Unknown':0,
          'Minimal':1,
          'Medium':2,
          'Heavy':3,
          'Severe':4,
          'One':1,
          'Two':2,
          'Three':3,
          'Four':4,
          'All':4}
damage_cols = ['BonnetSubmerged','BootOpens','DeployedAirbags','Front','FrontBonnet','FrontLeft','FrontRight','Left','LeftBackseat','LeftFrontWheel','LeftMirror','LeftRearWheel','LeftUnderside','Rear','RearLeft','RearRight','RearWindowDamage','Right','RightBackseat','RightFrontWheel','RightMirror','RightRearWheel','RightRoof','RightUnderside','RoofDamage','UnderbodyDamage','WindscreenDamage','DoorsOpen','Driveable','EngineDamage','EngineRunningInWater','ExhaustDamaged','HailDamage','LightsDamaged','PanelGaps','PassengerAreaSubmerged','RadiatorDamaged','SharpEdges','TheftDamage','WaterIngressWithinEngine','WheelsDamaged','WingMirrorDamaged','SidePanelDamage','StructuralDamage']
special_car_cols = ['IsHighValueVehicle','IsRightHandDrive','IsUnknown','isTaxi']
def convert_damage(d):
  try:
    return mapper[d]
  except:
    return d
damage_udf = udf(convert_damage)
for c in damage_cols + special_car_cols:
  dfJ = dfJ.withColumn(c,damage_udf(col(c)).cast('int'))

# create fields for components of date columns
dfJ = dfJ.withColumn('Notified_DOW',when(dayofweek('NotificationDate')==1,7).otherwise(dayofweek('NotificationDate')-1)).withColumn('Notified_Day',dayofmonth('NotificationDate')).withColumn('Notified_Month',month('NotificationDate')).withColumn('Notified_Year',year('NotificationDate')).withColumn('Incident_DOW',when(dayofweek('IncidentDate')==1,7).otherwise(dayofweek('IncidentDate')-1)).withColumn('Incident_Day',dayofmonth('IncidentDate'))
dfJ = dfJ.drop('NotificationDate','IncidentDate')
dfJ = dfJ.withColumn("PostcodeArea",regexp_extract("KeptAtPostcode", r'^\D+', 0)).drop('KeptAtPostcode')
dfJ = dfJ.withColumn('VehicleAge', when(col('VehicleAge')<=30, col('VehicleAge')).otherwise(None))
dfJ = dfJ.fillna(0,['DA_DR','DA_DTL','DA_UTL','DA_UR','DA_O','BodyKey_01','BodyKey_02','BodyKey_03','BodyKey_04','FuelKey_01','FuelKey_02'])
dfJ = dfJ.fillna("ZZ",['PostcodeArea'])
dfJ = dfJ.fillna("ZMISSING",['ManufacturerDescription'])

# Impute missing values with the median
impute_med_cols = ['VehicleAge','Incident_DOW','Incident_Day']
for c in impute_med_cols:
  median_value = dfJ.approxQuantile(c, [0.5], 0.25)[0]
  dfJ = dfJ.withColumn(c, when(col(c).isNotNull(), col(c)).otherwise(median_value))

#damage summary columns
sev_damage = ['Front','FrontBonnet','FrontLeft','FrontRight','Left','LeftBackseat','LeftFrontWheel','LeftMirror','LeftRearWheel','LeftUnderside','Rear','RearLeft','RearRight','RearWindowDamage','Right','RightBackseat','RightFrontWheel','RightMirror','RightRearWheel','RightRoof','RightUnderside','RoofDamage','UnderbodyDamage','WindscreenDamage']
dfJ = dfJ.withColumn("DamageSev_Total",reduce(add, [col(c) for c in sev_damage]))
dfJ = dfJ.withColumn("DamageSev_Count",reduce(add, [when(col(c) > 0, 1).otherwise(0) for c in sev_damage]))
dfJ = dfJ.withColumn("DamageSev_Mean",when(col('DamageSev_Count')>0,(col('DamageSev_Total')/col('DamageSev_Count'))).otherwise(lit(0)))
dfJ = dfJ.withColumn("DamageSev_Max",greatest('Front','FrontBonnet','FrontLeft','FrontRight','Left','LeftBackseat','LeftFrontWheel','LeftMirror','LeftRearWheel','LeftUnderside','Rear','RearLeft','RearRight','RearWindowDamage','Right','RightBackseat','RightFrontWheel','RightMirror','RightRearWheel','RightRoof','RightUnderside','RoofDamage','UnderbodyDamage','WindscreenDamage'))

# remove columns with constant values
dfJ = dfJ.drop('BonnetSubmerged','EngineRunningInWater','WaterIngressWithinEngine','SidePanelDamage','StructuralDamage','IsHighValueVehicle','IsTaxi','IsUnknown','IsRightHandDrive','TheftDamage')


# COMMAND ----------

#write
dfJ.write.mode("overwrite").parquet('/mnt/datalake/users/HarryBjarnason/TPC/TimeToRepair/all')

# COMMAND ----------

dfJ = spark.read.parquet('/mnt/datalake/users/HarryBjarnason/TPC/TimeToRepair/all')
print(dfJ.select(avg(dfJ.DaysInRepair)).collect()[0][0])
print(dfJ.select(stddev(dfJ.DaysInRepair)).collect()[0][0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Test Split

# COMMAND ----------

train_data, test_data = dfJ.randomSplit([0.8, 0.2], seed=55)
# Display the size of the training and testing sets
print("Training set size:", train_data.count())
print("Testing set size:", test_data.count())

# COMMAND ----------

train_data.write.mode("overwrite").parquet('/mnt/datalake/users/HarryBjarnason/TPC/TimeToRepair/train')
test_data.write.mode("overwrite").parquet('/mnt/datalake/users/HarryBjarnason/TPC/TimeToRepair/test')