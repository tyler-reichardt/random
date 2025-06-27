# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

dfClaimIncident = spark.read.parquet('/mnt/datalake/users/LukeRowland/claimextracts/IncidentDetail/*')

# COMMAND ----------

# filter down columns and cast timestamps
dfClaimIncident = dfClaimIncident.filter(col("IncidentType") == "Accident").select(['PMCClaimVersionKey','EventTimestamp','NotificationDate','ClaimNumber','ClaimVersion','IncidentType','IncidentCauseDescription','IncidentSubCauseDescription','ImpactSpeed','ImpactSpeedRange','IncidentDate','EmergencyServicesFire','EmergencyServicesPolice']).withColumn("EventTimestamp",dfClaimIncident.EventTimestamp.cast(TimestampType())).withColumn("NotificationDate",to_timestamp(dfClaimIncident.NotificationDate,"yyyyMMdd HH:mm:ss")).withColumn("IncidentDate",to_timestamp(dfClaimIncident.IncidentDate,"yyyyMMdd HH:mm:ss"))

# COMMAND ----------

# keep only the first available version. We can't currently see the FNOL timings to keep events within that so this is a workaround for now
wp1 = Window.partitionBy("ClaimNumber").orderBy(col("EventTimestamp").asc())

dfInitialIncident = dfClaimIncident.withColumn("rn",row_number().over(wp1)).filter(col("rn") == 1).drop("rn")

# COMMAND ----------

dfClaimItem = spark.read.parquet('/mnt/datalake/users/LukeRowland/claimextracts/ClaimItemDetail/*')

# COMMAND ----------

# select the relevant columns, cast timestamp,keep only accepted vehicles 

# accepted vehicle type list:
acceptedVehicleList = ["VanMotorVehicleClaimItem","CarMotorVehicleClaimItem","MotorcycleMotorVehicleClaimItem"]

dfClaimItem = dfClaimItem.where(col("ClaimItemType").isin(acceptedVehicleList)).select(['PMCClaimVersionKey','ClaimItemType','ClaimVersion','EventTimestamp','ClaimNumber','Nature','ClaimItemID','DamageAssessment','BonnetSubmerged','BootOpens','DeployedAirbags','Front','FrontBonnet','FrontLeft','FrontRight','Left','LeftBackseat','LeftFrontWheel','LeftMirror','LeftRearWheel','LeftUnderside','Rear','RearLeft','RearRight','RearWindowDamage','Right','RightBackseat','RightFrontWheel','RightMirror','RightRearWheel','RightRoof','RightUnderside','RoofDamage','UnderbodyDamage','WindscreenDamage','DoorsOpen','Driveable','EngineDamage','EngineRunningInWater','ExhaustDamaged','HailDamage','LightsDamaged','PanelGaps','PassengerAreaSubmerged','RadiatorDamaged','SharpEdges','TheftDamage','WaterIngressWithinEngine','WheelsDamaged','WingMirrorDamaged','SidePanelDamage','StructuralDamage','RecoveryStatus','BodyKey','ColourDescription','ColourKey','DashCam','Doors','EngineCapacity','FuelKey','IsHighValueVehicle','IsRightHandDrive','IsUnknown','isTaxi','Lights','ManualModelDescription','RegistrationDate','RegistrationUnknown','SeatingConfiguration','Seats','YearOfManufacture','InsurerName1','ManufacturerKey','ManufacturerDescription','ModelKey','ModelDescription','CategoryDescription','KeptAtCounty','KeptAtPostcode','Registration']).withColumn("EventTimestamp",dfClaimItem.EventTimestamp.cast(TimestampType()))

# COMMAND ----------

# keep only the first available version. We can't currently see the FNOL timings to keep events within that so this is a workaround for now
wp1 = Window.partitionBy(["ClaimNumber","ClaimItemID"]).orderBy(col("EventTimestamp").asc())

dfInitialItem = dfClaimItem.withColumn("rn",row_number().over(wp1)).filter(col("rn") == 1).drop("rn")

# COMMAND ----------

dfInitialItem.count()

# COMMAND ----------

# Join the initial incident and items to get the vehicle data set
dfInitialClaimVehicle = dfInitialIncident.join(dfInitialItem.drop("PMCClaimVersionKey","ClaimVersion","EventTimestamp"),on=["ClaimNumber"],how="inner")

# COMMAND ----------

dfInitialClaimVehicle.count()

# COMMAND ----------

dfInitialClaimVehicle.write.mode("overwrite").parquet('/mnt/datalake/users/AdrienneHall/TPC_ThirdPartyCapture/RepairerData/Exports/InitialClaimVehicleData')