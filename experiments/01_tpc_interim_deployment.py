# Databricks notebook source
pip install lightgbm==4.6.0

# COMMAND ----------

pip install scikit-learn==1.5.0

# COMMAND ----------

import pandas as pd
import datetime as dt
import numpy as np
import os
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.types import StructType,StructField, StringType
from functools import reduce
from operator import add
import mlflow
from io import StringIO
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transcache Read

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Identify Dates to Backfill

# COMMAND ----------

#read in run log
run_log = pd.read_csv('/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/interim_deployment/tpc_interim_deployment_run_log.csv').set_index('date')

#current date, datetime and hour 
today = dt.datetime.now().strftime('%Y-%m-%d')
now = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
hour = dt.datetime.now().hour

#check for an entire missing day in last 2 weeks and if so add to log with null values for runs
last_2_weeks = pd.date_range((dt.datetime.now() - dt.timedelta(days=14)).date(), 
                             (dt.datetime.now() - dt.timedelta(days=1)).date()).strftime('%Y-%m-%d')
               
missing_dates_entire = [x for x in last_2_weeks if x not in run_log.index] #entire dates missing from log

#check log that not missing a run from any other day in last 2 weeks (removing today as that will be run anyway)
missing_dates_partial = run_log[run_log.isna().sum(axis=1) > 0].index.to_list()
missing_dates_partial.remove(today) if today in missing_dates_partial else None

#add entire missing dates to log as empty - when run completes we will fillna() on all backfill dates with current date time to track backfills
for date in missing_dates_entire:
    run_log.loc[date] = [np.nan, np.nan, np.nan]

#create list of dates to backfill
backfill_dates = missing_dates_entire + missing_dates_partial

# update log to reflect date and time of extracts 
run_log.loc[backfill_dates] = run_log.loc[backfill_dates].fillna(now) #all missing backfill dates being updated now
run_hours = {'run_1': 8,'run_2': 12,'run_3': 16} #3 runs throughout the day
for run, hr in run_hours.items():
    if hour==hr:
        run_log.at[today, run] = now

#sort log and save off as a temp version - when run successfully completes we will read this back in and overwrite the master log
run_log.sort_index(inplace=True)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Temp Views

# COMMAND ----------

# define schema to use when importing json files
# general path
schemaPathGeneral = f'/mnt/datalake/general/live/transcache/claimseventpayload/2020-12-07'
sample_schemaGeneral = spark.read.json(schemaPathGeneral).schema

# secure path
schemaPathSecure = f'/mnt/datalake/secure/live/transcache/claimseventpayload/2020-12-07'
sample_schemaSecure = spark.read.json(schemaPathSecure).schema

# COMMAND ----------

path_general = "/mnt/datalake/general/live/transcache/claimseventpayload/"
path_secure = "/mnt/datalake/secure/live/transcache/claimseventpayload/"
spark.conf.set("spark.sql.caseSensitive", "true")

#get list of files to read from todays folder that are not the current hour as current hour file will be updating
files_today_general = [x for x in os.listdir('/dbfs/'+ path_general + today) if x.lstrip('0') != str(hour)] #remove leading 0 for evaluation
paths_today_general = [path_general + today + '/' + x + '/*' for x in files_today_general]
files_today_secure = [x for x in os.listdir('/dbfs/'+ path_secure + today) if x.lstrip('0') != str(hour)] #remove leading 0 for evaluation
paths_today_secure = [path_secure + today + '/' + x + '/*' for x in files_today_secure]

#list of file to read from backfill dates
paths_backfill_general = [path_general + x + '/*' for x in backfill_dates]
paths_backfill_secure = [path_secure + x  + '/*' for x in backfill_dates]

#combine to get list of paths to read for this run
paths_general = paths_backfill_general + paths_today_general
paths_secure = paths_backfill_secure + paths_today_secure

# read in required fields from general claims events:
df_general = spark.read.schema(sample_schemaGeneral).json(paths_general)
df_secure = spark.read.schema(sample_schemaSecure).json(paths_secure)

# create views, apply query and save
df_general.createOrReplaceTempView("vwEventsGeneral")
df_secure.createOrReplaceTempView("vwEventsSecure")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### SQL Queries

# COMMAND ----------

df_general_queried = spark.sql(
"""
---TPC Specific Extracts
SELECT 
body.ComponentEvent.EventTimestamp as EventTimestamp,
body.ComponentEvent.EventType as Type,
convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Number as ClaimNumber,
convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.SelectedClaimVersion as SelectedClaimVersion,
encodedvalues.PiiKey as piikey,
ClaimItemKey,
ClaimItemData.FirstParty as firstParty,
--instigator:
convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Instigator.InstigatorType as InstigatorType,
--claim version info
PMCClaimVersionKey,
PMCClaimVersionData.BusinessProcess as BusinessProcess,
PMCClaimVersionData.CreatedDate as CreatedDate,
PMCClaimVersionData.ChangedDate as ChangedDate,
PMCClaimVersionData.Incident.LiabilityDetails.LiabilityDecision as Liability,
PMCClaimVersionData.Incident.Cause.Data.Description as IncidentCause,
PMCClaimVersionData.Incident.SubCause.Data.Description as IncidentSubCause,
PMCClaimVersionData.Incident.MultiplePartiesInvolved as MultiplePartiesInvolved,
PMCClaimVersionData.Incident.NotificationMethod as NotificationMethod,
size(PMCClaimVersionData.Items.ClaimItem) as CountClaimItems,
--tppro:
--ClaimItemData.TPPros.TPPro.InterventionClaimType as InterventionClaimType,
--ClaimItemData.TPPros.TPPro.InterventionLetterSentDate as InterventionLetterSentDate,
--ClaimItemData.TPPros.TPPro.InterventionOutcome as InterventionOutcome,
--ClaimItemData.TPPros.TPPro.OutboundCommsSentDate as OutboundCommsSentDate,
--ClaimItemData.TPPros.TPPro.RejectionReason as RejectionReason,
--ClaimItemData.TPPros.TPPro.TPProUnsuitabilityReason as TPProUnsuitabilityReason,
--ClaimItemData.TPProCallLogs.TPProCallLog as TPProCallLog,
--contact details
ClaimItemData.Guid as ClaimItemID,
ClaimItemData.Claimant.Guid as PersonGuid,
ClaimItemData.Claimant.Id as PersonID,
ClaimItemData.Claimant.Name,
ClaimItemData.PositionStatus.OverrideReason.Data.Description as OverrideReason,

---Daily Job Claim Incident
concat(substr(convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Audit.CreatedDateTime,1,4),substr(convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Audit.CreatedDateTime,6,2),substr(convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Audit.CreatedDateTime,9,2)," ",substr(convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Audit.CreatedDateTime,12,8)) as NotificationDate,
convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.FirstPartyConfirmed,
convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.FirstPartyConfirmedTPNotifiedClaim,
PMCClaimVersionData.PolicyInfo.PolicyKey as ExternalReference,
PMCClaimVersionData.PolicyInfo.PolicyScheme,
concat(substr( PMCClaimVersionData.PolicyInfo.PolicyStartDate,1,4),substr( PMCClaimVersionData.PolicyInfo.PolicyStartDate,6,2),substr( PMCClaimVersionData.PolicyInfo.PolicyStartDate,9,2)) as PolicyStartDate,
PMCClaimVersionData.PositionStatus.Status as Status,
PMCClaimVersionData.PositionStatus.ClaimClosureReason,
PMCClaimVersionData.PublicVersion,
PMCClaimVersionData.Version as ClaimVersion,
PMCClaimVersionData.PositionStatus.Position.Data.Description as PositionStatusDescription,
PMCClaimVersionData.PositionStatus.Position.Data.PropertyName as PositionStatusPropertyName,
body.ComponentEvent.EventType as EventType,
PMCClaimVersionData.Incident.Type as IncidentType,
PMCClaimVersionData.Incident.Cause.Data.Description as IncidentCauseDescription,
PMCClaimVersionData.Incident.Cause.Data.PropertyName as IncidentCausePropertyName,
PMCClaimVersionData.Incident.SubCause.Data.Description as IncidentSubCauseDescription,
PMCClaimVersionData.Incident.SubCause.Data.PropertyName as IncidentSubCausePropertyName,
--PMCClaimVersion.PMCClaimVersionData.Incident.Circumstances as IncidentCircumstances, 
PMCClaimVersionData.Incident.ImpactSpeed as ImpactSpeed,
PMCClaimVersionData.Incident.ImpactSpeedRange as ImpactSpeedRange,
PMCClaimVersionData.Incident.ImpactSpeedUnit as ImpactSpeedUnit,
PMCClaimVersionData.Incident.WeatherConditions.Daylight.Data.Description as WeatherConditionsDaylight,
PMCClaimVersionData.Incident.WeatherConditions.Precipitation.Data.Description as WeatherConditionsPrecipitation,
PMCClaimVersionData.Incident.WeatherConditions.SystemCollected as WeatherConditionsSystemCollected,
PMCClaimVersionData.Incident.WeatherConditions.Type as WeatherConditionsType,
PMCClaimVersionData.Incident.WeatherConditions.Wind.Data.Description as WeatherConditionsWind,
PMCClaimVersionData.Incident.RoadConditions as RoadConditions,
PMCClaimVersionData.Incident.Location.RoadDescription as RoadDescription,
concat(substr(PMCClaimVersionData.Incident.ReportedDate,1,4),substr(PMCClaimVersionData.Incident.ReportedDate,6,2),substr(PMCClaimVersionData.Incident.ReportedDate,9,2)," ",substr(PMCClaimVersionData.Incident.ReportedDate,12,8)) as ReportedDate,    
concat(substr(PMCClaimVersionData.Incident.StartDate,1,4),substr(PMCClaimVersionData.Incident.StartDate,6,2),substr(PMCClaimVersionData.Incident.StartDate,9,2)," ",substr(PMCClaimVersionData.Incident.StartDate,12,8)) as IncidentDate,
CASE WHEN PMCClaimVersionData.Incident.EmergencyServices.Fire is not null then 'Yes' else 'No' END as EmergencyServicesFire,
CASE WHEN PMCClaimVersionData.Incident.EmergencyServices.Police is not null then 'Yes' else 'No' end as EmergencyServicesPolice,
PMCClaimVersionData.Incident.EmergencyServices.PoliceNotifedMethod as EmergencyServicesPoliceNotifedMethod,
CASE WHEN PMCClaimVersionData.Incident.Witnesses is not null then 'Yes' else 'No' end as Witnesses,
CASE WHEN PMCClaimVersionData.Incident.FootageEvidenceCCTV.FootageEvidenceCCTV IS NOT null then 'Yes' ELSE 'No' END as FootageEvidenceCCTV,
PMCClaimVersionData.Incident.Location.LocationAddress.Country.Description IncidentCountry,
PMCClaimVersionData.Incident.Location.LocationAddress.Country.Key as IncidentCountryKey,
PMCClaimVersionData.Incident.Location.LocationAddress.UKCountry.Description as IncidentUKCountry,

--- Daily Job Claim Item
ClaimItemData.ClaimItemType,
CASE WHEN ClaimItemData.Claimant.IsFirstParty = TRUE THEN 'CLAIM_PARTY' ELSE 'THIRD_PARTY' END as Nature,
ClaimItemData.Damage.Assessment.Category as DamageAssessment,
ClaimItemData.Damage.BonnetSubmerged,
ClaimItemData.Damage.BootOpens,
ClaimItemData.Damage.DeployedAirbags,
ClaimItemData.Damage.Details.Front.Severity as Front,
ClaimItemData.Damage.Details.FrontBonnet.Severity as FrontBonnet,
ClaimItemData.Damage.Details.FrontLeft.Severity as FrontLeft,
ClaimItemData.Damage.Details.FrontRight.Severity as FrontRight,
ClaimItemData.Damage.Details.Left.Severity as Left,	
ClaimItemData.Damage.Details.LeftBackseat.Severity as LeftBackseat,	
ClaimItemData.Damage.Details.LeftFrontWheel.Severity as LeftFrontWheel,	
ClaimItemData.Damage.Details.LeftMirror.Severity as LeftMirror,
ClaimItemData.Damage.Details.LeftRearWheel.Severity as LeftRearWheel,	
ClaimItemData.Damage.Details.LeftUnderside.Severity as LeftUnderside,	
ClaimItemData.Damage.Details.Rear.Severity as Rear,	
ClaimItemData.Damage.Details.RearLeft.Severity as RearLeft,	
ClaimItemData.Damage.Details.RearRight.Severity as RearRight,	
ClaimItemData.Damage.Details.RearWindowDamage.Severity as RearWindowDamage,	
ClaimItemData.Damage.Details.Right.Severity as Right,	
ClaimItemData.Damage.Details.RightBackseat.Severity as RightBackseat,	
ClaimItemData.Damage.Details.RightFrontWheel.Severity as RightFrontWheel,	
ClaimItemData.Damage.Details.RightMirror.Severity as RightMirror,	
ClaimItemData.Damage.Details.RightRearWheel.Severity as RightRearWheel,	
ClaimItemData.Damage.Details.RightRoof.Severity as RightRoof,
ClaimItemData.Damage.Details.RightUnderside.Severity as RightUnderside,	
ClaimItemData.Damage.Details.RoofDamage.Severity as RoofDamage,	
ClaimItemData.Damage.Details.UnderbodyDamage.Severity as UnderbodyDamage,	
ClaimItemData.Damage.Details.WindscreenDamage.Severity as WindscreenDamage,
ClaimItemData.Damage.DoorsOpen,
ClaimItemData.Damage.Driveable,
ClaimItemData.Damage.EngineDamage,
ClaimItemData.Damage.EngineRunningInWater,
ClaimItemData.Damage.ExhaustDamaged,
ClaimItemData.Damage.HailDamage,
ClaimItemData.Damage.LightsDamaged,
ClaimItemData.Damage.PanelGaps,
ClaimItemData.Damage.PassengerAreaSubmerged,
ClaimItemData.Damage.RadiatorDamaged,
ClaimItemData.Damage.SharpEdges,
ClaimItemData.Damage.TheftDamage,
ClaimItemData.Damage.WaterIngressWithinEngine,
ClaimItemData.Damage.WheelsDamaged,
ClaimItemData.Damage.WingMirrorDamaged,
ClaimItemData.Damage.BrokenBones,
ClaimItemData.Damage.Category as CategoryString,
ClaimItemData.Damage.CnfLocReceived,
ClaimItemData.Damage.Collapsed,
ClaimItemData.Damage.ConsciousnessLost,
ClaimItemData.Damage.DamageToForks,
ClaimItemData.Damage.DamageToStand,
ClaimItemData.Damage.DamagedCausedAnObstruction,
ClaimItemData.Damage.DangerousEdges,
ClaimItemData.Damage.Fatality,
ClaimItemData.Damage.FrameBent,
ClaimItemData.Damage.FrameDamaged,
ClaimItemData.Damage.GPAttended,
ClaimItemData.Damage.HandlebarsDamaged,
ClaimItemData.Damage.InjurySubType,
ClaimItemData.Damage.InjuryType,
ClaimItemData.Damage.LimbsLost,
--ClaimItemData.Damage.LoackRearAccess,
ClaimItemData.Damage.LockDoors,
ClaimItemData.Damage.LockRearAccess,
ClaimItemData.Damage.LockedCabDoors,
ClaimItemData.Damage.LockedRearAccess,
ClaimItemData.Damage.MirrorsSmashed,
ClaimItemData.Damage.NearbyPropertyDamaged,
ClaimItemData.Damage.PartialCollapsed,
ClaimItemData.Damage.PersonalInjuryRepresentation,
ClaimItemData.Damage.Rehabilitation.ReceivedTreatment as RehabilitationReceivedTreatment,
ClaimItemData.Damage.ReinsuranceApplicable,
ClaimItemData.Damage.SafetyHazard,
ClaimItemData.Damage.SeatBeltWorn,
ClaimItemData.Damage.Severity,
ClaimItemData.Damage.SidePanelDamage,
ClaimItemData.Damage.StructuralDamage,
--ClaimItemData.Damage.Summary,
ClaimItemData.Damage.TimeTakenOffWork,
ClaimItemData.Damage.TotalCollapsed,
ClaimItemData.LossType,
ClaimItemData.NotOnMID,
ClaimItemData.PassengersInvolved,
ClaimItemData.Theft.RecoveryStatus,
ClaimItemData.Usage.VehicleUsageComments,
ClaimItemData.Vehicle.Body.Key as BodyKey,
ClaimItemData.Vehicle.Colour.Description as ColourDescription,
ClaimItemData.Vehicle.Colour.Key as ColourKey,
ClaimItemData.Vehicle.CurrentMileage,
ClaimItemData.Vehicle.DashCam,
ClaimItemData.Vehicle.Doors,
ClaimItemData.Vehicle.EngineCapacity,
ClaimItemData.Vehicle.Fuel.Key as FuelKey,
ClaimItemData.Vehicle.GrossWeight,
ClaimItemData.Vehicle.IsHighValueVehicle,
ClaimItemData.Vehicle.IsRightHandDrive,
ClaimItemData.Vehicle.IsUnknown,
ClaimItemData.Vehicle.IsTaxi,
ClaimItemData.Vehicle.Lights,
ClaimItemData.Vehicle.ManualModelDescription,
ClaimItemData.Vehicle.Manufacturer as ManufacturerString,
ClaimItemData.Vehicle.MarketValue,
ClaimItemData.Vehicle.Model as ModelString,
ClaimItemData.Vehicle.OdometerInKilometres.Key as OdometerInKilometres,
ClaimItemData.Vehicle.OdometerReading,  
concat(substr(ClaimItemData.Vehicle.OdometerReadingDate,1,4),substr(ClaimItemData.Vehicle.OdometerReadingDate,6,2),substr(ClaimItemData.Vehicle.OdometerReadingDate,9,2)) as OdometerReadingDate,
ClaimItemData.Vehicle.OutstandingFinance,
ClaimItemData.Vehicle.OutstandingFinanceAmount,
ClaimItemData.Vehicle.PermitManualDefinition,
ClaimItemData.Vehicle.PleasureMileage,
concat(substr(ClaimItemData.Vehicle.PurchaseDate,1,4),substr(ClaimItemData.Vehicle.PurchaseDate,6,2),substr(ClaimItemData.Vehicle.PurchaseDate,9,2)) as PurchaseDate,
ClaimItemData.Vehicle.PurchasePrice,
concat(substr(ClaimItemData.Vehicle.RegistrationDate,1,4),substr(ClaimItemData.Vehicle.RegistrationDate,6,2),substr(ClaimItemData.Vehicle.RegistrationDate,9,2)) as RegistrationDate,
ClaimItemData.Vehicle.RegistrationUnknown,
ClaimItemData.Vehicle.SeatingConfiguration,
ClaimItemData.Vehicle.Seats,
ClaimItemData.Vehicle.Size as SizeString,
concat(substr(ClaimItemData.Vehicle.TpVehicleInsuranceExpiryDate,1,4),substr(ClaimItemData.Vehicle.TpVehicleInsuranceExpiryDate,6,2),substr(ClaimItemData.Vehicle.TpVehicleInsuranceExpiryDate,9,2)) as TpVehicleInsuranceExpiryDate,
concat(substr(ClaimItemData.Vehicle.TpVehicleMotExpiryDate,1,4),substr(ClaimItemData.Vehicle.TpVehicleMotExpiryDate,6,2),substr(ClaimItemData.Vehicle.TpVehicleMotExpiryDate,9,2)) as TpVehicleMotExpiryDate,
concat(substr(ClaimItemData.Vehicle.TpVehicleTaxExpiryDate,1,4),substr(ClaimItemData.Vehicle.TpVehicleTaxExpiryDate,6,2),substr(ClaimItemData.Vehicle.TpVehicleTaxExpiryDate,9,2)) as TpVehicleTaxExpiryDate,
ClaimItemData.Vehicle.OvernightLocation.Description as OvernightLocationDescription,
ClaimItemData.Vehicle.OvernightLocation.Key as OvernightLocationKey,
ClaimItemData.Vehicle.Owner,
ClaimItemData.Vehicle.Value,
ClaimItemData.Vehicle.VehicleId,
ClaimItemData.Vehicle.YearOfManufacture,
ClaimItemData.VehicleUnattended,
ClaimItemData.InsuranceDetails.InsuranceDetails[0].Insurer.Name as InsurerName1,
ClaimItemData.InsuranceDetails.InsuranceDetails[1].Insurer.Name as InsurerName2,
ClaimItemData.InsuranceDetails.InsuranceDetails[2].Insurer.Name as InsurerName3,
ClaimItemData.InsuranceDetails.InsuranceDetails[3].Insurer.Name as InsurerName4,
ClaimItemData.InsuranceDetails.InsuranceDetails[4].Insurer.Name as InsurerName5,
CASE WHEN ClaimItemData.Ambulance IS NOT null THEN 'Yes' ELSE 'No' END as Ambulance,
CASE WHEN ClaimItemData.Hospital IS NOT null THEN 'Yes' ELSE 'No' END as Hospital

FROM vwEventsGeneral
-- explode so have row for every claim version and claim item 
LATERAL VIEW posexplode(convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Versions.PMCClaimVersion) PMCClaimVersion as PMCClaimVersionKey, PMCClaimVersionData
LATERAL VIEW posexplode(PMCClaimVersionData.Items.ClaimItem) ClaimItem as ClaimItemKey, ClaimItemData
""").where((col("ClaimItemID").isNotNull()) & coalesce((col("OverrideReason") != 'Raised Incorrectly'), lit(True))) # remove items that are null or have been process incorrectly

# COMMAND ----------

df_secure_queried = spark.sql(
"""
SELECT 
    PMCClaimVersionKey as PMCClaimVersionKey,
    ClaimItemKey as ClaimItemKey,
    piikey as piikey,
    ClaimItemData.Vehicle.Registration as VehicleReg,
    ClaimItemData.Claimant.Customer.Forename as Forename,
    ClaimItemData.Claimant.Customer.Surname as Surname,
    concat(substr(ClaimItemData.Claimant.Customer.DateOfBirth,1,4),substr(ClaimItemData.Claimant.Customer.DateOfBirth,6,2),substr(ClaimItemData.Claimant.Customer.DateOfBirth,9,2)) as DateOfBirth,
    -- just take first known address and contact details for the claimant:
    ClaimItemData.Claimant.Customer.PartyDetails.Addresses.Address[0].Line1 as AddressLine1,
    ClaimItemData.Claimant.Customer.PartyDetails.Addresses.Address[0].Line2 as AddressLine2,
    ClaimItemData.Claimant.Customer.PartyDetails.Addresses.Address[0].Town as Town,
    ClaimItemData.Claimant.Customer.PartyDetails.Addresses.Address[0].County as County,
    ClaimItemData.Claimant.Customer.PartyDetails.Addresses.Address[0].Postcode as Postcode,
    FILTER(ClaimItemData.Claimant.Customer.PartyDetails.ContactMethods.ContactMethod, item -> item.`MethodType` = 'HomeEmail').Details[0]as HomeEmail,
    FILTER(ClaimItemData.Claimant.Customer.PartyDetails.ContactMethods.ContactMethod, item -> item.`MethodType` = 'WorkEmail').Details[0]as WorkEmail,
    FILTER(ClaimItemData.Claimant.Customer.PartyDetails.ContactMethods.ContactMethod, item -> item.`MethodType` = 'HomeTelephone').Details[0]as HomeTelephone,
    FILTER(ClaimItemData.Claimant.Customer.PartyDetails.ContactMethods.ContactMethod, item -> item.`MethodType` = 'WorkTelephone').Details[0]as WorkTelephone,
    FILTER(ClaimItemData.Claimant.Customer.PartyDetails.ContactMethods.ContactMethod, item -> item.`MethodType` = 'MobileTelephone').Details[0]as MobileTelephone,

    -- Daily Job Claim Item
    ClaimItemData.Vehicle.KeptAt.Line1 as KeptAtLine1,
    ClaimItemData.Vehicle.KeptAt.Line2 as KeptAtLine2,
    ClaimItemData.Vehicle.KeptAt.Town as KeptAtTown,
    ClaimItemData.Vehicle.KeptAt.County as KeptAtCounty,
    ClaimItemData.Vehicle.KeptAt.Postcode as KeptAtPostcode,
    ClaimItemData.Vehicle.Registration,

    -- in main build got this from SAS:
    PMCClaimVersionData.Incident.Location.LocationAddress.Postcode as IncidentPostcode

FROM vwEventsSecure
--explode so have row for every claim version and claim item
LATERAL VIEW posexplode(plaintextvalues.ConvertedValues.payload.ClaimsEventPayload.Claim.PMCClaim.Versions.PMCClaimVersion) PMCClaimVersion as PMCClaimVersionKey, PMCClaimVersionData
LATERAL VIEW posexplode(PMCClaimVersionData.Items.ClaimItem) ClaimItem as ClaimItemKey, ClaimItemData
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Daily Job Processing 

# COMMAND ----------

df_general_queried = df_general_queried.withColumn("EventTimestamp", col("EventTimestamp").cast("timestamp"))\
                                       .withColumn("PositionStatusDescription", regexp_replace(col("PositionStatusDescription"), '"', ""))\
                                       .withColumn("RoadDescription", regexp_replace(col("RoadDescription"), '"', ""))


# define schema of strings
manufacturerSchema = StructType().add("Key", StringType()).add("Description", StringType())
modelSchema = StructType().add("Key", StringType()).add("Description", StringType())
sizeSchema = StructType().add("Data",StructType().add("Description", StringType()))
categorySchema = StructType().add("Data",StructType().add("Description", StringType()))

#for some reason this has already been converted to struct type so change back to string so rest of code works
df_general_queried = df_general_queried.withColumn("ModelString", col("ModelString").cast(StringType()))

df_general_queried = df_general_queried\
    .select(
        "*",
        from_json("ManufacturerString",manufacturerSchema).alias("Manufacturer"), 
        from_json("ModelString",modelSchema).alias("Model"),
        from_json("SizeString",sizeSchema).alias("Size"),
        from_json("CategoryString",categorySchema).alias("Category"))\
    .select(
        "*",
        col("Manufacturer.Key").alias("ManufacturerKey"),
        col("Manufacturer.Description").alias("ManufacturerDescription"),
        col("Model.Key").alias("ModelKey"),
        col("Model.Description").alias("ModelDescription"),
        when(col("Size.Data.Description").isNull(), col("SizeString")).otherwise(col("Size.Data.Description")).alias("SizeDescription"),
        when(col("Category.Data.Description").isNull(), col("CategoryString")).otherwise(col("Category.Data.Description")).alias("CategoryDescription")
        )\
    .drop("ManufacturerString","Manufacturer","ModelString","Model","SizeString","Size","CategoryString", "Category")

# get latest version of claim and item
df_general_queried = df_general_queried.withColumn("rn", row_number().over(Window.partitionBy("ClaimNumber","ClaimItemKey").orderBy(col("EventTimestamp").desc(),col("ClaimVersion").desc())))
df_general_queried = df_general_queried.filter(col("rn") == 1).drop("rn").withColumn("EventTimestamp", col("EventTimestamp").cast("STRING"))
df_general_queried.withColumn("PositionStatusDescription", regexp_replace(col("PositionStatusDescription"), '"', "")).withColumn("VehicleUsageComments", regexp_replace(col("VehicleUsageComments"), '"', ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Join General & Secure

# COMMAND ----------

# join general and secure data
df_joined = df_general_queried.join(df_secure_queried, on =["piikey","PMCClaimVersionKey","ClaimItemKey"], how ="left")

#dedupe again
df_joined = df_joined.withColumn("rn", row_number().over(Window.partitionBy("ClaimNumber","ClaimItemKey").orderBy(col("EventTimestamp").desc(),col("ClaimVersion").desc())))
df_joined = df_joined.filter(col("rn") == 1).drop("rn").withColumn("EventTimestamp", col("EventTimestamp").cast("STRING"))

# clean missing claim item types
df_joined = df_joined.withColumn(
    'ClaimItemType', 
    when(col("ClaimItemType").isNotNull(),col("ClaimItemType"))
    .otherwise(
        when((col("ClaimItemType").isNull()) & (col("Registration").isNotNull()), 'CarMotorVehicleClaimItem')
        .otherwise( 
            when((col("ClaimItemType").isNull()) & (col("InjuryType").isNotNull()), 'PedestrianPersonalInjuryClaimItem')
            .otherwise('Unknown')
        )
    )
)

#rename to just 'df' so rest of pasted code works
df = df_joined.withColumn('IncidentDate', to_date(col("IncidentDate"), "yyyyMMdd HH:mm:ss"))\
              .withColumn('NotificationDate', to_date(col("NotificationDate"), "yyyyMMdd HH:mm:ss"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Processing

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Processing and Feature Engineering

# COMMAND ----------

#paste in data build fns commenting out parts that need to be dropped 

# this function cleans up some of the column fields, joins first party info onto the third party rows, drops FP rows and creates some new features
def tpc_clean_engineer_1(df):
    # convert nans in InsurerName 1 to 'unknown'
    df = df.fillna("Unknown", subset="InsurerName1")

    # airbags
    airbag_replacement_dict = {
        "None": 0,
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "All": 5,
    }
    for key, value in airbag_replacement_dict.items():
        df = df.withColumn(
            "DeployedAirbags",
            when(df["DeployedAirbags"] == key, value).otherwise(df["DeployedAirbags"]),
        )
    df = df.withColumn(
        "DeployedAirbags",
        when(
            col("DeployedAirbags").isin(list(airbag_replacement_dict.values())),
            col("DeployedAirbags"),
        )
        .otherwise(0)
        .cast(IntegerType()),
    )

    # damage cols
    damage_cols = [
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
    ]
    # indicator flagging if any info on damage or not (including if unknown entered)
    df = df.withColumn(
        "DamageAssessedInd",
        array_contains(
            array([df[column].isNull() for column in damage_cols]), False
        ).cast("int"),
    )
    # damage replacement dict
    damage_replacement_dict = {
        "Minimal": 1,
        "Medium": 2,
        "Heavy": 3,
        "Severe": 4,
        "Unknown": 0,
    }
    # apply mappings of damage replacement dict to damage cols
    for damage_col in damage_cols:
        for key, value in damage_replacement_dict.items():
            df = df.withColumn(
                damage_col, when(df[damage_col] == key, value).otherwise(df[damage_col])
            )
        # cast to int
        df = df.withColumn(damage_col, col(damage_col).cast("int"))
    # fill missing damage values
    df = df.fillna(value=0, subset=damage_cols)
    # area damage summary features
    df = (
        df.withColumn("DamageSev_Total", reduce(add, [col((c)) for c in damage_cols]))
        .withColumn(
            "DamageSev_Count",
            reduce(add, [when(col((c)) > 0, 1).otherwise(0) for c in damage_cols]),
        )
        .withColumn(
            "DamageSev_Mean",
            when(
                col("DamageSev_Count") > 0,
                (col("DamageSev_Total") / col("DamageSev_Count")),
            ).otherwise(lit(0)),
        )
        .withColumn("DamageSev_Max", greatest(*damage_cols))
    )

    # veh age
    df = df.withColumn(
        "VehicleAge", year(col("IncidentDate")) - col("YearOfManufacture").cast("int")
    ).drop("YearOfManufacture")

    # map fuel key values
    fuel_key_dict = {"001": "Diesel", "002": "Petrol", "004": "Electricity"}
    for key, value in fuel_key_dict.items():
        df = df.withColumn(
            "FuelKey", when(df["FuelKey"] == key, value).otherwise(df["FuelKey"])
        )

    # categorise claim items
    motorVehicleList = [
        "LorryMotorVehicleClaimItem",
        "UnknownVehicleClaimItem",
        "MinibusMotorVehicleClaimItem",
        "VanMotorVehicleClaimItem",
        "BusCoachMotorVehicleClaimItem",
        "HeavyPlantMotorVehicleClaimItem",
        "CarMotorVehicleClaimItem",
        "MotorhomeClaimItem",
        "MotorcycleMotorVehicleClaimItem",
    ]
    allowedVehicleList = [
        "VanMotorVehicleClaimItem",
        "CarMotorVehicleClaimItem",
        "MotorcycleMotorVehicleClaimItem",
    ]
    personalInjuryItemList = ["PedestrianPersonalInjuryClaimItem"]
    otherItemList = [
        "ReinsuranceClaimItem",
        "AnimalInjuryClaimItem",
        "Unknown",
        "KeyClaimItem",
        "PropertyClaimItem",
        "BicycleNonMotorVehicleClaimItem",
        "StreetFurnitureClaimItem",
    ]

    # claim item indicators
    df = (
        df.withColumn(
            "MotorVehicleFP",
            when(
                (col("Nature") == "CLAIM_PARTY")
                & (col("ClaimItemType").isin(motorVehicleList)),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "MotorVehicleTP",
            when(
                (col("Nature") == "THIRD_PARTY")
                & (col("ClaimItemType").isin(motorVehicleList)),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "PIItems",
            when(
                col("ClaimItemType") == "PedestrianPersonalInjuryClaimItem", 1
            ).otherwise(0),
        )
        .withColumn(
            "OtherItems", when(col("ClaimItemType").isin(otherItemList), 1).otherwise(0)
        )
        .withColumn(
            "AllowedVehicle",
            when(
                (col("Nature") == "THIRD_PARTY")
                & (col("ClaimItemType").isin(allowedVehicleList)),
                1,
            ).otherwise(0),
        )
    )

    # # get count of different types of claim items per incident and whether capture attempted and successful
    # df = df.withColumn(
    #     "CaptureSuccess_AdrienneVersion",
    #     when(
    #         (
    #             (col("InterventionOutcome") == "Captured")
    #             | (col("Has_Paid_Intervention") == True)
    #         ),
    #         1,
    #     ).otherwise(0),
    # ).withColumn(
    #     "CaptureCallAttempted",
    #     when(col("TPProCallLogKey").isNull(), 0).otherwise(1),
    # )

    # get item counts for each claim to help with filtering
    df_grouped = df.groupBy("ClaimNumber").agg(
        sum("MotorVehicleTP").alias("TPMotorVehicle_Count"),
        sum("PIItems").alias("PIItem_Count"),
        sum("OtherItems").alias("OtherItem_Count"),
        sum("AllowedVehicle").alias("AllowedVehicle_Count"),
    )

    df = df.join(df_grouped, on="ClaimNumber", how="left")
    df_grouped.unpersist()

    # clean some data types:
    int_veh_cols = ["Doors", "Seats", "EngineCapacity", "Value"]
    for c in int_veh_cols:
        df = df.withColumn(c, col(c).cast(IntegerType()))

    bool_veh_cols = [
        "DoorsOpen",
        "BootOpens",
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
        "Driveable",
    ]
    for c in bool_veh_cols:
        df = df.withColumn(c, col(c).cast(BooleanType()).cast(IntegerType()))

    # FP info damage:
    # these other columns need to be pulled for FP and joined onto TP rows (as well as damage sev columns):
    other_veh_info_cols = [
        "DamageAssessedInd",
        "ClaimItemType",
        "BootOpens",
        "DeployedAirbags",
        "Driveable",
        "WheelsDamaged",
        "WingMirrorDamaged",
        "DoorsOpen",
        "EngineDamage",
        "ExhaustDamaged",
        "LightsDamaged",
        "PanelGaps",
        "RadiatorDamaged",
        "HailDamage",
        "FuelKey",
        "Doors",
        "Seats",
        "Value",
        "ManufacturerDescription",
        "PassengerAreaSubmerged",
        "EngineCapacity",
        "SharpEdges",
        "KeptAtPostcode",
        "VehicleAge",
        "DamageSev_Total",
        "DamageSev_Count",
        "DamageSev_Mean",
        "DamageSev_Max",
        "BodyKey",
    ]
    df_fp = df.where(col("MotorVehicleFP") == 1).select(
        damage_cols
        + other_veh_info_cols
        + ["ClaimNumber", "PMCClaimVersionKey", "ClaimVersion"]
    )
    # drop all FP rows from main df, rename damage cols and join FP info back on to main df,
    df = df.where(col("MotorVehicleFP") != 1)
    for column in damage_cols + other_veh_info_cols:
        df = df.withColumnRenamed(column, "TP_" + column)
        df_fp = df_fp.withColumnRenamed(column, "FP_" + column)
    df = df.join(
        df_fp, on=["ClaimNumber", "PMCClaimVersionKey", "ClaimVersion"], how="left"
    )

    # filter so only looking at TP allowed vehicles
    df = df.where(col("MotorVehicleTP") == 1)

    return df

# this function does the more 'straightforward' feature engineering
def tpc_clean_engineer_2(df):
    # notification features
    df = (
        df.withColumn(
            "TimeToNotify", datediff(col("NotificationDate"), col("IncidentDate"))
        )
        .withColumn(
            "TimeToNotify",
            when(col("TimeToNotify") < 0, 0).otherwise(col("TimeToNotify")),
        )
        .withColumn(
            "TimeToNotify",
            when(col("TimeToNotify") > 30, 30).otherwise(col("TimeToNotify")),
        )
        .withColumn(
            "Notified_DOW",
            when(dayofweek("NotificationDate") == 1, 7).otherwise(
                dayofweek("NotificationDate") - 1
            ),
        )
        .withColumn("Notified_Day", dayofmonth("NotificationDate"))
        .withColumn("Notified_Month", month("NotificationDate"))
        .withColumn("Notified_Year", year("NotificationDate"))
        .withColumn(
            "Incident_DOW",
            when(dayofweek("IncidentDate") == 1, 7).otherwise(
                dayofweek("IncidentDate") - 1
            ),
        )
        .withColumn("Incident_Day", dayofmonth("IncidentDate"))
    )

    # postcode area
    df = df.withColumn(
        "PostcodeArea",
        when(
            col("IncidentPostcode").isNotNull(),
            regexp_extract("IncidentPostcode", r"^\D+", 0),
        ).otherwise(
            when(
                col("FP_KeptAtPostcode").isNotNull(),
                regexp_extract("FP_KeptAtPostcode", r"^\D+", 0),
            ).otherwise("ZZ")
        ),
    )

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Filter Capture Cases Only

# COMMAND ----------

#elligible claim filter to only select cases being routed to the capture team 
def tpc_filtering_interim_deployment(df):
    # A claim is in scope for capture if:
      
    # - It was an 'Accident' type claim
    df = df.where(col('IncidentType')=='Accident')
    # - Only 2 vehicles were involved, our party and 1 third party
    df = df.where(col("TPMotorVehicle_Count") == 1)
    # - The third party vehicle was a car, motorcycle or van
    df = df.where(col('AllowedVehicle_Count')==1)
    # - The claim was not instigated by the third party insurer or a third party representative
    df = df.where(~(col('InstigatorType').isin(['ThirdPartyInsurer','ThirdPartyRepresentative'])))
    # - There are contact details for third party recorded on the claim
    df = df.where(col("AddressLine1").isNotNull() | 
                  col("HomeEmail").isNotNull() | 
                  col("WorkEmail").isNotNull() | 
                  col("HomeTelephone").isNotNull() | 
                  col("WorkTelephone").isNotNull() | 
                  col("MobileTelephone").isNotNull())
    # - Liability = 'Fault'
    df = df.where(col("Liability") == "Fault")

    #also:
    
    # Claim should be open
    df = df.where(col("Status")!="Closed")
    # Claim should be notified within the last 2 weeks
    df = df.where(to_timestamp(col("NotificationDate"),"yyyyMMdd HH:mm:ss") >= (dt.datetime.now() - dt.timedelta(14)).date())

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time to Repair Scoring

# COMMAND ----------

# fn first performs some data prep, encoding and imputation, then selects modelling features and scores through ttr model
# then joins scores back onto initial df
def ttr_prep_score(df):
    # Different method of encoding body and fuel keys:
    df_ttr = (
        df.withColumn("TP_BodyKey_01", (col("TP_BodyKey") == "01").cast("int"))
        .withColumn("TP_BodyKey_02", (col("TP_BodyKey") == "02").cast("int"))
        .withColumn("TP_BodyKey_03", (col("TP_BodyKey") == "03").cast("int"))
        .withColumn("TP_BodyKey_04", (col("TP_BodyKey") == "04").cast("int"))
        .withColumn("TP_FuelKey_01", (col("TP_FuelKey") == "001").cast("int"))
        .withColumn("TP_FuelKey_02", (col("TP_FuelKey") == "002").cast("int"))
        .withColumn("FP_BodyKey_01", (col("FP_BodyKey") == "01").cast("int"))
        .withColumn("FP_BodyKey_02", (col("FP_BodyKey") == "02").cast("int"))
        .withColumn("FP_BodyKey_03", (col("FP_BodyKey") == "03").cast("int"))
        .withColumn("FP_BodyKey_04", (col("FP_BodyKey") == "04").cast("int"))
        .withColumn("FP_FuelKey_01", (col("FP_FuelKey") == "001").cast("int"))
        .withColumn("FP_FuelKey_02", (col("FP_FuelKey") == "002").cast("int"))
        .withColumn("DA_DR", (col("DamageAssessment") == "DriveableRepair").cast("int"))
        .withColumn("DA_DTL", (col("DamageAssessment") == "DriveableTotalLoss").cast("int"))
        .withColumn("DA_UR", (col("DamageAssessment") == "UnroadworthyRepair").cast("int"))
        .withColumn("DA_UTL", (col("DamageAssessment") == "UnroadworthyTotalLoss").cast("int"))
        .withColumn("DA_O",
            (~col("DamageAssessment").isin(
                "DriveableRepair",
                "DriveableTotalLoss",
                "UnroadworthyRepair",
                "UnroadworthyTotalLoss"
            )).cast("int")))

    # clean some data types:
    int_veh_cols = [
        "Doors",
        "Seats",
        "Value",
        "EngineCapacity",
        "DoorsOpen",
        "BootOpens",
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
        "Driveable",
    ]
    for c in int_veh_cols:
        df_ttr = df_ttr.withColumn("FP_" + c, col("FP_" + c).cast("int")).withColumn(
            "TP_" + c, col("TP_" + c).cast("int")
        )

    # create extra features needed for ttr model and tidy up missings and some col names
    df_ttr = (
        df_ttr.withColumn("Nature_TP", lit(1))
        .withColumn("Nature_PH", lit(0))
        # .drop("IncidentCauseDescription")
        # .withColumnRenamed("INCIDENT_CAUSE_DESC", "IncidentCauseDescription")
        .fillna(
            "ZMISSING", ["TP_ManufacturerDescription", "FP_ManufacturerDescription"]
        )
        .fillna(0)
    )

    # create list of modelling features for ttr
    car_fields = [
        "ExhaustDamaged",
        "BodyKey_04",
        "RightRoof",
        "RadiatorDamaged",
        "RightFrontWheel",
        "HailDamage",
        "RightMirror",
        "LeftMirror",
        "RearWindowDamage",
        "LeftRearWheel",
        "Front",
        "BootOpens",
        "FrontBonnet",
        "LeftFrontWheel",
        "RearRight",
        "Driveable",
        "FuelKey_02",
        "DamageSev_Count",
        "Right",
        "WindscreenDamage",
        "BodyKey_01",
        "Doors",
        "RightUnderside",
        "LeftBackseat",
        "WheelsDamaged",
        "Seats",
        "DamageSev_Total",
        "PanelGaps",
        "RearLeft",
        "RoofDamage",
        "DeployedAirbags",
        "Value",
        "ManufacturerDescription",
        "BodyKey_03",
        "DamageSev_Mean",
        "LeftUnderside",
        "FuelKey_01",
        "DamageSev_Max",
        "RightBackseat",
        "VehicleAge",
        "PassengerAreaSubmerged",
        "EngineCapacity",
        "SharpEdges",
        "Left",
        "Rear",
        "DoorsOpen",
        "EngineDamage",
        "BodyKey_02",
        "FrontRight",
        "RightRearWheel",
        "UnderbodyDamage",
        "FrontLeft",
        "WingMirrorDamaged",
        "LightsDamaged",
    ]
    clm_fields = [
        "Incident_DOW",
        "TimeToNotify",
        "Notified_Day",
        "PostcodeArea",
        "Nature_TP",
        "Nature_PH",
        "Notified_Month",
        "IncidentCauseDescription",
        "Incident_Day",
        "Notified_DOW",
        "DA_DR",
        "DA_DTL",
        "DA_UTL",
        "DA_UR",
        "DA_O",
        "Notified_Year",
    ]
    all = clm_fields + ["TP_" + c for c in car_fields]

    # define dataset to be scored through TimeToRepair model (and remove TP prefix from col names)
    df_ttr = df_ttr.select(
        "PMCClaimVersionKey", "ClaimItemID", "ClaimNumber", "ClaimVersion", *all
    )
    for c in car_fields:
        df_ttr = df_ttr.withColumnRenamed("TP_" + c, c)

    # fillna
    df_ttr = df_ttr.fillna(0)

    # Score cases with time to repair prediction
    logged_model = "runs:/432e6e59366142129c5643cc7667541f/model"
    # Load model as a Spark UDF.
    loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)
    # Predict on a Spark DataFrame.
    df_scored = df_ttr.withColumn(
        "prediction", loaded_model(struct(*map(col, df_ttr.columns)))[0]
    ).select(
        [
            "ClaimVersion",
            "PMCClaimVersionKey",
            "ClaimItemID",
            "ClaimNumber",
            "prediction",
        ]
    )

    # join back on
    df = df.join(
        df_scored,
        on=["ClaimVersion", "PMCClaimVersionKey", "ClaimItemID", "ClaimNumber"],
        how="left",
    )

    return df

# COMMAND ----------

#call processing functions up to this point:
df = tpc_clean_engineer_1(df)
df = tpc_clean_engineer_2(df)
df = tpc_filtering_interim_deployment(df)
df = ttr_prep_score(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Cleaning
# MAGIC The below is pasted in from the 03_Cleaning_and_Quality_Checks notebook

# COMMAND ----------

# for these features most of variance is only in 1 level so create indicators for that level and drop original columns to reduce noise
df = df.withColumn(
    "WetRoad_Ind", when(col("RoadConditions") == "Wet", 1).otherwise(0)
).withColumn(
    "RainyWeather_Ind", when(col("WeatherConditionsType") == "Rain", 1).otherwise(0)
)

# body key
body_key_mappings = {
    "01": "Saloon",
    "02": "Hatchback",
    "03": "Estate",
    "04": "Coupe",
    "06": "Pickup",
    "07": "Van",
    "08": "Convertible",
}

# loop through to replace keys with values
for k, v in body_key_mappings.items():
    df = df.withColumn(
        "TP_BodyKey", when(col("TP_BodyKey") == k, v).otherwise(col("TP_BodyKey"))
    ).withColumn(
        "FP_BodyKey", when(col("FP_BodyKey") == k, v).otherwise(col("FP_BodyKey"))
    )

# fill remaining low volume keys as 'Other'
df = df.withColumn(
    "TP_BodyKey",
    when(~col("TP_BodyKey").isin(list(body_key_mappings.values())), "Other").otherwise(
        col("TP_BodyKey")
    ),
).withColumn(
    "FP_BodyKey",
    when(~col("FP_BodyKey").isin(list(body_key_mappings.values())), "Other").otherwise(
        col("FP_BodyKey")
    ),
)

# impact speed range is more populated than speed, so clean the values converting to numeric
speed_range_mappings = {
    "Stationary": "0",
    "ZeroToSix": "3",
    "SevenToFourteen": "11",
    "FifteenToTwenty": "18",
    "TwentyOneToThirty": "26",
    "ThirtyOneToFourty": "36",
    "FourtyOneToFifty": "46",
    "FiftyOneToSixty": "56",
    "SixtyOneToSeventy": "66",
    "OverSeventy": "70",
    "Unknown": "-1",
}

df = df.withColumn("ImpactSpeedRange_int", lit(0))
for k, v in speed_range_mappings.items():
    df = df.withColumn(
        "ImpactSpeedRange_int",
        when(col("ImpactSpeedRange") == k, v).otherwise(col("ImpactSpeedRange_int")),
    )

# insurer bandings / mappings:
insurer_mappings = {
    "Admiral Insurance Services Ltd": "Admiral Insurance Services Ltd",
    "Aviva Insurance Ltd": "Aviva Insurance Ltd",
    "Unknown": "Unknown",
    "Hastings Direct": "Hastings Direct",
    "LV=": "LV=",
    "Esure Insurance Ltd": "Esure Insurance Ltd",
    "UK Insurance Limited": "UK Insurance Limited",
    "First Central Insurance Management Ltd": "First Central Insurance Management Ltd",
    "Ageas Insurance Limited": "Ageas Insurance Limited",
    "Direct Line Insurance": "Direct Line Insurance",
    "West Bay Insurance": "West Bay Insurance",
    "AXA Insurance": "AXA Insurance",
    "AXA insurance": "AXA Insurance",
    "RSA": "RSA",
    "Tesco Underwriting Ltd": "Tesco Underwriting Ltd",
    "Churchill Insurance": "Churchill Insurance",
    "Haven Insurance": "Haven Insurance",
    "Allianz Insurance PLC": "Allianz Insurance PLC",
    "Covea Insurance": "Covea Insurance",
    "AA Underwriting Insurance Company Limited": "AA Underwriting Insurance Company Limited",
    "Watford Insurance c/o Somerset Bridge Ltd ": "Watford Insurance c/o Somerset Bridge Ltd ",
    "Zurich Insurance": "Zurich Insurance",
    "Zenith Insurance": "Zenith Insurance",
    "NFU Mutual": "NFU Mutual",
    "Highway Insurance": "Highway Insurance",
    "Markerstudy Insurance": "Markerstudy Insurance",
    "RSA Motability": "RSA Motability",
    "Wakam": "Wakam",
    "QBE Insurance (Europe) Ltd": "QBE Insurance (Europe) Ltd",
    "Accredited Insurance (Formally R&Q)": "Accredited Insurance (Formally R&Q)",
    "Acromas Insurance Company Ltd": "Acromas Insurance Company Ltd",
    "One Insurance Limited": "One Insurance Limited",
    "AIG Europe Ltd": "AIG Europe Ltd",
    "ERS Syndicate Services Ltd": "ERS Syndicate Services Ltd",
    "Mulsanne Insurance": "Mulsanne Insurance",
    "Aioi Nissay Dowa Insurance UK Ltd": "Aioi Nissay Dowa Insurance UK Ltd",
    "NIG Insurance": "NIG Insurance",
    "Sabre Insurance": "Sabre Insurance",
    "QIC Europe Ltd - KGM Motor": "QIC Europe Ltd - KGM Motor",
    "Admiral Insurance": "Admiral Insurance Services Ltd",
    "Tradex Insurance": "Other",
    "Axa Insurance DAC": "Other",
    "Alwyn Insurance Company Ltd": "Other",
    "Privilege": "Other",
    "Marshmallow Insurance Limited": "Other",
    "New India Insurance": "Other",
    "KGM Motor": "Other",
    "Nelson Insurance Company Ltd": "Other",
    "Elephant Insurance": "Other",
    "Premier Insurance Company Ltd": "Other",
    "Tesco Bank": "Other",
    "Admiral Insurance (Gibraltar) Ltd": "Other",
    "Berkshire Hathaway Insurance": "Other",
    "Co-Operative Insurance": "Other",
    "Quote Me Happy": "Other",
    "Trinity Lane (Adrian Flux)": "Other",
    "WATFORD INSURANCE COMPANY EUROPE LIMITED": "Other",
    "ERS Claims Limited": "Other",
    "Great Lakes Insurance": "Other",
    "Advantage Insurance Company Ltd": "Other",
    "Somerset Bridge Limited": "Other",
    "QIC Europe Ltd - City Underwriting": "Other",
    "Calpe Insurance Company Ltd": "Other",
    "One Call Insurance": "Other",
    "Trinity Claims": "Other",
    "More Than": "Other",
    "Insurethebox Ltd": "Other",
    "Aviva Ireland": "Other",
    "Saga Insurance": "Other",
    "Hastings Insurance Services Ltd": "Other",
    "Diamond": "Other",
    "QIC Europe Ltd.": "Other",
    "Ace Insurance": "Other",
    "MCE Insurance": "Other",
    "Protector Insurance": "Other",
    "Collingwood Insurance Ltd": "Other",
    "Sheilas' Wheels": "Other",
    "Extracover Insurance Company limited": "Other",
    "XL Catlin Insurance Company": "Other",
    "Ageas Insurance Ltd": "Other",
    "Travelers Insurance": "Other",
    "Swiftcover": "Other",
    "HDI Global SE": "Other",
    "Axa Ireland": "Other",
    "General Accident": "Other",
    "Royal & Sun Alliance Insurance Ltd": "Other",
    "Amlin UK Limited": "Other",
    "RAC Insurance": "Other",
    "Royal & Sun Alliance Insurance PLC": "Other",
    "Zego Insurance": "Other",
    "Bell Insurance": "Other",
    "Bromley Garage Services": "Other",
    "Darwin insurance": "Other",
    "Skyfire Insurance Company": "Other",
    "Arch Insurance Company (UK) Limited ": "Other",
    "Allianz Insurance": "Other",
    "RSA Insurance Ireland Limited": "Other",
    "Green Realisation 123 Ltd (Formally MCE Insurance": "Other",
    "AXA": "Other",
    "Accelerant Insurance Ltd": "Other",
    "RSA insurance": "Other",
    "Go Skippy": "Other",
    "Freeway Insurance": "Other",
    "Acorn Insurance": "Other",
    "AIG UK Ltd": "Other",
    "Broker Direct": "Other",
    "Chubb European Group Ltd": "Other",
    "Equity Red Star": "Other",
    "Direct line insurance ": "Other",
    "UK Insurance Ltd": "Other",
    "AXA ROI": "Other",
    "Cornish Mutual Insurance Co Ltd": "Other",
    "ADVANTAGE INSURANCE COMPANY LTD": "Other",
    "AXA Fleet": "Other",
    "Churchill Insurance ": "Other",
    "Halifax Insurance": "Other",
    "Allianz Insurance ": "Other",
    "Broadspire": "Other",
    "Sainsbury's Insurance": "Other",
    "Esure Bulk": "Other",
    "LLoyds Bank  Home Insurance": "Other",
    "USAA Limited": "Other",
    "Tesco Motor Claims": "Other",
    "AXA Insurance UK": "Other",
    "Aviva Bulk": "Other",
    "CIS General Insurance Limited": "Other",
    "XS Direct": "Other",
    "ERS CLAIMS LTD": "Other",
    "HDI GLOBAL SE": "Other",
    "Direct Line Motor Claims": "Other",
    "Tradewise Insurance": "Other",
    "LV": "Other",
    "HISCOX INSURANCE CO LIMITED": "Other",
    "Pukka insure": "Other",
    "Walsingham Motor Insurance Limited ": "Other",
    "The Co-Operative": "Other",
    "Admiral Bulk": "Other",
    "Ageas- Household insurance": "Other",
    "Arch Insurance ": "Other",
    "Gallagher Insurance": "Other",
    "Great Lakes Insurance SE": "Other",
    "1st Central Insurance": "Other",
    "Ticker Claims": "Other",
    "Geoffrey Insurance Services": "Other",
    "Noble Claims": "Other",
    "Direct Line": "Other",
    "Rural Insurance C/o Zurich": "Other",
    "Insure 2 Drive (Sabre)": "Other",
    "Liverpool Victoria Insurance Company Ltd ": "Other",
    "Davies Group (AIOI)": "Other",
    "Toyota Motor Insurance": "Other",
    "WNS Assistance": "Other",
    "QIC Europe Ltd - Policy Expert": "Other",
    "CIS General Insurance": "Other",
    "Liberty Insurance": "Other",
    "Gladiator Insurance": "Other",
    "QIC Europe Ltd - Eridge Underwriting Agency Ltd": "Other",
    "Van Ameyde": "Other",
    "Ford Insure": "Other",
    "ALLIANZ INSURANCE ": "Other",
    "Gallagher Bassett": "Other",
    "HADLEIGH": "Other",
    "Amet Insurance": "Other",
    "CODEVE INSURANCE COMPANY DAC": "Other",
    "Davies Group": "Other",
    "Gefion - Insurance": "Other",
    "1st central": "Other",
    "Collingwood Insurance Company Ltd": "Other",
    "Ageas ": "Other",
    "Haven Claims": "Other",
    "CO-OP Insurance Services ": "Other",
    "ERS CLAIMS": "Other",
    "AXA COMMERCIAL": "Other",
    "Nelson Insurance Company Limited": "Other",
    "Prudential Car Claims": "Other",
    "Axa Insurance": "Other",
    "Budget Insurance": "Other",
    "Tesco": "Other",
    "Churchill Motor Claims": "Other",
    "Direct Commercial": "Other",
    "Centrica": "Other",
    "RSA MOTABILITY": "Other",
    "LV Bulk": "Other",
    "ERS Insurance Ltd": "Other",
    "Tesco Underwriting ": "Other",
    "BMW Car Insurance": "Other",
    "Enterprise Rent-A-Car UK Limited": "Other",
    "Zurich Insurance PLC": "Other",
    "Go Girl": "Other",
    "AA Insurance Services ": "Other",
    "PSA Car Insurance": "Other",
    "Insure The Box": "Other",
    "NFU Mutual ": "Other",
    "Hedgehog Insurance": "Other",
    "La Parisienne Assurances": "Other",
    "Capulus": "Other",
    "Qdos Accident Management": "Other",
    "Eldon Insurance": "Other",
    "AON": "Other",
    "AA Accident Assist ": "Other",
    "QBE UK  Ltd": "Other",
    "First Alternative": "Other",
    "ZMIS LTD CLAIMS": "Other",
    "admiral ": "Other",
    "Laurie Ross Insurance ": "Other",
    "FMG Fleet Incident Management": "Other",
    "Southern Rock Insurance Company Limited": "Other",
    "QIC Europe Ltd - PUKKA Insure": "Other",
    "AA Accident assist": "Other",
    "Axa Motor Insurance": "Other",
    "Diamond Insurance": "Other",
    "Admiral Insurance Services Limited": "Other",
    "Hastings Insurance Services Limited ": "Other",
    "Inshur UK": "Other",
    "Arnold Clark": "Other",
    "Hastings Bulk": "Other",
    "DCL DirectCommercial": "Other",
    "Enterprise Rent-A-Car": "Other",
    "Allianz Claims": "Other",
    "Covea Bulk": "Other",
    "Insure The Box Limited": "Other",
    "Carraig Insurance": "Other",
    "Accident Exchange": "Other",
    "AXA Insurance ": "Other",
    "Tesco Underwriting": "Other",
    "Towergate Insurance": "Other",
    "Aviva": "Other",
    "Liberty Global": "Other",
    "CO OP Claims and Underwriting ": "Other",
    "Arch Insurance": "Other",
    "BDELITE": "Other",
    "Accident Credit Group": "Other",
    "QIC Europe Ltd - KGM Claims": "Other",
    "Key Claims and Administration": "Other",
    "Das Claims": "Other",
    "Collision Solutions": "Other",
    "Insure The Box Limited ": "Other",
    "Ams Insurance / Broker": "Other",
    "Volkswagen Car Claims": "Other",
    "Romero Insurance": "Other",
    "Sagar Insurances": "Other",
    "Sedgwick": "Other",
    "GIS General Insurance Limited": "Other",
    "Insurance Blackburn": "Other",
    "Transcare Solutions": "Other",
    "Total Transparent Solutions Limited": "Other",
    "L V =": "Other",
    "Enterprise": "Other",
    "Coop Insurance": "Other",
    "Peugeot Motor Claims": "Other",
    "FMG": "Other",
    "Zurich Insurance Company Ltd": "Other",
    "QIC Europe Ltd - Nelson Policies @ Lloyds": "Other",
    "Tesco Bank Box Insurance": "Other",
    "R&Q Insurance (Malta) Limited": "Other",
    "West Midlands Travel LTD": "Other",
    "easi drive": "Other",
    "BD elite ": "Other",
    "MacDonald Group ": "Other",
    "Alpha Insurance": "Other",
    "churchill insurance": "Other",
    "Sovereign": "Other",
    "Privelege Insurance": "Other",
    "Waterfront Insurance ": "Other",
    "Auxillis": "Other",
    "AONE+": "Other",
    "allianz insurance": "Other",
    "4th Dimension": "Other",
    "Network Rail": "Other",
    "Bray Insurance Company Limited": "Other",
    "Co-Op Isurance Services Limited": "Other",
    "Miles Smith ": "Other",
    "One Call Claims": "Other",
    "Atlas Insurance PCC Ltd": "Other",
    "Carpenters Limited": "Other",
    "SAVA INSURANCE COMPANY D.D.": "Other",
    "DAS Group": "Other",
    "4th Dimension Innovation Ltd": "Other",
    "Geo Underwriting Services Limited": "Other",
    "Quote Me Happy Motor Claims": "Other",
    "Accredited Insurance (Formerly R&Q)": "Other",
    "Renault Insurance": "Other",
    "BROKER DIRECT PLC": "Other",
    "Acromos Insurance Co Ltd": "Other",
    "First underwriting ltd ": "Other",
    "Kingsway Claims": "Other",
    "Devitt": "Other",
    "Coversure": "Other",
    "Clegg Gifford": "Other",
    "Inter europe": "Other",
    "Prestige Underwriting": "Other",
    "Connexus Group ": "Other",
    "one call claims": "Other",
    "Lloyds banking": "Other",
    "City Insurance Group": "Other",
    "Wrightsure Insurance Services": "Other",
    "Aioi Nissay Dowa Insurance Europe": "Other",
}
for k, v in insurer_mappings.items():
    df = df.withColumn(
        "InsurerName1", when(col("InsurerName1") == k, v).otherwise(col("InsurerName1"))
    )
# fill remaining low volume levels as 'Other' - initially didnt need to do this but have reran with wider date range so new low volume values found
df = df.withColumn(
    "InsurerName1",
    when(~col("InsurerName1").isin(list(insurer_mappings.values())), "Other").otherwise(
        col("InsurerName1")
    ),
)

# manufacturer
manufacturer_mappings = {
    "FORD": "FORD",
    "VAUXHALL": "VAUXHALL",
    "VOLKSWAGEN": "VOLKSWAGEN",
    "BMW": "BMW",
    "AUDI": "AUDI",
    "MERCEDES-BENZ": "MERCEDES-BENZ",
    "TOYOTA": "TOYOTA",
    "NISSAN": "NISSAN",
    "PEUGEOT": "PEUGEOT",
    "HONDA": "HONDA",
    "RENAULT": "RENAULT",
    "KIA": "KIA",
    "HYUNDAI": "HYUNDAI",
    "CITROEN": "CITROEN",
    "SKODA": "SKODA",
    "MINI": "MINI",
    "SEAT": "SEAT",
    "LANDROVER": "LANDROVER",
    "FIAT": "FIAT",
    "MAZDA": "MAZDA",
    "VOLVO": "VOLVO",
    "SUZUKI": "SUZUKI",
    "JAGUAR": "JAGUAR",
    "MITSUBISHI": "MITSUBISHI",
    "DACIA": "DACIA",
    "LEXUS": "LEXUS",
    "TESLA": "TESLA",
    "MG": "MG",
    "YAMAHA": "YAMAHA",
    "PORSCHE": "PORSCHE",
    "ALFA ROMEO": "ALFA ROMEO",
    "SMART": "Other",
    "JEEP": "Other",
    "DS": "Other",
    "CHEVROLET": "Other",
    "LONDON TAXIS": "Other",
    "SUBARU": "Other",
    "SAAB": "Other",
    "KAWASAKI": "Other",
    "ISUZU": "Other",
    "CUPRA": "Other",
    "SSANGYONG": "Other",
    "LEXMOTO": "Other",
    "ALEXANDER DENNIS": "Other",
    "CHRYSLER": "Other",
    "TRIUMPH": "Other",
    "INFINITI": "Other",
    "VESPA-PIAGGIO": "Other",
    "KTM": "Other",
    "DAIHATSU": "Other",
    "IVECO": "Other",
    "ROVER": "Other",
    "BENTLEY": "Other",
    "DAF": "Other",
    "HILLMAN": "Other",
    "SCANIA": "Other",
    "HARLEY-DAVIDSON": "Other",
    "MASERATI": "Other",
    "SYM": "Other",
    "DODGE": "Other",
    "APRILIA": "Other",
    "KEEWAY": "Other",
    "SINNIS": "Other",
    "PIAGGIO": "Other",
    "LOTUS": "Other",
    "WRIGHTBUS": "Other",
    "UNKNOWN": "Other",
    "AJS": "Other",
    "PERODUA": "Other",
    "POLESTAR": "Other",
    "ASTON MARTIN": "Other",
    "BYD": "Other",
    "ABARTH": "Other",
    "DAF TRUCKS": "Other",
    "FERRARI": "Other",
    "MORGAN": "Other",
    "KYMCO": "Other",
    "LDV": "Other",
    "AUSTIN": "Other",
    "BENELLI": "Other",
    "MORRIS": "Other",
    "GENERIC": "Other",
    "MCLAREN": "Other",
    "RIEJU": "Other",
    "LAMBORGHINI": "Other",
    "KSR MOTO": "Other",
    "DUCATI": "Other",
    "ROYAL ENFIELD": "Other",
    "TVR": "Other",
    "ALVIS": "Other",
    "OPTARE": "Other",
    "LEVC": "Other",
    "MOTORINI": "Other",
    "ROLLS ROYCE": "Other",
    "ZONTES": "Other",
    "INDIAN": "Other",
    "AC": "Other",
    "CATERHAM": "Other",
    "WK": "Other",
    "MG-MOTOR UK": "Other",
    "DAIMLER": "Other",
    "MUTT": "Other",
    ".": "Other",
    "BULLIT": "Other",
    "MASSEY FERGUSON": "Other",
    "HARLEY DAVIDS": "Other",
    "Royal Enfield": "Other",
    "ZERO MOTORCYC": "Other",
    "INSURANCE GROUPS": "Other",
    "HYMER": "Other",
    "FENDT": "Other",
    "DENNIS": "Other",
    "London Electric Vehicle Company": "Other",
    "MGB": "Other",
    "BRP": "Other",
    "REVA": "Other",
    "HUSQVARNA": "Other",
    "PILOTE": "Other",
    "WESTFIELD": "Other",
    "LEYLAND": "Other",
    "MASH": "Other",
    "Kawasaki": "Other",
    "MONDIAL": "Other",
    "Other": "Other",
    "TX VISTA": "Other",
    "ALPINA": "Other",
    "LIFAN HONGDA": "Other",
    "TX VISTA COMFORT PLUS ": "Other",
    "BRIXTON": "Other",
    "Lexmoto": "Other",
    "ORA": "Other",
    "ALPINE": "Other",
    "COLEMAN MILNE": "Other",
    "Yamaha": "Other",
    "WANGYE": "Other",
    "PROTON": "Other",
    "Bluroc Motorcycles": "Other",
    "Maxus": "Other",
    "Lunar Landstar": "Other",
    "HARTFORD": "Other",
    "HYOSUNG": "Other",
    "JIANSHE": "Other",
    "SMITHS": "Other",
    "BAOTIAN": "Other",
    "HERALD": "Other",
    "LEVC TX": "Other",
    "Dayi Motor": "Other",
    "ANT": "Other",
    "VMOTO": "Other",
    "BOND": "Other",
    "HONGDOU": "Other",
    "SUPERBYKE": "Other",
    "WOLSELEY": "Other",
    "ROYAL ALLOY": "Other",
    "QINGQI": "Other",
    "Askoll": "Other",
    "OPEL": "Other",
    "RELIANT": "Other",
    "GENESIS": "Other",
    "L.R. D-TYPE REPLICA ": "Other",
    "MOTO GUZZI": "Other",
    "CADILLAC": "Other",
}
for k, v in manufacturer_mappings.items():
    df = df.withColumn(
        "TP_ManufacturerDescription",
        when(col("TP_ManufacturerDescription") == k, v).otherwise(
            col("TP_ManufacturerDescription")
        ),
    )
# fill remaining low volume levels as 'Other' - initially didnt need to do this but have reran with wider date range so new low volume values found
df = df.withColumn(
    "TP_ManufacturerDescription",
    when(
        ~col("TP_ManufacturerDescription").isin(list(manufacturer_mappings.values())),
        "Other",
    ).otherwise(col("TP_ManufacturerDescription")),
)

# tp veh age - some very high values around 2000 likely due to missing purchase date
# so introduce a floor at 0, a cap at 25 but map the erroneous values >100 to the average below the cap rather than the cap
df = (
    df.withColumn(
        "TP_VehicleAge",
        when(col("TP_VehicleAge") < 0, 0).otherwise(col("TP_VehicleAge")),
    )
    .withColumn(
        "TP_VehicleAge",
        when(col("TP_VehicleAge") > 100, 7.3).otherwise(col("TP_VehicleAge")),
    )
    .withColumn(
        "TP_VehicleAge",
        when(col("TP_VehicleAge") > 25, 25).otherwise(col("TP_VehicleAge")),
    )
)

# fix some dtypes:
int_cols = ["CurrentMileage", "ImpactSpeedRange_int"]
for c in int_cols:
    df = df.withColumn(c, col(c).cast(IntegerType()))

bool_cols = [
    "IsRightHandDrive",
    "NotOnMID",
    "VehicleUnattended"]
for c in bool_cols:
    df = df.withColumn(c, col(c).cast(BooleanType()).cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Feature Selection

# COMMAND ----------

key_cols = [
    "ClaimVersion",
    "ClaimItemID",
    "ClaimItemKey",
    "ClaimNumber",
    "EventTimestamp",
    "NotificationDate",
    "piikey",
    "PMCClaimVersionKey",
    "ReportedDate",
    "IncidentDate",
]

# predictive modelling_features
predictive_cols = [
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

df = df.select(key_cols + predictive_cols).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scoring

# COMMAND ----------

# MAGIC %md
# MAGIC ##### TPPD Cost Model Score

# COMMAND ----------

#load model
model = mlflow.sklearn.load_model('runs:/a88a2a66c63043d5aa710a8457af4246/model')

df_cap = df.copy()
df_cap['CaptureSuccess_AdrienneVersion'] = True

df_nocap = df.copy()
df_nocap['CaptureSuccess_AdrienneVersion'] = False


df['TPPD_Pred_Capture'] = model.predict(df_cap)
df['TPPD_Pred_No_Capture'] = model.predict(df_nocap)

#predicted value
df['Capture_Benefit'] = df.TPPD_Pred_No_Capture - df.TPPD_Pred_Capture

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Quantile Adjustments

# COMMAND ----------

#read in TPPD cost quantile bin edges
bin_edges = pd.read_csv("/Workspace/Users/harry.bjarnason@first-central.com/03_Projects/TPC_ThirdPartyCapture/2024_07_TPC_Capture_Benefit/artefacts/TPPD_Pred_No_Cap_Quantile_Bin_Edges_Dedupe_Fix.csv", index_col='Unnamed: 0')

#read in quantile adjustment multiplier
quantile_benefit_multiplier = pd.read_csv("/Workspace/Users/harry.bjarnason@first-central.com/03_Projects/TPC_ThirdPartyCapture/2024_07_TPC_Capture_Benefit/artefacts/quant_benefit_adjustments_dedupe_fix.csv", index_col='Unnamed: 0').drop('INC_TOT_TPPD_QCUT', axis=1)

#get TPPD cost quantile
#now use bin edges to get quantile for each case - we could have just used the qcut directly for this but wanted to illustrate how this would work in deployment
df['TPPD_Pred_No_Capture_Quantile'] = pd.cut(df['TPPD_Pred_No_Capture'], bins=bin_edges['0'].values, labels=False, include_lowest=True)

#now join on quantile multiplier to the bin edge quantile
df = df.merge(quantile_benefit_multiplier, left_on='TPPD_Pred_No_Capture_Quantile', right_on='INC_TOT_TPPD_QCUT_INT', how='inner')

#make adjustment to base benefit prediction
df['Capture_Benefit_Adjusted'] = df.Capture_Benefit * df.quant_benefit_multiplier

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Score Categorisation

# COMMAND ----------

#read in categorisation boundaries
boundaries = pd.read_csv("/Workspace/Users/harry.bjarnason@first-central.com/03_Projects/TPC_ThirdPartyCapture/2024_07_TPC_Capture_Benefit/artefacts/Capture_Benefit_Prioritisation_Thresholds_Dedupe_Fix.csv", index_col='Unnamed: 0')['0']

#use to assign bronze / silver / gold
df['Benefit_Priority'] = 'Silver'
df.loc[df.Capture_Benefit_Adjusted>boundaries[2], 'Benefit_Priority'] = 'Gold'
df.loc[df.Capture_Benefit_Adjusted<boundaries[1], 'Benefit_Priority'] = 'Bronze'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Logs

# COMMAND ----------

#read in existing results log
existing_results_log = pd.read_csv('/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/interim_deployment/tpc_interim_deployment_results_log.csv')

#append this run's scores to the full score log, with current datetime 
df_results_log = df[['ClaimNumber', 'NotificationDate', 'IncidentDate', 'TPPD_Pred_No_Capture', 'TPPD_Pred_Capture', 'Capture_Benefit', 'Capture_Benefit_Adjusted', 'quant_benefit_multiplier', 'TPPD_Pred_No_Capture_Quantile', 'Benefit_Priority']].copy()
df_results_log['Run_Datetime'] = now

existing_results_log = pd.concat([existing_results_log, df_results_log])

existing_results_log.to_csv('/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/interim_deployment/tpc_interim_deployment_results_log.csv', index=False)

# COMMAND ----------

#open most recently sent log of gold claims
df_gold_log = pd.read_csv('/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/interim_deployment/tpc_interim_deployment_gold_log.csv')

#check if any claims on log have been downgraded in most recent run and remove from gold log if so
downgrades = df_results_log[(df_results_log.Benefit_Priority!='Gold') & (df_results_log.ClaimNumber.isin(df_gold_log.ClaimNumber))]['ClaimNumber']
df_gold_log = df_gold_log[~df_gold_log.ClaimNumber.isin(downgrades)]

#check if any new gold claims scored in most recent run and add to gold log if so
new_golds = df_results_log[(df_results_log.Benefit_Priority=='Gold') & 
                           (~df_results_log.ClaimNumber.isin(df_gold_log.ClaimNumber))]
new_golds = new_golds[['ClaimNumber', 'NotificationDate', 'IncidentDate', 'Benefit_Priority']]
df_gold_log = pd.concat([df_gold_log, new_golds])

#remove any claims from gold log now out of scope (notified >14 days ago)
df_gold_log = df_gold_log[(dt.datetime.today() - pd.to_datetime(df_gold_log['NotificationDate'])).dt.days<=14]

#save off gold log
df_gold_log.to_csv('/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/interim_deployment/tpc_interim_deployment_gold_log.csv', index=False)

# COMMAND ----------

#finally can overwrite run log now that run was successful 
run_log.to_csv('/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/interim_deployment/tpc_interim_deployment_run_log.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Email Distribution

# COMMAND ----------

#create message text
msg_text = 'Please see latest gold case log attached\n\nBest regards,\nData Science Team\ndatascienceteam@1stcentral.co.uk'
print(msg_text)

#config email server info
server = smtplib.SMTP('smtp.office365.com', 587)
server.ehlo()
server.starttls()
server.login("data-lake@first-central.com", "XAib6jzfCzY1vj9seeLY")
sender = "data-lake@first-central.com"
recipients = 'Harry.Bjarnason@first-central.com'

#craft message
msg = MIMEMultipart()
msg['Subject'] = "TPC Gold Cases"
msg['From'] = sender
msg['To'] = recipients
msg.attach(MIMEText(msg_text))

#convert dataFrame to csv in memory
csv_buffer = StringIO()
df_gold_log.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

# Generate the filename with timestamp
filename = f"gold_log_{now.replace(' ', '_').replace(':', '_').replace('-', '_')}.csv"

#attach file
attachment = MIMEApplication(csv_data, Name=filename)
attachment['Content-Disposition'] = f'attachment; filename="{filename}"'
msg.attach(attachment)

# send the email
server.sendmail(msg['From'], msg['To'].split(","), msg.as_string())
server.quit()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quality Checks

# COMMAND ----------

df['Capture_Benefit_Adjusted'].hist(bins=50)

# COMMAND ----------

df.isna().sum().sort_values(ascending=False).head(50) / len(df)

# COMMAND ----------

df['Benefit_Priority'].value_counts() / len(df)

# COMMAND ----------

df.groupby('Benefit_Priority')['Capture_Benefit_Adjusted'].mean()

# COMMAND ----------

df.groupby('Benefit_Priority')['Capture_Benefit_Adjusted'].min()

# COMMAND ----------

import matplotlib.pyplot as plt
plt.scatter(df['Capture_Benefit_Adjusted'], df['TPPD_Pred_No_Capture'],  alpha=0.1)

# COMMAND ----------

plt.scatter(df['Capture_Benefit_Adjusted'], df['TP_VehicleAge'], alpha=0.1)

# COMMAND ----------

plt.scatter(df['Capture_Benefit_Adjusted'], df['FP_DamageSev_Total'], alpha=0.1)