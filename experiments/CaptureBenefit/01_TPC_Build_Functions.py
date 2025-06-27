# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd
from functools import reduce
from operator import add
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC #### SQL Extract

# COMMAND ----------

def tpc_sql_extracts(start_date, end_date):
    # requires case sensitive sql to interperet schema without error
    spark.conf.set("spark.sql.caseSensitive", "true")
    # create list of dates which we are interested in
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")
    # specify general path 
    path_general = "/mnt/datalake/general/live/transcache/claimseventpayload/"
    # now create list of paths to read JSON from
    paths_general = [path_general + x + "/*" for x in dates]
    # read in required fields from general claims events:
    df_general = spark.read.json(paths_general)
    # create views, apply query and save
    df_general.createOrReplaceTempView("vwEventsGeneral")

    df_general_queried = spark.sql(
        """
    SELECT 
    body.ComponentEvent.EventTimestamp as EventTimestamp,
    body.ComponentEvent.EventType as Type,
    convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Number as ClaimNumber,
    convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.SelectedClaimVersion as ClaimVersion,
    encodedvalues.PiiKey as piikey,
    ClaimItemKey,
    ClaimItemData.ClaimItemType as ClaimItemType,
    ClaimItemData.FirstParty as firstParty,
    --instigator:
    convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Instigator.InstigatorType as InstigatorType,
    --claim version info
    PMCClaimVersionKey,
    PMCClaimVersionData.BusinessProcess as BusinessProcess,
    PMCClaimVersionData.CreatedDate as CreatedDate,
    PMCClaimVersionData.ChangedDate as ChangedDate,
    PMCClaimVersionData.Incident.ReportedDate as ReportedDate,
    PMCClaimVersionData.Incident.LiabilityDetails.LiabilityDecision as Liability,
    PMCClaimVersionData.Incident.Cause.Data.Description as IncidentCause,
    PMCClaimVersionData.Incident.SubCause.Data.Description as IncidentSubCause,
    PMCClaimVersionData.Incident.MultiplePartiesInvolved as MultiplePartiesInvolved,
    PMCClaimVersionData.Incident.NotificationMethod as NotificationMethod,
    size(PMCClaimVersionData.Items.ClaimItem) as CountClaimItems,
    --tppro:
    ClaimItemData.TPPros.TPPro.InterventionClaimType as InterventionClaimType,
    ClaimItemData.TPPros.TPPro.InterventionLetterSentDate as InterventionLetterSentDate,
    ClaimItemData.TPPros.TPPro.InterventionOutcome as InterventionOutcome,
    ClaimItemData.TPPros.TPPro.OutboundCommsSentDate as OutboundCommsSentDate,
    ClaimItemData.TPPros.TPPro.RejectionReason as RejectionReason,
    ClaimItemData.TPPros.TPPro.TPProUnsuitabilityReason as TPProUnsuitabilityReason,
    ClaimItemData.TPProCallLogs.TPProCallLog as TPProCallLog,
    --contact details
    ClaimItemData.Guid as ClaimItemID,
    ClaimItemData.Claimant.Guid as PersonGuid,
    ClaimItemData.Claimant.Id as PersonID,
    ClaimItemData.Claimant.Name,
    ClaimItemData.PositionStatus.OverrideReason.Data.Description as OverrideReason,
    --hire cost details and claim notes 
    ClaimItemData.Jobs.ClaimJob.ABICarClass as ABICarClass,
    ClaimItemData.Jobs.ClaimJob.AbiCarClass as ABICarClass2,
    ClaimItemData.Jobs.ClaimJob.Notes as Notes,
    ClaimItemData.HireProgression.CreditHireDetails.HireDetails.CarClass as HireCarClass
    FROM vwEventsGeneral
    -- explode so have row for every claim version and claim item 
    LATERAL VIEW posexplode(convertedvalues.payload.ClaimsEventPayload.Claim.PMCClaim.Versions.PMCClaimVersion) PMCClaimVersion as PMCClaimVersionKey, PMCClaimVersionData
    LATERAL VIEW posexplode(PMCClaimVersionData.Items.ClaimItem) ClaimItem as ClaimItemKey, ClaimItemData
"""
    )
    
    return df_general_queried

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preprocessing

# COMMAND ----------

def tpc_preprocessing(df):
    # define schema for tpprocall log json entry data as currently read in as string
    tppro_call_log_entry_schema = StructType(
        [
            StructField("CallType", StringType(), True),
            StructField("Touch", StringType(), True),
            StructField("CustomerAnswered", StringType(), True),
            StructField("CustomerCaptured", StringType(), True),
            StructField("NonCaptureReason", StringType(), True),
        ]
    )
    # the call log can have multiple entries  so wrap the above schema in an array type
    tppro_call_log_schema = ArrayType(tppro_call_log_entry_schema)
    # convert string to structured JSON using our defined schema
    df = df.withColumn(
        "TPProCallLogParsed", from_json(col("TPProCallLog"), tppro_call_log_schema)
    )
    # explode so row for every entry into tpprocalllog - need to union with df where tpprocalllog null as exploding drops those rows
    df.createOrReplaceTempView("vwCallLogList")
    df = spark.sql(
        """
  SELECT *,
  null as TPProCallLogKey,
  null as TPProCallLogData,
  null as TPProCallLogCallType,
  null as TPProCallLogTouch,
  null as TPProCallLogCustomerAnswered,
  null as TPProCallLogCustomerCaptured,
  null as TPProCallLogNonCaptureReason  
  FROM vwCallLogList
  WHERE TPProCallLogParsed is null
  UNION
  SELECT *,
  TPProCallLogData.CallType as TPProCallLogCallType,
  TPProCallLogData.Touch as TPProCallLogTouch,
  TPProCallLogData.CustomerAnswered as TPProCallLogCustomerAnswered,
  TPProCallLogData.CustomerCaptured as TPProCallLogCustomerCaptured,
  TPProCallLogData.NonCaptureReason as TPProCallLogNonCaptureReason
  FROM vwCallLogList
  LATERAL VIEW posexplode(TPProCallLogParsed) TPProCallLogParsed as TPProCallLogKey, TPProCallLogData
  WHERE TPProCallLogParsed is not null
  """
    )

    # fix some column dtypes
    df = (
        df.withColumn("EventTimestamp", df.EventTimestamp.cast(TimestampType()))
        .withColumn("ClaimVersion", df.ClaimVersion.cast(IntegerType()))
        .withColumn("CreatedDate", df.CreatedDate.cast(TimestampType()))
        .withColumn("ChangedDate", df.ChangedDate.cast(TimestampType()))
        .withColumn("ReportedDate", df.ReportedDate.cast(TimestampType()))
        .withColumn("TPProCallLogKey", df.TPProCallLogKey.cast(IntegerType()))
        .withColumn(
            "TPProCallLogCustomerAnswered",
            df.TPProCallLogCustomerAnswered.cast(BooleanType()).cast(IntegerType()),
        )
        .withColumn(
            "TPProCallLogCustomerCaptured",
            df.TPProCallLogCustomerCaptured.cast(BooleanType()).cast(IntegerType()),
        )
    )

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Row Filtering 1
# MAGIC For capture value modelling we want one row per third party vehicle that would be eligible to capture, whereas for capture propensity modelling we want one row per capture attempt, as time since FNOL will likely be an important factor.
# MAGIC
# MAGIC We may want to build two propensity models - one for a full capture, and one where we only capture the hire car? 
# MAGIC
# MAGIC Care must be taken to ensure the correct claim item and claim event versions are selected
# MAGIC
# MAGIC We will also consider whether we want to include capture attempts made within the first 15 mins of FNOL or not, as currently the team give these cases the absolute priority
# MAGIC
# MAGIC perhaps come back to this as requires a bit of thought for each model?

# COMMAND ----------

# filtering for capture benefit value
def tpc_capture_benefit_row_filter_1(df):
    # dedupe keep latest claim version and tppro call log to reduce computation requirement
    # will need to do this again during data cleaning once looking at all time intervals together
    wp_first_tppro = Window.partitionBy(["ClaimNumber", "ClaimItemID"]).orderBy(
        col("EventTimestamp").desc(),
        col("ClaimVersion").desc(),
        col("PMCClaimVersionKey").desc(),
        col("TPProCallLogKey").desc(),
    )
    df = (
        df.withColumn("rn", row_number().over(wp_first_tppro))
        .filter(col("rn") == 1)
        .drop("rn")
    )

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Enrichment Join
# MAGIC Here we can join on claim item and claim incident info from the daily job, and claim summary and itemised claim costs from SAS
# MAGIC
# MAGIC Code in hidden cell below:

# COMMAND ----------

# MAGIC %md
# MAGIC SAS Extract Code (for reference):
# MAGIC
# MAGIC Note that table names and dates would need updating. To ensure a good match with extracts from data lake worth using a date range +/- 1 year from dates used for datalake extraction
# MAGIC ```
# MAGIC proc sql;
# MAGIC 	create table sas_claim_transactions as
# MAGIC 	select 
# MAGIC 		CLM_NUMBER,
# MAGIC 		HOD_PARENT,
# MAGIC 		TRANS_TYPE,
# MAGIC 		TRANS_DATE,
# MAGIC 		HOD_DESC,
# MAGIC 		INCIDENT_TYPE_DESC,
# MAGIC 		NOTIFICATION_DT,
# MAGIC 		FAULT_IND,
# MAGIC 		INCIDENT_CAUSE_DESC,
# MAGIC 		PAID_TOT_TPPD,
# MAGIC 		RESERVE_TPPD
# MAGIC 	from FCCLMMPU.claimtransactions_fcl_20240731
# MAGIC 	where NOTIFICATION_DT between '01Jan2020'd and '01Aug2024'd;
# MAGIC quit;
# MAGIC
# MAGIC
# MAGIC proc sql;
# MAGIC 	create table sas_claim_summaries as
# MAGIC 	select 
# MAGIC 		POLICYNUMBER,
# MAGIC 		CLM_NUMBER,
# MAGIC 		INCIDENT_DT,
# MAGIC 		NOTIFICATION_DT,
# MAGIC 		PAID_REC_AD,
# MAGIC 		PAID_REC_FIRE,
# MAGIC 		PAID_REC_THEFT,
# MAGIC 		PAID_REC_TPI,
# MAGIC 		PAID_REC_TPPD,
# MAGIC 		PAID_TOT_AD,
# MAGIC 		PAID_TOT_FIRE,
# MAGIC 		PAID_TOT_THEFT,
# MAGIC 		PAID_TOT_TPI,
# MAGIC 		PAID_TOT_TPPD,
# MAGIC 		RESERVE_AD,
# MAGIC 		RESERVE_FIRE,
# MAGIC 		RESERVE_THEFT,
# MAGIC 		RESERVE_TPI,
# MAGIC 		RESERVE_TPPD,
# MAGIC 		RESERVE_REC_AD,
# MAGIC 		RESERVE_REC_FIRE,
# MAGIC 		RESERVE_REC_THEFT,
# MAGIC 		RESERVE_REC_TPI,
# MAGIC 		RESERVE_REC_TPPD,
# MAGIC 		INC_TOT_AD,
# MAGIC 		INC_TOT_FIRE,
# MAGIC 		INC_TOT_THEFT,
# MAGIC 		INC_TOT_TPI,
# MAGIC 		INC_TOT_TPPD,
# MAGIC 		NUM_CLM,
# MAGIC 		NUM_CLM_AD,
# MAGIC 		NUM_CLM_FIRE,
# MAGIC 		NUM_CLM_THEFT,
# MAGIC 		NUM_CLM_TPPD,
# MAGIC 		NUM_CLM_TPI,
# MAGIC 		STATUS,
# MAGIC 		FAULT_IND,
# MAGIC 		INCIDENT_CAUSE_DESC,
# MAGIC 		INCIDENT_POSTCODE,
# MAGIC 		BONUS_ALLOWANCE_DESCRIPTION
# MAGIC 	from FCCLMMPU.POLICYCLAIMSUMMARY_FCL_20240731
# MAGIC 	where NOTIFICATION_DT between '01Jan2020'd and '01Aug2024'd;
# MAGIC
# MAGIC quit;
# MAGIC
# MAGIC ```

# COMMAND ----------

def tpc_sas_join(df, sas_claim_summaries_path, sas_claim_transactions_path):
    # read sas claim summary
    df_sas_summary = spark.read.csv(sas_claim_summaries_path, header=True)
    # Add TPPD_STATUS to check whether there are any remaining TPPD reserves (this would mean this HOD is still developing)
    df_sas_summary = (
        df_sas_summary.withColumn(
            "TPPD_STATUS",
            when(
                (col("RESERVE_TPPD") == 0) & (col("RESERVE_REC_TPPD") == 0), "CLOSED"
            ).otherwise("OPEN"),
        )
        .withColumn("NOTIFICATION_YEAR", substring(col("NOTIFICATION_DT"), 6, 4))
        .withColumnRenamed("STATUS", "STATUS_SAS_SUMMARY")
    )  # so different name to status in claim incident
    df_sas_summary = df_sas_summary.drop_duplicates()

    # read sas claim transactions data
    df_sas_transactions = spark.read.csv(sas_claim_transactions_path, header=True)
    # group by to get sum of PAID_TOT_TPPD and RESERVE_TPPD for each tppd HOD_DESC (very granular) and CLM_NUMBER
    df_sas_transactions_tppd = (
        df_sas_transactions.filter(col("HOD_PARENT").isin("TP_PROPERTY", "TP_VEHICLE"))
        .groupBy("CLM_NUMBER", "HOD_DESC")
        .agg(
            sum("PAID_TOT_TPPD").alias("PAID_TOT_TPPD"),
            round(sum("RESERVE_TPPD"), 2).alias("RESERVE_TPPD"),
        )
    )

    # pivot and agg so have 2 cols for every sub TPPD divison of 'HOD_DESC' (_PAID AND _RESERVE)
    df_sas_transactions_tppd_pivot = (
        df_sas_transactions_tppd.groupBy(col("CLM_NUMBER"))
        .pivot("HOD_DESC")
        .agg(sum("PAID_TOT_TPPD").alias("PAID"), sum("RESERVE_TPPD").alias("RESERVE"))
        .na.fill(value=0)
    )

    # join two sas df's together and create intervention paid + reserve flag colmns
    df_sas_joined = (
        df_sas_summary.join(
            df_sas_transactions_tppd_pivot,
            on=[df_sas_summary.CLM_NUMBER == df_sas_transactions_tppd_pivot.CLM_NUMBER],
            how="left",
        )
        .drop(df_sas_transactions_tppd_pivot.CLM_NUMBER)
        .withColumn(
            "Has_Paid_Intervention",
            when(
                (col("TP Intervention_PAID") > 0)
                | (col("T P Intervention Mobility_PAID") > 0)
                | (col("T P Intervention Uninsured Loss_PAID") > 0),
                True,
            ).otherwise(False),
        )
        .withColumn(
            "Has_Reserve_Intervention",
            when(
                (col("TP Intervention_RESERVE") > 0)
                | (col("T P Intervention Mobility_RESERVE") > 0)
                | (col("T P Intervention Uninsured Loss_RESERVE") > 0),
                True,
            ).otherwise(False),
        )
        .withColumn(
            "Has_Reserve_Intervention_Mobility",
            when((col("T P Intervention Mobility_RESERVE") > 0), True).otherwise(False),
        )
        .withColumn(
            "Has_Reserve_Intervention_Repair",
            when((col("TP Intervention_RESERVE") > 0), True).otherwise(False),
        )
        .withColumn(
            "Has_Paid_Intervention_Mobility",
            when((col("T P Intervention Mobility_PAID") > 0), True).otherwise(False),
        )
        .withColumn(
            "Has_Paid_Intervention_Repair",
            when((col("TP Intervention_PAID") > 0), True).otherwise(False),
        )
    )

    # join SAS data onto lake extract
    df = df.join(
        df_sas_joined, on=[df_sas_joined.CLM_NUMBER == df.ClaimNumber], how="left"
    )

    # sort out some dtypes
    float_cols = [
        "T P Intervention Mobility_PAID",
        "RESERVE_TPPD",
        "RESERVE_REC_TPPD",
        "PAID_TOT_TPPD",
        "PAID_REC_TPPD",
    ]
    for c in float_cols:
        df = df.withColumn(c, col(c).cast("float"))

    return df

# COMMAND ----------

def daily_job_join(df):
    # get the claim incident data
    df_claim_incident = spark.read.parquet(
        "/mnt/datalake/users/LukeRowland/claimextracts/IncidentDetail/*"
    )
    df_claim_incident = (
        df_claim_incident.withColumn(
            "EventTimestamp", df_claim_incident.EventTimestamp.cast(TimestampType())
        )
        .withColumn(
            "NotificationDate",
            to_timestamp(df_claim_incident.NotificationDate, "yyyyMMdd HH:mm:ss"),
        )
        .withColumn(
            "ReportedDate",
            to_timestamp(df_claim_incident.ReportedDate, "yyyyMMdd HH:mm:ss"),
        )
        .withColumn(
            "IncidentDate",
            to_timestamp(df_claim_incident.IncidentDate, "yyyyMMdd HH:mm:ss"),
        )
    )

    # drop columns that exist already in SQL extracts
    df_claim_incident = df_claim_incident.drop(
        "EventTimestamp",
        "MultiplePartiesInvolved",
        "NotificationMethod",
        "ReportedDate",
    )
    # join daily job claim incident data
    df = df.join(
        df_claim_incident,
        on=["ClaimNumber", "ClaimVersion", "PMCClaimVersionKey", "piikey"],
        how="left",
    )

    # get daily job claim item data
    df_claim_item = spark.read.parquet(
        "/mnt/datalake/users/LukeRowland/claimextracts/ClaimItemDetail/*"
    )
    df_claim_item = df_claim_item.withColumn(
        "EventTimestamp", df_claim_item.EventTimestamp.cast(TimestampType())
    )

    # drop columns that exist already in SQL extracts
    df_claim_item = df_claim_item.drop(
        "ClaimItemType",
        "EventTimestamp",
        "Status",
        "PositionStatusDescription",
        "PositionStatusPropertyName",
        "Name",
        "PersonGuid",
        "PersonID",
        "OverrideReason",
        "ClaimItemKey",
    )

    # join daily job claim item data
    df = df.join(
        df_claim_item,
        on=[
            "ClaimNumber",
            "ClaimItemID",
            "ClaimVersion",
            "PMCClaimVersionKey",
            "piikey",
        ],
        how="left",
    )

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Cleaning / Feature Engineering

# COMMAND ----------

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

    # get count of different types of claim items per incident and whether capture attempted and successful
    df = df.withColumn(
        "CaptureSuccess_AdrienneVersion",
        when(
            (
                (col("InterventionOutcome") == "Captured")
                | (col("Has_Paid_Intervention") == True)
            ),
            1,
        ).otherwise(0),
    ).withColumn(
        "CaptureCallAttempted",
        when(col("TPProCallLogKey").isNull(), 0).otherwise(1),
    )

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

# COMMAND ----------

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
            col("INCIDENT_POSTCODE").isNotNull(),
            regexp_extract("INCIDENT_POSTCODE", r"^\D+", 0),
        ).otherwise(
            when(
                col("FP_KeptAtPostcode").isNotNull(),
                regexp_extract("FP_KeptAtPostcode", r"^\D+", 0),
            ).otherwise("ZZ")
        ),
    )

    return df

    # int_veh_cols = ['Doors', 'Seats', 'Value', 'EngineCapacity', 'DoorsOpen', 'BootOpens', 'EngineDamage', 'ExhaustDamaged', 'HailDamage', 'LightsDamaged', 'PanelGaps', 'PassengerAreaSubmerged', 'RadiatorDamaged', 'SharpEdges', 'WheelsDamaged', 'WingMirrorDamaged', 'Driveable']
    # for c in int_veh_cols:
    #      df = df.withColumn('FP_'+c, col('FP_'+c).cast('int')).withColumn('TP_'+c, col('TP_'+c).cast('int'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Row Filtering 2

# COMMAND ----------

# tpc_capture_benefit_row_filter_2
def tpc_capture_benefit_row_filter_2(df):
    # filter for only 1 TP (as otherwise muddys the financial data) and where TP is an allowed vehicle (to make sure only modelling cases eligible for capture)
    df = df.where(
        (col("TPMotorVehicle_Count") == 1) & (col("AllowedVehicle_Count") == 1)
    )

    # where we have made a TPPD payment, and where reserve is closed
    df = df.where(
        (col("RESERVE_TPPD") == 0)
        & (col("RESERVE_REC_TPPD") == 0)
        & (col("PAID_TOT_TPPD") > 0)
    )

    # where we are liable
    df = df.where(col("Liability") == "Fault")

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC #### TTR Model Scoring

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
        .drop("IncidentCauseDescription")
        .withColumnRenamed("INCIDENT_CAUSE_DESC", "IncidentCauseDescription")
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
    logged_model = "runs:/0827ba5e08b94d5ea6741236c35f2791/model"
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

# MAGIC %md
# MAGIC #### Complete Build Processes
# MAGIC
# MAGIC

# COMMAND ----------

# tpc_capture_benefit_data_build
def tpc_capture_benefit_main(
    start_date,
    end_date,
    period_splits,
    write_path,
    sas_claim_summaries_path,
    sas_claim_trans_path,
):
    # create range of dates splitting into roughly monthly periods
    date_range = pd.date_range(start_date, end_date, periods=period_splits)

    # run and save build for each period
    for i in range(len(date_range) - 1):
        df = tpc_sql_extracts(date_range[i].date(), date_range[i + 1].date())
        df = tpc_preprocessing(df)
        df = tpc_capture_benefit_row_filter_1(df)
        df = tpc_sas_join(df, sas_claim_summaries_path, sas_claim_trans_path)
        df = daily_job_join(df)
        df = tpc_clean_engineer_1(df)
        df = tpc_clean_engineer_2(df)
        df = tpc_capture_benefit_row_filter_2(df)
        df = ttr_prep_score(df)
        df.write.mode("Overwrite").parquet(write_path + "_" + str(date_range[i].date()))
        print("df_" + str(date_range[i].date()), "written successfully")
        df.unpersist()