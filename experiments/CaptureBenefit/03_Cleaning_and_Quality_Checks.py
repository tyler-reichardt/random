# Databricks notebook source
# MAGIC %md
# MAGIC ###### Author: Harry Bjarnason
# MAGIC ###### Notebook Purpose: Perform data cleaning and quality checks

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd

# COMMAND ----------

df = spark.read.parquet(
    "/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/*"
)
print(df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dedupe

# COMMAND ----------

# dedupe - keep latest claim version
# don't care about keeping tppro call logs for capture benefit
wp = Window.partitionBy(["ClaimNumber", 'ClaimItemID']).orderBy(
    col("EventTimestamp").desc(),
    col("ClaimVersion").desc(),
    col("PMCClaimVersionKey").desc(),
    col("TPProCallLogKey").desc(),
)
df = df.withColumn("rn", row_number().over(wp)).filter(col("rn") == 1).drop("rn")
print(df.count())

#still have some duplicate claim numbers where some incidents with more than 1 TP have slipped through so remove these duplicates completely 
no_dupes = df.groupBy('ClaimNumber').count().filter(col("count")==1)
df = df.join(no_dupes, on='ClaimNumber', how='inner')

print(df.count())
print(df.dropDuplicates(subset=["ClaimNumber"]).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Row Filtering

# COMMAND ----------

# further row filtering
df = (
    df.where(col("IncidentType") == "Accident")
    .where(col("IncidentCountryKey") == "GB")
    .where(col("FAULT_IND") == "Yes")
    .where(col("Liability") == "Fault")
)

print(df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean Columns

# COMMAND ----------

# for these features most of variance is only in 1 level so create indicators for that level and drop original columns to reduce noise
df = df.withColumn(
    "WetRoad_Ind", when(col("RoadConditions") == "Wet", 1).otherwise(0)
).withColumn(
    "RainyWeather_Ind", when(col("WeatherConditionsType") == "Rain", 1).otherwise(0)
)

# COMMAND ----------

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

# COMMAND ----------

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

# COMMAND ----------

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

# COMMAND ----------

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

# COMMAND ----------

# tp veh age - some very high values around 2000 likely due to missing purchase date

# check avg veh age excluding outliers (some values of 2000)
print(df.where(col("TP_VehicleAge") < 25).agg({"TP_VehicleAge": "avg"}).collect()[0][0])
print(
    df.where(col("TP_VehicleAge") > 100).agg({"TP_VehicleAge": "avg"}).collect()[0][0]
)
print(df.where(col("TP_VehicleAge") > 100).count())

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

# reprint checks
print(df.where(col("TP_VehicleAge") < 25).agg({"TP_VehicleAge": "avg"}).collect()[0][0])
print(
    df.where(col("TP_VehicleAge") > 100).agg({"TP_VehicleAge": "avg"}).collect()[0][0]
)
print(df.where(col("TP_VehicleAge") > 100).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Organise Columns

# COMMAND ----------

# create different columns lists to order, group and view more easily

# join keys and lookups
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

# target columns and sense checks
target_sense_cols = [
    "CaptureCallAttempted",
    "CaptureCallSuccess",
    "CaptureSuccess_AdrienneVersion",
    "Has_Paid_Intervention",
    "Has_Paid_Intervention_Mobility",
    "Has_Paid_Intervention_Repair",
    "INC_TOT_TPPD",
    "InterventionClaimType",
    "InterventionOutcome",
    "PAID_REC_TPPD",
    "PAID_TOT_TPPD",
    "RejectionReason",
    "TP_ClaimItemType",
    "T P Damage Property_PAID",
    "T P Intervention Fees_PAID",
    "T P Intervention Mobility_PAID",
    "T P Intervention Uninsured Loss_PAID",
    "TP Credit Hire_PAID",
    "TP Damage_PAID",
    "TP Fees_PAID",
    "TP Intervention_PAID",
    "TPMotorVehicle_Count",
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


df = df.select(key_cols+target_sense_cols+predictive_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### DTypes

# COMMAND ----------

# fix some dtypes:
int_cols = ["CurrentMileage", "ImpactSpeedRange_int"]
for c in int_cols:
    df = df.withColumn(c, col(c).cast(IntegerType()))

bool_cols = [
    "IsRightHandDrive",
    "NotOnMID",
    "VehicleUnattended",
    "Has_Paid_Intervention",
    "Has_Paid_Intervention_Mobility",
    "CaptureAttempted_AdrienneVersion",
    "CaptureCallAttempted",
    "CaptureCallSuccess",
    "CaptureSuccess_AdrienneVersion",
    "Has_Paid_Intervention_Repair",
]
for c in bool_cols:
    df = df.withColumn(c, col(c).cast(BooleanType()).cast(IntegerType()))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Missing Values 
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# drop the 200 rows which have over 1/3 values missing
df_pd = df.toPandas()
print(len(df_pd))

df_pd = df_pd[df_pd[predictive_cols].isna().sum(axis=1) / len(predictive_cols) < 0.33]

print(len(df_pd))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write

# COMMAND ----------

# save off cleaned df as csv

df_pd.to_csv(
    "/dbfs/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2_cleaned/capture_benefit_df_20240806_dupe_clm_no_fix.csv"
)