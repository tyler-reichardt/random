# Databricks notebook source
# MAGIC %md
# MAGIC ###### Author: Harry Bjarnason
# MAGIC ###### Notebook Purpose: Import the build functions created in 01_TPC_Build_Functions and run in chunks to alleviate any memory issues or crashed 

# COMMAND ----------

# MAGIC %run "/Workspace/Users/harry.bjarnason@first-central.com/Projects/TPC Third Party Capture/2024_07_TPC_Capture_Benefit/deployment/CaptureBenefit/01_TPC_Build_Functions" 

# COMMAND ----------

tpc_capture_benefit_main(
    start_date="2021-01-01",
    end_date="2021-03-01",
    period_splits=2,
    write_path="/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/",
    sas_claim_summaries_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_SUMMARIES_20240801.csv",
    sas_claim_trans_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_TRANSACTIONS_20240801.csv",
)

# COMMAND ----------

tpc_capture_benefit_main(
    start_date="2021-03-01",
    end_date="2021-06-01",
    period_splits=3,
    write_path="/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/",
    sas_claim_summaries_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_SUMMARIES_20240801.csv",
    sas_claim_trans_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_TRANSACTIONS_20240801.csv",
)

# COMMAND ----------

tpc_capture_benefit_main(
    start_date="2021-06-01",
    end_date="2022-01-01",
    period_splits=7,
    write_path="/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/",
    sas_claim_summaries_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_SUMMARIES_20240801.csv",
    sas_claim_trans_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_TRANSACTIONS_20240801.csv",
)

# COMMAND ----------

tpc_capture_benefit_main(
    start_date="2022-01-01",
    end_date="2022-07-01",
    period_splits=7,
    write_path="/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/",
    sas_claim_summaries_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_SUMMARIES_20240801.csv",
    sas_claim_trans_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_TRANSACTIONS_20240801.csv",
)

# COMMAND ----------

tpc_capture_benefit_main(
    start_date="2022-07-01",
    end_date="2023-01-01",
    period_splits=7,
    write_path="/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/",
    sas_claim_summaries_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_SUMMARIES_20240801.csv",
    sas_claim_trans_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_TRANSACTIONS_20240801.csv",
)

# COMMAND ----------

tpc_capture_benefit_main(
    start_date="2023-01-01",
    end_date="2023-07-01",
    period_splits=7,
    write_path="/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/",
    sas_claim_summaries_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_SUMMARIES_20240801.csv",
    sas_claim_trans_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_TRANSACTIONS_20240801.csv",
)

# COMMAND ----------

tpc_capture_benefit_main(
    start_date="2023-07-01",
    end_date="2024-01-01",
    period_splits=7,
    write_path="/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/",
    sas_claim_summaries_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_SUMMARIES_20240801.csv",
    sas_claim_trans_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_TRANSACTIONS_20240801.csv",
)

# COMMAND ----------

tpc_capture_benefit_main(
    start_date="2024-01-01",
    end_date="2024-08-01",
    period_splits=7,
    write_path="/mnt/datalake/users/HarryBjarnason/2024_04_TPC_Capture_Benefit/CaptureBenefitBuild_v2/",
    sas_claim_summaries_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_SUMMARIES_20240801.csv",
    sas_claim_trans_path="/mnt/datalake/users/HarryBjarnason/2024_02 TPC/SAS_Extracts/SAS_CLAIM_TRANSACTIONS_20240801.csv",
)