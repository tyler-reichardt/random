# Databricks notebook source
# MAGIC %sql
# MAGIC --- =============================================
# MAGIC --- TPPD Calculation - Complete SQL Implementation
# MAGIC --- Replicating all stored procedure logic using Unity Catalog Delta Tables
# MAGIC --- =============================================
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 1: READ BASE DELTA TABLES AND CREATE CATEGORY MAPPING
# MAGIC ---=============================================================================
# MAGIC WITH category_mapping AS (
# MAGIC   SELECT * FROM VALUES
# MAGIC     ('TP Intervention (Mobility)', 'TP_VEHICLE'),
# MAGIC     ('TP Intervention (Vehicle)', 'TP_VEHICLE'),
# MAGIC     ('TP Credit Hire (Vehicle)', 'TP_VEHICLE'),
# MAGIC     ('TP Authorised Hire (Vehicle)', 'TP_VEHICLE'),
# MAGIC     ('TP Fees (Vehicle)', 'TP_VEHICLE'),
# MAGIC     ('Ex-Gratia (Vehicle)', 'TP_PROPERTY'),
# MAGIC     ('TP Authorized Hire Vehicle', 'TP_VEHICLE'),
# MAGIC     ('Medical Expenses', 'TP_PROPERTY'),
# MAGIC     ('Legal Expenses', 'TP_PROPERTY'),
# MAGIC     ('TP Fees (Property)', 'TP_PROPERTY'),
# MAGIC     ('TP Credit Hire (Property)', 'TP_PROPERTY'),
# MAGIC     ('TP Authorised Hire (Property)', 'TP_PROPERTY'),
# MAGIC     ('TP Intervention (Property)', 'TP_PROPERTY'),
# MAGIC     ('Unknown', 'TP_PROPERTY'),
# MAGIC     ('AD Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC     ('Fire Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC     ('OD Reinsurance', 'TP_PROPERTY'),
# MAGIC     ('Theft Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC     ('TP Damage (Property)', 'TP_PROPERTY'),
# MAGIC     ('TP Authorized Hire Property', 'TP_PROPERTY'),
# MAGIC     ('TP Intervention (Uninsured Loss)', 'TP_PROPERTY'),
# MAGIC     ('TP Intervention (Fees)', 'TP_PROPERTY'),
# MAGIC     ('TP Damage (Vehicle)', 'TP_PROPERTY')
# MAGIC   AS t(payment_category, skyfire_parent)
# MAGIC ),
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 2: PAYMENT COMPONENT JOINS (replicating fact.usp_load_payments)
# MAGIC ---=============================================================================
# MAGIC
# MAGIC --- GUID Join CTE - exclude 00000000-0000-0000-0000-000000000000 and don't use event_identity
# MAGIC guid_join_cte AS (
# MAGIC   SELECT
# MAGIC     p.payment_id,
# MAGIC     p.payment_reference,
# MAGIC     p.payment_guid,
# MAGIC     p.claim_version,
# MAGIC     p.claim_version_item_index,
# MAGIC     p.event_identity,
# MAGIC     p.head_of_damage,
# MAGIC     p.status,
# MAGIC     p.type,
# MAGIC     p.transaction_date,
# MAGIC     p.claim_id,
# MAGIC     pc.payment_component_id,
# MAGIC     pc.payment_category,
# MAGIC     pc.net_amount,
# MAGIC     pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_guid = pc.payment_guid
# MAGIC     AND p.claim_version = pc.claim_version
# MAGIC   WHERE
# MAGIC     pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC ),
# MAGIC
# MAGIC --- Invoice Join CTE - join on payment_reference = invoice_number
# MAGIC invoice_join_cte AS (
# MAGIC   SELECT DISTINCT
# MAGIC     p.payment_id,
# MAGIC     p.payment_reference,
# MAGIC     p.payment_guid,
# MAGIC     p.claim_version,
# MAGIC     p.claim_version_item_index,
# MAGIC     p.event_identity,
# MAGIC     p.head_of_damage,
# MAGIC     p.status,
# MAGIC     p.type,
# MAGIC     p.transaction_date,
# MAGIC     p.claim_id,
# MAGIC     pc.payment_component_id,
# MAGIC     pc.payment_category,
# MAGIC     pc.net_amount,
# MAGIC     pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_reference = pc.invoice_number
# MAGIC     AND p.event_identity = pc.event_identity
# MAGIC     AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN guid_join_cte g
# MAGIC     ON g.payment_id = p.payment_id
# MAGIC     AND g.claim_version = p.claim_version
# MAGIC   WHERE
# MAGIC     g.payment_id IS NULL --- not picked up via the guid join
# MAGIC     AND p.payment_reference IS NOT NULL --- exclude null payment refs
# MAGIC ),
# MAGIC
# MAGIC --- Payment ID Join CTE
# MAGIC paymentid_join_cte AS (
# MAGIC   SELECT
# MAGIC     p.payment_id,
# MAGIC     p.payment_reference,
# MAGIC     p.payment_guid,
# MAGIC     p.claim_version,
# MAGIC     p.claim_version_item_index,
# MAGIC     p.event_identity,
# MAGIC     p.head_of_damage,
# MAGIC     p.status,
# MAGIC     p.type,
# MAGIC     p.transaction_date,
# MAGIC     p.claim_id,
# MAGIC     pc.payment_component_id,
# MAGIC     pc.payment_category,
# MAGIC     pc.net_amount,
# MAGIC     pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_id = pc.payment_id
# MAGIC     AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN (
# MAGIC     SELECT payment_id, payment_component_id, claim_version FROM guid_join_cte
# MAGIC     UNION
# MAGIC     SELECT payment_id, payment_component_id, claim_version FROM invoice_join_cte
# MAGIC   ) g
# MAGIC     ON g.payment_id = p.payment_id
# MAGIC     AND g.payment_component_id = pc.payment_component_id
# MAGIC     AND g.claim_version = p.claim_version
# MAGIC   WHERE
# MAGIC     pc.payment_id != 0 --- exclude 0 ids
# MAGIC     AND g.payment_id IS NULL --- not picked up via the guid or invoice join
# MAGIC ),
# MAGIC
# MAGIC --- Combined payments with skyfire_parent mapping
# MAGIC combined_payments_cte AS (
# MAGIC   SELECT * FROM guid_join_cte
# MAGIC   UNION
# MAGIC   SELECT * FROM invoice_join_cte
# MAGIC   UNION
# MAGIC   SELECT * FROM paymentid_join_cte
# MAGIC ),
# MAGIC
# MAGIC combined_payments_with_parent AS (
# MAGIC   SELECT
# MAGIC     cp.*,
# MAGIC     COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent
# MAGIC   FROM combined_payments_cte cp
# MAGIC   LEFT JOIN category_mapping cm
# MAGIC     ON cp.payment_category = cm.payment_category
# MAGIC ),
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 3: NET INCURRED CLAIMS FLAG (replicating dim.usp_load_payment)
# MAGIC ---=============================================================================
# MAGIC payments_with_flag AS (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     CASE
# MAGIC       WHEN TRIM(status) IN ('Paid','MarkedAsPaid') AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC       WHEN TRIM(status) IN ('Reversed','Approved') AND TRIM(type) = 'RecoveryReceipt' THEN -1
# MAGIC       WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC       WHEN TRIM(status) = 'Cancelled' AND TRIM(type) = 'RecoveryReceipt' THEN 0
# MAGIC       WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePaymentReversal' THEN -1
# MAGIC       ELSE 0
# MAGIC     END AS net_incurred_claims_flag
# MAGIC   FROM combined_payments_with_parent
# MAGIC ),
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 4: PAYMENT DEDUPLICATION (replicating fact.usp_load_payments)
# MAGIC ---=============================================================================
# MAGIC payments_base AS (
# MAGIC   SELECT *
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       *,
# MAGIC       ROW_NUMBER() OVER (
# MAGIC         PARTITION BY payment_guid, claim_id, status, gross_amount, payment_category
# MAGIC         ORDER BY transaction_date ASC
# MAGIC       ) AS rn
# MAGIC     FROM payments_with_flag
# MAGIC   )
# MAGIC   WHERE rn = 1
# MAGIC ),
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 5: CALCULATE PAID_TOT_TPPD (replicating payment stored procedures)
# MAGIC ---=============================================================================
# MAGIC
# MAGIC --- Recovery Payments Logic (from usp_load_claims_transaction_recovery_payments)
# MAGIC recovery_payments AS (
# MAGIC   SELECT
# MAGIC     claim_id,
# MAGIC     CASE
# MAGIC       WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE')
# MAGIC       THEN CASE WHEN TRIM(status) = 'Approved' THEN gross_amount * net_incurred_claims_flag ELSE gross_amount END
# MAGIC       ELSE 0
# MAGIC     END AS paid_tot_tppd
# MAGIC   FROM payments_base
# MAGIC   WHERE
# MAGIC     TRIM(type) = 'RecoveryReceipt'
# MAGIC     AND TRIM(status) IN ('Approved', 'Cancelled')
# MAGIC ),
# MAGIC
# MAGIC --- Reserve Payments Logic (from usp_load_claims_transaction_reserve_payments)
# MAGIC reserve_payments AS (
# MAGIC   SELECT
# MAGIC     claim_id,
# MAGIC     CASE
# MAGIC       WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE')
# MAGIC       THEN CASE WHEN TRIM(status) = 'Reversed' THEN gross_amount * -1 ELSE gross_amount * net_incurred_claims_flag END
# MAGIC       ELSE 0
# MAGIC     END AS paid_tot_tppd
# MAGIC   FROM payments_base
# MAGIC   WHERE
# MAGIC     TRIM(type) = 'ReservePayment'
# MAGIC     AND gross_amount <> 0
# MAGIC ),
# MAGIC
# MAGIC --- Aggregate paid_tot_tppd
# MAGIC paid_tppd_agg AS (
# MAGIC   SELECT
# MAGIC     claim_id,
# MAGIC     SUM(paid_tot_tppd) AS paid_tot_tppd
# MAGIC   FROM (
# MAGIC     SELECT claim_id, paid_tot_tppd FROM recovery_payments
# MAGIC     UNION ALL
# MAGIC     SELECT claim_id, paid_tot_tppd FROM reserve_payments
# MAGIC   )
# MAGIC   GROUP BY claim_id
# MAGIC ),
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 6: BUILD RESERVE BASE DATA (replicating fact.usp_load_reserve)
# MAGIC ---=============================================================================
# MAGIC
# MAGIC --- Reserves CTE with proper joins
# MAGIC reserves_cte AS (
# MAGIC  SELECT DISTINCT
# MAGIC    vi.claim_id,
# MAGIC    cv.claim_version_id,
# MAGIC    vi.claim_version_item_id,
# MAGIC    vi.claim_version_item_index,
# MAGIC    r.head_of_damage,
# MAGIC    r.category_data_description,
# MAGIC    r.reserve_guid,
# MAGIC    r.reserve_value,
# MAGIC    CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value,
# MAGIC    r.type,
# MAGIC    -- Add this to get the most recent event_enqueued_utc_time for each reserve
# MAGIC    r.event_enqueued_utc_time,
# MAGIC    ROW_NUMBER() OVER (
# MAGIC      PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id
# MAGIC      ORDER BY
# MAGIC        r.event_enqueued_utc_time DESC,  -- Most recent first
# MAGIC        CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END,
# MAGIC        r.reserve_value
# MAGIC    ) AS rn
# MAGIC  FROM prod_adp_certified.claim.reserve r
# MAGIC   INNER JOIN prod_adp_certified.claim.claim_version cv
# MAGIC     ON r.event_identity = cv.event_identity
# MAGIC   INNER JOIN prod_adp_certified.claim.claim_version_item vi
# MAGIC     ON vi.event_identity = r.event_identity
# MAGIC     AND vi.claim_version_item_index = r.claim_version_item_index
# MAGIC ),
# MAGIC
# MAGIC reserves_filtered AS (
# MAGIC   SELECT * FROM reserves_cte WHERE rn = 1
# MAGIC ),
# MAGIC ---=============================================================================
# MAGIC --- STEP 7: PAYMENT AGGREGATIONS FOR RESERVES (replicating fact.usp_load_reserve payments_cte)
# MAGIC ---=============================================================================
# MAGIC payments_agg_cte AS (
# MAGIC   SELECT
# MAGIC     pb.claim_id,
# MAGIC     pb.claim_version,
# MAGIC     pb.payment_category,
# MAGIC     COALESCE(p.claim_version_item_index, 0) as claim_version_item_index,
# MAGIC     SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount,
# MAGIC     SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount
# MAGIC   FROM payments_base pb
# MAGIC   LEFT JOIN prod_adp_certified.claim.payment p
# MAGIC     ON pb.payment_id = p.payment_id
# MAGIC     AND pb.payment_guid = p.payment_guid
# MAGIC   WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt')
# MAGIC   GROUP BY
# MAGIC     pb.claim_id,
# MAGIC     pb.claim_version,
# MAGIC     pb.payment_category,
# MAGIC     p.claim_version_item_index
# MAGIC ),
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 8: CALCULATE ADJUSTED RESERVE VALUES
# MAGIC ---=============================================================================
# MAGIC reserves_adjusted AS (
# MAGIC   SELECT DISTINCT
# MAGIC     r.claim_id,
# MAGIC     r.claim_version_id,
# MAGIC     r.head_of_damage,
# MAGIC     r.category_data_description,
# MAGIC     r.reserve_guid,
# MAGIC     r.expected_recovery_value,
# MAGIC     r.reserve_value,
# MAGIC     COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value,
# MAGIC     COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value
# MAGIC   FROM reserves_filtered r
# MAGIC   LEFT JOIN payments_agg_cte p
# MAGIC     ON p.claim_id = r.claim_id
# MAGIC     AND p.claim_version = r.claim_version_id --- CORRECTED: claim_version_id = claim_version
# MAGIC     AND p.payment_category = r.category_data_description
# MAGIC     AND p.claim_version_item_index = r.claim_version_item_index
# MAGIC ),
# MAGIC
# MAGIC reserves_with_parent AS (
# MAGIC   SELECT
# MAGIC     ra.*,
# MAGIC     COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent
# MAGIC   FROM reserves_adjusted ra
# MAGIC   LEFT JOIN category_mapping cm
# MAGIC     ON ra.category_data_description = cm.payment_category
# MAGIC ),
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 9: CALCULATE RESERVE_TPPD (replicating reserve movement logic)
# MAGIC ---=============================================================================
# MAGIC -- Enhanced movements_cte with additional temporal handling
# MAGIC movements_cte AS (
# MAGIC  WITH ordered_reserves AS (
# MAGIC    SELECT
# MAGIC      claim_id,
# MAGIC      claim_version_id,
# MAGIC      head_of_damage,
# MAGIC      category_data_description,
# MAGIC      reserve_guid,
# MAGIC      expected_recovery_value,
# MAGIC      reserve_value,
# MAGIC      finance_adjusted_reserve_value,
# MAGIC      finance_adjusted_recovery_value,
# MAGIC      skyfire_parent,
# MAGIC      -- Create a unique sequence for each reserve across versions
# MAGIC      DENSE_RANK() OVER (
# MAGIC        PARTITION BY claim_id, reserve_guid
# MAGIC        ORDER BY claim_version_id
# MAGIC      ) AS version_sequence
# MAGIC    FROM reserves_with_parent
# MAGIC  )
# MAGIC  SELECT
# MAGIC    claim_id,
# MAGIC    claim_version_id,
# MAGIC    head_of_damage,
# MAGIC    category_data_description,
# MAGIC    reserve_guid,
# MAGIC    expected_recovery_value,
# MAGIC    reserve_value,
# MAGIC    finance_adjusted_reserve_value,
# MAGIC    finance_adjusted_recovery_value,
# MAGIC    skyfire_parent,
# MAGIC    version_sequence,
# MAGIC    -- Calculate movements only when version sequence changes
# MAGIC    CASE
# MAGIC      WHEN version_sequence > 1
# MAGIC      THEN COALESCE(finance_adjusted_reserve_value, 0) -
# MAGIC           COALESCE(LAG(finance_adjusted_reserve_value, 1, 0) OVER (
# MAGIC             PARTITION BY claim_id, reserve_guid
# MAGIC             ORDER BY version_sequence
# MAGIC           ), 0)
# MAGIC      ELSE COALESCE(finance_adjusted_reserve_value, 0)  -- First version movement is the full value
# MAGIC    END AS reserve_movement,
# MAGIC    -- Calculate recovery reserve movement
# MAGIC    CASE
# MAGIC      WHEN version_sequence > 1
# MAGIC      THEN (
# MAGIC        COALESCE(finance_adjusted_recovery_value, 0) -
# MAGIC        COALESCE(LAG(finance_adjusted_recovery_value, 1, 0) OVER (
# MAGIC          PARTITION BY claim_id, reserve_guid
# MAGIC          ORDER BY version_sequence
# MAGIC        ), 0)
# MAGIC      ) * -1
# MAGIC      ELSE COALESCE(finance_adjusted_recovery_value, 0) * -1  -- First version
# MAGIC    END AS recovery_reserve_movement
# MAGIC  FROM ordered_reserves
# MAGIC ),
# MAGIC
# MAGIC --- Reserve movements
# MAGIC reserve_part AS (
# MAGIC   SELECT
# MAGIC     claim_id,
# MAGIC     CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN reserve_movement ELSE 0 END AS reserve_tppd
# MAGIC   FROM movements_cte
# MAGIC   WHERE reserve_movement <> 0 
# MAGIC ),
# MAGIC
# MAGIC --- Recovery reserve movements
# MAGIC recovery_reserve_part AS (
# MAGIC   SELECT
# MAGIC     claim_id,
# MAGIC     CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN recovery_reserve_movement ELSE 0 END AS reserve_tppd
# MAGIC   FROM movements_cte
# MAGIC   WHERE recovery_reserve_movement <> 0
# MAGIC ),
# MAGIC
# MAGIC --- Aggregate reserve_tppd
# MAGIC reserve_tppd_agg AS (
# MAGIC   SELECT
# MAGIC     claim_id,
# MAGIC     SUM(reserve_tppd) AS reserve_tppd
# MAGIC   FROM (
# MAGIC     SELECT claim_id, reserve_tppd FROM reserve_part
# MAGIC     UNION ALL
# MAGIC     SELECT claim_id, reserve_tppd FROM recovery_reserve_part
# MAGIC   )
# MAGIC   GROUP BY claim_id
# MAGIC ),
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 10: FINAL COMBINATION AND RESULT
# MAGIC ---=============================================================================
# MAGIC
# MAGIC --- Get base claim data with latest version
# MAGIC base_claim AS (
# MAGIC   SELECT
# MAGIC     c.claim_id,
# MAGIC     cv.claim_version_id,
# MAGIC     cv.event_identity,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY c.claim_id ORDER BY cv.claim_version_id DESC) AS rn
# MAGIC   FROM prod_adp_certified.claim.claim c
# MAGIC   INNER JOIN prod_adp_certified.claim.claim_version cv
# MAGIC     ON c.claim_id = cv.claim_id
# MAGIC ),
# MAGIC
# MAGIC base_claim_latest AS (
# MAGIC   SELECT
# MAGIC     claim_id,
# MAGIC     claim_version_id,
# MAGIC     event_identity
# MAGIC   FROM base_claim
# MAGIC   WHERE rn = 1
# MAGIC ),
# MAGIC
# MAGIC --- Final result with TPPD calculations
# MAGIC result_with_tppd AS (
# MAGIC   SELECT
# MAGIC     bc.claim_id,
# MAGIC     bc.claim_version_id,
# MAGIC     COALESCE(pt.paid_tot_tppd, 0) AS paid_tot_tppd,
# MAGIC     COALESCE(rt.reserve_tppd, 0) AS reserve_tppd,
# MAGIC     COALESCE(pt.paid_tot_tppd, 0) + COALESCE(rt.reserve_tppd, 0) AS inc_tot_tppd
# MAGIC   FROM base_claim_latest bc
# MAGIC   LEFT JOIN paid_tppd_agg pt
# MAGIC     ON bc.claim_id = pt.claim_id
# MAGIC   LEFT JOIN reserve_tppd_agg rt
# MAGIC     ON bc.claim_id = rt.claim_id
# MAGIC ),
# MAGIC
# MAGIC --- Add payment category for reference
# MAGIC payment_category_ref AS (
# MAGIC   SELECT DISTINCT
# MAGIC     claim_id,
# MAGIC     FIRST_VALUE(payment_category) OVER (PARTITION BY claim_id ORDER BY payment_id) AS payment_category
# MAGIC   FROM payments_base
# MAGIC )
# MAGIC ---=============================================================================
# MAGIC --- FINAL SELECT - RESULTS FOR TEST CASES
# MAGIC ---=============================================================================
# MAGIC SELECT
# MAGIC   r.claim_id,
# MAGIC   r.claim_version_id,
# MAGIC   pcr.payment_category,
# MAGIC   r.paid_tot_tppd,
# MAGIC   r.reserve_tppd,
# MAGIC   r.inc_tot_tppd
# MAGIC FROM result_with_tppd r
# MAGIC LEFT JOIN payment_category_ref pcr
# MAGIC   ON r.claim_id = pcr.claim_id
# MAGIC WHERE
# MAGIC   r.claim_id IN (1901162, 1894827, 1789444) --- Your test cases
# MAGIC ORDER BY
# MAGIC   r.claim_id;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC --- =============================================
# MAGIC --- TPPD Calculation - Debugging Implementation
# MAGIC --- Replicating logic and printing each intermediate table for selected claims.
# MAGIC --- =============================================
# MAGIC --- TEST CASE FILTER
# MAGIC --- claims to be analyzed: (1901162, 1894827, 1789444)
# MAGIC --- =============================================
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- STEP 1: CATEGORY MAPPING (No output, used by later steps)
# MAGIC ---=============================================================================
# MAGIC CREATE OR REPLACE TEMP VIEW category_mapping AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('TP Intervention (Mobility)', 'TP_VEHICLE'),
# MAGIC   ('TP Intervention (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('TP Credit Hire (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('TP Authorised Hire (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('TP Fees (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('Ex-Gratia (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('TP Authorized Hire Vehicle', 'TP_VEHICLE'),
# MAGIC   ('Medical Expenses', 'TP_PROPERTY'),
# MAGIC   ('Legal Expenses', 'TP_PROPERTY'),
# MAGIC   ('TP Fees (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Credit Hire (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Authorised Hire (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Intervention (Property)', 'TP_PROPERTY'),
# MAGIC   ('Unknown', 'TP_PROPERTY'),
# MAGIC   ('AD Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC   ('Fire Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC   ('OD Reinsurance', 'TP_PROPERTY'),
# MAGIC   ('Theft Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Damage (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Authorized Hire Property', 'TP_PROPERTY'),
# MAGIC   ('TP Intervention (Uninsured Loss)', 'TP_PROPERTY'),
# MAGIC   ('TP Intervention (Fees)', 'TP_PROPERTY'),
# MAGIC   ('TP Damage (Vehicle)', 'TP_PROPERTY')
# MAGIC AS t(payment_category, skyfire_parent);
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 2.1: guid_join_cte
# MAGIC ---=============================================================================
# MAGIC WITH guid_join_cte AS (
# MAGIC   SELECT
# MAGIC     p.payment_id, p.payment_reference, p.payment_guid, p.claim_version,
# MAGIC     p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status,
# MAGIC     p.type, p.transaction_date, p.claim_id, pc.payment_component_id,
# MAGIC     pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444)
# MAGIC     AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC )
# MAGIC SELECT 'guid_join_cte' AS source_cte, * FROM guid_join_cte;
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 2.2: invoice_join_cte
# MAGIC ---=============================================================================
# MAGIC WITH guid_join_cte AS (
# MAGIC   SELECT
# MAGIC     p.payment_id, p.payment_reference, p.payment_guid, p.claim_version,
# MAGIC     p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status,
# MAGIC     p.type, p.transaction_date, p.claim_id, pc.payment_component_id,
# MAGIC     pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444)
# MAGIC     AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC ),
# MAGIC invoice_join_cte AS (
# MAGIC   SELECT DISTINCT
# MAGIC     p.payment_id, p.payment_reference, p.payment_guid, p.claim_version,
# MAGIC     p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status,
# MAGIC     p.type, p.transaction_date, p.claim_id, pc.payment_component_id,
# MAGIC     pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_reference = pc.invoice_number
# MAGIC     AND p.event_identity = pc.event_identity
# MAGIC     AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN guid_join_cte g ON g.payment_id = p.payment_id AND g.claim_version = p.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444)
# MAGIC     AND g.payment_id IS NULL
# MAGIC     AND p.payment_reference IS NOT NULL
# MAGIC )
# MAGIC SELECT 'invoice_join_cte' AS source_cte, * FROM invoice_join_cte;
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 2.3: paymentid_join_cte
# MAGIC ---=============================================================================
# MAGIC WITH guid_join_cte AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT
# MAGIC       p.payment_id, p.claim_version, pc.payment_component_id
# MAGIC     FROM prod_adp_certified.claim.payment p
# MAGIC     INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC       ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC     WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC   )
# MAGIC ),
# MAGIC invoice_join_cte AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT DISTINCT
# MAGIC       p.payment_id, p.claim_version, pc.payment_component_id
# MAGIC     FROM prod_adp_certified.claim.payment p
# MAGIC     INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC       ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC     LEFT JOIN guid_join_cte g ON g.payment_id = p.payment_id AND g.claim_version = p.claim_version
# MAGIC     WHERE p.claim_id IN (1901162, 1894827, 1789444) AND g.payment_id IS NULL AND p.payment_reference IS NOT NULL
# MAGIC   )
# MAGIC ),
# MAGIC paymentid_join_cte AS (
# MAGIC   SELECT
# MAGIC     p.payment_id, p.payment_reference, p.payment_guid, p.claim_version,
# MAGIC     p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status,
# MAGIC     p.type, p.transaction_date, p.claim_id, pc.payment_component_id,
# MAGIC     pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN (
# MAGIC     SELECT payment_id, payment_component_id, claim_version FROM guid_join_cte
# MAGIC     UNION
# MAGIC     SELECT payment_id, payment_component_id, claim_version FROM invoice_join_cte
# MAGIC   ) g ON g.payment_id = p.payment_id AND g.payment_component_id = pc.payment_component_id AND g.claim_version = p.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND g.payment_id IS NULL
# MAGIC )
# MAGIC SELECT 'paymentid_join_cte' AS source_cte, * FROM paymentid_join_cte;
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 4: payments_base
# MAGIC ---=============================================================================
# MAGIC WITH guid_join_cte AS (
# MAGIC   SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC ),
# MAGIC invoice_join_cte AS (
# MAGIC   SELECT DISTINCT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN guid_join_cte g ON g.payment_id = p.payment_id AND g.claim_version = p.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444) AND g.payment_id IS NULL AND p.payment_reference IS NOT NULL
# MAGIC ),
# MAGIC paymentid_join_cte AS (
# MAGIC   SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN (SELECT payment_id, payment_component_id, claim_version FROM guid_join_cte UNION SELECT payment_id, payment_component_id, claim_version FROM invoice_join_cte) g
# MAGIC   ON g.payment_id = p.payment_id AND g.payment_component_id = pc.payment_component_id AND g.claim_version = p.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND g.payment_id IS NULL
# MAGIC ),
# MAGIC combined_payments_cte AS (
# MAGIC   SELECT * FROM guid_join_cte UNION SELECT * FROM invoice_join_cte UNION SELECT * FROM paymentid_join_cte
# MAGIC ),
# MAGIC combined_payments_with_parent AS (
# MAGIC   SELECT cp.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent
# MAGIC   FROM combined_payments_cte cp LEFT JOIN category_mapping cm ON cp.payment_category = cm.payment_category
# MAGIC ),
# MAGIC payments_with_flag AS (
# MAGIC   SELECT *, CASE
# MAGIC       WHEN TRIM(status) IN ('Paid','MarkedAsPaid') AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC       WHEN TRIM(status) IN ('Reversed','Approved') AND TRIM(type) = 'RecoveryReceipt' THEN -1
# MAGIC       WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC       WHEN TRIM(status) = 'Cancelled' AND TRIM(type) = 'RecoveryReceipt' THEN 0
# MAGIC       WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePaymentReversal' THEN -1
# MAGIC       ELSE 0
# MAGIC     END AS net_incurred_claims_flag
# MAGIC   FROM combined_payments_with_parent
# MAGIC ),
# MAGIC payments_base AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT *, ROW_NUMBER() OVER (PARTITION BY payment_guid, claim_id, status, gross_amount, payment_category ORDER BY transaction_date ASC) AS rn
# MAGIC     FROM payments_with_flag
# MAGIC   )
# MAGIC   WHERE rn = 1
# MAGIC )
# MAGIC SELECT 'payments_base' AS source_cte, * FROM payments_base;
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 9: movements_cte
# MAGIC ---=============================================================================
# MAGIC WITH payments_base AS (
# MAGIC   -- This CTE is a condensed version of all prior payment steps
# MAGIC   SELECT * FROM (
# MAGIC     SELECT *, ROW_NUMBER() OVER (PARTITION BY payment_guid, claim_id, status, gross_amount, payment_category ORDER BY transaction_date ASC) AS rn
# MAGIC     FROM (
# MAGIC       SELECT cp.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent,
# MAGIC       CASE
# MAGIC         WHEN TRIM(status) IN ('Paid','MarkedAsPaid') AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC         WHEN TRIM(status) IN ('Reversed','Approved') AND TRIM(type) = 'RecoveryReceipt' THEN -1
# MAGIC         WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC         WHEN TRIM(status) = 'Cancelled' AND TRIM(type) = 'RecoveryReceipt' THEN 0
# MAGIC         WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePaymentReversal' THEN -1
# MAGIC         ELSE 0
# MAGIC       END AS net_incurred_claims_flag
# MAGIC       FROM (
# MAGIC           SELECT * FROM (
# MAGIC               SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC               FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC               WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC           )
# MAGIC           UNION
# MAGIC           SELECT * FROM (
# MAGIC               SELECT DISTINCT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC               FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC               WHERE p.claim_id IN (1901162, 1894827, 1789444) AND p.payment_reference IS NOT NULL AND NOT EXISTS (
# MAGIC                 SELECT 1 FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id = p.claim_id AND p_guid.payment_id = p.payment_id AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC               )
# MAGIC           )
# MAGIC           UNION
# MAGIC           SELECT * FROM (
# MAGIC             SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC             FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC             WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND NOT EXISTS (
# MAGIC                 SELECT 1 FROM (
# MAGIC                     SELECT p_guid.payment_id, p_guid.claim_version, pc_guid.payment_component_id FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id IN (1901162, 1894827, 1789444) AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC                     UNION
# MAGIC                     SELECT p_inv.payment_id, p_inv.claim_version, pc_inv.payment_component_id FROM prod_adp_certified.claim.payment p_inv JOIN prod_adp_certified.claim.payment_component pc_inv ON p_inv.payment_reference = pc_inv.invoice_number AND p_inv.event_identity = pc_inv.event_identity AND p_inv.claim_version = pc_inv.claim_version WHERE p_inv.claim_id IN (1901162, 1894827, 1789444) AND p_inv.payment_reference IS NOT NULL
# MAGIC                 ) ex WHERE ex.payment_id = p.payment_id AND ex.claim_version = p.claim_version AND ex.payment_component_id = pc.payment_component_id
# MAGIC             )
# MAGIC           )
# MAGIC       ) cp LEFT JOIN category_mapping cm ON cp.payment_category = cm.payment_category
# MAGIC     )
# MAGIC   ) WHERE rn = 1
# MAGIC ),
# MAGIC reserves_cte AS (
# MAGIC   SELECT DISTINCT vi.claim_id, cv.claim_version_id, vi.claim_version_item_id, vi.claim_version_item_index, r.head_of_damage,
# MAGIC     r.category_data_description, r.reserve_guid, r.reserve_value, CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value, r.type,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id ORDER BY CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END, r.reserve_value) AS rn
# MAGIC   FROM prod_adp_certified.claim.reserve r
# MAGIC   INNER JOIN prod_adp_certified.claim.claim_version cv ON r.event_identity = cv.event_identity
# MAGIC   INNER JOIN prod_adp_certified.claim.claim_version_item vi ON vi.event_identity = r.event_identity AND vi.claim_version_item_index = r.claim_version_item_index
# MAGIC   WHERE vi.claim_id IN (1901162, 1894827, 1789444)
# MAGIC ),
# MAGIC reserves_filtered AS ( SELECT * FROM reserves_cte WHERE rn = 1 ),
# MAGIC payments_agg_cte AS (
# MAGIC   SELECT pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index,
# MAGIC     SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount,
# MAGIC     SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount
# MAGIC   FROM payments_base pb LEFT JOIN prod_adp_certified.claim.payment p ON pb.payment_id = p.payment_id AND pb.payment_guid = p.payment_guid
# MAGIC   WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt')
# MAGIC   GROUP BY pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index
# MAGIC ),
# MAGIC reserves_adjusted AS (
# MAGIC   SELECT DISTINCT r.claim_id, r.claim_version_id, r.head_of_damage, r.category_data_description, r.reserve_guid, r.expected_recovery_value, r.reserve_value,
# MAGIC     COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value,
# MAGIC     COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value
# MAGIC   FROM reserves_filtered r LEFT JOIN payments_agg_cte p ON p.claim_id = r.claim_id AND p.claim_version = r.claim_version_id AND p.payment_category = r.category_data_description AND p.claim_version_item_index = r.claim_version_item_index
# MAGIC ),
# MAGIC reserves_with_parent AS (
# MAGIC   SELECT ra.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent
# MAGIC   FROM reserves_adjusted ra LEFT JOIN category_mapping cm ON ra.category_data_description = cm.payment_category
# MAGIC ),
# MAGIC movements_cte AS (
# MAGIC   SELECT *,
# MAGIC     COALESCE(finance_adjusted_reserve_value, 0) - LAG(COALESCE(finance_adjusted_reserve_value, 0), 1, 0) OVER (PARTITION BY claim_id, reserve_guid ORDER BY claim_version_id) AS reserve_movement,
# MAGIC     (COALESCE(finance_adjusted_recovery_value, 0) - LAG(COALESCE(finance_adjusted_recovery_value, 0), 1, 0) OVER (PARTITION BY claim_id, reserve_guid ORDER BY claim_version_id)) * -1 AS recovery_reserve_movement
# MAGIC   FROM reserves_with_parent
# MAGIC )
# MAGIC SELECT 'movements_cte' AS source_cte, * FROM movements_cte;
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- FINAL SELECT - RESULTS FOR TEST CASES
# MAGIC ---=============================================================================
# MAGIC WITH category_mapping AS (
# MAGIC   SELECT * FROM VALUES
# MAGIC     ('TP Intervention (Mobility)', 'TP_VEHICLE'), ('TP Intervention (Vehicle)', 'TP_VEHICLE'), ('TP Credit Hire (Vehicle)', 'TP_VEHICLE'), ('TP Authorised Hire (Vehicle)', 'TP_VEHICLE'), ('TP Fees (Vehicle)', 'TP_VEHICLE'), ('Ex-Gratia (Vehicle)', 'TP_VEHICLE'), ('TP Authorized Hire Vehicle', 'TP_VEHICLE'), ('Medical Expenses', 'TP_PROPERTY'), ('Legal Expenses', 'TP_PROPERTY'), ('TP Fees (Property)', 'TP_PROPERTY'), ('TP Credit Hire (Property)', 'TP_PROPERTY'), ('TP Authorised Hire (Property)', 'TP_PROPERTY'), ('TP Intervention (Property)', 'TP_PROPERTY'), ('Unknown', 'TP_PROPERTY'), ('AD Ex-Gratia (Property)', 'TP_PROPERTY'), ('Fire Ex-Gratia (Property)', 'TP_PROPERTY'), ('OD Reinsurance', 'TP_PROPERTY'), ('Theft Ex-Gratia (Property)', 'TP_PROPERTY'), ('TP Damage (Property)', 'TP_PROPERTY'), ('TP Authorized Hire Property', 'TP_PROPERTY'), ('TP Intervention (Uninsured Loss)', 'TP_PROPERTY'), ('TP Intervention (Fees)', 'TP_PROPERTY'), ('TP Damage (Vehicle)', 'TP_PROPERTY')
# MAGIC   AS t(payment_category, skyfire_parent)
# MAGIC ),
# MAGIC payments_base AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT *, ROW_NUMBER() OVER (PARTITION BY payment_guid, claim_id, status, gross_amount, payment_category ORDER BY transaction_date ASC) AS rn
# MAGIC     FROM (
# MAGIC       SELECT cp.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent,
# MAGIC         CASE WHEN TRIM(status) IN ('Paid','MarkedAsPaid') AND TRIM(type) = 'ReservePayment' THEN 1 WHEN TRIM(status) IN ('Reversed','Approved') AND TRIM(type) = 'RecoveryReceipt' THEN -1 WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePayment' THEN 1 WHEN TRIM(status) = 'Cancelled' AND TRIM(type) = 'RecoveryReceipt' THEN 0 WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePaymentReversal' THEN -1 ELSE 0 END AS net_incurred_claims_flag
# MAGIC       FROM (
# MAGIC         SELECT * FROM (
# MAGIC           SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC         ) UNION
# MAGIC         SELECT * FROM (
# MAGIC           SELECT DISTINCT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND p.payment_reference IS NOT NULL AND NOT EXISTS (SELECT 1 FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id = p.claim_id AND p_guid.payment_id = p.payment_id AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000')
# MAGIC         ) UNION
# MAGIC         SELECT * FROM (
# MAGIC           SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND NOT EXISTS (SELECT 1 FROM (SELECT p_guid.payment_id, p_guid.claim_version, pc_guid.payment_component_id FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id IN (1901162, 1894827, 1789444) AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000' UNION SELECT p_inv.payment_id, p_inv.claim_version, pc_inv.payment_component_id FROM prod_adp_certified.claim.payment p_inv JOIN prod_adp_certified.claim.payment_component pc_inv ON p_inv.payment_reference = pc_inv.invoice_number AND p_inv.event_identity = pc_inv.event_identity AND p_inv.claim_version = pc_inv.claim_version WHERE p_inv.claim_id IN (1901162, 1894827, 1789444) AND p_inv.payment_reference IS NOT NULL) ex WHERE ex.payment_id = p.payment_id AND ex.claim_version = p.claim_version AND ex.payment_component_id = pc.payment_component_id)
# MAGIC         )
# MAGIC       ) cp
# MAGIC       LEFT JOIN category_mapping cm ON cp.payment_category = cm.payment_category
# MAGIC     )
# MAGIC   ) WHERE rn = 1
# MAGIC ),
# MAGIC paid_tppd_agg AS (
# MAGIC   SELECT claim_id, SUM(paid_tot_tppd) AS paid_tot_tppd FROM (
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN CASE WHEN TRIM(status) = 'Approved' THEN gross_amount * net_incurred_claims_flag ELSE gross_amount END ELSE 0 END AS paid_tot_tppd FROM payments_base WHERE TRIM(type) = 'RecoveryReceipt' AND TRIM(status) IN ('Approved', 'Cancelled')
# MAGIC     UNION ALL
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN CASE WHEN TRIM(status) = 'Reversed' THEN gross_amount * -1 ELSE gross_amount * net_incurred_claims_flag END ELSE 0 END AS paid_tot_tppd FROM payments_base WHERE TRIM(type) = 'ReservePayment' AND gross_amount <> 0
# MAGIC   ) GROUP BY claim_id
# MAGIC ),
# MAGIC reserve_tppd_agg AS (
# MAGIC   SELECT claim_id, SUM(reserve_tppd) AS reserve_tppd FROM (
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN reserve_movement ELSE 0 END AS reserve_tppd
# MAGIC     FROM (SELECT *, COALESCE(finance_adjusted_reserve_value, 0) - LAG(COALESCE(finance_adjusted_reserve_value, 0), 1, 0) OVER (PARTITION BY claim_id, reserve_guid ORDER BY claim_version_id) AS reserve_movement FROM (SELECT ra.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent FROM (SELECT DISTINCT r.claim_id, r.claim_version_id, r.head_of_damage, r.category_data_description, r.reserve_guid, r.expected_recovery_value, r.reserve_value, COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value, COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value FROM (SELECT * FROM (SELECT DISTINCT vi.claim_id, cv.claim_version_id, vi.claim_version_item_id, vi.claim_version_item_index, r.head_of_damage, r.category_data_description, r.reserve_guid, r.reserve_value, CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value, r.type, ROW_NUMBER() OVER (PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id ORDER BY CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END, r.reserve_value) AS rn FROM prod_adp_certified.claim.reserve r JOIN prod_adp_certified.claim.claim_version cv ON r.event_identity = cv.event_identity JOIN prod_adp_certified.claim.claim_version_item vi ON vi.event_identity = r.event_identity AND vi.claim_version_item_index = r.claim_version_item_index WHERE vi.claim_id IN (1901162, 1894827, 1789444)) WHERE rn = 1) r LEFT JOIN (SELECT pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index, SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount, SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount FROM payments_base pb LEFT JOIN prod_adp_certified.claim.payment p ON pb.payment_id = p.payment_id AND pb.payment_guid = p.payment_guid WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt') GROUP BY pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index) p ON p.claim_id = r.claim_id AND p.claim_version = r.claim_version_id AND p.payment_category = r.category_data_description AND p.claim_version_item_index = r.claim_version_item_index) ra LEFT JOIN category_mapping cm ON ra.category_data_description = cm.payment_category)) WHERE reserve_movement <> 0
# MAGIC     UNION ALL
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN recovery_reserve_movement ELSE 0 END AS reserve_tppd
# MAGIC     FROM (SELECT *, (COALESCE(finance_adjusted_recovery_value, 0) - LAG(COALESCE(finance_adjusted_recovery_value, 0), 1, 0) OVER (PARTITION BY claim_id, reserve_guid ORDER BY claim_version_id)) * -1 AS recovery_reserve_movement FROM (SELECT ra.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent FROM (SELECT DISTINCT r.claim_id, r.claim_version_id, r.head_of_damage, r.category_data_description, r.reserve_guid, r.expected_recovery_value, r.reserve_value, COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value, COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value FROM (SELECT * FROM (SELECT DISTINCT vi.claim_id, cv.claim_version_id, vi.claim_version_item_id, vi.claim_version_item_index, r.head_of_damage, r.category_data_description, r.reserve_guid, r.reserve_value, CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value, r.type, ROW_NUMBER() OVER (PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id ORDER BY CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END, r.reserve_value) AS rn FROM prod_adp_certified.claim.reserve r JOIN prod_adp_certified.claim.claim_version cv ON r.event_identity = cv.event_identity JOIN prod_adp_certified.claim.claim_version_item vi ON vi.event_identity = r.event_identity AND vi.claim_version_item_index = r.claim_version_item_index WHERE vi.claim_id IN (1901162, 1894827, 1789444)) WHERE rn = 1) r LEFT JOIN (SELECT pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index, SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount, SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount FROM payments_base pb LEFT JOIN prod_adp_certified.claim.payment p ON pb.payment_id = p.payment_id AND pb.payment_guid = p.payment_guid WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt') GROUP BY pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index) p ON p.claim_id = r.claim_id AND p.claim_version = r.claim_version_id AND p.payment_category = r.category_data_description AND p.claim_version_item_index = r.claim_version_item_index) ra LEFT JOIN category_mapping cm ON ra.category_data_description = cm.payment_category)) WHERE recovery_reserve_movement <> 0
# MAGIC   ) GROUP BY claim_id
# MAGIC ),
# MAGIC base_claim_latest AS (
# MAGIC   SELECT claim_id, claim_version_id, event_identity FROM (
# MAGIC     SELECT c.claim_id, cv.claim_version_id, cv.event_identity, ROW_NUMBER() OVER (PARTITION BY c.claim_id ORDER BY cv.claim_version_id DESC) AS rn
# MAGIC     FROM prod_adp_certified.claim.claim c INNER JOIN prod_adp_certified.claim.claim_version cv ON c.claim_id = cv.claim_id
# MAGIC     WHERE c.claim_id IN (1901162, 1894827, 1789444)
# MAGIC   ) WHERE rn = 1
# MAGIC ),
# MAGIC result_with_tppd AS (
# MAGIC   SELECT bc.claim_id, bc.claim_version_id, COALESCE(pt.paid_tot_tppd, 0) AS paid_tot_tppd, COALESCE(rt.reserve_tppd, 0) AS reserve_tppd,
# MAGIC     COALESCE(pt.paid_tot_tppd, 0) + COALESCE(rt.reserve_tppd, 0) AS inc_tot_tppd
# MAGIC   FROM base_claim_latest bc
# MAGIC   LEFT JOIN paid_tppd_agg pt ON bc.claim_id = pt.claim_id
# MAGIC   LEFT JOIN reserve_tppd_agg rt ON bc.claim_id = rt.claim_id
# MAGIC ),
# MAGIC payment_category_ref AS (
# MAGIC   SELECT DISTINCT claim_id, FIRST_VALUE(payment_category) OVER (PARTITION BY claim_id ORDER BY payment_id) AS payment_category
# MAGIC   FROM payments_base
# MAGIC )
# MAGIC SELECT
# MAGIC   r.claim_id,
# MAGIC   r.claim_version_id,
# MAGIC   pcr.payment_category,
# MAGIC   r.paid_tot_tppd,
# MAGIC   r.reserve_tppd,
# MAGIC   r.inc_tot_tppd
# MAGIC FROM result_with_tppd r
# MAGIC LEFT JOIN payment_category_ref pcr ON r.claim_id = pcr.claim_id
# MAGIC ORDER BY r.claim_id;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC ---=============================================================================
# MAGIC --- FINAL SELECT - RESULTS FOR TEST CASES
# MAGIC ---=============================================================================
# MAGIC WITH category_mapping AS (
# MAGIC   SELECT * FROM VALUES
# MAGIC     ('TP Intervention (Mobility)', 'TP_VEHICLE'), ('TP Intervention (Vehicle)', 'TP_VEHICLE'), ('TP Credit Hire (Vehicle)', 'TP_VEHICLE'), ('TP Authorised Hire (Vehicle)', 'TP_VEHICLE'), ('TP Fees (Vehicle)', 'TP_VEHICLE'), ('Ex-Gratia (Vehicle)', 'TP_VEHICLE'), ('TP Authorized Hire Vehicle', 'TP_VEHICLE'), ('Medical Expenses', 'TP_PROPERTY'), ('Legal Expenses', 'TP_PROPERTY'), ('TP Fees (Property)', 'TP_PROPERTY'), ('TP Credit Hire (Property)', 'TP_PROPERTY'), ('TP Authorised Hire (Property)', 'TP_PROPERTY'), ('TP Intervention (Property)', 'TP_PROPERTY'), ('Unknown', 'TP_PROPERTY'), ('AD Ex-Gratia (Property)', 'TP_PROPERTY'), ('Fire Ex-Gratia (Property)', 'TP_PROPERTY'), ('OD Reinsurance', 'TP_PROPERTY'), ('Theft Ex-Gratia (Property)', 'TP_PROPERTY'), ('TP Damage (Property)', 'TP_PROPERTY'), ('TP Authorized Hire Property', 'TP_PROPERTY'), ('TP Intervention (Uninsured Loss)', 'TP_PROPERTY'), ('TP Intervention (Fees)', 'TP_PROPERTY'), ('TP Damage (Vehicle)', 'TP_PROPERTY')
# MAGIC   AS t(payment_category, skyfire_parent)
# MAGIC ),
# MAGIC payments_base AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT *, ROW_NUMBER() OVER (PARTITION BY payment_guid, claim_id, status, gross_amount, payment_category ORDER BY transaction_date ASC) AS rn
# MAGIC     FROM (
# MAGIC       SELECT cp.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent,
# MAGIC         CASE WHEN TRIM(status) IN ('Paid','MarkedAsPaid') AND TRIM(type) = 'ReservePayment' THEN 1 WHEN TRIM(status) IN ('Reversed','Approved') AND TRIM(type) = 'RecoveryReceipt' THEN -1 WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePayment' THEN 1 WHEN TRIM(status) = 'Cancelled' AND TRIM(type) = 'RecoveryReceipt' THEN 0 WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePaymentReversal' THEN -1 ELSE 0 END AS net_incurred_claims_flag
# MAGIC       FROM (
# MAGIC         SELECT * FROM (
# MAGIC           SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC         ) UNION
# MAGIC         SELECT * FROM (
# MAGIC           SELECT DISTINCT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND p.payment_reference IS NOT NULL AND NOT EXISTS (SELECT 1 FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id = p.claim_id AND p_guid.payment_id = p.payment_id AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000')
# MAGIC         ) UNION
# MAGIC         SELECT * FROM (
# MAGIC           SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND NOT EXISTS (SELECT 1 FROM (SELECT p_guid.payment_id, p_guid.claim_version, pc_guid.payment_component_id FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id IN (1901162, 1894827, 1789444) AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000' UNION SELECT p_inv.payment_id, p_inv.claim_version, pc_inv.payment_component_id FROM prod_adp_certified.claim.payment p_inv JOIN prod_adp_certified.claim.payment_component pc_inv ON p_inv.payment_reference = pc_inv.invoice_number AND p_inv.event_identity = pc_inv.event_identity AND p_inv.claim_version = pc_inv.claim_version WHERE p_inv.claim_id IN (1901162, 1894827, 1789444) AND p_inv.payment_reference IS NOT NULL) ex WHERE ex.payment_id = p.payment_id AND ex.claim_version = p.claim_version AND ex.payment_component_id = pc.payment_component_id)
# MAGIC         )
# MAGIC       ) cp
# MAGIC       LEFT JOIN category_mapping cm ON cp.payment_category = cm.payment_category
# MAGIC     )
# MAGIC   ) WHERE rn = 1
# MAGIC ),
# MAGIC paid_tppd_agg AS (
# MAGIC   SELECT claim_id, SUM(paid_tot_tppd) AS paid_tot_tppd FROM (
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN CASE WHEN TRIM(status) = 'Approved' THEN gross_amount * net_incurred_claims_flag ELSE gross_amount END ELSE 0 END AS paid_tot_tppd FROM payments_base WHERE TRIM(type) = 'RecoveryReceipt' AND TRIM(status) IN ('Approved', 'Cancelled')
# MAGIC     UNION ALL
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN CASE WHEN TRIM(status) = 'Reversed' THEN gross_amount * -1 ELSE gross_amount * net_incurred_claims_flag END ELSE 0 END AS paid_tot_tppd FROM payments_base WHERE TRIM(type) = 'ReservePayment' AND gross_amount <> 0
# MAGIC   ) GROUP BY claim_id
# MAGIC ),
# MAGIC reserve_tppd_agg AS (
# MAGIC   SELECT claim_id, SUM(reserve_tppd) AS reserve_tppd FROM (
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN reserve_movement ELSE 0 END AS reserve_tppd
# MAGIC     FROM (SELECT *, COALESCE(finance_adjusted_reserve_value, 0) - LAG(COALESCE(finance_adjusted_reserve_value, 0), 1, 0) OVER (PARTITION BY claim_id, reserve_guid ORDER BY claim_version_id) AS reserve_movement FROM (SELECT ra.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent FROM (SELECT DISTINCT r.claim_id, r.claim_version_id, r.head_of_damage, r.category_data_description, r.reserve_guid, r.expected_recovery_value, r.reserve_value, COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value, COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value FROM (SELECT * FROM (SELECT DISTINCT vi.claim_id, cv.claim_version_id, vi.claim_version_item_id, vi.claim_version_item_index, r.head_of_damage, r.category_data_description, r.reserve_guid, r.reserve_value, CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value, r.type, ROW_NUMBER() OVER (PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id ORDER BY CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END, r.reserve_value) AS rn FROM prod_adp_certified.claim.reserve r JOIN prod_adp_certified.claim.claim_version cv ON r.event_identity = cv.event_identity JOIN prod_adp_certified.claim.claim_version_item vi ON vi.event_identity = r.event_identity AND vi.claim_version_item_index = r.claim_version_item_index WHERE vi.claim_id IN (1901162, 1894827, 1789444)) WHERE rn = 1) r LEFT JOIN (SELECT pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index, SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount, SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount FROM payments_base pb LEFT JOIN prod_adp_certified.claim.payment p ON pb.payment_id = p.payment_id AND pb.payment_guid = p.payment_guid WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt') GROUP BY pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index) p ON p.claim_id = r.claim_id AND p.claim_version = r.claim_version_id AND p.payment_category = r.category_data_description AND p.claim_version_item_index = r.claim_version_item_index) ra LEFT JOIN category_mapping cm ON ra.category_data_description = cm.payment_category)) WHERE reserve_movement <> 0
# MAGIC     UNION ALL
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN recovery_reserve_movement ELSE 0 END AS reserve_tppd
# MAGIC     FROM (SELECT *, (COALESCE(finance_adjusted_recovery_value, 0) - LAG(COALESCE(finance_adjusted_recovery_value, 0), 1, 0) OVER (PARTITION BY claim_id, reserve_guid ORDER BY claim_version_id)) * -1 AS recovery_reserve_movement FROM (SELECT ra.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent FROM (SELECT DISTINCT r.claim_id, r.claim_version_id, r.head_of_damage, r.category_data_description, r.reserve_guid, r.expected_recovery_value, r.reserve_value, COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value, COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value FROM (SELECT * FROM (SELECT DISTINCT vi.claim_id, cv.claim_version_id, vi.claim_version_item_id, vi.claim_version_item_index, r.head_of_damage, r.category_data_description, r.reserve_guid, r.reserve_value, CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value, r.type, ROW_NUMBER() OVER (PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id ORDER BY CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END, r.reserve_value) AS rn FROM prod_adp_certified.claim.reserve r JOIN prod_adp_certified.claim.claim_version cv ON r.event_identity = cv.event_identity JOIN prod_adp_certified.claim.claim_version_item vi ON vi.event_identity = r.event_identity AND vi.claim_version_item_index = r.claim_version_item_index WHERE vi.claim_id IN (1901162, 1894827, 1789444)) WHERE rn = 1) r LEFT JOIN (SELECT pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index, SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount, SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount FROM payments_base pb LEFT JOIN prod_adp_certified.claim.payment p ON pb.payment_id = p.payment_id AND pb.payment_guid = p.payment_guid WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt') GROUP BY pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index) p ON p.claim_id = r.claim_id AND p.claim_version = r.claim_version_id AND p.payment_category = r.category_data_description AND p.claim_version_item_index = r.claim_version_item_index) ra LEFT JOIN category_mapping cm ON ra.category_data_description = cm.payment_category)) WHERE recovery_reserve_movement <> 0
# MAGIC   ) GROUP BY claim_id
# MAGIC ),
# MAGIC base_claim_latest AS (
# MAGIC   SELECT claim_id, claim_version_id, event_identity FROM (
# MAGIC     SELECT c.claim_id, cv.claim_version_id, cv.event_identity, ROW_NUMBER() OVER (PARTITION BY c.claim_id ORDER BY cv.claim_version_id DESC) AS rn
# MAGIC     FROM prod_adp_certified.claim.claim c INNER JOIN prod_adp_certified.claim.claim_version cv ON c.claim_id = cv.claim_id
# MAGIC     WHERE c.claim_id IN (1901162, 1894827, 1789444)
# MAGIC   ) WHERE rn = 1
# MAGIC ),
# MAGIC result_with_tppd AS (
# MAGIC   SELECT bc.claim_id, bc.claim_version_id, COALESCE(pt.paid_tot_tppd, 0) AS paid_tot_tppd, COALESCE(rt.reserve_tppd, 0) AS reserve_tppd,
# MAGIC     COALESCE(pt.paid_tot_tppd, 0) + COALESCE(rt.reserve_tppd, 0) AS inc_tot_tppd
# MAGIC   FROM base_claim_latest bc
# MAGIC   LEFT JOIN paid_tppd_agg pt ON bc.claim_id = pt.claim_id
# MAGIC   LEFT JOIN reserve_tppd_agg rt ON bc.claim_id = rt.claim_id
# MAGIC ),
# MAGIC payment_category_ref AS (
# MAGIC   SELECT DISTINCT claim_id, FIRST_VALUE(payment_category) OVER (PARTITION BY claim_id ORDER BY payment_id) AS payment_category
# MAGIC   FROM payments_base
# MAGIC )
# MAGIC SELECT
# MAGIC   r.claim_id,
# MAGIC   r.claim_version_id,
# MAGIC   pcr.payment_category,
# MAGIC   r.paid_tot_tppd,
# MAGIC   r.reserve_tppd,
# MAGIC   r.inc_tot_tppd
# MAGIC FROM result_with_tppd r
# MAGIC LEFT JOIN payment_category_ref pcr ON r.claim_id = pcr.claim_id
# MAGIC ORDER BY r.claim_id;
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
col, when, coalesce, create_map, lit, trim,
sum as _sum, lag, first, row_number, max as _max
)
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
# =============================================================================
# STEP 1: INITIALIZE SPARK AND LOAD BASE TABLES (as per your original code)
# =============================================================================
spark = SparkSession.builder.getOrCreate()
# Load base tables
claim = spark.table("prod_adp_certified.claim.claim").select("claim_id", "policy_number", 
"claim_number")
claim_version = spark.table("prod_adp_certified.claim.claim_version").select("claim_id", 
"claim_number", "policy_number", "claim_version_id", "event_identity")
claim_version_item = spark.table("prod_adp_certified.claim.claim_version_item").select("claim_id", 
"claim_version_item_index", "claim_version_item_id", "event_identity")
payment = spark.table("prod_adp_certified.claim.payment").select("claim_id", 
"payment_reference", "payment_id", "payment_guid", "claim_version", 
"claim_version_item_index", "event_identity", "head_of_damage", "status", "type", 
"transaction_date")
payment_component = spark.table("prod_adp_certified.claim.payment_component").select("payment_id", "payment_component_id", 
"payment_guid", "claim_version", "invoice_number", "event_identity", "payment_category", 
"net_amount", "gross_amount")
reserve = spark.table("prod_adp_certified.claim.reserve").select("reserve_guid", 
"head_of_damage", "event_identity", "claim_version_item_index", 
"category_data_description", "reserve_value", "type", "expected_recovery_value")
# =============================================================================
# STEP 2: SETUP CATEGORY MAPPING (as per your original code)
# =============================================================================
category_to_parent = {
# TP_VEHICLE
"TP Intervention (Mobility)": "TP_VEHICLE",
"TP Intervention (Vehicle)": "TP_VEHICLE",
"TP Credit Hire (Vehicle)": "TP_VEHICLE",
"TP Authorised Hire (Vehicle)": "TP_VEHICLE",
"TP Fees (Vehicle)": "TP_VEHICLE",
"Ex-Gratia (Vehicle)": "TP_VEHICLE",
"TP Authorized Hire Vehicle": "TP_VEHICLE",
# TP_PROPERTY
"Medical Expenses": "TP_PROPERTY",
"Legal Expenses": "TP_PROPERTY",
"TP Fees (Property)": "TP_PROPERTY",
"TP Credit Hire (Property)": "TP_PROPERTY",
"TP Authorised Hire (Property)": "TP_PROPERTY",
"TP Intervention (Property)": "TP_PROPERTY",
"Unknown": "TP_PROPERTY",
"AD Ex-Gratia (Property)": "TP_PROPERTY",
"Fire Ex-Gratia (Property)": "TP_PROPERTY",
"OD Reinsurance": "TP_PROPERTY",
"Theft Ex-Gratia (Property)": "TP_PROPERTY",
"TP Damage (Property)": "TP_PROPERTY",
"TP Authorized Hire Property": "TP_PROPERTY",
"TP Intervention (Uninsured Loss)": "TP_PROPERTY",
"TP Intervention (Fees)": "TP_PROPERTY",
"TP Damage (Vehicle)": "TP_PROPERTY",
}
mapping_expr = create_map([lit(x) for i in category_to_parent.items() for x in i])

# COMMAND ----------



# COMMAND ----------

# =============================================================================
# STEP 3: BUILD DIM_PAYMENT LOGIC (replicate usp_load_payment exactly)
# =============================================================================
from pyspark.sql.functions import col, lit, when
from functools import reduce

# It's a best practice to alias your dataframes in complex joins
# to avoid column ambiguity.
p = payment.alias("p")
pc = payment_component.alias("pc")

print("=== Building Payment Component Joins (replicating fact.usp_load_payments) ===")

# Strategy 1: GUID Join
# Join where payment_guid is present and not the null GUID.
# This is the highest-priority join.
guid_join_cte = p.join(
    pc.filter(pc.payment_guid != "00000000-0000-0000-0000-000000000000"),
    [p.payment_guid == pc.payment_guid, p.claim_version == pc.claim_version],
    "inner"
).select(
    "p.*",  # Select all columns from the 'payment' table
    pc.payment_category,
    pc.payment_component_id,
    pc.net_amount,
    pc.gross_amount
)

# Strategy 2: Invoice Number Join
# Join where payment_reference matches invoice_number, excluding records
# already matched by the GUID join.
guid_exclusions_for_invoice = guid_join_cte.select(
    col("payment_id"),
    col("claim_version")
).distinct().alias("g_inv")

invoice_join_cte = p.join(
    pc,
    [p.payment_reference == pc.invoice_number,
     p.event_identity == pc.event_identity,
     p.claim_version == pc.claim_version],
    "inner"
).filter(
    p.payment_reference.isNotNull()
).join(
    guid_exclusions_for_invoice,
    [p.payment_id == col("g_inv.payment_id"),
     p.claim_version == col("g_inv.claim_version")],
    "left_anti"  # Keep only records from the main join that are NOT in the guid_exclusions.
).select(
    "p.*",
    pc.payment_category,
    pc.payment_component_id,
    pc.net_amount,
    pc.gross_amount
)

# Strategy 3: Payment ID Join
# The fallback join, using payment_id. It excludes records matched by either GUID or Invoice joins.
guid_exclusions_for_paymentid = guid_join_cte.select("payment_id", "payment_component_id", "claim_version")
invoice_exclusions_for_paymentid = invoice_join_cte.select("payment_id", "payment_component_id", "claim_version")

guid_and_invoice_exclusions = guid_exclusions_for_paymentid.unionByName(invoice_exclusions_for_paymentid).distinct().alias("ex_pay")

paymentid_join_cte = p.join(
    pc.filter(pc.payment_id != 0),
    [p.payment_id == pc.payment_id,
     p.claim_version == pc.claim_version],
    "inner"
).join(
    guid_and_invoice_exclusions,
    # Explicitly reference columns from 'p' and 'pc' which are in scope from the first join.
    [p.payment_id == col("ex_pay.payment_id"),
     p.claim_version == col("ex_pay.claim_version"),
     pc.payment_component_id == col("ex_pay.payment_component_id")],
    "left_anti"
).select(
    "p.*",
    pc.payment_category,
    pc.payment_component_id,
    pc.net_amount,
    pc.gross_amount
)

# =============================================================================
# Combine all three join strategies
# =============================================================================
print("=== Combining join strategies ===")

# Union the results from the three strategies.
# Using dropDuplicates() as a safeguard, although the exclusion logic should prevent overlaps.
combined_payments = guid_join_cte.unionByName(invoice_join_cte).unionByName(paymentid_join_cte).dropDuplicates(["payment_id", "claim_version", "payment_component_id"])

# =============================================================================
# Add skyfire_parent mapping
# =============================================================================
print("=== Applying skyfire_parent mapping ===")

# Define the mapping for payment_category to skyfire_parent
mapping_expr = {
    "CAT1": "PARENT_A",
    "CAT2": "PARENT_B",
    "CAT3": "PARENT_C"
    # Add all other mappings here
}

# Create a column mapping expression for use in withColumn
# This iterates through the map keys and creates a nested when/otherwise expression.
mapping_col = reduce(
    lambda acc, key: when(col("payment_category") == key, lit(mapping_expr[key])).otherwise(acc),
    mapping_expr.keys(),
    lit("DEFAULT_PARENT") # Default value for categories not in the map
)

# Add the new column based on the mapping
final_payments = combined_payments.withColumn("skyfire_parent", mapping_col)

final_payments.printSchema()
final_payments.show(10, truncate=False)


# COMMAND ----------

display(invoice_join_base)

# COMMAND ----------



# COMMAND ----------

# =============================================================================
# STEP 4: BUILD NET_INCURRED_CLAIMS_FLAG (replicate dim.usp_load_payment exactly)
# =============================================================================
print("=== Building Net Incurred Claims Flag (replicating dim.usp_load_payment) ===")
# Exact replication of the SQL CASE statement with LTRIM(RTRIM()) logic
net_incurred_claims_flag_expr = (
when(
(trim(col("status")).isin("Paid", "MarkedAsPaid")) &
(trim(col("type")) == "ReservePayment"),
lit(1)
)
.when(
(trim(col("status")).isin("Reversed", "Approved")) &
(trim(col("type")) == "RecoveryReceipt"),
lit(-1)
)
.when(
(trim(col("status")) == "Reversed") &
(trim(col("type")) == "ReservePayment"),
lit(1)
)
.when(
(trim(col("status")) == "Cancelled") &
(trim(col("type")) == "RecoveryReceipt"),
lit(0)
)
.when(
(trim(col("status")) == "Reversed") &
(trim(col("type")) == "ReservePaymentReversal"),
lit(-1)
)
.otherwise(0)
)
payments_base = combined_payments.withColumn("net_incurred_claims_flag", 
net_incurred_claims_flag_expr)

# COMMAND ----------



# COMMAND ----------

# =============================================================================
# STEP 5: PAYMENT DEDUPLICATION (replicate fact.usp_load_payments deduplication)
# =============================================================================
print("=== Applying Payment Deduplication ===")
# Replicate the SQL deduplication window exactly
deduplication_window = Window.partitionBy(
"payment_guid",
"claim_id",
"status",
"gross_amount",
"payment_category"
).orderBy(col("transaction_date").asc())
payments_base_deduplicated = payments_base.withColumn(
"rn", row_number().over(deduplication_window)
).filter(col("rn") == 1).drop("rn")

# COMMAND ----------



# COMMAND ----------

# =============================================================================
# STEP 6: CALCULATE PAID_TOT_TPPD (replicate usp_load_claims_transaction_recovery_payments & usp_load_claims_transaction_reserve_payments)
# =============================================================================
print("=== Calculating Paid TPPD (replicating payment stored procedures) ===")
# Recovery Payments Logic (from usp_load_claims_transaction_recovery_payments)
recovery_payments_df = payments_base_deduplicated.filter(
(trim(col("type")) == 'RecoveryReceipt') &
(trim(col("status")).isin('Approved', 'Cancelled'))
)
# Exact replication of SQL recovery logic with nested CASE
paid_tppd_recovery_expr = when(
col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"),
when(trim(col("status")) == "Approved",
col("gross_amount") * col("net_incurred_claims_flag"))
.otherwise(col("gross_amount"))
).otherwise(0)
recovery_tppd = recovery_payments_df.withColumn("paid_tot_tppd", 
paid_tppd_recovery_expr)
# Reserve Payments Logic (from usp_load_claims_transaction_reserve_payments)
reserve_payments_df = payments_base_deduplicated.filter(
(trim(col("type")) == 'ReservePayment') &
(col("gross_amount") != 0)
)
# Exact replication of SQL reserve payment logic with 'Reversed' status handling
paid_tppd_reserve_expr = when(
col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"),
when(trim(col("status")) == 'Reversed', col("gross_amount") * -1)
.otherwise(col("gross_amount") * col("net_incurred_claims_flag"))
).otherwise(0)
reserve_tppd_payments = reserve_payments_df.withColumn("paid_tot_tppd", 
paid_tppd_reserve_expr)
# Combine and aggregate paid_tot_tppd
all_paid_transactions = recovery_tppd.select("claim_id", "paid_tot_tppd").unionByName(reserve_tppd_payments.select("claim_id", "paid_tot_tppd"))
paid_tppd_agg = all_paid_transactions.groupBy("claim_id").agg(_sum("paid_tot_tppd").alias("paid_tot_tppd"))

# COMMAND ----------



# COMMAND ----------

# =============================================================================
# STEP 7: BUILD FACT_RESERVE LOGIC (replicate fact.usp_load_reserve)
# =============================================================================
print("=== Building Reserve Base Data (replicating fact.usp_load_reserve) ===")
# Reserves CTE - exact replication with corrected ROW_NUMBER logic
reserve_base_with_versions = reserve.alias("r").join(
claim_version.alias("cv"),
col("r.event_identity") == col("cv.event_identity"),
"inner"
).join(
claim_version_item.alias("vi"),
[col("r.event_identity") == col("vi.event_identity"),
col("r.claim_version_item_index") == col("vi.claim_version_item_index")],
"inner"
).select(
col("r.*"),
col("cv.claim_id"),
col("cv.claim_version_id"),
col("vi.claim_version_item_id").alias("claim_version_item_id")
)
# Exact ROW_NUMBER replication - prioritize claim_version_item_id <> 0
# Note: SQL uses vi.claim_id in partition, not just claim_id
reserves_window_spec = Window.partitionBy("reserve_guid", "claim_id", 
"claim_version_id").orderBy(
when(col("claim_version_item_id") != 0, 0).otherwise(1),
col("reserve_value")
)
reserves_filtered = reserve_base_with_versions.withColumn(
"rn", row_number().over(reserves_window_spec)
).filter(col("rn") == 1)

# COMMAND ----------



# COMMAND ----------

# =============================================================================
# STEP 8: CALCULATE PAYMENT AGGREGATIONS FOR RESERVES (replicate fact.usp_load_reserve payments_cte)
# =============================================================================
print("=== Calculating Payment Aggregations for Reserves ===")
# Payments CTE from fact.usp_load_reserve - calculate finance amounts
finance_reserve_expr = when(
(col("type") == "ReservePayment") & (col("status").isin("MarkedAsPaid", "Paid")),
col("gross_amount")
).otherwise(0)
finance_recovery_expr = when(
(col("type") == "RecoveryReceipt") & (col("status") == "Approved"),
col("gross_amount")
).otherwise(0)
# Get payment_id and claim_version_item_index from payment table
payment_with_index = payment.select("payment_id", "payment_guid", 
"claim_version_item_index").dropDuplicates()
payments_agg_df = (
payments_base_deduplicated
.join(payment_with_index, ["payment_id", "payment_guid"], "left")
.groupBy("claim_id", "claim_version", "payment_category", "claim_version_item_index")
.agg(
_sum(finance_reserve_expr).alias("finance_reserve_gross_amount"),
_sum(finance_recovery_expr).alias("finance_recovery_gross_amount")
)
)

# COMMAND ----------



# COMMAND ----------

# =============================================================================
# STEP 9: CALCULATE ADJUSTED RESERVE VALUES (replicate fact.usp_load_reserve combined_cte)
# =============================================================================
print("=== Calculating Adjusted Reserve Values ===")
# Combined CTE - join reserves with payment aggregations
# CORRECTED: SQL joins claim_version_id = claim_version (not claim_version_id = claim_version_id)
reserves_adjusted = reserves_filtered.join(
payments_agg_df,
[
reserves_filtered.claim_id == payments_agg_df.claim_id,
reserves_filtered.claim_version_id == payments_agg_df.claim_version, 
# CORRECTED: claim_version not claim_version_id
reserves_filtered.category_data_description == payments_agg_df.payment_category,
reserves_filtered.claim_version_item_index == payments_agg_df.claim_version_item_index],"left").drop(payments_agg_df.claim_id).drop(payments_agg_df.claim_version_item_index).withColumn(
"finance_adjusted_reserve_value",
coalesce(col("reserve_value"), lit(0)) - coalesce(col("finance_reserve_gross_amount"), lit(0))
).withColumn(
"finance_adjusted_recovery_value",
coalesce(col("expected_recovery_value"), lit(0)) - 
coalesce(col("finance_recovery_gross_amount"), lit(0))
)
# Add skyfire_parent mapping to reserves
reserves_with_parent = reserves_adjusted.withColumn(
"skyfire_parent", mapping_expr[col("category_data_description")]
)

# COMMAND ----------



# COMMAND ----------

# =============================================================================
# STEP 10: CALCULATE RESERVE_TPPD (replicate usp_load_claims_transaction_reserve_recovery)
# =============================================================================
print("=== Calculating Reserve TPPD (replicating reserve movement logic) ===")
# Window for LAG function - must match SQL exactly
w_mov = Window.partitionBy("claim_id", "reserve_guid").orderBy("claim_version_id")
# Calculate reserve movements using LAG
movements_cte = reserves_with_parent.withColumn(
"reserve_movement",
coalesce(col("finance_adjusted_reserve_value"), lit(0)) -
lag(coalesce(col("finance_adjusted_reserve_value"), lit(0)), 1, 0).over(w_mov)
).withColumn(
"recovery_reserve_movement",
(coalesce(col("finance_adjusted_recovery_value"), lit(0)) -
lag(coalesce(col("finance_adjusted_recovery_value"), lit(0)), 1, 0).over(w_mov)) * -1
)
# Calculate reserve_tppd for both reserve and recovery movements
reserve_part = movements_cte.filter(col("reserve_movement") != 0)
reserve_tppd_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), 
col("reserve_movement")).otherwise(0)
reserve_part_final = reserve_part.withColumn("reserve_tppd", reserve_tppd_expr)
recovery_reserve_part = movements_cte.filter(col("recovery_reserve_movement") != 0)
recovery_reserve_tppd_expr = when(col("skyfire_parent").isin("TP_PROPERTY", 
"TP_VEHICLE"), col("recovery_reserve_movement")).otherwise(0)
recovery_reserve_part_final = recovery_reserve_part.withColumn("reserve_tppd", 
recovery_reserve_tppd_expr)
# Combine and aggregate reserve_tppd
all_reserve_transactions = reserve_part_final.select("claim_id", "reserve_tppd").unionByName(recovery_reserve_part_final.select("claim_id", "reserve_tppd"))
reserve_tppd_agg = all_reserve_transactions.groupBy("claim_id").agg(_sum("reserve_tppd").alias("reserve_tppd"))

# COMMAND ----------

# =============================================================================
# STEP 11: FINAL COMBINATION AND RESULT (replicate the final merge logic)
# =============================================================================
print("=== Creating Final Result ===")
# Get base claim data with latest version
claim_and_versions_df = claim.join(claim_version, "claim_id", "inner")
window_spec = Window.partitionBy("claim_id").orderBy(col("claim_version_id").desc())
base_claim_df = claim_and_versions_df.withColumn("rn", row_number().over(window_spec)).filter(col("rn") == 1).select("claim_id", "claim_version_id", "event_identity")
# Join paid and reserve aggregations
result = base_claim_df.join(paid_tppd_agg, "claim_id", "left").join(reserve_tppd_agg, "claim_id", "left")
# Calculate inc_tot_tppd
result = result.withColumn(
"inc_tot_tppd",
coalesce(col("paid_tot_tppd"), lit(0)) + coalesce(col("reserve_tppd"), lit(0))
)
# Add payment_category for reference (get any payment category for the claim)
payment_category_dedup = payments_base_deduplicated.select("claim_id", 
"payment_category").dropDuplicates(["claim_id"])
result = result.join(payment_category_dedup, "claim_id", "left")
# Final result with required columns
final_result = result.select(
"claim_id",
"claim_version_id",
"payment_category",
"paid_tot_tppd",
"reserve_tppd",
"inc_tot_tppd"
)
print("=== TPPD Calculation Complete ===")
print("Final schema:")
final_result.printSchema()

# COMMAND ----------



# COMMAND ----------

# Display results for your test cases
print("=== Results for Test Cases ===")
test_claims = [1901162, 1894827, 1789444]
for claim_id in test_claims:
    print(f"\n— Results for claim_id: {claim_id} —")
    result_df = final_result.filter(col("claim_id") == claim_id)
    result_df.show(truncate=False)

# Show intermediate results for debugging
print(f"Paid TPPD components for claim {claim_id}:")
all_paid_transactions.filter(col("claim_id") == claim_id).show(truncate=False)
print(f"Reserve TPPD components for claim {claim_id}:")
all_reserve_transactions.filter(col("claim_id") == claim_id).show(truncate=False)

# COMMAND ----------

# Optional: Show detailed breakdown for debugging
print("=== Detailed Payment Breakdown ===")
payments_base_deduplicated.filter(col("claim_id").isin(test_claims)).select(
"claim_id", "payment_id", "type", "status", "payment_category", "skyfire_parent", 
"gross_amount", "net_incurred_claims_flag"
).show(truncate=False)
print("=== Detailed Reserve Breakdown ===")
movements_cte.filter(col("claim_id").isin(test_claims)).select(
"claim_id", "reserve_guid", "category_data_description", "skyfire_parent",
"reserve_movement", "recovery_reserve_movement"
).show(truncate=False)

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, coalesce, create_map, lit,
    sum as _sum, lag, first, row_number, trim
)
from pyspark.sql.window import Window
from pyspark.sql.types import StringType

# Category to parent mapping
category_to_parent = {
    # TP_VEHICLE
    "TP Intervention (Mobility)": "TP_VEHICLE",
    "TP Intervention (Vehicle)": "TP_VEHICLE",
    "TP Credit Hire (Vehicle)": "TP_VEHICLE",
    "TP Authorised Hire (Vehicle)": "TP_VEHICLE",
    "TP Fees (Vehicle)": "TP_VEHICLE",
    "Ex-Gratia (Vehicle)": "TP_VEHICLE",
    "TP Authorized Hire Vehicle": "TP_VEHICLE",
    # TP_PROPERTY
    "Medical Expenses": "TP_PROPERTY",
    "Legal Expenses": "TP_PROPERTY",
    "TP Fees (Property)": "TP_PROPERTY",
    "TP Credit Hire (Property)": "TP_PROPERTY",
    "TP Authorised Hire (Property)": "TP_PROPERTY",
    "TP Intervention (Property)": "TP_PROPERTY",
    "Unknown": "TP_PROPERTY",
    "AD Ex-Gratia (Property)": "TP_PROPERTY",
    "Fire Ex-Gratia (Property)": "TP_PROPERTY",
    "OD Reinsurance": "TP_PROPERTY",
    "Theft Ex-Gratia (Property)": "TP_PROPERTY",
    "TP Damage (Property)": "TP_PROPERTY",
    "TP Authorized Hire Property": "TP_PROPERTY",
    "TP Intervention (Uninsured Loss)": "TP_PROPERTY",
    "TP Intervention (Fees)": "TP_PROPERTY",
    "TP Damage (Vehicle)": "TP_PROPERTY",
}

# — Initialize SparkSession and Load DataFrames —
spark = SparkSession.builder.getOrCreate()

# Load actual data from Databricks tables
claim = spark.table("prod_adp_certified.claim.claim").select("claim_id", "policy_number", "claim_number")
claim_version = spark.table("prod_adp_certified.claim.claim_version").select("claim_id", "claim_number", "policy_number", "claim_version_id", "event_identity")
# FIX: Added missing columns required for the join operation below.
claim_version_item = spark.table("prod_adp_certified.claim.claim_version_item").select("claim_id", "claim_version_id", "claim_version_item_id", "claim_version_item_index", "event_identity")
payment = spark.table("prod_adp_certified.claim.payment").select("claim_id", "payment_reference", "payment_id", "payment_guid", "claim_version", "claim_version_item_index", "event_identity", "head_of_damage", "status", "type", "transaction_date", "event_enqueued_utc_time", "metadata_insert_timestamp")
payment_component_raw = spark.table("prod_adp_certified.claim.payment_component").select("payment_id", "payment_guid", "claim_version", "invoice_number", "event_identity", "payment_category", "net_amount", "gross_amount")
reserve = spark.table("prod_adp_certified.claim.reserve").select("reserve_guid", "head_of_damage", "event_identity", "claim_version_item_index", "category_data_description", "reserve_value", "type", "expected_recovery_value", "event_enqueued_utc_time", "metadata_insert_timestamp")


# COMMAND ----------

# =============================================================================
# 1. Define Mappings
# =============================================================================
mapping_expr = create_map([lit(x) for i in category_to_parent.items() for x in i])
payment_component = payment_component_raw.withColumn("skyfire_parent", mapping_expr[col("payment_category")])


# COMMAND ----------

# =============================================================================
# 2. Correct Calculation of 'net_incurred_claims_flag'
# =============================================================================
net_incurred_claims_flag_expr = (
    when(
        (trim(col("status")).isin("Paid", "MarkedAsPaid")) &
        (trim(col("type")) == "ReservePayment"),
        lit(1)
    )
    .when(
        (trim(col("status")).isin("Reversed", "Approved")) &
        (trim(col("type")) == "RecoveryReceipt"),
        lit(-1)
    )
    .when(
        (trim(col("status")) == "Reversed") &
        (trim(col("type")) == "ReservePayment"),
        lit(1)
    )
    .when(
        (trim(col("status")) == "Cancelled") &
        (trim(col("type")) == "RecoveryReceipt"),
        lit(0)
    )
    .when(
        (trim(col("status")) == "Reversed") &
        (trim(col("type")) == "ReservePaymentReversal"),
        lit(-1)
    )
    .otherwise(0)
)
payment_with_flag = payment.withColumn("net_incurred_claims_flag", net_incurred_claims_flag_expr)
payments_with_flag = payment_with_flag.filter(
    col("event_enqueued_utc_time") >= col("metadata_insert_timestamp")
)


# COMMAND ----------

# =============================================================================
# 3. Corrected Joins and Calculation of `paid_tot_tppd` with Deduplication
# =============================================================================
join1 = payment_with_flag.join(payment_component,
    [payment_with_flag.payment_guid == payment_component.payment_guid,
     payment_with_flag.event_identity == payment_component.event_identity],
    "inner"
)

pc2 = payment_component.alias("pc2")
join2 = payment_with_flag.join(pc2,
    [payment_with_flag.payment_reference == pc2.invoice_number,
     payment_with_flag.event_identity == pc2.event_identity,
     payment_with_flag.claim_version == pc2.claim_version],
    "inner"
)

pc3 = payment_component.alias("pc3")
join3 = payment_with_flag.join(pc3,
    [payment_with_flag.payment_id == pc3.payment_id,
     payment_with_flag.claim_version == pc3.claim_version],
    "inner"
)

cols_to_select_join1 = [payment_with_flag[c] for c in payment_with_flag.columns] + [
    payment_component.payment_category,
    payment_component.net_amount,
    payment_component.gross_amount,
    payment_component.skyfire_parent
]
cols_to_select_join2 = [payment_with_flag[c] for c in payment_with_flag.columns] + [
    pc2.payment_category,
    pc2.net_amount,
    pc2.gross_amount,
    pc2.skyfire_parent
]
cols_to_select_join3 = [payment_with_flag[c] for c in payment_with_flag.columns] + [
    pc3.payment_category,
    pc3.net_amount,
    pc3.gross_amount,
    pc3.skyfire_parent
]
join1_selected = join1.select(*cols_to_select_join1)
join2_selected = join2.select(*cols_to_select_join2)
join3_selected = join3.select(*cols_to_select_join3)
payments_base_with_duplicates = join1_selected.unionByName(join2_selected).unionByName(join3_selected).dropDuplicates()

deduplication_window = Window.partitionBy(
    "payment_guid", "claim_id", "status", "gross_amount", "payment_category"
).orderBy(col("transaction_date").asc())
payments_base = payments_base_with_duplicates.withColumn("rn", row_number().over(deduplication_window)).filter(col("rn") == 1).drop("rn")

recovery_payments_df = payments_base.filter(
    (trim(col("type")) == 'RecoveryReceipt') &
    (trim(col("status")).isin('Approved', 'Cancelled'))
)
paid_tppd_recovery_expr = when(
    col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"),
    col("gross_amount") * col("net_incurred_claims_flag")
).otherwise(0)
recovery_tppd = recovery_payments_df.withColumn("paid_tot_tppd", paid_tppd_recovery_expr)

reserve_payments_df = payments_base.filter(
    (trim(col("type")) == 'ReservePayment') &
    (col("gross_amount") != 0)
)
paid_tppd_reserve_expr = when(
    col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"),
    when(trim(col("status")) == 'Reversed', col("gross_amount") * -1)
    .otherwise(col("gross_amount") * col("net_incurred_claims_flag"))
).otherwise(0)
reserve_tppd_payments = reserve_payments_df.withColumn("paid_tot_tppd", paid_tppd_reserve_expr)

all_paid_transactions = recovery_tppd.select("claim_id", "paid_tot_tppd").unionByName(reserve_tppd_payments.select("claim_id", "paid_tot_tppd"))
paid_tppd_agg = all_paid_transactions.groupBy("claim_id").agg(_sum("paid_tot_tppd").alias("paid_tot_tppd"))


# COMMAND ----------

# =============================================================================
# 4. Correct Calculation of `reserve_tppd`
# =============================================================================
reserve_with_version_item = reserve.alias("r").join(
    claim_version_item.alias("vi"),
    (col("r.event_identity") == col("vi.event_identity")) &
    (col("r.claim_version_item_index") == col("vi.claim_version_item_index")),
    "inner"
).select("r.*", "vi.claim_id", "vi.claim_version_id", "vi.claim_version_item_id")

reserve_with_version_item = reserve_with_version_item.filter(
    col("event_enqueued_utc_time") >= col("metadata_insert_timestamp")
)

reserves_window_spec = Window.partitionBy("reserve_guid", "claim_id", "claim_version_id").orderBy(
    when(col("claim_version_item_id") != 0, 0).otherwise(1), col("reserve_value")
)
reserves_filtered = reserve_with_version_item.withColumn("rn", row_number().over(reserves_window_spec)).filter(col("rn") == 1).drop("rn")

finance_reserve_expr = when((col("type") == "ReservePayment") & (col("status").isin("MarkedAsPaid", "Paid")), col("gross_amount")).otherwise(0)
finance_recovery_expr = when((col("type") == "RecoveryReceipt") & (col("status") == "Approved"), col("gross_amount")).otherwise(0)
payments_agg_df = (
    payments_base
    .groupBy("claim_id", "claim_version", "payment_category", "claim_version_item_index")
    .agg(
        _sum(finance_reserve_expr).alias("finance_reserve_gross_amount"),
        _sum(finance_recovery_expr).alias("finance_recovery_gross_amount")
    )
)

reserves_adjusted = reserves_filtered.join(
    payments_agg_df,
    [
        reserves_filtered.claim_id == payments_agg_df.claim_id,
        reserves_filtered.claim_version_id == payments_agg_df.claim_version,
        reserves_filtered.category_data_description == payments_agg_df.payment_category,
        reserves_filtered.claim_version_item_index == payments_agg_df.claim_version_item_index
    ],
    "left"
).drop(payments_agg_df.claim_id).drop(payments_agg_df.claim_version_item_index).withColumn(
    "finance_adjusted_reserve_value",
    coalesce(col("reserve_value"), lit(0)) - coalesce(col("finance_reserve_gross_amount"), lit(0))
).withColumn(
    "finance_adjusted_recovery_value",
    coalesce(col("expected_recovery_value"), lit(0)) - coalesce(col("finance_recovery_gross_amount"), lit(0))
)

reserves_with_parent = reserves_adjusted.withColumn("skyfire_parent", mapping_expr[col("category_data_description")])
w_mov = Window.partitionBy("claim_id", "reserve_guid").orderBy("claim_version_id")
movements_cte = reserves_with_parent.withColumn(
    "reserve_movement",
    coalesce(col("finance_adjusted_reserve_value"), lit(0)) -
    lag(coalesce(col("finance_adjusted_reserve_value"), lit(0)), 1, 0).over(w_mov)
).withColumn(
    "recovery_reserve_movement",
    (coalesce(col("finance_adjusted_recovery_value"), lit(0)) -
    lag(coalesce(col("finance_adjusted_recovery_value"), lit(0)), 1, 0).over(w_mov)) * -1
)

reserve_part = movements_cte.filter(col("reserve_movement") != 0)
reserve_tppd_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), col("reserve_movement")).otherwise(0)
reserve_part_final = reserve_part.withColumn("reserve_tppd", reserve_tppd_expr)
recovery_reserve_part = movements_cte.filter(col("recovery_reserve_movement") != 0)
recovery_reserve_tppd_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), col("recovery_reserve_movement")).otherwise(0)
recovery_reserve_part_final = recovery_reserve_part.withColumn("reserve_tppd", recovery_reserve_tppd_expr)
all_reserve_transactions = reserve_part_final.select("claim_id", "reserve_tppd").unionByName(recovery_reserve_part_final.select("claim_id", "reserve_tppd"))
reserve_tppd_agg = all_reserve_transactions.groupBy("claim_id").agg(_sum("reserve_tppd").alias("reserve_tppd"))


# COMMAND ----------

# =============================================================================
# 5. Final Combination
# =============================================================================
claim_and_versions_df = claim.join(claim_version.drop("claim_id"), ["claim_number", "policy_number"], "inner")
window_spec = Window.partitionBy("claim_id").orderBy(col("claim_version_id").desc())
base_claim_df = claim_and_versions_df.withColumn("rn", row_number().over(window_spec)).filter(col("rn") == 1).select("claim_id", "claim_version_id", "event_identity")

result = base_claim_df.join(paid_tppd_agg, "claim_id", "left").join(reserve_tppd_agg, "claim_id", "left")
result = result.withColumn(
    "inc_tot_tppd",
    coalesce(col("paid_tot_tppd"), lit(0)) + coalesce(col("reserve_tppd"), lit(0))
)
payment_component_dedup = payment_component.select("event_identity", "payment_category").dropDuplicates(["event_identity"])
result = result.join(payment_component_dedup, "event_identity", "left")

final_result = result.select(
    "claim_id", "claim_version_id", "payment_category",
    "paid_tot_tppd", "reserve_tppd", "inc_tot_tppd"
)

# Display final results for the specific test cases
print(f"\n--- Final Result for Claim ID: 1901162 ---")
display(final_result.filter(col("claim_id") == 1901162))
print(f"\n--- Final Result for Claim ID: 1894827 ---")
display(final_result.filter(col("claim_id") == 1894827))
print(f"\n--- Final Result for Claim ID: 1789444 ---")
display(final_result.filter(col("claim_id") == 1789444))

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, coalesce, create_map, lit, sum as _sum, lag, row_number, trim
from pyspark.sql.window import Window
from pyspark.sql.types import StringType

# =============================================================================
# Configuration for Tracing
# =============================================================================
# Set the claim_id you want to trace through the script execution
claim_id_to_trace = 1894827 

def display_filtered_df(df, df_name):
    """
    Prints a header and displays the content of a DataFrame filtered by the global claim_id_to_trace.
    """
    print(f"\n{'='*40}\nDisplaying DataFrame: {df_name}\n{'='*40}")
    
    # Check if 'claim_id' column exists to filter on
    if "claim_id" in df.columns:
        filtered_df = df.filter(col("claim_id") == claim_id_to_trace)
        print(f"--> Filtered for claim_id = {claim_id_to_trace}")
        display(filtered_df)
    else:
        # If no claim_id, we can't filter. Display the schema and a sample.
        print(f"--> 'claim_id' not in columns. Displaying sample of unfiltered data.")
        display(df)

# — Initialize SparkSession —
spark = SparkSession.builder.getOrCreate()

# =============================================================================
# 1. Load Initial DataFrames
# =============================================================================
print("--- 1. Loading Initial DataFrames ---")

claim = spark.table("prod_adp_certified.claim.claim").select("claim_id", "policy_number", "claim_number")
display_filtered_df(claim, "claim")

claim_version = spark.table("prod_adp_certified.claim.claim_version").select("claim_id", "claim_number", "policy_number", "claim_version_id", "event_identity")
display_filtered_df(claim_version, "claim_version")

claim_version_item = spark.table("prod_adp_certified.claim.claim_version_item").select("claim_id", "claim_version_id", "claim_version_item_id", "claim_version_item_index", "event_identity")
display_filtered_df(claim_version_item, "claim_version_item")

payment = spark.table("prod_adp_certified.claim.payment").select("claim_id", "payment_reference", "payment_id", "payment_guid", "claim_version", "claim_version_item_index", "event_identity", "head_of_damage", "status", "type", "transaction_date")
display_filtered_df(payment, "payment")

payment_component_raw = spark.table("prod_adp_certified.claim.payment_component").select("payment_id", "payment_guid", "claim_version", "invoice_number", "event_identity", "payment_category", "net_amount", "gross_amount")
# This table doesn't have claim_id, so we can't filter it directly yet.
# It will be filtered after being joined.
display_filtered_df(payment_component_raw, "payment_component_raw")

reserve = spark.table("prod_adp_certified.claim.reserve").select("reserve_guid", "head_of_damage", "event_identity", "claim_version_item_index", "category_data_description", "reserve_value", "type", "expected_recovery_value")
# This table also lacks claim_id and will be filtered after a join.
display_filtered_df(reserve, "reserve")


# =============================================================================
# 2. Map Payment Categories and Add Flags
# =============================================================================
print("\n--- 2. Processing Payments ---")

category_to_parent = {
    "TP Intervention (Mobility)": "TP_VEHICLE", "TP Intervention (Vehicle)": "TP_VEHICLE",
    "TP Credit Hire (Vehicle)": "TP_VEHICLE", "TP Authorised Hire (Vehicle)": "TP_VEHICLE",
    "TP Fees (Vehicle)": "TP_VEHICLE", "Ex-Gratia (Vehicle)": "TP_VEHICLE",
    "TP Authorized Hire Vehicle": "TP_VEHICLE", "Medical Expenses": "TP_PROPERTY",
    "Legal Expenses": "TP_PROPERTY", "TP Fees (Property)": "TP_PROPERTY",
    "TP Credit Hire (Property)": "TP_PROPERTY", "TP Authorised Hire (Property)": "TP_PROPERTY",
    "TP Intervention (Property)": "TP_PROPERTY", "Unknown": "TP_PROPERTY",
    "AD Ex-Gratia (Property)": "TP_PROPERTY", "Fire Ex-Gratia (Property)": "TP_PROPERTY",
    "OD Reinsurance": "TP_PROPERTY", "Theft Ex-Gratia (Property)": "TP_PROPERTY",
    "TP Damage (Property)": "TP_PROPERTY", "TP Authorized Hire Property": "TP_PROPERTY",
    "TP Intervention (Uninsured Loss)": "TP_PROPERTY", "TP Intervention (Fees)": "TP_PROPERTY",
    "TP Damage (Vehicle)": "TP_PROPERTY",
}
mapping_expr = create_map([lit(x) for i in category_to_parent.items() for x in i])
payment_component = payment_component_raw.withColumn("skyfire_parent", mapping_expr[col("payment_category")])
display_filtered_df(payment_component, "payment_component (with skyfire_parent)")

net_incurred_claims_flag_expr = (
    when((trim(col("status")).isin("Paid", "MarkedAsPaid")) & (trim(col("type")) == "ReservePayment"), lit(1))
    .when((trim(col("status")).isin("Reversed", "Approved")) & (trim(col("type")) == "RecoveryReceipt"), lit(-1))
    .when((trim(col("status")) == "Reversed") & (trim(col("type")) == "ReservePayment"), lit(1))
    .when((trim(col("status")) == "Cancelled") & (trim(col("type")) == "RecoveryReceipt"), lit(0))
    .when((trim(col("status")) == "Reversed") & (trim(col("type")) == "ReservePaymentReversal"), lit(-1))
    .otherwise(0)
)
payment_with_flag = payment.withColumn("net_incurred_claims_flag", net_incurred_claims_flag_expr)
display_filtered_df(payment_with_flag, "payment_with_flag")


# =============================================================================
# 3. Calculate Paid Totals
# =============================================================================
print("\n--- 3. Calculating Paid Totals (paid_tot_tppd) ---")

join1 = payment_with_flag.join(payment_component, [payment_with_flag.payment_guid == payment_component.payment_guid, payment_with_flag.event_identity == payment_component.event_identity], "inner")
pc2 = payment_component.alias("pc2")
join2 = payment_with_flag.join(pc2, [payment_with_flag.payment_reference == pc2.invoice_number, payment_with_flag.event_identity == pc2.event_identity, payment_with_flag.claim_version == pc2.claim_version], "inner")
pc3 = payment_component.alias("pc3")
join3 = payment_with_flag.join(pc3, [payment_with_flag.payment_id == pc3.payment_id, payment_with_flag.claim_version == pc3.claim_version], "inner")

cols_to_select = [payment_with_flag[c] for c in payment_with_flag.columns] + [payment_component.payment_category, payment_component.net_amount, payment_component.gross_amount, payment_component.skyfire_parent]
payments_base_with_duplicates = join1.select(*cols_to_select).unionByName(join2.select(*cols_to_select)).unionByName(join3.select(*cols_to_select)).dropDuplicates()
display_filtered_df(payments_base_with_duplicates, "payments_base_with_duplicates (after joins and union)")

deduplication_window = Window.partitionBy("payment_guid", "claim_id", "status", "gross_amount", "payment_category").orderBy(col("transaction_date").asc())
payments_base = payments_base_with_duplicates.withColumn("rn", row_number().over(deduplication_window)).filter(col("rn") == 1).drop("rn")
display_filtered_df(payments_base, "payments_base (after deduplication)")

recovery_payments_df = payments_base.filter((trim(col("type")) == 'RecoveryReceipt') & (trim(col("status")).isin('Approved', 'Cancelled')))
paid_tppd_recovery_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), col("gross_amount") * col("net_incurred_claims_flag")).otherwise(0)
recovery_tppd = recovery_payments_df.withColumn("paid_tot_tppd", paid_tppd_recovery_expr)
display_filtered_df(recovery_tppd, "recovery_tppd (calculated for recovery payments)")

reserve_payments_df = payments_base.filter((trim(col("type")) == 'ReservePayment') & (col("gross_amount") != 0))
paid_tppd_reserve_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), when(trim(col("status")) == 'Reversed', col("gross_amount") * -1).otherwise(col("gross_amount") * col("net_incurred_claims_flag"))).otherwise(0)
reserve_tppd_payments = reserve_payments_df.withColumn("paid_tot_tppd", paid_tppd_reserve_expr)
display_filtered_df(reserve_tppd_payments, "reserve_tppd_payments (calculated for reserve payments)")

all_paid_transactions = recovery_tppd.select("claim_id", "paid_tot_tppd").unionByName(reserve_tppd_payments.select("claim_id", "paid_tot_tppd"))
paid_tppd_agg = all_paid_transactions.groupBy("claim_id").agg(_sum("paid_tot_tppd").alias("paid_tot_tppd"))
display_filtered_df(paid_tppd_agg, "paid_tppd_agg (final aggregated paid total)")


# =============================================================================
# 4. Calculate Reserve Totals
# =============================================================================
print("\n--- 4. Calculating Reserve Totals (reserve_tppd) ---")

reserve_with_version_item = reserve.alias("r").join(claim_version_item.alias("vi"), (col("r.event_identity") == col("vi.event_identity")) & (col("r.claim_version_item_index") == col("vi.claim_version_item_index")), "inner").select("r.*", "vi.claim_id", "vi.claim_version_id", "vi.claim_version_item_id")
display_filtered_df(reserve_with_version_item, "reserve_with_version_item (reserve joined with claim_version_item)")

reserves_window_spec = Window.partitionBy("reserve_guid", "claim_id", "claim_version_id").orderBy(when(col("claim_version_item_id") != 0, 0).otherwise(1), col("reserve_value"))
reserves_filtered = reserve_with_version_item.withColumn("rn", row_number().over(reserves_window_spec)).filter(col("rn") == 1).drop("rn")
display_filtered_df(reserves_filtered, "reserves_filtered (after deduplication)")

finance_reserve_expr = when((col("type") == "ReservePayment") & (col("status").isin("MarkedAsPaid", "Paid")), col("gross_amount")).otherwise(0)
finance_recovery_expr = when((col("type") == "RecoveryReceipt") & (col("status") == "Approved"), col("gross_amount")).otherwise(0)
payments_agg_df = payments_base.groupBy("claim_id", "claim_version", "payment_category", "claim_version_item_index").agg(_sum(finance_reserve_expr).alias("finance_reserve_gross_amount"), _sum(finance_recovery_expr).alias("finance_recovery_gross_amount"))
display_filtered_df(payments_agg_df, "payments_agg_df (aggregated payments for reserve adjustment)")

reserves_adjusted = reserves_filtered.join(payments_agg_df, [reserves_filtered.claim_id == payments_agg_df.claim_id, reserves_filtered.claim_version_id == payments_agg_df.claim_version, reserves_filtered.category_data_description == payments_agg_df.payment_category, reserves_filtered.claim_version_item_index == payments_agg_df.claim_version_item_index], "left").drop(payments_agg_df.claim_id).drop(payments_agg_df.claim_version_item_index).withColumn("finance_adjusted_reserve_value", coalesce(col("reserve_value"), lit(0)) - coalesce(col("finance_reserve_gross_amount"), lit(0))).withColumn("finance_adjusted_recovery_value", coalesce(col("expected_recovery_value"), lit(0)) - coalesce(col("finance_recovery_gross_amount"), lit(0)))
display_filtered_df(reserves_adjusted, "reserves_adjusted (after adjusting for payments)")

reserves_with_parent = reserves_adjusted.withColumn("skyfire_parent", mapping_expr[col("category_data_description")])
w_mov = Window.partitionBy("claim_id", "reserve_guid").orderBy("claim_version_id")
movements_cte = reserves_with_parent.withColumn("reserve_movement", coalesce(col("finance_adjusted_reserve_value"), lit(0)) - lag(coalesce(col("finance_adjusted_reserve_value"), lit(0)), 1, 0).over(w_mov)).withColumn("recovery_reserve_movement", (coalesce(col("finance_adjusted_recovery_value"), lit(0)) - lag(coalesce(col("finance_adjusted_recovery_value"), lit(0)), 1, 0).over(w_mov)) * -1)
display_filtered_df(movements_cte, "movements_cte (calculated reserve movements)")

reserve_part = movements_cte.filter(col("reserve_movement") != 0)
reserve_tppd_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), col("reserve_movement")).otherwise(0)
reserve_part_final = reserve_part.withColumn("reserve_tppd", reserve_tppd_expr)
recovery_reserve_part = movements_cte.filter(col("recovery_reserve_movement") != 0)
recovery_reserve_tppd_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), col("recovery_reserve_movement")).otherwise(0)
recovery_reserve_part_final = recovery_reserve_part.withColumn("reserve_tppd", recovery_reserve_tppd_expr)
all_reserve_transactions = reserve_part_final.select("claim_id", "reserve_tppd").unionByName(recovery_reserve_part_final.select("claim_id", "reserve_tppd"))
reserve_tppd_agg = all_reserve_transactions.groupBy("claim_id").agg(_sum("reserve_tppd").alias("reserve_tppd"))
display_filtered_df(reserve_tppd_agg, "reserve_tppd_agg (final aggregated reserve total)")


# =============================================================================
# 5. Final Combination
# =============================================================================
print("\n--- 5. Final Combination ---")

claim_and_versions_df = claim.join(claim_version.drop("claim_id"), ["claim_number", "policy_number"], "inner")
window_spec = Window.partitionBy("claim_id").orderBy(col("claim_version_id").desc())
base_claim_df = claim_and_versions_df.withColumn("rn", row_number().over(window_spec)).filter(col("rn") == 1).select("claim_id", "claim_version_id", "event_identity")
display_filtered_df(base_claim_df, "base_claim_df (latest version of each claim)")

result = base_claim_df.join(paid_tppd_agg, "claim_id", "left").join(reserve_tppd_agg, "claim_id", "left")
display_filtered_df(result, "result (after joining paid and reserve aggregates)")

result = result.withColumn("inc_tot_tppd", coalesce(col("paid_tot_tppd"), lit(0)) + coalesce(col("reserve_tppd"), lit(0)))
payment_component_dedup = payment_component.select("event_identity", "payment_category").dropDuplicates(["event_identity"])
result = result.join(payment_component_dedup, "event_identity", "left")
display_filtered_df(result, "result (after calculating incurred total and joining payment category)")

final_result = result.select("claim_id", "claim_version_id", "payment_category", "paid_tot_tppd", "reserve_tppd", "inc_tot_tppd")
display_filtered_df(final_result, "final_result")



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, coalesce, create_map, lit,
    sum as _sum, lag, first
)
from pyspark.sql.window import Window
from pyspark.sql.types import StringType

# Category to parent mapping
category_to_parent = {
    # TP_VEHICLE
    "TP Intervention (Mobility)": "TP_VEHICLE",
    "TP Intervention (Vehicle)": "TP_VEHICLE",
    "TP Credit Hire (Vehicle)": "TP_VEHICLE",
    "TP Authorised Hire (Vehicle)": "TP_VEHICLE",
    "TP Fees (Vehicle)": "TP_VEHICLE",
    "Ex-Gratia (Vehicle)": "TP_VEHICLE",
    "TP Authorized Hire Vehicle": "TP_VEHICLE",
    # TP_PROPERTY
    "Medical Expenses": "TP_PROPERTY",
    "Legal Expenses": "TP_PROPERTY",
    "TP Fees (Property)": "TP_PROPERTY",
    "TP Credit Hire (Property)": "TP_PROPERTY",
    "TP Authorised Hire (Property)": "TP_PROPERTY",
    "TP Intervention (Property)": "TP_PROPERTY",
    "Unknown": "TP_PROPERTY",
    "AD Ex-Gratia (Property)": "TP_PROPERTY",
    "Fire Ex-Gratia (Property)": "TP_PROPERTY",
    "OD Reinsurance": "TP_PROPERTY",
    "Theft Ex-Gratia (Property)": "TP_PROPERTY",
    "TP Damage (Property)": "TP_PROPERTY",
    "TP Authorized Hire Property": "TP_PROPERTY",
    "TP Intervention (Uninsured Loss)": "TP_PROPERTY",
    "TP Intervention (Fees)": "TP_PROPERTY",
    "TP Damage (Vehicle)": "TP_PROPERTY",
}

# --- Initialize SparkSession and Load DataFrames ---
spark = SparkSession.builder.getOrCreate()
claim = spark.table("prod_adp_certified.claim.claim").select("claim_id", "policy_number", "claim_number")
claim_version = spark.table("prod_adp_certified.claim.claim_version").select("claim_id", "claim_number", "policy_number", "claim_version_id", "event_identity")
claim_version_item = spark.table("prod_adp_certified.claim.claim_version_item").select("claim_id", "claim_version_item_index")
payment = spark.table("prod_adp_certified.claim.payment").select("claim_id", "payment_reference", "payment_id", "payment_guid", "claim_version", "claim_version_item_index", "event_identity", "head_of_damage", "status", "type", "transaction_date")
payment_component = spark.table("prod_adp_certified.claim.payment_component").select("payment_id", "payment_guid", "claim_version", "invoice_number", "event_identity", "payment_category", "net_amount", "gross_amount")
reserve = spark.table("prod_adp_certified.claim.reserve").select("reserve_guid", "head_of_damage", "event_identity", "claim_version_item_index", "category_data_description", "reserve_value", "type", "expected_recovery_value")

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, when, lit, create_map, sum as _sum, coalesce, lag, row_number, trim

# Assume SparkSession 'spark' is already available.
# Assume DataFrames 'claim', 'claim_version', 'payment', 'payment_component', 'reserve' are loaded.

# =============================================================================
# 1. Define Mappings
# =============================================================================

mapping_expr = create_map([lit(x) for i in category_to_parent.items() for x in i])
payment_component = payment_component.withColumn("skyfire_parent", mapping_expr[col("payment_category")])

# COMMAND ----------

# =============================================================================
# 2. Correct Calculation of 'net_incurred_claims_flag'
# This logic is replicated from 'dim.usp_load_payment'.
# =============================================================================
# The original 'net_incurred_claims_flag' was incorrect and oversimplified.
# This new expression correctly mirrors the complex CASE statement in the SQL procedure.
net_incurred_claims_flag_expr = (
    when(
        (trim(col("status")).isin("Paid", "MarkedAsPaid")) & 
        (trim(col("type")) == "ReservePayment"), 
        lit(1)
    )
    .when(
        (trim(col("status")).isin("Reversed", "Approved")) & 
        (trim(col("type")) == "RecoveryReceipt"), 
        lit(-1)
    )
    .when(
        (trim(col("status")) == "Reversed") & 
        (trim(col("type")) == "ReservePayment"), 
        lit(1)
    )
    .when(
        (trim(col("status")) == "Cancelled") & 
        (trim(col("type")) == "RecoveryReceipt"), 
        lit(0)
    )
    .when(
        (trim(col("status")) == "Reversed") & 
        (trim(col("type")) == "ReservePaymentReversal"), 
        lit(-1)
    )
    .otherwise(0)
)

payment_with_flag = payment.withColumn("net_incurred_claims_flag", net_incurred_claims_flag_expr)

# COMMAND ----------

# =============================================================================
# 3. Corrected Joins and Calculation of `paid_tot_tppd` with Deduplication
# =============================================================================

# --- Replicate the SQL's multi-strategy join ---
join1 = payment_with_flag.join(payment_component, [payment_with_flag.payment_guid == payment_component.payment_guid, payment_with_flag.event_identity == payment_component.event_identity], "inner")
pc2 = payment_component.alias("pc2")
join2 = payment_with_flag.join(pc2, [payment_with_flag.payment_reference == pc2.invoice_number, payment_with_flag.event_identity == pc2.event_identity, payment_with_flag.claim_version == pc2.claim_version], "inner")
pc3 = payment_component.alias("pc3")
join3 = payment_with_flag.join(pc3, [payment_with_flag.payment_id == pc3.payment_id, payment_with_flag.claim_version == pc3.claim_version], "inner")

cols_to_select = [payment_with_flag[c] for c in payment_with_flag.columns] + [payment_component.payment_category, payment_component.net_amount, payment_component.gross_amount, payment_component.skyfire_parent]
join1_selected = join1.select(*cols_to_select)
join2_selected = join2.select(*cols_to_select)
join3_selected = join3.select(*cols_to_select)
payments_base_with_duplicates = join1_selected.unionByName(join2_selected).unionByName(join3_selected).dropDuplicates()


# --- Correctly Replicate the SQL Deduplication Logic ---
# This window identifies duplicate transactions based on their key attributes.
deduplication_window = Window.partitionBy(
    "payment_guid",
    "claim_id",
    "status",
    "gross_amount",
    "payment_category" 
).orderBy(col("transaction_date").asc())

# Apply the deduplication by keeping only the first record (rn=1) of each identical transaction.
payments_base = payments_base_with_duplicates.withColumn("rn", row_number().over(deduplication_window)).filter(col("rn") == 1).drop("rn")


# --- The rest of the logic now operates on the correctly deduplicated `payments_base` DataFrame ---

# --- Logic from usp_load_claims_transaction_recovery_payments ---
recovery_payments_df = payments_base.filter(
    (trim(col("type")) == 'RecoveryReceipt') &
    (trim(col("status")).isin('Approved', 'Cancelled'))
)

paid_tppd_recovery_expr = when(
    col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"),
    col("gross_amount") * col("net_incurred_claims_flag")
).otherwise(0)

recovery_tppd = recovery_payments_df.withColumn("paid_tot_tppd", paid_tppd_recovery_expr)


# --- Logic from usp_load_claims_transaction_reserve_payments ---
reserve_payments_df = payments_base.filter(
    (trim(col("type")) == 'ReservePayment') &
    (col("gross_amount") != 0)
)

paid_tppd_reserve_expr = when(
    col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"),
    when(trim(col("status")) == 'Reversed', col("gross_amount") * -1)
    .otherwise(col("gross_amount") * col("net_incurred_claims_flag"))
).otherwise(0)

reserve_tppd_payments = reserve_payments_df.withColumn("paid_tot_tppd", paid_tppd_reserve_expr)


# --- Combine and Aggregate `paid_tot_tppd` ---
all_paid_transactions = recovery_tppd.select("claim_id", "paid_tot_tppd") \
    .unionByName(reserve_tppd_payments.select("claim_id", "paid_tot_tppd"))

paid_tppd_agg = all_paid_transactions.groupBy("claim_id").agg(_sum("paid_tot_tppd").alias("paid_tot_tppd"))

# COMMAND ----------

#print("payment_with_flag")
#display(payment_with_flag.filter(col("claim_id") == 1901162))
#print("payments_base")
## claim_id = [1901162, 1894827, 1789444]
#display(payments_base.filter(col("claim_id") == 1901162))   #  .filter(col("prod_adp_certified.claim.claim_version.claim_id") == 1901162))
#
#print("recovery_payments_df")
#display(recovery_payments_df.filter(col("claim_id") == 1901162))
#print("recovery_tppd")
#display(recovery_tppd.filter(col("claim_id") == 1901162)) 
#print("reserve_tppd_payments") 
#display(reserve_tppd_payments.filter(col("claim_id") == 1901162)) 
#print("all_paid_transactions")
#display(all_paid_transactions.filter(col("claim_id") == 1901162)) 
#print("paid_tppd_agg") 
#display(paid_tppd_agg.filter(col("claim_id") == 1901162)) 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# =============================================================================
# 4. Correct Calculation of `reserve_tppd`
# =============================================================================

# --- 4.1 Replicate `reserves_cte` filtering from 'fact.usp_load_reserve' ---
reserve_base_with_versions = reserve.alias("r").join(
    claim_version.alias("cv"),
    col("r.event_identity") == col("cv.event_identity"),
    "inner"
).select(
    "r.*",
    "cv.claim_id",
    "cv.claim_version_id"
)
reserves_window_spec = Window.partitionBy("reserve_guid", "claim_id", "claim_version_id").orderBy(
    when(col("claim_version_item_index") != 0, 0).otherwise(1),
    col("reserve_value")
)
reserves_filtered = reserve_base_with_versions.withColumn("rn", row_number().over(reserves_window_spec)) \
                                            .filter(col("rn") == 1)

# --- 4.2 Replicate `payments_cte` from 'fact.usp_load_reserve' ---
finance_reserve_expr = when(
    (col("type") == "ReservePayment") & (col("status").isin("MarkedAsPaid", "Paid")),
    col("gross_amount")
).otherwise(0)
finance_recovery_expr = when(
    (col("type") == "RecoveryReceipt") & (col("status") == "Approved"),
    col("gross_amount")
).otherwise(0)
payments_agg_df = (
    payments_base
    .groupBy("claim_id", "claim_version", "payment_category", "claim_version_item_index")
    .agg(
        _sum(finance_reserve_expr).alias("finance_reserve_gross_amount"),
        _sum(finance_recovery_expr).alias("finance_recovery_gross_amount")
    )
)

# --- 4.3 Replicate `combined_cte` to get adjusted values ---
# CORRECTED LOGIC: The join condition now correctly uses 'claim_version' from the payments_agg_df.
reserves_adjusted = reserves_filtered.join(
    payments_agg_df,
    [
        reserves_filtered.claim_id == payments_agg_df.claim_id,
        reserves_filtered.claim_version_id == payments_agg_df.claim_version, # Corrected join condition
        reserves_filtered.category_data_description == payments_agg_df.payment_category,
        reserves_filtered.claim_version_item_index == payments_agg_df.claim_version_item_index
    ],
    "left"
).drop(payments_agg_df.claim_id) \
 .drop(payments_agg_df.claim_version_item_index) \
 .withColumn(
    "finance_adjusted_reserve_value",
    coalesce(col("reserve_value"), lit(0)) - coalesce(col("finance_reserve_gross_amount"), lit(0))
).withColumn(
    "finance_adjusted_recovery_value",
    coalesce(col("expected_recovery_value"), lit(0)) - coalesce(col("finance_recovery_gross_amount"), lit(0))
)

# --- 4.4 Calculate `reserve_tppd` using Adjusted Values ---

# CORRECTED LOGIC: 'skyfire_parent' must be derived directly from the reserve's own category,
# not by joining to the payment_component. This ensures the column is not null.
reserves_with_parent = reserves_adjusted.withColumn(
    "skyfire_parent", mapping_expr[col("category_data_description")]
)

w_mov = Window.partitionBy("claim_id", "reserve_guid").orderBy("claim_version_id")

movements_cte = reserves_with_parent.withColumn(
    "reserve_movement",
    coalesce(col("finance_adjusted_reserve_value"), lit(0)) - lag(coalesce(col("finance_adjusted_reserve_value"), lit(0)), 1, 0).over(w_mov)
).withColumn(
    "recovery_reserve_movement",
    (coalesce(col("finance_adjusted_recovery_value"), lit(0)) - lag(coalesce(col("finance_adjusted_recovery_value"), lit(0)), 1, 0).over(w_mov)) * -1
)

# This expression will now work because 'skyfire_parent' is correctly populated
reserve_part = movements_cte.filter(col("reserve_movement") != 0)
reserve_tppd_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), col("reserve_movement")).otherwise(0)
reserve_part_final = reserve_part.withColumn("reserve_tppd", reserve_tppd_expr)

recovery_reserve_part = movements_cte.filter(col("recovery_reserve_movement") != 0)
recovery_reserve_tppd_expr = when(col("skyfire_parent").isin("TP_PROPERTY", "TP_VEHICLE"), col("recovery_reserve_movement")).otherwise(0)
recovery_reserve_part_final = recovery_reserve_part.withColumn("reserve_tppd", recovery_reserve_tppd_expr)

# The final aggregation will now correctly sum the '1101' value
all_reserve_transactions = reserve_part_final.select("claim_id", "reserve_tppd") \
    .unionByName(recovery_reserve_part_final.select("claim_id", "reserve_tppd"))
    
reserve_tppd_agg = all_reserve_transactions.groupBy("claim_id").agg(_sum("reserve_tppd").alias("reserve_tppd"))

# COMMAND ----------

#print("payment_with_flag")
#display(payment_with_flag.filter(col("claim_id") == 1901162))
#print("payments_base")
## claim_id = [1901162, 1894827, 1789444]
#display(payments_base.filter(col("claim_id") == 1901162))   #  .filter(col("prod_adp_certified.claim.claim_version.claim_id") == 1901162))
#
#print("recovery_payments_df")
#display(recovery_payments_df.filter(col("claim_id") == 1901162))
#print("recovery_tppd")
#display(recovery_tppd.filter(col("claim_id") == 1901162)) 
#print("reserve_tppd_payments") 
#display(reserve_tppd_payments.filter(col("claim_id") == 1901162)) 
#print("all_paid_transactions")
#display(all_paid_transactions.filter(col("claim_id") == 1901162)) 
#print("paid_tppd_agg") 
#display(paid_tppd_agg.filter(col("claim_id") == 1901162)) 

# COMMAND ----------

# =============================================================================
# 5. Final Combination
# =============================================================================
claim_and_versions_df = claim.join(claim_version, "claim_id", "inner")
window_spec = Window.partitionBy("claim_id").orderBy(col("claim_version_id").desc())
base_claim_df = claim_and_versions_df.withColumn("rn", row_number().over(window_spec)) \
                                     .filter(col("rn") == 1) \
                                     .select("claim_id", "claim_version_id", "event_identity")
result = base_claim_df.join(paid_tppd_agg, "claim_id", "left") \
                      .join(reserve_tppd_agg, "claim_id", "left")
result = result.withColumn(
    "inc_tot_tppd",
    coalesce(col("paid_tot_tppd"), lit(0)) + coalesce(col("reserve_tppd"), lit(0))
)
payment_component_dedup = payment_component.select("event_identity", "payment_category").dropDuplicates(["event_identity"])
result = result.join(payment_component_dedup, "event_identity", "left")
final_result = result.select(
    "claim_id",
    "claim_version_id",
    "payment_category",
    "paid_tot_tppd",
    "reserve_tppd",
    "inc_tot_tppd"
)

#final_result.show()



# COMMAND ----------

# claim_id = [1901162, 1894827, 1789444]
display(final_result.filter(col("claim_id") == 1901162)) 

# COMMAND ----------

print("final_result") 
display(final_result.filter(col("claim_id") == 1894827)) 

# COMMAND ----------

# claim_id = [1901162, 1894827, 1789444]
display(final_result.filter(col("claim_id") == 1789444)) 

# COMMAND ----------

# MAGIC %sql
# MAGIC --- =============================================
# MAGIC --- TPPD Calculation - Debugging Implementation
# MAGIC --- Replicating logic and printing each intermediate table for selected claims.
# MAGIC --- =============================================
# MAGIC --- TEST CASE FILTER
# MAGIC --- claims to be analyzed: (1901162, 1894827, 1789444)
# MAGIC --- =============================================
# MAGIC  
# MAGIC ---=============================================================================
# MAGIC --- STEP 1: CATEGORY MAPPING (No output, used by later steps)
# MAGIC ---=============================================================================
# MAGIC CREATE OR REPLACE TEMP VIEW category_mapping AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('TP Intervention (Mobility)', 'TP_VEHICLE'),
# MAGIC   ('TP Intervention (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('TP Credit Hire (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('TP Authorised Hire (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('TP Fees (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('Ex-Gratia (Vehicle)', 'TP_VEHICLE'),
# MAGIC   ('TP Authorized Hire Vehicle', 'TP_VEHICLE'),
# MAGIC   ('Medical Expenses', 'TP_PROPERTY'),
# MAGIC   ('Legal Expenses', 'TP_PROPERTY'),
# MAGIC   ('TP Fees (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Credit Hire (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Authorised Hire (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Intervention (Property)', 'TP_PROPERTY'),
# MAGIC   ('Unknown', 'TP_PROPERTY'),
# MAGIC   ('AD Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC   ('Fire Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC   ('OD Reinsurance', 'TP_PROPERTY'),
# MAGIC   ('Theft Ex-Gratia (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Damage (Property)', 'TP_PROPERTY'),
# MAGIC   ('TP Authorized Hire Property', 'TP_PROPERTY'),
# MAGIC   ('TP Intervention (Uninsured Loss)', 'TP_PROPERTY'),
# MAGIC   ('TP Intervention (Fees)', 'TP_PROPERTY'),
# MAGIC   ('TP Damage (Vehicle)', 'TP_PROPERTY')
# MAGIC AS t(payment_category, skyfire_parent);
# MAGIC  
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 2.1: guid_join_cte
# MAGIC ---=============================================================================
# MAGIC WITH guid_join_cte AS (
# MAGIC   SELECT
# MAGIC     p.payment_id, p.payment_reference, p.payment_guid, p.claim_version,
# MAGIC     p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status,
# MAGIC     p.type, p.transaction_date, p.claim_id, pc.payment_component_id,
# MAGIC     pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444)
# MAGIC     AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC )
# MAGIC SELECT 'guid_join_cte' AS source_cte, * FROM guid_join_cte;
# MAGIC  
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 2.2: invoice_join_cte
# MAGIC ---=============================================================================
# MAGIC WITH guid_join_cte AS (
# MAGIC   SELECT
# MAGIC     p.payment_id, p.payment_reference, p.payment_guid, p.claim_version,
# MAGIC     p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status,
# MAGIC     p.type, p.transaction_date, p.claim_id, pc.payment_component_id,
# MAGIC     pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444)
# MAGIC     AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC ),
# MAGIC invoice_join_cte AS (
# MAGIC   SELECT DISTINCT
# MAGIC     p.payment_id, p.payment_reference, p.payment_guid, p.claim_version,
# MAGIC     p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status,
# MAGIC     p.type, p.transaction_date, p.claim_id, pc.payment_component_id,
# MAGIC     pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC     ON p.payment_reference = pc.invoice_number
# MAGIC     AND p.event_identity = pc.event_identity
# MAGIC     AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN guid_join_cte g ON g.payment_id = p.payment_id AND g.claim_version = p.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444)
# MAGIC     AND g.payment_id IS NULL
# MAGIC     AND p.payment_reference IS NOT NULL
# MAGIC )
# MAGIC SELECT 'invoice_join_cte' AS source_cte, * FROM invoice_join_cte;
# MAGIC  
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 2.3: paymentid_join_cte
# MAGIC ---=============================================================================
# MAGIC WITH guid_join_cte AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT
# MAGIC       p.payment_id, p.claim_version, pc.payment_component_id
# MAGIC     FROM prod_adp_certified.claim.payment p
# MAGIC     INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC       ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC     WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC   )
# MAGIC ),
# MAGIC invoice_join_cte AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT DISTINCT
# MAGIC       p.payment_id, p.claim_version, pc.payment_component_id
# MAGIC     FROM prod_adp_certified.claim.payment p
# MAGIC     INNER JOIN prod_adp_certified.claim.payment_component pc
# MAGIC       ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC     LEFT JOIN guid_join_cte g ON g.payment_id = p.payment_id AND g.claim_version = p.claim_version
# MAGIC     WHERE p.claim_id IN (1901162, 1894827, 1789444) AND g.payment_id IS NULL AND p.payment_reference IS NOT NULL
# MAGIC   )
# MAGIC ),
# MAGIC paymentid_join_cte AS (
# MAGIC   SELECT
# MAGIC     p.payment_id, p.payment_reference, p.payment_guid, p.claim_version,
# MAGIC     p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status,
# MAGIC     p.type, p.transaction_date, p.claim_id, pc.payment_component_id,
# MAGIC     pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p
# MAGIC   INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN (
# MAGIC     SELECT payment_id, payment_component_id, claim_version FROM guid_join_cte
# MAGIC     UNION
# MAGIC     SELECT payment_id, payment_component_id, claim_version FROM invoice_join_cte
# MAGIC   ) g ON g.payment_id = p.payment_id AND g.payment_component_id = pc.payment_component_id AND g.claim_version = p.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND g.payment_id IS NULL
# MAGIC )
# MAGIC SELECT 'paymentid_join_cte' AS source_cte, * FROM paymentid_join_cte;
# MAGIC  
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 4: payments_base
# MAGIC ---=============================================================================
# MAGIC WITH guid_join_cte AS (
# MAGIC   SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC ),
# MAGIC invoice_join_cte AS (
# MAGIC   SELECT DISTINCT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN guid_join_cte g ON g.payment_id = p.payment_id AND g.claim_version = p.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444) AND g.payment_id IS NULL AND p.payment_reference IS NOT NULL
# MAGIC ),
# MAGIC paymentid_join_cte AS (
# MAGIC   SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC   FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC   LEFT JOIN (SELECT payment_id, payment_component_id, claim_version FROM guid_join_cte UNION SELECT payment_id, payment_component_id, claim_version FROM invoice_join_cte) g
# MAGIC   ON g.payment_id = p.payment_id AND g.payment_component_id = pc.payment_component_id AND g.claim_version = p.claim_version
# MAGIC   WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND g.payment_id IS NULL
# MAGIC ),
# MAGIC combined_payments_cte AS (
# MAGIC   SELECT * FROM guid_join_cte UNION SELECT * FROM invoice_join_cte UNION SELECT * FROM paymentid_join_cte
# MAGIC ),
# MAGIC combined_payments_with_parent AS (
# MAGIC   SELECT cp.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent
# MAGIC   FROM combined_payments_cte cp LEFT JOIN category_mapping cm ON cp.payment_category = cm.payment_category
# MAGIC ),
# MAGIC payments_with_flag AS (
# MAGIC   SELECT *, CASE
# MAGIC       WHEN TRIM(status) IN ('Paid','MarkedAsPaid') AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC       WHEN TRIM(status) IN ('Reversed','Approved') AND TRIM(type) = 'RecoveryReceipt' THEN -1
# MAGIC       WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC       WHEN TRIM(status) = 'Cancelled' AND TRIM(type) = 'RecoveryReceipt' THEN 0
# MAGIC       WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePaymentReversal' THEN -1
# MAGIC       ELSE 0
# MAGIC     END AS net_incurred_claims_flag
# MAGIC   FROM combined_payments_with_parent
# MAGIC ),
# MAGIC payments_base AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT *, ROW_NUMBER() OVER (PARTITION BY payment_guid, claim_id, status, gross_amount, payment_category ORDER BY transaction_date ASC) AS rn
# MAGIC     FROM payments_with_flag
# MAGIC   )
# MAGIC   WHERE rn = 1
# MAGIC )
# MAGIC SELECT 'payments_base' AS source_cte, * FROM payments_base;
# MAGIC  
# MAGIC ---=============================================================================
# MAGIC --- DEBUG STEP 9: movements_cte
# MAGIC ---=============================================================================
# MAGIC WITH payments_base AS (
# MAGIC   -- This CTE is a condensed version of all prior payment steps
# MAGIC   SELECT * FROM (
# MAGIC     SELECT *, ROW_NUMBER() OVER (PARTITION BY payment_guid, claim_id, status, gross_amount, payment_category ORDER BY transaction_date ASC) AS rn
# MAGIC     FROM (
# MAGIC       SELECT cp.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent,
# MAGIC       CASE
# MAGIC         WHEN TRIM(status) IN ('Paid','MarkedAsPaid') AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC         WHEN TRIM(status) IN ('Reversed','Approved') AND TRIM(type) = 'RecoveryReceipt' THEN -1
# MAGIC         WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePayment' THEN 1
# MAGIC         WHEN TRIM(status) = 'Cancelled' AND TRIM(type) = 'RecoveryReceipt' THEN 0
# MAGIC         WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePaymentReversal' THEN -1
# MAGIC         ELSE 0
# MAGIC       END AS net_incurred_claims_flag
# MAGIC       FROM (
# MAGIC           SELECT * FROM (
# MAGIC               SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC               FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC               WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC           )
# MAGIC           UNION
# MAGIC           SELECT * FROM (
# MAGIC               SELECT DISTINCT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC               FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC               WHERE p.claim_id IN (1901162, 1894827, 1789444) AND p.payment_reference IS NOT NULL AND NOT EXISTS (
# MAGIC                 SELECT 1 FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id = p.claim_id AND p_guid.payment_id = p.payment_id AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC               )
# MAGIC           )
# MAGIC           UNION
# MAGIC           SELECT * FROM (
# MAGIC             SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC             FROM prod_adp_certified.claim.payment p INNER JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC             WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND NOT EXISTS (
# MAGIC                 SELECT 1 FROM (
# MAGIC                     SELECT p_guid.payment_id, p_guid.claim_version, pc_guid.payment_component_id FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id IN (1901162, 1894827, 1789444) AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC                     UNION
# MAGIC                     SELECT p_inv.payment_id, p_inv.claim_version, pc_inv.payment_component_id FROM prod_adp_certified.claim.payment p_inv JOIN prod_adp_certified.claim.payment_component pc_inv ON p_inv.payment_reference = pc_inv.invoice_number AND p_inv.event_identity = pc_inv.event_identity AND p_inv.claim_version = pc_inv.claim_version WHERE p_inv.claim_id IN (1901162, 1894827, 1789444) AND p_inv.payment_reference IS NOT NULL
# MAGIC                 ) ex WHERE ex.payment_id = p.payment_id AND ex.claim_version = p.claim_version AND ex.payment_component_id = pc.payment_component_id
# MAGIC             )
# MAGIC           )
# MAGIC       ) cp LEFT JOIN category_mapping cm ON cp.payment_category = cm.payment_category
# MAGIC     )
# MAGIC   ) WHERE rn = 1
# MAGIC ),
# MAGIC reserves_cte AS (
# MAGIC   SELECT DISTINCT vi.claim_id, cv.claim_version_id, vi.claim_version_item_id, vi.claim_version_item_index, r.head_of_damage,
# MAGIC     r.category_data_description, r.reserve_guid, r.reserve_value, CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value, r.type,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id ORDER BY CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END, r.reserve_value) AS rn
# MAGIC   FROM prod_adp_certified.claim.reserve r
# MAGIC   INNER JOIN prod_adp_certified.claim.claim_version cv ON r.event_identity = cv.event_identity
# MAGIC   INNER JOIN prod_adp_certified.claim.claim_version_item vi ON vi.event_identity = r.event_identity AND vi.claim_version_item_index = r.claim_version_item_index
# MAGIC   WHERE vi.claim_id IN (1901162, 1894827, 1789444)
# MAGIC ),
# MAGIC reserves_filtered AS ( SELECT * FROM reserves_cte WHERE rn = 1 ),
# MAGIC payments_agg_cte AS (
# MAGIC   SELECT pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index,
# MAGIC     SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount,
# MAGIC     SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount
# MAGIC   FROM payments_base pb LEFT JOIN prod_adp_certified.claim.payment p ON pb.payment_id = p.payment_id AND pb.payment_guid = p.payment_guid
# MAGIC   WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt')
# MAGIC   GROUP BY pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index
# MAGIC ),
# MAGIC reserves_adjusted AS (
# MAGIC   SELECT DISTINCT r.claim_id, r.claim_version_id, r.head_of_damage, r.category_data_description, r.reserve_guid, r.expected_recovery_value, r.reserve_value,
# MAGIC     COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value,
# MAGIC     COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value
# MAGIC   FROM reserves_filtered r LEFT JOIN payments_agg_cte p ON p.claim_id = r.claim_id AND p.claim_version = r.claim_version_id AND p.payment_category = r.category_data_description AND p.claim_version_item_index = r.claim_version_item_index
# MAGIC ),
# MAGIC reserves_with_parent AS (
# MAGIC   SELECT ra.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent
# MAGIC   FROM reserves_adjusted ra LEFT JOIN category_mapping cm ON ra.category_data_description = cm.payment_category
# MAGIC ),
# MAGIC -- Enhanced movements_cte with proper temporal handling
# MAGIC movements_cte AS (
# MAGIC  WITH ordered_reserves AS (
# MAGIC    SELECT
# MAGIC      *,
# MAGIC      -- Get the version sequence for each reserve
# MAGIC      DENSE_RANK() OVER (
# MAGIC        PARTITION BY claim_id, reserve_guid
# MAGIC        ORDER BY claim_version_id
# MAGIC      ) AS version_seq,
# MAGIC      -- Check if this is the first appearance of this reserve
# MAGIC      ROW_NUMBER() OVER (
# MAGIC        PARTITION BY claim_id, reserve_guid
# MAGIC        ORDER BY claim_version_id
# MAGIC      ) AS reserve_instance
# MAGIC    FROM reserves_with_parent
# MAGIC  )
# MAGIC  SELECT
# MAGIC    claim_id,
# MAGIC    claim_version_id,
# MAGIC    head_of_damage,
# MAGIC    category_data_description,
# MAGIC    reserve_guid,
# MAGIC    expected_recovery_value,
# MAGIC    reserve_value,
# MAGIC    finance_adjusted_reserve_value,
# MAGIC    finance_adjusted_recovery_value,
# MAGIC    skyfire_parent,
# MAGIC    version_seq,
# MAGIC    reserve_instance,
# MAGIC    -- Calculate reserve movement
# MAGIC    CASE
# MAGIC      -- First instance: movement is the adjusted value itself
# MAGIC      WHEN reserve_instance = 1
# MAGIC      THEN finance_adjusted_reserve_value
# MAGIC      -- Subsequent: calculate the change
# MAGIC      ELSE finance_adjusted_reserve_value -
# MAGIC           LAG(finance_adjusted_reserve_value, 1) OVER (
# MAGIC             PARTITION BY claim_id, reserve_guid
# MAGIC             ORDER BY claim_version_id
# MAGIC           )
# MAGIC    END AS reserve_movement,
# MAGIC    -- Calculate recovery reserve movement
# MAGIC    CASE
# MAGIC      -- First instance: movement is the negative of adjusted value
# MAGIC      WHEN reserve_instance = 1
# MAGIC      THEN finance_adjusted_recovery_value * -1
# MAGIC      -- Subsequent: calculate the change
# MAGIC      ELSE (finance_adjusted_recovery_value -
# MAGIC            LAG(finance_adjusted_recovery_value, 1) OVER (
# MAGIC              PARTITION BY claim_id, reserve_guid
# MAGIC              ORDER BY claim_version_id
# MAGIC            )) * -1
# MAGIC    END AS recovery_reserve_movement
# MAGIC  FROM ordered_reserves
# MAGIC )
# MAGIC SELECT 'movements_cte' AS source_cte, * FROM movements_cte;
# MAGIC
# MAGIC  
# MAGIC ---=============================================================================
# MAGIC --- FINAL SELECT - RESULTS FOR TEST CASES
# MAGIC ---=============================================================================
# MAGIC WITH category_mapping AS (
# MAGIC   SELECT * FROM VALUES
# MAGIC     ('TP Intervention (Mobility)', 'TP_VEHICLE'), ('TP Intervention (Vehicle)', 'TP_VEHICLE'), ('TP Credit Hire (Vehicle)', 'TP_VEHICLE'), ('TP Authorised Hire (Vehicle)', 'TP_VEHICLE'), ('TP Fees (Vehicle)', 'TP_VEHICLE'), ('Ex-Gratia (Vehicle)', 'TP_VEHICLE'), ('TP Authorized Hire Vehicle', 'TP_VEHICLE'), ('Medical Expenses', 'TP_PROPERTY'), ('Legal Expenses', 'TP_PROPERTY'), ('TP Fees (Property)', 'TP_PROPERTY'), ('TP Credit Hire (Property)', 'TP_PROPERTY'), ('TP Authorised Hire (Property)', 'TP_PROPERTY'), ('TP Intervention (Property)', 'TP_PROPERTY'), ('Unknown', 'TP_PROPERTY'), ('AD Ex-Gratia (Property)', 'TP_PROPERTY'), ('Fire Ex-Gratia (Property)', 'TP_PROPERTY'), ('OD Reinsurance', 'TP_PROPERTY'), ('Theft Ex-Gratia (Property)', 'TP_PROPERTY'), ('TP Damage (Property)', 'TP_PROPERTY'), ('TP Authorized Hire Property', 'TP_PROPERTY'), ('TP Intervention (Uninsured Loss)', 'TP_PROPERTY'), ('TP Intervention (Fees)', 'TP_PROPERTY'), ('TP Damage (Vehicle)', 'TP_PROPERTY')
# MAGIC   AS t(payment_category, skyfire_parent)
# MAGIC ),
# MAGIC payments_base AS (
# MAGIC   SELECT * FROM (
# MAGIC     SELECT *, ROW_NUMBER() OVER (PARTITION BY payment_guid, claim_id, status, gross_amount, payment_category ORDER BY transaction_date ASC) AS rn
# MAGIC     FROM (
# MAGIC       SELECT cp.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent,
# MAGIC         CASE WHEN TRIM(status) IN ('Paid','MarkedAsPaid') AND TRIM(type) = 'ReservePayment' THEN 1 WHEN TRIM(status) IN ('Reversed','Approved') AND TRIM(type) = 'RecoveryReceipt' THEN -1 WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePayment' THEN 1 WHEN TRIM(status) = 'Cancelled' AND TRIM(type) = 'RecoveryReceipt' THEN 0 WHEN TRIM(status) = 'Reversed' AND TRIM(type) = 'ReservePaymentReversal' THEN -1 ELSE 0 END AS net_incurred_claims_flag
# MAGIC       FROM (
# MAGIC         SELECT * FROM (
# MAGIC           SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_guid = pc.payment_guid AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_guid <> '00000000-0000-0000-0000-000000000000'
# MAGIC         ) UNION
# MAGIC         SELECT * FROM (
# MAGIC           SELECT DISTINCT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_reference = pc.invoice_number AND p.event_identity = pc.event_identity AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND p.payment_reference IS NOT NULL AND NOT EXISTS (SELECT 1 FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id = p.claim_id AND p_guid.payment_id = p.payment_id AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000')
# MAGIC         ) UNION
# MAGIC         SELECT * FROM (
# MAGIC           SELECT p.payment_id, p.payment_reference, p.payment_guid, p.claim_version, p.claim_version_item_index, p.event_identity, p.head_of_damage, p.status, p.type, p.transaction_date, p.claim_id, pc.payment_component_id, pc.payment_category, pc.net_amount, pc.gross_amount
# MAGIC           FROM prod_adp_certified.claim.payment p JOIN prod_adp_certified.claim.payment_component pc ON p.payment_id = pc.payment_id AND p.claim_version = pc.claim_version
# MAGIC           WHERE p.claim_id IN (1901162, 1894827, 1789444) AND pc.payment_id != 0 AND NOT EXISTS (SELECT 1 FROM (SELECT p_guid.payment_id, p_guid.claim_version, pc_guid.payment_component_id FROM prod_adp_certified.claim.payment p_guid JOIN prod_adp_certified.claim.payment_component pc_guid ON p_guid.payment_guid = pc_guid.payment_guid AND p_guid.claim_version = pc_guid.claim_version WHERE p_guid.claim_id IN (1901162, 1894827, 1789444) AND pc_guid.payment_guid <> '00000000-0000-0000-0000-000000000000' UNION SELECT p_inv.payment_id, p_inv.claim_version, pc_inv.payment_component_id FROM prod_adp_certified.claim.payment p_inv JOIN prod_adp_certified.claim.payment_component pc_inv ON p_inv.payment_reference = pc_inv.invoice_number AND p_inv.event_identity = pc_inv.event_identity AND p_inv.claim_version = pc_inv.claim_version WHERE p_inv.claim_id IN (1901162, 1894827, 1789444) AND p_inv.payment_reference IS NOT NULL) ex WHERE ex.payment_id = p.payment_id AND ex.claim_version = p.claim_version AND ex.payment_component_id = pc.payment_component_id)
# MAGIC         )
# MAGIC       ) cp
# MAGIC       LEFT JOIN category_mapping cm ON cp.payment_category = cm.payment_category
# MAGIC     )
# MAGIC   ) WHERE rn = 1
# MAGIC ),
# MAGIC paid_tppd_agg AS (
# MAGIC   SELECT claim_id, SUM(paid_tot_tppd) AS paid_tot_tppd FROM (
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN CASE WHEN TRIM(status) = 'Approved' THEN gross_amount * net_incurred_claims_flag ELSE gross_amount END ELSE 0 END AS paid_tot_tppd FROM payments_base WHERE TRIM(type) = 'RecoveryReceipt' AND TRIM(status) IN ('Approved', 'Cancelled')
# MAGIC     UNION ALL
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN CASE WHEN TRIM(status) = 'Reversed' THEN gross_amount * -1 ELSE gross_amount * net_incurred_claims_flag END ELSE 0 END AS paid_tot_tppd FROM payments_base WHERE TRIM(type) = 'ReservePayment' AND gross_amount <> 0
# MAGIC   ) GROUP BY claim_id
# MAGIC ),
# MAGIC reserve_tppd_agg AS (
# MAGIC   SELECT claim_id, SUM(reserve_tppd) AS reserve_tppd FROM (
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN reserve_movement ELSE 0 END AS reserve_tppd
# MAGIC     FROM (SELECT *, COALESCE(finance_adjusted_reserve_value, 0) - LAG(COALESCE(finance_adjusted_reserve_value, 0), 1, 0) OVER (PARTITION BY claim_id, reserve_guid ORDER BY claim_version_id) AS reserve_movement FROM (SELECT ra.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent FROM (SELECT DISTINCT r.claim_id, r.claim_version_id, r.head_of_damage, r.category_data_description, r.reserve_guid, r.expected_recovery_value, r.reserve_value, COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value, COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value FROM (SELECT * FROM (SELECT DISTINCT vi.claim_id, cv.claim_version_id, vi.claim_version_item_id, vi.claim_version_item_index, r.head_of_damage, r.category_data_description, r.reserve_guid, r.reserve_value, CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value, r.type, ROW_NUMBER() OVER (PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id ORDER BY CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END, r.reserve_value) AS rn FROM prod_adp_certified.claim.reserve r JOIN prod_adp_certified.claim.claim_version cv ON r.event_identity = cv.event_identity JOIN prod_adp_certified.claim.claim_version_item vi ON vi.event_identity = r.event_identity AND vi.claim_version_item_index = r.claim_version_item_index WHERE vi.claim_id IN (1901162, 1894827, 1789444)) WHERE rn = 1) r LEFT JOIN (SELECT pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index, SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount, SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount FROM payments_base pb LEFT JOIN prod_adp_certified.claim.payment p ON pb.payment_id = p.payment_id AND pb.payment_guid = p.payment_guid WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt') GROUP BY pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index) p ON p.claim_id = r.claim_id AND p.claim_version = r.claim_version_id AND p.payment_category = r.category_data_description AND p.claim_version_item_index = r.claim_version_item_index) ra LEFT JOIN category_mapping cm ON ra.category_data_description = cm.payment_category)) WHERE reserve_movement <> 0
# MAGIC     UNION ALL
# MAGIC     SELECT claim_id, CASE WHEN skyfire_parent IN ('TP_PROPERTY', 'TP_VEHICLE') THEN recovery_reserve_movement ELSE 0 END AS reserve_tppd
# MAGIC     FROM (SELECT *, (COALESCE(finance_adjusted_recovery_value, 0) - LAG(COALESCE(finance_adjusted_recovery_value, 0), 1, 0) OVER (PARTITION BY claim_id, reserve_guid ORDER BY claim_version_id)) * -1 AS recovery_reserve_movement FROM (SELECT ra.*, COALESCE(cm.skyfire_parent, 'OTHER') AS skyfire_parent FROM (SELECT DISTINCT r.claim_id, r.claim_version_id, r.head_of_damage, r.category_data_description, r.reserve_guid, r.expected_recovery_value, r.reserve_value, COALESCE(r.reserve_value, 0) - COALESCE(p.finance_reserve_gross_amount, 0) AS finance_adjusted_reserve_value, COALESCE(r.expected_recovery_value, 0) - COALESCE(p.finance_recovery_gross_amount, 0) AS finance_adjusted_recovery_value FROM (SELECT * FROM (SELECT DISTINCT vi.claim_id, cv.claim_version_id, vi.claim_version_item_id, vi.claim_version_item_index, r.head_of_damage, r.category_data_description, r.reserve_guid, r.reserve_value, CAST(r.expected_recovery_value AS DECIMAL(10, 2)) AS expected_recovery_value, r.type, ROW_NUMBER() OVER (PARTITION BY r.reserve_guid, vi.claim_id, cv.claim_version_id ORDER BY CASE WHEN vi.claim_version_item_id <> 0 THEN 0 ELSE 1 END, r.reserve_value) AS rn FROM prod_adp_certified.claim.reserve r JOIN prod_adp_certified.claim.claim_version cv ON r.event_identity = cv.event_identity JOIN prod_adp_certified.claim.claim_version_item vi ON vi.event_identity = r.event_identity AND vi.claim_version_item_index = r.claim_version_item_index WHERE vi.claim_id IN (1901162, 1894827, 1789444)) WHERE rn = 1) r LEFT JOIN (SELECT pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index, SUM(CASE WHEN pb.type = 'ReservePayment' AND pb.status IN ('MarkedAsPaid', 'Paid') THEN pb.gross_amount ELSE 0 END) AS finance_reserve_gross_amount, SUM(CASE WHEN pb.type = 'RecoveryReceipt' AND pb.status = 'Approved' THEN pb.gross_amount ELSE 0 END) AS finance_recovery_gross_amount FROM payments_base pb LEFT JOIN prod_adp_certified.claim.payment p ON pb.payment_id = p.payment_id AND pb.payment_guid = p.payment_guid WHERE pb.type IN ('ReservePayment', 'RecoveryReceipt') GROUP BY pb.claim_id, pb.claim_version, pb.payment_category, p.claim_version_item_index) p ON p.claim_id = r.claim_id AND p.claim_version = r.claim_version_id AND p.payment_category = r.category_data_description AND p.claim_version_item_index = r.claim_version_item_index) ra LEFT JOIN category_mapping cm ON ra.category_data_description = cm.payment_category)) WHERE recovery_reserve_movement <> 0
# MAGIC   ) GROUP BY claim_id
# MAGIC ),
# MAGIC base_claim_latest AS (
# MAGIC   SELECT claim_id, claim_version_id, event_identity FROM (
# MAGIC     SELECT c.claim_id, cv.claim_version_id, cv.event_identity, ROW_NUMBER() OVER (PARTITION BY c.claim_id ORDER BY cv.claim_version_id DESC) AS rn
# MAGIC     FROM prod_adp_certified.claim.claim c INNER JOIN prod_adp_certified.claim.claim_version cv ON c.claim_id = cv.claim_id
# MAGIC     WHERE c.claim_id IN (1901162, 1894827, 1789444)
# MAGIC   ) WHERE rn = 1
# MAGIC ),
# MAGIC result_with_tppd AS (
# MAGIC   SELECT bc.claim_id, bc.claim_version_id, COALESCE(pt.paid_tot_tppd, 0) AS paid_tot_tppd, COALESCE(rt.reserve_tppd, 0) AS reserve_tppd,
# MAGIC     COALESCE(pt.paid_tot_tppd, 0) + COALESCE(rt.reserve_tppd, 0) AS inc_tot_tppd
# MAGIC   FROM base_claim_latest bc
# MAGIC   LEFT JOIN paid_tppd_agg pt ON bc.claim_id = pt.claim_id
# MAGIC   LEFT JOIN reserve_tppd_agg rt ON bc.claim_id = rt.claim_id
# MAGIC ),
# MAGIC payment_category_ref AS (
# MAGIC   SELECT DISTINCT claim_id, FIRST_VALUE(payment_category) OVER (PARTITION BY claim_id ORDER BY payment_id) AS payment_category
# MAGIC   FROM payments_base
# MAGIC )
# MAGIC SELECT
# MAGIC   r.claim_id,
# MAGIC   r.claim_version_id,
# MAGIC   pcr.payment_category,
# MAGIC   r.paid_tot_tppd,
# MAGIC   r.reserve_tppd,
# MAGIC   r.inc_tot_tppd
# MAGIC FROM result_with_tppd r
# MAGIC LEFT JOIN payment_category_ref pcr ON r.claim_id = pcr.claim_id
# MAGIC ORDER BY r.claim_id;
# MAGIC  
# MAGIC  