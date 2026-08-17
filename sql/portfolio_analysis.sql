-- portfolio_analysis.sql
-- Run against data/processed/risklens.db
-- Purpose: portfolio-level view combining both risk domains for executive reporting.

-- 1. Side-by-side portfolio summary (fraud vs credit risk domains)
SELECT 'Fraud (Transactions)' AS domain,
       COUNT(*)                                   AS total_records,
       SUM(Class)                                  AS positive_cases,
       ROUND(100.0 * SUM(Class) / COUNT(*), 4)     AS positive_rate_pct
FROM fraud_transactions
UNION ALL
SELECT 'Credit (Customers)' AS domain,
       COUNT(*)                                          AS total_records,
       SUM(SeriousDlqin2yrs)                              AS positive_cases,
       ROUND(100.0 * SUM(SeriousDlqin2yrs) / COUNT(*), 4)  AS positive_rate_pct
FROM credit_customers;

-- 2. Credit customer segmentation by risk band (post-model; requires the
--    'default_probability' / 'risk_band' columns produced by
--    powerbi/export_for_powerbi.py -- run that script first to populate
--    data/processed/powerbi_risk_dataset.csv, or query credit_customers
--    directly for pre-model segmentation as below).
SELECT
    CASE
        WHEN NumberOfDependents = 0 THEN 'No dependents'
        WHEN NumberOfDependents <= 2 THEN '1-2 dependents'
        ELSE '3+ dependents'
    END AS dependents_band,
    COUNT(*)                                          AS total_customers,
    ROUND(AVG(MonthlyIncome), 0)                       AS avg_income,
    ROUND(100.0 * SUM(SeriousDlqin2yrs) / COUNT(*), 2)  AS default_rate_pct
FROM credit_customers
GROUP BY dependents_band;

-- 3. Monthly-style cohort view using Time-derived "day" bucket for fraud data
--    (dataset spans 48 hours; this simulates a day-over-day drift-style view)
SELECT
    CAST(Hour / 24 AS INTEGER) AS synthetic_day_bucket,  -- not meaningful beyond demo; real deployments use calendar date
    COUNT(*)                                    AS total_transactions,
    SUM(Class)                                  AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4)     AS fraud_rate_pct
FROM fraud_transactions
GROUP BY synthetic_day_bucket;
