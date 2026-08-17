-- fraud_analysis.sql
-- Run against data/processed/risklens.db (table: fraud_transactions)
-- Purpose: executive-level fraud pattern analysis for the dashboard/report.

-- 1. Overall fraud summary
SELECT
    COUNT(*)                                        AS total_transactions,
    SUM(Class)                                       AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4)          AS fraud_rate_pct,
    ROUND(AVG(Amount), 2)                            AS avg_amount,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS total_fraud_amount
FROM fraud_transactions;

-- 2. Fraud rate by hour of day (off-peak hours 2-5 AM expected to be highest, per EDA)
SELECT
    Hour,
    COUNT(*)                                    AS total_transactions,
    SUM(Class)                                  AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4)     AS fraud_rate_pct
FROM fraud_transactions
GROUP BY Hour
ORDER BY fraud_rate_pct DESC;

-- 3. Fraud rate by transaction size bucket
SELECT
    CASE
        WHEN Amount < 5 THEN '1. Micro (<5)'
        WHEN Amount < 50 THEN '2. Small (5-50)'
        WHEN Amount < 200 THEN '3. Medium (50-200)'
        WHEN Amount < 1000 THEN '4. Large (200-1000)'
        ELSE '5. Very Large (1000+)'
    END AS amount_bucket,
    COUNT(*)                                    AS total_transactions,
    SUM(Class)                                  AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4)     AS fraud_rate_pct
FROM fraud_transactions
GROUP BY amount_bucket
ORDER BY amount_bucket;

-- 4. Off-peak vs normal-hour fraud comparison
SELECT
    is_off_peak_hour,
    COUNT(*)                                    AS total_transactions,
    SUM(Class)                                  AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4)     AS fraud_rate_pct,
    ROUND(AVG(Amount), 2)                       AS avg_amount
FROM fraud_transactions
GROUP BY is_off_peak_hour;

-- 5. Top 20 highest-value confirmed fraud transactions
SELECT transaction_id, Amount, Hour, is_off_peak_hour
FROM fraud_transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 20;
