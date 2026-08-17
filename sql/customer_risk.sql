-- customer_risk.sql
-- Run against data/processed/risklens.db (table: credit_customers)
-- Purpose: customer-level credit risk segmentation.

-- 1. Overall default summary
SELECT
    COUNT(*)                                         AS total_customers,
    SUM(SeriousDlqin2yrs)                             AS default_count,
    ROUND(100.0 * SUM(SeriousDlqin2yrs) / COUNT(*), 2) AS default_rate_pct,
    ROUND(AVG(age), 1)                                AS avg_age,
    ROUND(AVG(MonthlyIncome), 0)                      AS avg_monthly_income
FROM credit_customers;

-- 2. Default rate by age band
SELECT
    CASE
        WHEN age < 26 THEN '18-25'
        WHEN age < 36 THEN '26-35'
        WHEN age < 46 THEN '36-45'
        WHEN age < 56 THEN '46-55'
        WHEN age < 66 THEN '56-65'
        ELSE '65+'
    END AS age_band,
    COUNT(*)                                          AS total_customers,
    SUM(SeriousDlqin2yrs)                              AS default_count,
    ROUND(100.0 * SUM(SeriousDlqin2yrs) / COUNT(*), 2)  AS default_rate_pct
FROM credit_customers
GROUP BY age_band
ORDER BY age_band;

-- 3. Default rate by delinquency history
SELECT
    CASE
        WHEN total_delinquencies = 0 THEN '0. None'
        WHEN total_delinquencies <= 2 THEN '1. Low (1-2)'
        WHEN total_delinquencies <= 5 THEN '2. Medium (3-5)'
        ELSE '3. High (6+)'
    END AS delinquency_band,
    COUNT(*)                                          AS total_customers,
    SUM(SeriousDlqin2yrs)                              AS default_count,
    ROUND(100.0 * SUM(SeriousDlqin2yrs) / COUNT(*), 2)  AS default_rate_pct
FROM credit_customers
GROUP BY delinquency_band
ORDER BY delinquency_band;

-- 4. Default rate by utilization quintile
WITH ranked AS (
    SELECT *, NTILE(5) OVER (ORDER BY RevolvingUtilizationOfUnsecuredLines) AS utilization_quintile
    FROM credit_customers
)
SELECT
    utilization_quintile,
    ROUND(MIN(RevolvingUtilizationOfUnsecuredLines), 3) AS min_utilization,
    ROUND(MAX(RevolvingUtilizationOfUnsecuredLines), 3) AS max_utilization,
    COUNT(*)                                            AS total_customers,
    SUM(SeriousDlqin2yrs)                                AS default_count,
    ROUND(100.0 * SUM(SeriousDlqin2yrs) / COUNT(*), 2)    AS default_rate_pct
FROM ranked
GROUP BY utilization_quintile
ORDER BY utilization_quintile;

-- 5. Highest-risk customers (most delinquencies + high utilization)
SELECT customer_id, age, total_delinquencies, RevolvingUtilizationOfUnsecuredLines,
       MonthlyIncome, SeriousDlqin2yrs
FROM credit_customers
WHERE has_any_delinquency = 1 AND high_utilization = 1
ORDER BY total_delinquencies DESC
LIMIT 20;
