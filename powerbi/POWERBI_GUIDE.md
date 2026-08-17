# Power BI Executive Dashboard — Setup Guide

RiskLens uses **two frontends for two audiences**, per the project design:
- **Streamlit** → operational risk investigation (transaction/customer lookup)
- **Power BI** → executive-level portfolio analytics (this guide)

Power BI Desktop is Windows-only proprietary software, so it can't be generated
headlessly here — this guide gets you to a finished `.pbix` in ~20 minutes using
the pre-scored CSVs already exported for you.

## 1. Data source

Two files are already generated at `data/processed/`:
- `powerbi_fraud_dataset.csv` — 284,807 scored transactions
- `powerbi_credit_dataset.csv` — 150,000 scored customers

Regenerate anytime after retraining with:
```bash
python -m powerbi.export_for_powerbi
```

## 2. Load into Power BI

1. Open Power BI Desktop → **Get Data → Text/CSV**
2. Load both CSVs as separate tables (`FraudTransactions`, `CreditCustomers`)
3. In Power Query Editor, confirm data types: `record_id` → Whole Number,
   probabilities/scores → Decimal Number, `*_risk_level` → Text
4. Close & Apply

## 3. Suggested DAX measures

```dax
Fraud Rate % = DIVIDE(SUM(FraudTransactions[Class]), COUNTROWS(FraudTransactions)) * 100

Default Rate % = DIVIDE(SUM(CreditCustomers[SeriousDlqin2yrs]), COUNTROWS(CreditCustomers)) * 100

High Risk Customers = CALCULATE(
    COUNTROWS(CreditCustomers),
    CreditCustomers[credit_risk_level] IN {"HIGH", "CRITICAL"}
)

Total Fraud Amount = CALCULATE(
    SUM(FraudTransactions[Amount]),
    FraudTransactions[Class] = 1
)

Avg Fraud Risk Score = AVERAGE(FraudTransactions[fraud_risk_score])
```

## 4. Suggested pages & visuals

**Page 1 — Portfolio Overview**
- KPI cards: Total Transactions, Fraud Rate %, Total Customers, Default Rate %
- Donut chart: `fraud_risk_level` distribution
- Donut chart: `credit_risk_level` distribution

**Page 2 — Fraud Analytics**
- Line chart: fraud rate by `Hour` (X-axis: Hour, Y-axis: Fraud Rate %)
- Bar chart: fraud count by amount bucket (add a calculated column bucketing `Amount`)
- Table: top 20 transactions by `fraud_risk_score`, filterable by `fraud_risk_level`

**Page 3 — Credit Risk Analytics**
- Bar chart: `Default Rate %` by age band (add calculated column bucketing `age`)
- Scatter plot: `RevolvingUtilizationOfUnsecuredLines` vs `default_probability`
- Table: high-risk customers (`credit_risk_level` = HIGH/CRITICAL), sortable

**Page 4 — Model Performance** (optional, load `models/fraud_metrics.json` and
`models/credit_metrics.json` manually as a small typed table, or paste the
values as a static table since they change only on retraining)
- Bar chart comparing Precision/Recall/F1/ROC-AUC across Logistic Regression,
  Random Forest, XGBoost for both models

## 5. Publish

If you have a Power BI Pro/student license: **Home → Publish** to share a live
link. Otherwise, export as PDF (**File → Export → Export to PDF**) to attach
to your resume/portfolio, or save the `.pbix` itself to the repo under
`powerbi/RiskLens.pbix` (add an exception to `.gitignore` for that one file if
you do — `.pbix` files are normally excluded since they're binary and can be
regenerated from the CSVs).
