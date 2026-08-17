"""
RiskLens Streamlit Dashboard -- operational risk investigation tool.

Run locally:
    streamlit run dashboard/app.py

Pages:
    1. Executive Overview   -- portfolio-level KPIs
    2. Fraud Investigation  -- look up a transaction by ID
    3. Credit Risk          -- look up a customer by ID
    4. Model Monitoring     -- metrics + drift status
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import sqlite3

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.risk_copilot import explain_risk
from src.models.credit_model import risk_band as credit_risk_band

st.set_page_config(page_title="RiskLens", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_models():
    fraud_bundle = joblib.load(config.FRAUD_MODEL_PATH)
    credit_bundle = joblib.load(config.CREDIT_MODEL_PATH)
    fraud_explainer = ShapExplainer(config.FRAUD_MODEL_PATH)
    credit_explainer = ShapExplainer(config.CREDIT_MODEL_PATH)
    return fraud_bundle, credit_bundle, fraud_explainer, credit_explainer


@st.cache_data
def load_summary_data():
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    fraud_summary = pd.read_sql(
        "SELECT Hour, COUNT(*) as total, SUM(Class) as fraud_count FROM fraud_transactions GROUP BY Hour",
        conn,
    )
    amount_bucket = pd.read_sql(
        """
        SELECT
            CASE
                WHEN Amount < 5 THEN '1. Micro (<5)'
                WHEN Amount < 50 THEN '2. Small (5-50)'
                WHEN Amount < 200 THEN '3. Medium (50-200)'
                WHEN Amount < 1000 THEN '4. Large (200-1000)'
                ELSE '5. Very Large (1000+)'
            END AS bucket,
            COUNT(*) as total, SUM(Class) as fraud_count
        FROM fraud_transactions GROUP BY bucket ORDER BY bucket
        """,
        conn,
    )
    age_band = pd.read_sql(
        """
        SELECT
            CASE
                WHEN age < 26 THEN '18-25' WHEN age < 36 THEN '26-35'
                WHEN age < 46 THEN '36-45' WHEN age < 56 THEN '46-55'
                WHEN age < 66 THEN '56-65' ELSE '65+'
            END AS age_band,
            COUNT(*) as total, SUM(SeriousDlqin2yrs) as default_count
        FROM credit_customers GROUP BY age_band ORDER BY age_band
        """,
        conn,
    )
    fraud_totals = pd.read_sql(
        "SELECT COUNT(*) as total, SUM(Class) as fraud_count, SUM(Amount) as total_amount, "
        "SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END) as fraud_amount FROM fraud_transactions",
        conn,
    ).iloc[0]
    credit_totals = pd.read_sql(
        "SELECT COUNT(*) as total, SUM(SeriousDlqin2yrs) as default_count FROM credit_customers",
        conn,
    ).iloc[0]
    conn.close()
    return fraud_summary, amount_bucket, age_band, fraud_totals, credit_totals


fraud_bundle, credit_bundle, fraud_explainer, credit_explainer = load_models()

st.sidebar.title("🛡️ RiskLens")
page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Fraud Investigation", "Credit Risk", "Model Monitoring"],
)

# ============================================================================
# PAGE 1: Executive Overview
# ============================================================================
if page == "Executive Overview":
    st.title("Executive Overview")

    fraud_summary, amount_bucket, age_band, fraud_totals, credit_totals = load_summary_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{int(fraud_totals['total']):,}")
    col2.metric("Fraud Detected", f"{int(fraud_totals['fraud_count']):,}",
                f"{fraud_totals['fraud_count']/fraud_totals['total']*100:.3f}% rate")
    col3.metric("Total Customers", f"{int(credit_totals['total']):,}")
    col4.metric("High-Risk Customers", f"{int(credit_totals['default_count']):,}",
                f"{credit_totals['default_count']/credit_totals['total']*100:.2f}% default rate")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fraud Rate by Hour of Day")
        fraud_summary["fraud_rate_pct"] = fraud_summary["fraud_count"] / fraud_summary["total"] * 100
        fig = px.line(fraud_summary, x="Hour", y="fraud_rate_pct", markers=True)
        fig.update_layout(yaxis_title="Fraud Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Fraud Rate by Transaction Size")
        amount_bucket["fraud_rate_pct"] = amount_bucket["fraud_count"] / amount_bucket["total"] * 100
        fig2 = px.bar(amount_bucket, x="bucket", y="fraud_rate_pct")
        fig2.update_layout(xaxis_title="Amount Bucket", yaxis_title="Fraud Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Default Rate by Age Band")
    age_band["default_rate_pct"] = age_band["default_count"] / age_band["total"] * 100
    fig3 = px.bar(age_band, x="age_band", y="default_rate_pct", color="default_rate_pct",
                  color_continuous_scale="Reds")
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# PAGE 2: Fraud Investigation
# ============================================================================
elif page == "Fraud Investigation":
    st.title("Fraud Investigation")
    st.caption("Look up a transaction by ID (0 - 284,806) to see its fraud risk score and SHAP explanation.")

    txn_id = st.number_input("Transaction ID", min_value=0, max_value=284806, value=541, step=1)

    if st.button("Investigate", type="primary"):
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM fraud_transactions WHERE transaction_id = ?", (txn_id,)
        ).fetchone()
        conn.close()

        if row is None:
            st.error("Transaction not found.")
        else:
            row_dict = dict(row)
            feature_cols = fraud_bundle["feature_cols"]
            X = pd.DataFrame([row_dict])[feature_cols]
            model = fraud_bundle["model"]
            proba = float(model.predict_proba(X)[0][1])
            score = round(proba * 100, 2)
            level = "CRITICAL" if score >= 75 else "HIGH" if score >= 50 else "MEDIUM" if score >= 20 else "LOW"
            factors = fraud_explainer.explain(row_dict)

            c1, c2, c3 = st.columns(3)
            c1.metric("Amount", f"₹{row_dict['Amount']:,.2f}")
            c2.metric("Fraud Probability", f"{proba*100:.1f}%")
            color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}[level]
            c3.metric("Risk Level", f"{color} {level}")

            if row_dict.get("Class") is not None:
                st.caption(f"Actual label in dataset: {'Fraud' if row_dict['Class']==1 else 'Legitimate'}")

            st.subheader("Why this score?")
            explanation = explain_risk("fraud", f"TXN-{txn_id}", proba, level, factors)
            st.code(explanation, language=None)

            factors_df = pd.DataFrame(factors)
            fig = px.bar(factors_df, x="shap_value", y="feature", orientation="h",
                         color="direction", color_discrete_map={
                             "increases_risk": "#C44E52", "decreases_risk": "#55A868"
                         })
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: Credit Risk
# ============================================================================
elif page == "Credit Risk":
    st.title("Credit Risk — Customer 360")
    st.caption("Look up a customer by ID (0 - 149,999) to see their default risk and top drivers.")

    cust_id = st.number_input("Customer ID", min_value=0, max_value=149999, value=5, step=1)

    if st.button("Look Up Customer", type="primary"):
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM credit_customers WHERE customer_id = ?", (cust_id,)
        ).fetchone()
        conn.close()

        if row is None:
            st.error("Customer not found.")
        else:
            row_dict = dict(row)
            feature_cols = credit_bundle["feature_cols"]
            X = pd.DataFrame([row_dict])[feature_cols]
            model = credit_bundle["model"]
            proba = float(model.predict_proba(X)[0][1])
            score = round(proba * 100, 2)
            level = credit_risk_band(proba)
            factors = credit_explainer.explain(row_dict)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Age", int(row_dict["age"]))
            c2.metric("Monthly Income", f"₹{row_dict['MonthlyIncome']:,.0f}")
            c3.metric("Default Probability", f"{proba*100:.1f}%")
            color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}[level]
            c4.metric("Risk Band", f"{color} {level}")

            st.subheader("Top Risk Drivers")
            explanation = explain_risk("credit", f"C{cust_id}", proba, level, factors)
            st.code(explanation, language=None)

            factors_df = pd.DataFrame(factors)
            fig = px.bar(factors_df, x="shap_value", y="feature", orientation="h",
                         color="direction", color_discrete_map={
                             "increases_risk": "#C44E52", "decreases_risk": "#55A868"
                         })
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: Model Monitoring
# ============================================================================
elif page == "Model Monitoring":
    st.title("Model Monitoring")

    tab1, tab2, tab3 = st.tabs(["Fraud Model", "Credit Model", "Data Drift"])

    with tab1:
        with open(config.FRAUD_METRICS_PATH) as f:
            fraud_metrics = json.load(f)
        st.subheader(f"Best model: {fraud_metrics['best_model']}")
        df = pd.DataFrame(fraud_metrics["comparison"])
        st.dataframe(df, use_container_width=True)
        fig = px.bar(df, x="model", y=["precision", "recall", "f1", "roc_auc", "pr_auc"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        with open(config.CREDIT_METRICS_PATH) as f:
            credit_metrics = json.load(f)
        st.subheader(f"Best model: {credit_metrics['best_model']}")
        df2 = pd.DataFrame(credit_metrics["comparison"])
        st.dataframe(df2, use_container_width=True)
        fig2 = px.bar(df2, x="model", y=["precision", "recall", "f1", "roc_auc"], barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        drift_path = config.DATA_PROCESSED_DIR / "drift_report.json"
        if drift_path.exists():
            with open(drift_path) as f:
                drift = json.load(f)
            st.metric("Drifted Columns", f"{int(drift.get('drifted_columns_count', 0))}",
                      f"{drift.get('drifted_columns_share', 0)*100:.1f}% of monitored columns")
            drift_df = pd.DataFrame([
                {"column": col, **vals} for col, vals in drift.get("columns", {}).items()
            ])
            if not drift_df.empty:
                st.dataframe(drift_df, use_container_width=True)
                fig3 = px.bar(drift_df, x="column", y="drift_score", color="drifted",
                              color_discrete_map={True: "#C44E52", False: "#55A868"})
                fig3.add_hline(y=0.1, line_dash="dash", annotation_text="drift threshold")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No drift report found. Run `python -m src.monitoring.drift` first.")
