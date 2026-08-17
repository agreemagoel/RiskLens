
"""
Export a model-scored dataset for the Power BI executive dashboard.

Runs both trained models over the full processed datasets and writes a
single flat CSV with risk scores, bands, and key attributes -- exactly the
shape Power BI wants for building visuals without needing any Python
runtime in Power BI itself.

Usage:
    python -m powerbi.export_for_powerbi
"""
import joblib
import pandas as pd

from src import config
from src.data.feature_engineering import engineer_fraud_features, engineer_credit_features
from src.models.credit_model import risk_band as credit_risk_band


def export_fraud_for_powerbi() -> pd.DataFrame:
    df = pd.read_csv(config.FRAUD_PROCESSED_CSV)
    df = engineer_fraud_features(df)

    bundle = joblib.load(config.FRAUD_MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    X = df[feature_cols]
    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["fraud_risk_score"] = (df["fraud_probability"] * 100).round(2)
    df["fraud_risk_level"] = pd.cut(
        df["fraud_risk_score"], bins=[-0.01, 20, 50, 75, 100.01],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )
    df["record_type"] = "transaction"
    df = df.reset_index().rename(columns={"index": "record_id"})

    out_cols = [
        "record_id", "record_type", "Amount", "Hour", "is_off_peak_hour",
        "Class", "fraud_probability", "fraud_risk_score", "fraud_risk_level",
    ]
    return df[out_cols]


def export_credit_for_powerbi() -> pd.DataFrame:
    df = pd.read_csv(config.CREDIT_PROCESSED_CSV, index_col=0)
    df = engineer_credit_features(df).drop(columns=["age_band"])

    bundle = joblib.load(config.CREDIT_MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    X = df[feature_cols]
    df["default_probability"] = model.predict_proba(X)[:, 1]
    df["credit_risk_score"] = (df["default_probability"] * 100).round(2)
    df["credit_risk_level"] = df["default_probability"].apply(credit_risk_band)
    df["record_type"] = "customer"
    df = df.reset_index().rename(columns={"index": "record_id"})

    out_cols = [
        "record_id", "record_type", "age", "MonthlyIncome",
        "RevolvingUtilizationOfUnsecuredLines", "total_delinquencies",
        "NumberOfDependents", "SeriousDlqin2yrs", "default_probability",
        "credit_risk_score", "credit_risk_level",
    ]
    return df[out_cols]


def main():
    fraud_export = export_fraud_for_powerbi()
    credit_export = export_credit_for_powerbi()

    fraud_path = config.DATA_PROCESSED_DIR / "powerbi_fraud_dataset.csv"
    credit_path = config.DATA_PROCESSED_DIR / "powerbi_credit_dataset.csv"

    fraud_export.to_csv(fraud_path, index=False)
    credit_export.to_csv(credit_path, index=False)

    print(f"Fraud export: {fraud_export.shape} -> {fraud_path}")
    print(f"Credit export: {credit_export.shape} -> {credit_path}")


if __name__ == "__main__":
    main()
