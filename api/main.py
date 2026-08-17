"""
RiskLens FastAPI service.

Exposes fraud detection, credit risk, and composite risk scoring as a REST
API, plus lookups against the SQLite-backed transaction/customer tables and
model performance metrics for the monitoring page.

Run locally:
    uvicorn api.main:app --reload --port 8000

Docs available at http://localhost:8000/docs
"""
import json
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.data.feature_engineering import engineer_fraud_features, engineer_credit_features
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.risk_copilot import explain_risk
from src.risk_engine.risk_score import RiskInputs, compute_final_risk, transaction_risk_heuristic
from src.models.credit_model import risk_band as credit_risk_band
from api.schemas import (
    FraudPredictRequest,
    FraudPredictResponse,
    CreditPredictRequest,
    CreditPredictResponse,
    CompositeRiskRequest,
    CompositeRiskResponse,
    RiskFactor,
)

app = FastAPI(
    title="RiskLens API",
    description="AI-powered credit & fraud risk intelligence platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load models once at startup ---
_fraud_bundle = joblib.load(config.FRAUD_MODEL_PATH)
_credit_bundle = joblib.load(config.CREDIT_MODEL_PATH)
_anomaly_bundle = joblib.load(config.ANOMALY_MODEL_PATH)

_fraud_explainer = ShapExplainer(config.FRAUD_MODEL_PATH)
_credit_explainer = ShapExplainer(config.CREDIT_MODEL_PATH)

_fraud_amount_p95 = None  # computed lazily on first request


def _get_amount_p95() -> float:
    global _fraud_amount_p95
    if _fraud_amount_p95 is None:
        df = pd.read_csv(config.FRAUD_PROCESSED_CSV, usecols=["Amount"])
        _fraud_amount_p95 = float(df["Amount"].quantile(0.95))
    return _fraud_amount_p95


def _fraud_row_to_features(payload: FraudPredictRequest) -> dict:
    row = payload.model_dump()
    hour = (row["Time"] // 3600) % 24
    row["Hour"] = hour
    row["is_off_peak_hour"] = int(2 <= hour <= 5)
    row["amount_log"] = __import__("numpy").log1p(row["Amount"])
    row["is_micro_transaction"] = int(row["Amount"] < 5)
    row["is_large_transaction"] = int(row["Amount"] > _get_amount_p95())
    return row


def _predict_fraud(payload: FraudPredictRequest) -> FraudPredictResponse:
    row = _fraud_row_to_features(payload)
    feature_cols = _fraud_bundle["feature_cols"]
    X = pd.DataFrame([row])[feature_cols]

    model = _fraud_bundle["model"]
    proba = float(model.predict_proba(X)[0][1])
    score = round(proba * 100, 2)
    level = "CRITICAL" if score >= 75 else "HIGH" if score >= 50 else "MEDIUM" if score >= 20 else "LOW"

    factors = _fraud_explainer.explain(row)
    explanation = explain_risk("fraud", "this transaction", proba, level, factors)

    return FraudPredictResponse(
        fraud_probability=round(proba, 4),
        risk_score=score,
        risk_level=level,
        top_risk_factors=[RiskFactor(**f) for f in factors],
        explanation=explanation,
    )


def _credit_row_to_features(payload: CreditPredictRequest) -> dict:
    row = payload.model_dump(by_alias=True)
    df = pd.DataFrame([row])
    df["MonthlyIncome_was_missing"] = 0
    df["NumberOfDependents_was_missing"] = 0
    df = engineer_credit_features(df).drop(columns=["age_band"])
    return df.iloc[0].to_dict()


def _predict_credit(payload: CreditPredictRequest) -> CreditPredictResponse:
    row = _credit_row_to_features(payload)
    feature_cols = _credit_bundle["feature_cols"]
    X = pd.DataFrame([row])[feature_cols]

    model = _credit_bundle["model"]
    proba = float(model.predict_proba(X)[0][1])
    score = round(proba * 100, 2)
    level = credit_risk_band(proba)

    factors = _credit_explainer.explain(row)
    explanation = explain_risk("credit", "this customer", proba, level, factors)

    return CreditPredictResponse(
        default_probability=round(proba, 4),
        risk_score=score,
        risk_level=level,
        top_risk_factors=[RiskFactor(**f) for f in factors],
        explanation=explanation,
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/")
def root():
    return {"service": "RiskLens API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict/fraud", response_model=FraudPredictResponse)
def predict_fraud(payload: FraudPredictRequest):
    return _predict_fraud(payload)


@app.post("/predict/credit", response_model=CreditPredictResponse)
def predict_credit(payload: CreditPredictRequest):
    return _predict_credit(payload)


@app.post("/risk/composite", response_model=CompositeRiskResponse)
def composite_risk(payload: CompositeRiskRequest):
    result = compute_final_risk(RiskInputs(**payload.model_dump()))
    return CompositeRiskResponse(**result)


@app.get("/transaction/{transaction_id}")
def get_transaction(transaction_id: int):
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM fraud_transactions WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    row_dict = dict(row)
    feature_cols = _fraud_bundle["feature_cols"]
    X = pd.DataFrame([row_dict])[feature_cols]
    model = _fraud_bundle["model"]
    proba = float(model.predict_proba(X)[0][1])
    score = round(proba * 100, 2)
    level = "CRITICAL" if score >= 75 else "HIGH" if score >= 50 else "MEDIUM" if score >= 20 else "LOW"
    factors = _fraud_explainer.explain(row_dict)

    return {
        "transaction_id": transaction_id,
        "amount": row_dict["Amount"],
        "actual_label": row_dict["Class"],
        "fraud_probability": round(proba, 4),
        "risk_score": score,
        "risk_level": level,
        "top_risk_factors": factors,
    }


@app.get("/customer/{customer_id}")
def get_customer(customer_id: int):
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM credit_customers WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    row_dict = dict(row)
    feature_cols = _credit_bundle["feature_cols"]
    X = pd.DataFrame([row_dict])[feature_cols]
    model = _credit_bundle["model"]
    proba = float(model.predict_proba(X)[0][1])
    score = round(proba * 100, 2)
    level = credit_risk_band(proba)
    factors = _credit_explainer.explain(row_dict)
    explanation = explain_risk("credit", f"C{customer_id}", proba, level, factors)

    return {
        "customer_id": customer_id,
        "default_probability": round(proba, 4),
        "risk_score": score,
        "risk_level": level,
        "top_risk_factors": factors,
        "explanation": explanation,
    }


@app.get("/model/metrics")
def model_metrics():
    with open(config.FRAUD_METRICS_PATH) as f:
        fraud_metrics = json.load(f)
    with open(config.CREDIT_METRICS_PATH) as f:
        credit_metrics = json.load(f)
    return {"fraud_model": fraud_metrics, "credit_model": credit_metrics}
