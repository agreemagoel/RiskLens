"""
Behavioral anomaly detection for transactions using Isolation Forest.

Separate from the supervised fraud classifier: this catches transactions
that look statistically unusual even without a fraud label, e.g. a customer
who normally spends Rs.500-3,000 suddenly transacting Rs.85,000 at 3 AM.
Isolation Forest outputs an anomaly score which becomes one input to the
composite risk engine (src/risk_engine/risk_score.py).
"""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config
from src.data.feature_engineering import engineer_fraud_features

ANOMALY_FEATURES = ["Amount", "amount_log", "Hour", "is_off_peak_hour", "is_large_transaction"]


def train_anomaly_model(contamination: float = 0.0017) -> IsolationForest:
    """contamination defaults to the real fraud rate found in EDA (0.17%),
    since Isolation Forest needs an estimate of the expected outlier proportion.
    """
    df = pd.read_csv(config.FRAUD_PROCESSED_CSV)
    df = engineer_fraud_features(df)

    X = df[ANOMALY_FEATURES]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)

    joblib.dump({"model": model, "feature_cols": ANOMALY_FEATURES}, config.ANOMALY_MODEL_PATH)
    print(f"Anomaly model trained on {len(X)} rows, saved to {config.ANOMALY_MODEL_PATH}")
    return model


def anomaly_score(model, row: dict) -> float:
    """Convert Isolation Forest's raw decision_function output (higher = more
    normal) into a 0-100 anomaly score (higher = more anomalous) for the risk
    engine.
    """
    X = pd.DataFrame([row])[ANOMALY_FEATURES]
    raw_score = model.decision_function(X)[0]  # roughly -0.5 (anomalous) to 0.5 (normal)
    normalized = max(0.0, min(1.0, (0.5 - raw_score)))
    return round(normalized * 100, 2)


if __name__ == "__main__":
    train_anomaly_model()
