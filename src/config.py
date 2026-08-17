"""
Central configuration for RiskLens.
All paths, filenames, and constants live here so nothing is hardcoded
across notebooks, scripts, the API, and the dashboard.
"""
from pathlib import Path

# --- Root paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"

for d in [DATA_PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Raw data files (real Kaggle datasets) ---
FRAUD_RAW_CSV = DATA_RAW_DIR / "creditcard.csv"                    # mlg-ulb/creditcardfraud
CREDIT_RAW_CSV = DATA_RAW_DIR / "cs-training.csv"                  # Give Me Some Credit

# --- Processed data ---
FRAUD_PROCESSED_CSV = DATA_PROCESSED_DIR / "fraud_processed.csv"
CREDIT_PROCESSED_CSV = DATA_PROCESSED_DIR / "credit_processed.csv"
SQLITE_DB_PATH = DATA_PROCESSED_DIR / "risklens.db"
POWERBI_EXPORT_CSV = DATA_PROCESSED_DIR / "powerbi_risk_dataset.csv"

# --- Model artifacts ---
FRAUD_MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
CREDIT_MODEL_PATH = MODELS_DIR / "credit_model.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
FRAUD_METRICS_PATH = MODELS_DIR / "fraud_metrics.json"
CREDIT_METRICS_PATH = MODELS_DIR / "credit_metrics.json"

# --- Modeling constants ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

FRAUD_TARGET_COL = "Class"
CREDIT_TARGET_COL = "SeriousDlqin2yrs"

# --- Risk engine weights ---
# Final Risk = 0.50 * Fraud Probability + 0.25 * Anomaly Score
#            + 0.15 * Customer (credit) Risk + 0.10 * Transaction Risk
RISK_WEIGHTS = {
    "fraud_probability": 0.50,
    "anomaly_score": 0.25,
    "customer_risk": 0.15,
    "transaction_risk": 0.10,
}

RISK_BANDS = [
    (0, 20, "LOW"),
    (20, 50, "MEDIUM"),
    (50, 75, "HIGH"),
    (75, 101, "CRITICAL"),
]

MLFLOW_EXPERIMENT_NAME = "RiskLens"
MLFLOW_TRACKING_URI = f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"
