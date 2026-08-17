"""
Credit default risk model.

Predicts probability of "serious delinquency" (default) within 2 years.
Moderate imbalance (~6.68% positive per EDA) -> still avoid raw accuracy,
report Precision/Recall/F1/ROC-AUC. Output probability is bucketed into
Low/Medium/High/Critical risk bands for downstream use in the risk engine
and dashboards.
"""
import json
import time

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data.feature_engineering import engineer_credit_features


def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.CREDIT_PROCESSED_CSV, index_col=0)
    df = engineer_credit_features(df)
    # age_band is used to build one-hot columns already in feature engineering;
    # drop the categorical original so the model only sees numeric columns.
    return df.drop(columns=["age_band"])


def _split(df: pd.DataFrame):
    y = df[config.CREDIT_TARGET_COL]
    X = df.drop(columns=[config.CREDIT_TARGET_COL])
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )


def risk_band(probability: float) -> str:
    pct = probability * 100
    for low, high, label in config.RISK_BANDS:
        if low <= pct < high:
            return label
    return "CRITICAL"


def _evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": model_name,
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def train_and_compare() -> dict:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    df = _load_dataset()
    X_train, X_test, y_train, y_test = _split(df)
    feature_cols = list(X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            scale_pos_weight=neg_pos_ratio, eval_metric="auc",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"credit_{name}"):
            start = time.time()
            if name == "logistic_regression":
                model.fit(X_train_scaled, y_train)
                metrics = _evaluate(model, X_test_scaled, y_test, name)
            else:
                model.fit(X_train, y_train)
                metrics = _evaluate(model, X_test, y_test, name)
            train_time = round(time.time() - start, 2)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
            mlflow.log_metric("train_time_seconds", train_time)

            print(f"[{name}] {metrics} (trained in {train_time}s)")
            results.append(metrics)
            fitted_models[name] = model

    best = max(results, key=lambda r: r["roc_auc"])
    best_name = best["model"]
    best_model = fitted_models[best_name]

    print(f"\nBest model: {best_name} (ROC-AUC={best['roc_auc']})")

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "scaler": scaler if best_name == "logistic_regression" else None,
            "feature_cols": feature_cols,
            "uses_scaler": best_name == "logistic_regression",
        },
        config.CREDIT_MODEL_PATH,
    )

    with open(config.CREDIT_METRICS_PATH, "w") as f:
        json.dump({"comparison": results, "best_model": best_name}, f, indent=2)

    return {"comparison": results, "best_model": best_name}


if __name__ == "__main__":
    train_and_compare()
