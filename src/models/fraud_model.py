"""
Fraud detection model.

Compares Logistic Regression, Random Forest, and XGBoost. Fraud is heavily
imbalanced (0.17% positive class per EDA), so we evaluate with Precision,
Recall, F1, ROC-AUC, and PR-AUC -- never accuracy alone -- and use
scale_pos_weight / class_weight to counter the imbalance instead of naive
resampling (keeps the full, real transaction distribution for evaluation).
"""
import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data.feature_engineering import engineer_fraud_features

FEATURE_COLS = None  # set at train time, saved alongside the model


def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.FRAUD_PROCESSED_CSV)
    return engineer_fraud_features(df)


def _split(df: pd.DataFrame):
    y = df[config.FRAUD_TARGET_COL]
    X = df.drop(columns=[config.FRAUD_TARGET_COL, "Time"])
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )


def _evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": model_name,
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
    }
    return metrics


def train_and_compare() -> dict:
    """Train Logistic Regression, Random Forest, XGBoost; log all to MLflow;
    persist the best (by PR-AUC, the right metric for severe imbalance) as
    the production fraud model.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    df = _load_dataset()
    X_train, X_test, y_train, y_test = _split(df)

    global FEATURE_COLS
    FEATURE_COLS = list(X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=fraud_ratio, eval_metric="aucpr",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"fraud_{name}"):
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

    # Best model by PR-AUC (correct metric for extreme imbalance)
    best = max(results, key=lambda r: r["pr_auc"])
    best_name = best["model"]
    best_model = fitted_models[best_name]

    print(f"\nBest model: {best_name} (PR-AUC={best['pr_auc']})")

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "scaler": scaler if best_name == "logistic_regression" else None,
            "feature_cols": FEATURE_COLS,
            "uses_scaler": best_name == "logistic_regression",
        },
        config.FRAUD_MODEL_PATH,
    )

    with open(config.FRAUD_METRICS_PATH, "w") as f:
        json.dump({"comparison": results, "best_model": best_name}, f, indent=2)

    return {"comparison": results, "best_model": best_name}


if __name__ == "__main__":
    train_and_compare()
