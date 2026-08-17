"""
Drift monitoring using Evidently.

Compares a "reference" window (e.g. training-time data) against a "current"
window (e.g. this month's transactions) and reports which features have
drifted. This is the piece most student projects skip -- it's what turns
RiskLens from a one-off model into something that can be talked about as an
MLOps-aware system in interviews.

Verified against evidently==0.7.21's actual Report/Dataset API (the
Evidently API changed significantly across versions, so this module avoids
guessing and uses the confirmed working call shape).
"""
import json
from datetime import datetime

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from src import config

DRIFT_REPORT_JSON = config.DATA_PROCESSED_DIR / "drift_report.json"


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict:
    """Run an Evidently data drift report and return a compact summary dict
    plus save the full report to disk as JSON.
    """
    if columns:
        reference_df = reference_df[columns]
        current_df = current_df[columns]

    definition = DataDefinition()
    ref_ds = Dataset.from_pandas(reference_df, data_definition=definition)
    cur_ds = Dataset.from_pandas(current_df, data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)
    raw = result.dict()

    # Extract the overall drifted-columns-count metric + per-column drift scores
    summary = {"generated_at": datetime.utcnow().isoformat(), "columns": {}}
    for m in raw["metrics"]:
        name = m["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            summary["drifted_columns_count"] = m["value"]["count"]
            summary["drifted_columns_share"] = m["value"]["share"]
        elif name.startswith("ValueDrift"):
            col = m["config"]["column"]
            summary["columns"][col] = {
                "drift_score": round(float(m["value"]), 4),
                "method": m["config"]["method"],
                "threshold": m["config"]["threshold"],
                "drifted": float(m["value"]) > m["config"]["threshold"],
            }

    with open(DRIFT_REPORT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def check_fraud_drift(split_point: int = 150000) -> dict:
    """Example drift check: split the fraud dataset by Time into an earlier
    'reference' window and later 'current' window, simulating comparing
    training-time data to newly arrived transactions.
    """
    df = pd.read_csv(config.FRAUD_PROCESSED_CSV)
    monitored_cols = ["Amount", "Hour", "V1", "V2", "V3", "V4", "V10", "V14", "V17"]

    reference = df[df["Time"] <= split_point]
    current = df[df["Time"] > split_point]

    summary = run_drift_report(reference, current, columns=monitored_cols)
    return summary


if __name__ == "__main__":
    result = check_fraud_drift()
    print(json.dumps(result, indent=2))
