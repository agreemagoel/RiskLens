"""
Feature engineering for RiskLens.

Adds derived features on top of the cleaned datasets from preprocessing.py.
Kept separate from preprocessing so cleaning (fixing bad data) and
engineering (creating new signal) are independently testable and reviewable.
"""
import pandas as pd
import numpy as np


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioral / time-based features to the fraud dataset.

    Rationale: EDA showed fraud rate spikes 2-5 AM and fraud amounts are
    bimodal (many very small "card-testing" transactions + some large ones).
    These features give the anomaly detector and classifier explicit signal
    for those patterns instead of relying on it to discover them implicitly.
    """
    df = df.copy()

    # Off-peak hour flag (2-5 AM showed highest fraud rate in EDA)
    df["is_off_peak_hour"] = df["Hour"].between(2, 5).astype(int)

    # Amount-based signals
    df["amount_log"] = np.log1p(df["Amount"])
    df["is_micro_transaction"] = (df["Amount"] < 5).astype(int)
    df["is_large_transaction"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

    return df


def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived risk features to the credit dataset.

    Rationale: EDA showed prior delinquency, utilization, and age band are
    the strongest linear risk drivers. These engineered features make those
    relationships more explicit for the model and for SHAP explanations.
    """
    df = df.copy()

    delinquency_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    df["total_delinquencies"] = df[delinquency_cols].sum(axis=1)
    df["has_any_delinquency"] = (df["total_delinquencies"] > 0).astype(int)

    df["high_utilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.8).astype(int)

    df["income_per_dependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    df["total_credit_lines"] = (
        df["NumberOfOpenCreditLinesAndLoans"] + df["NumberRealEstateLoansOrLines"]
    )

    df["age_band"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    # One-hot encode age band for modeling
    age_dummies = pd.get_dummies(df["age_band"], prefix="age_band", dtype=int)
    df = pd.concat([df, age_dummies], axis=1)

    return df
