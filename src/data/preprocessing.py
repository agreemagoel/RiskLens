"""
Preprocessing for RiskLens.

Handles two datasets with genuinely different data-quality issues:
- Fraud dataset: clean, PCA-anonymized, no missing values -> mainly dedup + scaling.
- Credit dataset: missing values, invalid sentinel values, extreme outliers
  -> imputation, capping, missingness flags.

Findings driving these choices are documented in notebooks/01_fraud_eda.ipynb
and notebooks/02_credit_eda.ipynb.
"""
import pandas as pd
import numpy as np

from src import config


def load_fraud_raw() -> pd.DataFrame:
    return pd.read_csv(config.FRAUD_RAW_CSV)


def load_credit_raw() -> pd.DataFrame:
    return pd.read_csv(config.CREDIT_RAW_CSV, index_col=0)


def clean_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the fraud transactions dataset.

    Decisions (see 01_fraud_eda.ipynb):
    - Keep duplicate rows: with 28 anonymized PCA features, duplicate rows can
      legitimately represent independent transactions with identical rounded
      feature values, and removing them risks discarding real fraud cases
      (only 19 of 1,081 duplicates are fraud, i.e. ~1.7% - dropping them
      would reduce our already-tiny fraud class further).
    - No missing values to handle.
    """
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    return df


def clean_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the credit risk dataset.

    Decisions (see 02_credit_eda.ipynb):
    - age == 0 is invalid -> replace with median age.
    - NumberOfTime*PastDue* columns contain sentinel codes (96/98) -> cap at 20.
    - RevolvingUtilizationOfUnsecuredLines and DebtRatio have extreme outliers
      -> cap at the 99.5th percentile rather than dropping rows, to preserve
      genuine high-risk signal.
    - MonthlyIncome missing (~19.8%) -> impute with median, add missing flag.
    - NumberOfDependents missing (~2.6%) -> impute with 0 (mode), add missing flag.
    """
    df = df.copy()

    # Missingness flags BEFORE imputation
    df["MonthlyIncome_was_missing"] = df["MonthlyIncome"].isnull().astype(int)
    df["NumberOfDependents_was_missing"] = df["NumberOfDependents"].isnull().astype(int)

    # Impute
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # Fix invalid age
    median_age = df.loc[df["age"] > 0, "age"].median()
    df.loc[df["age"] == 0, "age"] = median_age

    # Cap sentinel/outlier columns
    delinquency_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in delinquency_cols:
        df[col] = df[col].clip(upper=20)

    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        cap = df[col].quantile(0.995)
        df[col] = df[col].clip(upper=cap)

    return df


def preprocess_and_save() -> None:
    """Run full preprocessing pipeline for both datasets and persist to disk."""
    fraud_df = clean_fraud(load_fraud_raw())
    credit_df = clean_credit(load_credit_raw())

    fraud_df.to_csv(config.FRAUD_PROCESSED_CSV, index=False)
    credit_df.to_csv(config.CREDIT_PROCESSED_CSV, index=True)

    print(f"Fraud processed: {fraud_df.shape} -> {config.FRAUD_PROCESSED_CSV}")
    print(f"Credit processed: {credit_df.shape} -> {config.CREDIT_PROCESSED_CSV}")


if __name__ == "__main__":
    preprocess_and_save()
