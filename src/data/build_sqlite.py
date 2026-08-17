"""
Builds a local SQLite database from the processed datasets.

This gives the project a genuine SQL layer (used by sql/*.sql and by the
API's lookup-by-ID endpoints) without needing any database server --
SQLite ships with Python, per the project brief's "Tier 1 free local"
architecture.
"""
import sqlite3

import pandas as pd

from src import config
from src.data.feature_engineering import engineer_fraud_features, engineer_credit_features


def build_sqlite_db() -> None:
    fraud = pd.read_csv(config.FRAUD_PROCESSED_CSV)
    fraud = engineer_fraud_features(fraud)
    fraud = fraud.reset_index().rename(columns={"index": "transaction_id"})

    credit = pd.read_csv(config.CREDIT_PROCESSED_CSV, index_col=0)
    credit = engineer_credit_features(credit).drop(columns=["age_band"])
    credit = credit.reset_index().rename(columns={"index": "customer_id"})

    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    fraud.to_sql("fraud_transactions", conn, if_exists="replace", index=False)
    credit.to_sql("credit_customers", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_id ON fraud_transactions(transaction_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cust_id ON credit_customers(customer_id)")
    conn.commit()
    conn.close()

    print(f"SQLite DB built at {config.SQLITE_DB_PATH}")
    print(f"  fraud_transactions: {len(fraud)} rows")
    print(f"  credit_customers:   {len(credit)} rows")


if __name__ == "__main__":
    build_sqlite_db()
