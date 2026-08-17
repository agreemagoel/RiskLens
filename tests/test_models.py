"""
Sanity tests for trained models: they load, predict on real processed data,
and outputs are in valid ranges. Run with: pytest tests/
"""
import joblib
import pandas as pd
import pytest

from src import config
from src.data.feature_engineering import engineer_fraud_features, engineer_credit_features


@pytest.fixture(scope="module")
def fraud_bundle():
    return joblib.load(config.FRAUD_MODEL_PATH)


@pytest.fixture(scope="module")
def credit_bundle():
    return joblib.load(config.CREDIT_MODEL_PATH)


@pytest.fixture(scope="module")
def fraud_data():
    df = pd.read_csv(config.FRAUD_PROCESSED_CSV)
    return engineer_fraud_features(df)


@pytest.fixture(scope="module")
def credit_data():
    df = pd.read_csv(config.CREDIT_PROCESSED_CSV, index_col=0)
    return engineer_credit_features(df).drop(columns=["age_band"])


def test_fraud_model_loads(fraud_bundle):
    assert "model" in fraud_bundle
    assert "feature_cols" in fraud_bundle
    assert len(fraud_bundle["feature_cols"]) > 0


def test_credit_model_loads(credit_bundle):
    assert "model" in credit_bundle
    assert "feature_cols" in credit_bundle


def test_fraud_predictions_in_valid_range(fraud_bundle, fraud_data):
    X = fraud_data[fraud_bundle["feature_cols"]].head(100)
    proba = fraud_bundle["model"].predict_proba(X)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_credit_predictions_in_valid_range(credit_bundle, credit_data):
    X = credit_data[credit_bundle["feature_cols"]].head(100)
    proba = credit_bundle["model"].predict_proba(X)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_fraud_model_flags_known_fraud_higher_than_legit(fraud_bundle, fraud_data):
    """Sanity check: average predicted probability for known fraud cases
    should be much higher than for known legitimate cases."""
    fraud_rows = fraud_data[fraud_data["Class"] == 1][fraud_bundle["feature_cols"]]
    legit_rows = fraud_data[fraud_data["Class"] == 0][fraud_bundle["feature_cols"]].sample(
        n=len(fraud_rows), random_state=42
    )
    fraud_proba = fraud_bundle["model"].predict_proba(fraud_rows)[:, 1].mean()
    legit_proba = fraud_bundle["model"].predict_proba(legit_rows)[:, 1].mean()
    assert fraud_proba > legit_proba


def test_credit_model_flags_known_defaulters_higher_than_non_defaulters(credit_bundle, credit_data):
    default_rows = credit_data[credit_data["SeriousDlqin2yrs"] == 1][credit_bundle["feature_cols"]]
    nondefault_rows = credit_data[credit_data["SeriousDlqin2yrs"] == 0][credit_bundle["feature_cols"]].sample(
        n=len(default_rows), random_state=42
    )
    default_proba = credit_bundle["model"].predict_proba(default_rows)[:, 1].mean()
    nondefault_proba = credit_bundle["model"].predict_proba(nondefault_rows)[:, 1].mean()
    assert default_proba > nondefault_proba
