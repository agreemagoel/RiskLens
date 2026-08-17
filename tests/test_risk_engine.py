"""Tests for the composite risk engine (src/risk_engine/risk_score.py)."""
import pytest

from src.risk_engine.risk_score import (
    RiskInputs,
    compute_final_risk,
    risk_level,
    transaction_risk_heuristic,
)


def test_worked_example_from_project_brief():
    """Matches the exact example in the project brief:
    0.50*78 + 0.25*91 + 0.15*64 + 0.10*82 = 79.55 -> CRITICAL
    """
    inputs = RiskInputs(fraud_probability=78, anomaly_score=91, customer_risk=64, transaction_risk=82)
    result = compute_final_risk(inputs)
    assert result["final_risk_score"] == 79.55
    assert result["risk_level"] == "CRITICAL"


def test_all_zero_inputs_gives_zero_score():
    inputs = RiskInputs(0, 0, 0, 0)
    result = compute_final_risk(inputs)
    assert result["final_risk_score"] == 0.0
    assert result["risk_level"] == "LOW"


def test_all_max_inputs_gives_max_score():
    inputs = RiskInputs(100, 100, 100, 100)
    result = compute_final_risk(inputs)
    assert result["final_risk_score"] == 100.0
    assert result["risk_level"] == "CRITICAL"


@pytest.mark.parametrize("score,expected", [
    (0, "LOW"), (19.9, "LOW"), (20, "MEDIUM"), (49.9, "MEDIUM"),
    (50, "HIGH"), (74.9, "HIGH"), (75, "CRITICAL"), (100, "CRITICAL"),
])
def test_risk_bands(score, expected):
    assert risk_level(score) == expected


def test_transaction_risk_heuristic_scales_with_amount():
    p95 = 365.0
    low = transaction_risk_heuristic(100, p95)
    high = transaction_risk_heuristic(85000, p95)
    assert high > low
    assert 0 <= low <= 100
    assert 0 <= high <= 100
