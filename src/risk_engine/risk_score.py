"""
Composite Risk Engine.

Combines four independent signals into a single 0-100 Final Risk Score,
per the weighting from the project brief:

    Final Risk = 0.50 * Fraud Probability
               + 0.25 * Anomaly Score
               + 0.15 * Customer (credit) Risk
               + 0.10 * Transaction Risk

All four inputs are expected on a 0-100 scale before weighting.
"""
from dataclasses import dataclass

from src import config


@dataclass
class RiskInputs:
    fraud_probability: float   # 0-100
    anomaly_score: float       # 0-100
    customer_risk: float       # 0-100 (credit default probability)
    transaction_risk: float    # 0-100 (e.g. amount-based heuristic)


def risk_level(score: float) -> str:
    for low, high, label in config.RISK_BANDS:
        if low <= score < high:
            return label
    return "CRITICAL"


def compute_final_risk(inputs: RiskInputs) -> dict:
    weights = config.RISK_WEIGHTS
    final_score = (
        weights["fraud_probability"] * inputs.fraud_probability
        + weights["anomaly_score"] * inputs.anomaly_score
        + weights["customer_risk"] * inputs.customer_risk
        + weights["transaction_risk"] * inputs.transaction_risk
    )
    final_score = round(min(100.0, max(0.0, final_score)), 2)

    return {
        "final_risk_score": final_score,
        "risk_level": risk_level(final_score),
        "components": {
            "fraud_probability": inputs.fraud_probability,
            "anomaly_score": inputs.anomaly_score,
            "customer_risk": inputs.customer_risk,
            "transaction_risk": inputs.transaction_risk,
        },
        "weights": weights,
    }


def transaction_risk_heuristic(amount: float, amount_p95: float) -> float:
    """Simple 0-100 heuristic for 'transaction risk' based on how far the
    amount sits above the 95th percentile of normal transaction amounts.
    Kept intentionally simple and separate from the ML models so the risk
    engine has one non-ML, fully transparent input as a sanity check.
    """
    if amount_p95 <= 0:
        return 0.0
    ratio = amount / amount_p95
    score = min(100.0, max(0.0, (ratio - 1) * 50))  # scales smoothly past the p95 mark
    return round(score, 2)


if __name__ == "__main__":
    example = RiskInputs(
        fraud_probability=78, anomaly_score=91, customer_risk=64, transaction_risk=82
    )
    result = compute_final_risk(example)
    print(result)
