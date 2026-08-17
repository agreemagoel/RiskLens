"""
Risk Analyst Copilot -- FREE, template-based explanation engine.

Deliberately does NOT call an LLM API. Instead, it maps SHAP output to
human-readable sentences using a rule/template dictionary. This guarantees
the explanation is always grounded in the model's actual inputs (no
hallucination risk) and needs zero API cost -- see project brief:
"Version 1: Create a template-based explanation engine."

An optional local-LLM extension point is included at the bottom for anyone
who wants to route the same structured SHAP JSON through Ollama later.
"""
from src import config

# Human-readable descriptions per feature, keyed by dataset.
FRAUD_FEATURE_LABELS = {
    "Amount": "the transaction amount",
    "amount_log": "the transaction amount",
    "Hour": "the time of day the transaction occurred",
    "is_off_peak_hour": "the transaction happening during an unusual off-peak hour (2-5 AM)",
    "is_micro_transaction": "an unusually small 'test' transaction amount",
    "is_large_transaction": "an unusually large transaction amount",
}
# V1-V28 are anonymized PCA components; we describe them generically.
for i in range(1, 29):
    FRAUD_FEATURE_LABELS[f"V{i}"] = f"anonymized transaction pattern feature V{i}"

CREDIT_FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "credit utilization",
    "age": "the customer's age",
    "NumberOfTime30-59DaysPastDueNotWorse": "recent 30-59 day payment delinquencies",
    "NumberOfTime60-89DaysPastDueNotWorse": "recent 60-89 day payment delinquencies",
    "NumberOfTimes90DaysLate": "history of 90+ day late payments",
    "DebtRatio": "overall debt ratio",
    "MonthlyIncome": "monthly income",
    "NumberOfOpenCreditLinesAndLoans": "number of open credit lines and loans",
    "NumberRealEstateLoansOrLines": "number of real estate loans",
    "NumberOfDependents": "number of dependents",
    "total_delinquencies": "total delinquency history",
    "has_any_delinquency": "presence of any past delinquency",
    "high_utilization": "high credit utilization (>80%)",
    "income_per_dependent": "income relative to number of dependents",
    "total_credit_lines": "total number of credit lines",
    "MonthlyIncome_was_missing": "missing income information on file",
    "NumberOfDependents_was_missing": "missing dependents information on file",
}


def _label(dataset: str, feature: str) -> str:
    table = FRAUD_FEATURE_LABELS if dataset == "fraud" else CREDIT_FEATURE_LABELS
    if feature in table:
        return table[feature]
    if feature.startswith("age_band_"):
        return f"the customer's age band ({feature.replace('age_band_', '')})"
    return feature.replace("_", " ")


def explain_risk(
    dataset: str,
    entity_id: str,
    probability: float,
    risk_level: str,
    shap_factors: list[dict],
) -> str:
    """Build a plain-English explanation from SHAP factors.

    This is the "Risk Analyst Copilot" response -- entirely template-based,
    no LLM/API call. It only describes what the model actually computed.
    """
    increasing = [f for f in shap_factors if f["direction"] == "increases_risk"]
    decreasing = [f for f in shap_factors if f["direction"] == "decreases_risk"]

    label_type = "Transaction" if dataset == "fraud" else "Customer"
    lines = [f"{label_type} {entity_id} is classified as {risk_level} RISK."]
    lines.append(f"Estimated {'fraud' if dataset == 'fraud' else 'default'} probability: {probability*100:.1f}%.")
    lines.append("")

    if increasing:
        lines.append("Primary risk factors:")
        for f in increasing[:5]:
            lines.append(f"  - {_label(dataset, f['feature']).capitalize()}")

    if decreasing:
        lines.append("")
        lines.append("Risk-reducing factors:")
        for f in decreasing[:3]:
            lines.append(f"  - {_label(dataset, f['feature']).capitalize()}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OPTIONAL local-LLM extension point (NOT used by default; zero API cost).
# ---------------------------------------------------------------------------
def explain_risk_with_local_llm(structured_payload: dict) -> str:
    """Optional: route the same structured SHAP JSON through a local model
    (e.g. Ollama) for more natural phrasing. Not called anywhere in the app
    by default -- the template engine above is the production path so the
    project has zero dependency on any paid API or internet access at
    inference time.

    Example (commented out, requires `pip install ollama` and a running
    Ollama server with a small model pulled, e.g. `ollama pull llama3.2`):

        import ollama
        prompt = f\"\"\"You are a risk analyst. Using ONLY this structured
        data, write a 2-3 sentence explanation. Do not invent facts.
        Data: {structured_payload}\"\"\"
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    """
    raise NotImplementedError(
        "Local LLM path is optional and not wired up by default. "
        "See docstring for how to enable it with Ollama."
    )
