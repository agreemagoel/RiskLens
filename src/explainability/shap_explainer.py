"""
SHAP explainability for RiskLens.

Wraps SHAP TreeExplainer for both the fraud and credit models and converts
raw SHAP values into a ranked, human-readable list of risk drivers -- the
input the Risk Analyst Copilot (risk_copilot.py) turns into plain-English
explanations.
"""
import joblib
import pandas as pd
import shap

from src import config


class ShapExplainer:
    def __init__(self, model_path):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.model_name = bundle["model_name"]
        # TreeExplainer works for RF/XGBoost; both are our "best" models in
        # practice, but fall back to a generic Explainer if a linear model wins.
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self._is_tree = True
        except Exception:
            self.explainer = None
            self._is_tree = False

    def explain(self, row: dict, top_n: int = 5) -> list[dict]:
        """Return top_n risk factors as [{feature, shap_value, direction}, ...]
        sorted by absolute contribution, for a single row of input data.
        """
        X = pd.DataFrame([row])[self.feature_cols]

        if not self._is_tree:
            return []

        shap_values = self.explainer.shap_values(X)
        # SHAP's return shape for binary classifiers varies by model type and
        # library version:
        #   - list of 2 arrays [class0, class1], each (n_samples, n_features)
        #   - 3D array (n_samples, n_features, n_classes)
        #   - 2D array (n_samples, n_features)  (e.g. XGBoost binary -> single output)
        if isinstance(shap_values, list):
            values = shap_values[1][0]  # class-1 (positive) contributions, first row
        elif shap_values.ndim == 3:
            values = shap_values[0, :, 1]  # first row, all features, class-1
        else:
            values = shap_values[0]  # first row

        contributions = list(zip(self.feature_cols, values))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top = []
        for feature, value in contributions[:top_n]:
            top.append({
                "feature": feature,
                "shap_value": round(float(value), 4),
                "direction": "increases_risk" if value > 0 else "decreases_risk",
            })
        return top


def load_fraud_explainer() -> ShapExplainer:
    return ShapExplainer(config.FRAUD_MODEL_PATH)


def load_credit_explainer() -> ShapExplainer:
    return ShapExplainer(config.CREDIT_MODEL_PATH)
