"""Pydantic request/response schemas for the RiskLens API."""
from pydantic import BaseModel, Field


class FraudPredictRequest(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction in dataset")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 50000, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
                "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
                "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
                "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62,
            }
        }


class CreditPredictRequest(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0)
    age: int = Field(..., ge=18, le=110)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(0, ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0)
    MonthlyIncome: float = Field(..., ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(0, ge=0)
    NumberOfTimes90DaysLate: int = Field(0, ge=0)
    NumberRealEstateLoansOrLines: int = Field(0, ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(0, ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int = Field(0, ge=0)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.77,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 2,
                "DebtRatio": 0.80,
                "MonthlyIncome": 9120,
                "NumberOfOpenCreditLinesAndLoans": 13,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 6,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            }
        }


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str


class FraudPredictResponse(BaseModel):
    fraud_probability: float
    risk_score: float
    risk_level: str
    top_risk_factors: list[RiskFactor]
    explanation: str


class CreditPredictResponse(BaseModel):
    default_probability: float
    risk_score: float
    risk_level: str
    top_risk_factors: list[RiskFactor]
    explanation: str


class CompositeRiskRequest(BaseModel):
    fraud_probability: float = Field(..., ge=0, le=100)
    anomaly_score: float = Field(..., ge=0, le=100)
    customer_risk: float = Field(..., ge=0, le=100)
    transaction_risk: float = Field(..., ge=0, le=100)


class CompositeRiskResponse(BaseModel):
    final_risk_score: float
    risk_level: str
    components: dict
    weights: dict
