"""Reusable prediction and validation logic for the Streamlit churn app."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Mapping, MutableMapping

import joblib
import pandas as pd

from src.retention_strategy import generate_retention_recommendation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_churn_model.joblib")

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_FIELDS = {"tenure", "MonthlyCharges", "TotalCharges"}

CATEGORY_OPTIONS = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
}


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the saved final churn model artifact from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")

    try:
        artifact = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unable to load model artifact from {model_path}: {exc}") from exc

    required_attrs = {"predict_proba", "threshold", "model_name"}
    missing = sorted(required_attrs - set(dir(artifact)))
    if missing:
        raise ValueError(f"Model artifact is malformed or missing required attributes: {missing}")

    return artifact


def validate_customer_input(customer_input: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a single customer record before it reaches the saved pipeline."""
    if customer_input is None:
        raise ValueError("Customer input is required.")

    cleaned: Dict[str, Any] = {}

    for field in FEATURE_COLUMNS:
        if field not in customer_input:
            raise ValueError(f"Missing required field: {field}")

        value = customer_input[field]

        if field == "SeniorCitizen":
            if isinstance(value, bool):
                cleaned[field] = int(value)
                continue
            if isinstance(value, (int, float)) and value in {0, 1}:
                cleaned[field] = int(value)
                continue
            raise ValueError("SeniorCitizen must be a boolean or 0/1 value.")

        if field in NUMERIC_FIELDS:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a finite numeric value.") from exc
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(f"{field} must be a finite value >= 0.")
            cleaned[field] = numeric_value
            continue

        choice_set = CATEGORY_OPTIONS.get(field)
        if choice_set is None:
            raise ValueError(f"No validation rules defined for {field}.")

        if value is None:
            raise ValueError(f"{field} cannot be empty.")

        text_value = str(value).strip()
        if text_value not in {str(item) for item in choice_set}:
            allowed = ", ".join(sorted(str(item) for item in choice_set))
            raise ValueError(f"Invalid value for {field}: '{value}'. Allowed values: {allowed}")
        cleaned[field] = text_value

    return cleaned


def build_customer_dataframe(customer_input: Mapping[str, Any]) -> pd.DataFrame:
    """Construct the single-row DataFrame expected by the saved model pipeline."""
    validated = validate_customer_input(customer_input)
    df = pd.DataFrame([validated], columns=FEATURE_COLUMNS)
    return df


def predict_customer_churn(customer_input: Mapping[str, Any], model_path: str = DEFAULT_MODEL_PATH, artifact=None):
    """Predict churn probability and return the risk, threshold, and retention recommendation."""
    if artifact is None:
        artifact = load_model(model_path)

    validated = validate_customer_input(customer_input)
    input_df = build_customer_dataframe(validated)
    probabilities = artifact.predict_proba(input_df)
    if probabilities.ndim == 2:
        churn_probability = float(probabilities[0, 1])
    else:
        churn_probability = float(probabilities[0])
    threshold = float(artifact.threshold)
    predicted_churn = int(churn_probability >= threshold)
    risk_tier = "HIGH" if churn_probability >= threshold else "LOW"
    recommendation = generate_retention_recommendation(validated, churn_probability)

    return {
        "churn_probability": churn_probability,
        "predicted_churn": predicted_churn,
        "risk_tier": risk_tier,
        "threshold": threshold,
        "retention_recommendation": recommendation,
    }
