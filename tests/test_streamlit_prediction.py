"""Tests for the Phase 8 streamlit inference workflow."""

import os
import unittest

import pandas as pd

from src.prediction_service import (
    build_customer_dataframe,
    load_model,
    predict_customer_churn,
    validate_customer_input,
)


VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": False,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 69.90,
    "TotalCharges": 1800.00,
}


class TestStreamlitPrediction(unittest.TestCase):
    def test_final_model_loads(self):
        artifact = load_model()
        self.assertEqual(artifact.model_name, "Gradient Boosting")
        self.assertEqual(artifact.threshold, 0.35)

    def test_expected_feature_schema_is_accepted(self):
        customer = validate_customer_input(VALID_CUSTOMER.copy())
        self.assertIn("gender", customer)
        self.assertIn("Contract", customer)
        self.assertIn("MonthlyCharges", customer)

    def test_valid_customer_input_produces_probability_between_zero_and_one(self):
        result = predict_customer_churn(VALID_CUSTOMER.copy())
        self.assertGreaterEqual(result["churn_probability"], 0.0)
        self.assertLessEqual(result["churn_probability"], 1.0)

    def test_prediction_uses_artifact_threshold(self):
        artifact = load_model()
        result = predict_customer_churn(VALID_CUSTOMER.copy())
        self.assertEqual(result["threshold"], artifact.threshold)
        self.assertEqual(result["predicted_churn"], int(result["churn_probability"] >= artifact.threshold))

    def test_predicted_churn_consistent_with_threshold(self):
        result = predict_customer_churn(VALID_CUSTOMER.copy())
        if result["churn_probability"] >= result["threshold"]:
            self.assertEqual(result["risk_tier"], "HIGH")
        else:
            self.assertEqual(result["risk_tier"], "LOW")

    def test_low_risk_is_assigned_below_threshold(self):
        low_customer = VALID_CUSTOMER.copy()
        low_customer["Contract"] = "Two year"
        low_customer["tenure"] = 48
        low_customer["MonthlyCharges"] = 45
        low_customer["TotalCharges"] = 2000
        result = predict_customer_churn(low_customer)
        self.assertLess(result["churn_probability"], result["threshold"])
        self.assertEqual(result["risk_tier"], "LOW")

    def test_high_risk_is_assigned_at_or_above_threshold(self):
        high_customer = VALID_CUSTOMER.copy()
        high_customer["Contract"] = "Month-to-month"
        high_customer["tenure"] = 2
        high_customer["MonthlyCharges"] = 90
        high_customer["TotalCharges"] = 120
        high_customer["InternetService"] = "Fiber optic"
        high_customer["PaymentMethod"] = "Electronic check"
        high_customer["OnlineSecurity"] = "No"
        high_customer["TechSupport"] = "No"
        result = predict_customer_churn(high_customer)
        self.assertGreaterEqual(result["churn_probability"], result["threshold"])
        self.assertEqual(result["risk_tier"], "HIGH")

    def test_retention_recommendation_is_returned(self):
        result = predict_customer_churn(VALID_CUSTOMER.copy())
        self.assertIn("retention_recommendation", result)
        self.assertIn("priority", result["retention_recommendation"])
        self.assertIn("recommended_action", result["retention_recommendation"])

    def test_invalid_negative_numeric_input_is_rejected(self):
        bad = VALID_CUSTOMER.copy()
        bad["tenure"] = -1
        with self.assertRaises(ValueError):
            validate_customer_input(bad)

    def test_invalid_categorical_values_are_rejected(self):
        bad = VALID_CUSTOMER.copy()
        bad["Contract"] = "Invalid contract"
        with self.assertRaises(ValueError):
            validate_customer_input(bad)

    def test_missing_model_produces_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            load_model(model_path="models/does_not_exist.joblib")

    def test_repeated_prediction_for_same_input_is_deterministic(self):
        first = predict_customer_churn(VALID_CUSTOMER.copy())
        second = predict_customer_churn(VALID_CUSTOMER.copy())
        self.assertEqual(first["churn_probability"], second["churn_probability"])
        self.assertEqual(first["predicted_churn"], second["predicted_churn"])
        self.assertEqual(first["risk_tier"], second["risk_tier"])

    def test_customer_dataframe_matches_expected_schema(self):
        df = build_customer_dataframe(VALID_CUSTOMER.copy())
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(list(df.columns), [
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
        ])


if __name__ == "__main__":
    unittest.main()
