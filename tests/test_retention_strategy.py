"""Tests for the Phase 7 retention recommendation logic."""

import unittest

from src.retention_strategy import generate_retention_recommendation


class TestRetentionStrategy(unittest.TestCase):
    def test_low_risk_customer_classification(self):
        row = {
            "Contract": "Two year",
            "tenure": 48,
            "MonthlyCharges": 52.0,
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes",
            "PaymentMethod": "Credit card (automatic)",
        }
        result = generate_retention_recommendation(row, 0.12)
        self.assertEqual(result["risk_tier"], "LOW")
        self.assertEqual(result["priority"], "low")
        self.assertIn("monitor", result["recommended_action"].lower())

    def test_high_risk_customer_classification(self):
        row = {
            "Contract": "Month-to-month",
            "tenure": 2,
            "MonthlyCharges": 82.0,
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "PaymentMethod": "Electronic check",
        }
        result = generate_retention_recommendation(row, 0.75)
        self.assertEqual(result["risk_tier"], "HIGH")
        self.assertIn(result["priority"], {"medium", "high"})
        self.assertIn("risk", result["reason"].lower())

    def test_month_to_month_high_risk_logic(self):
        row = {
            "Contract": "Month-to-month",
            "tenure": 6,
            "MonthlyCharges": 67.0,
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "PaymentMethod": "Electronic check",
        }
        result = generate_retention_recommendation(row, 0.60)
        self.assertTrue("Month-to-month" in result["reason"] or "contract" in result["reason"].lower())
        self.assertIn("contract", result["recommended_action"].lower())

    def test_high_monthly_charge_logic(self):
        row = {
            "Contract": "One year",
            "tenure": 24,
            "MonthlyCharges": 90.0,
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
        }
        result = generate_retention_recommendation(row, 0.51)
        self.assertIn("high monthly charges", result["reason"].lower())
        self.assertIn("pricing", result["recommended_action"].lower())

    def test_missing_unknown_optional_attributes_do_not_crash(self):
        row = {
            "Contract": "Unknown",
            "tenure": None,
            "MonthlyCharges": None,
            "InternetService": "Unknown",
            "OnlineSecurity": None,
            "TechSupport": "Unknown",
            "PaymentMethod": None,
        }
        result = generate_retention_recommendation(row, 0.40)
        self.assertEqual(result["risk_tier"], "HIGH")
        self.assertIn("risk", result["reason"].lower())
        self.assertIn("review", result["recommended_action"].lower())

    def test_deterministic_output(self):
        row = {
            "Contract": "Month-to-month",
            "tenure": 8,
            "MonthlyCharges": 68.0,
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "PaymentMethod": "Electronic check",
        }
        first = generate_retention_recommendation(row, 0.62)
        second = generate_retention_recommendation(row, 0.62)
        self.assertEqual(first, second)

    def test_multiple_applicable_characteristics_handled_correctly(self):
        row = {
            "Contract": "Month-to-month",
            "tenure": 9,
            "MonthlyCharges": 79.0,
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "PaymentMethod": "Electronic check",
            "PaperlessBilling": "Yes",
        }
        result = generate_retention_recommendation(row, 0.74)
        reasons = result["reason"].lower()
        self.assertIn("month-to-month", reasons)
        self.assertIn("fiber optic", reasons)
        self.assertIn("electronic check", reasons)
        self.assertIn("online security", reasons)
        self.assertIn("tech support", reasons)


if __name__ == "__main__":
    unittest.main()
