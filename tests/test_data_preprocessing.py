"""
Unit tests for data preprocessing module using Python standard library unittest.
"""

import unittest
import pandas as pd
from src.data_preprocessing import (
    handle_total_charges,
    drop_identifiers,
    encode_target,
    clean_telco_data,
    build_preprocessing_pipeline,
    validate_cleaned_data,
)


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        """Create a minimal synthetic sample mimicking raw Telco data."""
        self.sample_raw_data = pd.DataFrame({
            "customerID": ["001-AAA", "002-BBB", "003-CCC"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [12, 0, 24],
            "PhoneService": ["Yes", "No", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "No", "No internet service"],
            "TechSupport": ["Yes", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "MonthlyCharges": [29.85, 50.00, 20.00],
            "TotalCharges": ["358.2", " ", "480.0"],
            "Churn": ["No", "No", "Yes"],
        })

    def test_handle_total_charges_coerces_whitespace(self):
        df_clean = handle_total_charges(self.sample_raw_data)
        self.assertTrue(pd.api.types.is_numeric_dtype(df_clean["TotalCharges"]))
        self.assertEqual(df_clean.loc[1, "TotalCharges"], 0.0)
        self.assertEqual(df_clean["TotalCharges"].isna().sum(), 0)

    def test_drop_identifiers(self):
        df_no_id = drop_identifiers(self.sample_raw_data)
        self.assertNotIn("customerID", df_no_id.columns)
        self.assertEqual(len(df_no_id.columns), len(self.sample_raw_data.columns) - 1)

    def test_encode_target(self):
        df_encoded = encode_target(self.sample_raw_data)
        self.assertTrue(set(df_encoded["Churn"].unique()).issubset({0, 1}))
        self.assertEqual(df_encoded.loc[0, "Churn"], 0)
        self.assertEqual(df_encoded.loc[2, "Churn"], 1)

    def test_clean_telco_data_full(self):
        df_clean = clean_telco_data(self.sample_raw_data)
        self.assertEqual(df_clean.shape, (3, 20))
        self.assertNotIn("customerID", df_clean.columns)
        self.assertEqual(df_clean.isna().sum().sum(), 0)
        self.assertEqual(df_clean.loc[1, "TotalCharges"], 0.0)
        self.assertTrue(set(df_clean["Churn"].unique()).issubset({0, 1}))

    def test_build_preprocessing_pipeline(self):
        pipeline = build_preprocessing_pipeline()
        self.assertIsNotNone(pipeline)
        transformer_names = [t[0] for t in pipeline.transformers]
        self.assertIn("num", transformer_names)
        self.assertIn("cat", transformer_names)


if __name__ == "__main__":
    unittest.main()
