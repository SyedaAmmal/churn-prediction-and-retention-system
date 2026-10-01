"""Tests for leakage-safe Phase 6 model optimization."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.model_optimization import (
    _make_search_spaces,
    run_phase6,
    select_threshold,
    threshold_analysis,
)


class TestModelOptimization(unittest.TestCase):
    @staticmethod
    def make_data(n_samples=120):
        rng = np.random.default_rng(42)
        features = {
            "tenure": rng.integers(0, 73, n_samples),
            "MonthlyCharges": rng.uniform(20, 120, n_samples),
            "TotalCharges": rng.uniform(0, 8500, n_samples),
        }
        category_values = {
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["Yes", "No"],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["Yes", "No", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service"],
            "DeviceProtection": ["Yes", "No", "No internet service"],
            "TechSupport": ["Yes", "No", "No internet service"],
            "StreamingTV": ["Yes", "No", "No internet service"],
            "StreamingMovies": ["Yes", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        }
        for feature in CATEGORICAL_FEATURES:
            features[feature] = rng.choice(category_values[feature], n_samples)
        frame = pd.DataFrame(features)
        frame["Churn"] = np.tile([0, 1, 0, 0], n_samples // 4)
        return frame

    def test_threshold_analysis_confusion_counts(self):
        labels = pd.Series([0, 0, 1, 1])
        probabilities = np.array([0.2, 0.7, 0.4, 0.8])
        result = threshold_analysis(labels, probabilities, np.array([0.5])).iloc[0]
        self.assertEqual((result["TP"], result["FP"], result["TN"], result["FN"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["Precision"], 0.5)
        self.assertAlmostEqual(result["Recall"], 0.5)
        self.assertAlmostEqual(result["F1"], 0.5)
        self.assertAlmostEqual(result["Predicted Positive Rate"], 0.5)

    def test_threshold_selection_uses_validation_table(self):
        labels = pd.Series([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.6, 0.4, 0.9])
        threshold, table = select_threshold(labels, probabilities)
        self.assertEqual(threshold, 0.15)
        self.assertEqual(len(table), 17)
        self.assertTrue(np.allclose(table["Threshold"], np.arange(0.1, 0.901, 0.05)))

    def test_search_estimators_keep_preprocessing_inside_pipeline(self):
        labels = pd.Series([0, 1] * 20)
        spaces = _make_search_spaces(labels, random_state=42)
        self.assertEqual(set(spaces), {
            "Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"
        })
        for estimator, _ in spaces.values():
            self.assertIsInstance(estimator, Pipeline)
            self.assertIn("preprocessor", estimator.named_steps)
            self.assertIn("model", estimator.named_steps)

    def test_pipeline_saves_loads_and_test_labels_do_not_select_threshold(self):
        frame = self.make_data()
        X = frame.drop(columns="Churn")
        y = frame["Churn"]
        X_train, X_test = X.iloc[:90], X.iloc[90:]
        y_train, y_test = y.iloc[:90], y.iloc[90:]

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = str(Path(temp_dir) / "final_churn_model.joblib")
            with patch("src.model_optimization._save_figures"):
                result = run_phase6(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    artifact_path=artifact_path,
                    reports_dir=temp_dir,
                    figures_dir=temp_dir,
                    cv_splits=2,
                    model_names=("Logistic Regression",),
                )
                flipped_test_result = run_phase6(
                    X_train,
                    y_train,
                    X_test,
                    1 - y_test,
                    artifact_path=artifact_path,
                    reports_dir=temp_dir,
                    figures_dir=temp_dir,
                    cv_splits=2,
                    model_names=("Logistic Regression",),
                )

            self.assertEqual(result["selected_threshold"], flipped_test_result["selected_threshold"])
            self.assertEqual(
                result["validation_comparison"]["Best Parameters"].iloc[0],
                flipped_test_result["validation_comparison"]["Best Parameters"].iloc[0],
            )
            loaded = joblib.load(artifact_path)
            self.assertEqual(loaded.model_name, "Logistic Regression")
            self.assertGreaterEqual(loaded.threshold, 0.1)
            self.assertLessEqual(loaded.threshold, 0.9)
            self.assertEqual(loaded.predict(X_test).shape[0], len(X_test))
            probabilities = loaded.predict_proba(X_test)
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
            scaler = loaded.pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
            self.assertEqual(scaler.n_samples_seen_, len(X_train))


if __name__ == "__main__":
    unittest.main()
