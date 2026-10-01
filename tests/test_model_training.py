"""
Unit tests for model training module using Python standard library unittest.
"""

import unittest
import pandas as pd
import numpy as np

from src.data_preprocessing import get_feature_target_split, build_preprocessing_pipeline
from src.model_training import (
    create_train_test_split,
    build_models,
    fit_preprocessor_and_transform,
    evaluate_model,
    compare_models,
    RANDOM_STATE,
)


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        """Create a representative synthetic dataset for modeling tests."""
        np.random.seed(RANDOM_STATE)
        n_samples = 100
        self.sample_df = pd.DataFrame({
            "tenure": np.random.randint(0, 72, size=n_samples),
            "MonthlyCharges": np.random.uniform(20.0, 120.0, size=n_samples),
            "TotalCharges": np.random.uniform(20.0, 8000.0, size=n_samples),
            "gender": np.random.choice(["Male", "Female"], size=n_samples),
            "SeniorCitizen": np.random.choice([0, 1], size=n_samples),
            "Partner": np.random.choice(["Yes", "No"], size=n_samples),
            "Dependents": np.random.choice(["Yes", "No"], size=n_samples),
            "PhoneService": np.random.choice(["Yes", "No"], size=n_samples),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n_samples),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples),
            "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples),
            "PaymentMethod": np.random.choice([
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], size=n_samples),
            "Churn": np.random.choice([0, 1], size=n_samples, p=[0.74, 0.26]),
        })

    def test_stratified_train_test_split(self):
        X_train, X_test, y_train, y_test = create_train_test_split(
            self.sample_df, target_col="Churn", test_size=0.2, random_state=42
        )
        # Check lengths
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        # Check no overlap
        self.assertTrue(set(X_train.index).isdisjoint(set(X_test.index)))
        # Check stratification consistency
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        self.assertAlmostEqual(train_rate, test_rate, delta=0.08)

    def test_build_models_returns_all_five(self):
        models = build_models(random_state=42)
        expected_keys = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
        ]
        self.assertEqual(list(models.keys()), expected_keys)

    def test_preprocessing_and_training(self):
        X_train, X_test, y_train, y_test = create_train_test_split(
            self.sample_df, target_col="Churn", test_size=0.2, random_state=42
        )
        preprocessor = build_preprocessing_pipeline()
        X_train_prep, X_test_prep = fit_preprocessor_and_transform(preprocessor, X_train, X_test)

        self.assertEqual(X_train_prep.shape[0], 80)
        self.assertEqual(X_test_prep.shape[0], 20)
        self.assertEqual(X_train_prep.shape[1], X_test_prep.shape[1])

        # Test evaluating one model
        models = build_models(random_state=42)
        lr = models["Logistic Regression"]
        lr.fit(X_train_prep, y_train)
        eval_dict = evaluate_model(lr, X_test_prep, y_test)

        self.assertTrue(set(eval_dict["y_pred"]).issubset({0, 1}))
        self.assertTrue(np.all((eval_dict["y_prob"] >= 0.0) & (eval_dict["y_prob"] <= 1.0)))
        self.assertIn("ROC-AUC", eval_dict)

    def test_compare_models_table_structure(self):
        X_train, X_test, y_train, y_test = create_train_test_split(
            self.sample_df, target_col="Churn", test_size=0.2, random_state=42
        )
        preprocessor = build_preprocessing_pipeline()
        X_train_prep, X_test_prep = fit_preprocessor_and_transform(preprocessor, X_train, X_test)

        models = build_models(random_state=42)
        comp_df, evals = compare_models(models, X_train_prep, y_train, X_test_prep, y_test)

        self.assertEqual(len(comp_df), 5)
        expected_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        self.assertEqual(list(comp_df.columns), expected_cols)
        self.assertEqual(len(evals), 5)


if __name__ == "__main__":
    unittest.main()
