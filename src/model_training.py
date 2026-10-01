"""
Model Training and Evaluation Module for Telco Customer Churn & Retention System.

This module provides modular, reproducible functions for:
1. Stratified train/test splitting
2. Baseline model construction (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
3. Leakage-free model training with scikit-learn ColumnTransformer
4. Comprehensive multi-metric model evaluation and comparison
5. Diagnostic visualization generation (ROC curves, confusion matrices)
"""

import os
import warnings
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress benign scipy optimization warnings from LogisticRegression
warnings.filterwarnings("ignore", message=".*Unknown solver options.*")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.compose import ColumnTransformer

# Global reproducibility constant
RANDOM_STATE: int = 42


def create_train_test_split(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and test sets using stratified sampling.

    Stratification is essential because customer churn is inherently imbalanced
    (~73.5% retained vs ~26.5% churned). Stratified splitting guarantees that
    the class proportion in both train and test partitions mirrors the population.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input DataFrame.
    target_col : str, default='Churn'
        Name of target column.
    test_size : float, default=0.2
        Fraction of dataset to allocate to the hold-out test set.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Instantiate the 5 baseline classification models with standard parameters.

    Models:
    1. Logistic Regression: max_iter=1000 for reliable convergence.
    2. Decision Tree: Default depth for unconstrained baseline.
    3. Random Forest: 100 trees, ensemble baseline.
    4. Gradient Boosting: 100 estimators, sequential gradient boosting.
    5. XGBoost: High-performance gradient boosted trees (eval_metric='logloss').

    Parameters
    ----------
    random_state : int, default=42
        Seed for stochastic algorithms.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to model instances.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }
    return models


def fit_preprocessor_and_transform(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit preprocessor ONLY on X_train and transform both X_train and X_test.

    Prevents test set information (scaling statistics, category encoders)
    from leaking into model training.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Unfitted ColumnTransformer instance.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_train_prep, X_test_prep)
    """
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    return X_train_prep, X_test_prep


def evaluate_model(
    model: Any,
    X_test_prep: np.ndarray,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Compute classification metrics and predictions for an already fitted model.

    Parameters
    ----------
    model : Any
        Fitted classification model.
    X_test_prep : np.ndarray
        Preprocessed test features.
    y_test : pd.Series
        Ground-truth binary labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary with scalar metrics, y_pred, and y_prob.
    """
    y_pred = model.predict(X_test_prep)

    # Obtain predicted probability for positive class (Churn = 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_prep)[:, 1]
    elif hasattr(model, "decision_function"):
        # Map decision function via sigmoid if no predict_proba
        y_prob = 1 / (1 + np.exp(-model.decision_function(X_test_prep)))
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    return {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def compare_models(
    models: Dict[str, Any],
    X_train_prep: np.ndarray,
    y_train: pd.Series,
    X_test_prep: np.ndarray,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Train all baseline models and evaluate them on identical hold-out test set.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to model instances.
    X_train_prep : np.ndarray
        Preprocessed training features.
    y_train : pd.Series
        Training target vector.
    X_test_prep : np.ndarray
        Preprocessed test features.
    y_test : pd.Series
        Test target vector.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        (comparison_df, detailed_evaluations)
    """
    comparison_records = []
    evaluations = {}

    for name, model in models.items():
        # Fit model on training partition only
        model.fit(X_train_prep, y_train)

        # Evaluate model on test partition
        eval_dict = evaluate_model(model, X_test_prep, y_test)
        evaluations[name] = eval_dict

        comparison_records.append({
            "Model": name,
            "Accuracy": eval_dict["Accuracy"],
            "Precision": eval_dict["Precision"],
            "Recall": eval_dict["Recall"],
            "F1": eval_dict["F1"],
            "ROC-AUC": eval_dict["ROC-AUC"],
        })

    comparison_df = pd.DataFrame(comparison_records)
    return comparison_df, evaluations


def plot_roc_curves(
    evaluations: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate and optionally save an overlay of ROC curves for all models.

    Parameters
    ----------
    evaluations : Dict[str, Dict[str, Any]]
        Evaluation outputs from compare_models.
    y_test : pd.Series
        Ground-truth labels.
    save_path : str, optional
        Path to save the generated figure.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706"]

    for (name, ev), color in zip(evaluations.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, ev["y_prob"])
        auc_score = ev["ROC-AUC"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", color=color, linewidth=2)

    # Reference random line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.5000)")

    ax.set_title("Baseline Model Comparison — ROC Curves", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curves saved to: {save_path}")

    return fig


def plot_confusion_matrices(
    evaluations: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate and optionally save a 2x3 grid of confusion matrices for all models.

    Parameters
    ----------
    evaluations : Dict[str, Dict[str, Any]]
        Evaluation outputs from compare_models.
    y_test : pd.Series
        Ground-truth labels.
    save_path : str, optional
        Path to save the generated figure.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (name, ev) in enumerate(evaluations.items()):
        cm = confusion_matrix(y_test, ev["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"],
        )
        axes[idx].set_title(
            f"{name}\n(Acc: {ev['Accuracy']:.2f}, F1: {ev['F1']:.2f}, Rec: {ev['Recall']:.2f})",
            fontsize=11,
            fontweight="bold",
        )
        axes[idx].set_ylabel("Actual Label", fontsize=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=10)

    # Hide extra unused subplot (axis 5)
    axes[5].axis("off")

    plt.suptitle("Baseline Confusion Matrices (Hold-out Test Set, N=1,409)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrices saved to: {save_path}")

    return fig


def save_comparison_results(comparison_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the model comparison metrics table to a CSV file.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame of evaluated model metrics.
    output_path : str
        Target filepath (e.g. reports/model_comparison.csv).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    print(f"Model comparison saved to: {output_path}")
