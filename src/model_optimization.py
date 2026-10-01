"""Leakage-safe model optimization for Telco customer churn prediction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data_preprocessing import build_preprocessing_pipeline
from src.model_training import RANDOM_STATE


@dataclass
class ChurnModelArtifact:
    """A fitted prediction pipeline and its validation-selected cutoff."""

    pipeline: Pipeline
    threshold: float
    model_name: str

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(features)[:, 1]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(int)


def _make_search_spaces(y_fit: pd.Series, random_state: int) -> Dict[str, Tuple[Pipeline, dict]]:
    positive_count = int((y_fit == 1).sum())
    negative_count = int((y_fit == 0).sum())
    positive_weight = negative_count / max(positive_count, 1)

    def pipeline(model: Any) -> Pipeline:
        return Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", model),
        ])

    return {
        "Logistic Regression": (
            pipeline(LogisticRegression(max_iter=1500, random_state=random_state)),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            pipeline(RandomForestClassifier(random_state=random_state, n_jobs=1)),
            {
                "model__n_estimators": [150],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 3],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting": (
            pipeline(GradientBoostingClassifier(random_state=random_state)),
            {
                "model__n_estimators": [100, 180],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [1, 2],
                "model__min_samples_leaf": [3],
            },
        ),
        "XGBoost": (
            pipeline(XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=1,
                tree_method="hist",
            )),
            {
                "model__n_estimators": [150],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__scale_pos_weight": [1.0, positive_weight],
            },
        ),
    }


def threshold_analysis(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Calculate binary metrics and confusion counts for each cutoff."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.901, 0.05), 2)

    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append({
            "Threshold": float(threshold),
            "Precision": precision_score(y_true, predictions, zero_division=0),
            "Recall": recall_score(y_true, predictions, zero_division=0),
            "F1": f1_score(y_true, predictions, zero_division=0),
            "Predicted Positive Rate": float(predictions.mean()),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        })
    return pd.DataFrame(rows)


def _metrics_at_threshold(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> Dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
    }


def select_threshold(
    y_validation: pd.Series, validation_probabilities: np.ndarray
) -> Tuple[float, pd.DataFrame]:
    """Select the validation cutoff with best F1, then recall, then precision."""
    table = threshold_analysis(y_validation, validation_probabilities)
    winner = table.sort_values(
        ["F1", "Recall", "Precision", "Threshold"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    return float(winner["Threshold"]), table


def _save_figures(
    comparison: pd.DataFrame,
    threshold_table: pd.DataFrame,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
    test_predictions: np.ndarray,
    figures_dir: str,
) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    ordered = comparison.sort_values("Validation F1", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(ordered))
    ax.barh(positions - 0.18, ordered["CV F1"], height=0.34, label="Mean CV F1")
    ax.barh(positions + 0.18, ordered["Validation F1"], height=0.34, label="Validation F1")
    ax.set_yticks(positions, ordered["Model"])
    ax.set_xlabel("F1 score")
    ax.set_title("Optimized Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "optimized_model_comparison.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ("Precision", "Recall", "F1"):
        ax.plot(threshold_table["Threshold"], threshold_table[metric], marker="o", label=metric)
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Validation Threshold Analysis")
    ax.set_xticks(threshold_table["Threshold"])
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "threshold_precision_recall_f1.png"), dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="Actual label", title="FINAL UNTOUCHED TEST EVALUATION\nConfusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "final_confusion_matrix.png"), dpi=160)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, test_probabilities)
    auc = roc_auc_score(y_test, test_probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(false_positive_rate, true_positive_rate, label=f"Selected model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="FINAL UNTOUCHED TEST EVALUATION\nROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "final_roc_curve.png"), dpi=160)
    plt.close(fig)


def run_phase6(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_path: str = "models/final_churn_model.joblib",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures",
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    model_names: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Tune, select, refit, and evaluate without using test data before finalization."""
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    spaces = _make_search_spaces(y_fit, random_state)
    if model_names is not None:
        spaces = {name: spaces[name] for name in model_names}

    comparison_rows = []
    threshold_tables = {}
    best_estimators = {}
    for name, (estimator, parameter_grid) in spaces.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_fit, y_fit)
        validation_probabilities = search.best_estimator_.predict_proba(X_validation)[:, 1]
        threshold, candidate_thresholds = select_threshold(y_validation, validation_probabilities)
        validation_metrics = _metrics_at_threshold(y_validation, validation_probabilities, threshold)
        best_index = search.best_index_
        comparison_rows.append({
            "Model": name,
            "Best Parameters": search.best_params_,
            "CV Accuracy": float(search.cv_results_["mean_test_accuracy"][best_index]),
            "CV Precision": float(search.cv_results_["mean_test_precision"][best_index]),
            "CV Recall": float(search.cv_results_["mean_test_recall"][best_index]),
            "CV F1": float(search.cv_results_["mean_test_f1"][best_index]),
            "CV ROC-AUC": float(search.cv_results_["mean_test_roc_auc"][best_index]),
            "Validation Threshold": threshold,
            "Validation Accuracy": validation_metrics["Accuracy"],
            "Validation Precision": validation_metrics["Precision"],
            "Validation Recall": validation_metrics["Recall"],
            "Validation F1": validation_metrics["F1"],
            "Validation ROC-AUC": validation_metrics["ROC-AUC"],
        })
        threshold_tables[name] = candidate_thresholds
        best_estimators[name] = search.best_estimator_

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["Validation F1", "Validation ROC-AUC", "Validation Precision"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["Model"])
    selected_threshold = float(comparison.iloc[0]["Validation Threshold"])
    selected_threshold_table = threshold_tables[selected_name]

    final_pipeline = clone(best_estimators[selected_name])
    final_pipeline.fit(X_train, y_train)
    artifact = ChurnModelArtifact(
        pipeline=final_pipeline,
        threshold=selected_threshold,
        model_name=selected_name,
    )

    test_probabilities = artifact.predict_proba(X_test)
    test_predictions = artifact.predict(X_test)
    final_metrics = _metrics_at_threshold(y_test, test_probabilities, selected_threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()
    final_metrics.update({"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)})

    os.makedirs(os.path.dirname(artifact_path) or ".", exist_ok=True)
    joblib.dump(artifact, artifact_path)
    os.makedirs(reports_dir, exist_ok=True)
    comparison.to_csv(os.path.join(reports_dir, "phase6_model_comparison.csv"), index=False)
    selected_threshold_table.to_csv(
        os.path.join(reports_dir, "phase6_threshold_analysis.csv"), index=False
    )
    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(reports_dir, "phase6_final_test_metrics.csv"), index=False
    )
    _save_figures(
        comparison,
        selected_threshold_table,
        y_test,
        test_probabilities,
        test_predictions,
        figures_dir,
    )

    return {
        "validation_comparison": comparison,
        "threshold_analysis": selected_threshold_table,
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "final_test_metrics": final_metrics,
        "artifact": artifact,
        "artifact_path": artifact_path,
    }