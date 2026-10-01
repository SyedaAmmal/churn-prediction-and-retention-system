"""Generate Phase 7 business-analysis reports and figures.

This module loads the verified Phase 6 final model artifact, scores the cleaned Telco
customer data at the selected threshold of 0.35, creates a business-ready customer
risk table, and exports the summary and visual artifacts required for the retention
analysis layer.
"""

from __future__ import annotations

import os
from typing import Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.retention_strategy import generate_retention_recommendation


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def build_customer_risk_analysis() -> pd.DataFrame:
    """Create customer-level churn-risk dataset using the final saved artifact."""
    raw_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "Telco-Customer-Churn.csv"))
    clean_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "telco_churn_cleaned.csv"))
    artifact = joblib.load(os.path.join(PROJECT_ROOT, "models", "final_churn_model.joblib"))

    if len(raw_df) != len(clean_df):
        raise ValueError("Raw and cleaned Telco datasets do not have the same row count.")

    analysis = clean_df.copy()
    analysis["customerID"] = raw_df["customerID"].values
    analysis["churn_probability"] = artifact.predict_proba(analysis.drop(columns=["Churn"]))
    analysis["predicted_churn"] = (analysis["churn_probability"] >= artifact.threshold).astype(int)
    analysis["risk_tier"] = np.where(analysis["churn_probability"] < artifact.threshold, "LOW", "HIGH")

    rec_rows = []
    for _, row in analysis.iterrows():
        rec_rows.append(generate_retention_recommendation(row.to_dict(), row["churn_probability"]))
    rec_df = pd.DataFrame(rec_rows)

    analysis["priority"] = rec_df["priority"].values
    analysis["reason"] = rec_df["reason"].values
    analysis["recommended_action"] = rec_df["recommended_action"].values
    return analysis


def build_retention_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk-tier summaries and major associated characteristics."""
    summary = (
        analysis.groupby("risk_tier", dropna=False)
        .agg(
            customer_count=("customerID", "count"),
            average_churn_probability=("churn_probability", "mean"),
            predicted_churn_rate=("predicted_churn", "mean"),
            observed_churn_rate=("Churn", "mean"),
        )
        .reset_index()
    )
    summary["percentage_of_customers"] = summary["customer_count"] / len(analysis)

    summarised_features: Dict[str, str] = {}
    for tier, group in analysis.groupby("risk_tier", dropna=False):
        contract = group["Contract"].value_counts(normalize=True)
        internet = group["InternetService"].value_counts(normalize=True)
        payment = group["PaymentMethod"].value_counts(normalize=True)
        parts = []
        for series in (contract, internet, payment):
            top = series.head(3)
            top_text = "; ".join(f"{label} ({value:.1%})" for label, value in top.items())
            parts.append(top_text)
        summarised_features[tier] = " | ".join(parts)

    summary["major_characteristics"] = summary["risk_tier"].map(summarised_features)
    summary = summary[
        [
            "risk_tier",
            "customer_count",
            "percentage_of_customers",
            "average_churn_probability",
            "predicted_churn_rate",
            "observed_churn_rate",
            "major_characteristics",
        ]
    ]
    return summary


def save_figures(analysis: pd.DataFrame, artifact) -> None:
    """Create the required business-analysis figures for Phase 7."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. risk tier distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = analysis["risk_tier"].value_counts().reindex(["LOW", "HIGH"], fill_value=0)
    ax.bar(counts.index, counts.values, color=["#66bb6a", "#ef5350"])
    ax.set_title("Phase 7 Risk Tier Distribution")
    ax.set_xlabel("Risk tier")
    ax.set_ylabel("Customer count")
    for container in ax.containers:
        ax.bar_label(container, padding=3, fmt="%d")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase7_risk_tier_distribution.png"), dpi=160)
    plt.close(fig)

    # 2. probability distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(analysis["churn_probability"], bins=25, color="#6c5ce7", edgecolor="black")
    ax.axvline(artifact.threshold, color="red", linestyle="--", label=f"Phase 6 threshold = {artifact.threshold}")
    ax.set_title("Phase 7 Churn Probability Distribution")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Customer count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase7_churn_probability_distribution.png"), dpi=160)
    plt.close(fig)

    # 3. top feature importance
    preprocessor = artifact.pipeline.named_steps["preprocessor"]
    model = artifact.pipeline.named_steps["model"]
    encoded_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "feature": [name.split("__", 1)[1] if "__" in name else name for name in encoded_names],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_title("Top 15 Gradient Boosting Feature Importances")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Encoded feature")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase7_top_feature_importance.png"), dpi=160)
    plt.close(fig)

    high = analysis[analysis["risk_tier"] == "HIGH"]

    # 4. contract distribution high risk
    fig, ax = plt.subplots(figsize=(8, 5))
    high["Contract"].value_counts().plot(kind="bar", color="salmon", ax=ax)
    ax.set_title("High-Risk Customer Contract Distribution")
    ax.set_xlabel("Contract type")
    ax.set_ylabel("Customer count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase7_high_risk_contract_distribution.png"), dpi=160)
    plt.close(fig)

    # 5. tenure distribution high risk
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(high["tenure"], bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_title("High-Risk Customer Tenure Distribution")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customer count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase7_high_risk_tenure_distribution.png"), dpi=160)
    plt.close(fig)

    # 6. monthly charges high risk
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(high["MonthlyCharges"], bins=20, kde=True, color="darkorange", ax=ax)
    ax.set_title("High-Risk Customer Monthly Charges")
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Customer count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase7_high_risk_monthly_charges.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    artifact = joblib.load(os.path.join(PROJECT_ROOT, "models", "final_churn_model.joblib"))
    analysis = build_customer_risk_analysis()
    summary = build_retention_summary(analysis)

    risk_columns = [
        "customerID",
        "churn_probability",
        "predicted_churn",
        "risk_tier",
        "priority",
        "reason",
        "recommended_action",
    ]
    analysis[risk_columns].sort_values("churn_probability", ascending=False).to_csv(
        os.path.join(REPORTS_DIR, "phase7_customer_risk_analysis.csv"),
        index=False,
    )
    summary.to_csv(os.path.join(REPORTS_DIR, "phase7_retention_summary.csv"), index=False)
    save_figures(analysis, artifact)

    print(f"Final model artifact: {artifact.model_name}, threshold={artifact.threshold}")
    print(f"Risk counts:\n{analysis['risk_tier'].value_counts().to_string()}")
    print(f"Rows exported: {len(analysis)}")


if __name__ == "__main__":
    main()
