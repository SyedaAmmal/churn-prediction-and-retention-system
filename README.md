# AI-Powered Customer Churn Prediction & Retention System

## Overview

An end-to-end machine-learning project that estimates customer churn probability and translates that estimate into transparent, deterministic retention guidance. The repository includes analysis notebooks, reusable preprocessing and inference code, a saved Gradient Boosting model, aggregate reports, and a Streamlit prediction interface.

## Business Problem

Customer churn can reduce recurring revenue and increase replacement costs. This project explores how historical customer and service attributes relate to churn, evaluates classification tradeoffs, and demonstrates how model probabilities can support prioritization. Predictions and suggested actions are decision-support outputs, not guarantees of future behavior or proven churn-reduction interventions.

## Dataset

The project uses the IBM Telco Customer Churn dataset, containing 7,043 customer records and 21 columns, with `Churn` as the target. The raw and processed data files are intentionally excluded from public Git.

Obtain the dataset separately from the [IBM Telco Customer Churn public repository](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) and place the downloaded CSV at:

```text
data/raw/Telco-Customer-Churn.csv
```

The notebooks and data-processing workflow expect this relative path. The processed dataset, when generated, is stored at `data/processed/telco_churn_cleaned.csv`. No licensing claim is made here; consult the dataset source for its applicable terms.

## Machine Learning Workflow

Raw data → data understanding → cleaning → preprocessing → exploratory analysis → feature engineering and preprocessing → baseline models → optimization → threshold analysis → final model → business interpretation → retention strategy → Streamlit prediction interface.

## Baseline Model Results

Phase 4 compared five baseline classifiers on the established stratified holdout partition at the default classification threshold.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8420 |
| Decision Tree | 0.7182 | 0.4698 | 0.4786 | 0.4742 | 0.6415 |
| Random Forest | 0.7821 | 0.6128 | 0.4866 | 0.5425 | 0.8183 |
| Gradient Boosting | 0.8027 | 0.6655 | 0.5160 | 0.5813 | 0.8433 |
| XGBoost | 0.7764 | 0.5897 | 0.5187 | 0.5519 | 0.8205 |

## Model Optimization

Phase 6 tuned Logistic Regression, Random Forest, Gradient Boosting, and XGBoost using training-only cross-validation and an internal validation split. Gradient Boosting was selected based on its validation F1 score. Its selected decision threshold is **0.35**. The final saved artifact is `models/final_churn_model.joblib`.

## Final Holdout Evaluation

At the selected threshold, the recorded Phase 6 holdout metrics are:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.7814 | 0.5699 | 0.7193 | 0.6359 | 0.8454 |

| Confusion matrix | Actual: No Churn | Actual: Churn |
| --- | ---: | ---: |
| Predicted: No Churn | TN = 832 | FN = 105 |
| Predicted: Churn | FP = 203 | TP = 269 |

**Evaluation caveat:** This holdout was excluded from Phase 6 tuning and threshold selection, but it had already been evaluated in Phases 4–5. It is an established confirmation set, not a completely never-before-seen dataset.

## Business Interpretation

The analysis observed associations between churn and month-to-month contracts, shorter tenure, fiber optic internet, higher monthly charges, electronic check payments, and lack of services such as Online Security and Tech Support. These are patterns in historical observational data; they do not establish causal relationships.

## Retention Strategy

Phase 7 applies deterministic, reviewable rules using customer attributes and model probability. Probability below 0.35 maps to LOW risk; probability at or above 0.35 maps to HIGH risk. Priority is low below 0.35, medium from 0.35 to below 0.60, and high at or above 0.60. The recommendation logic may suggest onboarding outreach, contract or plan review, pricing/value review, support/security offers, or payment support, depending on available attributes. LOW-risk customers receive routine monitoring guidance. These recommendations are candidate actions and have not been experimentally proven to reduce churn.

## Streamlit Application

The application accepts customer attributes, loads the saved model, produces a churn probability, applies the artifact's stored 0.35 threshold, determines LOW/HIGH risk, and displays the corresponding deterministic retention recommendation. It does not retrain the model.

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                         # Local input; excluded from Git
│   └── processed/                   # Local processed data; excluded from Git
├── models/
│   └── final_churn_model.joblib
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_model_interpretation.ipynb
│   ├── 05_model_optimization.ipynb
│   └── 06_business_interpretation_and_retention.ipynb
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── phase6_final_test_metrics.csv
│   ├── phase6_model_comparison.csv
│   ├── phase6_threshold_analysis.csv
│   ├── phase7_customer_risk_analysis.csv  # Local/private; excluded from Git
│   └── phase7_retention_summary.csv
├── src/
│   ├── data_preprocessing.py
│   ├── generate_phase7_business_outputs.py
│   ├── model_optimization.py
│   ├── model_training.py
│   ├── prediction_service.py
│   └── retention_strategy.py
├── tests/
├── README.md
└── requirements.txt
```

The raw dataset, processed dataset, and customer-level Phase 7 risk report remain local and are excluded from Git. Aggregate reports, figures, notebooks, code, tests, application files, and the final model artifact are intended to remain available in the public repository.

## Installation

From the project root, create and activate a fresh virtual environment, then install the listed dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Application

```powershell
streamlit run app/streamlit_app.py
```

## Run Tests

```powershell
python -m pytest -q
```

## Limitations

- The dataset is a historical, cross-sectional snapshot and does not include longitudinal usage, service-quality, competitor, or customer-satisfaction information.
- Observed patterns and model feature importance are not causal evidence; unobserved confounders may affect the results.
- Model probabilities are estimates, and the selected threshold may overfit the finite validation sample. Probability calibration was not the primary objective.
- The holdout was previously examined during Phases 4–5, so its Phase 6 results are not a fully independent, never-before-seen evaluation.
- No real business cost optimization, A/B testing, or causal inference was performed. Retention suggestions are unvalidated candidate actions.

## Technologies

Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly, Streamlit, Joblib, and Jupyter.
