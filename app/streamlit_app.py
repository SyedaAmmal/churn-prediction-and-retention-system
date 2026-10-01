"""Streamlit inference app for the final churn model and retention recommendations."""

from __future__ import annotations

import os

import streamlit as st

from src.prediction_service import DEFAULT_MODEL_PATH, build_customer_dataframe, load_model, predict_customer_churn


@st.cache_resource
def get_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the final model object once and reuse it for the app session."""
    return load_model(model_path)


def render_customer_form() -> dict:
    """Render the sidebar form and return the selected customer attributes."""
    with st.sidebar:
        st.header("Customer Information")
        customer = {
            "gender": st.selectbox("gender", ["Female", "Male"]),
            "SeniorCitizen": st.checkbox("Senior citizen", value=False),
            "Partner": st.selectbox("Partner", ["Yes", "No"]),
            "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
            "tenure": int(st.number_input("tenure (months)", min_value=0, step=1, value=12)),
            "PhoneService": st.selectbox("PhoneService", ["Yes", "No"]),
            "MultipleLines": st.selectbox("MultipleLines", ["Yes", "No", "No phone service"]),
            "InternetService": st.selectbox("InternetService", ["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"]),
            "OnlineBackup": st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"]),
            "DeviceProtection": st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"]),
            "TechSupport": st.selectbox("TechSupport", ["Yes", "No", "No internet service"]),
            "StreamingTV": st.selectbox("StreamingTV", ["Yes", "No", "No internet service"]),
            "StreamingMovies": st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"]),
            "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": st.selectbox("PaperlessBilling", ["Yes", "No"]),
            "PaymentMethod": st.selectbox(
                "PaymentMethod",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            ),
            "MonthlyCharges": float(st.number_input("MonthlyCharges ($)", min_value=0.0, step=0.01, value=65.0)),
            "TotalCharges": float(st.number_input("TotalCharges ($)", min_value=0.0, step=0.01, value=1200.0)),
        }
        return customer


def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction & Retention", layout="wide")
    st.title("Customer Churn Prediction & Retention")
    st.caption("Enter customer information to estimate churn risk and review potential retention actions.")

    try:
        artifact = get_model(DEFAULT_MODEL_PATH)
    except FileNotFoundError as exc:
        st.error(f"Model artifact is missing: {exc}")
        return
    except ValueError as exc:
        st.error(f"Model artifact is invalid: {exc}")
        return

    customer = render_customer_form()
    customer_df = build_customer_dataframe(customer)

    with st.form("prediction_form"):
        submitted = st.form_submit_button("Predict churn risk")

    if not submitted:
        st.info("Use the sidebar to enter customer attributes and click Predict churn risk.")
        return

    try:
        result = predict_customer_churn(customer, artifact=artifact)
    except ValueError as exc:
        st.error(f"Input validation failed: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Prediction failed: {exc}")
        return

    probability_pct = result["churn_probability"] * 100
    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn probability", f"{probability_pct:.1f}%")
    col2.metric("Prediction", "High churn risk" if result["predicted_churn"] == 1 else "Low churn risk")
    col3.metric("Risk tier", result["risk_tier"])

    st.progress(float(result["churn_probability"]))
    st.caption(f"Decision threshold: {artifact.threshold:.2f} (probability >= threshold is HIGH risk)")

    st.subheader("Risk Explanation")
    reason_text = result["retention_recommendation"]["reason"]
    st.write(reason_text)

    st.subheader("Retention Strategy")
    rec = result["retention_recommendation"]
    st.write(f"Priority: {rec['priority']}")
    st.write(f"Recommended action: {rec['recommended_action']}")

    st.subheader("Customer Summary")
    st.dataframe(customer_df, use_container_width=True)

    with st.expander("About this model"):
        st.markdown(
            """
            - Model: Gradient Boosting
            - Decision threshold: {threshold}
            - Purpose: customer churn risk estimation
            - Risk classification: probability-based
            - Important limitation: model predictions are probabilistic and historical associations do not establish causation.
            """.format(threshold=artifact.threshold)
        )


if __name__ == "__main__":
    main()
