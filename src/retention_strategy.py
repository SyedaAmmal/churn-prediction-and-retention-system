"""Transparent business-oriented retention recommendation rules for Phase 7.

The function below purposely avoids any hidden model logic, probabilistic guessing,
or claims of causality. It simply interprets available customer attributes and the
model-produced churn probability into clear, deterministic, reviewable actions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


RISK_THRESHOLD = 0.35


def _safe_value(customer_row: Any, key: str, default: Optional[Any] = None) -> Any:
    """Get a row value from either dict-like or pandas-like objects."""
    if customer_row is None:
        return default
    if isinstance(customer_row, dict):
        return customer_row.get(key, default)
    if hasattr(customer_row, "get"):
        try:
            return customer_row.get(key, default)
        except Exception:
            pass
    if hasattr(customer_row, key):
        return getattr(customer_row, key)
    return default


def _normalize_text(value: Any) -> str:
    """Convert an attribute to a normalized string for deterministic comparisons."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float while handling missing values safely."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_retention_recommendation(customer_row: Any, churn_probability: float) -> Dict[str, str]:
    """Return a deterministic, transparent customer retention recommendation.

    Parameters
    ----------
    customer_row : Any
        A dict-like or pandas Series containing the customer attributes available to
        the business analysis layer.
    churn_probability : float
        Model-estimated churn probability from the final Gradient Boosting pipeline.

    Returns
    -------
    Dict[str, str]
        Structured recommendation with risk tier, priority, reason, and action.
    """
    probability = _to_float(churn_probability, default=0.0)
    risk_tier = "HIGH" if probability >= RISK_THRESHOLD else "LOW"

    if probability >= 0.60:
        priority = "high"
    elif probability >= 0.35:
        priority = "medium"
    else:
        priority = "low"

    contract = _normalize_text(_safe_value(customer_row, "Contract"))
    tenure = _to_float(_safe_value(customer_row, "tenure"), default=float("nan"))
    monthly_charges = _to_float(_safe_value(customer_row, "MonthlyCharges"), default=float("nan"))
    internet_service = _normalize_text(_safe_value(customer_row, "InternetService"))
    online_security = _normalize_text(_safe_value(customer_row, "OnlineSecurity"))
    tech_support = _normalize_text(_safe_value(customer_row, "TechSupport"))
    payment_method = _normalize_text(_safe_value(customer_row, "PaymentMethod"))
    paperless = _normalize_text(_safe_value(customer_row, "PaperlessBilling"))
    streaming_tv = _normalize_text(_safe_value(customer_row, "StreamingTV"))
    streaming_movies = _normalize_text(_safe_value(customer_row, "StreamingMovies"))

    reasons: List[str] = []
    actions: List[str] = []

    if contract.lower() == "month-to-month":
        reasons.append("Month-to-month contract indicates elevated churn risk.")
        actions.append("Potential contract-retention outreach and plan review.")
    elif contract.lower() in {"one year", "two year"} and risk_tier == "LOW":
        reasons.append("Longer-term contract is associated with lower churn risk.")
        actions.append("Monitor and maintain relationship with routine service review.")

    if not pd_isnan(tenure) and tenure <= 12:
        reasons.append("Short tenure suggests early-life churn exposure.")
        actions.append("Prioritize onboarding and value-check outreach.")

    if not pd_isnan(monthly_charges) and monthly_charges >= 70:
        reasons.append("High monthly charges are associated with elevated churn risk.")
        actions.append("Review pricing, included value, and plan-fit options.")

    if internet_service.lower() == "fiber optic":
        reasons.append("Fiber optic internet service is associated with higher churn risk.")
        actions.append("Review connectivity value and service satisfaction touchpoints.")

    if online_security.lower() in {"no", "unknown", ""}:
        reasons.append("Online security is not active, which is associated with higher churn risk.")
        actions.append("Discuss security-product adoption and digital protection offer.")

    if tech_support.lower() in {"no", "unknown", ""}:
        reasons.append("Tech support is not active, which is associated with higher churn risk.")
        actions.append("Offer targeted support review and service guidance.")

    if payment_method.lower() == "electronic check":
        reasons.append("Electronic check payment is associated with higher churn risk.")
        actions.append("Discuss automatic payment enrollment and payment support options.")

    if paperless.lower() == "yes":
        reasons.append("Paperless billing is associated with elevated churn risk in this historical sample.")
        actions.append("Review billing communication preferences and account touchpoints.")

    if streaming_tv.lower() == "yes" or streaming_movies.lower() == "yes":
        reasons.append("Streaming service usage suggests a high-engagement service profile worth review.")
        actions.append("Check package fit and engagement-based retention conversations.")

    if not reasons:
        reasons.append("Customer risk is based on model probability and available attributes.")
        actions.append("Review account with routine service and retention monitoring.")

    if risk_tier == "HIGH":
        recommendation = "; ".join(actions)
        reason_text = "; ".join(reasons)
    else:
        recommendation = "Monitor account and maintain routine service review."
        reason_text = "; ".join(reasons) if reasons else "Customer risk is below the model operating threshold."

    return {
        "risk_tier": risk_tier,
        "priority": priority,
        "reason": reason_text,
        "recommended_action": recommendation,
    }


def pd_isnan(value: Any) -> bool:
    """Return True for NaN-like values without requiring pandas."""
    try:
        import math
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False
