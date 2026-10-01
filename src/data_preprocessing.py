"""
Data Preprocessing Module for Telco Customer Churn & Retention System.

This module provides clean, modular, and reproducible functions to ingest,
clean, and preprocess the Telco Customer Churn dataset according to Phase 3 requirements.
"""

import os
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Define canonical feature groupings
NUMERIC_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COL: str = "Churn"
IDENTIFIER_COLS: List[str] = ["customerID"]


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load the raw Telco churn dataset without altering source contents.

    Parameters
    ----------
    data_path : str
        Filepath to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
    return pd.read_csv(data_path)


def handle_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert TotalCharges column to numeric.

    In the raw Telco dataset, 11 records have whitespace ' ' in TotalCharges.
    All 11 records have tenure == 0 (new accounts that have not completed a billing cycle).
    These values are coerced to NaN and imputed with 0.0 to accurately represent
    their cumulative charges to date without dropping valid customer records.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with TotalCharges as float64 and missing values replaced with 0.0.
    """
    df = df.copy()
    # Coerce whitespace and non-numeric strings to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    
    # Impute tenure == 0 missing TotalCharges with 0.0
    zero_tenure_mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[zero_tenure_mask, "TotalCharges"] = 0.0
    
    # In case any other unexpected NaN exists, fill with 0.0 as safe fallback
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def drop_identifiers(df: pd.DataFrame, id_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove unique identifier columns (e.g., customerID) from the modeling dataset.

    `customerID` is a unique identifier with one distinct value per customer.
    It does not represent a meaningful, generalizable customer characteristic,
    so it is excluded from the predictive feature set.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_cols : list of str, optional
        List of identifier column names to drop. Defaults to IDENTIFIER_COLS.

    Returns
    -------
    pd.DataFrame
        DataFrame with identifier columns removed.
    """
    if id_cols is None:
        id_cols = IDENTIFIER_COLS
    cols_to_drop = [c for c in id_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Encode the target variable 'Churn' from string ('Yes'/'No') to binary integer (1/0).

    Mapping:
        'Yes' -> 1 (Customer churned)
        'No'  -> 0 (Customer retained)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Target column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary-encoded target column.
    """
    df = df.copy()
    if target_col in df.columns and not pd.api.types.is_numeric_dtype(df[target_col]):
        target_map = {"No": 0, "Yes": 1}
        df[target_col] = df[target_col].map(target_map).astype(int)
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute end-to-end data cleaning for Phase 3.

    1. Treats TotalCharges whitespace as missing and imputes tenure=0 records with 0.0.
    2. Drops customerID.
    3. Converts target 'Churn' from Yes/No to 1/0.
    4. Preserves discrete sub-categories ('No internet service', 'No phone service').

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for train/test split and modeling.
    """
    df_cleaned = df.copy()
    df_cleaned = handle_total_charges(df_cleaned)
    df_cleaned = drop_identifiers(df_cleaned)
    df_cleaned = encode_target(df_cleaned)
    return df_cleaned


def get_feature_target_split(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing features and target.
    target_col : str
        Name of target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Return lists of numeric and categorical feature names.

    Returns
    -------
    Tuple[List[str], List[str]]
        (numeric_features, categorical_features)
    """
    return NUMERIC_FEATURES.copy(), CATEGORICAL_FEATURES.copy()


def build_preprocessing_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True,
    drop_first: bool = False,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for feature transformations.

    IMPORTANT FOR LEAKAGE PREVENTION:
    This pipeline object must ONLY be fit on training data splits (X_train)
    during Phase 4 modeling, and transform applied to validation/test data (X_test).

    Parameters
    ----------
    numeric_features : list of str, optional
        List of numeric column names.
    categorical_features : list of str, optional
        List of categorical column names.
    scale_numeric : bool, default=True
        Whether to standardize numeric features using StandardScaler.
    drop_first : bool, default=False
        Whether OneHotEncoder should drop the first category per feature.

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer ready for integration into modeling pipelines.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    # Numeric sub-pipeline
    if scale_numeric:
        num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    else:
        num_transformer = "passthrough"

    # Categorical sub-pipeline: preserves 'No phone service' and 'No internet service'
    drop_param = "first" if drop_first else None
    cat_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    drop=drop_param,
                    sparse_output=False,
                    handle_unknown="ignore" if not drop_first else "infrequent_if_exist",
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned dataset to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    output_path : str
        Target destination filepath.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def validate_cleaned_data(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """
    Run automated validation checks on cleaned data against raw inputs.

    Checks:
    1. Raw dataset row count unchanged.
    2. Cleaned dataset contains 0 missing values.
    3. customerID is absent from cleaned data.
    4. Churn contains only binary values {0, 1}.
    5. TotalCharges is float dtype.
    6. Row count is preserved (7,043 rows).

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original raw DataFrame.
    cleaned_df : pd.DataFrame
        Cleaned DataFrame.

    Returns
    -------
    dict
        Dictionary of check results and boolean pass/fail status.
    """
    checks = {
        "raw_rows_unchanged": len(raw_df) == 7043,
        "cleaned_rows_preserved": len(cleaned_df) == len(raw_df),
        "no_missing_values": cleaned_df.isna().sum().sum() == 0,
        "customerID_absent": "customerID" not in cleaned_df.columns,
        "churn_binary_only": set(cleaned_df["Churn"].unique()).issubset({0, 1}),
        "totalcharges_is_numeric": pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"]),
        "totalcharges_zero_imputed": (
            cleaned_df.loc[raw_df["TotalCharges"].str.strip() == "", "TotalCharges"] == 0.0
        ).all(),
    }
    checks["all_passed"] = all(checks.values())
    return checks
