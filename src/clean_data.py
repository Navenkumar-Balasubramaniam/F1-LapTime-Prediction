"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value handling.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value handling.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:

    # Basic validation
    if df_raw is None:
        raise ValueError("Input dataframe is None.")

    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw must be a pandas DataFrame.")

    if target_column not in df_raw.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    # Work on a copy
    df_clean = df_raw.copy()

    # Drop rows where target is missing
    df_clean = df_clean.dropna(subset=[target_column])

    # Drop rows where position_y is missing (if it exists)
    if "position_y" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["position_y"])

    # Final safety check
    if target_column not in df_clean.columns:
        raise RuntimeError("Target column was removed during cleaning.")

    if df_clean is None or df_clean.empty:
        raise ValueError("Cleaning resulted in empty dataframe.")

    return df_clean