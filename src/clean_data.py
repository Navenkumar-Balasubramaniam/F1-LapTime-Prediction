"""
Educational Goal:
- Why this module exists in an MLOps system: Keep all dataset cleaning and
  leakage-prevention logic in one place so training, evaluation, and inference
  always see consistent inputs.
- Responsibility (separation of concerns): Cleaning rules should not be mixed
  into model training code; that makes systems brittle and hard to debug.
- Pipeline contract (inputs and outputs): Takes a raw pandas DataFrame and
  returns a cleaned pandas DataFrame ready for splitting and feature building.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean raw dataset before splitting and modeling.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataframe from load_data.load_raw_data.
    target_column : str
        Name of the modeling target column.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe safe for downstream pipeline stages.

    Raises
    ------
    ValueError
        If input is invalid or cleaning results in empty data.
    TypeError
        If df_raw is not a pandas DataFrame.
    """

    if df_raw is None:
        raise ValueError("Input dataframe is None.")

    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw must be a pandas DataFrame.")

    if df_raw.empty:
        raise ValueError("Input dataframe is empty.")

    if target_column not in df_raw.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    df_clean = df_raw.copy()

    # 1) Type enforcement for target (convert bad strings → NaN)
    df_clean[target_column] = pd.to_numeric(
        df_clean[target_column], errors="coerce"
    )

    # 2) Drop rows missing key modeling columns
    required_non_null = [target_column]
    if "position_y" in df_clean.columns:
        required_non_null.append("position_y")

    df_clean = df_clean.dropna(subset=required_non_null)

    # 3) Drop redundant ID-like columns
    cols_to_drop_if_present = [
        "Driver",
        "LapNumber",
        "time",
        "circuitId",
        "driverId",
        "statusId",
        "raceId",
    ]

    df_clean = df_clean.drop(
        columns=[c for c in cols_to_drop_if_present if c in df_clean.columns],
        errors="ignore",
    )

    # 4) Leakage prevention
    # IMPORTANT: do NOT drop position_y (used in modeling)
    leakage_cols = ["position_x", "status"]

    df_clean = df_clean.drop(
        columns=[c for c in leakage_cols if c in df_clean.columns],
        errors="ignore",
    )

    # 5) Normalize missing markers (object columns only)
    obj_cols = df_clean.select_dtypes(include=["object"]).columns

    if len(obj_cols) > 0:
        df_clean[obj_cols] = (
            df_clean[obj_cols]
            .replace({None: "UNKNOWN", "None": "UNKNOWN"})
            .fillna("UNKNOWN")
        )

    # 6) Outlier trimming using IQR (if sufficient rows)
    if df_clean[target_column].shape[0] > 10:
        q1 = df_clean[target_column].quantile(0.25)
        q3 = df_clean[target_column].quantile(0.75)
        iqr = q3 - q1

        if pd.notna(iqr) and iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df_clean = df_clean[
                (df_clean[target_column] >= lower)
                & (df_clean[target_column] <= upper)
            ]

    # 7) Drop known highly correlated column if present
    if "AirTemp" in df_clean.columns:
        df_clean = df_clean.drop(columns=["AirTemp"], errors="ignore")

    # Final contract checks
    if target_column not in df_clean.columns:
        raise RuntimeError("Target column was removed during cleaning.")

    if df_clean.empty:
        raise ValueError("Cleaning resulted in empty dataframe.")

    return df_clean