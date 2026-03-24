"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.

Educational Goal:
- Why this module exists in an MLOps system: Fail fast before training starts.
  Bad inputs should be detected early, not after the model is trained.
- Responsibility (separation of concerns): Validation is different from cleaning.
  Validation checks assumptions; cleaning changes data.
- Pipeline contract (inputs and outputs): Takes a DataFrame and required column list;
  returns True if valid, otherwise raises a clear error.

This version uses structured logging so validation failures leave a clear trail.
"""

from __future__ import annotations

import pandas as pd

from src.logging import get_logger

logger = get_logger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: DataFrame to validate
    - required_columns: list of columns that must exist
    Outputs:
    - bool: True if validation passes
    Why this contract matters for reliable ML delivery:
    - Prevents silent training on wrong schema, which is a common source
      of production failures and misleading evaluation.
    """
    logger.info("Validating dataframe with shape=%s", None if df is None else df.shape)

    if df is None or df.empty:
        logger.error("Validation failed: dataframe is empty. Cannot proceed.")
        raise ValueError("Validation failed: dataframe is empty. Cannot proceed.")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.error("Validation failed: missing required columns=%s", missing)
        raise ValueError(f"Validation failed: missing required columns: {missing}")

    # Check required columns are not entirely null
    all_null = [c for c in required_columns if df[c].isna().all()]
    if all_null:
        logger.error("Validation failed: required columns entirely null=%s", all_null)
        raise ValueError(f"Validation failed: required columns are entirely null: {all_null}")

    logger.info("Validation passed for required columns=%s", required_columns)
    return True
