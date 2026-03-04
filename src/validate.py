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

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

import pandas as pd


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
    print("[validate.validate_dataframe] Validating dataframe...")  # TODO: replace with logging later

    if df is None or df.empty:
        raise ValueError("Validation failed: dataframe is empty. Cannot proceed.")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: missing required columns: {missing}")
    # Check required columns are not entirely null
    all_null = [c for c in required_columns if df[c].isna().all()]
    if all_null:
        raise ValueError(f"Validation failed: required columns are entirely null: {all_null}")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add simple dataset-specific checks (types, ranges, unique constraints)
    # Why: Different datasets have different minimum quality bars.
    #
    # Examples:
    # 1) assert df[target_column].notna().all()
    # 2) assert (df['year'] >= 1950).all()
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return True