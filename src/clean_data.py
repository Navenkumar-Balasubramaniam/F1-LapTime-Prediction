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
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml
in a later session
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import yaml


def _load_config(config_path: Path = Path("config.yaml")) -> dict:
    """
    Internal helper to load YAML config.
    Keeps config parsing out of main logic.
    """
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


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

    # Load cleaning rules from config.yaml (single source of truth)
    cfg = _load_config()
    cleaning_cfg = (cfg.get("cleaning") or {})

    dropna_subset = cleaning_cfg.get("dropna_subset", [])
    drop_columns = cleaning_cfg.get("drop_columns", [])

    # Always enforce target exists in dropna_subset (if user forgot)
    if target_column not in dropna_subset:
        dropna_subset = [target_column] + list(dropna_subset)

    # 1) Type enforcement for target
    df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors="coerce")

    # 2) Drop rows missing required columns (from config)
    existing_dropna_cols = [c for c in dropna_subset if c in df_clean.columns]
    if existing_dropna_cols:
        df_clean = df_clean.dropna(subset=existing_dropna_cols)

    # 3) Drop columns (from config)
    existing_drop_cols = [c for c in drop_columns if c in df_clean.columns]
    if existing_drop_cols:
        df_clean = df_clean.drop(columns=existing_drop_cols, errors="ignore")

    # 4) Normalize missing markers (object columns only)
    obj_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        df_clean[obj_cols] = (
            df_clean[obj_cols]
            .replace({None: "UNKNOWN", "None": "UNKNOWN"})
            .fillna("UNKNOWN")
        )

    # 5) Outlier trimming using IQR (still notebook-aligned)
    if df_clean[target_column].shape[0] > 10:
        q1 = df_clean[target_column].quantile(0.25)
        q3 = df_clean[target_column].quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df_clean = df_clean[(df_clean[target_column] >= lower) & (df_clean[target_column] <= upper)]

    if df_clean.empty:
        raise ValueError("Cleaning resulted in empty dataframe.")

    return df_clean
