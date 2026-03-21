"""
Educational Goal:
- Why this module exists in an MLOps system: Keep all dataset cleaning and
  leakage-prevention logic in one place so training, evaluation, and inference
  always see consistent inputs.
- Responsibility (separation of concerns): Cleaning rules should not be mixed
  into model training code; that makes systems brittle and hard to debug.
- Pipeline contract (inputs and outputs): Takes a raw pandas DataFrame and
  returns a cleaned pandas DataFrame ready for splitting and feature building.

This version replaces print statements with standard logging so that runs are
traceable in local development, CI, and experiment tracking systems.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from src.logging import get_logger

logger = get_logger(__name__)


def _load_config(config_path: Path = Path("config.yaml")) -> dict:
    """
    Internal helper to load YAML config.
    Keeps config parsing out of main logic.
    """
    if not config_path.exists():
        logger.warning("Config file not found at %s. Using empty cleaning config.", config_path)
        return {}
    with config_path.open("r", encoding="utf-8") as f:
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
        logger.error("Input dataframe is None.")
        raise ValueError("Input dataframe is None.")

    if not isinstance(df_raw, pd.DataFrame):
        logger.error("df_raw must be a pandas DataFrame. Got type=%s", type(df_raw))
        raise TypeError("df_raw must be a pandas DataFrame.")
    
    logger.info("Starting dataframe cleaning. Input shape=%s", None if df_raw is None else df_raw.shape)

    if df_raw.empty:
        logger.error("Input dataframe is empty.")
        raise ValueError("Input dataframe is empty.")

    if target_column not in df_raw.columns:
        logger.error("Target column '%s' not found in dataframe.", target_column)
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
    logger.info("Target column '%s' converted to numeric with coercion.", target_column)

    # 2) Drop rows missing required columns (from config)
    existing_dropna_cols = [c for c in dropna_subset if c in df_clean.columns]
    if existing_dropna_cols:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=existing_dropna_cols)
        logger.info(
            "Dropped %s rows due to missing required values in columns=%s",
            before - len(df_clean),
            existing_dropna_cols,
        )

    # 3) Drop columns (from config)
    existing_drop_cols = [c for c in drop_columns if c in df_clean.columns and c != target_column]
    if existing_drop_cols:
        df_clean = df_clean.drop(columns=existing_drop_cols, errors="ignore")
        logger.info("Dropped configured columns: %s", existing_drop_cols)

    # 4) Normalize missing markers (object columns only)
    obj_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        df_clean[obj_cols] = (
            df_clean[obj_cols]
            .replace({None: "UNKNOWN", "None": "UNKNOWN"})
            .fillna("UNKNOWN")
        )
        logger.info("Filled missing values in object/string columns with 'UNKNOWN'.")

    # 5) Outlier trimming using IQR (still notebook-aligned)
    if df_clean[target_column].shape[0] > 10:
        q1 = df_clean[target_column].quantile(0.25)
        q3 = df_clean[target_column].quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            before = len(df_clean)
            df_clean = df_clean[(df_clean[target_column] >= lower) & (df_clean[target_column] <= upper)]
            logger.info(
                "Applied IQR outlier trimming on target '%s'. Removed %s rows.",
                target_column,
                before - len(df_clean),
            )

    if df_clean.empty:
        logger.error("Cleaning resulted in empty dataframe.")
        raise ValueError("Cleaning resulted in empty dataframe.")

    logger.info("Finished cleaning dataframe. Output shape=%s", df_clean.shape)
    return df_clean
