"""
Educational Goal:
- Why this module exists in an MLOps system: Simple, reliable I/O helpers to persist datasets
  and models so other modules avoid duplicating file handling logic.
- Responsibility (separation of concerns): Reading/writing CSVs and models only; no ML logic.
- Pipeline contract (inputs and outputs): Read/Write DataFrames and joblib model artifacts.

This version uses logging instead of print to make artifact I/O visible in both
console output and persisted run logs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.logging import get_logger

logger = get_logger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to CSV file to read
    Outputs:
    - pd.DataFrame loaded from CSV
    Why this contract matters for reliable ML delivery:
    - Centralizing CSV reading makes it easy to add parsing options, compression, or schema checks later.
    """
    logger.info("Loading CSV from %s", filepath)
    df = pd.read_csv(filepath)
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save
    - filepath: Path to write CSV to
    Outputs:
    - None (writes file to disk)
    Why this contract matters for reliable ML delivery:
    - Ensures parent dirs exist and all CSV writes use a consistent default (index=False).
    """
    logger.info("Saving DataFrame to %s", filepath)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)



def save_model(model: Any, filepath: Path) -> None:
    """
    Inputs:
    - model: Any picklable model object (sklearn Pipeline, etc.)
    - filepath: Path to write the model to
    Outputs:
    - None (writes model artifact)
    Why this contract matters for reliable ML delivery:
    - Having a single model save function makes it trivial to swap serialization formats later.
    """
    logger.info("Saving model to %s", filepath)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)



def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to model artifact
    Outputs:
    - model object loaded via joblib
    Why this contract matters for reliable ML delivery:
    - Centralized model loading ensures consistent behavior across evaluation and inference scripts.
    """
    logger.info("Loading model from %s", filepath)
    return joblib.load(filepath)
