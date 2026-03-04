"""
Educational Goal:
- Why this module exists in an MLOps system: Simple, reliable I/O helpers to persist datasets
  and models so other modules avoid duplicating file handling logic.
- Responsibility (separation of concerns): Reading/writing CSVs and models only; no ML logic.
- Pipeline contract (inputs and outputs): Read/Write DataFrames and joblib model artifacts.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to CSV file to read
    Outputs:
    - pd.DataFrame loaded from CSV
    Why this contract matters for reliable ML delivery:
    - Centralizing CSV reading makes it easy to add parsing options, compression, or schema checks later.
    """
    print(f"[utils.load_csv] Loading CSV from: {filepath}")  # TODO: replace with logging later
    df = pd.read_csv(filepath)
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: If your data requires custom parsing (date columns, separators, encoding),
    # add those parameters here. Why: Real datasets often require non-default read_csv args.
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder: nothing required for baseline
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
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
    print(f"[utils.save_csv] Saving DataFrame to: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: If you want different CSV options (compression, quoting, float_format),
    # change them here. Why: Large datasets often need compression or specific float formats.
    #
    # Placeholder: baseline uses index=False
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------


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
    print(f"[utils.save_model] Saving model to: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Consider adding metadata (config, timestamp, metrics) alongside the model.
    # Why: Experiments are reproducible when artifact metadata is stored.
    #
    # Placeholder: baseline saves the raw joblib file only
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to model artifact
    Outputs:
    - model object loaded via joblib
    Why this contract matters for reliable ML delivery:
    - Centralized model loading ensures consistent behavior across evaluation and inference scripts.
    """
    print(f"[utils.load_model] Loading model from: {filepath}")  # TODO: replace with logging later
    return joblib.load(filepath)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add validation of model's expected attributes (e.g., predict) if desired.
    # Why: Early failure helps diagnose mismatched pipelines or corrupted artifacts.
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
