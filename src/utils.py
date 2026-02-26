"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize file I/O so the
pipeline has one trusted way to read/write data and models.
- Responsibility (separation of concerns): Keep filesystem details out of
modeling code to reduce bugs and make deployment simpler.
- Pipeline contract (inputs and outputs): Reads/writes CSVs and joblib model
artifacts using Path objects.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

from pathlib import Path

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to a CSV file on disk.
    Outputs:
    - A pandas DataFrame loaded from the CSV.
    Why this contract matters for reliable ML delivery:
    - If every module loads data the same way, you eliminate silent
    differences (encoding, separators, index handling).
    """
    print(f"[utils.load_csv] Loading CSV from: {filepath}")
    # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    """
    TODO_STUDENT: Adjust read_csv parameters
    ONLY if needed (encoding, sep, dtype, etc.)
    """
    """
    Why: File formats differ across businesses; the pipeline should still
    have one controlled entry point.
    """
    # Examples:
    # 1. df = pd.read_csv(filepath, encoding="utf-8")
    # 2. df = pd.read_csv(filepath, sep=";")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to write.
    - filepath: Path where the CSV should be saved.
    Outputs:
    - None (writes a file as a side-effect).
    Why this contract matters for reliable ML delivery:
    - Artifact paths become predictable, enabling automation, debugging,
    and CI validation.
    """
    print(f"[utils.save_csv] Saving CSV to: {filepath}")
    # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    """
    TODO_STUDENT: Adjust to_csv parameters ONLY
    if needed (index, quoting, etc.)
    """
    """
    Why: Some teams want indexes saved or special
    formatting for downstream systems.
    """
    # Examples:
    # 1. df.to_csv(filepath, index=True)
    # 2. df.to_csv(filepath, index=False, encoding="utf-8")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: Any trained model object (typically a scikit-learn Pipeline).
    - filepath: Path where the model artifact should be saved.
    Outputs:
    - None (writes a file as a side-effect).
    Why this contract matters for reliable ML delivery:
    - A single save/load standard prevents “trained model can’t be reloaded”
    deployment failures.
    """
    print(f"[utils.save_model] Saving model to: {filepath}")
    # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    """
    TODO_STUDENT: Adjust joblib.dump parameters ONLY if
    needed (compression, protocol).
    """
    """
    Why: Model artifacts sometimes need compression or
    compatibility constraints.
    """
    # Examples:
    # 1. joblib.dump(model, filepath, compress=3)
    # 2. joblib.dump(model, filepath, protocol=4)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to a saved model artifact.
    Outputs:
    - The loaded model object.
    Why this contract matters for reliable ML delivery:
    - Inference and evaluation must load the exact same artifact
    produced by training.
    """
    print(f"[utils.load_model] Loading model from: {filepath}")
    # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add checks or alternate loading behavior ONLY if needed.
    """
    Why: Some teams version artifacts or store them differently;
    keep changes localized here.
    """
    # Examples:
    # 1. assert filepath.exists(), "Model file missing!"
    # 2. model = joblib.load(filepath)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return joblib.load(filepath)
