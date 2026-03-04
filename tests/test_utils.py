"""
Unit tests for src/utils.py

Purpose of this file:
- Verify that file I/O behavior is correct.
- Ensure reproducibility of artifacts (CSV + model files).
- Catch regressions if someone modifies utils.py later.

We use pytest and its built-in tmp_path fixture so that:
- Tests do NOT write to your real filesystem.
- Each test runs in an isolated temporary directory.
- Tests are safe and repeatable.
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# Import the functions we want to test
from src.utils import load_csv, load_model, save_csv, save_model


# ----------------------------------------------------------------------
# Test 1: save_csv creates parent directories and writes file correctly
# ----------------------------------------------------------------------
def test_save_csv_creates_parent_dirs_and_writes_file(tmp_path: Path) -> None:
    """
    Goal:
    - Ensure save_csv creates missing parent directories.
    - Ensure the saved CSV matches the original DataFrame.
    """

    # Create a simple test DataFrame
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    # Define a nested path inside the temporary directory
    # This path does NOT exist yet.
    out_path = tmp_path / "nested" / "dir" / "data.csv"

    # Verify that parent directory does not exist before saving
    assert not out_path.parent.exists()

    # Call the function under test
    save_csv(df, out_path)

    # After saving:
    # - Parent directory should exist
    # - File should exist
    assert out_path.parent.exists()
    assert out_path.exists()

    # Load file directly with pandas to verify content
    df_loaded = pd.read_csv(out_path)

    # Ensure data is identical (index=False in utils)
    pd.testing.assert_frame_equal(df_loaded, df)


# ----------------------------------------------------------------------
# Test 2: load_csv reads expected DataFrame correctly
# ----------------------------------------------------------------------
def test_load_csv_reads_expected_dataframe(tmp_path: Path) -> None:
    """
    Goal:
    - Ensure load_csv properly reads an existing CSV file.
    """

    # Create sample data
    df = pd.DataFrame({"num": [0.1, 0.2], "cat": ["A", "B"]})

    # Write it manually to disk using pandas
    csv_path = tmp_path / "input.csv"
    df.to_csv(csv_path, index=False)

    # Now use our utility function to load it
    df_loaded = load_csv(csv_path)

    # Verify loaded data matches original
    pd.testing.assert_frame_equal(df_loaded, df)


# ----------------------------------------------------------------------
# Test 3: save_model + load_model roundtrip behavior
# ----------------------------------------------------------------------
def test_save_model_and_load_model_roundtrip(tmp_path: Path) -> None:
    """
    Goal:
    - Ensure model artifacts can be saved and loaded.
    - Ensure loaded model behaves identically.
    """

    # Create simple dataset
    X = pd.DataFrame({"x1": [0, 1, 2, 3], "x2": [1, 1, 0, 0]})
    y = pd.Series([0, 1, 1, 0])

    # Train a simple model
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)

    # Define a nested model output path
    model_path = tmp_path / "models" / "nested" / "model.joblib"

    # Ensure parent directory does not exist before saving
    assert not model_path.parent.exists()

    # Save model using our utility
    save_model(model, model_path)

    # After saving:
    # - Directory must exist
    # - Model file must exist
    assert model_path.parent.exists()
    assert model_path.exists()

    # Load model back
    loaded = load_model(model_path)

    # Compare predictions from original vs loaded model
    preds_original = model.predict(X)
    preds_loaded = loaded.predict(X)

    # Ensure predictions are identical
    assert (preds_original == preds_loaded).all()


# ----------------------------------------------------------------------
# Test 4: load_model should fail if file does not exist
# ----------------------------------------------------------------------
def test_load_model_raises_if_missing(tmp_path: Path) -> None:
    """
    Goal:
    - Ensure load_model fails properly when artifact is missing.
    - This prevents silent deployment failures.
    """

    missing_path = tmp_path / "does_not_exist.joblib"

    # Expect FileNotFoundError from joblib.load
    with pytest.raises(FileNotFoundError):
        load_model(missing_path)
