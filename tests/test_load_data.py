"""
Unit tests for src/load_data.py

What we are testing:
- load_raw_data reads local parquet correctly.
- It fails loudly and clearly for missing files.
- It rejects unsupported file extensions.

Why these tests matter in MLOps:
- Data loading is a critical pipeline entry point.
- If loading is unreliable, training/evaluation/inference become
non-reproducible.
- Loud failures are better than silent corruption.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.load_data import load_raw_data


# ----------------------------------------------------------------------
# Test 1: Successful load of a parquet file
# ----------------------------------------------------------------------
def test_load_raw_data_reads_parquet_correctly(tmp_path: Path) -> None:
    """
    Goal:
    - Confirm load_raw_data() loads a parquet file and returns the
    same DataFrame.

    Why tmp_path:
    - pytest gives us a clean, temporary sandbox directory for each test.
    - No files are written into your real repo (no messy data/raw artifacts).
    """

    # Create a small deterministic DataFrame (simple types)
    df_expected = pd.DataFrame(
        {
            "driver": ["hamilton", "verstappen"],
            "points": [25, 18],
            "team": ["mercedes", "redbull"],
        }
    )

    # Write it as parquet inside the temporary directory
    parquet_path = tmp_path / "f1_all.parquet"
    df_expected.to_parquet(parquet_path, index=False)

    # Call the function under test
    df_loaded = load_raw_data(parquet_path)

    # Verify we got a DataFrame back
    assert isinstance(df_loaded, pd.DataFrame)

    # Verify contents match (column order & values)
    """
    Note: parquet roundtrips should preserve data types for
    these simple columns.
    """
    pd.testing.assert_frame_equal(df_loaded, df_expected)


# ----------------------------------------------------------------------
# Test 2: Missing file should raise FileNotFoundError
# ----------------------------------------------------------------------
def test_load_raw_data_raises_file_not_found_for_missing_path(tmp_path: Path) -> None:
    """
    Goal:
    - Ensure missing raw file triggers a loud failure (FileNotFoundError).

    Why this matters:
    - In production, missing data should fail the pipeline early,
      not continue with garbage defaults.
    """

    missing_path = tmp_path / "missing.parquet"

    # File doesn't exist
    assert not missing_path.exists()

    # Expect FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        load_raw_data(missing_path)

    # Optional: verify error message contains useful info
    assert "Raw data file not found" in str(exc_info.value)


# ----------------------------------------------------------------------
# Test 3: Unsupported extension should raise ValueError
# ----------------------------------------------------------------------
def test_load_raw_data_raises_value_error_for_unsupported_extension(tmp_path: Path) -> None:
    """
    Goal:
    - Ensure the loader rejects non-parquet formats (since your function
    supports only .parquet).

    Why this matters:
    - Prevents silent incorrect reads and inconsistent pipeline behavior.
    """

    # Create an empty dummy file with an unsupported suffix
    bad_path = tmp_path / "data.csv"
    bad_path.write_text("a,b\n1,2\n")

    # Expect ValueError due to unsupported suffix
    with pytest.raises(ValueError) as exc_info:
        load_raw_data(bad_path)

    # Optional: verify it explains the problem clearly
    assert "Unsupported raw data format" in str(exc_info.value)
