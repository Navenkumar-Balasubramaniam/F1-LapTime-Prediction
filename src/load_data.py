from __future__ import annotations

from pathlib import Path
import pandas as pd

"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""

"""
Data loading module.

Purpose:
- Provide a single, reliable entry point for reading raw data artifacts.
- Keep file-format specifics (parquet/csv/etc.) out of
    training/evaluation code.

Notes:
- This project expects a local parquet file (e.g., data/raw/f1_all.parquet).
- Parquet reading requires a parquet engine such as `pyarrow` (recommended)
    or `fastparquet`.
"""


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Load raw dataset from disk.

    Parameters
    ----------
    raw_data_path : Path
        Path to the raw dataset file. Expected: .parquet

    Returns
    -------
    pd.DataFrame
        Loaded raw data.

    Raises
    ------
    FileNotFoundError
        If raw_data_path does not exist.
    ValueError
        If file suffix is not supported.
    ImportError
        If parquet engine is missing.
    """
    raw_data_path = Path(raw_data_path)

    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {raw_data_path}\n"
            "Expected a local parquet file. Ensure it exists and is available."
        )

    suffix = raw_data_path.suffix.lower()

    if suffix == ".parquet":
        try:
            df = pd.read_parquet(raw_data_path)
        except ImportError as e:
            raise ImportError(
                "Parquet support requires a parquet engine.\n"
                "Install one of: pyarrow (recommended) or fastparquet.\n"
                "Example: conda install -c conda-forge pyarrow"
            ) from e
        return df

    raise ValueError(
        f"Unsupported raw data format: '{suffix}'. "
        "This pipeline currently supports: .parquet"
    )
