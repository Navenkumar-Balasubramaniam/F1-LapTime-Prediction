import pandas as pd
import pytest

from src.validate import validate_dataframe


def test_validate_passes():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert validate_dataframe(df, ["a", "b"]) is True


def test_validate_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        validate_dataframe(df, ["a", "b"])


def test_validate_empty_df():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_dataframe(df, ["a"])


def test_validate_all_null_column():
    df = pd.DataFrame({"a": [None, None], "b": [1, 2]})
    with pytest.raises(ValueError):
        validate_dataframe(df, ["a", "b"])