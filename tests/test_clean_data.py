import pandas as pd
import pytest

from src.clean_data import clean_dataframe


def test_clean_dataframe_happy_path_contract():
    df = pd.DataFrame({
        "TyreLife": [100, None, 200],
        "position_y": [1, 2, None]
    })

    cleaned = clean_dataframe(df, "TyreLife")

    assert isinstance(cleaned, pd.DataFrame)
    assert "TyreLife" in cleaned.columns
    assert cleaned.shape[0] == 1  # only first row survives


def test_clean_dataframe_missing_target_column():
    df = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError):
        clean_dataframe(df, "TyreLife")


def test_clean_dataframe_none_input():
    with pytest.raises(ValueError):
        clean_dataframe(None, "TyreLife")