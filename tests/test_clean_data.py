import pandas as pd
import pytest

from src.clean_data import clean_dataframe

# Columns your clean_data.py explicitly drops (if present)
DROPPED_COLUMNS = [
    # redundant ID-like columns
    "Driver",
    "LapNumber",
    "time",
    "circuitId",
    "driverId",
    "statusId",
    "raceId",
    # leakage prevention
    "position_x",
    "status",
    # correlated
    "AirTemp",
]


def make_raw_df_for_cleaning() -> pd.DataFrame:
    """
    Build a synthetic dataframe that:
    - Contains all columns that should be dropped
    - Contains required columns used by the pipeline
    - Includes NaNs in target and TyreLife to test row dropping
    - Includes NaNs in object columns to test fill with 'UNKNOWN'
    """
    return pd.DataFrame(
        {
            # target (some invalid values)
            "milliseconds": ["1000", "2000", None, "bad_value", "4000"],
            # required numeric feature with NaN
            "TyreLife": [10, None, 30, 40, 50],
            # extra numeric features (no NaN here to allow "no NaN anywhere" assertion)
            "lap": [1, 2, 3, 4, 5],
            "grid": [5, 4, 3, 2, 1],
            "Stint": [1, 1, 2, 2, 3],
            "TrackTemp": [25.0, 26.0, 27.0, 28.0, 29.0],
            "Humidity": [40.0, 41.0, 42.0, 43.0, 44.0],
            "Pressure": [1010.0, 1011.0, 1012.0, 1013.0, 1014.0],
            "Rainfall": [0.0, 0.0, 1.0, 0.0, 0.0],
            "WindSpeed": [10.0, 11.0, 12.0, 13.0, 14.0],
            "WindDirection": [180.0, 190.0, 200.0, 210.0, 220.0],
            # categorical features (include NaNs to test 'UNKNOWN' fill)
            "round": [1, 1, 2, 2, 3],
            "name": [None, None, "Spa", "Spa", "Monaco"],
            "constructorId": ["mercedes", "redbull", None, "ferrari", "mclaren"],
            "code": ["HAM", "VER", "LEC", None, "NOR"],
            "Compound": ["Soft", "Medium", "Hard", None, "Soft"],
            "FreshTyre": ["True", "False", None, "True", "False"],
            # columns to be dropped (present)
            "Driver": ["x", "y", "z", "a", "b"],
            "LapNumber": [10, 11, 12, 13, 14],
            "time": ["00:01", "00:02", "00:03", "00:04", "00:05"],
            "circuitId": [100, 100, 200, 200, 300],
            "driverId": [1, 2, 3, 4, 5],
            "statusId": [0, 0, 1, 0, 0],
            "raceId": [999, 999, 888, 888, 777],
            "position_x": [1, 2, 3, 4, 5],
            "status": ["OK", "OK", "OK", "DNF", "OK"],
            "AirTemp": [20.0, 21.0, 22.0, 23.0, 24.0],
            # not dropped by your code (kept unless you drop it in config later)
            "position_y": [1, 2, 3, 4, 5],
            "year": [2022, 2022, 2023, 2023, 2024],
        }
    )


def test_clean_dataframe_extensive_contract_drops_columns_and_removes_nans():
    """
    This is the main "extensive" test:
    - Ensures required cleaning contract behaviors.
    - Ensures all specified columns are dropped.
    - Ensures bad target strings are coerced and removed.
    - Ensures no NaNs remain anywhere in the cleaned df (strong guarantee).
    """
    df_raw = make_raw_df_for_cleaning()

    cleaned = clean_dataframe(df_raw, target_column="milliseconds")

    # Basic contract
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.empty
    assert "milliseconds" in cleaned.columns

    # 1) Check all expected dropped columns are gone
    for col in DROPPED_COLUMNS:
        assert col not in cleaned.columns, f"Expected column '{col}' to be dropped."

    # 2) Check that target is numeric after coercion
    assert pd.api.types.is_numeric_dtype(cleaned["milliseconds"]), "Target should be numeric after cleaning."

    # 3) Ensure no missing values remain in required columns
    assert cleaned["milliseconds"].isna().sum() == 0, "No NaN allowed in target after cleaning."
    assert "TyreLife" in cleaned.columns
    assert cleaned["TyreLife"].isna().sum() == 0, "No NaN allowed in TyreLife after cleaning."

    # 4) Ensure object columns have no NaNs (they should be filled with 'UNKNOWN')
    obj_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        assert cleaned[obj_cols].isna().sum().sum() == 0, "Object columns should have no NaNs after fill."
        # Optional: check at least one UNKNOWN exists because we inserted some missing
        assert (cleaned[obj_cols] == "UNKNOWN").any().any(), "Expected at least one 'UNKNOWN' fill value."

    # 5) Strong guarantee: no NaNs anywhere in the cleaned dataframe
    # (safe because our numeric feature columns in the test data have no NaNs)
    assert cleaned.isna().sum().sum() == 0, "Cleaned dataframe should have no NaNs anywhere."

    # 6) Row-level expectation:
    # Raw rows:
    # - row0: milliseconds="1000" ok, TyreLife=10 ok => keep
    # - row1: TyreLife=None => drop
    # - row2: milliseconds=None => drop
    # - row3: milliseconds="bad_value" => coerces to NaN => drop
    # - row4: milliseconds="4000" ok, TyreLife=50 ok => keep
    # So at least rows 0 and 4 should survive (unless IQR trimming removes something).
    # Because IQR trimming only runs if >10 rows, and we have 5 rows, trimming won't happen.
    assert cleaned.shape[0] == 2


def test_clean_dataframe_missing_target_column_raises():
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        clean_dataframe(df, "milliseconds")


def test_clean_dataframe_none_input_raises():
    with pytest.raises(ValueError):
        clean_dataframe(None, "milliseconds")


def test_clean_dataframe_wrong_type_input_raises():
    with pytest.raises(TypeError):
        clean_dataframe(["not", "a", "dataframe"], "milliseconds")


def test_clean_dataframe_empty_df_raises():
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame(), "milliseconds")


def test_clean_dataframe_does_not_remove_target_column():
    """
    Safety test: even if drop lists change later, the target should never disappear.
    """
    df = pd.DataFrame({"milliseconds": [1000, 2000], "TyreLife": [10, 20]})
    cleaned = clean_dataframe(df, "milliseconds")
    assert "milliseconds" in cleaned.columns
