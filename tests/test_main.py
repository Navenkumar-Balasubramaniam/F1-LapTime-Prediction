"""
Integration test for src.main.main.

This test:
- runs the end-to-end pipeline via main.main()
- asserts required artifacts exist:
    - data/processed/clean.csv
    - models/model.joblib
    - reports/predictions.csv
- verifies that reports/predictions.csv is a non-empty CSV with a single column "prediction"
"""

from pathlib import Path

import pandas as pd
import yaml

import src.main as main_module

ARTIFACTS = {
    "processed": Path("data/processed/clean.csv"),
    "model": Path("models/model.joblib"),
    "predictions": Path("reports/predictions.csv"),
}


def load_config():
    cfg_path = Path("config.yaml")
    assert cfg_path.exists(), "config.yaml must exist at repo root for integration test"
    with cfg_path.open("r") as f:
        return yaml.safe_load(f) or {}


def test_main_creates_artifacts_and_predictions_csv():
    cfg = load_config()
    target_col = cfg["task"]["target_column"]

    # Preconditions: raw parquet must exist (otherwise integration test can't run)
    raw_path = Path(cfg["data"]["raw_path"])
    assert raw_path.exists(), f"Raw data parquet not found at {raw_path}. Cannot run integration test."

    # 1) Run pipeline
    main_module.main()

    # 2) Assert artifacts exist
    assert ARTIFACTS["processed"].exists(), f"Missing artifact: {ARTIFACTS['processed']}"
    assert ARTIFACTS["model"].exists(), f"Missing artifact: {ARTIFACTS['model']}"
    assert ARTIFACTS["predictions"].exists(), f"Missing artifact: {ARTIFACTS['predictions']}"

    # 3) Load and sanity-check processed CSV
    df_processed = pd.read_csv(ARTIFACTS["processed"])
    assert not df_processed.empty, "Processed CSV should not be empty"
    assert target_col in df_processed.columns, f"Processed data must contain target column '{target_col}'"

    # 4) Load and sanity-check predictions CSV
    df_preds = pd.read_csv(ARTIFACTS["predictions"])
    assert not df_preds.empty, "Predictions CSV should not be empty"
    assert list(df_preds.columns) == ["prediction"], "Predictions CSV must have exactly one column named 'prediction'"

    # 5) prediction count should equal test-set rows (since your main runs inference on X_test)
    # Since X_test is built from df_test (year == test_year), it should match df_preds length.
    # The processed CSV includes train+test rows, so df_preds should be <= df_processed
    assert len(df_preds) <= len(df_processed), "Predictions rows cannot exceed processed data rows"