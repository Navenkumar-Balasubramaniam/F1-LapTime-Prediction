"""
Lightweight integration test for src.main.main.

This test verifies pipeline orchestration and artifact creation
without running expensive real model training.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

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


def test_main_creates_artifacts_and_predictions_csv(monkeypatch):
    cfg = load_config()
    target_col = cfg["task"]["target_column"]

    raw_path = Path(cfg["data"]["raw_path"])
    assert raw_path.exists(), f"Raw data parquet not found at {raw_path}. Cannot run integration test."

    # Replace expensive training with a tiny dummy regressor pipeline.
    def fake_train_model(X_train, y_train, preprocessor, problem_type, run=None):
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", DummyRegressor(strategy="mean")),
            ]
        )
        model.fit(X_train, y_train)
        return model

    monkeypatch.setattr(main_module, "train_model", fake_train_model)

    # Run pipeline
    main_module.main()

    # Assert artifacts exist
    assert ARTIFACTS["processed"].exists(), f"Missing artifact: {ARTIFACTS['processed']}"
    assert ARTIFACTS["model"].exists(), f"Missing artifact: {ARTIFACTS['model']}"
    assert ARTIFACTS["predictions"].exists(), f"Missing artifact: {ARTIFACTS['predictions']}"

    # Processed CSV checks
    df_processed = pd.read_csv(ARTIFACTS["processed"])
    assert not df_processed.empty, "Processed CSV should not be empty"
    assert target_col in df_processed.columns, f"Processed data must contain target column '{target_col}'"

    # Predictions CSV checks
    df_preds = pd.read_csv(ARTIFACTS["predictions"])
    assert not df_preds.empty, "Predictions CSV should not be empty"
    assert list(df_preds.columns) == ["prediction"], "Predictions CSV must have exactly one column named 'prediction'"

    assert len(df_preds) <= len(df_processed), "Predictions rows cannot exceed processed data rows"