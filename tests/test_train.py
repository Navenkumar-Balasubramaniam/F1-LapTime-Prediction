"""Unit tests for ``src.train.train_model``.

These tests were updated to support the new regression training flow:
- classification still trains a single logistic-regression pipeline
- regression now compares Linear Regression vs Random Forest and returns the
  best fitted pipeline based on cross-validated RMSE
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import get_feature_preprocessor
from src.train import train_model


def make_dummy_data(n: int = 50):
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        {
            "num_feature": np.linspace(0, 10, n),
            "cat_feature": rng.choice(["a", "b", "c"], size=n),
        }
    )

    y_cls = pd.Series((X["num_feature"] > 5).astype(int), name="target")
    y_reg = pd.Series(X["num_feature"] * 0.5 + rng.normal(0, 0.1, size=n), name="target")

    return X, y_cls, y_reg


def test_train_model_fits_and_predicts_classification():
    X, y_cls, _ = make_dummy_data(n=20)

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=[],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=["num_feature"],
        n_bins=3,
    )

    model = train_model(X_train=X, y_train=y_cls, preprocessor=preprocessor, problem_type="classification")

    assert hasattr(model, "predict")
    assert "preprocess" in model.named_steps
    assert "model" in model.named_steps

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.unique(preds).size >= 1


def test_train_model_fits_and_predicts_regression_with_best_pipeline_selection():
    X, _, y_reg = make_dummy_data(n=20)

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=[],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=["num_feature"],
        n_bins=3,
    )

    model = train_model(X_train=X, y_train=y_reg, preprocessor=preprocessor, problem_type="regression")

    assert hasattr(model, "predict")
    assert "preprocess" in model.named_steps
    assert "model" in model.named_steps

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert preds.dtype.kind in ("f", "i")

    # The selected estimator can now be either LinearRegression or
    # RandomForestRegressor. The test therefore checks behavior, not one exact
    # internal implementation detail.
    model_name = model.named_steps["model"].__class__.__name__
    assert model_name in {"LinearRegression", "RandomForestRegressor"}
