"""
Unit tests for src.train.train_model

This test verifies:
- train_model can accept a DataFrame and unfitted preprocessor,
- returns a fitted sklearn Pipeline with a working predict method.

Run with:
    pytest -q tests/test_train.py
"""

import numpy as np
import pandas as pd

from src.features import get_feature_preprocessor
from src.train import train_model


def make_dummy_data(n=50):
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
    X, y_cls, _ = make_dummy_data(n=50)

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=[],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=["num_feature"],
        n_bins=3,
    )

    model = train_model(X_train=X, y_train=y_cls, preprocessor=preprocessor, problem_type="classification")

    assert hasattr(model, "predict")
    preds = model.predict(X)

    assert preds.shape == (len(X),)
    assert np.unique(preds).size >= 1


def test_train_model_fits_and_predicts_regression():
    X, _, y_reg = make_dummy_data(n=50)

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=[],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=["num_feature"],
        n_bins=3,
    )

    model = train_model(X_train=X, y_train=y_reg, preprocessor=preprocessor, problem_type="regression")

    preds = model.predict(X)

    assert preds.shape == (len(X),)
    assert preds.dtype.kind in ("f", "i")

    # Extra: verify Lasso selection step exists for regression
    assert "select" in model.named_steps
