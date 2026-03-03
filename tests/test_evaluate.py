import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor, DummyClassifier

from src.evaluate import evaluate_model


def test_regression_rmse():
    X_train = pd.DataFrame({"x": [0, 1, 2, 3]})
    y_train = pd.Series([1.0, 2.0, 3.0, 4.0])

    model = Pipeline([("reg", DummyRegressor(strategy="mean"))])
    model.fit(X_train, y_train)

    X_test = pd.DataFrame({"x": [5, 6]})
    y_test = pd.Series([2.5, 2.5])

    score = evaluate_model(model, X_test, y_test, "regression")

    assert isinstance(score, float)


def test_classification_f1():
    X_train = pd.DataFrame({"x": [0, 1, 2, 3]})
    y_train = pd.Series([0, 0, 1, 1])

    model = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    model.fit(X_train, y_train)

    X_test = pd.DataFrame({"x": [10, 11, 12, 13]})
    y_test = pd.Series([0, 0, 1, 1])

    score = evaluate_model(model, X_test, y_test, "classification")

    assert isinstance(score, float)


def test_invalid_problem_type():
    X = pd.DataFrame({"x": [0, 1]})
    y = pd.Series([0, 1])

    model = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    model.fit(X, y)

    with pytest.raises(ValueError):
        evaluate_model(model, X, y, "invalid")
    