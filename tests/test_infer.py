import pandas as pd
from sklearn.linear_model import LinearRegression

from src.infer import run_inference


def test_run_inference_returns_dataframe_with_prediction_column():
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]}, index=[10, 11, 12])
    y = pd.Series([2.0, 4.0, 6.0], index=X.index)

    model = LinearRegression().fit(X, y)

    out = run_inference(model, X)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["prediction"]
    assert out.index.equals(X.index)
    assert len(out) == len(X)