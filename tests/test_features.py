from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import get_feature_preprocessor


def test_get_feature_preprocessor_returns_columntransformer():
    pre = get_feature_preprocessor(
        quantile_bin_cols=["q1", "q2"],
        categorical_onehot_cols=["cat1"],
        numeric_passthrough_cols=["num1"],
        n_bins=3,
    )
    assert isinstance(pre, ColumnTransformer)


def test_get_feature_preprocessor_contains_expected_transformers():
    pre = get_feature_preprocessor(
        quantile_bin_cols=["q1"],
        categorical_onehot_cols=["cat1"],
        numeric_passthrough_cols=["num1"],
    )

    # pre.transformers is a list of 3-tuples: (name, transformer, columns)
    names = [t[0] for t in pre.transformers]
    assert "num" in names
    assert "cat" in names

    # Get transformers safely (avoid dict(...) because items are 3-tuples)
    cat_transformer = next(t[1] for t in pre.transformers if t[0] == "cat")
    num_transformer = next(t[1] for t in pre.transformers if t[0] == "num")

    assert isinstance(cat_transformer, OneHotEncoder)
    assert cat_transformer.handle_unknown == "ignore"

    assert isinstance(num_transformer, StandardScaler)