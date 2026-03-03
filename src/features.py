"""
Educational Goal:
- Why this module exists in an MLOps system: Define a single, reusable feature
  transformation “recipe” so train/eval/infer all use identical preprocessing.
- Responsibility (separation of concerns): Feature engineering lives here,
  not scattered across training and evaluation code.
- Pipeline contract (inputs and outputs): Returns a scikit-learn ColumnTransformer
  that is NOT fit here (fitting happens on training data only in train.py).

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols: Optional list of columns (kept for professor API compatibility)
    - categorical_onehot_cols: Optional list of categorical columns to one-hot encode
    - numeric_passthrough_cols: Optional list of numeric columns to scale
    - n_bins: kept for professor API compatibility (unused in StandardScaler-only version)
    Outputs:
    - preprocessor: sklearn ColumnTransformer (unfitted)
    Why this contract matters for reliable ML delivery:
    - A reusable preprocessing recipe prevents train/serve skew and
      reduces leakage risk by fitting only on training data.
    """
    print("[features.get_feature_preprocessor] Building feature preprocessor...")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    # StandardScaler-only approach:
    # We'll treat BOTH numeric_passthrough_cols and quantile_bin_cols as numeric-to-scale,
    # because your notebook uses StandardScaler (not KBinsDiscretizer).
    numeric_to_scale = list(dict.fromkeys(numeric_passthrough_cols + quantile_bin_cols))

    # Required compatibility pattern for OneHotEncoder across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

    transformers = []

    if numeric_to_scale:
        transformers.append(("num", StandardScaler(), numeric_to_scale))

    if categorical_onehot_cols:
        transformers.append(("cat", ohe, categorical_onehot_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add/modify transformers (imputation, custom encoders, etc.)
    # Why: Feature engineering is highly dataset-dependent.
    #
    # Examples:
    # 1) Add SimpleImputer for missing numeric values
    # 2) Swap encoder to target encoding
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return preprocessor