"""
Educational Goal:
- Why this module exists in an MLOps system: Define a single, reusable feature
  transformation “recipe” so train/eval/infer all use identical preprocessing.
- Responsibility (separation of concerns): Feature engineering lives here,
  not scattered across training and evaluation code.
- Pipeline contract (inputs and outputs): Returns a scikit-learn ColumnTransformer
  that is NOT fit here (fitting happens on training data only in train.py).

This version logs what it builds so readers can understand the preprocessing
contract without needing to step through the code in a debugger.
"""

from __future__ import annotations

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logging import get_logger

logger = get_logger(__name__)


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
    logger.info("Building feature preprocessor.")

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    # StandardScaler-only approach:
    # We'll treat BOTH numeric_passthrough_cols and quantile_bin_cols as numeric-to-scale,
    # because the current notebook uses StandardScaler (not KBinsDiscretizer).
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
    # This project keeps the preprocessing intentionally simple so the flow is
    # easy to read:
    # - numeric features are standardized
    # - categorical features are one-hot encoded
    #
    # More advanced steps such as imputation or custom encoders can be added later.
    logger.info(
        "Preprocessor ready. Numeric columns=%s | Categorical columns=%s | n_bins placeholder=%s",
        numeric_to_scale,
        categorical_onehot_cols,
        n_bins,
    )
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return preprocessor
