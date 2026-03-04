"""
Educational Goal:
- Why this module exists in an MLOps system: Inference should be identical
  across notebooks, batch jobs, and deployment services.
- Responsibility (separation of concerns): Keep prediction formatting separate
  from model training/evaluation.
- Pipeline contract (inputs and outputs): Takes a fitted model and feature data;
  returns a DataFrame with one column: "prediction", preserving index.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

import pandas as pd


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: fitted sklearn Pipeline
    - X_infer: features for inference (DataFrame)
    Outputs:
    - df_pred: DataFrame with ONE column "prediction" preserving the index
    Why this contract matters for reliable ML delivery:
    - Standard output format makes downstream consumption (reports/APIs) predictable.
    """
    print("[infer.run_inference] Running inference...")  # TODO: replace with logging later

    preds = model.predict(X_infer)
    df_pred = pd.DataFrame({"prediction": preds}, index=X_infer.index)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add post-processing (rounding, clipping, inverse transforms).
    # Why: Many business systems require bounded or formatted outputs.
    #
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_pred