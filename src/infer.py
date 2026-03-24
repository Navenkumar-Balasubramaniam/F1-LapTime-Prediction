"""
Educational Goal:
- Why this module exists in an MLOps system: Inference should be identical
  across notebooks, batch jobs, and deployment services.
- Responsibility (separation of concerns): Keep prediction formatting separate
  from model training/evaluation.
- Pipeline contract (inputs and outputs): Takes a fitted model and feature data;
  returns a DataFrame with one column: "prediction", preserving index.

This version logs the inference flow so debugging prediction issues is easier.
"""

from __future__ import annotations

import pandas as pd

from src.logging import get_logger

logger = get_logger(__name__)


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
    logger.info("Running inference on %s rows.", len(X_infer))

    preds = model.predict(X_infer)
    df_pred = pd.DataFrame({"prediction": preds}, index=X_infer.index)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Post-processing can be added here later if the business problem requires
    # rounded outputs, clipped values, inverse transforms, or label mapping.
    logger.info("Inference complete. Produced prediction dataframe with shape=%s", df_pred.shape)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_pred
