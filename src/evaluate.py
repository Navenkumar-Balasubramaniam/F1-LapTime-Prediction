"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.

Educational Goal:
- Provide a consistent evaluation step for models before deployment.
- Metrics should live in one place.
- Return a single float score for CI.

This version logs useful evaluation details instead of printing them.
"""

from __future__ import annotations

import math
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from src.logging import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Evaluate a fitted model and return a single score used by the pipeline.

    For regression, the returned score is RMSE and lower is better.
    For classification, the returned score is weighted F1 and higher is better.
    """
    logger.info("Evaluating model for problem_type=%s on %s rows.", problem_type, len(X_test))

    preds = model.predict(X_test)
    problem_type = problem_type.lower().strip()

    if problem_type == "regression":
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        logger.info("Regression metrics | RMSE=%.4f | MAE=%.4f | R2=%.4f", rmse, mae, r2)
        return float(rmse)

    if problem_type == "classification":
        score = f1_score(y_test, preds, average="weighted")
        logger.info("Classification metric | Weighted F1=%.4f", score)
        return float(score)

    logger.error("Invalid problem_type=%s. Expected regression or classification.", problem_type)
    raise ValueError("problem_type must be either 'regression' or 'classification'")
