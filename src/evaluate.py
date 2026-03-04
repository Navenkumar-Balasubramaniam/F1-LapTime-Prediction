"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

"""""
Educational Goal:
- Provide a consistent evaluation step for models before deployment.
- Metrics should live in one place.
- Return a single float score for CI.

TODO: Replace print with logging later.
"""

import math
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    print("[evaluate.evaluate_model] Evaluating model...")

    preds = model.predict(X_test)

    problem_type = problem_type.lower().strip()

    if problem_type == "regression":
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")
        return float(rmse)

    if problem_type == "classification":
        score = f1_score(y_test, preds, average="weighted")
        print(f"Weighted F1: {score:.4f}")
        return float(score)

    raise ValueError("problem_type must be either 'regression' or 'classification'")