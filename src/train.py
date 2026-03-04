"""
Educational Goal:
- Why this module exists in an MLOps system: Train the model in a controlled,
  reproducible way that prevents leakage (fit preprocessing on train only).
- Responsibility (separation of concerns): Training logic and estimator choice
  belong here, not mixed into main.py.
- Pipeline contract (inputs and outputs): Accepts train split + preprocessor and
  returns a fitted sklearn Pipeline artifact.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline


def _load_config(config_path: Path = Path("config.yaml")) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
    """
    Inputs:
    - X_train: training features (DataFrame)
    - y_train: training labels/target (Series)
    - preprocessor: ColumnTransformer (unfitted)
    - problem_type: "regression" or "classification"
    Outputs:
    - model: fitted sklearn Pipeline(preprocess + estimator)
    Why this contract matters for reliable ML delivery:
    - The Pipeline ensures preprocessing learned from training data is reused
      identically during evaluation and inference.
    """
    print("[train.train_model] Training model pipeline...")  # TODO: replace with logging later

    if problem_type not in {"regression", "classification"}:
        raise ValueError("problem_type must be either 'regression' or 'classification'")

    cfg = _load_config()
    training_cfg = cfg.get("training", {})
    random_state = training_cfg.get("random_state", 42)

    if problem_type == "regression":
        reg_cfg = training_cfg.get("regression", {})
        use_lasso = reg_cfg.get("use_lasso_feature_selection", True)

        lr_params = (reg_cfg.get("linear_regression") or {})
        estimator = LinearRegression(**lr_params)

        steps = [("preprocess", preprocessor)]

        if use_lasso:
            lasso_cfg = reg_cfg.get("lasso", {})
            selector = SelectFromModel(
                estimator=LassoCV(
                    cv=lasso_cfg.get("cv", 5),
                    n_alphas=lasso_cfg.get("n_alphas", 100),
                    max_iter=lasso_cfg.get("max_iter", 20000),
                    random_state=random_state,
                ),
                threshold=lasso_cfg.get("threshold", 0.0),
            )
            steps.append(("select", selector))

        steps.append(("model", estimator))
        model = Pipeline(steps=steps)

    else:
        clf_cfg = training_cfg.get("classification", {})
        logreg_params = (clf_cfg.get("logistic_regression") or {})
        estimator = LogisticRegression(**logreg_params)

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add optional debug prints for selected features count, if enabled in config.
    # Why: Helps verify Lasso feature selection is actually doing something.
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return model
