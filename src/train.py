"""
Educational Goal:
- Why this module exists in an MLOps system: Train the model in a controlled,
  reproducible way that prevents leakage (fit preprocessing on train only).
- Responsibility (separation of concerns): Training logic and estimator choice
  belong here, not mixed into main.py.
- Pipeline contract (inputs and outputs): Accepts train split + preprocessor and
  returns fitted sklearn Pipeline artifacts plus model-selection metadata.

This version adds:
- Linear Regression as a baseline model
- Random Forest Regressor with GridSearchCV
- Simple model selection using the lowest RMSE
- Optional Weights & Biases sweep support for Random Forest
- Candidate model tracking so main.py can log both models as artifacts
"""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.logging import get_logger

logger = get_logger(__name__)

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None


def _load_config(config_path: Path = Path("config.yaml")) -> dict:
    """
    Load project configuration from YAML.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary, or an empty dictionary if the file
        does not exist.
    """
    if not config_path.exists():
        logger.warning("Config file not found at %s. Using empty training config.", config_path)
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_regression_pipeline(preprocessor, reg_cfg: dict, random_state: int, estimator) -> Pipeline:
    """
    Build a regression pipeline.

    A dedicated helper keeps the pipeline assembly logic in one place so both
    the baseline linear model and the random forest model reuse the same
    preprocessing contract.

    Parameters
    ----------
    preprocessor
        Unfitted sklearn-compatible preprocessing object.
    reg_cfg : dict
        Regression configuration section.
    random_state : int
        Random seed for reproducibility.
    estimator
        Final regression estimator to place at the end of the pipeline.

    Returns
    -------
    Pipeline
        Unfitted sklearn pipeline.
    """
    use_lasso = reg_cfg.get("use_lasso_feature_selection", True)
    steps = [("preprocess", deepcopy(preprocessor))]

    # Lasso-based feature selection is useful for linear models but not ideal for
    # tree models. We therefore enable it only for estimators that are linear-ish.
    if use_lasso and isinstance(estimator, LinearRegression):
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
    return Pipeline(steps=steps)


def _rmse(y_true: pd.Series, y_pred) -> float:
    """
    Compute RMSE from true and predicted values.

    Parameters
    ----------
    y_true : pd.Series
        True target values.
    y_pred
        Predicted target values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _maybe_init_wandb(cfg: dict, job_type: str, extra_config: Optional[dict] = None):
    """
    Initialize Weights & Biases if enabled and available.

    The function is intentionally defensive:
    - if wandb is not installed, the pipeline still runs
    - credentials are read from .env through python-dotenv
    - missing credentials produce a warning instead of crashing the run

    Parameters
    ----------
    cfg : dict
        Full project configuration.
    job_type : str
        W&B job type label.
    extra_config : Optional[dict]
        Extra metadata to include in the W&B run config.

    Returns
    -------
    wandb.sdk.wandb_run.Run | None
        Active run if initialization succeeds, else None.
    """
    load_dotenv()

    wandb_cfg = cfg.get("wandb", {}) or {}
    if not wandb_cfg.get("enabled", False):
        logger.info("W&B disabled in config. Continuing without experiment tracking.")
        return None

    if wandb is None:
        logger.warning("wandb package is not installed. Skipping W&B logging.")
        return None

    api_key = os.getenv(wandb_cfg.get("api_key_env", "WANDB_API_KEY"))
    entity = os.getenv(wandb_cfg.get("entity_env", "WANDB_ENTITY"))
    project = wandb_cfg.get("project", "mlops-project")

    if not api_key or not entity:
        logger.warning("W&B credentials are missing in .env. Expected API key and entity variables.")
        return None

    os.environ.setdefault("WANDB_API_KEY", api_key)
    os.environ.setdefault("WANDB_ENTITY", entity)

    run_config = {
        "job_type": job_type,
        "project_config": cfg,
    }
    if extra_config:
        run_config.update(extra_config)

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            job_type=job_type,
            config=run_config,
            reinit="finish_previous",
            settings=wandb.Settings(
                console="off",
                silent=True,
            ),
        )
        logger.info("Initialized W&B run | project=%s | entity=%s | job_type=%s", project, entity, job_type)
        return run
    except Exception as exc:  # pragma: no cover - network/service dependent
        logger.warning("Failed to initialize W&B. Continuing without it. Error: %s", exc)
        return None


def _log_model_summary_to_wandb(
    run,
    model_name: str,
    rmse: float,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log model-level summary metrics to W&B.

    Parameters
    ----------
    run
        Active W&B run or None.
    model_name : str
        Human-readable model name.
    rmse : float
        RMSE value for the model.
    params : Optional[Dict[str, Any]]
        Parameter dictionary to log alongside metrics.
    """
    if run is None or wandb is None:
        return

    payload = {f"{model_name}/rmse": rmse}
    if params:
        payload.update({f"{model_name}/{k}": v for k, v in params.items()})

    run.log(payload)


def run_random_forest_sweep(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> None:
    """
    Launch a simple W&B sweep for Random Forest hyperparameters.

    This helper is separate from the normal training path because sweeps usually
    run many experiments and are operationally different from a single pipeline run.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    preprocessor
        Unfitted preprocessing object shared with the main pipeline.
    """
    cfg = _load_config()
    if wandb is None:
        raise ImportError("wandb is required to run the sweep.")

    load_dotenv()
    sweep_cfg = (cfg.get("wandb", {}) or {}).get("sweep", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    reg_cfg = training_cfg.get("regression", {}) or {}
    random_state = training_cfg.get("random_state", 42)

    run = _maybe_init_wandb(cfg, job_type="sweep")
    if run is not None:
        run.finish()

    parameters = sweep_cfg.get("parameters") or {
        "n_estimators": {"values": [50]},
        "max_depth": {"values": [10]},
        "min_samples_split": {"values": [2]},
        "min_samples_leaf": {"values": [1]},
    }

    sweep_config = {
        "method": sweep_cfg.get("method", "grid"),
        "metric": {"name": "rmse", "goal": "minimize"},
        "parameters": parameters,
    }

    project = (cfg.get("wandb", {}) or {}).get("project", "mlops-project")
    entity = os.getenv((cfg.get("wandb", {}) or {}).get("entity_env", "WANDB_ENTITY"))
    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    logger.info("Created W&B sweep with id=%s", sweep_id)

    def _sweep_train():
        run = _maybe_init_wandb(cfg, job_type="rf_sweep_trial")
        if run is None:
            return

        rf = RandomForestRegressor(
            random_state=random_state,
            n_estimators=wandb.config.n_estimators,
            max_depth=wandb.config.max_depth,
            min_samples_split=wandb.config.min_samples_split,
            min_samples_leaf=wandb.config.min_samples_leaf,
            n_jobs=reg_cfg.get("random_forest", {}).get("n_jobs", 1),
        )
        pipeline = _build_regression_pipeline(preprocessor, reg_cfg, random_state, rf)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_train)
        rmse = _rmse(y_train, preds)
        run.log({"rmse": rmse})
        run.finish()

    wandb.agent(sweep_id, function=_sweep_train, count=sweep_cfg.get("count", 8))


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    problem_type: str,
    run=None,
) -> Dict[str, Any]:
    """
    Train candidate models and return the selected model plus metadata.

    Inputs:
    - X_train: training features (DataFrame)
    - y_train: training labels/target (Series)
    - preprocessor: ColumnTransformer (unfitted)
    - problem_type: "regression" or "classification"
    - run: optional active W&B run provided by main.py

    Outputs:
    - dict containing:
        - selected_model: fitted sklearn Pipeline
        - selected_name: selected model name
        - selected_score: selection score
        - candidate_models: dict of fitted candidate models
        - candidate_metrics: dict of candidate metrics/params

    Why this contract matters for reliable ML delivery:
    - The Pipeline ensures preprocessing learned from training data is reused
      identically during evaluation and inference.
    - Returning all fitted candidate models allows main.py to log both artifacts
      to W&B and later promote one to production using aliases such as `prod`.

    For regression we train two candidate models:
    1) Linear Regression baseline
    2) Random Forest tuned with GridSearchCV

    The candidate with the lowest cross-validated RMSE is selected as the final model.
    """
    logger.info("Training model pipeline for problem_type=%s", problem_type)

    if problem_type not in {"regression", "classification"}:
        logger.error("Invalid problem_type=%s", problem_type)
        raise ValueError("problem_type must be either 'regression' or 'classification'")

    cfg = _load_config()
    training_cfg = cfg.get("training", {}) or {}
    random_state = training_cfg.get("random_state", 42)

    if run is None:
        run = _maybe_init_wandb(cfg, job_type="train")
        created_local_run = True
    else:
        created_local_run = False

    candidate_models: Dict[str, Any] = {}
    candidate_metrics: Dict[str, Any] = {}

    if problem_type == "regression":
        reg_cfg = training_cfg.get("regression", {}) or {}
        cv_folds = reg_cfg.get("cv_folds", 2)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # ---------------------------------------------------------
        # Candidate 1: Linear Regression baseline
        # ---------------------------------------------------------
        lr_params = reg_cfg.get("linear_regression") or {}
        linear_estimator = LinearRegression(**lr_params)
        linear_pipeline = _build_regression_pipeline(
            preprocessor=preprocessor,
            reg_cfg=reg_cfg,
            random_state=random_state,
            estimator=linear_estimator,
        )
        linear_pipeline.fit(X_train, y_train)
        linear_rmse = cross_val_score_neg_rmse(linear_pipeline, X_train, y_train, cv)

        logger.info("Linear Regression CV RMSE=%.4f", linear_rmse)
        _log_model_summary_to_wandb(run, "linear_regression", linear_rmse, lr_params)

        candidate_models["linear_regression"] = linear_pipeline
        candidate_metrics["linear_regression"] = {
            "cv_rmse": linear_rmse,
            "params": lr_params,
        }

        # ---------------------------------------------------------
        # Candidate 2: Random Forest with grid search
        # ---------------------------------------------------------
        rf_cfg = reg_cfg.get("random_forest") or {}
        rf_enabled = rf_cfg.get("enabled", True)

        if rf_enabled:
            rf_param_grid = rf_cfg.get("param_grid") or {
                "model__n_estimators": [50],
                "model__max_depth": [10],
                "model__min_samples_split": [2],
                "model__min_samples_leaf": [1],
            }

            rf_estimator = RandomForestRegressor(
                random_state=random_state,
                n_jobs=rf_cfg.get("n_jobs", 1),
            )
            rf_pipeline = _build_regression_pipeline(
                preprocessor=preprocessor,
                reg_cfg=reg_cfg,
                random_state=random_state,
                estimator=rf_estimator,
            )
            rf_search = GridSearchCV(
                estimator=rf_pipeline,
                param_grid=rf_param_grid,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=rf_cfg.get("grid_search_n_jobs", 1),
                refit=True,
                verbose=1,
            )
            rf_search.fit(X_train, y_train)

            random_forest_rmse = float(-rf_search.best_score_)
            logger.info(
                "Random Forest best CV RMSE=%.4f with params=%s",
                random_forest_rmse,
                rf_search.best_params_,
            )
            _log_model_summary_to_wandb(
                run,
                "random_forest",
                random_forest_rmse,
                rf_search.best_params_,
            )

            candidate_models["random_forest"] = rf_search.best_estimator_
            candidate_metrics["random_forest"] = {
                "cv_rmse": random_forest_rmse,
                "params": rf_search.best_params_,
            }
        else:
            random_forest_rmse = float("inf")
            rf_search = None
            logger.info("Random Forest disabled in config. Skipping grid search.")

        # ---------------------------------------------------------
        # Final model selection based on lower RMSE
        # ---------------------------------------------------------
        if rf_enabled and rf_search is not None and random_forest_rmse < linear_rmse:
            selected_model = rf_search.best_estimator_
            selected_name = "random_forest"
            selected_score = random_forest_rmse
        else:
            selected_model = linear_pipeline
            selected_name = "linear_regression"
            selected_score = linear_rmse

        logger.info(
            "Selected final regression model=%s with CV RMSE=%.4f",
            selected_name,
            selected_score,
        )

        if run is not None and wandb is not None:
            run.summary["selected_model"] = selected_name
            run.summary["selected_model_cv_rmse"] = selected_score

    else:
        # ---------------------------------------------------------
        # Classification path
        # ---------------------------------------------------------
        clf_cfg = training_cfg.get("classification", {}) or {}
        logreg_params = clf_cfg.get("logistic_regression") or {}
        estimator = LogisticRegression(**logreg_params)

        selected_model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        selected_model.fit(X_train, y_train)

        selected_name = "logistic_regression"
        selected_score = None

        candidate_models["logistic_regression"] = selected_model
        candidate_metrics["logistic_regression"] = {
            "params": logreg_params,
        }

        logger.info("Classification pipeline trained with LogisticRegression.")

        if run is not None and wandb is not None:
            run.summary["selected_model"] = selected_name

    if created_local_run and run is not None:
        run.finish()

    return {
        "selected_model": selected_model,
        "selected_name": selected_name,
        "selected_score": selected_score,
        "candidate_models": candidate_models,
        "candidate_metrics": candidate_metrics,
    }


def cross_val_score_neg_rmse(model, X_train: pd.DataFrame, y_train: pd.Series, cv) -> float:
    """
    Compute the mean cross-validated RMSE for a model.

    The helper is kept explicit instead of importing cross_val_score directly into
    the training loop so the purpose of the score is easier to read.

    Parameters
    ----------
    model
        sklearn-compatible estimator or pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    cv
        Cross-validation splitter.

    Returns
    -------
    float
        Positive RMSE value averaged across folds.
    """
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
    )
    return float(-scores.mean())