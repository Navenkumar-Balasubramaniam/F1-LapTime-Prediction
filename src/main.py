"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single command entry point
  (python -m src.main) that runs the pipeline end-to-end reliably.
- Responsibility (separation of concerns): Orchestrates steps; does not contain
  the detailed logic of cleaning/features/training/evaluation.
- Pipeline contract (inputs and outputs): Produces pipeline artifacts and logs.

This version adds:
- standard logging instead of print statements
- optional Weights & Biases experiment tracking using .env credentials
- model selection for regression handled in train.py
- candidate model artifact logging to W&B so one can later be promoted to `prod`
"""

from __future__ import annotations

import logging as pylogging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import yaml
from dotenv import load_dotenv
from sklearn.metrics import r2_score

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.logging import configure_logging, get_logger
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None

logger = get_logger(__name__)


def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """
    Load YAML configuration from disk.

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    ValueError
        If config.yaml is empty or invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict) or not cfg:
        raise ValueError("config.yaml is empty or invalid (expected a YAML mapping).")

    return cfg


def _init_wandb_for_pipeline(cfg: dict):
    """
    Initialize a single W&B pipeline run.

    This run is owned by main.py and is the primary run used for:
    - pipeline-level metrics
    - pipeline artifacts
    - candidate model artifact logging

    Returns
    -------
    wandb.sdk.wandb_run.Run | None
        Active W&B run if enabled and initialized successfully, else None.
    """
    load_dotenv()
    wandb_cfg = cfg.get("wandb", {}) or {}

    if not wandb_cfg.get("enabled", False):
        return None

    if wandb is None:
        logger.warning("wandb package is not installed. Skipping pipeline W&B run.")
        return None

    api_key = os.getenv(wandb_cfg.get("api_key_env", "WANDB_API_KEY"))
    entity = os.getenv(wandb_cfg.get("entity_env", "WANDB_ENTITY"))
    project = wandb_cfg.get("project", "mlops-project")

    if not api_key or not entity:
        logger.warning("Missing W&B credentials in .env. Skipping pipeline W&B run.")
        return None

    os.environ.setdefault("WANDB_API_KEY", api_key)
    os.environ.setdefault("WANDB_ENTITY", entity)

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            job_type="pipeline",
            config=cfg,
            reinit="finish_previous",
            settings=wandb.Settings(
                console="off",
                silent=True,
            ),
        )
        logger.info("Started pipeline W&B run.")
        return run
    except Exception as exc:  # pragma: no cover - service/network dependent
        logger.warning("Failed to initialize pipeline W&B run. Continuing without W&B. Error: %s", exc)
        return None


def _log_artifacts_to_wandb(run, model_path: Path, processed_path: Path, predictions_path: Path) -> None:
    """
    Log core pipeline output artifacts to W&B.

    Parameters
    ----------
    run
        Active W&B run or None.
    model_path : Path
        Local selected-model path.
    processed_path : Path
        Processed CSV output path.
    predictions_path : Path
        Predictions CSV output path.
    """
    if run is None or wandb is None:
        return

    artifact = wandb.Artifact("pipeline_outputs", type="pipeline-output")
    for path in [model_path, processed_path, predictions_path]:
        if Path(path).exists():
            artifact.add_file(str(path))

    run.log_artifact(artifact)
    logger.info("Logged pipeline output artifact to W&B.")


def _log_model_candidates_to_wandb(run, cfg: dict, candidate_models: dict, selected_name: str) -> None:
    """
    Log each candidate model as a separate W&B model artifact.

    Aliases:
    - all candidate models get: 'candidate'
    - the selected model also gets: 'selected'

    Later, one of these artifacts can be promoted manually (or via script)
    to alias 'prod' for production inference.

    Parameters
    ----------
    run
        Active W&B run or None.
    cfg : dict
        Full project configuration.
    candidate_models : dict
        Mapping of model name -> fitted model object.
    selected_name : str
        Name of the chosen model.
    """
    if run is None or wandb is None:
        return

    wandb_cfg = cfg.get("wandb", {}) or {}
    registry_prefix = wandb_cfg.get("model_registry_name", "laptime-model")

    for model_name, model_obj in candidate_models.items():
        artifact_name = f"{registry_prefix}-{model_name}"

        with TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "model.joblib"
            joblib.dump(model_obj, model_file)

            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(str(model_file))

            aliases = ["candidate"]
            if model_name == selected_name:
                aliases.append("selected")

            run.log_artifact(artifact, aliases=aliases)
            logger.info("Logged candidate model artifact '%s' with aliases=%s", artifact_name, aliases)


def main() -> None:
    """
    Inputs:
    - None (reads config.yaml)

    Outputs:
    - None (creates artifacts on disk)

    Why this contract matters for reliable ML delivery:
    - A single script entry point makes runs reproducible and CI-friendly.
    """
    cfg = load_config()

    # Configure logging as early as possible so all subsequent steps are captured.
    logging_cfg = cfg.get("logging", {}) or {}
    configure_logging(
        log_file=logging_cfg.get("file_path", "logs/pipeline.log"),
        level=logging_cfg.get("level", "INFO"),
    )

    logger.info("Starting end-to-end pipeline.")

    # Ensure required directories exist (idempotent)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # Config sections
    data_cfg = cfg.get("data", {}) or {}
    art_cfg = cfg.get("artifacts", {}) or {}
    task_cfg = cfg.get("task", {}) or {}
    feat_cfg = cfg.get("features", {}) or {}

    # Required config keys (fail fast)
    raw_path = Path(data_cfg.get("raw_path", ""))
    processed_path = Path(data_cfg.get("processed_path", ""))
    model_path = Path(art_cfg.get("model_path", ""))
    predictions_path = Path(art_cfg.get("predictions_path", ""))

    problem_type = task_cfg.get("problem_type")
    target_column = task_cfg.get("target_column")
    year_column = task_cfg.get("year_column")
    test_year = task_cfg.get("test_year")

    if not raw_path:
        raise ValueError("config.yaml missing: data.raw_path")
    if not processed_path:
        raise ValueError("config.yaml missing: data.processed_path")
    if not model_path:
        raise ValueError("config.yaml missing: artifacts.model_path")
    if not predictions_path:
        raise ValueError("config.yaml missing: artifacts.predictions_path")

    if problem_type not in {"regression", "classification"}:
        raise ValueError("config.yaml task.problem_type must be 'regression' or 'classification'")
    if not target_column:
        raise ValueError("config.yaml missing: task.target_column")
    if not year_column:
        raise ValueError("config.yaml missing: task.year_column (year split is required)")
    if test_year is None:
        raise ValueError("config.yaml missing: task.test_year (year split is required)")

    numeric_cols = feat_cfg.get("numeric_passthrough", [])
    categorical_cols = feat_cfg.get("categorical_onehot", [])
    quantile_bin_cols = feat_cfg.get("quantile_bin", [])
    n_bins = feat_cfg.get("n_bins", 3)

    pipeline_run = _init_wandb_for_pipeline(cfg)

    try:
        # 1) Load raw data
        df_raw = load_raw_data(raw_path)

        # 2) Clean data
        df_clean = clean_dataframe(df_raw, target_column=target_column)

        # 3) Save processed CSV artifact
        save_csv(df_clean, processed_path)

        # 4) Validate schema
        required_columns = [target_column] + list(numeric_cols) + list(categorical_cols)
        validate_dataframe(df_clean, required_columns=required_columns)

        # 5) Year-based split (REQUIRED)
        if year_column not in df_clean.columns:
            raise ValueError(
                f"Year split required, but year column '{year_column}' not found in cleaned dataframe."
            )

        logger.info("Using year split: train < %s, test == %s", test_year, test_year)
        df_train = df_clean[df_clean[year_column] < test_year].copy()
        df_test = df_clean[df_clean[year_column] == test_year].copy()

        if df_train.empty:
            raise ValueError(
                f"Year split produced empty TRAIN set. No rows where {year_column} < {test_year}."
            )
        if df_test.empty:
            raise ValueError(
                f"Year split produced empty TEST set. No rows where {year_column} == {test_year}."
            )

        X_train = df_train.drop(columns=[target_column])
        y_train = df_train[target_column]
        X_test = df_test.drop(columns=[target_column])
        y_test = df_test[target_column]

        # 6) Fail-fast feature presence checks
        configured_cols = list(numeric_cols) + list(categorical_cols)
        missing_feats = [c for c in configured_cols if c not in X_train.columns]
        if missing_feats:
            raise ValueError(f"Configured feature columns missing from training data: {missing_feats}")

        # 7) Build preprocessing recipe (unfitted)
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=quantile_bin_cols,
            categorical_onehot_cols=categorical_cols,
            numeric_passthrough_cols=numeric_cols,
            n_bins=n_bins,
        )

        # 8) Train model and get candidate metadata
        train_result = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            problem_type=problem_type,
            run=pipeline_run,
        )

        model = train_result["selected_model"]
        selected_name = train_result["selected_name"]
        selected_score = train_result["selected_score"]
        candidate_models = train_result["candidate_models"]
        candidate_metrics = train_result["candidate_metrics"]

        logger.info("Training returned selected model '%s'.", selected_name)

        # 9) Save selected model artifact locally
        save_model(model, model_path)

        # 10) Evaluate on test set
        score = evaluate_model(model, X_test=X_test, y_test=y_test, problem_type=problem_type)
        logger.info("Final score returned (single float)=%.4f", score)

        # 11) Inference example: run on X_test
        df_pred = run_inference(model, X_infer=X_test)

        # 12) Save predictions artifact
        save_csv(df_pred, predictions_path)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        if problem_type == "regression":
            train_r2 = float(r2_score(y_train, train_preds))
            test_r2 = float(r2_score(y_test, test_preds))
            logger.info("Train R2=%.4f | Test R2=%.4f", train_r2, test_r2)

            if pipeline_run is not None:
                pipeline_run.log(
                    {
                        "selected_model_name": selected_name,
                        "selected_model_cv_rmse": selected_score,
                        "test_rmse": score,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                    }
                )

                # Also log candidate metrics so they are visible on the run page.
                for model_name, metrics_dict in candidate_metrics.items():
                    payload = {}
                    for key, value in metrics_dict.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                payload[f"{model_name}/{sub_key}"] = sub_value
                        else:
                            payload[f"{model_name}/{key}"] = value
                    if payload:
                        pipeline_run.log(payload)

        else:
            if pipeline_run is not None:
                pipeline_run.log(
                    {
                        "selected_model_name": selected_name,
                        "test_score": score,
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                    }
                )

        # 13) Log core pipeline artifacts
        _log_artifacts_to_wandb(
            pipeline_run,
            model_path=model_path,
            processed_path=processed_path,
            predictions_path=predictions_path,
        )

        # 14) Log both candidate models as separate W&B model artifacts
        _log_model_candidates_to_wandb(
            pipeline_run,
            cfg=cfg,
            candidate_models=candidate_models,
            selected_name=selected_name,
        )

        logger.info("Pipeline completed successfully.")

    except Exception:
        logger.exception("Pipeline failed.")
        if pipeline_run is not None:
            try:
                pipeline_run.alert(title="Pipeline failed", text="Check local logs for details.")
            except Exception:
                logger.warning("Could not send W&B alert for pipeline failure.")
        raise

    finally:
        if pipeline_run is not None:
            pipeline_run.finish()
        pylogging.shutdown()


if __name__ == "__main__":
    main()