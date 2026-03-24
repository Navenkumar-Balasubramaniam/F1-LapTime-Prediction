"""
API layer for serving predictions.

Educational Goal:
- Provide a clean separation between model logic and serving layer
- Reuse existing pipeline functions (no new ML logic here)
- Expose endpoints for health check and prediction
- Load the production model from the W&B model registry using the `prod` alias

Endpoints:
- GET /health -> service status
- POST /predict -> returns predictions for input records
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.infer import run_inference
from src.logging import get_logger
from src.main import load_config
from src.validate import validate_dataframe

try:
    import wandb
except ImportError:  # pragma: no cover - optional runtime dependency
    wandb = None

logger = get_logger(__name__)

app = FastAPI(title="F1 Lap Time Prediction API")


# -----------------------------
# Pydantic schemas
# -----------------------------
class PredictionRequest(BaseModel):
    data: List[dict]


class PredictionResponse(BaseModel):
    predictions: List[float]


# -----------------------------
# Load config + model on startup
# -----------------------------
cfg = load_config()


def _load_model_from_wandb_prod():
    """
    Load the production model artifact from Weights & Biases.

    Expected config keys under `wandb`:
    - enabled: true/false
    - entity_env: env var name for WANDB entity
    - api_key_env: env var name for WANDB API key
    - project: W&B project name
    - model_registry_name: prefix used when logging candidate model artifacts

    Expected artifact naming pattern from main.py:
    - <model_registry_name>-linear_regression
    - <model_registry_name>-random_forest

    One of those should later be promoted to alias `prod` in W&B.
    """
    load_dotenv()

    wandb_cfg = cfg.get("wandb", {}) or {}
    if not wandb_cfg.get("enabled", False):
        raise RuntimeError("W&B is disabled in config. Cannot load production model artifact.")

    if wandb is None:
        raise RuntimeError("wandb is not installed. Cannot load production model artifact.")

    api_key = os.getenv(wandb_cfg.get("api_key_env", "WANDB_API_KEY"))
    entity = os.getenv(wandb_cfg.get("entity_env", "WANDB_ENTITY"))
    project = wandb_cfg.get("project", "mlops-project")

    # This should match the artifact you promoted to alias `prod`
    prod_model_name = wandb_cfg.get("prod_candidate_name", "laptime-model-random_forest")

    if not api_key or not entity:
        raise RuntimeError("Missing W&B credentials in .env for API model loading.")

    os.environ.setdefault("WANDB_API_KEY", api_key)
    os.environ.setdefault("WANDB_ENTITY", entity)

    artifact_ref = f"{entity}/{project}/{prod_model_name}:prod"
    logger.info("Loading production model from W&B artifact %s", artifact_ref)

    try:
        api = wandb.Api()
        artifact = api.artifact(artifact_ref, type="model")
        artifact_dir = Path(artifact.download())
        model_file = artifact_dir / "model.joblib"

        if not model_file.exists():
            raise RuntimeError(f"Downloaded W&B artifact does not contain model.joblib: {artifact_dir}")

        model = joblib.load(model_file)
        logger.info("Successfully loaded production model from W&B.")
        return model

    except Exception as exc:
        raise RuntimeError(f"Failed to load production model from W&B artifact {artifact_ref}: {exc}") from exc


try:
    model = _load_model_from_wandb_prod()
except Exception as exc:
    logger.error("API startup failed while loading model: %s", exc)
    raise


# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(request.data)

        if df.empty:
            raise ValueError("Input data is empty.")

        # Get config fields
        feat_cfg = cfg.get("features", {}) or {}
        numeric_cols = feat_cfg.get("numeric_passthrough", [])
        categorical_cols = feat_cfg.get("categorical_onehot", [])

        # Validate required columns for inference
        required_columns = list(numeric_cols) + list(categorical_cols)
        validate_dataframe(df, required_columns=required_columns)

        # Run inference using the loaded production model
        preds_df = run_inference(model, df)

        return PredictionResponse(predictions=preds_df["prediction"].tolist())

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))