"""
API layer for serving predictions.

Educational Goal:
- Provide a clean separation between model logic and serving layer
- Reuse existing pipeline functions (no new ML logic here)
- Expose endpoints for health check and prediction

Endpoints:
- GET /health → service status
- POST /predict → returns predictions for input records
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.clean_data import clean_dataframe
from src.infer import run_inference
from src.logging import get_logger
from src.main import load_config
from src.validate import validate_dataframe

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

model_path = Path(cfg.get("artifacts", {}).get("model_path", "models/model.joblib"))

if not model_path.exists():
    raise RuntimeError(f"Model not found at {model_path}. Run training first.")

logger.info("Loading model from %s", model_path)
model = joblib.load(model_path)


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
        task_cfg = cfg.get("task", {})
        feat_cfg = cfg.get("features", {})

        target_column = task_cfg.get("target_column")

        numeric_cols = feat_cfg.get("numeric_passthrough", [])
        categorical_cols = feat_cfg.get("categorical_onehot", [])

        # Clean data (target may not exist at inference, so skip if missing)
        if target_column in df.columns:
            df = clean_dataframe(df, target_column=target_column)

        # Validate required columns
        required_columns = list(numeric_cols) + list(categorical_cols)
        validate_dataframe(df, required_columns=required_columns)

        # Run inference
        preds_df = run_inference(model, df)

        return PredictionResponse(predictions=preds_df["prediction"].tolist())

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))