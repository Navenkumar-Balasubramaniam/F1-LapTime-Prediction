"""
API layer for serving predictions.

Educational Goal:
- Provide a clean separation between model logic and serving layer
- Reuse existing pipeline functions (no new ML logic here)
- Expose endpoints for health check and prediction
- Load the production model from the W&B model registry using the `prod` alias

Endpoints:
- GET / -> simple interactive UI
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
from fastapi.responses import HTMLResponse
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
    - prod_candidate_name: artifact name promoted to alias `prod`

    Expected artifact naming pattern from main.py:
    - <model_registry_name>-linear_regression
    - <model_registry_name>-random_forest

    One of those should be promoted to alias `prod` in W&B.
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

        loaded_model = joblib.load(model_file)
        logger.info("Successfully loaded production model from W&B.")
        return loaded_model

    except Exception as exc:
        raise RuntimeError(f"Failed to load production model from W&B artifact {artifact_ref}: {exc}") from exc


try:
    model = _load_model_from_wandb_prod()
except Exception as exc:
    logger.error("API startup failed while loading model: %s", exc)
    raise


# -----------------------------
# Simple homepage UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>F1 Lap Time Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 960px;
                margin: 30px auto;
                padding: 20px;
                line-height: 1.5;
                color: #222;
                background: #f8f9fb;
            }
            h1 {
                color: #c8102e;
                margin-bottom: 8px;
            }
            .subtitle {
                color: #555;
                margin-bottom: 24px;
            }
            .card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(220px, 1fr));
                gap: 12px 16px;
            }
            label {
                display: block;
                font-weight: bold;
                margin-bottom: 4px;
            }
            input, select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-sizing: border-box;
            }
            button {
                background: #c8102e;
                color: white;
                border: none;
                padding: 12px 18px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 16px;
            }
            button:hover {
                opacity: 0.92;
            }
            .links a {
                color: #c8102e;
                text-decoration: none;
                font-weight: bold;
                margin-right: 16px;
            }
            .result {
                margin-top: 16px;
                padding: 12px;
                background: #eef7ee;
                border: 1px solid #cde8cd;
                border-radius: 8px;
                font-weight: bold;
            }
            .error {
                margin-top: 16px;
                padding: 12px;
                background: #fff1f1;
                border: 1px solid #efc2c2;
                border-radius: 8px;
                color: #a40000;
                font-weight: bold;
            }
            code {
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <h1>F1 Lap Time Prediction API</h1>
        <p class="subtitle">
            Enter lap and race context features below to get a predicted lap time from the production model.
        </p>

        <div class="card links">
            <a href="/docs">Swagger Docs</a>
            <a href="/health">Health Check</a>
        </div>

        <div class="card">
            <h2>Prediction Form</h2>
            <div class="grid">
                <div><label>lap</label><input id="lap" type="number" value="12"></div>
                <div><label>grid</label><input id="grid" type="number" value="4"></div>
                <div><label>Stint</label><input id="Stint" type="number" value="2"></div>
                <div><label>TyreLife</label><input id="TyreLife" type="number" value="8"></div>
                <div><label>TrackTemp</label><input id="TrackTemp" type="number" step="any" value="32.5"></div>
                <div><label>Humidity</label><input id="Humidity" type="number" step="any" value="45"></div>
                <div><label>Pressure</label><input id="Pressure" type="number" step="any" value="1012"></div>
                <div><label>Rainfall</label><input id="Rainfall" type="number" step="any" value="0"></div>
                <div><label>WindSpeed</label><input id="WindSpeed" type="number" step="any" value="3.2"></div>
                <div><label>WindDirection</label><input id="WindDirection" type="number" step="any" value="180"></div>
                <div><label>round</label><input id="round" type="number" value="10"></div>
                <div><label>name</label><input id="name" type="text" value="Silverstone"></div>
                <div><label>constructorId</label><input id="constructorId" type="number" value="1"></div>
                <div><label>code</label><input id="code" type="text" value="HAM"></div>

                <div>
                    <label>Compound</label>
                    <select id="Compound">
                        <option value="SOFT">SOFT</option>
                        <option value="MEDIUM" selected>MEDIUM</option>
                        <option value="HARD">HARD</option>
                        <option value="INTERMEDIATE">INTERMEDIATE</option>
                        <option value="WET">WET</option>
                    </select>
                </div>

                <div>
                    <label>FreshTyre</label>
                    <select id="FreshTyre">
                        <option value="0" selected>0</option>
                        <option value="1">1</option>
                    </select>
                </div>
            </div>

            <button onclick="predictLapTime()">Predict</button>

            <div id="result"></div>
        </div>

        <div class="card">
            <h2>API Payload Shape</h2>
            <p>The UI submits the same JSON structure expected by <code>POST /predict</code>.</p>
        </div>

        <script>
            async function predictLapTime() {
                const payload = {
                    data: [
                        {
                            lap: Number(document.getElementById("lap").value),
                            grid: Number(document.getElementById("grid").value),
                            Stint: Number(document.getElementById("Stint").value),
                            TyreLife: Number(document.getElementById("TyreLife").value),
                            TrackTemp: Number(document.getElementById("TrackTemp").value),
                            Humidity: Number(document.getElementById("Humidity").value),
                            Pressure: Number(document.getElementById("Pressure").value),
                            Rainfall: Number(document.getElementById("Rainfall").value),
                            WindSpeed: Number(document.getElementById("WindSpeed").value),
                            WindDirection: Number(document.getElementById("WindDirection").value),
                            round: Number(document.getElementById("round").value),
                            name: document.getElementById("name").value,
                            constructorId: Number(document.getElementById("constructorId").value),
                            code: document.getElementById("code").value,
                            Compound: document.getElementById("Compound").value,
                            FreshTyre: Number(document.getElementById("FreshTyre").value)
                        }
                    ]
                };

                const resultBox = document.getElementById("result");
                resultBox.innerHTML = "";

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        resultBox.innerHTML = `<div class="error">Prediction failed: ${data.detail || "Unknown error"}</div>`;
                        return;
                    }

                    const prediction = data.predictions[0];
                    resultBox.innerHTML = `<div class="result">Predicted lap time: ${prediction.toFixed(2)} milliseconds</div>`;
                } catch (error) {
                    resultBox.innerHTML = `<div class="error">Request failed: ${error}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """


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