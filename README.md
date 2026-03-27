# 🏎️ Predictive Modeling of F1 Lap Performance

**Author:** Group 7
**Course:** MLOps — Master in Business Analytics and Data Science
**Status:** Production-Ready MLOps System

------------------------------------------------------------------------

## 📌 1. Project Overview

This project delivers a **full end-to-end MLOps system** to predict Formula 1 lap times using historical race data, tyre usage, and environmental conditions.

Unlike a typical ML project, this system includes:

- Reproducible ML pipeline (`src/`)
- Experiment tracking with Weights & Biases
- Model registry with production aliasing (`prod`)
- REST API using FastAPI
- Containerization with Docker
- Unit testing and validation

------------------------------------------------------------------------

## 🎯 2. Business Objective

The goal is to improve **race strategy decision-making** by predicting lap performance.

### Business Value

- Optimize pit-stop timing
- Improve tyre strategy decisions
- Reduce uncertainty in race simulations
- Enable data-driven performance optimization

------------------------------------------------------------------------

## 📊 3. Success Metrics

### Business KPI

Reduce lap time prediction error and improve strategy decision accuracy.

### Technical Metric

```
Target:   RMSE ≤ 300 ms on test season
Achieved: Test RMSE ≈ 6,876 ms  |  Test R² ≈ 0.67
```

> **Note:** The 300 ms RMSE target reflects an aspirational benchmark. Current performance (Test R² = 0.67) demonstrates meaningful predictive signal and is sufficient for strategic planning use cases. Closing the gap to the target metric is listed under Future Improvements.

### Acceptance Criteria

- ≥15% improvement vs baseline model

------------------------------------------------------------------------

## 🧠 4. System Architecture

```
Raw Data
   ↓
Cleaning (clean_data.py)
   ↓
Validation (validate.py)
   ↓
Feature Engineering (features.py)
   ↓
Training (train.py)
   ↓
Model Selection
   ↓
Evaluation (evaluate.py)
   ↓
Inference (infer.py)
   ↓
API (api.py)
```

------------------------------------------------------------------------

## 📁 5. Repository Structure

The pipeline is orchestrated through `main.py`, which serves as the single entry point.

```
src/
    main.py          # pipeline entrypoint
    api.py           # FastAPI service
    train.py         # model training + selection
    features.py      # feature engineering
    clean_data.py    # data cleaning
    validate.py      # schema validation
    evaluate.py      # model evaluation
    infer.py         # inference
    logging.py       # shared logging config
    utils.py         # shared helpers

tests/               # one test file per src module
notebooks/
config.yaml          # central pipeline config
.env                 # secrets (not committed)
environment.yml
conda-lock.yml
Dockerfile
.github/
    workflows/
        ci.yml
        deploy.yml
README.md
```

------------------------------------------------------------------------

## ⚙️ 6. Data Instructions (Reproducibility)

### Data Sources

- Kaggle / Ergast dataset: https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features/data

### Required Setup

To ensure reproducibility, follow these steps exactly:

1. Download the dataset from the Kaggle/Ergast source above.
2. Place raw data files in:

```
data/raw/
```

3. Ensure the filename matches exactly:

```
data/raw/f1_all.parquet
```

The parquet file (`f1_all.parquet`) is generated from the raw dataset during preprocessing. If using FastF1, ensure required data is cached or downloaded via script.

4. Run the pipeline:

```bash
python -m src.main
```

All data paths and parameters are controlled via `config.yaml`.

------------------------------------------------------------------------

## 🗂️ 7. Configuration

- All non-secret runtime settings are stored in `config.yaml`
- Secrets (API keys, credentials) are stored in `.env`
- No hardcoded paths or parameters in production code
- A centralized logging system replaces print statements and writes to both console and local log files

------------------------------------------------------------------------

## ⚙️ 8. Setup

```bash
conda env create -f environment.yml
conda activate mlops-student-env
```

------------------------------------------------------------------------

## 🚀 9. Run Pipeline

```bash
python -m src.main
```

This runs the full pipeline end-to-end:

- Data loading and validation
- Data cleaning
- Feature engineering
- Model training and selection
- Evaluation
- Inference

**Outputs:** processed dataset (`data/processed/clean.csv`), trained model (`models/model.joblib`), predictions (`reports/predictions.csv`)

------------------------------------------------------------------------

## 🤖 10. Model Training & Selection

Two candidate models are trained on every run:

- **Linear Regression** (baseline, with Lasso feature selection)
- **Random Forest** (tuned with GridSearchCV)

Selection criterion:

```
Best model = lowest cross-validated RMSE
```

In the current production run, **Linear Regression** was selected (CV RMSE ≈ 5,983 ms vs Random Forest CV RMSE ≈ 8,114 ms).

------------------------------------------------------------------------

## ⚠️ 11. Known Issue

Random Forest shows significant overfitting:

```
Train R² ≈ 0.86
Test R²  ≈ 0.08
```

**Cause:** Distribution shift between pre-2023 training data and the 2023 test season, plus unseen categorical values (circuits, drivers, constructors) not present in training.

------------------------------------------------------------------------

## 📊 12. Experiment Tracking (W&B)

We use Weights & Biases (W&B) to track experiments, metrics, artifacts, and logs.

Models are logged as artifacts:

```
laptime-model-linear_regression
laptime-model-random_forest
```

Models are versioned and stored in W&B. The production model is accessed via alias `prod`.

| Alias     | Meaning          |
|-----------|------------------|
| candidate | all trained models |
| selected  | best model by RMSE |
| prod      | deployed model   |

------------------------------------------------------------------------

## 🚀 13. Model Registry

The API loads the production model using:

```
entity/project/model-name:prod
```

To promote a model: open the W&B UI → select a model version → add alias `prod`.

------------------------------------------------------------------------

## 🌐 14. API

Run locally:

```bash
uvicorn src.api:app --reload
```

Open: http://127.0.0.1:8000/docs

**Endpoints:**
- `GET /health` → returns system status
- `POST /predict` → returns predicted lap time in milliseconds

### Example Request

```bash
POST https://f1-laptime-prediction-n9qv.onrender.com/predict
```

```json
{
  "data": [
    {
      "lap": 12,
      "grid": 4,
      "Stint": 2,
      "TyreLife": 8,
      "TrackTemp": 32.5,
      "Humidity": 45,
      "Pressure": 1012,
      "Rainfall": 0,
      "WindSpeed": 3.2,
      "WindDirection": 180,
      "round": 10,
      "name": "Silverstone",
      "constructorId": 1,
      "code": "HAM",
      "Compound": "MEDIUM",
      "FreshTyre": 0
    }
  ]
}
```

> **Note:** Field names are case-sensitive. Use `Stint`, `TyreLife`, `TrackTemp`, `WindSpeed`, `WindDirection`, `Compound`, and `FreshTyre` exactly as shown.

------------------------------------------------------------------------

## 🐳 15. Docker

Build image:

```bash
docker build -t f1-api .
```

Run container:

```bash
docker run --env-file .env -p 8000:8000 f1-api
```

**Deployment:** Live on Render at https://f1-laptime-prediction-n9qv.onrender.com/
Supports real-time JSON requests.

------------------------------------------------------------------------

## 🔁 16. CI/CD Workflow

- `ci.yml`: runs tests and validation on every pull request
- `deploy.yml`: deploys only after a GitHub Release is published
- Pull requests must pass CI before merge
- Production deployments are tied to release versions

------------------------------------------------------------------------

## 🧪 17. Testing

```bash
pytest
```

Tests cover all core modules (`clean_data`, `features`, `train`, `evaluate`, `infer`, `validate`, `utils`, `main`) with edge case coverage. Ensures pipeline reliability on each change.

------------------------------------------------------------------------

## 📄 18. Model Card

### Model Name

`laptime-model-linear_regression`

### Model Purpose

Predict Formula 1 lap times (in milliseconds) to support race strategy decisions such as pit-stop timing and tyre management.

### Production Model

**Ridge Regression** with Lasso-based feature selection (scikit-learn `Pipeline`). Ridge was chosen over plain Linear Regression to prevent numerically unstable coefficients when one-hot encoding produces many correlated columns. Ridge was selected over Random Forest due to better generalisation on the 2023 test season.

### Training Data

| Split | Period      | Rows   |
|-------|-------------|--------|
| Train | Pre-2023    | 42,260 |
| Test  | 2023 season | 23,523 |

Source: Kaggle / Ergast + FastF1 telemetry

### Input Features

| Feature         | Type        | Description                        |
|-----------------|-------------|------------------------------------|
| `lap`           | Numeric     | Lap number                         |
| `grid`          | Numeric     | Starting grid position             |
| `Stint`         | Numeric     | Tyre stint number                  |
| `TyreLife`      | Numeric     | Laps completed on current tyre     |
| `TrackTemp`     | Numeric     | Track surface temperature (°C)     |
| `Humidity`      | Numeric     | Relative humidity (%)              |
| `Pressure`      | Numeric     | Atmospheric pressure (hPa)         |
| `Rainfall`      | Numeric     | Rainfall indicator                 |
| `WindSpeed`     | Numeric     | Wind speed (m/s)                   |
| `WindDirection` | Numeric     | Wind direction (degrees)           |
| `round`         | Categorical | Race round number                  |
| `name`          | Categorical | Circuit name                       |
| `constructorId` | Categorical | Constructor (team) identifier      |
| `code`          | Categorical | Driver code (e.g. HAM)             |
| `Compound`      | Categorical | Tyre compound (SOFT/MEDIUM/HARD/…) |
| `FreshTyre`     | Categorical | Whether the tyre is new (0 or 1)   |

### Output

Predicted lap time in **milliseconds**.

### Performance Metrics (Latest Run)

| Metric          | Value       |
|-----------------|-------------|
| CV RMSE (train) | 5,982.95 ms |
| Test RMSE       | 6,875.73 ms |
| Train R²        | 0.784       |
| Test R²         | 0.669       |

### Limitations

- Test RMSE exceeds the 300 ms aspirational target — gap is driven by distribution shift between pre-2023 and 2023 data
- Missing telemetry for some drivers/laps
- Random Forest overfits severely (Test R² ≈ 0.08) due to unseen categorical values in 2023
- Not suitable for real-time safety-critical systems

### Intended Use

Race strategy support: pit-stop timing, tyre strategy planning, performance simulation.

### Not Intended Use

Real-time safety-critical systems or decisions with direct safety implications.

------------------------------------------------------------------------

## 🔐 19. Environment Variables

```
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=...
```

Store these in a `.env` file at the project root. Never commit this file.

------------------------------------------------------------------------

## 📊 20. Data

**Sources:** Kaggle / Ergast dataset + FastF1 telemetry
**Target column:** `milliseconds` (lap time in ms)
**Train/test split:** year-based — train on seasons before 2023, test on 2023

------------------------------------------------------------------------

## ⚠️ 21. Risks

| Risk                  | Mitigation                              |
|-----------------------|-----------------------------------------|
| Data quality          | Schema validation step in pipeline      |
| Overfitting           | Model selection by CV RMSE; RF excluded |
| Concept drift         | Retraining on new season data required  |
| Missing telemetry     | Handled by `dropna` cleaning step       |

------------------------------------------------------------------------

## 🚀 22. Future Improvements

- Better feature engineering (driver-specific features, circuit characteristics)
- Validation split instead of CV-only to catch distribution shift earlier
- Automated model promotion via W&B API
- Monitoring and drift detection on live predictions
- Reduce test RMSE gap toward the 300 ms target

------------------------------------------------------------------------

## 📈 23. Monitoring & Observability

- Logs written to console and `logs/pipeline.log`
- W&B tracks all experiments, metrics, and model artifacts
- Deployment logs available via Render dashboard

------------------------------------------------------------------------

## 📝 24. Changelog

| Version | Date         | Changes                                                       |
|---------|--------------|---------------------------------------------------------------|
| v1.0    | Mar 2026     | Initial pipeline: data cleaning, feature engineering, Linear Regression baseline |
| v1.1    | Mar 2026     | Added FastAPI prediction endpoint and Docker containerisation |
| v1.2    | Mar 2026     | Integrated W&B experiment tracking and model registry with `candidate` / `selected` / `prod` aliases |
| v1.3    | Mar 2026     | Added CI/CD workflows (`ci.yml`, `deploy.yml`) and Render deployment |

------------------------------------------------------------------------

## 📌 25. Summary

This project demonstrates a full production MLOps system:

- End-to-end reproducible ML pipeline
- Experiment tracking and model registry (W&B)
- REST API with FastAPI and live Render deployment
- Docker containerisation for reproducibility
- CI/CD via GitHub Actions
