# 🏎️ Predictive Modeling of F1 Lap Performance

**Author:** Group 7\
**Course:** MLOps -- Master in Business Analytics and Data Science\
**Status:** Production-Ready MLOps System

------------------------------------------------------------------------

# 📌 1. Project Overview

This project delivers a **full end-to-end MLOps system** to predict
Formula 1 lap times using historical race data, tyre usage, and
environmental conditions.

Unlike a typical ML project, this system includes:

-   Reproducible ML pipeline (`src/`)
-   Experiment tracking with Weights & Biases
-   Model registry with production aliasing (`prod`)
-   REST API using FastAPI
-   Containerization with Docker
-   Unit testing and validation

------------------------------------------------------------------------

# 🎯 2. Business Objective

The goal is to improve **race strategy decision-making** by predicting
lap performance.

### Business Value

-   Optimize pit-stop timing
-   Improve tyre strategy decisions
-   Reduce uncertainty in race simulations
-   Enable data-driven performance optimization

------------------------------------------------------------------------

# 📊 3. Success Metrics

### Business KPI

Reduce lap time prediction error and improve strategy decision accuracy

### Technical Metric

``` text
RMSE ≤ 300 ms on test season
```

### Acceptance Criteria

-   ≥15% improvement vs baseline model

------------------------------------------------------------------------

# 🧠 4. System Architecture

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

------------------------------------------------------------------------

# 📁 5. Repository Structure

    src/
      main.py          # pipeline entrypoint
      api.py           # FastAPI service
      train.py         # model training + selection
      features.py
      clean_data.py
      validate.py
      evaluate.py
      infer.py

    tests/
    notebooks/
    config.yaml
    environment.yml
    Dockerfile

------------------------------------------------------------------------

# ⚙️ 6. Setup

``` bash
conda env create -f environment.yml
conda activate mlops-student-env
```

------------------------------------------------------------------------

# 🚀 7. Run Pipeline

``` bash
python -m src.main
```

Outputs: - processed dataset - trained model - predictions

------------------------------------------------------------------------

# 🤖 8. Model Training & Selection

Models: - Linear Regression (baseline) - Random Forest (grid search)

Selection:

``` text
Best model = lowest RMSE
```

------------------------------------------------------------------------

# ⚠️ 9. Known Issue

Random Forest shows overfitting:

    Train R² ≈ 0.86
    Test R² ≈ 0.08

Cause: - distribution shift (pre-2023 vs 2023) - unseen categorical
values

------------------------------------------------------------------------

# 📊 10. Experiment Tracking (W&B)

Models are logged as artifacts:

    laptime-model-linear_regression
    laptime-model-random_forest

Aliases:

  Alias       Meaning
  ----------- ----------------
  candidate   all models
  selected    best model
  prod        deployed model

------------------------------------------------------------------------

# 🚀 11. Model Registry

API loads:

    entity/project/model-name:prod

To promote model: - open W&B UI - select model version - add alias
`prod`

------------------------------------------------------------------------

# 🌐 12. API

Run locally:

``` bash
uvicorn src.api:app --reload
```

Open:

    http://127.0.0.1:8000/docs

Endpoints: - GET /health - POST /predict

------------------------------------------------------------------------

# 🐳 13. Docker

``` bash
docker build -t f1-api .
docker run --env-file .env -p 8000:8000 f1-api
```

------------------------------------------------------------------------

# 🧪 14. Testing

``` bash
pytest
```

------------------------------------------------------------------------

# 🔐 15. Environment Variables

    WANDB_API_KEY=...
    WANDB_ENTITY=...
    WANDB_PROJECT=...

------------------------------------------------------------------------

# 📊 16. Data

Sources: - Kaggle / Ergast dataset - FastF1 telemetry

Target: - Lap time (milliseconds)

------------------------------------------------------------------------

# ⚠️ 17. Risks

-   Data quality → handled by validation
-   Overfitting → model tuning needed
-   Concept drift → retraining required

------------------------------------------------------------------------

# 🚀 18. Future Improvements

-   Better feature engineering
-   Validation split instead of CV-only
-   Automated model promotion
-   Monitoring & drift detection

------------------------------------------------------------------------

# 📌 19. Summary

This project demonstrates:

-   Full ML pipeline
-   Experiment tracking
-   Model registry
-   API deployment
-   Docker reproducibility
