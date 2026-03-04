# Predictive Modeling of F1 Lap Performance

**Author:** Group 7
**Course:** MLOps: Master in Business Analytics and Data Sciense
**Status:** Session 1 (Initialization)

---

## 1. Business Objective
This project develops a data-driven model to predict Formula 1 lap times using historical race performance, tyre strategy, and weather data.

The goal is to transform raw race data into actionable performance insights that can support strategic decision-making in competitive motorsport environments.

* **The Goal:** What business value does this model create?
  > The model creates business value by improving race strategy decisions, quantifying performance drivers, reducing uncertainty, and enabling data-driven optimization in a highly competitive, millisecond-sensitive environment.

* **The User:** Who consumes the output and how?
  > The primary consumers of the model are race strategists and performance engineers, who use predicted lap times to simulate race scenarios, optimize tyre and pit strategies, and benchmark performance. Senior management consumes aggregated insights to support strategic and resource allocation decisions.

---

## 2. Success Metrics
*How do we know if the project is successful?*
The project is successful if it delivers accurate, generalizable lap time predictions that meaningfully improve race strategy decisions while remaining interpretable, reliable, and scalable.

* **Business KPI (The "Why"):**
  > Reduce lap time forecast error to under 0.3 seconds and improve race strategy decision accuracy by at least 20% compared to historical benchmarks.

* **Technical Metric (The "How"):**
  > Root Mean Squared Error (RMSE) ≤ 0.30 seconds (300 ms) on the test season

* **Acceptance Criteria:**
  > The model must outperform a naïve baseline (e.g., moving average of last 3 laps or season average lap time) by at least 15% in RMSE on the test season.

---

## 3. The Data

* **Source:** 
1. Kaggle / Ergast F1 Dataset – Historical race results, lap times, driver and constructor information.
2. FastF1 API – Lap-level telemetry and weather data, including tyre compounds, tyre life, and track conditions..
* **Target Variable:** Lap Time (milliseconds)
* **Sensitive Info:** Are there emails, credit cards, or any PII (Personally Identifiable Information)?
  > *⚠️ **WARNING:** If the dataset contains sensitive data, it must NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.*

The dataset contains:

- Driver names / IDs
- Team names
- Race metadata

It does NOT contain:

- Emails
- Financial data
- Credit card information
- Personal contact details
- Any Personally Identifiable Information (PII)

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── yourbaseline.ipynb   # From previous work
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── features.py          # Feature engineering
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Automated tests
```

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python -m src.main`



