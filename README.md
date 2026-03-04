# Predictive Modeling of F1 Lap Performance

**Author:** Group 7  
**Course:** MLOps: Master in Business Analytics and Data Sciense  
**Status:** Session 1 (Initialization)

---

## 1. Business Objective

This project develops a data-driven model to predict Formula 1 lap times using historical race performance, tyre strategy, and weather data.

The goal is to transform raw race data into actionable performance insights that can support strategic decision-making in competitive motorsport environments.

**The Goal:** What business value does this model create?

> The model creates business value by improving race strategy decisions, quantifying performance drivers, reducing uncertainty, and enabling data-driven optimization in a highly competitive, millisecond-sensitive environment.

**The User:** Who consumes the output and how?

> The primary consumers of the model are race strategists and performance engineers, who use predicted lap times to simulate race scenarios, optimize tyre and pit strategies, and benchmark performance. Senior management consumes aggregated insights to support strategic and resource allocation decisions.

---

## 2. Success Metrics

How do we know if the project is successful?

The project is successful if it delivers accurate, generalizable lap time predictions that meaningfully improve race strategy decisions while remaining interpretable, reliable, and scalable.

### Business KPI (The "Why")

> Reduce lap time forecast error to under 0.3 seconds and improve race strategy decision accuracy by at least 20% compared to historical benchmarks.

### Technical Metric (The "How")

> Root Mean Squared Error (RMSE) ≤ 0.30 seconds (300 ms) on the test season

### Acceptance Criteria

> The model must outperform a naïve baseline (e.g., moving average of last 3 laps or season average lap time) by at least 15% in RMSE on the test season.

---

## 3. Client Maturity Assessment

Formula 1 teams operate in an environment where large volumes of telemetry, race performance data, and engineering information are constantly generated. However, the level of analytical maturity varies between teams.

Many organizations already collect and analyze race data extensively, yet some strategic decisions still rely heavily on expert judgment and manual analysis rather than predictive systems.

This project assumes a **medium to high level of analytical maturity**, where:

- Historical performance data is available and structured  
- Teams already employ analytics for monitoring performance  
- Predictive modeling capabilities are not yet fully integrated into strategic decision-making processes  

The goal is therefore to extend existing analytical practices into **predictive, decision-support tools** that can enhance race strategy planning.

---

## 4. Explicit Business Pain Baseline

Race strategy decisions in Formula 1 must be made under significant uncertainty and time pressure.

Teams face several operational challenges:

- Difficulty predicting lap performance under varying track and weather conditions
- Uncertainty surrounding tyre degradation and optimal stint lengths
- Limited ability to simulate multiple strategy scenarios in real time
- Heavy reliance on manual forecasting and engineer experience

These limitations can lead to:

- Suboptimal pit-stop timing
- Inefficient tyre strategies
- Missed competitive opportunities during races

Because Formula 1 performance margins are extremely small, even minor prediction errors (for example **0.3–0.5 seconds per lap**) can have a substantial impact on final race outcomes.

---

## 5. Executive-Level Solution Description

This project introduces a machine learning pipeline designed to forecast Formula 1 lap times using historical performance data, tyre usage information, and environmental conditions.

The system will:

1. Ingest historical race and telemetry data  
2. Perform automated cleaning and validation of raw datasets  
3. Engineer predictive features capturing tyre degradation, driver performance, and track conditions  
4. Train machine learning models capable of predicting lap performance  
5. Generate interpretable evaluation metrics and performance insights  

The resulting model can be integrated into race strategy workflows, enabling teams to simulate race scenarios and evaluate strategic options more effectively.

---

## 6. Tangible Benefits

The deployment of a predictive lap time model offers several tangible benefits to race teams.

### Improved Strategy Decisions

More accurate lap time predictions allow strategists to optimize:

- Pit-stop timing
- Tyre compound selection
- Stint length planning

### Competitive Advantage

Improved prediction accuracy helps teams anticipate opportunities such as:

- Undercut and overcut strategies
- Relative pace compared to competitors

### Faster Strategic Analysis

Automated predictions allow race engineers to evaluate multiple race scenarios quickly, improving responsiveness during race events.

### Knowledge Retention

Machine learning models capture and formalize insights that might otherwise remain implicit in individual engineers' experience, strengthening organizational learning.

---

## 7. Scalability

The project architecture is designed to support scalability across multiple seasons and datasets.

The modular pipeline structure allows:

- Integration of additional race seasons and telemetry datasets
- Extension to related predictive problems such as tyre degradation modeling or pit-stop optimization
- Adaptation to real-time or near real-time race simulation environments

Because the project separates **data ingestion, feature engineering, modeling, and inference**, it can be integrated into larger data infrastructure without major restructuring.

---

## 8. Risk and Mitigation

### Data Quality Risk

Telemetry and race datasets may contain missing values or inconsistencies.

**Mitigation:**  
Automated validation procedures implemented in `validate.py` ensure data integrity before model training.

### Model Overfitting

Models may perform well on historical races but fail to generalize to new seasons.

**Mitigation:**  
Use proper train-test splits across seasons and cross-validation techniques.

### Concept Drift

Race regulations, vehicle technologies, and track conditions evolve over time.

**Mitigation:**  
Regular model retraining and monitoring of performance metrics across seasons.

### Operational Integration Risk

Even accurate models may fail to generate business value if they are not integrated into race strategy tools.

**Mitigation:**  
Ensure model outputs are interpretable and compatible with existing race strategy workflows.

---

## 9. Cost Estimation

The cost of implementing this predictive modeling system is relatively modest compared to the strategic value it can deliver.

### Development Costs

- Data engineering and pipeline development
- Model training and validation
- Experimentation and testing

### Operational Costs

- Compute resources for model training and inference
- Storage of historical race data
- Ongoing monitoring and maintenance

### Estimated Cost Profile

| Cost Component | Estimated Level |
|----------------|----------------|
| Development | Medium |
| Infrastructure | Low–Medium |
| Maintenance | Low |

Given the high financial stakes associated with competitive performance in Formula 1, even small improvements in strategic decision accuracy can generate returns far exceeding the operational costs of maintaining this system.

---

## 10. The Data

**Source:**

1. Kaggle / Ergast F1 Dataset – Historical race results, lap times, driver and constructor information.  
2. FastF1 API – Lap-level telemetry and weather data, including tyre compounds, tyre life, and track conditions.

**Target Variable:** Lap Time (milliseconds)

**Sensitive Info:** Are there emails, credit cards, or any PII (Personally Identifiable Information)?

⚠️ **WARNING:** If the dataset contains sensitive data, it must NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.

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

## 11. Repository Structure

This project follows a strict separation between **Sandbox (Notebooks)** and **Production (Src)**.
```
.
├── README.md ## Project overview, business bjective, success metrics
├── LICENSE # License information
├── .gitignore # Git ignore rules
├── environment.yml # Conda environment definition (mlops-student-env)
├── config.yaml # Global configuration (paths, params, settings)
├── pytest.ini # Pytest configuration
├── notebooks/ # Exploration / prototyping notebooks
│ └── ML_Group7_Final_MLOPS.ipynb
├── src/ # Production pipeline code (reusable modules)
│ ├── init.py
│ ├── main.py # Orchestrates the full pipeline (entry point)
│ ├── load_data.py # Data ingestion / loading
│ ├── clean_data.py # Data cleaning / preprocessing
│ ├── validate.py # Data validation checks
│ ├── features.py # Feature engineering
│ ├── train.py # Model training
│ ├── evaluate.py # Model evaluation + metrics
│ ├── infer.py # Inference / prediction generation
│ └── utils.py # Shared helper functions
├── tests/ # Unit tests for src/ modules
│ ├── init.py
│ ├── test_load_data.py
│ ├── test_clean_data.py
│ ├── test_validate.py
│ ├── test_features.py
│ ├── test_train.py
│ ├── test_evaluate.py
│ ├── test_infer.py
│ ├── test_utils.py
│ └── test_main.py
├── reports/ # Outputs and generated artifacts for reporting
│ ├── .gitkeep
│ └── predictions.csv
└── .pytest_cache/ # Local pytest cache (can be ignored in git)
```
### Notes
- `src/` contains the reproducible pipeline (the code you can run end-to-end).
- `notebooks/` is for exploration; important logic should be migrated into `src/`.
- `reports/` is where prediction outputs and evaluation artifacts should live.

## 12. Execution Model

1. Clone the Repository
```
git clone <your-repo-url>
cd 1-mlops-kickoff-repo
```

2. Create and Activate the Conda Environment
```
conda env create -f environment.yml
conda activate mlops-student-env
```

3. Run the Full Pipeline

From the project root:
```
python -m src.main
```

This will execute the full workflow:

    1. Data loading and validation
    2. Data cleaning and feature engineering
    3. Model training and evaluation
    4. Prediction generation
Outputs (e.g., predictions.csv) will be saved in the reports/ directory.

4. Run Unit Tests
```
pytest
```



