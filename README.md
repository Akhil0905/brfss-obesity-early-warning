# BRFSS Obesity Early Warning

**A structured public health analytics pipeline using CDC BRFSS obesity indicator data.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

This project builds a modular, interpretable machine learning pipeline to model and predict
**obesity prevalence** across U.S. states and demographic subgroups using the CDC/Data.gov
**Behavioral Risk Factor Surveillance System (BRFSS)** Nutrition, Physical Activity, and Obesity dataset.

The project answers:
- Which socioeconomic and demographic factors are most associated with high obesity prevalence?
- Can we identify high-risk groups or state contexts from historical BRFSS data?
- Can we build an **early-warning model** that predicts whether a state/demographic-group
  combination will cross into a high-risk obesity category in the following year?

### Modeling Tracks

| Track | Target | Models |
|-------|--------|--------|
| **Regression** | `Data_Value` — obesity prevalence % (continuous) | Linear, Ridge, Lasso, Random Forest, GBM, XGBoost |
| **Classification** | `high_risk_obesity` — above P75 threshold (binary) | Logistic Regression, Random Forest, GBM, XGBoost |
| **Early Warning** | `early_warning` — will cross into high-risk next year (binary) | Same as classification (if data permits) |

---

## Repository Structure

```
brfss-obesity-early-warning/
├── PROJECT_PLAN.md          # Full project design document
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .gitignore
│
├── configs/
│   └── config.yaml          # All tunable parameters
│
├── data/
│   ├── raw/                 # Place raw CSV here (see below)
│   ├── interim/             # Cleaned intermediate data (auto-generated)
│   └── processed/           # Final feature + target matrices (auto-generated)
│
├── models/
│   ├── regression/          # Serialized regression models (.joblib)
│   └── classification/      # Serialized classification models (.joblib)
│
├── notebooks/
│   ├── 01_phase1_reference.ipynb      # Phase 1 EDA reference
│   ├── 02_modeling_experiments.ipynb  # Interactive exploration
│   └── 03_final_analysis.ipynb        # Final structured analysis
│
├── reports/
│   ├── figures/             # PNG plots (auto-generated)
│   └── metrics/             # JSON metric files (auto-generated)
│
├── src/
│   ├── data/                # Data loading, validation, preprocessing
│   ├── features/            # Feature and target engineering
│   ├── models/              # Train, evaluate, explain
│   ├── pipelines/           # End-to-end pipeline runner
│   └── utils/               # Paths, logging, helpers
│
└── tests/                   # Unit tests
```

---

## Dataset Setup

### Download

The dataset is **not included** in this repository. Download it from Data.gov:

> **[Nutrition, Physical Activity, and Obesity — Behavioral Risk Factor Surveillance System](https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system)**

Click **CSV** to download the dataset file.

### Placement

Place the downloaded file in `data/raw/` **without renaming it**:

```
data/raw/Nutrition__Physical_Activity__and_Obesity__-_Behavioral_Risk_Factor_Surveillance_System.csv
```

> If the filename differs slightly, update `data.raw_filename` in `configs/config.yaml`.

---

## Setup

### Prerequisites

- Python 3.9 or higher
- `pip`

### Install dependencies

```bash
# From the project root
pip install -r requirements.txt
```

Optional (for XGBoost support):
```bash
pip install xgboost
```

Optional (for SHAP interpretability):
```bash
pip install shap
```

---

## Running the Pipeline

All commands are run from the **project root directory**.

### Full pipeline (both regression and classification)

```bash
python -m src.pipelines.run_pipeline
```

### Regression track only

```bash
python -m src.pipelines.run_pipeline --track regression
```

### Classification track only

```bash
python -m src.pipelines.run_pipeline --track classification
```

### Run unit tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All parameters are controlled in `configs/config.yaml`:

| Section | Key Settings |
|---------|-------------|
| `data` | Raw filename, question filter, columns to keep |
| `features` | Group key for panel structure, lag years, rolling window |
| `targets` | High-risk percentile threshold, early warning minimum pairs |
| `split` | Train/val/test year boundaries |
| `regression` | Model list, hyperparameters |
| `classification` | Target selection, model list, hyperparameters |
| `interpretability` | Top-N features, SHAP on/off, permutation importance |
| `output` | Save models/figures/metrics flags, DPI |

### Changing the train/test split

Edit `configs/config.yaml`:
```yaml
split:
  train_max_year: 2017    # train on years ≤ this
  val_min_year: 2018
  val_max_year: 2019
  test_min_year: 2020     # test on years ≥ this
```

---

## Outputs

After running the pipeline, outputs are written to:

| Location | Content |
|----|--|
| `data/interim/brfss_cleaned.csv` | Cleaned, encoded dataset |
| `data/processed/features.csv` | Model-ready feature matrix |
| `data/processed/targets.csv` | Regression + classification targets |
| `models/regression/*.joblib` | Serialized regression models |
| `models/classification/*.joblib` | Serialized classification models |
| `reports/metrics/regression_metrics.json` | MAE, RMSE, R² per model |
| `reports/metrics/classification_metrics.json` | Accuracy, F1, AUC per model |
| `reports/metrics/feature_importances_*.json` | Feature importance rankings |
| `reports/figures/` | All visualization plots |

---

## Split Strategy

This project uses a **temporal split** to prevent data leakage:

- **Train:** Years ≤ 2017
- **Validation:** Years 2018–2019
- **Test:** Years ≥ 2020

This is critical because the early warning target explicitly links year T to year T+1.
Using future years in training would constitute look-ahead bias.

---

## Notebooks

The notebooks in `notebooks/` are companion artifacts for exploration and narrative analysis:

| Notebook | Purpose |
|----------|---------|
| `01_phase1_reference.ipynb` | Brief reference to Phase 1 EDA findings |
| `02_modeling_experiments.ipynb` | Interactive model exploration, ablations |
| `03_final_analysis.ipynb` | Structured final analysis using `src/` pipeline outputs |

The core logic lives in `src/` — notebooks call into it rather than reimplementing it.

---

## Project Context

- **Course:** DSCI 521 (Drexel University)
- **Dataset:** CDC BRFSS Nutrition, Physical Activity, and Obesity Indicators
- **Phase 1:** Exploratory data analysis (state-level trends, income disparities)
- **Phase 2:** This repository — full modeling pipeline

---

## Limitations

- Data is **aggregated indicator-level**, not person-level microdata.
- Survey values are **self-reported estimates** with sampling uncertainty.
- Cross-year and cross-state comparability may be affected by methodology changes.
- All models estimate **group-level patterns** — individual-level inferences should not be drawn.

---

*Last updated: March 2026*
