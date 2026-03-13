# BRFSS Obesity Early Warning

**A structured public health analytics pipeline using CDC BRFSS obesity indicator data.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-31%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

This project builds a modular, interpretable machine learning pipeline to model and predict
**obesity prevalence** across U.S. states and demographic subgroups using the CDC/Data.gov
**Behavioral Risk Factor Surveillance System (BRFSS)** Nutrition, Physical Activity, and Obesity dataset.

**Course:** DSCI 521 — Data Analysis and Interpretation · Drexel University  
**Team:** The OGs — Akhil Tom · Sapan Parikh · Rishabh Gujarathi · Yugeshkanna Venkatesh

The project answers:
- Which socioeconomic and demographic factors are most associated with high obesity prevalence?
- Can we build an **early-warning model** that predicts whether a state/demographic-group combination will cross into a high-risk obesity category in the following year?

### Modeling Tracks

| Track | Target | Methodology | Best Model | Results |
|-------|--------|-------------|------------|---------|
| **Regression** | `Data_Value` (%) | Group K-Fold (Robustness) | XGBoost | R² = 0.715 (CV) |
| **Early Warning** | `early_warning` | Binary Classification | Random Forest | AUC = 0.860 |
| **Time Series** | `Data_Value` (next year) | LSTM (Temporal Sequencing)| Neural Network | MAE = 2.45 pp |

---

## Repository Structure

```
brfss-obesity-early-warning/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .gitignore
│
├── configs/
│   └── config.yaml                   # All tunable parameters (split years, models, etc.)
│
├── data/
│   ├── raw/                          # Raw BRFSS CSV (~40 MB, included)
│   ├── interim/                      # Cleaned intermediate data (auto-generated)
│   └── processed/                    # Feature + target matrices (auto-generated)
│
├── models/
│   ├── regression/                   # Serialized regression models (.joblib)
│   └── classification/               # Serialized classification models (.joblib)
│
├── notebooks/
│   ├── Project(Phase1)_The OGs_DSCI521.ipynb   # Phase 1 EDA notebook
│   └── Project(Phase2)_The OGs_DSCI521.ipynb   # Phase 2 final analysis notebook
│
├── reports/
│   ├── figures/                      # All generated PNG plots
│   └── metrics/                      # JSON metric files (regression + classification)
│
├── src/
│   ├── analysis/                     # statistical_impact.py (ANOVA/Tukey)
│   ├── data/                         # load_data, validate_data, preprocess
│   ├── features/                     # build_features, build_targets
│   ├── models/                       # train, evaluate, cross_validation, explain
│   ├── pipelines/                    # run_pipeline.py (entry point)
│   └── utils/                        # paths, helpers, logging
│
└── tests/                            # 31 unit tests (pytest)
    ├── test_preprocess.py
    └── test_features.py
```

---

## Dataset

The raw dataset is included in this repository under `data/raw/`.

**Source:** [CDC BRFSS — Nutrition, Physical Activity, and Obesity](https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system)

| Property | Value |
|---|---|
| Raw rows | 110,880 |
| After filtering | 19,078 (obesity prevalence question only) |
| Years | 2011–2024 (14 years) |
| States / territories | 55 |
| Strata | Total, Income, Age, Education, Sex, Race/Ethnicity |

---

## Setup

### Prerequisites

- Python 3.9+

### Install dependencies

```bash
pip install -r requirements.txt
```

Optional extras:
```bash
pip install xgboost   # for XGBoost models
pip install shap      # for SHAP interpretability
```

---

## Running the Pipeline

All commands from the **project root directory**.

```bash
# Full pipeline (regression + classification)
python -m src.pipelines.run_pipeline

# Regression only
python -m src.pipelines.run_pipeline --track regression

# Classification only
python -m src.pipelines.run_pipeline --track classification

# Run unit tests
python -m pytest tests/ -v
```

Pipeline completes in **~16 seconds** on a standard laptop.

---

## Configuration

All parameters are in `configs/config.yaml`:

| Section | Key Settings |
|---------|-------------|
| `data` | Raw filename, question filter, columns to keep |
| `features` | Group key, lag years, rolling window |
| `targets` | High-risk percentile threshold (P75 = 33.60%) |
| `split` | Train ≤ 2017 · Val 2018–2019 · Test ≥ 2020 |
| `regression` | Model list, hyperparameters |
| `classification` | Target column, model list, hyperparameters |
| `interpretability` | Top-N features, SHAP on/off, permutation importance |
| `output` | Save models/figures/metrics flags |

---

## Outputs

| Location | Content |
|---|---|
| `data/interim/brfss_cleaned.csv` | Cleaned, encoded dataset |
| `data/processed/features.csv` | Model-ready feature matrix (13 features) |
| `data/processed/targets.csv` | Regression + classification targets |
| `models/regression/*.joblib` | 6 serialized regression models |
| `models/classification/*.joblib` | 4 serialized classification models |
| `reports/metrics/regression_metrics.json` | MAE, RMSE, R² per model |
| `reports/metrics/classification_metrics.json` | Accuracy, Precision, Recall, F1, AUC per model |
| `reports/metrics/feature_importances_*.json` | Feature importance rankings |
| `reports/figures/` | 26 visualization plots |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `Project(Phase1)_The OGs_DSCI521.ipynb` | Phase 1 EDA — temporal trends, geographic & demographic patterns |
| `Project(Phase2)_The OGs_DSCI521.ipynb` | Phase 2 final analysis — model results, feature importances, conclusions |

---

| Model | Track | Metric | Test Score |
|-------|-------|--------|------------|
| XGBoost | Regression | Test R² | **0.729** |
| Group K-Fold | Robustness | Avg R² | **0.715** |
| Random Forest | Early Warning | Test AUC | **0.860** |
| LSTM | Time Series | Test MAE | **2.45 pp** |
| ANOVA | Correlation | Income/Edu | **p < 0.0001** |

**Top predictors (across all models):** `rolling_mean_3` · `lag_1` · `income_enc` · `education_enc`

---

## Split & Validation Strategy

1. **Strict Temporal Split** (to prevent data leakage):
   - **Train:** Years ≤ 2017 (9,499 rows)
   - **Validation:** Years 2018–2019 (2,712 rows)
   - **Test:** Years ≥ 2020 (6,867 rows)

2. **5-Fold GroupKFold Cross-Validation** (Grouped by State):
   - Used during the training phase to ensure geographic generalization.
   - Prevents "geographic leakage" by ensuring state-specific data is never shared across training/validation folds.
   - **Mean CV $R^2$ = 0.715** (XGBoost).

High-risk threshold (33.60%) is computed on **training data only** and applied to validation/test.

---

## Limitations

- Data is **aggregated group-level**, not individual person-level.
- Survey values are **self-reported estimates** with sampling uncertainty.
- 2020–2021 data affected by COVID-19 survey methodology changes.
- Group-level patterns cannot be used to infer individual behavior (ecological fallacy).

---

*DSCI 521 · The OGs · March 2026*
