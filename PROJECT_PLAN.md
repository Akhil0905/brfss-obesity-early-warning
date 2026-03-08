# BRFSS Obesity Early Warning — Project Plan

## 1. Project Objective

This project builds a **structured, interpretable public health analytics pipeline**
using the CDC/Data.gov BRFSS (*Behavioral Risk Factor Surveillance System*) Nutrition,
Physical Activity, and Obesity dataset.

The central question is whether historical BRFSS state-level obesity indicators —
stratified by demographics and socioeconomic characteristics — can be used to:

1. **Model obesity prevalence** as a function of year, geography, and demographic subgroup.
2. **Classify high-risk contexts** where obesity prevalence exceeds a meaningful threshold.
3. **Generate early-warning signals** that identify whether a state/demographic-group
   combination is likely to cross into the high-risk category in the *next observed year*.

The project is positioned as a **graduate-level data science portfolio project** combining
data engineering, temporal feature construction, predictive modeling, and interpretability.

---

## 2. Dataset

| Property | Detail |
|---|---|
| **Source** | CDC / Data.gov — Nutrition, Physical Activity, and Obesity — Behavioral Risk Factor Surveillance System |
| **URL** | https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system |
| **Primary observation unit** | Aggregated survey estimate: state × year × stratification (indicator/question/subgroup) |
| **Key target field** | `Data_Value` — survey-estimated percentage of adults with obesity |
| **Primary indicator** | "Percent of adults aged 18 years and older who have obesity" |

### Dataset Characteristics & Limitations

- This is **aggregated indicator-level** data, not person-level survey microdata.
- Values are self-reported estimates with confidence intervals.
- Not all state × year × subgroup cells are populated (sparse coverage).
- Stratification categories (income, age, education, race/ethnicity, sex) vary in completeness.
- Cross-year comparability is affected by survey methodology changes.
- All modeling must respect the **panel structure** (observations indexed by state, year, subgroup).

---

## 3. Repository Structure

```
brfss-obesity-early-warning/
│
├── PROJECT_PLAN.md            ← this file
├── README.md                  ← getting started guide
├── requirements.txt           ← pinned Python deps
├── .gitignore
│
├── configs/
│   └── config.yaml            ← all tunable parameters in one place
│
├── data/
│   ├── raw/                   ← original unmodified dataset (CSV from Data.gov)
│   ├── interim/               ← cleaned/merged intermediate outputs
│   └── processed/             ← final model-ready feature matrix + targets
│
├── models/
│   ├── regression/            ← serialized regression models (.joblib)
│   └── classification/        ← serialized classification models (.joblib)
│
├── notebooks/
│   ├── 01_phase1_reference.ipynb   ← brief reference to Phase 1 EDA work
│   ├── 02_modeling_experiments.ipynb ← interactive exploration / ablations
│   └── 03_final_analysis.ipynb     ← final structured analysis using src/ outputs
│
├── reports/
│   ├── figures/               ← saved plots (.png)
│   └── metrics/               ← saved JSON metrics per model
│
├── src/
│   ├── data/
│   │   ├── load_data.py       ← raw data loading with validation hooks
│   │   ├── validate_data.py   ← schema + quality checks
│   │   └── preprocess.py      ← cleaning, encoding, imputation
│   ├── features/
│   │   ├── build_features.py  ← derived features (lag, rolling, indicators)
│   │   └── build_targets.py   ← regression target + classification targets
│   ├── models/
│   │   ├── train_regression.py
│   │   ├── evaluate_regression.py
│   │   ├── train_classification.py
│   │   ├── evaluate_classification.py
│   │   └── explain.py         ← feature importance, coefficients, SHAP
│   ├── pipelines/
│   │   └── run_pipeline.py    ← end-to-end pipeline entry point
│   └── utils/
│       ├── paths.py           ← centralized path definitions
│       └── helpers.py         ← logging setup, I/O helpers
│
└── tests/
    ├── test_preprocess.py
    └── test_features.py
```

---

## 4. Modeling Tracks

### 4.1 Regression Track

**Goal:** Predict obesity prevalence (`Data_Value`) as a continuous outcome.

| Model | Rationale |
|---|---|
| Linear Regression | Baseline, maximum interpretability |
| Ridge / Lasso | Regularized linear, handles correlated features |
| Random Forest Regressor | Non-linear, robust baseline |
| Gradient Boosting Regressor | Strong ensemble, captures interactions |
| XGBoost Regressor | If it outperforms GBM with comparable complexity |

**Evaluation metrics:** MAE, RMSE, R²

### 4.2 Classification / Early Warning Track

**Goal:** Detect and predict high-risk obesity conditions.

**Primary target — `high_risk_obesity`**
- Binary: 1 if `Data_Value` ≥ top quartile threshold (computed on training split only)
- Interpreted as: "this state/group/year context exceeds the high-obesity threshold"

**Early warning target — `early_warning`** *(constructed if dataset permits)*
- Binary: 1 if a state/demographic-group combination that is *not currently* in the
  high-risk category **will be in the high-risk category in the next observed year**
- Requires reliable year-over-year linking by a composite group key:
  `LocationAbbr + Class + Question + Stratification1` (or similar)
- If coverage is too sparse to construct reliable next-year pairs, the pipeline will
  fall back gracefully to `high_risk_obesity` and document the limitation.

| Model | Rationale |
|---|---|
| Logistic Regression | Baseline, probability calibration, interpretable coefficients |
| Random Forest Classifier | Strong ensemble, feature importance |
| Gradient Boosting Classifier | Handles imbalanced classes well, strong AUC |
| XGBoost Classifier | Optional if it adds clear lift |

**Evaluation metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

---

## 5. Early Warning Idea — Design Details

### Concept

The early warning model asks: *given what we know about a state/group combination today
(year T), can we predict whether they will cross into the high-risk obesity category next
year (year T+1)?*

### Construction
1. Sort observations by `(group_key, YearStart)`.
2. For each group_key, compute `next_year_high_risk` = `high_risk_obesity` value at
   year T+1.
3. Filter: keep only observations at year T where year T+1 also exists for the same group.
4. This defines the "early warning" training sample.

### Feasibility Checkpoint
- If fewer than 500 valid next-year pairs exist, the script logs a warning and the
  pipeline defaults to the `high_risk_obesity` target.
- The code is written to be extensible: if more data becomes available, enabling the
  early warning path requires only changing one flag in `config.yaml`.

---

## 6. Feature Engineering Strategy

### Available Stratification Variables
From the BRFSS dataset structure:
- `YearStart` — temporal feature
- `LocationAbbr`, `LocationDesc` — geography
- `Class`, `Topic`, `Question` — indicator hierarchy
- `Data_Value_Type` — unit (e.g., "Value" vs. age-adjusted)
- `StratificationCategory1`, `Stratification1` — e.g., Income, Education, Age, Race, Sex

### Derived Features
- **Lag features:** prior-year prevalence for same group
- **Rolling mean:** 2-3 year rolling average of prevalence
- **Trend:** year-over-year change in prevalence
- **Region encoding:** US Census region from state abbreviation
- **Encoded categoricals:** OrdinalEncoder / one-hot for stratification subgroups

### Feature Selection
Features are selected after preprocessing using domain knowledge + permutation importance from the random forest models. Highly collinear features are identified via correlation analysis and pruned before final model training.

---

## 7. Train / Test Split Strategy

**Approach: Temporal split (time-aware, no look-ahead)**

- **Training set:** years ≤ 2017
- **Validation set:** years 2018–2019
- **Test set:** years 2020 and beyond

*Rationale:* The dataset represents a time series of repeated cross-sections.
Using future data to train would constitute data leakage since the early warning target
explicitly links year T to year T+1. Temporal splits preserve the causal ordering.

**Secondary strategy:** Group-aware cross-validation using `GroupKFold` stratified by
`LocationAbbr` for hyperparameter tuning, ensuring no single state is seen in both
train and validation folds within a CV round.

---

## 8. Assumptions & Risks

| # | Assumption / Risk | Mitigation |
|---|---|---|
| 1 | Dataset file is present in `data/raw/` before pipeline runs | README documents exact filename expected; loader raises clear error if missing |
| 2 | Stratification coverage is uneven across states/years | Missingness analysis in validation; indicator-level filters applied before feature building |
| 3 | Early warning links may be sparse | Feasibility check in `build_targets.py`; fallback documented |
| 4 | Aggregated data ≠ person-level; statistical models estimate group-level patterns only | Clearly stated throughout; no individual-level inferences drawn |
| 5 | Top-quartile threshold for high-risk computed on training data | Threshold stored in config/metadata after training split; test set labeled using training threshold |
| 6 | Some years or states may be missing entirely | Logged during validation; models trained on available data only |
| 7 | Class imbalance in classification targets | Evaluated using stratified sampling + `class_weight='balanced'` in linear models |

---

## 9. Implementation Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Project planning, structure design | ✅ |
| **Phase 1** | Scaffold — directories, configs, utilities | 🔄 |
| **Phase 2** | Data layer — load, validate, preprocess | 🔄 |
| **Phase 3** | Feature & target engineering | 🔄 |
| **Phase 4** | Regression track — train + evaluate | 🔄 |
| **Phase 5** | Classification track — train + evaluate | 🔄 |
| **Phase 6** | Interpretability | 🔄 |
| **Phase 7** | Pipeline runner | 🔄 |
| **Phase 8** | Notebooks + README + requirements | 🔄 |
| **Phase 9** | Tests | 🔄 |

---

## 10. Output Artifacts

| Output | Location | Description |
|---|---|---|
| Cleaned dataset | `data/interim/brfss_cleaned.csv` | After preprocessing |
| Feature matrix | `data/processed/features.csv` | Model-ready features |
| Target series | `data/processed/targets.csv` | Regression + classification targets |
| Regression model(s) | `models/regression/*.joblib` | Serialized scikit-learn models |
| Classification model(s) | `models/classification/*.joblib` | Serialized scikit-learn models |
| Regression metrics | `reports/metrics/regression_metrics.json` | MAE, RMSE, R² per model |
| Classification metrics | `reports/metrics/classification_metrics.json` | Accuracy, F1, AUC per model |
| Feature importance plots | `reports/figures/` | PNG plots |

---

*Plan version: 2026-03-07 | Author: AI-assisted, DSCI 521 Phase 2*
