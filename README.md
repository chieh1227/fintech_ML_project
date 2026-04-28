# Credit Risk Prediction (TCRI)

Predicting Taiwan Corporate Credit Risk Indicator (TCRI) using multiple machine learning models. Built as part of a Fintech ML course project.

## Objective

Develop a binary classification model to predict whether a company's TCRI score will reach **≥ 7** (high credit risk) in the next year, using historical financial ratios and industry classification data.

## Models Compared

All metrics below are computed on Platt-calibrated probabilities to match the report.

| Model | PR-AUC | ROC-AUC | F1 | Brier Score |
|-------|--------|---------|----|-------------|
| XGBoost | **0.686** | **0.822** | 0.638 | 0.150 |
| Gradient Boosting | 0.684 | 0.817 | 0.624 | **0.147** |
| Neural Network (MLP) | 0.667 | 0.815 | 0.638 | 0.155 |
| Random Forest | 0.666 | 0.813 | **0.639** | 0.150 |
| Logistic Regression | 0.654 | 0.802 | 0.598 | 0.158 |
| SVM | 0.554 | 0.768 | 0.588 | 0.175 |

## Data

- **Sources**: TCRI ratings + financial ratio data for 1,028 Taiwanese listed/OTC companies (21,293 firm-year rows)
- **Period** (forecast-year convention, predicting next-year TCRI from current-year features):
  - Train: forecast years 2015–2022 (features 2014–2021)
  - Validation: forecast year 2023 (features 2022)
  - Test: forecast year 2024 (features 2023)
- **Features**: 5 financial ratios (Working Capital / Total Assets, Retained Earnings / Total Assets, Cash Flow / Total Debt, Total Debt / Total Assets, Current Ratio) + stock prefix + 12 GICS industry sectors (one-hot encoded)
- **Label**: Binary — whether TCRI ≥ 7 at t+1 year

## Methodology

1. **Data Pipeline**: Merge TCRI scores with financial ratios, forward-shift labels by 1 year, time-based train/valid/test split (no data leakage)
2. **Preprocessing**: Median imputation for numeric features, one-hot encoding for categorical features
3. **Model Selection**: Hyperparameter grid search on validation set, optimized for PR-AUC
4. **Calibration**: Platt scaling (sigmoid) applied to calibrate predicted probabilities
5. **Explainability**: SHAP analysis for feature importance across all models
6. **Robustness**: Sensitivity analysis on different train/test splits and TCRI thresholds (tau)

## Project Structure

```
├── src/
│   ├── data_prep.py        # Data loading, merging, label creation, splitting
│   ├── metrics.py           # PR-AUC, ROC-AUC, F1, Brier, ECE, Logloss
│   ├── explain.py           # SHAP summary and bar plots
│   ├── reporting.py         # Save metrics, predictions, feature weights
│   └── modeling/            # Model-specific utilities
├── baseline_model.ipynb          # Logistic Regression baseline
├── baseline_model_gics.ipynb     # Logistic Regression with GICS features
├── baseline_model_no_gics.ipynb  # Logistic Regression without GICS (ablation)
├── xgboost_model.ipynb           # XGBoost
├── rf_model.ipynb                # Random Forest
├── nn_model.ipynb                # Neural Network (MLP)
├── svm_model.ipynb               # SVM
├── gb_model.ipynb                # Gradient Boosting
├── robustness_splits.ipynb       # Robustness: alternative data splits
├── robustness_tau.ipynb          # Robustness: TCRI threshold sensitivity
└── outputs/                      # Saved predictions, metrics, plots
```

## Tech Stack

Python · scikit-learn · XGBoost · SHAP · pandas · matplotlib · Jupyter Notebook
