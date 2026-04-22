# Credit Risk Prediction (TCRI)

Predicting Taiwan Corporate Credit Risk Indicator (TCRI) using multiple machine learning models. Built as part of a Fintech ML course project.

## Objective

Develop a binary classification model to predict whether a company's TCRI score will reach **≥ 7** (high credit risk) in the next month, using historical financial ratios and industry classification data.

## Models Compared

| Model | PR-AUC | ROC-AUC | F1 | Brier Score |
|-------|--------|---------|----|-------------|
| **XGBoost** | **0.686** | **0.822** | **0.636** | 0.150 |
| Gradient Boosting | 0.684 | 0.817 | 0.638 | 0.149 |
| Neural Network (MLP) | 0.667 | 0.815 | 0.637 | 0.152 |
| Random Forest | 0.666 | 0.813 | 0.637 | 0.148 |
| Logistic Regression | 0.654 | 0.802 | 0.613 | 0.170 |
| SVM | 0.554 | 0.768 | 0.586 | 0.177 |

## Data

- **Sources**: TCRI ratings + financial ratio data for ~1000+ Taiwanese companies
- **Period**: Train (2014–2021), Validation (2022), Test (2023)
- **Features**: ~80+ financial ratios + GICS industry classification (one-hot encoded)
- **Label**: Binary — whether TCRI ≥ 7 at t+1

## Methodology

1. **Data Pipeline**: Merge TCRI scores with financial ratios, forward-shift labels by 1 month, time-based train/valid/test split (no data leakage)
2. **Preprocessing**: Median imputation for numeric features, one-hot encoding for categorical features
3. **Model Selection**: Hyperparameter grid search on validation set, optimized for PR-AUC
4. **Calibration**: Platt scaling (sigmoid) and Isotonic regression applied to calibrate predicted probabilities
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
├── baseline_model.ipynb     # Logistic Regression baseline
├── xgboost_model.ipynb      # XGBoost
├── rf_model.ipynb           # Random Forest
├── nn_model.ipynb           # Neural Network (MLP)
├── svm_model.ipynb          # SVM
├── qrt_model.ipynb          # Gradient Boosting
├── robustness_splits.ipynb  # Robustness: alternative data splits
├── robustness_tau.ipynb     # Robustness: TCRI threshold sensitivity
└── outputs/                 # Saved predictions, metrics, plots
```

## Tech Stack

Python · scikit-learn · XGBoost · SHAP · pandas · matplotlib · Jupyter Notebook
