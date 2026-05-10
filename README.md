# Heart Disease Prediction with Explainable AI (SHAP)

SEDS 537 – Machine Learning | Term Project | İzmir Institute of Technology | Spring 2026

## Project Overview

This project builds a machine learning pipeline for binary heart disease classification using structured EHR data. The pipeline combines strong predictive models with SHAP-based post-hoc explainability to produce clinically transparent predictions.

## Dataset

Cardiovascular Disease Dataset from Kaggle (~70,000 records).
Download from: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
Place the file as `cardio_train.csv` in the project root.

## Setup

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn
```

## Run

```bash
python3 analysis.py
```

Results will be saved to `results.csv` and figures to the `figures/` folder.

## Models

- Logistic Regression (Baseline 1)
- Random Forest (Baseline 2)
- Support Vector Machine (Baseline 3)
- XGBoost + SHAP (Proposed Method)

## Results

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 72.85% | 70.70% | 0.7947 |
| Random Forest | 71.09% | 70.37% | 0.7646 |
| SVM | 73.54% | 71.45% | 0.7926 |
| XGBoost | 73.57% | 72.05% | 0.7978 |
