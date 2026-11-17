# Supervised Classification Modeling & Tuning

This repository contains a Jupyter Notebook that compares four supervised classification algorithms on the Adult Census Income dataset.

## Models
- Logistic Regression
- Random Forest
- Extra Trees
- HistGradientBoosting

## Workflow
- Train/test split before preprocessing
- Numerical median imputation and standard scaling
- Categorical mode imputation and one-hot encoding
- Stratified 3-fold `GridSearchCV` hyperparameter tuning
- Model comparison using Precision, Recall, F1-score, and ROC-AUC
- ROC curves and champion confusion matrix
- Champion selection using cross-validation ROC-AUC
- Serialization of the full fitted pipeline using Joblib

## Run
Open the notebook in Google Colab or Jupyter and run all cells. The first run downloads the Adult dataset from OpenML, so an internet connection is required.

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

The notebook generates:
- `champion_model.joblib`
- `model_comparison.csv`
- `roc_auc_comparison.png`
- `champion_confusion_matrix.png`

**Note:** Run all notebook cells before submission so the generated model and evidence files are available to upload.
