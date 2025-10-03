# 🏡 Ames Housing Prices — Machine Learning Regression Project

## 📌 Overview
This project uses the **Ames Housing dataset** (a well-known Kaggle dataset) to predict house sale prices based on various property features.  
It demonstrates a full end-to-end machine learning workflow:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Preprocessing pipelines (numeric + categorical features)
- Multiple regression models (linear & tree-based)
- Hyperparameter tuning with cross-validation
- Model comparison and experiment logging
- Error analysis and feature importance

---

## ⚙️ Project structure

housing-prices/
│
├── data/                  # raw and processed datasets (ignored in git)
├── notebooks/             # Jupyter notebooks for experiments
├── src/                   # source code (preprocessing, features, models)
├── models/                # saved trained models (.joblib)
├── plots/                 # visualizations (residuals, feature importance)
├── experiments_log.csv    # experiment history
├── requirements.in / .txt # dependencies
├── environment.yml        # conda environment
├── README.md              # project documentation
└── .gitignore


---

## 📊 Models & Results
All models were trained using a unified pipeline with preprocessing and feature engineering.  
Hyperparameter tuning was performed via **GridSearchCV (5-fold CV)**, results logged in `experiments_log.csv`.

| Model        | CV RMSE | Test RMSE ($) | MAE ($) | R²   |
|--------------|---------|---------------|---------|------|
| Ridge        | …       | …             | …       | …    |
| Lasso        | …       | …             | …       | …    |
| RandomForest | …       | …             | …       | …    |
| XGBoost      | …       | …             | …       | …    |
| LightGBM     | …       | …             | …       | …    |
| CatBoost     | …       | …             | …       | …    |

*(XGBoost currently achieves the best performance with R² ≈ 0.936.)*

---

## 📈 Visualizations
- Residuals vs Predicted values  
- Residuals histogram  
- Q–Q plot of residuals  
- Built-in feature importance (tree-based models)  
- Permutation importance (model-agnostic)  

Example:  
![Residuals vs Predicted](plots/xgb_residuals_scatter.png)  
![Feature Importance](plots/xgb_feature_importance_builtin.png)

---
