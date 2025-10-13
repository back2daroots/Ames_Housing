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
```
housing_prices/   
├─ data/                    # raw and processed datasets (ignored in git)
├─ notebooks/               # Jupyter notebooks for experiments
│  └─ housing_prices.ipynb 
├─ src/                     # source code (preprocessing, features, models)
├─ models/                  # saved trained models (.joblib)
├─ plots/                   # visualizations (residuals, feature importance)
├─ experiments_log.csv      # experiment history
├─ README.md                # project documentation
├─ requirements.in / .txt   # dependencies
├── environment.yml         # conda environment
└─ .gitignore
```
---

## 📊 Models & Results
All models were trained using a unified pipeline with preprocessing and feature engineering.  
Hyperparameter tuning was performed via **GridSearchCV (5-fold CV)**, results logged in `experiments_log.csv`.

| Model         |   CV RMSE ($) |   Test RMSE ($) |   MAE ($) |     R² |
|:--------------|--------------:|----------------:|----------:|-------:|
| CatBoost      |       22809.9 |         22748.8 |   13796.2 | 0.9355 |
| XGB           |       24396.2 |         23127   |   14008.9 | 0.9333 |
| LightGBM      |       24338.4 |         26107.6 |   14947.1 | 0.915  |
| RandomForest  |       27425.4 |         26621.8 |   15804.8 | 0.9116 |
| XGB (cleaned) |         ---   |         27160.7 |   13928.7 | 0.908  |
| Ridge         |       30062.6 |         31011.9 |   18624.9 | 0.88   |
| Lasso         |       31078.2 |         31371.3 |   18815.4 | 0.8772 |



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
