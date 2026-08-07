# 🏆 Ames Housing Price Prediction | Top 20% Ensemble Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FF6F00?style=for-the-badge&logo=catboost&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)

An end-to-end machine learning pipeline for the **Ames Housing regression task** on Kaggle.

By leveraging domain-driven feature engineering, skewness correction, 10-Fold Stratified Cross-Validation, and 4-model weighted ensemble (CatBoost 35%, LightGBM 25%, XGBoost 20%, Ridge 20%), this solution achieved a **Global Leaderboard Rank of 776 (Top 20%)** with an **RMSE score of ~0.1228**.

---

🏆 Kaggle Result: Top 20% · Best RMSLE: 0.12286

---

## 📌 Competition Overview
* **Objective:** Predict house sale prices in Ames, Iowa using 79 explanatory variables.
* **Evaluation Metric:** Root Mean Squared Logarithmic Error (RMSLE) between predicted and actual values.
* **Kaggle Notebook:** [View Interactive Kaggle Solution](https://www.kaggle.com/code/ryhorabramovich/ames-housing-4-model-super-blend-ensemble)

---

## 🛠️ Key Pipeline Highlights

### 1. Data Preprocessing & Outlier Handling
* **Outlier Removal:** Identified and dropped 2 extreme outliers (`GrLivArea > 4000` with `SalePrice < $300,000`) that heavily distorted model boundaries.
* **Log Transformation:** Applied `np.log1p` to target variable `SalePrice` and all numerical features with skewness $> 0.75$ to align with normal distribution.

### 2. Domain-Driven Feature Engineering
* **Ordinal Quality Mapping:** Encoded categorical structural qualities (`ExterQual`, `BsmtQual`, `KitchenQual`, etc.) into ordered integers ($1-5$).
* **Aggregated Square Footage (`TotalSF`):** Combined basement, 1st floor, and 2nd floor square footage into a unified size metric.
* **Porch & Bath Metrics:** Created aggregated total porch area (`TotalPorch`) and total bathroom counts (`TotalBath`).
* **Interaction Terms:** Generated `OverallQual_SF` to highlight non-linear impacts on high-value properties.

### 3. Model Architecture & Ensembling
To avoid overfitting and maximize generalization, predictions were generated using **10-Fold Cross-Validation** across a blended ensemble:

| Model | Weight | Purpose / Strengths |
| :--- | :---: | :--- |
| **CatBoost Regressor** | **35%** | Primary engine, excellent handling of categorical relationships |
| **LightGBM Regressor** | **25%** | Fast, high-capacity decision tree splitting |
| **XGBoost Regressor** | **20%** | Fine-tuned depth-wise boosting for robust residual learning |
| **Ridge Regression** | **20%** | L2 Regularized Linear Anchor for baseline variance reduction |

---

## 🚀 How to Run Locally

1. **Clone Repository:**
   ```bash
   git clone [https://github.com/ryhorabramovich/ames-housing-prediction.git](https://github.com/ryhorabramovich/ames-housing-prediction.git)
   cd ames-housing-prediction
