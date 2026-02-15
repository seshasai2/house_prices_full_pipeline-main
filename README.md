# House Prices — End-to-End ML Pipeline

**Goal:** Predict house sale prices (Kaggle: House Prices — Advanced Regression Techniques) using an end-to-end pipeline: EDA → Feature Engineering → Modeling → Ensembling → Explainability.

---

## 🚀 Project Summary
Built a full regression pipeline to predict `SalePrice` with the following highlights:
- Robust feature engineering (TotalSF, OverallGrade, Age, log transforms).
- Models trained: Ridge, Lasso, RandomForest, XGBoost.
- Advanced stacking using Out-Of-Fold (OOF) predictions and a Ridge meta-model.
- Final public Kaggle RMSE (stacked model): **`<0.132>`**.
- Notebook (published): <https://www.kaggle.com/code/seshasai2409/house-prices-full-pipeline/output?scriptVersionId=269845793>

---

## 🔧 Tech Stack
- Python, NumPy, Pandas
- scikit-learn (Ridge, Lasso, RandomForest, KFold)
- XGBoost
- Matplotlib, Seaborn
- Optional: SHAP for explainability


---

## 📈 Leaderboard Progress
| Day | Model / Step | Public RMSE |
|-----|--------------|-------------|
| Day 4 | Linear Regression baseline | 0.14 |
| Day 5 | Ridge / Lasso | 0.13 |
| Day 6 | RandomForest / XGBoost | 0.125 |
| Day 7 | OOF Stacking (Ridge meta) | **<0.132>** |

> Update the table as you improve. Recruiters like to see the iterative progression.

---
