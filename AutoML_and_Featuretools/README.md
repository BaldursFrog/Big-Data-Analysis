# AutoML and Feature Engineering Tools — Laboratory Work

This laboratory work focuses on **Automated Machine Learning (AutoML)** and **automated feature engineering** in the context of a regression task for **house price prediction**.

The goal of the lab is to compare different levels of automation in a machine learning pipeline and analyze their impact on model performance, development effort, and interpretability.

---

## Objective

To compare three approaches to solving a regression problem:

1. **Manual Feature Engineering**
2. **Automated Feature Engineering using FeatureTools**
3. **End-to-End AutoML using TPOT**

---

## Task Description

The task is to predict house prices based on numerical characteristics of residential areas.  
The dataset contains features such as:

- Median income  
- House age  
- Average number of rooms and bedrooms  
- Population  
- Average occupancy  
- Geographic coordinates (latitude and longitude)  
- Target variable: house price  

All approaches are evaluated on the same dataset using identical train–test splits.

---

## Implemented Approaches

### 1. Manual Feature Engineering

In this approach, new features are created manually based on domain knowledge.  
Examples include:

- rooms per person  
- bedroom-to-room ratio  
- population density  
- location-based features  
- interaction features  

A **Random Forest Regressor** is trained on the extended feature set.

---

### 2. Automated Feature Engineering (FeatureTools)

FeatureTools is used to automatically generate features using **Deep Feature Synthesis (DFS)**.

Key characteristics:
- No manual feature design
- Automatic generation of transformation-based features
- Feature matrix construction from the original dataset

A **Random Forest Regressor** is trained on the generated feature matrix.

---

### 3. AutoML (TPOT)

TPOT is used as a full **AutoML system** that automatically:
- selects models,
- optimizes hyperparameters,
- builds the entire ML pipeline.

The model is trained directly on the original feature set without manual feature engineering.

---

## Evaluation Metrics

The following regression metrics are used:

- **RMSE** — Root Mean Square Error  
- **MAE** — Mean Absolute Error  
- **R²** — Coefficient of Determination  

These metrics allow a quantitative comparison of prediction accuracy and model quality.

---

## Results Summary

The experimental results show that:

- Manual feature engineering improves performance but depends strongly on domain expertise.
- FeatureTools outperforms manual features by automatically discovering informative patterns.
- AutoML (TPOT) achieves the best overall performance by optimizing the full pipeline.

This demonstrates the effectiveness of higher levels of automation in machine learning workflows.

---

## Conclusion

This laboratory work illustrates the practical advantages of automated approaches in machine learning:

- **Manual feature engineering** offers interpretability but limited scalability.
- **Automated feature engineering** reduces manual effort and improves performance.
- **AutoML** provides the highest performance with minimal human intervention.

The experiment highlights the trade-offs between interpretability, computational cost, and predictive accuracy when choosing between different ML automation strategies.

---

## How to Run

1. Install required dependencies:
```bash
pip install pandas numpy scikit-learn featuretools tpot
