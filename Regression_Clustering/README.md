# Regression and Clustering Experiments

This laboratory work is devoted to the study and comparison of **regression** and **clustering** methods on both **synthetic** and **real-world datasets**.  
The work combines classical machine learning algorithms and neural network–based approaches, with an emphasis on evaluation metrics, generalization ability, training cost, and interpretability.

---

## Objectives

The main objectives of this laboratory work are:

- To compare linear regression models and neural networks on regression tasks
- To analyze the influence of data complexity and noise on model performance
- To study clustering methods on image data
- To compare traditional clustering (K-Means) with a deep learning–based clustering approach
- To evaluate models using quantitative metrics and statistical analysis

---

## Tasks Overview

### Part 1: Regression Experiments

Regression experiments are conducted on:
- **Synthetic datasets** with different structures (linear, polynomial, exponential, interaction-based, piecewise)
- **California Housing dataset** (real-world data)

Models used:
- Ordinary Least Squares (Linear Regression)
- Ridge Regression
- Lasso Regression
- Neural Network Regressor (MLP)

Evaluation metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² score
- Training time
- Cross-validation statistics

Additional analysis:
- Generalization gap
- Interpretability comparison
- Statistical significance tests (paired t-test, Wilcoxon test)
- Bootstrap confidence intervals

---

### Part 2: Clustering Experiments

Clustering experiments are performed on the **MNIST dataset**.

Methods used:
- K-Means clustering
- MiniBatch K-Means
- DeepCluster-like neural network approach

Feature representations:
- Raw pixel values
- PCA-based features
- HOG (Histogram of Oriented Gradients) features
- Learned features from a neural network

Evaluation metrics:
- Clustering Accuracy
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score
- Training time

Visual analysis:
- Cluster sample visualization
- Metric comparison plots

---

## Project Structure

- `experiment1_main.py` — regression experiments on synthetic and California Housing datasets  
- `experiment2_mnist.py` — clustering experiments on MNIST  
- `BDA_Topic6_Regression_and_Clustering_Report.docx` — detailed laboratory report  
- `outputs/` — saved models, metrics, and plots (generated during execution)  
- `screenshots/` — screenshots used in the report  

---

## Methodology

1. **Data generation and loading**
   - Synthetic data generation with controlled parameters
   - Loading and preprocessing of real-world datasets

2. **Preprocessing**
   - Train–test split
   - Feature scaling and normalization

3. **Model training**
   - Classical ML models
   - Neural network models with hyperparameter tuning

4. **Evaluation**
   - Quantitative metrics
   - Cross-validation
   - Statistical hypothesis testing

5. **Visualization**
   - Performance plots
   - Cluster visualization

---

## Results Summary

- Neural network regressors outperform linear models on nonlinear datasets but require significantly more training time.
- Regularized linear models (Ridge, Lasso) provide a good balance between performance and interpretability.
- K-Means clustering performs competitively on MNIST when combined with suitable feature extraction.
- Deep clustering methods require careful feature learning to outperform traditional approaches.
- Statistical analysis confirms the significance of performance differences across models.

---

## How to Run

### Requirements
Python 3.9+

Required libraries include:
- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- scipy

### Run regression experiments
```bash
python experiment1_main.py
