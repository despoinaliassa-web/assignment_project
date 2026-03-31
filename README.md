# Epigenetic Age Predictor (DNA Methylation Clock)

This project develops a computational model (Epigenetic Clock) to predict biological age using DNA methylation data (CpG sites)

## Project Structure

* `notebooks/`
    * `preprocessing.ipynb`: Data cleaning, Data Preprocessing, EDA
    * `models.ipynb`: Training of regression models and feature selection 
* `src/`
    * `functions.py`: Shared functions for metrics, bootstrap evaluation, and visualization
* `figures/`: Plots (Venn, correlation plots, feature importance)
* `results/`: Results tables and model comparisons in CSV format

## 🚀 Work Progress

### Task 1 & 2: Preprocessing & Initial Selection
* Data splitting, handling missing values, data scaling
* Training and evaluation of Regression models (Ordinary Least Squares (OLS) Linear Regression, ElasticNet Regression,Support Vector Regression (SVR) with RBF Kernel, Bayesian Ridge Regression)

### Task 3: Feature Selection & Evaluation
* Method Comparison : 
    * Stability Selection (185 features)
    * mRMR Selection (150 features) 
* Use of **Bootstrap (1000 resamples)** to calculate 95% confidence intervals (CI)

## 📈 Next Steps
* **Task 4:** Hyperparameter tuning to minimize prediction error
* Final evaluation on the Evaluation set