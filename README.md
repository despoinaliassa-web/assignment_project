# Epigenetic Age Prediction Project

This repository contains a machine learning pipeline for predicting biological age using DNA methylation data (CpG sites) and demographic metadata.

## Project Overview
The goal is to develop and evaluate an **Epigenetic Clock** based on 1000 selected CpG features and categorical metadata (Sex and Ethnicity) from 456 samples.

## Project Structure
*   `notebooks/`: Jupyter Notebooks for Exploratory Data Analysis (EDA) and Model Training.
*   `src/`: Contains `functions.py` with reusable preprocessing and modeling logic.
*   `figures/`: Visualizations generated during EDA (150 dpi resolution).
*   `requirements.txt`: List of dependencies (scikit-learn, pandas, seaborn, etc.).

## Methodology & Constraints
*   **Preprocessing:** One-Hot Encoding for categorical data and Robust Scaling for DNA methylation.
*   **Data Leakage Prevention:** All parameters (imputation, scaling) are fitted **only** on the training split and applied to validation/evaluation sets.
*   **Reproducibility:** Constant `seed=42` used across all experiments.
*   **Evaluation:** Metrics include 95% Confidence Intervals (CI) estimated via 1000-fold resampling with replacement.

## Key Visualizations
Below is a sample distribution of the development set:

![Gender and Ethnicity Distribution](figures/Correlation%20gender-ethnicity.png)

## 🚀 Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
