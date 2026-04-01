import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.utils import resample
from scipy.stats import spearmanr
from sklearn.model_selection import RandomizedSearchCV

def create_stratify_bins(series, num_bins=4):
    """Create bins for stratification based on the distribution of a numeric series (e.g., age)
     - series: The numeric series to be binned (e.g., age)
    - num_bins: The number of bins to create (default is 4 for quartiles)
     - Returns: An array of bin labels corresponding to each value in the series
    """
    bins = np.linspace(series.min(), series.max(), num_bins + 1)
    # digitize: returns which bin each value belongs to
    return np.digitize(series, bins[:-1])

def data_split(df, stratify_col=None, test_size=0.2, seed=42):
    """
    Split the data into training and validation sets, ensuring that the distribution of the stratify_col is preserved in both sets
     - stratify_col: The column used for stratification (e.g., age bins)
     - test_size: Proportion of the dataset to include in the validation split
     - seed: Random seed for reproducibility 
        - Returns: train_set, val_set
    """
    train_set, val_set = train_test_split(
        df, 
        test_size=test_size, 
        random_state=seed, 
        stratify=stratify_col
    )
    return train_set, val_set


def get_preprocessing_pipeline(numeric_features, categorical_features, n_components=None, seed=42):
    # Define the PCA step
    # If n_components is 'passthrough', the 'pca' step will do nothing
    if n_components == 'passthrough':
        pca_step = ('pca', FunctionTransformer(lambda x: x)) 
    else:
        pca_step = ('pca', PCA(n_components=n_components, random_state=seed))

    # 1. Pipeline for CpG features (Numeric)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        pca_step # here we dynamically include either PCA or the "passthrough" step
    ])

    # 2. Pipeline for Categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False)) 
    ])

    # 3. ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

# Statistics for each split
"""- n (Samples): The number of samples in the split   
    - Age (Mean ± Std): The mean and standard deviation of the age distribution in the split
    - Age Range: The minimum and maximum age in the split
    - Sex Balance (%  Male): The percentage of male samples in the split (calculated as the proportion of samples where sex is 'M')
"""
def get_split_stats(name, df_split):
    stats = {
        'Split': name,
        'n (Samples)': len(df_split),
        'Age (Mean ± Std)': f"{df_split['age'].mean():.2f} ± {df_split['age'].std():.2f}",
        'Age Range': f"{df_split['age'].min()} - {df_split['age'].max()}",
        'Sex Balance (% Male)': f"{(df_split['sex'] == 'M').mean()*100:.1f}%"
    }
    return stats

def evaluate_with_bootstrap(y_true, y_pred, n_resamples=1000, seed=42):
    """
    Calculation of RMSE, MAE, R2 και Pearson r with 95% Confidence Intervals 
    using bootstrap resampling only in the predictions of the validation set to assess the stability of the model's performance metrics.
     - y_true: The true target values of the validation set
    - y_pred: The predicted target values
    - n_resamples: The number of bootstrap resamples to perform (default is 1000)
    - seed: Random seed for reproducibility
     - Returns: A dictionary with the mean and 95% CI for each metric
    """
    metrics = {'rmse': [], 'mae': [], 'r2': [], 'pearson_r': []}
    
    # Transformation to numpy arrays for certainty in indices
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for i in range(n_resamples):
        # Bootstrap Resampling  with replacement of the predictions
        # We use seed+i to have different but reproducible samples
        y_t_boot, y_p_boot = resample(y_true, y_pred, replace=True, random_state=seed + i)
        
        # Calculation of the 4 metrics
        metrics['rmse'].append(np.sqrt(mean_squared_error(y_t_boot, y_p_boot)))
        metrics['mae'].append(mean_absolute_error(y_t_boot, y_p_boot))
        metrics['r2'].append(r2_score(y_t_boot, y_p_boot))
        
        corr, _ = pearsonr(y_t_boot, y_p_boot)
        metrics['pearson_r'].append(corr)
    
    # Calculation of Mean and 95% CI (2.5th and 97.5th percentile)
    results = {}
    for m in metrics:
        mean_val = np.mean(metrics[m])
        lower = np.percentile(metrics[m], 2.5)
        upper = np.percentile(metrics[m], 97.5)
        results[m] = (mean_val, lower, upper)
        
    return results

def calculate_regression_metrics(y_true, y_pred):
    """Calculation of the 4 basic metrics"""
    corr, _ = pearsonr(y_true, y_pred)
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson_r': corr
    }

def get_bootstrap_samples(y_true, y_pred, n_resamples=1000, seed=42):
    """Creates only the samples of the metrics."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    boot_results = {m: [] for m in ['rmse', 'mae', 'r2', 'pearson_r']}
    
    for i in range(n_resamples):
        yt_b, yp_b = resample(y_true, y_pred, replace=True, random_state=seed + i)
        metrics = calculate_regression_metrics(yt_b, yp_b)
        for m in metrics:
            boot_results[m].append(metrics[m])
            
    return boot_results

def calculate_confidence_intervals(samples, confidence=95):
    """Calculates Mean and Confidence Intervals for a list of samples."""
    lower_p = (100 - confidence) / 2
    upper_p = 100 - lower_p
    return (np.mean(samples), np.percentile(samples, lower_p), np.percentile(samples, upper_p))


def perform_stability_selection(X, y, n_subsamples=50, sub_size=0.8, top_k=200, threshold=0.5, seed=42):
    """
    Implements Stability Selection based on Spearman Correlation.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    feature_names = X.columns if hasattr(X, 'columns') else np.arange(n_features)
    
    # Matrix for counting how many times each feature was selected
    selection_counts = np.zeros(n_features)
    
    sample_size = int(n_samples * sub_size)
    
    for i in range(n_subsamples):
        # 1. Subsample without replacement
        indices = np.random.choice(n_samples, size=sample_size, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
        
        # 2. Calculation of Spearman correlation for each feature
        correlations = []
        for j in range(n_features):
            score, _ = spearmanr(X_sub[:, j], y_sub)
            correlations.append(abs(score))
        
        # 3. Ordering and selection of the top 200
        correlations = np.array(correlations)
        # Take the indices of the 200 largest values
        top_indices = np.argsort(correlations)[-top_k:]
        selection_counts[top_indices] += 1
        
    # Conversion to frequency (0.0 to 1.0)
    selection_frequencies = selection_counts / n_subsamples
    
    # Selection of features with frequency > threshold
    stable_features_indices = np.where(selection_frequencies > threshold)[0]
    
    return stable_features_indices, selection_frequencies


def run_hyperparameter_tuning(model, param_dist, X, y, n_iter=40, seed=42):
    """
    Runs RandomizedSearchCV to find optimal hyperparameters 
    """
    search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist, 
        n_iter=n_iter, 
        cv=5, 
        scoring='neg_root_mean_squared_error', 
        random_state=seed,
        n_jobs=-1,
        refit=True # automatic refit in full dev set
    )
    search.fit(X, y)
    return search