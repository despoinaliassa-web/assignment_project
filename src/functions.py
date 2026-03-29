import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

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

def get_preprocessing_pipeline(numeric_features, categorical_features, n_components=None):
    """
    Creation of a preprocessing pipeline for both numeric and categorical features.
     - Numeric features: Imputation with median, Standard Scaling and PCA for dimensionality reduction (n_components can be set based on the Elbow Plot) 
     - PCA is included in the numeric pipeline to ensure that it is applied only to the numeric features after imputation and scaling
     - Categorical features: Imputation with most frequent + One-Hot Encoding (drop first to avoid multicollinearity)
     - ColumnTransformer to combine both pipelines  
    """
    
    # 1. Pipeline for CpG features (Numeric)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Missing values replaced with Median 
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=n_components, random_state=42))                                    # Scaling with StandardScaler 
    ])

    # 2. Pipeline for Categorical features (Ethnicity&Sex)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Keep this to handle any potential missing values in categorical features in the future
        ('onehot', OneHotEncoder(drop='first', sparse_output=False)) 
    ])

    # 3. ColumnTransformer units 1 & 2
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Ensure that no other column is calculated (e.g. sex, sample_id)
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