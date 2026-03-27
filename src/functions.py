import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_stratify_bins(series, num_bins=3):
    bins = np.linspace(series.min(), series.max(), num_bins + 1)
    # digitize: returns which bin each value belongs to
    return np.digitize(series, bins[:-1])

def data_split(df, stratify_col=None, test_size=0.2, seed=42):
    """
    Γενική συνάρτηση split. 
    Δέχεται έτοιμο το stratify_col (array-like).
    """
    train_set, val_set = train_test_split(
        df, 
        test_size=test_size, 
        random_state=seed, 
        stratify=stratify_col
    )
    return train_set, val_set

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Δημιουργεί το αντικείμενο προεπεξεργασίας σύμφωνα με τις οδηγίες:
    - Numeric (CpGs): Median Imputation & StandardScaler (σε Pipeline)
    - Categorical (Ethnicity): Most Frequent Imputation & One-Hot Encoding
    """
    
    # 1. Pipeline for CpG features (Numeric)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Missing values replaced with Median 
        ('scaler', StandardScaler())                   # Scaling with StandardScaler 
    ])

    # 2. Pipeline for Categorical features (Ethnicity&Sex)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
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
def get_split_stats(name, df_split):
    stats = {
        'Split': name,
        'n (Samples)': len(df_split),
        'Age (Mean ± Std)': f"{df_split['age'].mean():.2f} ± {df_split['age'].std():.2f}",
        'Age Range': f"{df_split['age'].min()} - {df_split['age'].max()}",
        'Sex Balance (% Male)': f"{(df_split['sex'] == 'M').mean()*100:.1f}%"
    }
    return stats