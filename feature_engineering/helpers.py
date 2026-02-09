import numpy as np
import pandas as pd

def transform_feature_1(df):
    """Transform feature 1"""
    return df['input_col_1'].apply(lambda x: np.log1p(x))

def transform_feature_2(df):
    """Transform feature 2"""
    return df['input_col_2'].apply(lambda x: x ** 2)

def validate_data(df):
    """Validate data quality"""
    if df.isnull().sum().sum() > 0:
        print('Warning: Missing values detected')
        df = df.dropna()
    return df

def get_feature_statistics(df):
    """Get statistical summary of features"""
    return df.describe()