import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import helpers

def load_data(bucket_name, file_path):
    """Load data from Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = pd.read_csv(f'gs://{bucket_name}/{file_path}')
    return data

def engineer_features(df):
    """Apply feature engineering transformations"""
    # Create new features
    df['feature_1'] = helpers.transform_feature_1(df)
    df['feature_2'] = helpers.transform_feature_2(df)
    
    # Normalize features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def save_data(df, bucket_name, output_path):
    """Save processed data to Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(output_path)
    df.to_csv(f'gs://{bucket_name}/{output_path}', index=False)

if __name__ == '__main__':
    # Load raw data
    raw_data = load_data('your-bucket-name', 'data/raw_data.csv')
    
    # Engineer features
    processed_data = engineer_features(raw_data)
    
    # Save processed data
    save_data(processed_data, 'your-bucket-name', 'data/processed_data.csv')
    
    print('Feature engineering completed successfully!')