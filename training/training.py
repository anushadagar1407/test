import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from google.cloud import storage
import joblib
import helpers

def load_training_data(bucket_name, file_path):
    """Load training data from Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = pd.read_csv(f'gs://{bucket_name}/{file_path}')
    return data

def prepare_data(df):
    """Prepare data for training"""
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train machine learning model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    score = model.score(X_test, y_test)
    print(f'Model Accuracy: {score:.4f}')
    return score

def save_model(model, bucket_name, model_path):
    """Save trained model to Google Cloud Storage"""
    local_path = '/tmp/model.joblib'
    joblib.dump(model, local_path)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(local_path)

if __name__ == '__main__':
    # Load data
    training_data = load_training_data('your-bucket-name', 'data/processed_data.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(training_data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, 'your-bucket-name', 'models/trained_model.joblib')
    
    print('Training completed successfully!')
