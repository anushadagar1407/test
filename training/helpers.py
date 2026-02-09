import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def get_class_distribution(y):
    """Get class distribution"""
    return y.value_counts()

def print_metrics(y_true, y_pred):
    """Print evaluation metrics"""
    print('Classification Report:')
    print(classification_report(y_true, y_pred))
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

def handle_imbalance(X, y):
    """Handle class imbalance using oversampling"""
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation"""
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=cv)
    print(f'Cross-validation scores: {scores}')
    print(f'Mean score: {scores.mean():.4f} (+/- {scores.std():.4f})')
    return scores