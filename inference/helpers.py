import numpy as np
import pandas as pd

def preprocess_input(data):
    """Preprocess input data for inference"""
    df = pd.DataFrame([data])
    # Apply same transformations as training
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df

def format_response(prediction, probability, confidence_threshold=0.7):
    """Format inference response"""
    confidence = max(probability)
    is_confident = confidence >= confidence_threshold
    
    return {
        'prediction': int(prediction),
        'probability': probability.tolist(),
        'confidence': float(confidence),
        'is_confident': bool(is_confident)
    }

def validate_input(data, required_features):
    """Validate input data"""
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        raise ValueError(f'Missing required features: {missing_features}')
    return True
