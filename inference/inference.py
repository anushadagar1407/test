import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from google.cloud import storage
import helpers

app = Flask(__name__)

# Load model at startup
model = None

def load_model_from_gcs(bucket_name, model_path):
    """Load model from Google Cloud Storage"""
    global model
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename('/tmp/model.joblib')
    model = joblib.load('/tmp/model.joblib')
    print('Model loaded successfully')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on input data"""
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Load model from GCS
    load_model_from_gcs('your-bucket-name', 'models/trained_model.joblib')
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)