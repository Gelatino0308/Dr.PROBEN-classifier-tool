from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np

# Add the diabetes directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'diabetes'))
from predict_diabetes import load_model, predict_diabetes

app = Flask(__name__)
CORS(app)  # This should allow all origins

# Load the model once when the server starts
try:
    model, scaler = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler = None, None

@app.route('/api/predict/diabetes', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded properly"}), 500
            
        data = request.json
        print("Received data:", data)  
        
        # Create a DataFrame from the submitted form data
        input_data = pd.DataFrame([{
            'Pregnancies': float(data['pregnancies']),
            'Glucose': float(data['plasma']),
            'BloodPressure': float(data['BP']),
            'SkinThickness': float(data['skin']),
            'Insulin': float(data['insulin']),
            'BMI': float(data['BMI']),
            'DiabetesPedigreeFunction': float(data['pedigree']),
            'Age': float(data['age'])
        }])
        
        # Make prediction
        prediction, probability = predict_diabetes(input_data, model, scaler)

        # Ensure both are arrays for safe indexing
        prediction = np.atleast_1d(prediction)
        probability = np.atleast_1d(probability)
        
        result = {
            'prediction': int(prediction[0]),  # 0 or 1
            'probability': float(probability[0]),  # Probability as a decimal
            'percentage': round(float(probability[0] * 100))  # Percentage for display
        }
        
        print("Prediction result:", result)  # Debug print
        return jsonify(result)
        
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)