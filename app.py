from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Add disease directories to path
for disease_dir in ['diabetes', 'heart', 'cancer']:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), disease_dir))

# Import prediction functions
from predict_diabetes import load_model as load_diabetes_model, predict_diabetes
from predict_heart import load_model as load_heart_model, predict_heart
from predict_cancer import load_model as load_cancer_model, predict_cancer

# Load all models
models = {}
try:
    models['diabetes'] = load_diabetes_model()
    print("Diabetes model loaded successfully!")
except Exception as e:
    print(f"Error loading diabetes model: {e}")
    models['diabetes'] = (None, None)

try:
    models['heart'] = load_heart_model()
    print("Heart model loaded successfully!")
except Exception as e:
    print(f"Error loading heart model: {e}")
    models['heart'] = (None, None)

try:
    models['cancer'] = load_cancer_model()
    print("Cancer model loaded successfully!")
except Exception as e:
    print(f"Error loading cancer model: {e}")
    models['cancer'] = (None, None)

@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes_endpoint():
    try:
        model, scaler = models['diabetes']
        if model is None or scaler is None:
            return jsonify({"error": "Diabetes model not loaded properly"}), 500
            
        data = request.json
        
        # Create DataFrame for diabetes prediction
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
        
        prediction, probability = predict_diabetes(input_data, model, scaler)
        
        prediction = np.atleast_1d(prediction)
        probability = np.atleast_1d(probability)
        
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'percentage': round(float(probability[0] * 100))
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/heart', methods=['POST'])
def predict_heart_endpoint():
    try:
        model, scaler = models['heart']
        if model is None or scaler is None:
            return jsonify({"error": "Heart model not loaded properly"}), 500
            
        data = request.json
        
        # Create DataFrame for heart prediction
        input_data = pd.DataFrame([{
            'age': float(data['age']),
            'sex': float(data['sex']),
            'cp': float(data['cp']),
            'trestbps': float(data['trestbps']),
            'chol': float(data['chol']),
            'fbs': float(data['fbs']),
            'restecg': float(data['restecg']),
            'thalach': float(data['thalach']),
            'exang': float(data['exang']),
            'oldpeak': float(data['oldpeak']),
            'slope': float(data['slope']),
            'ca': float(data['ca']),
            'thal': float(data['thal'])
        }])
        
        prediction, probability = predict_heart(input_data, model, scaler)
        
        prediction = np.atleast_1d(prediction)
        probability = np.atleast_1d(probability)
        
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'percentage': round(float(probability[0] * 100))
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/cancer', methods=['POST'])
def predict_cancer_endpoint():
    try:
        model, scaler = models['cancer']
        if model is None or scaler is None:
            return jsonify({"error": "Cancer model not loaded properly"}), 500
            
        data = request.json
        
        # Create DataFrame for cancer prediction
        input_data = pd.DataFrame([{
            'clump_thickness': float(data['clump_thickness']),
            'uniformity_cell_size': float(data['uniformity_cell_size']),
            'uniformity_cell_shape': float(data['uniformity_cell_shape']),
            'marginal_adhesion': float(data['marginal_adhesion']),
            'single_epithelial_cell_size': float(data['single_epithelial_cell_size']),
            'bare_nuclei': float(data['bare_nuclei']),
            'bland_chromatin': float(data['bland_chromatin']),
            'normal_nucleoli': float(data['normal_nucleoli']),
            'mitoses': float(data['mitoses'])
        }])
        
        prediction, probability = predict_cancer(input_data, model, scaler)
        
        prediction = np.atleast_1d(prediction)
        probability = np.atleast_1d(probability)
        
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'percentage': round(float(probability[0] * 100))
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)