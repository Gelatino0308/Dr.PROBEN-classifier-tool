from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np
import pickle
import traceback

# Import preprocessing modules
from preprocess import get_preprocessor, DiabetesPreprocessor, HeartPreprocessor, CancerPreprocessor

app = Flask(__name__)
CORS(app)

# Define paths to new models
MODELS_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'tool-models', 'ACOR-with-reportgen')

class ACORNeuralNetwork:
    """Wrapper class to reconstruct ACOR neural network from saved weights."""
    
    def __init__(self, weights, architecture, scaler=None):
        self.weights = weights
        self.architecture = architecture
        self.scaler = scaler
        
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward_pass(self, X):
        """Perform forward pass through the network."""
        activations = X
        
        # Get layer sizes from architecture
        if isinstance(self.architecture, dict):
            hidden_size = self.architecture.get('hidden', 
                          self.architecture.get('hidden_size', 
                          self.architecture.get('hidden_layer_size', 10)))
            output_size = self.architecture.get('output', 
                          self.architecture.get('output_size', 1))
            input_size = self.architecture.get('input', 
                         self.architecture.get('input_size', X.shape[1]))
        else:
            hidden_size = 10
            output_size = 1
            input_size = X.shape[1]
        
        # Validate input size matches
        if X.shape[1] != input_size:
            raise ValueError(f"Input size mismatch: expected {input_size}, got {X.shape[1]}")
        
        # Calculate weight matrix sizes
        w1_size = input_size * hidden_size
        b1_size = hidden_size
        w2_size = hidden_size * output_size
        b2_size = output_size
        
        # Extract weights and biases
        W1 = self.weights[:w1_size].reshape(input_size, hidden_size)
        b1 = self.weights[w1_size:w1_size + b1_size]
        W2 = self.weights[w1_size + b1_size:w1_size + b1_size + w2_size].reshape(hidden_size, output_size)
        b2 = self.weights[w1_size + b1_size + w2_size:w1_size + b1_size + w2_size + b2_size]
        
        # Hidden layer
        z1 = np.dot(activations, W1) + b1
        a1 = self._sigmoid(z1)
        
        # Output layer
        z2 = np.dot(a1, W2) + b2
        a2 = self._sigmoid(z2)
        
        return a2
    
    def predict(self, X):
        """Predict class labels."""
        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self._forward_pass(X)
        
        # Convert to binary predictions
        predictions = (probabilities.flatten() > 0.5).astype(int)
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self._forward_pass(X)
        
        # Return as [prob_class_0, prob_class_1]
        prob_class_1 = probabilities.flatten()
        prob_class_0 = 1 - prob_class_1
        return np.column_stack([prob_class_0, prob_class_1])

def load_disease_model(disease_name):
    """Load the ACOR model for the specified disease."""
    model_path = os.path.join(MODELS_BASE_PATH, disease_name, f'{disease_name}_acor_lm_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Check if this is an ACOR model with weights
        if isinstance(loaded_data, dict) and 'best_model_weights' in loaded_data:
            print(f"Reconstructing ACOR neural network for {disease_name}...")
            
            # Extract components
            weights = loaded_data['best_model_weights']
            architecture = loaded_data.get('architecture', {})
            scaler = loaded_data.get('scaler', None)
            
            # Create model wrapper
            model = ACORNeuralNetwork(weights, architecture, scaler)
            
            print(f"✓ {disease_name.capitalize()} ACOR model reconstructed successfully")
            print(f"  - Weights shape: {weights.shape}")
            print(f"  - Architecture: {architecture}")
            print(f"  - Scaler: {'Yes' if scaler is not None else 'No'}")
            
            return model
        
        # Fallback: try to find a predict method in the dict
        elif isinstance(loaded_data, dict):
            for key in ['model', 'classifier', 'best_model', 'estimator']:
                if key in loaded_data and hasattr(loaded_data[key], 'predict'):
                    print(f"Using model from key: '{key}'")
                    return loaded_data[key]
            
            for key, value in loaded_data.items():
                if hasattr(value, 'predict'):
                    print(f"Using model from key: '{key}'")
                    return value
            
            raise ValueError(f"Could not find model object in dictionary. Keys: {loaded_data.keys()}")
        
        else:
            if not hasattr(loaded_data, 'predict'):
                raise ValueError(f"Loaded object does not have 'predict' method. Type: {type(loaded_data)}")
            return loaded_data
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading {disease_name} model: {e}")
        traceback.print_exc()
        return None

# Load all models
models = {}
preprocessors = {}

print("Loading models and preprocessors...")
for disease in ['diabetes', 'heart', 'cancer']:
    print(f"\nLoading {disease} model...")
    models[disease] = load_disease_model(disease)
    preprocessors[disease] = get_preprocessor(disease)
    
    if models[disease] is None:
        print(f"WARNING: {disease} model failed to load!")
    else:
        print(f"✓ {disease} model loaded successfully")

# Helper function for diabetes: raw to unscaled
def prepare_diabetes_input(raw_values):
    """Convert raw diabetes values to array without scaling (model scaler will handle it)"""
    return np.array(raw_values, dtype=float).reshape(1, -1)

# Helper function for heart: expand 13 to 35
def prepare_heart_input(raw_values):
    """Expand 13 raw heart features to 35 features"""
    from preprocess import expand_heart_features
    expanded = expand_heart_features(raw_values)
    return expanded.reshape(1, -1)

# Helper function for cancer: raw to unscaled
def prepare_cancer_input(raw_values):
    """Convert raw cancer values to array without scaling (model scaler will handle it)"""
    return np.array(raw_values, dtype=float).reshape(1, -1)

# Single prediction endpoints
@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes_endpoint():
    try:
        model = models['diabetes']
        
        if model is None:
            return jsonify({"error": "Diabetes model not loaded properly"}), 500
            
        data = request.json
        
        # Extract values in correct order for diabetes
        raw_values = [
            float(data['pregnancies']),
            float(data['plasma']),      # Glucose
            float(data['BP']),           # BloodPressure
            float(data['skin']),         # SkinThickness
            float(data['insulin']),
            float(data['BMI']),
            float(data['pedigree']),     # DiabetesPedigreeFunction
            float(data['age'])
        ]
        
        # Prepare input (no preprocessing - model scaler handles it)
        X = prepare_diabetes_input(raw_values)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0][1]
        else:
            probability = float(prediction)
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'percentage': round(float(probability * 100))
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Diabetes prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/heart', methods=['POST'])
def predict_heart_endpoint():
    try:
        model = models['heart']
        
        if model is None:
            return jsonify({"error": "Heart model not loaded properly"}), 500
            
        data = request.json
        
        # Extract 13 raw heart features
        raw_values = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]
        
        # Expand to 35 features
        X = prepare_heart_input(raw_values)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0][1]
        else:
            probability = float(prediction)
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'percentage': round(float(probability * 100))
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Heart prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/cancer', methods=['POST'])
def predict_cancer_endpoint():
    try:
        model = models['cancer']
        
        if model is None:
            return jsonify({"error": "Cancer model not loaded properly"}), 500
            
        data = request.json
        
        # Extract 9 cancer features
        raw_values = [
            float(data['clump_thickness']),
            float(data['uniformity_cell_size']),
            float(data['uniformity_cell_shape']),
            float(data['marginal_adhesion']),
            float(data['single_epithelial_cell_size']),
            float(data['bare_nuclei']),
            float(data['bland_chromatin']),
            float(data['normal_nucleoli']),
            float(data['mitoses'])
        ]
        
        # Prepare input (no preprocessing - model scaler handles it)
        X = prepare_cancer_input(raw_values)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0][1]
        else:
            probability = float(prediction)
        
        result = {
            'prediction': int(prediction),
            'percentage': round(float(probability * 100))
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Cancer prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Batch prediction endpoints
@app.route('/api/predict/diabetes/batch', methods=['POST'])
def predict_diabetes_batch_endpoint():
    try:
        model = models['diabetes']
        
        if model is None:
            return jsonify({"error": "Diabetes model not loaded properly"}), 500
            
        data = request.json
        batch_data = data['data']
        
        # Prepare batch input (no preprocessing scaling)
        raw_batch = []
        for row in batch_data:
            raw_batch.append([
                float(row['pregnancies']),
                float(row['plasma']),
                float(row['BP']),
                float(row['skin']),
                float(row['insulin']),
                float(row['BMI']),
                float(row['pedigree']),
                float(row['age'])
            ])
        
        # Convert to numpy array (model scaler will handle scaling)
        X = np.array(raw_batch, dtype=float)
        
        # Make predictions
        predictions = model.predict(X).tolist()
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1].tolist()
        else:
            probabilities = [float(p) for p in predictions]
        
        result = {
            'predictions': [int(p) for p in predictions],
            'probabilities': probabilities,
            'messages': ['Batch prediction completed successfully']
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Batch diabetes prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/heart/batch', methods=['POST'])
def predict_heart_batch_endpoint():
    try:
        model = models['heart']
        from preprocess import expand_heart_features
        
        if model is None:
            return jsonify({"error": "Heart model not loaded properly"}), 500
            
        data = request.json
        batch_data = data['data']
        
        # Prepare batch input (13 raw features per row, expand to 35)
        expanded_batch = []
        for row in batch_data:
            raw_values = [
                float(row['age']),
                float(row['sex']),
                float(row['cp']),
                float(row['trestbps']),
                float(row['chol']),
                float(row['fbs']),
                float(row['restecg']),
                float(row['thalach']),
                float(row['exang']),
                float(row['oldpeak']),
                float(row['slope']),
                float(row['ca']),
                float(row['thal'])
            ]
            expanded = expand_heart_features(raw_values)
            expanded_batch.append(expanded)
        
        # Convert to numpy array
        X = np.array(expanded_batch, dtype=float)
        
        # Make predictions
        predictions = model.predict(X).tolist()
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1].tolist()
        else:
            probabilities = [float(p) for p in predictions]
        
        result = {
            'predictions': [int(p) for p in predictions],
            'probabilities': probabilities,
            'messages': ['Batch prediction completed successfully (13 features expanded to 35)']
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Batch heart prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/cancer/batch', methods=['POST'])
def predict_cancer_batch_endpoint():
    try:
        model = models['cancer']
        
        if model is None:
            return jsonify({"error": "Cancer model not loaded properly"}), 500
            
        data = request.json
        batch_data = data['data']
        
        # Prepare batch input (no preprocessing scaling)
        raw_batch = []
        for row in batch_data:
            raw_batch.append([
                float(row['clump_thickness']),
                float(row['uniformity_cell_size']),
                float(row['uniformity_cell_shape']),
                float(row['marginal_adhesion']),
                float(row['single_epithelial_cell_size']),
                float(row['bare_nuclei']),
                float(row['bland_chromatin']),
                float(row['normal_nucleoli']),
                float(row['mitoses'])
            ])
        
        # Convert to numpy array (model scaler will handle scaling)
        X = np.array(raw_batch, dtype=float)
        
        # Make predictions
        predictions = model.predict(X).tolist()
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1].tolist()
        else:
            probabilities = [float(p) for p in predictions]
        
        result = {
            'predictions': [int(p) for p in predictions],
            'probabilities': probabilities,
            'messages': ['Batch prediction completed successfully']
        }
        
        return jsonify(result)
        
    except Exception as e:
        print("Batch cancer prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "diabetes": models['diabetes'] is not None,
            "heart": models['heart'] is not None,
            "cancer": models['cancer'] is not None
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)