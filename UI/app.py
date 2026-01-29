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
MODELS_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'tool-models', 'ACOR_Final-Def')

class ACORNeuralNetwork:
    """Wrapper class to reconstruct ACOR neural network from saved weights."""
    
    def __init__(self, weights, architecture, scaler=None):
        self.weights = weights
        self.architecture = architecture
        self.scaler = scaler
        
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
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
        
        # Hidden layer with ReLU (matching FNN architecture)
        z1 = np.dot(activations, W1) + b1
        a1 = self._relu(z1)  # Changed from sigmoid to ReLU
        
        # Output layer with Sigmoid
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
    model_path = os.path.join(MODELS_BASE_PATH, disease_name, f'acor_lm_{disease_name}_model.pkl')
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

# Helper function for diabetes: use preprocessor to normalize raw values to 0-1 range
def prepare_diabetes_input(raw_values):
    """
    Convert raw diabetes values to normalized array (0-1 scale).
    Uses DiabetesPreprocessor.preprocess_single() for consistency with batch processing.
    
    Raw value ranges (from Pima Indians dataset):
    - Pregnancies: 0-17
    - Glucose: 0-199
    - BloodPressure: 0-122
    - SkinThickness: 0-99
    - Insulin: 0-846
    - BMI: 0-67.1
    - DiabetesPedigreeFunction: 0.078-2.42
    - Age: 21-81
    """
    preprocessor = preprocessors['diabetes']
    return preprocessor.preprocess_single(raw_values).reshape(1, -1)

# Helper function for heart: expand 13 to 35
def prepare_heart_input(raw_values):
    """Expand 13 raw heart features to 35 features"""
    from preprocess import expand_heart_features
    expanded = expand_heart_features(raw_values)
    return expanded.reshape(1, -1)

# Helper function for cancer: use preprocessor to normalize 1-10 values to 0-1 range
def prepare_cancer_input(raw_values):
    """
    Convert raw cancer values (1-10 scale) to normalized array (0-1 scale).
    Uses CancerPreprocessor.preprocess_single() for consistency with batch processing.
    """
    preprocessor = preprocessors['cancer']
    return preprocessor.preprocess_single(raw_values).reshape(1, -1)

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
            float(data['Number of Pregnancies']),
            float(data['Plasma Glucose Concentration']),
            float(data['Diastolic Blood Pressure']),
            float(data['Triceps Skin Fold Thickness']),
            float(data['2-Hour Serum Insulin']),
            float(data['Body Mass Index']),
            float(data['Diabetes Pedigree Function']),
            float(data['Age'])
        ]
        
        # Prepare input
        X = prepare_diabetes_input(raw_values)
        
        # Get probability for class 1 (which is actually NON-DIABETIC in preprocessed data)
        if hasattr(model, 'predict_proba'):
            prob_class_1 = model.predict_proba(X)[0][1]
        else:
            prediction = model.predict(X)[0]
            prob_class_1 = float(prediction)
        
        # Determine prediction based on probability
        # If prob_class_1 > 0.5, predict DIABETIC (1), else NON-DIABETIC (0)
        final_prediction = 1 if prob_class_1 > 0.5 else 0
        
        # Calculate percentage for the predicted class (always >= 50%)
        if final_prediction == 1:
            # Predicted DIABETIC
            final_percentage = round(prob_class_1 * 100)
        else:
            # Predicted NON-DIABETIC
            final_percentage = round((1 - prob_class_1) * 100)
        
        result = {
            'prediction': final_prediction,  # 0 = NON-DIABETIC, 1 = DIABETIC (corrected)
            'probability': float(prob_class_1),  # Corrected probability of DIABETIC
            'percentage': final_percentage  # Percentage of predicted class (>= 50)
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
            float(data['Age']),
            float(data['Sex']),
            float(data['Chest Pain Type']),
            float(data['Resting Blood Pressure']),
            float(data['Serum Cholesterol']),
            float(data['FBS > 120mg/dL']),
            float(data['Resting ECG Results']),
            float(data['Maximum Heart Rate']),
            float(data['Exercise Induced Angina']),
            float(data['ST Depression (Oldpeak)']),
            float(data['Slope of Peak Exercise ST']),
            float(data['Number of Major Vessels']),
            float(data['Thalassemia'])
        ]
        
        # Expand to 35 features
        X = prepare_heart_input(raw_values)
        
        # Get probability for class 1 (positive/heart disease)
        if hasattr(model, 'predict_proba'):
            prob_class_1 = model.predict_proba(X)[0][1]
        else:
            prediction = model.predict(X)[0]
            prob_class_1 = float(prediction)
        
        # Determine prediction based on probability
        final_prediction = 1 if prob_class_1 > 0.5 else 0
        
        # Calculate percentage for the predicted class (always >= 50%)
        if final_prediction == 1:
            final_percentage = round(prob_class_1 * 100)
        else:
            final_percentage = round((1 - prob_class_1) * 100)
        
        result = {
            'prediction': final_prediction,
            'probability': float(prob_class_1),
            'percentage': final_percentage
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
            float(data['Clump Thickness']),
            float(data['Uniformity of Cell Size']),
            float(data['Uniformity of Cell Shape']),
            float(data['Marginal Adhesion']),
            float(data['Single Epithelial Cell Size']),
            float(data['Bare Nuclei']),
            float(data['Bland Chromatin']),
            float(data['Normal Nucleoli']),
            float(data['Mitoses'])
        ]
        
        # Prepare input
        X = prepare_cancer_input(raw_values)
        
        # Get probability for class 1 (positive/malignant)
        if hasattr(model, 'predict_proba'):
            prob_class_1 = model.predict_proba(X)[0][1]
        else:
            prediction = model.predict(X)[0]
            prob_class_1 = float(prediction)
        
        # Determine prediction based on probability
        final_prediction = 1 if prob_class_1 > 0.5 else 0
        
        # Calculate percentage for the predicted class (always >= 50%)
        if final_prediction == 1:
            final_percentage = round(prob_class_1 * 100)
        else:
            final_percentage = round((1 - prob_class_1) * 100)
        
        result = {
            'prediction': final_prediction,
            'percentage': final_percentage
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
        preprocessor = preprocessors['diabetes']
        
        if model is None:
            return jsonify({"error": "Diabetes model not loaded properly"}), 500
            
        data = request.json
        batch_data = data['data']
        
        # Prepare raw batch input
        raw_batch = []
        for row in batch_data:
            raw_batch.append([
                float(row['Number of Pregnancies']),
                float(row['Plasma Glucose Concentration']),
                float(row['Diastolic Blood Pressure']),
                float(row['Triceps Skin Fold Thickness']),
                float(row['2-Hour Serum Insulin']),
                float(row['Body Mass Index']),
                float(row['Diabetes Pedigree Function']),
                float(row['Age'])
            ])
        
        # Convert to numpy array
        raw_array = np.array(raw_batch, dtype=float)
        
        # Use preprocessor to normalize (handles raw -> 0-1 scaling)
        X, _, msgs = preprocessor.preprocess_batch_matrix(raw_array)
        
        # Get probabilities for class 1 (which is actually NON-DIABETIC in preprocessed data)
        if hasattr(model, 'predict_proba'):
            prob_class_1_array = model.predict_proba(X)[:, 1]  # Prob of class 1
        else:
            raw_predictions = model.predict(X)
            prob_class_1_array = raw_predictions.astype(float)
    
        # Determine predictions based on probability
        corrected_predictions = [1 if prob > 0.5 else 0 for prob in prob_class_1_array]
        
        # Calculate percentages for predicted class (always >= 50%)
        corrected_percentages = []
        for i, prediction in enumerate(corrected_predictions):
            prob_class_1 = prob_class_1_array[i]
            if prediction == 1:
                # Predicted DIABETIC
                percentage = round(prob_class_1 * 100)
            else:
                # Predicted NON-DIABETIC
                percentage = round((1 - prob_class_1) * 100)
            corrected_percentages.append(percentage)
        
        result = {
            'predictions': corrected_predictions,  # Corrected predictions (0 or 1)
            'probabilities': corrected_percentages,  # Corrected percentages (always >= 50)
            'messages': msgs + ['Batch prediction completed successfully (labels corrected)']
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
                float(row['Age']),
                float(row['Sex']),
                float(row['Chest Pain Type']),
                float(row['Resting Blood Pressure']),
                float(row['Serum Cholesterol']),
                float(row['FBS > 120mg/dL']),
                float(row['Resting ECG Results']),
                float(row['Maximum Heart Rate']),
                float(row['Exercise Induced Angina']),
                float(row['ST Depression (Oldpeak)']),
                float(row['Slope of Peak Exercise ST']),
                float(row['Number of Major Vessels']),
                float(row['Thalassemia'])
            ]
            expanded = expand_heart_features(raw_values)
            expanded_batch.append(expanded)
        
        # Convert to numpy array
        X = np.array(expanded_batch, dtype=float)
        
        # Get probabilities for class 1 (POSITIVE/HEART DISEASE)
        if hasattr(model, 'predict_proba'):
            prob_class_1_array = model.predict_proba(X)[:, 1]
        else:
            raw_predictions = model.predict(X)
            prob_class_1_array = raw_predictions.astype(float)
        
        # Determine predictions based on probability (threshold 0.5)
        predictions = [1 if prob > 0.5 else 0 for prob in prob_class_1_array]
        
        # Calculate percentages for predicted class (always >= 50%)
        percentages = []
        for i, prediction in enumerate(predictions):
            prob_class_1 = prob_class_1_array[i]
            if prediction == 1:
                # Predicted POSITIVE (HEART DISEASE)
                percentage = round(prob_class_1 * 100)
            else:
                # Predicted NEGATIVE (NO HEART DISEASE)
                percentage = round((1 - prob_class_1) * 100)
            percentages.append(percentage)
        
        result = {
            'predictions': predictions,
            'probabilities': percentages,  # Now returns percentages >= 50
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
        preprocessor = preprocessors['cancer']
        
        if model is None:
            return jsonify({"error": "Cancer model not loaded properly"}), 500
            
        data = request.json
        batch_data = data['data']
        
        # Prepare raw batch input
        raw_batch = []
        for row in batch_data:
            raw_values = [
                float(row['Clump Thickness']),
                float(row['Uniformity of Cell Size']),
                float(row['Uniformity of Cell Shape']),
                float(row['Marginal Adhesion']),
                float(row['Single Epithelial Cell Size']),
                float(row['Bare Nuclei']),
                float(row['Bland Chromatin']),
                float(row['Normal Nucleoli']),
                float(row['Mitoses'])
            ]
            raw_batch.append(raw_values)
        
        # Convert to numpy array
        raw_array = np.array(raw_batch, dtype=float)
        
        # Use preprocessor to normalize (handles 1-10 -> 0-1 scaling)
        X, _, msgs = preprocessor.preprocess_batch_matrix(raw_array)
        
        # Get probabilities for class 1 (MALIGNANT)
        if hasattr(model, 'predict_proba'):
            prob_class_1_array = model.predict_proba(X)[:, 1]
        else:
            raw_predictions = model.predict(X)
            prob_class_1_array = raw_predictions.astype(float)
        
        # Determine predictions based on probability (threshold 0.5)
        predictions = [1 if prob > 0.5 else 0 for prob in prob_class_1_array]
        
        # Calculate percentages for predicted class (always >= 50%)
        percentages = []
        for i, prediction in enumerate(predictions):
            prob_class_1 = prob_class_1_array[i]
            if prediction == 1:
                # Predicted MALIGNANT
                percentage = round(prob_class_1 * 100)
            else:
                # Predicted BENIGN
                percentage = round((1 - prob_class_1) * 100)
            percentages.append(percentage)
        
        result = {
            'predictions': predictions,
            'probabilities': percentages,  # Now returns percentages >= 50
            'messages': msgs + ['Batch prediction completed successfully']
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