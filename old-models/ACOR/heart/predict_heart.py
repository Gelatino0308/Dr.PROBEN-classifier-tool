import numpy as np
import pandas as pd
import pickle
import os

class FNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def set_weights(self, weights):
        idx = 0
        self.W1 = weights[idx:idx+self.input_dim*self.hidden_dim].reshape(self.input_dim, self.hidden_dim)
        idx += self.input_dim*self.hidden_dim
        self.b1 = weights[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        self.W2 = weights[idx:idx+self.hidden_dim*self.output_dim].reshape(self.hidden_dim, self.output_dim)
        idx += self.hidden_dim*self.output_dim
        self.b2 = weights[idx:idx+self.output_dim]

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = 1 / (1 + np.exp(-z2))
        return a2.squeeze()

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

def load_model(model_path='heart_model.pkl'):
    """Load the trained heart disease classification model."""
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = FNN(model_data['input_dim'], model_data['hidden_dim'], model_data['output_dim'])
    model.set_weights(model_data['weights'])
    return model, model_data['scaler']

def predict_heart(data, model=None, scaler=None):
    """
    Predict heart disease classification for given data.
    
    Parameters:
    data: pandas DataFrame with columns:
        - age: age in years
        - sex: sex (1 = male; 0 = female)
        - cp: chest pain type (0-3)
        - trestbps: resting blood pressure
        - chol: serum cholesterol in mg/dl
        - fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        - restecg: resting electrocardiographic results (0-2)
        - thalach: maximum heart rate achieved
        - exang: exercise induced angina (1 = yes; 0 = no)
        - oldpeak: ST depression induced by exercise relative to rest
        - slope: slope of the peak exercise ST segment (0-2)
        - ca: number of major vessels (0-3) colored by flourosopy
        - thal: thalassemia (1-3)
    
    Returns:
    predictions: array of predictions (0 = no heart disease, 1 = heart disease)
    probabilities: array of probabilities (probability of heart disease)
    """
    if model is None or scaler is None:
        model, scaler = load_model()
    
    if isinstance(data, pd.DataFrame):
        # Remove target column if present
        if 'target' in data.columns:
            data = data.drop('target', axis=1)
        X = data.values
    else:
        X = data
    
    # Standardize the input data
    X_scaled = scaler.transform(X)
    
    # Get probabilities and predictions
    probabilities = model.forward(X_scaled)
    predictions = (probabilities > 0.5).astype(int)
    
    return predictions, probabilities

if __name__ == "__main__":
    # Example usage
    model, scaler = load_model()
    
    # Example: Sample data for heart disease prediction
    # These are typical values for a no heart disease case
    sample_no_disease = {
        'age': 45,
        'sex': 0,
        'cp': 0,
        'trestbps': 120,
        'chol': 200,
        'fbs': 0,
        'restecg': 0,
        'thalach': 170,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 2,
        'ca': 0,
        'thal': 2
    }
    
    # Example: Sample data for a heart disease case
    sample_heart_disease = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    print("Testing Heart Disease Classification Model:")
    print("=" * 50)
    
    # Test no heart disease sample
    sample_df_no_disease = pd.DataFrame([sample_no_disease])
    pred, prob = predict_heart(sample_df_no_disease, model, scaler)
    
    print(f"No Heart Disease Sample Test:")
    print(f"Input: {sample_no_disease}")
    # Handle both scalar and array cases
    if np.isscalar(pred):
        print(f"Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
        print(f"Probability of heart disease: {prob:.4f}")
    else:
        print(f"Prediction: {'Heart Disease' if pred[0] == 1 else 'No Heart Disease'}")
        print(f"Probability of heart disease: {prob[0]:.4f}")
    print()
    
    # Test heart disease sample
    sample_df_heart_disease = pd.DataFrame([sample_heart_disease])
    pred, prob = predict_heart(sample_df_heart_disease, model, scaler)
    
    print(f"Heart Disease Sample Test:")
    print(f"Input: {sample_heart_disease}")
    # Handle both scalar and array cases
    if np.isscalar(pred):
        print(f"Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
        print(f"Probability of heart disease: {prob:.4f}")
    else:
        print(f"Prediction: {'Heart Disease' if pred[0] == 1 else 'No Heart Disease'}")
        print(f"Probability of heart disease: {prob[0]:.4f}")