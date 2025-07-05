import numpy as np
import pandas as pd
import pickle
import os

class FNN:
    """
    Feedforward Neural Network (FNN) with a single hidden layer and 1 output neuron.
    Architecture: Input -> Hidden -> Output(1)
    """
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

def load_model(model_path='cancer_model.pkl'):
    """Load the trained cancer classification model."""
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = FNN(model_data['input_dim'], model_data['hidden_dim'], model_data['output_dim'])
    model.set_weights(model_data['weights'])
    return model, model_data['scaler']

def predict_cancer(data, model=None, scaler=None):
    """
    Predict cancer classification for given data.
    
    Parameters:
    data: pandas DataFrame with columns:
        - clump_thickness (1-10)
        - uniformity_cell_size (1-10)
        - uniformity_cell_shape (1-10)
        - marginal_adhesion (1-10)
        - single_epithelial_cell_size (1-10)
        - bare_nuclei (1-10)
        - bland_chromatin (1-10)
        - normal_nucleoli (1-10)
        - mitoses (1-10)
    
    Returns:
    predictions: array of predictions (0 = benign, 1 = malignant)
    probabilities: array of probabilities (probability of malignancy)
    """
    if model is None or scaler is None:
        model, scaler = load_model()
    
    if isinstance(data, pd.DataFrame):
        # Remove class column if present
        if 'class' in data.columns:
            data = data.drop('class', axis=1)
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
    
    # Example: Sample data for breast cancer prediction
    # These are typical values for a benign case
    sample_benign = {
        'clump_thickness': 5,
        'uniformity_cell_size': 1,
        'uniformity_cell_shape': 1,
        'marginal_adhesion': 1,
        'single_epithelial_cell_size': 2,
        'bare_nuclei': 1,
        'bland_chromatin': 3,
        'normal_nucleoli': 1,
        'mitoses': 1
    }
    
    # Example: Sample data for a malignant case
    sample_malignant = {
        'clump_thickness': 8,
        'uniformity_cell_size': 7,
        'uniformity_cell_shape': 5,
        'marginal_adhesion': 10,
        'single_epithelial_cell_size': 7,
        'bare_nuclei': 9,
        'bland_chromatin': 5,
        'normal_nucleoli': 5,
        'mitoses': 4
    }
    
    print("Testing Cancer Classification Model:")
    print("=" * 50)
    
    # Test benign sample
    sample_df_benign = pd.DataFrame([sample_benign])
    pred, prob = predict_cancer(sample_df_benign, model, scaler)
    
    print(f"Benign Sample Test:")
    print(f"Input: {sample_benign}")
    # Fix: Handle both scalar and array cases like in predict_diabetes.py
    if np.isscalar(pred):
        print(f"Prediction: {'Malignant' if pred == 1 else 'Benign'}")
        print(f"Probability of malignancy: {prob:.4f}")
    else:
        print(f"Prediction: {'Malignant' if pred[0] == 1 else 'Benign'}")
        print(f"Probability of malignancy: {prob[0]:.4f}")
    print()
    
    # Test malignant sample
    sample_df_malignant = pd.DataFrame([sample_malignant])
    pred, prob = predict_cancer(sample_df_malignant, model, scaler)
    
    print(f"Malignant Sample Test:")
    print(f"Input: {sample_malignant}")
    # Fix: Handle both scalar and array cases like in predict_diabetes.py
    if np.isscalar(pred):
        print(f"Prediction: {'Malignant' if pred == 1 else 'Benign'}")
        print(f"Probability of malignancy: {prob:.4f}")
    else:
        print(f"Prediction: {'Malignant' if pred[0] == 1 else 'Benign'}")
        print(f"Probability of malignancy: {prob[0]:.4f}")