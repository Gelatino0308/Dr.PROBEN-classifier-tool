import pickle
import numpy as np

# Load the model
model_path = r"c:\Users\Angelo\Documents\CS 4th Year Files\Thesis Writing 2\Dr.PROBEN-classifier-tool\tool-models\ACOR_Final-Def\cancer\acor_lm_cancer_model.pkl"

with open(model_path, 'rb') as f:
    data = pickle.load(f)

scaler = data['scaler']
weights = data['best_model_weights']
arch = data['architecture']

# Extract network parameters
input_dim = arch['input']
hidden_dim = arch['hidden']
output_dim = arch['output']

idx = 0
W1 = weights[idx:idx + input_dim*hidden_dim].reshape(input_dim, hidden_dim)
idx += input_dim*hidden_dim
b1 = weights[idx:idx + hidden_dim]
idx += hidden_dim
W2 = weights[idx:idx + hidden_dim*output_dim].reshape(hidden_dim, output_dim)
idx += hidden_dim*output_dim
b2 = weights[idx:idx + output_dim]

def predict_with_normalization(raw_values):
    """Predict with proper normalization (1-10 -> 0-1 -> StandardScaler)"""
    # Step 1: Normalize from 1-10 to 0-1 (PROBEN1 preprocessing)
    normalized = [(v - 1) / 9.0 for v in raw_values]
    X = np.array([normalized], dtype=float)
    
    # Step 2: Apply StandardScaler (fit on normalized data)
    X_scaled = scaler.transform(X)
    
    # Step 3: Forward pass
    z1 = np.dot(X_scaled, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = np.dot(a1, W2) + b2
    prob = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))
    
    return prob[0][0]

print("Testing with PROPER normalization (1-10 -> 0-1 -> StandardScaler):")
print("="*60)

test_cases = [
    ([1, 1, 1, 1, 1, 1, 1, 1, 1], "All 1s (minimum - likely benign)"),
    ([10, 10, 10, 10, 10, 10, 10, 10, 10], "All 10s (maximum - likely malignant)"),
    ([5, 5, 5, 5, 5, 5, 5, 5, 5], "All 5s (middle values)"),
    ([1, 1, 1, 1, 2, 1, 3, 1, 1], "Mostly benign pattern"),
    ([8, 10, 10, 8, 7, 10, 9, 7, 1], "Malignant pattern"),
]

for raw, description in test_cases:
    prob = predict_with_normalization(raw)
    pred = int(prob > 0.5)
    label = "Malignant" if pred == 1 else "Benign"
    print(f"{description}")
    print(f"  Input: {raw}")
    print(f"  Probability: {prob:.4f} -> {label}")
    print()