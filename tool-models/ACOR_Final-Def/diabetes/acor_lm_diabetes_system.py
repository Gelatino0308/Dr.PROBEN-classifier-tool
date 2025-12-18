"""
ACOR-LM with Multiple Colonies for Diabetes Classification - System Training

This script trains the model on 100% of the dataset and saves it as a .pkl file
for UI integration. No testing or evaluation outputs are generated.

Architecture: 8 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
Total weights: 8*6 + 6 + 6*1 + 1 = 61 weights
"""

import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import lm_local_search
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lm_local_search import MultipleColonyACOR

# Set random seed for reproducibility
np.random.seed(42)


# ==============================================================================
# 1. FEEDFORWARD NEURAL NETWORK (FNN)
# ==============================================================================
class FNN:
    """
    Feedforward Neural Network matching thesis specifications.
    
    Architecture: Input(8) -> Hidden(6, ReLU) -> Output(1, Sigmoid)
    Total weights: 8*6 + 6 + 6*1 + 1 = 61 weights
    """
    def __init__(self, input_dim=8, hidden_dim=6, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def set_weights(self, weights):
        """Unpack flat weight vector into layer matrices"""
        idx = 0
        self.W1 = weights[idx:idx+self.input_dim*self.hidden_dim].reshape(self.input_dim, self.hidden_dim)
        idx += self.input_dim*self.hidden_dim
        self.b1 = weights[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        self.W2 = weights[idx:idx+self.hidden_dim*self.output_dim].reshape(self.hidden_dim, self.output_dim)
        idx += self.hidden_dim*self.output_dim
        self.b2 = weights[idx:idx+self.output_dim]
    
    def _stable_sigmoid(self, z):
        """Numerically stable sigmoid to prevent overflow"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """Forward pass: Input -> ReLU -> Sigmoid"""
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = a1 @ self.W2 + self.b2
        a2 = self._stable_sigmoid(z2)  # Sigmoid activation
        return a2.squeeze()

    def predict(self, X):
        """Binary classification with 0.5 threshold"""
        return (self.forward(X) > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim=8, hidden_dim=6, output_dim=1):
        """Calculate total number of weights in the network"""
        return input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim


# ==============================================================================
# 2. OBJECTIVE FUNCTION
# ==============================================================================
def objective_function(weights, model, X_train, y_train):
    """Binary Cross-Entropy Loss as fitness function"""
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
    return loss


# ==============================================================================
# 3. TRAINING AND SAVING
# ==============================================================================
def train_and_save_model():
    """Train on 100% data and save model as .pkl file"""
    
    # Load data
    data = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'diabetes1.dat'),
        sep=' ',
        header=None
    )

    X = data.iloc[:, :-2].values
    y_onehot = data.iloc[:, -2:].values
    y = np.argmax(y_onehot, axis=1)

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize model
    input_dim = 8
    hidden_dim = 6
    output_dim = 1
    num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
    
    model = FNN(input_dim, hidden_dim, output_dim)
    
    # Create objective function wrapper
    def obj_func(weights):
        return objective_function(weights, model, X_scaled, y)
    
    print("\nTraining Hybrid ACOR-LM on 100% data...")
    print("=" * 50)
    
    # Initialize and run ACOR-LM
    acor_lm = MultipleColonyACOR(
        obj_func=obj_func,
        dim=num_weights,
        n_colonies=5,
        n_ants=2,
        n_samples=136,
        q=0.01,
        xi=0.95,
        max_iter=100,
        patience=15,
        local_patience=3,
        sharing_frequency=5,
        sharing_ratio=0.15,
        seed=42
    )
    
    best_weights, best_loss, iterations, _ = acor_lm.optimize(
        lb=-3, ub=3, model=model, X_train=X_scaled, y_train=y,
        loss_threshold=None  # No threshold tracking needed for system training
    )
    
    # Set final weights to model
    model.set_weights(best_weights)
    
    print(f"Training completed in {iterations} iterations")
    print(f"Final loss: {best_loss:.6f}")
    
    # Save ONLY weights and metadata (NOT the model object)
    model_data = {
        'best_model_weights': best_weights,
        'scaler': scaler,
        'architecture': {
            'input': input_dim,
            'hidden': hidden_dim,
            'output': output_dim
        },
        'final_loss': best_loss,
        'iterations': iterations,
        'algorithm': 'Hybrid ACOR-LM'
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'acor_lm_diabetes_model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {output_path}")
    
    return model_data


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID ACOR-LM - DIABETES CLASSIFICATION SYSTEM TRAINING")
    print("=" * 60)
    
    model_data = train_and_save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)