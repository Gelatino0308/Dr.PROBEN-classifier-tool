"""
Baseline ACOR (SOCHA-ACOR) for Diabetes Classification - System Training

This script trains the model on 100% of the dataset and saves it as a .pkl file
for UI integration. No testing or evaluation outputs are generated.

Architecture: 8 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
Total weights: 8*6 + 6 + 6*1 + 1 = 61 weights
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler

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
# 2. SOCHA-ACOR ALGORITHM (Baseline)
# ==============================================================================
class SOCHA_ACOR:
    """
    SOCHA's Ant Colony Optimization for Continuous Domains (ACOR)
    """
    
    def __init__(self, obj_func, dim, n_ants=2, n_samples=136, q=0.01, xi=0.95, 
                 max_iter=100, patience=15, seed=42):
        self.obj_func = obj_func
        self.dim = dim
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience
        self.seed = seed
        
        if seed > 0:
            np.random.seed(seed)

    def optimize(self, lb, ub):
        """Main SOCHA-ACOR optimization loop"""
        neighbor_list = np.empty((self.n_samples, self.n_samples - 1), dtype=int)
        
        best_weights = np.full(self.dim, np.nan)
        best_loss = np.inf
        best_iteration = 0
        
        archive_solutions = None
        archive_fitness = []
        
        # Archive Initialization
        for i in range(self.n_samples):
            solution = np.random.uniform(lb, ub, self.dim)
            fitness = self.obj_func(solution)
            
            if archive_solutions is None:
                archive_solutions = solution.reshape(1, -1)
            else:
                archive_solutions = np.vstack([archive_solutions, solution.reshape(1, -1)])
            archive_fitness.append(float(fitness))
            
            if fitness < best_loss:
                best_loss = float(fitness)
                best_weights = solution.copy()

        archive_fitness = np.array(archive_fitness, dtype=float)
        archive_ranks = self._rank_ascending_with_random_ties(archive_fitness)
        
        for i in range(self.n_samples):
            neighbor_list[i] = np.delete(np.arange(self.n_samples), i)

        # Main Optimization Loop
        iteration = 0
        for iteration in range(self.max_iter):
            if np.sum(np.std(archive_solutions, axis=0)) == 0:
                break
            
            new_solutions = self._generate_new_solutions(
                archive_solutions, archive_ranks,
                self.n_ants, self.q, self.n_samples, self.xi
            )

            if new_solutions is None or len(new_solutions) == 0:
                break

            new_fitness = self.obj_func(new_solutions)

            archive_solutions = np.vstack([archive_solutions, new_solutions])
            archive_fitness = np.concatenate([archive_fitness, new_fitness])
            archive_ranks = self._rank_ascending_with_random_ties(archive_fitness)

            keep_indices = archive_ranks <= self.n_samples
            archive_fitness = archive_fitness[keep_indices]
            archive_ranks = archive_ranks[keep_indices]
            archive_solutions = archive_solutions[keep_indices]

            for i in range(self.n_samples):
                neighbor_list[i] = np.delete(np.arange(self.n_samples), i)

            current_best_fitness = np.min(new_fitness)
            if current_best_fitness < best_loss:
                best_loss = float(current_best_fitness)
                best_idx = int(np.argmin(new_fitness))
                best_weights = new_solutions[best_idx].copy()
                best_iteration = iteration

            if iteration - best_iteration > self.patience:
                break

        return best_weights, best_loss, iteration + 1

    def _rank_ascending_with_random_ties(self, values):
        """Rank values in ascending order with random tie-breaking"""
        n = len(values)
        permutation = np.random.permutation(n)
        shuffled_values = values[permutation]
        
        sort_order = np.argsort(shuffled_values, kind='mergesort')
        ranks = np.empty(n, dtype=int)
        ranks[sort_order] = np.arange(1, n + 1)
        
        inverse_permutation = np.empty(n, dtype=int)
        inverse_permutation[permutation] = np.arange(n)
        
        return ranks[inverse_permutation]

    def _generate_new_solutions(self, archive_solutions, archive_ranks, 
                                n_new_solutions, q, k, xi):
        """Generate new solutions using Standard ACOR"""
        num_solutions, num_dimensions = archive_solutions.shape
        new_solutions = np.empty((n_new_solutions, num_dimensions), dtype=float)
        
        ranks = np.arange(1, num_solutions + 1)
        weights = self._gaussian_kernel_pdf(ranks, mean=1.0, std=q * k)
        weights = weights / weights.sum()
        
        for ant_idx in range(n_new_solutions):
            guide_idx = np.random.choice(num_solutions, p=weights)
            guide_solution = archive_solutions[guide_idx]
            
            sigma = np.zeros(num_dimensions)
            for d in range(num_dimensions):
                distances = np.abs(archive_solutions[:, d] - guide_solution[d])
                sigma[d] = xi * np.sum(distances) / (k - 1)
            
            new_solutions[ant_idx] = np.random.normal(loc=guide_solution, scale=sigma)
        
        return new_solutions

    def _gaussian_kernel_pdf(self, x, mean, std):
        """Compute Gaussian probability density function"""
        if std <= 0:
            out = np.zeros_like(x, dtype=float)
            out[np.isclose(x, mean)] = 1.0
            return out
        z = (x - mean) / std
        return np.exp(-0.5 * z * z) / (std * np.sqrt(2.0 * np.pi))


# ==============================================================================
# 3. OBJECTIVE FUNCTION
# ==============================================================================
def objective_function(weights, model, X_train, y_train):
    """Binary Cross-Entropy Loss as fitness function"""
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
    return loss


# ==============================================================================
# 4. TRAINING AND SAVING
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
        if weights.ndim == 1:
            return objective_function(weights, model, X_scaled, y)
        else:
            return np.array([objective_function(w, model, X_scaled, y) for w in weights])
    
    print("\nTraining Baseline ACOR on 100% data...")
    print("=" * 50)
    
    # Initialize and run SOCHA-ACOR
    acor = SOCHA_ACOR(
        obj_func=obj_func,
        dim=num_weights,
        n_ants=2,
        n_samples=136,
        q=0.01,
        xi=0.95,
        max_iter=100,
        patience=15,
        seed=42
    )
    
    best_weights, best_loss, iterations = acor.optimize(lb=-3, ub=3)
    
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
        'algorithm': 'Baseline SOCHA-ACOR'
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'acor_diabetes_model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {output_path}")
    
    return model_data


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE ACOR - DIABETES CLASSIFICATION SYSTEM TRAINING")
    print("=" * 60)
    
    model_data = train_and_save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)