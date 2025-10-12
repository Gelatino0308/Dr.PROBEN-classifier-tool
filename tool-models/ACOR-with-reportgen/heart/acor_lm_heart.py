"""
ACOR-LM with Multiple Colonies for Heart Disease Classification

This implementation follows the thesis specifications:
- Multiple Colony ACOR with Levenberg-Marquardt local search
- FNN Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)
- Binary Cross-Entropy Loss
- Single train-test split (80-20)
- 50 independent runs
- Uses preprocessed heart1.dat with 35 features
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import pickle
import datetime

# Add parent directory to path to import lm_local_search
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lm_local_search import MultipleColonyACOR, LevenbergMarquardt

# --- CHANGE: Import the function from your new reporting file ---
from reportgen_heart_ACORLM import generate_summary_chart

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the data from heart1.dat
# --------------------------------------------------
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'heart1.dat'),
    sep=' ',
    header=None
)
X = data.iloc[:, :-2].values
y_onehot = data.iloc[:, -2:].values
y = np.argmax(y_onehot, axis=1)

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")


# 2. Define FNN matching thesis specifications (35, 6, 1)
# --------------------------------------------------
class FNN_Thesis:
    def __init__(self, input_dim=35, hidden_dim=6, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.total_weights = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

    def set_weights(self, weights):
        if len(weights) != self.total_weights:
            raise ValueError(f"Expected {self.total_weights} weights, got {len(weights)}")
        idx = 0
        self.W1 = weights[idx:idx + self.input_dim * self.hidden_dim].reshape(self.input_dim, self.hidden_dim)
        idx += self.input_dim * self.hidden_dim
        self.b1 = weights[idx:idx + self.hidden_dim]
        idx += self.hidden_dim
        self.W2 = weights[idx:idx + self.hidden_dim * self.output_dim].reshape(self.hidden_dim, self.output_dim)
        idx += self.hidden_dim * self.output_dim
        self.b2 = weights[idx:idx + self.output_dim]

    def _stable_sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._stable_sigmoid(z2)
        return a2.squeeze()

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim=35, hidden_dim=6, output_dim=1):
        return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

# 3. Objective function for ACOR-LM (Binary Cross-Entropy Loss)
# --------------------------------------------------
def objective_function(weights, model, X_train, y_train):
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    return -np.mean(y_train * np.log(y_pred + eps) + (1 - y_train) * np.log(1 - y_pred + eps))

# 4. Multiple runs evaluation
# --------------------------------------------------
def evaluate_acor_lm(X_train, X_test, y_train, y_test, n_runs=50):
    results = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
        'confusion_matrices': [], 'best_losses': [], 'iterations': []
    }
    best_overall_accuracy = -1
    best_model_weights = None
    best_run_idx = -1
    
    print(f"\nRunning {n_runs} independent experiments...")
    print("=" * 60)
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}", end=" ")
        model = FNN_Thesis(input_dim=35, hidden_dim=6, output_dim=1)
        num_weights = FNN_Thesis.get_num_weights(35, 6, 1)
        
        def obj_func(weights):
            return objective_function(weights, model, X_train, y_train)
        
        acor_lm = MultipleColonyACOR(
            obj_func=obj_func, dim=num_weights, n_colonies=3, n_ants=2,
            n_samples=230, q=0.6, xi=0.9, max_iter=100, patience=15,
            sharing_frequency=10, sharing_ratio=0.1, seed=42 + run
        )
        
        best_weights, best_loss, iterations = acor_lm.optimize(
            lb=-3.0, ub=3.0, model=model, X_train=X_train, y_train=y_train
        )
        
        model.set_weights(best_weights)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        results['accuracy'].append(acc)
        results['precision'].append(prec)
        results['recall'].append(rec)
        results['f1_score'].append(f1)
        results['confusion_matrices'].append(cm)
        results['best_losses'].append(best_loss)
        results['iterations'].append(iterations)
        
        if acc > best_overall_accuracy:
            best_overall_accuracy = acc
            best_model_weights = best_weights.copy()
            best_run_idx = run
        
        print(f"Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, Loss: {best_loss:.3f}")
    
    results['best_model_weights'] = best_model_weights
    results['best_run_index'] = best_run_idx
    results['best_overall_accuracy'] = best_overall_accuracy
    return results

# 5. Main execution
# --------------------------------------------------
if __name__ == "__main__":
    print("ACOR-LM with Multiple Colonies for Heart Disease Classification")
    print("=" * 60)
    print(f"Architecture: 35 inputs, 6 hidden (ReLU), 1 output (Sigmoid)")
    print(f"Total weights: {FNN_Thesis.get_num_weights(35, 6, 1)}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Evaluation: 50 independent runs")
    
    results = evaluate_acor_lm(X_train, X_test, y_train, y_test, n_runs=50)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Averaged across 50 runs)")
    print("=" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"Best Loss: {np.mean(results['best_losses']):.6f} ± {np.std(results['best_losses']):.6f}")
    print(f"Iterations: {np.mean(results['iterations']):.1f} ± {np.std(results['iterations']):.1f}")
    
    avg_cm = np.mean(results['confusion_matrices'], axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(avg_cm)
    
    print(f"\nBest Model: Run {results['best_run_index'] + 1} with accuracy {results['best_overall_accuracy']:.4f}")
    
    # Define directories and create them if they don't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, 'reports')
    models_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Get a single timestamp for all files from this run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save the .pkl file in the 'saved_models' directory
    pkl_filename = f'heart_acor_lm_model_{timestamp}.pkl'
    full_pkl_path = os.path.join(models_dir, pkl_filename)
    
    model_data = {
        'best_model_weights': results['best_model_weights'], 'best_run_index': results['best_run_index'],
        'best_overall_accuracy': results['best_overall_accuracy'],
        'metrics_summary': {
            'accuracy_mean': np.mean(results['accuracy']), 'accuracy_std': np.std(results['accuracy']),
            'precision_mean': np.mean(results['precision']), 'precision_std': np.std(results['precision']),
            'recall_mean': np.mean(results['recall']), 'recall_std': np.std(results['recall']),
            'f1_mean': np.mean(results['f1_score']), 'f1_std': np.std(results['f1_score']),
            'loss_mean': np.mean(results['best_losses']), 'loss_std': np.std(results['best_losses']),
            'iterations_mean': np.mean(results['iterations']), 'iterations_std': np.std(results['iterations']),
        },
        'average_confusion_matrix': avg_cm,
        'architecture': {'input': 35, 'hidden': 6, 'output': 1, 'weights': 223},
        'evaluation': {'train_samples': len(X_train), 'test_samples': len(X_test), 'runs': 50},
        'algorithm': 'ACOR-LM with Multiple Colonies',
        'dataset': 'heart1.dat (35 features)',
        'scaler': scaler
    }
    
    with open(full_pkl_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {full_pkl_path}")

    # Generate and save the .png report in the 'reports' directory
    generate_summary_chart(results, reports_dir, timestamp)