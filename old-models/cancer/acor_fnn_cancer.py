import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and preprocess the data
# --------------------------------------------------
# Load the breast-cancer-wisconsin.data file
data_path = os.path.join(os.path.dirname(__file__), 'breast-cancer-wisconsin.data')
column_names = [
    'id',
    'clump_thickness',
    'uniformity_cell_size',
    'uniformity_cell_shape',
    'marginal_adhesion',
    'single_epithelial_cell_size',
    'bare_nuclei',
    'bland_chromatin',
    'normal_nucleoli',
    'mitoses',
    'class'
]

# Read the data with column names
data = pd.read_csv(data_path, names=column_names, na_values='?')

# Drop the 'id' column as it's not needed for prediction
data = data.drop('id', axis=1)

# Handle missing values (replace with median)
# The 'bare_nuclei' column often has missing values marked as '?'
data = data.dropna()  # For simplicity, drop rows with missing values

# Map class values: 2 (benign) -> 0, 4 (malignant) -> 1
data['class'] = data['class'].map({2: 0, 4: 1})

# Separate features and target
X = data.drop('class', axis=1).values
y = data['class'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Debug: Check data and labels
print('Dataset shape:', X.shape)
print('Features:', column_names[1:-1])  # Exclude 'id' and 'class'
print('Unique values in y_train:', np.unique(y_train, return_counts=True))
print('Unique values in y_test:', np.unique(y_test, return_counts=True))
print('First 5 rows of X_train:', X_train[:5])

# 2. Define a simple FNN (single hidden layer, 6 neurons)
# --------------------------------------------------
class FNN:
    """
    Feedforward Neural Network (FNN) with a single hidden layer (6 neurons) and 1 output neuron.
    Architecture: Input -> Hidden(6) -> Output(1)
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
        a2 = 1 / (1 + np.exp(-z2))  # Sigmoid for binary classification
        return a2.squeeze()

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    @staticmethod
    def get_num_weights(input_dim, hidden_dim, output_dim):
        return input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim

# 3. Improved ACOR implementation
# --------------------------------------------------
class ACOR:
    def __init__(self, obj_func, dim, n_ants=40, n_samples=80, q=0.1, xi=0.85, max_iter=60, patience=12):
        self.obj_func = obj_func
        self.dim = dim
        self.n_ants = n_ants
        self.n_samples = n_samples
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.patience = patience

    def optimize(self, lb, ub):
        # --- Step 1: Initialization ---
        # Generate P (= n_samples) random solutions
        solutions = np.random.uniform(lb, ub, (self.n_samples, self.dim))
        fitness = np.array([self.obj_func(sol) for sol in solutions])
        idx = np.argsort(fitness)
        solutions = solutions[idx]
        fitness = fitness[idx]
        best_sol = solutions[0].copy()
        best_fit = fitness[0]
        best_iter = 0
        
        for it in range(self.max_iter):
            # --- Step 2: Solution Construction ---
            # Construct G Gaussian mixture pdfs (weights and stds)
            w = 1/(self.q*self.n_samples*np.sqrt(2*np.pi)) * np.exp(-0.5*(np.arange(self.n_samples)/(self.q*self.n_samples))**2)
            w /= w.sum()
            s = np.zeros((self.n_samples, self.dim))
            for i in range(self.n_samples):
                s[i] = self.xi * np.std(solutions, axis=0) + 1e-10
            
            # Generate Q (= n_ants) new trial solutions by sampling the GMM
            new_solutions = np.zeros((self.n_ants, self.dim))
            for k in range(self.n_ants):
                idx = np.random.choice(self.n_samples, p=w)
                new_solutions[k] = np.random.normal(solutions[idx], s[idx])
                new_solutions[k] = np.clip(new_solutions[k], lb, ub)
            
            new_fitness = np.array([self.obj_func(sol) for sol in new_solutions])
            
            # --- Step 3: Pheromone Update ---
            # Add new solutions to population and discard worst Q
            all_solutions = np.vstack([solutions, new_solutions])
            all_fitness = np.hstack([fitness, new_fitness])
            idx = np.argsort(all_fitness)
            solutions = all_solutions[idx][:self.n_samples]
            fitness = all_fitness[idx][:self.n_samples]
            
            # --- Step 4: Convergence Check ---
            if fitness[0] < best_fit:
                best_fit = fitness[0]
                best_sol = solutions[0].copy()
                best_iter = it
            
            print(f"Iteration {it+1}/{self.max_iter}, Best Loss: {best_fit:.4f}")
            
            # Early stopping (convergence)
            if it - best_iter > self.patience:
                print(f"Early stopping at iteration {it+1}")
                return best_sol, best_fit, it + 1
        
        return best_sol, best_fit, self.max_iter

# 4. Objective function for ACOR (binary cross-entropy loss)
# --------------------------------------------------
def objective(weights):
    model.set_weights(weights)
    y_pred = model.forward(X_train)
    eps = 1e-8
    loss = -np.mean(y_train*np.log(y_pred+eps) + (1-y_train)*np.log(1-y_pred+eps))
    return loss

# 5. Model and ACOR parameters
# --------------------------------------------------
input_dim = X_train.shape[1]  # Should be 9 for the new dataset
hidden_dim = 6  # single hidden layer size
output_dim = 1
model = FNN(input_dim, hidden_dim, output_dim)
num_weights = FNN.get_num_weights(input_dim, hidden_dim, output_dim)
acor = ACOR(obj_func=objective, dim=num_weights, n_ants=30, n_samples=80, q=0.1, xi=0.85, max_iter=80, patience=15)
lb = -3
ub = 3

# Debug: Model output before optimization
model = FNN(input_dim, hidden_dim, output_dim)
random_weights = np.random.uniform(-3, 3, num_weights)
model.set_weights(random_weights)
pre_opt_output = model.forward(X_train)
print('Model output (forward) on X_train before optimization (first 10):', pre_opt_output[:10])

# Debug: Loss before optimization
pre_opt_loss = -np.mean(y_train*np.log(pre_opt_output+1e-8) + (1-y_train)*np.log(1-pre_opt_output+1e-8))
print('Loss before optimization:', pre_opt_loss)

# 6. Run ACOR to optimize FNN weights
# --------------------------------------------------
print("Starting ACOR optimization...")
best_weights, best_loss, n_iterations = acor.optimize(lb, ub)

# Debug: Model output after optimization
model.set_weights(best_weights)
post_opt_output = model.forward(X_train)
print('Model output (forward) on X_train after optimization (first 10):', post_opt_output[:10])

# Debug: Loss after optimization
post_opt_loss = -np.mean(y_train*np.log(post_opt_output+1e-8) + (1-y_train)*np.log(1-post_opt_output+1e-8))
print('Loss after optimization:', post_opt_loss)

# 7. Evaluate on test set and save results
# --------------------------------------------------
model.set_weights(best_weights)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Class distribution
unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
unique_true, counts_true = np.unique(y_test, return_counts=True)

# Warn if only one class is predicted
if len(unique_pred) == 1:
    warnings.warn(f"Model predicted only one class: {unique_pred[0]}. Metrics may be misleading.")

# Print metrics
print(f"\nTest Results:")
print(f"Test Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(f"  True Negatives (benign predicted correctly): {cm[0,0]}")
print(f"  False Positives (benign predicted as malignant): {cm[0,1]}")
print(f"  False Negatives (malignant predicted as benign): {cm[1,0]}")
print(f"  True Positives (malignant predicted correctly): {cm[1,1]}")
print(f"Predicted class distribution: {y_pred.tolist().count(0)} predicted benign, {y_pred.tolist().count(1)} predicted malignant")
print(f"True class distribution: {y_test.tolist().count(0)} actually benign, {y_test.tolist().count(1)} actually malignant")

# Save results to a txt file
output_dir = os.path.dirname(__file__)
with open(os.path.join(output_dir, 'cancer_result.txt'), 'w') as f:
    f.write(f"Breast Cancer Classification Results\n")
    f.write(f"=====================================\n")
    f.write(f"Dataset: Wisconsin Breast Cancer Database\n")
    f.write(f"Features: 9 attributes (Clump Thickness, Uniformity of Cell Size, etc.)\n")
    f.write(f"Architecture: {input_dim} -> {hidden_dim} -> {output_dim}\n")
    f.write(f"Optimization: ACOR Algorithm\n\n")
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"Number of Iterations until Convergence: {n_iterations}\n\n")
    f.write("Confusion Matrix (for test set):\n")
    f.write(f"  True Negatives (benign predicted correctly): {cm[0,0]}\n")
    f.write(f"  False Positives (benign predicted as malignant): {cm[0,1]}\n")
    f.write(f"  False Negatives (malignant predicted as benign): {cm[1,0]}\n")
    f.write(f"  True Positives (malignant predicted correctly): {cm[1,1]}\n\n")
    f.write(f"Class Distribution:\n")
    f.write(f"Predicted: {y_pred.tolist().count(0)} benign, {y_pred.tolist().count(1)} malignant\n")
    f.write(f"Actual: {y_test.tolist().count(0)} benign, {y_test.tolist().count(1)} malignant\n")
    if len(unique_pred) == 1:
        f.write(f"WARNING: Model predicted only one class: {unique_pred[0]}. Metrics may be misleading.\n")

# Plot and save confusion matrix
def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label', xlabel='Predicted label',
        title='Breast Cancer Classification - Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_confusion_matrix(cm, class_names=["Benign", "Malignant"], 
                     save_path=os.path.join(output_dir, "cancer_confusion_matrix.png"))

# Plot and save metrics bar chart
metrics = [acc, prec, rec, f1]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
plt.figure(figsize=(8, 6))
bars = plt.bar(metric_names, metrics, color=['skyblue', 'orange', 'green', 'red'])
plt.ylim(0, 1)
plt.title('Breast Cancer Classification - Performance Metrics')
plt.ylabel('Score')
for bar, value in zip(bars, metrics):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cancer_metrics_bar_chart.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save the trained model (weights and scaler)
model_data = {
    'weights': best_weights,
    'scaler': scaler,
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'output_dim': output_dim,
    'feature_names': column_names[1:-1]  # Store feature names for reference
}

with open(os.path.join(output_dir, 'cancer_model.pkl'), 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to cancer_model.pkl")
print(f"Results saved to cancer_result.txt")
print(f"Visualizations saved as PNG files")