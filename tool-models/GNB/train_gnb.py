import os
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_proben1_dat(path_to_dat_file, n_inputs, n_outputs):
    X, y = [], []
    with open(path_to_dat_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            values = [float(v) for v in parts]
            xi = values[:n_inputs]
            yi = values[n_inputs:]
            if n_outputs == 1:
                yi_val = int(round(yi[0]))
            else:
                yi_val = int(np.argmax(yi))
            X.append(xi)
            y.append(yi_val)
    return np.array(X), np.array(y)

class MyGaussianNB:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.vars = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-9  # avoid div/0
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_pdf(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        probs = []
        for x in X:
            log_posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self._gaussian_pdf(c, x) + 1e-12))  # prevent log(0)
                log_posteriors.append(prior + likelihood)

            log_posteriors = np.array(log_posteriors)
            max_log = np.max(log_posteriors)  
            exp_post = np.exp(log_posteriors - max_log)
            probs.append(exp_post / exp_post.sum())

        return np.array(probs)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def train_and_save(dataset_path, n_inputs, n_outputs, name, label_names):
    print(f"\n🔹 Training model for {name}...")
    if not os.path.isabs(dataset_path):
        candidates = [
            os.path.join(BASE_DIR, dataset_path),
            os.path.join(BASE_DIR, "datasets", dataset_path),
        ]
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = p
                break
        if found is None:
            tried = '\n'.join(candidates)
            raise FileNotFoundError(f"Dataset file not found. Tried:\n{tried}")
        dataset_path = found

    X, y = load_proben1_dat(dataset_path, n_inputs, n_outputs)

    # 50/25/25 split
    n_total = len(X)
    n_train = int(0.5 * n_total)
    n_val   = int(0.25 * n_total)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

    # --- Class distribution per split ---
    def _dist(y_split):
        counts = np.bincount(y_split, minlength=len(label_names))
        total = len(y_split)
        return [(int(c), (c/total*100.0) if total>0 else 0.0) for c in counts]

    train_dist = _dist(y_train)
    val_dist   = _dist(y_val)
    test_dist  = _dist(y_test)

    print("🔎 Class distribution (count | % within split):")
    header = "        " + " | ".join([f"{lbl}" for lbl in label_names])
    print(header)
    def _row(name, dist):
        cells = [f"{c} | {pct:.1f}%" for c, pct in dist]
        print(f"  {name:<5}: " + " | ".join(cells))
    _row("Train", train_dist)
    _row("Val",   val_dist)
    _row("Test",  test_dist)

    model = MyGaussianNB()
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_acc = (y_pred_train == y_train).mean()
    test_acc  = (y_pred_test == y_test).mean()

    train_cep = (1.0 - train_acc) * 100.0
    test_cep  = (1.0 - test_acc) * 100.0

    cm = np.zeros((len(label_names), len(label_names)), dtype=int)
    for true, pred in zip(y_test, y_pred_test):
        cm[int(true), int(pred)] += 1

    print(f"Dataset size: {n_total} (Train={n_train}, Val={n_val}, Test={len(X_test)})")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy    : {test_acc:.4f}")
    print(f"Training CEP (%) : {train_cep:.2f}")
    print(f"Test CEP (%)     : {test_cep:.2f}")
    print("📊 Confusion Matrix (Test):\n", cm)
    print("📑 Classification Report (Test):")
    for idx, label in enumerate(label_names):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{label}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    model_path = os.path.join(MODELS_DIR, f"gnb_{name.lower()}.joblib")
    dump(model, model_path)
    print(f"💾 Saved model to: {model_path}")


if __name__ == "__main__":
    train_and_save("cancer1.dat",   n_inputs=9,  n_outputs=2, name="Cancer",   label_names=["Benin", "Malignant"])
    train_and_save("heart1.dat",    n_inputs=35, n_outputs=2, name="Heart",    label_names=["Negative", "Positive"])
    train_and_save("diabetes1.dat", n_inputs=8,  n_outputs=2, name="Diabetes", label_names=["Non-Diabetic", "Diabetic"])
