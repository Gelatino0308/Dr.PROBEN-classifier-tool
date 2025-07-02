# Dr. PROBEN Classifier Tool - Setup and Run Guide

This guide will help you set up and run the Dr. PROBEN classifier tool for diabetes prediction using machine learning.

---

## Prerequisites

- Python 3.7 or higher  
- `pip` (Python package installer)

---

## Installation Steps

### 1. Install Required Python Packages

Open a terminal/command prompt in the project directory and run:

```bash
pip install flask flask-cors numpy pandas scikit-learn matplotlib pickle-mixin
```

### 2. Train the Machine Learning Model

Before running the web application, you need to train the diabetes prediction model:

```bash
python diabetes/acor_fnn_diabetes.py
```

This will:

- Train the neural network using the ACOR algorithm

- Generate performance metrics and charts

- Save the trained model to diabetes_model.pkl

- Create result files showing model accuracy (~71.43%)

### 3. Start the Backend API Server

Run the Flask backend server:

```bash
python app.py
```

You should see output like:

```nginx
Model loaded successfully!
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**⚠️ Keep this terminal window open - the server needs to stay running.**

### Start the Frontend

Open the index.html file in the browser.


