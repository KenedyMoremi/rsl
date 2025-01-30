from flask import Flask, request, jsonify
import joblib
import torch
import torch.nn as nn
import numpy as np
import re

app = Flask(__name__)

# Load Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

# Define Neural Network Model
class DataValidationNN(nn.Module):
    def __init__(self, input_size):
        super(DataValidationNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

# Load Neural Network model
input_size = 4  # Adjust based on features
nn_model = DataValidationNN(input_size)
nn_model.load_state_dict(torch.load("neural_network_model.pth"))
nn_model.eval()

# Function to check if Name is valid (No numbers or special characters)
def is_valid_name(name):
    return bool(re.match(r"^[A-Za-z ]+$", name))  # Only letters and spaces

# Function to check if InvoiceNumber is valid (Must be alphanumeric)
def is_valid_invoice(invoice):
    return bool(re.match(r"^[A-Za-z0-9]+$", invoice))  # Alphanumeric only

# Function to convert the input JSON to model-compatible format
def preprocess_input(data):
    name_valid = 1 if is_valid_name(data["Name"]) else 0
    date_valid = 1 if data["Date"] else 0  # Assuming any non-empty date is valid
    invoice_valid = 1 if is_valid_invoice(data["InvoiceNumber"]) else 0
    supplier_valid = 1 if data["Supplier"] else 0  # Assuming any non-empty supplier is valid
    
    return np.array([name_valid, date_valid, invoice_valid, supplier_valid, 1]).reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.json

        # Validate required fields
        required_fields = ["Name", "Date", "InvoiceNumber", "Supplier"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Preprocess input
        features = preprocess_input(data)

        # Random Forest Prediction
        rf_prediction = rf_model.predict(features)[0]

        # Neural Network Prediction
        with torch.no_grad():
            tensor_features = torch.tensor(features, dtype=torch.float32)
            nn_prediction = nn_model(tensor_features).item()
            nn_prediction = 1 if nn_prediction > 0.5 else 0  # Convert probability to class

        return jsonify({
            "random_forest_prediction": rf_prediction,
            "neural_network_prediction": nn_prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ensure the app runs only when using Flask directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Ignored when using Gunicorn
