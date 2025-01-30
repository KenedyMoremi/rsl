from flask import Flask, request, jsonify
import joblib
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Load the Random Forest model
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

# Load the Neural Network model
input_size = 5  # Adjust based on features
nn_model = DataValidationNN(input_size)
nn_model.load_state_dict(torch.load("neural_network_model.pth"))
nn_model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)

    # Random Forest Prediction
    rf_prediction = rf_model.predict(features)[0]

    # Neural Network Prediction
    with torch.no_grad():
        tensor_features = torch.tensor(features, dtype=torch.float32)
        nn_prediction = nn_model(tensor_features).item()
        nn_prediction = 1 if nn_prediction > 0.5 else 0  # Convert probability to class

    return jsonify({"random_forest_prediction": rf_prediction, "neural_network_prediction": nn_prediction})

if __name__ == "__main__":
    app.run(debug=True)
    
