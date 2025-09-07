from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
file_path = "random_forest_model.pkl"  # Path to the model file
with open(file_path, 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return "Welcome to the ML Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({'prediction': prediction.tolist()})
