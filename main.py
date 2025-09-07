from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "random_forest_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return "Welcome to the ML Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # Ensure JSON parsing
        features = data['features']

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
