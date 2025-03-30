import os
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import json
from flask import send_from_directory

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "../model/best_model.pkl"
SCALER_PATH = "../model/scaler.pkl"
MODEL_DETAILS_PATH = "../model/model_details.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found! Train the model first.")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Scaler file not found! Train the model first.")

if not os.path.exists(MODEL_DETAILS_PATH):
    raise FileNotFoundError("Model details file not found! Train the model first.")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(MODEL_DETAILS_PATH, "r") as details_file:
    model_details = json.load(details_file)

best_accuracy = model_details.get("accuracy", "N/A")  # Retrieve stored accuracy

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    """Render the homepage with the form."""
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/predict", methods=["POST"])
def predict():
    """Handles manual form entry and JSON input for predictions."""
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model or Scaler not found!"})

        if request.is_json:
            data = request.get_json()
            features = np.array(list(data.values())).reshape(1, -1)
        else:
            # Get form data
            features = [float(request.form[key]) for key in request.form]
            features = np.array(features).reshape(1, -1)

        # Apply the scaler transformation
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        return jsonify({
            "diagnosis": diagnosis,
            "input_data": request.form if not request.is_json else data,
            "accuracy": str(best_accuracy * 100)[:5]
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for JSON-based cancer prediction."""
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model or Scaler not found!"})

        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        return jsonify({
            "diagnosis": diagnosis,
            "accuracy": best_accuracy
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/upload", methods=["POST"])
def upload_scan():
    """Handles file uploads and displays results in a webpage."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        filename = secure_filename(file.filename)
        file_path = os.path.join("./uploads/", filename)
        file.save(file_path)

        # TODO: Extract features from the image (currently simulating)
        extracted_features = np.random.rand(1, 30)

        # Scale features
        features_scaled = scaler.transform(extracted_features)

        # Make prediction
        prediction = model.predict(features_scaled)
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        return render_template(
            "result.html",
            file_path=file_path,
            diagnosis=diagnosis,
            accuracy=f"{best_accuracy:.2%}",
            features=extracted_features.tolist()[0]
        )

    except Exception as e:
        return render_template("error.html", error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)

