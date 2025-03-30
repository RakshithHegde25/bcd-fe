import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and scaler
with open("../model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("../model/model_details.json", "r") as f:
    model_details = json.load(f)

# Feature names for the form
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

@app.route("/")
def home():
    """Render the main page with form"""
    return render_template(
        "index.html",
        model_accuracy=round(model_details.get("accuracy", 0) * 100),
        feature_names=FEATURE_NAMES
    )

@app.route("/predict", methods=["POST"])
def predict():
    """Handle JSON input for prediction with validation"""
    try:
        # Get and validate JSON
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        
        # Validate all required features are present
        missing_features = [f for f in FEATURE_NAMES if f not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing features: {', '.join(missing_features)}",
                "required_features": FEATURE_NAMES
            }), 400
            
        # Validate feature values are numbers
        invalid_features = {}
        for feature in FEATURE_NAMES:
            try:
                float(data[feature])
            except (ValueError, TypeError):
                invalid_features[feature] = data[feature]
                
        if invalid_features:
            return jsonify({
                "error": "Invalid feature values",
                "invalid_features": invalid_features
            }), 400
            
        # Convert all features to floats
        features = [float(data[feature]) for feature in FEATURE_NAMES]
        
        # Make prediction
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max() * 100
        
        return jsonify({
            "diagnosis": "Malignant" if prediction == 1 else "Benign",
            "confidence": round(confidence, 2),
            "accuracy": round(model_details.get("accuracy", 0) * 100, 2),
            "features_used": {name: val for name, val in zip(FEATURE_NAMES, features)}
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file uploads"""
    if 'file' not in request.files:
        return render_template("error.html", error_message="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template("error.html", error_message="No selected file")
    
    try:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # In a real app, you would extract features from the image here
        # For demo, we'll use random features scaled to typical ranges
        features = np.random.rand(30).tolist()
        features[0] = features[0] * 10 + 10  # radius_mean
        features[1] = features[1] * 5 + 10   # texture_mean
        features[2] = features[2] * 50 + 50   # perimeter_mean
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max() * 100
        
        return render_template(
            "result.html",
            file_path=os.path.join("uploads", filename),
            diagnosis="Malignant" if prediction == 1 else "Benign",
            confidence=round(confidence, 2),
            accuracy=round(model_details.get("accuracy", 0) * 100, 2),
            features=features,
            feature_names=FEATURE_NAMES
        )
    except Exception as e:
        return render_template("error.html", error_message=str(e))

@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)