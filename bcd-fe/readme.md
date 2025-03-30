# Breast Cancer Detection

## 📌 Project Overview
This project is a **Breast Cancer Prediction System** that consists of:
- A **machine learning model** trained to classify breast cancer as **Malignant (M)** or **Benign (B)**.
- A **Flask web application** that allows users to predict breast cancer using either JSON data or uploaded breast scans.
- **Feature extraction** from breast scan images using **OpenCV**.

---

## 📂 Project Structure
```bash
.
├── model/         # Contains trained ML model and details
│   ├── best_model.pkl     # Trained ML model (saved as pickle file)
│   ├── scaler.pkl         # StandardScaler for feature scaling
│   ├── model_details.json # Information about the best model
│   ├── train_model.py     # Script to train and save the model
│   └── data.csv           # Breast cancer dataset (if applicable)
│
├── web/          # Flask application for prediction
│   ├── templates/        # HTML Templates
│   │   ├── index.html    # Home page
│   │   ├── result.html   # Results page
│   │   ├── error.html    # Error page
│   ├── uploads/         # Uploaded breast scan images
│   ├── app.py           # Flask API and Web Server
│   ├── requirements.txt # Python dependencies
│   ├── static/          # CSS, JS, and other static files
│   └── utils.py         # Feature extraction with OpenCV
│
├── README.md     # Project documentation
└── requirements.txt # Dependencies for the project
```

---

## 🚀 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/s4hil/breast-cancer-detection
cd breast-cancer-detection
```

### **2️⃣ Install Dependencies**
Ensure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### **3️⃣ Train the Model (Optional)**
If you want to retrain the model, navigate to the `model/` directory and run:
```sh
python train_model.py
```
This will create `best_model.pkl`, `scaler.pkl`, and `model_details.json` inside `model/`.

### **4️⃣ Run the Flask App**
Go to the `web/` directory and start the Flask server:
```sh
cd web
python app.py
```
The app will be available at: `http://127.0.0.1:5000/`

---

## 🌟 Features
✅ **Breast Cancer Prediction** based on input features or an uploaded scan  
✅ **JSON API** to send input data programmatically  
✅ **Model Trained with Multiple Algorithms**, selecting the best one  
✅ **Web UI with Bootstrap Styling**  
✅ **Displays Prediction Accuracy and Extracted Features**  
✅ **File Upload Support** for image-based predictions  
✅ **Feature Extraction with OpenCV** for image analysis  

---

## 📌 API Endpoints
| Endpoint          | Method | Description |
|------------------|--------|-------------|
| `/`              | GET    | Renders the home page |
| `/model-info`    | GET    | Renders model details |
| `/api/predict`   | POST   | Accepts JSON input with 30 features and returns the cancer diagnosis |
| `/upload`        | POST   | Uploads a breast scan image and predicts if cancer is Malignant or Benign |

### **🔹 JSON Prediction API**
- **`POST /api/predict`** → Accepts JSON input with 30 features and returns the cancer diagnosis.

  **Example Request:**
  ```json
  {
      "radius_mean": 17.99,
      "texture_mean": 10.38,
      "perimeter_mean": 122.8,
      "area_mean": 1001.0,
      "smoothness_mean": 0.1184,
      "compactness_mean": 0.2776,
      "concavity_mean": 0.3001,
      "concave_points_mean": 0.1471,
      "symmetry_mean": 0.2419,
      "fractal_dimension_mean": 0.07871,
      "radius_se": 1.095,
      "texture_se": 0.9053,
      "perimeter_se": 8.589,
      "area_se": 153.4,
      "smoothness_se": 0.006399,
      "compactness_se": 0.04904,
      "concavity_se": 0.05373,
      "concave_points_se": 0.01587,
      "symmetry_se": 0.03003,
      "fractal_dimension_se": 0.006193,
      "radius_worst": 25.38,
      "texture_worst": 17.33,
      "perimeter_worst": 184.6,
      "area_worst": 2019.0,
      "smoothness_worst": 0.1622,
      "compactness_worst": 0.6656,
      "concavity_worst": 0.7119,
      "concave_points_worst": 0.2654,
      "symmetry_worst": 0.4601,
      "fractal_dimension_worst": 0.1189
  }
  ```
  **Example Response:**
  ```json
  {
      "diagnosis": "Malignant",
      "input_data": { ... }
  }
  ```

### **🔹 Image Upload Prediction**
- **`POST /upload`** → Uploads a breast scan image and predicts if cancer is Malignant or Benign.

  **Example Response:**
  ```json
  {
      "file_path": "uploads/sample.jpg",
      "diagnosis": "Benign",
      "accuracy": "97.8%",
      "extracted_features": [0.3, 1.2, 2.5, ...]
  }
  ```

---

## 📷 Screenshots
### **1️⃣ Home Page**
![Home Page](static/home_screenshot.png)

### **2️⃣ Prediction Result**
![Result Page](static/result_screenshot.png)

---

## 🛠 Technologies Used
- **Python** (Flask, NumPy, Pandas, Scikit-Learn, OpenCV)
- **Machine Learning** (SVM, Decision Tree, KNN, Naive Bayes)
- **Bootstrap** (UI Styling)
- **HTML, CSS, JavaScript** (Frontend)

---

## 🐟 License
This project is **open-source** and available under the [MIT License](LICENSE).

---

## 💡 Acknowledgments
- **Dataset:** [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Machine Learning Reference:** Scikit-learn documentation
- **UI Framework:** Bootstrap

---

🚀 **Happy Coding & Stay Healthy!** 🏥✨