import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model_dir = "../model/"
os.makedirs(model_dir, exist_ok=True)

# Load Data
df = pd.read_csv('data.csv')
df = df.set_index('id')
df.drop(['Unnamed: 32'], axis=1, inplace=True)

df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0).astype(float)

# Prepare Data
Y = df['diagnosis'].values
X = df.drop('diagnosis', axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=21)

# Standardize the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the Scaler
with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Define Models
models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVM', SVC(probability=True)),
    ('Naive Bayes', GaussianNB()),
    ('KNN', KNeighborsClassifier())
]

best_model = None
best_accuracy = 0
best_model_name = ""
best_params = {}

# Evaluate Models
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring='accuracy')
    mean_accuracy = cv_results.mean()
    
    print(f"Model: {name} | Accuracy: {mean_accuracy:.4f}")
    
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_model = model
        best_model_name = name
        best_params = model.get_params()

# Train Best Model
best_model.fit(X_train_scaled, Y_train)
predictions = best_model.predict(X_test_scaled)
final_accuracy = accuracy_score(Y_test, predictions)

# Save Best Model as Pickle File
with open(os.path.join(model_dir, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

# Save Model Details to JSON
model_details = {
    "best_model": best_model_name,
    "accuracy": final_accuracy,
    "parameters": best_params
}
with open(os.path.join(model_dir, "model_details.json"), "w") as f:
    json.dump(model_details, f, indent=4)

print("✅ Best Model Saved: best_model.pkl")
print("✅ Scaler Saved: scaler.pkl")
print("✅ Model Details Saved: model_details.json")
