import joblib
import pandas as pd

model= joblib.load('random_forest_model.pkl')

new_data = pd.DataFrame([{
    'season': 1,
    'yr': 1,
    'mnth': 1,
    'hr': 10,
    'holiday': 0,
    'weekday': 6,
    'workingday': 0,
    'weathersit': 1,
    'temp': 0.24,
    'atemp': 0.28,
    'hum': 0.70,
    'windspeed': 0.10
}])

prediction = model.predict(new_data)

print("Predicted bike rentals:", prediction[0])




# Classification
print("Classification Example")


# Load model and scaler
model = joblib.load('logistic_regression.pkl')
scaler = joblib.load('scaler.pkl')

# New input (30 features)
new_data = pd.DataFrame([{
    'mean radius': 0.0,
    'mean texture': 0.0,
    'mean perimeter': 0.0,
    'mean area': 0.0,
    'mean smoothness': 0.0,
    'mean compactness': 0.0,
    'mean concavity': 0.0,
    'mean concave points': 0.0,
    'mean symmetry': 0.0,
    'mean fractal dimension': 0.0,
    'radius error': 0.0,
    'texture error': 0.0,
    'perimeter error': 0.0,
    'area error': 0.0,
    'smoothness error': 0.0,
    'compactness error': 0.0,
    'concavity error': 0.0,
    'concave points error': 0.0,
    'symmetry error': 0.0,
    'fractal dimension error': 0.0,
    'worst radius': 0.0,
    'worst texture': 0.0,
    'worst perimeter': 0.0,
    'worst area': 0.0,
    'worst smoothness': 0.0,
    'worst compactness': 0.0,
    'worst concavity': 0.0,
    'worst concave points': 0.0,
    'worst symmetry': 0.0,
    'worst fractal dimension': 0.0
}])

# Scale input
new_data_scaled = scaler.transform(new_data.to_numpy()) # Convert DataFrame to numpy array before scaling

# Prediction
prediction = model.predict(new_data_scaled)

print("Prediction:", prediction[0])

if prediction[0] == 0:
    print("Malignant")
else:
    print("Benign")
