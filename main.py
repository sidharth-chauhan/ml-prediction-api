from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd 
import joblib

model=joblib.load('random_forest_model.pkl')

app=FastAPI()

class PredictRegression(BaseModel):
  season: float
  yr: float
  mnth: float
  hr: float
  holiday: float
  weekday: float
  workingday: float
  weathersit: float
  temp: float
  atemp: float
  hum: float
  windspeed: float

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(data: PredictRegression):
    df = [data.model_dump()]
    new_data = pd.DataFrame(df)

    prediction=model.predict(new_data)

    return {"prediction": float(prediction[0])}


class PredictClassification(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

model_2=joblib.load('logistic_regression.pkl')
model_2_scaler=joblib.load('scaler.pkl')

@app.post("/predict_classification")
async def predict_classification(data: PredictClassification):
    # pydantic model to dict - > list of dict -> dataframe
    df = [data.model_dump()]
    new_data = pd.DataFrame(df)

    # Scale input
    new_data_scaled = model_2_scaler.transform(new_data.to_numpy())

    # Prediction
    prediction = model_2.predict(new_data_scaled)

    if prediction[0] == 0:
      result = "Malignant"
    else:
      result = "Benign"
    return {"prediction": result, "raw_prediction": int(prediction[0])}

  
    