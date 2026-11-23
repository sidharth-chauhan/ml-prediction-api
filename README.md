<h1 align="center">🚴 Bike Demand Prediction & ML Classification API</h1>

<p align="center">
  <strong>Regression · Classification · FastAPI · Scikit-learn</strong><br/>
  Two machine learning applications served through a single REST API.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Git_LFS-Storage-orange?style=for-the-badge&logo=git&logoColor=white" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Installation and Setup](#installation-and-setup)
- [API Documentation](#api-documentation)
- [Bike Demand Prediction](#bike-demand-prediction)
- [Breast Cancer Classification](#breast-cancer-classification)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Evaluation](#model-evaluation)
- [API Endpoints](#api-endpoints)
- [Future Improvements](#future-improvements)

---

## Overview

This project develops and deploys machine learning models through a RESTful API built with **FastAPI** and **Scikit-learn**.

It combines two applications in one service:

- 🚴 **Bike Demand Prediction** – a regression model that predicts bike rental demand from environmental, seasonal, and time-related features.
- 🧬 **Breast Cancer Classification** – a binary classifier that labels a tumor as malignant or benign from cell-nuclei measurements.

Both trained models are serialized with Joblib and loaded by the API to generate predictions over HTTP.

---

## Key Features

- **Two ML applications** – regression and classification in one API
- **Random Forest Regression** – predicts bike rental demand
- **Logistic Regression** – classifies breast tumor measurements
- **Data preprocessing** – `StandardScaler` standardizes classification inputs
- **Model persistence** – trained models saved and loaded with Joblib
- **Git LFS integration** – handles large model files
- **Input validation** – Pydantic validates every request
- **Interactive documentation** – Swagger UI and OpenAPI docs generated automatically
- **Modular inference** – each model has its own endpoint

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | REST API development |
| Scikit-learn | Machine learning algorithms |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computing |
| Joblib | Model serialization and loading |
| Pydantic | Request validation |
| Uvicorn | ASGI server |
| Git LFS | Large model file storage |

---

## Architecture

```text
                 +--------------------------+
                 |          Client          |
                 |  Postman / Browser / cURL|
                 +------------+-------------+
                              |
                              v
                 +--------------------------+
                 |         FastAPI          |
                 |    Request Validation    |
                 +------+-------------+-----+
                        |             |
          POST /predict |             | POST /predict_classification
                        v             v
          +------------------+   +--------------------+
          | Random Forest    |   |   StandardScaler   |
          | Regressor        |   +---------+----------+
          +--------+---------+             |
                   |                       v
                   |             +--------------------+
                   |             | Logistic Regression|
                   |             +---------+----------+
                   |                       |
                   +-----------+-----------+
                               v
                 +--------------------------+
                 |      JSON Response       |
                 +--------------------------+
```

---

## Project Structure

```text
ml-prediction-api/
│
├── main.py
├── predict.py
│
├── random_forest_model.pkl
├── logistic_regression.pkl
├── scaler.pkl
│
├── bike+sharing+dataset/
│   ├── Readme.txt
│   ├── day.csv
│   └── hour.csv
│
├── notebook_classification/
├── notebook_regression/
│
├── requirements.txt
├── .gitattributes
├── .gitignore
└── README.md
```

### Model Artifacts

| File | Description |
|---|---|
| `random_forest_model.pkl` | Trained Random Forest Regressor |
| `logistic_regression.pkl` | Trained Logistic Regression Classifier |
| `scaler.pkl` | Fitted StandardScaler used for classification |

> Model files are stored with Git Large File Storage (Git LFS).

---

## Machine Learning Models

Multiple algorithms were explored in the notebooks before one model per task was selected for deployment.

### 1. Bike Demand Prediction (Regression)

**Algorithms explored:** Linear Regression, Decision Tree Regressor, Random Forest Regressor

**Selected model: Random Forest Regressor**

Random Forest is an ensemble method that trains many decision trees on different samples of the data, uses random feature subsets when building each tree, and averages their predictions.

**Why Random Forest?**

- Captures nonlinear relationships between features
- Handles complex feature interactions
- Reduces overfitting compared with a single decision tree in many settings
- Works well on structured, tabular data

Saved as `random_forest_model.pkl`.

### 2. Breast Cancer Classification (Binary Classification)

**Algorithms explored:** Logistic Regression, Decision Tree, Gaussian Naive Bayes, K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM)

**Selected model: Logistic Regression**

Logistic Regression estimates the probability that an input belongs to a class using the logistic (sigmoid) function.

**Why Logistic Regression?**

- Well suited to binary classification
- Produces probability-based predictions
- Relatively easy to interpret
- Works well with standardized numerical features
- Provides a strong, simple baseline

Saved as `logistic_regression.pkl`, with the fitted scaler saved as `scaler.pkl`.

### Model Selection Summary

| Application | Algorithms Explored | Selected Model |
|---|---|---|
| Bike Demand Prediction | Linear Regression, Decision Tree, Random Forest | **Random Forest Regressor** |
| Breast Cancer Classification | Logistic Regression, Decision Tree, Naive Bayes, KNN, Random Forest, SVM | **Logistic Regression** |

---

## Installation and Setup

### 1. Clone the Repository

Install [Git LFS](https://git-lfs.com/) first so the `.pkl` model files download correctly.

```bash
git lfs install
git clone https://github.com/sidharth-chauhan/ml-prediction-api.git
cd ml-prediction-api
git lfs pull
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI Server

```bash
uvicorn main:app --reload
```

The application runs at `http://127.0.0.1:8000`.

---

## API Documentation

FastAPI generates interactive documentation automatically.

| Documentation | URL |
|---|---|
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |
| Root Endpoint | http://127.0.0.1:8000/ |

---

## Bike Demand Prediction

**Endpoint:** `POST /predict`
**Algorithm:** Random Forest Regressor
**Problem Type:** Regression

Predicts bike rental demand from seasonal, environmental, and time-related features.

### Input Features

| Feature | Description |
|---|---|
| `season` | Season of the year |
| `yr` | Year indicator |
| `mnth` | Month |
| `hr` | Hour of the day |
| `holiday` | Whether the day is a holiday |
| `weekday` | Day of the week |
| `workingday` | Whether the day is a working day |
| `weathersit` | Weather situation |
| `temp` | Normalized temperature |
| `atemp` | Normalized feeling temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |

### Example Request

```json
{
  "season": 2,
  "yr": 1,
  "mnth": 6,
  "hr": 17,
  "holiday": 0,
  "weekday": 3,
  "workingday": 1,
  "weathersit": 1,
  "temp": 0.65,
  "atemp": 0.62,
  "hum": 0.45,
  "windspeed": 0.15
}
```

### Example Response

```json
{
  "prediction": 245.0
}
```

> The prediction above is illustrative. Actual output depends on the trained model and input.

---

## Breast Cancer Classification

**Endpoint:** `POST /predict_classification`
**Algorithm:** Logistic Regression
**Problem Type:** Binary Classification

Uses 30 numerical features describing cell nuclei, computed from digitized images of fine needle aspirates (FNA) of breast masses.

### Prediction Classes

| Output | Meaning |
|---|---|
| `0` | Malignant |
| `1` | Benign |

### Preprocessing (StandardScaler)

Input features are standardized before prediction using this formula:

```text
z = (x - mean) / std
```

- `x` = original feature value
- `mean` = mean from the training data
- `std` = standard deviation from the training data
- `z` = standardized feature value

The same fitted scaler used during training is loaded at inference time, so prediction inputs are transformed consistently with the training data.

### Input Features

The API expects these 30 features:

| # | Mean | Error | Worst |
|---|---|---|---|
| 1 | `mean_radius` | `radius_error` | `worst_radius` |
| 2 | `mean_texture` | `texture_error` | `worst_texture` |
| 3 | `mean_perimeter` | `perimeter_error` | `worst_perimeter` |
| 4 | `mean_area` | `area_error` | `worst_area` |
| 5 | `mean_smoothness` | `smoothness_error` | `worst_smoothness` |
| 6 | `mean_compactness` | `compactness_error` | `worst_compactness` |
| 7 | `mean_concavity` | `concavity_error` | `worst_concavity` |
| 8 | `mean_concave_points` | `concave_points_error` | `worst_concave_points` |
| 9 | `mean_symmetry` | `symmetry_error` | `worst_symmetry` |
| 10 | `mean_fractal_dimension` | `fractal_dimension_error` | `worst_fractal_dimension` |

### Example Request

```json
{
  "mean_radius": 14.5,
  "mean_texture": 20.0,
  "mean_perimeter": 95.0,
  "mean_area": 650.0,
  "mean_smoothness": 0.1,
  "mean_compactness": 0.12,
  "mean_concavity": 0.1,
  "mean_concave_points": 0.05,
  "mean_symmetry": 0.19,
  "mean_fractal_dimension": 0.06,
  "radius_error": 0.3,
  "texture_error": 1.0,
  "perimeter_error": 2.0,
  "area_error": 25.0,
  "smoothness_error": 0.006,
  "compactness_error": 0.02,
  "concavity_error": 0.02,
  "concave_points_error": 0.01,
  "symmetry_error": 0.02,
  "fractal_dimension_error": 0.003,
  "worst_radius": 17.0,
  "worst_texture": 25.0,
  "worst_perimeter": 115.0,
  "worst_area": 900.0,
  "worst_smoothness": 0.14,
  "worst_compactness": 0.25,
  "worst_concavity": 0.25,
  "worst_concave_points": 0.1,
  "worst_symmetry": 0.3,
  "worst_fractal_dimension": 0.08
}
```

### Example Response

```json
{
  "prediction": "Benign",
  "raw_prediction": 1
}
```

> The prediction above is illustrative. Actual output depends on the trained model and input.

---

## Machine Learning Workflow

```text
Data Collection
      |
      v
Exploratory Data Analysis
      |
      v
Data Preprocessing
      |
      v
Feature Preparation
      |
      v
Train-Test Split
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
Model Selection
      |
      v
Model Serialization
      |
      v
FastAPI Deployment
      |
      v
Prediction / Inference
```

---

## Model Evaluation

Different metrics are used for regression and classification.

### Regression Metrics

| Metric | Purpose |
|---|---|
| MAE | Average absolute prediction error |
| MSE | Average squared prediction error |
| RMSE | Prediction error in the target's original scale |
| R² Score | Proportion of variance explained by the model |

### Classification Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall share of correct predictions |
| Precision | Share of positive predictions that are correct |
| Recall | Share of actual positives that are identified |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Correct and incorrect predictions per class |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Welcome endpoint |
| `POST` | `/predict` | Predict bike rental demand |
| `POST` | `/predict_classification` | Classify tumor measurements |

### Quick Test

1. Start the FastAPI server.
2. Open `http://127.0.0.1:8000/docs`.
3. Select the endpoint you want to test.
4. Click **Try it out**.
5. Enter the required JSON input.
6. Click **Execute** to receive the prediction.

You can also test with Postman or any HTTP client.

---

<p align="center">
  <strong>Built with ❤️ using Python, Scikit-learn and FastAPI</strong>
</p>
