
# 🚴 Bike Demand Prediction & ML Classification API

<p align="center">
  <strong>Machine Learning · Regression · Classification · FastAPI · Scikit-learn</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.165.0-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.9.1-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-3.1.1-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-2.5.5-013243?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

---

## 📌 Overview

This project demonstrates the development and deployment of machine learning models through a RESTful API built using **FastAPI**.

It integrates two machine learning applications into a single API:

- 🚴 **Bike Demand Prediction:** A regression model that predicts bike rental demand based on environmental, seasonal, and time-related features.
- 🧬 **Breast Cancer Classification:** A classification model that predicts whether a breast tumor is malignant or benign using cell-nuclei measurements.

Both trained models are serialized and loaded into the FastAPI application to generate predictions through HTTP requests.

## ✨ Key Features

- **Two Machine Learning Applications** – Regression and classification in one API.
- **Random Forest Regression** – Predicts bike rental demand.
- **Logistic Regression** – Classifies breast tumor measurements.
- **Data Preprocessing** – Uses StandardScaler for classification input normalization.
- **Model Persistence** – Saves and loads trained models using Joblib.
- **RESTful API** – Exposes prediction endpoints through FastAPI.
- **Input Validation** – Uses Pydantic models to validate incoming request data.
- **Interactive API Documentation** – Supports Swagger UI and automatic OpenAPI documentation.
- **Modular Prediction Workflow** – Keeps model inference accessible through separate endpoints.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | REST API development |
| Scikit-learn | Machine learning algorithms |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Joblib | Model serialization and loading |
| Pydantic | Request validation |
| Uvicorn | ASGI server |

## 🏗️ Project Architecture

```text
                ┌──────────────────────┐
                │      Client          │
                │   Postman / Browser  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │     FastAPI          │
                │   Request Validation │
                └──────────┬───────────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
       ┌──────────────────┐  ┌──────────────────┐
       │ Bike Prediction  │  │ Cancer Prediction│
       │  Regression      │  │  Classification  │
       └────────┬─────────┘  └────────┬─────────┘
                │                     │
                ▼                     ▼
       ┌──────────────────┐  ┌──────────────────┐
       │ Random Forest    │  │ Logistic         │
       │ Regressor        │  │ Regression       │
       └────────┬─────────┘  └────────┬─────────┘
                │                     │
                ▼                     ▼
       ┌──────────────────┐  ┌──────────────────┐
       │ Demand Prediction│  │ StandardScaler   │
       │                  │  │ + Classification │
       └────────┬─────────┘  └────────┬─────────┘
                │                     │
                └──────────┬──────────┘
                           ▼
                ┌──────────────────────┐
                │   JSON Response      │
                └──────────────────────┘
```

## 📂 Project Structure

```text
bike-demand-prediction/
│
├── main.py
│
├── random_forest_model.pkl
├── logistic_regression.pkl
├── scaler.pkl
│
├── requirements.txt
├── .gitignore
├── README.md
│
└── venv/
```

**Model artifacts:**

| File | Description |
|---|---|
| `random_forest_model.pkl` | Trained Random Forest Regressor |
| `logistic_regression.pkl` | Trained Logistic Regression Classifier |
| `scaler.pkl` | Fitted StandardScaler used for classification |

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd bike-demand-prediction
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

The application will run at:

```text
http://127.0.0.1:8000
```

## 📖 API Documentation

FastAPI automatically generates interactive API documentation.

| Documentation | URL |
|---|---|
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |
| Root Endpoint | http://127.0.0.1:8000/ |

## 🚴 1. Bike Demand Prediction

**Endpoint:** `POST /predict`

**Algorithm:** Random Forest Regressor

**Problem Type:** Regression

### Input Features

The model accepts 12 input features:

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

*The prediction above is illustrative; actual output depends on the trained model and input.*

## 🧬 2. Breast Cancer Classification

**Endpoint:** `POST /predict_classification`

**Algorithm:** Logistic Regression

**Problem Type:** Binary Classification

The classification model uses 30 numerical features describing characteristics of cell nuclei computed from digitized images of fine needle aspirates (FNA) of breast masses.

### Prediction Classes

| Output | Meaning |
|---|---|
| `0` | Malignant |
| `1` | Benign |

### Preprocessing

The classification model uses **StandardScaler** to standardize the input features before prediction.

Standardization transforms features using:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) = Original feature value
- \(\mu\) = Training mean
- \(\sigma\) = Training standard deviation

The same fitted scaler used during model training is loaded during API inference.

### Input Features

The API expects these 30 features in the following order:

1. mean_radius
2. mean_texture
3. mean_perimeter
4. mean_area
5. mean_smoothness
6. mean_compactness
7. mean_concavity
8. mean_concave_points
9. mean_symmetry
10. mean_fractal_dimension
11. radius_error
12. texture_error
13. perimeter_error
14. area_error
15. smoothness_error
16. compactness_error
17. concavity_error
18. concave_points_error
19. symmetry_error
20. fractal_dimension_error
21. worst_radius
22. worst_texture
23. worst_perimeter
24. worst_area
25. worst_smoothness
26. worst_compactness
27. worst_concavity
28. worst_concave_points
29. worst_symmetry
30. worst_fractal_dimension

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

*The prediction above is illustrative; actual output depends on the trained model and input.*

## 🧠 Machine Learning Workflow

The project follows a typical machine learning development workflow:

```text
Data Collection
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Selection & Preparation
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Serialization
      │
      ▼
FastAPI Deployment
      │
      ▼
Real-Time Inference
```

## 📊 Model Evaluation

The project uses different evaluation metrics depending on the machine learning problem.

### Regression Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome endpoint |
| POST | `/predict` | Predict bike rental demand |
| POST | `/predict_classification` | Classify breast tumor measurements |

## 🚀 Running the Application

Once the server is running, you can:

1. Open Swagger UI at `/docs`.
2. Select the prediction endpoint.
3. Click **Try it out**.
4. Enter the required JSON input.
5. Click **Execute**.
6. View the prediction response.

You can also test the endpoints using Postman or any HTTP client.

## 🔮 Future Improvements

- Add model comparison and experiment tracking.
- Improve prediction performance through additional feature engineering.
- Add automated model evaluation.
- Introduce Docker-based deployment.
- Deploy the API to a cloud platform.
- Add automated testing with Pytest.
- Implement CI/CD workflows.
- Add structured logging and API monitoring.

## 📄 License

This project is available under the MIT License.

---

<p align="center">
  <strong>Built with Python, Scikit-learn and FastAPI.</strong>
</p>

<p align="center">
  ⭐ If you find this project useful, consider giving it a star!
</p>
