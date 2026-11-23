<h1 align="center">🚴 Bike Demand Prediction & Breast Cancer Classification API</h1>

<p align="center">
  <strong>Regression · Classification · FastAPI · Scikit-learn</strong><br/>
  Two trained machine learning models served through a single REST API.
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

## Results at a Glance

| Task | Selected Model | Key Metrics |
|---|---|---|
| 🚴 Bike demand (regression) | Random Forest Regressor | **R² 0.9448** · MAE 24.77 · RMSE 41.79 |
| 🧬 Breast cancer (classification) | Logistic Regression | **98.25% test accuracy** · weighted F1 0.99 |

Full comparisons are in [Model Evaluation](#model-evaluation).

---

## Quick Start

**1. Clone the repository**

Git LFS is required so the `.pkl` model files download correctly.

```bash
git lfs install
git clone https://github.com/sidharth-chauhan/ml-prediction-api.git
cd ml-prediction-api
git lfs pull
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies and run**

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

**4. Open the API**

| Page | URL |
|---|---|
| Swagger UI (try requests in the browser) | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |
| Root | http://127.0.0.1:8000/ |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Welcome message |
| `POST` | `/predict` | Predict bike rental demand |
| `POST` | `/predict_classification` | Classify a tumor as malignant or benign |

### `POST /predict` : Bike Demand

**Request body**

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

**cURL**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"season":2,"yr":1,"mnth":6,"hr":17,"holiday":0,"weekday":3,"workingday":1,"weathersit":1,"temp":0.65,"atemp":0.62,"hum":0.45,"windspeed":0.15}'
```

**Response**

```json
{
  "prediction": 245.0
}
```

### `POST /predict_classification` : Breast Cancer

**Request body**

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

**Response**

```json
{
  "prediction": "Benign",
  "raw_prediction": 1
}
```

| `raw_prediction` | Meaning |
|---|---|
| `0` | Malignant |
| `1` | Benign |

> Example responses are illustrative. Actual output depends on the trained model and the values you send.

---

## How Your Inputs Drive the Prediction

You choose the input values, and the model predicts from exactly those values. Change an input and the prediction changes.

### Bike demand: what you choose

| Field | What you choose | Values |
|---|---|---|
| `season` | Season | 1 = spring, 2 = summer, 3 = fall, 4 = winter |
| `yr` | Year | 0 = 2011, 1 = 2012 |
| `mnth` | Month | 1 to 12 |
| `hr` | Hour of the day | 0 to 23 |
| `holiday` | Holiday | 0 = no, 1 = yes |
| `weekday` | Day of the week | 0 to 6 |
| `workingday` | Working day | 0 = no, 1 = yes |
| `weathersit` | Weather | 1 = clear, 2 = mist or cloudy, 3 = light rain or snow, 4 = heavy rain or snow |
| `temp` | Temperature | Normalized, 0 to 1 |
| `atemp` | Feeling temperature | Normalized, 0 to 1 |
| `hum` | Humidity | Normalized, 0 to 1 |
| `windspeed` | Wind speed | Normalized, 0 to 1 |

Value meanings follow the UCI Bike Sharing dataset documentation in `bike+sharing+dataset/Readme.txt`.

### Breast cancer: what you choose

You provide 30 measurements of cell nuclei, computed from digitized images of fine needle aspirates (FNA) of breast masses. Each of the 10 characteristics is given in three forms: the **mean**, the **standard error** and the **worst** (largest) value.

| # | Characteristic | Mean | Error | Worst |
|---|---|---|---|---|
| 1 | Radius | `mean_radius` | `radius_error` | `worst_radius` |
| 2 | Texture | `mean_texture` | `texture_error` | `worst_texture` |
| 3 | Perimeter | `mean_perimeter` | `perimeter_error` | `worst_perimeter` |
| 4 | Area | `mean_area` | `area_error` | `worst_area` |
| 5 | Smoothness | `mean_smoothness` | `smoothness_error` | `worst_smoothness` |
| 6 | Compactness | `mean_compactness` | `compactness_error` | `worst_compactness` |
| 7 | Concavity | `mean_concavity` | `concavity_error` | `worst_concavity` |
| 8 | Concave points | `mean_concave_points` | `concave_points_error` | `worst_concave_points` |
| 9 | Symmetry | `mean_symmetry` | `symmetry_error` | `worst_symmetry` |
| 10 | Fractal dimension | `mean_fractal_dimension` | `fractal_dimension_error` | `worst_fractal_dimension` |

**Preprocessing:** the values are standardized with the fitted `StandardScaler` saved during training, so inputs are transformed exactly as the training data was.

```text
z = (x - mean) / std
```

> This API is a machine learning demonstration and not a medical diagnostic tool.

---

## Model Evaluation

### Bike Demand Prediction (Regression)

| Metric | Result |
|---|---|
| MAE | 24.77 |
| RMSE | 41.79 |
| R² Score | 0.9448 |

### Breast Cancer Classification

| Algorithm | Training Accuracy | Testing Accuracy |
|---|---|---|
| **Logistic Regression** | 98.90% | **98.25%** |
| **SVM** | 98.24% | **98.25%** |
| Random Forest | 100% | 95.61% |
| KNN | 97.36% | 95.61% |
| Gaussian Naive Bayes | 94.07% | 93.86% |
| Decision Tree | 100% | 91.23% |

**Logistic Regression and SVM: detailed metrics**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Malignant | 0.98 | 0.98 | 0.98 |
| Benign | 0.99 | 0.99 | 0.99 |
| **Overall (weighted)** | | | **0.99** |

**Conclusion:** Logistic Regression and SVM achieved the highest test accuracy (98.25%). Logistic Regression was selected as the final model.

---

## Model Selection

| Task | Algorithms Explored | Selected Model |
|---|---|---|
| Bike demand | Linear Regression, Decision Tree, Random Forest | **Random Forest Regressor** |
| Breast cancer | Logistic Regression, SVM, Random Forest, KNN, Gaussian Naive Bayes, Decision Tree | **Logistic Regression** |

**Why Random Forest for bike demand?**
It is an ensemble of decision trees that averages their predictions. It captures nonlinear relationships and feature interactions (for example hour × working day × weather) and works well on tabular data.

**Why Logistic Regression for cancer classification?**
It matched the best test accuracy of all six algorithms with a small gap between training and testing scores (98.90% vs 98.25%). By contrast, Random Forest and Decision Tree reached 100% training accuracy but dropped on the test set, a sign of overfitting. Logistic Regression is also interpretable and works well on standardized features.

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

## Machine Learning Workflow

```text
Data Collection → Exploratory Data Analysis → Preprocessing → Feature Preparation
      → Train-Test Split → Model Training → Evaluation → Model Selection
      → Serialization (Joblib) → FastAPI Deployment → Inference
```

---

## Project Structure

```text
ml-prediction-api/
│
├── main.py                     # FastAPI app and endpoints
├── predict.py                  # Prediction logic
│
├── random_forest_model.pkl     # Bike demand model      (Git LFS)
├── logistic_regression.pkl     # Cancer classifier      (Git LFS)
├── scaler.pkl                  # Fitted StandardScaler  (Git LFS)
│
├── bike+sharing+dataset/
│   ├── Readme.txt
│   ├── day.csv
│   └── hour.csv
│
├── notebook_regression/        # Regression experiments
├── notebook_classification/    # Classification experiments
│
├── requirements.txt
├── .gitattributes
├── .gitignore
└── README.md
```

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| FastAPI | REST API framework |
| Scikit-learn | ML algorithms and preprocessing |
| Pandas / NumPy | Data manipulation and numerical computing |
| Joblib | Model serialization and loading |
| Pydantic | Request validation |
| Uvicorn | ASGI server |
| Git LFS | Large model file storage |

---

**Sidharth Chauhan**
