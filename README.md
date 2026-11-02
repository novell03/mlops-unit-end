# Titanic Survival Prediction — End-to-End MLOps Pipeline

## 1. Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline to predict whether passengers survived the Titanic disaster using demographic and travel-related information.

The system covers data exploration, preprocessing, classification model development, experiment tracking, dataset versioning, automated testing, API deployment, model monitoring, and documentation.

The objective is not only to build an accurate classification model but also to demonstrate a reproducible, maintainable, and monitorable machine learning workflow.

## 2. Problem Statement

Develop a machine learning classification system capable of predicting Titanic passenger survival using demographic and travel features.

The project aims to:

- Identify important factors influencing passenger survival.
- Train and evaluate classification algorithms.
- Track model experiments and versions.
- Version datasets for reproducibility.
- Automate testing and training through CI/CD.
- Deploy the model through a REST API.
- Monitor data drift and prediction performance.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas / NumPy | Data manipulation and analysis |
| Scikit-learn | Preprocessing, model training, evaluation |
| Matplotlib / Seaborn | Data visualization |
| MLflow | Experiment tracking and model registry |
| DVC | Dataset version control |
| Git / GitHub | Source code version control |
| GitHub Actions | CI/CD automation |
| FastAPI | REST API for predictions |
| Pydantic | Request validation |
| Uvicorn | ASGI application server |
| Evidently AI | Data drift and model performance monitoring |
| Pytest | Automated testing |
| Joblib | Model serialization |

## 4. Dataset Description

The Titanic dataset contains 891 passenger records and 12 attributes.

The target variable is `Survived`:

- `0` — Passenger did not survive.
- `1` — Passenger survived.

Important features include passenger class, sex, age, fare, family relationships, and embarkation port.

### Missing Values

| Feature | Missing Records |
|---|---:|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

The original dataset contains 891 passengers.

The dataset is split into:

- Training: 712 records (80%).
- Testing: 179 records (20%).

A stratified split preserves the survival class proportions.

## 5. Project Architecture

The MLOps workflow follows this structure:

```text
Titanic Dataset
       |
       v
Data Exploration & Preprocessing
       |
       v
Feature Engineering
       |
       v
Model Training & Evaluation
       |
       +----> MLflow Experiment Tracking
       |
       +----> DVC Dataset Versioning
       |
       v
Best Classification Model
       |
       v
FastAPI Prediction Service
       |
       v
Model Predictions
       |
       v
Evidently AI Monitoring
       |
       v
Monitoring Reports & Drift Alerts

GitHub Actions:
Automated Tests → Training → Evaluation → Artifacts
```

## 6. Project Structure

```text
titanic-mlops/
├── api/
│   ├── main.py
│   └── schemas.py
├── src/
│   ├── __init__.py
│   ├── explore.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── monitor.py
│   └── monitor_alerts.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_api.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── best_model.pkl
├── reports/
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── monitoring/
│       ├── monitoring_report.html
│       ├── monitoring_report.json
│       └── drift_alert.json
├── .github/
│   └── workflows/
│       ├── mlops.yml
│       └── monitoring.yml
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

DVC metadata files and configuration are also maintained in the repository.

## 7. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/titanic-mlops.git
cd titanic-mlops
```

### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

Python 3.12 is used in the GitHub Actions configuration. The locked dependencies should be generated and tested using a compatible environment.

### Restore the Dataset

If a DVC remote has been configured and populated:

```bash
dvc pull
```

Otherwise, place `titanic.csv` inside `data/raw/`.

## 8. Data Exploration and Preprocessing

Run data exploration:

```bash
python -m src.explore
```

Run preprocessing:

```bash
python -m src.preprocess
```

Preprocessing includes:

- Missing-value imputation.
- Categorical feature encoding.
- Numerical feature scaling.
- Feature engineering.
- Fare transformation.
- Stratified train/test splitting.

Engineered features include:

- `FamilySize`: Number of family members traveling together, including the passenger.
- `IsAlone`: Whether the passenger traveled alone.

The preprocessing operations are included in the trained model pipeline to ensure consistent inference.

## 9. Model Development and Evaluation

Two classification algorithms are implemented:

1. Logistic Regression.
2. Random Forest Classifier.

### Start MLflow

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

MLflow dashboard:

http://127.0.0.1:5000

### Train Models

Open another terminal:

```bash
source .venv/bin/activate
python -m src.train
```

### Evaluate the Best Model

```bash
python -m src.evaluate
```

The following metrics are calculated:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.

The implemented selection rule chooses the model with the highest test-set F1-score.

For a more rigorous production workflow, model selection should use training-set cross-validation and reserve the test set for final evaluation.

### Model Results

Record the actual metrics from `reports/model_metrics.json`.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | To be filled | To be filled |
| Precision | To be filled | To be filled |
| Recall | To be filled | To be filled |
| F1-score | To be filled | To be filled |
| ROC-AUC | To be filled | To be filled |

Selected model: **To be filled from training output.**

The trained model is saved as:

`models/best_model.pkl`

## 10. MLflow Model Tracking and Versioning

MLflow records:

- Experiment names and run identifiers.
- Model hyperparameters.
- Evaluation metrics.
- Model artifacts.
- Input/output signatures.
- Registered model versions.

The registered model name is:

`TitanicSurvivalClassifier`

The registered model can be inspected in the MLflow dashboard under the Models section.

## 11. DVC Data Versioning

Initialize DVC:

```bash
dvc init
```

Track the datasets:

```bash
dvc add data/raw/titanic.csv
dvc add data/processed/train.csv
dvc add data/processed/test.csv
```

Check the tracked data:

```bash
dvc status
```

After configuring a remote:

```bash
dvc push
```

Restore tracked datasets:

```bash
dvc pull
```

DVC stores content hashes and dataset metadata, while Git tracks the associated `.dvc` files.

A remote accessible to GitHub Actions is required for automated data restoration.

## 12. CI/CD Pipeline

GitHub Actions is used to automate the machine learning workflow.

The CI pipeline performs:

1. Repository checkout.
2. Python environment setup.
3. Dependency installation.
4. DVC dataset restoration.
5. Preprocessing tests.
6. Data preparation.
7. Model training.
8. Model and API tests.
9. Model evaluation.
10. Artifact upload.

Workflow file:

`.github/workflows/mlops.yml`

### Run Tests Locally

```bash
python -m pytest tests/ -v
```

### Trigger CI

```bash
git add .
git commit -m "Update Titanic MLOps pipeline"
git push origin main
```

Pushing to the `main` branch triggers the workflow.

The workflow performs continuous integration and produces downloadable artifacts. Automatic production deployment is not yet configured.

## 13. FastAPI Model Deployment

The trained model is exposed through a REST API.

### Start the API

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### Available Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check model availability |
| `/predict` | POST | Predict passenger survival |
| `/docs` | GET | Interactive API documentation |

Swagger UI:

http://127.0.0.1:8000/docs

### Example Prediction Request

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 80,
  "Embarked": "C"
}
```

### Example Response Format

```json
{
  "prediction": 1,
  "survived": true,
  "survival_probability": 0.91,
  "message": "Passenger predicted to survive"
}
```

The example values are illustrative and may differ from the actual model output.


A production deployment should require authentication, use HTTPS, and include rate limiting.

## 14. Evidently AI Model Monitoring

Evidently AI is used to compare reference data with current data.

Monitoring includes:

- Numerical feature drift.
- Categorical feature drift.
- Prediction probability drift.
- Classification performance metrics when labels are available.
- Drift alerts.

### Generate Monitoring Reports

```bash
python -m src.monitor
```

### Generate Drift Alerts

```bash
python -m src.monitor_alerts
```

### Open the HTML Report

On macOS:

```bash
open reports/monitoring/monitoring_report.html
```

Monitoring artifacts include:

- `monitoring_report.html`
- `monitoring_report.json`
- `drift_alert.json`

The alert script uses a 30% drifted-feature threshold and a statistical significance threshold of 0.05.

### Scheduled Monitoring

The GitHub Actions monitoring workflow is configured to run weekly.

Workflow file:

`.github/workflows/monitoring.yml`

The current implementation compares static training and test datasets. Continuous production monitoring would require collecting new prediction inputs and actual outcomes.

## 15. Testing

Run all automated tests:

```bash
python -m pytest tests/ -v
```

The test suite covers:

- Feature engineering.
- Missing-value preprocessing.
- Model prediction validity.
- Prediction probability validity.
- API health checks.
- API input validation.
- API-key authentication.

## 16. Reproducibility and Best Practices

The project follows these practices:

- Fixed random seed for reproducible data splitting and training.
- Stratified train/test splitting.
- Scikit-learn pipelines for consistent preprocessing.
- MLflow experiment tracking.
- DVC dataset versioning.
- Git-based source control.
- Dependency version locking.
- Automated testing.
- API input validation.
- Monitoring and drift reporting.
- Separation of training, evaluation, serving, and monitoring code.

Model artifacts should only be loaded from trusted sources.

## 17. Limitations

- The dataset contains only 891 records.
- Missing age and cabin information limits the available features.
- The model is trained on historical passenger data.
- The API currently runs locally.
- Monitoring uses a static dataset rather than live production traffic.
- Model selection currently uses the test set.
- Continuous deployment to a public hosting environment has not been implemented.

## 18. Future Improvements

Potential improvements include:

- Hyperparameter optimization with cross-validation.
- XGBoost model comparison.
- Additional feature engineering using passenger titles.
- Docker containerization.
- Cloud deployment on AWS, Azure, or another provider.
- Persistent MLflow tracking infrastructure.
- Automated retraining triggered by meaningful data drift.
- Real-time prediction logging.
- Centralized monitoring dashboards.
- Stronger API authentication and observability.

## 19. Conclusion

This project demonstrates the core stages of an end-to-end MLOps lifecycle using the Titanic survival classification problem.

It integrates data preparation, machine learning, experiment tracking, dataset versioning, automated testing, model serving, and monitoring into a structured Python project.

The implementation provides a foundation for building reproducible and maintainable machine learning applications.

## 20. Commands Summary

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements-lock.txt

# Explore data
python -m src.explore

# Preprocess data
python -m src.preprocess

# Start MLflow
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts

# Train models
python -m src.train

# Evaluate model
python -m src.evaluate

# Check DVC status
dvc status

# Push DVC data to remote
dvc push

# Restore DVC data
dvc pull

# Run tests
python -m pytest tests/ -v

# Start API
uvicorn api.main:app --reload

# Generate monitoring report
python -m src.monitor

# Generate drift alerts
python -m src.monitor_alerts

# Open monitoring report on macOS
open reports/monitoring/monitoring_report.html

# Commit and push code
git add .
git commit -m "Complete Titanic MLOps project"
git push origin main
```

---

**Project:** Titanic Survival Prediction using MLOps  
**Dataset:** Titanic passenger dataset  
**Tools:** Python, Scikit-learn, MLflow, DVC, FastAPI, GitHub Actions, Evidently AI  
**Status:** Local implementation and testing; hosted deployment and live monitoring remain future improvements.
