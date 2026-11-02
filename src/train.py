



import json
from pathlib import Path
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.features import FeatureEngineer
from src.preprocess import build_preprocessor


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

#mlflow.set_tracking_uri("http://127.0.0.1:5000")



mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)

mlflow.set_experiment("Titanic Survival Prediction")


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train.drop(columns=["Survived"])
    y_train = train["Survived"]

    X_test = test.drop(columns=["Survived"])
    y_test = test["Survived"]

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        )
    }

    return metrics, predictions


def train_models():
    X_train, X_test, y_train, y_test = load_data()

    numeric_columns = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    X_train[numeric_columns] = X_train[numeric_columns].astype("float64")
    X_test[numeric_columns] = X_test[numeric_columns].astype("float64")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    trained_models = {}

    for name, classifier in models.items():

        with mlflow.start_run(run_name=name):

            pipeline = Pipeline([
                ("feature_engineering", FeatureEngineer()),
                ("preprocessing", build_preprocessor()),
                ("classifier", classifier)
            ])

            pipeline.fit(X_train, y_train)

            metrics, predictions = evaluate_model(
                pipeline, X_test, y_test
            )

            results[name] = metrics
            trained_models[name] = pipeline

            # Log model parameters
            mlflow.log_params(classifier.get_params())

            # Log evaluation metrics
            mlflow.log_metrics(metrics)

            mlflow.set_tag("model_type", name)
            mlflow.set_tag("dataset", "Titanic")

            # Model input/output signature
            signature = infer_signature(
                X_train,
                pipeline.predict(X_train)
            )

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=X_train.head(5),
                serialization_format="cloudpickle",
                registered_model_name="TitanicSurvivalClassifier",
                skops_trusted_types=[
                    "numpy.dtype",
                    "src.features.FeatureEngineer"
                ]
            )

            # mlflow.sklearn.log_model(
            #     sk_model=pipeline,
            #     name="model",
            #     signature=signature,
            #     input_example=X_train.head(3),
            #     serialization_format="cloudpickle"
            # )

            # Log trained model
            # mlflow.sklearn.log_model(
            #     sk_model=pipeline,
            #     name="model",
            #     signature=signature,
            #     input_example=X_train.head(5)
            # )



            print(f"\n========== {name} ==========")

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    predictions,
                    zero_division=0
                )
            )

    best_name = max(
        results,
        key=lambda name: results[name]["f1_score"]
    )

    best_model = trained_models[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.pkl"
    )

    with open(
        REPORT_DIR / "model_metrics.json", "w"
    ) as file:
        json.dump(results, file, indent=4)

    print("\n========== BEST MODEL ==========")
    print(f"Best Model: {best_name}")
    print(f"F1 Score: {results[best_name]['f1_score']:.4f}")
    print("Model saved successfully.")


if __name__ == "__main__":
    train_models()
