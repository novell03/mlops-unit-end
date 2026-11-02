
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.core.datasets import BinaryClassification
from evidently.metrics import (
    DriftedColumnsCount,
    ValueDrift,
    Accuracy,
    Precision,
    Recall,
    F1Score
)

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/best_model.pkl")
REPORT_DIR = Path("reports/monitoring")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

NUMERICAL = [
    "Age", "SibSp", "Parch", "Fare"
]

CATEGORICAL = [
    "Pclass", "Sex", "Embarked"
]


def load_data():
    reference = pd.read_csv(DATA_DIR / "train.csv")
    current = pd.read_csv(DATA_DIR / "test.csv")

    return reference, current


def prepare_monitoring_data(df, model):
    df = df.copy()

    X = df[FEATURES].copy()

    # Ensure the same numeric input types as training
    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        X[col] = X[col].astype("float64")

    df["prediction"] = model.predict(X).astype(int)

    df["prediction_probability"] = (
        model.predict_proba(X)[:, 1]
    )

    return df


def generate_monitoring_report():
    reference, current = load_data()

    model = joblib.load(MODEL_PATH)

    reference = prepare_monitoring_data(reference, model)
    current = prepare_monitoring_data(current, model)

    definition = DataDefinition(
        numerical_columns=NUMERICAL + [
            "prediction_probability"
        ],
        categorical_columns=CATEGORICAL + [
            "prediction"
        ],
        classification=[
            BinaryClassification(
                target="Survived",
                prediction_labels="prediction",
                prediction_probas="prediction_probability"
            )
        ]
    )

    reference_dataset = Dataset.from_pandas(
        reference,
        data_definition=definition
    )

    current_dataset = Dataset.from_pandas(
        current,
        data_definition=definition
    )

    report = Report(metrics=[
        DriftedColumnsCount(
            columns=FEATURES
        ),
        ValueDrift(column="Age"),
        ValueDrift(column="Fare"),
        ValueDrift(column="Sex"),
        ValueDrift(column="prediction_probability"),
        Accuracy(),
        Precision(),
        Recall(),
        F1Score()
    ])

    snapshot = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset
    )

    snapshot.save_html(
        str(REPORT_DIR / "monitoring_report.html")
    )

    snapshot.save_json(
        str(REPORT_DIR / "monitoring_report.json")
    )

    print("Monitoring reports generated successfully.")
    print("HTML: reports/monitoring/monitoring_report.html")
    print("JSON: reports/monitoring/monitoring_report.json")


if __name__ == "__main__":
    generate_monitoring_report()
