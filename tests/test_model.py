
from pathlib import Path

import joblib
import pandas as pd
import pytest


MODEL_PATH = Path("models/best_model.pkl")


@pytest.fixture
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Train model before running model tests")

    return joblib.load(MODEL_PATH)


def test_model_prediction(model):
    passenger = pd.DataFrame([{
        "Pclass": 3,
        "Sex": "male",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }])

    prediction = model.predict(passenger)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_model_probability(model):
    passenger = pd.DataFrame([{
        "Pclass": 1,
        "Sex": "female",
        "Age": 30.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Embarked": "C"
    }])

    probabilities = model.predict_proba(passenger)

    assert probabilities.shape == (1, 2)
    assert abs(probabilities.sum() - 1.0) < 1e-6
