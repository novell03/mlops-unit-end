
import os

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction(client):
    passenger = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80,
        "Embarked": "C"
    }

    response = client.post("/predict", json=passenger)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0 <= data["survival_probability"] <= 1


def test_invalid_passenger(client):
    passenger = {
        "Pclass": 5,
        "Sex": "female",
        "Age": -10,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80,
        "Embarked": "C"
    }

    response = client.post("/predict", json=passenger)

    assert response.status_code == 422


def test_api_key_required(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")

    passenger = {
        "Pclass": 2,
        "Sex": "male",
        "Age": 30,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 15,
        "Embarked": "S"
    }

    response = client.post("/predict", json=passenger)

    assert response.status_code == 401

    response = client.post(
        "/predict",
        json=passenger,
        headers={"X-API-Key": "test-secret"}
    )

    assert response.status_code == 200
