
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader

from api.schemas import PassengerInput, PredictionResponse

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "models/best_model.pkl")
)

model = None

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(api_key: str | None = Depends(api_key_header)):
    expected_key = os.getenv("API_KEY")

    if expected_key is None:
        return

    if api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run python -m src.train first."
        )

    # Load only model artifacts created by your trusted training pipeline.
    model = joblib.load(MODEL_PATH)

    print("Titanic prediction model loaded successfully.")

    yield

    model = None


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict passenger survival using a trained ML model.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)]
)
def predict(passenger: PassengerInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    input_data = pd.DataFrame([
        passenger.model_dump()
    ])

    # Match the numerical types used for MLflow model signatures.
    numeric_columns = [
        "Pclass", "Age", "SibSp", "Parch", "Fare"
    ]

    input_data[numeric_columns] = (
        input_data[numeric_columns].astype("float64")
    )

    prediction = int(model.predict(input_data)[0])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    return PredictionResponse(
        prediction=prediction,
        survived=bool(prediction),
        survival_probability=round(probability, 4),
        message=(
            "Passenger predicted to survive"
            if prediction == 1
            else "Passenger predicted not to survive"
        )
    )
