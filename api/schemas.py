
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class PassengerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=120)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float | None = Field(default=None, ge=0)
    Embarked: Literal["C", "Q", "S"] | None = None


class PredictionResponse(BaseModel):
    prediction: int
    survived: bool
    survival_probability: float
    message: str
