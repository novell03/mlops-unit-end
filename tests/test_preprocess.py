
import numpy as np
import pandas as pd

from src.preprocess import (
    engineer_features,
    build_preprocessor
)


def test_feature_engineering():
    df = pd.DataFrame({
        "SibSp": [1, 0],
        "Parch": [2, 0],
        "Fare": [20.0, 10.0]
    })

    result = engineer_features(df)

    assert result["FamilySize"].tolist() == [4, 1]
    assert result["IsAlone"].tolist() == [0, 1]
    assert np.isclose(result["Fare"].iloc[0], np.log1p(20))


def test_preprocessing_pipeline():
    df = pd.DataFrame({
        "Age": [22, np.nan, 35],
        "Fare": [7.25, 15.0, 30.0],
        "SibSp": [1, 0, 0],
        "Parch": [0, 0, 1],
        "FamilySize": [2, 1, 2],
        "IsAlone": [0, 1, 0],
        "Pclass": [3, 2, 1],
        "Sex": ["male", "female", "female"],
        "Embarked": ["S", "C", None]
    })

    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(df)

    assert result.shape[0] == 3
    assert np.isfinite(result.toarray() if hasattr(result, "toarray") else result).all()
