
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RAW_PATH = Path("data/raw/titanic.csv")
OUTPUT_DIR = Path("data/processed")

NUMERICAL_FEATURES = [
    "Age", "Fare", "SibSp", "Parch",
    "FamilySize", "IsAlone"
]

CATEGORICAL_FEATURES = [
    "Pclass", "Sex", "Embarked"
]

FEATURE_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]


def engineer_features(df):
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Fare"] = np.log1p(df["Fare"].clip(lower=0))

    return df


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def prepare_data():
    df = pd.read_csv(RAW_PATH)

    X = df[FEATURE_COLUMNS].copy()
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["Survived"] = y_train

    test_df = X_test.copy()
    test_df["Survived"] = y_test

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("Preprocessing data split completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining survival distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting survival distribution:")
    print(y_test.value_counts(normalize=True))


if __name__ == "__main__":
    prepare_data()
