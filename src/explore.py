
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/titanic.csv")


def explore_data():
    df = pd.read_csv(DATA_PATH)

    print("DATASET SHAPE")
    print(df.shape)

    print("\nFIRST FIVE ROWS")
    print(df.head())

    print("\nDATASET INFORMATION")
    df.info()

    print("\nSUMMARY STATISTICS")
    print(df.describe(include="all"))

    print("\nMISSING VALUES")
    print(df.isnull().sum())

    print("\nDUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\nSURVIVAL DISTRIBUTION")
    print(df["Survived"].value_counts())

    print("\nSURVIVAL PERCENTAGE")
    print(
        df["Survived"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nSURVIVAL RATE BY SEX")
    print(df.groupby("Sex")["Survived"].mean())

    print("\nSURVIVAL RATE BY PASSENGER CLASS")
    print(df.groupby("Pclass")["Survived"].mean())


if __name__ == "__main__":
    explore_data()
