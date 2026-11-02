
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)

MODEL_PATH = Path("models/best_model.pkl")
TEST_PATH = Path("data/processed/test.csv")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(exist_ok=True)


def evaluate():
    model = joblib.load(MODEL_PATH)

    test = pd.read_csv(TEST_PATH)

    X_test = test.drop(columns=["Survived"])
    y_test = test["Survived"]

    predictions = model.predict(X_test)

    print("\nBEST MODEL CLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Not Survived", "Survived"],
        cmap="Blues"
    )

    plt.title("Titanic Survival Confusion Matrix")
    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # ROC Curve
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title("Titanic Survival ROC Curve")
    plt.savefig(
        REPORT_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    evaluate()
