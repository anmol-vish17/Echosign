
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CSV_FILES = [
    ROOT / "data" / "samples.csv",
    ROOT / "data" / "samples_member2.csv",
]
MODEL_FILE = ROOT / "model.pkl"
MATRIX_FILE = ROOT / "confusion_matrix.png"

EXPECTED_COLUMNS = 127


def main():
    frames = []

    for file in CSV_FILES:
        if file.is_file():
            frames.append(pd.read_csv(file, header=None))
            print(f"Loaded: {file.name}")

    if not frames:
        raise SystemExit("No CSV dataset found.")

    data = pd.concat(frames, ignore_index=True)

    if data.shape[1] != EXPECTED_COLUMNS:
        raise SystemExit(
            f"Expected 127 columns, found {data.shape[1]}."
        )

    labels = data.iloc[:, 0].astype(str).str.strip()
    features = data.iloc[:, 1:].apply(
        pd.to_numeric, errors="coerce"
    )

    if features.isna().any().any():
        raise SystemExit("Dataset contains invalid feature values.")

    values = features.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise SystemExit("Dataset contains non-finite values.")

    counts = labels.value_counts()

    if (counts < 5).any():
        raise SystemExit("Some signs have fewer than 5 samples.")

    x_train, x_test, y_train, y_test = train_test_split(
        values,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("\nTotal samples:", len(data))
    print("Total signs:", labels.nunique())
    print("\nSign counts:")
    print(counts.sort_index().to_string())

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print(
        "Accuracy:",
        f"{accuracy_score(y_test, predictions):.2%}",
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=model.classes_,
        xticks_rotation=45,
    )

    plt.tight_layout()
    plt.savefig(MATRIX_FILE)
    plt.close()

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved:", MODEL_FILE)
    print("Confusion matrix saved:", MATRIX_FILE)


if __name__ == "__main__":
    main()
