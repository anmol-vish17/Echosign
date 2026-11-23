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
SAMPLES_FILE = ROOT / "data" / "final_dataset.csv"
MODEL_FILE = ROOT / "model.pkl"
MATRIX_FILE = ROOT / "confusion_matrix.png"
PROJECT_LABELS = (
    "CASH",
    "I",
    "WANT",
    "NEED",
    "WATER",
    "FOOD",
    "HELP",
    "MEDICINE",
    "TOILET",
    "YES",
    "NO",
    "HELLO",
    "THANK YOU",
    "PLEASE",
    "GOOD",
    "BAD",    
    "GOOD MORNING",
    "HI",
    "SORRY",
    "TIME",
    "HURT",
    "STOP",
    "IDLE",
)
EXPECTED_COLUMNS = 127


def main():
    if not SAMPLES_FILE.is_file():
        raise SystemExit(f"Sample file not found: {SAMPLES_FILE}")

    try:
        data = pd.read_csv(SAMPLES_FILE)
    except pd.errors.EmptyDataError:
        raise SystemExit("The sample file is empty. Record sign samples first.")

    if data.shape[1] != EXPECTED_COLUMNS:
        raise SystemExit(
            f"Expected {EXPECTED_COLUMNS} columns (label + 126 features), "
            f"but found {data.shape[1]}."
        )

    labels = data.iloc[:, 0].astype(str).str.strip().str.upper()
    extra_labels = sorted(set(labels) - set(PROJECT_LABELS))
    if extra_labels:
        print("Excluding out-of-scope labels from training and testing: " + ", ".join(extra_labels))

    project_rows = labels.isin(PROJECT_LABELS)
    data = data.loc[project_rows].copy()
    labels = labels.loc[project_rows]
    class_labels = [label for label in PROJECT_LABELS if label in set(labels)]
    missing_labels = [label for label in PROJECT_LABELS if label not in class_labels]
    if missing_labels:
        raise SystemExit("Missing planned labels: " + ", ".join(missing_labels))

    counts = labels.value_counts()
    low_count_labels = [label for label in class_labels if counts[label] < 5]
    if low_count_labels:
        raise SystemExit("Need at least 5 samples for: " + ", ".join(low_count_labels))

    features = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise SystemExit("Feature data contains missing or non-numeric values.")
    feature_values = features.to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise SystemExit("Feature data contains infinite or non-finite values.")

    x_train, x_test, y_train, y_test = train_test_split(
        feature_values,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=42,
    )

    model = RandomForestClassifier(
    n_estimators=250,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Training on {len(class_labels)} labels: " + ", ".join(class_labels))
    print("Per-class results from a random 80/20 split:")
    print(classification_report(y_test, predictions, labels=class_labels, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("Note: a random split can overestimate accuracy; also test on data from a person excluded from training.")

    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=class_labels,
        display_labels=class_labels,
        xticks_rotation=45,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(MATRIX_FILE)
    plt.close()

    joblib.dump(model, MODEL_FILE)
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved confusion matrix: {MATRIX_FILE}")


if __name__ == "__main__":
    main()
