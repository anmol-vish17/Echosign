
from pathlib import Path
import csv

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from features import build_features


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "model.pkl"
DATA_FOLDER = ROOT / "data"

CSV_FILES = [
    "samples.csv",
    "samples_member2.csv",
]

app = Flask(__name__)
CORS(app)

model = joblib.load(MODEL_FILE) if MODEL_FILE.is_file() else None


# ---------------- LIBRARY DATA ----------------

library_samples = []
sample_lookup = {}


def load_library_data():
    global library_samples, sample_lookup

    library_samples = []
    sample_lookup = {}

    for filename in CSV_FILES:
        file_path = DATA_FOLDER / filename

        if not file_path.is_file():
            print(f"CSV file not found: {file_path}")
            continue

        with open(file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)

            for row_number, row in enumerate(reader, start=1):
                if not row:
                    continue

                if len(row) != 127:
                    continue

                try:
                    sign_name = row[0].strip()
                    features = [float(value) for value in row[1:]]

                    if not sign_name or not np.isfinite(features).all():
                        continue

                    sample_id = f"{Path(filename).stem}_{row_number}"

                    sample = {
                        "sample_id": sample_id,
                        "sign": sign_name,
                        "source_file": filename,
                        "features": features,
                    }

                    library_samples.append(sample)
                    sample_lookup[sample_id] = sample

                except (ValueError, TypeError):
                    continue

    print(f"Library loaded: {len(library_samples)} samples")


load_library_data()


# ---------------- EXISTING APIs ----------------

@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    hands = payload.get("hands", [])

    if not isinstance(hands, list):
        return jsonify(error="'hands' must be a list."), 400

    if not hands:
        return jsonify(label=None, confidence=0.0)

    if len(hands) > 2:
        return jsonify(error="At most two hands are supported."), 400

    try:
        points = np.asarray(hands, dtype=float)
    except (TypeError, ValueError):
        return jsonify(error="Hand points must contain numeric [x, y, z] coordinates."), 400

    if points.shape != (len(hands), 21, 3):
        return jsonify(error="Each hand must contain exactly 21 [x, y, z] points."), 400

    if not np.isfinite(points).all():
        return jsonify(error="Hand coordinates must be finite numbers."), 400

    if model is None:
        return jsonify(error="No trained model found. Run train.py first."), 503

    feature_row = np.asarray(
        build_features(points.tolist()), dtype=float
    ).reshape(1, -1)

    probabilities = model.predict_proba(feature_row)[0]
    best_index = int(np.argmax(probabilities))

    return jsonify(
        label=str(model.classes_[best_index]),
        confidence=float(probabilities[best_index]),
    )


# ---------------- LIBRARY APIs ----------------

@app.get("/library/signs")
def get_library_signs():
    sign_counts = {}

    for sample in library_samples:
        sign = sample["sign"]
        sign_counts[sign] = sign_counts.get(sign, 0) + 1

    signs = [
        {
            "sign": sign,
            "sample_count": count,
        }
        for sign, count in sorted(sign_counts.items())
    ]

    return jsonify(
        total_signs=len(signs),
        total_samples=len(library_samples),
        signs=signs,
    )


@app.get("/library/signs/<sign_name>")
def get_sign_samples(sign_name):
    matching_samples = [
        sample for sample in library_samples
        if sample["sign"].casefold() == sign_name.casefold()
    ]

    if not matching_samples:
        return jsonify(error="Sign not found."), 404

    return jsonify(
        sign=matching_samples[0]["sign"],
        sample_count=len(matching_samples),
        samples=matching_samples,
    )


@app.get("/library/sample/<sample_id>")
def get_single_sample(sample_id):
    sample = sample_lookup.get(sample_id)

    if sample is None:
        return jsonify(error="Sample not found."), 404

    return jsonify(sample)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
