from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from features import build_features


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "model.pkl"

app = Flask(__name__)
CORS(app)
model = joblib.load(MODEL_FILE) if MODEL_FILE.is_file() else None


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

    points[:, :, 0] = 1.0 - points[:, :, 0]

    feature_row = np.asarray(build_features(points.tolist()), dtype=float).reshape(1, -1)
    probabilities = model.predict_proba(feature_row)[0]
    best_index = int(np.argmax(probabilities))

    return jsonify(
        label=str(model.classes_[best_index]),
        confidence=float(probabilities[best_index]),
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)