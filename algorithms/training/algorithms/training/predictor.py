from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("/opt/ml/model/model.joblib")
model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/ping", methods=["GET"])
def ping():
    try:
        load_model()
        return "OK", 200
    except Exception as e:
        return str(e), 500


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        payload = request.get_json()

        if payload is None:
            return jsonify({"error": "No se recibió JSON válido"}), 400

        # payload esperado: lista de diccionarios
        # ejemplo:
        # [
        #   {"lag_1": 19.9, "lag_3": 3.0, "lag_6": 19.9, "lag_12": 0.0}
        # ]
        df = pd.DataFrame(payload)

        preds = load_model().predict(df)

        return jsonify({"predictions": preds.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500