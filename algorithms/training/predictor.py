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
    payload = request.get_json()
    df = pd.DataFrame(payload)
    preds = load_model().predict(df)
    return jsonify({"predictions": preds.tolist()})
