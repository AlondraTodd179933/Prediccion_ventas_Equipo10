"""
Training script.

This script:
- Loads the prepared monthly dataset
- Builds lag features if needed
- Splits train/validation
- Trains a RandomForestRegressor
- Saves the trained model bundle to joblib
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from algorithms.utils.features import make_lag_features
from algorithms.utils.logging_config import setup_logger

DEFAULT_DATA_PATH = Path("artifacts/data/monthly_clean.csv")
DEFAULT_MODEL_DIR = Path("artifacts/models")
MODEL_FILENAME = "model.joblib"


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target/leakage columns and keep model features."""
    drop_cols = ["item_cnt_month", "y", "target", "label"]
    return df.drop(columns=drop_cols, errors="ignore")


def main(
    input_path: str | None = None,
    output_path: str | None = None,
    n_estimators: int = 200,
    max_depth: int | None = 6,
) -> None:
    """Run training end-to-end."""
    start_time = time.time()
    logger = setup_logger("train")

    data_file = Path(input_path) if input_path else DEFAULT_DATA_PATH
    model_file = Path(output_path) if output_path else (DEFAULT_MODEL_DIR / MODEL_FILENAME)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"No encontré data en: {data_file.as_posix()}")

    logger.info("Iniciando entrenamiento")
    logger.info("Usando dataset: %s", data_file.as_posix())
    logger.info("Hiperparámetros: n_estimators=%s, max_depth=%s", n_estimators, max_depth)

    df = pd.read_csv(data_file)

    if not {"lag_1", "lag_2", "lag_3", "lag_mean_1_2"}.issubset(df.columns):
        logger.info("No encontré features lag en el dataset. Las voy a construir.")
        df = make_lag_features(df)

    if "item_cnt_month" not in df.columns:
        raise ValueError("El dataset no contiene la columna objetivo 'item_cnt_month'.")

    X = _build_features(df)
    y = df["item_cnt_month"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Entrenando modelo...")
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = mse ** 0.5
    logger.info("RMSE validación: %.4f", rmse)

    bundle = {
        "model": model,
        "features": list(X.columns),
        "rmse_val": rmse,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    joblib.dump(bundle, model_file)
    logger.info("Modelo guardado en: %s", model_file.as_posix())

    duration = time.time() - start_time
    logger.info("Entrenamiento finalizado en %.2f segundos", duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default=None,
        help="Ruta al dataset de entrenamiento",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Ruta donde guardar el modelo .joblib",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Número de árboles del Random Forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Profundidad máxima de los árboles",
    )

    args = parser.parse_args()

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )