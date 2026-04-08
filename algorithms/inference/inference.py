"""
Inference script.

Loads the trained model and the monthly dataset, ensures feature
columns match those used during training (including lag features),
and writes predictions to CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from algorithms.utils.features import make_lag_features
from algorithms.utils.logging_config import setup_logger

DEFAULT_DATA_PATH = Path("artifacts/data/monthly_clean.csv")
DEFAULT_MODEL_DIR = Path("artifacts/models")
DEFAULT_OUTPUT_DIR = Path("artifacts/predictions")
PREDICTIONS_FILENAME = "predictions.csv"


def _latest_file(folder: Path, pattern: str) -> Optional[Path]:
    """Return the newest file in a folder matching a glob pattern."""
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_model(model_path: Path):
    """
    Load model from joblib.

    Supports bundles saved as dict with keys like:
    model, estimator, pipeline, ridge, regressor, clf.
    """
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        for key in ("model", "estimator", "pipeline", "ridge", "regressor", "clf"):
            if key in bundle:
                return bundle[key]
        keys = list(bundle.keys())
        raise ValueError(f"El .joblib es un dict pero no trae estimador. Keys: {keys}")

    return bundle


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the features dataframe for prediction."""
    drop_cols = ["item_cnt_month", "y", "target", "label"]
    return df.drop(columns=drop_cols, errors="ignore")


def _save_predictions(df: pd.DataFrame, preds, output_dir: Path, logger) -> Path:
    """Write predictions CSV to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / PREDICTIONS_FILENAME

    out_df = df.copy()
    out_df["prediction"] = preds
    out_df.to_csv(pred_file, index=False, encoding="utf-8-sig")

    logger.info("Predicciones guardadas en: %s", pred_file.as_posix())
    return pred_file


def main(
    input_path: str | None = None,
    model_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """Run inference end-to-end."""
    logger = setup_logger("inference")

    data_file = Path(input_path) if input_path else DEFAULT_DATA_PATH
    model_file = (
        Path(model_path) if model_path else _latest_file(DEFAULT_MODEL_DIR, "*.joblib")
    )
    output_dir = Path(output_path) if output_path else DEFAULT_OUTPUT_DIR

    if model_file is None or not model_file.exists():
        raise FileNotFoundError(
            "No encontré modelo .joblib en artifacts/models. Corre train primero."
        )

    if not data_file.exists():
        raise FileNotFoundError(f"No encontré data en: {data_file.as_posix()}")

    logger.info("Usando data: %s", data_file.as_posix())
    logger.info("Usando modelo: %s", model_file.as_posix())

    df = pd.read_csv(data_file)

    if not {"lag_1", "lag_2", "lag_3", "lag_mean_1_2"}.issubset(df.columns):
        logger.info("No encontré features lag en el dataset. Las voy a construir.")
        df = make_lag_features(df)

    model = _load_model(model_file)
    X_features = _build_features(df)

    preds = model.predict(X_features)
    _save_predictions(df, preds, output_dir, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default=None,
        help="Ruta al dataset para inferencia",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Ruta al modelo .joblib",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta de salida para predicciones",
    )

    args = parser.parse_args()

    main(args.input_path, args.model_path, args.output_path)