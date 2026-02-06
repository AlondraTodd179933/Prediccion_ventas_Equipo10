# # inference.py
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .utils.logging_config import setup_logger


def _latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_model(model_path: str):
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        for key in ["model", "estimator", "pipeline", "ridge", "regressor", "clf"]:
            if key in bundle:
                return bundle[key]
        raise ValueError(
            f"El modelo viene como dict, pero no encontré el estimador. Keys: {list(bundle.keys())}"
        )

    return bundle


def _get_expected_features(model):
    """
    Intenta recuperar las columnas con las que se entrenó el modelo (si sklearn las guardó).
    Si no existen, usamos las que sabemos por el error.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return ["lag_1", "lag_2", "lag_3", "lag_mean_1_2"]


def _build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye lags a partir de item_cnt_month agrupando por (shop_id, item_id)
    y ordenando por date_block_num.
    """
    needed_base = {"shop_id", "item_id", "date_block_num", "item_cnt_month"}
    missing = needed_base - set(df.columns)
    if missing:
        raise ValueError(
            f"Para construir lags faltan columnas: {sorted(missing)}"
        )

    df = df.sort_values(["shop_id", "item_id", "date_block_num"]).copy()

    g = df.groupby(["shop_id", "item_id"])["item_cnt_month"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_3"] = g.shift(3)

    df["lag_mean_1_2"] = df[["lag_1", "lag_2"]].mean(axis=1)

    return df


def main(data_path: str | None, model_path: str | None, out_dir: str):
    logger = setup_logger("inference")

    repo_root = Path(__file__).resolve().parents[2]
    artifacts = repo_root / "artifacts"

    if data_path is None:
        data_path = artifacts / "data" / "monthly_clean.csv"
    else:
        data_path = Path(data_path)

    if model_path is None:
        model_path = _latest_file(artifacts / "models", "*.joblib")
    else:
        model_path = Path(model_path)

    if model_path is None or not model_path.exists():
        raise FileNotFoundError("No encontré el modelo .joblib en artifacts/models. Corre train primero.")

    logger.info("Usando data: %s", data_path)
    logger.info("Usando modelo: %s", model_path)

    df = pd.read_csv(data_path)

    # cargar modelo
    model = _load_model(str(model_path))

    # columnas esperadas por el modelo
    expected = _get_expected_features(model)

    # si el modelo espera lags y no están, los construimos
    if any(c.startswith("lag_") for c in expected) and not set(expected).issubset(df.columns):
        logger.info("No encontré features lag en el dataset. Las voy a construir...")
        df = _build_lag_features(df)

    # armar X con EXACTAMENTE lo que el modelo espera
    # (si queda algún NaN por lags iniciales, lo quitamos)
    X = df.reindex(columns=expected)
    mask_ok = ~X.isna().any(axis=1)
    X_ok = X.loc[mask_ok].copy()

    if X_ok.empty:
        raise ValueError(
            "Después de crear lags, no quedaron filas válidas para predecir. "
            "Revisa que date_block_num tenga suficientes meses para generar lag_1/lag_2/lag_3."
        )

    preds = model.predict(X_ok)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pred_file = out_path / "predictions.csv"

    out_df = df.loc[mask_ok].copy()
    out_df["prediction"] = preds
    out_df.to_csv(pred_file, index=False, encoding="utf-8-sig")

    logger.info("Predicciones guardadas en: %s", pred_file.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Ruta a monthly_clean.csv")
    parser.add_argument("--model", default=None, help="Ruta al modelo .joblib")
    parser.add_argument("--out_dir", default="artifacts/predictions", help="Carpeta de salida")
    args = parser.parse_args()

    main(args.data, args.model, args.out_dir)

