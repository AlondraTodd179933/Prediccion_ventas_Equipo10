######### -*- coding: utf-8 -*-
"""
Entrenamiento del modelo (baseline) para predicción de ventas.

- Lee el dataset mensual preparado por prep.py
- Crea features (lags por shop-item)
- Entrena un modelo Ridge (rápido y estable)
- Guarda modelo + métricas
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils.logging_config import setup_logger
from sklearn.metrics import root_mean_squared_error



def _find_prepared_file(default_dir: Path) -> Path:
    """
    Busca un .csv dentro de artifacts/data por si el nombre exacto cambia.
    Prioriza archivos que contengan 'monthly' o 'prepared'.
    """
    if not default_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {default_dir.as_posix()}")

    candidates = list(default_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No encontré ningún .csv en {default_dir.as_posix()} (ejecuta prep.py primero)."
        )

    # Prioridad por nombre
    preferred = [p for p in candidates if ("monthly" in p.name.lower() or "prepared" in p.name.lower())]
    return preferred[0] if preferred else candidates[0]


def make_lag_features(df: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    """
    Asume que df tiene al menos:
      - shop_id, item_id, date_block_num
      - item_cnt_month (target)
    """
    df = df.copy()
    df = df.sort_values(["shop_id", "item_id", "date_block_num"])

    for l in lags:
        df[f"lag_{l}"] = (
            df.groupby(["shop_id", "item_id"])["item_cnt_month"].shift(l)
        )

    # Rolling mean simple (sobre lag_1 y lag_2 si existen)
    if "lag_1" in df.columns and "lag_2" in df.columns:
        df["lag_mean_1_2"] = df[["lag_1", "lag_2"]].mean(axis=1)

    return df


def main(data_path: str | None, out_dir: str) -> None:
    logger = setup_logger("train")

    base_dir = Path(out_dir)
    models_dir = base_dir / "models"
    metrics_dir = base_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ########## 1) Cargar datos preparados 
    if data_path is None:
        prepared = _find_prepared_file(Path("artifacts") / "data")
        logger.info("No se pasó --data. Usando: %s", prepared.as_posix())
        data_path = str(prepared)

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {data_path.as_posix()}")

    df = pd.read_csv(data_path)
    logger.info("Datos cargados: %s filas, %s cols", df.shape[0], df.shape[1])

    required = {"shop_id", "item_id", "date_block_num", "item_cnt_month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset preparado: {sorted(missing)}")

    ########## 2) Features 
    df_feat = make_lag_features(df, lags=(1, 2, 3))

    # Quitamos filas con NaN por lags (primeros meses de cada shop-item)
    df_feat = df_feat.dropna().reset_index(drop=True)
    logger.info("Datos después de lags (dropna): %s filas", df_feat.shape[0])

    feature_cols = [c for c in df_feat.columns if c.startswith("lag_")] + (["lag_mean_1_2"] if "lag_mean_1_2" in df_feat.columns else [])
    X = df_feat[feature_cols]
    y = df_feat["item_cnt_month"]

    ########## 3) Train/Val por tiempo
    # Validación sencilla: último date_block_num como validation
    last_block = df_feat["date_block_num"].max()
    train_mask = df_feat["date_block_num"] < last_block
    val_mask = df_feat["date_block_num"] == last_block

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    logger.info("Train: %s | Val: %s | last_block=%s", X_train.shape[0], X_val.shape[0], last_block)

    ########## 4) Modelo 
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)


    rmse = root_mean_squared_error(y_val, pred_val)

    logger.info("RMSE validación (último mes): %.6f", rmse)

    ########## 5) Guardar artefactos 
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"ridge_{stamp}.joblib"
    dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "last_block": int(last_block),
        },
        model_path,
    )
    logger.info("Modelo guardado en: %s", model_path.as_posix())

    metrics = {
        "rmse_val_last_block": float(rmse),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "last_block": int(last_block),
        "features": feature_cols,
        "data_used": data_path.as_posix(),
    }
    metrics_path = metrics_dir / f"train_metrics_{stamp}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Métricas guardadas en: %s", metrics_path.as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Ruta del CSV preparado por prep.py (si no se pasa, busca en artifacts/data).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts",
        help="Carpeta donde se guardan modelos/métricas/logs.",
    )
    args = parser.parse_args()
    main(args.data, args.out_dir)
