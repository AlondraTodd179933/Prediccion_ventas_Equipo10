"""
Training script.

Reads the monthly dataset produced by prep.py, builds lag features,
trains a regression model, logs metrics, and saves artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

from algorithms.utils.logging_config import setup_logger

DEFAULT_DATA_PATH = Path("artifacts/data/monthly_clean.csv")
DEFAULT_OUT_DIR = Path("artifacts")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for time-based train/validation split."""

    val_last_block: int


def resolve_paths(data_path: str | None, out_dir: str) -> Tuple[Path, Path]:
    """Resolve input/output paths with safe defaults."""
    data_file = Path(data_path) if data_path else DEFAULT_DATA_PATH
    out_root = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    return data_file, out_root


def load_data(data_file: Path) -> pd.DataFrame:
    """Load training dataset from disk."""
    if not data_file.exists():
        raise FileNotFoundError(f"No encontré el dataset en: {data_file.as_posix()}")
    return pd.read_csv(data_file)


def make_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features used for training.

    Expected base column: item_cnt_month
    """
    if "item_cnt_month" not in df.columns:
        return df

    ordered = df.sort_values(["shop_id", "item_id", "date_block_num"]).copy()
    grouped = ordered.groupby(["shop_id", "item_id"])["item_cnt_month"]

    ordered["lag_1"] = grouped.shift(1)
    ordered["lag_2"] = grouped.shift(2)
    ordered["lag_3"] = grouped.shift(3)
    ordered["lag_mean_1_2"] = ordered[["lag_1", "lag_2"]].mean(axis=1)

    return ordered.dropna().reset_index(drop=True)


def split_time_last_block(
    df: pd.DataFrame, cfg: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/validation by date_block_num using last block as validation."""
    df_train = df[df["date_block_num"] < cfg.val_last_block].copy()
    df_val = df[df["date_block_num"] == cfg.val_last_block].copy()
    return df_train, df_val


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build features matrix X and target y."""
    y_target = df["item_cnt_month"]
    x_features = df.drop(columns=["item_cnt_month"], errors="ignore")
    return x_features, y_target


def save_json(payload: dict, out_file: Path) -> None:
    """Save dictionary as JSON."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train_and_score(df: pd.DataFrame, logger) -> Tuple[Ridge, float, int]:
    """
    Train model and compute RMSE using last month as validation.

    Returns:
      model, rmse, last_block
    """
    df_lags = make_lag_features(df)
    logger.info("Datos después de lags (dropna): %s filas", f"{len(df_lags):,}")

    last_block = int(df_lags["date_block_num"].max())
    cfg = SplitConfig(val_last_block=last_block)

    df_train, df_val = split_time_last_block(df_lags, cfg)
    logger.info(
        "Train: %s | Val: %s | last_block=%s",
        f"{len(df_train):,}",
        f"{len(df_val):,}",
        last_block,
    )

    x_train, y_train = build_xy(df_train)
    x_val, y_val = build_xy(df_val)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(x_train, y_train)

    preds_val = model.predict(x_val)
    rmse = float(root_mean_squared_error(y_val, preds_val))
    return model, rmse, last_block


def main(data_path: str | None, out_dir: str) -> None:
    """Run the full training pipeline."""
    start_time = time.time()
    logger = setup_logger("train")

    data_file, out_root = resolve_paths(data_path, out_dir)

    if not data_path:
        logger.info("No se pasó --data. Usando: %s", data_file.as_posix())

    df = load_data(data_file)
    logger.info("Datos cargados: %s filas, %s columnas", f"{len(df):,}", df.shape[1])

    model, rmse, last_block = train_and_score(df, logger)
    logger.info("RMSE validación (último mes): %.6f", rmse)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = out_root / "models" / f"ridge_{timestamp}.joblib"
    metrics_path = out_root / "metrics" / f"train_metrics_{timestamp}.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Modelo guardado en: %s", model_path.as_posix())

    save_json({"rmse_val_last_block": rmse, "val_last_block": last_block}, metrics_path)
    logger.info("Métricas guardadas en: %s", metrics_path.as_posix())

    duration = time.time() - start_time
    logger.info("Tiempo de ejecución: %.2f segundos", duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Ruta a monthly_clean.csv")
    parser.add_argument(
        "--out_dir", default=str(DEFAULT_OUT_DIR), help="Carpeta raíz de artifacts"
    )
    args = parser.parse_args()

    main(args.data, args.out_dir)
