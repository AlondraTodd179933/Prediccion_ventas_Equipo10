"""
Data preparation script.

This script:
- Loads raw Kaggle datasets
- Cleans invalid observations
- Aggregates sales to monthly level
- Merges item category info
- Saves the final dataset for training/inference
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from algorithms.utils.logging_config import setup_logger

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("artifacts/data")
OUTPUT_FILENAME = "monthly_clean.csv"


def main(input_path: str | None = None, output_path: str | None = None) -> None:
    """Run the full data preparation pipeline."""
    start_time = time.time()
    logger = setup_logger("prep")

    data_dir = Path(input_path) if input_path else DEFAULT_DATA_DIR
    output_file = Path(output_path) if output_path else (DEFAULT_OUTPUT_DIR / OUTPUT_FILENAME)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Iniciando preparación de datos")
    logger.info("Leyendo datos desde: %s", data_dir.as_posix())

    sales_train_path = data_dir / "sales_train.csv"
    items_path = data_dir / "items.csv"

    if not sales_train_path.exists():
        raise FileNotFoundError(f"No encontré: {sales_train_path.as_posix()}")

    if not items_path.exists():
        raise FileNotFoundError(f"No encontré: {items_path.as_posix()}")

    logger.info("Cargando datasets originales")
    sales_train = pd.read_csv(sales_train_path)
    items = pd.read_csv(items_path)
    logger.info("Ventas cargadas: %s filas", f"{len(sales_train):,}")

    sales_train = sales_train[
        (sales_train["item_price"] > 0) & (sales_train["item_cnt_day"] >= 0)
    ].copy()
    logger.info("Ventas después de limpieza: %s filas", f"{len(sales_train):,}")

    monthly = (
        sales_train.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
        .agg(item_cnt_month=("item_cnt_day", "sum"))
        .merge(items[["item_id", "item_category_id"]], on="item_id", how="left")
    )
    logger.info("Datos agregados a nivel mensual")

    monthly.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info("Dataset guardado en: %s", output_file.as_posix())

    duration = time.time() - start_time
    logger.info("Preparación finalizada en %.2f segundos", duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default=None,
        help="Carpeta donde viven sales_train.csv e items.csv",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Ruta completa del archivo CSV de salida",
    )
    args = parser.parse_args()

    main(args.input_path, args.output_path)