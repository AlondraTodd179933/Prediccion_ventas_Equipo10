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

import time
from pathlib import Path

import pandas as pd

from algorithms.utils.logging_config import setup_logger

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("artifacts/data")
OUTPUT_FILENAME = "monthly_clean.csv"


def main() -> None:
    """Run the full data preparation pipeline."""
    start_time = time.time()
    logger = setup_logger("prep")

    logger.info("Iniciando preparación de datos")

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando datasets originales")
    sales_train = pd.read_csv(DEFAULT_DATA_DIR / "sales_train.csv")
    items = pd.read_csv(DEFAULT_DATA_DIR / "items.csv")
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

    output_path = DEFAULT_OUTPUT_DIR / OUTPUT_FILENAME
    monthly.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Dataset guardado en: %s", output_path.as_posix())

    duration = time.time() - start_time
    logger.info("Preparación finalizada en %.2f segundos", duration)


if __name__ == "__main__":
    main()
