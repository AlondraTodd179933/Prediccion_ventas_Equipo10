"""
Script de preparación de datos para el modelo de predicción de ventas.

Este script se encarga de:
- Cargar datos crudos
- Limpiar observaciones inválidas
- Agregar ventas a nivel mensual
- Construir features temporales (lags)
- Guardar el dataset listo para entrenamiento
"""

import time
from pathlib import Path

import pandas as pd

from .utils.logging_config import setup_logger


def main() -> None:
    """Ejecuta el pipeline completo de preparación de datos."""
    start_time = time.time()
    logger = setup_logger("prep")

    logger.info("Iniciando preparación de datos")

    ################### Rutas
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("artifacts/data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    ################### Carga de datos
    logger.info("Cargando datasets originales")

    sales_train = pd.read_csv(DATA_DIR / "sales_train.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")

    logger.info("Ventas cargadas: %s filas", len(sales_train))

    
    #################Limpieza básica
    sales_train = sales_train[
        (sales_train["item_price"] > 0) &
        (sales_train["item_cnt_day"] >= 0)
    ]

    logger.info("Ventas después de limpieza: %s filas", len(sales_train))

    ################## Agregación mensual
    monthly = (
        sales_train
        .groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
        .agg(item_cnt_month=("item_cnt_day", "sum"))
    )

    logger.info("Datos agregados a nivel mensual")

    ################## Merge con categorías
    monthly = monthly.merge(
        items[["item_id", "item_category_id"]],
        on="item_id",
        how="left"
    )

    
    ##############Guardar resultado
    output_path = OUTPUT_DIR / "monthly_clean.csv"
    monthly.to_csv(output_path, index=False)

    duration = time.time() - start_time
    logger.info("Preparación finalizada en %.2f segundos", duration)


if __name__ == "__main__":
    main()
