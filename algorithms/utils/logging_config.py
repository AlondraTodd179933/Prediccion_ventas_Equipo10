"""
Configuración centralizada de logging para el proyecto.

Este módulo define una función estándar para crear loggers que escriben
tanto en consola como en archivos, con el fin de mantener trazabilidad
en los scripts de preparación, entrenamiento e inferencia.
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(script_name: str, log_dir: str = "artifacts/logs") -> logging.Logger:
    """
    Configura un logger con salida a consola y a archivo con timestamp.

    Parameters
    ----------
    script_name : str
        Nombre del script que usa el logger (por ejemplo: 'prep', 'train', 'inference').
    log_dir : str, optional
        Carpeta donde se guardarán los logs. Por defecto 'artifacts/logs'.

    Returns
    -------
    logging.Logger
        Logger configurado y listo para usarse.
    """
    # Crear directorio de logs si no existe
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Evitar agregar handlers duplicados (útil si se ejecuta varias veces)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Handler para archivo
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger configurado correctamente. Log file: %s", log_file)

    return logger
