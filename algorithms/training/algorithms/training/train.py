if __name__ == "__main__":def _resolve_sagemaker_train_file(train_data_path: str) -> Path:
    """Return the CSV file mounted by SageMaker in the train channel."""
    path = Path(train_data_path)

    if path.is_file():
        return path

    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en el canal de entrenamiento: {path.as_posix()}"
            )
        return csv_files[0]

    raise FileNotFoundError(
        f"La ruta de entrenamiento de SageMaker no existe: {path.as_posix()}"
    )


def train_and_evaluate(
    train_data_path: str,
    model_dir: str,
    hyperparams: dict | None = None,
) -> None:
    """Train the model using SageMaker input/output conventions."""
    start_time = time.time()
    logger = setup_logger("train_sagemaker")
    hyperparams = hyperparams or {}

    data_file = _resolve_sagemaker_train_file(train_data_path)
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando datos desde: %s", data_file.as_posix())
    df = load_data(data_file)
    logger.info("Datos cargados: %s filas, %s columnas", f"{len(df):,}", df.shape[1])

    # Por ahora leemos hyperparameters pero usamos la lógica actual del repositorio.
    # Eso es suficiente para la tarea 5.
    model, rmse, last_block = train_and_score(df, logger)

    model_path = model_root / "model.joblib"
    metrics_path = model_root / "training_metrics.json"

    joblib.dump(model, model_path)
    logger.info("Modelo SageMaker guardado en: %s", model_path.as_posix())

    save_json(
        {
            "rmse_val_last_block": rmse,
            "val_last_block": last_block,
            "hyperparameters": hyperparams,
        },
        metrics_path,
    )
    logger.info("Métricas SageMaker guardadas en: %s", metrics_path.as_posix())

    duration = time.time() - start_time
    logger.info("Tiempo total de entrenamiento SageMaker: %.2f segundos", duration)
    