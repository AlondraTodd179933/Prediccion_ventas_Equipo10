import argparse
from .inference import main

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
        default="artifacts/predictions",
        help="Carpeta de salida para predicciones",
    )

    args = parser.parse_args()

    main(
        input_path=args.input_path,
        model_path=args.model_path,
        output_path=args.output_path,
    )