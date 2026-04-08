import argparse
from .train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default=None,
        help="Ruta al dataset de entrenamiento",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Ruta donde guardar el modelo .joblib",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Número de árboles del Random Forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Profundidad máxima de los árboles",
    )

    args = parser.parse_args()

    main(
        input_path=args.input_path,
        output_path=args.output_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )