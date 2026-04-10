import argparse
from .prep import main

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