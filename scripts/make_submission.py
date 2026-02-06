# scripts/make_submission.py
"""
Genera el archivo submission.csv a partir de las predicciones del modelo.
Este script prepara el formato requerido para Kaggle.
"""

from pathlib import Path
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    preds_path = repo_root / "artifacts" / "predictions" / "predictions.csv"
    out_dir = repo_root / "artifacts" / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        raise FileNotFoundError(
            "No se encontró predictions.csv. Ejecuta inference.py primero."
        )

    df = pd.read_csv(preds_path)

    # Formato típico Kaggle: id + target
    # Ajusta si tu competencia pide otro nombre
    df["id"] = df.index
    submission = df[["id", "prediction"]]
    submission = submission.rename(columns={"prediction": "item_cnt_month"})

    out_file = out_dir / "submission.csv"
    submission.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"Submission generada en: {out_file}")


if __name__ == "__main__":
    main()
