"""One command from raw data to saved models.

    python -m src.pipeline                    # always retrains
    python -m src.pipeline --skip-if-trained  # used by docker compose up
    python -m src.pipeline --reload-data      # reload the CSV into PostgreSQL

Steps:
    1. If the CSV does not exist, generate it (synthetic data, fixed seed).
    2. If there is a database and its table is empty, load the CSV into it.
    3. Train, compare and save the models (src.models.train).
"""

import argparse

import pandas as pd

from src import db
from src.config import MODEL_PATH, ROOT, load_params
from src.data.generate import generate
from src.models import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-if-trained", action="store_true",
                        help="do nothing if artifacts/model.joblib already exists")
    parser.add_argument("--reload-data", action="store_true",
                        help="replace the database table with the CSV")
    args = parser.parse_args()
    params = load_params()

    csv_path = ROOT / params["data"]["raw_path"]
    if not csv_path.exists():
        print(f"[1/3] Generando datos sintéticos en {csv_path}")
        generate(params["data"]["n_reviews"], params["seed"]).to_csv(csv_path, index=False)
    else:
        print(f"[1/3] Datos crudos: {csv_path}")

    engine = db.get_engine()
    if engine is None:
        print("[2/3] Sin DATABASE_URL: se entrena directamente desde el CSV")
    elif args.reload_data or db.count_reviews(engine) == 0:
        raw = pd.read_csv(csv_path, parse_dates=["fecha"])
        db.replace_reviews(engine, raw)
        print(f"[2/3] {len(raw)} reseñas cargadas en PostgreSQL (tabla resenas)")
    else:
        print(f"[2/3] PostgreSQL ya tiene {db.count_reviews(engine)} reseñas")

    if args.skip_if_trained and MODEL_PATH.exists():
        print(f"[3/3] Ya existe {MODEL_PATH}; no se reentrena "
              "(usa 'python -m src.pipeline' para forzarlo)")
        return
    print("[3/3] Entrenando")
    train.main()


if __name__ == "__main__":
    main()
