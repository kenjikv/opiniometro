"""PostgreSQL access. The schema lives in db/init.sql."""

import json
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import database_url

REVIEW_COLUMNS = ["id", "restaurante", "ciudad", "fecha", "texto", "estrellas",
                  "sentimiento", "fenomeno"]


@lru_cache
def get_engine() -> Engine | None:
    url = database_url()
    return create_engine(url, pool_pre_ping=True) if url else None


def is_available() -> bool:
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def count_reviews(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM resenas")).scalar_one()


def replace_reviews(engine: Engine, df: pd.DataFrame) -> None:
    """Load the raw CSV into the database, replacing what was there."""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE resenas"))
        df[REVIEW_COLUMNS].to_sql("resenas", conn, if_exists="append", index=False)


def read_reviews(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM resenas ORDER BY id"), conn)


def record_training(engine: Engine, run: dict) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO entrenamientos (version, modelo, configuracion, cv_f1_macro,
                test_f1_macro, test_accuracy, n_entrenamiento, n_prueba)
            VALUES (:version, :modelo, CAST(:configuracion AS JSONB), :cv_f1_macro,
                :test_f1_macro, :test_accuracy, :n_entrenamiento, :n_prueba)
        """), {**run, "configuracion": json.dumps(run["configuracion"])})


def record_predictions(engine: Engine, rows: list[dict]) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO predicciones (texto, sentimiento, probabilidad, probabilidades,
                version_modelo, origen)
            VALUES (:texto, :sentimiento, :probabilidad, CAST(:probabilidades AS JSONB),
                :version_modelo, :origen)
        """), [{**r, "probabilidades": json.dumps(r["probabilidades"])} for r in rows])


def recent_predictions(engine: Engine, limit: int) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, fecha, texto, sentimiento, probabilidad, version_modelo, origen
            FROM predicciones ORDER BY fecha DESC, id DESC LIMIT :limit
        """), {"limit": limit})
        return [dict(row._mapping) for row in result]


def training_history(engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT version, fecha, modelo, cv_f1_macro, test_f1_macro, test_accuracy
            FROM entrenamientos ORDER BY fecha DESC
        """))
        return [dict(row._mapping) for row in result]
