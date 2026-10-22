"""Paths and parameters shared by the data, training and API code."""

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", ROOT / "params.yaml"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
EXPERIMENTS_PATH = ARTIFACTS_DIR / "experiments.csv"
ERRORS_PATH = ARTIFACTS_DIR / "errors.csv"

LABELS = ["negativo", "neutro", "positivo"]


@lru_cache
def load_params() -> dict:
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def database_url() -> str | None:
    """None means "run without a database" (tests, notebook)."""
    return os.getenv("DATABASE_URL") or None
