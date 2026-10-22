"""Loads the saved pipeline and runs predictions with it.

Nothing here preprocesses text: the pipeline's first step does it, with the
exact code and parameters used during training.
"""

import json
import logging

import joblib

from src import db
from src.config import METRICS_PATH, MODEL_PATH

log = logging.getLogger("opiniometro")


class ModelService:
    def __init__(self):
        self.bundle = None
        self.metrics = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            log.warning("No existe %s: ejecuta el entrenamiento", MODEL_PATH)
            return
        self.bundle = joblib.load(MODEL_PATH)
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        # Warm-up: the first prediction loads spaCy and the stemmer (~0.5 s).
        self.pipeline.predict(["calentamiento del modelo"])
        log.info("Modelo %s cargado (versión %s)", self.bundle["model"], self.bundle["version"])

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    @property
    def pipeline(self):
        return self.bundle["pipeline"]

    def predict(self, texts: list[str]) -> list[dict]:
        probabilities = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.classes_)
        results = []
        for row in probabilities:
            best = int(row.argmax())
            results.append({
                "sentimiento": classes[best],
                "probabilidad": round(float(row[best]), 4),
                "probabilidades": {c: round(float(p), 4) for c, p in zip(classes, row)},
            })
        return results

    def explain(self, text: str) -> dict:
        preprocessor = self.pipeline.named_steps["preprocess"]
        steps = preprocessor.explain(text)
        return {
            "original": steps["original"],
            "normalizado": steps["normalized"],
            "tokens": steps["tokens"],
            "stopwords_eliminadas": steps["removed_stopwords"],
            "sin_stopwords": steps["without_stopwords"],
            "raices": steps["roots"],
            "final": steps["final"],
            "metodo": preprocessor.normalizer,
        }

    def save(self, texts: list[str], results: list[dict], origin: str) -> None:
        """Store predictions; a database failure never breaks a prediction."""
        engine = db.get_engine()
        if engine is None:
            return
        try:
            db.record_predictions(engine, [
                {"texto": t, **r, "version_modelo": self.bundle["version"], "origen": origin}
                for t, r in zip(texts, results)
            ])
        except Exception as exc:  # noqa: BLE001
            log.warning("No se pudo guardar la predicción: %s", exc)
