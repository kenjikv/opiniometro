"""REST API of the Opiniómetro. Interactive documentation at /api/docs."""

import csv
import io
import logging
import time
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas import (BatchResponse, ErrorResponse, HealthResponse, PredictRequest,
                         PredictResponse, validate_text)
from api.service import ModelService
from src import db
from src.config import LABELS, database_url

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

MAX_BATCH_ROWS = 500
MAX_FILE_BYTES = 1_000_000

service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()  # once, at startup; not on every request
    yield


app = FastAPI(
    title="Opiniómetro API",
    version="1.0.0",
    description="Clasifica el sentimiento de reseñas de restaurantes en español "
                "(negativo, neutro o positivo) y muestra cómo se preprocesó el texto.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
    lifespan=lifespan,
)

ERRORS = {
    422: {"model": ErrorResponse, "description": "Entrada inválida"},
    503: {"model": ErrorResponse, "description": "Modelo o base de datos no disponible"},
}

# Pydantic error types translated for the user.
MESSAGES = {
    "missing": "Falta este campo obligatorio.",
    "string_type": "Debe ser un texto entre comillas.",
    "json_invalid": "El cuerpo de la petición no es un JSON válido.",
    "model_attributes_type": "El cuerpo debe ser un objeto JSON, por ejemplo {\"texto\": \"...\"}.",
    "dict_type": "El cuerpo debe ser un objeto JSON, por ejemplo {\"texto\": \"...\"}.",
    "extra_forbidden": "Campo no permitido; solo se acepta 'texto'.",
}


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = []
    for error in exc.errors():
        message = MESSAGES.get(error["type"], error["msg"]).removeprefix("Value error, ")
        field = ".".join(str(p) for p in error["loc"] if p != "body") or "cuerpo"
        details.append({"campo": field, "mensaje": message})
    return JSONResponse(status_code=422,
                        content={"error": details[0]["mensaje"], "detalles": details})


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "detalles": []})


def require_model() -> None:
    if not service.loaded:
        raise HTTPException(503, "El modelo no está entrenado. Ejecuta: python -m src.pipeline")


def require_database():
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(503, "La base de datos no está configurada (DATABASE_URL).")
    return engine


@app.get("/api/health", response_model=HealthResponse, tags=["estado"])
def health():
    """Estado del servicio: modelo cargado y conexión a la base de datos."""
    if database_url() is None:
        database = "no configurada"
    else:
        database = "conectada" if db.is_available() else "sin conexión"
    return HealthResponse(
        estado="ok" if service.loaded else "degradado",
        modelo_cargado=service.loaded,
        modelo=service.bundle["label"] if service.loaded else None,
        version=service.bundle["version"] if service.loaded else None,
        base_de_datos=database,
    )


@app.post("/api/predict", response_model=PredictResponse, responses=ERRORS, tags=["predicción"])
def predict(request: PredictRequest):
    """Sentimiento de una reseña, con su probabilidad y cada paso del preprocesamiento."""
    require_model()
    start = time.perf_counter()
    result = service.predict([request.texto])[0]
    steps = service.explain(request.texto)
    elapsed = (time.perf_counter() - start) * 1000
    service.save([request.texto], [result], origin="individual")
    return {
        **result,
        "preprocesamiento": steps,
        "modelo": {"nombre": service.bundle["label"], "version": service.bundle["version"]},
        "tiempo_ms": round(elapsed, 1),
    }


@app.post("/api/predict/batch", response_model=BatchResponse,
          responses={**ERRORS, 413: {"model": ErrorResponse, "description": "Archivo muy grande"}},
          tags=["predicción"])
async def predict_batch(archivo: UploadFile = File(
        description="Un .txt con una reseña por línea o un .csv con una columna 'texto'.")):
    """Clasifica varias reseñas de un archivo. Las filas inválidas se informan y se omiten."""
    require_model()
    name = (archivo.filename or "").lower()
    if not name.endswith((".txt", ".csv")):
        raise HTTPException(422, "Formato no soportado: sube un .txt (una reseña por línea) "
                                 "o un .csv con una columna 'texto'.")
    content = await archivo.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, "El archivo supera el máximo de 1 MB.")
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, "El archivo debe estar codificado en UTF-8.") from None

    if name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(decoded))
        if "texto" not in (reader.fieldnames or []):
            raise HTTPException(422, "El CSV debe tener una columna llamada 'texto'.")
        rows = [row["texto"] or "" for row in reader]
    else:
        rows = [line for line in decoded.splitlines() if line.strip()]
    if len(rows) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"El archivo tiene {len(rows)} reseñas; "
                                 f"el máximo es {MAX_BATCH_ROWS}.")

    valid, skipped = [], []
    for number, text in enumerate(rows, start=1):
        try:
            valid.append((number, validate_text(text)))
        except ValueError as exc:
            skipped.append({"fila": number, "motivo": str(exc)})
    if not valid:
        raise HTTPException(422, "El archivo no contiene ninguna reseña válida.")

    texts = [text for _, text in valid]
    results = service.predict(texts)
    service.save(texts, results, origin="lote")
    counts = Counter(r["sentimiento"] for r in results)
    return {
        "total": len(results),
        "resumen": {label: counts.get(label, 0) for label in LABELS},
        "resultados": [{"fila": n, "texto": t, **r} for (n, t), r in zip(valid, results)],
        "omitidas": skipped,
    }


@app.get("/api/metrics", responses=ERRORS, tags=["modelo"])
def metrics():
    """Métricas del entrenamiento: comparación de modelos, experimentos y errores."""
    require_model()
    return service.metrics


@app.get("/api/history", responses=ERRORS, tags=["historial"])
def history(limit: int = Query(20, ge=1, le=200, description="Cantidad de predicciones")):
    """Últimas predicciones guardadas en PostgreSQL."""
    engine = require_database()
    try:
        return db.recent_predictions(engine, limit)
    except Exception:
        raise HTTPException(503, "No se pudo leer la base de datos.") from None


@app.get("/api/trainings", responses=ERRORS, tags=["historial"])
def trainings():
    """Entrenamientos registrados: versión, modelo y métricas de cada uno."""
    engine = require_database()
    try:
        return db.training_history(engine)
    except Exception:
        raise HTTPException(503, "No se pudo leer la base de datos.") from None
