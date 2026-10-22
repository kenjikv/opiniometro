"""Request and response contracts. FastAPI validates every request against
these models and documents them in /api/docs."""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import load_params

MAX_CHARS = load_params()["data"]["max_chars"]
LETTER_RE = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]")


def validate_text(value: str) -> str:
    """Shared by single and batch predictions."""
    value = value.strip()
    if not value:
        raise ValueError("El texto está vacío.")
    if len(value) > MAX_CHARS:
        raise ValueError(f"El texto tiene {len(value)} caracteres; el máximo es {MAX_CHARS}.")
    if not LETTER_RE.search(value):
        raise ValueError("El texto no contiene palabras.")
    return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"texto": "El majadito estaba riquísimo, pero tardaron bastante."}]})

    texto: str = Field(description=f"Reseña en español, de 1 a {MAX_CHARS} caracteres.")

    @field_validator("texto")
    @classmethod
    def check_text(cls, value: str) -> str:
        return validate_text(value)


class PreprocessingSteps(BaseModel):
    original: str
    normalizado: str = Field(description="Minúsculas, sin URLs, números ni puntuación; "
                                         "emojis convertidos en palabras.")
    tokens: list[str]
    stopwords_eliminadas: list[str]
    sin_stopwords: list[str]
    raices: list[str] = Field(description="Raíces (stemming) o lemas, según el modelo.")
    final: list[str] = Field(description="Lo que recibe TF-IDF: raíces sin tildes.")
    metodo: str = Field(description="stem o lemma")


class ModelInfo(BaseModel):
    nombre: str
    version: str


class PredictResponse(BaseModel):
    sentimiento: str
    probabilidad: float
    probabilidades: dict[str, float]
    preprocesamiento: PreprocessingSteps
    modelo: ModelInfo
    tiempo_ms: float


class BatchItem(BaseModel):
    fila: int
    texto: str
    sentimiento: str
    probabilidad: float
    probabilidades: dict[str, float]


class SkippedRow(BaseModel):
    fila: int
    motivo: str


class BatchResponse(BaseModel):
    total: int
    resumen: dict[str, int]
    resultados: list[BatchItem]
    omitidas: list[SkippedRow]


class HealthResponse(BaseModel):
    estado: str = Field(description="ok si el modelo está cargado; degradado si no.")
    modelo_cargado: bool
    modelo: str | None
    version: str | None
    base_de_datos: str = Field(description="conectada, sin conexión o no configurada")


class ErrorResponse(BaseModel):
    error: str
    detalles: list[dict] = []
