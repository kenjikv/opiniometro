-- Runs once, the first time the PostgreSQL container starts with an empty volume.

-- Raw dataset, exactly as it arrives (with empty texts and duplicates).
-- Cleaning happens in the training script, not here.
CREATE TABLE IF NOT EXISTS resenas (
    id           INTEGER PRIMARY KEY,
    restaurante  TEXT NOT NULL,
    ciudad       TEXT,
    fecha        DATE NOT NULL,
    texto        TEXT,
    estrellas    SMALLINT NOT NULL CHECK (estrellas BETWEEN 1 AND 5),
    sentimiento  TEXT NOT NULL CHECK (sentimiento IN ('negativo', 'neutro', 'positivo')),
    fenomeno     TEXT NOT NULL
);

-- One row per training run: what was trained and how well it did.
CREATE TABLE IF NOT EXISTS entrenamientos (
    id              SERIAL PRIMARY KEY,
    version         TEXT NOT NULL UNIQUE,
    fecha           TIMESTAMPTZ NOT NULL DEFAULT now(),
    modelo          TEXT NOT NULL,
    configuracion   JSONB NOT NULL,
    cv_f1_macro     DOUBLE PRECISION NOT NULL,
    test_f1_macro   DOUBLE PRECISION NOT NULL,
    test_accuracy   DOUBLE PRECISION NOT NULL,
    n_entrenamiento INTEGER NOT NULL,
    n_prueba        INTEGER NOT NULL
);

-- Every prediction served by the API.
CREATE TABLE IF NOT EXISTS predicciones (
    id             SERIAL PRIMARY KEY,
    fecha          TIMESTAMPTZ NOT NULL DEFAULT now(),
    texto          TEXT NOT NULL,
    sentimiento    TEXT NOT NULL,
    probabilidad   DOUBLE PRECISION NOT NULL,
    probabilidades JSONB NOT NULL,
    version_modelo TEXT NOT NULL,
    origen         TEXT NOT NULL CHECK (origen IN ('individual', 'lote'))
);

CREATE INDEX IF NOT EXISTS predicciones_fecha_idx ON predicciones (fecha DESC);
