# API

Base: `http://localhost:8000/api` (directo) o `http://localhost:8080/api` (a través de nginx).
Documentación interactiva, con botón «Try it out»: http://localhost:8000/api/docs

Todas las respuestas son JSON. Los errores tienen siempre esta forma:

```json
{"error": "El texto está vacío.", "detalles": [{"campo": "texto", "mensaje": "El texto está vacío."}]}
```

## `GET /api/health`

Estado del servicio.

```bash
curl http://localhost:8000/api/health
```

```json
{"estado": "ok", "modelo_cargado": true, "modelo": "Regresión logística",
 "version": "20261022-185649", "base_de_datos": "conectada"}
```

`estado` es `degradado` si no hay modelo entrenado. `base_de_datos` puede ser `conectada`, `sin conexión` o `no configurada`.

## `POST /api/predict`

Clasifica una reseña y devuelve cada paso del preprocesamiento.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "Las salteñas no estaban nada mal y no tuvimos que esperar 😋"}'
```

```json
{
  "sentimiento": "positivo",
  "probabilidad": 0.7871,
  "probabilidades": {"negativo": 0.1753, "neutro": 0.0377, "positivo": 0.7871},
  "preprocesamiento": {
    "original": "Las salteñas no estaban nada mal y no tuvimos que esperar 😋",
    "normalizado": "las salteñas no estaban nada mal y no tuvimos que esperar emojipositivo",
    "tokens": ["las", "salteñas", "no", "estaban", "nada", "mal", "no", "tuvimos", "que", "esperar", "emojipositivo"],
    "stopwords_eliminadas": ["las", "estaban", "que"],
    "sin_stopwords": ["salteñas", "no", "nada", "mal", "no", "tuvimos", "esperar", "emojipositivo"],
    "raices": ["salteñ", "no", "nad", "mal", "no", "tuv", "esper", "emojipositivo"],
    "final": ["salteñ", "no", "nad", "mal", "no", "tuv", "esper", "emojipositivo"],
    "metodo": "stem"
  },
  "modelo": {"nombre": "Regresión logística", "version": "20261022-185649"},
  "tiempo_ms": 1.2
}
```

### Validación

| Entrada | Código | `error` |
|---|---|---|
| `{"texto": ""}` o solo espacios | 422 | El texto está vacío. |
| Más de 1000 caracteres | 422 | El texto tiene N caracteres; el máximo es 1000. |
| Sin letras (`"123 !!!"`) | 422 | El texto no contiene palabras. |
| `{"texto": 5}` | 422 | Debe ser un texto entre comillas. |
| `{}` | 422 | Falta este campo obligatorio. |
| `{"texto": "...", "otro": 1}` | 422 | Campo no permitido; solo se acepta 'texto'. |
| JSON mal formado | 422 | El cuerpo de la petición no es un JSON válido. |
| Modelo sin entrenar | 503 | El modelo no está entrenado. Ejecuta: python -m src.pipeline |

El límite de caracteres está en `params.yaml` (`data.max_chars`).

## `POST /api/predict/batch`

Clasifica un archivo: `.txt` con una reseña por línea, o `.csv` con una columna `texto`. Máximo 500 reseñas y 1 MB. Las filas inválidas se omiten e informan; el resto se procesa.

```bash
curl -F archivo=@frontend/public/ejemplo_lote.csv http://localhost:8000/api/predict/batch
```

```json
{
  "total": 8,
  "resumen": {"negativo": 4, "neutro": 3, "positivo": 1},
  "resultados": [
    {"fila": 1, "texto": "El fricasé estaba delicioso…", "sentimiento": "positivo",
     "probabilidad": 0.781, "probabilidades": {"negativo": 0.0515, "neutro": 0.1674, "positivo": 0.781}}
  ],
  "omitidas": [
    {"fila": 7, "motivo": "El texto está vacío."},
    {"fila": 8, "motivo": "El texto no contiene palabras."}
  ]
}
```

Errores: extensión distinta de `.txt`/`.csv` (422), CSV sin columna `texto` (422), archivo que no está en UTF-8 (422), más de 1 MB o más de 500 reseñas (413).

## `GET /api/metrics`

Todo lo que produjo el último entrenamiento (`artifacts/metrics.json`): modelo en producción, comparación de modelos con métricas por clase y matriz de confusión, los 17 experimentos, el efecto de cada decisión, el resumen del EDA, los términos más influyentes y el análisis de errores.

## `GET /api/history?limit=20`

Últimas predicciones guardadas en PostgreSQL (`limit` entre 1 y 200). Responde 503 si no hay base de datos.

## `GET /api/trainings`

Entrenamientos registrados en la tabla `entrenamientos`: versión, fecha, modelo y métricas.
