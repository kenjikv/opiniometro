# Contribuciones

> **Este archivo es una plantilla.** En el ejemplo, todo el código lo escribió el docente. La tabla muestra cómo debe verse este archivo en un proyecto de grupo: cada fila tiene que coincidir con el historial de commits de esa persona.

| Integrante | Rol | Aporte principal | Commits clave | Aporte en NLP o ML |
|---|---|---|---|---|
| Integrante 1 | Datos y NLP | Dataset y documentación de origen (`data/`), EDA (`notebooks/`), `TextPreprocessor` (`src/nlp/`) | `feat(nlp): add emoji normalization`, `feat(nlp): adjust spaCy stop word list for negations` | Lista de stop words ajustada (+2,1 puntos de F1) |
| Integrante 2 | Modelado | Búsqueda con validación cruzada (`src/models/experiments.py`), evaluación y análisis de errores (`src/models/evaluate.py`) | `feat(models): add grid search with stratified CV`, `feat(models): error analysis by phenomenon` | Comparación Naive Bayes vs. regresión logística |
| Integrante 3 | Backend | API (`api/`), esquema y acceso a PostgreSQL (`db/`, `src/db.py`), pruebas (`tests/test_api.py`), Docker | `feat(api): validate empty and long texts`, `feat(api): batch prediction from CSV` | Endpoint que expone los pasos del preprocesamiento |
| Integrante 4 | Frontend | Interfaz React (`frontend/`), paneles de métricas y datos | `feat(frontend): step-by-step preprocessing view`, `feat(frontend): confusion matrix` | Comparación stemming vs. lematización (experimento y gráfico) |

## Reglas que este archivo debe cumplir

- Cada integrante tiene **al menos un commit significativo en NLP o ML**: todos serán preguntados sobre ello (última columna).
- El trabajo en pareja se registra en el commit con una línea `Co-authored-by: Nombre <correo>`.
- Lo que no está en el historial de una persona no cuenta como trabajo suyo.

## Cómo verificarlo

```bash
git shortlog -sn --no-merges                    # commits por persona
git log --author="Nombre" --oneline             # commits de una persona
git log --author="Nombre" --stat -- src/nlp     # qué tocó en NLP
```
