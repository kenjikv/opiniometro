# Informe técnico · Opiniómetro

## 1. Problema

Un restaurante recibe decenas de reseñas por semana y no tiene tiempo de leerlas todas. Queremos clasificar automáticamente cada reseña como **negativa, neutra o positiva**, para atender primero las negativas y medir cómo evoluciona la opinión de los clientes.

**Tarea:** clasificación de texto en tres clases.
**Entrada:** el texto de la reseña, en español.
**Salida:** la clase y la probabilidad de cada una.

## 2. Dataset

3.045 reseñas sintéticas de restaurantes bolivianos (origen, licencia y diccionario en [`data/README.md`](../data/README.md)). Hallazgos del análisis exploratorio (`notebooks/01_eda_y_experimentos.ipynb`):

| Hallazgo | Dato | Decisión |
|---|---|---|
| Textos vacíos | 31 (1 %) | Se eliminan: no hay nada que aprender |
| Ciudad vacía | 92 (3 %) | Se conservan: el modelo no usa la ciudad |
| Duplicados exactos | 54 | Se eliminan **antes** de dividir, para que una copia no quede en entrenamiento y otra en prueba |
| Desbalance | 48 % positivas, 31 % negativas, 21 % neutras | F1 macro como métrica principal; división estratificada; `class_weight` y `fit_prior` en la búsqueda |
| Longitud | mediana 13 palabras, máximo 42 | No separa las clases; no se usa como variable |
| Estrellas contra etiqueta | 21 reseñas negativas con 5 estrellas, 13 positivas con 1 | Etiquetas ruidosas: no se corrigen a mano, se reportan en el análisis de errores |

**División:** 80 % entrenamiento (2.368) y 20 % prueba (592), estratificada, semilla 42. El conjunto de prueba no se usa para elegir nada.

## 3. Pipeline de NLP

Implementado en la clase `TextPreprocessor` (`src/nlp/preprocessing.py`), un transformador de scikit-learn que forma parte del pipeline guardado. Entrenamiento y API ejecutan el mismo código.

| Paso | Antes | Después | Justificación |
|---|---|---|---|
| Normalización | `Q RICOOO el majadito!!! 😍 Bs 45 https://… #ComidaBoliviana` | `que rico el majadito emojipositivo bs comidaboliviana` | Minúsculas; se eliminan URLs, menciones, números y puntuación; se reducen letras repetidas; se expande la jerga (`q`, `x`, `tb`); **los emojis se convierten en palabras** porque expresan sentimiento |
| Tokenización | `las salteñas no estaban nada mal` | `[las, salteñas, no, estaban, nada, mal]` | Después de normalizar solo quedan letras: basta separar por espacios. Se descartan tokens de una letra |
| Stop words | `[las, salteñas, no, estaban, nada, mal]` | `[salteñas, no, nada, mal]` | Lista de spaCy **ajustada**: la original incluye «no», «nada», «bueno», «mejor», «mal» y borraría el sentimiento. Se agregan palabras del dominio presentes en todas las clases («restaurante», «local», «lugar») |
| Stemming | `[salteñas, no, nada, mal]` | `[salteñ, no, nad, mal]` | Snowball para español. Se comparó con la lematización de spaCy (ver resultados) |
| Tildes | `[atención]` | `[atencion]` | Se quitan **al final**: el lematizador necesita las tildes, pero los usuarios a menudo las omiten |
| N-gramas | `no recom` | se agrega el bigrama | Captura negaciones y nombres de platos compuestos («piqu mach») |
| TF-IDF | | 1.948 términos | `max_features=5000`, `min_df=2` (descarta errores de tipeo únicos), `max_df=0.9`, `sublinear_tf=True` |

Todos los parámetros están en `params.yaml`.

## 4. Modelos

### Por qué estos modelos

- **Naive Bayes multinomial.** Estima, para cada clase, la probabilidad de cada término y combina esas probabilidades con la regla de Bayes, suponiendo que los términos son independientes. Es el modelo clásico para texto: entrena en milisegundos y funciona bien con vocabularios grandes y pocos datos. Su debilidad es justamente la independencia: «no» y «recom» se cuentan por separado.
- **Regresión logística multinomial.** Aprende un peso por término y por clase y convierte la suma ponderada en probabilidades con softmax. No supone independencia, así que puede dar un peso fuerte al bigrama «no recom» y uno distinto a «recom». Sus coeficientes se pueden interpretar (pestaña **Métricas**, «Términos más influyentes»).
- **Línea base.** `DummyClassifier` que siempre predice la clase más frecuente. Cualquier modelo útil debe superarla con claridad.

### Búsqueda de la configuración

`src/models/experiments.py` evalúa con validación cruzada estratificada de 5 particiones, **solo sobre el conjunto de entrenamiento**:

- 2 normalizaciones (stemming, lematización) × 2 listas de stop words (completa, ajustada) × 2 rangos de n-gramas (1-1, 1-2) × 2 modelos = 16 configuraciones.
- Cada configuración con su búsqueda de hiperparámetros (`GridSearchCV`):
  - Naive Bayes: `alpha` ∈ {0,1; 0,5; 1,0} y `fit_prior` ∈ {sí, no}.
  - Regresión logística: `C` ∈ {0,5; 1; 2; 5} y `class_weight` ∈ {ninguno, balanced}.

El preprocesamiento no aprende nada de los datos, así que se calcula una vez por variante. El vocabulario de TF-IDF sí se aprende dentro de cada partición, sin fuga de información.

### Efecto de cada decisión (F1 macro en validación cruzada, mejor configuración)

| Decisión | Opción A | Opción B | Diferencia |
|---|---|---|---|
| Stop words | completa: 0,854 | **ajustada: 0,875** | +2,1 puntos |
| N-gramas | unigramas: 0,864 | **uni + bigramas: 0,875** | +1,1 puntos |
| Normalización | lematización: 0,874 | **stemming: 0,875** | empate (0,04 puntos, menor que la desviación estándar de 2,8 puntos) |
| Modelo | Naive Bayes: 0,857 | **regresión logística: 0,875** | +1,7 puntos |

Stemming y lematización empatan. Se queda stemming porque es más simple y más rápido, y no depende de un modelo de spaCy que comete errores con palabras del dominio («recomiendo» → «recomeir»).

### Comparación en el conjunto de prueba

| Modelo | Configuración | F1 macro CV | Exactitud | F1 macro | F1 neg. | F1 neutro | F1 pos. |
|---|---|---|---|---|---|---|---|
| Línea base | siempre «positivo» | 0,217 | 48,3 % | 0,217 | 0,000 | 0,000 | 0,651 |
| Naive Bayes | stemming, ajustada, 1-2, `alpha=0.1`, `fit_prior=False` | 0,857 ± 0,026 | 88,9 % | 0,874 | 0,907 | 0,794 | 0,920 |
| **Regresión logística** | stemming, ajustada, 1-2, `C=2`, `class_weight=balanced` | **0,875 ± 0,028** | **89,5 %** | **0,880** | 0,891 | **0,813** | **0,935** |

**En producción: regresión logística**, elegida por F1 macro en validación cruzada. En prueba confirma la ventaja, sobre todo en la clase neutra, la más difícil.

Matriz de confusión de la regresión logística (filas: real; columnas: predicho):

| | negativo | neutro | positivo |
|---|---|---|---|
| **negativo** | **160** | 13 | 9 |
| **neutro** | 12 | **104** | 8 |
| **positivo** | 5 | 15 | **266** |

Casi todos los errores involucran a la clase neutra; solo 14 de 592 reseñas se confunden entre negativo y positivo.

## 5. Análisis de errores

Exactitud del modelo en producción según el fenómeno de cada reseña de prueba:

| Fenómeno | Reseñas | Exactitud |
|---|---|---|
| Etiqueta ruidosa | 20 | 0 % |
| Opinión mixta | 169 | 79 % |
| Simple | 274 | 98 % |
| Negación | 114 | 99 % |
| Ironía | 15 | 100 % |

- **Etiquetas ruidosas.** «El sabor es estándar. Habrá que darle otra oportunidad» está etiquetada como negativa; el modelo dice neutra con 99 %. El texto le da la razón al modelo: el error es del dato. Estas 20 reseñas son el techo de cualquier modelo.
- **Opiniones mixtas.** «El fricasé estaba súper salado, sin embargo el personal es cordial» (real: negativa, predicho: neutra). TF-IDF suma evidencia positiva y negativa sin saber cuál pesa más para el autor.
- **Dobles negaciones.** «Las empanadas no estaban nada mal y no salimos con hambre» es positiva, pero tiene tres negaciones y el modelo la clasifica como negativa. Una bolsa de palabras no sabe que las negaciones se anulan.
- **Ironía.** Aquí se acierta siempre, pero es un efecto de los datos sintéticos: las frases irónicas se repiten entre entrenamiento y prueba. Con reseñas reales sería uno de los peores casos.

**Mejoras posibles:** más ejemplos de opiniones mixtas, trigramas para negaciones dobles, o modelos que leen contexto (embeddings, transformers), fuera del alcance de este proyecto.

## 6. Pipeline reproducible

- **Un comando:** `python -m src.pipeline` (o `docker compose up`) genera los datos si faltan, los carga en PostgreSQL, entrena las 16 configuraciones, elige la mejor y guarda `artifacts/model.joblib`, `metrics.json`, `experiments.csv` y `errors.csv`.
- **El backend no reimplementa nada:** carga `model.joblib`, un `Pipeline` de scikit-learn con tres pasos: `TextPreprocessor` → `TfidfVectorizer` → modelo.
- **Versiones fijas** en `requirements.txt`, y una sola imagen de Docker para entrenar y servir: el modelo siempre se carga con las mismas librerías que lo crearon.
- **Semilla 42** en el generador, la división, la validación cruzada y la regresión logística. Dos ejecuciones producen las mismas métricas.
- **Cada entrenamiento** queda registrado en la tabla `entrenamientos` con su versión y métricas.

## 7. Arquitectura

| Servicio | Imagen | Responsabilidad |
|---|---|---|
| `db` | postgres:16-alpine | Tablas `resenas` (datos crudos), `entrenamientos` y `predicciones` |
| `trainer` | `Dockerfile` (Python 3.12) | Ejecuta `src.pipeline` y termina |
| `api` | la misma imagen | FastAPI en el puerto 8000; carga el modelo al iniciar |
| `frontend` | `frontend/Dockerfile` (Node para compilar, nginx para servir) | React en el puerto 8080; reenvía `/api` a la API |

Endpoints (detalle en [`api.md`](api.md)): `GET /api/health`, `POST /api/predict`, `POST /api/predict/batch`, `GET /api/metrics`, `GET /api/history`, `GET /api/trainings`.

**Por qué FastAPI:** valida las entradas a partir de los modelos de Pydantic, genera la documentación Swagger sin trabajo extra y es el framework más usado en Python para servir modelos.

**Por qué React:** permite construir la vista paso a paso del preprocesamiento y los paneles de métricas con componentes reutilizables. Vite compila en segundos y nginx sirve el resultado sin Node en producción.

**Pruebas automatizadas** (`tests/`, 34 casos): cada paso de NLP, que la lista ajustada conserve las negaciones, que el preprocesador sobreviva a guardarse y cargarse, que los modelos superen a la línea base, y todas las validaciones de la API.

## 8. Limitaciones

- Los datos son sintéticos: las métricas no representan el desempeño con reseñas reales.
- El modelo no entiende contexto largo, ironía real ni comparaciones («mejor que el de enfrente»).
- No hay autenticación: es un ejemplo para ejecutar en local.

## 9. Uso de asistentes de IA

El código, los datos sintéticos y esta documentación se elaboraron con ayuda de un asistente de IA (Claude) y fueron revisados y ejecutados por el docente. Todas las cifras de este informe salen de una ejecución real de `python -m src.pipeline`.
