# Modificaciones en vivo

En la defensa, el docente pide un cambio pequeño. Hay que **implementarlo, ejecutarlo, interpretar el resultado y subirlo con un commit propio**. Esta guía resuelve los ejemplos del enunciado sobre este proyecto. Cada receta se probó: las cifras salen de una ejecución real.

## Antes de empezar

```bash
git switch -c live/<tu-nombre>-<cambio>    # rama propia para el cambio
docker compose up -d                       # sistema levantado
```

Ciclo para cualquier cambio en `params.yaml` o `src/`:

```bash
docker compose run --rm trainer python -m src.pipeline   # reentrena (~30 s)
docker compose restart api                               # la API carga el modelo nuevo
```

La pestaña **Historial → Entrenamientos** muestra la versión anterior y la nueva lado a lado. Ahí se compara.

> **Regla de oro al interpretar:** se decide con **F1 macro de validación cruzada**, no con el conjunto de prueba. Con 592 reseñas de prueba, una diferencia de medio punto puede ser ruido.

---

## NLP · Agregar o quitar una stop word

**Pedido típico:** «Haz que "pero" sea stop word y mira qué pasa».

1. En `params.yaml`, quita `pero` de `nlp.keep_words`. Al salir de esa lista, vuelve a ser stop word, como en la lista original de spaCy.
2. Reentrena y reinicia la API.
3. Prueba en **Analizar**: «Rico pero caro». En el paso 3, «pero» aparece ahora tachado.

**Resultado medido:** F1 macro CV baja de 0,875 a 0,871 y el F1 de la clase **neutra** en prueba, de 0,813 a 0,802.

**Interpretación:** «pero» era el término más influyente de la clase neutra (pestaña **Métricas**, «Términos más influyentes»). Las reseñas neutras suelen ser mixtas («rico, pero caro») y «pero» es la señal de esa mezcla. Sin esa palabra el modelo confunde más neutras con positivas o negativas.

**Variante:** quitar `"no"` de `keep_words`. F1 macro CV baja de 0,875 a 0,865, pero F1 en prueba **sube** de 0,880 a 0,883. Buen momento para explicar la regla de oro: la validación cruzada promedia 5 particiones y dice que empeora; la subida en prueba está dentro del ruido.

**Para agregar** una stop word del dominio, súmala a `nlp.extra_stopwords`.

```bash
git commit -am "feat(nlp): treat 'pero' as a stop word

Neutral F1 drops from 0.813 to 0.802: 'pero' signals mixed reviews."
```

---

## NLP · Cambiar el rango de n-gramas o stemming por lematización

El script elige la mejor combinación entre las opciones de `experiments`. Para **forzar** una opción, deja solo esa en la lista:

```yaml
experiments:
  normalizers: [lemma]          # antes: [stem, lemma]
  ngram_ranges: [[1, 3]]        # antes: [[1, 1], [1, 2]]
```

Reentrena y compara en **Historial → Entrenamientos**. Con la lematización forzada, el F1 macro CV queda en 0,874 frente a 0,875: empate técnico. Para mostrar el efecto en una oración, usa **Analizar** con «Los meseros nos atendieron rápido»: con stemming se ve `meser`, `atend`; con lematización, `mesero`, `atender`.

```bash
git commit -am "chore(nlp): force lemmatization to compare it with stemming"
```

---

## Modelado · Cambiar un hiperparámetro y comparar

```yaml
experiments:
  models:
    logistic_regression:
      C: [0.1]                  # antes: [0.5, 1.0, 2.0, 5.0]
      class_weight: [balanced]
```

Reentrena. Un `C` pequeño significa **más regularización**: los coeficientes se encogen y el modelo se vuelve más simple.

**Resultado medido:** el F1 macro CV de la regresión logística baja de 0,875 a 0,852 (subajuste: la regularización fuerte le quita información útil). Como ahora Naive Bayes (0,857) la supera en validación cruzada, **el script pone a Naive Bayes en producción** sin tocar código. La pestaña **Métricas** lo muestra en la tarjeta «En producción». Es la mejor demostración de que la elección del modelo depende de los datos y de la búsqueda, no de una preferencia.

Para Naive Bayes, el equivalente es `alpha` (suavizado de Laplace): más grande da probabilidades más parejas entre términos.

---

## Modelado · Agregar una métrica

**Pedido típico:** «Calcula la exactitud balanceada» o «el error por categoría».

`src/models/evaluate.py`:

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ...

def classification_metrics(y_true, y_pred) -> dict:
    ...
    return {
        "accuracy": ...,
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        ...
```

`src/models/train.py`, en `_print_summary`, agrega `balanced={m['test']['balanced_accuracy']:.3f}` al `print`.

Opcional, para verla en la interfaz, en `frontend/src/components/Metrics.jsx` (tabla «Comparación de modelos») agrega la columna:

```jsx
<th className="num">Exactitud balanceada</th>
...
<td className="num">{pct(m.test.balanced_accuracy)}</td>
```

y reconstruye el frontend: `docker compose up -d --build frontend`.

**Interpretación:** la exactitud balanceada es el promedio del recall de cada clase. La línea base obtiene 33 % (acierta todas las positivas y ninguna de las otras dos), aunque su exactitud normal es 48 %. Por eso, con clases desbalanceadas, la exactitud normal engaña.

El **error por categoría** ya existe: es `1 − recall` de cada clase en la tabla de métricas por clase. La matriz de confusión dice con qué clase se confunde cada una.

---

## Backend · Nueva validación: rechazar textos de menos de 3 palabras

`api/schemas.py`, al final de `validate_text`:

```python
    if len(value.split()) < 3:
        raise ValueError("El texto debe tener al menos 3 palabras.")
    return value
```

La validación vale para `/api/predict` y para cada fila de `/api/predict/batch`, porque ambos usan `validate_text`. No hace falta reentrenar: `docker compose restart api`.

Ejecuta las pruebas: `docker compose --profile dev run --rm --build tests`. **Fallan dos**, y es lo esperado:

- `test_invalid_input_gets_a_clear_message[... Campo no permitido]` envía `"rico"` (una palabra): ahora el primer error es el de las 3 palabras. Cambia el texto a `"muy rico todo"`.
- `test_batch_from_txt_skips_invalid_lines` usa «Riquísimo todo» (2 palabras). Cambia las líneas a «Riquísimo todo, volveremos» y «Pésima atención del mesero».

Agrega la prueba de la regla nueva en `tests/test_api.py`:

```python
def test_rejects_texts_with_fewer_than_three_words(client):
    response = client.post("/api/predict", json={"texto": "muy rico"})
    assert response.status_code == 422
    assert response.json()["error"] == "El texto debe tener al menos 3 palabras."
```

Con los ajustes: 35 pruebas pasan. Mensaje de commit sugerido: `feat(api): reject reviews with fewer than three words`.

---

## Backend · Nuevo endpoint

**Pedido típico:** «Agrega un endpoint que devuelva las stop words que usa el modelo».

`api/main.py`:

```python
from src.nlp.preprocessing import stopwords

@app.get("/api/stopwords", responses=ERRORS, tags=["modelo"])
def stopword_list():
    """Stop words que usa el modelo en producción."""
    require_model()
    pre = service.pipeline.named_steps["preprocess"]
    words = stopwords(pre.stopword_list, tuple(pre.keep_words), tuple(pre.extra_stopwords))
    return {"lista": pre.stopword_list, "total": len(words), "palabras": sorted(words)}
```

`docker compose restart api` y prueba en http://localhost:8000/api/docs. Las stop words se leen **del pipeline cargado**, no de `params.yaml`: así reflejan el modelo que realmente está sirviendo.

Prueba:

```python
def test_stopwords_endpoint(client):
    body = client.get("/api/stopwords").json()
    assert "no" not in body["palabras"] and "el" in body["palabras"]
```

---

## Frontend · Mostrar un dato nuevo de la API

**Pedido típico:** «Muestra cuántos términos llegan al modelo y avisa si no llega ninguno».

`frontend/src/components/Analyze.jsx`, debajo de `<SentimentBars … />`:

```jsx
<p className="muted small">
  Términos que llegan al modelo: {result.preprocesamiento.final.length}
</p>
{result.preprocesamiento.final.length === 0 && (
  <div className="error">
    El texto quedó vacío después del preprocesamiento: la predicción no es confiable.
  </div>
)}
```

`docker compose up -d --build frontend` y prueba con «el de la que» (solo stop words). Sin términos, el modelo solo puede usar su intercepto: responde «neutro» con 43 %, una predicción que no dice nada del texto.

## Frontend · Cambiar una visualización

**Pedido típico:** «Ordena las barras de probabilidad de mayor a menor».

`frontend/src/components/ui.jsx`, en `SentimentBars`, ordena las filas antes de pasarlas a `Bars`:

```jsx
rows={Object.keys(SENTIMENTS)
  .map((key) => ({ label: SENTIMENTS[key].name, value: probabilities[key] ?? 0, className: SENTIMENTS[key].className }))
  .sort((a, b) => b.value - a.value)}
```

---

## Al terminar cualquier cambio

```bash
docker compose --profile dev run --rm tests      # las pruebas siguen pasando
git add -A
git commit -m "<tipo>(<área>): <qué cambió y por qué>"
git push -u origin HEAD
```

Abre el pull request y explica el resultado: qué métrica cambió, cuánto y por qué.
