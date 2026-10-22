# Datos

## Origen, licencia y fecha

| Campo | Valor |
|---|---|
| Origen | Sintético. Generado por `src/data/generate.py` con semilla 42 (`python -m src.data.generate`). |
| Licencia | Creado para este ejemplo. Uso libre, sin restricciones (CC0). |
| Fecha de obtención | 22 de octubre de 2026 |
| Tamaño | 3.045 filas crudas; 2.960 después de limpiar |
| Idioma | Español de Bolivia, con jerga, errores de tipeo y emojis |

Los nombres de restaurantes son inventados. No hay datos de personas.

**Por qué sintético.** Para que el ejemplo funcione sin Internet, sin claves y con exactamente los mismos resultados en cualquier computadora. El generador produce a propósito los problemas de un dataset real: clases desbalanceadas, textos vacíos, duplicados, negaciones, ironía, opiniones mixtas y un 4 % de etiquetas equivocadas.

**Limitación.** Las frases salen de plantillas, así que se repiten más que en la realidad. Por eso fenómenos difíciles, como la ironía, resultan más fáciles aquí que con reseñas reales. Los grupos deben trabajar con datos reales.

## Diccionario

`raw/resenas.csv`, también cargado en la tabla `resenas` de PostgreSQL:

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | entero | Identificador único |
| `restaurante` | texto | Nombre del restaurante (ficticio) |
| `ciudad` | texto | Santa Cruz, La Paz, Cochabamba, Sucre o Tarija. ~3 % vacía |
| `fecha` | fecha | Fecha de la reseña, entre 2024 y 2026 |
| `texto` | texto | La reseña. ~1 % vacía. **Entrada del modelo** |
| `estrellas` | 1 a 5 | Puntuación del autor |
| `sentimiento` | `negativo`, `neutro`, `positivo` | **Etiqueta a predecir** |
| `fenomeno` | texto | Solo existe porque los datos son sintéticos: `simple`, `negacion`, `mixta`, `ironia` o `etiqueta_ruidosa`. Se usa **únicamente** para agrupar el análisis de errores; nunca como variable del modelo |

## Distribución

| Sentimiento | Reseñas limpias | % |
|---|---|---|
| Positivo | 1.428 | 48 % |
| Negativo | 911 | 31 % |
| Neutro | 621 | 21 % |
