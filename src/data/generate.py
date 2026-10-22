"""Synthetic restaurant reviews written in Bolivian Spanish.

The generator lets the example run offline and be repeated exactly with the
same seed. It deliberately produces the problems a real dataset has: class
imbalance, negations, irony, mixed opinions, typos, emojis, URLs, missing
values, duplicates and a few wrong labels.

The column ``fenomeno`` exists only because the data is synthetic: it records
which phenomenon each review contains, so the error analysis can be grouped by
it. It is never used as a model feature. Real projects must use real data.

Usage:
    python -m src.data.generate            # writes data/raw/resenas.csv
"""

import argparse
import random
import re
from datetime import date, timedelta

import pandas as pd

from src.config import ROOT, load_params

RESTAURANTS = [
    "La Cuchara de Barro", "Brasa y Leña", "El Patio Vallegrandino", "Sazón Andino",
    "Fogón del Sur", "Casa del Majadito", "Punto Salteña", "La Olla Feliz",
    "Mirador Chapaco", "El Tacú", "Leña y Carbón", "Doña Tola Cocina",
]
CITIES = ["Santa Cruz", "La Paz", "Cochabamba", "Sucre", "Tarija"]

# (name, gender, number) so adjectives can agree with the dish.
DISHES = [
    ("majadito", "m", "s"), ("pique macho", "m", "s"), ("silpancho", "m", "s"),
    ("sopa de maní", "f", "s"), ("salteñas", "f", "p"), ("chicharrón", "m", "s"),
    ("fricasé", "m", "s"), ("lomo montado", "m", "s"), ("sajta de pollo", "f", "s"),
    ("pollo a la broaster", "m", "s"), ("cuñapés", "m", "p"), ("charque", "m", "s"),
    ("laping", "m", "s"), ("empanadas", "f", "p"), ("costillas a la parrilla", "f", "p"),
]
DRINKS = ["mocochinchi", "somó", "jugo de tumbo", "café", "refresco de linaza", "api"]

ARTICLES = {("m", "s"): "el", ("f", "s"): "la", ("m", "p"): "los", ("f", "p"): "las"}

# Adjective forms: masculine singular, feminine singular, masculine plural, feminine plural.
ADJECTIVES = {
    "rico": ("rico", "rica", "ricos", "ricas"),
    "delicioso": ("delicioso", "deliciosa", "deliciosos", "deliciosas"),
    "sabroso": ("sabroso", "sabrosa", "sabrosos", "sabrosas"),
    "exquisito": ("exquisito", "exquisita", "exquisitos", "exquisitas"),
    "frio": ("frío", "fría", "fríos", "frías"),
    "salado": ("salado", "salada", "salados", "saladas"),
    "crudo": ("crudo", "cruda", "crudos", "crudas"),
    "insipido": ("insípido", "insípida", "insípidos", "insípidas"),
    "recalentado": ("recalentado", "recalentada", "recalentados", "recalentadas"),
    "aceitoso": ("aceitoso", "aceitosa", "aceitosos", "aceitosas"),
    "normal": ("normal", "normal", "normales", "normales"),
    "aceptable": ("aceptable", "aceptable", "aceptables", "aceptables"),
    "fresco": ("fresco", "fresca", "frescos", "frescas"),
}

# Aspect phrases talk about one thing (food, service, price...). Global
# phrases are a verdict on the whole visit and never appear in mixed reviews.
POSITIVE_ASPECT = [
    "{D} {V} {INT}{A:rico}", "{D} {V} {A:delicioso}", "{D} {V} {A:exquisito}",
    "{D} {V} {INT}{A:sabroso} y bien servido", "la sazón es excelente",
    "{D} tiene sabor de casa", "las porciones son abundantes", "la atención fue {INT}amable",
    "el mesero fue atento en todo momento", "nos atendieron rápido", "el personal es cordial",
    "los precios son justos", "buena relación calidad precio", "por Bs {PRICE} comes muy bien",
    "el ambiente es agradable", "es acogedor y tranquilo", "todo estaba impecable",
    "los baños estaban limpios", "el {DRINK} bien helado, perfecto para el calor",
    "la música en vivo le da un toque especial", "el pedido llegó en {MINS_SHORT} minutos",
    "{D} es lo mejor que probé en {CITY}",
]
POSITIVE_GLOBAL = ["nos encantó", "una experiencia excelente", "el trato fue de diez",
                   "todo muy bueno"]
NEGATIVE_ASPECT = [
    "{D} {V} {A:frio}", "{D} {V} {INT}{A:salado}", "{D} {V} {A:crudo}",
    "{D} {V} {A:insipido}", "{D} {V} {A:recalentado}", "{D} {V} {INT}{A:aceitoso}",
    "la atención fue pésima", "el mesero fue grosero", "nos ignoraron por completo",
    "esperamos {MINS} minutos para que nos atiendan", "el pedido tardó más de una hora",
    "es caro para lo que sirven", "las porciones son ridículamente pequeñas",
    "el baño estaba sucio", "encontré un pelo en el plato", "había moscas por todas partes",
    "me cobraron de más", "el {DRINK} estaba aguado", "me cayó mal al estómago",
    "huele a fritura vieja", "la mesa estaba pegajosa", "se equivocaron de pedido dos veces",
]
NEGATIVE_GLOBAL = ["una decepción total", "pésima experiencia", "un desastre"]
POSITIVE = POSITIVE_ASPECT + POSITIVE_GLOBAL
NEGATIVE = NEGATIVE_ASPECT + NEGATIVE_GLOBAL
# A negation flips a positive phrase: the meaning is negative. Most of these
# reuse the vocabulary of POSITIVE_ASPECT, so only "no" tells them apart.
NEGATED_POSITIVE = [
    "{D} no {V} {A:rico}", "{D} no {V} {A:delicioso}", "{D} no {V} {INT}{A:sabroso}",
    "{D} no {V} {A:fresco}", "la atención no fue amable", "el mesero no fue atento",
    "no nos atendieron rápido", "los precios no son justos", "las porciones no son abundantes",
    "el ambiente no es agradable", "no es acogedor", "los baños no estaban limpios",
    "no vale lo que cuesta", "no me gustó para nada", "no lo recomiendo",
    "no es tan bueno como dicen", "no fue una buena experiencia", "el servicio no es nada bueno",
    "no nos atendieron bien", "nunca más vuelvo", "nada recomendable",
]
# A negation flips a negative phrase: the meaning is positive.
NEGATED_NEGATIVE = [
    "{D} no {V} nada mal", "{D} no {V} {A:salado} como en otros lados",
    "{D} no {V} {A:aceitoso}", "no tuvimos que esperar", "no es caro", "no tardaron nada",
    "nunca nos decepciona", "no hay nada que reclamar", "no me arrepiento de haber ido",
    "no encontramos ni un solo problema", "no salimos con hambre",
    "nunca tuvimos una mala experiencia", "nada que ver con la mala fama que tiene",
]
NEUTRAL = [
    "{D} {V} {A:normal}", "{D} {V} {A:aceptable}", "la atención fue correcta",
    "el precio es el promedio de la zona", "nada fuera de lo común", "cumple con lo básico",
    "ni bueno ni malo", "es lo que esperaba", "es pequeño", "abren de {HOUR} a 22:00",
    "queda cerca de la plaza principal", "un domingo al mediodía", "{D} y un {DRINK}",
    "tienen estacionamiento en la calle", "aceptan pago con QR",
    "hay que reservar los fines de semana", "el menú es corto", "regular, como cualquier otro",
    "el sabor es estándar", "la porción es la de siempre", "estuvo bien, sin más",
    "ni fu ni fa", "esta vez estuvo más o menos", "tiene mesas afuera",
    # Neutral, but written with sentiment words.
    "no está mal pero tampoco es para tanto", "esperaba más", "podría ser mejor",
    "bien pero nada especial", "rico pero caro", "ni muy rico ni muy feo",
    "algunas cosas buenas, otras no tanto", "{D} bien, lo demás normal",
]
# Irony: positive words, negative meaning. The hardest case for a bag of words.
# The second half has no negative word at all.
IRONY = [
    "excelente servicio, solo esperamos {MINS} minutos",
    "qué rapidez, una hora para un {DRINK}",
    "genial, {D} llegó {A:frio}. gracias por nada",
    "muy recomendable si te gusta esperar",
    "cinco estrellas en paciencia, porque la comida nunca llegó",
    "felicitaciones al chef por {D} más {A:salado} de {CITY}",
    "increíble, lograron arruinar algo tan simple como {D}",
    "lo mejor fue la cuenta, rapidísima de cobrar",
    "atención de primera: nos ignoraron con mucha elegancia",
    "buenísimo, si lo que buscas es pasar hambre",
    "qué maravilla de atención, súper atentos a su celular",
    "todo perfecto, sobre todo el pelo en la sopa",
    "excelente lugar para practicar la paciencia",
    "la mejor dieta: pedir aquí y esperar sentado",
]

OPENERS = [
    "fuimos en familia", "vine por recomendación de un amigo", "primera vez que voy",
    "fuimos por el cumpleaños de mi mamá", "pedí por delivery", "almuerzo del día",
    "segunda vez que vamos", "salida con compañeros de trabajo", "cena de aniversario",
]
CLOSERS = {
    "positivo": ["volvería sin dudar", "100% recomendado", "10/10", "lo recomiendo",
                 "seguro volvemos", "felicitaciones al equipo"],
    "negativo": ["no vuelvo", "no lo recomiendo", "qué decepción",
                 "cero estrellas si se pudiera", "una pena"],
    "neutro": ["habrá que darle otra oportunidad", "puede mejorar",
               "está bien para salir del paso", "en general regular", "ni fu ni fa"],
}
INTENSIFIERS = ["", "", "muy ", "súper ", "bastante ", "re "]
EMOJIS = {
    "positivo": ["😍", "👍", "😋", "🔥", "❤️", "👌"],
    "negativo": ["😡", "👎", "🤢", "😤", "💩"],
    "neutro": ["😐", "🤷", "🙂"],
    "ironia": ["🙃", "🙄", "😒"],
}
SLANG = [(r"\bque\b", "q"), (r"\bpor\b", "x"), (r"\bporque\b", "pq"),
         (r"\btambién\b", "tb"), (r"\bmuy\b", "mui")]
ACCENTS = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")


def _fill(template: str, rng: random.Random, dish: tuple) -> str:
    name, gender, number = dish
    form = {("m", "s"): 0, ("f", "s"): 1, ("m", "p"): 2, ("f", "p"): 3}[(gender, number)]
    text = re.sub(r"\{A:(\w+)\}", lambda m: ADJECTIVES[m.group(1)][form], template)
    return (text
            .replace("{D}", f"{ARTICLES[(gender, number)]} {name}")
            .replace("{V}", "estaba" if number == "s" else "estaban")
            .replace("{INT}", rng.choice(INTENSIFIERS))
            .replace("{DRINK}", rng.choice(DRINKS))
            .replace("{CITY}", rng.choice(CITIES))
            .replace("{PRICE}", str(rng.choice([25, 30, 35, 40, 45, 50, 60])))
            .replace("{MINS_SHORT}", str(rng.choice([10, 15, 20])))
            .replace("{MINS}", str(rng.choice([40, 50, 60, 75, 90])))
            .replace("{HOUR}", rng.choice(["11:00", "12:00", "18:00"])))


def _clauses(rng: random.Random, bank: list, k: int) -> list[str]:
    """k different phrases from a bank, each one about a random dish."""
    return [_fill(t, rng, rng.choice(DISHES)) for t in rng.sample(bank, k)]


def _join(parts: list[str], rng: random.Random) -> str:
    text = parts[0]
    for part in parts[1:]:
        if part.startswith(("pero ", "aunque ", "sin embargo ")):
            text += ", " + part
        else:
            text += rng.choice([". ", ", ", " y ", ". "]) + part
    return text


def _compose(label: str, rng: random.Random) -> tuple[str, str]:
    """Return (text, phenomenon) for a review whose true label is ``label``."""
    phenomenon = "simple"
    parts = [rng.choice(OPENERS)] if rng.random() < 0.35 else []
    contrast = rng.choice(["pero ", "aunque ", "", "sin embargo "])

    if label == "positivo":
        parts += _clauses(rng, POSITIVE, rng.choice([1, 1, 2, 2, 3]))
        if rng.random() < 0.25:
            parts += _clauses(rng, NEGATED_NEGATIVE, 1)
            phenomenon = "negacion"
        if rng.random() < 0.25:
            parts.append(contrast + _clauses(rng, NEUTRAL + NEGATIVE_ASPECT, 1)[0])
            phenomenon = "mixta"
    elif label == "negativo":
        if rng.random() < 0.12:
            parts += _clauses(rng, IRONY, 1)
            phenomenon = "ironia"
        else:
            parts += _clauses(rng, NEGATIVE, rng.choice([1, 1, 2, 3]))
            if rng.random() < 0.35:
                parts += _clauses(rng, NEGATED_POSITIVE, 1)
                phenomenon = "negacion"
            if rng.random() < 0.20:
                parts.append(contrast + _clauses(rng, POSITIVE_ASPECT, 1)[0])
                phenomenon = "mixta"
    else:
        if rng.random() < 0.5:
            parts += _clauses(rng, NEUTRAL, rng.choice([1, 1, 2]))
        else:
            good = _clauses(rng, POSITIVE_ASPECT, 1)[0]
            bad = _clauses(rng, NEGATIVE_ASPECT, 1)[0]
            first, second = (good, bad) if rng.random() < 0.5 else (bad, good)
            parts += [first, contrast + second]
            phenomenon = "mixta"

    if rng.random() < 0.40 and phenomenon != "ironia":
        parts.append(rng.choice(CLOSERS[label]))
    return _join(parts, rng), phenomenon


def _add_noise(text: str, label: str, phenomenon: str, rng: random.Random) -> str:
    """Make the text look like something people actually type."""
    sentences = [s.strip() for s in text.split(". ")]
    if rng.random() < 0.7:
        sentences = [s[:1].upper() + s[1:] for s in sentences]
    text = ". ".join(sentences) + rng.choice([".", ".", "", "!", "!!!"])

    if rng.random() < 0.35:
        text = text.translate(ACCENTS)
    if rng.random() < 0.20:
        pattern, short = rng.choice(SLANG)
        text = re.sub(pattern, short, text)
    if rng.random() < 0.15:  # elongation: "riiicooo"
        words = text.split()
        i = rng.randrange(len(words))
        if words[i][-1:].isalpha():
            words[i] += words[i][-1] * rng.randint(2, 4)
        text = " ".join(words)
    if rng.random() < 0.08:  # one word in capitals
        words = text.split()
        i = rng.randrange(len(words))
        words[i] = words[i].upper()
        text = " ".join(words)
    if rng.random() < 0.12:  # typo: drop one letter
        words = text.split()
        i = rng.randrange(len(words))
        if len(words[i]) > 4:
            j = rng.randrange(1, len(words[i]) - 1)
            words[i] = words[i][:j] + words[i][j + 1:]
        text = " ".join(words)
    if rng.random() < 0.30:
        key = "ironia" if phenomenon == "ironia" else label
        text += " " + "".join(rng.choices(EMOJIS[key], k=rng.randint(1, 2)))
    if rng.random() < 0.03:
        text += f" fotos: https://fotos.ejemplo.bo/r/{rng.getrandbits(32):08x}"
    if rng.random() < 0.05:
        text += " #ComidaBoliviana"
    return text


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    labels = rng.choices(["positivo", "negativo", "neutro"], weights=[0.50, 0.30, 0.20], k=n)
    start = date(2024, 1, 1)
    rows = []
    for i, label in enumerate(labels, start=1):
        text, phenomenon = _compose(label, rng)
        text = _add_noise(text, label, phenomenon, rng)
        stars = {"positivo": rng.choice([4, 5, 5]), "negativo": rng.choice([1, 1, 2]),
                 "neutro": 3}[label]
        if rng.random() < 0.04:  # annotation error: the label disagrees with the text
            label = rng.choice([x for x in ["positivo", "negativo", "neutro"] if x != label])
            phenomenon = "etiqueta_ruidosa"
        rows.append({
            "id": i,
            "restaurante": rng.choice(RESTAURANTS),
            "ciudad": rng.choice(CITIES) if rng.random() > 0.03 else None,
            "fecha": start + timedelta(days=rng.randrange(1000)),
            "texto": text if rng.random() > 0.01 else None,
            "estrellas": stars,
            "sentimiento": label,
            "fenomeno": phenomenon,
        })
    df = pd.DataFrame(rows)

    # Duplicates: the same review submitted twice, with a new id.
    dup = df.sample(frac=0.015, random_state=seed).copy()
    dup["id"] = range(n + 1, n + 1 + len(dup))
    return pd.concat([df, dup], ignore_index=True)


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=params["data"]["n_reviews"])
    parser.add_argument("--seed", type=int, default=params["seed"])
    parser.add_argument("--out", default=str(ROOT / params["data"]["raw_path"]))
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} reseñas guardadas en {args.out}")
    print(df["sentimiento"].value_counts().to_string())


if __name__ == "__main__":
    main()
