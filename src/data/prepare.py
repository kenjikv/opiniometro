"""Cleaning, exploratory summary and train/test split."""

from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABELS
from src.nlp.preprocessing import normalize, stopwords, strip_accents, tokenize


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop reviews without text and exact duplicates. Returns the clean data
    and a report of what was removed, for the EDA panel."""
    missing = {col: int(n) for col, n in raw.isna().sum().items() if n}
    df = raw.copy()
    df["texto"] = df["texto"].fillna("").str.strip()
    empty = df["texto"].eq("")
    df = df[~empty]
    duplicated = df.duplicated(subset="texto", keep="first")
    df = df[~duplicated].reset_index(drop=True)
    report = {
        "raw_rows": len(raw),
        "missing_values": missing,
        "empty_texts_removed": int(empty.sum()),
        "duplicates_removed": int(duplicated.sum()),
        "clean_rows": len(df),
    }
    return df, report


def describe(df: pd.DataFrame, keep_words: tuple, extra_stopwords: tuple,
             top_n: int = 12) -> dict:
    """Class distribution, text length and most frequent words per class."""
    words = df["texto"].str.split().str.len()
    stop = stopwords("adjusted", keep_words, extra_stopwords)
    top_words = {}
    for label in LABELS:
        counter = Counter()
        for text in df.loc[df["sentimiento"] == label, "texto"]:
            counter.update(t for t in tokenize(normalize(text))
                           if strip_accents(t) not in stop)
        top_words[label] = counter.most_common(top_n)

    return {
        "class_counts": df["sentimiento"].value_counts().reindex(LABELS).to_dict(),
        "length_words": {
            label: {
                "mean": round(float(words[df["sentimiento"] == label].mean()), 1),
                "median": float(words[df["sentimiento"] == label].median()),
                "max": int(words[df["sentimiento"] == label].max()),
            } for label in LABELS
        },
        "length_histogram": _histogram(words),
        "top_words": top_words,
        "stars_by_label": pd.crosstab(df["sentimiento"], df["estrellas"])
                            .reindex(LABELS).to_dict(orient="index"),
    }


def _histogram(values: pd.Series, width: int = 5) -> list[dict]:
    bins = (values // width) * width
    counts = bins.value_counts().sort_index()
    return [{"from": int(b), "to": int(b + width - 1), "count": int(c)}
            for b, c in counts.items()]


def split(df: pd.DataFrame, test_size: float, seed: int):
    """Stratified split with a fixed seed: same proportions of each class in
    train and test, and the same split every time."""
    return train_test_split(df, test_size=test_size, stratify=df["sentimiento"],
                            random_state=seed)
