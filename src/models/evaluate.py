"""Test-set metrics, error analysis and the most influential terms."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from src.config import LABELS


def classification_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro",
                                         labels=LABELS, zero_division=0)), 4),
        "per_class": {
            label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                    "f1": round(float(f), 4), "support": int(s)}
            for label, p, r, f, s in zip(LABELS, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def error_table(pipeline: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    """All test reviews with prediction and confidence; errors flagged."""
    proba = pipeline.predict_proba(test["texto"])
    classes = list(pipeline.classes_)
    predicted = [classes[i] for i in proba.argmax(axis=1)]
    return pd.DataFrame({
        "id": test["id"].to_numpy(),
        "texto": test["texto"].to_numpy(),
        "real": test["sentimiento"].to_numpy(),
        "predicho": predicted,
        "probabilidad": proba.max(axis=1).round(4),
        "fenomeno": test["fenomeno"].to_numpy(),
    }).assign(correcto=lambda d: d["real"] == d["predicho"])


def accuracy_by_phenomenon(table: pd.DataFrame) -> list[dict]:
    grouped = table.groupby("fenomeno")["correcto"].agg(["count", "mean"])
    return [{"fenomeno": name, "n": int(row["count"]), "accuracy": round(float(row["mean"]), 4)}
            for name, row in grouped.sort_values("mean").iterrows()]


def error_examples(table: pd.DataFrame, n: int = 12) -> list[dict]:
    """The most confident mistakes, at most three per phenomenon."""
    errors = (table[~table["correcto"]]
              .sort_values("probabilidad", ascending=False)
              .groupby("fenomeno").head(3)
              .head(n))
    return errors.drop(columns="correcto").to_dict(orient="records")


def top_terms(pipeline: Pipeline, n: int = 12) -> dict:
    """Terms that push each class up the most.

    Logistic regression: largest coefficients. Naive Bayes: largest log
    probability relative to the average of the classes.
    """
    vocabulary = pipeline.named_steps["tfidf"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        weights = model.coef_
    elif hasattr(model, "feature_log_prob_"):
        weights = model.feature_log_prob_ - model.feature_log_prob_.mean(axis=0)
    else:
        return {}
    return {
        label: [[vocabulary[i], round(float(weights[c, i]), 3)]
                for i in np.argsort(weights[c])[::-1][:n]]
        for c, label in enumerate(model.classes_)
    }
