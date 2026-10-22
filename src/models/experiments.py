"""Model definitions, search space and the production pipeline."""

from itertools import product

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.nlp.preprocessing import TextPreprocessor

MODEL_LABELS = {
    "baseline": "Línea base (clase más frecuente)",
    "naive_bayes": "Naive Bayes multinomial",
    "logistic_regression": "Regresión logística",
}


def make_model(name: str, seed: int):
    if name == "naive_bayes":
        return MultinomialNB()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "baseline":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model: {name}")


def make_vectorizer(ngram_range, tfidf: dict) -> TfidfVectorizer:
    # The preprocessor already lowercased and tokenized: split on spaces only.
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False,
                           ngram_range=tuple(ngram_range), **tfidf)


def make_preprocessor(normalizer: str, stopword_list: str, nlp: dict) -> TextPreprocessor:
    return TextPreprocessor(normalizer=normalizer, stopword_list=stopword_list,
                            keep_words=tuple(nlp["keep_words"]),
                            extra_stopwords=tuple(nlp["extra_stopwords"]))


def build_pipeline(config: dict, params: dict) -> Pipeline:
    """The object that is trained, saved and served: raw text in, label out."""
    model = make_model(config["model"], params["seed"])
    model.set_params(**config.get("hyperparameters", {}))
    return Pipeline([
        ("preprocess", make_preprocessor(config["normalizer"], config["stopword_list"],
                                         params["nlp"])),
        ("tfidf", make_vectorizer(config["ngram_range"], params["tfidf"])),
        ("model", model),
    ])


def run_experiments(texts: pd.Series, y: pd.Series, params: dict) -> pd.DataFrame:
    """Every preprocessing variant x n-gram range x model, each with its own
    hyperparameter grid, scored with stratified cross-validation on train only.

    The preprocessor is stateless, so each variant is computed once and reused
    by all folds. The TF-IDF vocabulary is still learned inside each fold.
    """
    exp = params["experiments"]
    cv = StratifiedKFold(n_splits=exp["cv_folds"], shuffle=True, random_state=params["seed"])
    rows = []

    for normalizer, stopword_list in product(exp["normalizers"], exp["stopword_lists"]):
        prepared = make_preprocessor(normalizer, stopword_list, params["nlp"]).transform(texts)
        print(f"  preprocesamiento {normalizer} + {stopword_list}")

        for ngram_range, (name, grid) in product(exp["ngram_ranges"], exp["models"].items()):
            pipeline = Pipeline([
                ("tfidf", make_vectorizer(ngram_range, params["tfidf"])),
                ("model", make_model(name, params["seed"])),
            ])
            search = GridSearchCV(
                pipeline, {f"model__{k}": v for k, v in grid.items()},
                scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
                refit=exp["scoring"], cv=cv, n_jobs=-1,
            )
            search.fit(prepared, y)
            best = search.best_index_
            results = search.cv_results_
            rows.append({
                "model": name,
                "normalizer": normalizer,
                "stopword_list": stopword_list,
                "ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
                "hyperparameters": {k.removeprefix("model__"): v
                                    for k, v in search.best_params_.items()},
                "cv_f1_macro": round(float(results["mean_test_f1_macro"][best]), 4),
                "cv_f1_macro_std": round(float(results["std_test_f1_macro"][best]), 4),
                "cv_accuracy": round(float(results["mean_test_accuracy"][best]), 4),
            })

    baseline = cross_validate(make_model("baseline", params["seed"]), texts, y, cv=cv,
                              scoring=["f1_macro", "accuracy"])
    rows.append({
        "model": "baseline", "normalizer": "-", "stopword_list": "-", "ngram_range": "-",
        "hyperparameters": {"strategy": "most_frequent"},
        "cv_f1_macro": round(float(baseline["test_f1_macro"].mean()), 4),
        "cv_f1_macro_std": round(float(baseline["test_f1_macro"].std()), 4),
        "cv_accuracy": round(float(baseline["test_accuracy"].mean()), 4),
    })
    return (pd.DataFrame(rows)
            .sort_values("cv_f1_macro", ascending=False)
            .reset_index(drop=True))


def row_to_config(row: pd.Series) -> dict:
    return {
        "model": row["model"],
        "normalizer": row["normalizer"],
        "stopword_list": row["stopword_list"],
        "ngram_range": [int(n) for n in row["ngram_range"].split("-")],
        "hyperparameters": row["hyperparameters"],
    }
