"""Train, compare and save the models.

    python -m src.models.train

Reads the reviews from PostgreSQL when DATABASE_URL is set, otherwise from
the CSV. Writes everything to artifacts/:
    model.joblib     production pipeline (preprocessing + TF-IDF + model)
    metrics.json     what the API and the frontend show
    experiments.csv  cross-validation result of every configuration
    errors.csv       test reviews the production model gets wrong
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src import db
from src.config import ARTIFACTS_DIR, LABELS, ROOT, load_params
from src.data.prepare import clean, describe, split
from src.models.evaluate import (accuracy_by_phenomenon, classification_metrics,
                                 error_examples, error_table, top_terms)
from src.models.experiments import (MODEL_LABELS, build_pipeline, make_model,
                                    row_to_config, run_experiments)


def load_raw(params: dict) -> tuple[pd.DataFrame, str]:
    engine = db.get_engine()
    if engine is not None:
        return db.read_reviews(engine), "postgresql"
    return pd.read_csv(ROOT / params["data"]["raw_path"]), "csv"


def factor_effects(experiments: pd.DataFrame) -> dict:
    """Best CV F1 reached with each value of each decision, e.g. stem vs lemma."""
    models = experiments[experiments["model"] != "baseline"]
    return {
        factor: models.groupby(factor)["cv_f1_macro"].max().round(4).to_dict()
        for factor in ["normalizer", "stopword_list", "ngram_range", "model"]
    }


def train(raw: pd.DataFrame, params: dict, artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    seed = params["seed"]
    df, cleaning = clean(raw)
    eda = describe(df, tuple(params["nlp"]["keep_words"]), tuple(params["nlp"]["extra_stopwords"]))
    train_df, test_df = split(df, params["data"]["test_size"], seed)
    print(f"Entrenamiento: {len(train_df)} | prueba: {len(test_df)}")

    print("Validación cruzada de todas las configuraciones...")
    experiments = run_experiments(train_df["texto"], train_df["sentimiento"], params)

    # Each model keeps its best configuration; the overall best goes to production.
    # The test set is used only now, once, to report the final numbers.
    models, fitted = [], {}
    for name in [*params["experiments"]["models"], "baseline"]:
        row = experiments[experiments["model"] == name].iloc[0]
        if name == "baseline":
            pipeline = make_model("baseline", seed)
            config = {"model": "baseline", "hyperparameters": {"strategy": "most_frequent"}}
        else:
            config = row_to_config(row)
            pipeline = build_pipeline(config, params)
        pipeline.fit(train_df["texto"], train_df["sentimiento"])
        fitted[name] = pipeline
        models.append({
            "name": name,
            "label": MODEL_LABELS[name],
            "config": config,
            "cv_f1_macro": float(row["cv_f1_macro"]),
            "cv_f1_macro_std": float(row["cv_f1_macro_std"]),
            "test": classification_metrics(test_df["sentimiento"],
                                           pipeline.predict(test_df["texto"])),
        })

    production = next(m for m in sorted(models, key=lambda m: -m["cv_f1_macro"])
                      if m["name"] != "baseline")
    pipeline = fitted[production["name"]]
    table = error_table(pipeline, test_df)

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    metrics = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "labels": LABELS,
        "data_source": raw.attrs.get("source", "csv"),
        "dataset": {
            "cleaning": cleaning,
            "eda": eda,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_counts": train_df["sentimiento"].value_counts().reindex(LABELS).to_dict(),
            "test_counts": test_df["sentimiento"].value_counts().reindex(LABELS).to_dict(),
        },
        "production": {k: production[k] for k in ["name", "label", "config", "cv_f1_macro"]},
        "models": models,
        "experiments": experiments.to_dict(orient="records"),
        "factor_effects": factor_effects(experiments),
        "tfidf": params["tfidf"],
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "accuracy_by_phenomenon": accuracy_by_phenomenon(table),
        "error_examples": error_examples(table),
        "top_terms": top_terms(pipeline),
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "version": version, "model": production["name"],
                 "label": production["label"], "config": production["config"]},
                artifacts_dir / "model.joblib")
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    experiments.assign(hyperparameters=experiments["hyperparameters"].map(json.dumps)) \
        .to_csv(artifacts_dir / "experiments.csv", index=False)
    table[~table["correcto"]].drop(columns="correcto") \
        .to_csv(artifacts_dir / "errors.csv", index=False)

    _print_summary(experiments, models, production)
    return metrics


def _print_summary(experiments: pd.DataFrame, models: list, production: dict) -> None:
    print("\nMejores configuraciones (validación cruzada, F1 macro):")
    print(experiments.head(8)[["model", "normalizer", "stopword_list", "ngram_range",
                               "cv_f1_macro", "cv_f1_macro_std"]].to_string(index=False))
    print("\nConjunto de prueba:")
    for m in models:
        print(f"  {m['label']:<36} accuracy={m['test']['accuracy']:.3f}  "
              f"f1_macro={m['test']['f1_macro']:.3f}")
    print(f"\nEn producción: {production['label']} {production['config']}")


def main() -> None:
    params = load_params()
    raw, source = load_raw(params)
    raw.attrs["source"] = source
    print(f"{len(raw)} reseñas leídas desde {source}")
    metrics = train(raw, params)

    engine = db.get_engine()
    if engine is not None:
        prod = next(m for m in metrics["models"] if m["name"] == metrics["production"]["name"])
        db.record_training(engine, {
            "version": metrics["version"],
            "modelo": prod["name"],
            "configuracion": prod["config"],
            "cv_f1_macro": prod["cv_f1_macro"],
            "test_f1_macro": prod["test"]["f1_macro"],
            "test_accuracy": prod["test"]["accuracy"],
            "n_entrenamiento": metrics["dataset"]["train_size"],
            "n_prueba": metrics["dataset"]["test_size"],
        })
        print("Entrenamiento registrado en la tabla entrenamientos")


if __name__ == "__main__":
    main()
