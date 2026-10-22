import pandas as pd

from src.config import ARTIFACTS_DIR, LABELS
from src.data.generate import generate
from src.data.prepare import clean, split


def test_generator_is_reproducible():
    assert generate(100, seed=1).equals(generate(100, seed=1))


def test_clean_removes_empty_texts_and_duplicates():
    raw = pd.DataFrame({"texto": ["rico", None, "  ", "rico", "malo"],
                        "sentimiento": ["positivo"] * 3 + ["positivo", "negativo"]})
    df, report = clean(raw)
    assert df["texto"].tolist() == ["rico", "malo"]
    assert report["empty_texts_removed"] == 2 and report["duplicates_removed"] == 1


def test_split_is_stratified():
    df, _ = clean(generate(600, seed=3))
    train, test = split(df, test_size=0.2, seed=42)
    train_share = train["sentimiento"].value_counts(normalize=True)
    test_share = test["sentimiento"].value_counts(normalize=True)
    assert (train_share - test_share).abs().max() < 0.02


def test_training_writes_all_artifacts(trained):
    for name in ["model.joblib", "metrics.json", "experiments.csv", "errors.csv"]:
        assert (ARTIFACTS_DIR / name).exists()


def test_models_beat_the_baseline(trained):
    scores = {m["name"]: m["test"]["f1_macro"] for m in trained["models"]}
    assert scores["naive_bayes"] > scores["baseline"]
    assert scores["logistic_regression"] > scores["baseline"]
    assert trained["production"]["name"] in {"naive_bayes", "logistic_regression"}


def test_confusion_matrix_covers_the_test_set(trained):
    production = next(m for m in trained["models"]
                      if m["name"] == trained["production"]["name"])
    matrix = production["test"]["confusion_matrix"]
    assert len(matrix) == len(LABELS)
    assert sum(map(sum, matrix)) == trained["dataset"]["test_size"]
