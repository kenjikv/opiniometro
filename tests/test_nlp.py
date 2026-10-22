import joblib
import pytest

from src.nlp.preprocessing import TextPreprocessor, normalize, stopwords, tokenize

KEEP = ("no", "bueno", "nada", "mal")


def test_normalize_removes_noise_and_keeps_accents():
    text = "Q RICOOO el majadito!!! Bs 45 https://fotos.bo/x @pepe #ComidaBoliviana"
    assert normalize(text) == "que rico el majadito bs comidaboliviana"
    assert normalize("La atención fue rápida") == "la atención fue rápida"


def test_emojis_become_words():
    assert normalize("buenísimo 😍👎🙃") == "buenísimo emojipositivo emojinegativo emojiironia"
    assert normalize("raro 🦄") == "raro"


def test_tokenize_drops_single_letters():
    assert tokenize("y el pique macho a la mesa") == ["el", "pique", "macho", "la", "mesa"]


def test_adjusted_list_keeps_sentiment_words():
    full = stopwords("spacy_full")
    adjusted = stopwords("adjusted", KEEP, ("restaurante",))
    for word in KEEP:
        assert word in full and word not in adjusted
    assert "restaurante" in adjusted and "restaurante" not in full


def test_negation_survives_with_adjusted_list():
    adjusted = TextPreprocessor("stem", "adjusted", KEEP).explain("No estaba bueno")
    full = TextPreprocessor("stem", "spacy_full").explain("No estaba bueno")
    assert "no" in adjusted["final"] and "buen" in adjusted["final"]
    assert full["final"] == []  # spaCy's full list deletes the whole review


def test_final_tokens_have_no_accents():
    steps = TextPreprocessor("lemma", "adjusted", KEEP).explain("La atención fue pésima")
    assert all(not set(t) & set("áéíóú") for t in steps["final"])


@pytest.mark.parametrize("normalizer", ["stem", "lemma"])
def test_transform_matches_explain(normalizer):
    pre = TextPreprocessor(normalizer, "adjusted", KEEP)
    text = "Las salteñas estaban frías y no nos atendieron rápido"
    assert pre.transform([text]) == [" ".join(pre.explain(text)["final"])]


def test_stem_and_lemma_differ():
    text = "los meseros fueron atentos"
    stem = TextPreprocessor("stem", "adjusted", KEEP).transform([text])[0]
    lemma = TextPreprocessor("lemma", "adjusted", KEEP).transform([text])[0]
    assert stem != lemma


def test_survives_saving_and_loading(tmp_path):
    pre = TextPreprocessor("lemma", "adjusted", KEEP)
    joblib.dump(pre, tmp_path / "pre.joblib")
    loaded = joblib.load(tmp_path / "pre.joblib")
    text = "El majadito no estaba nada mal 😋"
    assert loaded.transform([text]) == pre.transform([text])


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        TextPreprocessor(normalizer="porter").fit(["texto"])
