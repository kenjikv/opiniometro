"""Text preprocessing for Spanish reviews.

The same ``TextPreprocessor`` object is used to train the model and, inside
the saved pipeline, to answer every API request. The API never reimplements
any of these steps.

Steps, in order:
    1. normalize   lowercase, URLs, mentions, emojis, numbers, slang,
                   elongations and punctuation
    2. tokenize    split the normalized text into words
    3. stop words  drop words without sentiment
    4. stem/lemma  reduce each word to its root (Snowball) or lemma (spaCy)
    5. accents     strip accents at the very end, so "atención" == "atencion"
"""

import re
import unicodedata
from functools import lru_cache

import emoji
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NUMBER_RE = re.compile(r"\d+([.,/:]\d+)*%?")
ELONGATION_RE = re.compile(r"(\w)\1{2,}")  # Spanish never repeats a letter 3 times
NON_LETTER_RE = re.compile(r"[^a-záéíóúüñ_\s]")
SPACES_RE = re.compile(r"\s+")

# Informal writing found in the reviews. Expanded so that "q" and "que" are
# the same token (and are then removed as a stop word).
SLANG = {
    "q": "que", "k": "que", "x": "por", "pq": "porque", "xq": "porque",
    "tb": "también", "tmb": "también", "mui": "muy", "bn": "bien",
}

# Emojis carry sentiment, so they become words instead of being deleted.
# Any emoji not listed here is removed.
EMOJI_WORDS = {
    "emojipositivo": "😍👍😋🔥❤👌😀😃😄😊🥰🤩💯👏🙌😁",
    "emojinegativo": "😡👎🤢😤💩😠😞😢🤮😭😫",
    "emojiironia": "🙃🙄😒",
}
EMOJI_TO_WORD = {e: word for word, chars in EMOJI_WORDS.items() for e in chars}

STOPWORD_LISTS = ("spacy_full", "adjusted")
NORMALIZERS = ("stem", "lemma")


def strip_accents(text: str) -> str:
    """Remove accents but keep ñ: "año" and "ano" are different words."""
    text = text.replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")
    return text.replace("\0", "ñ")


def _emoji_to_word(chars: str, _data: dict) -> str:
    return f" {EMOJI_TO_WORD.get(chars.replace(chr(0xFE0F), ''), '')} "


def normalize(text: str) -> str:
    """Step 1. Accents are kept here because the lemmatizer needs them."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = emoji.replace_emoji(text, replace=_emoji_to_word)
    text = NUMBER_RE.sub(" ", text)
    text = ELONGATION_RE.sub(r"\1", text)
    text = NON_LETTER_RE.sub(" ", text)
    words = [SLANG.get(w, w) for w in text.split()]
    return SPACES_RE.sub(" ", " ".join(words)).strip()


def tokenize(normalized: str) -> list[str]:
    """Step 2. After normalization only letters remain, so splitting on
    spaces is enough. Single letters ("a", "y", "o") are dropped."""
    return [t for t in normalized.split(" ") if len(t) > 1]


@lru_cache
def stopwords(list_name: str, keep_words: tuple = (), extra: tuple = ()) -> frozenset:
    """Step 3. spaCy's Spanish list, optionally adjusted for sentiment.

    Words are stored without accents and compared without accents, because
    people often omit them ("mas", "tambien").
    """
    from spacy.lang.es.stop_words import STOP_WORDS

    if list_name not in STOPWORD_LISTS:
        raise ValueError(f"Unknown stop word list: {list_name}")
    words = {strip_accents(w) for w in STOP_WORDS}
    if list_name == "adjusted":
        words -= {strip_accents(w) for w in keep_words}
        words |= {strip_accents(w) for w in extra}
    return frozenset(words)


@lru_cache
def _stemmer():
    from nltk.stem import SnowballStemmer
    return SnowballStemmer("spanish")


@lru_cache
def _spacy():
    import spacy
    # The parser and the entity recognizer are not needed to lemmatize.
    return spacy.load("es_core_news_sm", disable=["parser", "ner"])


def _is_special(token: str) -> bool:
    return token.startswith("emoji")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: raw texts in, preprocessed strings out.

    It learns nothing from the data (``fit`` does nothing), so applying it
    before cross-validation does not leak information between folds.
    """

    def __init__(self, normalizer: str = "lemma", stopword_list: str = "adjusted",
                 keep_words: tuple = (), extra_stopwords: tuple = ()):
        self.normalizer = normalizer
        self.stopword_list = stopword_list
        self.keep_words = keep_words
        self.extra_stopwords = extra_stopwords

    def fit(self, X, y=None):
        if self.normalizer not in NORMALIZERS:
            raise ValueError(f"normalizer must be one of {NORMALIZERS}")
        if self.stopword_list not in STOPWORD_LISTS:
            raise ValueError(f"stopword_list must be one of {STOPWORD_LISTS}")
        return self

    def transform(self, X) -> list[str]:
        return [" ".join(steps["final"]) for steps in self._analyze(list(X))]

    def explain(self, text: str) -> dict:
        """Every intermediate step for one text, for the API and the frontend."""
        return self._analyze([text])[0]

    def _stopwords(self) -> frozenset:
        return stopwords(self.stopword_list, tuple(self.keep_words),
                         tuple(self.extra_stopwords))

    def _roots(self, token_lists: list[list[str]]) -> list[list[str]]:
        """Step 4 on the full token list, before removing stop words, so that
        spaCy sees each word in its sentence context."""
        if self.normalizer == "stem":
            stem = _stemmer().stem
            return [[t if _is_special(t) else stem(t) for t in tokens]
                    for tokens in token_lists]

        from spacy.tokens import Doc
        nlp = _spacy()
        docs = (Doc(nlp.vocab, words=tokens) for tokens in token_lists)
        return [[t.text if _is_special(t.text) else t.lemma_.lower() for t in doc]
                for doc in nlp.pipe(docs, batch_size=256)]

    def _analyze(self, texts: list[str]) -> list[dict]:
        normalized = [normalize(t or "") for t in texts]
        token_lists = [tokenize(n) for n in normalized]
        root_lists = self._roots(token_lists)
        stop = self._stopwords()

        results = []
        for text, norm, tokens, roots in zip(texts, normalized, token_lists, root_lists):
            kept = [(t, r) for t, r in zip(tokens, roots) if strip_accents(t) not in stop]
            results.append({
                "original": text,
                "normalized": norm,
                "tokens": tokens,
                "removed_stopwords": [t for t in tokens if strip_accents(t) in stop],
                "without_stopwords": [t for t, _ in kept],
                "roots": [r for _, r in kept],
                "final": [strip_accents(r) for _, r in kept],
            })
        return results
