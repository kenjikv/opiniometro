// Display names for the values the API returns.

export const SENTIMENTS = {
  negativo: { name: "Negativo", className: "neg" },
  neutro: { name: "Neutro", className: "neu" },
  positivo: { name: "Positivo", className: "pos" },
};

export const FACTORS = {
  stopword_list: {
    title: "Lista de stop words",
    values: { spacy_full: "Lista completa de spaCy", adjusted: "Ajustada (conserva negaciones)" },
  },
  ngram_range: {
    title: "N-gramas",
    values: { "1-1": "Unigramas", "1-2": "Unigramas + bigramas" },
  },
  normalizer: {
    title: "Stemming vs. lematización",
    values: { stem: "Stemming (Snowball)", lemma: "Lematización (spaCy)" },
  },
  model: {
    title: "Modelo",
    values: { naive_bayes: "Naive Bayes", logistic_regression: "Regresión logística" },
  },
};

export const PHENOMENA = {
  simple: "Simple",
  negacion: "Negación",
  mixta: "Opinión mixta",
  ironia: "Ironía",
  etiqueta_ruidosa: "Etiqueta ruidosa",
};

export const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)} %`;
