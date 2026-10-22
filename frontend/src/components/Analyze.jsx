import { useState } from "react";
import { api } from "../api.js";
import { SENTIMENTS, pct } from "../labels.js";
import { Card, ErrorBox, SentimentBars, Tokens } from "./ui.jsx";

const MAX_CHARS = 1000;

const EXAMPLES = [
  { label: "Positiva", text: "El majadito estaba riquísimo y la atención fue muy amable 😋 Volvería sin dudar" },
  { label: "Negativa", text: "Esperamos 50 minutos y el pique macho llegó frío. Pésima experiencia!!!" },
  { label: "Negación", text: "Las salteñas no estaban nada mal y no tuvimos que esperar" },
  { label: "Mixta", text: "La sopa de maní estaba rica, pero es caro para lo que sirven" },
  { label: "Ironía", text: "Excelente servicio, solo esperamos una hora para un café 🙃" },
];

function Steps({ steps }) {
  const method = steps.metodo === "lemma" ? "Lematización (spaCy)" : "Stemming (Snowball)";
  const pairs = steps.sin_stopwords.map((token, i) => `${token} → ${steps.raices[i]}`);
  return (
    <ol className="steps">
      <li>
        <h3>Texto original</h3>
        <p className="mono">{steps.original}</p>
      </li>
      <li>
        <h3>1 · Normalización</h3>
        <p className="muted">Minúsculas, sin URLs, menciones, números ni puntuación; emojis convertidos en palabras; jerga expandida.</p>
        <p className="mono">{steps.normalizado || "(vacío)"}</p>
      </li>
      <li>
        <h3>2 · Tokenización</h3>
        <Tokens items={steps.tokens} />
      </li>
      <li>
        <h3>3 · Stop words</h3>
        <p className="muted">Eliminadas:</p>
        <Tokens items={steps.stopwords_eliminadas} variant="removed" />
        <p className="muted">Se conservan:</p>
        <Tokens items={steps.sin_stopwords} />
      </li>
      <li>
        <h3>4 · {method}</h3>
        <Tokens items={pairs} variant="pair" />
      </li>
      <li>
        <h3>5 · Entrada a TF-IDF</h3>
        <p className="muted">Sin tildes, para que «atención» y «atencion» sean el mismo término.</p>
        <p className="mono">{steps.final.join(" ") || "(vacío)"}</p>
      </li>
    </ol>
  );
}

export default function Analyze() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function analyze(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      setResult(await api.predict(text));
    } catch (err) {
      setResult(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function pickExample(example) {
    setText(example);
    setResult(null);
    setError("");
  }

  const meta = result && SENTIMENTS[result.sentimiento];

  return (
    <div className="grid two">
      <Card title="Escribe una reseña" subtitle="El modelo la clasifica como negativa, neutra o positiva.">
        <form onSubmit={analyze}>
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setResult(null);
            }}
            rows={6}
            placeholder="Ej.: El silpancho estaba frío y tardaron muchísimo…"
            aria-label="Texto de la reseña"
          />
          <div className="row between">
            <span className={text.length > MAX_CHARS ? "counter over" : "counter"}>
              {text.length} / {MAX_CHARS}
            </span>
            <button className="primary" disabled={loading}>
              {loading ? "Analizando…" : "Analizar"}
            </button>
          </div>
        </form>
        <div className="examples">
          <span className="muted">Probar con:</span>
          {EXAMPLES.map((ex) => (
            <button key={ex.label} className="ghost" onClick={() => pickExample(ex.text)}>
              {ex.label}
            </button>
          ))}
        </div>
        <ErrorBox message={error} />

        {result && (
          <div className={`result result-${meta.className}`}>
            <div className="result-head">
              <span className="result-label">{meta.name}</span>
              <span className="result-prob">{pct(result.probabilidad)}</span>
            </div>
            <SentimentBars probabilities={result.probabilidades} />
            <p className="muted small">
              {result.modelo.nombre} · versión {result.modelo.version} · {result.tiempo_ms} ms
            </p>
          </div>
        )}
      </Card>

      <Card title="Cómo se procesó el texto" subtitle="Los mismos pasos que se usaron al entrenar, ejecutados por el pipeline guardado.">
        {result ? <Steps steps={result.preprocesamiento} /> : <p className="muted">Analiza una reseña para ver cada paso.</p>}
      </Card>
    </div>
  );
}
