import { useEffect, useState } from "react";
import { api } from "../api.js";
import { FACTORS, PHENOMENA, SENTIMENTS, pct } from "../labels.js";
import { Bars, Card, ErrorBox, Loading, SentimentChip } from "./ui.jsx";

const f3 = (value) => value.toFixed(3);

function describeConfig(config) {
  if (config.model === "baseline") return "Predice siempre «positivo»";
  const params = Object.entries(config.hyperparameters)
    .map(([k, v]) => `${k}=${v ?? "None"}`)
    .join(", ");
  return [
    FACTORS.normalizer.values[config.normalizer],
    FACTORS.stopword_list.values[config.stopword_list],
    FACTORS.ngram_range.values[config.ngram_range.join("-")],
    params,
  ].join(" · ");
}

function ConfusionMatrix({ matrix, labels }) {
  return (
    <div className="table-wrap">
      <table className="confusion">
        <thead>
          <tr>
            <th>Real ↓ / Predicho →</th>
            {labels.map((l) => <th key={l}>{SENTIMENTS[l].name}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={labels[i]}>
                <th>{SENTIMENTS[labels[i]].name}</th>
                {row.map((value, j) => (
                  <td
                    key={j}
                    className={i === j ? "hit" : "miss"}
                    style={{ "--alpha": (value / total).toFixed(2) }}
                    title={`${pct(value / total)} de los ${SENTIMENTS[labels[i]].name.toLowerCase()}s`}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FactorEffects({ effects }) {
  return (
    <div className="factors">
      {Object.entries(FACTORS).map(([factor, meta]) => {
        const values = effects[factor] || {};
        const best = Math.max(...Object.values(values));
        return (
          <div key={factor} className="factor">
            <h3>{meta.title}</h3>
            {Object.entries(values).map(([value, score]) => (
              <div key={value} className="factor-row">
                <span>{meta.values[value] || value}</span>
                <span className="mono">{f3(score)}</span>
                <span className={score === best ? "delta best" : "delta"}>
                  {score === best ? "mejor" : `−${((best - score) * 100).toFixed(2)} pts`}
                </span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export default function Metrics() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch((err) => setError(err.message));
  }, []);

  if (error) return <ErrorBox message={error} />;
  if (!metrics) return <Loading />;

  const productionName = metrics.production.name;
  const shown = metrics.models.find((m) => m.name === (selected || productionName));
  const production = metrics.models.find((m) => m.name === productionName);

  return (
    <div className="stack">
      <Card title={`En producción: ${production.label}`} subtitle={describeConfig(production.config)}>
        <div className="stats">
          <div className="stat"><span>{f3(production.cv_f1_macro)} ± {f3(production.cv_f1_macro_std)}</span>F1 macro (validación cruzada)</div>
          <div className="stat"><span>{f3(production.test.f1_macro)}</span>F1 macro (prueba)</div>
          <div className="stat"><span>{pct(production.test.accuracy)}</span>Exactitud (prueba)</div>
          <div className="stat"><span>{metrics.vocabulary_size}</span>Términos en el vocabulario TF-IDF</div>
        </div>
        <p className="muted small">
          Elegido por F1 macro en validación cruzada sobre entrenamiento. El conjunto de prueba solo se usó para
          reportar estas cifras. Versión {metrics.version}.
        </p>
      </Card>

      <Card title="Comparación de modelos" subtitle="Cada modelo con su mejor configuración. La línea base demuestra que el modelo aporta valor.">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Configuración</th>
                <th className="num">F1 macro CV</th>
                <th className="num">Exactitud prueba</th>
                <th className="num">F1 macro prueba</th>
                {metrics.labels.map((l) => <th key={l} className="num">F1 {SENTIMENTS[l].name.toLowerCase()}</th>)}
              </tr>
            </thead>
            <tbody>
              {metrics.models.map((m) => (
                <tr key={m.name} className={m.name === productionName ? "highlight" : ""}>
                  <td>{m.label}</td>
                  <td className="small">{describeConfig(m.config)}</td>
                  <td className="num">{f3(m.cv_f1_macro)}</td>
                  <td className="num">{pct(m.test.accuracy)}</td>
                  <td className="num">{f3(m.test.f1_macro)}</td>
                  {metrics.labels.map((l) => <td key={l} className="num">{f3(m.test.per_class[l].f1)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Matriz de confusión y métricas por clase" subtitle="Conjunto de prueba.">
        <div className="row">
          {metrics.models.map((m) => (
            <button key={m.name} className={m.name === shown.name ? "ghost active" : "ghost"} onClick={() => setSelected(m.name)}>
              {m.label}
            </button>
          ))}
        </div>
        <div className="grid two">
          <ConfusionMatrix matrix={shown.test.confusion_matrix} labels={metrics.labels} />
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Clase</th><th className="num">Precisión</th><th className="num">Recall</th><th className="num">F1</th><th className="num">Soporte</th></tr>
              </thead>
              <tbody>
                {metrics.labels.map((l) => {
                  const c = shown.test.per_class[l];
                  return (
                    <tr key={l}>
                      <td><SentimentChip value={l} /></td>
                      <td className="num">{f3(c.precision)}</td>
                      <td className="num">{f3(c.recall)}</td>
                      <td className="num">{f3(c.f1)}</td>
                      <td className="num">{c.support}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </Card>

      <Card title="Efecto de cada decisión" subtitle="Mejor F1 macro en validación cruzada alcanzado con cada opción.">
        <FactorEffects effects={metrics.factor_effects} />
      </Card>

      <Card title="Todos los experimentos" subtitle={`Validación cruzada estratificada de ${metrics.experiments.length - 1} configuraciones, cada una con su búsqueda de hiperparámetros.`}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th><th>Modelo</th><th>Normalización</th><th>Stop words</th><th>N-gramas</th>
                <th>Hiperparámetros</th><th className="num">F1 macro CV</th><th className="num">Exactitud CV</th>
              </tr>
            </thead>
            <tbody>
              {metrics.experiments.map((e, i) => (
                <tr key={i} className={i === 0 ? "highlight" : ""}>
                  <td>{i + 1}</td>
                  <td>{FACTORS.model.values[e.model] || "Línea base"}</td>
                  <td>{FACTORS.normalizer.values[e.normalizer] || "—"}</td>
                  <td>{FACTORS.stopword_list.values[e.stopword_list] || "—"}</td>
                  <td>{FACTORS.ngram_range.values[e.ngram_range] || "—"}</td>
                  <td className="small mono">{JSON.stringify(e.hyperparameters)}</td>
                  <td className="num">{f3(e.cv_f1_macro)} ± {f3(e.cv_f1_macro_std)}</td>
                  <td className="num">{pct(e.cv_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Términos más influyentes" subtitle="Los que más empujan la predicción hacia cada clase en el modelo en producción.">
        <div className="grid three">
          {metrics.labels.map((l) => (
            <div key={l}>
              <SentimentChip value={l} />
              <ol className="terms">
                {(metrics.top_terms[l] || []).map(([term, weight]) => (
                  <li key={term}><span className="mono">{term}</span><span className="muted">{weight}</span></li>
                ))}
              </ol>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Análisis de errores" subtitle="Exactitud del modelo en producción según el fenómeno de cada reseña de prueba.">
        <Bars
          rows={metrics.accuracy_by_phenomenon.map((p) => ({
            label: `${PHENOMENA[p.fenomeno] || p.fenomeno} (${p.n})`,
            value: p.accuracy,
            className: p.accuracy < 0.8 ? "neg" : "pos",
          }))}
        />
        <h3>Errores con mayor confianza</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Texto</th><th>Real</th><th>Predicho</th><th className="num">Confianza</th><th>Fenómeno</th></tr>
            </thead>
            <tbody>
              {metrics.error_examples.map((e) => (
                <tr key={e.id}>
                  <td>{e.texto}</td>
                  <td><SentimentChip value={e.real} /></td>
                  <td><SentimentChip value={e.predicho} /></td>
                  <td className="num">{pct(e.probabilidad)}</td>
                  <td>{PHENOMENA[e.fenomeno] || e.fenomeno}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
