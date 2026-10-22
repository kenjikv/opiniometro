import { useEffect, useState } from "react";
import { api } from "../api.js";
import { SENTIMENTS, pct } from "../labels.js";
import { Bars, Card, ErrorBox, Loading, SentimentChip } from "./ui.jsx";

export default function Data() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.metrics().then(setMetrics).catch((err) => setError(err.message));
  }, []);

  if (error) return <ErrorBox message={error} />;
  if (!metrics) return <Loading />;

  const { cleaning, eda, train_counts, test_counts } = metrics.dataset;
  const labels = metrics.labels;
  const total = cleaning.clean_rows;
  const maxBin = Math.max(...eda.length_histogram.map((b) => b.count));
  const stars = [1, 2, 3, 4, 5];

  return (
    <div className="stack">
      <Card title="Limpieza" subtitle={`Datos leídos desde ${metrics.data_source === "postgresql" ? "PostgreSQL (tabla resenas)" : "el CSV"}.`}>
        <div className="stats">
          <div className="stat"><span>{cleaning.raw_rows}</span>Filas crudas</div>
          <div className="stat"><span>{cleaning.empty_texts_removed}</span>Textos vacíos eliminados</div>
          <div className="stat"><span>{cleaning.duplicates_removed}</span>Duplicados eliminados</div>
          <div className="stat"><span>{cleaning.clean_rows}</span>Filas para modelar</div>
        </div>
        <p className="muted small">
          Valores faltantes en los datos crudos:{" "}
          {Object.entries(cleaning.missing_values).map(([col, n]) => `${col} (${n})`).join(", ") || "ninguno"}.
          La ciudad faltante no se usa en el modelo, así que esas filas se conservan.
        </p>
      </Card>

      <div className="grid two">
        <Card title="Distribución de clases" subtitle="Desbalanceada: por eso la métrica principal es F1 macro.">
          <Bars
            max={total}
            format={(v) => `${v} (${pct(v / total, 0)})`}
            rows={labels.map((l) => ({ label: SENTIMENTS[l].name, value: eda.class_counts[l], className: SENTIMENTS[l].className }))}
          />
          <table>
            <thead><tr><th>Partición</th>{labels.map((l) => <th key={l} className="num">{SENTIMENTS[l].name}</th>)}</tr></thead>
            <tbody>
              <tr><td>Entrenamiento ({metrics.dataset.train_size})</td>{labels.map((l) => <td key={l} className="num">{train_counts[l]}</td>)}</tr>
              <tr><td>Prueba ({metrics.dataset.test_size})</td>{labels.map((l) => <td key={l} className="num">{test_counts[l]}</td>)}</tr>
            </tbody>
          </table>
          <p className="muted small">División estratificada 80/20 con semilla {metrics.seed}.</p>
        </Card>

        <Card title="Longitud de los textos" subtitle="Palabras por reseña.">
          <table>
            <thead><tr><th>Clase</th><th className="num">Media</th><th className="num">Mediana</th><th className="num">Máximo</th></tr></thead>
            <tbody>
              {labels.map((l) => (
                <tr key={l}>
                  <td><SentimentChip value={l} /></td>
                  <td className="num">{eda.length_words[l].mean}</td>
                  <td className="num">{eda.length_words[l].median}</td>
                  <td className="num">{eda.length_words[l].max}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Bars
            max={maxBin}
            format={(v) => String(v)}
            rows={eda.length_histogram.map((b) => ({ label: `${b.from}–${b.to}`, value: b.count }))}
          />
        </Card>
      </div>

      <Card title="Palabras más frecuentes por clase" subtitle="Después de normalizar y quitar stop words (lista ajustada).">
        <div className="grid three">
          {labels.map((l) => (
            <div key={l}>
              <SentimentChip value={l} />
              <ol className="terms">
                {eda.top_words[l].map(([word, count]) => (
                  <li key={word}><span className="mono">{word}</span><span className="muted">{count}</span></li>
                ))}
              </ol>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Estrellas frente a etiqueta" subtitle="Las celdas fuera de lo esperado (por ejemplo, 5 estrellas con etiqueta negativa) revelan etiquetas ruidosas.">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Etiqueta</th>{stars.map((s) => <th key={s} className="num">{"★".repeat(s)}</th>)}</tr></thead>
            <tbody>
              {labels.map((l) => (
                <tr key={l}>
                  <td><SentimentChip value={l} /></td>
                  {stars.map((s) => <td key={s} className="num">{eda.stars_by_label[l]?.[s] ?? eda.stars_by_label[l]?.[String(s)] ?? 0}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
