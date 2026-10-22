import { useState } from "react";
import { api } from "../api.js";
import { SENTIMENTS, pct } from "../labels.js";
import { Bars, Card, ErrorBox, SentimentChip } from "./ui.jsx";

function downloadCsv(results) {
  const escape = (value) => `"${String(value).replaceAll('"', '""')}"`;
  const lines = [
    "fila,texto,sentimiento,probabilidad",
    ...results.map((r) => [r.fila, escape(r.texto), r.sentimiento, r.probabilidad].join(",")),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "predicciones.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

export default function Batch() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function upload(event) {
    event.preventDefault();
    if (!file) return setError("Elige un archivo .txt o .csv.");
    setLoading(true);
    setError("");
    try {
      setData(await api.batch(file));
    } catch (err) {
      setData(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="stack">
      <Card
        title="Analizar un archivo"
        subtitle="Un .txt con una reseña por línea, o un .csv con una columna llamada «texto». Máximo 500 reseñas y 1 MB."
      >
        <form onSubmit={upload} className="row">
          <input
            type="file"
            accept=".txt,.csv"
            onChange={(e) => {
              setFile(e.target.files[0] || null);
              setData(null);
            }}
          />
          <button className="primary" disabled={loading}>
            {loading ? "Procesando…" : "Clasificar"}
          </button>
        </form>
        <p className="muted small">
          Prueba con <a href="/ejemplo_lote.csv" download>ejemplo_lote.csv</a>.
        </p>
        <ErrorBox message={error} />
      </Card>

      {data && (
        <>
          <div className="grid two">
            <Card title={`${data.total} reseñas clasificadas`}>
              <Bars
                max={data.total}
                format={(v) => `${v} (${pct(v / data.total, 0)})`}
                rows={Object.keys(SENTIMENTS).map((key) => ({
                  label: SENTIMENTS[key].name,
                  value: data.resumen[key],
                  className: SENTIMENTS[key].className,
                }))}
              />
              <button className="ghost" onClick={() => downloadCsv(data.resultados)}>
                Descargar resultados (.csv)
              </button>
            </Card>
            <Card title="Filas omitidas" subtitle="No cumplen la validación de la API.">
              {data.omitidas.length === 0 ? (
                <p className="muted">Ninguna.</p>
              ) : (
                <ul className="plain">
                  {data.omitidas.map((row) => (
                    <li key={row.fila}>
                      Fila {row.fila}: {row.motivo}
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>
          <Card title="Resultados">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Fila</th>
                    <th>Texto</th>
                    <th>Sentimiento</th>
                    <th className="num">Probabilidad</th>
                  </tr>
                </thead>
                <tbody>
                  {data.resultados.map((r) => (
                    <tr key={r.fila}>
                      <td>{r.fila}</td>
                      <td>{r.texto}</td>
                      <td><SentimentChip value={r.sentimiento} /></td>
                      <td className="num">{pct(r.probabilidad)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
