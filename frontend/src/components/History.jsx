import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { pct } from "../labels.js";
import { Card, ErrorBox, SentimentChip } from "./ui.jsx";

const formatDate = (value) =>
  new Date(value).toLocaleString("es-BO", { dateStyle: "short", timeStyle: "medium" });

export default function History() {
  const [predictions, setPredictions] = useState([]);
  const [trainings, setTrainings] = useState([]);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    setError("");
    Promise.all([api.history(30), api.trainings()])
      .then(([p, t]) => {
        setPredictions(p);
        setTrainings(t);
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(load, [load]);

  return (
    <div className="stack">
      <ErrorBox message={error} />
      <Card title="Últimas predicciones" subtitle="Guardadas por la API en la tabla predicciones de PostgreSQL.">
        <button className="ghost" onClick={load}>Actualizar</button>
        {predictions.length === 0 ? (
          <p className="muted">Todavía no hay predicciones.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Fecha</th><th>Texto</th><th>Sentimiento</th><th className="num">Probabilidad</th><th>Origen</th><th>Versión</th></tr>
              </thead>
              <tbody>
                {predictions.map((p) => (
                  <tr key={p.id}>
                    <td className="nowrap">{formatDate(p.fecha)}</td>
                    <td>{p.texto}</td>
                    <td><SentimentChip value={p.sentimiento} /></td>
                    <td className="num">{pct(p.probabilidad)}</td>
                    <td>{p.origen}</td>
                    <td className="mono small">{p.version_modelo}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="Entrenamientos" subtitle="Cada ejecución de python -m src.pipeline queda registrada en la tabla entrenamientos.">
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Fecha</th><th>Versión</th><th>Modelo</th><th className="num">F1 macro CV</th><th className="num">F1 macro prueba</th><th className="num">Exactitud prueba</th></tr>
            </thead>
            <tbody>
              {trainings.map((t) => (
                <tr key={t.version}>
                  <td className="nowrap">{formatDate(t.fecha)}</td>
                  <td className="mono small">{t.version}</td>
                  <td>{t.modelo}</td>
                  <td className="num">{t.cv_f1_macro.toFixed(3)}</td>
                  <td className="num">{t.test_f1_macro.toFixed(3)}</td>
                  <td className="num">{pct(t.test_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
