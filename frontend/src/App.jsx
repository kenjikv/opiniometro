import { useEffect, useState } from "react";
import { api } from "./api.js";
import Analyze from "./components/Analyze.jsx";
import Batch from "./components/Batch.jsx";
import Data from "./components/Data.jsx";
import History from "./components/History.jsx";
import Metrics from "./components/Metrics.jsx";

const TABS = [
  { id: "analyze", label: "Analizar", component: Analyze },
  { id: "batch", label: "Lote", component: Batch },
  { id: "metrics", label: "Métricas", component: Metrics },
  { id: "data", label: "Datos", component: Data },
  { id: "history", label: "Historial", component: History },
];

function StatusPill() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setError(true));
  }, []);

  if (error) return <span className="pill pill-bad">API sin conexión</span>;
  if (!health) return <span className="pill">Conectando…</span>;
  return (
    <span className={`pill ${health.estado === "ok" ? "pill-ok" : "pill-bad"}`}>
      {health.modelo_cargado ? `${health.modelo} · v${health.version}` : "Modelo sin entrenar"}
      {" · BD "}
      {health.base_de_datos}
    </span>
  );
}

export default function App() {
  const [tab, setTab] = useState("analyze");
  const Current = TABS.find((t) => t.id === tab).component;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Opiniómetro</h1>
          <p className="subtitle">Sentimiento de reseñas de restaurantes en español</p>
        </div>
        <StatusPill />
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={tab === t.id ? "tab active" : "tab"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        <Current />
      </main>

      <footer className="footer">
        Ejemplo del Proyecto Integrador · datos sintéticos · API en <a href="/api/docs">/api/docs</a>
      </footer>
    </div>
  );
}
