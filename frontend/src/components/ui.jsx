import { SENTIMENTS, pct } from "../labels.js";

export function Card({ title, subtitle, children, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {title && <h2>{title}</h2>}
      {subtitle && <p className="muted">{subtitle}</p>}
      {children}
    </section>
  );
}

export function SentimentChip({ value }) {
  const meta = SENTIMENTS[value] || { name: value, className: "" };
  return <span className={`chip chip-${meta.className}`}>{meta.name}</span>;
}

export function ErrorBox({ message }) {
  return message ? <div className="error" role="alert">{message}</div> : null;
}

// One horizontal bar per row: [{ label, value, className }], value in [0, max].
export function Bars({ rows, max = 1, format = pct }) {
  return (
    <div className="bars">
      {rows.map((row) => (
        <div className="bar-row" key={row.label}>
          <span className="bar-label">{row.label}</span>
          <span className="bar-track">
            <span
              className={`bar-fill ${row.className || ""}`}
              style={{ width: `${Math.max(0, (row.value / max) * 100)}%` }}
            />
          </span>
          <span className="bar-value">{format(row.value)}</span>
        </div>
      ))}
    </div>
  );
}

export function SentimentBars({ probabilities }) {
  return (
    <Bars
      rows={Object.keys(SENTIMENTS).map((key) => ({
        label: SENTIMENTS[key].name,
        value: probabilities[key] ?? 0,
        className: SENTIMENTS[key].className,
      }))}
    />
  );
}

export function Tokens({ items, variant = "" }) {
  if (!items.length) return <span className="muted">(vacío)</span>;
  return (
    <div className="tokens">
      {items.map((token, i) => (
        <span className={`token ${variant}`} key={`${token}-${i}`}>
          {token}
        </span>
      ))}
    </div>
  );
}

export function Loading({ text = "Cargando…" }) {
  return <p className="muted">{text}</p>;
}
