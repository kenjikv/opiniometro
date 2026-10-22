// Every call goes to /api on the same origin; nginx (or Vite in development)
// forwards it to FastAPI. Errors carry the backend's message in Spanish.

async function request(path, options) {
  let response;
  try {
    response = await fetch(`/api${path}`, options);
  } catch {
    throw new Error("No se pudo conectar con la API. ¿Está levantado el contenedor api?");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Error ${response.status}`);
  return body;
}

export const api = {
  health: () => request("/health"),
  predict: (texto) =>
    request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto }),
    }),
  batch: (file) => {
    const form = new FormData();
    form.append("archivo", file);
    return request("/predict/batch", { method: "POST", body: form });
  },
  metrics: () => request("/metrics"),
  history: (limit = 30) => request(`/history?limit=${limit}`),
  trainings: () => request("/trainings"),
};
