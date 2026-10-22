import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development (npm run dev) Vite forwards /api to the backend, just like
// nginx does in the container. The browser always talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.API_URL || "http://localhost:8000" },
  },
});
