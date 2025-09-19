import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Proxy server to avoid CORS
  server: {
    proxy: {"/api": "http://localhost:8000"}
  }
});