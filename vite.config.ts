import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// "single" mode inlines all JavaScript and CSS into one HTML file for static
// hosting; the demo data bundle stays in /demo as separate gzip files.
export default defineConfig(({ mode }) => ({
  plugins: mode === "single" ? [react(), viteSingleFile()] : [react()],
  base: "./",
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
  build: {
    outDir: mode === "single" ? "dist-single" : "dist",
    target: "es2022",
    chunkSizeWarningLimit: 900,
  },
}));
