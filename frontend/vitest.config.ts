import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react-swc";
import path from "path";

export default defineConfig({
  plugins: [react()],

  // GitHub Pages repo name:
  // https://chowdharyaarna.github.io/TartanHacks26/
  base: "/TartanHacks26/",

  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },

  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
});
