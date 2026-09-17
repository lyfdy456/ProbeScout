import vinext from "vinext";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { defineConfig } from "vite";
import { privateDevBoundary } from "./scripts/private-dev-boundary.mjs";

const WEB_ROOT = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const viteRoot = path.dirname(require.resolve("vite/package.json"));
const viteBrowserEnvironment = path.join(viteRoot, "dist/client/env.mjs").replaceAll("\\", "/");
const vinextRuntime = path.dirname(fileURLToPath(import.meta.resolve("vinext"))).replaceAll("\\", "/");
const rscRuntime = path.dirname(fileURLToPath(import.meta.resolve("@vitejs/plugin-rsc"))).replaceAll("\\", "/");
const publicDevModules = [`/@fs/${viteBrowserEnvironment}`, `/@fs/${vinextRuntime}/`, `/@fs/${rscRuntime}/`];
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";

export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 3000,
    strictPort: true,
    fs: {
      strict: true,
      allow: [WEB_ROOT, viteBrowserEnvironment, vinextRuntime, rscRuntime],
      deny: ["runtime/**", "**/.cookie-secret", "**/*.sqlite", "**/*.sqlite3", "**/*.sqlite3-shm", "**/*.sqlite3-wal"],
    },
    proxy: {
      "/api/tuning": {
        target: process.env.PCP_TUNING_API_URL ?? "http://127.0.0.1:8787",
        changeOrigin: false,
      },
    },
    ...(isCodexSeatbeltSandbox ? { watch: { useFsEvents: false, usePolling: true } } : {}),
  },
  plugins: [privateDevBoundary(publicDevModules), vinext()],
});
