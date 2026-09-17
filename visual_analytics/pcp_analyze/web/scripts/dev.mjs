import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { resolvePython } from "./python-runtime.mjs";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const tuningScript = path.join(webRoot, "scripts", "tuning_server.py");
const python = resolvePython();
const vinextCli = path.join(webRoot, "node_modules", "vinext", "dist", "cli.js");
const apiPort = process.env.PROBESCOUT_API_PORT || "8787";
const webPort = process.env.PROBESCOUT_WEB_PORT || "3000";

let stopping = false;
let sidecar;
let frontend;

function stop(exitCode = 0) {
  if (stopping) return;
  stopping = true;
  if (frontend && !frontend.killed) frontend.kill();
  if (sidecar && !sidecar.killed) sidecar.kill();
  setTimeout(() => process.exit(exitCode), 250).unref();
}

async function waitForSidecar() {
  const deadline = Date.now() + 30_000;
  const endpoint = `http://127.0.0.1:${apiPort}/api/tuning/health`;
  while (Date.now() < deadline) {
    if (sidecar.exitCode !== null) {
      throw new Error(`Tuning service exited with code ${sidecar.exitCode}`);
    }
    try {
      const response = await fetch(endpoint, { signal: AbortSignal.timeout(1_000) });
      if (response.ok) return;
    } catch {
      // The Python process imports the scientific environment before listening.
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`Timed out waiting for the tuning service on port ${apiPort}`);
}

process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));

sidecar = spawn(python, [tuningScript, "--port", apiPort], {
  cwd: webRoot,
  env: { ...process.env, PYTHONUNBUFFERED: "1" },
  stdio: "inherit",
  windowsHide: true,
});
sidecar.on("error", (error) => {
  console.error(`Unable to start the tuning service: ${error.message}`);
  stop(1);
});
sidecar.on("exit", (code) => {
  if (!stopping) {
    console.error(`Tuning service stopped unexpectedly (code ${code ?? "unknown"}).`);
    stop(code || 1);
  }
});

try {
  await waitForSidecar();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  stop(1);
  await new Promise(() => {});
}

frontend = spawn(
  process.execPath,
  [vinextCli, "dev", "--hostname", "127.0.0.1", "--port", webPort],
  {
  cwd: webRoot,
  env: { ...process.env, PCP_TUNING_API_URL: `http://127.0.0.1:${apiPort}` },
  stdio: "inherit",
  windowsHide: true,
  },
);
frontend.on("error", (error) => {
  console.error(`Unable to start Vinext: ${error.message}`);
  stop(1);
});
frontend.on("exit", (code) => stop(code || 0));
