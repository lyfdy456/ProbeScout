import { spawn } from "node:child_process";
import { resolvePython } from "./python-runtime.mjs";

const args = process.argv.slice(2);
if (!args.length) {
  console.error("Usage: node scripts/run-python.mjs <python arguments...>");
  process.exit(2);
}
const child = spawn(resolvePython(), args, {
  stdio: "inherit", env: process.env, windowsHide: true,
});
child.on("error", (error) => {
  console.error(`Unable to launch Python: ${error.message}`);
  process.exitCode = 1;
});
child.on("exit", (code) => { process.exitCode = code ?? 1; });
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => { if (!child.killed) child.kill(signal); });
}
