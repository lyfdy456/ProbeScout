import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(webRoot, "..", "..", "..");

export function resolvePython() {
  const override = process.env.PROBESCOUT_PYTHON || process.env.PCP_TUNING_PYTHON;
  if (override) return override;
  const executable = process.platform === "win32" ? ["Scripts", "python.exe"] : ["bin", "python"];
  const candidates = [
    path.join(repositoryRoot, "probe_learning", ".venv", ...executable),
    path.join(repositoryRoot, ".venv", ...executable),
  ];
  return candidates.find(existsSync) || "python";
}
