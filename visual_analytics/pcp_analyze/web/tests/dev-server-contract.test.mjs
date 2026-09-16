import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { isPrivateDevRequest } from "../scripts/private-dev-boundary.mjs";

const source = async (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("the integrated launcher binds one IPv4 port-3000 frontend to the loopback sidecar", async () => {
  const [launcher, vite, packageJson] = await Promise.all([
    source("scripts/dev.mjs"),
    source("vite.config.ts"),
    source("package.json"),
  ]);
  assert.match(launcher, /"dev", "--hostname", "127\.0\.0\.1", "--port", "3000"/);
  assert.match(launcher, /http:\/\/127\.0\.0\.1:8787\/api\/tuning\/health/);
  assert.match(vite, /host: "127\.0\.0\.1"/);
  assert.match(vite, /port: 3000,[\s\S]*?strictPort: true/);
  assert.match(vite, /"\/api\/tuning"[\s\S]*?http:\/\/127\.0\.0\.1:8787/);
  const pkg = JSON.parse(packageJson);
  assert.equal(pkg.scripts["dev:web"], "vinext dev --hostname 127.0.0.1 --port 3000");
});

test("the dev server denies mutable runtime data and private project files", async () => {
  const vite = await source("vite.config.ts");
  assert.match(vite, /privateDevBoundary\(\)/);
  assert.match(vite, /fs: \{[\s\S]*?strict: true,[\s\S]*?allow: \[WEB_ROOT\]/);

  for (const url of [
    "/runtime/tuning/workbench.sqlite3",
    "/runtime/tuning/.cookie-secret?url&inline",
    "/%72untime/tuning/users/example/effective_labels.json",
    "/%2572untime/tuning/workbench.sqlite3",
    "/@fs/X:/private/workbench.sqlite3",
    "/scripts/tuning_server.py",
    "/package.json",
    "/.openai/hosting.json",
  ]) {
    assert.equal(isPrivateDevRequest(url), true, url);
  }

  for (const url of [
    "/",
    "/api/tuning/health",
    "/data/catalog.json",
    "/app/Dashboard.tsx",
    "/@vite/client",
  ]) {
    assert.equal(isPrivateDevRequest(url), false, url);
  }
});
