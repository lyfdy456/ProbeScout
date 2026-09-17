import assert from "node:assert/strict";
import test from "node:test";
import { isPrivateDevRequest } from "../scripts/private-dev-boundary.mjs";

test("the exact Vite environment module loads while other filesystem paths stay private", () => {
  const module = "/@fs/C:/repo/node_modules/vite/dist/client/env.mjs";
  assert.equal(isPrivateDevRequest(module, [module]), false);
  assert.equal(isPrivateDevRequest(`${module}?t=123`, [module]), false);
  const runtime = "/@fs/C:/repo/node_modules/vinext/dist/";
  assert.equal(isPrivateDevRequest(`${runtime}server/app-browser-entry.js?v=1`, [runtime]), false);
  for (const suffix of [".env", ".cookie-secret", "state.sqlite3", "%2e%2e%2f%2e%2e%2fprivate.js"]) {
    assert.equal(isPrivateDevRequest(runtime + suffix, [runtime]), true, suffix);
  }
  for (const path of ["/@fs/C:/repo/runtime/workbench.sqlite3", "/runtime/x", "/scripts/tuning_server.py",
    "/@fs/C:/repo/node_modules/vite/dist/client/other.mjs", "/.env", `${module}/../secret`]) {
    assert.equal(isPrivateDevRequest(path, [module]), true, path);
  }
});
