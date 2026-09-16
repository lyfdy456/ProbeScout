import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("rank-profile selector uses the requested sentence case without changing its controls", async () => {
  const [dashboard, css] = await Promise.all([source("app/Dashboard.tsx"), source("app/globals.css")]);
  assert.match(dashboard, /<span className="rank-profile-clusters-label">Rank-profile clusters<\/span>/);
  assert.match(css, /\.toolbar-controls label > \.rank-profile-clusters-label\s*\{\s*text-transform: none;/);
  assert.doesNotMatch(dashboard, /PCP clusters/);
  assert.match(dashboard, /value=\{rankClusterScheme\}/);
});

test("staged refinement label retains the same backend mode and action guards", async () => {
  const actions = await source("app/components/TuningFunctionActions.tsx");
  assert.match(actions, /mode: "weight_staged",\s*label: "Staged Weight Refinement"/);
  assert.doesNotMatch(actions, /Staged Weight Tune/);
  assert.match(actions, /state\.runTuning\(mode\)/);
  assert.match(actions, /onClick=\{\(\) => launch\(action\.mode\)\}/);
});

test("ranking selector presents aliases but keeps canonical method option values", async () => {
  const dashboard = await source("app/Dashboard.tsx");
  assert.match(dashboard, /<option key=\{method\} value=\{method\}>[\s\S]*?methodDisplayLabel\(method\)/);
  assert.match(dashboard, /id: method,\s*label: methodDisplayLabel\(method\)/);
  assert.match(dashboard, /methodDisplayLabel\(rankMethod\)/);
});
