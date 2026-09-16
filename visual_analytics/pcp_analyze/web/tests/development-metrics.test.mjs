import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createElement } from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";
import * as tuningApi from "../app/lib/tuningApi.ts";
import * as tuningEvaluation from "../app/lib/tuningEvaluation.ts";
import * as hierarchicalPcp from "../app/lib/hierarchicalPcp.ts";
import * as methodDisplay from "../app/lib/methodDisplay.js";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

// Transpile only these pure presentation components in memory: no build,
// backend, session creation, data access or generated files are involved.
async function component(name) {
  const text = await source(`app/components/${name}.tsx`);
  const compiled = ts.transpileModule(text, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const imports = {
    "react/jsx-runtime": jsxRuntime,
    "../lib/tuningApi": tuningApi,
    "../lib/tuningEvaluation": tuningEvaluation,
    "../lib/hierarchicalPcp": hierarchicalPcp,
    "../lib/methodDisplay.js": methodDisplay,
  };
  const componentModule = { exports: {} };
  new Function("require", "module", "exports", compiled)((id) => {
    assert.ok(id in imports, `unexpected runtime import ${id}`);
    return imports[id];
  }, componentModule, componentModule.exports);
  return componentModule.exports[name];
}

function state(scope = "vqa-validation") {
  const metrics = (ap) => ({ ap, evaluationCount: 40, positiveCount: 10, bestF1: .4, tpAt30: 8, tpAt50: 10, tpAt100: 10, tpAt200: 10 });
  return {
    serviceState: "ready", user: { displayName: "Researcher", identitySource: "anonymous" },
    session: { id: "session-one", taskId: "task-one", targetId: "joint", baseMethod: "Ours-Full" },
    annotations: new Map(), counts: { usablePositive: 1, usableNegative: 0 },
    feedbackBreakdown: { unresolvedCount: 0, totalCount: 1, newCount: 1, existingCount: 0 },
    run: { id: "run-one", mode: "weight_staged", evaluationScope: scope, status: "succeeded", annotationCount: 1,
      before: metrics(.2), after: metrics(.3), testEvaluation: { before: metrics(.4), after: metrics(.5) } },
    busy: false, probeUpdateInProgress: false, pendingAnnotationCount: 0, resultStale: false,
    probeSource: "original", selectedProbeUpdate: null, probeUpdates: [],
    applyRun: async () => {}, revertRun: () => {}, saveDisplayName: async () => {}, startNewSession: async () => {},
  };
}

test("Development contains Validation metrics only, including closed details", async () => {
  const Panel = await component("TuningPanel");
  const html = renderToStaticMarkup(createElement(Panel, { state: state() }));
  const summaryStart = html.indexOf('class="tuning-val-summary"');
  const detailsStart = html.indexOf('<details class="tuning-evaluation-result">');
  assert.ok(summaryStart >= 0 && detailsStart > summaryStart);
  const summary = html.slice(summaryStart, detailsStart);
  for (const label of ["Val AP:", "Before AP", "After AP", "20.00%", "30.00%", "→", "+10.00 pp"]) assert.ok(summary.includes(label), label);
  assert.doesNotMatch(summary, /Test|40\.00%|50\.00%/);
  assert.doesNotMatch(html, /<details[^>]*\bopen(?:[= >])/);
  const details = html.slice(detailsStart);
  for (const label of ["Evaluation details", "Validation AP comparison", "Val · TP@K", "Best F1", "Use in Top ranking"]) assert.ok(details.includes(label), label);
  assert.doesNotMatch(html, /Test|50\.00%/);
  assert.ok(details.includes("Apply ranking"), "closed evaluation summary points to the apply action");
});

test("historical runs never acquire a current Val AP summary", async () => {
  const Panel = await component("TuningPanel");
  for (const [scope, label] of [["test", "Saved run"], ["validation", "legacy Web Validation"], ["clean-validation", "Clean Validation"], ["probe-validation", "Probe Val"]]) {
    const html = renderToStaticMarkup(createElement(Panel, { state: state(scope) }));
    assert.doesNotMatch(html, /class="tuning-val-summary"/);
    assert.ok(html.includes(label), scope);
    assert.ok(html.includes("Historical evaluation · original scope"));
    assert.doesNotMatch(html, /<details[^>]*\bopen(?:[= >])/);
  }
});

test("Development never renders metrics from Test-only, missing or unknown evaluation scopes", async () => {
  const Panel = await component("TuningPanel");
  for (const scope of ["test", undefined, "unknown-scope"]) {
    const input = state();
    input.run.evaluationScope = scope;
    const html = renderToStaticMarkup(createElement(Panel, { state: input }));
    assert.doesNotMatch(html, /Test|20\.00%|30\.00%|40\.00%|50\.00%|Best F1|TP@|AP comparison/);
    assert.match(html, /No Validation metrics recorded/);
    assert.match(html, /Use in Top ranking/);
  }
});

test("Development and Frozen Test rendering stays separate for both schedules and probe sources", async () => {
  const Development = await component("TuningPanel");
  const TestView = await component("FrozenTestEvaluation");
  for (const mode of ["weight_staged", "weight_joint"]) {
    for (const probeSource of ["original", "updated"]) {
      const input = state();
      Object.assign(input.run, { mode, probeSource });
      input.run.testEvaluation.before.bestF1 = .6713;
      input.run.testEvaluation.after.bestF1 = .7861;
      input.run.testEvaluation.after.tpAt50 = 37;
      const development = renderToStaticMarkup(createElement(Development, { state: input }));
      assert.doesNotMatch(development, /Test|67\.13%|78\.61%|50\.00%|>37</);
      assert.match(development, /20\.00%/);
      assert.match(development, /30\.00%/);
      const frozen = renderToStaticMarkup(createElement(TestView, { run: input.run }));
      for (const value of ["Test AP comparison", "40.00%", "50.00%", "67.13%", "78.61%", "37", "Test · TP@K"]) {
        assert.ok(frozen.includes(value), value);
      }
      assert.doesNotMatch(frozen, /20\.00%|30\.00%|<button/);
    }
  }
});

test("Frozen Test preserves old Test-only results and missing secondary metrics without Val fallback", async () => {
  const Panel = await component("FrozenTestEvaluation");
  const legacy = state("test");
  const oldHtml = renderToStaticMarkup(createElement(Panel, { run: legacy.run }));
  assert.match(oldHtml, /Test \(legacy\)/);
  assert.match(oldHtml, /20\.00%/);
  assert.match(oldHtml, /30\.00%/);
  assert.doesNotMatch(oldHtml, /50\.00%/);
  const missing = state();
  delete missing.run.testEvaluation;
  const missingHtml = renderToStaticMarkup(createElement(Panel, { run: missing.run }));
  assert.match(missingHtml, /Not recorded/);
  assert.doesNotMatch(missingHtml, /20\.00%|30\.00%|40\.00%|50\.00%/);
});
