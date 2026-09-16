import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createElement } from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";
import * as tuningApi from "../app/lib/tuningApi.ts";

const source = await readFile(new URL("../app/components/TuningFunctionActions.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022,
} }).outputText;
const componentModule = { exports: {} };
new Function("require", "module", "exports", compiled)((id) => {
  if (id === "react/jsx-runtime") return jsxRuntime;
  if (id === "../lib/tuningApi") return tuningApi;
  throw new Error(`Unexpected import: ${id}`);
}, componentModule, componentModule.exports);
const { TuningFunctionActions } = componentModule.exports;
const ready = () => ({
  serviceState: "ready", session: { taskId: "toy", targetId: "joint", baseMethod: "Ours-Full" },
  counts: { usablePositive: 0, usableNegative: 0 }, annotations: new Map(),
  feedbackBreakdown: { unresolvedCount: 0 }, pendingAnnotationCount: 0, busy: false,
  probeSource: "original", selectedProbeUpdate: null, run: null, probeUpdateInProgress: false,
  refinementCapabilities: {
    weight_staged: { available: true, algorithmVersion: tuningApi.TUNING_LAUNCH_ALGORITHMS.weight_staged },
    weight_joint: { available: true, algorithmVersion: tuningApi.TUNING_LAUNCH_ALGORITHMS.weight_joint },
    update_probes: { available: true, algorithmVersion: tuningApi.PROBE_UPDATE_ALGORITHM },
  },
});
function buttons(state = ready(), props = {}) {
  const html = renderToStaticMarkup(createElement(TuningFunctionActions, {
    state, taskId: "toy", targetId: "joint", baseMethod: "Ours-Full", validationReady: true, ...props,
  }));
  return Object.fromEntries([...html.matchAll(/<button\b([^>]*)>(.*?)<\/button>/g)]
    .map((match) => [match[2], { disabled: /\bdisabled=/.test(match[1]), hidden: /\bhidden=/.test(match[1]), attributes: match[1] }]));
}

test("only Staged Weight Refinement is visible; hidden operations keep their existing guards", () => {
  const rendered = buttons();
  assert.deepEqual(Object.keys(rendered).filter((label) => !rendered[label].hidden), ["Staged Weight Refinement"]);
  assert.equal(rendered["Update Probes"].hidden, true);
  assert.equal(rendered["Joint Weight Tune"].hidden, true);
});

test("both weight buttons accept empty or uncertain-only feedback; Probe update still requires correction", () => {
  for (const annotations of [new Map(), new Map([["one", { label: 0 }]])]) {
    const rendered = buttons({ ...ready(), annotations });
    assert.equal(rendered["Staged Weight Refinement"].disabled, false);
    assert.equal(rendered["Joint Weight Tune"].disabled, false);
    assert.equal(rendered["Update Probes"].disabled, true);
    assert.match(rendered["Joint Weight Tune"].attributes, /原 VQA 训练监督/);
  }
});

test("zero feedback does not bypass loading, concurrency, Val, identity or Probe snapshot guards", () => {
  for (const change of [{ serviceState: "offline" }, { busy: true }, { pendingAnnotationCount: 1 },
    { run: { status: "running" } }, { probeUpdateInProgress: true },
    { feedbackBreakdown: { unresolvedCount: 1 } }, { probeSource: "updated" },
    { refinementCapabilities: null }, { session: { taskId: "another" } }]) {
    const rendered = buttons({ ...ready(), ...change });
    assert.equal(rendered["Staged Weight Refinement"].disabled, true, JSON.stringify(change));
    assert.equal(rendered["Joint Weight Tune"].disabled, true, JSON.stringify(change));
  }
  assert.equal(buttons(ready(), { validationReady: false })["Joint Weight Tune"].disabled, true);
});

test("held-out-only corrections and unconfirmed negatives are not treated as absent feedback", () => {
  for (const [annotation, counts] of [
    [{ label: 1, supervision: { includedInTune: false } }, { usablePositive: 0, usableNegative: 0 }],
    [{ label: -1, failureAttributionConfirmed: false }, { usablePositive: 0, usableNegative: 1 }],
  ]) {
    const rendered = buttons({ ...ready(), annotations: new Map([["one", annotation]]), counts });
    assert.equal(rendered["Joint Weight Tune"].disabled, true);
    assert.equal(rendered["Staged Weight Refinement"].disabled, true);
  }
});

test("confirmed empty attribution permits only robust Joint weight tuning, never native updates", () => {
  const task = "059_hico_task_hico_hugging_cat_robust_test";
  const relation = { label: -1, failureAttributionConfirmed: true, failedAttributeIds: [] };
  assert.equal(tuningApi.hasConfirmedNegativeFeedback(relation, task, "joint"), true);
  for (const [taskId, targetId] of [["032_hico_task_hico_hugging_cat", "joint"], [task, "cat"], [undefined, "joint"]]) {
    assert.equal(tuningApi.hasConfirmedNegativeFeedback(relation, taskId, targetId), false);
  }
  for (const annotation of [relation, { ...relation, failureAttributionConfirmed: false }, { ...relation, failedAttributeIds: undefined }]) {
    const state = { ...ready(), session: { taskId: task, targetId: "joint", baseMethod: "Ours-Full" },
      annotations: new Map([["one", annotation]]), counts: { usablePositive: 0, usableNegative: 1 } };
    const rendered = buttons(state, { taskId: task });
    assert.equal(rendered["Staged Weight Refinement"].disabled, annotation !== relation);
    assert.equal(rendered["Joint Weight Tune"].disabled, annotation !== relation);
    assert.equal(rendered["Update Probes"].disabled, true);
  }
  assert.equal(buttons({ ...ready(), annotations: new Map([["one", relation]]),
    counts: { usablePositive: 0, usableNegative: 1 } })["Staged Weight Refinement"].disabled, true);
});
