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

test("identity and advanced controls live once at the top; only DG exposes source and training", async () => {
  const dashboard = await source("app/Dashboard.tsx");
  const top = dashboard.slice(dashboard.indexOf('<div className="workspace-controls">'), dashboard.indexOf('className={`scope-purpose-banner'));
  assert.match(top, /<UserSessionControls state=\{tuning\}/);
  assert.match(top, /<details className="advanced-settings">[\s\S]*?<summary>Advanced Settings<\/summary>/);
  assert.match(top, /onClick=\{reloadCurrentTask\}>Reload/);
  assert.match(top, /developmentMode && !hierarchicalFusionSelected && !legacyBaselineSelected && \([\s\S]*?<ProbeSourceControls/);
  assert.doesNotMatch(top, /TuningFunctionActions|\bopen[=>\s]/);
  const right = dashboard.slice(dashboard.indexOf('<div className="refinement-actions-block">'), dashboard.indexOf('<section className="workspace-panel test-audit-panel"'));
  assert.match(right, /<span>Refinement and Validation<\/span>/);
  assert.match(right, /developmentMode && !hierarchicalFusionSelected && !legacyBaselineSelected && \([\s\S]*?<TuningFunctionActions/);
  assert.match(right, /<TuningPanel[\s\S]*?onApplyRun=\{applyTunedRanking\}[\s\S]*?onRevertRun=\{revertTunedRanking\}/);
  assert.doesNotMatch(dashboard, /className="function-block"/);
  for (const name of ["UserSessionControls", "ProbeSourceControls", "TuningFunctionActions"]) {
    assert.equal((dashboard.match(new RegExp(`<${name}\\b`, "g")) ?? []).length, 1, name);
  }
  assert.equal((dashboard.match(/= useTuningSession\(/g) ?? []).length, 1);
});

test("extracted controls reuse session callbacks and preserve source guards without local state", async () => {
  const [identity, sourceControl, actions, panel] = await Promise.all([
    source("app/components/UserSessionControls.tsx"), source("app/components/ProbeSourceControls.tsx"),
    source("app/components/TuningFunctionActions.tsx"), source("app/components/TuningPanel.tsx"),
  ]);
  assert.match(identity, /void saveDisplayName\(displayName\)\.catch/);
  assert.match(identity, /void startNewSession\(\)\.catch/);
  assert.match(identity, /busy \|\| runInProgress \|\| state\.probeUpdateInProgress/);
  assert.doesNotMatch(panel, /saveDisplayName|startNewSession|name="displayName"/);
  assert.doesNotMatch(actions, /selectProbeSource|<select/);
  for (const guard of ["state.busy", "annotationsPending", "runInProgress", "state.probeUpdateInProgress", 'state.serviceState !== "ready"', "!validationReady", "!sessionMatches"]) {
    assert.ok(sourceControl.includes(guard), guard);
  }
  assert.match(sourceControl, /compatibleProbeUpdates\(state\.probeUpdates, state\.session\.id, taskId\)/);
  assert.match(sourceControl, /state\.selectProbeSource\("updated", event\.target\.value\)/);
  assert.doesNotMatch(`${identity}\n${sourceControl}`, /useState|useEffect|useTuningSession\(|createTuningRun|updateAnnotation/);
});

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

test("training audit is collapsed after the result and never hides pending, error or stale states", async () => {
  const Panel = await component("TuningPanel");
  const input = state();
  input.resultStale = true;
  input.pendingAnnotationCount = 2;
  input.feedbackBreakdown.unresolvedCount = 1;
  input.error = "Request failed";
  const html = renderToStaticMarkup(createElement(Panel, { state: input, applyError: "Apply failed" }));
  const firstDetails = html.indexOf("<details");
  const visible = html.slice(0, firstDetails);
  for (const message of ["Saving 2 labels", "Checking 1 saved label", "Request failed", "Apply failed", "Labels changed · rerun"]) {
    assert.ok(visible.includes(message), message);
  }
  const auditStart = html.indexOf('<details class="tuning-training-details">');
  assert.ok(auditStart > html.indexOf('class="tuning-val-summary"'));
  const audit = html.slice(auditStart);
  for (const label of ["Training details", "Val protocol", "feedback"]) assert.ok(audit.includes(label), label);
  assert.doesNotMatch(visible, /tuning-feedback-breakdown|Val protocol/);
  assert.doesNotMatch(html, /<details[^>]*\bopen(?:[= >])/);
  assert.match(html, /disabled=""[^>]*>Use in Top ranking/);
});

test("visible Staged action fills one row and summary uses compact inline evidence", async () => {
  const css = await source("app/globals.css");
  assert.match(css, /\.refinement-training-actions\s*\{[^}]*grid-template-columns: minmax\(0, 1fr\)/);
  assert.match(css, /\.refinement-training-actions \[hidden\]\s*\{[^}]*display: none !important/);
  assert.doesNotMatch(css, /\.refinement-training-actions \.function-probe-update\s*\{[^}]*grid-column: 1 \/ -1/);
  assert.match(css, /\.refinement-training-actions \.function-probe-update\s*\{[^}]*grid-column: auto;/, "override the legacy shared probe/source full-width selector");
  assert.match(css, /\.tuning-val-summary\s*\{[^}]*display: flex;[^}]*flex-wrap: wrap;/);
  const Panel = await component("TuningPanel");
  const input = state();
  input.run.before.ap = .9197;
  input.run.after.ap = .9271;
  const html = renderToStaticMarkup(createElement(Panel, { state: input, isApplied: true }));
  const visible = html.slice(0, html.indexOf("<details"));
  for (const value of ["91.97%", "92.71%", "+0.74 pp"]) assert.ok(visible.includes(value), value);
  assert.ok(html.includes("Ranking applied"));
  assert.ok(html.includes("Return to baseline"));
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

test("top identity and source controls render existing session values without duplicating training", async () => {
  const Identity = await component("UserSessionControls");
  const Source = await component("ProbeSourceControls");
  const props = { state: state(), taskId: "task-one", targetId: "joint", targetLabel: "Joint", baseMethod: "Ours-Full", validationReady: true };
  const identity = renderToStaticMarkup(createElement(Identity, props));
  assert.match(identity, /value="Researcher"/);
  assert.match(identity, /session-one/);
  assert.match(identity, /Private session/);
  const controls = renderToStaticMarkup(createElement(Source, props));
  assert.match(controls, /aria-label="Probe source"/);
  assert.match(controls, /value="updated" disabled=""/);
  assert.doesNotMatch(controls, /Update Probes|Staged Weight Refinement|Joint Weight Tune/);
  const blocked = renderToStaticMarkup(createElement(Source, { ...props, validationReady: false }));
  assert.match(blocked, /<select[^>]*aria-label="Probe source"[^>]*disabled=""/);
});
