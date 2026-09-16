import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createElement } from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

// Exercise the real, pure display component without a build or runtime data.
const source = await readFile(new URL("../app/components/SelectionRuleSummaryPanel.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
}).outputText;
const componentModule = { exports: {} };
new Function("require", "module", "exports", compiled)((id) => {
  assert.equal(id, "react/jsx-runtime", "the disclosure must not add runtime data dependencies");
  return jsxRuntime;
}, componentModule, componentModule.exports);
const { SelectionRuleSummaryPanel } = componentModule.exports;

const result = {
  kind: "summary",
  summary: {
    selectedCount: 213, baseCount: 1000, lowerQuantile: 0.05, upperQuantile: 0.95,
    attributes: [{
      attributeId: "cat", attributeLabel: "Cat", sampleCount: 213,
      rank: { minimum: 0.9998, maximum: 1, lowerQuantile: 0.99991, upperQuantile: 0.99997 },
      calibratedScore: { minimum: 0.01, maximum: 1, lowerQuantile: 0.15149, upperQuantile: 0.9977 },
    }],
  },
};

test("selected range summary is a native, initially closed disclosure with a compact count", () => {
  const html = renderToStaticMarkup(createElement(SelectionRuleSummaryPanel, { result, active: true }));
  const detailsTag = html.match(/<details\b[^>]*>/)?.[0];
  assert.ok(detailsTag);
  assert.match(detailsTag, /selection-rule-disclosure/);
  assert.doesNotMatch(detailsTag, /\bopen(?:=|\s|>)/);
  const summary = html.match(/<summary\b[^>]*>([\s\S]*?)<\/summary>/)?.[1];
  assert.match(summary, /213 selected/);
  assert.match(summary, /5–95% ranges/);
  assert.doesNotMatch(summary, /Selected rows|<table|Cat/);
  assert.match(html, /<summary[^>]*aria-live="polite"/);
});

test("expanded body retains exact rank precision, three-decimal scores and audit tooltips", () => {
  const before = structuredClone(result);
  const html = renderToStaticMarkup(createElement(SelectionRuleSummaryPanel, { result, active: true }));
  assert.match(html, /<\/summary><div class="selection-rule-body">/);
  assert.match(html, /<table class="selection-rule-table">/);
  assert.match(html, /Rank <small>1 = best<\/small>/);
  assert.match(html, /0\.9999100<span aria-hidden="true">–<\/span>0\.9999700/);
  assert.match(html, /0\.151<span aria-hidden="true">–<\/span>0\.998/);
  assert.match(html, /Rank observed min 0\.999800 · max 1\.000000/);
  assert.match(html, /Calibrated score observed min 0\.010000 · max 1\.000000/);
  assert.match(html, /Descriptive only; it does not add another filter\./);
  assert.deepEqual(result, before);
});

test("inactive and empty selections keep the existing explanatory text without an empty disclosure", () => {
  const inactive = renderToStaticMarkup(createElement(SelectionRuleSummaryPanel, { result, active: false }));
  assert.doesNotMatch(inactive, /<details|213 selected/);
  assert.match(inactive, /Brush or choose a cluster/);
  const empty = renderToStaticMarkup(createElement(SelectionRuleSummaryPanel, {
    result: { kind: "none", reason: "empty-selection" }, active: true,
  }));
  assert.doesNotMatch(empty, /<details|<table/);
  assert.match(empty, /The current selection is empty\./);
});
