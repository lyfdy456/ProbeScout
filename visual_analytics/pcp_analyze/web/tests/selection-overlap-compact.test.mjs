import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createElement } from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";
import { summarizeSelectionOverlap } from "../app/lib/selectionOverlap.ts";

const source = await readFile(new URL("../app/components/SelectionOverlapPanel.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
}).outputText;
const componentModule = { exports: {} };
new Function("require", "module", "exports", compiled)((id) => {
  assert.equal(id, "react/jsx-runtime", "display must not add data dependencies");
  return jsxRuntime;
}, componentModule, componentModule.exports);
const { SelectionOverlapPanel } = componentModule.exports;
const makeMask = (...values) => Uint8Array.from(values);
const baseMask = makeMask(1, 1, 1, 1, 1, 1);
const conditions = [
  { id: "pcp", label: "PCP cluster 50", mask: makeMask(1, 1, 1, 0, 0, 0), rule: { kind: "category", field: "rank-cluster", value: "Fine-grained K50 · 50" }, ruleResult: { kind: "none", reason: "no-attributes" } },
  { id: "umap", label: "Embedding brush · UMAP", mask: makeMask(0, 1, 1, 1, 0, 0), rule: { kind: "projection", projection: "umap", xDomain: [2, 3], yDomain: [4, 5] }, ruleResult: { kind: "none", reason: "no-attributes" } },
];
function markup(extra = {}) {
  const result = summarizeSelectionOverlap({ baseMask, conditions, finalMask: makeMask(0, 1, 1, 0, 0, 0), ...extra });
  const before = structuredClone(result);
  const html = renderToStaticMarkup(createElement(SelectionOverlapPanel, { result, active: true }));
  assert.deepEqual(result, before);
  return html;
}

test("compact overlap shows independent counts and true intersection with shared collapsed rules", () => {
  const html = markup();
  assert.match(html, /selection-overlap-compact/);
  assert.match(html, /selection-overlap-diagram/);
  const compact = html.slice(html.indexOf('class="selection-overlap-counts"'), html.indexOf('<details'));
  assert.match(compact, />PCP:<\/span><strong>3<\/strong>/);
  assert.match(compact, />UMAP:<\/span><strong>3<\/strong>/);
  assert.match(compact, />Intersection:<\/span><strong>2<\/strong>/);
  assert.doesNotMatch(compact, /Final:/);
  assert.match(html, /<details class="selection-overlap-details"><summary>Rules &amp; counts<\/summary>/);
  assert.doesNotMatch(html, /<details[^>]*\bopen/);
  assert.match(html, /Direct rule/);
  assert.match(html, /Exact independent filter overlap counts/);
  assert.equal((html.match(/class="selection-overlap-circle /g) ?? []).length, 2);
  assert.match(html, /class="selection-overlap-set-code">A<\/text>/);
  assert.match(html, /class="selection-overlap-set-code">B<\/text>/);
});

test("a diagnostic has a separate final count and does not replace the intersection", () => {
  const html = markup({ stages: [{ id: "diagnostic", label: "Diagnostic Top 50", mask: makeMask(0, 1, 0, 0, 0, 0) }], finalMask: makeMask(0, 1, 0, 0, 0, 0) });
  const compact = html.slice(html.indexOf('class="selection-overlap-counts"'), html.indexOf('<details'));
  assert.match(compact, />Intersection:<\/span><strong>2<\/strong>/);
  assert.match(compact, />Final:<\/span><strong>1<\/strong>/);
  assert.match(html, /Diagnostic Top 50/);
});

test("more than three conditions retain every circle, count, exact AND and rule", () => {
  const extraConditions = [
    { id: "a", label: "Attr A", mask: makeMask(1, 1, 0, 1, 1, 0) },
    { id: "b", label: "Attr B", mask: makeMask(0, 1, 1, 1, 1, 1) },
  ].map((item) => ({ ...item, rule: { kind: "range", axisId: item.id, valueKind: "rank", lower: 0.9, upper: 1, source: "brush" }, ruleResult: { kind: "none", reason: "no-attributes" } }));
  const html = markup({ conditions: [...conditions, ...extraConditions], finalMask: makeMask(0, 1, 0, 0, 0, 0) });
  assert.equal((html.match(/class="selection-overlap-circle /g) ?? []).length, 4);
  const compact = html.slice(html.indexOf('class="selection-overlap-counts"'), html.indexOf('<details'));
  assert.match(compact, /PCP · Attr A:/);
  assert.match(compact, /PCP · Attr B:/);
  assert.match(compact, />Intersection:<\/span><strong>1<\/strong>/);
  assert.match(html, /Cumulative independent filter intersection/);
  assert.match(html, /AND 1/);
});

test("empty intersections stay visible as zero and inactive panels stay absent", () => {
  const disjoint = [conditions[0], { ...conditions[1], mask: makeMask(0, 0, 0, 1, 1, 1) }];
  const html = markup({ conditions: disjoint, finalMask: makeMask(0, 0, 0, 0, 0, 0) });
  assert.match(html, />Intersection:<\/span><strong>0<\/strong>/);
  assert.equal(renderToStaticMarkup(createElement(SelectionOverlapPanel, { result: { kind: "none", reason: "no-active-selection" }, active: false })), "");
});

test("diagram and summary are side by side without hiding expanded details", async () => {
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(css, /\.selection-overlap-compact\s*\{[^}]*grid-template-columns:\s*minmax\(112px, 176px\) minmax\(0, 1fr\)/);
  assert.match(css, /@container \(max-width: 400px\)[\s\S]*grid-template-columns: 112px minmax\(0, 1fr\)/);
  assert.doesNotMatch(css, /\.selection-overlap-(?:compact|details|panel)\s*\{[^}]*max-height:/);
});
