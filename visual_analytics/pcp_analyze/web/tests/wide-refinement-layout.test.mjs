import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
const dashboard = await readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8");

test("desktop restores the original three tracks without expanding the page", () => {
  assert.match(dashboard, /className="workspace-width-reference"[\s\S]*?className=\{`workspace-grid/);
  assert.match(css, /\.workspace-width-reference\s*\{[^}]*container-type: inline-size;[^}]*max-width: 1920px/s);
  assert.match(css, /\.workspace-grid\s*\{[^}]*grid-template-columns: minmax\(415px, 0\.95fr\) minmax\(500px, 1\.16fr\) minmax\(365px, 0\.86fr\)/s);
  assert.doesNotMatch(css, /--original-track-space|width: calc\(100cqw \+/);
});

test("diagnostics keep two columns but expose one scrollable row on desktop and narrow screens", () => {
  assert.match(css, /\.smart-filter-grid\s*\{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/s);
  assert.match(css, /@media \(max-width: 760px\) \{[^}]*\}[^}]*\.smart-filter-grid \{ grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/s);
  assert.match(css, /\.smart-filter-scroll\s*\{[^}]*height: var\(--diagnostic-card-height, 96px\);[^}]*overflow-y: auto;/s);
  assert.match(css, /\.smart-filter-grid\s*\{[^}]*grid-auto-rows: var\(--diagnostic-card-height, 96px\)/s);
});

test("feedback keeps two horizontally scrollable images beside its shared editor", () => {
  assert.match(css, /\.annotation-review-grid\s*\{[^}]*--review-visible-columns: 2;[^}]*grid-auto-flow: column;[^}]*grid-auto-columns: calc\(\(100% - \(var\(--review-visible-columns\) - 1\) \* 6px\) \/ var\(--review-visible-columns\)\);[^}]*grid-template-columns: none;/s);
  assert.doesNotMatch(css, /--review-visible-columns: [345]/);
  assert.doesNotMatch(css, /@container \(max-width: 700px\)\s*\{\s*\.annotation-review-workspace/);
  assert.match(css, /@container \(max-width: 300px\)\s*\{\s*\.annotation-review-workspace\s*\{\s*grid-template-columns: minmax\(0, 1fr\)/);
  assert.doesNotMatch(css, /\.annotation-review-grid\s*\{[^}]*grid-template-columns: repeat/s);
  assert.match(css, /\.annotation-review-workspace\s*\{[^}]*grid-template-columns: minmax\(0, 1fr\) minmax\(150px, 40%\)/s);
  assert.match(css, /\.annotation-review-scroll\s*\{[^}]*overflow-x: auto;[^}]*overflow-y: hidden;/s);
  assert.match(css, /\.annotation-review-editor \.gallery-preference\s*\{[^}]*flex-wrap: wrap;/s);
  assert.match(css, /\.annotation-review-editor \.failure-attribute-options label span\s*\{[^}]*overflow-wrap: anywhere/s);
});

test("PCP legend retains all clusters inside two fixed-height scrollable rows", async () => {
  const component = await readFile(new URL("../app/components/ClusterSummaryParallelCoordinates.tsx", import.meta.url), "utf8");
  assert.match(css, /\.pcp-cluster-legend\s*\{[^}]*grid-auto-rows: 30px;[^}]*gap: 6px 8px;[^}]*max-height: 66px;[^}]*overflow-y: auto;/s);
  assert.match(component, /className="pcp-cluster-legend"[\s\S]*?aria-label="PCP cluster legend"[\s\S]*?tabIndex=\{0\}/);
  assert.match(component, /profilePaths\.map\(\(profile\)/);
  assert.doesNotMatch(component, /maxHeight: denseProfiles|overflowY: denseProfiles/);
  assert.match(component, /legend\.scrollTop/);
});

test("selection disclosure has visible state and keyboard focus cues", () => {
  assert.match(css, /\.selection-rule-disclosure\[open\] > summary::before/);
  assert.match(css, /\.selection-rule-disclosure > summary:focus-visible/);
});
