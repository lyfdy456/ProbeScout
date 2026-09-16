import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import * as d3 from "d3";
import ts from "typescript";
import * as chartTypography from "../app/lib/chartTypography.ts";
import * as navigation from "../app/lib/clusterSummaryNavigation.ts";
import * as hierarchy from "../app/lib/hierarchicalPcp.ts";
import * as methodDisplay from "../app/lib/methodDisplay.js";
import * as feedback from "../app/lib/feedbackLabels.ts";
import * as tuningApi from "../app/lib/tuningApi.ts";
import * as smartFilter from "../app/smartFilter.ts";

const source = (name) => readFile(new URL(`../${name}`, import.meta.url), "utf8");
async function component(name, hooks = React) {
  const text = await source(`app/components/${name}.tsx`);
  const imports = {
    react: hooks, "react/jsx-runtime": jsxRuntime, d3,
    "../lib/chartTypography": chartTypography,
    "../lib/clusterSummaryNavigation": navigation,
    "../lib/hierarchicalPcp": hierarchy,
    "../lib/methodDisplay.js": methodDisplay,
    "../lib/feedbackLabels": feedback, "../smartFilter": smartFilter,
    "../lib/tuningApi": tuningApi,
    "./TopGallery": { GalleryThumbnail: () => null, PreferenceControls: () => null },
    "./GalleryLightbox": { GalleryLightbox: () => null },
  };
  const compiled = ts.transpileModule(text, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022,
  } }).outputText;
  const mod = { exports: {} };
  new Function("require", "module", "exports", compiled)((id) => {
    assert.ok(id in imports, id);
    return imports[id];
  }, mod, mod.exports);
  return mod.exports[name];
}
const render = (Component, props) => renderToStaticMarkup(React.createElement(Component, props));
const nodes = (tree) => !tree || typeof tree !== "object" ? []
  : Array.isArray(tree) ? tree.flatMap(nodes) : [tree, ...nodes(tree.props?.children)];

test("PCP legend starts collapsed, preserves every cluster and retains selected status", async () => {
  const Component = await component("ClusterSummaryParallelCoordinates");
  const html = render(Component, {
    methods: ["a", "b"], enabledMethods: ["a", "b"], values: new Float32Array([.1, .2, .4, .5, .8, .9]),
    labels: new Uint8Array([0, 1, 2]), selectedCluster: 1, onClusterClick: () => {},
    title: "Hierarchical Evidence",
  });
  assert.match(html, /<details class="cluster-disclosure pcp-cluster-disclosure">/);
  assert.doesNotMatch(html, /<details[^>]*\bopen[= >]/);
  assert.equal((html.match(/aria-pressed=/g) ?? []).length, 3);
  assert.match(html, /data-cluster-selected="true"/);
  const text = await source("app/components/ClusterSummaryParallelCoordinates.tsx");
  assert.match(text, /onToggle=\{\(event\) => setLegendExpanded\(event.currentTarget.open\)\}/);
  assert.match(text, /\[legendExpanded, selectedKey, selectedPosition\]/);
});

test("visual legend disclosure keeps all labels and filters; compact scatter keeps selection", async () => {
  const dashboard = await source("app/Dashboard.tsx");
  assert.match(dashboard, /<details key=\{selectedTaskId\} className="cluster-disclosure visual-cluster-disclosure">/);
  assert.match(dashboard, /visualClusterSummaries\.map\(\(summary\) =>/);
  assert.match(dashboard, /onClick=\{\(\) => toggleVisualClusterSummary\(summary.cluster_id\)\}/);
  const Scatter = await component("ProjectionScatter");
  const html = render(Scatter, { points: [], projection: "umap", showHeading: false, height: 250 });
  assert.doesNotMatch(html, /<h3|class="projection-help"/);
  assert.match(html, /height:250px/);
  assert.match(html, /Reset selection/);
  assert.match(html, /UMAP projection of 0 images\. Drag a rectangle to select points/);
  assert.match(dashboard, /onBoxSelect=\{handleProjectionSelection\}/);
});

test("all four diagnostics stay actionable and rescue selector is outside the row viewport", async () => {
  const Panel = await component("SmartFilterPanel");
  const props = {
    activeKind: null, learnerMethod: "MLP", learnerMethods: ["MLP", "K-Fold"], targetLabel: "Joint",
    comparisonMethod: "F0", sourceLabel: "F0", sourceDetail: "Applied snapshot only",
    prototypeComparisonAvailable: true, result: null, onKindChange: () => {}, onLearnerMethodChange: () => {},
  };
  const html = render(Panel, props);
  assert.equal((html.match(/aria-pressed="false"/g) ?? []).length, 4);
  assert.match(html, /class="smart-filter-scroll"[^>]*tabindex="0"/);
  for (const title of ["SoftGate Boundary", "Probe Disagreement", "Individual-Probe Rescue", "Prototype–Ranking Mismatch"]) {
    assert.ok(html.includes(title), title);
  }
  const rescue = render(Panel, { ...props, activeKind: "learner-fusion-gap" });
  assert.match(rescue, /<\/div><\/div><label class="smart-filter-learner">/);
  assert.match(rescue, /aria-label="Probe for Individual-Probe Rescue"/);
  const unavailable = render(Panel, { ...props, unavailableReason: "Snapshot missing" });
  assert.equal((unavailable.match(/disabled=""/g) ?? []).length, 4);
});

test("Clear all is a feedback-header sibling, counts all labels and preserves disable guards", async () => {
  const Review = await component("AnnotationReviewGallery");
  const annotations = new Map([["one", { imageId: "one", label: 1 }], ["unavailable", { imageId: "unavailable", label: 0 }]]);
  const props = { annotations, itemById: new Map(), preferences: new Map(), onClearAll: async () => {} };
  const html = render(Review, props);
  assert.match(html, /aria-label="Clear all 2 feedback labels"/);
  assert.match(html, /<\/button><button[^>]*class="text-button annotation-review-clear"/);
  assert.doesNotMatch(html, /<button[^>]*>[\s\S]*?<button[^>]*>[\s\S]*?<\/button>[\s\S]*?<\/button>/);
  for (const guards of [{ busy: true }, { clearDisabled: true }]) {
    assert.match(render(Review, { ...props, ...guards }), /aria-label="Clear all 2 feedback labels" disabled=""/);
  }
  assert.doesNotMatch(render(Review, { ...props, annotations: new Map() }), /Clear all/);
  let calls = 0;
  const hooks = { ...React, useState: (initial) => [initial, () => {}], useMemo: (fn) => fn(), useRef: (value) => ({ current: value }), useEffect: () => {} };
  const TestReview = await component("AnnotationReviewGallery", hooks);
  const tree = TestReview({ ...props, onClearAll: async () => { calls += 1; } });
  nodes(tree).find((node) => node.props?.className === "text-button annotation-review-clear").props.onClick();
  assert.equal(calls, 1);
});
