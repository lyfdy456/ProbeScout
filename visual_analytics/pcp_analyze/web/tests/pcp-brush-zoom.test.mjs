import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  applyPcpBrushZoomMask,
  createPcpBrushZoomState,
  drillIntoPcpBrushZoom,
  formatPcpAxisTick,
  hasPcpBrushZoom,
  pcpBrushZoomDepth,
  pcpBrushZoomFingerprint,
  resetPcpBrushZoom,
  stepBackPcpBrushZoom,
} from "../app/lib/pcpBrushZoom.ts";

test("both PCP renderers fit the measured pane beside the evidence rail", async () => {
  for (const file of ["ParallelCoordinates.tsx", "ClusterSummaryParallelCoordinates.tsx"]) {
    const source = await readFile(new URL(`../app/components/${file}`, import.meta.url), "utf8");
    assert.match(source, /Math\.max\(1, Math\.round\(\w+\.getBoundingClientRect\(\)\.width/);
    assert.doesNotMatch(source, /Math\.max\(280, Math\.round/);
  }
});

test("PCP brush zoom commits absolute ranges and supports nested back navigation", () => {
  const initial = createPcpBrushZoomState();
  const first = drillIntoPcpBrushZoom(initial, { full: [0.9, 1] }, ["full", "cat"]);
  assert.notEqual(first, initial);
  assert.deepEqual(first.ranges, { full: [0.9, 1] });
  assert.equal(pcpBrushZoomDepth(first), 1);
  assert.equal(hasPcpBrushZoom(first), true);

  const second = drillIntoPcpBrushZoom(
    first,
    { full: [0.99, 1], cat: [0.4, 0.6] },
    ["full", "cat"],
  );
  assert.deepEqual(second.ranges, {
    full: [0.99, 1],
    cat: [0.4, 0.6],
  });
  assert.equal(pcpBrushZoomDepth(second), 2);
  assert.deepEqual(first.ranges, { full: [0.9, 1] }, "inputs stay immutable");

  const backOne = stepBackPcpBrushZoom(second);
  assert.deepEqual(backOne.ranges, first.ranges);
  assert.equal(pcpBrushZoomDepth(backOne), 1);
  const backToRoot = stepBackPcpBrushZoom(backOne);
  assert.deepEqual(backToRoot, createPcpBrushZoomState());
  assert.strictEqual(stepBackPcpBrushZoom(backToRoot), backToRoot);
  assert.deepEqual(resetPcpBrushZoom(), createPcpBrushZoomState());
});

test("PCP brush zoom ignores empty/full-width brushes and clips reversed ranges", () => {
  const initial = createPcpBrushZoomState();
  assert.strictEqual(
    drillIntoPcpBrushZoom(initial, {}, ["full"]),
    initial,
  );
  assert.strictEqual(
    drillIntoPcpBrushZoom(initial, { full: [0, 1] }, ["full"]),
    initial,
  );
  assert.strictEqual(
    drillIntoPcpBrushZoom(initial, { full: [Number.NaN, 1] }, ["full"]),
    initial,
  );
  assert.strictEqual(
    drillIntoPcpBrushZoom(initial, { full: [0.5, 0.5] }, ["full"]),
    initial,
  );
  assert.strictEqual(
    drillIntoPcpBrushZoom(initial, { full: [0.5, Number.POSITIVE_INFINITY] }, ["full"]),
    initial,
  );
  assert.strictEqual(
    drillIntoPcpBrushZoom(initial, { missing: [0.8, 1] }, ["full"]),
    initial,
  );
  assert.deepEqual(
    drillIntoPcpBrushZoom(initial, { full: [1.2, 0.8] }, ["full"]).ranges,
    { full: [0.8, 1] },
  );

  const parent = drillIntoPcpBrushZoom(initial, { full: [0.8, 1] }, ["full"]);
  assert.strictEqual(
    drillIntoPcpBrushZoom(parent, { full: [0.7, 1] }, ["full"]),
    parent,
    "a child brush that covers the parent does not create a redundant level",
  );
});

test("committed PCP ranges remain an AND-filtered candidate scope", () => {
  const values = new Float32Array([
    0.95, 0.50,
    0.995, 0.55,
    0.999, 0.80,
    0.70, 0.50,
  ]);
  const base = new Uint8Array([1, 1, 1, 0]);
  const first = applyPcpBrushZoomMask({
    values,
    axisIds: ["full", "cat"],
    candidateMask: base,
    ranges: { full: [0.9, 1] },
  });
  assert.deepEqual([...first], [1, 1, 1, 0]);

  const nested = applyPcpBrushZoomMask({
    values,
    axisIds: ["full", "cat"],
    candidateMask: base,
    ranges: { full: [0.99, 1], cat: [0.4, 0.6] },
  });
  assert.deepEqual([...nested], [0, 1, 0, 0]);
  assert.deepEqual([...base], [1, 1, 1, 0], "the upstream mask stays immutable");

  assert.throws(() => applyPcpBrushZoomMask({
    values,
    axisIds: ["full"],
    candidateMask: base,
    ranges: { full: [0.9, 1] },
  }), /not row-aligned/);

  const boundaries = applyPcpBrushZoomMask({
    values: new Float32Array([0.5, 1, Number.NaN]),
    axisIds: ["full"],
    candidateMask: new Uint8Array([1, 1, 1]),
    ranges: { full: [0.5, 1] },
  });
  assert.deepEqual([...boundaries], [1, 1, 0], "range endpoints are inclusive and NaN is rejected");
});

test("deep PCP zoom tick labels stay distinct and fingerprints follow axis order", () => {
  assert.notEqual(
    formatPcpAxisTick(0.99, [0.99, 1]),
    formatPcpAxisTick(1, [0.99, 1]),
  );
  assert.notEqual(
    formatPcpAxisTick(0.99999, [0.99999, 1]),
    formatPcpAxisTick(1, [0.99999, 1]),
  );
  const ranges = { cat: [0.4, 0.6], full: [0.99, 1] };
  assert.notEqual(
    pcpBrushZoomFingerprint(ranges, ["full", "cat"]),
    pcpBrushZoomFingerprint(ranges, ["cat", "full"]),
  );
});

test("sample PCP exposes per-axis zoom scales and compact drill controls", async () => {
  const [component, dashboard, styles] = await Promise.all([
    readFile(new URL("../app/components/ParallelCoordinates.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(component, /axisDomains\?: PcpBrushMap/);
  assert.ok(
    component.match(/domainsByMethod\.get\(method\) \?\? domain/g)?.length >= 4,
    "canvas, SVG axes, initial brush move and controlled brush move all use per-axis domains",
  );
  assert.match(component, /const scaleXs = methods\.map/);
  assert.match(component, /colorLearnerDomain/);
  assert.match(component, /scaleX\.invert\(pixelRange\[0\]\)[\s\S]*?axisDomain/);
  assert.match(component, /brush from \$\{axisDomain\[0\]\} to \$\{axisDomain\[1\]\}/);
  assert.match(component, /const axisDomain = domainsByMethod\.get\(method\)[\s\S]*?\.domain\(axisDomain\)[\s\S]*?brush\.move/);
  assert.match(component, /zoomedMethodSet\.has\(method\)/);
  assert.match(component, /formatPcpAxisTick\(Number\(value\), axisDomain\)/);
  assert.match(component, />\s*Zoom\s*</);
  assert.match(component, />\s*Back\s*</);
  assert.match(component, />\s*Reset\s*</);
  assert.match(styles, /\.pcp-brush-action\s*\{[^}]*pointer-events:\s*auto;/s);
  assert.match(styles, /@media[\s\S]*?\.pcp-brush-action,[\s\S]*?min-height:\s*32px;/);

  assert.match(dashboard, /applyPcpBrushZoomMask\(\{/);
  assert.match(dashboard, /candidateMask=\{pcpCandidateMask\}/);
  assert.match(dashboard, /axisDomains=\{pcpBrushZoom\.ranges\}/);
  assert.match(dashboard, /onBrushZoom=\{zoomCurrentPcpBrush\}/);
  assert.match(dashboard, /const hasPcpRestriction = effectivePcpRanges\.length > 0/);
  assert.match(dashboard, /const pcpIndependentMask = useMemo/);
  const zoomReset = dashboard.slice(
    dashboard.indexOf("const clearPcpBrushZoom"),
    dashboard.indexOf("const clearRankClusterInteraction"),
  );
  assert.doesNotMatch(zoomReset, /setRankClusterFilter|setClusterDrilldown/);
  const globalReset = dashboard.slice(
    dashboard.indexOf("const clearFilters"),
    dashboard.indexOf("const changeResultScope"),
  );
  assert.match(globalReset, /clearPcpSelection\(\)/);
});
