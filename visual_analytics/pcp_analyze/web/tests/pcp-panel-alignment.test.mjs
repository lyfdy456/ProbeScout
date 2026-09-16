import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { pcpPanelDimensions } from "../app/lib/usePcpPanelAlignment.ts";

test("PCP frame reaches natural exploration bottom with a matching drawable height", () => {
  assert.deepEqual(pcpPanelDimensions(true, 1590, 500, 2, 260), {
    frameHeight: 1090, plotHeight: 1088,
  });
});

test("summary controls consume frame space without shortening the frame", () => {
  assert.deepEqual(pcpPanelDimensions(true, 1590, 500, 188, 260), {
    frameHeight: 1090, plotHeight: 902,
  });
});

test("many expanded axes retain readable geometry and scroll inside the fixed frame", () => {
  assert.deepEqual(pcpPanelDimensions(true, 1590, 500, 2, 1700), {
    frameHeight: 1090, plotHeight: 1700,
  });
});

test("stacked mobile layout and short/empty galleries keep a bounded readable PCP", () => {
  assert.deepEqual(pcpPanelDimensions(false, 9000, 500, 2, 260), {
    frameHeight: 715, plotHeight: 713,
  });
  assert.deepEqual(pcpPanelDimensions(true, 690, 500, 2, 260), {
    frameHeight: 420, plotHeight: 418,
  });
});

test("compact desktop exploration no longer forces a 715px PCP frame", () => {
  assert.deepEqual(pcpPanelDimensions(true, 1050, 500, 80, 260), {
    frameHeight: 550, plotHeight: 470,
  });
});

test("fractional pixel geometry rounds the outer frame and never overflows the inner plot", () => {
  assert.deepEqual(pcpPanelDimensions(true, 1500.25, 400.75, 2.25, 260), {
    frameHeight: 1100, plotHeight: 1097,
  });
});

test("height updates share rail/plot geometry and are independent of right-side refinement", async () => {
  const [dashboard, hook] = await Promise.all([
    readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/lib/usePcpPanelAlignment.ts", import.meta.url), "utf8"),
  ]);
  assert.match(dashboard, /const pcpHeight = pcpPanel\.plotHeight/);
  assert.match(dashboard, /const summaryPcpHeight = pcpPanel\.plotHeight/);
  assert.match(dashboard, /ref=\{explorationContentRef\} className="exploration-content"/);
  assert.equal([...dashboard.matchAll(/height=\{pcpHeight\}/g)].length, 2);
  assert.equal([...dashboard.matchAll(/height=\{summaryPcpHeight\}/g)].length, 2);
  assert.match(hook, /frame\.scrollTop/);
  assert.match(hook, /exploration\.getBoundingClientRect\(\)\.bottom/);
  assert.doesNotMatch(hook, /optimization-panel|setBrush|setProjection|resultMask|rank/);
  assert.match(hook, /current\.frameHeight === next\.frameHeight/);
  assert.match(hook, /observer\?\.disconnect\(\)/);
  assert.match(dashboard, /onBrushChange=\{\(next\) => \{\s*setBrushes\(\{ \.\.\.next \}\);\s*setSelectedId\(null\);\s*setHoveredId\(null\);/);
  assert.doesNotMatch(dashboard, /onSelectionChange=\{\(\) => \{\s*setSelectedId\(null\)/);
});
