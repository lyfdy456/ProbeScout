import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { manualHighlightScope } from "../app/lib/dashboardState.ts";

const readDashboard = () => readFile(
  new URL("../app/Dashboard.tsx", import.meta.url),
  "utf8",
);

function componentCall(contents, name) {
  const start = contents.indexOf(`<${name}`);
  assert.notEqual(start, -1, `${name} must be rendered`);
  const end = contents.indexOf("/>", start);
  assert.notEqual(end, -1, `${name} must have a closing tag`);
  return contents.slice(start, end + 2);
}

test("manual highlight scope separates visible and out-of-range annotations", () => {
  const highlights = [
    { rowIndex: 1, label: 2 },
    { rowIndex: 3, label: -1 },
    { rowIndex: 5, label: 1 },
  ];
  const mask = Uint8Array.from([0, 1, 0, 0, 0, 1]);

  const result = manualHighlightScope(highlights, mask);

  assert.deepEqual(result.visible, [highlights[0], highlights[2]]);
  assert.equal(result.hiddenCount, 1);
  assert.deepEqual(
    highlights.map(({ rowIndex }) => rowIndex),
    [1, 3, 5],
    "scope calculation must not mutate the annotation snapshot",
  );
});

test("manual highlight scope handles empty and fully hidden selections", () => {
  assert.deepEqual(manualHighlightScope([], new Uint8Array(3)), {
    visible: [],
    hiddenCount: 0,
  });

  const hidden = [{ rowIndex: 4, label: -2 }];
  assert.deepEqual(manualHighlightScope(hidden, new Uint8Array(5)), {
    visible: [],
    hiddenCount: 1,
  });
});

test("manual highlighting is opt-in and controlled by an accessible pressed button", async () => {
  const dashboard = await readDashboard();

  assert.match(
    dashboard,
    /const \[showManualHighlights, setShowManualHighlights\] = useState\(false\)/,
    "manual highlights must be off on first load",
  );
  assert.match(
    dashboard,
    /<button[\s\S]*?aria-pressed=\{showManualHighlights\}[\s\S]*?setShowManualHighlights\(\(current\) => !current\)[\s\S]*?>[\s\S]*?Highlight selected[\s\S]*?<\/button>/,
    "the toggle must expose its state to keyboard and assistive-technology users",
  );
});

test("all three analysis views receive only enabled highlights inside the current PCP range", async () => {
  const dashboard = await readDashboard();
  const samplePcp = componentCall(dashboard, "ParallelCoordinates");
  const clusterPcp = componentCall(dashboard, "ClusterSummaryParallelCoordinates");
  const projection = componentCall(dashboard, "ProjectionScatter");

  assert.match(
    dashboard,
    /manualHighlightScope\(manualHighlights, pcpEligibleMask\)/,
    "visibility must follow the current PCP brush, structural, and zoom scope",
  );
  assert.match(
    dashboard,
    /const visibleManualHighlights = useMemo\([\s\S]*?showManualHighlights \? scopedManualHighlights\.visible : \[\]/,
    "turning highlighting off must supply an empty selection",
  );
  assert.match(dashboard, /const hiddenManualHighlightCount = [^;]+\.hiddenCount/);

  for (const call of [samplePcp, clusterPcp, projection]) {
    assert.match(call, /manualHighlights=\{visibleManualHighlights\}/);
    assert.doesNotMatch(call, /manualHighlights=\{manualHighlights\}/);
  }
});

test("the PCP reports visible and out-of-range selection counts only while highlighting", async () => {
  const dashboard = await readDashboard();

  assert.match(
    dashboard,
    /showManualHighlights && manualHighlights\.length > 0[\s\S]*?scopedManualHighlights\.visible\.length[\s\S]*?hiddenManualHighlightCount/,
  );
  assert.match(dashboard, /aria-live="polite"/);
  assert.match(dashboard, /outside (?:the )?current (?:PCP )?range/i);
});
