import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("sample PCP exposes a candidate-scoped human-selection highlight layer", async () => {
  const source = await readFile(
    new URL("../app/components/ParallelCoordinates.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /selectedRowIndices\?: PcpSelectedRowIndices/,
    "callers can provide original matrix row indices without changing brush state",
  );
  assert.match(source, /manualHighlights\?: readonly PcpManualHighlight\[\]/);
  assert.match(source, /focusedRowIndex\?: number \| null/);
  assert.match(
    source,
    /requested\.has\(image\)[\s\S]*?!candidateMask \|\| candidateMask\[image\]/,
    "only selected rows visible in the current candidate scope are highlighted",
  );
  assert.match(source, /className="pcp-canvas pcp-canvas--selected"/);
  assert.match(source, /drawLayer\(canvas, visibleHighlightRowIndices, maxForegroundLines, "selected"\)/);
  assert.match(source, /hasVisibleManualSelection \? "muted-context" : "context"/);
  assert.match(source, /--pcp-selected-line-color/);
  assert.match(source, /--pcp-selected-line-halo/);
  assert.match(source, /--pcp-positive-line-color/);
  assert.match(source, /--pcp-negative-line-color/);
  assert.match(source, /--pcp-uncertain-line-color/);
  assert.match(source, /--pcp-focused-line-color/);
  assert.match(source, /lineWidth: 3\.15, opacity: 1, haloExtraWidth: 4\.2/);

  const foreground = source.indexOf('className="pcp-canvas pcp-canvas--foreground"');
  const selected = source.indexOf('className="pcp-canvas pcp-canvas--selected"');
  const overlay = source.indexOf('className="pcp-overlay"');
  assert.ok(foreground >= 0 && selected > foreground && overlay > selected);
});

test("cluster PCP aggregates selected members and marks both legend and curve", async () => {
  const source = await readFile(
    new URL("../app/components/ClusterSummaryParallelCoordinates.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /selectedRowIndices\?: PcpSelectedRowIndices/);
  assert.match(source, /manualHighlights\?: readonly PcpManualHighlight\[\]/);
  assert.match(source, /focusedRowIndex\?: number \| null/);
  assert.match(source, /highlightedRowSet\.has\(rowIndex\)/);
  assert.match(source, /accumulator\.selectedCount \+= 1/);
  assert.match(source, /data-manual-selected-count=/);
  assert.match(source, /stroke=\{selectedHighlightColor\}/);
  assert.match(source, /profile\.lineWidth \+ 6/);
  assert.match(source, /\{selectedCountText\}/);
  assert.match(source, /hasVisibleManualSelection[\s\S]*?emphasized \? 0\.94 : 0\.1/);
});

test("dashboard derives highlights from the private annotation session and wires every view", async () => {
  const source = await readFile(
    new URL("../app/Dashboard.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /const manualHighlights = useMemo\([\s\S]*?tuning\.annotations\.values\(\)[\s\S]*?rowIndex: annotation\.rowIndex[\s\S]*?label: annotation\.label/,
    "persistent highlights must come from this tuning session's human annotations",
  );
  assert.match(
    source,
    /const focusedRowIndex = selectedId[\s\S]*?galleryItemById\.get\(selectedId\)\?\.rowIndex/,
    "the currently opened image remains a separate transient focus",
  );
  assert.equal(
    source.match(/manualHighlights=\{visibleManualHighlights\}/g)?.length,
    3,
    "sample PCP, cluster-summary PCP, and visual embedding must share the same enabled, in-scope selection",
  );
  assert.equal(
    source.match(/focusedRowIndex=\{visibleFocusedRowIndex\}/g)?.length,
    2,
    "both PCP modes must receive only the enabled, in-scope current-image focus",
  );
});
