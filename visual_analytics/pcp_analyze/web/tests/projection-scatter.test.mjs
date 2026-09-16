import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = () => readFile(
  new URL("../app/components/ProjectionScatter.tsx", import.meta.url),
  "utf8",
);

test("a common-scope expansion recomputes the committed rectangle independently", async () => {
  const component = await source();
  const start = component.indexOf("// A genuine scope change");
  const end = component.indexOf("lastEmittedRowsRef.current = new Set", start);
  assert.notEqual(start, -1, "upstream candidate reconciliation block must exist");
  assert.ok(end > start, "upstream candidate reconciliation block must emit its result");

  const reconciliation = component.slice(start, end);
  const selectionCallStart = reconciliation.indexOf("selectionForRect(");
  const selectionCallEnd = reconciliation.indexOf("    );", selectionCallStart);
  assert.notEqual(selectionCallStart, -1);
  assert.ok(selectionCallEnd > selectionCallStart);
  const selectionCall = reconciliation.slice(selectionCallStart, selectionCallEnd + 6);
  assert.match(reconciliation, /selectionBaseMaskRef\.current = nextMask/);
  assert.match(
    selectionCall,
    /selectionForRect\([\s\S]*?projection,[\s\S]*?nextMask,?\s*\)/,
  );
  assert.doesNotMatch(
    selectionCall,
    /lastEmittedRowsRef\.current/,
    "newly eligible points inside the committed rectangle must not be restricted to the old selection",
  );
});

test("a second embedding drag replaces the selection against the common-scope base", async () => {
  const component = await source();
  const start = component.indexOf("const handlePointerDown");
  const end = component.indexOf("const handlePointerCancel", start);
  assert.notEqual(start, -1, "pointer interaction block must exist");
  assert.ok(end > start, "pointer interaction block must be complete");

  const pointerInteraction = component.slice(start, end);
  assert.match(
    pointerInteraction,
    /if \(!selectionActiveRef\.current\) \{[\s\S]*?selectionBaseMaskRef\.current = snapshotCandidateMask\(boxSelectionMask\)/,
  );
  assert.match(
    pointerInteraction,
    /const interactionMask = selectionActiveRef\.current[\s\S]*?\? selectionBaseMaskRef\.current[\s\S]*?: boxSelectionMask/,
  );
  assert.match(
    pointerInteraction,
    /selectionForRect\([\s\S]*?interactionMask,[\s\S]*?\)[\s\S]*?lastEmittedRowsRef\.current = new Set\(selection\.rowIndices\)[\s\S]*?onBoxSelect\?\.\(selection\)/,
  );
});

test("PCP highlight is presentation-only while box selection uses the independent scope", async () => {
  const component = await source();
  const drawStart = component.indexOf("const dense = layout.validCount > 10_000");
  const drawEnd = component.indexOf("context.restore();", drawStart);
  assert.notEqual(drawStart, -1, "point drawing block must exist");
  assert.ok(drawEnd > drawStart, "point drawing block must be complete");
  const drawing = component.slice(drawStart, drawEnd);

  assert.match(component, /highlightMask\?: CandidateMask/);
  assert.match(drawing, /const hasHighlight = highlightMask != null/);
  assert.match(drawing, /const presentationMask = hasHighlight \? highlightMask : candidateMask/);
  assert.match(
    drawing,
    /colorMode === "neutral"[\s\S]*?"#aeb5bf"/,
    "PCP-unselected points must use one neutral color instead of cluster colors",
  );
  assert.match(
    drawing,
    /drawMaskPass\(false, dense \? 0\.075 : 0\.13, pointRadius, "neutral"\)/,
  );
  assert.match(
    drawing,
    /drawMaskPass\(true, dense \? 0\.86 : 0\.94, pointRadius \* 1\.32, "cluster"\)/,
  );
  assert.match(
    drawing,
    /highlightCount <= 2_000[\s\S]*?pointRadius \* 1\.8, "halo"/,
  );
  assert.match(
    drawing,
    /layout\.colors\.get\(cluster\) \|\| DEFAULT_CLUSTER_COLORS\[0\]/,
    "highlighted points must retain their embedding-cluster color",
  );
  assert.match(
    drawing,
    /const passes = candidateMask \? \[false, true\] : \[true\][\s\S]*?0\.075/,
    "without a highlight, the existing candidate-mask rendering must remain",
  );

  const interactionStart = component.indexOf("const handlePointerDown");
  const interactionEnd = component.indexOf("const handlePointerCancel", interactionStart);
  const pointerInteraction = component.slice(interactionStart, interactionEnd);
  assert.match(component, /boxSelectionMask\?: CandidateMask/);
  assert.match(pointerInteraction, /interactionMask[\s\S]*?: boxSelectionMask/);
  assert.doesNotMatch(pointerInteraction, /highlightMask/);
  assert.match(component, /const interactionMask = candidateMask;[\s\S]*?const column =/);
  assert.match(component, /const cluster = String\(point\.cluster\)/);
  assert.match(component, /<span>PCP selected <\/span>[\s\S]*?highlightCount\.toLocaleString\(\)/);
});

test("human selections are redrawn as a distinct top-most scatter layer", async () => {
  const component = await source();

  assert.match(
    component,
    /selectedRowIndices\?: SelectionCollection<number>/,
    "callers may identify reviewed points by original row index",
  );
  assert.match(
    component,
    /selectedIds\?: SelectionCollection<string>/,
    "callers may alternatively identify reviewed points by image ID",
  );
  assert.match(
    component,
    /manualHighlights\?: readonly ProjectionManualHighlight\[\]/,
    "label-aware feedback is accepted without changing the point model",
  );
  assert.match(
    component,
    /hasFeedbackLabel[\s\S]*?selectionIncludes\(selectedRowIndices, point\.rowIndex\)[\s\S]*?\|\| selectionIncludes\(selectedIds, point\.id\)/,
    "row and ID selections must combine as a union",
  );

  const overlayStart = component.indexOf("const drawRect =");
  const overlayEnd = component.indexOf("const nearestPoint", overlayStart);
  assert.notEqual(overlayStart, -1);
  assert.ok(overlayEnd > overlayStart);
  const overlay = component.slice(overlayStart, overlayEnd);
  assert.match(overlay, /if \(committedRect\) drawRect/);
  assert.match(overlay, /const drawManualSelectionLayer/);
  assert.match(
    overlay,
    /drawManualSelectionLayer\(manualHighlightEntries\)[\s\S]*?if \(selectedId\)[\s\S]*?if \(hoverIndex !== null\)/,
    "manual selection must sit above the brush while preserving transient focus",
  );
  assert.match(overlay, /label > 0[\s\S]*?color: "#16a34a"/);
  assert.match(overlay, /label < 0[\s\S]*?color: "#dc2626"/);
  assert.match(overlay, /color: "#ca8a04"/);
  assert.match(overlay, /style\.strong \? 3\.5 : 2\.5/);

  const interactionStart = component.indexOf("const nearestPoint");
  const interactionEnd = component.indexOf("const updateHover", interactionStart);
  const interaction = component.slice(interactionStart, interactionEnd);
  assert.doesNotMatch(
    interaction,
    /selectedRowIndices|selectedIds|manualHighlights|manualHighlightEntries/,
    "manual highlight must not alter point hit-testing",
  );
});
