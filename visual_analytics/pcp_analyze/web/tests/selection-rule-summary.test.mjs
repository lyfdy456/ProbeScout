import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { summarizeSelectionRules } from "../app/lib/selectionRuleSummary.ts";

function masks(selectedIndices, baseIndices, rowCount = 10) {
  const selectionMask = new Uint8Array(rowCount);
  const baseMask = new Uint8Array(rowCount);
  for (const index of selectedIndices) selectionMask[index] = 1;
  for (const index of baseIndices) baseMask[index] = 1;
  return { selectionMask, baseMask };
}

function closeTo(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be close to ${expected}`);
}

test("selection summary uses selected-in-base rows for exact rank and score quantiles", () => {
  const { selectionMask, baseMask } = masks(
    [0, 1, 2, 3, 4, 9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
  );
  const result = summarizeSelectionRules({
    selectionMask,
    baseMask,
    attributes: [
      {
        id: "attr1",
        label: "Attr 1",
        ranks: Float32Array.from([0.91, 0.92, 0.93, 0.94, 0.95, 0, 0, 0, 0, 0.99]),
        calibratedScores: Float32Array.from([0.511, 0.563, 0.612, 0.667, 0.694, 0, 0, 0, 0, 0.99]),
      },
      {
        id: "attr2",
        label: "Attr 2",
        ranks: Float32Array.from([0.81, 0.83, 0.85, 0.87, 0.89, 0, 0, 0, 0, 0.99]),
        calibratedScores: Float32Array.from([0.311, 0.357, 0.392, 0.426, 0.483, 0, 0, 0, 0, 0.99]),
      },
    ],
  });

  assert.equal(result.kind, "summary");
  assert.equal(result.summary.selectedCount, 5);
  assert.equal(result.summary.baseCount, 9);
  const first = result.summary.attributes[0];
  closeTo(first.rank.lowerQuantile, 0.912);
  closeTo(first.rank.upperQuantile, 0.948);
  closeTo(first.calibratedScore.lowerQuantile, 0.5214);
  closeTo(first.calibratedScore.upperQuantile, 0.6886);
  closeTo(first.calibratedScore.minimum, 0.511);
  closeTo(first.calibratedScore.maximum, 0.694);
  assert.notEqual(first.calibratedScore.lowerQuantile, 0.5, "scores must not snap to a 0.05 grid");
  assert.notEqual(first.calibratedScore.upperQuantile, 0.7, "scores must not snap to a 0.05 grid");
});

test("quantiles limit isolated outliers while observed min and max remain available", () => {
  const rowCount = 22;
  const selectionMask = new Uint8Array(rowCount);
  const baseMask = new Uint8Array(rowCount);
  selectionMask.fill(1, 0, 20);
  baseMask.fill(1);
  const scores = new Float32Array(rowCount);
  scores[0] = 0.01;
  scores.fill(0.56, 1, 19);
  scores[19] = 0.99;

  const result = summarizeSelectionRules({
    selectionMask,
    baseMask,
    attributes: [{
      id: "cat",
      label: "Cat",
      ranks: Float32Array.from(scores, (value) => Math.min(1, value + 0.3)),
      calibratedScores: scores,
    }],
  });

  assert.equal(result.kind, "summary");
  const range = result.summary.attributes[0].calibratedScore;
  assert.equal(range.minimum, scores[0]);
  assert.equal(range.maximum, scores[19]);
  assert.ok(range.lowerQuantile > range.minimum);
  assert.ok(range.upperQuantile < range.maximum);
  closeTo(range.lowerQuantile, 0.5325);
  closeTo(range.upperQuantile, 0.5815);
});

test("empty, entire-base, and absent-attribute selections explicitly return no summary", () => {
  const base = [0, 1, 2];
  const scores = Float32Array.from([0.1, 0.2, 0.3, 0.4]);

  assert.deepEqual(
    summarizeSelectionRules({
      ...masks([], base, 4),
      attributes: [{ id: "a", label: "A", ranks: scores, calibratedScores: scores }],
    }),
    { kind: "none", reason: "empty-selection" },
  );
  assert.deepEqual(
    summarizeSelectionRules({
      ...masks(base, base, 4),
      attributes: [{ id: "a", label: "A", ranks: scores, calibratedScores: scores }],
    }),
    { kind: "none", reason: "entire-base-selected" },
  );
  assert.deepEqual(
    summarizeSelectionRules({ ...masks([0], base, 4), attributes: [] }),
    { kind: "none", reason: "no-attributes" },
  );
  assert.deepEqual(
    summarizeSelectionRules({
      selectionMask: new Uint8Array(3),
      baseMask: new Uint8Array(3),
      attributes: [{
        id: "a",
        label: "A",
        ranks: new Float32Array(3),
        calibratedScores: new Float32Array(3),
      }],
    }),
    { kind: "none", reason: "empty-base" },
  );
});

test("a selected non-finite rank or score produces no summary and identifies its source", () => {
  const result = summarizeSelectionRules({
    ...masks([0, 2], [0, 1, 2], 3),
    attributes: [{
      id: "joint-a",
      label: "Joint A",
      ranks: Float64Array.from([0.7, 0.8, Number.NaN]),
      calibratedScores: Float64Array.from([0.2, 0.4, 0.6]),
    }],
  });

  assert.deepEqual(result, {
    kind: "none",
    reason: "non-finite-value",
    attributeId: "joint-a",
    rowIndex: 2,
    valueKind: "rank",
  });

  assert.deepEqual(
    summarizeSelectionRules({
      ...masks([0, 2], [0, 1, 2], 3),
      attributes: [{
        id: "joint-a",
        label: "Joint A",
        ranks: Float64Array.from([0.7, 0.8, 0.9]),
        calibratedScores: Float64Array.from([0.2, 0.4, Number.POSITIVE_INFINITY]),
      }],
    }),
    {
      kind: "none",
      reason: "non-finite-value",
      attributeId: "joint-a",
      rowIndex: 2,
      valueKind: "calibrated-score",
    },
  );
});

test("mask/configuration and score alignment are validated", () => {
  assert.throws(
    () => summarizeSelectionRules({
      selectionMask: Uint8Array.from([1, 2]),
      baseMask: Uint8Array.from([1, 1]),
      attributes: [],
    }),
    /binary/,
  );
  assert.throws(
    () => summarizeSelectionRules({
      ...masks([0], [0, 1], 2),
      attributes: [{
        id: "a",
        label: "A",
        ranks: new Float32Array(2),
        calibratedScores: new Float32Array(1),
      }],
    }),
    /row-aligned/,
  );
  assert.throws(
    () => summarizeSelectionRules({
      ...masks([0], [0, 1], 2),
      quantileRange: [0.95, 0.05],
      attributes: [{
        id: "a",
        label: "A",
        ranks: new Float32Array(2),
        calibratedScores: new Float32Array(2),
      }],
    }),
    /ordered/,
  );
});

test("selection panel renders rank and ungridded three-decimal calibrated score together", async () => {
  const [dashboardSource, panelSource, cssSource] = await Promise.all([
    readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/SelectionRuleSummaryPanel.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  assert.match(dashboardSource, /<span>Selected Image Analysis<\/span>/);
  assert.match(panelSource, /Rank <small>1 = best<\/small>/);
  assert.match(panelSource, /Score <small>calibrated<\/small>/);
  assert.match(panelSource, /calibratedScore\.lowerQuantile\.toFixed\(3\)/);
  assert.match(panelSource, /calibratedScore\.upperQuantile\.toFixed\(3\)/);
  assert.doesNotMatch(panelSource, /lowerBound|upperBound|toFixed\(2\)/);
  assert.match(cssSource, /\.selection-rule-table[\s\S]*?table-layout:\s*fixed/);
});
