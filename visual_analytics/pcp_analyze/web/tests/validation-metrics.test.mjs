import assert from "node:assert/strict";
import test from "node:test";

import {
  evaluateRankedScope,
  evaluateTruePositivesAtKs,
} from "../app/lib/validationMetrics.ts";

function evaluate(overrides = {}) {
  return evaluateRankedScope({
    ranks: Float32Array.from([0.9, 0.8, 0.7, 0.6, 0.5]),
    groundTruth: Uint8Array.from([
      0, 1,
      0, 0,
      0, 1,
      0, 0,
      0, 1,
    ]),
    scopeMask: Uint8Array.from([1, 1, 1, 1, 0]),
    rowCount: 5,
    targetIndex: 1,
    targetCount: 2,
    k: 2,
    ...overrides,
  });
}

test("validation metrics compute AP, best F1, P@K and Recall@K inside the scope", () => {
  const metrics = evaluate();
  assert.deepEqual(metrics, {
    averagePrecision: (1 + 2 / 3) / 2,
    bestF1: 0.8,
    bestCutoff: 3,
    precisionAtK: 0.5,
    recallAtK: 0.5,
    positiveCount: 2,
    evaluatedCount: 4,
    truePositiveAtK: 1,
  });
});

test("rank ties use stable row order and target selection follows row-major layout", () => {
  const metrics = evaluate({
    ranks: Float32Array.from([0.5, 0.5, 0.5]),
    groundTruth: Uint8Array.from([
      1, 0,
      0, 1,
      1, 0,
    ]),
    scopeMask: Uint8Array.from([1, 1, 1]),
    rowCount: 3,
    targetIndex: 0,
    targetCount: 2,
    k: 2,
  });
  assert.equal(metrics.averagePrecision, (1 + 2 / 3) / 2);
  assert.equal(metrics.truePositiveAtK, 1);
  assert.equal(metrics.precisionAtK, 0.5);
  assert.equal(metrics.bestCutoff, 3);
});

test("empty scopes and scopes without positives return stable zero metrics", () => {
  assert.deepEqual(evaluate({ scopeMask: new Uint8Array(5) }), {
    averagePrecision: 0,
    bestF1: 0,
    bestCutoff: 0,
    precisionAtK: 0,
    recallAtK: 0,
    positiveCount: 0,
    evaluatedCount: 0,
    truePositiveAtK: 0,
  });

  const noPositives = evaluate({
    groundTruth: new Uint8Array(10),
    k: 50,
  });
  assert.equal(noPositives.averagePrecision, 0);
  assert.equal(noPositives.bestF1, 0);
  assert.equal(noPositives.bestCutoff, 0);
  assert.equal(noPositives.precisionAtK, 0);
  assert.equal(noPositives.recallAtK, 0);
  assert.equal(noPositives.evaluatedCount, 4);
});

test("K is clamped to the scoped population and non-finite ranks sort deterministically", () => {
  const metrics = evaluate({
    ranks: Float32Array.from([Number.NaN, Number.POSITIVE_INFINITY, 0.8, Number.NEGATIVE_INFINITY, 0.2]),
    scopeMask: Uint8Array.from([1, 1, 1, 1, 0]),
    k: 100,
  });
  assert.equal(metrics.evaluatedCount, 4);
  assert.equal(metrics.truePositiveAtK, 2);
  assert.equal(metrics.precisionAtK, 0.5);
  assert.equal(metrics.recallAtK, 1);
});

test("invalid row-target and mask contracts fail explicitly", () => {
  assert.throws(() => evaluate({ rowCount: 4 }), /align|row-target/);
  assert.throws(() => evaluate({ targetIndex: 2 }), /target index/);
  assert.throws(() => evaluate({ scopeMask: Uint8Array.from([2, 0, 0, 0, 0]) }), /binary/);
});

test("multi-cutoff TP audit sorts once, honors the split mask, and clamps large K", () => {
  const rowCount = 245;
  const ranks = new Float32Array(rowCount);
  const reranked = new Float32Array(rowCount);
  const groundTruth = new Uint8Array(rowCount);
  const scopeMask = new Uint8Array(rowCount);
  const positivePositions = new Set([
    1, 5, 9, 13, 17, 21, 25, 29,
    31, 35, 39, 43, 47,
    ...Array.from({ length: 12 }, (_, index) => 51 + index * 4),
    ...Array.from({ length: 25 }, (_, index) => 101 + index * 4),
    ...Array.from({ length: 10 }, (_, index) => 201 + index),
  ]);

  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const inScope = rowIndex < 240;
    const positive = positivePositions.has(rowIndex + 1) || !inScope;
    scopeMask[rowIndex] = inScope ? 1 : 0;
    groundTruth[rowIndex] = positive ? 1 : 0;
    ranks[rowIndex] = inScope ? 1 - rowIndex / 1000 : 100;
    reranked[rowIndex] = inScope
      ? (positive ? 2 - rowIndex / 1000 : 1 - rowIndex / 1000)
      : 100;
  }

  const shared = {
    groundTruth,
    scopeMask,
    rowCount,
    targetIndex: 0,
    targetCount: 1,
    ks: [30, 50, 100, 200, 300],
  };
  const baseline = evaluateTruePositivesAtKs({ ...shared, ranks });
  const after = evaluateTruePositivesAtKs({ ...shared, ranks: reranked });

  assert.deepEqual(baseline, {
    positiveCount: 60,
    evaluatedCount: 240,
    truePositiveAtK: { 30: 8, 50: 13, 100: 25, 200: 50, 300: 60 },
  });
  assert.deepEqual(after.truePositiveAtK, { 30: 30, 50: 50, 100: 60, 200: 60, 300: 60 });
  assert.throws(
    () => evaluateTruePositivesAtKs({ ...shared, ranks, ks: [30, -1] }),
    /cutoffs/,
  );
});
