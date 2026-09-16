import assert from "node:assert/strict";
import test from "node:test";

import {
  buildSmartFilterMask,
  SMART_FILTER_LIMIT,
} from "../app/smartFilter.ts";

function createSource(rowCount) {
  const methodCount = 10;
  const targetCount = 2;
  const rawScores = new Float32Array(rowCount * methodCount * targetCount);
  const ranks = new Float32Array(rowCount * methodCount * targetCount);
  const candidateMask = new Uint8Array(rowCount);
  candidateMask.fill(1);
  const offset = (row, method, target) => (
    (row * methodCount + method) * targetCount + target
  );
  return {
    source: {
      rowCount,
      methodCount,
      targetCount,
      targetIndex: 1,
      targetMemberIndices: [0, 1],
      learnerMethodIndices: [0, 1, 2, 3, 4, 5, 6, 7],
      fusionMethodIndex: 8,
      prototypeMethodIndex: 9,
      selectedLearnerIndex: 0,
      comparisonMethodIndex: 8,
      rawScores,
      ranks,
      candidateMask,
    },
    setRaw(row, method, target, value) {
      rawScores[offset(row, method, target)] = value;
    },
    setRank(row, method, target, value) {
      ranks[offset(row, method, target)] = value;
    },
  };
}

test("SoftGate boundary uses the nearest member gate and keeps exactly Top 50", () => {
  const fixture = createSource(60);
  for (let row = 0; row < 60; row += 1) {
    fixture.setRaw(row, 8, 0, 0.5 + row / 1000);
    fixture.setRaw(row, 8, 1, 0.9);
  }

  const result = buildSmartFilterMask("softgate-boundary", fixture.source);

  assert.equal(result.selectedCount, SMART_FILTER_LIMIT);
  assert.equal(result.eligibleCount, 60);
  assert.deepEqual(result.rowIndices, Array.from({ length: 50 }, (_, index) => index));
  assert.ok(Math.abs(result.cutoff - 0.049) < 1e-6);
  assert.equal(result.mask[49], 1);
  assert.equal(result.mask[50], 0);
});

test("learner disagreement is population rank standard deviation within candidates", () => {
  const fixture = createSource(60);
  for (let row = 0; row < 60; row += 1) {
    const deviation = row / 200;
    for (let method = 0; method < 8; method += 1) {
      fixture.setRank(row, method, 1, method < 4 ? 0.5 - deviation : 0.5 + deviation);
    }
  }
  fixture.source.candidateMask[59] = 0;

  const result = buildSmartFilterMask("learner-disagreement", fixture.source);

  assert.equal(result.candidateCount, 59);
  assert.equal(result.selectedCount, 50);
  assert.deepEqual(result.rowIndices.slice(0, 3), [58, 57, 56]);
  assert.equal(result.rowIndices.at(-1), 9);
});

test("learner-over-fusion applies inclusive rank limits before gap Top 50", () => {
  const fixture = createSource(6);
  const values = [
    [0.95, 0.40],
    [0.90, 0.50],
    [0.89, 0.10],
    [0.99, 0.55],
    [0.95, 0.50],
    [0.96, 0.41],
  ];
  values.forEach(([learner, fusion], row) => {
    fixture.setRank(row, 0, 1, learner);
    fixture.setRank(row, 8, 1, fusion);
  });

  const result = buildSmartFilterMask("learner-fusion-gap", fixture.source);

  assert.equal(result.eligibleCount, 4);
  assert.deepEqual(result.rowIndices, [5, 0, 4, 1]);
  assert.deepEqual([...result.mask], [1, 1, 0, 0, 1, 1]);
});

test("prototype-over-current compares Image Prototype with the selected rank method", () => {
  const fixture = createSource(5);
  const values = [
    [0.95, 0.20],
    [0.90, 0.50],
    [0.89, 0.10],
    [0.99, 0.55],
    [0.95, 0.20],
  ];
  values.forEach(([prototype, comparison], row) => {
    fixture.setRank(row, 9, 1, prototype);
    fixture.setRank(row, 8, 1, comparison);
  });
  fixture.source.candidateMask[0] = 0;

  const result = buildSmartFilterMask("prototype-rank-gap", fixture.source);

  assert.equal(result.candidateCount, 4);
  assert.equal(result.eligibleCount, 2);
  assert.deepEqual(result.rowIndices, [4, 1]);
  assert.deepEqual([...result.mask], [0, 1, 0, 0, 1]);
});

test("smart filters reject a learner contract other than the audited eight methods", () => {
  const fixture = createSource(1);
  fixture.source.learnerMethodIndices = [0, 1];

  assert.throws(
    () => buildSmartFilterMask("learner-disagreement", fixture.source),
    /method or target indices/,
  );
});

function liveValues(rowCount, gateCount = 1) {
  return {
    gateCount,
    gateScores: new Float32Array(rowCount * gateCount).fill(0.8),
    learnerRanks: new Float32Array(rowCount * 8).fill(0.5),
    fusionRanks: new Float32Array(rowCount).fill(0.5),
    comparisonRanks: new Float32Array(rowCount).fill(0.5),
  };
}

test("live gate and learner ranks change Top-50 membership without changing static tensors", () => {
  const fixture = createSource(51);
  const current = liveValues(51, 2);
  fixture.source.liveValues = current;
  const beforeGate = buildSmartFilterMask("softgate-boundary", fixture.source);
  const beforeDisagreement = buildSmartFilterMask("learner-disagreement", fixture.source);
  assert.equal(beforeGate.mask[50], 0);
  assert.equal(beforeDisagreement.mask[50], 0);

  current.gateScores[50 * 2 + 1] = 0.5;
  current.learnerRanks[50 * 8] = 1;
  const afterGate = buildSmartFilterMask("softgate-boundary", fixture.source);
  const afterDisagreement = buildSmartFilterMask("learner-disagreement", fixture.source);
  for (const result of [afterGate, afterDisagreement]) {
    assert.equal(result.selectedCount, 50);
    assert.equal(result.mask[50], 1);
    assert.equal(result.mask[49], 0);
  }
  assert.ok(fixture.source.rawScores.every((value) => value === 0));
  assert.ok(fixture.source.ranks.every((value) => value === 0));
});

test("live learner-fusion gap uses mapped learner columns and current fusion ranks", () => {
  const fixture = createSource(3);
  const current = liveValues(3);
  fixture.source.liveValues = current;
  fixture.source.learnerMethodIndices = [7, 6, 5, 4, 3, 2, 1, 0];
  fixture.source.selectedLearnerIndex = 2;
  current.learnerRanks[5] = 0.95;
  current.learnerRanks[8 + 5] = 0.95;
  current.fusionRanks.set([0.4, 0.8, 0.1]);
  assert.deepEqual(buildSmartFilterMask("learner-fusion-gap", fixture.source).rowIndices, [0]);
  current.fusionRanks.set([0.8, 0.4, 0.1]);
  assert.deepEqual(buildSmartFilterMask("learner-fusion-gap", fixture.source).rowIndices, [1]);
});

test("live prototype gap keeps fixed prototype ranks but follows the current comparison", () => {
  const fixture = createSource(3);
  const current = liveValues(3);
  fixture.source.liveValues = current;
  for (let row = 0; row < 3; row += 1) {
    fixture.setRank(row, 9, 1, row === 2 ? 0.1 : 0.95);
    fixture.setRank(row, 8, 1, 0.1);
  }
  current.comparisonRanks.set([0.8, 0.4, 0.1]);
  assert.deepEqual(buildSmartFilterMask("prototype-rank-gap", fixture.source).rowIndices, [1]);
  current.comparisonRanks.set([0.4, 0.8, 0.1]);
  assert.deepEqual(buildSmartFilterMask("prototype-rank-gap", fixture.source).rowIndices, [0]);
  delete fixture.source.liveValues;
  assert.deepEqual(buildSmartFilterMask("prototype-rank-gap", fixture.source).rowIndices, [0, 1]);
});

test("candidate scope only selects from fixed live gallery ranks and never renormalizes them", () => {
  const fixture = createSource(3);
  fixture.source.liveValues = liveValues(3);
  fixture.source.liveValues.learnerRanks[8] = 0.8;
  fixture.source.liveValues.fusionRanks[1] = 0.1;
  fixture.source.candidateMask.set([0, 1, 0]);
  const result = buildSmartFilterMask("learner-fusion-gap", fixture.source);
  assert.equal(result.candidateCount, 1);
  assert.equal(result.selectedCount, 0, "the sole scoped candidate's .8 gallery rank must not become 1");
});

test("malformed current values fail closed instead of falling back to static diagnostics", () => {
  const variants = [
    (source) => { source.liveValues.gateCount = 0; },
    (source) => { source.liveValues.gateScores = new Float32Array(0); },
    (source) => { source.liveValues.learnerRanks = new Float32Array(1); },
    (source) => { source.liveValues.fusionRanks = new Float32Array(0); },
    (source) => { source.liveValues.comparisonRanks = new Float32Array(0); },
    (source) => { source.liveValues.gateScores[0] = Number.NaN; },
    (source) => { source.liveValues.learnerRanks[0] = Number.POSITIVE_INFINITY; },
    (source) => { source.liveValues.fusionRanks[0] = -0.1; },
    (source) => { source.liveValues.comparisonRanks[0] = 1.1; },
    (source) => { source.selectedLearnerIndex = 9; },
    (source) => { source.learnerMethodIndices = [0, 0, 1, 2, 3, 4, 5, 6]; },
  ];
  for (const change of variants) {
    const fixture = createSource(2);
    fixture.source.liveValues = liveValues(2);
    change(fixture.source);
    assert.throws(() => buildSmartFilterMask("softgate-boundary", fixture.source), /Live diagnostic/);
  }
});
