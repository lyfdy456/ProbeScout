import assert from "node:assert/strict";
import test from "node:test";

import {
  analysisScopeMask,
  applyResultScope,
  countRemainingPositives,
  filterAnalysisRows,
  isGroundTruthPositive,
  scopeMaskFor,
  validateScopeMasks,
} from "../app/lib/resultScope.ts";

function splitMasks() {
  return {
    developmentMask: Uint8Array.from([1, 0, 0, 0, 0, 0]),
    validationMask: Uint8Array.from([0, 1, 0, 1, 0, 0]),
    testMask: Uint8Array.from([0, 0, 1, 0, 0, 0]),
  };
}

test("three analysis scopes return their explicit aligned mask without copying", () => {
  const masks = splitMasks();
  assert.equal(scopeMaskFor(6, masks, "development"), masks.developmentMask);
  assert.equal(analysisScopeMask(6, masks, "validation"), masks.validationMask);
  assert.equal(analysisScopeMask(6, masks, "test"), masks.testMask);
});

test("scope validation allows all-zero Query rows but rejects overlap and malformed masks", () => {
  assert.doesNotThrow(() => validateScopeMasks(6, splitMasks()));
  assert.throws(
    () => validateScopeMasks(2, {
      developmentMask: Uint8Array.from([1, 0]),
      validationMask: Uint8Array.from([1, 0]),
      testMask: Uint8Array.from([0, 0]),
    }),
    /overlap at row 0/,
  );
  assert.throws(
    () => validateScopeMasks(2, {
      developmentMask: Uint8Array.from([2, 0]),
      validationMask: Uint8Array.from([0, 0]),
      testMask: Uint8Array.from([0, 0]),
    }),
    /binary/,
  );
  assert.throws(() => validateScopeMasks(3, splitMasks()), /aligned/);
  assert.throws(() => validateScopeMasks(-1, splitMasks()), /non-negative integer/);
});

test("analysis row filtering preserves source order and rejects invalid row indices", () => {
  const rows = [
    { rowIndex: 5 },
    { rowIndex: 1 },
    { rowIndex: 3 },
    { rowIndex: -1 },
    { rowIndex: 99 },
  ];
  const filtered = filterAnalysisRows(
    rows,
    (row) => row.rowIndex,
    splitMasks().validationMask,
  );
  assert.deepEqual(filtered, [{ rowIndex: 1 }, { rowIndex: 3 }]);
});

test("candidate scope intersection supports development, validation, and test", () => {
  const candidates = Uint8Array.from([1, 1, 1, 0, 1, 1]);
  const masks = splitMasks();
  assert.deepEqual([...applyResultScope(candidates, masks, "development")], [1, 0, 0, 0, 0, 0]);
  assert.deepEqual([...applyResultScope(candidates, masks, "validation")], [0, 1, 0, 0, 0, 0]);
  assert.deepEqual([...applyResultScope(candidates, masks, "test")], [0, 0, 1, 0, 0, 0]);
});

test("ground-truth lookup follows the exported row-target layout", () => {
  const labels = Uint8Array.from([
    1, 0, 0,
    0, 1, 1,
  ]);

  assert.equal(isGroundTruthPositive(labels, 0, 0, 3), true);
  assert.equal(isGroundTruthPositive(labels, 0, 1, 3), false);
  assert.equal(isGroundTruthPositive(labels, 1, 2, 3), true);
  assert.equal(isGroundTruthPositive(labels, 2, 0, 3), false);
});

test("positive remainder never falls below zero", () => {
  assert.equal(countRemainingPositives(19, 16), 3);
  assert.equal(countRemainingPositives(19, 19), 0);
  assert.equal(countRemainingPositives(19, 20), 0);
});
