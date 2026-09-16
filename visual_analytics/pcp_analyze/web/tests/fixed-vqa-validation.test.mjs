import assert from "node:assert/strict";
import test from "node:test";
import {
  fixedVqaValidationMask,
  developmentWithoutFixedVal,
  fixedValAllowsFeedback,
} from "../app/lib/fixedVqaValidation.ts";

const payload = {
  taskId: "task-a", version: "vqa-val-v1", manifestSha256: "a".repeat(64),
  rowIndices: [1, 4], count: 2, protocol: "fixed-vqa-joint-seed0-holdout-v1",
  labelSource: "original-vqa-supervision", initialModelHoldoutIndependent: false,
};

test("fixed Val validates provenance, exact count, unique task-aligned rows", () => {
  assert.deepEqual([...fixedVqaValidationMask(payload, "task-a", 6)], [0, 1, 0, 0, 1, 0]);
  for (const change of [
    { taskId: "task-b" }, { taskId: undefined }, { version: "" }, { manifestSha256: "bad" },
    { protocol: "" }, { protocol: "different-protocol" }, { referenceOnly: false },
    { labelSource: "ground-truth" }, { initialModelHoldoutIndependent: true },
    { count: 0, rowIndices: [] }, { count: 1 }, { rowIndices: [1, 1] },
    { rowIndices: [-1, 4] }, { rowIndices: [1, 6] }, { rowIndices: [1.5, 4] },
  ]) assert.throws(() => fixedVqaValidationMask({ ...payload, ...change }, "task-a", 6));
});

test("DG subtracts fixed Val without reclaiming old Web Validation or changing Test", () => {
  const development = Uint8Array.from([1, 1, 0, 0, 1, 0]);
  const validation = fixedVqaValidationMask(payload, "task-a", 6);
  const testMask = Uint8Array.from([0, 0, 0, 1, 0, 0]);
  assert.deepEqual([...developmentWithoutFixedVal(development, validation)], [1, 0, 0, 0, 0, 0]);
  assert.deepEqual([...development], [1, 1, 0, 0, 1, 0]);
  assert.deepEqual([...testMask], [0, 0, 0, 1, 0, 0]);
  assert.deepEqual([...developmentWithoutFixedVal(development, null)], [0, 0, 0, 0, 0, 0]);
  assert.throws(() => developmentWithoutFixedVal(development, new Uint8Array(2)));
});

test("all feedback modes fail closed on protected, pending, or misaligned rows", () => {
  const imageIds = ["a", "b", "c", "d", "e", "f"];
  const development = Uint8Array.from([1, 1, 0, 0, 1, 0]);
  const validation = fixedVqaValidationMask(payload, "task-a", 6);
  assert.equal(fixedValAllowsFeedback({ rowIndex: 0, id: "a" }, imageIds, development, validation), true);
  for (const item of [{ rowIndex: 1, id: "b" }, { rowIndex: 4, id: "e" },
    { rowIndex: 2, id: "c" }, { rowIndex: 3, id: "d" }, { rowIndex: -1, id: "a" },
    { rowIndex: 0, id: "wrong" }, { rowIndex: 0.5, id: "a" }]) {
    assert.equal(fixedValAllowsFeedback(item, imageIds, development, validation), false);
  }
  assert.equal(fixedValAllowsFeedback({ rowIndex: 0, id: "a" }, imageIds, development, null), false);
});
