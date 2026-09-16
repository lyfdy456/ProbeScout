import assert from "node:assert/strict";
import test from "node:test";

import {
  WEIGHTED_FUSION_ALGORITHM,
  WEIGHTED_FUSION_LAYOUT,
  WEIGHTED_FUSION_LEARNERS,
  areWeightedFusionWeightsEquivalent,
  createEqualWeightedFusionWeights,
  normalizeWeightedFusionWeights,
  parseWeightedFusionResponse,
} from "../app/lib/weightedFusion.ts";

test("normalizes all eight learner weights and treats common scaling as equivalent", () => {
  const equal = createEqualWeightedFusionWeights();
  const normalized = normalizeWeightedFusionWeights(equal);
  assert.equal(WEIGHTED_FUSION_LEARNERS.length, 8);
  for (const learner of WEIGHTED_FUSION_LEARNERS) {
    assert.equal(normalized[learner], 0.125);
  }
  assert.equal(
    areWeightedFusionWeightsEquivalent(equal, createEqualWeightedFusionWeights(7)),
    true,
  );
});

test("rejects negative, non-finite, and all-zero weight sets", () => {
  const negative = createEqualWeightedFusionWeights();
  negative.MLP = -1;
  assert.throws(() => normalizeWeightedFusionWeights(negative), /non-negative/);

  const nonFinite = createEqualWeightedFusionWeights();
  nonFinite["K-Fold"] = Number.NaN;
  assert.throws(() => normalizeWeightedFusionWeights(nonFinite), /finite non-negative/);

  const zero = createEqualWeightedFusionWeights();
  for (const learner of WEIGHTED_FUSION_LEARNERS) zero[learner] = 0;
  assert.throws(() => normalizeWeightedFusionWeights(zero), /greater than zero/);
});

test("parses the raw, calibrated, rank binary layout and audit headers", async () => {
  const values = new Float32Array([
    0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 0.25, 0.75,
  ]);
  const response = new Response(values.buffer, {
    headers: {
      "Content-Type": "application/octet-stream",
      "X-PCP-Algorithm": WEIGHTED_FUSION_ALGORITHM,
      "X-PCP-Layout": WEIGHTED_FUSION_LAYOUT.join(","),
      "X-PCP-Row-Count": "2",
      "X-PCP-Target-Count": "2",
      "X-PCP-Equal-Weights": "1",
      "X-PCP-Baseline-Max-Abs-Error": "0.000001",
    },
  });
  const result = await parseWeightedFusionResponse(
    response,
    { taskId: "task-a", rowCount: 2, targetCount: 2 },
    createEqualWeightedFusionWeights(),
  );

  assert.deepEqual([...result.rawScores], [...values.slice(0, 4)]);
  assert.deepEqual([...result.calibratedScores], [...values.slice(4, 8)]);
  assert.deepEqual([...result.ranks], [...values.slice(8, 12)]);
  assert.equal(result.equalWeights, true);
  assert.equal(result.baselineMaxAbsError, 0.000001);
});

test("rejects response shapes and byte lengths that do not match the active task", async () => {
  const headers = {
    "Content-Type": "application/octet-stream",
    "X-PCP-Algorithm": WEIGHTED_FUSION_ALGORITHM,
    "X-PCP-Layout": WEIGHTED_FUSION_LAYOUT.join(","),
    "X-PCP-Row-Count": "2",
    "X-PCP-Target-Count": "1",
    "X-PCP-Equal-Weights": "0",
  };
  await assert.rejects(
    parseWeightedFusionResponse(
      new Response(new Float32Array(6).buffer, { headers }),
      { taskId: "task-a", rowCount: 3, targetCount: 1 },
      createEqualWeightedFusionWeights(),
    ),
    /does not match/,
  );
  await assert.rejects(
    parseWeightedFusionResponse(
      new Response(new Float32Array(5).buffer, { headers }),
      { taskId: "task-a", rowCount: 2, targetCount: 1 },
      createEqualWeightedFusionWeights(),
    ),
    /bytes; expected/,
  );
});
