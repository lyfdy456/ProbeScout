import assert from "node:assert/strict";
import test from "node:test";

import {
  buildRuleMask,
  matchesRuleComparator,
} from "../app/ruleFilter.ts";

test("supports scalar and range rule comparators", () => {
  assert.equal(matchesRuleComparator(0.8, ">", 0.75, 0.9), true);
  assert.equal(matchesRuleComparator(0.75, ">=", 0.75, 0.9), true);
  assert.equal(matchesRuleComparator(0.2, "<", 0.25, 0.9), true);
  assert.equal(matchesRuleComparator(0.25, "<=", 0.25, 0.9), true);
  assert.equal(matchesRuleComparator(0.5, "between", 0.7, 0.3), true);
  assert.equal(matchesRuleComparator(0.8, "outside", 0.3, 0.7), true);
  assert.equal(matchesRuleComparator(Number.NaN, ">", 0, 1), false);
});

test("combines model-output clauses with NOT, grouped AND, and root OR", () => {
  const source = {
    rowCount: 4,
    methodCount: 1,
    targetCount: 2,
    methodIndex: 0,
    targetIndices: new Map([["a", 0], ["b", 1]]),
    // Layout is [row, method, target].
    calibratedScores: new Float32Array([
      0.8, 0.2,
      0.9, 0.8,
      0.4, 0.1,
      0.5, 0.7,
    ]),
    ranks: new Float32Array([
      0.9, 0.1,
      0.7, 0.5,
      0.4, 0.8,
      0.3, 0.2,
    ]),
  };
  const groups = [
    {
      id: "g1",
      logic: "AND",
      clauses: [
        {
          id: "a-high",
          target: "a",
          valueKind: "calibrated",
          comparator: ">=",
          threshold: 0.7,
          upperThreshold: 1,
          negated: false,
        },
        {
          id: "not-b-high",
          target: "b",
          valueKind: "calibrated",
          comparator: ">",
          threshold: 0.5,
          upperThreshold: 1,
          negated: true,
        },
      ],
    },
    {
      id: "g2",
      logic: "OR",
      clauses: [
        {
          id: "b-tail-rank",
          target: "b",
          valueKind: "rank",
          comparator: "outside",
          threshold: 0.3,
          upperThreshold: 0.7,
          negated: false,
        },
      ],
    },
  ];

  assert.deepEqual([...buildRuleMask(groups, "OR", source)], [1, 0, 1, 1]);
  assert.deepEqual([...buildRuleMask(groups, "AND", source)], [1, 0, 0, 0]);
});

test("an empty rule builder keeps every row eligible", () => {
  const mask = buildRuleMask([], "AND", {
    rowCount: 3,
    methodCount: 1,
    targetCount: 1,
    methodIndex: 0,
    targetIndices: new Map([["target", 0]]),
    calibratedScores: new Float32Array(3),
    ranks: new Float32Array(3),
  });
  assert.deepEqual([...mask], [1, 1, 1]);
});

test("a personal tune overrides Rule values only for its tuned target", () => {
  const source = {
    rowCount: 3,
    methodCount: 1,
    targetCount: 2,
    methodIndex: 0,
    targetIndices: new Map([["tuned", 0], ["untouched", 1]]),
    // Base layout is [row, method, target]. The tuned target's base values
    // deliberately disagree with the runtime output.
    calibratedScores: new Float32Array([
      0.1, 0.8,
      0.9, 0.4,
      0.2, 0.9,
    ]),
    ranks: new Float32Array([
      0.1, 0.9,
      0.9, 0.8,
      0.2, 0.3,
    ]),
    targetOverride: {
      targetIndex: 0,
      calibratedScores: new Float32Array([0.9, 0.2, 0.8]),
      ranks: new Float32Array([0.8, 0.1, 0.7]),
    },
  };
  const groups = [{
    id: "personal-tune-rule",
    logic: "AND",
    clauses: [
      {
        id: "new-score",
        target: "tuned",
        valueKind: "calibrated",
        comparator: ">=",
        threshold: 0.75,
        upperThreshold: 1,
        negated: false,
      },
      {
        id: "new-rank",
        target: "tuned",
        valueKind: "rank",
        comparator: ">=",
        threshold: 0.75,
        upperThreshold: 1,
        negated: false,
      },
      {
        id: "base-other-target",
        target: "untouched",
        valueKind: "rank",
        comparator: ">=",
        threshold: 0.5,
        upperThreshold: 1,
        negated: false,
      },
    ],
  }];

  assert.deepEqual([...buildRuleMask(groups, "AND", source)], [1, 0, 0]);
});

test("a rank-only legacy override leaves calibrated Rule clauses on the base score", () => {
  const source = {
    rowCount: 2,
    methodCount: 1,
    targetCount: 1,
    methodIndex: 0,
    targetIndices: new Map([["target", 0]]),
    calibratedScores: new Float32Array([0.8, 0.2]),
    ranks: new Float32Array([0.1, 0.9]),
    targetOverride: {
      targetIndex: 0,
      ranks: new Float32Array([0.9, 0.1]),
    },
  };
  const groups = [{
    id: "legacy-safe-rule",
    logic: "AND",
    clauses: [
      {
        id: "base-calibrated",
        target: "target",
        valueKind: "calibrated",
        comparator: ">=",
        threshold: 0.5,
        upperThreshold: 1,
        negated: false,
      },
      {
        id: "tuned-rank",
        target: "target",
        valueKind: "rank",
        comparator: ">=",
        threshold: 0.5,
        upperThreshold: 1,
        negated: false,
      },
    ],
  }];

  assert.deepEqual([...buildRuleMask(groups, "AND", source)], [1, 0]);
});
