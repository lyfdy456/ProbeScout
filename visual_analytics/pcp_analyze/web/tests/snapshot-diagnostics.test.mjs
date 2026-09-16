import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register(new URL("./support/extensionless-typescript-loader.mjs", import.meta.url), import.meta.url);
const { buildSnapshotDiagnosticValues } = await import("../app/lib/snapshotDiagnostics.ts");
const { HIERARCHICAL_PCP_LEARNERS } = await import("../app/lib/hierarchicalPcp.ts");
const { buildSmartFilterMask } = await import("../app/smartFilter.ts");
const learners = [...HIERARCHICAL_PCP_LEARNERS];

function createSnapshot(rowCount = 4, attributeIds = ["cat", "blue"], embeddingMethods = ["Query MaxSim", "Text Prompt Ensemble"]) {
  const componentCount = 3 + attributeIds.length * 9 + embeddingMethods.length;
  const snapshot = {
    rowCount,
    componentCount,
    scores: new Float32Array(rowCount * componentCount).fill(0.5),
    ranks: new Float32Array(rowCount * componentCount).fill(0.5),
    attributeIds: [...attributeIds],
    embeddingMethods: [...embeddingMethods],
  };
  return {
    snapshot,
    gate(row, attribute, score, rank = score) {
      const column = 2 + snapshot.attributeIds.indexOf(attribute) * 9;
      snapshot.scores[row * componentCount + column] = score;
      snapshot.ranks[row * componentCount + column] = rank;
    },
    probe(row, attribute, method, score, rank = score) {
      const column = 3 + snapshot.attributeIds.indexOf(attribute) * 9 + learners.indexOf(method);
      snapshot.scores[row * componentCount + column] = score;
      snapshot.ranks[row * componentCount + column] = rank;
    },
    fusion(row, rank) { snapshot.ranks[row * componentCount] = rank; },
  };
}

const close = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-6, `${value} != ${expected[index]}`));
};

test("attribute diagnostics read current g and stored z ranks by identity, without mutating F0/Tune input", () => {
  const fixture = createSnapshot(3, ["blue", "cat"]);
  for (let row = 0; row < 3; row += 1) {
    fixture.gate(row, "cat", 0.2 + row * 0.1, 1 - row * 0.5);
    fixture.fusion(row, 0.123);
    learners.forEach((method, index) => fixture.probe(row, "cat", method, 0.1, (index + row) / 10));
  }
  const beforeScores = fixture.snapshot.scores.slice();
  const beforeRanks = fixture.snapshot.ranks.slice();
  const input = Object.freeze({ ...fixture.snapshot, attributeIds: Object.freeze([...fixture.snapshot.attributeIds]) });
  const methodOrder = [...learners].reverse();
  const values = buildSnapshotDiagnosticValues(input, "cat", methodOrder);
  assert.equal(values.gateCount, 1);
  close([...values.gateScores], [0.2, 0.3, 0.4]);
  close([...values.fusionRanks], [1, 0.5, 0]);
  close([...values.comparisonRanks], [1, 0.5, 0]);
  close([...values.learnerRanks.slice(0, 8)], [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]);
  close([...values.learnerRanks.slice(8, 16)], [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]);
  assert.deepEqual(fixture.snapshot.scores, beforeScores);
  assert.deepEqual(fixture.snapshot.ranks, beforeRanks);
  values.gateScores.fill(0);
  values.learnerRanks.fill(0);
  values.comparisonRanks.fill(0);
  assert.deepEqual(fixture.snapshot.scores, beforeScores);
  assert.deepEqual(fixture.snapshot.ranks, beforeRanks);
  close([...values.fusionRanks], [1, 0.5, 0]);
});

test("Joint uses all current gates, F rank, and per-learner gallery rank of the current z product", () => {
  const fixture = createSnapshot(4);
  const pairs = [[0.8, 0.1], [0.2, 0.8], [0.5, 0.5], [0.9, 0.9]];
  pairs.forEach(([cat, blue], row) => {
    fixture.gate(row, "cat", 0.2 + row * 0.1, 1);
    fixture.gate(row, "blue", 0.8 - row * 0.1, 0);
    fixture.fusion(row, 1 - row / 3);
    fixture.probe(row, "cat", "MLP", cat, 1 - row / 3);
    fixture.probe(row, "blue", "MLP", blue, 1 - row / 3);
  });
  const values = buildSnapshotDiagnosticValues(fixture.snapshot, "joint", learners);
  assert.equal(values.gateCount, 2);
  close([...values.gateScores], [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]);
  close([...values.fusionRanks], [1, 2 / 3, 1 / 3, 0]);
  close(Array.from({ length: 4 }, (_, row) => values.learnerRanks[row * 8]), [0, 1 / 3, 2 / 3, 1]);
  close(Array.from({ length: 4 }, (_, row) => values.learnerRanks[row * 8 + 1]), [0.5, 0.5, 0.5, 0.5]);
  fixture.probe(0, "blue", "MLP", 1, 0);
  const changed = buildSnapshotDiagnosticValues(fixture.snapshot, "joint", learners);
  close(Array.from({ length: 4 }, (_, row) => changed.learnerRanks[row * 8]), [2 / 3, 0, 1 / 3, 1]);
});

test("Joint log products preserve tiny positive scores, exact-zero ties, average ties and singleton rank", () => {
  const fixture = createSnapshot(5);
  const pairs = [[0, 1], [1, 0], [1e-30, 1e-30], [1e-20, 1e-20], [0.5, 0.5]];
  pairs.forEach(([cat, blue], row) => {
    fixture.probe(row, "cat", "MLP", cat);
    fixture.probe(row, "blue", "MLP", blue);
  });
  const values = buildSnapshotDiagnosticValues(fixture.snapshot, "joint", learners);
  close(Array.from({ length: 5 }, (_, row) => values.learnerRanks[row * 8]), [0.125, 0.125, 0.5, 0.75, 1]);
  const allTied = createSnapshot(3);
  for (let row = 0; row < 3; row += 1) allTied.probe(row, "cat", "MLP", 0);
  const tied = buildSnapshotDiagnosticValues(allTied.snapshot, "joint", learners);
  close(Array.from({ length: 3 }, (_, row) => tied.learnerRanks[row * 8]), [0.5, 0.5, 0.5]);
  const singleton = buildSnapshotDiagnosticValues(createSnapshot(1).snapshot, "joint", learners);
  assert.ok(singleton.learnerRanks.every((rank) => rank === 1));
});

test("reordered attribute/embedding manifests and UI learner order retain identity-based values", () => {
  const first = createSnapshot(3, ["cat", "blue"], ["Query MaxSim", "Text Prompt Ensemble"]);
  const reordered = createSnapshot(3, ["blue", "cat"], ["Text Prompt Ensemble", "Query MaxSim"]);
  for (const fixture of [first, reordered]) {
    for (let row = 0; row < 3; row += 1) {
      fixture.gate(row, "cat", 0.1 + row * 0.1);
      fixture.gate(row, "blue", 0.6 + row * 0.1);
      learners.forEach((method, index) => {
        fixture.probe(row, "cat", method, (row + index + 1) / 11, (index + row) / 10);
        fixture.probe(row, "blue", method, (3 - row + index) / 11, (8 - index + row) / 10);
      });
    }
  }
  const reversedMethods = [...learners].reverse();
  for (const target of ["joint", "cat", "blue"]) {
    const left = buildSnapshotDiagnosticValues(first.snapshot, target, learners);
    const right = buildSnapshotDiagnosticValues(reordered.snapshot, target, reversedMethods);
    assert.deepEqual(left.fusionRanks, right.fusionRanks);
    for (let row = 0; row < 3; row += 1) {
      assert.deepEqual(
        [...left.learnerRanks.slice(row * 8, row * 8 + 8)],
        [...right.learnerRanks.slice(row * 8, row * 8 + 8)].reverse(),
      );
      const rightGates = [...right.gateScores.slice(row * right.gateCount, (row + 1) * right.gateCount)];
      assert.deepEqual(
        [...left.gateScores.slice(row * left.gateCount, (row + 1) * left.gateCount)],
        target === "joint" ? rightGates.reverse() : rightGates,
      );
    }
  }
});

test("snapshot adapters change actual boundary/disagreement/gap membership after an applied update", () => {
  const fixture = createSnapshot(51, ["cat"]);
  for (let row = 0; row < 51; row += 1) fixture.gate(row, "cat", 0.8, 0.8);
  const staticRanks = new Float32Array(51 * 10);
  for (let row = 0; row < 51; row += 1) staticRanks[row * 10 + 9] = 0.95;
  const source = {
    rowCount: 51, methodCount: 10, targetCount: 1, targetIndex: 0,
    targetMemberIndices: [0], learnerMethodIndices: [0, 1, 2, 3, 4, 5, 6, 7],
    fusionMethodIndex: 8, prototypeMethodIndex: 9, selectedLearnerIndex: 0, comparisonMethodIndex: 8,
    rawScores: new Float32Array(51 * 10), ranks: staticRanks, candidateMask: new Uint8Array(51).fill(1),
    liveValues: buildSnapshotDiagnosticValues(fixture.snapshot, "cat", learners),
  };
  assert.equal(buildSmartFilterMask("softgate-boundary", source).mask[50], 0);
  assert.equal(buildSmartFilterMask("learner-disagreement", source).mask[50], 0);
  assert.equal(buildSmartFilterMask("learner-fusion-gap", source).selectedCount, 0);
  assert.equal(buildSmartFilterMask("prototype-rank-gap", source).selectedCount, 0);
  fixture.gate(50, "cat", 0.5, 0.2);
  fixture.probe(50, "cat", "MLP", 0.95, 1);
  source.liveValues = buildSnapshotDiagnosticValues(fixture.snapshot, "cat", learners);
  for (const kind of ["softgate-boundary", "learner-disagreement", "learner-fusion-gap", "prototype-rank-gap"]) {
    assert.equal(buildSmartFilterMask(kind, source).mask[50], 1, kind);
  }
});

test("invalid identities, layouts, targets and values never become legacy data", () => {
  const variants = [
    (snapshot) => { snapshot.attributeIds = []; },
    (snapshot) => { snapshot.attributeIds = ["cat", "cat"]; },
    (snapshot) => { snapshot.attributeIds = ["cat", ""]; },
    (snapshot) => { snapshot.attributeIds = ["cat", "joint"]; },
    (snapshot) => { snapshot.embeddingMethods = ["unknown"]; },
    (snapshot) => { snapshot.embeddingMethods = ["Query MaxSim", "Query MaxSim"]; },
    (snapshot) => { snapshot.componentCount -= 1; },
    (snapshot) => { snapshot.rowCount = 0; },
    (snapshot) => { snapshot.rowCount = 1.5; },
    (snapshot) => { snapshot.scores = new Float32Array(1); },
    (snapshot) => { snapshot.ranks = new Float32Array(1); },
    (snapshot) => { snapshot.scores[0] = Number.NaN; },
    (snapshot) => { snapshot.scores[0] = -0.1; },
    (snapshot) => { snapshot.ranks[0] = Number.POSITIVE_INFINITY; },
    (snapshot) => { snapshot.ranks[0] = 1.1; },
  ];
  for (const change of variants) {
    const { snapshot } = createSnapshot();
    change(snapshot);
    assert.throws(() => buildSnapshotDiagnosticValues(snapshot, "joint", learners), RangeError);
  }
  assert.throws(() => buildSnapshotDiagnosticValues(createSnapshot().snapshot, "missing", learners), /target/);
  for (const methods of [learners.slice(1), [...learners.slice(1), "unknown"], [...learners.slice(1), learners[1]]]) {
    assert.throws(() => buildSnapshotDiagnosticValues(createSnapshot().snapshot, "joint", methods), /identities/);
  }
});
