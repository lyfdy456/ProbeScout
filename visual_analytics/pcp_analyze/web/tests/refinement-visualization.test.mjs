import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

register(
  new URL("./support/extensionless-typescript-loader.mjs", import.meta.url),
  import.meta.url,
);

const {
  WEIGHT_JOINT_ALGORITHM,
  WEIGHT_STAGED_ALGORITHM,
  WEIGHT_JOINT_ALGORITHM_V2,
  WEIGHT_STAGED_ALGORITHM_V2,
  WEIGHT_JOINT_ALGORITHM_V3,
  WEIGHT_STAGED_ALGORITHM_V3,
  WEIGHT_JOINT_ALGORITHM_V4,
  WEIGHT_STAGED_ALGORITHM_V4,
  PROBE_STAGED_ALGORITHM,
  PROBE_JOINT_ALGORITHM,
  PROBE_STAGED_ALGORITHM_V2,
  PROBE_JOINT_ALGORITHM_V2,
  LEGACY_REFINEMENT_EMBEDDING_METHODS,
  REFINEMENT_EMBEDDING_METHODS,
  refinementEmbeddingMethods,
  isWeightRefinementRun,
} = await import("../app/lib/tuningApi.ts");
const {
  REFINEMENT_CLUSTER_ALGORITHM,
  REFINEMENT_CLUSTER_FEATURE_BASIS,
  REFINEMENT_VISUALIZATION_ALGORITHM,
  REFINEMENT_VISUALIZATION_LAYOUT,
  requestRefinementClusters,
  requestRefinementVisualization,
} = await import("../app/lib/refinementVisualization.ts");

function refinementRun(mode = "weight_staged", overrides = {}) {
  return {
    id: `run-${mode}`,
    sessionId: "session-1",
    mode,
    baseMethod: "Ours-Full",
    status: "succeeded",
    createdAt: "2026-09-05T00:00:00Z",
    attributeIds: ["cat"],
    embeddingMethods: [...REFINEMENT_EMBEDDING_METHODS],
    algorithmVersion: mode === "weight_joint"
      ? WEIGHT_JOINT_ALGORITHM
      : WEIGHT_STAGED_ALGORITHM,
    visualizationUrl: `/api/tuning/runs/run-${mode}/visualization.f32`,
    clusterUrl: `/api/tuning/runs/run-${mode}/clusters.u8`,
    ...overrides,
  };
}

function installFetch(implementation) {
  const previous = globalThis.fetch;
  globalThis.fetch = implementation;
  return () => {
    globalThis.fetch = previous;
  };
}

function visualizationResponse(run, {
  rowCount = 3,
  componentCount = 3 + run.attributeIds.length * 9 + run.embeddingMethods.length,
  scores = Float32Array.from({ length: rowCount * componentCount }, (_, index) => index / (rowCount * componentCount)),
  ranks = Float32Array.from({ length: rowCount * componentCount }, (_, index) => 1 - index / (rowCount * componentCount)),
  headers = {},
  body,
  status = 200,
} = {}) {
  const values = new Float32Array(scores.length + ranks.length);
  values.set(scores, 0);
  values.set(ranks, scores.length);
  return new Response(body ?? values, {
    status,
    headers: {
      "Content-Type": "application/octet-stream",
      "X-PCP-Algorithm": REFINEMENT_VISUALIZATION_ALGORITHM,
      "X-PCP-Layout": REFINEMENT_VISUALIZATION_LAYOUT.join(","),
      "X-PCP-Run-Id": run.id,
      "X-PCP-Row-Count": String(rowCount),
      "X-PCP-Component-Count": String(componentCount),
      "X-PCP-Source-Fingerprint": "source-sha-1",
      "X-PCP-Base-Fingerprint": "base-sha-1",
      ...headers,
    },
  });
}

function clusterResponse(run, snapshot, scheme, {
  clusterCount = 2,
  fitRowCount = 2,
  headers = {},
  body = Uint8Array.from({ length: snapshot.rowCount }, (_, index) => index % clusterCount),
  status = 200,
} = {}) {
  return new Response(body, {
    status,
    headers: {
      "Content-Type": "application/octet-stream",
      "X-PCP-Algorithm": REFINEMENT_CLUSTER_ALGORITHM,
      "X-PCP-Feature-Basis": REFINEMENT_CLUSTER_FEATURE_BASIS,
      "X-PCP-Run-Id": run.id,
      "X-PCP-Source-Fingerprint": snapshot.sourceFingerprint,
      "X-PCP-Row-Count": String(snapshot.rowCount),
      "X-PCP-Cluster-Count": String(clusterCount),
      "X-PCP-Feature-Count": String(snapshot.componentCount),
      "X-PCP-Fit-Row-Count": String(fitRowCount),
      "X-PCP-Cluster-Scheme": scheme,
      "X-PCP-Fit-Scope": "development",
      ...headers,
    },
  });
}

test("only exact Weight refinement runs advertise the full PCP visualization contract", () => {
  assert.equal(isWeightRefinementRun(refinementRun("weight_staged")), true);
  assert.equal(isWeightRefinementRun(refinementRun("weight_joint")), true);
  assert.equal(
    isWeightRefinementRun(refinementRun("weight_staged", { algorithmVersion: "other" })),
    false,
  );
  assert.equal(
    isWeightRefinementRun(refinementRun("weight_joint", { visualizationUrl: undefined })),
    false,
  );
  assert.equal(
    isWeightRefinementRun(refinementRun("weight_joint", { clusterUrl: undefined })),
    false,
  );
  assert.equal(
    isWeightRefinementRun(refinementRun("residual", {
      algorithmVersion: "residual-tune-v1",
    })),
    false,
  );
});

test("refinement visualization decodes one immutable score/rank matrix for the whole PCP", async () => {
  const run = refinementRun("weight_staged", { id: "run-visualization-contract" });
  const scores = Float32Array.from({ length: 42 }, (_, index) => index / 42);
  const ranks = Float32Array.from({ length: 42 }, (_, index) => (41 - index) / 41);
  const calls = [];
  const restore = installFetch(async (url, init) => {
    calls.push({ url: String(url), init });
    return visualizationResponse(run, { scores, ranks });
  });
  try {
    const snapshot = await requestRefinementVisualization(run, 3);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, run.visualizationUrl);
    assert.equal(calls[0].init.credentials, "same-origin");
    assert.equal(calls[0].init.headers.Accept, "application/octet-stream");
    assert.equal(snapshot.runId, run.id);
    assert.equal(snapshot.rowCount, 3);
    assert.equal(snapshot.componentCount, 14);
    assert.deepEqual(snapshot.attributeIds, ["cat"]);
    assert.deepEqual(snapshot.embeddingMethods, [...REFINEMENT_EMBEDDING_METHODS]);
    assert.equal(snapshot.sourceFingerprint, "source-sha-1");
    assert.equal(snapshot.baseFingerprint, "base-sha-1");
    assert.deepEqual([...snapshot.scores], [...scores]);
    assert.deepEqual([...snapshot.ranks], [...ranks]);
  } finally {
    restore();
  }
});

test("refinement visualization rejects a stale, cross-run, or malformed matrix", async (t) => {
  const run = refinementRun("weight_joint", { id: "run-invalid-visualization" });
  const cases = [
    {
      name: "five-channel payload for a two-channel run",
      response: () => visualizationResponse(run, { componentCount: 17 }),
      pattern: /component count does not match/,
    },
    {
      name: "old layout without C",
      response: () => visualizationResponse(run, { componentCount: 13 }),
      pattern: /component count does not match/,
    },
    {
      name: "old algorithm",
      response: () => visualizationResponse(run, {
        headers: { "X-PCP-Algorithm": "conjunction-holistic-visualization-v1" },
      }),
      pattern: /Unsupported refinement visualization algorithm/,
    },
    {
      name: "nonfinite C score",
      response: () => visualizationResponse(run, {
        scores: Float32Array.from({ length: 42 }, (_, index) => index === 1 ? Number.NaN : 0.5),
      }),
      pattern: /must be finite values/,
    },
    {
      name: "out of range rank",
      response: () => visualizationResponse(run, { ranks: new Float32Array(42).fill(1.1) }),
      pattern: /must be finite values/,
    },
    {
      name: "wrong run",
      response: () => visualizationResponse(run, {
        headers: { "X-PCP-Run-Id": "another-run" },
      }),
      pattern: /belongs to another run/,
    },
    {
      name: "wrong row count",
      response: () => visualizationResponse(run, { rowCount: 2 }),
      pattern: /row count changed/,
    },
    {
      name: "wrong layout",
      response: () => visualizationResponse(run, {
        headers: { "X-PCP-Layout": "rank,score" },
      }),
      pattern: /Unsupported refinement visualization layout/,
    },
    {
      name: "truncated payload",
      response: () => visualizationResponse(run, { body: new Uint8Array(7) }),
      pattern: /wrong size/,
    },
    {
      name: "missing source fingerprint",
      response: () => visualizationResponse(run, {
        headers: { "X-PCP-Source-Fingerprint": "" },
      }),
      pattern: /Missing X-PCP-Source-Fingerprint/,
    },
  ];

  for (const current of cases) {
    await t.test(current.name, async () => {
      const restore = installFetch(async () => current.response());
      try {
        await assert.rejects(requestRefinementVisualization(run, 3), current.pattern);
      } finally {
        restore();
      }
    });
  }
});

test("two-channel v5 and five-channel historical v2 retain distinct method identities", async () => {
  const current = refinementRun("weight_joint");
  const historical = refinementRun("weight_joint", {
    id: "legacy-five-channel",
    algorithmVersion: WEIGHT_JOINT_ALGORITHM_V2,
    embeddingMethods: [...LEGACY_REFINEMENT_EMBEDDING_METHODS],
  });
  assert.equal(isWeightRefinementRun(historical), true);
  assert.deepEqual(refinementEmbeddingMethods(current), ["Query MaxSim", "Text Prompt Ensemble"]);
  assert.deepEqual(refinementEmbeddingMethods(historical), [...LEGACY_REFINEMENT_EMBEDDING_METHODS]);
  assert.equal(refinementEmbeddingMethods({ ...current, embeddingMethods: [...LEGACY_REFINEMENT_EMBEDDING_METHODS] }), null);
  assert.equal(refinementEmbeddingMethods({ ...current, embeddingMethods: [...REFINEMENT_EMBEDDING_METHODS].reverse() }), null);
  assert.equal(refinementEmbeddingMethods({ ...current, after: { modelSummary: {
    embeddingWeights: { "Query MaxSim": 0.5, "Text Prompt Ensemble": 0.5 },
  } } })?.length, 2);
  assert.equal(refinementEmbeddingMethods({ ...current, after: { modelSummary: {
    embeddingWeights: { "Query MaxSim": 0.2, "Text Prompt Ensemble": 0.2 },
  } } }), null);
  const restore = installFetch(async () => visualizationResponse(historical));
  try {
    const snapshot = await requestRefinementVisualization(historical, 3);
    assert.equal(snapshot.componentCount, 17);
    assert.deepEqual(snapshot.embeddingMethods, [...LEGACY_REFINEMENT_EMBEDDING_METHODS]);
    await assert.rejects(requestRefinementVisualization({
      ...current, embeddingMethods: [...LEGACY_REFINEMENT_EMBEDDING_METHODS],
    }, 3), /run-pinned refinement method identities/);
    await assert.rejects(requestRefinementVisualization({ ...current, attributeIds: [] }, 3), /run-pinned/);
  } finally {
    restore();
  }
});

test("both v5 modes and historical v4/v3/v2 keep their exact mode and embedding contracts", async () => {
  assert.equal(WEIGHT_STAGED_ALGORITHM, "conjunction-holistic-weight-staged-v5");
  assert.equal(WEIGHT_JOINT_ALGORITHM, "conjunction-holistic-weight-joint-v5");
  assert.equal(WEIGHT_STAGED_ALGORITHM_V4, "conjunction-holistic-weight-staged-v4");
  assert.equal(WEIGHT_JOINT_ALGORITHM_V4, "conjunction-holistic-weight-joint-v4");
  const contracts = [
    ["weight_staged", WEIGHT_STAGED_ALGORITHM, REFINEMENT_EMBEDDING_METHODS],
    ["weight_joint", WEIGHT_JOINT_ALGORITHM, REFINEMENT_EMBEDDING_METHODS],
    ["weight_staged", WEIGHT_STAGED_ALGORITHM_V4, REFINEMENT_EMBEDDING_METHODS],
    ["weight_joint", WEIGHT_JOINT_ALGORITHM_V4, REFINEMENT_EMBEDDING_METHODS],
    ["weight_staged", WEIGHT_STAGED_ALGORITHM_V3, REFINEMENT_EMBEDDING_METHODS],
    ["weight_joint", WEIGHT_JOINT_ALGORITHM_V3, REFINEMENT_EMBEDDING_METHODS],
    ["weight_staged", WEIGHT_STAGED_ALGORITHM_V2, LEGACY_REFINEMENT_EMBEDDING_METHODS],
    ["weight_joint", WEIGHT_JOINT_ALGORITHM_V2, LEGACY_REFINEMENT_EMBEDDING_METHODS],
  ];
  for (const [mode, algorithmVersion, methods] of contracts) {
    const run = refinementRun(mode, { id: algorithmVersion, algorithmVersion, embeddingMethods: [...methods] });
    assert.equal(isWeightRefinementRun(run), true);
    assert.deepEqual([...refinementEmbeddingMethods(run)], [...methods]);
    const wrongMode = { ...run, mode: mode === "weight_joint" ? "weight_staged" : "weight_joint" };
    assert.equal(isWeightRefinementRun(wrongMode), false);
    assert.equal(refinementEmbeddingMethods(wrongMode), null);
    assert.equal(refinementEmbeddingMethods({ ...run, embeddingMethods: [...methods].reverse() }), null);
    const wrongMethods = methods.length === 2 ? LEGACY_REFINEMENT_EMBEDDING_METHODS : REFINEMENT_EMBEDDING_METHODS;
    assert.equal(refinementEmbeddingMethods({ ...run, embeddingMethods: [...wrongMethods] }), null);
    const restore = installFetch(async () => visualizationResponse(run));
    try {
      const snapshot = await requestRefinementVisualization(run, 3);
      assert.equal(snapshot.componentCount, 12 + methods.length);
      assert.deepEqual(snapshot.embeddingMethods, [...methods]);
    } finally {
      restore();
    }
  }
});

test("native v3 and historical v2 runs apply exact two-channel PCP snapshots", async () => {
  assert.equal(PROBE_STAGED_ALGORITHM, "conjunction-holistic-probe-staged-v3");
  assert.equal(PROBE_JOINT_ALGORITHM, "conjunction-holistic-probe-joint-v3");
  assert.equal(PROBE_STAGED_ALGORITHM_V2, "conjunction-holistic-probe-staged-v2");
  assert.equal(PROBE_JOINT_ALGORITHM_V2, "conjunction-holistic-probe-joint-v2");
  for (const [mode, algorithmVersion] of [
    ["probe_staged", PROBE_STAGED_ALGORITHM],
    ["probe_joint", PROBE_JOINT_ALGORITHM],
    ["probe_staged", PROBE_STAGED_ALGORITHM_V2],
    ["probe_joint", PROBE_JOINT_ALGORITHM_V2],
  ]) {
    const run = refinementRun(mode, { id: algorithmVersion, algorithmVersion });
    assert.equal(isWeightRefinementRun(run), true);
    assert.deepEqual([...refinementEmbeddingMethods(run)], [...REFINEMENT_EMBEDDING_METHODS]);
    const legacy = { ...run, algorithmVersion: algorithmVersion.replace(/-v[23]$/, "-v1") };
    assert.equal(isWeightRefinementRun(legacy), false);
    assert.equal(refinementEmbeddingMethods(legacy), null);
    const wrongMode = { ...run, mode: mode === "probe_staged" ? "probe_joint" : "probe_staged" };
    assert.equal(isWeightRefinementRun(wrongMode), false);
    assert.equal(refinementEmbeddingMethods(wrongMode), null);
    assert.equal(refinementEmbeddingMethods({ ...run, embeddingMethods: [...REFINEMENT_EMBEDDING_METHODS].reverse() }), null);
    assert.equal(refinementEmbeddingMethods({ ...run, embeddingMethods: [...LEGACY_REFINEMENT_EMBEDDING_METHODS] }), null);
    const restore = installFetch(async (url) => String(url).includes("clusters.u8")
      ? clusterResponse(run, { rowCount: 3, componentCount: 14, sourceFingerprint: `native-${mode}` }, "fine30")
      : visualizationResponse(run, { headers: { "X-PCP-Source-Fingerprint": `native-${mode}` } }));
    try {
      const snapshot = await requestRefinementVisualization(run, 3);
      assert.equal(snapshot.componentCount, 14);
      const clusters = await requestRefinementClusters({ run, snapshot, scheme: "fine30" });
      assert.equal(clusters.runId, run.id);
      assert.equal(clusters.sourceFingerprint, snapshot.sourceFingerprint);
      assert.equal(clusters.featureCount, snapshot.componentCount);
    } finally { restore(); }
  }
});

test("refinement clusters are keyed to the exact run snapshot and PCP component count", async () => {
  const run = refinementRun("weight_joint", {
    id: "run-cluster-contract",
    clusterUrl: "/api/tuning/runs/run-cluster-contract/clusters.u8",
  });
  const snapshot = {
    runId: run.id,
    rowCount: 6,
    componentCount: 5,
    scores: new Float32Array(30),
    ranks: new Float32Array(30),
    sourceFingerprint: "cluster-source-sha",
    baseFingerprint: "cluster-base-sha",
  };
  const calls = [];
  const restore = installFetch(async (url, init) => {
    calls.push({ url: String(url), init });
    return clusterResponse(run, snapshot, "fine30", { clusterCount: 3, fitRowCount: 4 });
  });
  try {
    const first = await requestRefinementClusters({ run, snapshot, scheme: "fine30" });
    const second = await requestRefinementClusters({ run, snapshot, scheme: "fine30" });
    assert.strictEqual(first, second);
    assert.equal(calls.length, 1, "the same run+artifact+scheme must share its cached labels");
    assert.equal(calls[0].url, `${run.clusterUrl}?scheme=fine30`);
    assert.equal(calls[0].init.credentials, "same-origin");
    assert.equal(first.runId, run.id);
    assert.equal(first.sourceFingerprint, snapshot.sourceFingerprint);
    assert.equal(first.featureCount, snapshot.componentCount);
    assert.equal(first.fitRowCount, 4);
    assert.deepEqual([...first.labels], [0, 1, 2, 0, 1, 2]);
  } finally {
    restore();
  }
});

test("refinement clusters reject labels produced for another score snapshot", async () => {
  const run = refinementRun("weight_staged", {
    id: "run-stale-cluster",
    clusterUrl: "/api/tuning/runs/run-stale-cluster/clusters.u8",
  });
  const snapshot = {
    runId: run.id,
    rowCount: 4,
    componentCount: 3,
    scores: new Float32Array(12),
    ranks: new Float32Array(12),
    sourceFingerprint: "current-source-sha",
    baseFingerprint: "base-sha",
  };
  const restore = installFetch(async () => clusterResponse(run, snapshot, "fine50", {
    headers: { "X-PCP-Source-Fingerprint": "stale-source-sha" },
  }));
  try {
    await assert.rejects(
      requestRefinementClusters({ run, snapshot, scheme: "fine50" }),
      /belong to another model snapshot/,
    );
  } finally {
    restore();
  }
});

test("useTuningSession preflights the matching clusters before atomically publishing Weight Tune", async () => {
  const source = await readFile(
    new URL("../app/lib/useTuningSession.ts", import.meta.url),
    "utf8",
  );
  const applyStart = source.indexOf("const applyRun = useCallback");
  const revertStart = source.indexOf("const revertRun = useCallback", applyStart);
  assert.ok(applyStart >= 0 && revertStart > applyStart);
  const applyBlock = source.slice(applyStart, revertStart);

  assert.match(applyBlock, /isWeightRefinementRun\(run\)/);
  assert.match(applyBlock, /requestRefinementVisualization\(run, rowCount\)/);
  assert.match(applyBlock, /Promise\.all\(\[/);
  const preflightAt = applyBlock.indexOf("await requestRefinementClusters");
  const ranksCommitAt = applyBlock.indexOf("setAppliedRanks(ranks)");
  const scoresCommitAt = applyBlock.indexOf("setAppliedScores(scores)");
  const visualizationCommitAt = applyBlock.indexOf("setAppliedVisualization(visualization)");
  const runCommitAt = applyBlock.indexOf("setAppliedRunId(run.id)");
  assert.ok(preflightAt >= 0, "Weight Tune must preflight current-scheme clusters");
  for (const commitAt of [ranksCommitAt, scoresCommitAt, visualizationCommitAt, runCommitAt]) {
    assert.ok(
      commitAt > preflightAt,
      "Top, PCP, and their run identity must commit only after cluster preflight succeeds",
    );
  }
  const catchStart = applyBlock.indexOf("} catch (reason)");
  const finallyStart = applyBlock.indexOf("} finally {", catchStart);
  assert.ok(catchStart > runCommitAt && finallyStart > catchStart);
  assert.doesNotMatch(
    applyBlock.slice(catchStart, finallyStart),
    /setAppliedRanks|setAppliedScores|setAppliedVisualization|setAppliedRunId/,
    "a failed visualization or cluster request must preserve the previously applied model",
  );

  const revertBlock = source.slice(revertStart, source.indexOf("return {", revertStart));
  assert.match(revertBlock, /setAppliedRanks\(null\)/);
  assert.match(revertBlock, /setAppliedScores\(null\)/);
  assert.match(revertBlock, /setAppliedVisualization\(null\)/);
  assert.match(revertBlock, /setAppliedRunId\(null\)/);
});
