import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";
register(new URL("./support/extensionless-typescript-loader.mjs", import.meta.url), import.meta.url);
const {
  INITIAL_BASELINE_VERSION, initialBaselineSourceKey, initialBaselineTargetColumn,
  requestInitialBaselineManifest, requestInitialBaselineVisualization, requestInitialBaselineClusters,
} = await import("../app/lib/initialBaseline.ts");
const { HIERARCHICAL_PCP_LEARNERS, refinementPcpComponentIds, hierarchicalAttributeAxisId } = await import("../app/lib/hierarchicalPcp.ts");
const { REFINEMENT_EMBEDDING_METHODS } = await import("../app/lib/tuningApi.ts");
const {
  REFINEMENT_VISUALIZATION_ALGORITHM, REFINEMENT_CLUSTER_ALGORITHM, REFINEMENT_CLUSTER_FEATURE_BASIS,
} = await import("../app/lib/refinementVisualization.ts");
const sha = (number) => number.toString(16).padStart(64, "0");
let counter = 100;
function manifest() {
  const id = ++counter;
  return {
    available: true, taskId: `task-${id}`, version: INITIAL_BASELINE_VERSION,
    fingerprint: sha(id), baseStateFingerprint: sha(id), sourceFingerprint: sha(id + 200),
    normalizationFingerprint: sha(id + 300), initialModelHoldoutIndependent: true, referenceOnly: false,
    publicationVersion: "publication-test-v1", rowCount: 60, attributeIds: ["cat", "hugging"],
    learnerMethods: [...HIERARCHICAL_PCP_LEARNERS], embeddingMethods: [...REFINEMENT_EMBEDDING_METHODS],
    visualizationVersion: REFINEMENT_VISUALIZATION_ALGORITHM, clusterAlgorithm: REFINEMENT_CLUSTER_ALGORITHM,
    modelSummary: {
      beta: Object.fromEntries(["cat", "hugging"].map((attribute) => [attribute,
        Object.fromEntries(HIERARCHICAL_PCP_LEARNERS.map((method) => [method, 1 / 8]))])),
      gamma: { cat: 1, hugging: 1 },
      embeddingWeights: Object.fromEntries(REFINEMENT_EMBEDDING_METHODS.map((method) => [method, 0.5])),
      embeddingFusionStrength: 0.25,
    },
  };
}
const input = (m) => ({ taskId: m.taskId, rowCount: m.rowCount, attributeIds: m.attributeIds });
const json = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
function visualization(m, overrideHeaders = {}, transform) {
  const columns = 23;
  const values = Float32Array.from({ length: m.rowCount * columns * 2 }, (_, i) => (i % 61) / 61);
  transform?.(values);
  return new Response(values.buffer, { headers: {
    "Content-Type": "application/octet-stream", "X-PCP-Algorithm": REFINEMENT_VISUALIZATION_ALGORITHM,
    "X-PCP-Layout": "score,rank", "X-PCP-Baseline-Fingerprint": m.fingerprint,
    "X-PCP-Base-Fingerprint": m.baseStateFingerprint, "X-PCP-Source-Fingerprint": m.sourceFingerprint,
    "X-PCP-Row-Count": String(m.rowCount), "X-PCP-Component-Count": String(columns), ...overrideHeaders,
  } });
}
function clusterResponse(m, overrideHeaders = {}, labels = Uint8Array.from({ length: m.rowCount }, (_, i) => i % 4)) {
  return new Response(labels, { headers: {
    "Content-Type": "application/octet-stream", "X-PCP-Algorithm": REFINEMENT_CLUSTER_ALGORITHM,
    "X-PCP-Layout": "labels", "X-PCP-Base-Fingerprint": m.baseStateFingerprint,
    "X-PCP-Feature-Basis": REFINEMENT_CLUSTER_FEATURE_BASIS,
    "X-PCP-Baseline-Fingerprint": m.fingerprint, "X-PCP-Source-Fingerprint": m.sourceFingerprint,
    "X-PCP-Row-Count": String(m.rowCount), "X-PCP-Feature-Count": "23",
    "X-PCP-Cluster-Count": "4", "X-PCP-Fit-Row-Count": "40",
    "X-PCP-Fit-Scope": "development", "X-PCP-Cluster-Scheme": "absolute", ...overrideHeaders,
  } });
}
async function mockedFetch(callback) {
  const oldFetch = globalThis.fetch;
  try { await callback((mock) => { globalThis.fetch = mock; }); } finally { globalThis.fetch = oldFetch; }
}

test("unpublished and older services are pending, never a synthetic F0", async () => {
  const m = manifest();
  await mockedFetch(async (install) => {
    install(async () => json({ taskId: m.taskId, available: false }));
    assert.equal(await requestInitialBaselineManifest(input(m)), null);
    install(async () => json({ error: "Endpoint not found" }, 404));
    assert.equal(await requestInitialBaselineManifest(input(m)), null);
    install(async () => json({ taskId: "other", available: false }));
    await assert.rejects(requestInitialBaselineManifest(input(m)), /another task/);
  });
});

test("Val-calibrated publication explicitly declares selection provenance and cutoff", async () => {
  const m = { ...manifest(), initialModelHoldoutIndependent: false, calibrationUsedValidation: true,
    calibration: "full-only-per-task-vqa-val-ap-gates-val-f1-cutoff-v1", classificationThreshold: 0.38 };
  await mockedFetch(async (install) => {
    install(async () => json(m));
    assert.deepEqual(await requestInitialBaselineManifest(input(m)), m);
    for (const change of [{ classificationThreshold: null }, { calibrationUsedValidation: false },
      { initialModelHoldoutIndependent: true }, { calibration: "unknown" }]) {
      install(async () => json({ ...m, ...change }));
      await assert.rejects(requestInitialBaselineManifest(input(m)), /identity/);
    }
  });
});

test("F0 manifest pins ordered models, all initial weights and publication identity", async () => {
  const m = manifest();
  await mockedFetch(async (install) => {
    install(async (url, options) => {
      assert.equal(url, `/api/tuning/tasks/${m.taskId}/initial-baseline`);
      assert.equal(options.credentials, "same-origin");
      assert.equal(options.cache, "no-store");
      return json(m);
    });
    assert.deepEqual(await requestInitialBaselineManifest(input(m)), m);
    const variants = [
      { ...m, available: "true" }, { ...m, version: "legacy" }, { ...m, rowCount: 61 },
      { ...m, sourceFingerprint: "missing" }, { ...m, attributeIds: ["hugging", "cat"] },
      { ...m, baseStateFingerprint: sha(999) },
      { ...m, initialModelHoldoutIndependent: false }, { ...m, referenceOnly: true },
      { ...m, learnerMethods: [...m.learnerMethods].reverse() },
      { ...m, embeddingMethods: [...m.embeddingMethods].reverse() },
      { ...m, embeddingMethods: ["Image Prototype", ...m.embeddingMethods] },
      { ...m, modelSummary: { ...m.modelSummary, gamma: { cat: 0.5, hugging: 1.5 } } },
      { ...m, modelSummary: { ...m.modelSummary, embeddingFusionStrength: 0.5 } },
    ];
    for (const value of variants) {
      install(async () => json(value));
      await assert.rejects(requestInitialBaselineManifest(input(m)), /identity|initial weights/);
    }
  });
});

test("F0 Top single attributes and Joint index the exact shared PCP columns", async () => {
  const m = manifest();
  await mockedFetch(async (install) => {
    install(async (url) => {
      assert.equal(url, `/api/tuning/tasks/${m.taskId}/initial-baseline/visualization?fingerprint=${m.fingerprint}`);
      return visualization(m);
    });
    const snapshot = await requestInitialBaselineVisualization(m);
    assert.equal("runId" in snapshot, false, "baseline is not a fabricated personal run");
    assert.equal(snapshot.componentCount, 23);
    assert.equal(initialBaselineTargetColumn(m, "joint"), 0);
    assert.equal(initialBaselineTargetColumn(m, "cat"), 2);
    assert.equal(initialBaselineTargetColumn(m, "hugging"), 11);
    assert.equal(initialBaselineTargetColumn(m, "unknown"), null);
    const componentIds = refinementPcpComponentIds(m.attributeIds, "joint", m.embeddingMethods);
    for (const attribute of m.attributeIds) {
      const topColumn = initialBaselineTargetColumn(m, attribute);
      const pcpColumn = componentIds.indexOf(hierarchicalAttributeAxisId(attribute));
      assert.equal(topColumn, pcpColumn);
      for (let row = 0; row < m.rowCount; row += 1) {
        assert.equal(snapshot.ranks[row * 23 + topColumn], snapshot.ranks[row * 23 + pcpColumn]);
      }
    }
    install(async () => { throw new Error("cached snapshot should not be downloaded again"); });
    assert.equal(await requestInitialBaselineVisualization(m), snapshot);
  });
});

test("F0 rejects wrong binary identities, dimensions, range and truncation", async () => {
  await mockedFetch(async (install) => {
    for (const headers of [
      { "X-PCP-Baseline-Fingerprint": sha(999) }, { "X-PCP-Base-Fingerprint": sha(999) },
      { "X-PCP-Source-Fingerprint": sha(999) }, { "X-PCP-Component-Count": "26" },
      { "X-PCP-Layout": "rank,score" }, { "X-PCP-Row-Count": "61" },
    ]) {
      const m = manifest(); install(async () => visualization(m, headers));
      await assert.rejects(requestInitialBaselineVisualization(m), /mismatch|dimensions/);
    }
    const m = manifest();
    install(async () => visualization(m, {}, (values) => { values[2] = Number.NaN; }));
    await assert.rejects(requestInitialBaselineVisualization(m), /finite/);
    install(async () => { const full = visualization(m); return new Response(new ArrayBuffer(4), { headers: full.headers }); });
    await assert.rejects(requestInitialBaselineVisualization(m), /wrong size/);
  });
});

test("F0 cluster cache never crosses task, publication, model source or scheme", async () => {
  const m = manifest();
  await mockedFetch(async (install) => {
    install(async () => visualization(m));
    const snapshot = await requestInitialBaselineVisualization(m);
    for (const headers of [
      { "X-PCP-Baseline-Fingerprint": sha(999) }, { "X-PCP-Source-Fingerprint": sha(999) },
      { "X-PCP-Base-Fingerprint": sha(999) }, { "X-PCP-Layout": "score,rank" },
      { "X-PCP-Feature-Count": "22" }, { "X-PCP-Fit-Scope": "test" },
      { "X-PCP-Cluster-Scheme": "shape" }, { "X-PCP-Cluster-Count": "50" },
    ]) {
      install(async () => clusterResponse(m, headers));
      await assert.rejects(requestInitialBaselineClusters(m, snapshot, "absolute"), /mismatch|dimensions/);
    }
    install(async () => clusterResponse(m, {}, new Uint8Array(m.rowCount).fill(4)));
    await assert.rejects(requestInitialBaselineClusters(m, snapshot, "absolute"), /labels/);
    install(async (url) => {
      assert.equal(url, `/api/tuning/tasks/${m.taskId}/initial-baseline/clusters?fingerprint=${m.fingerprint}&scheme=absolute`);
      return clusterResponse(m);
    });
    const result = await requestInitialBaselineClusters(m, snapshot, "absolute");
    assert.equal(result.baselineFingerprint, m.fingerprint);
    assert.equal("runId" in result, false);
    assert.equal(result.labels.length, 60);
    assert.notEqual(initialBaselineSourceKey(m, "absolute"), initialBaselineSourceKey(m, "shape"));
    assert.notEqual(initialBaselineSourceKey(m, "absolute"), initialBaselineSourceKey({ ...m, fingerprint: sha(1) }, "absolute"));
  });
});

test("default F0 is atomic and does not mutate sessions, runs or exported score arrays", async () => {
  const dashboard = await readFile(new URL("../app/Dashboard.tsx", import.meta.url), "utf8");
  const hook = await readFile(new URL("../app/lib/useInitialBaseline.ts", import.meta.url), "utf8");
  assert.doesNotMatch(hook, /createTuningRun|updateAnnotation|useTuningSession|POST|PUT/);
  assert.match(hook, /await requestInitialBaselineVisualization[\s\S]*?await requestInitialBaselineClusters[\s\S]*?requestGeneration !== generation\.current[\s\S]*?data: \{ manifest, snapshot, clusters \}/);
  assert.match(hook, /controller\.abort\(\)/);
  assert.match(dashboard, /const activeInitialBaseline = !legacyBaselineSelected[\s\S]*?!activeTunedOutputMatches && !activeRuntimeFusionResult/);
  assert.match(dashboard, /activePersonalRefinementVisualization\s*\?\? activeInitialBaseline\?\.snapshot/);
  assert.match(dashboard, /activeInitialBaseline\.clusters\.scheme === rankClusterScheme/);
  assert.match(dashboard, /await initialBaseline\.prepareScheme\(scheme\)/);
  assert.match(dashboard, /F₀ pending publication · showing Legacy Ours-Full/);
  assert.match(dashboard, /<option value=\{LEGACY_OURS_FULL_OPTION\}>Legacy Ours-Full/);
  assert.match(dashboard, /`Legacy \$\{methodDisplayLabel\(method\)\}`/);
  assert.match(dashboard, /<span>Diagnostic Filters<\/span>/);
  assert.match(dashboard, /diagnosticSnapshotRequired \? "Current model · snapshot pending" : "Legacy static scores"/);
  assert.match(dashboard, /const diagnosticInitialExpected = !legacyBaselineSelected && rankMethod === "Ours-Full"/);
  assert.match(dashboard, /diagnosticInitialExpected[\s\S]*?Waiting for a valid current model snapshot/);
  assert.match(dashboard, /activePersonalRefinementVisualization\s*\?\? activeInitialBaseline\?\.snapshot \?\? null/);
  assert.match(dashboard, /!legacyBaselineSelected && \([\s\S]*?<TuningFunctionActions/);
  assert.match(dashboard, /initialBaselineTargetColumn\(activeInitialBaseline\.manifest, retrievalTarget\)/);
});
