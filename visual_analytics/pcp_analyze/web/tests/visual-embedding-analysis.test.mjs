import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  clearVisualEmbeddingAnalysisMemoryCache,
  loadVisualEmbeddingAnalysis,
} from "../app/lib/visualEmbeddingAnalysis.ts";

function float32LittleEndian(values) {
  const buffer = new ArrayBuffer(values.length * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return new Uint8Array(buffer);
}

function hex(character) {
  return character.repeat(64);
}

function fixture(rowCount = 120) {
  const pcaValues = Array.from({ length: rowCount * 2 }, (_, index) => index / 10);
  const umapValues = Array.from({ length: rowCount * 2 }, (_, index) => 100 + index / 5);
  const bodies = new Map([
    ["visual-pca.f32", float32LittleEndian(pcaValues)],
    ["visual-umap.f32", float32LittleEndian(umapValues)],
    ["visual-k30.u8", Uint8Array.from({ length: rowCount }, (_, index) => index % 30)],
    ["visual-k50.u8", Uint8Array.from({ length: rowCount }, (_, index) => index % 50)],
    ["visual-k100.u8", Uint8Array.from({ length: rowCount }, (_, index) => index % 100)],
  ]);
  const files = {
    visualPca2d: {
      path: "visual-pca.f32",
      dtype: "float32",
      byteOrder: "little",
      layout: "row-major",
      shape: [rowCount, 2],
      bytes: rowCount * 2 * 4,
      sha256: hex("1"),
    },
    visualUmap2d: {
      path: "visual-umap.f32",
      dtype: "float32",
      byteOrder: "little",
      layout: "row-major",
      shape: [rowCount, 2],
      bytes: rowCount * 2 * 4,
      sha256: hex("2"),
    },
    visualFine30Labels: {
      path: "visual-k30.u8",
      dtype: "uint8",
      byteOrder: "not-applicable",
      layout: "row-major",
      shape: [rowCount],
      bytes: rowCount,
      sha256: hex("3"),
    },
    visualFine50Labels: {
      path: "visual-k50.u8",
      dtype: "uint8",
      byteOrder: "not-applicable",
      layout: "row-major",
      shape: [rowCount],
      bytes: rowCount,
      sha256: hex("4"),
    },
    visualFine100Labels: {
      path: "visual-k100.u8",
      dtype: "uint8",
      byteOrder: "not-applicable",
      layout: "row-major",
      shape: [rowCount],
      bytes: rowCount,
      sha256: hex("5"),
    },
  };
  return {
    bodies,
    pcaValues,
    umapValues,
    source: {
      taskId: "visual-static-task",
      dataRoot: "/data/tasks/visual-static-task",
      rowCount,
      files,
      manifest: {
        contractVersion: 1,
        available: true,
        loading: "lazy-static",
        algorithmVersion: "visual-embedding-analysis-v3",
        backbone: "siglip",
        embeddingDimension: 768,
        fitScope: "development",
        fitMaskFileKey: "developmentMask",
        fitRowCount: 80,
        assignmentScope: "Development fit; held-out transform/predict",
        groundTruthUsed: false,
        pca: {
          fileKey: "visualPca2d",
          explainedVarianceRatio: [0.42, 0.18],
        },
        umap: {
          fileKey: "visualUmap2d",
          inputDimension: 50,
          nNeighbors: 30,
          minDist: 0.1,
          metric: "cosine",
          randomState: 42,
        },
        clusterings: [30, 50, 100].map((clusters) => ({
          id: `fine${clusters}`,
          label: "Fine-grained",
          clusters,
          labelsFileKey: `visualFine${clusters}Labels`,
        })),
      },
    },
  };
}

function installStaticFetch(bodies, calls) {
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    calls.push(url);
    const filename = url.split("/").at(-1).split("?")[0];
    const body = bodies.get(filename);
    return body
      ? new Response(body)
      : new Response("Not found", { status: 404 });
  };
  return () => {
    globalThis.fetch = previousFetch;
  };
}

test("visual analysis loads PCA, UMAP and labels from immutable static files", async () => {
  clearVisualEmbeddingAnalysisMemoryCache();
  const sample = fixture();
  const calls = [];
  const restore = installStaticFetch(sample.bodies, calls);
  try {
    const result = await loadVisualEmbeddingAnalysis(sample.source, 30);
    assert.equal(result.rowCount, 120);
    assert.equal(result.pca2d.length, 240);
    assert.equal(result.umap2d.length, 240);
    assert.equal(result.labels.length, 120);
    assert.equal(result.staticExport, true);
    assert.ok(Math.abs(result.pca2d[3] - sample.pcaValues[3]) < 1e-6);
    assert.ok(Math.abs(result.umap2d[3] - sample.umapValues[3]) < 1e-5);
    assert.deepEqual([...result.labels.slice(0, 30)], Array.from({ length: 30 }, (_, index) => index));
    assert.equal(calls.length, 3);
    assert.ok(calls.every((url) => url.startsWith("/data/tasks/visual-static-task/visual-")));
    assert.ok(calls.every((url) => !url.includes("/api/tuning/")));
  } finally {
    restore();
  }
});

test("switching visual K reuses static projections and fetches only the new labels", async () => {
  clearVisualEmbeddingAnalysisMemoryCache();
  const sample = fixture();
  const calls = [];
  const restore = installStaticFetch(sample.bodies, calls);
  try {
    const k30 = await loadVisualEmbeddingAnalysis(sample.source, 30);
    const k50 = await loadVisualEmbeddingAnalysis(sample.source, 50);
    assert.strictEqual(k30.pca2d, k50.pca2d);
    assert.strictEqual(k30.umap2d, k50.umap2d);
    assert.equal(calls.filter((url) => url.includes("visual-pca.f32")).length, 1);
    assert.equal(calls.filter((url) => url.includes("visual-umap.f32")).length, 1);
    assert.equal(calls.filter((url) => url.includes("visual-k30.u8")).length, 1);
    assert.equal(calls.filter((url) => url.includes("visual-k50.u8")).length, 1);
  } finally {
    restore();
  }
});

test("visual analysis rejects truncated static files and invalid labels", async () => {
  clearVisualEmbeddingAnalysisMemoryCache();
  const truncated = fixture();
  truncated.bodies.set("visual-umap.f32", new Uint8Array(4));
  let calls = [];
  let restore = installStaticFetch(truncated.bodies, calls);
  try {
    await assert.rejects(
      loadVisualEmbeddingAnalysis(truncated.source, 30),
      /invalid byte length/,
    );
  } finally {
    restore();
  }

  clearVisualEmbeddingAnalysisMemoryCache();
  const invalid = fixture();
  invalid.bodies.get("visual-k30.u8")[0] = 30;
  calls = [];
  restore = installStaticFetch(invalid.bodies, calls);
  try {
    await assert.rejects(
      loadVisualEmbeddingAnalysis(invalid.source, 30),
      /out-of-range cluster label/,
    );
  } finally {
    restore();
  }
});

test("Dashboard permanently assigns visual coordinates and labels to ProjectionScatter", async () => {
  const dashboard = await readFile(
    new URL("../app/Dashboard.tsx", import.meta.url),
    "utf8",
  );
  const loader = await readFile(
    new URL("../app/lib/visualEmbeddingAnalysis.ts", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(dashboard, /type ClusterBasis|clusterBasis|Cluster basis/);
  assert.match(dashboard, /activeVisualAnalysis\.umap2d/);
  assert.match(dashboard, /const pcaSource = activeVisualAnalysis\.pca2d/);
  assert.match(dashboard, /const umapSource = activeVisualAnalysis\.umap2d/);
  assert.match(dashboard, /cluster: visualClusterLabels\[index\]/);
  assert.match(dashboard, /umap: umapSource[\s\S]*?umapSource\[index \* 2\]/);
  assert.match(dashboard, /<option value="umap">UMAP<\/option>/);
  assert.match(dashboard, /const activePcpHighlightMask = hasPcpRestriction \? pcpEligibleMask : null/);
  assert.match(dashboard, /highlightMask=\{activePcpHighlightMask\}/);
  assert.match(dashboard, /clusterColors=\{resolveVisualClusterColor\}/);
  assert.match(dashboard, /cluster: activeVisualAnalysis \? visualClusterLabels\[rowIndex\] : undefined/);
  assert.match(dashboard, /Loading precomputed visual embedding/);
  assert.doesNotMatch(loader, /\/api\/tuning\/visual-analysis/);
});

test("rank PCP and visual embedding clustering use independent controls and filters", async () => {
  const dashboard = await readFile(
    new URL("../app/Dashboard.tsx", import.meta.url),
    "utf8",
  );
  assert.match(dashboard, /const \[rankClusterScheme, setRankClusterScheme\]/);
  assert.match(dashboard, /const \[rankClusterFilter, setRankClusterFilter\]/);
  assert.match(dashboard, /const \[visualClusterScheme, setVisualClusterScheme\]/);
  assert.match(dashboard, /const \[visualClusterFilter, setVisualClusterFilter\]/);
  assert.match(dashboard, /labels=\{rankClusterLabels\}/);
  assert.match(dashboard, /labels=\{rankClusterLabels\}[\s\S]*?activeClusters=\{rankClusterSummaries/);
  assert.match(dashboard, /intersectMasks\(pcpEligibleMask, visualStructuralMask\)/);
  assert.match(dashboard, /manifest\.visualEmbedding\.clusterings/);
  assert.match(dashboard, /changeVisualClusterScheme[\s\S]*?setProjectionSelection\(null\)/);
  const visualToggle = dashboard.slice(
    dashboard.indexOf("const toggleVisualClusterSummary"),
    dashboard.indexOf("const openClusterDrilldown"),
  );
  assert.doesNotMatch(
    visualToggle,
    /setProjectionSelection\(null\)/,
    "visual-cluster selection and embedding rectangle are independent conditions",
  );
  assert.match(dashboard, /candidateMask=\{projectionEligibleMask\}[\s\S]*?boxSelectionMask=\{analysisMask\}/);
  assert.doesNotMatch(
    dashboard.slice(
      dashboard.indexOf("const changeVisualClusterScheme"),
      dashboard.indexOf("const changeDataset"),
    ),
    /clearPcpSelection|setBrushes|setPcpSelection/,
  );
});
