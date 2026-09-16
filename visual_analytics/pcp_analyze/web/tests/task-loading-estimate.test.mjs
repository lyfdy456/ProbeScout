import assert from "node:assert/strict";
import test from "node:test";

import {
  estimateLoadTime,
  initialDownloadBytesFromCatalog,
  initialDownloadBytesFromManifest,
  resolveInitialDownloadBytes,
} from "../app/lib/taskLoadingEstimate.ts";

const REQUIRED_KEYS = [
  "metadata",
  "imageIds",
  "rawScores",
  "calibratedScores",
  "ranks",
  "pca2d",
  "metrics",
  "groundTruth",
  "developmentMask",
  "validationMask",
  "testMask",
];
const CLUSTER_FILE_KEYS = [
  "fine30Labels",
  "fine50Labels",
  "fine100Labels",
  "absoluteLabels",
  "shapeLabels",
];

function manifestWithFiles(bytesForKey = () => 100) {
  return {
    files: Object.fromEntries([
      ...REQUIRED_KEYS.map((key) => [key, { path: `${key}.bin`, bytes: bytesForKey(key) }]),
      ...CLUSTER_FILE_KEYS.map((key) => [
        key,
        { path: `${key}.bin`, bytes: bytesForKey(key) },
      ]),
      ["umap2d", { path: "umap.bin", bytes: bytesForKey("umap2d") }],
      ["drawOrder", { path: "draw-order.bin", bytes: 999_999 }],
    ]),
    clusters: {
      schemes: CLUSTER_FILE_KEYS.map((labelsFileKey) => ({ labelsFileKey })),
    },
    projections: { umap: { available: true } },
  };
}

test("sums only files fetched before the workspace becomes interactive", () => {
  const manifest = manifestWithFiles(() => 100);
  const result = initialDownloadBytesFromManifest(manifest);

  assert.equal(result.complete, true);
  assert.equal(result.source, "manifest");
  assert.equal(result.bytes, (REQUIRED_KEYS.length + CLUSTER_FILE_KEYS.length + 1) * 100);
});

test("does not count UMAP when that projection is unavailable", () => {
  const manifest = manifestWithFiles(() => 100);
  manifest.projections.umap.available = false;

  assert.equal(
    initialDownloadBytesFromManifest(manifest).bytes,
    (REQUIRED_KEYS.length + CLUSTER_FILE_KEYS.length) * 100,
  );
});

test("estimates a conservative 3 Mbps range", () => {
  const bytes = 30 * 1024 * 1024;
  const estimate = estimateLoadTime(bytes, 3);

  assert.equal(estimate.minSeconds, 84);
  assert.ok(estimate.maxSeconds > estimate.minSeconds);
  assert.ok(estimate.maxSeconds < 140);
});

test("creates an immediate estimate from optional catalog metadata", () => {
  assert.deepEqual(initialDownloadBytesFromCatalog(42_000), {
    bytes: 42_000,
    complete: true,
    source: "catalog",
    unresolvedFiles: [],
  });
  assert.equal(initialDownloadBytesFromCatalog(undefined), null);
});

test("uses catalog metadata before probing missing file sizes", async () => {
  const manifest = manifestWithFiles((key) => key === "rawScores" ? undefined : 100);
  const result = await resolveInitialDownloadBytes(manifest, {
    dataRoot: "/data/task",
    catalogInitialDownloadBytes: 42_000,
    fetchImpl: async () => {
      throw new Error("HEAD should not run when catalog metadata is complete");
    },
  });

  assert.deepEqual(result, {
    bytes: 42_000,
    complete: true,
    source: "catalog",
    unresolvedFiles: [],
  });
});

test("uses HEAD Content-Length without issuing another body download", async () => {
  const manifest = manifestWithFiles((key) => key === "rawScores" ? undefined : 100);
  const requests = [];
  const result = await resolveInitialDownloadBytes(manifest, {
    dataRoot: "/data/task",
    fetchImpl: async (input, init) => {
      requests.push({ input: String(input), method: init?.method });
      return new Response(null, {
        status: 200,
        headers: { "content-length": "250" },
      });
    },
  });

  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0], {
    input: "/data/task/rawScores.bin",
    method: "HEAD",
  });
  assert.equal(result.complete, true);
  assert.equal(result.source, "manifest+head");
  assert.equal(
    result.bytes,
    (REQUIRED_KEYS.length + CLUSTER_FILE_KEYS.length) * 100 + 250,
  );
});

test("returns a lower bound when HEAD has no usable Content-Length", async () => {
  const manifest = manifestWithFiles((key) => key === "rawScores" ? undefined : 100);
  const result = await resolveInitialDownloadBytes(manifest, {
    dataRoot: "/data/task",
    fetchImpl: async () => new Response(null, { status: 200 }),
  });

  assert.equal(result.complete, false);
  assert.equal(result.source, "partial");
  assert.deepEqual(result.unresolvedFiles, ["rawScores.bin"]);
});

test("reports a cluster scheme without a label-file contract", () => {
  const manifest = manifestWithFiles(() => 100);
  delete manifest.clusters.schemes[0].labelsFileKey;
  const result = initialDownloadBytesFromManifest(manifest);

  assert.equal(result.complete, false);
  assert.ok(result.unresolvedFiles.includes("clusters.schemes[0].labelsFileKey"));
});
