"use client";

import { useEffect, useMemo, useState } from "react";

export type VisualClusterCount = 30 | 50 | 100;
export type VisualClusterScheme = "fine30" | "fine50" | "fine100";

export interface VisualEmbeddingFileSpec {
  path: string;
  dtype?: string;
  byteOrder?: string;
  layout?: string;
  shape: number[];
  bytes?: number;
  sha256?: string;
}

export interface VisualEmbeddingClustering {
  id: VisualClusterScheme;
  label: string;
  clusters: VisualClusterCount;
  labelsFileKey: string;
  algorithm?: string;
  featureSpace?: string;
}

export interface VisualEmbeddingManifest {
  contractVersion: number;
  available: boolean;
  loading: "lazy-static";
  algorithmVersion: string;
  backbone: "siglip";
  embeddingDimension: number;
  fitScope: "development";
  fitMaskFileKey: "developmentMask";
  fitRowCount: number;
  assignmentScope: string;
  groundTruthUsed: false;
  pca: {
    fileKey: string;
    explainedVarianceRatio: [number, number];
  };
  umap: {
    fileKey: string;
    inputDimension: number;
    nNeighbors: number;
    minDist: number;
    metric: string;
    randomState: number;
  };
  clusterings: VisualEmbeddingClustering[];
}

export interface VisualEmbeddingSource {
  taskId: string;
  dataRoot: string;
  rowCount: number;
  manifest: VisualEmbeddingManifest;
  files: Record<string, VisualEmbeddingFileSpec | undefined>;
}

export interface VisualEmbeddingAnalysis {
  taskId: string;
  clusterCount: VisualClusterCount;
  rowCount: number;
  fitRowCount: number;
  embeddingDimension: number;
  algorithmVersion: string;
  explainedVarianceRatio: readonly [number, number];
  /** Row-major [image, xy] coordinates fitted on Development. */
  pca2d: Float32Array;
  /** Row-major Development-fitted UMAP [image, xy] coordinates. */
  umap2d: Float32Array;
  /** Development-fitted MiniBatchKMeans assignment for every aligned row. */
  labels: Uint8Array;
  staticExport: true;
}

export interface VisualEmbeddingAnalysisState {
  status: "idle" | "loading" | "ready" | "error";
  data: VisualEmbeddingAnalysis | null;
  error: Error | null;
}

interface ProjectionData {
  pca2d: Float32Array;
  umap2d: Float32Array;
}

const MAX_PROJECTION_CACHE_ENTRIES = 4;
const MAX_LABEL_CACHE_ENTRIES = 12;
const MAX_ANALYSIS_CACHE_ENTRIES = 12;
const projectionCache = new Map<string, ProjectionData>();
const labelCache = new Map<string, Uint8Array>();
const completedCache = new Map<string, VisualEmbeddingAnalysis>();

function remember<K, V>(cache: Map<K, V>, key: K, value: V, maximum: number) {
  cache.delete(key);
  cache.set(key, value);
  while (cache.size > maximum) {
    const oldest = cache.keys().next().value as K | undefined;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

function normalizeDataRoot(dataRoot: string) {
  const rooted = dataRoot.startsWith("/") ? dataRoot : `/${dataRoot}`;
  return rooted.replace(/\/$/, "");
}

function versionedDataPath(dataRoot: string, spec: VisualEmbeddingFileSpec) {
  const parts = spec.path.replaceAll("\\", "/").split("/");
  if (spec.path.startsWith("/") || parts.includes("..") || parts.some((part) => !part)) {
    throw new Error("Visual embedding manifest contains an unsafe static path.");
  }
  const version = typeof spec.sha256 === "string" ? spec.sha256.slice(0, 12) : "unversioned";
  return `${normalizeDataRoot(dataRoot)}/${parts.join("/")}?v=${version}`;
}

function sourcePrefix(source: VisualEmbeddingSource) {
  return `${source.taskId}\u0000${normalizeDataRoot(source.dataRoot)}\u0000`;
}

function requiredFile(
  source: VisualEmbeddingSource,
  fileKey: string,
  dtype: "float32" | "uint8",
  shape: readonly number[],
) {
  const spec = source.files[fileKey];
  const expectedBytes = shape.reduce((size, value) => size * value, 1)
    * (dtype === "float32" ? Float32Array.BYTES_PER_ELEMENT : Uint8Array.BYTES_PER_ELEMENT);
  if (
    !spec
    || spec.dtype !== dtype
    || spec.layout !== "row-major"
    || spec.shape.length !== shape.length
    || spec.shape.some((value, index) => value !== shape[index])
    || spec.bytes !== expectedBytes
    || typeof spec.sha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(spec.sha256)
    || (dtype === "float32" && spec.byteOrder !== "little")
    || (dtype === "uint8" && spec.byteOrder !== "not-applicable")
  ) {
    throw new Error(`Visual embedding file ${fileKey} does not match its static contract.`);
  }
  return spec;
}

function clusteringFor(source: VisualEmbeddingSource, clusterCount: VisualClusterCount) {
  const clustering = source.manifest.clusterings.find(
    (candidate) => candidate.clusters === clusterCount,
  );
  if (!clustering || clustering.id !== `fine${clusterCount}`) {
    throw new Error(`Visual embedding manifest does not provide K${clusterCount}.`);
  }
  return clustering;
}

function validateSource(source: VisualEmbeddingSource) {
  const manifest = source.manifest;
  if (
    manifest.contractVersion !== 1
    || manifest.available !== true
    || manifest.loading !== "lazy-static"
    || manifest.backbone !== "siglip"
    || manifest.fitScope !== "development"
    || manifest.fitMaskFileKey !== "developmentMask"
    || manifest.groundTruthUsed !== false
    || !Number.isSafeInteger(source.rowCount)
    || source.rowCount <= 0
    || !Number.isSafeInteger(manifest.fitRowCount)
    || manifest.fitRowCount <= 0
    || manifest.fitRowCount > source.rowCount
    || !Number.isSafeInteger(manifest.embeddingDimension)
    || manifest.embeddingDimension <= 1
    || !manifest.algorithmVersion
    || manifest.clusterings.length !== 3
  ) {
    throw new Error("Visual embedding manifest has an invalid static analysis contract.");
  }
}

function projectionKey(
  source: VisualEmbeddingSource,
  pcaSpec: VisualEmbeddingFileSpec,
  umapSpec: VisualEmbeddingFileSpec,
) {
  return `${sourcePrefix(source)}${source.manifest.algorithmVersion}\u0000${pcaSpec.sha256}\u0000${umapSpec.sha256}`;
}

function labelsKey(
  source: VisualEmbeddingSource,
  clusterCount: VisualClusterCount,
  spec: VisualEmbeddingFileSpec,
) {
  return `${sourcePrefix(source)}${source.manifest.algorithmVersion}\u0000${clusterCount}\u0000${spec.sha256}`;
}

function analysisKey(source: VisualEmbeddingSource, clusterCount: VisualClusterCount) {
  const pcaSpec = source.files[source.manifest.pca.fileKey];
  const umapSpec = source.files[source.manifest.umap.fileKey];
  const labelSpec = source.files[clusteringFor(source, clusterCount).labelsFileKey];
  return `${sourcePrefix(source)}${source.manifest.algorithmVersion}\u0000${clusterCount}\u0000${pcaSpec?.sha256 ?? ""}\u0000${umapSpec?.sha256 ?? ""}\u0000${labelSpec?.sha256 ?? ""}`;
}

async function fetchStaticBuffer(
  source: VisualEmbeddingSource,
  spec: VisualEmbeddingFileSpec,
  signal?: AbortSignal,
) {
  const path = versionedDataPath(source.dataRoot, spec);
  const response = await fetch(path, { signal, cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`Unable to load precomputed visual embedding ${spec.path} (${response.status}).`);
  }
  const buffer = await response.arrayBuffer();
  signal?.throwIfAborted();
  if (buffer.byteLength !== spec.bytes) {
    throw new Error(`Precomputed visual embedding ${spec.path} has an invalid byte length.`);
  }
  return buffer;
}

function littleEndianFloat32(buffer: ArrayBuffer, valueCount: number) {
  const nativeLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  if (nativeLittleEndian) return new Float32Array(buffer, 0, valueCount);
  const values = new Float32Array(valueCount);
  const view = new DataView(buffer);
  for (let index = 0; index < valueCount; index += 1) {
    values[index] = view.getFloat32(index * 4, true);
  }
  return values;
}

function assertFiniteCoordinates(values: Float32Array, label: string) {
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error(`Precomputed visual ${label} contains a non-finite coordinate.`);
    }
  }
}

async function loadProjections(source: VisualEmbeddingSource, signal?: AbortSignal) {
  const pcaSpec = requiredFile(
    source,
    source.manifest.pca.fileKey,
    "float32",
    [source.rowCount, 2],
  );
  const umapSpec = requiredFile(
    source,
    source.manifest.umap.fileKey,
    "float32",
    [source.rowCount, 2],
  );
  const key = projectionKey(source, pcaSpec, umapSpec);
  const cached = projectionCache.get(key);
  if (cached) {
    signal?.throwIfAborted();
    remember(projectionCache, key, cached, MAX_PROJECTION_CACHE_ENTRIES);
    return cached;
  }
  const [pcaBuffer, umapBuffer] = await Promise.all([
    fetchStaticBuffer(source, pcaSpec, signal),
    fetchStaticBuffer(source, umapSpec, signal),
  ]);
  const pca2d = littleEndianFloat32(pcaBuffer, source.rowCount * 2);
  const umap2d = littleEndianFloat32(umapBuffer, source.rowCount * 2);
  assertFiniteCoordinates(pca2d, "PCA");
  assertFiniteCoordinates(umap2d, "UMAP");
  const result = { pca2d, umap2d };
  remember(projectionCache, key, result, MAX_PROJECTION_CACHE_ENTRIES);
  return result;
}

async function loadLabels(
  source: VisualEmbeddingSource,
  clusterCount: VisualClusterCount,
  signal?: AbortSignal,
) {
  const clustering = clusteringFor(source, clusterCount);
  const spec = requiredFile(
    source,
    clustering.labelsFileKey,
    "uint8",
    [source.rowCount],
  );
  const key = labelsKey(source, clusterCount, spec);
  const cached = labelCache.get(key);
  if (cached) {
    signal?.throwIfAborted();
    remember(labelCache, key, cached, MAX_LABEL_CACHE_ENTRIES);
    return cached;
  }
  const labels = new Uint8Array(await fetchStaticBuffer(source, spec, signal));
  const observed = new Uint8Array(clusterCount);
  for (const label of labels) {
    if (label >= clusterCount) {
      throw new Error("Precomputed visual embedding contains an out-of-range cluster label.");
    }
    observed[label] = 1;
  }
  if (observed.some((value) => value === 0)) {
    throw new Error("Precomputed visual embedding is missing at least one cluster label.");
  }
  remember(labelCache, key, labels, MAX_LABEL_CACHE_ENTRIES);
  return labels;
}

/** Load one immutable task-level visual analysis directly from static files. */
export async function loadVisualEmbeddingAnalysis(
  source: VisualEmbeddingSource,
  clusterCount: VisualClusterCount,
  signal?: AbortSignal,
): Promise<VisualEmbeddingAnalysis> {
  validateSource(source);
  const key = analysisKey(source, clusterCount);
  const cached = completedCache.get(key);
  if (cached) {
    signal?.throwIfAborted();
    remember(completedCache, key, cached, MAX_ANALYSIS_CACHE_ENTRIES);
    return cached;
  }
  const [projection, labels] = await Promise.all([
    loadProjections(source, signal),
    loadLabels(source, clusterCount, signal),
  ]);
  const explained = source.manifest.pca.explainedVarianceRatio;
  if (
    explained.length !== 2
    || explained.some((value) => !Number.isFinite(value) || value < 0)
  ) {
    throw new Error("Visual embedding manifest has invalid PCA variance metadata.");
  }
  const result: VisualEmbeddingAnalysis = {
    taskId: source.taskId,
    clusterCount,
    rowCount: source.rowCount,
    fitRowCount: source.manifest.fitRowCount,
    embeddingDimension: source.manifest.embeddingDimension,
    algorithmVersion: source.manifest.algorithmVersion,
    explainedVarianceRatio: [explained[0], explained[1]],
    pca2d: projection.pca2d,
    umap2d: projection.umap2d,
    labels,
    staticExport: true,
  };
  remember(completedCache, key, result, MAX_ANALYSIS_CACHE_ENTRIES);
  return result;
}

export function clearVisualEmbeddingAnalysisMemoryCache(taskId?: string) {
  if (!taskId) {
    projectionCache.clear();
    labelCache.clear();
    completedCache.clear();
    return;
  }
  const prefix = `${taskId}\u0000`;
  for (const cache of [projectionCache, labelCache, completedCache]) {
    for (const key of cache.keys()) {
      if (key.startsWith(prefix)) cache.delete(key);
    }
  }
}

/** Lazily loads static visual data only while its UI basis is active. */
export function useVisualEmbeddingAnalysis(
  source: VisualEmbeddingSource | null,
  clusterCount: VisualClusterCount,
  enabled: boolean,
): VisualEmbeddingAnalysisState {
  const key = useMemo(
    () => source ? analysisKey(source, clusterCount) : null,
    [clusterCount, source],
  );
  const [settled, setSettled] = useState<{
    key: string;
    data: VisualEmbeddingAnalysis | null;
    error: Error | null;
  } | null>(null);

  useEffect(() => {
    if (!enabled || !source || !key) return;
    const cached = completedCache.get(key);
    if (cached) return;
    const controller = new AbortController();
    loadVisualEmbeddingAnalysis(source, clusterCount, controller.signal)
      .then((data) => setSettled({ key, data, error: null }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setSettled({
          key,
          data: null,
          error: error instanceof Error ? error : new Error(String(error)),
        });
      });
    return () => controller.abort();
  }, [clusterCount, enabled, key, source]);

  if (!enabled || !source || !key) {
    return { status: "idle", data: null, error: null };
  }
  const cached = completedCache.get(key);
  if (cached) return { status: "ready", data: cached, error: null };
  if (settled?.key === key && settled.error) {
    return { status: "error", data: null, error: settled.error };
  }
  if (settled?.key === key && settled.data) {
    return { status: "ready", data: settled.data, error: null };
  }
  return { status: "loading", data: null, error: null };
}
