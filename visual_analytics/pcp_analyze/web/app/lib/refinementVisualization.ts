"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { refinementEmbeddingMethods, type TuningRun } from "./tuningApi";

export const REFINEMENT_VISUALIZATION_ALGORITHM =
  "conjunction-holistic-visualization-v2";
export const REFINEMENT_VISUALIZATION_LAYOUT = ["score", "rank"] as const;
export const REFINEMENT_CLUSTER_ALGORITHM =
  "dynamic-refinement-pcp-clusters-v2";
export const REFINEMENT_CLUSTER_FEATURE_BASIS =
  "exact-refinement-visible-rank-profile-v2";

export interface RefinementVisualizationSnapshot {
  runId: string;
  rowCount: number;
  componentCount: number;
  scores: Float32Array;
  ranks: Float32Array;
  sourceFingerprint: string;
  baseFingerprint: string;
  attributeIds: readonly string[];
  embeddingMethods: readonly string[];
}

export interface RefinementClusterResult {
  runId: string;
  rowCount: number;
  clusterCount: number;
  featureCount: number;
  fitRowCount: number;
  scheme: string;
  sourceFingerprint: string;
  labels: Uint8Array;
}

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;
const IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const clusterCache = new Map<string, RefinementClusterResult>();

export class RefinementVisualizationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RefinementVisualizationError";
  }
}

function requiredHeader(response: Response, name: string) {
  const value = response.headers.get(name)?.trim();
  if (!value) throw new RefinementVisualizationError(`Missing ${name}.`);
  return value;
}

function integerHeader(response: Response, name: string) {
  const value = Number(requiredHeader(response, name));
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RefinementVisualizationError(`Invalid ${name}.`);
  }
  return value;
}

function floatSlice(buffer: ArrayBuffer, offset: number, length: number) {
  if (IS_LITTLE_ENDIAN) return new Float32Array(buffer, offset, length);
  const view = new DataView(buffer, offset, length * FLOAT_BYTES);
  const values = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    values[index] = view.getFloat32(index * FLOAT_BYTES, true);
  }
  return values;
}

async function responseError(response: Response) {
  let message = `Refinement visualization returned ${response.status}.`;
  try {
    const payload = await response.json() as { error?: unknown };
    if (typeof payload.error === "string" && payload.error.trim()) {
      message = payload.error.trim();
    }
  } catch {
    // Retain the status fallback for non-JSON responses.
  }
  return new RefinementVisualizationError(message);
}

function assertBinaryResponse(response: Response) {
  if (!(response.headers.get("Content-Type") ?? "").toLowerCase().startsWith("application/octet-stream")) {
    throw new RefinementVisualizationError("Refinement response is not binary data.");
  }
}

export async function requestRefinementVisualization(
  run: TuningRun,
  expectedRowCount: number,
  options: { signal?: AbortSignal } = {},
): Promise<RefinementVisualizationSnapshot> {
  if (!run.visualizationUrl) {
    throw new RefinementVisualizationError("This run has no refinement visualization snapshot.");
  }
  const embeddingMethods = refinementEmbeddingMethods(run);
  const attributeIds = run.attributeIds;
  if (!embeddingMethods || !attributeIds?.length
    || new Set(attributeIds).size !== attributeIds.length
    || attributeIds.some((id) => typeof id !== "string" || !id.trim())) {
    throw new RefinementVisualizationError("Invalid run-pinned refinement method identities.");
  }
  const expectedComponentCount = 3 + attributeIds.length * 9 + embeddingMethods.length;
  const response = await fetch(run.visualizationUrl, {
    credentials: "same-origin",
    headers: { Accept: "application/octet-stream" },
    signal: options.signal,
  });
  if (!response.ok) throw await responseError(response);
  assertBinaryResponse(response);
  if (requiredHeader(response, "X-PCP-Algorithm") !== REFINEMENT_VISUALIZATION_ALGORITHM) {
    throw new RefinementVisualizationError("Unsupported refinement visualization algorithm.");
  }
  if (requiredHeader(response, "X-PCP-Layout") !== REFINEMENT_VISUALIZATION_LAYOUT.join(",")) {
    throw new RefinementVisualizationError("Unsupported refinement visualization layout.");
  }
  if (requiredHeader(response, "X-PCP-Run-Id") !== run.id) {
    throw new RefinementVisualizationError("Refinement visualization belongs to another run.");
  }
  const rowCount = integerHeader(response, "X-PCP-Row-Count");
  const componentCount = integerHeader(response, "X-PCP-Component-Count");
  if (rowCount !== expectedRowCount) {
    throw new RefinementVisualizationError("Refinement visualization row count changed.");
  }
  if (componentCount !== expectedComponentCount) {
    throw new RefinementVisualizationError("Refinement component count does not match this run's methods.");
  }
  const sourceFingerprint = requiredHeader(response, "X-PCP-Source-Fingerprint");
  const baseFingerprint = requiredHeader(response, "X-PCP-Base-Fingerprint");
  const valueCount = rowCount * componentCount;
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== valueCount * REFINEMENT_VISUALIZATION_LAYOUT.length * FLOAT_BYTES) {
    throw new RefinementVisualizationError("Refinement visualization payload has the wrong size.");
  }
  const scores = floatSlice(buffer, 0, valueCount);
  const ranks = floatSlice(buffer, valueCount * FLOAT_BYTES, valueCount);
  if (scores.some((value) => !Number.isFinite(value) || value < 0 || value > 1)
    || ranks.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new RefinementVisualizationError("Refinement scores and ranks must be finite values in [0, 1].");
  }
  return {
    runId: run.id,
    rowCount,
    componentCount,
    scores,
    ranks,
    sourceFingerprint,
    baseFingerprint,
    attributeIds: [...attributeIds],
    embeddingMethods: [...embeddingMethods],
  };
}

function clusterSourceKey(
  runId: string,
  sourceFingerprint: string,
  scheme: string,
) {
  return ["refinement", runId, sourceFingerprint, scheme].join("|");
}

export async function requestRefinementClusters(input: {
  run: TuningRun;
  snapshot: RefinementVisualizationSnapshot;
  scheme: string;
  signal?: AbortSignal;
}): Promise<RefinementClusterResult> {
  const { run, snapshot, scheme, signal } = input;
  if (!run.clusterUrl) {
    throw new RefinementVisualizationError("This run has no refinement cluster endpoint.");
  }
  const key = clusterSourceKey(run.id, snapshot.sourceFingerprint, scheme);
  const cached = clusterCache.get(key);
  if (cached) return cached;
  const response = await fetch(`${run.clusterUrl}?scheme=${encodeURIComponent(scheme)}`, {
    credentials: "same-origin",
    headers: { Accept: "application/octet-stream" },
    signal,
  });
  if (!response.ok) throw await responseError(response);
  assertBinaryResponse(response);
  if (requiredHeader(response, "X-PCP-Algorithm") !== REFINEMENT_CLUSTER_ALGORITHM) {
    throw new RefinementVisualizationError("Unsupported refinement cluster algorithm.");
  }
  if (requiredHeader(response, "X-PCP-Feature-Basis") !== REFINEMENT_CLUSTER_FEATURE_BASIS) {
    throw new RefinementVisualizationError("Unsupported refinement cluster feature basis.");
  }
  if (
    requiredHeader(response, "X-PCP-Run-Id") !== run.id
    || requiredHeader(response, "X-PCP-Source-Fingerprint") !== snapshot.sourceFingerprint
  ) {
    throw new RefinementVisualizationError("Refinement clusters belong to another model snapshot.");
  }
  const rowCount = integerHeader(response, "X-PCP-Row-Count");
  const clusterCount = integerHeader(response, "X-PCP-Cluster-Count");
  const featureCount = integerHeader(response, "X-PCP-Feature-Count");
  const fitRowCount = integerHeader(response, "X-PCP-Fit-Row-Count");
  if (
    rowCount !== snapshot.rowCount
    || featureCount !== snapshot.componentCount
    || requiredHeader(response, "X-PCP-Cluster-Scheme") !== scheme
    || requiredHeader(response, "X-PCP-Fit-Scope") !== "development"
  ) {
    throw new RefinementVisualizationError("Refinement cluster metadata does not match the PCP snapshot.");
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== rowCount) {
    throw new RefinementVisualizationError("Refinement cluster label count is invalid.");
  }
  const result: RefinementClusterResult = {
    runId: run.id,
    rowCount,
    clusterCount,
    featureCount,
    fitRowCount,
    scheme,
    sourceFingerprint: snapshot.sourceFingerprint,
    labels: new Uint8Array(buffer),
  };
  if (clusterCache.size >= 40) clusterCache.delete(clusterCache.keys().next().value as string);
  clusterCache.set(key, result);
  return result;
}

export function useRefinementClusters(input: {
  run: TuningRun | null;
  snapshot: RefinementVisualizationSnapshot | null;
  scheme: string;
  disabled?: boolean;
}) {
  const { run, snapshot, scheme, disabled = false } = input;
  const sourceKey = run && snapshot
    ? clusterSourceKey(run.id, snapshot.sourceFingerprint, scheme)
    : "refinement|inactive";
  const [state, setState] = useState<{
    sourceKey: string;
    status: "idle" | "loading" | "succeeded" | "error";
    data: RefinementClusterResult | null;
    error: string | null;
  }>({ sourceKey, status: "idle", data: null, error: null });
  const generation = useRef(0);

  const load = useCallback(async () => {
    if (disabled || !run || !snapshot) return null;
    const requestGeneration = ++generation.current;
    setState({ sourceKey, status: "loading", data: null, error: null });
    try {
      const data = await requestRefinementClusters({ run, snapshot, scheme });
      if (requestGeneration !== generation.current) return null;
      setState({ sourceKey, status: "succeeded", data, error: null });
      return data;
    } catch (reason) {
      if (requestGeneration !== generation.current) return null;
      const message = reason instanceof Error ? reason.message : String(reason);
      setState({ sourceKey, status: "error", data: null, error: message });
      return null;
    }
  }, [disabled, run, scheme, snapshot, sourceKey]);

  useEffect(() => {
    generation.current += 1;
    if (disabled || !run || !snapshot) {
      const timer = window.setTimeout(() => {
        setState({ sourceKey, status: "idle", data: null, error: null });
      }, 0);
      return () => window.clearTimeout(timer);
    }
    const cached = clusterCache.get(sourceKey);
    if (cached) {
      const timer = window.setTimeout(() => {
        setState({ sourceKey, status: "succeeded", data: cached, error: null });
      }, 0);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [disabled, load, run, snapshot, sourceKey]);

  const cached = disabled ? null : clusterCache.get(sourceKey) ?? null;
  if (cached && (state.sourceKey !== sourceKey || state.data !== cached)) {
    return { status: "succeeded" as const, data: cached, error: null, sourceKey, reload: load };
  }
  if (state.sourceKey !== sourceKey) {
    return {
      status: disabled ? "idle" as const : "loading" as const,
      data: null,
      error: null,
      sourceKey,
      reload: load,
    };
  }
  return { ...state, sourceKey, reload: load };
}
