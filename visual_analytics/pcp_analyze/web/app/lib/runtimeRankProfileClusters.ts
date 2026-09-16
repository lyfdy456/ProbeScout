"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  HIERARCHICAL_PCP_METHODS,
  cloneHierarchicalPcpConfig,
  hierarchicalPcpConfigFingerprint,
  normalizeHierarchicalPcpConfig,
  type HierarchicalPcpConfig,
} from "./hierarchicalPcp";

export type RuntimeRankProfileClusterScheme =
  | "fine30"
  | "fine50"
  | "fine100"
  | "absolute"
  | "shape";

export const RUNTIME_RANK_PROFILE_CLUSTER_API_PATH = "/api/tuning/rank-fusion-clusters";
export const RUNTIME_RANK_PROFILE_CLUSTER_ALGORITHM =
  "dynamic-attribute-rank-fusion-clusters-v1";
export const RUNTIME_RANK_PROFILE_CLUSTER_LAYOUT = "labels";
export const RUNTIME_RANK_PROFILE_CLUSTER_FEATURE_BASIS =
  "attribute-hierarchy-13-rank-v1";
export const RUNTIME_RANK_PROFILE_CLUSTER_FIT_SCOPE = "development";
export const RUNTIME_RANK_PROFILE_CLUSTER_COUNTS: Readonly<
  Record<RuntimeRankProfileClusterScheme, number>
> = Object.freeze({
  fine30: 30,
  fine50: 50,
  fine100: 100,
  absolute: 4,
  shape: 3,
});

export interface RuntimeRankProfileClustersInput {
  taskId: string;
  rowCount: number;
  targetCount: number;
  attributeIds: readonly string[];
  config: Readonly<HierarchicalPcpConfig>;
  scheme: RuntimeRankProfileClusterScheme;
  disabled?: boolean;
}

export interface RuntimeRankProfileClustersResult {
  taskId: string;
  rowCount: number;
  targetCount: number;
  attributeIds: readonly string[];
  scheme: RuntimeRankProfileClusterScheme;
  clusterCount: number;
  config: HierarchicalPcpConfig;
  configFingerprint: string;
  labels: Uint8Array;
  cacheKey: string;
  featureCount: number | null;
  fitRowCount: number | null;
}

export type RuntimeRankProfileClustersStatus =
  | "idle"
  | "loading"
  | "succeeded"
  | "error";

export interface RuntimeRankProfileClustersState {
  status: RuntimeRankProfileClustersStatus;
  data: RuntimeRankProfileClustersResult | null;
  error: string | null;
  sourceKey: string;
  reload: () => void;
  clear: () => void;
}

interface PreparedRequest {
  taskId: string;
  rowCount: number;
  targetCount: number;
  attributeIds: readonly string[];
  scheme: RuntimeRankProfileClusterScheme;
  clusterCount: number;
  config: HierarchicalPcpConfig;
  configFingerprint: string;
  cacheKey: string;
}

interface CachedRequest {
  epoch: number;
  promise: Promise<RuntimeRankProfileClustersResult>;
}

const RESULT_CACHE_LIMIT = 24;
const resultCache = new Map<string, RuntimeRankProfileClustersResult>();
const pendingCache = new Map<string, CachedRequest>();
const keyEpochs = new Map<string, number>();
let globalCacheEpoch = 0;

export class RuntimeRankProfileClustersError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "RuntimeRankProfileClustersError";
    this.status = status;
  }
}

function positiveInteger(value: number, label: string) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RuntimeRankProfileClustersError(`${label} must be a positive integer.`);
  }
  return value;
}

function prepareRequest(input: RuntimeRankProfileClustersInput): PreparedRequest {
  const taskId = input.taskId.trim();
  if (!taskId) {
    throw new RuntimeRankProfileClustersError(
      "Runtime rank-profile clustering requires a task ID.",
    );
  }
  const rowCount = positiveInteger(input.rowCount, "rowCount");
  const targetCount = positiveInteger(input.targetCount, "targetCount");
  if (!Object.prototype.hasOwnProperty.call(
    RUNTIME_RANK_PROFILE_CLUSTER_COUNTS,
    input.scheme,
  )) {
    throw new RuntimeRankProfileClustersError(
      `Unsupported runtime rank-profile cluster scheme: ${String(input.scheme)}.`,
    );
  }
  const attributeIds = [...input.attributeIds];
  const config = normalizeHierarchicalPcpConfig(input.config, attributeIds);
  const configFingerprint = hierarchicalPcpConfigFingerprint(config, attributeIds);
  const cacheKey = JSON.stringify([
    taskId,
    rowCount,
    targetCount,
    attributeIds,
    configFingerprint,
    input.scheme,
  ]);
  return {
    taskId,
    rowCount,
    targetCount,
    attributeIds,
    scheme: input.scheme,
    clusterCount: RUNTIME_RANK_PROFILE_CLUSTER_COUNTS[input.scheme],
    config,
    configFingerprint,
    cacheKey,
  };
}

export function runtimeRankProfileClusterKey(input: RuntimeRankProfileClustersInput) {
  return prepareRequest(input).cacheKey;
}

function currentKeyEpoch(cacheKey: string) {
  return keyEpochs.get(cacheKey) ?? 0;
}

function invalidateCacheKey(cacheKey: string) {
  resultCache.delete(cacheKey);
  pendingCache.delete(cacheKey);
  keyEpochs.set(cacheKey, currentKeyEpoch(cacheKey) + 1);
}

function rememberResult(result: RuntimeRankProfileClustersResult) {
  resultCache.delete(result.cacheKey);
  resultCache.set(result.cacheKey, result);
  while (resultCache.size > RESULT_CACHE_LIMIT) {
    const oldest = resultCache.keys().next().value as string | undefined;
    if (oldest === undefined) break;
    resultCache.delete(oldest);
  }
}

function cachedResult(cacheKey: string) {
  const result = resultCache.get(cacheKey) ?? null;
  if (result) {
    resultCache.delete(cacheKey);
    resultCache.set(cacheKey, result);
  }
  return result;
}

export function clearRuntimeRankProfileClustersMemoryCache() {
  globalCacheEpoch += 1;
  resultCache.clear();
  pendingCache.clear();
  keyEpochs.clear();
}

function requiredIntegerHeader(response: Response, name: string) {
  const raw = response.headers.get(name);
  const value = raw === null ? Number.NaN : Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile clustering returned an invalid ${name}.`,
    );
  }
  return value;
}

function optionalIntegerHeader(response: Response, name: string) {
  const raw = response.headers.get(name);
  if (raw === null || raw.trim() === "") return null;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile clustering returned an invalid ${name}.`,
    );
  }
  return value;
}

async function responseError(response: Response) {
  let message = `Runtime rank-profile clustering service returned ${response.status}.`;
  try {
    const payload = await response.json() as { error?: unknown; message?: unknown };
    if (typeof payload.error === "string" && payload.error.trim()) {
      message = payload.error.trim();
    } else if (typeof payload.message === "string" && payload.message.trim()) {
      message = payload.message.trim();
    }
  } catch {
    // Preserve the status-based fallback for non-JSON failures.
  }
  return new RuntimeRankProfileClustersError(message, response.status);
}

async function fetchRuntimeRankProfileClusters(
  prepared: PreparedRequest,
): Promise<RuntimeRankProfileClustersResult> {
  const response = await fetch(RUNTIME_RANK_PROFILE_CLUSTER_API_PATH, {
    method: "POST",
    credentials: "same-origin",
    headers: {
      Accept: "application/octet-stream",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      taskId: prepared.taskId,
      scheme: prepared.scheme,
      attributeWeights: prepared.config.attributeWeights,
      methodWeightsByAttribute: prepared.config.methodWeightsByAttribute,
    }),
  });
  if (!response.ok) throw await responseError(response);
  if (!(response.headers.get("Content-Type") ?? "").toLowerCase().startsWith(
    "application/octet-stream",
  )) {
    throw new RuntimeRankProfileClustersError(
      "Runtime rank-profile clustering response is not an octet-stream.",
    );
  }
  if (response.headers.get("X-PCP-Algorithm") !== RUNTIME_RANK_PROFILE_CLUSTER_ALGORITHM) {
    throw new RuntimeRankProfileClustersError(
      "Unsupported runtime rank-profile clustering algorithm.",
    );
  }
  if (response.headers.get("X-PCP-Layout") !== RUNTIME_RANK_PROFILE_CLUSTER_LAYOUT) {
    throw new RuntimeRankProfileClustersError(
      "Unsupported runtime rank-profile clustering response layout.",
    );
  }
  const rowCount = requiredIntegerHeader(response, "X-PCP-Row-Count");
  const clusterCount = requiredIntegerHeader(response, "X-PCP-Cluster-Count");
  const responseScheme = response.headers.get("X-PCP-Cluster-Scheme");
  if (rowCount !== prepared.rowCount) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile clustering row count ${rowCount} does not match ${prepared.rowCount}.`,
    );
  }
  if (clusterCount !== prepared.clusterCount) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile clustering K${clusterCount} does not match K${prepared.clusterCount}.`,
    );
  }
  if (responseScheme !== prepared.scheme) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile clustering scheme ${String(responseScheme)} does not match ${prepared.scheme}.`,
    );
  }
  const responseTargetCount = optionalIntegerHeader(response, "X-PCP-Target-Count");
  if (responseTargetCount !== null && responseTargetCount !== prepared.targetCount) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile clustering target count ${responseTargetCount} does not match ${prepared.targetCount}.`,
    );
  }
  const featureCount = optionalIntegerHeader(response, "X-PCP-Feature-Count");
  const fitRowCount = optionalIntegerHeader(response, "X-PCP-Fit-Row-Count");
  if (
    response.headers.get("X-PCP-Feature-Basis")
    !== RUNTIME_RANK_PROFILE_CLUSTER_FEATURE_BASIS
  ) {
    throw new RuntimeRankProfileClustersError(
      "Unsupported runtime rank-profile clustering feature basis.",
    );
  }
  if (response.headers.get("X-PCP-Fit-Scope") !== RUNTIME_RANK_PROFILE_CLUSTER_FIT_SCOPE) {
    throw new RuntimeRankProfileClustersError(
      "Runtime rank-profile clustering was not fitted on Development.",
    );
  }
  const expectedFeatureCount = 1
    + prepared.attributeIds.length * (1 + HIERARCHICAL_PCP_METHODS.length);
  if (featureCount !== null && featureCount !== expectedFeatureCount) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile clustering feature count ${featureCount} does not match ${expectedFeatureCount}.`,
    );
  }
  if (fitRowCount !== null && fitRowCount > rowCount) {
    throw new RuntimeRankProfileClustersError(
      "Runtime rank-profile clustering fit row count exceeds the gallery row count.",
    );
  }

  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== prepared.rowCount) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile cluster labels have ${buffer.byteLength} bytes; expected ${prepared.rowCount}.`,
    );
  }
  const labels = new Uint8Array(buffer);
  const observed = new Uint8Array(prepared.clusterCount);
  for (let rowIndex = 0; rowIndex < labels.length; rowIndex += 1) {
    const clusterId = labels[rowIndex];
    if (clusterId >= prepared.clusterCount) {
      throw new RuntimeRankProfileClustersError(
        `Runtime rank-profile clustering returned out-of-range label ${clusterId} at row ${rowIndex}.`,
      );
    }
    observed[clusterId] = 1;
  }
  const missingCluster = observed.findIndex((value) => value === 0);
  if (missingCluster >= 0) {
    throw new RuntimeRankProfileClustersError(
      `Runtime rank-profile clustering did not assign cluster ${missingCluster}.`,
    );
  }

  return {
    taskId: prepared.taskId,
    rowCount: prepared.rowCount,
    targetCount: prepared.targetCount,
    attributeIds: [...prepared.attributeIds],
    scheme: prepared.scheme,
    clusterCount: prepared.clusterCount,
    config: cloneHierarchicalPcpConfig(prepared.config, prepared.attributeIds),
    configFingerprint: prepared.configFingerprint,
    labels,
    cacheKey: prepared.cacheKey,
    featureCount,
    fitRowCount,
  };
}

function abortError() {
  return new DOMException("The operation was aborted.", "AbortError");
}

function waitWithSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      reject(abortError());
    };
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    signal.addEventListener("abort", onAbort, { once: true });
    void promise.then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      },
    );
  });
}

/**
 * Fetches one immutable runtime cluster assignment. Completed and in-flight
 * requests share a semantic cache key, so Dashboard may prefetch before it
 * commits a new Fusion cube and the reactive hook will reuse that exact result.
 */
export function requestRuntimeRankProfileClusters(
  input: RuntimeRankProfileClustersInput,
  options: { signal?: AbortSignal } = {},
): Promise<RuntimeRankProfileClustersResult> {
  const prepared = prepareRequest(input);
  const cached = cachedResult(prepared.cacheKey);
  if (cached) return waitWithSignal(Promise.resolve(cached), options.signal);

  const existing = pendingCache.get(prepared.cacheKey);
  if (existing && existing.epoch === currentKeyEpoch(prepared.cacheKey)) {
    return waitWithSignal(existing.promise, options.signal);
  }

  const keyEpoch = currentKeyEpoch(prepared.cacheKey);
  const cacheEpoch = globalCacheEpoch;
  const promise = fetchRuntimeRankProfileClusters(prepared).then((result) => {
    if (
      globalCacheEpoch === cacheEpoch
      && currentKeyEpoch(prepared.cacheKey) === keyEpoch
    ) {
      rememberResult(result);
    }
    return result;
  }).finally(() => {
    const current = pendingCache.get(prepared.cacheKey);
    if (current?.promise === promise) pendingCache.delete(prepared.cacheKey);
  });
  pendingCache.set(prepared.cacheKey, { epoch: keyEpoch, promise });
  return waitWithSignal(promise, options.signal);
}

function disabledSourceKey(input: RuntimeRankProfileClustersInput) {
  return JSON.stringify([
    "disabled",
    input.taskId,
    input.rowCount,
    input.targetCount,
    input.attributeIds,
    input.scheme,
  ]);
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}

export function useRuntimeRankProfileClusters(
  input: RuntimeRankProfileClustersInput,
): RuntimeRankProfileClustersState {
  const {
    taskId,
    rowCount,
    targetCount,
    attributeIds,
    config,
    scheme,
    disabled: inputDisabled = false,
  } = input;
  const disabled = inputDisabled;
  const sourceKey = disabled ? disabledSourceKey(input) : runtimeRankProfileClusterKey(input);
  const [reloadRevision, setReloadRevision] = useState(0);
  const [state, setState] = useState<{
    sourceKey: string;
    status: RuntimeRankProfileClustersStatus;
    data: RuntimeRankProfileClustersResult | null;
    error: string | null;
  }>(() => {
    const cached = disabled ? null : cachedResult(sourceKey);
    return {
      sourceKey,
      status: cached ? "succeeded" : "idle",
      data: cached,
      error: null,
    };
  });
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    const requestGeneration = generation.current;
    if (disabled) {
      const timer = window.setTimeout(() => {
        if (requestGeneration === generation.current) {
          setState({ sourceKey, status: "idle", data: null, error: null });
        }
      }, 0);
      return () => window.clearTimeout(timer);
    }

    const requestController = new AbortController();
    controller.current = requestController;
    const requestInput: RuntimeRankProfileClustersInput = {
      taskId,
      rowCount,
      targetCount,
      attributeIds,
      config,
      scheme,
    };
    const cached = cachedResult(sourceKey);
    const stateTimer = window.setTimeout(() => {
      if (requestGeneration !== generation.current) return;
      setState({
        sourceKey,
        status: cached ? "succeeded" : "loading",
        data: cached,
        error: null,
      });
    }, 0);
    if (cached) {
      return () => {
        window.clearTimeout(stateTimer);
        requestController.abort();
      };
    }

    void requestRuntimeRankProfileClusters(requestInput, {
      signal: requestController.signal,
    }).then((result) => {
      if (requestGeneration !== generation.current) return;
      window.clearTimeout(stateTimer);
      setState({ sourceKey, status: "succeeded", data: result, error: null });
    }).catch((error: unknown) => {
      if (isAbortError(error) || requestGeneration !== generation.current) return;
      window.clearTimeout(stateTimer);
      const message = error instanceof Error ? error.message : String(error);
      setState({ sourceKey, status: "error", data: null, error: message });
    }).finally(() => {
      if (controller.current === requestController) controller.current = null;
    });
    return () => {
      window.clearTimeout(stateTimer);
      requestController.abort();
    };
  }, [
    attributeIds,
    config,
    disabled,
    reloadRevision,
    rowCount,
    scheme,
    sourceKey,
    targetCount,
    taskId,
  ]);

  const reload = useCallback(() => {
    if (!disabled) invalidateCacheKey(sourceKey);
    setReloadRevision((current) => current + 1);
  }, [disabled, sourceKey]);

  const clear = useCallback(() => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    if (!disabled) invalidateCacheKey(sourceKey);
    setState({ sourceKey, status: "idle", data: null, error: null });
  }, [disabled, sourceKey]);

  // A direct Dashboard prefetch may populate the cache in the same render that
  // activates a new runtime Fusion. Surface it synchronously rather than
  // exposing one mismatched frame before this hook's effect runs.
  const prefetched = disabled ? null : cachedResult(sourceKey);
  if (prefetched && (state.sourceKey !== sourceKey || state.data !== prefetched)) {
    return {
      status: "succeeded",
      data: prefetched,
      error: null,
      sourceKey,
      reload,
      clear,
    };
  }
  if (state.sourceKey !== sourceKey) {
    return {
      status: disabled ? "idle" : "loading",
      data: null,
      error: null,
      sourceKey,
      reload,
      clear,
    };
  }
  return {
    status: state.status,
    data: state.data,
    error: state.error,
    sourceKey,
    reload,
    clear,
  };
}
