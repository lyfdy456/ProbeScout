"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  cloneHierarchicalPcpConfig,
  normalizeHierarchicalPcpConfig,
  type HierarchicalPcpConfig,
} from "./hierarchicalPcp";

export const HIERARCHICAL_FUSION_METHOD_ID = "Weighted-Rank-Fusion";
export const HIERARCHICAL_FUSION_METHOD_LABEL = "Weighted Fusion";
export const HIERARCHICAL_FUSION_API_PATH = "/api/tuning/rank-fusion";
export const HIERARCHICAL_FUSION_ALGORITHM = "hierarchical-rank-fusion-13-v1";
export const HIERARCHICAL_FUSION_LAYOUT = [
  "raw",
  "calibrated",
  "rank",
] as const;

export interface HierarchicalFusionResult {
  taskId: string;
  rowCount: number;
  targetCount: number;
  attributeIds: readonly string[];
  config: HierarchicalPcpConfig;
  rawScores: Float32Array;
  calibratedScores: Float32Array;
  ranks: Float32Array;
}

export type HierarchicalFusionStatus = "idle" | "loading" | "succeeded" | "error";

export interface UseHierarchicalFusionInput {
  taskId: string;
  rowCount: number;
  targetCount: number;
  attributeIds: readonly string[];
  disabled?: boolean;
}

export interface HierarchicalFusionState {
  status: HierarchicalFusionStatus;
  result: HierarchicalFusionResult | null;
  error: string | null;
  apply: (config: Readonly<HierarchicalPcpConfig>) => Promise<HierarchicalFusionResult | null>;
  clear: () => void;
}

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;
const IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export class HierarchicalFusionError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "HierarchicalFusionError";
    this.status = status;
  }
}

function integerHeader(response: Response, name: string) {
  const value = Number(response.headers.get(name));
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new HierarchicalFusionError(`Hierarchical Fusion returned an invalid ${name}.`);
  }
  return value;
}

function floatSlice(buffer: ArrayBuffer, offset: number, length: number) {
  if (IS_LITTLE_ENDIAN) return new Float32Array(buffer, offset, length);
  const source = new DataView(buffer, offset, length * FLOAT_BYTES);
  const values = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    values[index] = source.getFloat32(index * FLOAT_BYTES, true);
  }
  return values;
}

async function responseError(response: Response) {
  let message = `Hierarchical Fusion service returned ${response.status}.`;
  try {
    const payload = await response.json() as { error?: unknown };
    if (typeof payload.error === "string" && payload.error.trim()) message = payload.error.trim();
  } catch {
    // Keep status-based fallback for non-JSON failures.
  }
  return new HierarchicalFusionError(message, response.status);
}

export async function requestHierarchicalFusion(
  input: UseHierarchicalFusionInput & { config: Readonly<HierarchicalPcpConfig> },
  options: { signal?: AbortSignal } = {},
): Promise<HierarchicalFusionResult> {
  const taskId = input.taskId.trim();
  if (!taskId) throw new HierarchicalFusionError("Hierarchical Fusion requires a task ID.");
  if (!Number.isSafeInteger(input.rowCount) || input.rowCount <= 0) {
    throw new HierarchicalFusionError("Hierarchical Fusion requires a positive row count.");
  }
  if (!Number.isSafeInteger(input.targetCount) || input.targetCount <= 0) {
    throw new HierarchicalFusionError("Hierarchical Fusion requires a positive target count.");
  }
  const config = normalizeHierarchicalPcpConfig(input.config, input.attributeIds);
  const response = await fetch(HIERARCHICAL_FUSION_API_PATH, {
    method: "POST",
    credentials: "same-origin",
    signal: options.signal,
    headers: { Accept: "application/octet-stream", "Content-Type": "application/json" },
    body: JSON.stringify({
      taskId,
      attributeWeights: config.attributeWeights,
      methodWeightsByAttribute: config.methodWeightsByAttribute,
    }),
  });
  if (!response.ok) throw await responseError(response);
  if (!(response.headers.get("Content-Type") ?? "").toLowerCase().startsWith("application/octet-stream")) {
    throw new HierarchicalFusionError("Hierarchical Fusion response is not an octet-stream.");
  }
  if (response.headers.get("X-PCP-Algorithm") !== HIERARCHICAL_FUSION_ALGORITHM) {
    throw new HierarchicalFusionError("Unsupported Hierarchical Fusion response algorithm.");
  }
  if (response.headers.get("X-PCP-Layout") !== HIERARCHICAL_FUSION_LAYOUT.join(",")) {
    throw new HierarchicalFusionError("Unsupported Hierarchical Fusion response layout.");
  }
  const rowCount = integerHeader(response, "X-PCP-Row-Count");
  const targetCount = integerHeader(response, "X-PCP-Target-Count");
  if (rowCount !== input.rowCount || targetCount !== input.targetCount) {
    throw new HierarchicalFusionError(
      `Hierarchical Fusion shape ${rowCount}x${targetCount} does not match this task.`,
    );
  }
  const count = rowCount * targetCount;
  const buffer = await response.arrayBuffer();
  const expectedBytes = count * HIERARCHICAL_FUSION_LAYOUT.length * FLOAT_BYTES;
  if (buffer.byteLength !== expectedBytes) {
    throw new HierarchicalFusionError(
      `Hierarchical Fusion payload has ${buffer.byteLength} bytes; expected ${expectedBytes}.`,
    );
  }
  return {
    taskId,
    rowCount,
    targetCount,
    attributeIds: [...input.attributeIds],
    config: cloneHierarchicalPcpConfig(config, input.attributeIds),
    rawScores: floatSlice(buffer, 0, count),
    calibratedScores: floatSlice(buffer, count * FLOAT_BYTES, count),
    ranks: floatSlice(buffer, count * FLOAT_BYTES * 2, count),
  };
}

function makeScopeKey(input: UseHierarchicalFusionInput) {
  return [input.taskId, input.rowCount, input.targetCount, ...input.attributeIds].join("\u0000");
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}

export function useHierarchicalFusion(input: UseHierarchicalFusionInput): HierarchicalFusionState {
  const {
    taskId,
    rowCount,
    targetCount,
    attributeIds,
    disabled = false,
  } = input;
  const scopeKey = makeScopeKey({ taskId, rowCount, targetCount, attributeIds });
  const [state, setState] = useState<{
    scopeKey: string;
    status: HierarchicalFusionStatus;
    result: HierarchicalFusionResult | null;
    error: string | null;
  }>({ scopeKey, status: "idle", result: null, error: null });
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    const timer = window.setTimeout(() => {
      setState({ scopeKey, status: "idle", result: null, error: null });
    }, 0);
    return () => window.clearTimeout(timer);
  }, [disabled, scopeKey]);

  const apply = useCallback(async (config: Readonly<HierarchicalPcpConfig>) => {
    if (disabled) throw new HierarchicalFusionError("Hierarchical Fusion is disabled.");
    controller.current?.abort();
    const requestController = new AbortController();
    controller.current = requestController;
    const requestGeneration = ++generation.current;
    // Keep the last committed result visible while a replacement is being
    // computed. The caller activates the new configuration only after this
    // request succeeds, so a failed/retried Apply cannot expose half-updated
    // ranks and PCP weights.
    setState((current) => ({
      scopeKey,
      status: "loading",
      result: current.scopeKey === scopeKey ? current.result : null,
      error: null,
    }));
    try {
      const result = await requestHierarchicalFusion(
        { taskId, rowCount, targetCount, attributeIds, disabled, config },
        { signal: requestController.signal },
      );
      if (requestGeneration !== generation.current) return null;
      setState({ scopeKey, status: "succeeded", result, error: null });
      return result;
    } catch (error) {
      if (isAbortError(error) || requestGeneration !== generation.current) return null;
      const message = error instanceof Error ? error.message : String(error);
      setState((current) => ({
        scopeKey,
        status: "error",
        result: current.scopeKey === scopeKey ? current.result : null,
        error: message,
      }));
      throw error;
    } finally {
      if (controller.current === requestController) controller.current = null;
    }
  }, [attributeIds, disabled, rowCount, scopeKey, targetCount, taskId]);

  const clear = useCallback(() => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    setState({ scopeKey, status: "idle", result: null, error: null });
  }, [scopeKey]);

  if (state.scopeKey !== scopeKey) {
    return { status: "idle", result: null, error: null, apply, clear };
  }
  return { status: state.status, result: state.result, error: state.error, apply, clear };
}
