"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export const WEIGHTED_FUSION_METHOD_ID = "Ours-Weighted";
export const WEIGHTED_FUSION_METHOD_LABEL = "Ours-Weighted";
export const WEIGHTED_FUSION_API_PATH = "/api/tuning/weighted-fusion";
export const WEIGHTED_FUSION_ALGORITHM = "weighted-ours-full-v1";
export const WEIGHTED_FUSION_LAYOUT = ["raw", "calibrated", "rank"] as const;

export const WEIGHTED_FUSION_LEARNERS = [
  "MLP",
  "K-Fold",
  "Triplet Loss",
  "Attention Pooling",
  "Attribute-conditioned Attention",
  "nnPU",
  "DC-PU",
  "Ours-PURA",
] as const;

export type WeightedFusionLearner = typeof WEIGHTED_FUSION_LEARNERS[number];
export type WeightedFusionWeights = Record<WeightedFusionLearner, number>;
export type WeightedFusionStatus = "idle" | "loading" | "succeeded" | "error";

export interface WeightedFusionRequest {
  taskId: string;
  weights: WeightedFusionWeights;
}

export interface WeightedFusionResult {
  taskId: string;
  rowCount: number;
  targetCount: number;
  weights: WeightedFusionWeights;
  rawScores: Float32Array;
  calibratedScores: Float32Array;
  ranks: Float32Array;
  algorithm: string;
  layout: typeof WEIGHTED_FUSION_LAYOUT;
  equalWeights: boolean;
  baselineMaxAbsError: number | null;
}

export interface WeightedFusionDimensions {
  taskId: string;
  rowCount: number;
  targetCount: number;
}

export interface UseWeightedFusionInput extends WeightedFusionDimensions {
  disabled?: boolean;
}

export interface WeightedFusionState {
  status: WeightedFusionStatus;
  result: WeightedFusionResult | null;
  error: string | null;
  apply: (weights: WeightedFusionWeights) => Promise<WeightedFusionResult | null>;
  clear: () => void;
}

interface InternalState {
  scopeKey: string;
  status: WeightedFusionStatus;
  result: WeightedFusionResult | null;
  error: string | null;
}

interface RequestOptions {
  signal?: AbortSignal;
}

const FLOAT_SIZE = Float32Array.BYTES_PER_ELEMENT;
const EQUAL_WEIGHT_EPSILON = 1e-8;
const IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export class WeightedFusionError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "WeightedFusionError";
    this.status = status;
  }
}

export function createEqualWeightedFusionWeights(value = 1): WeightedFusionWeights {
  if (!Number.isFinite(value) || value <= 0) {
    throw new WeightedFusionError("Equal weights must use a finite value greater than zero.");
  }
  return Object.fromEntries(
    WEIGHTED_FUSION_LEARNERS.map((learner) => [learner, value]),
  ) as WeightedFusionWeights;
}

export const DEFAULT_WEIGHTED_FUSION_WEIGHTS: Readonly<WeightedFusionWeights> =
  Object.freeze(createEqualWeightedFusionWeights());

export function cloneWeightedFusionWeights(
  weights: Readonly<WeightedFusionWeights>,
): WeightedFusionWeights {
  return Object.fromEntries(
    WEIGHTED_FUSION_LEARNERS.map((learner) => [learner, weights[learner]]),
  ) as WeightedFusionWeights;
}

export function weightedFusionWeightTotal(weights: Readonly<WeightedFusionWeights>) {
  let total = 0;
  for (const learner of WEIGHTED_FUSION_LEARNERS) {
    const weight = weights[learner];
    if (!Number.isFinite(weight) || weight < 0) {
      throw new WeightedFusionError(`${learner} weight must be a finite non-negative number.`);
    }
    total += weight;
  }
  if (!Number.isFinite(total) || total <= 0) {
    throw new WeightedFusionError("At least one learner weight must be greater than zero.");
  }
  return total;
}

export function normalizeWeightedFusionWeights(
  weights: Readonly<WeightedFusionWeights>,
): WeightedFusionWeights {
  const total = weightedFusionWeightTotal(weights);
  return Object.fromEntries(
    WEIGHTED_FUSION_LEARNERS.map((learner) => [learner, weights[learner] / total]),
  ) as WeightedFusionWeights;
}

export function areWeightedFusionWeightsEquivalent(
  left: Readonly<WeightedFusionWeights>,
  right: Readonly<WeightedFusionWeights>,
  epsilon = EQUAL_WEIGHT_EPSILON,
) {
  let normalizedLeft: WeightedFusionWeights;
  let normalizedRight: WeightedFusionWeights;
  try {
    normalizedLeft = normalizeWeightedFusionWeights(left);
    normalizedRight = normalizeWeightedFusionWeights(right);
  } catch {
    return false;
  }
  return WEIGHTED_FUSION_LEARNERS.every(
    (learner) => Math.abs(normalizedLeft[learner] - normalizedRight[learner]) <= epsilon,
  );
}

function requiredIntegerHeader(response: Response, name: string) {
  const raw = response.headers.get(name);
  const parsed = raw === null ? Number.NaN : Number(raw);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new WeightedFusionError(`Weighted Fusion response has an invalid ${name} header.`);
  }
  return parsed;
}

function optionalNonNegativeNumberHeader(response: Response, name: string) {
  const raw = response.headers.get(name);
  if (raw === null || raw.trim() === "") return null;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new WeightedFusionError(`Weighted Fusion response has an invalid ${name} header.`);
  }
  return parsed;
}

function readFloat32LittleEndian(buffer: ArrayBuffer, byteOffset: number, length: number) {
  if (IS_LITTLE_ENDIAN) return new Float32Array(buffer, byteOffset, length);
  const view = new DataView(buffer, byteOffset, length * FLOAT_SIZE);
  const values = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    values[index] = view.getFloat32(index * FLOAT_SIZE, true);
  }
  return values;
}

async function responseError(response: Response) {
  let message = `Weighted Fusion service returned ${response.status}.`;
  try {
    const payload = await response.json() as { error?: unknown };
    if (typeof payload.error === "string" && payload.error.trim()) {
      message = payload.error.trim();
    }
  } catch {
    // Keep the status-based fallback for a non-JSON error response.
  }
  return new WeightedFusionError(message, response.status);
}

export async function parseWeightedFusionResponse(
  response: Response,
  expected: WeightedFusionDimensions,
  weights: Readonly<WeightedFusionWeights>,
): Promise<WeightedFusionResult> {
  if (!response.ok) throw await responseError(response);

  const contentType = response.headers.get("Content-Type")?.toLowerCase() ?? "";
  if (!contentType.startsWith("application/octet-stream")) {
    throw new WeightedFusionError("Weighted Fusion response is not an octet-stream.");
  }

  const algorithm = response.headers.get("X-PCP-Algorithm") ?? "";
  if (algorithm !== WEIGHTED_FUSION_ALGORITHM) {
    throw new WeightedFusionError(`Unsupported Weighted Fusion algorithm: ${algorithm || "missing"}.`);
  }
  const layoutHeader = response.headers.get("X-PCP-Layout") ?? "";
  const layout = layoutHeader.split(",").map((part) => part.trim());
  if (
    layout.length !== WEIGHTED_FUSION_LAYOUT.length
    || WEIGHTED_FUSION_LAYOUT.some((part, index) => part !== layout[index])
  ) {
    throw new WeightedFusionError(`Unsupported Weighted Fusion layout: ${layoutHeader || "missing"}.`);
  }

  const rowCount = requiredIntegerHeader(response, "X-PCP-Row-Count");
  const targetCount = requiredIntegerHeader(response, "X-PCP-Target-Count");
  if (rowCount !== expected.rowCount || targetCount !== expected.targetCount) {
    throw new WeightedFusionError(
      `Weighted Fusion shape ${rowCount}×${targetCount} does not match `
      + `${expected.rowCount}×${expected.targetCount} for ${expected.taskId}.`,
    );
  }

  const equalWeightsHeader = response.headers.get("X-PCP-Equal-Weights");
  if (equalWeightsHeader !== "0" && equalWeightsHeader !== "1") {
    throw new WeightedFusionError("Weighted Fusion response has an invalid X-PCP-Equal-Weights header.");
  }
  const baselineMaxAbsError = optionalNonNegativeNumberHeader(
    response,
    "X-PCP-Baseline-Max-Abs-Error",
  );

  const valueCount = rowCount * targetCount;
  const expectedBytes = valueCount * WEIGHTED_FUSION_LAYOUT.length * FLOAT_SIZE;
  if (!Number.isSafeInteger(expectedBytes)) {
    throw new WeightedFusionError("Weighted Fusion response dimensions are too large.");
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== expectedBytes) {
    throw new WeightedFusionError(
      `Weighted Fusion payload has ${buffer.byteLength} bytes; expected ${expectedBytes}.`,
    );
  }

  return {
    taskId: expected.taskId,
    rowCount,
    targetCount,
    weights: normalizeWeightedFusionWeights(weights),
    rawScores: readFloat32LittleEndian(buffer, 0, valueCount),
    calibratedScores: readFloat32LittleEndian(buffer, valueCount * FLOAT_SIZE, valueCount),
    ranks: readFloat32LittleEndian(buffer, valueCount * FLOAT_SIZE * 2, valueCount),
    algorithm,
    layout: WEIGHTED_FUSION_LAYOUT,
    equalWeights: equalWeightsHeader === "1",
    baselineMaxAbsError,
  };
}

export async function requestWeightedFusion(
  input: WeightedFusionRequest & Pick<WeightedFusionDimensions, "rowCount" | "targetCount">,
  options: RequestOptions = {},
) {
  const taskId = input.taskId.trim();
  if (!taskId) throw new WeightedFusionError("Weighted Fusion requires a task ID.");
  if (!Number.isSafeInteger(input.rowCount) || input.rowCount <= 0) {
    throw new WeightedFusionError("Weighted Fusion requires a positive row count.");
  }
  if (!Number.isSafeInteger(input.targetCount) || input.targetCount <= 0) {
    throw new WeightedFusionError("Weighted Fusion requires a positive target count.");
  }
  const normalizedWeights = normalizeWeightedFusionWeights(input.weights);
  const response = await fetch(WEIGHTED_FUSION_API_PATH, {
    method: "POST",
    credentials: "same-origin",
    signal: options.signal,
    headers: {
      Accept: "application/octet-stream",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ taskId, weights: normalizedWeights }),
  });
  return parseWeightedFusionResponse(
    response,
    { taskId, rowCount: input.rowCount, targetCount: input.targetCount },
    normalizedWeights,
  );
}

function scopeKey(input: WeightedFusionDimensions) {
  return `${input.taskId}\u0000${input.rowCount}\u0000${input.targetCount}`;
}

function isAbortError(reason: unknown) {
  return reason instanceof DOMException && reason.name === "AbortError";
}

export function useWeightedFusion(input: UseWeightedFusionInput): WeightedFusionState {
  const { taskId, rowCount, targetCount, disabled = false } = input;
  const currentScopeKey = useMemo(
    () => scopeKey({ taskId, rowCount, targetCount }),
    [rowCount, targetCount, taskId],
  );
  const [state, setState] = useState<InternalState>({
    scopeKey: currentScopeKey,
    status: "idle",
    result: null,
    error: null,
  });
  const generationRef = useRef(0);
  const controllerRef = useRef<AbortController | null>(null);
  const activeScopeRef = useRef(currentScopeKey);

  useEffect(() => {
    activeScopeRef.current = currentScopeKey;
    generationRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    const timer = window.setTimeout(() => {
      setState({ scopeKey: currentScopeKey, status: "idle", result: null, error: null });
    }, 0);
    return () => {
      window.clearTimeout(timer);
      generationRef.current += 1;
      controllerRef.current?.abort();
      controllerRef.current = null;
    };
  }, [currentScopeKey, disabled]);

  const apply = useCallback(async (weights: WeightedFusionWeights) => {
    if (disabled) throw new WeightedFusionError("Weighted Fusion is disabled.");
    const requestScopeKey = currentScopeKey;
    const controller = new AbortController();
    controllerRef.current?.abort();
    controllerRef.current = controller;
    const generation = ++generationRef.current;
    setState({
      scopeKey: requestScopeKey,
      status: "loading",
      result: null,
      error: null,
    });
    try {
      const result = await requestWeightedFusion({
        taskId,
        rowCount,
        targetCount,
        weights,
      }, { signal: controller.signal });
      if (
        generation !== generationRef.current
        || activeScopeRef.current !== requestScopeKey
      ) return null;
      setState({
        scopeKey: requestScopeKey,
        status: "succeeded",
        result,
        error: null,
      });
      return result;
    } catch (reason) {
      if (
        isAbortError(reason)
        || generation !== generationRef.current
        || activeScopeRef.current !== requestScopeKey
      ) return null;
      const message = reason instanceof Error ? reason.message : String(reason);
      setState({
        scopeKey: requestScopeKey,
        status: "error",
        result: null,
        error: message,
      });
      throw reason;
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  }, [currentScopeKey, disabled, rowCount, targetCount, taskId]);

  const clear = useCallback(() => {
    generationRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    setState({
      scopeKey: currentScopeKey,
      status: "idle",
      result: null,
      error: null,
    });
  }, [currentScopeKey]);

  if (state.scopeKey !== currentScopeKey) {
    return { status: "idle", result: null, error: null, apply, clear };
  }
  return {
    status: state.status,
    result: state.result,
    error: state.error,
    apply,
    clear,
  };
}
