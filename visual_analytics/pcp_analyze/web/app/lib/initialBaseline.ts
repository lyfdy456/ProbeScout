import { HIERARCHICAL_PCP_LEARNERS } from "./hierarchicalPcp";
import { REFINEMENT_EMBEDDING_METHODS, type TuningEvaluationMetrics } from "./tuningApi";
import {
  REFINEMENT_CLUSTER_ALGORITHM, REFINEMENT_CLUSTER_FEATURE_BASIS,
  REFINEMENT_VISUALIZATION_ALGORITHM, type RefinementVisualizationSnapshot,
} from "./refinementVisualization";

export const INITIAL_BASELINE_VERSION = "isolated-initial-baseline-v1";
export const LEGACY_OURS_FULL_OPTION = "legacy:Ours-Full";
export interface InitialBaselineManifest {
  available: true;
  taskId: string;
  version: string;
  fingerprint: string;
  baseStateFingerprint: string;
  sourceFingerprint: string;
  normalizationFingerprint: string;
  initialModelHoldoutIndependent: boolean;
  calibration?: string;
  calibrationUsedValidation?: boolean;
  classificationThreshold?: number | null;
  referenceOnly: false;
  publicationVersion: string;
  visualizationVersion: string;
  clusterAlgorithm: string;
  rowCount: number;
  attributeIds: string[];
  learnerMethods: string[];
  embeddingMethods: string[];
  modelSummary: NonNullable<TuningEvaluationMetrics["modelSummary"]>;
}
export type InitialBaselineSnapshot = Omit<RefinementVisualizationSnapshot, "runId"> & {
  baselineFingerprint: string;
};
export interface InitialBaselineClusters {
  baselineFingerprint: string;
  sourceFingerprint: string;
  scheme: string;
  rowCount: number;
  clusterCount: number;
  featureCount: number;
  fitRowCount: number;
  labels: Uint8Array;
}
export interface InitialBaselineData {
  manifest: InitialBaselineManifest;
  snapshot: InitialBaselineSnapshot;
  clusters: InitialBaselineClusters;
}
const snapshots = new Map<string, InitialBaselineSnapshot>();
const clusters = new Map<string, InitialBaselineClusters>();
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const fingerprintValid = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const same = (actual: unknown, expected: readonly string[]) => Array.isArray(actual)
  && actual.length === expected.length && actual.every((value, index) => value === expected[index]);
const prefix = (taskId: string) => `/api/tuning/tasks/${encodeURIComponent(taskId)}/initial-baseline`;

async function readResponse(url: string, signal?: AbortSignal, binary = false, allowPending = false) {
  const response = await fetch(url, {
    credentials: "same-origin", cache: "no-store", signal,
    headers: { Accept: binary ? "application/octet-stream" : "application/json" },
  });
  if (allowPending && response.status === 404) return response;
  if (!response.ok) {
    let detail = `Initial F₀ returned ${response.status}.`;
    try { const payload = await response.json(); if (typeof payload.error === "string") detail = payload.error; } catch { /* status fallback */ }
    throw new Error(detail);
  }
  if (binary && !(response.headers.get("Content-Type") ?? "").startsWith("application/octet-stream")) {
    throw new Error("Initial F₀ response is not binary data.");
  }
  return response;
}
function header(response: Response, name: string) {
  const value = response.headers.get(name)?.trim();
  if (!value) throw new Error(`Initial F₀ missing ${name}.`);
  return value;
}
function integer(response: Response, name: string) {
  const value = Number(header(response, name));
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`Initial F₀ invalid ${name}.`);
  return value;
}
function assertHeader(response: Response, name: string, expected: string) {
  if (header(response, name) !== expected) throw new Error(`Initial F₀ ${name} identity mismatch.`);
}
function remember<T>(cache: Map<string, T>, key: string, value: T, limit: number) {
  if (cache.size >= limit) cache.delete(cache.keys().next().value as string);
  cache.set(key, value);
  return value;
}
function assertInitialWeights(manifest: InitialBaselineManifest) {
  const summary = manifest.modelSummary;
  const equal = (value: unknown, expected: number) => typeof value === "number"
    && Number.isFinite(value) && Math.abs(value - expected) < 1e-6;
  if (!summary || !equal(summary.embeddingFusionStrength, 0.25)
    || !manifest.attributeIds.every((attributeId) => equal(summary.gamma?.[attributeId], 1)
      && HIERARCHICAL_PCP_LEARNERS.every((method) => equal(summary.beta?.[attributeId]?.[method], 1 / 8)))
    || !REFINEMENT_EMBEDDING_METHODS.every((method) => equal(summary.embeddingWeights?.[method], 0.5))) {
    throw new Error("Initial F₀ does not contain the fixed initial weights.");
  }
}

export async function requestInitialBaselineManifest(input: {
  taskId: string; rowCount: number; attributeIds: readonly string[]; signal?: AbortSignal;
}): Promise<InitialBaselineManifest | null> {
  const response = await readResponse(prefix(input.taskId), input.signal, false, true);
  if (response.status === 404) return null;
  const payload = await response.json();
  if (payload.taskId !== input.taskId) throw new Error("Initial F₀ belongs to another task.");
  if (payload.available === false) return null;
  const calibrationValid = payload.calibration === "full-only-per-task-vqa-val-ap-gates-val-f1-cutoff-v1"
    ? payload.initialModelHoldoutIndependent === false && payload.calibrationUsedValidation === true
      && typeof payload.classificationThreshold === "number" && Number.isFinite(payload.classificationThreshold)
      && payload.classificationThreshold >= 0 && payload.classificationThreshold <= 1
    : payload.initialModelHoldoutIndependent === true && payload.calibrationUsedValidation !== true;
  if (payload.available !== true || payload.version !== INITIAL_BASELINE_VERSION
    || payload.rowCount !== input.rowCount || !Number.isSafeInteger(input.rowCount) || input.rowCount <= 0
    || !input.attributeIds.length || new Set(input.attributeIds).size !== input.attributeIds.length
    || !same(payload.attributeIds, input.attributeIds)
    || !same(payload.learnerMethods, HIERARCHICAL_PCP_LEARNERS)
    || !same(payload.embeddingMethods, REFINEMENT_EMBEDDING_METHODS)
    || payload.visualizationVersion !== REFINEMENT_VISUALIZATION_ALGORITHM
    || payload.clusterAlgorithm !== REFINEMENT_CLUSTER_ALGORITHM
    || !fingerprintValid(payload.fingerprint) || !fingerprintValid(payload.baseStateFingerprint)
    || payload.fingerprint !== payload.baseStateFingerprint
    || !fingerprintValid(payload.sourceFingerprint)
    || !fingerprintValid(payload.normalizationFingerprint)
    || !calibrationValid || payload.referenceOnly !== false
    || typeof payload.publicationVersion !== "string" || !payload.publicationVersion.trim()) {
    throw new Error("Initial F₀ publication identity is invalid.");
  }
  assertInitialWeights(payload);
  return payload;
}
function endpoint(manifest: InitialBaselineManifest, kind: string) {
  return `${prefix(manifest.taskId)}/${kind}?fingerprint=${encodeURIComponent(manifest.fingerprint)}`;
}

export async function requestInitialBaselineVisualization(manifest: InitialBaselineManifest, signal?: AbortSignal) {
  const key = `${manifest.taskId}|${manifest.fingerprint}|${manifest.sourceFingerprint}`;
  const cached = snapshots.get(key);
  if (cached) return cached;
  const response = await readResponse(endpoint(manifest, "visualization"), signal, true);
  assertHeader(response, "X-PCP-Algorithm", REFINEMENT_VISUALIZATION_ALGORITHM);
  assertHeader(response, "X-PCP-Layout", "score,rank");
  assertHeader(response, "X-PCP-Baseline-Fingerprint", manifest.fingerprint);
  assertHeader(response, "X-PCP-Base-Fingerprint", manifest.baseStateFingerprint);
  assertHeader(response, "X-PCP-Source-Fingerprint", manifest.sourceFingerprint);
  const rowCount = integer(response, "X-PCP-Row-Count");
  const componentCount = integer(response, "X-PCP-Component-Count");
  if (rowCount !== manifest.rowCount || componentCount !== 3 + 9 * manifest.attributeIds.length + 2) {
    throw new Error("Initial F₀ evidence dimensions changed.");
  }
  const count = rowCount * componentCount;
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== count * 8) throw new Error("Initial F₀ evidence payload has the wrong size.");
  const values = LITTLE_ENDIAN ? new Float32Array(buffer) : Float32Array.from(
    { length: count * 2 }, (_, index) => new DataView(buffer).getFloat32(index * 4, true),
  );
  if (values.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error("Initial F₀ evidence must be finite values in [0,1].");
  }
  return remember(snapshots, key, {
    baselineFingerprint: manifest.fingerprint, sourceFingerprint: manifest.sourceFingerprint,
    baseFingerprint: manifest.baseStateFingerprint, rowCount, componentCount,
    attributeIds: manifest.attributeIds, embeddingMethods: manifest.embeddingMethods,
    scores: values.subarray(0, count), ranks: values.subarray(count),
  }, 3);
}

export function initialBaselineSourceKey(manifest: InitialBaselineManifest, scheme: string) {
  return `initial|${manifest.taskId}|${manifest.fingerprint}|${manifest.sourceFingerprint}|${scheme}`;
}
export async function requestInitialBaselineClusters(
  manifest: InitialBaselineManifest, snapshot: InitialBaselineSnapshot, scheme: string, signal?: AbortSignal,
) {
  const expectedK = ({ fine30: 30, fine50: 50, fine100: 100, absolute: 4, shape: 3 } as Record<string, number>)[scheme];
  if (!expectedK || snapshot.baselineFingerprint !== manifest.fingerprint
    || snapshot.baseFingerprint !== manifest.baseStateFingerprint
    || snapshot.sourceFingerprint !== manifest.sourceFingerprint) throw new Error("Initial F₀ cluster source is invalid.");
  const key = initialBaselineSourceKey(manifest, scheme);
  const cached = clusters.get(key);
  if (cached) return cached;
  const response = await readResponse(`${endpoint(manifest, "clusters")}&scheme=${encodeURIComponent(scheme)}`, signal, true);
  assertHeader(response, "X-PCP-Algorithm", REFINEMENT_CLUSTER_ALGORITHM);
  assertHeader(response, "X-PCP-Layout", "labels");
  assertHeader(response, "X-PCP-Feature-Basis", REFINEMENT_CLUSTER_FEATURE_BASIS);
  assertHeader(response, "X-PCP-Baseline-Fingerprint", manifest.fingerprint);
  assertHeader(response, "X-PCP-Base-Fingerprint", manifest.baseStateFingerprint);
  assertHeader(response, "X-PCP-Source-Fingerprint", snapshot.sourceFingerprint);
  assertHeader(response, "X-PCP-Cluster-Scheme", scheme);
  assertHeader(response, "X-PCP-Fit-Scope", "development");
  const rowCount = integer(response, "X-PCP-Row-Count");
  const clusterCount = integer(response, "X-PCP-Cluster-Count");
  const featureCount = integer(response, "X-PCP-Feature-Count");
  const fitRowCount = integer(response, "X-PCP-Fit-Row-Count");
  if (rowCount !== snapshot.rowCount || featureCount !== snapshot.componentCount
    || clusterCount !== expectedK || fitRowCount > rowCount) throw new Error("Initial F₀ cluster dimensions changed.");
  const buffer = await response.arrayBuffer();
  const labels = new Uint8Array(buffer);
  if (labels.length !== rowCount || labels.some((label) => label >= clusterCount)) {
    throw new Error("Initial F₀ cluster labels are invalid.");
  }
  return remember(clusters, key, {
    baselineFingerprint: manifest.fingerprint, sourceFingerprint: snapshot.sourceFingerprint,
    scheme, rowCount, clusterCount, featureCount, fitRowCount, labels,
  }, 40);
}

/** F0 targets have model scores: Joint=F; a modeled single attribute=g_a. */
export function initialBaselineTargetColumn(manifest: InitialBaselineManifest, targetId: string) {
  if (targetId === "joint") return 0;
  const attributeIndex = manifest.attributeIds.indexOf(targetId);
  return attributeIndex < 0 ? null : 2 + attributeIndex * 9;
}
