export type FeedbackValue = -2 | -1 | 0 | 1 | 2;
export type TuningLaunchMode =
  | "weight_staged"
  | "weight_joint"
  | "probe_staged"
  | "probe_joint";
export type LegacyTuningMode = "prototype" | "label" | "fusion-weight" | "residual";
export type TuningMode = TuningLaunchMode | LegacyTuningMode;
export type TuningRunStatus = "queued" | "running" | "succeeded" | "failed";
export type ProbeSource = "original" | "updated";
export type WeightTuningMode = "weight_staged" | "weight_joint";
export interface ProbeUpdateJob {
  id: string;
  sessionId: string;
  taskId: string;
  status: TuningRunStatus;
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  error?: string | null;
  snapshotId?: string | null;
  annotationCount: number;
  positiveCount: number;
  negativeCount: number;
  annotationStateSha256?: string;
  baseStateFingerprint?: string;
  compatible: boolean;
  stale: boolean;
  updatedAttributes?: string[];
}
export const MAX_BULK_TUNING_ANNOTATIONS = 1_000;
export const PROBE_UPDATE_ALGORITHM = "native-probe-update-then-freeze-v1";
export const PER_ATTRIBUTE_FUSION_ALGORITHM_V2 = "fusion-weight-per-attribute-tune-v2";
export const PER_ATTRIBUTE_FUSION_ALGORITHM_V3 =
  "fusion-weight-per-attribute-joint-product-tune-v3";
export const PER_ATTRIBUTE_FUSION_ALGORITHM = PER_ATTRIBUTE_FUSION_ALGORITHM_V3;
export const HIERARCHICAL_RANK_FUSION_ALGORITHM =
  "fusion-weight-hierarchical-rank-13-v4";
export const WEIGHT_STAGED_ALGORITHM_V2 = "conjunction-holistic-weight-staged-v2";
export const WEIGHT_JOINT_ALGORITHM_V2 = "conjunction-holistic-weight-joint-v2";
export const WEIGHT_STAGED_ALGORITHM_V3 = "conjunction-holistic-weight-staged-v3";
export const WEIGHT_JOINT_ALGORITHM_V3 = "conjunction-holistic-weight-joint-v3";
export const WEIGHT_STAGED_ALGORITHM_V4 = "conjunction-holistic-weight-staged-v4";
export const WEIGHT_JOINT_ALGORITHM_V4 = "conjunction-holistic-weight-joint-v4";
export const WEIGHT_STAGED_ALGORITHM = "conjunction-holistic-weight-staged-v5";
export const WEIGHT_JOINT_ALGORITHM = "conjunction-holistic-weight-joint-v5";
export const PROBE_STAGED_ALGORITHM_V2 = "conjunction-holistic-probe-staged-v2";
export const PROBE_JOINT_ALGORITHM_V2 = "conjunction-holistic-probe-joint-v2";
export const PROBE_STAGED_ALGORITHM = "conjunction-holistic-probe-staged-v3";
export const PROBE_JOINT_ALGORITHM = "conjunction-holistic-probe-joint-v3";
export const TUNING_LAUNCH_ALGORITHMS: Readonly<Record<TuningLaunchMode, string>> = {
  weight_staged: WEIGHT_STAGED_ALGORITHM,
  weight_joint: WEIGHT_JOINT_ALGORITHM,
  probe_staged: PROBE_STAGED_ALGORITHM,
  probe_joint: PROBE_JOINT_ALGORITHM,
};
export const REFINEMENT_EMBEDDING_METHODS = ["Query MaxSim", "Text Prompt Ensemble"] as const;
export const LEGACY_REFINEMENT_EMBEDDING_METHODS = [
  "Image Prototype", "Query MaxSim", "Image--Text Fusion",
  "Text Prompt Ensemble", "Z-score Image--Text Fusion",
] as const;

function isNativeProbeRefinementAlgorithm(mode: string, algorithmVersion?: string): boolean {
  return (mode === "probe_staged" && (
    algorithmVersion === PROBE_STAGED_ALGORITHM || algorithmVersion === PROBE_STAGED_ALGORITHM_V2
  )) || (mode === "probe_joint" && (
    algorithmVersion === PROBE_JOINT_ALGORITHM || algorithmVersion === PROBE_JOINT_ALGORITHM_V2
  ));
}

export function tuningModeLabel(mode: string, algorithmVersion?: string): string {
  if (mode === "weight_staged") return "Weight-only · Staged";
  if (mode === "weight_joint") return "Weight-only · Joint";
  if (mode === "probe_staged") return algorithmVersion && !isNativeProbeRefinementAlgorithm(mode, algorithmVersion)
    ? "Legacy Probe-adaptive · Staged" : "Native → Fusion · Staged";
  if (mode === "probe_joint") return algorithmVersion && !isNativeProbeRefinementAlgorithm(mode, algorithmVersion)
    ? "Legacy Probe-adaptive · Joint" : "Native → Fusion · Joint";
  if (mode === "fusion-weight") return "Fusion Weight";
  if (mode === "residual") return "Residual";
  if (mode === "prototype") return "Legacy Prototype";
  if (mode === "label") return "Legacy Label Head";
  return "Legacy tuning";
}

export function tuningRankingLabel(mode: string, algorithmVersion?: string): string {
  return `${tuningModeLabel(mode, algorithmVersion)} Tune`;
}

export interface TuningRefinementCapability {
  available: boolean;
  reason?: string | null;
  algorithmVersion?: string;
}

export type TuningRefinementCapabilities = Partial<Record<TuningLaunchMode | "update_probes", TuningRefinementCapability>>;

/** A missing or incompatible capability never unlocks a training action. */
export function refinementCapability(
  capabilities: TuningRefinementCapabilities | null | undefined,
  mode: TuningLaunchMode,
): TuningRefinementCapability {
  const capability = capabilities?.[mode];
  if (capability?.available !== true) return {
    available: false,
    reason: typeof capability?.reason === "string" && capability.reason.trim()
      ? capability.reason : "Tuning availability has not been confirmed.",
  };
  if (capability.algorithmVersion !== TUNING_LAUNCH_ALGORITHMS[mode]) return {
    available: false,
    reason: "Tuning implementation mismatch; reload the service.",
  };
  return { available: true, algorithmVersion: capability.algorithmVersion };
}

export function probeUpdateCapability(capabilities: TuningRefinementCapabilities | null | undefined): TuningRefinementCapability {
  const capability = capabilities?.update_probes;
  if (capability?.available !== true) return {
    available: false,
    reason: capability?.reason || "Probe update availability has not been confirmed.",
  };
  if (capability.algorithmVersion !== PROBE_UPDATE_ALGORITHM) return {
    available: false, reason: "Probe update implementation mismatch; reload the service.",
  };
  return capability;
}

export function tuningBaseMethod(mode: TuningLaunchMode, currentBaseMethod: string): string {
  void mode;
  void currentBaseMethod;
  return "Ours-Full";
}

export function isCurrentTuningMode(mode: string): mode is TuningLaunchMode {
  return mode === "weight_staged"
    || mode === "weight_joint"
    || mode === "probe_staged"
    || mode === "probe_joint";
}

export interface TuningUser {
  id: string;
  displayName: string;
  identitySource: "openai" | "local-cookie" | string;
}

export interface TuningSession {
  id: string;
  taskId: string;
  targetId: string;
  baseMethod: string;
  createdAt: string;
  updatedAt: string;
}

export type TuningFeedbackOrigin = "new" | "existing";
export type TuningFeedbackRelation = "new" | "override" | "reinforce" | "uncertain";

export interface TuningAnnotationSupervision {
  origin: TuningFeedbackOrigin;
  relation: TuningFeedbackRelation;
  originalLabel: 0 | 1 | null;
  effectiveLabel: 0 | 1 | null;
  includedInTune: boolean;
  excludedReason?: "fixed-vqa-validation" | string;
}

export interface TuningAnnotation {
  rowIndex: number;
  imageId: string;
  label: FeedbackValue;
  source: string;
  updatedAt: string;
  failedAttributeIds?: string[];
  suggestedFailedAttributeId?: string | null;
  failureAttributionConfirmed?: boolean;
  supervision?: TuningAnnotationSupervision;
}

/** The robustness pilot can explicitly reject a relation without labelling an attribute. */
export function supportsRelationMismatch(taskId: string | undefined, targetId: string): boolean {
  return taskId === "059_hico_task_hico_hugging_cat_robust_test" && targetId === "joint";
}

export function hasConfirmedNegativeFeedback(
  annotation: Pick<TuningAnnotation, "label" | "failureAttributionConfirmed" | "failedAttributeIds">,
  taskId: string | undefined,
  targetId: string,
): boolean {
  return annotation.label < 0 && annotation.failureAttributionConfirmed === true
    && (Boolean(annotation.failedAttributeIds?.length)
      || (Array.isArray(annotation.failedAttributeIds) && supportsRelationMismatch(taskId, targetId)));
}

export interface TuningAnnotationCounts {
  strongNegative: number;
  negative: number;
  uncertain: number;
  positive: number;
  strongPositive: number;
  usablePositive: number;
  usableNegative: number;
}

export interface TuningEvaluationMetrics {
  ap: number;
  bestF1?: number;
  bestF1Cutoff?: number;
  evaluationCount?: number;
  positiveCount?: number;
  testCount?: number;
  tpAt30: number;
  tpAt50: number;
  tpAt100: number;
  tpAt200: number;
  modelSummary?: {
    methodWeightsByAttribute?: Record<string, Record<string, number>>;
    methodWeightCount?: number;
    optimizedMethodWeightCount?: number;
    learnerWeightsByAttribute?: Record<string, Record<string, number>>;
    activeAttributeIds?: string[];
    attributeWeights?: Record<string, number>;
    activeJointAttributeIds?: string[];
    attributeWeightCount?: number;
    optimizedAttributeWeightCount?: number;
    jointAggregation?: "product" | string;
    jointWeightConstraint?: string;
    learnerWeights?: Record<string, number>;
    beta?: Record<string, Record<string, number>>;
    gamma?: Record<string, number>;
    embeddingWeights?: Record<string, number>;
    embeddingFusionStrength?: number;
    theta?: Record<string, number>;
    /** Run-pinned policies; older runs may omit these or retain their original calibration policy. */
    gateCalibrationPolicy?: string;
    thetaCalibration?: string;
    temperaturePolicy?: string;
    probeUpdated?: boolean;
    probeState?: {
      updated: boolean;
      source: string;
      snapshotId?: string;
      snapshotFingerprint?: string;
      baseBankFingerprint?: string;
      sharedAcrossFusionSchedules?: boolean;
      frozenDuringFusion?: boolean;
    };
    nativeProbeUpdate?: Record<string, unknown>;
    minMaxStats?: {
      probe: Record<string, Record<string, { min: number; max: number }>>;
      embedding: Record<string, { min: number; max: number }>;
    };
    trainingHistory?: Array<Record<string, unknown>>;
    residualCoefficients?: Record<string, number>;
    bias?: number;
    biasAffectsRanking?: boolean;
    initialObjective?: number;
    finalObjective?: number;
    iterations?: number;
  };
  supervisionSummary?: {
    originalDevelopmentCount: number;
    originalProbeSupervisionCount?: number;
    originalProbeFitCount?: number;
    probeValidationCount?: number;
    feedbackSnapshotCount?: number;
    feedbackHoldoutExcludedCount?: number;
    feedbackCount: number;
    newFeedbackCount?: number;
    feedbackExistingCount?: number;
    feedbackOverrideCount?: number;
    feedbackReinforceCount?: number;
    feedbackReviewOnlyCount?: number;
    feedbackWeight: number;
    combinedCount: number;
  };
  splitAudit?: TuningSplitAudit;
}

export interface TuningSplitAudit {
  schemaVersion: number;
  protocol: string;
  replayKind?: "exact-attribute-seed0" | "exact-attribute-seed" | "derived-joint-probe-style" | string;
  seed?: number;
  validationFraction?: number;
  targetId: string;
  targetKind: string;
  sourceCount?: number;
  fitCount?: number;
  validationCount: number;
  validationPositiveCount: number;
  validationNegativeCount: number;
  sourceFingerprint?: string;
  fitFingerprint?: string;
  validationFingerprint: string;
  feedbackHoldoutExcludedCount?: number;
  vqaValidationVersion?: string;
  vqaValidationManifestSha256?: string;
  labelSource?: string;
  initialModelHoldoutIndependent?: boolean;
  frozenVersion?: string;
  frozenManifestSha256?: string;
  evaluationLabelSource?: string;
  referenceOnly?: boolean;
}

/** Read-only evaluation of the finalized run, never an optimization input. */
export interface TuningTestEvaluation {
  before: TuningEvaluationMetrics;
  after: TuningEvaluationMetrics;
  targetId: string;
  beforeMethod: string;
  testMaskSha256: string;
  groundTruthSha256: string;
}

export interface TuningRun {
  id: string;
  sessionId: string;
  mode: TuningMode;
  baseMethod: string;
  beforeMethod?: string;
  status: TuningRunStatus;
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  error?: string | null;
  before?: TuningEvaluationMetrics;
  after?: TuningEvaluationMetrics;
  deltaAp?: number;
  testEvaluation?: TuningTestEvaluation;
  testEvaluationError?: string;
  ranksUrl?: string;
  scoresUrl?: string;
  visualizationUrl?: string;
  clusterUrl?: string;
  annotationCount?: number;
  positiveCount?: number;
  negativeCount?: number;
  evaluationScope?: "vqa-validation" | "clean-validation" | "probe-validation" | "validation" | "test" | string;
  vqaValidationVersion?: string;
  vqaValidationManifestSha256?: string;
  evaluationLabelSource?: string;
  initialModelHoldoutIndependent?: boolean;
  referenceOnly?: boolean;
  splitAudit?: TuningSplitAudit;
  algorithmVersion?: string;
  gateCalibrationPolicy?: string;
  thetaCalibration?: string;
  temperaturePolicy?: string;
  probeSource?: ProbeSource;
  probeUpdateId?: string | null;
  probeSnapshotId?: string | null;
  /** Ordered identities frozen in this run, not the current task defaults. */
  attributeIds?: string[];
  embeddingMethods?: string[];
  supervisionPolicy?: string;
  legacyMode?: boolean;
  stale?: boolean;
}

export function isHierarchicalRankFusionRun(
  run: TuningRun | null | undefined,
): boolean {
  return Boolean(
    run
    && run.mode === "fusion-weight"
    && run.algorithmVersion === HIERARCHICAL_RANK_FUSION_ALGORITHM
    && run.after?.modelSummary?.methodWeightsByAttribute
  );
}

export function isWeightRefinementRun(
  run: TuningRun | null | undefined,
): boolean {
  return Boolean(
    run
    && (
      (run.mode === "weight_staged" && (
        run.algorithmVersion === WEIGHT_STAGED_ALGORITHM
        || run.algorithmVersion === WEIGHT_STAGED_ALGORITHM_V4
        || run.algorithmVersion === WEIGHT_STAGED_ALGORITHM_V3
        || run.algorithmVersion === WEIGHT_STAGED_ALGORITHM_V2
      ))
      || (run.mode === "weight_joint" && (
        run.algorithmVersion === WEIGHT_JOINT_ALGORITHM
        || run.algorithmVersion === WEIGHT_JOINT_ALGORITHM_V4
        || run.algorithmVersion === WEIGHT_JOINT_ALGORITHM_V3
        || run.algorithmVersion === WEIGHT_JOINT_ALGORITHM_V2
      ))
      || isNativeProbeRefinementAlgorithm(run.mode, run.algorithmVersion)
    )
    && run.visualizationUrl
    && run.clusterUrl,
  );
}

/** Resolve run-pinned holistic identities; native-updated probes still use two embeddings. */
export function refinementEmbeddingMethods(run: TuningRun | null | undefined) {
  if (!run) return null;
  const twoChannel = (run.mode === "weight_staged" && (
    run.algorithmVersion === WEIGHT_STAGED_ALGORITHM
    || run.algorithmVersion === WEIGHT_STAGED_ALGORITHM_V4
    || run.algorithmVersion === WEIGHT_STAGED_ALGORITHM_V3
  )) || (run.mode === "weight_joint" && (
    run.algorithmVersion === WEIGHT_JOINT_ALGORITHM
    || run.algorithmVersion === WEIGHT_JOINT_ALGORITHM_V4
    || run.algorithmVersion === WEIGHT_JOINT_ALGORITHM_V3
  )) || isNativeProbeRefinementAlgorithm(run.mode, run.algorithmVersion);
  const historical = (run.mode === "weight_staged" && run.algorithmVersion === WEIGHT_STAGED_ALGORITHM_V2)
    || (run.mode === "weight_joint" && run.algorithmVersion === WEIGHT_JOINT_ALGORITHM_V2);
  const methods = twoChannel ? REFINEMENT_EMBEDDING_METHODS
    : historical ? LEGACY_REFINEMENT_EMBEDDING_METHODS : null;
  if (!methods) return null;
  if (run.embeddingMethods && (
    run.embeddingMethods.length !== methods.length
    || run.embeddingMethods.some((method, index) => method !== methods[index])
  )) return null;
  const weights = run.after?.modelSummary?.embeddingWeights;
  if (weights && (
    Object.keys(weights).length !== methods.length
    || methods.some((method) => !Number.isFinite(weights[method]) || weights[method] < 0)
    || Math.abs(methods.reduce((sum, method) => sum + weights[method], 0) - 1) > 1e-5
  )) return null;
  return methods;
}

export function isPerAttributeFusionRun(run: TuningRun | null | undefined): boolean {
  return Boolean(
    isHierarchicalRankFusionRun(run)
    || (
      run
      && run.mode === "fusion-weight"
      && (
        run.algorithmVersion === PER_ATTRIBUTE_FUSION_ALGORITHM_V3
        || run.algorithmVersion === PER_ATTRIBUTE_FUSION_ALGORITHM_V2
      )
      && run.after?.modelSummary?.learnerWeightsByAttribute
    )
  );
}

export interface TuningBootstrap {
  user: TuningUser;
  session: TuningSession;
  annotations: TuningAnnotation[];
  runs: TuningRun[];
  refinementCapabilities?: TuningRefinementCapabilities;
  probeUpdates?: ProbeUpdateJob[];
}

export type OriginalVqaLabel = 0 | 1;

export interface OriginalVqaSupervisionItem {
  rowIndex: number;
  imageId: string;
  label: OriginalVqaLabel;
  /** Whether this original label is eligible for Personal Tune training. */
  trainableInDevelopment: boolean;
}

export interface OriginalVqaSupervisionResponse {
  taskId: string;
  targetId: string;
  selectedCount: number;
  developmentCount: number;
  items: OriginalVqaSupervisionItem[];
}

/** Task-wide original VQA holdout; labels are not needed by exploration views. */
export interface FixedVqaValidationResponse {
  taskId: string;
  version: string;
  manifestSha256: string;
  rowIndices: number[];
  count: number;
  protocol: string;
  labelSource: "original-vqa-supervision" | "user-provided-supervision";
  initialModelHoldoutIndependent: false;
  referenceOnly?: true;
}

const API_ROOT = "/api/tuning";

async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    credentials: "same-origin",
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = `Tuning service returned ${response.status}.`;
    try {
      const payload = await response.json() as { error?: string; message?: string };
      message = payload.error || payload.message || message;
    } catch {
      // Keep the status-based fallback when the service did not return JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function bootstrapTuning(input: {
  taskId: string;
  targetId: string;
  baseMethod: string;
  displayName?: string;
  newSession?: boolean;
}) {
  return requestJson<TuningBootstrap>("/bootstrap", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchOriginalVqaSupervision(
  taskId: string,
  targetId: string,
  signal?: AbortSignal,
) {
  return requestJson<OriginalVqaSupervisionResponse>(
    `/tasks/${encodeURIComponent(taskId)}/targets/${encodeURIComponent(targetId)}/original-supervision`,
    { signal },
  );
}

export function fetchFixedVqaValidation(taskId: string, signal?: AbortSignal) {
  return requestJson<FixedVqaValidationResponse>(
    `/tasks/${encodeURIComponent(taskId)}/validation`,
    { signal, cache: "no-store" },
  );
}

export async function fetchRefinementCapabilities(taskId: string, signal?: AbortSignal) {
  const payload = await requestJson<{
    taskId: string;
    refinementCapabilities?: TuningRefinementCapabilities;
  }>(`/tasks/${encodeURIComponent(taskId)}/refinement-capabilities`, { signal, cache: "no-store" });
  if (payload.taskId !== taskId) throw new Error("Tuning capabilities belong to a different task.");
  return payload.refinementCapabilities ?? null;
}

export function putTuningAnnotation(
  sessionId: string,
  annotation: Pick<TuningAnnotation, "rowIndex" | "imageId" | "label"> & {
    source?: string;
    failedAttributeIds?: string[];
    suggestedFailedAttributeId?: string | null;
    failureAttributionConfirmed?: boolean;
  },
) {
  return requestJson<{
    annotation: TuningAnnotation;
    counts: TuningAnnotationCounts;
  }>(`/sessions/${encodeURIComponent(sessionId)}/annotations/${annotation.rowIndex}`, {
    method: "PUT",
    body: JSON.stringify({
      imageId: annotation.imageId,
      label: annotation.label,
      source: annotation.source ?? "top-gallery",
      ...(annotation.failedAttributeIds !== undefined
        ? { failedAttributeIds: annotation.failedAttributeIds }
        : {}),
      ...(annotation.suggestedFailedAttributeId !== undefined
        ? { suggestedFailedAttributeId: annotation.suggestedFailedAttributeId }
        : {}),
      ...(annotation.failureAttributionConfirmed !== undefined
        ? { failureAttributionConfirmed: annotation.failureAttributionConfirmed }
        : {}),
    }),
  });
}

export function putTuningAnnotationsBulk(
  sessionId: string,
  annotations: ReadonlyArray<Pick<TuningAnnotation, "rowIndex" | "imageId" | "label">>,
  source = "selection-bulk",
) {
  if (annotations.length === 0) {
    throw new Error("Choose at least one image before applying a bulk label.");
  }
  if (annotations.length > MAX_BULK_TUNING_ANNOTATIONS) {
    throw new Error(
      `Bulk feedback is limited to ${MAX_BULK_TUNING_ANNOTATIONS.toLocaleString()} images.`,
    );
  }
  return requestJson<{
    annotations: TuningAnnotation[];
    counts: TuningAnnotationCounts;
  }>(`/sessions/${encodeURIComponent(sessionId)}/annotations/bulk`, {
    method: "POST",
    body: JSON.stringify({
      source,
      annotations: annotations.map((annotation) => ({
        rowIndex: annotation.rowIndex,
        imageId: annotation.imageId,
        label: annotation.label,
      })),
    }),
  });
}

export function deleteTuningAnnotation(sessionId: string, rowIndex: number) {
  return requestJson<{ deleted: boolean; counts: TuningAnnotationCounts }>(
    `/sessions/${encodeURIComponent(sessionId)}/annotations/${rowIndex}`,
    { method: "DELETE" },
  );
}

export function createTuningRun(
  sessionId: string,
  input: { mode: TuningLaunchMode; baseMethod: string; probeSource?: ProbeSource; probeUpdateId?: string },
) {
  return requestJson<{ run: TuningRun }>(
    `/sessions/${encodeURIComponent(sessionId)}/runs`,
    { method: "POST", body: JSON.stringify(input) },
  );
}

export function createProbeUpdate(sessionId: string) {
  return requestJson<{ probeUpdate: ProbeUpdateJob }>(
    `/sessions/${encodeURIComponent(sessionId)}/probe-updates`,
    { method: "POST", body: JSON.stringify({}) },
  );
}

export function fetchProbeUpdates(sessionId: string, signal?: AbortSignal) {
  return requestJson<{ probeUpdates: ProbeUpdateJob[] }>(
    `/sessions/${encodeURIComponent(sessionId)}/probe-updates`,
    { signal, cache: "no-store" },
  );
}

export function fetchProbeUpdate(updateId: string, signal?: AbortSignal) {
  return requestJson<{ probeUpdate: ProbeUpdateJob }>(
    `/probe-updates/${encodeURIComponent(updateId)}`,
    { signal, cache: "no-store" },
  );
}

export function compatibleProbeUpdates(jobs: readonly ProbeUpdateJob[], sessionId: string, taskId: string) {
  return jobs.filter((job) => (
    job.sessionId === sessionId && job.taskId === taskId
    && job.status === "succeeded" && job.compatible === true && Boolean(job.snapshotId)
  )).sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

/** A saved Updated selection is an exact job, never a floating "latest" alias. */
export function resolveProbeSelection(jobs: readonly ProbeUpdateJob[], sessionId: string, taskId: string, updateId: string | null) {
  return updateId ? compatibleProbeUpdates(jobs, sessionId, taskId).find((job) => job.id === updateId) ?? null : null;
}

export function fetchTuningRun(runId: string) {
  return requestJson<{ run: TuningRun }>(`/runs/${encodeURIComponent(runId)}`);
}

export async function fetchTunedRanks(run: TuningRun, expectedRows: number) {
  const path = run.ranksUrl || `${API_ROOT}/runs/${encodeURIComponent(run.id)}/tuned-ranks.f32`;
  const response = await fetch(path, { credentials: "same-origin" });
  if (!response.ok) throw new Error(`Unable to load tuned ranks (${response.status}).`);
  const buffer = await response.arrayBuffer();
  const ranks = new Float32Array(buffer);
  if (ranks.length !== expectedRows) {
    throw new Error(`Tuned rank length ${ranks.length} does not match ${expectedRows} images.`);
  }
  return ranks;
}

export async function fetchTunedScores(run: TuningRun, expectedRows: number) {
  const path = run.scoresUrl || `${API_ROOT}/runs/${encodeURIComponent(run.id)}/tuned-scores.f32`;
  const response = await fetch(path, { credentials: "same-origin" });
  if (!response.ok) throw new Error(`Unable to load tuned scores (${response.status}).`);
  const buffer = await response.arrayBuffer();
  const scores = new Float32Array(buffer);
  if (scores.length !== expectedRows) {
    throw new Error(`Tuned score length ${scores.length} does not match ${expectedRows} images.`);
  }
  return scores;
}
