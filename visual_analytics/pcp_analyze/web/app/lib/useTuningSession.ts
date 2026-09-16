"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { GalleryItem } from "../components/TopGallery";
import {
  feedbackLabel,
  feedbackValue,
  type FeedbackLabel,
} from "./feedbackLabels";
import {
  bootstrapTuning,
  createTuningRun,
  createProbeUpdate,
  fetchProbeUpdate,
  fetchProbeUpdates,
  compatibleProbeUpdates,
  resolveProbeSelection,
  probeUpdateCapability,
  deleteTuningAnnotation,
  fetchTunedRanks,
  fetchTunedScores,
  fetchTuningRun,
  fetchRefinementCapabilities,
  refinementCapability,
  supportsRelationMismatch,
  hasConfirmedNegativeFeedback,
  MAX_BULK_TUNING_ANNOTATIONS,
  putTuningAnnotation,
  putTuningAnnotationsBulk,
  type TuningAnnotation,
  type TuningAnnotationCounts,
  type TuningBootstrap,
  type TuningLaunchMode,
  type TuningRun,
  type TuningRefinementCapabilities,
  type TuningSession,
  type TuningUser,
  type ProbeUpdateJob,
  type ProbeSource,
  isWeightRefinementRun,
  tuningBaseMethod,
} from "./tuningApi";
import {
  requestRefinementClusters,
  requestRefinementVisualization,
  type RefinementVisualizationSnapshot,
} from "./refinementVisualization";

type ServiceState = "idle" | "loading" | "ready" | "offline";

const EMPTY_COUNTS: TuningAnnotationCounts = {
  strongNegative: 0,
  negative: 0,
  uncertain: 0,
  positive: 0,
  strongPositive: 0,
  usablePositive: 0,
  usableNegative: 0,
};

function countsFromAnnotations(annotations: ReadonlyMap<string, TuningAnnotation>) {
  const counts = { ...EMPTY_COUNTS };
  for (const annotation of annotations.values()) {
    if (annotation.label === -2) counts.strongNegative += 1;
    else if (annotation.label === -1) counts.negative += 1;
    else if (annotation.label === 0) counts.uncertain += 1;
    else if (annotation.label === 1) counts.positive += 1;
    else if (annotation.label === 2) counts.strongPositive += 1;
    if (annotation.supervision?.includedInTune !== false) {
      if (annotation.label > 0) counts.usablePositive += 1;
      else if (annotation.label < 0) counts.usableNegative += 1;
    }
  }
  return counts;
}

export interface TuningFeedbackBreakdown {
  totalCount: number;
  newCount: number;
  existingCount: number;
  trainableNewCount: number;
  overrideCount: number;
  reinforceCount: number;
  reviewOnlyCount: number;
  unresolvedCount: number;
  holdoutExcludedCount: number;
}

function feedbackBreakdownFromAnnotations(
  annotations: ReadonlyMap<string, TuningAnnotation>,
): TuningFeedbackBreakdown {
  const summary: TuningFeedbackBreakdown = {
    totalCount: annotations.size,
    newCount: 0,
    existingCount: 0,
    trainableNewCount: 0,
    overrideCount: 0,
    reinforceCount: 0,
    reviewOnlyCount: 0,
    unresolvedCount: 0,
    holdoutExcludedCount: 0,
  };
  for (const annotation of annotations.values()) {
    const supervision = annotation.supervision;
    if (!supervision) {
      summary.unresolvedCount += 1;
      continue;
    }
    if (supervision.origin === "new") summary.newCount += 1;
    else summary.existingCount += 1;
    if (supervision.excludedReason === "fixed-vqa-validation") {
      summary.holdoutExcludedCount += 1;
    } else if (!supervision.includedInTune) {
      summary.reviewOnlyCount += 1;
    } else if (supervision.relation === "new") {
      summary.trainableNewCount += 1;
    } else if (supervision.relation === "override") {
      summary.overrideCount += 1;
    } else if (supervision.relation === "reinforce") {
      summary.reinforceCount += 1;
    }
  }
  return summary;
}

function annotationMap(payload: TuningBootstrap) {
  return new Map(payload.annotations.map((annotation) => [annotation.imageId, annotation]));
}

function trainableFeedbackValue(annotation: TuningAnnotation | null | undefined) {
  return annotation && annotation.label !== 0 ? annotation.label : null;
}

function probeSelectionStorageKey(userId: string, sessionId: string, taskId: string) {
  return `pcp:probe-source:${userId}:${sessionId}:${taskId}`;
}

function readProbeSelection(userId: string, sessionId: string, taskId: string): string | null {
  try { return window.localStorage.getItem(probeSelectionStorageKey(userId, sessionId, taskId)); }
  catch { return null; }
}

function persistProbeSelection(userId: string, sessionId: string, taskId: string, updateId: string | null) {
  try {
    const key = probeSelectionStorageKey(userId, sessionId, taskId);
    if (updateId) window.localStorage.setItem(key, updateId);
    else window.localStorage.removeItem(key);
  } catch { /* Storage is optional; the live request still pins the exact snapshot. */ }
}

export interface TuningSessionState {
  serviceState: ServiceState;
  user: TuningUser | null;
  session: TuningSession | null;
  refinementCapabilities: TuningRefinementCapabilities | null;
  probeUpdates: readonly ProbeUpdateJob[];
  probeUpdateInProgress: boolean;
  probeSource: ProbeSource;
  selectedProbeUpdate: ProbeUpdateJob | null;
  selectProbeSource: (source: ProbeSource, updateId?: string) => void;
  updateProbes: () => Promise<void>;
  annotations: ReadonlyMap<string, TuningAnnotation>;
  preferences: ReadonlyMap<string, FeedbackLabel>;
  counts: TuningAnnotationCounts;
  feedbackBreakdown: TuningFeedbackBreakdown;
  run: TuningRun | null;
  appliedRanks: Float32Array | null;
  appliedScores: Float32Array | null;
  appliedVisualization: RefinementVisualizationSnapshot | null;
  appliedRunId: string | null;
  resultStale: boolean;
  pendingAnnotationCount: number;
  busy: boolean;
  error: string | null;
  saveDisplayName: (displayName: string) => Promise<void>;
  startNewSession: () => Promise<void>;
  updateAnnotation: (item: GalleryItem, label: FeedbackLabel, source?: string) => Promise<void>;
  bulkUpdateAnnotations: (
    items: readonly GalleryItem[],
    label: FeedbackLabel,
    source?: string,
  ) => Promise<void>;
  updateFailureAttributes: (
    item: GalleryItem,
    failedAttributeIds: readonly string[],
  ) => Promise<void>;
  clearAnnotations: () => Promise<void>;
  runTuning: (mode: TuningLaunchMode) => Promise<void>;
  applyRun: (options?: { refinementClusterScheme?: string }) => Promise<void>;
  revertRun: () => void;
}

export function useTuningSession(input: {
  enabled: boolean;
  taskId: string;
  targetId: string;
  baseMethod: string;
  rowCount: number;
  canWriteFeedback: (item: GalleryItem) => boolean;
  canRemoveFeedback: (item: GalleryItem) => boolean;
  validationReady: boolean;
}): TuningSessionState {
  const { enabled, taskId, targetId, baseMethod, rowCount } = input;
  const [serviceState, setServiceState] = useState<ServiceState>("idle");
  const [user, setUser] = useState<TuningUser | null>(null);
  const [session, setSession] = useState<TuningSession | null>(null);
  const [refinementCapabilities, setRefinementCapabilities] = useState<TuningRefinementCapabilities | null>(null);
  const [probeUpdates, setProbeUpdates] = useState<ProbeUpdateJob[]>([]);
  const [selectedProbeUpdateId, setSelectedProbeUpdateId] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<Map<string, TuningAnnotation>>(new Map());
  const [run, setRun] = useState<TuningRun | null>(null);
  const [appliedRanks, setAppliedRanks] = useState<Float32Array | null>(null);
  const [appliedScores, setAppliedScores] = useState<Float32Array | null>(null);
  const [appliedVisualization, setAppliedVisualization] =
    useState<RefinementVisualizationSnapshot | null>(null);
  const [appliedRunId, setAppliedRunId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [pendingAnnotationCount, setPendingAnnotationCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [annotationRevision, setAnnotationRevision] = useState(0);
  const [runRevision, setRunRevision] = useState(0);
  const requestGeneration = useRef(0);
  const probeListRevisionRef = useRef(0);
  const activeSessionIdRef = useRef<string | null>(null);
  const currentRunIdRef = useRef<string | null>(null);
  const annotationRevisionRef = useRef(0);
  const rankRequestGenerationRef = useRef(0);
  const sessionMutationLockRef = useRef<symbol | null>(null);
  const busyOperationRef = useRef<symbol | null>(null);
  const rankBusyTokenRef = useRef<symbol | null>(null);
  const confirmedAnnotationsRef = useRef(new Map<string, TuningAnnotation>());
  const annotationSequenceRef = useRef(new Map<string, number>());
  const annotationTailsRef = useRef(new Map<string, Promise<void>>());
  const pendingAnnotationWritesRef = useRef(new Map<
    Promise<void>,
    { generation: number; sessionId: string }
  >());
  const submittedRevisionByRun = useRef(new Map<string, number>());
  const feedbackContextRef = useRef({ input, session });
  useLayoutEffect(() => {
    feedbackContextRef.current = { input, session };
  }, [input, session]);
  const assertFeedbackAllowed = useCallback((item: GalleryItem, sessionId: string, removal = false) => {
    const context = feedbackContextRef.current;
    if (
      !context.input.enabled
      || !context.input.validationReady
      || context.session?.id !== sessionId
      || context.session.taskId !== context.input.taskId
      || context.session.targetId !== context.input.targetId
      || context.session.baseMethod !== context.input.baseMethod
      || !(removal ? context.input.canRemoveFeedback(item) : context.input.canWriteFeedback(item))
    ) throw new Error("Feedback is locked outside Development Gallery or on fixed Val images.");
  }, []);

  const applyBootstrap = useCallback((payload: TuningBootstrap) => {
    const nextAnnotations = annotationMap(payload);
    const nextRun = payload.runs[0] ?? null;
    activeSessionIdRef.current = payload.session.id;
    currentRunIdRef.current = nextRun?.id ?? null;
    confirmedAnnotationsRef.current = new Map(nextAnnotations);
    annotationRevisionRef.current = 0;
    setUser(payload.user);
    setSession(payload.session);
    setRefinementCapabilities(payload.refinementCapabilities ?? null);
    const updates = (payload.probeUpdates ?? []).filter((job) => (
      job.sessionId === payload.session.id && job.taskId === payload.session.taskId
    ));
    setProbeUpdates(updates);
    const savedId = readProbeSelection(payload.user.id, payload.session.id, payload.session.taskId);
    // Never silently replace a saved Updated source with Original when its
    // snapshot becomes incompatible; keep the choice visibly locked for review.
    setSelectedProbeUpdateId(savedId);
    setAnnotations(nextAnnotations);
    setRun(nextRun);
    setAppliedRanks(null);
    setAppliedScores(null);
    setAppliedVisualization(null);
    setAppliedRunId(null);
    setAnnotationRevision(0);
    setRunRevision(nextRun?.stale ? -1 : 0);
    setServiceState("ready");
    setError(null);
  }, []);

  const isActiveSessionOperation = useCallback((generation: number, sessionId: string) => (
    generation === requestGeneration.current
    && activeSessionIdRef.current === sessionId
  ), []);

  const refreshPendingAnnotationCount = useCallback(() => {
    const generation = requestGeneration.current;
    const sessionId = activeSessionIdRef.current;
    let count = 0;
    if (sessionId) {
      for (const pending of pendingAnnotationWritesRef.current.values()) {
        if (pending.generation === generation && pending.sessionId === sessionId) count += 1;
      }
    }
    setPendingAnnotationCount(count);
  }, []);

  const trackPendingAnnotation = useCallback((
    operation: Promise<void>,
    generation: number,
    sessionId: string,
  ) => {
    pendingAnnotationWritesRef.current.set(operation, { generation, sessionId });
    refreshPendingAnnotationCount();
    void operation.then(
      () => {
        pendingAnnotationWritesRef.current.delete(operation);
        refreshPendingAnnotationCount();
      },
      () => {
        pendingAnnotationWritesRef.current.delete(operation);
        refreshPendingAnnotationCount();
      },
    );
    return operation;
  }, [refreshPendingAnnotationCount]);

  const waitForPendingAnnotations = useCallback(async (
    generation: number,
    sessionId: string,
  ) => {
    while (isActiveSessionOperation(generation, sessionId)) {
      const pending = [...pendingAnnotationWritesRef.current.entries()]
        .filter(([, owner]) => (
          owner.generation === generation && owner.sessionId === sessionId
        ))
        .map(([operation]) => operation);
      if (pending.length === 0) return;
      const settled = await Promise.allSettled(pending);
      if (!isActiveSessionOperation(generation, sessionId)) return;
      const failed = settled.find(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (failed) throw failed.reason;
    }
  }, [isActiveSessionOperation]);

  const advanceAnnotationRevision = useCallback(() => {
    annotationRevisionRef.current += 1;
    setAnnotationRevision(annotationRevisionRef.current);
    probeListRevisionRef.current += 1;
    setProbeUpdates((jobs) => jobs.map((job) => job.status === "succeeded" ? { ...job, stale: true } : job));
  }, []);

  const bootstrap = useCallback(async (options?: {
    displayName?: string;
    newSession?: boolean;
  }) => {
    if (!enabled || !taskId || !targetId || !baseMethod) return;
    const generation = ++requestGeneration.current;
    activeSessionIdRef.current = null;
    currentRunIdRef.current = null;
    sessionMutationLockRef.current = null;
    busyOperationRef.current = null;
    rankBusyTokenRef.current = null;
    rankRequestGenerationRef.current += 1;
    annotationSequenceRef.current.clear();
    annotationTailsRef.current.clear();
    setPendingAnnotationCount(0);
    setServiceState("loading");
    setRefinementCapabilities(null);
    setProbeUpdates([]);
    setSelectedProbeUpdateId(null);
    setBusy(true);
    try {
      const payload = await bootstrapTuning({
        taskId,
        targetId,
        baseMethod,
        ...options,
      });
      if (generation === requestGeneration.current) applyBootstrap(payload);
    } catch (reason) {
      if (generation !== requestGeneration.current) return;
      setServiceState("offline");
      setError(reason instanceof Error ? reason.message : String(reason));
      setSession(null);
      setAnnotations(new Map());
      confirmedAnnotationsRef.current = new Map();
      setRun(null);
      currentRunIdRef.current = null;
      setAppliedRanks(null);
      setAppliedScores(null);
      setAppliedVisualization(null);
      setAppliedRunId(null);
    } finally {
      if (generation === requestGeneration.current) setBusy(false);
    }
  }, [applyBootstrap, baseMethod, enabled, targetId, taskId]);

  useEffect(() => {
    if (!enabled) {
      requestGeneration.current += 1;
      activeSessionIdRef.current = null;
      currentRunIdRef.current = null;
      sessionMutationLockRef.current = null;
      busyOperationRef.current = null;
      rankBusyTokenRef.current = null;
      rankRequestGenerationRef.current += 1;
      const timer = window.setTimeout(() => {
        setPendingAnnotationCount(0);
        setServiceState("idle");
        setUser(null);
        setSession(null);
        setRefinementCapabilities(null);
        setProbeUpdates([]);
        setSelectedProbeUpdateId(null);
        setAnnotations(new Map());
        confirmedAnnotationsRef.current = new Map();
        setRun(null);
        setAppliedRanks(null);
        setAppliedScores(null);
        setAppliedVisualization(null);
        setAppliedRunId(null);
        setBusy(false);
        setError(null);
      }, 0);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(() => void bootstrap(), 0);
    return () => {
      window.clearTimeout(timer);
      requestGeneration.current += 1;
      activeSessionIdRef.current = null;
      currentRunIdRef.current = null;
      sessionMutationLockRef.current = null;
      busyOperationRef.current = null;
      rankBusyTokenRef.current = null;
      rankRequestGenerationRef.current += 1;
    };
  }, [bootstrap, enabled]);

  useEffect(() => {
    if (!run || run.status !== "queued" && run.status !== "running") return;
    const generation = requestGeneration.current;
    const sessionId = run.sessionId;
    const runId = run.id;
    let cancelled = false;
    let timer: number | null = null;
    let retryDelay = 1200;
    const schedulePoll = (delay: number) => {
      timer = window.setTimeout(() => {
        void fetchTuningRun(runId)
        .then(({ run: nextRun }) => {
          if (
            cancelled
            || !isActiveSessionOperation(generation, sessionId)
            || currentRunIdRef.current !== runId
          ) return;
          if (nextRun.id !== runId || nextRun.sessionId !== sessionId) {
            setError("The tuning service returned polling data for a different run.");
            retryDelay = Math.min(retryDelay * 2, 9600);
            schedulePoll(retryDelay);
            return;
          }
          currentRunIdRef.current = nextRun.id;
          setRun(nextRun);
          setError(null);
          if (nextRun.status === "succeeded") {
            setRunRevision(submittedRevisionByRun.current.get(nextRun.id) ?? runRevision);
          } else if (nextRun.status === "failed") {
            setError(nextRun.error || "The tuning run failed.");
          }
        })
        .catch((reason) => {
          if (
            cancelled
            || !isActiveSessionOperation(generation, sessionId)
            || currentRunIdRef.current !== runId
          ) return;
          setError(reason instanceof Error ? reason.message : String(reason));
          retryDelay = Math.min(retryDelay * 2, 9600);
          schedulePoll(retryDelay);
        });
      }, delay);
    };
    schedulePoll(retryDelay);
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [isActiveSessionOperation, run, runRevision]);

  const probeUpdateInProgress = probeUpdates.some((job) => job.status === "queued" || job.status === "running");
  const selectedProbeUpdate = session
    ? resolveProbeSelection(probeUpdates, session.id, taskId, selectedProbeUpdateId)
    : null;
  const probeSource: ProbeSource = selectedProbeUpdateId ? "updated" : "original";

  // Probe jobs are separate from ranking runs: completing an update never applies
  // ranks, replaces the previous Tune card, or silently changes the chosen source.
  useEffect(() => {
    if (!enabled || serviceState !== "ready" || !session) return;
    const generation = requestGeneration.current;
    const sessionId = session.id;
    const controller = new AbortController();
    let timer: number | null = null;
    const refresh = async () => {
      const listRevision = probeListRevisionRef.current;
      try {
        const response = await fetchProbeUpdates(sessionId, controller.signal);
        if (controller.signal.aborted || !isActiveSessionOperation(generation, sessionId)) return;
        if (listRevision !== probeListRevisionRef.current) return;
        if (response.probeUpdates.some((job) => job.sessionId !== sessionId || job.taskId !== taskId)) {
          throw new Error("Probe updates belong to a different session.");
        }
        setProbeUpdates(response.probeUpdates);
        if (response.probeUpdates.some((job) => job.status === "queued" || job.status === "running")) {
          timer = window.setTimeout(() => void refresh(), 1600);
        }
      } catch (reason) {
        if (controller.signal.aborted || !isActiveSessionOperation(generation, sessionId)) return;
        setError(reason instanceof Error ? reason.message : String(reason));
        if (probeUpdateInProgress) timer = window.setTimeout(() => void refresh(), 3200);
      }
    };
    timer = window.setTimeout(() => void refresh(), probeUpdateInProgress ? 900 : 150);
    return () => {
      controller.abort();
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [annotationRevision, enabled, isActiveSessionOperation, probeUpdateInProgress, serviceState, session, taskId]);

  const selectProbeSource = useCallback((source: ProbeSource, updateId?: string) => {
    if (!session || !user || serviceState !== "ready" || busy || sessionMutationLockRef.current || probeUpdateInProgress
      || run?.status === "queued" || run?.status === "running"
      || !isActiveSessionOperation(requestGeneration.current, session.id)) return;
    const candidates = compatibleProbeUpdates(probeUpdates, session.id, taskId);
    const selected = source === "updated"
      ? (updateId ? candidates.find((job) => job.id === updateId) : candidates[0]) : null;
    if (source === "updated" && !selected) {
      setError("Update Probes first to create a compatible frozen snapshot.");
      return;
    }
    const selectedId = selected?.id ?? null;
    setSelectedProbeUpdateId(selectedId);
    persistProbeSelection(user.id, session.id, taskId, selectedId);
  }, [busy, isActiveSessionOperation, probeUpdateInProgress, probeUpdates, run?.status, serviceState, session, taskId, user]);

  const preferences = useMemo(
    () => new Map(
      [...annotations].map(([imageId, annotation]) => [imageId, feedbackLabel(annotation.label)]),
    ),
    [annotations],
  );
  const counts = useMemo(() => countsFromAnnotations(annotations), [annotations]);
  const feedbackBreakdown = useMemo(
    () => feedbackBreakdownFromAnnotations(annotations),
    [annotations],
  );

  const saveDisplayName = useCallback(async (displayName: string) => {
    const normalized = displayName.trim();
    if (!normalized) throw new Error("Enter a user name before saving.");
    await bootstrap({ displayName: normalized });
  }, [bootstrap]);

  const startNewSession = useCallback(async () => {
    await bootstrap({ newSession: true });
  }, [bootstrap]);

  const updateAnnotation = useCallback((
    item: GalleryItem,
    label: FeedbackLabel,
    source = "top-gallery",
  ) => {
    if (!session) throw new Error("The tuning session is not ready.");
    assertFeedbackAllowed(item, session.id, label === "unmarked");
    const generation = requestGeneration.current;
    const sessionId = session.id;
    if (!isActiveSessionOperation(generation, sessionId)) {
      throw new Error("The tuning session is changing; wait for it to finish loading.");
    }
    if (sessionMutationLockRef.current) {
      throw new Error("Wait for the current tuning operation to finish before changing feedback.");
    }
    const numeric = feedbackValue(label);
    if (label !== "unmarked" && numeric === null) return Promise.resolve();
    const sequenceKey = `${sessionId}\u0000${item.rowIndex}`;
    const sequence = (annotationSequenceRef.current.get(sequenceKey) ?? 0) + 1;
    annotationSequenceRef.current.set(sequenceKey, sequence);
    setError(null);
    setAnnotations((current) => {
      const next = new Map(current);
      if (label === "unmarked") next.delete(item.id);
      else {
        if (numeric === null) return current;
        next.set(item.id, {
          rowIndex: item.rowIndex,
          imageId: item.id,
          label: numeric,
          source,
          updatedAt: new Date().toISOString(),
        });
      }
      return next;
    });

    // Serialize writes for one row. This preserves the user's click order while
    // still allowing different rows to save concurrently.
    const previousTail = annotationTailsRef.current.get(sequenceKey) ?? Promise.resolve();
    const operation = previousTail.catch(() => undefined).then(async () => {
      if (!isActiveSessionOperation(generation, sessionId)) return;
      try {
        assertFeedbackAllowed(item, sessionId, label === "unmarked");
        const previousConfirmed = confirmedAnnotationsRef.current.get(item.id);
        let confirmed: TuningAnnotation | null = null;
        if (label === "unmarked") {
          await deleteTuningAnnotation(sessionId, item.rowIndex);
        } else if (numeric !== null) {
          const response = await putTuningAnnotation(sessionId, {
            rowIndex: item.rowIndex,
            imageId: item.id,
            label: numeric,
            source,
          });
          confirmed = response.annotation;
        }
        if (!isActiveSessionOperation(generation, sessionId)) return;
        if (confirmed) confirmedAnnotationsRef.current.set(item.id, confirmed);
        else confirmedAnnotationsRef.current.delete(item.id);
        if (trainableFeedbackValue(previousConfirmed) !== trainableFeedbackValue(confirmed)) {
          advanceAnnotationRevision();
        }
        if (annotationSequenceRef.current.get(sequenceKey) === sequence) {
          setAnnotations((current) => {
            const next = new Map(current);
            if (confirmed) next.set(item.id, confirmed);
            else next.delete(item.id);
            return next;
          });
          setError(null);
        }
      } catch (reason) {
        if (
          isActiveSessionOperation(generation, sessionId)
          && annotationSequenceRef.current.get(sequenceKey) === sequence
        ) {
          const confirmed = confirmedAnnotationsRef.current.get(item.id);
          setAnnotations((current) => {
            const next = new Map(current);
            if (confirmed) next.set(item.id, confirmed);
            else next.delete(item.id);
            return next;
          });
          setError(reason instanceof Error ? reason.message : String(reason));
        }
        throw reason;
      }
    });
    const settledTail = operation.then(() => undefined, () => undefined);
    annotationTailsRef.current.set(sequenceKey, settledTail);
    void settledTail.then(() => {
      if (annotationTailsRef.current.get(sequenceKey) === settledTail) {
        annotationTailsRef.current.delete(sequenceKey);
      }
    });
    return trackPendingAnnotation(operation, generation, sessionId);
  }, [
    assertFeedbackAllowed,
    advanceAnnotationRevision,
    isActiveSessionOperation,
    session,
    trackPendingAnnotation,
  ]);

  const bulkUpdateAnnotations = useCallback(async (
    items: readonly GalleryItem[],
    label: FeedbackLabel,
    source = "selection-bulk",
  ) => {
    if (!session) throw new Error("The tuning session is not ready.");
    const numeric = feedbackValue(label);
    if (numeric === null) {
      throw new Error("Bulk feedback requires a positive, negative, or uncertain label.");
    }

    const uniqueByRow = new Map<number, GalleryItem>();
    for (const item of items) {
      assertFeedbackAllowed(item, session.id);
      const existing = uniqueByRow.get(item.rowIndex);
      if (existing && existing.id !== item.id) {
        throw new Error(`Row ${item.rowIndex} refers to more than one image.`);
      }
      uniqueByRow.set(item.rowIndex, item);
    }
    const uniqueItems = [...uniqueByRow.values()];
    if (uniqueItems.length === 0) {
      throw new Error("Choose at least one image before applying a bulk label.");
    }
    if (uniqueItems.length > MAX_BULK_TUNING_ANNOTATIONS) {
      throw new Error(
        `Bulk feedback is limited to ${MAX_BULK_TUNING_ANNOTATIONS.toLocaleString()} images.`,
      );
    }

    const generation = requestGeneration.current;
    const sessionId = session.id;
    if (!isActiveSessionOperation(generation, sessionId)) {
      throw new Error("The tuning session is changing; wait for it to finish loading.");
    }
    if (sessionMutationLockRef.current || busyOperationRef.current) {
      throw new Error("Another tuning operation is already in progress.");
    }

    const operationToken = Symbol("bulk-update-annotations");
    sessionMutationLockRef.current = operationToken;
    busyOperationRef.current = operationToken;
    setBusy(true);
    setError(null);
    try {
      await waitForPendingAnnotations(generation, sessionId);
      if (!isActiveSessionOperation(generation, sessionId)) return;

      const expected = new Map<string, (typeof uniqueItems)[number]>(
        uniqueItems.map((item) => [`${item.rowIndex}\u0000${item.id}`, item] as const),
      );
      for (const item of uniqueItems) assertFeedbackAllowed(item, sessionId);
      const response = await putTuningAnnotationsBulk(
        sessionId,
        uniqueItems.map((item) => ({
          rowIndex: item.rowIndex,
          imageId: item.id,
          label: numeric,
        })),
        source,
      );
      if (!isActiveSessionOperation(generation, sessionId)) return;

      const returnedKeys = new Set<string>();
      for (const annotation of response.annotations) {
        const key = `${annotation.rowIndex}\u0000${annotation.imageId}`;
        if (!expected.has(key) || returnedKeys.has(key) || annotation.label !== numeric) {
          throw new Error("The tuning service returned an invalid bulk annotation response.");
        }
        returnedKeys.add(key);
      }
      if (returnedKeys.size !== expected.size) {
        throw new Error("The tuning service returned an incomplete bulk annotation response.");
      }

      const previousConfirmed = confirmedAnnotationsRef.current;
      const nextConfirmed = new Map(previousConfirmed);
      let revisionChanged = false;
      for (const annotation of response.annotations) {
        if (
          trainableFeedbackValue(previousConfirmed.get(annotation.imageId))
          !== trainableFeedbackValue(annotation)
        ) {
          revisionChanged = true;
        }
        nextConfirmed.set(annotation.imageId, annotation);
      }
      confirmedAnnotationsRef.current = nextConfirmed;
      setAnnotations(new Map(nextConfirmed));
      if (revisionChanged) advanceAnnotationRevision();
      setError(null);
    } catch (reason) {
      if (isActiveSessionOperation(generation, sessionId)) {
        setAnnotations(new Map(confirmedAnnotationsRef.current));
        setError(reason instanceof Error ? reason.message : String(reason));
      }
      throw reason;
    } finally {
      if (sessionMutationLockRef.current === operationToken) {
        sessionMutationLockRef.current = null;
      }
      if (busyOperationRef.current === operationToken) {
        busyOperationRef.current = null;
        setBusy(false);
      }
    }
  }, [
    assertFeedbackAllowed,
    advanceAnnotationRevision,
    isActiveSessionOperation,
    session,
    waitForPendingAnnotations,
  ]);

  const updateFailureAttributes = useCallback((
    item: GalleryItem,
    failedAttributeIds: readonly string[],
  ) => {
    if (!session) throw new Error("The tuning session is not ready.");
    assertFeedbackAllowed(item, session.id);
    const generation = requestGeneration.current;
    const sessionId = session.id;
    if (!isActiveSessionOperation(generation, sessionId)) {
      throw new Error("The tuning session is changing; wait for it to finish loading.");
    }
    if (sessionMutationLockRef.current) {
      throw new Error("Wait for the current tuning operation to finish before changing feedback.");
    }
    const currentAnnotation = confirmedAnnotationsRef.current.get(item.id);
    if (!currentAnnotation || currentAnnotation.label >= 0) {
      throw new Error("Failed attributes can only be confirmed for a negative Joint label.");
    }
    const normalizedAttributeIds = [...new Set(
      failedAttributeIds.map((attributeId) => attributeId.trim()).filter(Boolean),
    )].sort();
    if (normalizedAttributeIds.length === 0 && !supportsRelationMismatch(session.taskId, session.targetId)) {
      throw new Error("Choose at least one failed attribute.");
    }

    const sequenceKey = `${sessionId}\u0000${item.rowIndex}`;
    const sequence = (annotationSequenceRef.current.get(sequenceKey) ?? 0) + 1;
    annotationSequenceRef.current.set(sequenceKey, sequence);
    const optimistic: TuningAnnotation = {
      ...currentAnnotation,
      failedAttributeIds: normalizedAttributeIds,
      failureAttributionConfirmed: true,
      updatedAt: new Date().toISOString(),
    };
    setError(null);
    setAnnotations((current) => {
      const next = new Map(current);
      next.set(item.id, optimistic);
      return next;
    });

    const previousTail = annotationTailsRef.current.get(sequenceKey) ?? Promise.resolve();
    const operation = previousTail.catch(() => undefined).then(async () => {
      if (!isActiveSessionOperation(generation, sessionId)) return;
      try {
        assertFeedbackAllowed(item, sessionId);
        const previousConfirmed = confirmedAnnotationsRef.current.get(item.id);
        if (!previousConfirmed || previousConfirmed.label >= 0) {
          throw new Error("The negative feedback label changed before confirmation.");
        }
        const response = await putTuningAnnotation(sessionId, {
          rowIndex: item.rowIndex,
          imageId: item.id,
          label: previousConfirmed.label,
          source: previousConfirmed.source || "failure-attribution",
          failedAttributeIds: normalizedAttributeIds,
          suggestedFailedAttributeId: previousConfirmed.suggestedFailedAttributeId,
          failureAttributionConfirmed: true,
        });
        if (!isActiveSessionOperation(generation, sessionId)) return;
        confirmedAnnotationsRef.current.set(item.id, response.annotation);
        advanceAnnotationRevision();
        if (annotationSequenceRef.current.get(sequenceKey) === sequence) {
          setAnnotations((current) => {
            const next = new Map(current);
            next.set(item.id, response.annotation);
            return next;
          });
          setError(null);
        }
      } catch (reason) {
        if (
          isActiveSessionOperation(generation, sessionId)
          && annotationSequenceRef.current.get(sequenceKey) === sequence
        ) {
          const confirmed = confirmedAnnotationsRef.current.get(item.id);
          setAnnotations((current) => {
            const next = new Map(current);
            if (confirmed) next.set(item.id, confirmed);
            else next.delete(item.id);
            return next;
          });
          setError(reason instanceof Error ? reason.message : String(reason));
        }
        throw reason;
      }
    });
    const settledTail = operation.then(() => undefined, () => undefined);
    annotationTailsRef.current.set(sequenceKey, settledTail);
    void settledTail.then(() => {
      if (annotationTailsRef.current.get(sequenceKey) === settledTail) {
        annotationTailsRef.current.delete(sequenceKey);
      }
    });
    return trackPendingAnnotation(operation, generation, sessionId);
  }, [
    assertFeedbackAllowed,
    advanceAnnotationRevision,
    isActiveSessionOperation,
    session,
    trackPendingAnnotation,
  ]);

  const clearAnnotations = useCallback(async () => {
    if (!session || annotations.size === 0) return;
    const generation = requestGeneration.current;
    const sessionId = session.id;
    if (!isActiveSessionOperation(generation, sessionId)) return;
    if (sessionMutationLockRef.current) {
      throw new Error("Another tuning operation is already in progress.");
    }
    const operationToken = Symbol("clear-annotations");
    sessionMutationLockRef.current = operationToken;
    busyOperationRef.current = operationToken;
    setBusy(true);
    setError(null);
    let snapshot: Map<string, TuningAnnotation> | null = null;
    try {
      await waitForPendingAnnotations(generation, sessionId);
      if (!isActiveSessionOperation(generation, sessionId)) return;
      snapshot = new Map(confirmedAnnotationsRef.current);
      for (const annotation of snapshot.values()) {
        assertFeedbackAllowed({ id: annotation.imageId, rowIndex: annotation.rowIndex }, sessionId, true);
      }
      setAnnotations(new Map());
      for (const annotation of snapshot.values()) {
        assertFeedbackAllowed({ id: annotation.imageId, rowIndex: annotation.rowIndex }, sessionId, true);
        await deleteTuningAnnotation(sessionId, annotation.rowIndex);
        if (!isActiveSessionOperation(generation, sessionId)) return;
      }
      confirmedAnnotationsRef.current = new Map();
      if ([...snapshot.values()].some((annotation) => annotation.label !== 0)) {
        advanceAnnotationRevision();
      }
    } catch (reason) {
      if (isActiveSessionOperation(generation, sessionId)) {
        if (snapshot) setAnnotations(snapshot);
        setError(reason instanceof Error ? reason.message : String(reason));
      }
      throw reason;
    } finally {
      if (sessionMutationLockRef.current === operationToken) {
        sessionMutationLockRef.current = null;
      }
      if (busyOperationRef.current === operationToken) {
        busyOperationRef.current = null;
        setBusy(false);
      }
    }
  }, [
    assertFeedbackAllowed,
    advanceAnnotationRevision,
    annotations.size,
    isActiveSessionOperation,
    session,
    waitForPendingAnnotations,
  ]);

  const runTuning = useCallback(async (mode: TuningLaunchMode) => {
    if (mode !== "weight_staged" && mode !== "weight_joint") {
      throw new Error("Use Update Probes separately, then choose a Weight Tune schedule.");
    }
    if (!session) throw new Error("The tuning session is not ready.");
    if (!feedbackContextRef.current.input.validationReady) {
      throw new Error("Wait for fixed Val membership to load before tuning.");
    }
    if (
      !enabled
      || session.taskId !== taskId
      || session.targetId !== targetId
      || session.baseMethod !== baseMethod
    ) {
      throw new Error("The selected tuning session is still preparing.");
    }
    if (baseMethod !== "Ours-Full") {
      throw new Error("Select Ours-Full before running a refinement method.");
    }
    if (targetId !== "joint") {
      throw new Error("Select Joint before running a refinement method.");
    }
    const generation = requestGeneration.current;
    const sessionId = session.id;
    if (!isActiveSessionOperation(generation, sessionId)) return;
    if (sessionMutationLockRef.current) {
      throw new Error("Another tuning operation is already in progress.");
    }
    if (probeUpdateInProgress || run?.status === "queued" || run?.status === "running") {
      throw new Error("Wait for the current training operation to finish.");
    }
    const pinnedProbeUpdateId = selectedProbeUpdateId;
    const operationToken = Symbol("create-tuning-run");
    sessionMutationLockRef.current = operationToken;
    busyOperationRef.current = operationToken;
    setBusy(true);
    setError(null);
    try {
      await waitForPendingAnnotations(generation, sessionId);
      if (!isActiveSessionOperation(generation, sessionId)) return;
      if (!feedbackContextRef.current.input.validationReady) {
        throw new Error("Fixed Val membership is unavailable; tuning is locked.");
      }
      const capabilities = await fetchRefinementCapabilities(taskId);
      if (!isActiveSessionOperation(generation, sessionId)) return;
      setRefinementCapabilities(capabilities);
      const capability = refinementCapability(capabilities, mode);
      if (!capability.available) throw new Error(capability.reason ?? "Tuning is unavailable.");
      if (!feedbackContextRef.current.input.validationReady) {
        throw new Error("Fixed Val membership is unavailable; tuning is locked.");
      }
      if (
        [...confirmedAnnotationsRef.current.values()].some((annotation) => (
          annotation.label < 0
          && annotation.supervision?.includedInTune !== false
          && !hasConfirmedNegativeFeedback(annotation, taskId, targetId)
        ))
      ) {
        throw new Error("Confirm failed attributes for every Joint negative before tuning.");
      }
      const submittedRevision = annotationRevisionRef.current;
      if (pinnedProbeUpdateId) {
        const { probeUpdate } = await fetchProbeUpdate(pinnedProbeUpdateId);
        if (!isActiveSessionOperation(generation, sessionId)) return;
        if (!resolveProbeSelection([probeUpdate], sessionId, taskId, pinnedProbeUpdateId)) {
          throw new Error("The selected Probe snapshot is unavailable or incompatible. Choose a source again.");
        }
        // New labels may train the weights while this exact older Probe snapshot
        // stays frozen; stale feedback is informational, never an implicit update.
        setProbeUpdates((jobs) => jobs.map((job) => job.id === probeUpdate.id ? probeUpdate : job));
      }
      rankRequestGenerationRef.current += 1;
      setAppliedRanks(null);
      setAppliedScores(null);
      setAppliedVisualization(null);
      setAppliedRunId(null);
      const runBaseMethod = tuningBaseMethod(mode, baseMethod);
      const response = await createTuningRun(sessionId, {
        mode,
        baseMethod: runBaseMethod,
        probeSource: pinnedProbeUpdateId ? "updated" : "original",
        ...(pinnedProbeUpdateId ? { probeUpdateId: pinnedProbeUpdateId } : {}),
      });
      if (!isActiveSessionOperation(generation, sessionId)) return;
      if (response.run.sessionId !== sessionId) {
        throw new Error("The tuning service returned a run for a different session.");
      }
      currentRunIdRef.current = response.run.id;
      submittedRevisionByRun.current.set(response.run.id, submittedRevision);
      setRun(response.run);
      setRunRevision(submittedRevision);
    } catch (reason) {
      if (isActiveSessionOperation(generation, sessionId)) {
        setError(reason instanceof Error ? reason.message : String(reason));
      }
      throw reason;
    } finally {
      if (sessionMutationLockRef.current === operationToken) {
        sessionMutationLockRef.current = null;
      }
      if (busyOperationRef.current === operationToken) {
        busyOperationRef.current = null;
        setBusy(false);
      }
    }
  }, [
    baseMethod,
    enabled,
    isActiveSessionOperation,
    session,
    selectedProbeUpdateId,
    probeUpdateInProgress,
    run?.status,
    targetId,
    taskId,
    waitForPendingAnnotations,
  ]);

  const updateProbes = useCallback(async () => {
    if (!session || !enabled || serviceState !== "ready" || session.taskId !== taskId
      || session.targetId !== targetId || session.baseMethod !== baseMethod) {
      throw new Error("The selected tuning session is still preparing.");
    }
    if (targetId !== "joint" || baseMethod !== "Ours-Full") throw new Error("Select Joint and Ours-Full before updating Probes.");
    if (!feedbackContextRef.current.input.validationReady) throw new Error("Wait for fixed Val membership before updating Probes.");
    if (sessionMutationLockRef.current || probeUpdateInProgress || run?.status === "queued" || run?.status === "running") {
      throw new Error("Wait for the current training operation to finish.");
    }
    const generation = requestGeneration.current;
    const sessionId = session.id;
    if (!isActiveSessionOperation(generation, sessionId)) return;
    const operationToken = Symbol("create-probe-update");
    probeListRevisionRef.current += 1;
    sessionMutationLockRef.current = operationToken;
    busyOperationRef.current = operationToken;
    setBusy(true);
    setError(null);
    try {
      await waitForPendingAnnotations(generation, sessionId);
      if (!isActiveSessionOperation(generation, sessionId)) return;
      const capabilities = await fetchRefinementCapabilities(taskId);
      if (!isActiveSessionOperation(generation, sessionId)) return;
      setRefinementCapabilities(capabilities);
      const capability = probeUpdateCapability(capabilities);
      if (!capability.available) throw new Error(capability.reason ?? "Probe update is unavailable.");
      if (!feedbackContextRef.current.input.validationReady) throw new Error("Fixed Val membership is unavailable.");
      const usable = [...confirmedAnnotationsRef.current.values()].filter((annotation) => annotation.label !== 0 && annotation.supervision?.includedInTune !== false);
      if (!usable.length) throw new Error("Add at least one positive or negative correction.");
      if (usable.some((annotation) => annotation.label < 0 && (!annotation.failureAttributionConfirmed || !annotation.failedAttributeIds?.length))) {
        throw new Error("Confirm failed attributes for every Joint negative before updating Probes.");
      }
      const { probeUpdate } = await createProbeUpdate(sessionId);
      if (!isActiveSessionOperation(generation, sessionId)) return;
      if (probeUpdate.sessionId !== sessionId || probeUpdate.taskId !== taskId) throw new Error("The Probe update belongs to a different session.");
      probeListRevisionRef.current += 1;
      setProbeUpdates((jobs) => [probeUpdate, ...jobs.filter((job) => job.id !== probeUpdate.id)]);
    } catch (reason) {
      if (isActiveSessionOperation(generation, sessionId)) setError(reason instanceof Error ? reason.message : String(reason));
      throw reason;
    } finally {
      if (sessionMutationLockRef.current === operationToken) sessionMutationLockRef.current = null;
      if (busyOperationRef.current === operationToken) {
        busyOperationRef.current = null;
        setBusy(false);
      }
    }
  }, [baseMethod, enabled, isActiveSessionOperation, probeUpdateInProgress, run?.status, serviceState, session, targetId, taskId, waitForPendingAnnotations]);

  const applyRun = useCallback(async (
    options: { refinementClusterScheme?: string } = {},
  ) => {
    if (!session || !run || run.status !== "succeeded") return;
    const generation = requestGeneration.current;
    const sessionId = session.id;
    if (
      !isActiveSessionOperation(generation, sessionId)
      || run.sessionId !== sessionId
      || currentRunIdRef.current !== run.id
    ) return;
    const rankGeneration = ++rankRequestGenerationRef.current;
    const operationToken = Symbol("fetch-tuned-output");
    busyOperationRef.current = operationToken;
    rankBusyTokenRef.current = operationToken;
    setBusy(true);
    setError(null);
    try {
      const visualizationPromise = isWeightRefinementRun(run)
        ? requestRefinementVisualization(run, rowCount)
        : Promise.resolve(null);
      const [ranks, scores, visualization] = await Promise.all([
        fetchTunedRanks(run, rowCount),
        fetchTunedScores(run, rowCount),
        visualizationPromise,
      ]);
      if (visualization && options.refinementClusterScheme) {
        // Populate the run+fingerprint cluster cache before committing any
        // Top/PCP state. A failed cluster fit therefore leaves the old model
        // visible everywhere instead of exposing a half-applied result.
        await requestRefinementClusters({
          run,
          snapshot: visualization,
          scheme: options.refinementClusterScheme,
        });
      }
      if (
        rankGeneration !== rankRequestGenerationRef.current
        || !isActiveSessionOperation(generation, sessionId)
        || currentRunIdRef.current !== run.id
      ) return;
      setAppliedRanks(ranks);
      setAppliedScores(scores);
      setAppliedVisualization(visualization);
      setAppliedRunId(run.id);
    } catch (reason) {
      if (
        rankGeneration !== rankRequestGenerationRef.current
        || !isActiveSessionOperation(generation, sessionId)
        || currentRunIdRef.current !== run.id
      ) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      throw reason;
    } finally {
      if (rankBusyTokenRef.current === operationToken) {
        rankBusyTokenRef.current = null;
      }
      if (busyOperationRef.current === operationToken) {
        busyOperationRef.current = null;
        setBusy(false);
      }
    }
  }, [isActiveSessionOperation, rowCount, run, session]);

  const revertRun = useCallback(() => {
    rankRequestGenerationRef.current += 1;
    const rankBusyToken = rankBusyTokenRef.current;
    rankBusyTokenRef.current = null;
    if (rankBusyToken && busyOperationRef.current === rankBusyToken) {
      busyOperationRef.current = null;
      setBusy(false);
    }
    setAppliedRanks(null);
    setAppliedScores(null);
    setAppliedVisualization(null);
    setAppliedRunId(null);
  }, []);

  return {
    serviceState,
    user,
    session,
    refinementCapabilities,
    probeUpdates,
    probeUpdateInProgress,
    probeSource,
    selectedProbeUpdate,
    selectProbeSource,
    updateProbes,
    annotations,
    preferences,
    counts,
    feedbackBreakdown,
    run,
    appliedRanks,
    appliedScores,
    appliedVisualization,
    appliedRunId,
    resultStale: Boolean(run && run.status === "succeeded" && runRevision !== annotationRevision),
    pendingAnnotationCount,
    busy: busy || pendingAnnotationCount > 0,
    error,
    saveDisplayName,
    startNewSession,
    updateAnnotation,
    bulkUpdateAnnotations,
    updateFailureAttributes,
    clearAnnotations,
    runTuning,
    applyRun,
    revertRun,
  };
}
