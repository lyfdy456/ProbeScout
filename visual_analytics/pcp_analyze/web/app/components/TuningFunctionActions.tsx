"use client";

import type { TuningSessionState } from "../lib/useTuningSession";
import { hasConfirmedNegativeFeedback, probeUpdateCapability, refinementCapability, type WeightTuningMode } from "../lib/tuningApi";

const TUNING_ACTIONS: ReadonlyArray<{
  mode: WeightTuningMode;
  label: string;
  description: string;
}> = [
  {
    mode: "weight_staged",
    label: "Staged Weight Refinement",
    description: "冻结所选 Probe 快照；先训练 β，再训练 γ、η、λ。θ/T 全程固定。",
  },
  {
    mode: "weight_joint",
    label: "Joint Weight Tune",
    description: "冻结所选 Probe 快照；同时训练 β、γ、η、λ。θ/T 全程固定。",
  },
];

interface TuningFunctionActionsProps {
  state: TuningSessionState;
  taskId: string;
  targetId: string;
  baseMethod: string;
  validationReady?: boolean;
}

function unavailableReason(input: {
  mode: WeightTuningMode | "update_probes";
  state: TuningSessionState;
  runInProgress: boolean;
  hasUsableFeedback: boolean;
  hasSignedFeedback: boolean;
  supervisionResolved: boolean;
  sessionMatches: boolean;
  baseMethod: string;
  targetId: string;
  unconfirmedJointNegativeCount: number;
}) {
  const {
    mode,
    state,
    runInProgress,
    hasUsableFeedback,
    hasSignedFeedback,
    supervisionResolved,
    sessionMatches,
    baseMethod,
    targetId,
    unconfirmedJointNegativeCount,
  } = input;
  if (state.serviceState === "offline") return "Tuning service is offline.";
  if (!sessionMatches) return "Preparing the selected tuning session.";
  if (state.serviceState !== "ready") return "Preparing the tuning session.";
  if (state.pendingAnnotationCount > 0) return "Saving feedback labels before tuning.";
  if (runInProgress) return "A tuning run is already in progress.";
  if (state.probeUpdateInProgress) return "A Probe update is in progress.";
  if (state.busy) return "The tuning session is busy.";
  if (targetId !== "joint") return "Select Joint as the retrieval target first.";
  if (baseMethod !== "Ours-Full") {
    return "Select Ours-Full in Top Results first.";
  }
  const capability = mode === "update_probes" ? probeUpdateCapability(state.refinementCapabilities)
    : refinementCapability(state.refinementCapabilities, mode);
  if (!capability.available) return capability.reason ?? "Tuning is unavailable.";
  if (unconfirmedJointNegativeCount > 0) {
    return `Confirm failed attributes for ${unconfirmedJointNegativeCount} Joint negative${
      unconfirmedJointNegativeCount === 1 ? "" : "s"
    } before tuning.`;
  }
  if (!supervisionResolved) return "Waiting for original-supervision checks.";
  if (!hasUsableFeedback && mode === "update_probes") return "Add at least one positive or negative correction.";
  if (!hasUsableFeedback && hasSignedFeedback) return "All corrections are held out; remove them or add Development feedback.";
  if (mode !== "update_probes" && state.probeSource === "updated" && !state.selectedProbeUpdate) return "Choose a compatible updated Probe snapshot.";
  if (mode === "update_probes") return "用原监督和人工反馈更新 Probes，保存并冻结快照；不自动训练权重。";
  if (!hasUsableFeedback) return "无人工反馈：仅使用原 VQA 训练监督。";
  return TUNING_ACTIONS.find((action) => action.mode === mode)?.description
    ?? "Run tuning.";
}

export function TuningFunctionActions({
  state,
  taskId,
  targetId,
  baseMethod,
  validationReady = false,
}: TuningFunctionActionsProps) {
  const runInProgress = state.run?.status === "queued" || state.run?.status === "running";
  const annotationsPending = state.pendingAnnotationCount > 0;
  const supervisionResolved = state.feedbackBreakdown.unresolvedCount === 0;
  const fusionBaseReady = baseMethod === "Ours-Full";
  const jointTargetReady = targetId === "joint";
  const hasUsableFeedback = state.counts.usablePositive + state.counts.usableNegative >= 1;
  const hasSignedFeedback = [...state.annotations.values()].some((annotation) => annotation.label !== 0);
  const unconfirmedJointNegativeCount = [...state.annotations.values()].filter((annotation) => (
    annotation.label < 0
    && annotation.supervision?.includedInTune !== false
    && (
      !annotation.failureAttributionConfirmed
      || !annotation.failedAttributeIds?.length
    )
  )).length;
  const unconfirmedWeightNegativeCount = [...state.annotations.values()].filter((annotation) => (
    annotation.label < 0
    && annotation.supervision?.includedInTune !== false
    && !hasConfirmedNegativeFeedback(annotation, taskId, targetId)
  )).length;
  const sessionMatches = Boolean(
    state.session
    && state.session.taskId === taskId
    && state.session.targetId === targetId
    && state.session.baseMethod === baseMethod,
  );
  const tuningReady = jointTargetReady
    && fusionBaseReady
    && (hasUsableFeedback || !hasSignedFeedback)
    && supervisionResolved
    && unconfirmedWeightNegativeCount === 0;
  const commonDisabled = (
    state.busy
    || annotationsPending
    || runInProgress
    || state.probeUpdateInProgress
    || state.serviceState !== "ready"
    || !validationReady
    || !sessionMatches
  );

  const launch = (mode: WeightTuningMode) => {
    void state.runTuning(mode).catch(() => undefined);
  };

  const reasonFor = (mode: WeightTuningMode | "update_probes") => !validationReady
    ? "Loading fixed Val membership before tuning."
    : unavailableReason({
    mode,
    state,
    runInProgress,
    hasUsableFeedback,
    hasSignedFeedback,
    supervisionResolved,
    sessionMatches,
    baseMethod,
    targetId,
    unconfirmedJointNegativeCount: mode === "update_probes"
      ? unconfirmedJointNegativeCount : unconfirmedWeightNegativeCount,
  });
  const titleFor = (mode: WeightTuningMode) => {
    const description = TUNING_ACTIONS.find((action) => action.mode === mode)?.description ?? "";
    const reason = reasonFor(mode);
    return reason === description ? description : `${description} ${reason}`.trim();
  };
  const chosenUpdate = state.selectedProbeUpdate;
  const updateDisabled = commonDisabled || !tuningReady || unconfirmedJointNegativeCount > 0
    || !hasUsableFeedback || !probeUpdateCapability(state.refinementCapabilities).available;
  const missingSnapshot = state.probeSource === "updated" && !chosenUpdate;

  return (
    <>
      <button
        type="button"
        className="button-secondary function-probe-update"
        hidden
        title={reasonFor("update_probes")}
        disabled={updateDisabled}
        onClick={() => void state.updateProbes().catch(() => undefined)}
      >
        Update Probes
      </button>
      {TUNING_ACTIONS.map((action) => (
        <button
          key={action.mode}
          type="button"
          className="button-secondary function-tune-action"
          hidden={action.mode !== "weight_staged"}
          aria-label={action.label}
          title={titleFor(action.mode)}
          disabled={!refinementCapability(state.refinementCapabilities, action.mode).available || !tuningReady || commonDisabled || missingSnapshot}
          onClick={() => launch(action.mode)}
        >
          {action.label}
        </button>
      ))}
    </>
  );
}

export default TuningFunctionActions;
