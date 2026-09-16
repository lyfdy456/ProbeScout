"use client";

import { HIERARCHICAL_PCP_METHODS } from "../lib/hierarchicalPcp";
import { methodDisplayLabel } from "../lib/methodDisplay.js";
import type { TuningSessionState } from "../lib/useTuningSession";
import { isCurrentTuningMode, tuningModeLabel, type FixedVqaValidationResponse } from "../lib/tuningApi";
import { apDeltaTone, formatAp, formatApDelta, isValidationEvaluation, tuningApRows, tuningEvaluationLabel } from "../lib/tuningEvaluation";

interface TuningPanelProps {
  state: TuningSessionState;
  validation?: FixedVqaValidationResponse | null;
  /** Optional Dashboard transaction used by per-attribute Fusion Weight v2-v4. */
  isApplied?: boolean;
  applyBusy?: boolean;
  applyError?: string | null;
  onApplyRun?: () => Promise<void>;
  onRevertRun?: () => void;
}

function percent(value: number | undefined) {
  return value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

function shortId(value: string | undefined) {
  if (!value) return "—";
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-5)}` : value;
}

function feedbackRelationSummary(
  newCount = 0,
  overrideCount = 0,
  reinforceCount = 0,
  reviewOnlyCount = 0,
) {
  return [
    newCount > 0 ? `${newCount} New` : null,
    overrideCount > 0 ? `${overrideCount} Override` : null,
    reinforceCount > 0 ? `${reinforceCount} Reinforce` : null,
    reviewOnlyCount > 0 ? `${reviewOnlyCount} review-only (?)` : null,
  ].filter((value): value is string => Boolean(value)).join(" · ");
}

export function TuningPanel({
  state,
  validation,
  isApplied: controlledIsApplied,
  applyBusy = false,
  applyError = null,
  onApplyRun,
  onRevertRun,
}: TuningPanelProps) {
  const {
    serviceState,
    counts,
    feedbackBreakdown,
    run,
    appliedRunId,
    resultStale,
    pendingAnnotationCount,
    busy,
    error,
    applyRun,
    revertRun,
  } = state;
  const runInProgress = run?.status === "queued" || run?.status === "running";
  const annotationsPending = pendingAnnotationCount > 0;
  const supervisionResolved = feedbackBreakdown.unresolvedCount === 0;
  const isApplied = controlledIsApplied ?? Boolean(run && appliedRunId === run.id);
  const applying = busy || applyBusy;
  const applyCurrentRun = onApplyRun ?? applyRun;
  const revertCurrentRun = onRevertRun ?? revertRun;
  const methodWeightsByAttribute = run?.after?.modelSummary?.methodWeightsByAttribute;
  const learnerWeightsByAttribute = run?.after?.modelSummary?.learnerWeightsByAttribute;
  const beta = run?.after?.modelSummary?.beta;
  const displayedLearnerWeights = learnerWeightsByAttribute ?? beta;
  const activeAttributeIds = new Set(
    run?.after?.modelSummary?.activeAttributeIds
      ?? Object.keys(methodWeightsByAttribute ?? displayedLearnerWeights ?? {}),
  );
  const attributeWeights = run?.after?.modelSummary?.attributeWeights;
  const activeJointAttributeIds = new Set(
    run?.after?.modelSummary?.activeJointAttributeIds ?? [],
  );
  const currentFeedbackRelations = feedbackRelationSummary(
    feedbackBreakdown.trainableNewCount,
    feedbackBreakdown.overrideCount,
    feedbackBreakdown.reinforceCount,
    feedbackBreakdown.reviewOnlyCount,
  );
  const runSupervision = run?.after?.supervisionSummary;
  const splitAudit = run?.splitAudit ?? run?.after?.splitAudit;
  const probeState = run?.after?.modelSummary?.probeState;
  const cleanValidation = run?.evaluationScope === "clean-validation";
  const vqaValidation = run?.evaluationScope === "vqa-validation";
  const validationMetricsAvailable = isValidationEvaluation(run?.evaluationScope);
  const apRows = run ? tuningApRows(run, "validation") : [];
  const evaluationLabel = tuningEvaluationLabel(run?.evaluationScope);
  const runValidationVersion = run?.vqaValidationVersion ?? splitAudit?.vqaValidationVersion ?? splitAudit?.frozenVersion;
  const runValidationManifestSha256 = run?.vqaValidationManifestSha256 ?? splitAudit?.vqaValidationManifestSha256 ?? splitAudit?.frozenManifestSha256;
  const evaluationCount = run?.after?.evaluationCount
    ?? run?.before?.evaluationCount
    ?? (cleanValidation || vqaValidation ? splitAudit?.validationCount : undefined);
  const evaluationPositiveCount = run?.after?.positiveCount
    ?? run?.before?.positiveCount
    ?? (cleanValidation || vqaValidation ? splitAudit?.validationPositiveCount : undefined);
  const runFeedbackRelations = runSupervision
    ? feedbackRelationSummary(
        runSupervision.newFeedbackCount,
        runSupervision.feedbackOverrideCount,
        runSupervision.feedbackReinforceCount,
        runSupervision.feedbackReviewOnlyCount,
      )
    : "";

  return (
    <div className="tuning-panel">
      {serviceState === "loading" && (
        <div className="tuning-status" aria-live="polite">Preparing tuning session…</div>
      )}
      {!supervisionResolved && (
        <div className="tuning-status" aria-live="polite">
          Checking {feedbackBreakdown.unresolvedCount} saved label
          {feedbackBreakdown.unresolvedCount === 1 ? "" : "s"} against original DG supervision…
        </div>
      )}
      {serviceState === "ready" && annotationsPending && (
        <div className="tuning-status" aria-live="polite">
          Saving {pendingAnnotationCount} {pendingAnnotationCount === 1 ? "label" : "labels"}…
        </div>
      )}
      {serviceState === "offline" && (
        <div className="tuning-status tuning-status-error" role="alert">
          Tuning service is offline.
        </div>
      )}
      {error && serviceState !== "offline" && (
        <div className="tuning-status tuning-status-error" role="alert">{error}</div>
      )}
      {applyError && applyError !== error && (
        <div className="tuning-status tuning-status-error" role="alert">{applyError}</div>
      )}
      {runInProgress && (
        <div className="tuning-status" aria-live="polite">
          {tuningModeLabel(run.mode, run.algorithmVersion)} · {run.status === "queued"
            ? "Run queued…"
            : vqaValidation
              ? "Computing Val metrics…"
              : run.evaluationScope === "clean-validation"
              ? "Computing Clean Validation metrics…"
              : run.evaluationScope === "probe-validation"
                ? "Computing Probe Val metrics…"
                : "Computing Validation metrics…"}
        </div>
      )}
      {resultStale && run?.status === "succeeded" && (
        <div className="tuning-status tuning-result-state" role="status">Labels changed · rerun</div>
      )}

      {run?.status === "succeeded" && run.before && run.after && (
        <div className="tuning-result-overview" key={run.id}>
          {vqaValidation && (
            <div className="tuning-val-summary" aria-label="Val AP summary" aria-live="polite">
              <strong className="tuning-val-summary-label">Val AP:</strong>
              <span className="tuning-val-summary-values">
                <strong aria-label={`Before AP ${formatAp(run.before.ap)}`}>{formatAp(run.before.ap)}</strong>
                <span aria-hidden="true"> → </span>
                <strong aria-label={`After AP ${formatAp(run.after.ap)}`}>{formatAp(run.after.ap)}</strong>
              </span>
              <span aria-hidden="true">·</span>
              <strong
                className={`ap-delta-${apDeltaTone(apRows[0]?.delta)}`}
                aria-label={`AP change ${formatApDelta(apRows[0]?.delta)} percentage points`}
              >{formatApDelta(apRows[0]?.delta)} pp</strong>
            </div>
          )}
        <details className="tuning-evaluation-result">
          <summary>
            {vqaValidation ? "Evaluation details" : "Historical evaluation · original scope"}
            <span>{isApplied ? " · Ranking applied" : " · Apply ranking"}</span>
          </summary>
        <div className={`tuning-result ${resultStale ? "stale" : ""}`.trim()} aria-live="polite">
          <div className="tuning-result-heading">
            <strong>
              {tuningModeLabel(run.mode, run.algorithmVersion)} · {
                vqaValidation
                  ? "Val"
                  : run.evaluationScope === "clean-validation"
                  ? "Clean Validation"
                  : run.evaluationScope === "probe-validation"
                    ? "Probe Val"
                    : run.evaluationScope === "validation"
                      ? "legacy Web Validation"
                      : "Saved run"
              }
            </strong>
            <em>{resultStale ? "Labels changed · rerun" : "Complete"}</em>
          </div>
          {run.probeSource && <div className="tuning-session-meta">
            <span>Probes · {run.probeSource === "updated" ? "Updated (frozen)" : "Original (frozen)"}</span>
            {run.probeUpdateId && <code title={run.probeSnapshotId ?? run.probeUpdateId}>{shortId(run.probeUpdateId)}</code>}
          </div>}
          {validationMetricsAvailable ? <>
          <div className="tuning-ap-comparison">
            <table aria-label="Validation AP comparison">
              <thead><tr>
                <th scope="col">AP</th>
                <th scope="col" aria-label="Before AP">Before</th>
                <th scope="col" aria-label="After AP">After</th>
                <th scope="col" title="Change in percentage points">ΔAP (pp)</th>
              </tr></thead>
              <tbody>{apRows.map((row) => (
                <tr key={row.key}>
                  <th scope="row">
                    {row.label}
                    {row.note && <small>{row.note}</small>}
                  </th>
                  <td title={`Baseline: ${methodDisplayLabel(run.beforeMethod ?? run.baseMethod)}`}>
                    {formatAp(row.before)}
                  </td>
                  <td>{formatAp(row.after)}</td>
                  <td className={`ap-delta-${apDeltaTone(row.delta)}`}>{formatApDelta(row.delta)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          {run.before.bestF1 !== undefined && run.after.bestF1 !== undefined && (
            <div className="tuning-tp-row tuning-f1-row">
              <span>{evaluationLabel} · Best F1 {percent(run.before.bestF1)} → {percent(run.after.bestF1)}</span>
              {(cleanValidation || vqaValidation) && evaluationCount !== undefined && (
                <span>
                  {evaluationCount} {vqaValidation ? "Val" : "clean val"}
                  {evaluationPositiveCount !== undefined ? ` · ${evaluationPositiveCount} positive` : ""}
                </span>
              )}
              {!cleanValidation && !vqaValidation && splitAudit && (
                <span>
                  {splitAudit.validationCount} val · {splitAudit.validationPositiveCount} positive
                </span>
              )}
            </div>
          )}
          <span className="tuning-metric-caption">{evaluationLabel} · TP@K</span>
          <div className="tuning-tp-row">
            <span>TP@30 {run.before.tpAt30} → {run.after.tpAt30}</span>
            <span>TP@50 {run.before.tpAt50} → {run.after.tpAt50}</span>
            <span>TP@100 {run.before.tpAt100} → {run.after.tpAt100}</span>
            <span>TP@200 {run.before.tpAt200} → {run.after.tpAt200}</span>
          </div>
          </> : (
            <p className="tuning-integrity-note">No Validation metrics recorded for this run.</p>
          )}
          <details className="tuning-run-details">
            <summary>
              {(cleanValidation || vqaValidation) && runSupervision
                ? `${runSupervision.originalProbeFitCount ?? splitAudit?.fitCount ?? runSupervision.originalDevelopmentCount} Probe train + ${runSupervision.feedbackCount} feedback`
                : runSupervision && splitAudit
                  ? `${runSupervision.originalProbeFitCount ?? splitAudit.fitCount ?? runSupervision.originalDevelopmentCount} Probe train + ${runSupervision.feedbackCount} feedback · ${splitAudit.validationCount} val`
                : runSupervision
                  ? `${runSupervision.originalDevelopmentCount} DG + ${runSupervision.feedbackCount} feedback · ${runSupervision.combinedCount} rows`
                  : `${run.annotationCount ?? counts.usablePositive + counts.usableNegative} feedback labels`}
            </summary>
            <div className="tuning-run-details-body">
              {probeState?.source === "native-updated-frozen-probe-snapshot" && (
                <p className="tuning-integrity-note" title={probeState.snapshotFingerprint}>
                  Native Probe update → Frozen probes → Fusion
                  {probeState.snapshotId ? ` · snapshot ${shortId(probeState.snapshotId)}` : ""}
                  {probeState.sharedAcrossFusionSchedules ? " · shared by Staged / Joint" : ""}
                  {probeState.frozenDuringFusion ? " · no Probe updates during fusion" : ""}
                </p>
              )}
              {vqaValidation && (
                <p className="tuning-integrity-note" title={runValidationManifestSha256}>
                  Val · original VQA labels · fixed 20%
                  {runValidationVersion ? ` · ${runValidationVersion}` : ""}
                  {runValidationManifestSha256 ? ` · ${runValidationManifestSha256.slice(0, 12)}` : ""}
                  {splitAudit ? ` · members ${splitAudit.validationFingerprint.slice(0, 8)}` : ""}
                  {" · not independent of original-model training"}
                </p>
              )}
              {cleanValidation && splitAudit && (
                <p className="tuning-integrity-note">
                  Clean Validation · {splitAudit.validationCount} val · {splitAudit.validationPositiveCount} positive
                  {` · label-isolated · ${splitAudit.validationFingerprint.slice(0, 8)}`}
                </p>
              )}
              {!cleanValidation && !vqaValidation && splitAudit && (
                <p className="tuning-integrity-note">
                  {splitAudit.seed !== undefined ? `Seed ${splitAudit.seed}` : "Fixed split"}
                  {splitAudit.validationFraction !== undefined
                    ? ` · stratified ${(splitAudit.validationFraction * 100).toFixed(0)}% holdout`
                    : ""}
                  {splitAudit.replayKind === "derived-joint-probe-style" ? " · Joint-derived" : " · original attribute split"}
                  {` · ${splitAudit.validationFingerprint.slice(0, 8)}`}
                </p>
              )}
              {(runSupervision?.feedbackHoldoutExcludedCount ?? 0) > 0 && (
                <p className="tuning-integrity-note">
                  {runSupervision?.feedbackHoldoutExcludedCount} feedback held out from tuning.
                </p>
              )}
              {runSupervision && (
                <p className="tuning-integrity-note">
                  Feedback weight {runSupervision.feedbackWeight}× · strong {runSupervision.feedbackWeight * 2}×
                  {runFeedbackRelations ? ` · ${runFeedbackRelations}` : ""}
                </p>
              )}
              {methodWeightsByAttribute && (
                <div className="tuning-attribute-weights">
                  <span>Learner weights by attribute</span>
                  {Object.entries(methodWeightsByAttribute).map(([attribute, weights]) => (
                    <p className="tuning-integrity-note" key={attribute}>
                      <strong>{attribute}:</strong>
                      {!activeAttributeIds.has(attribute) && (
                        <em className="tuning-kept-equal">kept equal</em>
                      )}{" "}
                      {HIERARCHICAL_PCP_METHODS
                        .map((name) => `${methodDisplayLabel(name)} ${(weights[name] * 100).toFixed(1)}%`)
                        .join(" · ")}
                    </p>
                  ))}
                </div>
              )}
              {displayedLearnerWeights && (
                <div className="tuning-attribute-weights">
                  <span>Learner weights by attribute</span>
                  {Object.entries(displayedLearnerWeights).map(([attribute, weights]) => (
                    <p className="tuning-integrity-note" key={attribute}>
                      <strong>{attribute}:</strong>
                      {!activeAttributeIds.has(attribute) && (
                        <em className="tuning-kept-equal">kept equal</em>
                      )}{" "}
                      {Object.entries(weights)
                        .map(([name, value]) => `${methodDisplayLabel(name)} ${(value * 100).toFixed(1)}%`)
                        .join(" · ")}
                    </p>
                  ))}
                </div>
              )}
              {attributeWeights && (
                <div className="tuning-attribute-weights">
                  <span>Attribute weights</span>
                  <p className="tuning-integrity-note">
                    {Object.entries(attributeWeights)
                      .map(([attribute, value]) => (
                        methodWeightsByAttribute
                          ? `${attribute} ${(value * 100).toFixed(1)}%`
                          : `${attribute} ${value.toFixed(3)}${
                            activeJointAttributeIds.has(attribute) ? "" : " (fixed)"
                          }`
                      ))
                      .join(" · ")}
                  </p>
                </div>
              )}
              {run.after.modelSummary?.gamma && (
                <div className="tuning-attribute-weights">
                  <span>
                    Attribute exponents γ
                    {cleanValidation || vqaValidation ? " · independent [0.05, 3.00]" : ""}
                  </span>
                  <p className="tuning-integrity-note">
                    {Object.entries(run.after.modelSummary.gamma)
                      .map(([attribute, value]) => `${attribute} ${value.toFixed(3)}`)
                      .join(" · ")}
                  </p>
                </div>
              )}
              {run.after.modelSummary?.embeddingWeights && (
                <div className="tuning-attribute-weights">
                  <span>Embedding weights</span>
                  <p className="tuning-integrity-note">
                    {Object.entries(run.after.modelSummary.embeddingWeights)
                      .map(([name, value]) => `${methodDisplayLabel(name)} ${(value * 100).toFixed(1)}%`)
                      .join(" · ")}
                  </p>
                </div>
              )}
              {run.after.modelSummary?.embeddingFusionStrength !== undefined && (
                <p className="tuning-integrity-note">
                  Embedding fusion λ {run.after.modelSummary.embeddingFusionStrength.toFixed(3)}
                  {probeState?.frozenDuringFusion
                    ? " · updated probes frozen"
                    : run.after.modelSummary.probeUpdated ? " · probes updated" : " · probes fixed"}
                </p>
              )}
              {run.after.modelSummary?.theta && (
                <p className="tuning-integrity-note">
                  SoftGate θ {Object.entries(run.after.modelSummary.theta)
                    .map(([attribute, value]) => `${attribute} ${value.toFixed(3)}`)
                    .join(" · ")}
                </p>
              )}
              {!methodWeightsByAttribute
                && !learnerWeightsByAttribute
                && run.after.modelSummary?.learnerWeights && (
                <p className="tuning-integrity-note">
                  Legacy shared weights: {Object.entries(run.after.modelSummary.learnerWeights)
                    .map(([name, value]) => `${methodDisplayLabel(name)} ${(value * 100).toFixed(1)}%`)
                    .join(" · ")}
                </p>
              )}
              {run.after.modelSummary?.residualCoefficients && (
                <p className="tuning-integrity-note">
                  Residual coefficients: {Object.entries(run.after.modelSummary.residualCoefficients)
                    .map(([name, value]) => `${methodDisplayLabel(name)} ${value >= 0 ? "+" : ""}${value.toFixed(3)}`)
                    .join(" · ")}
                </p>
              )}
              {run.after.modelSummary?.bias !== undefined && (
                <p className="tuning-integrity-note">
                  Bias {run.after.modelSummary.bias >= 0 ? "+" : ""}{run.after.modelSummary.bias.toFixed(3)}
                  {run.after.modelSummary.biasAffectsRanking === false ? " · ranking unchanged" : ""}
                </p>
              )}
            </div>
          </details>
          <div className="tuning-result-actions">
            <button
              type="button"
              className="button-primary"
              disabled={applying || resultStale || isApplied}
              onClick={() => void applyCurrentRun().catch(() => undefined)}
            >
              {isApplied
                ? "Applied ranking active"
                : applyBusy
                  ? "Applying weights…"
                  : isCurrentTuningMode(run.mode)
                    ? "Use in Top ranking"
                    : methodWeightsByAttribute || displayedLearnerWeights
                    ? "Apply weights & ranking"
                    : "Use in Top ranking"}
            </button>
            {isApplied && (
              <button type="button" className="button-secondary" onClick={revertCurrentRun}>
                Return to baseline
              </button>
            )}
          </div>
        </div>
        </details>
        </div>
      )}
      <details className="tuning-training-details">
        <summary>Training details</summary>
        <div className="tuning-training-details-body">
          <div className="tuning-integrity-note">
            Val · {validation ? `${validation.count} images` : "loading membership…"}
            <details>
              <summary>Val protocol</summary>
              Original VQA 20% · fixed per task · excluded from training, exploration, and feedback.
              This is a tuning holdout, not an independent holdout for the original model.
              {validation && <p title={validation.manifestSha256}>{validation.version} · {validation.manifestSha256.slice(0, 12)}</p>}
            </details>
          </div>
          <div className="tuning-feedback-breakdown" aria-live="polite" aria-atomic="true">
            <div className="tuning-feedback-primary">
              <strong>{feedbackBreakdown.totalCount} feedback</strong>
              {supervisionResolved && <span>
                {feedbackBreakdown.newCount} new · {feedbackBreakdown.existingCount} existing
              </span>}
            </div>
            {supervisionResolved && currentFeedbackRelations && <span>{currentFeedbackRelations}</span>}
            {feedbackBreakdown.holdoutExcludedCount > 0 && (
              <span>{feedbackBreakdown.holdoutExcludedCount} historical labels on Val · excluded from tuning</span>
            )}
          </div>
        </div>
      </details>
    </div>
  );
}

export default TuningPanel;
