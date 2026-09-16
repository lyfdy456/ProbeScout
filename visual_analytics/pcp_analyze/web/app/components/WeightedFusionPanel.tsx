"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { methodDisplayLabel } from "../lib/methodDisplay.js";
import {
  DEFAULT_WEIGHTED_FUSION_WEIGHTS,
  WEIGHTED_FUSION_LEARNERS,
  areWeightedFusionWeightsEquivalent,
  cloneWeightedFusionWeights,
  createEqualWeightedFusionWeights,
  normalizeWeightedFusionWeights,
  useWeightedFusion,
  type WeightedFusionResult,
  type WeightedFusionWeights,
} from "../lib/weightedFusion";

export interface WeightedFusionPanelProps {
  taskId: string;
  rowCount: number;
  targetCount: number;
  appliedResult?: WeightedFusionResult | null;
  testAudit?: WeightedFusionTestAudit | null;
  onApplied: (result: WeightedFusionResult) => void;
  onRevert: () => void;
  disabled?: boolean;
  disabledReason?: string;
}

export interface WeightedFusionTestAudit {
  targetLabel: string;
  positiveCount: number;
  evaluatedCount: number;
  changes: readonly {
    k: number;
    before: number;
    after: number;
  }[];
}

interface EditorState {
  scopeKey: string;
  draftWeights: WeightedFusionWeights;
  appliedWeights: WeightedFusionWeights | null;
}

const WEIGHT_MIN = 0;
const WEIGHT_MAX = 10;
const WEIGHT_STEP = 0.05;

function editorScopeKey(taskId: string, rowCount: number, targetCount: number) {
  return `${taskId}\u0000${rowCount}\u0000${targetCount}`;
}

function createEditorState(
  scopeKey: string,
  appliedResult: WeightedFusionResult | null = null,
): EditorState {
  return {
    scopeKey,
    draftWeights: cloneWeightedFusionWeights(
      appliedResult?.weights ?? DEFAULT_WEIGHTED_FUSION_WEIGHTS,
    ),
    appliedWeights: appliedResult
      ? cloneWeightedFusionWeights(appliedResult.weights)
      : null,
  };
}

function inputWeight(value: string) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.min(WEIGHT_MAX, Math.max(WEIGHT_MIN, parsed));
}

function displayedWeight(value: number) {
  return Number(value.toFixed(6));
}

export function WeightedFusionPanel({
  taskId,
  rowCount,
  targetCount,
  appliedResult = null,
  testAudit = null,
  onApplied,
  onRevert,
  disabled = false,
  disabledReason,
}: WeightedFusionPanelProps) {
  const scopeKey = editorScopeKey(taskId, rowCount, targetCount);
  const [editor, setEditor] = useState<EditorState>(() => (
    createEditorState(scopeKey, appliedResult)
  ));
  const currentEditor = editor.scopeKey === scopeKey
    ? editor
    : createEditorState(scopeKey, appliedResult);
  const { draftWeights, appliedWeights } = currentEditor;
  const fusion = useWeightedFusion({ taskId, rowCount, targetCount, disabled });
  const deliveredResultRef = useRef<WeightedFusionResult | null>(null);

  const normalizedDraft = useMemo(() => {
    try {
      return normalizeWeightedFusionWeights(draftWeights);
    } catch {
      return null;
    }
  }, [draftWeights]);
  const unchanged = Boolean(
    appliedWeights
    && areWeightedFusionWeightsEquivalent(draftWeights, appliedWeights),
  );
  const busy = fusion.status === "loading";
  const awaitingAppliedResult = Boolean(
    fusion.result
    && (!appliedWeights
      || !areWeightedFusionWeightsEquivalent(fusion.result.weights, appliedWeights)),
  );

  useEffect(() => {
    const result = fusion.result;
    if (!result || deliveredResultRef.current === result) return;
    deliveredResultRef.current = result;
    setEditor((current) => {
      const scoped = current.scopeKey === scopeKey ? current : createEditorState(scopeKey);
      return {
        ...scoped,
        appliedWeights: cloneWeightedFusionWeights(result.weights),
      };
    });
    onApplied(result);
  }, [fusion.result, onApplied, scopeKey]);

  const updateDraftWeight = (learner: keyof WeightedFusionWeights, value: number) => {
    setEditor((current) => {
      const scoped = current.scopeKey === scopeKey ? current : createEditorState(scopeKey);
      return {
        ...scoped,
        draftWeights: { ...scoped.draftWeights, [learner]: value },
      };
    });
  };

  const normalizeDraft = () => {
    if (!normalizedDraft) return;
    setEditor((current) => {
      const scoped = current.scopeKey === scopeKey ? current : createEditorState(scopeKey);
      return {
        ...scoped,
        draftWeights: Object.fromEntries(
          WEIGHTED_FUSION_LEARNERS.map((learner) => [
            learner,
            displayedWeight(normalizedDraft[learner]),
          ]),
        ) as WeightedFusionWeights,
      };
    });
  };

  const resetEqual = () => {
    setEditor((current) => {
      const scoped = current.scopeKey === scopeKey ? current : createEditorState(scopeKey);
      return { ...scoped, draftWeights: createEqualWeightedFusionWeights() };
    });
  };

  const apply = () => {
    if (!normalizedDraft || busy || disabled) return;
    void fusion.apply(normalizedDraft).catch(() => undefined);
  };

  const revert = () => {
    fusion.clear();
    setEditor((current) => {
      const scoped = current.scopeKey === scopeKey ? current : createEditorState(scopeKey);
      return { ...scoped, appliedWeights: null };
    });
    onRevert();
  };

  return (
    <section className="weighted-fusion-panel" aria-labelledby="weighted-fusion-heading">
      <div className="weighted-fusion-heading">
        <div>
          <span className="panel-index">CUSTOM RANKING</span>
          <h3 id="weighted-fusion-heading">Weighted Fusion</h3>
        </div>
        <span className="panel-kicker">8 learners</span>
      </div>
      <p className="section-description">
        Set non-negative weights (normalized to 100%); 0 excludes a learner. Apply updates Top.
      </p>

      <fieldset className="weighted-fusion-weights" disabled={disabled || busy}>
        <legend className="sr-only">Learner weights</legend>
        {WEIGHTED_FUSION_LEARNERS.map((learner, index) => {
          const weight = draftWeights[learner];
          const percent = normalizedDraft ? normalizedDraft[learner] * 100 : null;
          const inputId = `weighted-fusion-${index}`;
          return (
            <div className="weighted-fusion-weight-row" key={learner}>
              <label htmlFor={`${inputId}-number`}>
                <span>{methodDisplayLabel(learner)}</span>
                <small>{percent === null ? "—" : `${percent.toFixed(2)}%`}</small>
              </label>
              <input
                id={`${inputId}-range`}
                type="range"
                min={WEIGHT_MIN}
                max={WEIGHT_MAX}
                step={WEIGHT_STEP}
                value={weight}
                aria-label={`${methodDisplayLabel(learner)} weight slider`}
                onChange={(event) => updateDraftWeight(learner, inputWeight(event.target.value))}
              />
              <input
                id={`${inputId}-number`}
                type="number"
                min={WEIGHT_MIN}
                max={WEIGHT_MAX}
                step={WEIGHT_STEP}
                value={weight}
                aria-label={`${methodDisplayLabel(learner)} numeric weight`}
                onChange={(event) => updateDraftWeight(learner, inputWeight(event.target.value))}
              />
            </div>
          );
        })}
      </fieldset>

      <div className="weighted-fusion-actions">
        <button
          type="button"
          className="button-secondary"
          disabled={disabled || busy || !normalizedDraft}
          onClick={normalizeDraft}
        >
          Normalize
        </button>
        <button
          type="button"
          className="button-secondary"
          disabled={disabled || busy}
          onClick={resetEqual}
        >
          Reset equal
        </button>
        <button
          type="button"
          className="button-primary"
          disabled={disabled || busy || !normalizedDraft || unchanged}
          onClick={apply}
        >
          {busy ? "Computing…" : unchanged ? "Applied" : "Apply to Top"}
        </button>
        <button
          type="button"
          className="text-button"
          disabled={disabled || busy || !appliedWeights}
          onClick={revert}
        >
          Revert
        </button>
      </div>

      {disabled && disabledReason && (
        <p className="weighted-fusion-status" role="note">{disabledReason}</p>
      )}
      {!disabled && !normalizedDraft && (
        <p className="weighted-fusion-status tuning-status-error" role="alert">
          At least one learner weight must be greater than zero.
        </p>
      )}
      {fusion.status === "loading" && (
        <p className="weighted-fusion-status" aria-live="polite">
          Computing exact weighted SoftGate scores and ranks…
        </p>
      )}
      {fusion.error && (
        <p className="weighted-fusion-status tuning-status-error" role="alert">{fusion.error}</p>
      )}
      {fusion.result && (
        <p className="weighted-fusion-status" aria-live="polite">
          Applied {fusion.result.equalWeights ? "equal" : "custom"} weights
          {fusion.result.baselineMaxAbsError === null
            ? "."
            : ` · baseline max |error| ${fusion.result.baselineMaxAbsError.toExponential(2)}`}
        </p>
      )}
      {!busy && !awaitingAppliedResult && appliedWeights && testAudit && (
        <section
          className="weighted-fusion-test-audit"
          aria-label="Weighted Fusion full Frozen Test comparison"
          aria-live="polite"
        >
          <div className="weighted-fusion-test-audit-heading">
            <div>
              <span>FULL FROZEN TEST</span>
              <strong>{testAudit.targetLabel}</strong>
            </div>
            <small>Ours-Full → Weighted</small>
          </div>
          <div className="weighted-fusion-test-tp-grid" role="list">
            {testAudit.changes.map(({ k, before, after }) => {
              const delta = after - before;
              return (
                <div className="weighted-fusion-test-tp" key={k} role="listitem">
                  <span>TP@{k}</span>
                  <strong>{before} → {after}</strong>
                  <small className={delta > 0 ? "positive" : delta < 0 ? "negative" : "neutral"}>
                    {delta > 0 ? `+${delta}` : delta}
                  </small>
                </div>
              );
            })}
          </div>
          <p>
            {testAudit.positiveCount.toLocaleString()} positives · {testAudit.evaluatedCount.toLocaleString()} test images
          </p>
        </section>
      )}
    </section>
  );
}

export default WeightedFusionPanel;
