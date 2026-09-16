"use client";

import { useLayoutEffect, useRef } from "react";
import { methodDisplayLabel } from "../lib/methodDisplay.js";
import {
  SMART_FILTER_LIMIT,
  type SmartFilterKind,
  type SmartFilterResult,
} from "../smartFilter";

interface SmartFilterPanelProps {
  activeKind: SmartFilterKind | null;
  learnerMethod: string;
  learnerMethods: readonly string[];
  targetLabel: string;
  comparisonMethod: string;
  sourceLabel: string;
  sourceDetail: string;
  unavailableReason?: string | null;
  prototypeComparisonAvailable: boolean;
  result: SmartFilterResult | null;
  onKindChange: (kind: SmartFilterKind | null) => void;
  onLearnerMethodChange: (method: string) => void;
}

const FILTERS: ReadonlyArray<{
  kind: SmartFilterKind;
  index: string;
  title: string;
  rule: string;
}> = [
  {
    kind: "softgate-boundary",
    index: "01",
    title: "SoftGate Boundary",
    rule: "Smallest min |gate − 0.50|",
  },
  {
    kind: "learner-disagreement",
    index: "02",
    title: "Probe Disagreement",
    rule: "Largest rank std across 8 probes",
  },
  {
    kind: "learner-fusion-gap",
    index: "03",
    title: "Individual-Probe Rescue",
    rule: "probe ≥ .90 · fusion ≤ .50 · gap ≥ .40",
  },
  {
    kind: "prototype-rank-gap",
    index: "04",
    title: "Prototype–Ranking Mismatch",
    rule: "prototype ≥ .90 · current ≤ .50 · gap ≥ .40",
  },
];

function cutoffText(result: SmartFilterResult) {
  if (result.cutoff === null) return "No finite cutoff";
  if (result.kind === "softgate-boundary") {
    return `50th boundary margin ≤ ${result.cutoff.toFixed(4)}`;
  }
  if (result.kind === "learner-disagreement") {
    return `50th rank standard deviation ≥ ${result.cutoff.toFixed(4)}`;
  }
  return `50th rank gap ≥ ${result.cutoff.toFixed(4)}`;
}

export function SmartFilterPanel({
  activeKind,
  learnerMethod,
  learnerMethods,
  targetLabel,
  comparisonMethod,
  sourceLabel,
  sourceDetail,
  unavailableReason,
  prototypeComparisonAvailable,
  result,
  onKindChange,
  onLearnerMethodChange,
}: SmartFilterPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const viewport = scrollRef.current;
    const grid = gridRef.current;
    if (!viewport || !grid) return;
    const buttons = Array.from(grid.querySelectorAll<HTMLButtonElement>(".smart-filter-card > button"));
    let frame: number | null = null;
    const measure = () => {
      frame = null;
      // Equal-height rows fit both cards, even when a rule wraps or fonts resize.
      const height = Math.max(80, ...buttons.map((button) => button.offsetHeight + 2));
      viewport.style.setProperty("--diagnostic-card-height", `${height}px`);
    };
    const schedule = () => {
      if (frame === null) frame = requestAnimationFrame(measure);
    };
    measure();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(schedule);
    buttons.forEach((button) => observer?.observe(button));
    window.addEventListener("resize", schedule);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", schedule);
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }, [unavailableReason, prototypeComparisonAvailable]);

  return (
    <div className="smart-filter-panel">
      {unavailableReason && (
        <p className="inline-error" role="alert">Diagnostics unavailable: {unavailableReason}</p>
      )}
      <div ref={scrollRef} className="smart-filter-scroll" role="region"
        aria-label="Diagnostic filters: scroll for all four recommendations" tabIndex={0}>
      <div ref={gridRef} className="smart-filter-grid" role="group" aria-label="Diagnostic Filters · Top 50">
        {FILTERS.map((filter) => {
          const active = activeKind === filter.kind;
          const disabled = Boolean(unavailableReason)
            || (filter.kind === "prototype-rank-gap" && !prototypeComparisonAvailable);
          return (
            <div key={filter.kind} className={`smart-filter-card ${active ? "active" : ""}`.trim()}>
              <button
                type="button"
                aria-pressed={active}
                disabled={disabled}
                title={unavailableReason ?? (disabled ? "Choose a Ranked by method other than Image Prototype." : `${sourceLabel}: ${sourceDetail}`)}
                onClick={() => onKindChange(active ? null : filter.kind)}
              >
                <span className="smart-filter-index">{filter.index}</span>
                <strong>{filter.title}</strong>
                <small>{unavailableReason ? "Current snapshot unavailable"
                  : disabled ? "Choose another Ranked by method" : filter.rule}</small>
                <em>Top {SMART_FILTER_LIMIT}</em>
              </button>
            </div>
          );
        })}
      </div>
      </div>

      {activeKind === "learner-fusion-gap" && (
        <label className="smart-filter-learner">
          <span>Probe · Individual-Probe Rescue</span>
          <select aria-label="Probe for Individual-Probe Rescue" value={learnerMethod}
            disabled={Boolean(unavailableReason)}
            onChange={(event) => onLearnerMethodChange(event.target.value)}>
            {learnerMethods.map((method) => (
              <option key={method} value={method}>{methodDisplayLabel(method)}</option>
            ))}
          </select>
        </label>
      )}

      {activeKind && unavailableReason && (
        <button type="button" className="text-button" onClick={() => onKindChange(null)}>Clear</button>
      )}

      {activeKind && result && (
        <div className="smart-filter-status" aria-live="polite">
          <strong>{result.selectedCount} / {SMART_FILTER_LIMIT} recommended</strong>
          <span>
            {result.eligibleCount.toLocaleString()} eligible from {result.candidateCount.toLocaleString()} scoped candidates
          </span>
          <small>
            {cutoffText(result)} · target {targetLabel}
            {activeKind === "prototype-rank-gap" ? ` · compared with ${comparisonMethod}` : ""}
          </small>
          <button type="button" className="text-button" onClick={() => onKindChange(null)}>
            Clear
          </button>
        </div>
      )}
    </div>
  );
}

export default SmartFilterPanel;
