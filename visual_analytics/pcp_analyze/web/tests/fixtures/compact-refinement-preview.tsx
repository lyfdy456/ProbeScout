// Synthetic, memory-only browser QA. Never connects to the workbench API.
import { useState } from "react";
import { createRoot } from "react-dom/client";
import { AnnotationReviewGallery } from "../../app/components/AnnotationReviewGallery";
import { SelectionOverlapPanel } from "../../app/components/SelectionOverlapPanel";
import { TuningPanel } from "../../app/components/TuningPanel";
import { FEEDBACK_LABELS } from "../../app/lib/feedbackLabels";
import { summarizeSelectionOverlap } from "../../app/lib/selectionOverlap";
import type { TuningAnnotation } from "../../app/lib/tuningApi";
import type { TuningSessionState } from "../../app/lib/useTuningSession";

const items = Array.from({ length: 8 }, (_, i) => ({
  id: `preview-${i + 1}`, rowIndex: i, label: `Image ${i + 1}`,
  imageSrc: `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 120"><rect width="160" height="120" fill="hsl(${i * 35} 30% 80%)"/><text x="80" y="65" text-anchor="middle" font-family="sans-serif" font-size="22">Image ${i + 1}</text></svg>`)}`,
}));
const itemById = new Map(items.map(item => [item.id, item]));
const initialAnnotations = new Map<string, TuningAnnotation>(items.map((item, i) => [item.id, {
  imageId: item.id, rowIndex: i, label: i < 2 ? -1 : i === 2 ? 0 : 1,
  source: "synthetic-preview", updatedAt: "initial", suggestedFailedAttributeId: i === 0 ? "brand" : "body",
}]));
const baseMask = new Uint8Array(400).fill(1);
const pcp = Uint8Array.from(baseMask, (_, i) => Number(i < 213));
const umap = Uint8Array.from(baseMask, (_, i) => Number(i >= 158 && i < 361));
const finalMask = Uint8Array.from(baseMask, (_, i) => Number(pcp[i] && umap[i]));
const overlap = summarizeSelectionOverlap({
  baseMask,
  conditions: [
    { id: "pcp", label: "PCP cluster 50", mask: pcp, rule: { kind: "category", field: "rank-cluster", value: "Fine-grained K50 · 50" }, ruleResult: { kind: "none", reason: "no-attributes" } },
    { id: "umap", label: "UMAP brush", mask: umap, rule: { kind: "projection", projection: "umap", xDomain: [1, 2], yDomain: [3, 4] }, ruleResult: { kind: "none", reason: "no-attributes" } },
  ],
  stages: [], finalMask,
});
const metrics = (ap: number) => ({ ap, evaluationCount: 44, positiveCount: 27, bestF1: .9, tpAt30: 26, tpAt50: 27, tpAt100: 27, tpAt200: 27 });
const state = {
  serviceState: "ready", counts: { usablePositive: 5, usableNegative: 2 },
  annotations: initialAnnotations, feedbackBreakdown: { unresolvedCount: 0, totalCount: 8, newCount: 8, existingCount: 0 },
  session: { id: "preview-only", taskId: "preview-only", targetId: "joint", baseMethod: "Ours-Full" },
  run: { id: "preview-run", mode: "weight_staged", evaluationScope: "vqa-validation", status: "succeeded", annotationCount: 8,
    before: metrics(.9197), after: metrics(.9271), testEvaluation: { before: metrics(.5), after: metrics(.6) } },
  busy: false, pendingAnnotationCount: 0, probeUpdateInProgress: false, resultStale: false,
  applyRun: async () => {}, revertRun: () => {},
} as unknown as TuningSessionState;

function Preview() {
  const [annotations, setAnnotations] = useState(initialAnnotations);
  const [events, setEvents] = useState<string[]>([]);
  const preferences = new Map([...annotations].map(([id, label]) => [id, FEEDBACK_LABELS.find(option => option.numeric === label.label)!.value]));
  return <main style={{ padding: 16, maxWidth: 408 }}>
    <h1 style={{ fontSize: 18 }}>Synthetic layout QA · no production data</h1>
    <SelectionOverlapPanel result={overlap} active />
    <div className="section-label"><span>Human Feedback</span></div>
    <AnnotationReviewGallery annotations={annotations} itemById={itemById} preferences={preferences}
      jointSession attributeTargets={[{ id: "brand", label: "Brand identity" }, { id: "body", label: "Convertible body type with soft-top roof" }]}
      onPreferenceChange={async (item, preference) => {
        setAnnotations(current => {
          const next = new Map(current);
          if (preference === "unmarked") next.delete(item.id);
          else next.set(item.id, { ...next.get(item.id)!, label: FEEDBACK_LABELS.find(option => option.value === preference)!.numeric, updatedAt: String(Date.now()) });
          return next;
        });
        setEvents(current => [...current, `${item.id}: ${preference}`]);
      }}
      onConfirmFailureAttributes={async (item, ids) => {
        await new Promise(resolve => setTimeout(resolve, 8000));
        setAnnotations(current => new Map(current).set(item.id, { ...current.get(item.id)!, failedAttributeIds: [...ids], failureAttributionConfirmed: true, updatedAt: String(Date.now()) }));
        setEvents(current => [...current, `${item.id}: ${ids.join(",")}`]);
      }} />
    <div className="section-label"><span>Refinement and Validation</span></div>
    <div className="refinement-training-actions">{["Update Probes", "Staged Weight Refinement", "Joint Weight Tune"].map((label, index) => <button key={label} className={`button-secondary ${index === 0 ? "function-probe-update" : "function-tune-action"}`} disabled>{label}</button>)}</div>
    <TuningPanel state={state} />
    <output aria-label="Synthetic saved events">{events.join(" | ") || "No writes"}</output>
  </main>;
}
createRoot(document.getElementById("root")!).render(<Preview />);
