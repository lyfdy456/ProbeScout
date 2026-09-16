"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { FEEDBACK_LABELS, type FeedbackLabel } from "../lib/feedbackLabels";
import type {
  OriginalVqaLabel,
  TuningAnnotation,
} from "../lib/tuningApi";
import { supportsRelationMismatch } from "../lib/tuningApi";
import type { AttributeStrengthPoint } from "./AttributeStrengthProfile";
import { GalleryLightbox } from "./GalleryLightbox";
import {
  GalleryThumbnail,
  PreferenceControls,
  type GalleryItem,
  type PreferenceChangeHandler,
  type PreferenceLookup,
} from "./TopGallery";

type ReviewFilter = "all" | "positive" | "negative" | "uncertain";

interface ReviewEntry {
  annotation: TuningAnnotation;
  item: GalleryItem;
}

export interface FailureAttributeTarget {
  id: string;
  label: string;
}

export type FailureAttributeConfirmHandler = (
  item: GalleryItem,
  failedAttributeIds: readonly string[],
) => Promise<boolean | void>;

export interface AnnotationReviewGalleryProps {
  taskId?: string;
  annotations: ReadonlyMap<string, TuningAnnotation>;
  itemById: ReadonlyMap<string, GalleryItem>;
  preferences: PreferenceLookup;
  busy?: boolean;
  clearDisabled?: boolean;
  onClearAll?: () => Promise<unknown>;
  canAnnotate?: (item: GalleryItem) => boolean;
  canRemoveAnnotation?: (item: GalleryItem) => boolean;
  isValidationRow?: (item: GalleryItem) => boolean;
  getOriginalVqaLabel?: (item: GalleryItem) => OriginalVqaLabel | null | undefined;
  getAttributeStrengths?: (item: GalleryItem) => readonly AttributeStrengthPoint[];
  attributeStrengthSourceLabel?: string;
  onPreferenceChange?: PreferenceChangeHandler;
  jointSession?: boolean;
  attributeTargets?: readonly FailureAttributeTarget[];
  onConfirmFailureAttributes?: FailureAttributeConfirmHandler;
}

function reviewGroup(label: TuningAnnotation["label"]): Exclude<ReviewFilter, "all"> {
  if (label > 0) return "positive";
  if (label < 0) return "negative";
  return "uncertain";
}

function supervisionLabel(annotation: TuningAnnotation): string {
  if (annotation.supervision?.excludedReason === "fixed-vqa-validation") return "Val · excluded";
  const relation = annotation.supervision?.relation;
  if (relation === "new") return "New";
  if (relation === "override") return "Override";
  if (relation === "reinforce") return "Reinforce";
  if (relation === "uncertain") return "Review only";
  return "Saved";
}

function currentLabel(annotation: TuningAnnotation) {
  return FEEDBACK_LABELS.find((option) => option.numeric === annotation.label);
}

function currentPreference(preferences: PreferenceLookup, imageId: string) {
  const map = preferences as ReadonlyMap<string, FeedbackLabel>;
  if (typeof map.get === "function") return map.get(imageId) ?? "unmarked";
  return (preferences as Readonly<Record<string, FeedbackLabel>>)[imageId] ?? "unmarked";
}

function initialFailedAttributeIds(annotation: TuningAnnotation, relationMismatchAllowed = false) {
  if (relationMismatchAllowed && annotation.failureAttributionConfirmed && annotation.failedAttributeIds?.length === 0) return [];
  if (annotation.failedAttributeIds?.length) return [...annotation.failedAttributeIds].sort();
  return annotation.suggestedFailedAttributeId ? [annotation.suggestedFailedAttributeId] : [];
}

function sameAttributeIds(left: readonly string[], right: readonly string[]) {
  if (left.length !== right.length) return false;
  return [...left].sort().every((attributeId, index) => attributeId === [...right].sort()[index]);
}

function FailureAttributeSelector({
  annotation,
  item,
  attributeTargets,
  busy,
  onConfirm,
  relationMismatchAllowed = false,
}: {
  annotation: TuningAnnotation;
  item: GalleryItem;
  attributeTargets: readonly FailureAttributeTarget[];
  busy: boolean;
  onConfirm: FailureAttributeConfirmHandler;
  relationMismatchAllowed?: boolean;
}) {
  const [selectedIds, setSelectedIds] = useState<string[]>(() => initialFailedAttributeIds(annotation, relationMismatchAllowed));
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState("");

  const confirmedIds = annotation.failedAttributeIds ?? [];
  const selectionIsConfirmed = Boolean(
    annotation.failureAttributionConfirmed
    && sameAttributeIds(selectedIds, confirmedIds),
  );
  const relationIsConfirmed = relationMismatchAllowed && annotation.failureAttributionConfirmed
    && confirmedIds.length === 0;
  const stateLabel = selectionIsConfirmed
    ? relationIsConfirmed ? "Relation mismatch confirmed" : "Confirmed"
    : annotation.suggestedFailedAttributeId
      ? "Suggested · unconfirmed"
      : "Unconfirmed";

  const confirm = async () => {
    if (selectedIds.length === 0 || saving || busy) return;
    setSaving(true);
    setStatus("Saving…");
    try {
      const committed = await onConfirm(item, selectedIds);
      setStatus(committed === false ? "Not saved" : "Confirmed");
    } catch {
      setStatus("Not saved");
    } finally {
      setSaving(false);
    }
  };

  const confirmRelationMismatch = async () => {
    if (!relationMismatchAllowed || saving || busy) return;
    setSaving(true);
    setStatus("Saving…");
    try {
      const committed = await onConfirm(item, []);
      if (committed !== false) setSelectedIds([]);
      setStatus(committed === false ? "Not saved" : "Relation mismatch confirmed");
    } catch {
      setStatus("Not saved");
    } finally {
      setSaving(false);
    }
  };

  return (
    <fieldset className="failure-attribute-selector" disabled={saving || busy}>
      <legend>
        {relationMismatchAllowed ? "Failure reason" : "Failed attribute"} <span>{status || stateLabel}</span>
      </legend>
      <div className="failure-attribute-options">
        {attributeTargets.map((target) => {
          const checked = selectedIds.includes(target.id);
          const suggested = annotation.suggestedFailedAttributeId === target.id;
          return (
            <label key={target.id}>
              <input
                type="checkbox"
                checked={checked}
                onChange={() => {
                  setSelectedIds((current) => (
                    current.includes(target.id)
                      ? current.filter((attributeId) => attributeId !== target.id)
                      : [...current, target.id]
                  ));
                  setStatus("");
                }}
              />
              <span>{target.label}{suggested ? " · suggested" : ""}</span>
            </label>
          );
        })}
      </div>
      <button
        type="button"
        disabled={selectedIds.length === 0 || selectionIsConfirmed || saving || busy}
        onClick={() => void confirm()}
      >
        {selectionIsConfirmed ? "Confirmed" : "Confirm"}
      </button>
      {relationMismatchAllowed && (
        <button
          type="button"
          disabled={Boolean(relationIsConfirmed) || saving || busy}
          title="Reject the complete relation; leave every attribute unknown. This feedback only updates holistic fusion weights."
          onClick={() => void confirmRelationMismatch()}
        >
          {relationIsConfirmed ? "Relation mismatch confirmed" : "Confirm relation mismatch"}
        </button>
      )}
    </fieldset>
  );
}

export function AnnotationReviewGallery({
  taskId,
  annotations,
  itemById,
  preferences,
  busy = false,
  clearDisabled = false,
  onClearAll,
  canAnnotate,
  canRemoveAnnotation,
  isValidationRow,
  getOriginalVqaLabel,
  getAttributeStrengths,
  attributeStrengthSourceLabel,
  onPreferenceChange,
  jointSession = false,
  attributeTargets = [],
  onConfirmFailureAttributes,
}: AnnotationReviewGalleryProps) {
  const [expanded, setExpanded] = useState(false);
  const [filter, setFilter] = useState<ReviewFilter>("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState("");
  const mutationPending = useRef(false);

  const entries = useMemo<ReviewEntry[]>(() => (
    [...annotations.values()]
      .map((annotation) => {
        const item = itemById.get(annotation.imageId);
        return item ? { annotation, item } : null;
      })
      .filter((entry): entry is ReviewEntry => entry !== null)
      .sort((left, right) => (
        left.item.rowIndex - right.item.rowIndex
        || left.item.id.localeCompare(right.item.id)
      ))
  ), [annotations, itemById]);

  const counts = useMemo(() => {
    const result = { positive: 0, negative: 0, uncertain: 0 };
    for (const entry of entries) result[reviewGroup(entry.annotation.label)] += 1;
    return result;
  }, [entries]);
  const unavailableCount = Math.max(0, annotations.size - entries.length);

  const filteredEntries = useMemo(
    () => filter === "all"
      ? entries
      : entries.filter((entry) => reviewGroup(entry.annotation.label) === filter),
    [entries, filter],
  );
  const filteredItems = useMemo(
    () => filteredEntries.map((entry) => entry.item),
    [filteredEntries],
  );
  const selectedEntry = filteredEntries.find((entry) => entry.item.id === selectedId)
    ?? filteredEntries[0]
    ?? null;
  const previewIndex = previewId
    ? filteredItems.findIndex((item) => item.id === previewId)
    : -1;
  const reviewScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!reviewScrollRef.current) return;
    reviewScrollRef.current.scrollLeft = 0;
    reviewScrollRef.current.scrollTop = 0;
  }, [expanded, filter]);

  const changePreference: PreferenceChangeHandler = async (item, preference) => {
    if (!onPreferenceChange || mutationPending.current || busy) return false;
    if (preference === "unmarked"
      ? !(canRemoveAnnotation?.(item) ?? canAnnotate?.(item) ?? true)
      : !(canAnnotate?.(item) ?? true)) return false;
    mutationPending.current = true;
    setSaving(true);
    setStatus("Saving…");
    try {
      const committed = await onPreferenceChange(item, preference);
      if (committed === false) {
        setStatus("Feedback was not saved.");
        return false;
      }
      setStatus(preference === "unmarked" ? "Feedback removed." : "Feedback saved.");
      return true;
    } catch {
      setStatus("Feedback was not saved.");
      return false;
    } finally {
      mutationPending.current = false;
      setSaving(false);
    }
  };

  const confirmFailureAttributes: FailureAttributeConfirmHandler = async (item, failedAttributeIds) => {
    const annotation = annotations.get(item.id);
    if (!onConfirmFailureAttributes || mutationPending.current || busy
      || !jointSession || !annotation || annotation.label >= 0
      || !(canAnnotate?.(item) ?? true)) return false;
    mutationPending.current = true;
    setSaving(true);
    setStatus(`Saving ${item.label || item.id}…`);
    try {
      // Capture the image and draft at submission. Navigation never retargets a save.
      const committed = await onConfirmFailureAttributes(item, [...failedAttributeIds]);
      setStatus(committed === false ? "Failed attributes were not saved." : `Saved ${item.label || item.id}.`);
      return committed !== false;
    } catch {
      setStatus("Failed attributes were not saved.");
      return false;
    } finally {
      mutationPending.current = false;
      setSaving(false);
    }
  };

  const renderPreferenceControls = (item: GalleryItem, expandedControls = false) => (
    <PreferenceControls
      item={item}
      value={currentPreference(preferences, item.id)}
      canAnnotate={Boolean(onPreferenceChange) && (canAnnotate?.(item) ?? true)}
      onChange={changePreference}
      expanded={expandedControls}
      pending={saving || busy}
    />
  );

  const renderFailureAttributes = (item: GalleryItem) => {
    const annotation = annotations.get(item.id);
    if (
      !jointSession
      || !annotation
      || annotation.label >= 0
      || attributeTargets.length === 0
      || !onConfirmFailureAttributes
    ) return null;
    return (
      <FailureAttributeSelector
        key={`${item.id}:${annotation.updatedAt}:${annotation.failureAttributionConfirmed ? "confirmed" : "open"}`}
        annotation={annotation}
        item={item}
        attributeTargets={attributeTargets}
        busy={saving || busy || !(canAnnotate?.(item) ?? true)}
        onConfirm={confirmFailureAttributes}
        relationMismatchAllowed={supportsRelationMismatch(taskId, jointSession ? "joint" : "")}
      />
    );
  };

  const renderReviewControls = (item: GalleryItem, expandedControls = false) => (
    <div className="annotation-review-feedback-controls">
      {isValidationRow?.(item) && <span className="tuning-integrity-note">Val · excluded</span>}
      {renderPreferenceControls(item, expandedControls)}
      {renderFailureAttributes(item)}
      {jointSession && annotations.get(item.id)?.label !== undefined
        && (annotations.get(item.id)?.label ?? 0) >= 0 && (
          <span className="annotation-review-state">
            {(annotations.get(item.id)?.label ?? 0) > 0 ? "Relevant · all required attributes" : "Unsure · review only"}
          </span>
        )}
      {canAnnotate?.(item) === false && canRemoveAnnotation?.(item) && (
        <button type="button" className="text-button" disabled={saving || busy}
          onClick={() => void changePreference(item, "unmarked")}>
          Remove label
        </button>
      )}
    </div>
  );

  const filterOptions: ReadonlyArray<{
    id: ReviewFilter;
    label: string;
    count: number;
  }> = [
    { id: "all", label: "All", count: entries.length },
    { id: "positive", label: "Positive", count: counts.positive },
    { id: "negative", label: "Negative", count: counts.negative },
    { id: "uncertain", label: "Unsure", count: counts.uncertain },
  ];

  return (
    <section className="annotation-review" aria-label="Current session feedback">
      <div className="annotation-review-header">
      <button
        type="button"
        className="annotation-review-toggle"
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        <span>
          <strong>Review feedback</strong>
        </span>
        <span className="annotation-review-toggle-count">
          {annotations.size.toLocaleString()}
          <i aria-hidden="true">{expanded ? "−" : "+"}</i>
        </span>
      </button>
      {annotations.size > 0 && onClearAll && (
        <button type="button" className="text-button annotation-review-clear"
          aria-label={`Clear all ${annotations.size} feedback labels`}
          disabled={clearDisabled || busy || saving}
          onClick={() => void onClearAll().catch(() => undefined)}>
          Clear all
        </button>
      )}
      </div>

      {expanded && (
        <div className="annotation-review-body">
          <div className="annotation-review-toolbar">
            <div className="annotation-review-filters" role="group" aria-label="Filter feedback">
              {filterOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className={filter === option.id ? "annotation-review-filter-active" : ""}
                  aria-pressed={filter === option.id}
                  onClick={() => {
                    setFilter(option.id);
                    setSelectedId(null);
                    setPreviewId(null);
                  }}
                >
                  {option.label} <span>{option.count.toLocaleString()}</span>
                </button>
              ))}
            </div>
            <span className="annotation-review-status" role="status" aria-live="polite">
              {status || (unavailableCount > 0
                ? `${unavailableCount.toLocaleString()} labels are unavailable in the loaded gallery.`
                : "")}
            </span>
          </div>

          {filteredEntries.length > 0 ? (
            <div className="annotation-review-workspace">
            <div
              ref={reviewScrollRef}
              className="annotation-review-scroll"
              role="region"
              aria-label={`${filterOptions.find((option) => option.id === filter)?.label ?? "All"} feedback images`}
              tabIndex={0}
            >
              <div className="annotation-review-grid">
                {filteredEntries.map(({ annotation, item }) => {
                  const label = currentLabel(annotation);
                  const group = reviewGroup(annotation.label);
                  const originalVqaLabel = getOriginalVqaLabel?.(item) ?? null;
                  return (
                    <article key={item.id} className={`gallery-card annotation-review-card annotation-review-${group}${selectedEntry?.item.id === item.id ? " annotation-review-card-selected" : ""}`}>
                      <button
                        type="button"
                        className="gallery-card-main"
                        aria-label={`Select feedback image: ${item.label || item.id}`}
                        aria-pressed={selectedEntry?.item.id === item.id}
                        onClick={() => {
                          setSelectedId(item.id);
                        }}
                      >
                        <GalleryThumbnail item={item} />
                        <span className="gallery-card-badge-stack">
                          <span className={`annotation-review-label-badge annotation-review-label-${group}`}>
                            {label?.shortLabel ?? "Saved"}
                          </span>
                          {originalVqaLabel !== null && (
                            <span className={`gallery-original-vqa-badge gallery-original-vqa-${
                              originalVqaLabel === 1 ? "positive" : "negative"
                            }`}>
                              VQA {originalVqaLabel === 1 ? "+" : "−"}
                            </span>
                          )}
                        </span>
                      </button>
                      <div className="gallery-meta">
                        <span className="gallery-label" title={item.id}>
                          {item.label || item.id.split("/").at(-1) || item.id}
                        </span>
                        <span className="annotation-review-relation">
                          {supervisionLabel(annotation)}
                        </span>
                      </div>
                      <button type="button" className="annotation-review-expand"
                        aria-label={`Open feedback image: ${item.label || item.id}`}
                        aria-haspopup="dialog" aria-expanded={previewId === item.id}
                        onClick={() => {
                          setSelectedId(item.id);
                          setPreviewId(item.id);
                        }}>
                        Enlarge
                      </button>
                    </article>
                  );
                })}
              </div>
            </div>
            {selectedEntry && (
              <aside className="annotation-review-editor" aria-label="Selected feedback image">
                <div className="annotation-review-editor-heading">
                  <strong title={selectedEntry.item.id}>{selectedEntry.item.label || selectedEntry.item.id}</strong>
                  <span>{supervisionLabel(selectedEntry.annotation)}</span>
                </div>
                {previewIndex < 0 && renderReviewControls(selectedEntry.item)}
                {previewIndex >= 0 && <span className="annotation-review-state">Editing in enlarged view</span>}
              </aside>
            )}
            </div>
          ) : (
            <div className="annotation-review-empty">
              {entries.length === 0
                ? "No feedback labels in this session yet."
                : `No ${filter} feedback labels.`}
            </div>
          )}
        </div>
      )}

      {previewIndex >= 0 && (
        <GalleryLightbox
          items={filteredItems}
          activeIndex={previewIndex}
          onActiveIndexChange={(index) => {
            const nextId = filteredItems[index]?.id ?? null;
            setPreviewId(nextId);
            setSelectedId(nextId);
          }}
          onClose={() => {
            setPreviewId(null);
            setStatus("");
          }}
          feedbackBusy={saving || busy}
          feedbackStatus={status}
          getOriginalVqaLabel={getOriginalVqaLabel}
          getAttributeStrengths={getAttributeStrengths}
          attributeStrengthSourceLabel={attributeStrengthSourceLabel}
          renderFeedback={(item) => renderReviewControls(item, true)}
          navigationHint="Esc closes · ←/→ reviews the filtered labels"
        />
      )}
    </section>
  );
}

export default AnnotationReviewGallery;
