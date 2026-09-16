"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import type { CandidateMask, ClusterId } from "./ProjectionScatter";
import { isCandidateRow } from "./ProjectionScatter";
import { countRemainingPositives } from "../lib/resultScope";
import { useGalleryRowWindow } from "../lib/useGalleryRowWindow";
import {
  FEEDBACK_LABELS,
  type FeedbackLabel,
} from "../lib/feedbackLabels";
import type { OriginalVqaLabel } from "../lib/tuningApi";
import {
  AttributeStrengthProfile,
  type AttributeStrengthPoint,
} from "./AttributeStrengthProfile";
import { GalleryLightbox } from "./GalleryLightbox";

export type PreferenceState = FeedbackLabel;

export interface ThumbnailAtlasDescriptor {
  atlasUrl: string;
  row: number;
  col: number;
  cols?: number;
  rows?: number;
}

export interface GalleryItem {
  id: string;
  rowIndex: number;
  label?: string;
  imageSrc?: string | null;
  /** Loaded only when the user opens the enlarged preview. */
  fullImageSrc?: string | null;
  thumbnailAtlas?: ThumbnailAtlasDescriptor | null;
  cluster?: ClusterId;
}

export interface GalleryLearner {
  id: string;
  label: string;
}

export interface GalleryEvaluation {
  label: string;
  totalPositiveCount: number;
  isPositive: (item: GalleryItem) => boolean;
  filtered?: boolean;
  note?: string;
}

export type PreferenceLookup =
  | ReadonlyMap<string, PreferenceState>
  | Readonly<Record<string, PreferenceState>>;

export type PreferenceChangeHandler = (
  item: GalleryItem,
  preference: PreferenceState,
) => boolean | void | Promise<boolean | void>;

export interface TopGalleryProps {
  items: readonly GalleryItem[];
  learner: GalleryLearner;
  /** Return the normalized rank for an item. The existing PCP convention is 1 = best. */
  getRank: (item: GalleryItem, learnerId: string) => number;
  candidateMask?: CandidateMask;
  preferences?: PreferenceLookup;
  selectedId?: string | null;
  limit?: number;
  rankDirection?: "descending" | "ascending";
  clusterColors?:
    | Readonly<Record<string, string>>
    | ((cluster: ClusterId) => string);
  title?: string;
  className?: string;
  evaluation?: GalleryEvaluation;
  canAnnotate?: (item: GalleryItem) => boolean;
  annotationDisabledReason?: (item: GalleryItem) => string;
  /** Original VQA label for the active Retrieval target; null means not selected by VQA. */
  getOriginalVqaLabel?: (item: GalleryItem) => OriginalVqaLabel | null | undefined;
  /** Model-only per-attribute calibrated strengths for this image. */
  getAttributeStrengths?: (item: GalleryItem) => readonly AttributeStrengthPoint[];
  attributeStrengthSourceLabel?: string;
  onItemSelect?: (item: GalleryItem) => void;
  onPreferenceChange?: PreferenceChangeHandler;
}

interface RankedItem {
  item: GalleryItem;
  score: number;
}

const GALLERY_CLUSTER_COLORS = [
  "#d85d3a",
  "#3f75c6",
  "#35a27c",
  "#8b63bb",
  "#d29a35",
  "#2f95a5",
  "#c95e8a",
  "#758345",
];

function getPreference(
  preferences: PreferenceLookup | undefined,
  id: string,
): PreferenceState {
  if (!preferences) return "unmarked";
  if (typeof (preferences as ReadonlyMap<string, PreferenceState>).get === "function") {
    return (preferences as ReadonlyMap<string, PreferenceState>).get(id) || "unmarked";
  }
  return (preferences as Readonly<Record<string, PreferenceState>>)[id] || "unmarked";
}

function compareRanked(
  left: RankedItem,
  right: RankedItem,
  direction: "descending" | "ascending",
): number {
  const scoreDifference =
    direction === "descending" ? right.score - left.score : left.score - right.score;
  if (scoreDifference !== 0) return scoreDifference;
  const rowDifference = left.item.rowIndex - right.item.rowIndex;
  return rowDifference || left.item.id.localeCompare(right.item.id);
}

/** Keeps only N records while scanning all 47k rows, avoiding a full-array sort. */
function insertIntoTop(
  ranked: RankedItem[],
  candidate: RankedItem,
  limit: number,
  direction: "descending" | "ascending",
) {
  let low = 0;
  let high = ranked.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (compareRanked(candidate, ranked[middle], direction) < 0) high = middle;
    else low = middle + 1;
  }
  if (low >= limit && ranked.length >= limit) return;
  ranked.splice(low, 0, candidate);
  if (ranked.length > limit) ranked.pop();
}

function hashCluster(cluster: ClusterId): number {
  if (typeof cluster === "number" && Number.isFinite(cluster)) return Math.abs(cluster);
  const value = String(cluster);
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) | 0;
  }
  return Math.abs(hash);
}

function resolveClusterColor(
  cluster: ClusterId | undefined,
  custom:
    | Readonly<Record<string, string>>
    | ((cluster: ClusterId) => string)
    | undefined,
): string {
  if (cluster === undefined) return "#8390a3";
  if (typeof custom === "function") return custom(cluster);
  const customColor = custom?.[String(cluster)];
  return (
    customColor || GALLERY_CLUSTER_COLORS[hashCluster(cluster) % GALLERY_CLUSTER_COLORS.length]
  );
}

export function GalleryThumbnail({ item }: { item: GalleryItem }) {
  const [directFailed, setDirectFailed] = useState(false);
  const [directLoaded, setDirectLoaded] = useState(false);
  const directUrl = item.imageSrc || null;
  const atlas = item.thumbnailAtlas || null;

  const showDirect = Boolean(directUrl && !directFailed);
  const atlasStyle = useMemo<CSSProperties>(() => {
    if (!atlas) return {};
    const columns = Math.max(1, atlas.cols ?? 10);
    const rows = Math.max(1, atlas.rows ?? 10);
    const column = Math.max(0, Math.min(columns - 1, atlas.col));
    const row = Math.max(0, Math.min(rows - 1, atlas.row));
    const x = columns === 1 ? 0 : (column / (columns - 1)) * 100;
    const y = rows === 1 ? 0 : (row / (rows - 1)) * 100;
    const escapedUrl = atlas.atlasUrl.replaceAll('"', '\\"');
    return {
      backgroundImage: `url("${escapedUrl}")`,
      backgroundPosition: `${x}% ${y}%`,
      backgroundRepeat: "no-repeat",
      backgroundSize: `${columns * 100}% ${rows * 100}%`,
    };
  }, [atlas]);

  const showAtlas = !showDirect && Boolean(atlas?.atlasUrl);
  const hasUsableSource = showDirect || showAtlas;
  const alt = item.label || item.id;

  return (
    <div className="gallery-thumb">
      {showDirect && directUrl && (
        // Native lazy loading is intentional: gallery images can point outside Next's image host list.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          className={`gallery-thumb-image ${directLoaded ? "gallery-thumb-image-loaded" : ""}`.trim()}
          src={directUrl}
          alt={alt}
          loading="lazy"
          decoding="async"
          draggable={false}
          onLoad={() => setDirectLoaded(true)}
          onError={() => {
            setDirectFailed(true);
            setDirectLoaded(false);
          }}
        />
      )}
      {showAtlas && (
        <span
          className="gallery-thumb-atlas"
          style={atlasStyle}
          role="img"
          aria-label={alt}
        />
      )}
      {((showDirect && !directLoaded) || !hasUsableSource) && (
        <span
          className={`gallery-thumb-placeholder ${!hasUsableSource ? "gallery-thumb-missing" : ""}`.trim()}
          aria-hidden="true"
        >
          {!hasUsableSource ? "No image" : ""}
        </span>
      )}
    </div>
  );
}

export function PreferenceControls({
  item,
  value,
  canAnnotate,
  disabledReason,
  onChange,
  expanded = false,
  pending = false,
}: {
  item: GalleryItem;
  value: PreferenceState;
  canAnnotate: boolean;
  disabledReason?: string;
  onChange?: PreferenceChangeHandler;
  expanded?: boolean;
  pending?: boolean;
}) {
  if (!canAnnotate) {
    return (
      <span className="gallery-feedback-locked" title={disabledReason}>
        {expanded && disabledReason ? disabledReason : "Feedback unavailable"}
      </span>
    );
  }

  return (
    <div
      className={`gallery-preference ${expanded ? "gallery-preference-expanded" : ""}`.trim()}
      role="group"
      aria-label={`Feedback label for ${item.label || item.id}`}
    >
      {FEEDBACK_LABELS.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`gallery-preference-button gallery-preference-${option.value} ${
            value === option.value ? "gallery-preference-active" : ""
          }`.trim()}
          aria-label={option.label}
          aria-pressed={value === option.value}
          title={option.label}
          disabled={!onChange || pending}
          onClick={() => {
            const nextPreference = value === option.value ? "unmarked" : option.value;
            void Promise.resolve(onChange?.(item, nextPreference)).catch(() => undefined);
          }}
        >
          <span aria-hidden="true">{option.glyph}</span>
          {expanded && <span className="gallery-preference-text">{option.shortLabel}</span>}
        </button>
      ))}
    </div>
  );
}

export function TopGallery({
  items,
  learner,
  getRank,
  candidateMask,
  preferences,
  selectedId = null,
  limit = 30,
  rankDirection = "descending",
  clusterColors,
  title,
  className = "",
  evaluation,
  canAnnotate,
  annotationDisabledReason,
  getOriginalVqaLabel,
  getAttributeStrengths,
  attributeStrengthSourceLabel,
  onItemSelect,
  onPreferenceChange,
}: TopGalleryProps) {
  const [previewId, setPreviewId] = useState<string | null>(null);
  const previewIdRef = useRef<string | null>(null);
  const previewNavigationRevisionRef = useRef(0);
  const galleryScrollRef = useRef<HTMLDivElement>(null);
  const galleryGridRef = useRef<HTMLDivElement>(null);
  const lightboxFeedbackSavingRef = useRef(false);
  const [lightboxFeedbackSaving, setLightboxFeedbackSaving] = useState(false);
  const [lightboxFeedbackStatus, setLightboxFeedbackStatus] = useState("");
  const safeLimit = Math.max(0, Math.floor(limit));
  const resolvedTitle = title ?? "Ranked Gallery";
  const { ranked, candidateCount } = useMemo(() => {
    const top: RankedItem[] = [];
    let candidates = 0;
    if (safeLimit === 0) {
      return { ranked: top, candidateCount: candidates };
    }

    for (const item of items) {
      if (!isCandidateRow(candidateMask, item.rowIndex)) continue;
      candidates += 1;
      const score = Number(getRank(item, learner.id));
      if (!Number.isFinite(score)) continue;
      insertIntoTop(top, { item, score }, safeLimit, rankDirection);
    }
    return { ranked: top, candidateCount: candidates };
  }, [candidateMask, getRank, items, learner.id, rankDirection, safeLimit]);

  const truePositiveCount = evaluation
    ? ranked.reduce(
        (count, entry) => count + (evaluation.isPositive(entry.item) ? 1 : 0),
        0,
      )
    : 0;
  const evaluatedK = ranked.length;
  const precision = evaluatedK > 0 ? truePositiveCount / evaluatedK : null;
  const recall = evaluation && evaluation.totalPositiveCount > 0
    ? truePositiveCount / evaluation.totalPositiveCount
    : null;
  const remainingPositiveCount = evaluation
    ? countRemainingPositives(evaluation.totalPositiveCount, truePositiveCount)
    : 0;
  const formatPercent = (value: number | null) => (
    value === null ? "—" : `${(value * 100).toFixed(1)}%`
  );
  const rankedItems = useMemo(() => ranked.map((entry) => entry.item), [ranked]);
  useGalleryRowWindow(galleryScrollRef, galleryGridRef, rankedItems);
  const previewIndex = previewId === null
    ? -1
    : rankedItems.findIndex((item) => item.id === previewId);

  useEffect(() => {
    if (galleryScrollRef.current) galleryScrollRef.current.scrollTop = 0;
  }, [rankedItems]);

  useEffect(() => {
    if (previewId === null || previewIndex >= 0) return;
    const stalePreviewId = previewId;
    queueMicrotask(() => {
      if (previewIdRef.current !== stalePreviewId) return;
      previewNavigationRevisionRef.current += 1;
      previewIdRef.current = null;
      setPreviewId(null);
      setLightboxFeedbackStatus("");
    });
  }, [previewId, previewIndex]);

  const handlePreviewIndexChange = useCallback((index: number) => {
    const nextItem = rankedItems[index];
    previewNavigationRevisionRef.current += 1;
    if (!nextItem) {
      previewIdRef.current = null;
      setPreviewId(null);
      setLightboxFeedbackStatus("");
      return;
    }
    previewIdRef.current = nextItem.id;
    setPreviewId(nextItem.id);
    setLightboxFeedbackStatus("");
    onItemSelect?.(nextItem);
  }, [onItemSelect, rankedItems]);

  const handleLightboxPreferenceChange = useCallback(async (
    item: GalleryItem,
    preference: PreferenceState,
  ) => {
    if (
      !onPreferenceChange
      || lightboxFeedbackSavingRef.current
      || !(canAnnotate?.(item) ?? true)
    ) return false;

    lightboxFeedbackSavingRef.current = true;
    const navigationRevision = previewNavigationRevisionRef.current;
    setLightboxFeedbackSaving(true);
    setLightboxFeedbackStatus("Saving feedback…");
    try {
      const committed = await onPreferenceChange(item, preference);
      if (committed === false) {
        if (previewIdRef.current === item.id) {
          setLightboxFeedbackStatus("Feedback was not saved.");
        }
        return false;
      }
      if (
        previewIdRef.current !== item.id
        || previewNavigationRevisionRef.current !== navigationRevision
      ) return committed;
      if (preference === "unmarked") {
        setLightboxFeedbackStatus("Feedback removed.");
        return committed;
      }

      const currentIndex = rankedItems.findIndex((rankedItem) => rankedItem.id === item.id);
      const nextItem = currentIndex >= 0 ? rankedItems[currentIndex + 1] : undefined;
      if (!nextItem) {
        setLightboxFeedbackStatus("Saved · end of the current Top list.");
        return committed;
      }

      previewNavigationRevisionRef.current += 1;
      previewIdRef.current = nextItem.id;
      setPreviewId(nextItem.id);
      setLightboxFeedbackStatus("Saved · moved to the next image.");
      onItemSelect?.(nextItem);
      return committed;
    } catch {
      if (previewIdRef.current === item.id) {
        setLightboxFeedbackStatus("Feedback was not saved.");
      }
      return false;
    } finally {
      lightboxFeedbackSavingRef.current = false;
      setLightboxFeedbackSaving(false);
    }
  }, [canAnnotate, onItemSelect, onPreferenceChange, rankedItems]);

  const handlePreviewClose = useCallback(() => {
    previewNavigationRevisionRef.current += 1;
    previewIdRef.current = null;
    setPreviewId(null);
    setLightboxFeedbackStatus("");
  }, []);

  const renderPreferenceControls = (
    item: GalleryItem,
    expanded = false,
    onChange: PreferenceChangeHandler | undefined = onPreferenceChange,
    pending = false,
  ) => (
    <PreferenceControls
      item={item}
      value={getPreference(preferences, item.id)}
      canAnnotate={canAnnotate?.(item) ?? Boolean(onPreferenceChange)}
      disabledReason={annotationDisabledReason?.(item)}
      onChange={onChange}
      expanded={expanded}
      pending={pending}
    />
  );

  return (
    <section className={`gallery-root ${className}`.trim()}>
      <div className="gallery-toolbar">
        <div>
          <h3 className="gallery-heading">{resolvedTitle}</h3>
          <span className="gallery-method" title={`Ranked by ${learner.label}`}>Top {safeLimit}</span>
        </div>
        <div className="gallery-count" aria-live="polite">
          <strong>{ranked.length}</strong>
          <span> from {candidateCount.toLocaleString()}</span>
        </div>
      </div>

      {evaluation && (
        <div className="gallery-evaluation" aria-live="polite">
          <div className="gallery-evaluation-metrics">
            <strong>
              {evaluation.filtered ? "Filtered " : ""}TP@{evaluatedK}: {truePositiveCount}
            </strong>
            <span>P@{evaluatedK}: {formatPercent(precision)}</span>
            <span>Recall@{evaluatedK}: {formatPercent(recall)}</span>
            <span>
              {remainingPositiveCount.toLocaleString()} / {evaluation.totalPositiveCount.toLocaleString()} positives remaining
            </span>
          </div>
          <small>
            {evaluation.label}{evaluation.note ? ` · ${evaluation.note}` : ""}
            {" · Green border = GT positive"}
          </small>
        </div>
      )}

      {ranked.length > 0 ? (
        <div
          ref={galleryScrollRef}
          className="top-gallery-scroll"
          role="region"
          aria-label={`${resolvedTitle} ranked images`}
          tabIndex={0}
        >
          <div ref={galleryGridRef} className="gallery-grid">
            {ranked.map(({ item, score }, position) => {
            const clusterColor = resolveClusterColor(item.cluster, clusterColors);
            const evaluationPositive = Boolean(evaluation?.isPositive(item));
            const originalVqaLabel = getOriginalVqaLabel?.(item) ?? null;
            const attributeStrengths = getAttributeStrengths?.(item) ?? [];
            const cardStyle = {
              "--gallery-cluster-color": clusterColor,
            } as CSSProperties;
            return (
              <article
                key={`${item.id}:${item.imageSrc ?? item.thumbnailAtlas?.atlasUrl ?? ""}`}
                className={`gallery-card ${
                  selectedId === item.id ? "gallery-card-selected" : ""
                } ${
                  evaluationPositive ? "gallery-card-evaluation-positive" : ""
                }`.trim()}
                style={cardStyle}
              >
                <button
                  type="button"
                  className="gallery-card-main"
                  aria-label={`Open enlarged rank ${position + 1}: ${item.label || item.id}${
                    evaluationPositive
                      ? `, ${evaluation?.label ?? "Evaluation"} ground-truth positive`
                      : ""
                  }`}
                  aria-haspopup="dialog"
                  aria-expanded={previewId === item.id}
                  aria-pressed={selectedId === item.id}
                  onClick={() => {
                    onItemSelect?.(item);
                    previewNavigationRevisionRef.current += 1;
                    previewIdRef.current = item.id;
                    setPreviewId(item.id);
                    setLightboxFeedbackStatus("");
                  }}
                >
                  <GalleryThumbnail item={item} />
                  <span className="gallery-rank">#{position + 1}</span>
                  <span className="gallery-card-badge-stack">
                    {originalVqaLabel !== null && (
                      <span
                        className={`gallery-original-vqa-badge gallery-original-vqa-${
                          originalVqaLabel === 1 ? "positive" : "negative"
                        }`}
                        aria-label={`Existing VQA supervision: ${
                          originalVqaLabel === 1 ? "positive" : "negative"
                        }`}
                        title={`Existing VQA supervision: ${
                          originalVqaLabel === 1 ? "positive" : "negative"
                        }`}
                      >
                        <span aria-hidden="true">VQA {originalVqaLabel === 1 ? "✓" : "✕"}</span>
                      </span>
                    )}
                    {evaluationPositive && (
                      <span className="gallery-evaluation-positive-badge" aria-hidden="true">
                        GT positive
                      </span>
                    )}
                  </span>
                  <span className="gallery-zoom-hint" aria-hidden="true">Expand</span>
                </button>
                {attributeStrengths.length > 0 && (
                  <div className="gallery-card-insights">
                    <AttributeStrengthProfile points={attributeStrengths} variant="compact" />
                  </div>
                )}
                <div className="gallery-meta">
                  <span className="gallery-label" title={item.id}>
                    {item.label || item.id.split("/").at(-1) || item.id}
                  </span>
                  <span className="gallery-score">{score.toFixed(3)}</span>
                </div>
                <div className="gallery-card-footer">
                  <span className="gallery-cluster">
                    {item.cluster === undefined ? "Unclustered" : `Cluster ${String(item.cluster)}`}
                  </span>
                  {renderPreferenceControls(item)}
                </div>
              </article>
            );
            })}
          </div>
        </div>
      ) : (
        <div className="gallery-empty">
          <strong>No ranked images</strong>
          <span>Adjust the current brush or filters to restore candidates.</span>
        </div>
      )}

      {previewIndex >= 0 && (
        <GalleryLightbox
          items={rankedItems}
          activeIndex={previewIndex}
          onActiveIndexChange={handlePreviewIndexChange}
          onClose={handlePreviewClose}
          evaluation={evaluation}
          feedbackBusy={lightboxFeedbackSaving}
          feedbackStatus={lightboxFeedbackStatus}
          getOriginalVqaLabel={getOriginalVqaLabel}
          getAttributeStrengths={getAttributeStrengths}
          attributeStrengthSourceLabel={attributeStrengthSourceLabel}
          renderFeedback={(item) => renderPreferenceControls(
            item,
            true,
            handleLightboxPreferenceChange,
            lightboxFeedbackSaving,
          )}
        />
      )}
    </section>
  );
}

export default TopGallery;
