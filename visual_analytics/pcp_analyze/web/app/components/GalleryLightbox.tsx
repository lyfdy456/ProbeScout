"use client";

import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import type { OriginalVqaLabel } from "../lib/tuningApi";
import {
  AttributeStrengthProfile,
  type AttributeStrengthPoint,
} from "./AttributeStrengthProfile";
import type {
  GalleryEvaluation,
  GalleryItem,
  ThumbnailAtlasDescriptor,
} from "./TopGallery";

export interface GalleryLightboxProps {
  items: readonly GalleryItem[];
  activeIndex: number;
  onActiveIndexChange: (index: number) => void;
  onClose: () => void;
  navigationHint?: string;
  evaluation?: Pick<GalleryEvaluation, "label" | "isPositive">;
  feedbackBusy?: boolean;
  feedbackStatus?: string;
  getOriginalVqaLabel?: (item: GalleryItem) => OriginalVqaLabel | null | undefined;
  getAttributeStrengths?: (item: GalleryItem) => readonly AttributeStrengthPoint[];
  attributeStrengthSourceLabel?: string;
  renderFeedback?: (item: GalleryItem) => ReactNode;
}

function atlasBackgroundStyle(
  atlas: ThumbnailAtlasDescriptor | null | undefined,
): CSSProperties {
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
}

function LightboxVisual({ item }: { item: GalleryItem }) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const directUrl = item.fullImageSrc || item.imageSrc || null;
  const atlas = item.thumbnailAtlas || null;
  const showDirect = Boolean(directUrl && !failed);
  const showAtlas = !showDirect && Boolean(atlas?.atlasUrl);
  const label = item.label || item.id;

  return (
    <div className="gallery-lightbox-visual">
      {showDirect && directUrl && (
        // The full image is requested only after the lightbox opens.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          className={`gallery-lightbox-image ${loaded ? "gallery-lightbox-image-loaded" : ""}`.trim()}
          src={directUrl}
          alt={label}
          decoding="async"
          draggable={false}
          onLoad={() => setLoaded(true)}
          onError={() => {
            setFailed(true);
            setLoaded(false);
          }}
        />
      )}
      {showDirect && !loaded && (
        <span className="gallery-lightbox-loading" role="status">
          Loading original image…
        </span>
      )}
      {showAtlas && (
        <span
          className="gallery-lightbox-atlas"
          style={atlasBackgroundStyle(atlas)}
          role="img"
          aria-label={label}
        />
      )}
      {!showDirect && !showAtlas && (
        <span className="gallery-lightbox-unavailable">Image unavailable</span>
      )}
      {failed && showAtlas && (
        <span className="gallery-lightbox-fallback">Original unavailable · thumbnail preview</span>
      )}
    </div>
  );
}

export function GalleryLightbox({
  items,
  activeIndex,
  onActiveIndexChange,
  onClose,
  evaluation,
  feedbackBusy = false,
  feedbackStatus = "",
  getOriginalVqaLabel,
  getAttributeStrengths,
  attributeStrengthSourceLabel,
  renderFeedback,
  navigationHint = "Esc closes · ←/→ moves through the current Top results",
}: GalleryLightboxProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const safeIndex = Math.max(0, Math.min(items.length - 1, activeIndex));
  const item = items[safeIndex];
  const hasMultiple = items.length > 1;

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();
    return () => {
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, []);

  useEffect(() => {
    const move = (delta: number) => {
      if (items.length <= 1) return;
      onActiveIndexChange((safeIndex + delta + items.length) % items.length);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      const feedbackOwnsArrowKey = (
        (event.key === "ArrowLeft" || event.key === "ArrowRight")
        && event.target instanceof Element
        && event.target.closest("[data-gallery-lightbox-feedback]") !== null
      );
      if (feedbackOwnsArrowKey) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        move(-1);
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        move(1);
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = Array.from(
        dialogRef.current?.querySelectorAll<HTMLElement>("button:not([disabled])") ?? [],
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [items.length, onActiveIndexChange, onClose, safeIndex]);

  if (!item || typeof document === "undefined") return null;
  const evaluationPositive = Boolean(evaluation?.isPositive(item));
  const originalVqaLabel = getOriginalVqaLabel?.(item) ?? null;
  const attributeStrengths = getAttributeStrengths?.(item) ?? [];

  const move = (delta: number) => {
    if (!hasMultiple) return;
    onActiveIndexChange((safeIndex + delta + items.length) % items.length);
  };
  const closeFromBackdrop = (event: MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) onClose();
  };

  return createPortal(
    <div className="gallery-lightbox-backdrop" onMouseDown={closeFromBackdrop}>
      <div
        ref={dialogRef}
        className={`gallery-lightbox-dialog ${
          evaluationPositive ? "gallery-lightbox-dialog-evaluation-positive" : ""
        }`.trim()}
        role="dialog"
        aria-modal="true"
        aria-label={`Image preview: ${item.label || item.id}${
          evaluationPositive
            ? `, ${evaluation?.label ?? "Evaluation"} ground-truth positive`
            : ""
        }`}
      >
        <div className="gallery-lightbox-toolbar">
          <div>
            <strong>{item.label || item.id.split("/").at(-1) || item.id}</strong>
            <span aria-live="polite">
              {safeIndex + 1} / {items.length}
              {evaluationPositive
                ? ` · ${evaluation?.label ?? "Evaluation"} ground-truth positive`
                : ""}
            </span>
            {evaluationPositive && (
              <span className="gallery-lightbox-evaluation-badge" aria-hidden="true">
                GT positive
              </span>
            )}
            {originalVqaLabel !== null && (
              <span
                className={`gallery-lightbox-vqa-badge gallery-original-vqa-${
                  originalVqaLabel === 1 ? "positive" : "negative"
                }`}
                aria-label={`Existing VQA supervision: ${
                  originalVqaLabel === 1 ? "positive" : "negative"
                }`}
              >
                VQA {originalVqaLabel === 1 ? "positive ✓" : "negative ✕"}
              </span>
            )}
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className="gallery-lightbox-close"
            aria-label="Close enlarged image"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <div className="gallery-lightbox-stage">
          <button
            type="button"
            className="gallery-lightbox-nav gallery-lightbox-prev"
            aria-label="Previous image"
            disabled={!hasMultiple}
            onClick={() => move(-1)}
          >
            ←
          </button>
          <LightboxVisual key={`${item.id}:${item.fullImageSrc ?? "atlas"}`} item={item} />
          <button
            type="button"
            className="gallery-lightbox-nav gallery-lightbox-next"
            aria-label="Next image"
            disabled={!hasMultiple}
            onClick={() => move(1)}
          >
            →
          </button>
        </div>

        {attributeStrengths.length > 0 && (
          <div className="gallery-lightbox-insights">
            <AttributeStrengthProfile
              points={attributeStrengths}
              variant="expanded"
              sourceLabel={attributeStrengthSourceLabel ?? "Per-attribute calibrated strength"}
            />
          </div>
        )}

        {renderFeedback && (
          <div
            className="gallery-lightbox-feedback"
            data-gallery-lightbox-feedback
            aria-busy={feedbackBusy}
          >
            <span className="gallery-lightbox-feedback-label">Feedback</span>
            {renderFeedback(item)}
            <span className="gallery-lightbox-feedback-status" role="status" aria-live="polite">
              {feedbackStatus}
            </span>
          </div>
        )}

        <div className="gallery-lightbox-caption">
          <span title={item.id}>{item.id}</span>
          <small>{navigationHint}</small>
        </div>
      </div>
    </div>,
    document.body,
  );
}

export default GalleryLightbox;
