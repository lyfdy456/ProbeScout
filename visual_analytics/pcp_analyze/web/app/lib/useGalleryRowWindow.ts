"use client";

import { useLayoutEffect, type RefObject } from "react";
import { galleryRowWindowHeight } from "./galleryRowWindow";

const HEIGHT_PROPERTY = "--gallery-row-window-height";

function pixels(value: string): number {
  return Number.parseFloat(value) || 0;
}

/** A height-only enhancement: never changes which images are ranked or selected. */
export function useGalleryRowWindow(
  scrollRef: RefObject<HTMLDivElement | null>,
  gridRef: RefObject<HTMLDivElement | null>,
  contentKey: unknown,
) {
  useLayoutEffect(() => {
    const viewport = scrollRef.current;
    const grid = gridRef.current;
    if (!viewport || !grid) return;
    const cards = Array.from(grid.children).filter(
      (child): child is HTMLElement => child instanceof HTMLElement && child.classList.contains("gallery-card"),
    );
    let frame: number | null = null;

    const measure = () => {
      frame = null;
      const gridStyle = getComputedStyle(grid);
      const viewportStyle = getComputedStyle(viewport);
      const height = galleryRowWindowHeight(
        // Layout offsets ignore the decorative translateY on hovered/selected cards.
        cards.map((card) => ({ top: card.offsetTop, height: card.offsetHeight })),
        2,
        pixels(gridStyle.paddingTop),
        pixels(gridStyle.paddingBottom),
      );
      if (height === null) return;
      const viewportChrome = viewportStyle.boxSizing === "border-box"
        ? pixels(viewportStyle.paddingTop) + pixels(viewportStyle.paddingBottom)
          + pixels(viewportStyle.borderTopWidth) + pixels(viewportStyle.borderBottomWidth)
        : 0;
      const gridMargins = pixels(gridStyle.marginTop) + pixels(gridStyle.marginBottom);
      const nextHeight = `${Math.max(0, height + viewportChrome + gridMargins)}px`;
      if (viewport.style.getPropertyValue(HEIGHT_PROPERTY) !== nextHeight) {
        viewport.style.setProperty(HEIGHT_PROPERTY, nextHeight);
      }
    };
    const scheduleMeasure = () => {
      if (frame === null) frame = requestAnimationFrame(measure);
    };

    measure();
    // Observe content, not the capped viewport; coalesce callbacks outside ResizeObserver.
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(scheduleMeasure);
    observer?.observe(grid);
    cards.forEach((card) => observer?.observe(card));
    window.addEventListener("resize", scheduleMeasure);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", scheduleMeasure);
      if (frame !== null) cancelAnimationFrame(frame);
      viewport.style.removeProperty(HEIGHT_PROPERTY);
    };
  }, [contentKey, gridRef, scrollRef]);
}
