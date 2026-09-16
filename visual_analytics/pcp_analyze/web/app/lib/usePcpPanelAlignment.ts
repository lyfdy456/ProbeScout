"use client";

import { useLayoutEffect, useState, type RefObject } from "react";

export function pcpPanelDimensions(
  aligned: boolean,
  explorationBottom: number,
  frameTop: number,
  chartChrome: number,
  minimumPlotHeight: number,
) {
  const frameHeight = aligned && Number.isFinite(explorationBottom - frameTop)
    ? Math.max(420, Math.round(explorationBottom - frameTop))
    : 715;
  return {
    frameHeight,
    plotHeight: Math.max(minimumPlotHeight, Math.floor(frameHeight - chartChrome)),
  };
}

/** Measure natural middle-column content, never the grid-stretched panel height. */
export function usePcpPanelAlignment({
  frameRef,
  explorationRef,
  layoutKey,
  minimumPlotHeight,
  enabled,
}: {
  frameRef: RefObject<HTMLDivElement | null>;
  explorationRef: RefObject<HTMLDivElement | null>;
  layoutKey: string;
  minimumPlotHeight: number;
  enabled: boolean;
}) {
  const [dimensions, setDimensions] = useState({ frameHeight: 715, plotHeight: 710 });

  useLayoutEffect(() => {
    const frame = frameRef.current;
    const exploration = explorationRef.current;
    if (!enabled || !frame || !exploration) return;
    let pending = 0;
    const media = window.matchMedia("(min-width: 1025px)");
    const update = () => {
      pending = 0;
      const plot = frame.querySelector<HTMLElement>(".cluster-summary-chart-stage, .pcp-root");
      if (!plot) return;
      const frameBounds = frame.getBoundingClientRect();
      const style = getComputedStyle(frame);
      // Account for summary controls, border and padding; scrolling must not alter geometry.
      const chrome = plot.getBoundingClientRect().top - frameBounds.top + frame.scrollTop
        + (parseFloat(style.paddingBottom) || 0) + (parseFloat(style.borderBottomWidth) || 0);
      const next = pcpPanelDimensions(
        media.matches, exploration.getBoundingClientRect().bottom,
        frameBounds.top, chrome, minimumPlotHeight,
      );
      setDimensions((current) => current.frameHeight === next.frameHeight
        && current.plotHeight === next.plotHeight ? current : next);
    };
    const schedule = () => {
      if (!pending) pending = window.requestAnimationFrame(update);
    };
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(schedule);
    const observe = () => {
      observer?.disconnect();
      observer?.observe(exploration);
      observer?.observe(frame);
      // Left controls/breadcrumbs can change the plot's starting position.
      if (frame.parentElement) {
        for (const child of frame.parentElement.children) observer?.observe(child);
      }
      const summaryHeader = frame.querySelector(".cluster-summary-pcp")?.firstElementChild;
      if (summaryHeader) observer?.observe(summaryHeader);
      schedule();
    };
    const mutations = new MutationObserver(observe);
    if (frame.parentElement) mutations.observe(frame.parentElement, { childList: true });
    observe();
    window.cancelAnimationFrame(pending);
    update();
    media.addEventListener("change", schedule);
    window.addEventListener("resize", schedule);
    document.fonts?.addEventListener("loadingdone", schedule);
    return () => {
      observer?.disconnect();
      mutations.disconnect();
      window.cancelAnimationFrame(pending);
      media.removeEventListener("change", schedule);
      window.removeEventListener("resize", schedule);
      document.fonts?.removeEventListener("loadingdone", schedule);
    };
  }, [enabled, explorationRef, frameRef, layoutKey, minimumPlotHeight]);

  return dimensions;
}
