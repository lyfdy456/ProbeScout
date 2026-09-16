/** Shared chart face. SVG/CSS resolves the variable; Canvas needs its computed value. */
export const CHART_UI_FONT = "var(--font-ui, Inter, ui-sans-serif, system-ui, sans-serif)";

export const CHART_UI_FONT_FALLBACK = "Inter, ui-sans-serif, system-ui, sans-serif";

export function chartCanvasFont(element: Element | null, size = 11): string {
  const view = element?.ownerDocument?.defaultView;
  const family = element && view
    ? view.getComputedStyle(element).fontFamily.trim()
    : "";
  // CanvasRenderingContext2D.font cannot resolve a CSS var() expression itself.
  return `${size}px ${family || CHART_UI_FONT_FALLBACK}`;
}
