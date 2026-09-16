"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from "react";
import { scaleLinear, scaleOrdinal } from "d3";
import { CHART_UI_FONT, chartCanvasFont } from "../lib/chartTypography";

export type ProjectionKind = "pca" | "umap";
export type ClusterId = string | number;
export type SelectionCollection<T> =
  | ReadonlySet<T>
  | ReadonlyArray<T>
  | null;

/** A mask is indexed by the dataset row, not by the filtered array position. */
export type CandidateMask =
  | Uint8Array
  | ReadonlyArray<boolean | 0 | 1>
  | ReadonlySet<number>
  | null;

export interface ProjectionPoint {
  id: string;
  rowIndex: number;
  cluster: ClusterId;
  pca: ArrayLike<number>;
  umap?: ArrayLike<number> | null;
  label?: string;
}

export interface ProjectionSelection {
  projection: ProjectionKind;
  ids: string[];
  rowIndices: number[];
  xDomain: readonly [number, number];
  yDomain: readonly [number, number];
}

export interface ProjectionManualHighlight {
  rowIndex: number;
  /** Feedback label: 2/1 positive, 0 uncertain, and -1/-2 negative. */
  label: number;
}

export interface ProjectionScatterProps {
  points: readonly ProjectionPoint[];
  projection: ProjectionKind;
  /**
   * Presentation-only PCP selection. It never changes which points can be hit
   * or box-selected, nor their projection coordinates or cluster membership.
   */
  highlightMask?: CandidateMask;
  candidateMask?: CandidateMask;
  /**
   * Common-scope mask used only to evaluate the embedding rectangle. Keeping
   * it separate from candidateMask makes the rectangle an independent rule.
   */
  boxSelectionMask?: CandidateMask;
  clusterColors?:
    | Readonly<Record<string, string>>
    | ((cluster: ClusterId) => string);
  /** Lets a parent-level reset clear the locally drawn selection rectangle. */
  boxSelectionActive?: boolean;
  /**
   * Human-reviewed rows/IDs to emphasize without changing filtering or point
   * interaction. When both collections are supplied, a point matching either
   * one is highlighted.
   */
  selectedRowIndices?: SelectionCollection<number>;
  selectedIds?: SelectionCollection<string>;
  /** Label-aware human feedback; takes precedence over the generic selectors. */
  manualHighlights?: readonly ProjectionManualHighlight[];
  selectedId?: string | null;
  height?: number;
  pointRadius?: number;
  title?: string;
  showHeading?: boolean;
  /** Parent-owned projection/cluster selectors displayed inside the chart card. */
  controls?: ReactNode;
  className?: string;
  onPointHover?: (point: ProjectionPoint | null) => void;
  onPointClick?: (point: ProjectionPoint | null) => void;
  onBoxSelect?: (selection: ProjectionSelection | null) => void;
}

interface PixelPoint {
  x: number;
  y: number;
}

interface PixelRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface PointerState {
  pointerId: number;
  start: PixelPoint;
  last: PixelPoint;
  moved: boolean;
}

type ProjectionScale = ((value: number) => number) & {
  invert(value: number): number;
  ticks(count?: number): number[];
};

interface ScatterLayout {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
  xScale: ProjectionScale;
  yScale: ProjectionScale;
  screenX: Float32Array;
  screenY: Float32Array;
  groups: Map<string, number[]>;
  colors: Map<string, string>;
  buckets: Map<number, number[]>;
  bucketColumns: number;
  bucketSize: number;
  validCount: number;
}

const DEFAULT_CLUSTER_COLORS = [
  "#d85d3a",
  "#3f75c6",
  "#35a27c",
  "#8b63bb",
  "#d29a35",
  "#2f95a5",
  "#c95e8a",
  "#758345",
  "#68718c",
  "#b66f3e",
];

function isSetMask(mask: Exclude<CandidateMask, null>): mask is ReadonlySet<number> {
  return typeof (mask as ReadonlySet<number>).has === "function";
}

function selectionIncludes<T>(
  selection: SelectionCollection<T> | undefined,
  value: T,
): boolean {
  if (!selection) return false;
  if (typeof (selection as ReadonlySet<T>).has === "function") {
    return (selection as ReadonlySet<T>).has(value);
  }
  return (selection as ReadonlyArray<T>).includes(value);
}

export function isCandidateRow(
  mask: CandidateMask | undefined,
  rowIndex: number,
): boolean {
  if (!mask) return true;
  if (isSetMask(mask)) return mask.has(rowIndex);
  return rowIndex >= 0 && rowIndex < mask.length && Boolean(mask[rowIndex]);
}

function snapshotCandidateMask(mask: CandidateMask | undefined): CandidateMask | undefined {
  if (mask == null) return mask;
  if (isSetMask(mask)) return new Set(mask);
  if (mask instanceof Uint8Array) return mask.slice();
  return mask.slice();
}

function coordinatesFor(
  point: ProjectionPoint,
  projection: ProjectionKind,
): readonly [number, number] | null {
  const source = projection === "pca" ? point.pca : point.umap;
  if (!source || source.length < 2) return null;
  const x = Number(source[0]);
  const y = Number(source[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

function paddedDomain(minimum: number, maximum: number): [number, number] {
  if (minimum === maximum) {
    const padding = Math.abs(minimum) > 0 ? Math.abs(minimum) * 0.08 : 1;
    return [minimum - padding, maximum + padding];
  }
  const padding = (maximum - minimum) * 0.055;
  return [minimum - padding, maximum + padding];
}

function normalizeRect(rect: PixelRect): PixelRect {
  return {
    x0: Math.min(rect.x0, rect.x1),
    y0: Math.min(rect.y0, rect.y1),
    x1: Math.max(rect.x0, rect.x1),
    y1: Math.max(rect.y0, rect.y1),
  };
}

function candidateMasksMatch(
  first: CandidateMask | undefined,
  second: CandidateMask | undefined,
  points: readonly ProjectionPoint[],
): boolean {
  if (first === second) return true;
  for (const point of points) {
    if (
      isCandidateRow(first, point.rowIndex) !==
      isCandidateRow(second, point.rowIndex)
    ) {
      return false;
    }
  }
  return true;
}

function candidateMaskMatchesRows(
  mask: CandidateMask | undefined,
  rows: ReadonlySet<number> | null,
  points: readonly ProjectionPoint[],
): boolean {
  if (!rows) return false;
  let candidateCount = 0;
  for (const point of points) {
    if (!isCandidateRow(mask, point.rowIndex)) continue;
    candidateCount += 1;
    if (!rows.has(point.rowIndex)) return false;
  }
  return candidateCount === rows.size;
}

function selectionForRect(
  rect: PixelRect,
  layout: ScatterLayout,
  points: readonly ProjectionPoint[],
  projection: ProjectionKind,
  mask: CandidateMask | undefined,
): ProjectionSelection | null {
  const normalized = normalizeRect(rect);
  const ids: string[] = [];
  const rowIndices: number[] = [];
  points.forEach((point, index) => {
    if (!isCandidateRow(mask, point.rowIndex)) return;
    const x = layout.screenX[index];
    const y = layout.screenY[index];
    if (
      x >= normalized.x0 &&
      x <= normalized.x1 &&
      y >= normalized.y0 &&
      y <= normalized.y1
    ) {
      ids.push(point.id);
      rowIndices.push(point.rowIndex);
    }
  });
  if (rowIndices.length === 0) return null;

  const dataX0 = layout.xScale.invert(normalized.x0);
  const dataX1 = layout.xScale.invert(normalized.x1);
  const dataY0 = layout.yScale.invert(normalized.y1);
  const dataY1 = layout.yScale.invert(normalized.y0);
  return {
    projection,
    ids,
    rowIndices,
    xDomain: [Math.min(dataX0, dataX1), Math.max(dataX0, dataX1)],
    yDomain: [Math.min(dataY0, dataY1), Math.max(dataY0, dataY1)],
  };
}

function prepareCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): CanvasRenderingContext2D | null {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.max(1, Math.round(width * ratio));
  const pixelHeight = Math.max(1, Math.round(height * ratio));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return context;
}

function drawSmallPoint(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  dense: boolean,
) {
  if (dense) {
    const diameter = Math.max(1.25, radius * 1.55);
    context.fillRect(x - diameter / 2, y - diameter / 2, diameter, diameter);
  } else {
    context.moveTo(x + radius, y);
    context.arc(x, y, radius, 0, Math.PI * 2);
  }
}

export function ProjectionScatter({
  points,
  projection,
  highlightMask,
  candidateMask,
  boxSelectionMask,
  clusterColors,
  boxSelectionActive,
  selectedRowIndices,
  selectedIds,
  manualHighlights,
  selectedId = null,
  height = 310,
  pointRadius = 2.1,
  title = "Visual Embedding Exploration",
  showHeading = true,
  controls,
  className = "",
  onPointHover,
  onPointClick,
  onBoxSelect,
}: ProjectionScatterProps) {
  const stageRef = useRef<HTMLDivElement>(null);
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const pointerStateRef = useRef<PointerState | null>(null);
  const hoverIndexRef = useRef<number | null>(null);
  // Preserve the common selection population from before the first drag so a
  // later drag replaces the old rectangle instead of nesting inside it.
  const selectionBaseMaskRef = useRef<CandidateMask | undefined>(
    snapshotCandidateMask(boxSelectionMask),
  );
  const previousBoxSelectionMaskRef = useRef<CandidateMask | undefined>(
    snapshotCandidateMask(boxSelectionMask),
  );
  const lastEmittedRowsRef = useRef<ReadonlySet<number> | null>(null);
  const onBoxSelectRef = useRef(onBoxSelect);
  const selectionActiveRef = useRef(false);
  const [size, setSize] = useState({ width: 0, height });
  const [fontRevision, setFontRevision] = useState(0);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<PixelPoint | null>(null);
  const [dragRect, setDragRect] = useState<PixelRect | null>(null);
  const [committedSelection, setCommittedSelection] = useState<{
    projection: ProjectionKind;
    rect: PixelRect;
  } | null>(null);
  const committedRect =
    boxSelectionActive === false || committedSelection?.projection !== projection
      ? null
      : committedSelection.rect;

  useEffect(() => {
    onBoxSelectRef.current = onBoxSelect;
  }, [onBoxSelect]);

  useEffect(() => {
    if (!document.fonts) return;
    let disposed = false;
    const redrawWithLoadedFont = () => {
      if (!disposed) setFontRevision((revision) => revision + 1);
    };
    // Unlike SVG text, pixels already painted on Canvas do not update when
    // the local UI font finishes loading.
    void document.fonts.ready.then(redrawWithLoadedFont);
    document.fonts.addEventListener("loadingdone", redrawWithLoadedFont);
    return () => {
      disposed = true;
      document.fonts.removeEventListener("loadingdone", redrawWithLoadedFont);
    };
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const measure = () => {
      const nextWidth = Math.max(1, stage.getBoundingClientRect().width);
      setSize((current) =>
        Math.abs(current.width - nextWidth) < 0.5 && current.height === height
          ? current
          : { width: nextWidth, height },
      );
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [height]);

  const idToIndex = useMemo(() => {
    const lookup = new Map<string, number>();
    points.forEach((point, index) => lookup.set(point.id, index));
    return lookup;
  }, [points]);

  const manualHighlightEntries = useMemo(() => {
    if (!selectedRowIndices && !selectedIds && !manualHighlights?.length) return [];
    const labelByRow = new Map<number, number>();
    for (const highlight of manualHighlights ?? []) {
      labelByRow.set(highlight.rowIndex, highlight.label);
    }
    const entries: Array<{ index: number; label: number | null }> = [];
    points.forEach((point, index) => {
      const hasFeedbackLabel = labelByRow.has(point.rowIndex);
      if (
        hasFeedbackLabel
        || selectionIncludes(selectedRowIndices, point.rowIndex)
        || selectionIncludes(selectedIds, point.id)
      ) {
        entries.push({
          index,
          label: hasFeedbackLabel ? (labelByRow.get(point.rowIndex) ?? 0) : null,
        });
      }
    });
    return entries;
  }, [manualHighlights, points, selectedIds, selectedRowIndices]);

  const candidateCount = useMemo(() => {
    if (!candidateMask) return points.length;
    let count = 0;
    for (const point of points) {
      if (isCandidateRow(candidateMask, point.rowIndex)) count += 1;
    }
    return count;
  }, [candidateMask, points]);
  const highlightCount = useMemo(() => {
    if (highlightMask == null) return null;
    let count = 0;
    for (const point of points) {
      if (isCandidateRow(highlightMask, point.rowIndex)) count += 1;
    }
    return count;
  }, [highlightMask, points]);

  const layout = useMemo<ScatterLayout | null>(() => {
    if (size.width <= 1 || size.height <= 1 || points.length === 0) return null;

    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let validCount = 0;
    const dataX = new Float32Array(points.length);
    const dataY = new Float32Array(points.length);
    dataX.fill(Number.NaN);
    dataY.fill(Number.NaN);
    const groups = new Map<string, number[]>();

    points.forEach((point, index) => {
      const coordinate = coordinatesFor(point, projection);
      if (!coordinate) return;
      dataX[index] = coordinate[0];
      dataY[index] = coordinate[1];
      minX = Math.min(minX, coordinate[0]);
      maxX = Math.max(maxX, coordinate[0]);
      minY = Math.min(minY, coordinate[1]);
      maxY = Math.max(maxY, coordinate[1]);
      validCount += 1;
      const cluster = String(point.cluster);
      const group = groups.get(cluster);
      if (group) group.push(index);
      else groups.set(cluster, [index]);
    });

    if (validCount === 0) return null;

    const left = size.width < 390 ? 36 : 44;
    const right = 14;
    const top = 15;
    const bottom = 31;
    const xScale = scaleLinear<number, number>()
      .domain(paddedDomain(minX, maxX))
      .nice(5)
      .range([left, size.width - right]) as ProjectionScale;
    const yScale = scaleLinear<number, number>()
      .domain(paddedDomain(minY, maxY))
      .nice(5)
      .range([size.height - bottom, top]) as ProjectionScale;
    const screenX = new Float32Array(points.length);
    const screenY = new Float32Array(points.length);
    screenX.fill(Number.NaN);
    screenY.fill(Number.NaN);

    const clusterDomain = [...groups.keys()];
    const ordinal = scaleOrdinal<string, string>()
      .domain(clusterDomain)
      .range(
        clusterDomain.map(
          (_, index) => DEFAULT_CLUSTER_COLORS[index % DEFAULT_CLUSTER_COLORS.length],
        ),
      );
    const colors = new Map<string, string>();
    for (const cluster of clusterDomain) {
      const custom =
        typeof clusterColors === "function"
          ? clusterColors(points[groups.get(cluster)![0]].cluster)
          : clusterColors?.[cluster];
      colors.set(cluster, custom || ordinal(cluster));
    }

    const bucketSize = 14;
    const bucketColumns = Math.max(1, Math.ceil(size.width / bucketSize));
    const buckets = new Map<number, number[]>();
    dataX.forEach((xValue, index) => {
      const yValue = dataY[index];
      if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) return;
      const x = xScale(xValue);
      const y = yScale(yValue);
      screenX[index] = x;
      screenY[index] = y;
      const column = Math.max(0, Math.min(bucketColumns - 1, Math.floor(x / bucketSize)));
      const row = Math.max(0, Math.floor(y / bucketSize));
      const key = row * bucketColumns + column;
      const bucket = buckets.get(key);
      if (bucket) bucket.push(index);
      else buckets.set(key, [index]);
    });

    return {
      width: size.width,
      height: size.height,
      left,
      right,
      top,
      bottom,
      xScale,
      yScale,
      screenX,
      screenY,
      groups,
      colors,
      buckets,
      bucketColumns,
      bucketSize,
      validCount,
    };
  }, [clusterColors, points, projection, size]);

  useEffect(() => {
    const previousMask = previousBoxSelectionMaskRef.current;
    const nextMask = snapshotCandidateMask(boxSelectionMask);
    const selectionScopeChanged = !candidateMasksMatch(previousMask, nextMask, points);
    previousBoxSelectionMaskRef.current = nextMask;

    if (boxSelectionActive === false) {
      selectionActiveRef.current = false;
      selectionBaseMaskRef.current = nextMask;
      lastEmittedRowsRef.current = null;
      queueMicrotask(() => {
        if (!selectionActiveRef.current) {
          setCommittedSelection(null);
          setDragRect(null);
        }
      });
      return;
    }

    if (!selectionActiveRef.current) {
      selectionBaseMaskRef.current = nextMask;
      return;
    }
    if (!selectionScopeChanged) return;

    // A parent may echo the emitted rows through boxSelectionMask. That echo is
    // not a new scope and must not replace the broader pre-selection base.
    if (candidateMaskMatchesRows(nextMask, lastEmittedRowsRef.current, points)) {
      return;
    }

    // A genuine scope change establishes a new redraw base. Re-evaluate the
    // committed rectangle against the complete new common population.
    selectionBaseMaskRef.current = nextMask;
    if (!layout || !committedSelection || committedSelection.projection !== projection) {
      return;
    }
    const nextSelection = selectionForRect(
      committedSelection.rect,
      layout,
      points,
      projection,
      nextMask,
    );
    if (!nextSelection) {
      selectionActiveRef.current = false;
      lastEmittedRowsRef.current = null;
      queueMicrotask(() => {
        if (!selectionActiveRef.current) setCommittedSelection(null);
      });
      onBoxSelectRef.current?.(null);
      return;
    }
    lastEmittedRowsRef.current = new Set(nextSelection.rowIndices);
    onBoxSelectRef.current?.(nextSelection);
  }, [boxSelectionActive, boxSelectionMask, committedSelection, layout, points, projection]);

  useEffect(() => {
    const canvas = baseCanvasRef.current;
    if (!canvas) return;
    if (!layout) {
      prepareCanvas(canvas, size.width, size.height);
      return;
    }
    const context = prepareCanvas(canvas, layout.width, layout.height);
    if (!context) return;

    const xTicks = layout.xScale.ticks(5);
    const yTicks = layout.yScale.ticks(5);
    context.save();
    context.lineWidth = 1;
    context.strokeStyle = "rgba(80, 96, 117, 0.15)";
    context.fillStyle = "rgba(65, 78, 96, 0.72)";
    context.font = chartCanvasFont(canvas, 11);

    for (const tick of xTicks) {
      const x = layout.xScale(tick);
      context.beginPath();
      context.moveTo(x, layout.top);
      context.lineTo(x, layout.height - layout.bottom);
      context.stroke();
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(tick.toPrecision(3), x, layout.height - layout.bottom + 7);
    }
    for (const tick of yTicks) {
      const y = layout.yScale(tick);
      context.beginPath();
      context.moveTo(layout.left, y);
      context.lineTo(layout.width - layout.right, y);
      context.stroke();
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(tick.toPrecision(3), layout.left - 6, y);
    }

    const dense = layout.validCount > 10_000;
    const hasHighlight = highlightMask != null;
    const presentationMask = hasHighlight ? highlightMask : candidateMask;
    const drawMaskPass = (
      includedPass: boolean,
      alpha: number,
      radius: number,
      colorMode: "cluster" | "halo" | "neutral" = "cluster",
    ) => {
      context.globalAlpha = alpha;
      for (const [cluster, indices] of layout.groups) {
        context.fillStyle = colorMode === "halo"
          ? "#ffffff"
          : colorMode === "neutral"
            ? "#aeb5bf"
            : (layout.colors.get(cluster) || DEFAULT_CLUSTER_COLORS[0]);
        if (!dense) context.beginPath();
        for (const index of indices) {
          const included = isCandidateRow(presentationMask, points[index].rowIndex);
          if (included !== includedPass) continue;
          drawSmallPoint(
            context,
            layout.screenX[index],
            layout.screenY[index],
            radius,
            dense,
          );
        }
        if (!dense) context.fill();
      }
    };

    if (hasHighlight) {
      // Keep every embedding point in place. PCP-unselected points become a
      // neutral context layer so dense overlap
      // cannot recreate visual-cluster colors through alpha accumulation.
      drawMaskPass(false, dense ? 0.075 : 0.13, pointRadius, "neutral");
      if (highlightCount !== null && highlightCount > 0 && highlightCount <= 2_000) {
        drawMaskPass(true, 0.86, pointRadius * 1.8, "halo");
      }
      drawMaskPass(true, dense ? 0.86 : 0.94, pointRadius * 1.32, "cluster");
    } else {
      const passes = candidateMask ? [false, true] : [true];
      for (const candidatePass of passes) {
        drawMaskPass(
          candidatePass,
          candidatePass ? (dense ? 0.62 : 0.78) : 0.075,
          pointRadius,
        );
      }
    }
    context.restore();
  }, [candidateMask, fontRevision, highlightCount, highlightMask, layout, pointRadius, points, size]);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    if (!layout) {
      prepareCanvas(canvas, size.width, size.height);
      return;
    }
    const context = prepareCanvas(canvas, layout.width, layout.height);
    if (!context) return;

    const drawRect = (source: PixelRect, active: boolean) => {
      const rect = normalizeRect(source);
      context.save();
      context.fillStyle = active
        ? "rgba(41, 104, 193, 0.13)"
        : "rgba(41, 104, 193, 0.075)";
      context.strokeStyle = active ? "#2968c1" : "rgba(41, 104, 193, 0.62)";
      context.lineWidth = active ? 1.5 : 1;
      context.setLineDash(active ? [] : [4, 3]);
      context.fillRect(rect.x0, rect.y0, rect.x1 - rect.x0, rect.y1 - rect.y0);
      context.strokeRect(rect.x0, rect.y0, rect.x1 - rect.x0, rect.y1 - rect.y0);
      context.restore();
    };
    if (committedRect) drawRect(committedRect, false);
    if (dragRect) drawRect(dragRect, true);

    const drawFocus = (index: number, radius: number, stroke: string) => {
      const x = layout.screenX[index];
      const y = layout.screenY[index];
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const cluster = String(points[index].cluster);
      context.save();
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fillStyle = layout.colors.get(cluster) || DEFAULT_CLUSTER_COLORS[0];
      context.fill();
      context.lineWidth = 2;
      context.strokeStyle = stroke;
      context.stroke();
      context.restore();
    };

    const drawManualSelectionLayer = (
      entries: readonly { index: number; label: number | null }[],
    ) => {
      if (entries.length === 0) return;

      const feedbackStyle = (label: number | null) => {
        if (label === null) {
          return { color: "#ea580c", glow: "rgba(234, 88, 12, 0.72)", strong: false };
        }
        if (label > 0) {
          return { color: "#16a34a", glow: "rgba(22, 163, 74, 0.72)", strong: label >= 2 };
        }
        if (label < 0) {
          return { color: "#dc2626", glow: "rgba(220, 38, 38, 0.72)", strong: label <= -2 };
        }
        return { color: "#ca8a04", glow: "rgba(202, 138, 4, 0.72)", strong: false };
      };

      // White-backed feedback halos stay legible over every cluster color and
      // remain semantically distinct from the blue box-selection brush.
      context.save();
      context.globalAlpha = 1;
      context.fillStyle = "rgba(255, 255, 255, 0.92)";
      for (const { index, label } of entries) {
        const x = layout.screenX[index];
        const y = layout.screenY[index];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        const style = feedbackStyle(label);
        context.shadowColor = style.glow;
        context.shadowBlur = style.strong ? 9 : 7;
        context.strokeStyle = style.color;
        context.lineWidth = style.strong ? 3.5 : 2.5;
        context.beginPath();
        context.arc(x, y, pointRadius + (style.strong ? 6 : 5), 0, Math.PI * 2);
        context.fill();
        context.stroke();
      }
      context.restore();

      // Repaint every selected point after all halos so selected neighbours do
      // not erase one another's cluster-colored center.
      context.save();
      context.globalAlpha = 1;
      for (const { index } of entries) {
        const x = layout.screenX[index];
        const y = layout.screenY[index];
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        const cluster = String(points[index].cluster);
        context.beginPath();
        context.arc(x, y, pointRadius + 1.35, 0, Math.PI * 2);
        context.fillStyle = layout.colors.get(cluster) || DEFAULT_CLUSTER_COLORS[0];
        context.fill();
      }
      context.restore();
    };

    // Draw reviewed points above normal points and the embedding brush. Hover
    // and the legacy single-point focus remain the final transient affordances.
    drawManualSelectionLayer(manualHighlightEntries);

    if (selectedId) {
      const selectedIndex = idToIndex.get(selectedId);
      if (selectedIndex !== undefined) drawFocus(selectedIndex, pointRadius + 3.5, "#17233a");
    }
    if (hoverIndex !== null) drawFocus(hoverIndex, pointRadius + 2.5, "#ffffff");
  }, [
    committedRect,
    dragRect,
    hoverIndex,
    idToIndex,
    layout,
    manualHighlightEntries,
    pointRadius,
    points,
    selectedId,
    size,
  ]);

  const nearestPoint = useCallback(
    (position: PixelPoint): number | null => {
      if (!layout) return null;
      const interactionMask = candidateMask;
      const column = Math.floor(position.x / layout.bucketSize);
      const row = Math.floor(position.y / layout.bucketSize);
      const maximumDistance = Math.max(8, pointRadius * 3.5);
      let bestDistanceSquared = maximumDistance * maximumDistance;
      let bestIndex: number | null = null;
      for (let yOffset = -1; yOffset <= 1; yOffset += 1) {
        for (let xOffset = -1; xOffset <= 1; xOffset += 1) {
          const candidateColumn = column + xOffset;
          const candidateRow = row + yOffset;
          if (candidateColumn < 0 || candidateRow < 0) continue;
          const bucket = layout.buckets.get(
            candidateRow * layout.bucketColumns + candidateColumn,
          );
          if (!bucket) continue;
          for (const index of bucket) {
            if (!isCandidateRow(interactionMask, points[index].rowIndex)) continue;
            const deltaX = layout.screenX[index] - position.x;
            const deltaY = layout.screenY[index] - position.y;
            const distanceSquared = deltaX * deltaX + deltaY * deltaY;
            if (distanceSquared < bestDistanceSquared) {
              bestDistanceSquared = distanceSquared;
              bestIndex = index;
            }
          }
        }
      }
      return bestIndex;
    },
    [candidateMask, layout, pointRadius, points],
  );

  const updateHover = useCallback(
    (nextIndex: number | null, position: PixelPoint | null) => {
      if (hoverIndexRef.current !== nextIndex) {
        hoverIndexRef.current = nextIndex;
        setHoverIndex(nextIndex);
        onPointHover?.(nextIndex === null ? null : points[nextIndex]);
      }
      setTooltipPosition(nextIndex === null ? null : position);
    },
    [onPointHover, points],
  );

  const eventPosition = useCallback(
    (event: PointerEvent<HTMLCanvasElement>): PixelPoint => {
      const bounds = event.currentTarget.getBoundingClientRect();
      return {
        x: ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * size.width,
        y: ((event.clientY - bounds.top) / Math.max(1, bounds.height)) * size.height,
      };
    },
    [size],
  );

  const handlePointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    if (!layout) return;
    if (boxSelectionActive === false) selectionActiveRef.current = false;
    if (!selectionActiveRef.current) {
      selectionBaseMaskRef.current = snapshotCandidateMask(boxSelectionMask);
    }
    const position = eventPosition(event);
    pointerStateRef.current = {
      pointerId: event.pointerId,
      start: position,
      last: position,
      moved: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    updateHover(null, null);
  };

  const handlePointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    const position = eventPosition(event);
    const pointer = pointerStateRef.current;
    if (pointer?.pointerId === event.pointerId) {
      pointer.last = position;
      const deltaX = position.x - pointer.start.x;
      const deltaY = position.y - pointer.start.y;
      if (deltaX * deltaX + deltaY * deltaY >= 16) pointer.moved = true;
      if (pointer.moved) {
        setDragRect({
          x0: pointer.start.x,
          y0: pointer.start.y,
          x1: position.x,
          y1: position.y,
        });
      }
      return;
    }
    updateHover(nearestPoint(position), position);
  };

  const handlePointerUp = (event: PointerEvent<HTMLCanvasElement>) => {
    const pointer = pointerStateRef.current;
    if (!pointer || pointer.pointerId !== event.pointerId || !layout) return;
    const position = eventPosition(event);
    pointer.last = position;
    pointerStateRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    if (pointer.moved) {
      const rect = normalizeRect({
        x0: pointer.start.x,
        y0: pointer.start.y,
        x1: position.x,
        y1: position.y,
      });
      if (rect.x1 - rect.x0 >= 4 && rect.y1 - rect.y0 >= 4) {
        const interactionMask = selectionActiveRef.current
          ? selectionBaseMaskRef.current
          : boxSelectionMask;
        const selection = selectionForRect(
          rect,
          layout,
          points,
          projection,
          interactionMask,
        );
        if (!selection) {
          clearBoxSelection();
        } else {
          selectionActiveRef.current = true;
          lastEmittedRowsRef.current = new Set(selection.rowIndices);
          setCommittedSelection({ projection, rect });
          onBoxSelect?.(selection);
        }
      }
    } else {
      const index = nearestPoint(position);
      onPointClick?.(index === null ? null : points[index]);
      updateHover(index, position);
    }
    setDragRect(null);
  };

  const handlePointerCancel = () => {
    pointerStateRef.current = null;
    setDragRect(null);
    updateHover(null, null);
  };

  const clearBoxSelection = useCallback(() => {
    selectionActiveRef.current = false;
    selectionBaseMaskRef.current = snapshotCandidateMask(boxSelectionMask);
    previousBoxSelectionMaskRef.current = snapshotCandidateMask(boxSelectionMask);
    lastEmittedRowsRef.current = null;
    setCommittedSelection(null);
    setDragRect(null);
    onBoxSelect?.(null);
  }, [boxSelectionMask, onBoxSelect]);

  const handleKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      clearBoxSelection();
    }
  };

  const hoveredPoint = hoverIndex === null ? null : points[hoverIndex];
  const projectionLabel = projection.toUpperCase();

  return (
    <section className={`projection-root ${className}`.trim()} aria-label={title}>
      <div className="projection-toolbar">
        {showHeading && (
        <div className="projection-heading">
          <h3 className="projection-title">{title}</h3>
          <button
            type="button"
            className="projection-reset text-button"
            disabled={!committedRect}
            onClick={clearBoxSelection}
          >
            Reset selection
          </button>
        </div>
        )}
        {controls}
        <div className="projection-meta">
          {showHeading && <span className="projection-help">Drag to select · double-click to clear</span>}
          <div className="projection-count" aria-live="polite">
            {highlightCount === null ? (
              <>
                <strong>{candidateCount.toLocaleString()}</strong>
                <span> / {points.length.toLocaleString()} · {projectionLabel}</span>
              </>
            ) : (
              <>
                <span>PCP selected </span>
                <strong>{highlightCount.toLocaleString()}</strong>
                <span> / {points.length.toLocaleString()} · {projectionLabel}</span>
              </>
            )}
          </div>
          {!showHeading && (
            <button type="button" className="projection-reset text-button"
              disabled={!committedRect} onClick={clearBoxSelection}>
              Reset selection
            </button>
          )}
        </div>
      </div>

      <div
        ref={stageRef}
        className="projection-stage"
        style={{ height, position: "relative", width: "100%", fontFamily: CHART_UI_FONT }}
      >
        <canvas
          ref={baseCanvasRef}
          className="projection-canvas projection-canvas-base"
          aria-hidden="true"
          style={{ height: "100%", inset: 0, position: "absolute", width: "100%" }}
        />
        <canvas
          ref={overlayCanvasRef}
          className="projection-canvas projection-canvas-overlay"
          role="img"
          tabIndex={0}
          aria-label={`${projectionLabel} projection of ${points.length.toLocaleString()} images. Drag a rectangle to select points.`}
          style={{
            height: "100%",
            inset: 0,
            position: "absolute",
            touchAction: "pan-y pinch-zoom",
            width: "100%",
          }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
          onPointerLeave={() => {
            if (!pointerStateRef.current) updateHover(null, null);
          }}
          onDoubleClick={(event) => {
            event.preventDefault();
            clearBoxSelection();
          }}
          onKeyDown={handleKeyDown}
        />

        {!layout && (
          <div className="projection-empty">
            {projection === "umap"
              ? "UMAP coordinates are unavailable for this dataset."
              : "No projection coordinates are available."}
          </div>
        )}

        {hoveredPoint && tooltipPosition && layout && (
          <div
            className="projection-tooltip"
            style={{
              left: Math.max(8, Math.min(layout.width - 176, tooltipPosition.x + 12)),
              pointerEvents: "none",
              position: "absolute",
              top: Math.max(8, Math.min(layout.height - 70, tooltipPosition.y + 12)),
            }}
          >
            <strong>{hoveredPoint.label || hoveredPoint.id}</strong>
            <span>Cluster {String(hoveredPoint.cluster)}</span>
            <span>Row {hoveredPoint.rowIndex.toLocaleString()}</span>
          </div>
        )}
      </div>
    </section>
  );
}

export default ProjectionScatter;
