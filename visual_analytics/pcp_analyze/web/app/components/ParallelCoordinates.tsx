"use client";

import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as d3 from "d3";
import { CHART_UI_FONT } from "../lib/chartTypography";
import { formatPcpAxisTick } from "../lib/pcpBrushZoom";

export type PcpColorMode = "uniform" | "cluster" | "learner";
export type PcpBrushRange = readonly [number, number];
export type PcpBrushMap = Readonly<Record<string, PcpBrushRange | undefined>>;
export type PcpClusterId = string | number;
export type PcpSelectedRowIndices = ReadonlySet<number> | readonly number[];
export interface PcpManualHighlight {
  rowIndex: number;
  /** Positive/negative polarity; absolute values >= 2 are rendered more strongly. */
  label: number;
}
export type PcpClusterColorResolver =
  | Readonly<Record<string, string>>
  | ReadonlyMap<PcpClusterId, string>
  | ((cluster: PcpClusterId) => string | undefined);

export interface ParallelCoordinatesProps {
  /** Stable axis IDs in the same column order as `ranks`. */
  methods: readonly string[];
  /** Optional display labels keyed by stable axis ID. */
  axisLabels?: Readonly<Record<string, string>>;
  /** Hide duplicated SVG labels when an external learner rail names every row. */
  showAxisLabels?: boolean;
  /** Row-major rank matrix: image 0 / every axis, image 1 / every axis, ... */
  ranks: Float32Array;
  /** Cluster label per image. Numeric typed arrays are accepted. */
  labels?: ArrayLike<PcpClusterId>;
  /** Axis IDs which remain active for polylines and brush filtering. */
  enabledMethods: ReadonlySet<string> | readonly string[];
  colorMode?: PcpColorMode;
  /** Optional stable cluster palette shared with scatter plots and galleries. */
  clusterColors?: PcpClusterColorResolver;
  /** Human-readable name for the normalized value domain. */
  valueLabel?: string;
  /** Axis ID whose rank colors a whole polyline when `colorMode="learner"`. */
  colorLearner?: string;
  /** Upstream query/filter mask. A non-zero value means that image is a candidate. */
  candidateMask?: Uint8Array;
  /**
   * Original matrix row indices selected by a human. Visible selected rows are
   * rendered on a dedicated top Canvas layer; contextual rows remain visible.
   */
  selectedRowIndices?: PcpSelectedRowIndices;
  /** Persistent human labels, colored by polarity independently of brush state. */
  manualHighlights?: readonly PcpManualHighlight[];
  /** A clicked/previewed row drawn last with a dark stroke and white halo. */
  focusedRowIndex?: number | null;
  /** Optional controlled brush ranges, expressed in rank-domain values. */
  brushes?: PcpBrushMap;
  /** Receives the exact candidate AND brush mask, plus its selected row indices. */
  onSelectionChange?: (mask: Uint8Array, indices: number[]) => void;
  /** Receives all active axis brush ranges after a user brush interaction. */
  onBrushChange?: (brushes: PcpBrushMap) => void;
  /** Fallback value domain; normalized PCP data uses the default [0, 1]. */
  domain?: PcpBrushRange;
  /** Optional per-axis display domains used by committed brush drill-down levels. */
  axisDomains?: PcpBrushMap;
  /** Number of committed brush drill-down levels. */
  brushZoomDepth?: number;
  /** Commit the current working brushes and zoom the corresponding axes. */
  onBrushZoom?: (brushes: PcpBrushMap) => void;
  /** Restore the previous committed brush drill-down level. */
  onBrushZoomBack?: () => void;
  /** Restore every axis and candidate to the pre-drill-down scope. */
  onBrushZoomReset?: () => void;
  height?: number;
  maxBackgroundLines?: number;
  maxForegroundLines?: number;
  uniformColor?: string;
  selectedLineColor?: string;
  focusedLineColor?: string;
  ariaLabel?: string;
  className?: string;
}

interface Viewport {
  width: number;
  height: number;
}

interface Geometry extends Viewport {
  left: number;
  right: number;
  top: number;
  bottom: number;
  rowGap: number;
  brushHalfHeight: number;
}

type PcpCanvasLayer = "context" | "muted-context" | "brush" | "muted-brush" | "selected";

const CLUSTER_PALETTE = [
  "#4f67d2",
  "#e56b5d",
  "#42a887",
  "#d29b38",
  "#8a64c7",
  "#36a0b8",
  "#d35f9a",
  "#728148",
  "#ba7445",
  "#61758f",
  "#9b71bd",
  "#58a866",
] as const;

const DEFAULT_DOMAIN: PcpBrushRange = [0, 1];
const MANUAL_HIGHLIGHT_STYLES = [
  {
    id: "strong-positive",
    accepts: (label: number) => label >= 2,
    cssProperty: "--pcp-strong-positive-line-color",
    fallback: "#15803d",
    lineWidth: 2.9,
  },
  {
    id: "positive",
    accepts: (label: number) => label > 0,
    cssProperty: "--pcp-positive-line-color",
    fallback: "#16a34a",
    lineWidth: 2.15,
  },
  {
    id: "uncertain",
    accepts: (label: number) => label === 0,
    cssProperty: "--pcp-uncertain-line-color",
    fallback: "#ca8a04",
    lineWidth: 2.15,
  },
  {
    id: "strong-negative",
    accepts: (label: number) => label <= -2,
    cssProperty: "--pcp-strong-negative-line-color",
    fallback: "#b91c1c",
    lineWidth: 2.9,
  },
  {
    id: "negative",
    accepts: (label: number) => label < 0,
    cssProperty: "--pcp-negative-line-color",
    fallback: "#dc2626",
    lineWidth: 2.15,
  },
] as const;
const LEARNER_COLOR = d3.interpolateRgbBasis([
  "#3155a4",
  "#2f9c95",
  "#f1c453",
  "#dc534b",
]);

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function normalizeDomain(domain: PcpBrushRange): [number, number] {
  const first = Number.isFinite(domain[0]) ? domain[0] : 0;
  const second = Number.isFinite(domain[1]) ? domain[1] : 1;
  if (first === second) return [first, first + 1];
  return first < second ? [first, second] : [second, first];
}

function normalizeBrushRange(
  range: PcpBrushRange,
  domain: [number, number],
): [number, number] | null {
  if (!Number.isFinite(range[0]) || !Number.isFinite(range[1])) return null;
  const lower = clamp(Math.min(range[0], range[1]), domain[0], domain[1]);
  const upper = clamp(Math.max(range[0], range[1]), domain[0], domain[1]);
  return upper - lower > Number.EPSILON ? [lower, upper] : null;
}

function rangesEqual(
  first: PcpBrushRange | undefined,
  second: PcpBrushRange | undefined,
) {
  if (!first || !second) return first === second;
  return (
    Math.abs(first[0] - second[0]) < 1e-7 &&
    Math.abs(first[1] - second[1]) < 1e-7
  );
}

function sanitizeBrushes(
  brushes: PcpBrushMap | undefined,
  methodSet: ReadonlySet<string>,
  domains: ReadonlyMap<string, [number, number]>,
  fallbackDomain: [number, number],
): Record<string, [number, number]> {
  const next: Record<string, [number, number]> = {};
  if (!brushes) return next;

  for (const [method, range] of Object.entries(brushes)) {
    if (!methodSet.has(method) || !range) continue;
    const normalized = normalizeBrushRange(range, domains.get(method) ?? fallbackDomain);
    if (normalized) next[method] = normalized;
  }
  return next;
}

function cloneActiveBrushes(
  brushes: Readonly<Record<string, PcpBrushRange | undefined>>,
  enabledSet: ReadonlySet<string>,
): Record<string, [number, number]> {
  const copy: Record<string, [number, number]> = {};
  for (const [method, range] of Object.entries(brushes)) {
    if (enabledSet.has(method) && range) {
      copy[method] = [range[0], range[1]];
    }
  }
  return copy;
}

function sampleEvenly(indices: readonly number[], maximum: number): readonly number[] {
  if (maximum <= 0 || indices.length <= maximum) return indices;
  const sampled = new Array<number>(maximum);
  const step = indices.length / maximum;
  for (let index = 0; index < maximum; index += 1) {
    sampled[index] = indices[Math.floor(index * step)];
  }
  return sampled;
}

function truncateLabel(label: string, maximum = 23) {
  return label.length <= maximum ? label : `${label.slice(0, maximum - 1)}\u2026`;
}

function cssColor(element: HTMLElement | null, property: string, fallback: string) {
  if (!element || typeof window === "undefined") return fallback;
  return window.getComputedStyle(element).getPropertyValue(property).trim() || fallback;
}

function resolveClusterColor(
  cluster: PcpClusterId,
  index: number,
  colors: PcpClusterColorResolver | undefined,
) {
  let customColor: string | undefined;
  if (typeof colors === "function") {
    customColor = colors(cluster);
  } else if (
    colors &&
    typeof (colors as ReadonlyMap<PcpClusterId, string>).get === "function"
  ) {
    const colorMap = colors as ReadonlyMap<PcpClusterId, string>;
    customColor = colorMap.get(cluster) ?? colorMap.get(String(cluster));
  } else {
    customColor = (colors as Readonly<Record<string, string>> | undefined)?.[
      String(cluster)
    ];
  }
  return (
    customColor ||
    CLUSTER_PALETTE[index] ||
    `hsl(${(index * 137.508) % 360} 55% 52%)`
  );
}

/**
 * Hybrid parallel-coordinates renderer: Canvas batches the high-volume paths,
 * while D3 owns axes and one horizontal brush per active axis ID in an SVG layer.
 */
export function ParallelCoordinates({
  methods,
  axisLabels,
  showAxisLabels = true,
  ranks,
  labels,
  enabledMethods,
  colorMode = "uniform",
  clusterColors,
  valueLabel = "rank",
  colorLearner,
  candidateMask,
  selectedRowIndices,
  manualHighlights,
  focusedRowIndex = null,
  brushes,
  onSelectionChange,
  onBrushChange,
  domain: domainProp = DEFAULT_DOMAIN,
  axisDomains,
  brushZoomDepth = 0,
  onBrushZoom,
  onBrushZoomBack,
  onBrushZoomReset,
  height,
  maxBackgroundLines = 14_000,
  maxForegroundLines = 18_000,
  uniformColor = "#52647a",
  selectedLineColor = "#ea580c",
  focusedLineColor = "#1f2937",
  ariaLabel = "Parallel coordinates of image ranks by axis",
  className = "",
}: ParallelCoordinatesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const backgroundCanvasRef = useRef<HTMLCanvasElement>(null);
  const foregroundCanvasRef = useRef<HTMLCanvasElement>(null);
  const selectedCanvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const brushBehaviorsRef = useRef<Map<string, d3.BrushBehavior<unknown>>>(new Map());
  const brushGroupsRef = useRef<Map<string, SVGGElement>>(new Map());
  const suppressBrushEventsRef = useRef(false);
  const currentBrushesRef = useRef<Record<string, [number, number]>>({});
  const controlledBrushesRef = useRef(brushes);
  const selectionMaskRef = useRef<Uint8Array>(new Uint8Array());
  const selectionIndicesRef = useRef<number[]>([]);
  const selectionHasBrushRef = useRef(false);
  const evaluationFrameRef = useRef<number | null>(null);
  const backgroundFrameRef = useRef<number | null>(null);
  const foregroundFrameRef = useRef<number | null>(null);
  const selectedFrameRef = useRef<number | null>(null);
  const notifyBrushRef = useRef(false);
  const scheduleEvaluationRef = useRef<(notifyBrush?: boolean) => void>(() => undefined);
  const onSelectionChangeRef = useRef(onSelectionChange);
  const onBrushChangeRef = useRef(onBrushChange);
  const statusId = useId();

  useLayoutEffect(() => {
    controlledBrushesRef.current = brushes;
  }, [brushes]);

  useEffect(() => {
    onSelectionChangeRef.current = onSelectionChange;
  }, [onSelectionChange]);

  useEffect(() => {
    onBrushChangeRef.current = onBrushChange;
  }, [onBrushChange]);

  const chartHeight = height ?? Math.max(520, methods.length * 43 + 76);
  const [viewport, setViewport] = useState<Viewport>({
    width: 720,
    height: chartHeight,
  });

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const updateSize = () => {
      // The evidence rail can leave less than 280px for the plot on narrow panels.
      const width = Math.max(1, Math.round(container.getBoundingClientRect().width || 720));
      setViewport((current) =>
        current.width === width && current.height === chartHeight
          ? current
          : { width, height: chartHeight },
      );
    };

    updateSize();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateSize);
      return () => window.removeEventListener("resize", updateSize);
    }
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, [chartHeight]);

  const [domainMinimum, domainMaximum] = normalizeDomain(domainProp);
  const domain = useMemo<[number, number]>(
    () => [domainMinimum, domainMaximum],
    [domainMaximum, domainMinimum],
  );
  const methodSet = useMemo(() => new Set(methods), [methods]);
  const domainsByMethod = useMemo(() => {
    const resolved = new Map<string, [number, number]>();
    for (const method of methods) {
      const requested = axisDomains?.[method];
      const clipped = requested ? normalizeBrushRange(requested, domain) : null;
      resolved.set(method, clipped ?? domain);
    }
    return resolved;
  }, [axisDomains, domain, methods]);
  const zoomedMethodSet = useMemo(() => {
    const zoomed = new Set<string>();
    for (const method of methods) {
      const resolved = domainsByMethod.get(method);
      if (resolved && !rangesEqual(resolved, domain)) zoomed.add(method);
    }
    return zoomed;
  }, [domain, domainsByMethod, methods]);
  const enabledSet = useMemo(() => new Set<string>(enabledMethods), [enabledMethods]);
  const methodIndices = useMemo(
    () => new Map(methods.map((method, index) => [method, index])),
    [methods],
  );
  const activeMethodIndices = useMemo(
    () => methods.flatMap((method, index) => (enabledSet.has(method) ? [index] : [])),
    [enabledSet, methods],
  );
  const rowCount = methods.length > 0 ? Math.floor(ranks.length / methods.length) : 0;

  const candidateIndices = useMemo(() => {
    const indices: number[] = [];
    for (let image = 0; image < rowCount; image += 1) {
      if (!candidateMask || candidateMask[image]) indices.push(image);
    }
    return indices;
  }, [candidateMask, rowCount]);

  const visibleManualHighlights = useMemo(() => {
    const resolved = new Map<number, number>();
    for (const highlight of manualHighlights ?? []) {
      const rowIndex = highlight.rowIndex;
      if (
        !Number.isInteger(rowIndex)
        || rowIndex < 0
        || rowIndex >= rowCount
        || (candidateMask && !candidateMask[rowIndex])
      ) continue;
      resolved.set(rowIndex, Number.isFinite(highlight.label) ? highlight.label : 0);
    }
    return [...resolved].map(([rowIndex, label]) => ({ rowIndex, label }));
  }, [candidateMask, manualHighlights, rowCount]);
  const visibleManualRowSet = useMemo(
    () => new Set(visibleManualHighlights.map((highlight) => highlight.rowIndex)),
    [visibleManualHighlights],
  );
  const visibleNeutralSelectedRowIndices = useMemo(() => {
    if (!selectedRowIndices) return [];
    const requested = new Set<number>(selectedRowIndices);
    const indices: number[] = [];
    for (let image = 0; image < rowCount; image += 1) {
      if (
        requested.has(image)
        && !visibleManualRowSet.has(image)
        && (!candidateMask || candidateMask[image])
      ) {
        indices.push(image);
      }
    }
    return indices;
  }, [candidateMask, rowCount, selectedRowIndices, visibleManualRowSet]);
  const visibleFocusedRowIndex = focusedRowIndex !== null
    && Number.isInteger(focusedRowIndex)
    && focusedRowIndex >= 0
    && focusedRowIndex < rowCount
    && (!candidateMask || candidateMask[focusedRowIndex])
    ? focusedRowIndex
    : null;
  const visibleHighlightRowIndices = useMemo(() => {
    const indices = new Set(visibleNeutralSelectedRowIndices);
    for (const highlight of visibleManualHighlights) indices.add(highlight.rowIndex);
    if (visibleFocusedRowIndex !== null) indices.add(visibleFocusedRowIndex);
    return [...indices].sort((left, right) => left - right);
  }, [visibleFocusedRowIndex, visibleManualHighlights, visibleNeutralSelectedRowIndices]);
  const hasVisibleManualSelection = visibleHighlightRowIndices.length > 0;

  const labelColors = useMemo(() => {
    const colors = new Map<string, string>();
    for (let image = 0; image < rowCount; image += 1) {
      const cluster = labels?.[image] ?? "unlabelled";
      const label = String(cluster);
      if (!colors.has(label)) {
        const index = colors.size;
        colors.set(label, resolveClusterColor(cluster, index, clusterColors));
      }
    }
    return colors;
  }, [clusterColors, labels, rowCount]);

  const geometry = useMemo<Geometry>(() => {
    const { width, height: currentHeight } = viewport;
    const left = showAxisLabels
      ? Math.min(210, Math.max(72, width * 0.24))
      : 18;
    const right = Math.max(left + 80, width - 18);
    const top = 54;
    const bottom = Math.max(top + 20, currentHeight - 34);
    const rowGap = methods.length > 1 ? (bottom - top) / (methods.length - 1) : 0;
    return {
      width,
      height: currentHeight,
      left,
      right,
      top,
      bottom,
      rowGap,
      brushHalfHeight: clamp(rowGap * 0.28 || 10, 7, 13),
    };
  }, [methods.length, showAxisLabels, viewport]);

  const colorLearnerIndex = colorLearner ? methodIndices.get(colorLearner) : undefined;
  const colorLearnerDomain = colorLearner
    ? domainsByMethod.get(colorLearner) ?? domain
    : domain;

  const prepareCanvas = useCallback(
    (canvas: HTMLCanvasElement) => {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const pixelWidth = Math.max(1, Math.round(geometry.width * ratio));
      const pixelHeight = Math.max(1, Math.round(geometry.height * ratio));
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      canvas.style.width = `${geometry.width}px`;
      canvas.style.height = `${geometry.height}px`;
      const context = canvas.getContext("2d", { alpha: true });
      if (!context) return null;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, geometry.width, geometry.height);
      return context;
    },
    [geometry],
  );

  useLayoutEffect(() => {
    // Canvas bitmap dimensions are independent from their CSS dimensions.
    // Synchronize all Canvas layers before paint whenever geometry changes, including
    // an empty foreground layer which would otherwise retain its previous size.
    if (backgroundFrameRef.current !== null) {
      window.cancelAnimationFrame(backgroundFrameRef.current);
      backgroundFrameRef.current = null;
    }
    if (foregroundFrameRef.current !== null) {
      window.cancelAnimationFrame(foregroundFrameRef.current);
      foregroundFrameRef.current = null;
    }
    if (selectedFrameRef.current !== null) {
      window.cancelAnimationFrame(selectedFrameRef.current);
      selectedFrameRef.current = null;
    }
    const background = backgroundCanvasRef.current;
    const foreground = foregroundCanvasRef.current;
    const selected = selectedCanvasRef.current;
    if (background) prepareCanvas(background);
    if (foreground) prepareCanvas(foreground);
    if (selected) prepareCanvas(selected);
  }, [prepareCanvas]);

  const drawLayer = useCallback(
    (
      canvas: HTMLCanvasElement,
      sourceIndices: readonly number[],
      maximum: number,
      layer: PcpCanvasLayer,
    ) => {
      const context = prepareCanvas(canvas);
      if (!context || sourceIndices.length === 0 || activeMethodIndices.length === 0) return;

      const indices = sampleEvenly(sourceIndices, maximum);
      const scaleXs = methods.map((method) => d3
        .scaleLinear()
        .domain(domainsByMethod.get(method) ?? domain)
        .range([geometry.left, geometry.right])
        .clamp(true));
      const lineOpacity = layer === "selected"
        ? indices.length > 8_000
          ? 0.22
          : indices.length > 2_000
            ? 0.34
            : indices.length > 500
              ? 0.56
              : 0.94
        : layer === "brush"
          ? indices.length > 8_000
            ? 0.075
            : indices.length > 2_000
              ? 0.12
              : 0.2
          : layer === "muted-brush"
            ? 0.055
            : layer === "muted-context"
              ? 0.024
              : 0.045;
      const lineWidth = layer === "selected"
        ? 2.15
        : layer === "brush"
          ? 1.05
          : layer === "muted-brush"
            ? 0.8
            : 0.65;
      const fallbackLine = cssColor(containerRef.current, "--pcp-line-color", uniformColor);
      const highlightLine = cssColor(
        containerRef.current,
        "--pcp-selected-line-color",
        selectedLineColor,
      );
      const highlightHalo = cssColor(
        containerRef.current,
        "--pcp-selected-line-halo",
        "#ffffff",
      );
      const focusLine = cssColor(
        containerRef.current,
        "--pcp-focused-line-color",
        focusedLineColor,
      );
      const methodCount = methods.length;

      const strokeBatch = (
        batch: readonly number[],
        color: string,
        style?: { lineWidth?: number; opacity?: number; haloExtraWidth?: number },
      ) => {
        if (batch.length === 0) return;
        const batchLineWidth = style?.lineWidth ?? lineWidth;
        const batchOpacity = style?.opacity ?? lineOpacity;
        context.beginPath();
        for (const image of batch) {
          let started = false;
          let points = 0;
          let onlyX = 0;
          let onlyY = 0;
          for (const methodIndex of activeMethodIndices) {
            const value = ranks[image * methodCount + methodIndex];
            if (!Number.isFinite(value)) {
              started = false;
              points = 0;
              continue;
            }
            const x = scaleXs[methodIndex](value);
            const y =
              methods.length > 1
                ? geometry.top + methodIndex * geometry.rowGap
                : (geometry.top + geometry.bottom) / 2;
            if (!started) context.moveTo(x, y);
            else context.lineTo(x, y);
            started = true;
            points += 1;
            onlyX = x;
            onlyY = y;
          }
          if (points === 1) context.lineTo(onlyX + 0.01, onlyY);
        }
        context.lineJoin = "round";
        context.lineCap = "round";
        if (layer === "selected") {
          context.globalAlpha = Math.min(0.72, batchOpacity * 0.72);
          context.lineWidth = batchLineWidth + (style?.haloExtraWidth ?? 3.2);
          context.strokeStyle = highlightHalo;
          context.stroke();
        }
        context.globalAlpha = batchOpacity;
        context.lineWidth = batchLineWidth;
        context.strokeStyle = color;
        context.stroke();
      };

      if (layer === "selected") {
        const sampledRows = new Set(indices);
        const annotatedRows = new Set<number>();
        for (const style of MANUAL_HIGHLIGHT_STYLES) {
          const rows = visibleManualHighlights.flatMap((highlight) => {
            if (
              annotatedRows.has(highlight.rowIndex)
              || !sampledRows.has(highlight.rowIndex)
              || !style.accepts(highlight.label)
            ) return [];
            annotatedRows.add(highlight.rowIndex);
            return [highlight.rowIndex];
          });
          strokeBatch(
            rows,
            cssColor(containerRef.current, style.cssProperty, style.fallback),
            { lineWidth: style.lineWidth },
          );
        }
        strokeBatch(
          visibleNeutralSelectedRowIndices.filter((rowIndex) => (
            sampledRows.has(rowIndex) && !annotatedRows.has(rowIndex)
          )),
          highlightLine,
        );
        if (visibleFocusedRowIndex !== null && sampledRows.has(visibleFocusedRowIndex)) {
          strokeBatch(
            [visibleFocusedRowIndex],
            focusLine,
            { lineWidth: 3.15, opacity: 1, haloExtraWidth: 4.2 },
          );
        }
      } else if (colorMode === "cluster" && labels) {
        const groups = new Map<string, number[]>();
        for (const image of indices) {
          const label = String(labels[image] ?? "unlabelled");
          const group = groups.get(label);
          if (group) group.push(image);
          else groups.set(label, [image]);
        }
        for (const [label, group] of groups) {
          strokeBatch(group, labelColors.get(label) ?? fallbackLine);
        }
      } else if (colorMode === "learner" && colorLearnerIndex !== undefined) {
        const bins = Array.from({ length: 16 }, () => [] as number[]);
        const span = colorLearnerDomain[1] - colorLearnerDomain[0];
        for (const image of indices) {
          const value = ranks[image * methodCount + colorLearnerIndex];
          const normalized = Number.isFinite(value)
            ? (value - colorLearnerDomain[0]) / span
            : 0;
          const bin = clamp(Math.floor(normalized * bins.length), 0, bins.length - 1);
          bins[bin].push(image);
        }
        bins.forEach((bin, index) => {
          strokeBatch(bin, LEARNER_COLOR((index + 0.5) / bins.length));
        });
      } else {
        strokeBatch(indices, fallbackLine);
      }

      context.globalAlpha = 1;
    },
    [
      activeMethodIndices,
      colorLearnerDomain,
      colorLearnerIndex,
      colorMode,
      domain,
      domainsByMethod,
      geometry,
      focusedLineColor,
      labelColors,
      labels,
      methods,
      prepareCanvas,
      ranks,
      selectedLineColor,
      uniformColor,
      visibleFocusedRowIndex,
      visibleManualHighlights,
      visibleNeutralSelectedRowIndices,
    ],
  );

  const scheduleBackgroundDraw = useCallback(() => {
    // A mode/height change can happen while a draw for the previous geometry is
    // still queued.  Replacing that frame is important: otherwise the stale
    // callback can resize the canvas back to the old (usually 710px) height and
    // the new, taller SVG axes continue below an apparently clipped canvas.
    if (backgroundFrameRef.current !== null) {
      window.cancelAnimationFrame(backgroundFrameRef.current);
    }
    backgroundFrameRef.current = window.requestAnimationFrame(() => {
      backgroundFrameRef.current = null;
      const canvas = backgroundCanvasRef.current;
      if (canvas) {
        drawLayer(
          canvas,
          candidateIndices,
          maxBackgroundLines,
          hasVisibleManualSelection ? "muted-context" : "context",
        );
      }
    });
  }, [candidateIndices, drawLayer, hasVisibleManualSelection, maxBackgroundLines]);

  const scheduleForegroundDraw = useCallback(() => {
    if (foregroundFrameRef.current !== null) {
      window.cancelAnimationFrame(foregroundFrameRef.current);
    }
    foregroundFrameRef.current = window.requestAnimationFrame(() => {
      foregroundFrameRef.current = null;
      const canvas = foregroundCanvasRef.current;
      if (!canvas) return;
      drawLayer(
        canvas,
        selectionHasBrushRef.current ? selectionIndicesRef.current : [],
        maxForegroundLines,
        hasVisibleManualSelection ? "muted-brush" : "brush",
      );
    });
  }, [drawLayer, hasVisibleManualSelection, maxForegroundLines]);

  const scheduleSelectedDraw = useCallback(() => {
    if (selectedFrameRef.current !== null) {
      window.cancelAnimationFrame(selectedFrameRef.current);
    }
    selectedFrameRef.current = window.requestAnimationFrame(() => {
      selectedFrameRef.current = null;
      const canvas = selectedCanvasRef.current;
      if (!canvas) return;
      drawLayer(canvas, visibleHighlightRowIndices, maxForegroundLines, "selected");
    });
  }, [drawLayer, maxForegroundLines, visibleHighlightRowIndices]);

  const [selectedCount, setSelectedCount] = useState(candidateIndices.length);
  const [activeBrushCount, setActiveBrushCount] = useState(0);

  const evaluateSelection = useCallback(() => {
    const activeBrushes = cloneActiveBrushes(currentBrushesRef.current, enabledSet);
    const activeEntries = Object.entries(activeBrushes).flatMap(([method, range]) => {
      const methodIndex = methodIndices.get(method);
      return methodIndex === undefined ? [] : [{ methodIndex, range }];
    });
    const mask = new Uint8Array(rowCount);
    const selected: number[] = [];
    const methodCount = methods.length;

    candidateLoop: for (const image of candidateIndices) {
      for (const { methodIndex, range } of activeEntries) {
        const value = ranks[image * methodCount + methodIndex];
        if (!Number.isFinite(value) || value < range[0] || value > range[1]) {
          continue candidateLoop;
        }
      }
      mask[image] = 1;
      selected.push(image);
    }

    selectionMaskRef.current = mask;
    selectionIndicesRef.current = selected;
    selectionHasBrushRef.current = activeEntries.length > 0;
    setSelectedCount(selected.length);
    setActiveBrushCount(activeEntries.length);
    onSelectionChangeRef.current?.(mask, selected);
    scheduleForegroundDraw();

    if (notifyBrushRef.current) {
      notifyBrushRef.current = false;
      onBrushChangeRef.current?.(activeBrushes);
    }
  }, [candidateIndices, enabledSet, methodIndices, methods.length, ranks, rowCount, scheduleForegroundDraw]);

  const scheduleEvaluation = useCallback(
    (notifyBrush = false) => {
      notifyBrushRef.current ||= notifyBrush;
      if (evaluationFrameRef.current !== null) return;
      evaluationFrameRef.current = window.requestAnimationFrame(() => {
        evaluationFrameRef.current = null;
        evaluateSelection();
      });
    },
    [evaluateSelection],
  );

  useLayoutEffect(() => {
    scheduleEvaluationRef.current = scheduleEvaluation;
  }, [scheduleEvaluation]);

  useEffect(() => {
    scheduleBackgroundDraw();
  }, [scheduleBackgroundDraw]);

  useEffect(() => {
    scheduleForegroundDraw();
    scheduleSelectedDraw();
  }, [scheduleForegroundDraw, scheduleSelectedDraw]);

  useEffect(() => {
    scheduleEvaluation(false);
  }, [scheduleEvaluation]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    const root = d3.select(svg);
    root.selectAll("*").remove();
    brushBehaviorsRef.current.clear();
    brushGroupsRef.current.clear();

    root.append("title").text(ariaLabel);
    const incoming = sanitizeBrushes(
      controlledBrushesRef.current ?? currentBrushesRef.current,
      methodSet,
      domainsByMethod,
      domain,
    );
    currentBrushesRef.current = incoming;

    methods.forEach((method, methodIndex) => {
      const axisLabel = axisLabels?.[method] ?? method;
      const enabled = enabledSet.has(method);
      const axisDomain = domainsByMethod.get(method) ?? domain;
      const scaleX = d3
        .scaleLinear()
        .domain(axisDomain)
        .range([geometry.left, geometry.right])
        .clamp(true);
      const tickValues = [
        axisDomain[0],
        (axisDomain[0] + axisDomain[1]) / 2,
        axisDomain[1],
      ];
      const y =
        methods.length > 1
          ? geometry.top + methodIndex * geometry.rowGap
          : (geometry.top + geometry.bottom) / 2;
      const row = root
        .append("g")
        .attr("class", `pcp-method${enabled ? "" : " pcp-method--disabled"}`)
        .attr("transform", `translate(0,${y})`)
        .attr("role", "group")
        .attr(
          "aria-label",
          enabled
            ? `${axisLabel}: active axis with ${valueLabel} brush`
            : `${axisLabel}: disabled axis`,
        );

      if (enabled) {
        const brush = d3
          .brushX<unknown>()
          .extent([
            [geometry.left, -geometry.brushHalfHeight],
            [geometry.right, geometry.brushHalfHeight],
          ])
          .on("brush end", (event: d3.D3BrushEvent<unknown>) => {
            if (suppressBrushEventsRef.current) return;
            const pixelRange = event.selection as [number, number] | null;
            if (!pixelRange) {
              delete currentBrushesRef.current[method];
            } else {
              const next = normalizeBrushRange(
                [scaleX.invert(pixelRange[0]), scaleX.invert(pixelRange[1])],
                axisDomain,
              );
              if (next) currentBrushesRef.current[method] = next;
              else delete currentBrushesRef.current[method];
            }
            scheduleEvaluationRef.current(true);
          });

        const brushGroup = row
          .append("g")
          .attr("class", "pcp-brush")
          .attr("role", "group")
          .attr("aria-label", `Drag horizontally to filter ${axisLabel} ${valueLabel}`);
        brushGroup.call(brush);
        brushGroup
          .selectAll<SVGRectElement, unknown>(".overlay")
          .attr("fill", "transparent")
          .attr("cursor", "crosshair")
          .attr(
            "aria-label",
            `${axisLabel} ${valueLabel} brush from ${axisDomain[0]} to ${axisDomain[1]}`,
          );
        brushGroup
          .selectAll<SVGRectElement, unknown>(".selection")
          .attr("fill", "rgba(71, 99, 212, 0.2)")
          .attr("stroke", "#4f67d2")
          .attr("rx", 2);
        brushGroup
          .selectAll<SVGRectElement, unknown>(".handle")
          .attr("fill", "#4f67d2")
          .attr("rx", 1);
        const node = brushGroup.node();
        if (node) {
          brushBehaviorsRef.current.set(method, brush);
          brushGroupsRef.current.set(method, node);
        }
      }

      const axis = d3
        .axisBottom(scaleX)
        .tickValues(tickValues)
        .tickSize(4)
        .tickSizeOuter(0)
        .tickFormat((value) =>
          methodIndex === methods.length - 1 || zoomedMethodSet.has(method)
            ? formatPcpAxisTick(Number(value), axisDomain)
            : "",
        );
      row
        .append("g")
        .attr("class", "pcp-axis")
        .attr("aria-hidden", "true")
        .style("pointer-events", "none")
        .call(axis)
        .style("font-family", CHART_UI_FONT)
        .style("font-size", "11px");
      if (showAxisLabels) {
        const label = row
          .append("text")
          .attr("class", "pcp-method-label")
          .attr("x", geometry.left - 12)
          .attr("y", 0)
          .attr("dy", "0.34em")
          .attr("text-anchor", "end")
          .attr("aria-hidden", "true")
          .style("font-family", CHART_UI_FONT)
          .style("font-size", "11px")
          .text(truncateLabel(axisLabel));
        label.append("title").text(axisLabel);
      }
    });

    suppressBrushEventsRef.current = true;
    try {
      for (const [method, brush] of brushBehaviorsRef.current) {
        const group = brushGroupsRef.current.get(method);
        const range = currentBrushesRef.current[method];
        const axisDomain = domainsByMethod.get(method) ?? domain;
        const scaleX = d3
          .scaleLinear()
          .domain(axisDomain)
          .range([geometry.left, geometry.right])
          .clamp(true);
        if (group) {
          d3.select(group).call(
            brush.move,
            range ? [scaleX(range[0]), scaleX(range[1])] : null,
          );
        }
      }
    } finally {
      suppressBrushEventsRef.current = false;
    }
    scheduleEvaluationRef.current(false);
    const brushBehaviors = brushBehaviorsRef.current;
    const brushGroups = brushGroupsRef.current;

    return () => {
      root.selectAll("*").remove();
      brushBehaviors.clear();
      brushGroups.clear();
    };
  }, [
    ariaLabel,
    axisLabels,
    domain,
    domainsByMethod,
    enabledSet,
    geometry,
    methodSet,
    methods,
    showAxisLabels,
    valueLabel,
    zoomedMethodSet,
  ]);

  useEffect(() => {
    if (!brushes) return;
    const next = sanitizeBrushes(brushes, methodSet, domainsByMethod, domain);
    const current = currentBrushesRef.current;
    const keys = new Set([...Object.keys(current), ...Object.keys(next)]);
    let changed = false;
    for (const key of keys) {
      if (!rangesEqual(current[key], next[key])) {
        changed = true;
        break;
      }
    }
    if (!changed) return;

    currentBrushesRef.current = next;
    suppressBrushEventsRef.current = true;
    try {
      for (const [method, brush] of brushBehaviorsRef.current) {
        const group = brushGroupsRef.current.get(method);
        const range = next[method];
        const axisDomain = domainsByMethod.get(method) ?? domain;
        const scaleX = d3
          .scaleLinear()
          .domain(axisDomain)
          .range([geometry.left, geometry.right]);
        if (group) {
          d3.select(group).call(
            brush.move,
            range ? [scaleX(range[0]), scaleX(range[1])] : null,
          );
        }
      }
    } finally {
      suppressBrushEventsRef.current = false;
    }
    scheduleEvaluation(false);
  }, [
    brushes,
    domain,
    domainsByMethod,
    geometry.left,
    geometry.right,
    methodSet,
    scheduleEvaluation,
  ]);

  const clearBrushes = useCallback(() => {
    currentBrushesRef.current = {};
    suppressBrushEventsRef.current = true;
    try {
      for (const [method, brush] of brushBehaviorsRef.current) {
        const group = brushGroupsRef.current.get(method);
        if (group) d3.select(group).call(brush.move, null);
      }
    } finally {
      suppressBrushEventsRef.current = false;
    }
    scheduleEvaluation(true);
  }, [scheduleEvaluation]);

  const zoomToBrush = useCallback(() => {
    if (!onBrushZoom) return;
    const activeBrushes = cloneActiveBrushes(currentBrushesRef.current, enabledSet);
    if (Object.keys(activeBrushes).length === 0) return;
    onBrushZoom(activeBrushes);
  }, [enabledSet, onBrushZoom]);

  useEffect(
    () => () => {
      if (evaluationFrameRef.current !== null) {
        cancelAnimationFrame(evaluationFrameRef.current);
        evaluationFrameRef.current = null;
      }
      if (backgroundFrameRef.current !== null) {
        cancelAnimationFrame(backgroundFrameRef.current);
        backgroundFrameRef.current = null;
      }
      if (foregroundFrameRef.current !== null) {
        cancelAnimationFrame(foregroundFrameRef.current);
        foregroundFrameRef.current = null;
      }
      if (selectedFrameRef.current !== null) {
        cancelAnimationFrame(selectedFrameRef.current);
        selectedFrameRef.current = null;
      }
    },
    [],
  );

  const highlightStatus = hasVisibleManualSelection
    ? ` · ${visibleHighlightRowIndices.length.toLocaleString()} human-selected highlighted`
    : "";
  const statusText = `${selectedCount.toLocaleString()} / ${candidateIndices.length.toLocaleString()} selected${highlightStatus}`;
  const rootClassName = `pcp-root${className ? ` ${className}` : ""}`;

  return (
    <div
      ref={containerRef}
      className={rootClassName}
      role="group"
      aria-label={ariaLabel}
      aria-describedby={statusId}
      style={{
        position: "relative",
        width: "100%",
        height: chartHeight,
        minHeight: chartHeight,
      }}
    >
      <div
        className="pcp-toolbar"
        style={{ position: "absolute", top: 0, right: 0, zIndex: 2 }}
      >
        <span id={statusId} className="pcp-selection-status" aria-live="polite">
          {statusText}
        </span>
        {brushZoomDepth > 0 && (
          <span className="pcp-zoom-status">Zoom {brushZoomDepth}</span>
        )}
        <button
          type="button"
          className="pcp-brush-action pcp-clear-brushes"
          onClick={clearBrushes}
          disabled={activeBrushCount === 0}
          aria-label="Clear all parallel-coordinate brushes"
          title="Clear only the working brushes; keep the committed zoom scope"
        >
          Clear
        </button>
        {onBrushZoom && (
          <button
            type="button"
            className="pcp-brush-action pcp-zoom-brush"
            onClick={zoomToBrush}
            disabled={activeBrushCount === 0}
            title="Keep this selection as the parent scope and enlarge its axis ranges"
          >
            Zoom
          </button>
        )}
        {onBrushZoomBack && (
          <button
            type="button"
            className="pcp-brush-action pcp-zoom-back"
            onClick={onBrushZoomBack}
            disabled={brushZoomDepth === 0}
            aria-label="Return to the previous PCP brush zoom level"
          >
            Back
          </button>
        )}
        {onBrushZoomReset && (
          <button
            type="button"
            className="pcp-brush-action pcp-zoom-reset"
            onClick={onBrushZoomReset}
            disabled={brushZoomDepth === 0}
            aria-label="Reset all PCP brush zoom levels"
          >
            Reset
          </button>
        )}
      </div>
      <canvas
        ref={backgroundCanvasRef}
        className="pcp-canvas pcp-canvas--background"
        aria-hidden="true"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: geometry.width,
          height: geometry.height,
          maxWidth: "none",
          maxHeight: "none",
          pointerEvents: "none",
        }}
      />
      <canvas
        ref={foregroundCanvasRef}
        className="pcp-canvas pcp-canvas--foreground"
        aria-hidden="true"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: geometry.width,
          height: geometry.height,
          maxWidth: "none",
          maxHeight: "none",
          pointerEvents: "none",
        }}
      />
      <canvas
        ref={selectedCanvasRef}
        className="pcp-canvas pcp-canvas--selected"
        aria-hidden="true"
        data-highlighted-row-count={visibleHighlightRowIndices.length}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: geometry.width,
          height: geometry.height,
          maxWidth: "none",
          maxHeight: "none",
          pointerEvents: "none",
        }}
      />
      <svg
        ref={svgRef}
        className="pcp-overlay"
        viewBox={`0 0 ${geometry.width} ${geometry.height}`}
        width="100%"
        height={geometry.height}
        role="img"
        aria-label={ariaLabel}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: geometry.width,
          height: geometry.height,
          overflow: "visible",
        }}
      />
    </div>
  );
}

export default ParallelCoordinates;
