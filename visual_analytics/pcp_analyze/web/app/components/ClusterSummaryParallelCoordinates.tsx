"use client";

import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import * as d3 from "d3";
import { CHART_UI_FONT } from "../lib/chartTypography";
import {
  buildClusterSummaryNavigation,
  clusterSummaryDirectionForKey,
  moveClusterSummarySelection,
} from "../lib/clusterSummaryNavigation";
import {
  HIERARCHICAL_PCP_BOTTOM_PADDING,
  HIERARCHICAL_PCP_TOP,
} from "../lib/hierarchicalPcp";
import type {
  PcpManualHighlight,
  PcpSelectedRowIndices,
} from "./ParallelCoordinates";

export type ClusterSummaryId = string | number;
export type ClusterColorResolver =
  | Readonly<Record<string, string>>
  | ((cluster: ClusterSummaryId) => string);

export interface ClusterSummaryParallelCoordinatesProps {
  /** Stable axis IDs in the same column order as `values`. */
  methods: readonly string[];
  /** Optional display labels keyed by stable axis ID. */
  axisLabels?: Readonly<Record<string, string>>;
  /** Optional learner/group label rendered beside the first axis in a group. */
  axisGroupLabels?: Readonly<Record<string, string>>;
  /** Row-major values: sample 0 / every axis, sample 1 / every axis, ... */
  values: Float32Array;
  /** Cluster ID per sample. */
  labels: ArrayLike<ClusterSummaryId>;
  /** Axis IDs included in each cluster centroid line. */
  enabledMethods: ReadonlySet<string> | readonly string[];
  /** Optional upstream mask. A non-zero value includes that sample. */
  candidateMask?: Uint8Array;
  /** Neutral selected rows, expressed as original matrix row indices. */
  selectedRowIndices?: PcpSelectedRowIndices;
  /** Human annotations; cluster summaries aggregate their polarity-free union. */
  manualHighlights?: readonly PcpManualHighlight[];
  /** A currently clicked/previewed row included in the aggregate highlight. */
  focusedRowIndex?: number | null;
  /** Optional ordered cluster subset. Omit to show every cluster in `labels`. */
  activeClusters?: readonly ClusterSummaryId[];
  clusterColors?: ClusterColorResolver;
  clusterLabels?:
    | Readonly<Record<string, string>>
    | ((cluster: ClusterSummaryId) => string);
  selectedCluster?: ClusterSummaryId | null;
  onClusterClick?: (cluster: ClusterSummaryId) => void;
  /** Opens the selected centroid as a sample-level PCP drill-down. */
  onClusterDoubleClick?: (cluster: ClusterSummaryId) => void;
  /** Selects a cluster without click-to-toggle semantics (used by keyboard navigation). */
  onClusterSelect?: (cluster: ClusterSummaryId) => void;
  domain?: readonly [number, number];
  valueLabel?: string;
  title?: string;
  height?: number;
  lineWidthRange?: readonly [number, number];
  selectedHighlightColor?: string;
  /** Optional axis-aligned hierarchy controls rendered inside the chart stage. */
  axisRail?: ReactNode;
  /** Hides duplicate chart labels when an aligned axis rail supplies them. */
  showAxisLabels?: boolean;
  ariaLabel?: string;
  className?: string;
}

interface ClusterAccumulator {
  id: ClusterSummaryId;
  key: string;
  count: number;
  selectedCount: number;
  sums: Float64Array;
  validCounts: Uint32Array;
}

interface ClusterProfile {
  id: ClusterSummaryId;
  key: string;
  count: number;
  selectedCount: number;
  means: Float64Array;
}

interface Viewport {
  width: number;
  height: number;
}

const DEFAULT_CLUSTER_COLORS = ["#ee6a4b", "#4169c1", "#2d9a74", "#8b62b5"];
const DEFAULT_DOMAIN = [0, 1] as const;
const DEFAULT_LINE_WIDTH = [2.5, 10] as const;

function normalizeDomain(domain: readonly [number, number]): [number, number] {
  const first = Number.isFinite(domain[0]) ? domain[0] : 0;
  const second = Number.isFinite(domain[1]) ? domain[1] : 1;
  if (first === second) return [first, first + 1];
  return first < second ? [first, second] : [second, first];
}

function clusterKey(cluster: ClusterSummaryId) {
  return String(cluster);
}

function clusterName(
  cluster: ClusterSummaryId,
  labels: ClusterSummaryParallelCoordinatesProps["clusterLabels"],
) {
  if (typeof labels === "function") return labels(cluster);
  return labels?.[clusterKey(cluster)] ?? `Cluster ${clusterKey(cluster)}`;
}

function resolveClusterColor(
  cluster: ClusterSummaryId,
  index: number,
  colors: ClusterColorResolver | undefined,
) {
  if (typeof colors === "function") return colors(cluster);
  return colors?.[clusterKey(cluster)] ?? DEFAULT_CLUSTER_COLORS[index % DEFAULT_CLUSTER_COLORS.length];
}

function truncateLabel(label: string, maximum = 23) {
  return label.length <= maximum ? label : `${label.slice(0, maximum - 1)}\u2026`;
}

const ROOT_STYLE: CSSProperties = {
  background: "#fff",
  border: "1px solid var(--line, #d9dce1)",
  borderRadius: 10,
  overflow: "hidden",
  width: "100%",
};

/**
 * A compact PCP summary of a clustering. Every visible cluster becomes one
 * centroid polyline (per-axis arithmetic mean); stroke width linearly encodes
 * the cluster's current sample count.
 */
export function ClusterSummaryParallelCoordinates({
  methods,
  axisLabels,
  axisGroupLabels,
  values,
  labels,
  enabledMethods,
  candidateMask,
  selectedRowIndices,
  manualHighlights,
  focusedRowIndex = null,
  activeClusters,
  clusterColors,
  clusterLabels,
  selectedCluster = null,
  onClusterClick,
  onClusterDoubleClick,
  onClusterSelect,
  domain: domainProp = DEFAULT_DOMAIN,
  valueLabel = "rank",
  title = "Cluster PCP",
  height = 390,
  lineWidthRange = DEFAULT_LINE_WIDTH,
  selectedHighlightColor = "#ea580c",
  axisRail,
  showAxisLabels = true,
  ariaLabel = "Cluster centroid parallel coordinates",
  className = "",
}: ClusterSummaryParallelCoordinatesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartPaneRef = useRef<HTMLDivElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);
  const [legendExpanded, setLegendExpanded] = useState(false);
  const chartHeight = Math.max(220, height);
  const hasAxisRail = Boolean(axisRail);
  const [viewport, setViewport] = useState<Viewport>({ width: 640, height: chartHeight });

  useLayoutEffect(() => {
    const chartPane = chartPaneRef.current;
    if (!chartPane) return;

    const updateSize = () => {
      // Match the available plot pane rather than clipping behind the evidence rail.
      const width = Math.max(1, Math.round(chartPane.getBoundingClientRect().width || 640));
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
    observer.observe(chartPane);
    return () => observer.disconnect();
  }, [chartHeight, hasAxisRail]);

  const enabledSet = useMemo(() => new Set<string>(enabledMethods), [enabledMethods]);
  const activeMethodIndices = useMemo(
    () => methods.flatMap((method, index) => (enabledSet.has(method) ? [index] : [])),
    [enabledSet, methods],
  );
  const requestedClusterOrder = useMemo(
    () => activeClusters?.map(clusterKey) ?? null,
    [activeClusters],
  );
  const requestedClusters = useMemo(
    () => requestedClusterOrder ? new Set(requestedClusterOrder) : null,
    [requestedClusterOrder],
  );
  const rowCount = methods.length > 0
    ? Math.min(labels.length, Math.floor(values.length / methods.length))
    : 0;
  const highlightedRowSet = useMemo(() => {
    const rows = new Set<number>();
    for (const rowIndex of selectedRowIndices ?? []) {
      if (Number.isInteger(rowIndex) && rowIndex >= 0 && rowIndex < rowCount) rows.add(rowIndex);
    }
    for (const highlight of manualHighlights ?? []) {
      if (
        Number.isInteger(highlight.rowIndex)
        && highlight.rowIndex >= 0
        && highlight.rowIndex < rowCount
      ) rows.add(highlight.rowIndex);
    }
    if (
      focusedRowIndex !== null
      && Number.isInteger(focusedRowIndex)
      && focusedRowIndex >= 0
      && focusedRowIndex < rowCount
    ) rows.add(focusedRowIndex);
    return rows;
  }, [focusedRowIndex, manualHighlights, rowCount, selectedRowIndices]);

  const profiles = useMemo<ClusterProfile[]>(() => {
    const accumulators = new Map<string, ClusterAccumulator>();
    for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
      if (candidateMask && !candidateMask[rowIndex]) continue;
      const id = labels[rowIndex];
      const key = clusterKey(id);
      if (requestedClusters && !requestedClusters.has(key)) continue;

      let accumulator = accumulators.get(key);
      if (!accumulator) {
        accumulator = {
          id,
          key,
          count: 0,
          selectedCount: 0,
          sums: new Float64Array(methods.length),
          validCounts: new Uint32Array(methods.length),
        };
        accumulators.set(key, accumulator);
      }
      accumulator.count += 1;
      if (highlightedRowSet.has(rowIndex)) accumulator.selectedCount += 1;

      const rowOffset = rowIndex * methods.length;
      for (const methodIndex of activeMethodIndices) {
        const value = values[rowOffset + methodIndex];
        if (!Number.isFinite(value)) continue;
        accumulator.sums[methodIndex] += value;
        accumulator.validCounts[methodIndex] += 1;
      }
    }

    const discovered = [...accumulators.values()];
    if (requestedClusterOrder) {
      const order = new Map(requestedClusterOrder.map((key, index) => [key, index]));
      discovered.sort((left, right) =>
        (order.get(left.key) ?? Number.MAX_SAFE_INTEGER)
        - (order.get(right.key) ?? Number.MAX_SAFE_INTEGER),
      );
    }

    return discovered.map((accumulator) => {
      const means = new Float64Array(methods.length);
      means.fill(Number.NaN);
      for (const methodIndex of activeMethodIndices) {
        const validCount = accumulator.validCounts[methodIndex];
        if (validCount > 0) means[methodIndex] = accumulator.sums[methodIndex] / validCount;
      }
      return {
        id: accumulator.id,
        key: accumulator.key,
        count: accumulator.count,
        selectedCount: accumulator.selectedCount,
        means,
      };
    });
  }, [activeMethodIndices, candidateMask, highlightedRowSet, labels, methods.length, requestedClusterOrder, requestedClusters, rowCount, values]);

  const selectedProfileCount = profiles.reduce(
    (count, profile) => count + (profile.selectedCount > 0 ? 1 : 0),
    0,
  );
  const visibleSelectedRowCount = profiles.reduce(
    (count, profile) => count + profile.selectedCount,
    0,
  );
  const hasVisibleManualSelection = visibleSelectedRowCount > 0;

  const clusterNavigation = useMemo(
    () => buildClusterSummaryNavigation(profiles, activeMethodIndices),
    [activeMethodIndices, profiles],
  );
  const orderedProfiles = useMemo(
    () => clusterNavigation.entries.map((entry) => entry.profile),
    [clusterNavigation],
  );
  const selectedKey = selectedCluster === null || selectedCluster === undefined
    ? null
    : clusterKey(selectedCluster);
  const selectedPosition = selectedKey === null
    ? undefined
    : clusterNavigation.indexByKey.get(selectedKey);
  const selectedEntry = selectedPosition === undefined
    ? null
    : clusterNavigation.entries[selectedPosition];
  const selectFromKeyboard = onClusterSelect ?? onClusterClick;

  useEffect(() => {
    if (!selectFromKeyboard) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented
        || event.isComposing
        || event.ctrlKey
        || event.metaKey
        || event.altKey
      ) return;
      if (document.querySelector('dialog[open], [role="dialog"][aria-modal="true"]')) return;

      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest([
        "input",
        "select",
        "textarea",
        '[contenteditable]:not([contenteditable="false"])',
        '[role="textbox"]',
        '[role="combobox"]',
        '[role="slider"]',
        '[role="spinbutton"]',
        '[role="listbox"]',
      ].join(","))) return;
      const interactiveTarget = target?.closest('button, a[href], summary, canvas[tabindex]');
      if (interactiveTarget && !containerRef.current?.contains(interactiveTarget)) return;

      const direction = clusterSummaryDirectionForKey(event.key);
      if (direction === null) return;
      event.preventDefault();
      const nextCluster = moveClusterSummarySelection(
        clusterNavigation,
        selectedCluster,
        direction,
      );
      if (nextCluster === null) return;
      selectFromKeyboard(nextCluster);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [clusterNavigation, selectFromKeyboard, selectedCluster]);

  useLayoutEffect(() => {
    if (!legendExpanded) return;
    const legend = legendRef.current;
    const selected = legend?.querySelector<HTMLElement>('[data-cluster-selected="true"]');
    if (!legend || !selected) return;
    const legendBounds = legend.getBoundingClientRect();
    const selectedBounds = selected.getBoundingClientRect();
    if (selectedBounds.top < legendBounds.top) {
      legend.scrollTop -= legendBounds.top - selectedBounds.top;
    } else if (selectedBounds.bottom > legendBounds.bottom) {
      legend.scrollTop += selectedBounds.bottom - legendBounds.bottom;
    }
  }, [legendExpanded, selectedKey, selectedPosition]);

  const [domainMinimum, domainMaximum] = normalizeDomain(domainProp);
  const hasAxisGroupLabels = showAxisLabels
    && Boolean(axisGroupLabels && Object.keys(axisGroupLabels).length > 0);
  const left = showAxisLabels
    ? hasAxisGroupLabels
      ? Math.min(230, Math.max(154, viewport.width * 0.4))
      : Math.min(210, Math.max(112, viewport.width * 0.3))
    : 18;
  const right = Math.max(left + 80, viewport.width - 18);
  const top = hasAxisRail ? HIERARCHICAL_PCP_TOP : 24;
  const bottom = Math.max(
    top + 20,
    viewport.height - (hasAxisRail ? HIERARCHICAL_PCP_BOTTOM_PADDING : 34),
  );
  const rowGap = methods.length > 1 ? (bottom - top) / (methods.length - 1) : 0;
  const scaleX = useMemo(
    () => d3.scaleLinear().domain([domainMinimum, domainMaximum]).range([left, right]).clamp(true),
    [domainMaximum, domainMinimum, left, right],
  );
  const maxCount = d3.max(orderedProfiles, (profile) => profile.count) ?? 1;
  const minimumWidth = Math.min(lineWidthRange[0], lineWidthRange[1]);
  const maximumWidth = Math.max(lineWidthRange[0], lineWidthRange[1]);

  const profilePaths = useMemo(() => {
    const pathBuilder = d3
      .line<{ methodIndex: number; value: number }>()
      .defined((point) => Number.isFinite(point.value))
      .x((point) => scaleX(point.value))
      .y((point) => methods.length > 1
        ? top + point.methodIndex * rowGap
        : (top + bottom) / 2);
    return orderedProfiles.map((profile, index) => {
      const points = activeMethodIndices.map((methodIndex) => ({
        methodIndex,
        value: profile.means[methodIndex],
      }));
      const lastPoint = [...points].reverse().find((point) => Number.isFinite(point.value));
      return {
        ...profile,
        color: resolveClusterColor(profile.id, index, clusterColors),
        label: clusterName(profile.id, clusterLabels),
        lineWidth: minimumWidth
          + (maximumWidth - minimumWidth) * (profile.count / Math.max(1, maxCount)),
        path: pathBuilder(points) ?? "",
        selectedCountAnchor: lastPoint ? {
          x: scaleX(lastPoint.value),
          y: methods.length > 1
            ? top + lastPoint.methodIndex * rowGap
            : (top + bottom) / 2,
        } : null,
      };
    });
  }, [
    activeMethodIndices,
    bottom,
    clusterColors,
    clusterLabels,
    maxCount,
    maximumWidth,
    methods.length,
    minimumWidth,
    orderedProfiles,
    rowGap,
    scaleX,
    top,
  ]);

  const denseProfiles = profilePaths.length > 12;
  const tickValues = [domainMinimum, (domainMinimum + domainMaximum) / 2, domainMaximum];
  const tickFormat = d3.format(".2~f");
  const meanFormat = d3.format(".3~f");

  return (
    <div
      ref={containerRef}
      className={className}
      role="group"
      aria-label={ariaLabel}
      aria-keyshortcuts="ArrowLeft ArrowRight A D"
      tabIndex={selectFromKeyboard ? 0 : undefined}
      style={ROOT_STYLE}
    >
      <div style={{ padding: "11px 12px 6px" }}>
        <details className="cluster-disclosure pcp-cluster-disclosure" open={legendExpanded}
          onToggle={(event) => setLegendExpanded(event.currentTarget.open)}>
        <summary>
        <strong className="pcp-summary-title">{title}</strong>
          <small aria-live="polite">
            {selectedEntry && selectedPosition !== undefined
              ? `${selectedPosition + 1}/${clusterNavigation.entries.length} · ${clusterName(selectedEntry.profile.id, clusterLabels)} · mean ${selectedEntry.mean === null ? "n/a" : meanFormat(selectedEntry.mean)}${selectedEntry.profile.selectedCount > 0 ? ` · ${selectedEntry.profile.selectedCount.toLocaleString()} selected` : ""}`
              : `${clusterNavigation.entries.length} clusters${hasVisibleManualSelection ? ` · ${visibleSelectedRowCount.toLocaleString()} selected in ${selectedProfileCount.toLocaleString()}` : ""}`}
          </small>
        </summary>
        <div
          ref={legendRef}
          className="pcp-cluster-legend"
          role="group"
          aria-label="PCP cluster legend"
          tabIndex={0}
        >
          {profilePaths.map((profile) => {
            const selected = selectedCluster !== null && clusterKey(selectedCluster) === profile.key;
            const manuallyHighlighted = profile.selectedCount > 0;
            return (
              <button
                key={profile.key}
                type="button"
                disabled={!onClusterClick && !onClusterDoubleClick}
                aria-pressed={selected}
                data-cluster-selected={selected ? "true" : undefined}
                data-manual-selected-count={manuallyHighlighted ? profile.selectedCount : undefined}
                title={`${profile.label} · ${profile.count.toLocaleString()}${manuallyHighlighted ? ` · ${profile.selectedCount.toLocaleString()} selected` : ""}`}
                onClick={(event) => {
                  // The second click in a double-click must not toggle the
                  // selected cluster back off before the drill-down opens.
                  if (event.detail > 1) return;
                  onClusterClick?.(profile.id);
                }}
                onDoubleClick={() => onClusterDoubleClick?.(profile.id)}
                style={{
                  alignItems: "center",
                  background: selected
                    ? "var(--paper, #f4f3ee)"
                    : manuallyHighlighted
                      ? "rgba(234, 88, 12, 0.09)"
                      : "transparent",
                  border: manuallyHighlighted
                    ? `1px solid ${selectedHighlightColor}`
                    : selected
                      ? "1px solid var(--line, #d9dce1)"
                      : "1px solid transparent",
                  borderRadius: 5,
                  color: "var(--ink, #20242b)",
                  cursor: onClusterClick || onClusterDoubleClick ? "pointer" : "default",
                  display: "inline-flex",
                  fontSize: 11,
                  gap: 5,
                  minHeight: 30,
                  opacity: manuallyHighlighted || selected
                    ? 1
                    : hasVisibleManualSelection
                      ? 0.48
                      : selectedCluster === null
                        ? 1
                        : 0.58,
                  padding: "2px 4px",
                }}
              >
                <i
                  aria-hidden="true"
                  style={{
                    background: profile.color,
                    borderRadius: 99,
                    display: "inline-block",
                    height: Math.max(2, profile.lineWidth),
                    width: 22,
                  }}
                />
                <span>{profile.label}</span>
                {manuallyHighlighted && (
                  <small
                    title={`${profile.selectedCount.toLocaleString()} selected samples in this cluster`}
                    style={{
                      background: selectedHighlightColor,
                      borderRadius: 99,
                      color: "#fff",
                      fontSize: 11,
                      fontWeight: 750,
                      lineHeight: 1,
                      padding: "3px 5px",
                    }}
                  >
                    {profile.selectedCount.toLocaleString()} selected
                  </small>
                )}
                <small style={{ color: "var(--muted, #717986)", fontSize: 11 }}>
                  {profile.count.toLocaleString()}
                </small>
              </button>
            );
          })}
        </div>
        </details>
      </div>

      <div
        className={`cluster-summary-chart-stage${hasAxisRail ? " cluster-summary-chart-stage--with-rail" : ""}`}
        style={{ height: viewport.height }}
      >
        {axisRail}
        <div ref={chartPaneRef} className="cluster-summary-chart-pane">
          <svg
            viewBox={`0 0 ${viewport.width} ${viewport.height}`}
            width="100%"
            height={viewport.height}
            role="img"
            aria-label={ariaLabel}
            style={{ display: "block", overflow: "visible", fontFamily: CHART_UI_FONT }}
          >
            <title>{ariaLabel}</title>
            {methods.map((method, methodIndex) => {
          const enabled = enabledSet.has(method);
          const y = methods.length > 1
            ? top + methodIndex * rowGap
            : (top + bottom) / 2;
          const label = axisLabels?.[method] ?? method;
          const groupLabel = axisGroupLabels?.[method];
          return (
            <g key={method} opacity={enabled ? 1 : 0.25}>
              {groupLabel && (
                <text
                  x={4}
                  y={y}
                  dy="0.32em"
                  fill="var(--ink, #20242b)"
                  fontFamily={CHART_UI_FONT}
                  fontSize={11}
                  fontWeight={700}
                >
                  <title>{groupLabel}</title>
                  {truncateLabel(groupLabel, 18)}
                </text>
              )}
              <line x1={left} x2={right} y1={y} y2={y} stroke="#aeb4bb" />
              {tickValues.map((tick) => {
                const x = scaleX(tick);
                return (
                  <g key={tick} transform={`translate(${x},${y})`}>
                    <line y2={4} stroke="#aeb4bb" />
                    {methodIndex === methods.length - 1 && (
                      <text
                        dy="1.5em"
                        fill="var(--muted, #717986)"
                        fontFamily={CHART_UI_FONT}
                        fontSize={11}
                        textAnchor="middle"
                      >
                        {tickFormat(tick)}
                      </text>
                    )}
                  </g>
                );
              })}
              {showAxisLabels && (
                <text
                  x={left - 12}
                  y={y}
                  dy="0.34em"
                  fill="var(--ink, #20242b)"
                  fontSize={11}
                  fontWeight={650}
                  textAnchor="end"
                >
                  <title>{label}</title>
                  {truncateLabel(label)}
                </text>
              )}
            </g>
          );
        })}

        {[...profilePaths]
          .sort((leftProfile, rightProfile) => {
            const leftSelected = selectedCluster !== null
              && clusterKey(selectedCluster) === leftProfile.key;
            const rightSelected = selectedCluster !== null
              && clusterKey(selectedCluster) === rightProfile.key;
            const leftPriority = (leftProfile.selectedCount > 0 ? 2 : 0) + (leftSelected ? 1 : 0);
            const rightPriority = (rightProfile.selectedCount > 0 ? 2 : 0) + (rightSelected ? 1 : 0);
            if (leftPriority !== rightPriority) return leftPriority - rightPriority;
            return rightProfile.count - leftProfile.count;
          })
          .map((profile) => {
            const selected = selectedCluster !== null && clusterKey(selectedCluster) === profile.key;
            const manuallyHighlighted = profile.selectedCount > 0;
            const emphasized = selected || manuallyHighlighted;
            const whiteOpacity = hasVisibleManualSelection
              ? (emphasized ? 0.92 : 0.08)
              : selectedCluster === null
                ? (denseProfiles ? 0.34 : 0.92)
                : (selected ? 0.92 : 0.12);
            const colorOpacity = hasVisibleManualSelection
              ? (emphasized ? 0.94 : 0.1)
              : selectedCluster === null
                ? (denseProfiles ? 0.38 : 0.88)
                : (selected ? 0.94 : 0.14);
            const selectedCountText = profile.selectedCount.toLocaleString();
            const badgeWidth = Math.max(18, 10 + selectedCountText.length * 6);
            const badgeX = profile.selectedCountAnchor
              ? Math.max(
                  left + badgeWidth / 2,
                  Math.min(right - badgeWidth / 2, profile.selectedCountAnchor.x),
                )
              : null;
            return (
              <g
                key={profile.key}
                aria-label={`${profile.label}: ${profile.count.toLocaleString()} samples${manuallyHighlighted ? `, ${selectedCountText} selected` : ""}`}
                data-manual-selected-count={manuallyHighlighted ? profile.selectedCount : undefined}
                onClick={(event) => {
                  if (event.detail > 1) return;
                  onClusterClick?.(profile.id);
                }}
                onDoubleClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  onClusterDoubleClick?.(profile.id);
                }}
                onPointerDown={() => containerRef.current?.focus({ preventScroll: true })}
                style={{
                  cursor: onClusterClick || onClusterDoubleClick ? "pointer" : "default",
                }}
              >
                {manuallyHighlighted && (
                  <path
                    d={profile.path}
                    fill="none"
                    opacity={0.9}
                    stroke={selectedHighlightColor}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={profile.lineWidth + 6}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                <path
                  d={profile.path}
                  fill="none"
                  opacity={whiteOpacity}
                  stroke="#fff"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={profile.lineWidth + 2}
                  vectorEffect="non-scaling-stroke"
                />
                <path
                  d={profile.path}
                  fill="none"
                  opacity={colorOpacity}
                  stroke={profile.color}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={profile.lineWidth}
                  vectorEffect="non-scaling-stroke"
                >
                  <title>{`${profile.label}: ${profile.count.toLocaleString()} samples, mean ${valueLabel}${manuallyHighlighted ? `; ${selectedCountText} selected` : ""}`}</title>
                </path>
                {manuallyHighlighted && profile.selectedCountAnchor && badgeX !== null && (
                  <g
                    aria-hidden="true"
                    pointerEvents="none"
                    transform={`translate(${badgeX},${profile.selectedCountAnchor.y})`}
                  >
                    <rect
                      x={-badgeWidth / 2}
                      y={-9}
                      width={badgeWidth}
                      height={18}
                      rx={9}
                      fill={selectedHighlightColor}
                      stroke="#fff"
                      strokeWidth={1.5}
                      vectorEffect="non-scaling-stroke"
                    />
                    <text
                      y={0}
                      dy="0.34em"
                      fill="#fff"
                      fontFamily={CHART_UI_FONT}
                      fontSize={11}
                      fontWeight={750}
                      textAnchor="middle"
                    >
                      {selectedCountText}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
          </svg>

          {profilePaths.length === 0 && (
            <p className="cluster-summary-empty">
              No cluster samples in the current scope.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default ClusterSummaryParallelCoordinates;
