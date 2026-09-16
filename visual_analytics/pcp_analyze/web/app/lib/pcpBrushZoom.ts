export type PcpZoomRange = readonly [number, number];
export type PcpZoomRangeMap = Readonly<Record<string, PcpZoomRange | undefined>>;

export interface PcpBrushZoomState {
  /** Absolute value ranges which remain active as the parent drill-down scope. */
  ranges: PcpZoomRangeMap;
  /** Previous parent scopes, oldest first, for one-level-at-a-time navigation. */
  history: readonly PcpZoomRangeMap[];
}

const DEFAULT_DOMAIN: PcpZoomRange = [0, 1];
const RANGE_EPSILON = Number.EPSILON;

function normalizeRange(range: PcpZoomRange | undefined): [number, number] | null {
  if (!range || !Number.isFinite(range[0]) || !Number.isFinite(range[1])) return null;
  const lower = Math.min(range[0], range[1]);
  const upper = Math.max(range[0], range[1]);
  return upper - lower > RANGE_EPSILON ? [lower, upper] : null;
}

function cloneRanges(ranges: PcpZoomRangeMap): Record<string, [number, number]> {
  const copy: Record<string, [number, number]> = {};
  for (const [axisId, range] of Object.entries(ranges)) {
    const normalized = normalizeRange(range);
    if (normalized) copy[axisId] = normalized;
  }
  return copy;
}

function rangesEqual(first: PcpZoomRange, second: PcpZoomRange) {
  return Math.abs(first[0] - second[0]) <= RANGE_EPSILON
    && Math.abs(first[1] - second[1]) <= RANGE_EPSILON;
}

export function createPcpBrushZoomState(): PcpBrushZoomState {
  return { ranges: {}, history: [] };
}

export function hasPcpBrushZoom(state: PcpBrushZoomState) {
  return Object.keys(state.ranges).length > 0;
}

export function pcpBrushZoomDepth(state: PcpBrushZoomState) {
  return state.history.length;
}

/**
 * Commit the current visible brushes as a persistent parent scope. Only axes
 * whose brush is strictly narrower than their displayed domain create a new
 * level. Ranges stay in absolute score/rank coordinates across every level.
 */
export function drillIntoPcpBrushZoom(
  state: PcpBrushZoomState,
  brushes: PcpZoomRangeMap,
  axisIds: readonly string[],
  baseDomain: PcpZoomRange = DEFAULT_DOMAIN,
): PcpBrushZoomState {
  const normalizedBase = normalizeRange(baseDomain) ?? [0, 1];
  const allowedAxes = new Set(axisIds);
  const nextRanges = cloneRanges(state.ranges);
  let narrowed = false;

  for (const [axisId, brush] of Object.entries(brushes)) {
    if (!allowedAxes.has(axisId)) continue;
    const normalizedBrush = normalizeRange(brush);
    if (!normalizedBrush) continue;
    const parent = normalizeRange(state.ranges[axisId]) ?? normalizedBase;
    const clipped: [number, number] = [
      Math.max(parent[0], normalizedBrush[0]),
      Math.min(parent[1], normalizedBrush[1]),
    ];
    if (clipped[1] - clipped[0] <= RANGE_EPSILON || rangesEqual(clipped, parent)) continue;
    nextRanges[axisId] = clipped;
    narrowed = true;
  }

  if (!narrowed) return state;
  return {
    ranges: nextRanges,
    history: [...state.history, cloneRanges(state.ranges)],
  };
}

export function stepBackPcpBrushZoom(state: PcpBrushZoomState): PcpBrushZoomState {
  const previous = state.history.at(-1);
  if (!previous) return state;
  return {
    ranges: cloneRanges(previous),
    history: state.history.slice(0, -1),
  };
}

export function resetPcpBrushZoom(): PcpBrushZoomState {
  return createPcpBrushZoomState();
}

/** Apply committed parent ranges to the current upstream row mask. */
export function applyPcpBrushZoomMask({
  values,
  axisIds,
  candidateMask,
  ranges,
}: {
  values: Float32Array;
  axisIds: readonly string[];
  candidateMask: Uint8Array;
  ranges: PcpZoomRangeMap;
}): Uint8Array {
  const axisCount = axisIds.length;
  if (axisCount === 0) {
    if (values.length !== 0) throw new Error("PCP zoom values require at least one axis.");
    return candidateMask.slice();
  }
  if (values.length !== candidateMask.length * axisCount) {
    throw new Error("PCP zoom values and candidate mask are not row-aligned.");
  }

  const axisIndex = new Map(axisIds.map((axisId, index) => [axisId, index]));
  const activeRanges = Object.entries(ranges).flatMap(([axisId, range]) => {
    const index = axisIndex.get(axisId);
    const normalized = normalizeRange(range);
    return index === undefined || !normalized ? [] : [{ index, range: normalized }];
  });
  if (activeRanges.length === 0) return candidateMask.slice();

  const result = new Uint8Array(candidateMask.length);
  candidateLoop: for (let rowIndex = 0; rowIndex < candidateMask.length; rowIndex += 1) {
    if (!candidateMask[rowIndex]) continue;
    const rowOffset = rowIndex * axisCount;
    for (const active of activeRanges) {
      const value = values[rowOffset + active.index];
      if (!Number.isFinite(value) || value < active.range[0] || value > active.range[1]) {
        continue candidateLoop;
      }
    }
    result[rowIndex] = 1;
  }
  return result;
}

export function pcpBrushZoomFingerprint(
  ranges: PcpZoomRangeMap,
  axisIds: readonly string[],
) {
  return axisIds.flatMap((axisId) => {
    const range = normalizeRange(ranges[axisId]);
    return range ? [`${axisId}:${range[0].toPrecision(10)}:${range[1].toPrecision(10)}`] : [];
  }).join("|") || "full-domain";
}

/** Keep deep zoom endpoints (for example 0.99999 and 1) visibly distinct. */
export function formatPcpAxisTick(value: number, domain: PcpZoomRange) {
  if (!Number.isFinite(value)) return "";
  const normalizedDomain = normalizeRange(domain) ?? [0, 1];
  const span = normalizedDomain[1] - normalizedDomain[0];
  const precision = Math.min(
    8,
    Math.max(2, Math.ceil(-Math.log10(span)) + 2),
  );
  const rounded = Number(value.toFixed(precision));
  return Object.is(rounded, -0) ? "0" : String(rounded);
}
