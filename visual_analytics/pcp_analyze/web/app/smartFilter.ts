export const SMART_FILTER_LIMIT = 50;
export const SMART_FILTER_LEARNER_METHODS = [
  "MLP",
  "K-Fold",
  "Triplet Loss",
  "Attention Pooling",
  "Attribute-conditioned Attention",
  "nnPU",
  "DC-PU",
  "Ours-PURA",
] as const;

export type SmartFilterKind =
  | "softgate-boundary"
  | "learner-disagreement"
  | "learner-fusion-gap"
  | "prototype-rank-gap";

/** Current-model values, with learner columns ordered like learnerMethodIndices. */
export interface SmartFilterLiveValues {
  gateScores: Float32Array;
  gateCount: number;
  learnerRanks: Float32Array;
  fusionRanks: Float32Array;
  comparisonRanks: Float32Array;
}

export interface SmartFilterSource {
  rowCount: number;
  methodCount: number;
  targetCount: number;
  targetIndex: number;
  targetMemberIndices: readonly number[];
  learnerMethodIndices: readonly number[];
  fusionMethodIndex: number;
  prototypeMethodIndex: number;
  selectedLearnerIndex: number;
  comparisonMethodIndex: number;
  rawScores: Float32Array;
  ranks: Float32Array;
  candidateMask: Uint8Array;
  /** Omit only for an explicitly selected legacy/static diagnostic source. */
  liveValues?: SmartFilterLiveValues;
}

export interface SmartFilterResult {
  kind: SmartFilterKind;
  mask: Uint8Array;
  rowIndices: number[];
  selectedCount: number;
  eligibleCount: number;
  candidateCount: number;
  cutoff: number | null;
}

interface RankedCandidate {
  rowIndex: number;
  metric: number;
  secondary: number;
  tertiary: number;
}

interface DiagnosticMetric {
  metric: number;
  secondary?: number;
  tertiary?: number;
}

const THRESHOLD_EPSILON = 1e-6;

function scoreOffset(
  rowIndex: number,
  methodIndex: number,
  targetIndex: number,
  methodCount: number,
  targetCount: number,
) {
  return ((rowIndex * methodCount + methodIndex) * targetCount) + targetIndex;
}

function compareCandidate(
  left: RankedCandidate,
  right: RankedCandidate,
  direction: "ascending" | "descending",
) {
  const metricDifference = direction === "ascending"
    ? left.metric - right.metric
    : right.metric - left.metric;
  if (metricDifference !== 0) return metricDifference;
  const secondaryDifference = right.secondary - left.secondary;
  if (secondaryDifference !== 0) return secondaryDifference;
  const tertiaryDifference = left.tertiary - right.tertiary;
  return tertiaryDifference || left.rowIndex - right.rowIndex;
}

function insertBounded(
  selected: RankedCandidate[],
  candidate: RankedCandidate,
  direction: "ascending" | "descending",
) {
  let low = 0;
  let high = selected.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (compareCandidate(candidate, selected[middle], direction) < 0) high = middle;
    else low = middle + 1;
  }
  if (low >= SMART_FILTER_LIMIT && selected.length >= SMART_FILTER_LIMIT) return;
  selected.splice(low, 0, candidate);
  if (selected.length > SMART_FILTER_LIMIT) selected.pop();
}

function readRank(
  source: SmartFilterSource,
  rowIndex: number,
  methodIndex: number,
) {
  return source.ranks[scoreOffset(
    rowIndex,
    methodIndex,
    source.targetIndex,
    source.methodCount,
    source.targetCount,
  )];
}

function softGateMargin(source: SmartFilterSource, rowIndex: number) {
  let margin = Number.POSITIVE_INFINITY;
  if (source.liveValues) {
    const { gateScores, gateCount } = source.liveValues;
    for (let memberIndex = 0; memberIndex < gateCount; memberIndex += 1) {
      margin = Math.min(margin, Math.abs(gateScores[rowIndex * gateCount + memberIndex] - 0.5));
    }
    return margin;
  }
  const memberIndices = source.targetMemberIndices.length > 0
    ? source.targetMemberIndices
    : [source.targetIndex];
  for (const targetIndex of memberIndices) {
    const gate = source.rawScores[scoreOffset(
      rowIndex,
      source.fusionMethodIndex,
      targetIndex,
      source.methodCount,
      source.targetCount,
    )];
    if (!Number.isFinite(gate)) return null;
    margin = Math.min(margin, Math.abs(gate - 0.5));
  }
  return Number.isFinite(margin) ? margin : null;
}

function learnerDisagreement(source: SmartFilterSource, rowIndex: number) {
  if (source.learnerMethodIndices.length < 2) return null;
  let sum = 0;
  let squareSum = 0;
  for (const [learnerIndex, methodIndex] of source.learnerMethodIndices.entries()) {
    const value = source.liveValues
      ? source.liveValues.learnerRanks[rowIndex * source.learnerMethodIndices.length + learnerIndex]
      : readRank(source, rowIndex, methodIndex);
    if (!Number.isFinite(value)) return null;
    sum += value;
    squareSum += value * value;
  }
  const count = source.learnerMethodIndices.length;
  const mean = sum / count;
  return Math.sqrt(Math.max(0, squareSum / count - mean * mean));
}

function learnerFusionGap(source: SmartFilterSource, rowIndex: number) {
  const learnerRank = source.liveValues
    ? source.liveValues.learnerRanks[
        rowIndex * source.learnerMethodIndices.length
        + source.learnerMethodIndices.indexOf(source.selectedLearnerIndex)
      ]
    : readRank(source, rowIndex, source.selectedLearnerIndex);
  const fusionRank = source.liveValues?.fusionRanks[rowIndex]
    ?? readRank(source, rowIndex, source.fusionMethodIndex);
  const gap = learnerRank - fusionRank;
  if (
    !Number.isFinite(learnerRank)
    || !Number.isFinite(fusionRank)
    || learnerRank + THRESHOLD_EPSILON < 0.9
    || fusionRank - THRESHOLD_EPSILON > 0.5
    || gap + THRESHOLD_EPSILON < 0.4
  ) return null;
  return { metric: gap, secondary: learnerRank, tertiary: fusionRank };
}

function prototypeRankGap(source: SmartFilterSource, rowIndex: number) {
  const prototypeRank = readRank(source, rowIndex, source.prototypeMethodIndex);
  const comparisonRank = source.liveValues?.comparisonRanks[rowIndex]
    ?? readRank(source, rowIndex, source.comparisonMethodIndex);
  const gap = prototypeRank - comparisonRank;
  if (
    !Number.isFinite(prototypeRank)
    || !Number.isFinite(comparisonRank)
    || prototypeRank + THRESHOLD_EPSILON < 0.9
    || comparisonRank - THRESHOLD_EPSILON > 0.5
    || gap + THRESHOLD_EPSILON < 0.4
  ) return null;
  return { metric: gap, secondary: prototypeRank, tertiary: comparisonRank };
}

function validateSource(source: SmartFilterSource) {
  const scoreLength = source.rowCount * source.methodCount * source.targetCount;
  if (source.rowCount < 0 || source.methodCount <= 0 || source.targetCount <= 0) {
    throw new RangeError("Smart filter dimensions must be positive.");
  }
  if (
    source.rawScores.length !== scoreLength
    || source.ranks.length !== scoreLength
    || source.candidateMask.length !== source.rowCount
  ) {
    throw new RangeError("Smart filter arrays do not match their row contract.");
  }
  const methodIndices = [
    ...source.learnerMethodIndices,
    source.fusionMethodIndex,
    source.prototypeMethodIndex,
    source.selectedLearnerIndex,
    source.comparisonMethodIndex,
  ];
  const targetIndices = [source.targetIndex, ...source.targetMemberIndices];
  if (
    source.learnerMethodIndices.length !== 8
    || methodIndices.some((index) => !Number.isInteger(index) || index < 0 || index >= source.methodCount)
    || targetIndices.some((index) => !Number.isInteger(index) || index < 0 || index >= source.targetCount)
  ) {
    throw new RangeError("Smart filter method or target indices are invalid.");
  }
  if (source.liveValues) {
    const { gateScores, gateCount, learnerRanks, fusionRanks, comparisonRanks } = source.liveValues;
    if (
      !Number.isSafeInteger(gateCount) || gateCount <= 0
      || gateScores.length !== source.rowCount * gateCount
      || learnerRanks.length !== source.rowCount * source.learnerMethodIndices.length
      || fusionRanks.length !== source.rowCount
      || comparisonRanks.length !== source.rowCount
      || new Set(source.learnerMethodIndices).size !== source.learnerMethodIndices.length
      || !source.learnerMethodIndices.includes(source.selectedLearnerIndex)
    ) {
      throw new RangeError("Live diagnostic arrays do not match their row or learner contract.");
    }
    if ([gateScores, learnerRanks, fusionRanks, comparisonRanks].some((values) => (
      values.some((value) => !Number.isFinite(value) || value < 0 || value > 1)
    ))) {
      throw new RangeError("Live diagnostic values must be finite values in [0, 1].");
    }
  }
}

/** Selects at most 50 diagnostic rows without using ground-truth labels. */
export function buildSmartFilterMask(
  kind: SmartFilterKind,
  source: SmartFilterSource,
): SmartFilterResult {
  validateSource(source);
  const direction = kind === "softgate-boundary" ? "ascending" : "descending";
  const selected: RankedCandidate[] = [];
  let candidateCount = 0;
  let eligibleCount = 0;

  for (let rowIndex = 0; rowIndex < source.rowCount; rowIndex += 1) {
    if (!source.candidateMask[rowIndex]) continue;
    candidateCount += 1;
    let diagnostic: DiagnosticMetric | null;
    if (kind === "softgate-boundary") {
      const metric = softGateMargin(source, rowIndex);
      diagnostic = metric === null ? null : { metric };
    } else if (kind === "learner-disagreement") {
      const metric = learnerDisagreement(source, rowIndex);
      diagnostic = metric === null ? null : { metric };
    } else if (kind === "learner-fusion-gap") {
      diagnostic = learnerFusionGap(source, rowIndex);
    } else {
      diagnostic = prototypeRankGap(source, rowIndex);
    }
    if (diagnostic === null || !Number.isFinite(diagnostic.metric)) continue;
    eligibleCount += 1;
    insertBounded(selected, {
      rowIndex,
      metric: diagnostic.metric,
      secondary: diagnostic.secondary ?? 0,
      tertiary: diagnostic.tertiary ?? 0,
    }, direction);
  }

  const mask = new Uint8Array(source.rowCount);
  for (const candidate of selected) mask[candidate.rowIndex] = 1;
  return {
    kind,
    mask,
    rowIndices: selected.map((candidate) => candidate.rowIndex),
    selectedCount: selected.length,
    eligibleCount,
    candidateCount,
    cutoff: selected.at(-1)?.metric ?? null,
  };
}
