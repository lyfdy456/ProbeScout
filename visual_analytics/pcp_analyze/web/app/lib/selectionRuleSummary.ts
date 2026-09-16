export type NumericValueColumn = Float32Array | Float64Array;

export interface SelectionRuleAttributeInput {
  id: string;
  label: string;
  ranks: NumericValueColumn;
  calibratedScores: NumericValueColumn;
}

export interface SelectionRuleSummaryInput {
  /** Rows selected by a brush or cluster click. */
  selectionMask: Uint8Array;
  /** Rows that are currently eligible in the active scope/filter. */
  baseMask: Uint8Array;
  attributes: readonly SelectionRuleAttributeInput[];
  quantileRange?: readonly [number, number];
}

export interface SelectionRuleValueRange {
  minimum: number;
  maximum: number;
  lowerQuantile: number;
  upperQuantile: number;
}

export interface SelectionRuleAttributeRange {
  attributeId: string;
  attributeLabel: string;
  sampleCount: number;
  rank: SelectionRuleValueRange;
  calibratedScore: SelectionRuleValueRange;
}

export interface SelectionRuleSummary {
  selectedCount: number;
  baseCount: number;
  lowerQuantile: number;
  upperQuantile: number;
  attributes: readonly SelectionRuleAttributeRange[];
}

export type SelectionRuleSummaryNoneReason =
  | "empty-base"
  | "empty-selection"
  | "entire-base-selected"
  | "no-attributes"
  | "non-finite-value";

export type SelectionRuleSummaryResult =
  | { kind: "summary"; summary: SelectionRuleSummary }
  | {
      kind: "none";
      reason: SelectionRuleSummaryNoneReason;
      attributeId?: string;
      rowIndex?: number;
      valueKind?: "rank" | "calibrated-score";
    };

const DEFAULT_QUANTILES = [0.05, 0.95] as const;

function assertBinaryMask(mask: Uint8Array, rowCount: number, label: string): void {
  if (!(mask instanceof Uint8Array) || mask.length !== rowCount) {
    throw new RangeError(`${label} must be a row-aligned Uint8Array.`);
  }
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    if (mask[rowIndex] !== 0 && mask[rowIndex] !== 1) {
      throw new RangeError(`${label} must contain only binary 0/1 values.`);
    }
  }
}

function assertConfiguration(quantileRange: readonly [number, number]): void {
  const [lowerQuantile, upperQuantile] = quantileRange;
  if (
    !Number.isFinite(lowerQuantile)
    || !Number.isFinite(upperQuantile)
    || lowerQuantile < 0
    || upperQuantile > 1
    || lowerQuantile > upperQuantile
  ) {
    throw new RangeError("quantileRange must be ordered within [0, 1].");
  }
}

function swap(values: Float64Array, left: number, right: number): void {
  const value = values[left];
  values[left] = values[right];
  values[right] = value;
}

/** Average O(n) in-place order statistic; all callers preclude NaN. */
function selectKth(values: Float64Array, target: number): number {
  let left = 0;
  let right = values.length - 1;
  while (left < right) {
    const pivot = values[Math.floor((left + right) / 2)];
    let low = left;
    let high = right;
    while (low <= high) {
      while (values[low] < pivot) low += 1;
      while (values[high] > pivot) high -= 1;
      if (low <= high) {
        swap(values, low, high);
        low += 1;
        high -= 1;
      }
    }
    if (target <= high) right = high;
    else if (target >= low) left = low;
    else return values[target];
  }
  return values[left];
}

function quantile(values: Float64Array, probability: number): number {
  if (values.length === 1) return values[0];
  const rank = (values.length - 1) * probability;
  const lowerIndex = Math.floor(rank);
  const upperIndex = Math.ceil(rank);
  const lower = selectKth(values, lowerIndex);
  if (lowerIndex === upperIndex) return lower;
  const upper = selectKth(values, upperIndex);
  return lower + ((upper - lower) * (rank - lowerIndex));
}

function summarizeValues(
  values: Float64Array,
  quantileRange: readonly [number, number],
): SelectionRuleValueRange {
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return {
    minimum,
    maximum,
    lowerQuantile: quantile(values, quantileRange[0]),
    upperQuantile: quantile(values, quantileRange[1]),
  };
}

/**
 * Describes a brush/cluster selection using model ranks and calibrated scores.
 * Ground truth is intentionally not accepted by this API. Both intervals are
 * the selected rows' exact 5th–95th percentiles by default.
 *
 * Runtime is expected O(rows * attributes); no full-column sort is performed.
 */
export function summarizeSelectionRules(
  input: SelectionRuleSummaryInput,
): SelectionRuleSummaryResult {
  const rowCount = input.selectionMask.length;
  assertBinaryMask(input.selectionMask, rowCount, "selectionMask");
  assertBinaryMask(input.baseMask, rowCount, "baseMask");

  const quantileRange = input.quantileRange ?? DEFAULT_QUANTILES;
  assertConfiguration(quantileRange);

  let baseCount = 0;
  let selectedCount = 0;
  const selectedRows = new Uint32Array(rowCount);
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    if (input.baseMask[rowIndex] !== 1) continue;
    baseCount += 1;
    if (input.selectionMask[rowIndex] === 1) {
      selectedRows[selectedCount] = rowIndex;
      selectedCount += 1;
    }
  }

  if (baseCount === 0) return { kind: "none", reason: "empty-base" };
  if (selectedCount === 0) return { kind: "none", reason: "empty-selection" };
  if (selectedCount === baseCount) {
    return { kind: "none", reason: "entire-base-selected" };
  }
  if (input.attributes.length === 0) return { kind: "none", reason: "no-attributes" };

  const attributeIds = new Set<string>();
  for (const attribute of input.attributes) {
    if (!attribute.id.trim() || attributeIds.has(attribute.id)) {
      throw new RangeError("Attribute IDs must be unique and non-empty.");
    }
    attributeIds.add(attribute.id);
    if (
      attribute.ranks.length !== rowCount
      || attribute.calibratedScores.length !== rowCount
    ) {
      throw new RangeError(`Ranks and scores for ${attribute.id} must be row-aligned.`);
    }
  }

  const ranges: SelectionRuleAttributeRange[] = [];
  for (const attribute of input.attributes) {
    const selectedRanks = new Float64Array(selectedCount);
    const selectedScores = new Float64Array(selectedCount);
    for (let selectedIndex = 0; selectedIndex < selectedCount; selectedIndex += 1) {
      const rowIndex = selectedRows[selectedIndex];
      const rank = attribute.ranks[rowIndex];
      const calibratedScore = attribute.calibratedScores[rowIndex];
      if (!Number.isFinite(rank) || !Number.isFinite(calibratedScore)) {
        return {
          kind: "none",
          reason: "non-finite-value",
          attributeId: attribute.id,
          rowIndex,
          valueKind: Number.isFinite(rank) ? "calibrated-score" : "rank",
        };
      }
      selectedRanks[selectedIndex] = rank;
      selectedScores[selectedIndex] = calibratedScore;
    }

    ranges.push({
      attributeId: attribute.id,
      attributeLabel: attribute.label,
      sampleCount: selectedCount,
      rank: summarizeValues(selectedRanks, quantileRange),
      calibratedScore: summarizeValues(selectedScores, quantileRange),
    });
  }

  return {
    kind: "summary",
    summary: {
      selectedCount,
      baseCount,
      lowerQuantile: quantileRange[0],
      upperQuantile: quantileRange[1],
      attributes: ranges,
    },
  };
}
