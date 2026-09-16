import type { SelectionRuleSummaryResult } from "./selectionRuleSummary";
import type { PcpZoomRangeMap } from "./pcpBrushZoom";

export type SelectionOverlapRule =
  | {
      kind: "range";
      axisId: string;
      valueKind: "rank" | "calibrated";
      lower: number;
      upper: number;
      source: "zoom" | "brush" | "zoom+brush";
    }
  | {
      kind: "category";
      field: "rank-cluster" | "visual-cluster";
      value: string;
    }
  | {
      kind: "projection";
      projection: "pca" | "umap";
      xDomain: readonly [number, number];
      yDomain: readonly [number, number];
    };

export interface SelectionOverlapConditionInput {
  id: string;
  label: string;
  rule: SelectionOverlapRule;
  mask: Uint8Array;
  /** Precomputed from this condition alone, never from the combined intersection. */
  ruleResult: SelectionRuleSummaryResult;
}

export interface SelectionOverlapStageInput {
  id: string;
  label: string;
  /** Cumulative mask after this downstream stage has been applied. */
  mask: Uint8Array;
}

export interface SelectionOverlapConditionSummary {
  id: string;
  label: string;
  code: string;
  count: number;
  cumulativeCount: number;
  rule: SelectionOverlapRule;
  ruleResult: SelectionRuleSummaryResult;
}

export interface SelectionOverlapRegion {
  /** Bit i is set when the row belongs to visible condition i. */
  key: number;
  codes: readonly string[];
  count: number;
}

export interface SelectionOverlapStageSummary {
  id: string;
  label: string;
  count: number;
}

export interface SelectionOverlapSummary {
  baseCount: number;
  conditions: readonly SelectionOverlapConditionSummary[];
  visibleConditions: readonly SelectionOverlapConditionSummary[];
  hiddenConditions: readonly SelectionOverlapConditionSummary[];
  regions: readonly SelectionOverlapRegion[];
  outsideVisibleCount: number;
  visibleIntersectionCount: number;
  coreIntersectionCount: number;
  stages: readonly SelectionOverlapStageSummary[];
  finalCount: number;
}

export type SelectionOverlapResult =
  | { kind: "none"; reason: "no-active-selection" }
  | { kind: "summary"; summary: SelectionOverlapSummary };

const VISIBLE_SET_LIMIT = 3;
const RANGE_EPSILON = Number.EPSILON;

export interface EffectivePcpSelectionRange {
  axisId: string;
  range: readonly [number, number];
  source: "zoom" | "brush" | "zoom+brush";
}

function normalizeRange(range: readonly [number, number] | undefined) {
  if (!range || !Number.isFinite(range[0]) || !Number.isFinite(range[1])) return null;
  const lower = Math.min(range[0], range[1]);
  const upper = Math.max(range[0], range[1]);
  return upper - lower > RANGE_EPSILON ? [lower, upper] as const : null;
}

/** Build one canonical effective condition per active PCP axis. */
export function effectivePcpSelectionRanges(
  axisIds: readonly string[],
  committedRanges: PcpZoomRangeMap,
  workingBrushes: PcpZoomRangeMap,
): readonly EffectivePcpSelectionRange[] {
  const seen = new Set<string>();
  return axisIds.flatMap<EffectivePcpSelectionRange>((axisId) => {
    if (!axisId.trim() || seen.has(axisId)) {
      throw new Error("PCP overlap axis ids must be unique and non-empty.");
    }
    seen.add(axisId);
    const committed = normalizeRange(committedRanges[axisId]);
    const working = normalizeRange(workingBrushes[axisId]);
    if (!committed && !working) return [];
    if (committed && working) {
      const range = [
        Math.max(committed[0], working[0]),
        Math.min(committed[1], working[1]),
      ] as const;
      if (range[1] - range[0] <= RANGE_EPSILON) {
        throw new Error(`PCP zoom and brush do not overlap for axis ${axisId}.`);
      }
      return [{ axisId, range, source: "zoom+brush" as const }];
    }
    return [{
      axisId,
      range: (committed ?? working)!,
      source: committed ? "zoom" as const : "brush" as const,
    }];
  });
}

function assertBinaryMask(mask: Uint8Array, rowCount: number, label: string) {
  if (mask.length !== rowCount) {
    throw new Error(`${label} must be row-aligned with the base mask.`);
  }
  for (const value of mask) {
    if (value !== 0 && value !== 1) {
      throw new Error(`${label} must be a binary mask.`);
    }
  }
}

function countWithinBase(mask: Uint8Array, baseMask: Uint8Array) {
  let count = 0;
  for (let index = 0; index < baseMask.length; index += 1) {
    if (baseMask[index] && mask[index]) count += 1;
  }
  return count;
}

function conditionCode(index: number) {
  return index < 26 ? String.fromCharCode(65 + index) : `S${index + 1}`;
}

export function summarizeSelectionOverlap({
  baseMask,
  conditions,
  stages = [],
  finalMask,
}: {
  baseMask: Uint8Array;
  conditions: readonly SelectionOverlapConditionInput[];
  stages?: readonly SelectionOverlapStageInput[];
  finalMask: Uint8Array;
}): SelectionOverlapResult {
  assertBinaryMask(baseMask, baseMask.length, "Base mask");
  assertBinaryMask(finalMask, baseMask.length, "Final mask");

  const seenIds = new Set<string>();
  for (const entry of [...conditions, ...stages]) {
    if (!entry.id.trim() || !entry.label.trim()) {
      throw new Error("Selection overlap ids and labels must be non-empty.");
    }
    if (seenIds.has(entry.id)) {
      throw new Error(`Selection overlap id ${entry.id} is duplicated.`);
    }
    seenIds.add(entry.id);
    assertBinaryMask(entry.mask, baseMask.length, `${entry.label} mask`);
  }

  if (conditions.length === 0 && stages.length === 0) {
    return { kind: "none", reason: "no-active-selection" };
  }

  const baseCount = countWithinBase(baseMask, baseMask);
  const cumulativeCounts = new Uint32Array(conditions.length);
  for (let rowIndex = 0; rowIndex < baseMask.length; rowIndex += 1) {
    if (!baseMask[rowIndex]) continue;
    let included = true;
    for (let conditionIndex = 0; conditionIndex < conditions.length; conditionIndex += 1) {
      included &&= conditions[conditionIndex].mask[rowIndex] === 1;
      if (included) cumulativeCounts[conditionIndex] += 1;
    }
  }
  const allConditionSummaries = conditions.map((condition, index) => ({
    id: condition.id,
    label: condition.label,
    code: conditionCode(index),
    count: countWithinBase(condition.mask, baseMask),
    cumulativeCount: cumulativeCounts[index],
    rule: condition.rule,
    ruleResult: condition.ruleResult,
  }));
  const visibleConditions = allConditionSummaries.length <= VISIBLE_SET_LIMIT
    ? allConditionSummaries
    : [];
  const hiddenConditions = allConditionSummaries.length <= VISIBLE_SET_LIMIT
    ? []
    : allConditionSummaries;
  const regionConditionSummaries = allConditionSummaries.slice(0, VISIBLE_SET_LIMIT);
  const visibleInputs = conditions.slice(0, VISIBLE_SET_LIMIT);
  const regionCounts = new Uint32Array(1 << visibleInputs.length);
  let coreIntersectionCount = 0;
  let visibleIntersectionCount = 0;
  const coreMask = new Uint8Array(baseMask.length);

  for (let rowIndex = 0; rowIndex < baseMask.length; rowIndex += 1) {
    if (!baseMask[rowIndex]) continue;
    let regionKey = 0;
    for (let conditionIndex = 0; conditionIndex < visibleInputs.length; conditionIndex += 1) {
      if (visibleInputs[conditionIndex].mask[rowIndex]) regionKey |= 1 << conditionIndex;
    }
    regionCounts[regionKey] += 1;
    if (regionKey === regionCounts.length - 1) visibleIntersectionCount += 1;

    let insideCore = true;
    for (const condition of conditions) {
      if (!condition.mask[rowIndex]) {
        insideCore = false;
        break;
      }
    }
    if (insideCore) {
      coreMask[rowIndex] = 1;
      coreIntersectionCount += 1;
    }
  }

  const regions = Array.from(
    { length: Math.max(0, regionCounts.length - 1) },
    (_, offset): SelectionOverlapRegion => {
      const key = offset + 1;
      return {
        key,
        codes: regionConditionSummaries
          .filter((_, index) => (key & (1 << index)) !== 0)
          .map((condition) => condition.code),
        count: regionCounts[key],
      };
    },
  );

  const stageSummaries: SelectionOverlapStageSummary[] = [];
  let previousMask: Uint8Array = coreMask;
  for (const stage of stages) {
    for (let rowIndex = 0; rowIndex < baseMask.length; rowIndex += 1) {
      if (baseMask[rowIndex] && stage.mask[rowIndex] && !previousMask[rowIndex]) {
        throw new Error(`${stage.label} must be a subset of the preceding selection stage.`);
      }
    }
    stageSummaries.push({
      id: stage.id,
      label: stage.label,
      count: countWithinBase(stage.mask, baseMask),
    });
    previousMask = stage.mask;
  }

  for (let rowIndex = 0; rowIndex < baseMask.length; rowIndex += 1) {
    if (!baseMask[rowIndex]) continue;
    if (finalMask[rowIndex] !== previousMask[rowIndex]) {
      throw new Error("Final selection must equal the preceding selection stage.");
    }
  }

  return {
    kind: "summary",
    summary: {
      baseCount,
      conditions: allConditionSummaries,
      visibleConditions,
      hiddenConditions,
      regions,
      outsideVisibleCount: regionCounts[0] ?? 0,
      visibleIntersectionCount,
      coreIntersectionCount,
      stages: stageSummaries,
      finalCount: countWithinBase(finalMask, baseMask),
    },
  };
}
