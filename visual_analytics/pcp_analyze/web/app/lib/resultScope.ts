export type ResultScope = "development" | "validation" | "test";

export interface ScopeMasks {
  developmentMask: Uint8Array;
  validationMask: Uint8Array;
  testMask: Uint8Array;
}

const SCOPE_MASK_KEYS = {
  development: "developmentMask",
  validation: "validationMask",
  test: "testMask",
} as const satisfies Record<ResultScope, keyof ScopeMasks>;

function assertRowCount(rowCount: number) {
  if (!Number.isInteger(rowCount) || rowCount < 0) {
    throw new RangeError("Scope row count must be a non-negative integer.");
  }
}

/**
 * Validates three explicit, row-aligned split masks.
 *
 * A row may be absent from all three masks (fixed Query rows use this state),
 * but it may never belong to more than one split.
 */
export function validateScopeMasks(rowCount: number, masks: ScopeMasks): void {
  assertRowCount(rowCount);
  const entries = Object.entries(masks) as Array<[keyof ScopeMasks, Uint8Array]>;
  for (const [key, mask] of entries) {
    if (!(mask instanceof Uint8Array) || mask.length !== rowCount) {
      throw new RangeError(`${key} must be a Uint8Array aligned to the scope row count.`);
    }
  }

  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    let memberships = 0;
    for (const [key, mask] of entries) {
      const value = mask[rowIndex];
      if (value !== 0 && value !== 1) {
        throw new RangeError(`${key} must contain only binary 0/1 membership values.`);
      }
      memberships += value;
    }
    if (memberships > 1) {
      throw new RangeError(`Scope masks overlap at row ${rowIndex}.`);
    }
  }
}

/** Returns the selected split mask without copying it. */
export function scopeMaskFor(
  rowCount: number,
  masks: ScopeMasks,
  scope: ResultScope,
): Uint8Array {
  validateScopeMasks(rowCount, masks);
  return masks[SCOPE_MASK_KEYS[scope]];
}

/** The dataset-wide rows visible to analysis views before downstream filters. */
export function analysisScopeMask(
  rowCount: number,
  masks: ScopeMasks,
  scope: ResultScope,
): Uint8Array {
  return scopeMaskFor(rowCount, masks, scope);
}

/** Keeps row-aligned items inside one explicit split mask, preserving order. */
export function filterAnalysisRows<Row>(
  rows: readonly Row[],
  rowIndex: (row: Row) => number,
  scopeMask: Uint8Array,
): readonly Row[] {
  return rows.filter((row) => {
    const index = rowIndex(row);
    return Number.isInteger(index)
      && index >= 0
      && index < scopeMask.length
      && scopeMask[index] === 1;
  });
}

/** Intersects existing candidates with the selected explicit data split. */
export function applyResultScope(
  candidateMask: Uint8Array,
  masks: ScopeMasks,
  scope: ResultScope,
): Uint8Array {
  const scopeMask = scopeMaskFor(candidateMask.length, masks, scope);
  const result = new Uint8Array(candidateMask.length);
  for (let index = 0; index < result.length; index += 1) {
    result[index] = candidateMask[index] && scopeMask[index] ? 1 : 0;
  }
  return result;
}

export function isGroundTruthPositive(
  groundTruth: Uint8Array,
  rowIndex: number,
  targetIndex: number,
  targetCount: number,
): boolean {
  if (
    !Number.isInteger(rowIndex)
    || !Number.isInteger(targetIndex)
    || !Number.isInteger(targetCount)
    || rowIndex < 0
    || targetIndex < 0
    || targetCount <= 0
    || targetIndex >= targetCount
  ) return false;
  const offset = rowIndex * targetCount + targetIndex;
  return offset >= 0 && offset < groundTruth.length && groundTruth[offset] === 1;
}

export function countRemainingPositives(
  totalPositiveCount: number,
  topTruePositiveCount: number,
): number {
  const total = Math.max(0, Math.floor(totalPositiveCount));
  const found = Math.max(0, Math.floor(topTruePositiveCount));
  return Math.max(0, total - found);
}
