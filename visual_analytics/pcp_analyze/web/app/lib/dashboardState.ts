import type { ResultScope } from "./resultScope";

export interface ScopeActivitySummary {
  count: number;
  label: "active" | "comparison rows";
}

export interface ManualHighlightLike {
  rowIndex: number;
}

export interface ManualHighlightScope<T extends ManualHighlightLike> {
  visible: T[];
  hiddenCount: number;
}

/** Splits human-selected rows by the samples currently present in the PCP. */
export function manualHighlightScope<T extends ManualHighlightLike>(
  highlights: readonly T[],
  candidateMask: ArrayLike<number>,
): ManualHighlightScope<T> {
  const visible: T[] = [];
  let hiddenCount = 0;
  for (const highlight of highlights) {
    if (
      Number.isInteger(highlight.rowIndex)
      && highlight.rowIndex >= 0
      && highlight.rowIndex < candidateMask.length
      && Boolean(candidateMask[highlight.rowIndex])
    ) {
      visible.push(highlight);
    } else {
      hiddenCount += 1;
    }
  }
  return { visible, hiddenCount };
}

/** Removes a virtual method while keeping at least one real PCP method enabled. */
export function enabledMethodsAfterVirtualMethodRevert(
  enabledMethods: ReadonlySet<string>,
  virtualMethod: string,
  fallbackMethod: string,
): Set<string> {
  const next = new Set(enabledMethods);
  next.delete(virtualMethod);
  if (next.size === 0) next.add(fallbackMethod);
  return next;
}

/** A static exported ranking is current unless a personal tuned rank overrides it. */
export function isStaticValidationRankingCurrent(
  method: string,
  rankMethod: string,
  hasActiveTunedRanks: boolean,
): boolean {
  return method === rankMethod && !hasActiveTunedRanks;
}

/** Validation always reports its fixed comparison population, never legacy UI filters. */
export function scopeActivitySummary(
  scope: ResultScope,
  validationRowCount: number,
  resultCandidateCount: number,
): ScopeActivitySummary {
  if (scope === "validation") {
    return { count: validationRowCount, label: "comparison rows" };
  }
  return { count: resultCandidateCount, label: "active" };
}
