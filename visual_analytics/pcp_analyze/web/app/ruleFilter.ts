export type RuleComparator = ">" | ">=" | "<" | "<=" | "between" | "outside";
export type RuleLogic = "AND" | "OR";
export type RuleValueKind = "calibrated" | "rank";

export interface RuleClause {
  id: string;
  target: string;
  valueKind: RuleValueKind;
  comparator: RuleComparator;
  threshold: number;
  upperThreshold: number;
  negated: boolean;
}

export interface RuleGroup {
  id: string;
  logic: RuleLogic;
  clauses: RuleClause[];
}

export interface RuleEvaluationSource {
  rowCount: number;
  methodCount: number;
  targetCount: number;
  methodIndex: number;
  targetIndices: ReadonlyMap<string, number>;
  calibratedScores: Float32Array;
  ranks: Float32Array;
  targetOverride?: {
    targetIndex: number;
    calibratedScores?: Float32Array;
    ranks: Float32Array;
  };
}

function scoreOffset(
  rowIndex: number,
  methodIndex: number,
  targetIndex: number,
  methodCount: number,
  targetCount: number,
) {
  return ((rowIndex * methodCount + methodIndex) * targetCount) + targetIndex;
}

export function matchesRuleComparator(
  value: number,
  comparator: RuleComparator,
  threshold: number,
  upperThreshold: number,
) {
  if (!Number.isFinite(value)) return false;
  const lower = Math.min(threshold, upperThreshold);
  const upper = Math.max(threshold, upperThreshold);
  if (comparator === ">") return value > threshold;
  if (comparator === ">=") return value >= threshold;
  if (comparator === "<") return value < threshold;
  if (comparator === "<=") return value <= threshold;
  if (comparator === "between") return value >= lower && value <= upper;
  return value < lower || value > upper;
}

function evaluateClause(
  rowIndex: number,
  clause: RuleClause,
  source: RuleEvaluationSource,
) {
  const targetIndex = source.targetIndices.get(clause.target);
  if (targetIndex === undefined) return false;
  const targetOverride = source.targetOverride?.targetIndex === targetIndex
    ? source.targetOverride
    : null;
  const overrideValues = clause.valueKind === "rank"
    ? targetOverride?.ranks
    : targetOverride?.calibratedScores;
  const value = overrideValues
    ? overrideValues[rowIndex]
    : (clause.valueKind === "rank" ? source.ranks : source.calibratedScores)[scoreOffset(
        rowIndex,
        source.methodIndex,
        targetIndex,
        source.methodCount,
        source.targetCount,
      )];
  const matched = matchesRuleComparator(
    value,
    clause.comparator,
    clause.threshold,
    clause.upperThreshold,
  );
  return clause.negated ? !matched : matched;
}

/**
 * Evaluates parenthesized rule groups without using ground-truth labels.
 * Clauses inside a group share its logic, then non-empty groups share rootLogic.
 */
export function buildRuleMask(
  groups: readonly RuleGroup[],
  rootLogic: RuleLogic,
  source: RuleEvaluationSource,
) {
  const activeGroups = groups.filter((group) => group.clauses.length > 0);
  const mask = new Uint8Array(source.rowCount);
  if (activeGroups.length === 0) {
    mask.fill(1);
    return mask;
  }

  for (let rowIndex = 0; rowIndex < source.rowCount; rowIndex += 1) {
    const matchesGroup = (group: RuleGroup) => {
      if (group.logic === "AND") {
        return group.clauses.every((clause) => evaluateClause(rowIndex, clause, source));
      }
      return group.clauses.some((clause) => evaluateClause(rowIndex, clause, source));
    };
    const matched = rootLogic === "AND"
      ? activeGroups.every(matchesGroup)
      : activeGroups.some(matchesGroup);
    mask[rowIndex] = matched ? 1 : 0;
  }
  return mask;
}
