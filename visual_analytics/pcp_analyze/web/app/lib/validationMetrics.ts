interface RankedScopeContractInput {
  /** Row-aligned score or normalized-rank vector; larger values rank first. */
  ranks: ArrayLike<number>;
  /** Row-major [row, target] binary ground-truth tensor. */
  groundTruth: ArrayLike<number>;
  scopeMask: Uint8Array;
  rowCount: number;
  targetIndex: number;
  targetCount: number;
}

export interface RankedScopeEvaluationInput extends RankedScopeContractInput {
  k: number;
}

export interface RankedScopeTopKInput extends RankedScopeContractInput {
  ks: readonly number[];
}

export interface RankedScopeEvaluation {
  averagePrecision: number;
  bestF1: number;
  bestCutoff: number;
  precisionAtK: number;
  recallAtK: number;
  positiveCount: number;
  evaluatedCount: number;
  truePositiveAtK: number;
}

export interface RankedScopeTopKEvaluation {
  positiveCount: number;
  evaluatedCount: number;
  truePositiveAtK: Record<number, number>;
}

interface RankedRow {
  rowIndex: number;
  rank: number;
  positive: boolean;
}

function assertEvaluationContract(input: RankedScopeContractInput) {
  const { groundTruth, ranks, rowCount, scopeMask, targetCount, targetIndex } = input;
  if (!Number.isInteger(rowCount) || rowCount < 0) {
    throw new RangeError("Evaluation row count must be a non-negative integer.");
  }
  if (!Number.isInteger(targetCount) || targetCount <= 0) {
    throw new RangeError("Evaluation target count must be a positive integer.");
  }
  if (!Number.isInteger(targetIndex) || targetIndex < 0 || targetIndex >= targetCount) {
    throw new RangeError("Evaluation target index is outside the target contract.");
  }
  if (ranks.length !== rowCount || scopeMask.length !== rowCount) {
    throw new RangeError("Ranks and scope mask must align with the evaluation rows.");
  }
  if (groundTruth.length !== rowCount * targetCount) {
    throw new RangeError("Ground truth must use the declared row-target layout.");
  }
}

function descendingRankThenRow(left: RankedRow, right: RankedRow) {
  const leftNaN = Number.isNaN(left.rank);
  const rightNaN = Number.isNaN(right.rank);
  if (leftNaN !== rightNaN) return leftNaN ? 1 : -1;
  if (!leftNaN && left.rank !== right.rank) return left.rank > right.rank ? -1 : 1;
  return left.rowIndex - right.rowIndex;
}

function sortedRowsInScope(input: RankedScopeContractInput) {
  assertEvaluationContract(input);
  const { groundTruth, ranks, rowCount, scopeMask, targetCount, targetIndex } = input;
  const rows: RankedRow[] = [];
  let positiveCount = 0;

  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const membership = scopeMask[rowIndex];
    if (membership !== 0 && membership !== 1) {
      throw new RangeError("Evaluation scope mask must contain only binary 0/1 values.");
    }
    if (!membership) continue;
    const positive = groundTruth[rowIndex * targetCount + targetIndex] === 1;
    positiveCount += positive ? 1 : 0;
    rows.push({ rowIndex, rank: Number(ranks[rowIndex]), positive });
  }

  rows.sort(descendingRankThenRow);
  return { positiveCount, rows };
}

/**
 * Computes post-hoc retrieval metrics within one explicit split.
 * Equal ranks use source row order, matching the workbench's stable ranking.
 */
export function evaluateRankedScope(
  input: RankedScopeEvaluationInput,
): RankedScopeEvaluation {
  const { positiveCount, rows } = sortedRowsInScope(input);
  const requestedK = Number.isFinite(input.k) ? Math.max(0, Math.floor(input.k)) : 0;
  const evaluatedK = Math.min(requestedK, rows.length);
  let truePositives = 0;
  let truePositiveAtK = 0;
  let precisionSum = 0;
  let bestF1 = 0;
  let bestCutoff = 0;

  for (let index = 0; index < rows.length; index += 1) {
    if (rows[index].positive) {
      truePositives += 1;
      precisionSum += truePositives / (index + 1);
    }
    if (index + 1 === evaluatedK) truePositiveAtK = truePositives;
    if (positiveCount > 0) {
      const f1 = (2 * truePositives) / (index + 1 + positiveCount);
      if (f1 > bestF1) {
        bestF1 = f1;
        bestCutoff = index + 1;
      }
    }
  }

  return {
    averagePrecision: positiveCount > 0 ? precisionSum / positiveCount : 0,
    bestF1,
    bestCutoff,
    precisionAtK: evaluatedK > 0 ? truePositiveAtK / evaluatedK : 0,
    recallAtK: positiveCount > 0 ? truePositiveAtK / positiveCount : 0,
    positiveCount,
    evaluatedCount: rows.length,
    truePositiveAtK,
  };
}

/**
 * Computes TP at several cutoffs after one stable sort of one explicit split.
 * This intentionally returns only audit counts, so it cannot be mistaken for
 * a training objective or a source of fusion weights.
 */
export function evaluateTruePositivesAtKs(
  input: RankedScopeTopKInput,
): RankedScopeTopKEvaluation {
  if (input.ks.some((k) => !Number.isInteger(k) || k < 0)) {
    throw new RangeError("Evaluation cutoffs must be non-negative integers.");
  }
  const { positiveCount, rows } = sortedRowsInScope(input);
  const cutoffs = [...new Set(input.ks)].sort((left, right) => left - right);
  const truePositiveAtK: Record<number, number> = {};
  let cursor = 0;
  let truePositives = 0;

  for (const cutoff of cutoffs) {
    const evaluatedK = Math.min(cutoff, rows.length);
    while (cursor < evaluatedK) {
      truePositives += rows[cursor].positive ? 1 : 0;
      cursor += 1;
    }
    truePositiveAtK[cutoff] = truePositives;
  }

  return {
    positiveCount,
    evaluatedCount: rows.length,
    truePositiveAtK,
  };
}
