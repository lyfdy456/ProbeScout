import type { SmartFilterLiveValues } from "../smartFilter";
import {
  HIERARCHICAL_PCP_EMBEDDING_BASELINES,
  HIERARCHICAL_PCP_LEARNERS,
} from "./hierarchicalPcp";
import type { RefinementVisualizationSnapshot } from "./refinementVisualization";

export type DiagnosticSnapshot = Pick<
  RefinementVisualizationSnapshot,
  "rowCount" | "componentCount" | "scores" | "ranks" | "attributeIds" | "embeddingMethods"
>;

function validIdentities(values: readonly string[]) {
  return values.length > 0
    && new Set(values).size === values.length
    && values.every((value) => typeof value === "string" && value.trim().length > 0);
}

/** Average-tie gallery percentiles, matching the server convention: 1 = best. */
function normalizedRanks(values: Float64Array): Float32Array {
  const count = values.length;
  if (count === 1) return new Float32Array([1]);
  const ranks = new Float32Array(count);
  const order = Array.from({ length: count }, (_, rowIndex) => rowIndex);
  order.sort((left, right) => (
    values[left] < values[right] ? -1 : values[left] > values[right] ? 1 : left - right
  ));
  for (let start = 0; start < count;) {
    let stop = start + 1;
    while (stop < count && values[order[stop]] === values[order[start]]) stop += 1;
    const rank = (start + stop - 1) / (2 * (count - 1));
    for (let index = start; index < stop; index += 1) ranks[order[index]] = rank;
    start = stop;
  }
  return ranks;
}

/**
 * Read the current F0 or applied Tune snapshot without mutating it. The layout is
 * F, C, [g_a, eight canonical probe z columns] per attribute, H, embeddings.
 * For an attribute, use its stored g and z ranks. For Joint, each learner's
 * diagnostic rank is the all-gallery average-tie rank of the product of its
 * current z values across attributes. This is a diagnostic conjunction, not a
 * standalone learner probability or its legacy Joint score. Log products avoid
 * underflow; every exact-zero product shares the same lowest-score tie group.
 */
export function buildSnapshotDiagnosticValues(
  snapshot: DiagnosticSnapshot,
  targetId: string,
  learnerMethods: readonly string[],
): SmartFilterLiveValues {
  const { rowCount, componentCount, scores, ranks, attributeIds, embeddingMethods } = snapshot;
  if (
    !validIdentities(attributeIds) || attributeIds.includes("joint")
    || !validIdentities(embeddingMethods)
    || embeddingMethods.some((method) => !HIERARCHICAL_PCP_EMBEDDING_BASELINES.some((value) => value === method))
    || !validIdentities(learnerMethods)
    || learnerMethods.length !== HIERARCHICAL_PCP_LEARNERS.length
    || learnerMethods.some((method) => !HIERARCHICAL_PCP_LEARNERS.some((value) => value === method))
  ) {
    throw new RangeError("Snapshot diagnostic attribute and method identities are invalid.");
  }
  const expectedComponentCount = 3 + attributeIds.length * (1 + HIERARCHICAL_PCP_LEARNERS.length)
    + embeddingMethods.length;
  if (
    !Number.isSafeInteger(rowCount) || rowCount <= 0
    || componentCount !== expectedComponentCount
    || !Number.isSafeInteger(rowCount * componentCount)
    || scores.length !== rowCount * componentCount
    || ranks.length !== rowCount * componentCount
  ) {
    throw new RangeError("Snapshot diagnostic matrices do not match the component layout.");
  }
  if ([scores, ranks].some((values) => values.some((value) => (
    !Number.isFinite(value) || value < 0 || value > 1
  )))) {
    throw new RangeError("Snapshot diagnostic scores and ranks must be finite values in [0, 1].");
  }
  const joint = targetId === "joint";
  const targetAttributeIndex = attributeIds.indexOf(targetId);
  if (!joint && targetAttributeIndex < 0) {
    throw new RangeError("Snapshot diagnostic target is not an attribute or Joint.");
  }
  const memberIndices = joint
    ? attributeIds.map((_, index) => index)
    : [targetAttributeIndex];
  // Stable summation order makes a reordered attribute manifest equivalent.
  const productMemberIndices = [...memberIndices].sort((left, right) => (
    attributeIds[left] < attributeIds[right] ? -1 : attributeIds[left] > attributeIds[right] ? 1 : 0
  ));
  const gateCount = memberIndices.length;
  const gateScores = new Float32Array(rowCount * gateCount);
  const learnerRanks = new Float32Array(rowCount * learnerMethods.length);
  const fusionRanks = new Float32Array(rowCount);
  const gateColumn = (attributeIndex: number) => 2 + attributeIndex * (1 + HIERARCHICAL_PCP_LEARNERS.length);
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const offset = rowIndex * componentCount;
    for (const [memberIndex, attributeIndex] of memberIndices.entries()) {
      gateScores[rowIndex * gateCount + memberIndex] = scores[offset + gateColumn(attributeIndex)];
    }
    fusionRanks[rowIndex] = ranks[offset + (joint ? 0 : gateColumn(targetAttributeIndex))];
  }
  for (const [learnerIndex, method] of learnerMethods.entries()) {
    const canonicalIndex = HIERARCHICAL_PCP_LEARNERS.findIndex((value) => value === method);
    if (!joint) {
      const column = gateColumn(targetAttributeIndex) + 1 + canonicalIndex;
      for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
        learnerRanks[rowIndex * learnerMethods.length + learnerIndex] = ranks[rowIndex * componentCount + column];
      }
      continue;
    }
    const logProducts = new Float64Array(rowCount);
    for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
      let logProduct = 0;
      for (const attributeIndex of productMemberIndices) {
        const score = scores[rowIndex * componentCount + gateColumn(attributeIndex) + 1 + canonicalIndex];
        if (score === 0) {
          logProduct = Number.NEGATIVE_INFINITY;
          break;
        }
        logProduct += Math.log(score);
      }
      logProducts[rowIndex] = logProduct;
    }
    const jointRanks = normalizedRanks(logProducts);
    for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
      learnerRanks[rowIndex * learnerMethods.length + learnerIndex] = jointRanks[rowIndex];
    }
  }
  return { gateScores, gateCount, learnerRanks, fusionRanks, comparisonRanks: fusionRanks.slice() };
}
