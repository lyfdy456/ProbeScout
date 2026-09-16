import type { FixedVqaValidationResponse } from "./tuningApi";

/** Validate the whole payload before making any row eligible for feedback. */
export function fixedVqaValidationMask(
  response: FixedVqaValidationResponse,
  taskId: string,
  rowCount: number,
): Uint8Array {
  if (
    !Number.isInteger(rowCount) || rowCount <= 0
    || response.taskId !== taskId
    || typeof response.version !== "string" || !response.version.trim()
    || !/^[a-f0-9]{64}$/i.test(response.manifestSha256)
    || response.protocol !== "fixed-vqa-joint-seed0-holdout-v1"
    || response.labelSource !== "original-vqa-supervision"
    || response.initialModelHoldoutIndependent !== false
    || (response.referenceOnly !== undefined && response.referenceOnly !== true)
    || !Array.isArray(response.rowIndices)
    || !Number.isInteger(response.count) || response.count <= 0
    || response.count !== response.rowIndices.length
  ) throw new Error("The fixed Val membership contract is invalid.");
  const mask = new Uint8Array(rowCount);
  for (const rowIndex of response.rowIndices) {
    if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= rowCount || mask[rowIndex]) {
      throw new Error("The fixed Val membership contains an invalid or duplicate row.");
    }
    mask[rowIndex] = 1;
  }
  return mask;
}

/** Pending/missing membership fails closed, without changing the source DG mask. */
export function developmentWithoutFixedVal(
  developmentMask: Uint8Array,
  validationMask: Uint8Array | null,
): Uint8Array {
  const result = new Uint8Array(developmentMask.length);
  if (!validationMask) return result;
  if (validationMask.length !== developmentMask.length) {
    throw new Error("Fixed Val membership does not match the task row count.");
  }
  for (let row = 0; row < result.length; row += 1) {
    result[row] = developmentMask[row] === 1 && validationMask[row] === 0 ? 1 : 0;
  }
  return result;
}

export function fixedValAllowsFeedback(
  item: { rowIndex: number; id: string },
  imageIds: readonly string[],
  developmentMask: Uint8Array,
  validationMask: Uint8Array | null,
): boolean {
  return Boolean(
    validationMask
    && validationMask.length === imageIds.length
    && developmentMask.length === imageIds.length
    && Number.isInteger(item.rowIndex)
    && item.rowIndex >= 0 && item.rowIndex < imageIds.length
    && imageIds[item.rowIndex] === item.id
    && developmentMask[item.rowIndex] === 1
    && validationMask[item.rowIndex] === 0,
  );
}
