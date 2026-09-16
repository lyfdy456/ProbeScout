/**
 * UI-only method aliases. Keep the original IDs in data, state, and API payloads.
 * @param {string} methodId
 * @returns {string}
 */
export function methodDisplayLabel(methodId) {
  return methodId === "Ours-PURA" ? "PURA" : methodId;
}
