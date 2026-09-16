import type { TuningRun } from "./tuningApi";

export interface TuningApRow {
  key: "primary" | "test";
  label: string;
  before?: number;
  after?: number;
  delta?: number;
  note?: string;
}

function validAp(value: number | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1
    ? value : undefined;
}

export function tuningEvaluationLabel(scope: TuningRun["evaluationScope"]): string {
  switch (scope) {
    case "vqa-validation": return "Val";
    case "clean-validation": return "Clean Val (legacy)";
    case "probe-validation": return "Probe Val (legacy)";
    case "test": return "Test (legacy)";
    default: return "Web Val (legacy)";
  }
}

export function isValidationEvaluation(scope: TuningRun["evaluationScope"]): boolean {
  return scope === "vqa-validation" || scope === "clean-validation"
    || scope === "probe-validation" || scope === "validation";
}

export function tuningApRows(
  run: TuningRun,
  view: "validation" | "test" | "all" = "all",
): TuningApRow[] {
  const before = validAp(run.before?.ap);
  const after = validAp(run.after?.ap);
  const primary: TuningApRow = {
    key: "primary", label: tuningEvaluationLabel(run.evaluationScope), before, after,
    delta: before !== undefined && after !== undefined ? after - before : undefined,
  };
  // Development may show only explicitly identified Validation evaluations.
  // Missing/unknown scopes and old Test-only runs must not become Val results.
  if (view === "validation") return isValidationEvaluation(run.evaluationScope) ? [primary] : [];
  // An old Test-only evaluation must not be relabeled as Val or duplicated.
  if (run.evaluationScope === "test") return [primary];
  const testBefore = validAp(run.testEvaluation?.before?.ap);
  const testAfter = validAp(run.testEvaluation?.after?.ap);
  const test: TuningApRow = {
    key: "test", label: "Test", before: testBefore, after: testAfter,
    delta: testBefore !== undefined && testAfter !== undefined ? testAfter - testBefore : undefined,
    note: testBefore === undefined || testAfter === undefined
      ? run.testEvaluationError ? "Unavailable" : "Not recorded" : undefined,
  };
  return view === "test" ? [test] : [primary, test];
}

export function formatAp(value: number | undefined): string {
  const ap = validAp(value);
  return ap === undefined ? "—" : `${(ap * 100).toFixed(2)}%`;
}

export function apDeltaTone(value: number | undefined): "positive" | "negative" | "neutral" {
  // Avoid a green/red or negative-zero change that rounds to 0.00 pp.
  if (value === undefined || !Number.isFinite(value) || Math.abs(value) < 0.00005) return "neutral";
  return value > 0 ? "positive" : "negative";
}

export function formatApDelta(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  const tone = apDeltaTone(value);
  if (tone === "neutral") return "0.00";
  return `${tone === "positive" ? "+" : ""}${(value * 100).toFixed(2)}`;
}
