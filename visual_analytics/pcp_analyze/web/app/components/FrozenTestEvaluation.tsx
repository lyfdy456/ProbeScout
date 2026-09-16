"use client";

import { methodDisplayLabel } from "../lib/methodDisplay.js";
import type { TuningRun } from "../lib/tuningApi";
import { apDeltaTone, formatAp, formatApDelta, tuningApRows } from "../lib/tuningEvaluation";

/** Read-only results of the last completed refinement, shown only in Frozen Test. */
export function FrozenTestEvaluation({ run, stale = false }: { run: TuningRun; stale?: boolean }) {
  if (run.status !== "succeeded") return null;
  const [row] = tuningApRows(run, "test");
  const evaluation = run.evaluationScope === "test" ? run : run.testEvaluation;
  const beforeMethod = run.evaluationScope === "test"
    ? run.beforeMethod ?? run.baseMethod
    : run.testEvaluation?.beforeMethod ?? run.beforeMethod ?? run.baseMethod;

  return (
    <div className="tuning-panel">
      <div className="tuning-result-heading">
        <strong>Last completed refinement · {row.label}</strong>
        <em>{stale ? "Labels changed since this run" : "Read-only"}</em>
      </div>
      <div className="tuning-ap-comparison">
        <table aria-label="Test AP comparison">
          <thead><tr>
            <th scope="col">AP</th>
            <th scope="col">Before</th>
            <th scope="col">After</th>
            <th scope="col">ΔAP (pp)</th>
          </tr></thead>
          <tbody><tr>
            <th scope="row" title={run.testEvaluationError}>
              {row.label}{row.note && <small>{row.note}</small>}
            </th>
            <td title={`Baseline: ${methodDisplayLabel(beforeMethod)}`}>{formatAp(row.before)}</td>
            <td>{formatAp(row.after)}</td>
            <td className={`ap-delta-${apDeltaTone(row.delta)}`}>{formatApDelta(row.delta)}</td>
          </tr></tbody>
        </table>
      </div>
      {evaluation?.before?.bestF1 !== undefined && evaluation?.after?.bestF1 !== undefined && (
        <div className="tuning-tp-row tuning-f1-row">
          <span>Test · Best F1 {formatAp(evaluation.before.bestF1)} → {formatAp(evaluation.after.bestF1)}</span>
        </div>
      )}
      <span className="tuning-metric-caption">Test · TP@K</span>
      <div className="tuning-tp-row" role="group" aria-label="Test TP before and after">
        <span>TP@30 {evaluation?.before?.tpAt30 ?? "—"} → {evaluation?.after?.tpAt30 ?? "—"}</span>
        <span>TP@50 {evaluation?.before?.tpAt50 ?? "—"} → {evaluation?.after?.tpAt50 ?? "—"}</span>
        <span>TP@100 {evaluation?.before?.tpAt100 ?? "—"} → {evaluation?.after?.tpAt100 ?? "—"}</span>
        <span>TP@200 {evaluation?.before?.tpAt200 ?? "—"} → {evaluation?.after?.tpAt200 ?? "—"}</span>
      </div>
      <p className="tuning-integrity-note">Recorded on the full Frozen Test split; Gallery filters do not change these results.</p>
    </div>
  );
}
